"""Create a five-backbone Unified ensemble-adjudication figure on DGX."""

import argparse
import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register custom model layers and preprocessing functions before loading models.
from src import dgx_models  # noqa: E402,F401
from src.dgx_dataloader import load_all_images, process_single_image  # noqa: E402
from src.ensemble_adjudication import (  # noqa: E402
    CLASS_NAMES,
    adjudicate,
    select_figure_cases,
)


MODEL_NAMES = (
    "ResNet-50",
    "DenseNet-121",
    "EfficientNet-B4",
    "ConvNeXt-Tiny",
    "ViT-B-16",
)
CLASS_COLORS = ("#b2182b", "#f4a582", "#d6604d", "#b2182b")
CASE_ORDER = ("Unanimous", "Majority rescue", "Tie-break", "Safety fallback", "Ensemble error")


def load_mixed_data(base_dir, cache_dir=None):
    """Load real labeled examples from the mixed dataset in new_drive."""
    mixed_root = os.path.join(base_dir, "Dataset+Code", "MES Mixed Data")
    if not os.path.isdir(mixed_root):
        raise FileNotFoundError(
            f"Mixed dataset folder not found: {mixed_root}. Expected MES0-MES3 folders."
        )
    mixed = load_all_images([mixed_root], "Unified", cache_dir=cache_dir)
    mixed_images, mixed_features, mixed_labels, mixed_paths = mixed
    label_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    unknown_labels = sorted(set(mixed_labels) - set(label_to_index))
    if unknown_labels:
        raise ValueError(f"Unexpected mixed dataset labels: {unknown_labels}")

    return (
        np.asarray(mixed_images, dtype=np.uint8),
        np.asarray(mixed_features, dtype=np.float32),
        np.asarray([label_to_index[label] for label in mixed_labels], dtype=np.int64),
        mixed_paths,
    )


def load_explicit_samples(base_dir, sample_list_path):
    """Load a small, user-curated list of paths relative to Dataset+Code."""
    dataset_root = os.path.join(base_dir, "Dataset+Code")
    with open(sample_list_path, "r", encoding="utf-8") as sample_file:
        relative_paths = [
            line.strip()
            for line in sample_file
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not relative_paths:
        raise ValueError(f"No image paths found in sample list: {sample_list_path}")

    images, features, labels, paths = [], [], [], []
    label_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    for relative_path in relative_paths:
        relative_path = relative_path.removeprefix("Dataset+Code/")
        if relative_path.startswith("./"):
            relative_path = relative_path[2:]
        if os.path.isabs(relative_path):
            raise ValueError("Sample paths must be relative to Dataset+Code/.")
        image_path = os.path.join(dataset_root, relative_path)
        if os.path.commonpath((dataset_root, os.path.abspath(image_path))) != os.path.abspath(
            dataset_root
        ):
            raise ValueError("Sample paths must stay inside Dataset+Code/.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Selected figure image does not exist: {image_path}")
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name not in label_to_index:
            raise ValueError(
                f"Expected a MES0-MES3 parent folder for selected image: {image_path}"
            )
        processed = process_single_image(image_path, class_name)
        if processed is None:
            raise ValueError(f"Could not preprocess selected figure image: {image_path}")
        image, feature, label, resolved_path = processed
        images.append(image)
        features.append(feature)
        labels.append(label_to_index[label])
        paths.append(resolved_path)

    return (
        np.asarray(images, dtype=np.uint8),
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        paths,
    )


def _rank_figure_candidates(labels, adjudications, probabilities):
    candidates = {
        "Unanimous": [],
        "Majority rescue": [],
        "Tie-break": [],
        "Safety fallback": [],
        "Ensemble error": [],
    }
    for index, result in enumerate(adjudications):
        correct = result["label"] == int(labels[index])
        if result["rule"] == "Unanimous consensus" and correct:
            candidates["Unanimous"].append(index)
        elif (
            result["rule"] == "Majority vote"
            and correct
            and max(result["vote_counts"].values()) == 3
        ):
            candidates["Majority rescue"].append(index)
        elif result["rule"] == "Mean-probability tie-break" and correct:
            candidates["Tie-break"].append(index)
        elif result["rule"] == "Safety fallback: most severe" and correct:
            candidates["Safety fallback"].append(index)
        if not correct:
            candidates["Ensemble error"].append(index)

    def confidence(index):
        return sum(
            float(np.max(model_probabilities[index]))
            for model_probabilities in probabilities
        )

    for case_name in candidates:
        candidates[case_name].sort(
            key=lambda index: (confidence(index), -index), reverse=True
        )
    return candidates


def _same_grade_image_candidates(image_path):
    image_directory = os.path.dirname(image_path)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return [
        os.path.join(image_directory, filename)
        for filename in sorted(os.listdir(image_directory))
        if os.path.splitext(filename)[1].lower() in extensions
    ]


def _custom_objects():
    import tensorflow_hub as hub

    return {
        "KerasLayer": hub.KerasLayer,
        "ViT_B16_Wrapper": dgx_models.ViT_B16_Wrapper,
        "resnet50_preprocess": dgx_models.resnet50_preprocess,
        "densenet_preprocess": dgx_models.densenet_preprocess,
        "efficientnet_preprocess": dgx_models.efficientnet_preprocess,
        "convnext_preprocess": dgx_models.convnext_preprocess,
        "vit_preprocess": dgx_models.vit_preprocess,
    }


def _load_hybrid_model(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Missing trained model: {model_path}. Check --models-dir and confirm the "
            "Unified model weights exist on DGX."
        )
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=_custom_objects(),
        safe_mode=False,
    )


def _model_artifact_paths(models_dir, model_name):
    experiment_dir = os.path.join(models_dir, f"{model_name}_Experiment")
    return {
        "model": os.path.join(experiment_dir, f"{model_name}_hybrid.keras"),
        "scaler": os.path.join(experiment_dir, "base_scaler.pkl"),
        "umap": os.path.join(experiment_dir, "umap_model.pkl"),
    }


def predict_models(models_dir, images, features, batch_size):
    predictions = []
    probabilities = []

    for model_name in MODEL_NAMES:
        paths = _model_artifact_paths(models_dir, model_name)
        missing = [path for path in paths.values() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"Required Unified artifacts for {model_name} are missing:\n  "
                + "\n  ".join(missing)
            )

        print(f"Loading {model_name} for Unified held-out inference...")
        model = _load_hybrid_model(paths["model"])
        scaler = joblib.load(paths["scaler"])
        umap_model = joblib.load(paths["umap"])
        scaled_features = scaler.transform(features)
        umap_features = umap_model.transform(scaled_features)
        model_probabilities = []

        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch_images = np.asarray(images[start:end], dtype=np.float32)
            batch_probabilities = model(
                [
                    batch_images,
                    scaled_features[start:end],
                    umap_features[start:end],
                ],
                training=False,
            )
            model_probabilities.append(np.asarray(batch_probabilities))

        model_probabilities = np.concatenate(model_probabilities, axis=0)
        probabilities.append(model_probabilities)
        predictions.append(np.argmax(model_probabilities, axis=1))
        print(f"  Inferred {len(model_probabilities)} input images.")
        del model, scaler, umap_model, scaled_features, umap_features
        tf.keras.backend.clear_session()

    return np.asarray(predictions), np.asarray(probabilities)


def _resize_and_normalize_heatmap(heatmap, output_shape, sigma, power=1.0):
    heatmap = tf.convert_to_tensor(heatmap, dtype=tf.float32)
    heatmap = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis], output_shape, method="bilinear"
    )[0, ..., 0]
    heatmap = tf.pow(tf.maximum(heatmap, 0.0), power)
    heatmap = gaussian_filter(heatmap.numpy(), sigma=sigma)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(heatmap, 99))
    magnitude = float(np.max(np.abs(heatmap)))
    if (
        not np.isfinite(scale)
        or magnitude <= np.finfo(np.float64).tiny
        or float(np.ptp(heatmap)) <= magnitude * 1e-7
    ):
        return None
    return np.clip(heatmap / scale, 0, 1)


def _smoothed_input_saliency(gradients, image):
    heatmap = tf.reduce_mean(tf.abs(gradients), axis=-1)[0]
    height, width = int(image.shape[1]), int(image.shape[2])
    coarse_shape = (max(24, height // 8), max(24, width // 8))
    heatmap = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis], coarse_shape, method="bilinear"
    )[0, ..., 0]
    heatmap = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis], (height, width), method="bilinear"
    )[0, ..., 0]
    normalized = _resize_and_normalize_heatmap(
        heatmap, (height, width), sigma=5.0, power=0.7
    )
    if normalized is not None:
        return normalized, "Smoothed input-gradient saliency"

    gradient_times_input = tf.reduce_mean(
        tf.abs(gradients * tf.cast(image, gradients.dtype)), axis=-1
    )[0]
    normalized = _resize_and_normalize_heatmap(
        gradient_times_input, (height, width), sigma=5.0, power=0.7
    )
    if normalized is None:
        raise ValueError("ViT produced no spatially varying input-gradient saliency map.")
    return normalized, "Gradient×input saliency fallback"


def _nested_models(parent):
    """Yield nested Keras models from deepest to shallowest."""
    found = []
    for layer in getattr(parent, "layers", []):
        if isinstance(layer, tf.keras.Model):
            found.extend(_nested_models(layer))
            found.append(layer)
    return found


def _is_spatial_conv(layer):
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
    output_shape = getattr(getattr(layer, "output", None), "shape", ())
    return isinstance(layer, conv_types) and len(output_shape) == 4


def _standalone_backbone(model_name, input_shape, trained_candidates):
    builders = {
        "ResNet-50": tf.keras.applications.ResNet50,
        "DenseNet-121": tf.keras.applications.DenseNet121,
        "EfficientNet-B4": tf.keras.applications.EfficientNetB4,
        "ConvNeXt-Tiny": tf.keras.applications.ConvNeXtTiny,
    }
    try:
        builder = builders[model_name]
    except KeyError as error:
        raise ValueError(f"No standalone CNN builder for {model_name}.") from error

    backbone = builder(
        include_top=False,
        weights=None,
        input_shape=tuple(int(dimension) for dimension in input_shape[1:]),
    )
    expected_shapes = [tuple(weight.shape) for weight in backbone.get_weights()]
    matching_candidates = [
        candidate
        for candidate in trained_candidates
        if [tuple(weight.shape) for weight in candidate.get_weights()] == expected_shapes
    ]
    if matching_candidates:
        backbone.set_weights(matching_candidates[0].get_weights())
        return backbone, matching_candidates[0]
    return backbone, None


def _copy_backbone_weights_by_name(backbone, source_branch):
    source_layers = {}
    ordered_sources = []

    def visit(parent):
        for layer in getattr(parent, "layers", []):
            source_layers.setdefault(layer.name, []).append(layer)
            if layer.get_weights():
                ordered_sources.append(layer)
            if isinstance(layer, tf.keras.Model):
                visit(layer)

    visit(source_branch)
    missing = []
    used_sources = set()
    for target_layer in backbone.layers:
        target_weights = target_layer.get_weights()
        if not target_weights:
            continue
        target_shapes = [tuple(weight.shape) for weight in target_weights]
        source = next(
            (
                candidate
                for candidate in source_layers.get(target_layer.name, [])
                if id(candidate) not in used_sources
                if [tuple(weight.shape) for weight in candidate.get_weights()] == target_shapes
            ),
            None,
        )
        if source is None:
            source = next(
                (
                    candidate
                    for candidate in ordered_sources
                    if id(candidate) not in used_sources
                    and [tuple(weight.shape) for weight in candidate.get_weights()]
                    == target_shapes
                ),
                None,
            )
        if source is None:
            missing.append(target_layer.name)
        else:
            target_layer.set_weights(source.get_weights())
            used_sources.add(id(source))
    if missing:
        raise ValueError(
            f"Could not restore {len(missing)} weighted layers in the standalone "
            f"backbone: {missing[:8]}."
        )


def _input_gradient_or_occlusion_fallback(
    model, model_name, image, scaled_features, umap_features, class_index
):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    feature_tensor = tf.convert_to_tensor(scaled_features, dtype=tf.float32)
    umap_tensor = tf.convert_to_tensor(umap_features, dtype=tf.float32)
    logit_predictor = _logit_predictor(model)
    with tf.GradientTape() as input_tape:
        input_tape.watch(image_tensor)
        logits = _apply_logit_predictor(
            logit_predictor, [image_tensor, feature_tensor, umap_tensor]
        )
        score = logits[:, class_index]
    input_gradients = input_tape.gradient(score, image_tensor)
    gradient_spread = (
        float(tf.math.reduce_std(input_gradients).numpy())
        if input_gradients is not None
        else None
    )
    if input_gradients is not None:
        try:
            input_map, input_method = _smoothed_input_saliency(
                input_gradients, image_tensor
            )
            return input_map, f"{input_method} fallback"
        except ValueError:
            pass

    output_shape = (int(image.shape[1]), int(image.shape[2]))
    occlusion_map = _occlusion_saliency(
        model,
        image_tensor,
        feature_tensor,
        umap_tensor,
        class_index,
        grid_size=12,
        batch_size=12,
        logit_predictor=logit_predictor,
    )
    normalized = _resize_and_normalize_heatmap(
        occlusion_map, output_shape, sigma=2.5
    )
    if normalized is None:
        raise ValueError(
            f"{model_name} produced no spatially varying Grad-CAM, gradient, "
            f"or occlusion map (input-gradient std={gradient_spread}, "
            f"occlusion range={float(np.min(occlusion_map)):.6g}.."
            f"{float(np.max(occlusion_map)):.6g})."
        )
    return normalized, "Occlusion sensitivity fallback"


def _build_cnn_gradcam(model, model_name, image, scaled_features, umap_features, class_index):
    branch = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and layer.name.lower().endswith("_branch")
        ),
        None,
    )
    if branch is None:
        raise ValueError("Could not identify the CNN image branch in the saved model.")

    nested_backbones = [
        candidate
        for candidate in _nested_models(branch)
        if len(getattr(candidate.output, "shape", ())) == 4
        and any(_is_spatial_conv(layer) for layer in candidate.layers)
    ]
    grad_image = tf.convert_to_tensor(image, dtype=tf.float32)
    grad_features = tf.convert_to_tensor(scaled_features, dtype=tf.float32)
    grad_umap = tf.convert_to_tensor(umap_features, dtype=tf.float32)

    if nested_backbones:
        backbone, trained_backbone = _standalone_backbone(
            model_name=model_name,
            input_shape=image.shape,
            trained_candidates=nested_backbones,
        )
        if trained_backbone is None:
            _copy_backbone_weights_by_name(backbone, branch)
            last_augmentation_index = next(
                (
                    index
                    for index in range(len(branch.layers) - 1, -1, -1)
                    if isinstance(branch.layers[index], tf.keras.layers.RandomContrast)
                ),
                None,
            )
            if last_augmentation_index is None:
                raise ValueError(f"Could not locate preprocessing layers in {branch.name}.")
            branch_backbone_index = last_augmentation_index + 1
            branch_tail_index = next(
                (
                    index
                    for index, layer in enumerate(branch.layers[branch_backbone_index:], start=branch_backbone_index)
                    if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
                ),
                None,
            )
            if branch_tail_index is None:
                raise ValueError(f"Could not locate the pooling layer after {model_name}.")
        else:
            branch_backbone_index = next(
                index
                for index, layer in enumerate(branch.layers)
                if layer is trained_backbone
            )
            branch_tail_index = branch_backbone_index
        x = grad_image
        for layer in branch.layers[1:branch_backbone_index]:
            x = layer(x, training=False)
        conv_layers = [
            layer for layer in reversed(backbone.layers) if _is_spatial_conv(layer)
        ]

        def evaluate_layer(layer):
            probe = tf.keras.Model(backbone.input, [layer.output, backbone.output])
            with tf.GradientTape() as tape:
                activations, backbone_output = probe(x, training=False)
                branch_output = backbone_output
                for tail_layer in branch.layers[
                    branch_tail_index + (trained_backbone is not None) :
                ]:
                    branch_output = tail_layer(branch_output, training=False)
                logits = _apply_unified_fusion_head(
                    model, branch_output, grad_features, grad_umap, return_logits=True
                )
                target_score = logits[:, class_index]
            return activations, tape.gradient(target_score, activations)
    else:
        conv_layers = [
            layer for layer in reversed(branch.layers) if _is_spatial_conv(layer)
        ]
        if not conv_layers:
            raise ValueError(f"No spatial Conv2D feature map found in {branch.name}.")
        def evaluate_layer(layer):
            probe = tf.keras.Model(branch.input, [layer.output, branch.output])
            with tf.GradientTape() as tape:
                activations, branch_features = probe(grad_image, training=False)
                logits = _apply_unified_fusion_head(
                    model, branch_features, grad_features, grad_umap, return_logits=True
                )
                target_score = logits[:, class_index]
            return activations, tape.gradient(target_score, activations)

    output_shape = (int(image.shape[1]), int(image.shape[2]))
    for conv_layer in conv_layers:
        activations, gradients = evaluate_layer(conv_layer)
        if gradients is None:
            continue
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heatmap = tf.nn.relu(tf.reduce_sum(weights * activations, axis=-1))[0]
        normalized = _resize_and_normalize_heatmap(
            heatmap, output_shape, sigma=2.5
        )
        if normalized is not None:
            return normalized, f"Grad-CAM ({conv_layer.name})"

        # Some backbones have a flat positive CAM at the final stage. Check earlier
        # spatial features before switching to a different attribution method.
        gradient_activation = tf.reduce_sum(
            tf.abs(gradients) * tf.abs(activations), axis=-1
        )[0]
        normalized = _resize_and_normalize_heatmap(
            gradient_activation, output_shape, sigma=2.5
        )
        if normalized is not None:
            return normalized, f"Gradient×activation ({conv_layer.name})"

    return _input_gradient_or_occlusion_fallback(
        model, model_name, grad_image, grad_features, grad_umap, class_index
    )


def _occlusion_saliency(model, image, scaled_features, umap_features, class_index,
                        grid_size=12, batch_size=12, logit_predictor=None):
    """Estimate target-class sensitivity by replacing local image patches."""
    image_array = np.asarray(image, dtype=np.float32)[0]
    height, width = image_array.shape[:2]
    patch_height = max(1, height // grid_size)
    patch_width = max(1, width // grid_size)
    baseline = np.mean(image_array, axis=(0, 1), keepdims=True)
    if logit_predictor is None:
        logit_predictor = _logit_predictor(model)
    locations = [
        (row, column)
        for row in range(grid_size)
        for column in range(grid_size)
    ]
    scores = []

    for start in range(0, len(locations), batch_size):
        batch_locations = locations[start : start + batch_size]
        masked_batch = np.repeat(image_array[np.newaxis, ...], len(batch_locations), axis=0)
        for image_index, (row, column) in enumerate(batch_locations):
            top = min(row * patch_height, height - 1)
            left = min(column * patch_width, width - 1)
            bottom = min(top + patch_height, height)
            right = min(left + patch_width, width)
            masked_batch[image_index, top:bottom, left:right] = baseline

        repeated_features = tf.repeat(scaled_features, len(batch_locations), axis=0)
        repeated_umap = tf.repeat(umap_features, len(batch_locations), axis=0)
        masked_batch_tensor = tf.convert_to_tensor(masked_batch, dtype=tf.float32)
        batch_logits = _apply_logit_predictor(
            logit_predictor,
            [masked_batch_tensor, repeated_features, repeated_umap],
        )
        scores.extend(np.asarray(batch_logits)[:, class_index].tolist())

    original_logits = _apply_logit_predictor(
        logit_predictor, [image, scaled_features, umap_features]
    )
    original_score = float(original_logits[0, class_index])
    sensitivity = np.abs(original_score - np.asarray(scores, dtype=np.float32))
    return sensitivity.reshape(grid_size, grid_size)


def _logit_predictor(model):
    output_layer = next(
        layer
        for layer in reversed(model.layers)
        if isinstance(layer, tf.keras.layers.Dense)
    )
    feature_model = tf.keras.Model(model.inputs, output_layer.input)
    return feature_model, output_layer


def _has_spatial_input_signal(model, image, scaled_features, umap_features, class_index):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    predictor = _logit_predictor(model)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        logits = _apply_logit_predictor(
            predictor,
            [
                image_tensor,
                tf.convert_to_tensor(scaled_features, dtype=tf.float32),
                tf.convert_to_tensor(umap_features, dtype=tf.float32),
            ],
        )
        target_logit = logits[:, class_index]
    gradients = tape.gradient(target_logit, image_tensor)
    if gradients is None:
        return False
    try:
        _smoothed_input_saliency(gradients, image_tensor)
        return True
    except ValueError:
        return False


def _apply_logit_predictor(predictor, inputs):
    feature_model, output_layer = predictor
    fused = feature_model(inputs, training=False)
    return _dense_logits(fused, output_layer)


def _dense_logits(fused, output_layer):
    logits = tf.linalg.matmul(fused, output_layer.kernel)
    if output_layer.use_bias:
        logits = tf.nn.bias_add(logits, output_layer.bias)
    return logits


def _apply_unified_fusion_head(
    model, image_features, scaled_features, umap_features, return_logits=False
):
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    norm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
    concat_layer = next(
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Concatenate)
    )
    if len(dense_layers) != 5 or len(norm_layers) != 3 or len(dropout_layers) != 4:
        raise ValueError("Saved model does not match the expected Unified fusion head.")

    image_features = dropout_layers[0](
        norm_layers[0](dense_layers[0](image_features), training=False), training=False
    )
    handcrafted = dropout_layers[1](
        norm_layers[1](dense_layers[1](scaled_features), training=False), training=False
    )
    embedding = dropout_layers[2](
        norm_layers[2](dense_layers[2](umap_features), training=False), training=False
    )
    fused = concat_layer([image_features, handcrafted, embedding])
    fused = dropout_layers[3](dense_layers[3](fused), training=False)
    output_layer = dense_layers[4]
    if return_logits:
        return _dense_logits(fused, output_layer)
    return output_layer(fused)


def _input_gradient_saliency(model, image, scaled_features, umap_features, class_index):
    """Use input-gradient saliency for ViT's pooled TF-Hub features."""
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        probabilities = model(
            [
                image_tensor,
                tf.convert_to_tensor(scaled_features, dtype=tf.float32),
                tf.convert_to_tensor(umap_features, dtype=tf.float32),
            ],
            training=False,
        )
        score = probabilities[:, class_index]
    gradients = tape.gradient(score, image_tensor)
    if gradients is None:
        raise ValueError("Input gradients are unavailable for the ViT model.")
    return _smoothed_input_saliency(gradients, image)


def create_heatmap(model, model_name, image, scaled_features, umap_features, class_index):
    if model_name == "ViT-B-16":
        return _input_gradient_saliency(
            model, image, scaled_features, umap_features, class_index
        )
    return _build_cnn_gradcam(
        model, model_name, image, scaled_features, umap_features, class_index
    )


def _overlay(image, heatmap):
    height, width = image.shape[:2]
    resized = tf.image.resize(
        heatmap[..., np.newaxis], (height, width), method="bilinear"
    ).numpy()[..., 0]
    normalized = np.clip(resized, 0, 1)
    color_map = plt.get_cmap("jet")(normalized)[..., :3]
    original = np.clip(image / 255.0, 0, 1)
    alpha = 0.68 * np.power(normalized, 0.75)
    return np.clip(original * (1 - alpha[..., None]) + color_map * alpha[..., None], 0, 1)


def _center_crop_zoom(image, zoom):
    """Crop only the displayed view, preserving full-frame model inference."""
    if zoom <= 1:
        return image
    height, width = image.shape[:2]
    crop_height = max(1, int(round(height / zoom)))
    crop_width = max(1, int(round(width / zoom)))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return image[top : top + crop_height, left : left + crop_width]


def _legacy_endoscopy_crop(image_path, resized_image):
    """Apply the repository's older right-shifted crop for figure display."""
    raw_bgr = cv2.imread(image_path)
    if raw_bgr is None:
        raise ValueError(f"Could not read selected image for figure: {image_path}")
    raw_image = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    raw_height, raw_width = raw_image.shape[:2]
    if raw_height > 450 and raw_width > 550:
        y0, y1, x0, x1 = 30, min(430, raw_height), 200, min(550, raw_width)
    else:
        y0, y1, x0, x1 = 0, raw_height, 0, raw_width

    model_height, model_width = resized_image.shape[:2]
    bounds = (
        int(round(y0 * model_height / raw_height)),
        int(round(y1 * model_height / raw_height)),
        int(round(x0 * model_width / raw_width)),
        int(round(x1 * model_width / raw_width)),
    )
    cropped = raw_image[y0:y1, x0:x1]
    display_image = cv2.resize(cropped, (model_width, model_height), interpolation=cv2.INTER_AREA)
    return display_image, bounds


def _crop_heatmap_to_bounds(heatmap, bounds, output_shape):
    top, bottom, left, right = bounds
    cropped = heatmap[top:bottom, left:right]
    return tf.image.resize(
        cropped[..., np.newaxis], output_shape, method="bilinear"
    ).numpy()[..., 0]


def _normalize_display_crop(heatmap):
    """Renormalize surviving attribution inside the display crop, not removed margins."""
    heatmap = np.nan_to_num(np.asarray(heatmap), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(heatmap, (1, 99))
    if high - low <= 1e-12:
        low, high = float(np.min(heatmap)), float(np.max(heatmap))
    if high - low <= 1e-12:
        return None
    return np.clip((heatmap - low) / (high - low), 0, 1)


def _make_figure(
    output_path,
    display_images,
    labels,
    predictions,
    probabilities,
    selected,
    adjudications,
    heatmaps,
    figure_zoom,
    case_order,
):
    column_titles = [
        "Case",
        "Reference",
        "ResNet-50",
        "DenseNet-121",
        "EfficientNet-B4",
        "ConvNeXt-Tiny",
        "ViT-B/16",
        "Ensemble adjudication",
        "Illustrative output",
    ]
    fig = plt.figure(figsize=(21, 12), facecolor="white")
    grid = fig.add_gridspec(
        len(case_order) + 1,
        len(column_titles),
        height_ratios=[0.32] + [1] * len(case_order),
        width_ratios=[0.9, 0.85, 1, 1, 1, 1, 1, 1.35, 1.05],
        hspace=0.10,
        wspace=0.08,
        left=0.025,
        right=0.985,
        top=0.90,
        bottom=0.09,
    )

    for column, title in enumerate(column_titles):
        axis = fig.add_subplot(grid[0, column])
        axis.axis("off")
        axis.text(
            0.5,
            0.25,
            title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#26384a",
            wrap=True,
        )

    for row, case_name in enumerate(case_order, start=1):
        sample_index = selected[case_name]
        true_label = int(labels[sample_index])
        result = adjudications[sample_index]
        row_title = fig.add_subplot(grid[row, 0])
        row_title.axis("off")
        row_title.text(
            0.02,
            0.84,
            case_name,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#26384a",
        )
        row_title.text(
            0.02,
            0.42,
            result["rule"],
            ha="left",
            va="center",
            fontsize=7,
            color="#526579",
            wrap=True,
        )

        reference_axis = fig.add_subplot(grid[row, 1])
        reference_image = _center_crop_zoom(display_images[sample_index], figure_zoom)
        reference_axis.imshow(np.clip(reference_image / 255.0, 0, 1))
        reference_axis.set_title(CLASS_NAMES[true_label], fontsize=8, pad=3)
        reference_axis.axis("off")

        for model_index, model_name in enumerate(MODEL_NAMES):
            axis = fig.add_subplot(grid[row, model_index + 2])
            predicted_label = int(predictions[model_index, sample_index])
            axis.imshow(heatmaps[model_name][sample_index])
            matched = predicted_label == true_label
            marker = "OK" if matched else "ERR"
            color = "#17824b" if matched else "#c7342e"
            confidence = float(probabilities[model_index, sample_index, predicted_label])
            axis.set_title(
                f"{CLASS_NAMES[predicted_label]} - {confidence:.2f}",
                fontsize=8,
                color=color,
                pad=3,
            )
            axis.text(
                0.97,
                0.04,
                marker,
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color=color,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            axis.axis("off")

        adjudication_axis = fig.add_subplot(grid[row, 7])
        adjudication_axis.axis("off")
        adjudication_axis.text(
            0.5,
            0.72,
            f'{max(result["vote_counts"].values())}/5 votes',
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#26384a",
        )
        adjudication_axis.text(
            0.5,
            0.42,
            result["rule"],
            ha="center",
            va="center",
            fontsize=8,
            color="#526579",
            wrap=True,
        )
        adjudication_axis.text(
            0.5,
            0.12,
            " ".join(
                f'{CLASS_NAMES[index]}:{result["vote_counts"][CLASS_NAMES[index]]}'
                for index in range(len(CLASS_NAMES))
            ),
            ha="center",
            va="center",
            fontsize=7,
            color="#526579",
        )

        output_axis = fig.add_subplot(grid[row, 8])
        correct = result["label"] == true_label
        output_axis.axis("off")
        output_axis.text(
            0.5,
            0.62,
            CLASS_NAMES[result["label"]],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=CLASS_COLORS[result["label"]],
        )
        output_axis.text(
            0.5,
            0.25,
            "Matches reference" if correct else "Differs from reference",
            ha="center",
            va="center",
            fontsize=7,
            color="#17824b" if correct else "#c7342e",
            wrap=True,
        )

    method_note = (
        "CNN backbones use Grad-CAM; degenerate maps use input-gradient or occlusion "
        "saliency fallbacks. "
        "ViT-B/16 uses input-gradient saliency because the saved TF-Hub model exposes "
        "pooled features. Illustrative model output only."
    )
    fig.suptitle(
        "Unified Five-Backbone Ensemble Adjudication",
        fontsize=16,
        fontweight="bold",
        y=0.965,
        color="#182b3d",
    )
    fig.text(
        0.03,
        0.035,
        method_note,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#526579",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Generate mixed-dataset ensemble adjudication examples and saliency maps."
    )
    parser.add_argument(
        "--base-dir",
        default="/home/D13K48009/raid/Clara/new_drive",
        help="DGX directory containing Dataset/ and Dataset+Code/.",
    )
    parser.add_argument(
        "--models-dir",
        default="/home/D13K48009/raid/Clara/colonomind-train/Result/Intra_Unified",
        help="Folder containing the five *_Experiment directories.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "Result", "Unified_Adjudication_Figure.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON path for selected sample indices, labels, and votes.",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--sample-list",
        default=None,
        help="Optional server-local text file of image paths relative to Dataset+Code/.",
    )
    parser.add_argument(
        "--figure-zoom",
        type=float,
        default=1.0,
        help="Optional extra center crop after the legacy display crop; does not affect inference.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.figure_zoom < 1.0 or args.figure_zoom > 2.0:
        parser.error("--figure-zoom must be between 1.0 and 2.0.")
    if not os.path.isdir(args.models_dir):
        raise FileNotFoundError(f"Unified model directory does not exist: {args.models_dir}")
    required_artifacts = [
        path
        for model_name in MODEL_NAMES
        for path in _model_artifact_paths(args.models_dir, model_name).values()
        if not os.path.isfile(path)
    ]
    if required_artifacts:
        raise FileNotFoundError(
            "Missing Unified model artifacts:\n  " + "\n  ".join(required_artifacts)
        )

    if args.sample_list:
        print(f"Loading explicit figure samples from {args.sample_list}...")
        images, features, labels, image_paths = load_explicit_samples(
            args.base_dir, args.sample_list
        )
        case_order = tuple(f"Sample {index + 1}" for index in range(len(labels)))
    else:
        print("Loading labeled examples from the mixed dataset in new_drive...")
        images, features, labels, image_paths = load_mixed_data(args.base_dir, args.cache_dir)
        case_order = CASE_ORDER
    if len(labels) == 0:
        raise ValueError("Mixed dataset is empty; verify its MES0-MES3 class folders.")
    source_label = "curated" if args.sample_list else "mixed-dataset"
    print(f"Loaded {len(labels)} {source_label} images.")

    predictions, probabilities = predict_models(
        args.models_dir, images, features, args.batch_size
    )
    if args.sample_list:
        selected = {case_name: index for index, case_name in enumerate(case_order)}
        adjudications = [
            adjudicate(
                [model_predictions[index] for model_predictions in predictions],
                [model_probabilities[index] for model_probabilities in probabilities],
            )
            for index in range(len(labels))
        ]
    else:
        selected, adjudications = select_figure_cases(labels, predictions, probabilities)

    dense_map_cache = {}
    sample_replacements = {}
    dense_name = "DenseNet-121"
    dense_paths = _model_artifact_paths(args.models_dir, dense_name)
    dense_model = _load_hybrid_model(dense_paths["model"])
    dense_scaler = joblib.load(dense_paths["scaler"])
    dense_umap_model = joblib.load(dense_paths["umap"])
    dense_input_shape = dense_model.inputs[0].shape
    dense_height, dense_width = int(dense_input_shape[1]), int(dense_input_shape[2])
    candidate_sets = (
        _rank_figure_candidates(labels, adjudications, probabilities)
        if not args.sample_list
        else None
    )
    for case_name in case_order:
        sample_index = selected[case_name]
        if args.sample_list:
            original_path = image_paths[sample_index]
            candidate_specs = [(None, None)] + [
                (path, None)
                for path in _same_grade_image_candidates(original_path)
                if os.path.abspath(path) != os.path.abspath(original_path)
            ][:99]
        else:
            candidate_specs = [(None, index) for index in candidate_sets[case_name][:100]]

        last_error = None
        for candidate_position, (candidate_path, candidate_index) in enumerate(candidate_specs):
            if args.sample_list and candidate_position:
                print(
                    f"Checking same-grade alternative {candidate_position}/"
                    f"{len(candidate_specs) - 1} for {case_name}: "
                    f"{os.path.basename(candidate_path)}"
                )
            row_index = sample_index if candidate_index is None else candidate_index
            if candidate_path is None:
                candidate_image = images[row_index]
                candidate_features = features[row_index : row_index + 1]
                candidate_label = int(labels[row_index])
                candidate_source = image_paths[row_index]
            else:
                candidate_class = os.path.basename(os.path.dirname(original_path))
                processed = process_single_image(candidate_path, candidate_class)
                if processed is None:
                    continue
                candidate_image, feature, label, candidate_source = processed
                candidate_features = np.asarray(feature, dtype=np.float32)[np.newaxis, ...]
                candidate_label = CLASS_NAMES.index(label)

            sample_image = tf.image.resize(
                candidate_image, (dense_height, dense_width), method="bilinear"
            )[tf.newaxis, ...]
            scaled = dense_scaler.transform(candidate_features)
            embedded = dense_umap_model.transform(scaled)
            if candidate_position == 0 and candidate_index is None:
                class_index = int(predictions[MODEL_NAMES.index(dense_name), row_index])
            elif candidate_index is not None:
                class_index = int(predictions[MODEL_NAMES.index(dense_name), row_index])
            else:
                dense_probabilities = dense_model(
                    [sample_image, scaled, embedded], training=False
                )
                class_index = int(tf.argmax(dense_probabilities[0]).numpy())
            if candidate_position and not _has_spatial_input_signal(
                dense_model, sample_image, scaled, embedded, class_index
            ):
                continue

            try:
                heatmap, method = create_heatmap(
                    dense_model, dense_name, sample_image, scaled, embedded, class_index
                )
                display_image, bounds = _legacy_endoscopy_crop(
                    candidate_source, candidate_image
                )
                aligned = _crop_heatmap_to_bounds(
                    heatmap, bounds, display_image.shape[:2]
                )
                if _normalize_display_crop(
                    _center_crop_zoom(aligned, args.figure_zoom)
                ) is None:
                    raise ValueError(
                        "DenseNet attribution disappears in the display crop."
                    )
            except ValueError as error:
                last_error = error
                continue

            if candidate_path is not None:
                images[sample_index] = candidate_image
                features[sample_index] = candidate_features[0]
                labels[sample_index] = candidate_label
                image_paths[sample_index] = candidate_source
                sample_replacements[case_name] = {
                    "requested": original_path,
                    "selected": candidate_source,
                    "reference": CLASS_NAMES[candidate_label],
                }
                print(
                    f"Replaced {case_name} with a same-grade image that supports "
                    f"DenseNet attribution: {candidate_source}"
                )
            else:
                selected[case_name] = row_index
            dense_map_cache[case_name] = (heatmap, method, class_index)
            print(
                f"Selected DenseNet-attributable {case_name} case "
                f"{sample_index + 1} (MES{labels[sample_index]})."
            )
            break
        else:
            raise ValueError(
                f"No DenseNet-attributable image found for {case_name} among "
                f"{len(candidate_specs)} same-grade/eligible candidates. "
                f"Last error: {last_error}"
            )

    if args.sample_list and sample_replacements:
        predictions, probabilities = predict_models(
            args.models_dir, images, features, args.batch_size
        )
        adjudications = [
            adjudicate(
                [model_predictions[index] for model_predictions in predictions],
                [model_probabilities[index] for model_probabilities in probabilities],
            )
            for index in range(len(labels))
        ]
    del dense_model, dense_scaler, dense_umap_model
    tf.keras.backend.clear_session()

    heatmaps = {name: {} for name in MODEL_NAMES}
    map_methods = {name: {} for name in MODEL_NAMES}
    selected_indices = [selected[name] for name in case_order]
    selected_features = features[selected_indices]
    display_images = {}
    crop_bounds = {}
    for sample_index in selected_indices:
        display_images[sample_index], crop_bounds[sample_index] = _legacy_endoscopy_crop(
            image_paths[sample_index], images[sample_index]
        )

    for model_index, model_name in enumerate(MODEL_NAMES):
        paths = _model_artifact_paths(args.models_dir, model_name)
        model = _load_hybrid_model(paths["model"])
        scaler = joblib.load(paths["scaler"])
        umap_model = joblib.load(paths["umap"])
        scaled_features = scaler.transform(selected_features)
        umap_features = umap_model.transform(scaled_features)
        for case_row, case_name in enumerate(case_order):
            sample_index = selected[case_name]
            image_shape = model.inputs[0].shape
            image_height, image_width = int(image_shape[1]), int(image_shape[2])
            sample_image = tf.image.resize(
                images[sample_index], (image_height, image_width), method="bilinear"
            )[tf.newaxis, ...]
            if (
                model_name == "DenseNet-121"
                and case_name in dense_map_cache
                and dense_map_cache[case_name][2]
                == int(predictions[model_index, sample_index])
            ):
                heatmap, method, _ = dense_map_cache[case_name]
            else:
                heatmap, method = create_heatmap(
                    model,
                    model_name,
                    sample_image,
                    scaled_features[case_row : case_row + 1],
                    umap_features[case_row : case_row + 1],
                    int(predictions[model_index, sample_index]),
                )
            display_image = _center_crop_zoom(
                display_images[sample_index], args.figure_zoom
            )
            aligned_heatmap = _crop_heatmap_to_bounds(
                heatmap, crop_bounds[sample_index], display_images[sample_index].shape[:2]
            )
            display_heatmap = _center_crop_zoom(aligned_heatmap, args.figure_zoom)
            display_heatmap = _normalize_display_crop(display_heatmap)
            if display_heatmap is None:
                raise ValueError(
                    f"No spatial attribution remains inside the display crop for "
                    f"{case_name} / {model_name}; refusing to save a blank heatmap panel."
                )
            heatmaps[model_name][sample_index] = _overlay(
                display_image, display_heatmap
            )
            map_methods[model_name][case_name] = method
            print(f"Prepared {case_name} with {model_name}.")
        del model, scaler, umap_model, scaled_features, umap_features
        tf.keras.backend.clear_session()

    _make_figure(
        args.output,
        display_images,
        labels,
        predictions,
        probabilities,
        selected,
        adjudications,
        heatmaps,
        args.figure_zoom,
        case_order,
    )
    print(f"Figure saved to: {args.output}")

    metadata_path = args.metadata or os.path.splitext(args.output)[0] + ".json"
    metadata = {
        "scenario": (
            "Curated illustrative examples from MES classification_20250724"
            if args.sample_list
            else "Unified models evaluated on the mixed dataset"
        ),
        "models": list(MODEL_NAMES),
        "sample_source": (
            "Explicit server-local sample list relative to Dataset+Code/"
            if args.sample_list
            else os.path.join(args.base_dir, "Dataset+Code", "MES Mixed Data")
        ),
        "figure_zoom": args.figure_zoom,
        "display_crop": "legacy crop [30:430, 200:550] when raw image height>450 and width>550",
        "display_crop_applies_to": (
            "figure and aligned heatmaps only; inference inputs are unchanged. "
            "Each cropped heatmap is contrast-normalized within the displayed colon region."
        ),
        "same_grade_sample_replacements": sample_replacements,
        "selected_cases": {
            case_name: {
                "dataset_index": int(index),
                "image_path": image_paths[index],
                "reference": CLASS_NAMES[int(labels[index])],
                "model_predictions": {
                    model_name: CLASS_NAMES[int(predictions[model_index, index])]
                    for model_index, model_name in enumerate(MODEL_NAMES)
                },
                "model_confidences": {
                    model_name: float(probabilities[model_index, index].max())
                    for model_index, model_name in enumerate(MODEL_NAMES)
                },
                "adjudication": CLASS_NAMES[adjudications[index]["label"]],
                "rule": adjudications[index]["rule"],
                "vote_counts": adjudications[index]["vote_counts"],
            }
            for case_name, index in selected.items()
        },
        "visualization_methods": map_methods,
        "note": (
            "Illustrative examples only; they may overlap training data and are not an "
            "independent test set. CNN backbones use Grad-CAM with input-gradient or "
            "occlusion fallback "
            "for degenerate maps; ViT-B/16 uses input-gradient saliency."
            if args.sample_list
            else "CNN backbones use Grad-CAM with input-gradient or occlusion fallback "
            "ViT-B/16 uses input-gradient saliency."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"Figure metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
