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
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register custom model layers and preprocessing functions before loading models.
from src import dgx_models  # noqa: E402,F401
from src.dgx_dataloader import load_all_images  # noqa: E402
from src.ensemble_adjudication import (  # noqa: E402
    CLASS_NAMES,
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
            batch_probabilities = model(
                [
                    images[start:end],
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


def _smoothed_input_saliency(gradients, image):
    heatmap = tf.reduce_mean(tf.abs(gradients), axis=-1)[0]
    height, width = int(image.shape[1]), int(image.shape[2])
    heatmap = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis],
        (max(24, height // 8), max(24, width // 8)),
        method="bilinear",
    )[0, ..., 0]
    heatmap = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis], (height, width), method="bilinear"
    )[0, ..., 0]
    heatmap = tf.pow(heatmap, 0.7)
    heatmap = gaussian_filter(heatmap.numpy(), sigma=5.0)
    scale = np.percentile(heatmap, 99)
    return np.clip(heatmap / max(scale, 1e-8), 0, 1)


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


def _build_cnn_gradcam(model, image, scaled_features, umap_features, class_index):
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
        backbone = nested_backbones[0]
        conv_layer = next(
            layer
            for layer in reversed(backbone.layers)
            if _is_spatial_conv(layer)
        )
        backbone_probe = tf.keras.Model(
            backbone.input, [conv_layer.output, backbone.output]
        )
        branch_backbone_index = next(
            index for index, layer in enumerate(branch.layers) if layer is backbone
        )
        x = grad_image
        for layer in branch.layers[1:branch_backbone_index]:
            x = layer(x, training=False)
        with tf.GradientTape() as tape:
            activations, x = backbone_probe(x, training=False)
            for layer in branch.layers[branch_backbone_index + 1 :]:
                x = layer(x, training=False)
            class_probabilities = _apply_unified_fusion_head(
                model, x, grad_features, grad_umap
            )
            score = class_probabilities[:, class_index]
    else:
        conv_layer = next(
            (
                layer
                for layer in reversed(branch.layers)
                if _is_spatial_conv(layer)
            ),
            None,
        )
        if conv_layer is None:
            raise ValueError(f"No spatial Conv2D feature map found in {branch.name}.")
        branch_probe = tf.keras.Model(
            branch.input, [conv_layer.output, branch.output]
        )
        with tf.GradientTape() as tape:
            activations, branch_features = branch_probe(grad_image, training=False)
            class_probabilities = _apply_unified_fusion_head(
                model, branch_features, grad_features, grad_umap
            )
            score = class_probabilities[:, class_index]

    gradients = tape.gradient(score, activations)
    if gradients is None:
        raise ValueError(
            f"Grad-CAM gradient is disconnected for the {branch.name} feature map."
        )
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(weights * activations, axis=-1))[0]
    heatmap = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis],
        (int(image.shape[1]), int(image.shape[2])),
        method="bilinear",
    )[0, ..., 0]
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, maximum)
    heatmap = gaussian_filter(heatmap.numpy(), sigma=2.5)
    scale = np.percentile(heatmap, 99)
    return np.clip(heatmap / max(scale, 1e-8), 0, 1), "Grad-CAM"


def _apply_unified_fusion_head(model, image_features, scaled_features, umap_features):
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
    return dense_layers[4](fused)


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
    return _smoothed_input_saliency(gradients, image), "Smoothed input-gradient saliency"


def create_heatmap(model, model_name, image, scaled_features, umap_features, class_index):
    if model_name == "ViT-B-16":
        return _input_gradient_saliency(
            model, image, scaled_features, umap_features, class_index
        )
    return _build_cnn_gradcam(
        model, image, scaled_features, umap_features, class_index
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


def _make_figure(
    output_path,
    images,
    labels,
    predictions,
    probabilities,
    selected,
    adjudications,
    heatmaps,
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
        len(CASE_ORDER) + 1,
        len(column_titles),
        height_ratios=[0.32] + [1] * len(CASE_ORDER),
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

    for row, case_name in enumerate(CASE_ORDER, start=1):
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
        reference_axis.imshow(np.clip(images[sample_index] / 255.0, 0, 1))
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
        "CNN backbones use Grad-CAM; ViT-B/16 uses input-gradient saliency because "
        "the saved TF-Hub model exposes pooled features. Illustrative model output only."
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
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
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

    print("Loading labeled examples from the mixed dataset in new_drive...")
    images, features, labels, image_paths = load_mixed_data(args.base_dir, args.cache_dir)
    if len(labels) == 0:
        raise ValueError("Mixed dataset is empty; verify its MES0-MES3 class folders.")
    print(f"Loaded {len(labels)} mixed-dataset images.")

    predictions, probabilities = predict_models(
        args.models_dir, images, features, args.batch_size
    )
    selected, adjudications = select_figure_cases(labels, predictions, probabilities)

    heatmaps = {name: {} for name in MODEL_NAMES}
    map_methods = {}
    selected_indices = [selected[name] for name in CASE_ORDER]
    selected_features = features[selected_indices]
    for model_index, model_name in enumerate(MODEL_NAMES):
        paths = _model_artifact_paths(args.models_dir, model_name)
        model = _load_hybrid_model(paths["model"])
        scaler = joblib.load(paths["scaler"])
        umap_model = joblib.load(paths["umap"])
        scaled_features = scaler.transform(selected_features)
        umap_features = umap_model.transform(scaled_features)
        for case_row, case_name in enumerate(CASE_ORDER):
            sample_index = selected[case_name]
            image_shape = model.inputs[0].shape
            image_height, image_width = int(image_shape[1]), int(image_shape[2])
            sample_image = tf.image.resize(
                images[sample_index], (image_height, image_width), method="bilinear"
            )[tf.newaxis, ...]
            heatmap, method = create_heatmap(
                model,
                model_name,
                sample_image,
                scaled_features[case_row : case_row + 1],
                umap_features[case_row : case_row + 1],
                int(predictions[model_index, sample_index]),
            )
            heatmaps[model_name][sample_index] = _overlay(
                sample_image[0].numpy(), heatmap
            )
            map_methods[model_name] = method
            print(f"Prepared {case_name} with {model_name}.")
        del model, scaler, umap_model, scaled_features, umap_features
        tf.keras.backend.clear_session()

    _make_figure(
        args.output,
        images,
        labels,
        predictions,
        probabilities,
        selected,
        adjudications,
        heatmaps,
    )
    print(f"Figure saved to: {args.output}")

    metadata_path = args.metadata or os.path.splitext(args.output)[0] + ".json"
    metadata = {
        "scenario": "Unified models evaluated on the mixed dataset",
        "models": list(MODEL_NAMES),
        "sample_source": os.path.join(args.base_dir, "Dataset+Code", "MES Mixed Data"),
        "selected_cases": {
            case_name: {
                "test_index": int(index),
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
        "note": "CNN backbones use Grad-CAM; ViT-B/16 uses input-gradient saliency.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"Figure metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
