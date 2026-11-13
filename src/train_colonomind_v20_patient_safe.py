#!/usr/bin/env python3
"""Train the V20 Mod-SE(2) model on the patient-aware manifest."""

import argparse
import json
import os
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


import numpy as np
import pandas as pd
import cv2
import joblib
import tensorflow as tf

gpu_devices = tf.config.list_physical_devices("GPU")
if not gpu_devices:
    raise RuntimeError(
        "TensorFlow cannot see a GPU; refusing to run V20 training on CPU. "
        "Check nvidia-smi, CUDA_VISIBLE_DEVICES, and the TensorFlow GPU install."
    )
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"TensorFlow {tf.__version__} using GPU devices: {gpu_devices}", flush=True)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Multiply,
    Reshape,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence, to_categorical

NUM_CLASSES = 4
CLASS_NAMES = ["MES0", "MES1", "MES2", "MES3"]


def se_block(x, ratio=8):
    channels = x.shape[-1]
    weights = GlobalAveragePooling2D()(x)
    weights = Reshape((1, 1, channels))(weights)
    weights = Dense(max(1, channels // ratio), activation="relu", use_bias=False)(weights)
    weights = Dense(channels, activation="sigmoid", use_bias=False)(weights)
    return Multiply()([x, weights])


def build_modse2_hybrid(img_shape=(256, 256, 3), feat_dim=28, dropout_rate=0.15):
    image_input = Input(shape=img_shape, name="image_input_se2")

    x = Conv2D(32, 3, padding="same")(image_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = se_block(x)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.05)(x)

    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = se_block(x)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.05)(x)

    x3 = Conv2D(128, 3, padding="same")(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    x3 = se_block(x3)
    gap_x3 = GlobalAveragePooling2D()(x3)
    x = Dropout(0.10)(MaxPooling2D(2)(x3))

    x4 = Conv2D(256, 3, padding="same")(x)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)
    x4 = se_block(x4)
    gap_x4 = GlobalAveragePooling2D()(x4)
    x = Dropout(0.15)(MaxPooling2D(2)(x4))

    x5 = Conv2D(512, 3, padding="same")(x)
    x5 = BatchNormalization()(x5)
    x5 = Activation("relu")(x5)
    x5 = se_block(x5)
    gap_x5 = GlobalAveragePooling2D()(x5)

    visual = Concatenate(name="Visual_MultiScale")([gap_x3, gap_x4, gap_x5])
    visual = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(visual)
    visual = BatchNormalization()(visual)
    visual = Dropout(dropout_rate, name="feat_cnn_proj")(visual)

    feature_input = Input(shape=(feat_dim,), name="feat_input")
    hand = BatchNormalization()(feature_input)
    hand = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(hand)
    hand = BatchNormalization()(hand)
    hand = Dropout(dropout_rate, name="feat_hand_proj")(hand)

    manifold_input = Input(shape=(2,), name="umap_feat_input")
    manifold = Dense(32, activation="relu", kernel_regularizer=l2(1e-4))(manifold_input)
    manifold = BatchNormalization()(manifold)
    manifold = Dropout(0.2, name="feat_umap_proj")(manifold)

    fused = Concatenate(name="Fusion")([visual, hand, manifold])
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4), name="deep_ext_1")(fused)
    x = BatchNormalization()(x)
    x = Dropout(0.20)(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4), name="deep_ext_2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4), name="deep_ext_3")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.10)(x)
    output = Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    return Model([image_input, feature_input, manifold_input], output, name="ColonoMind_ModSE2_V20")


def ordinal_soft_labels(labels, sigma=0.55):
    grades = np.arange(NUM_CLASSES)[None, :]
    weights = np.exp(-((grades - labels[:, None]) ** 2) / (2 * sigma ** 2))
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def make_cdw_ce_loss(alpha=2.0, mix=0.5):
    distances = tf.constant(
        [[abs(i - j) ** alpha for j in range(NUM_CLASSES)] for i in range(NUM_CLASSES)],
        dtype=tf.float32,
    )

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        true_idx = tf.argmax(y_true, axis=-1)
        weights = tf.gather(distances, true_idx)
        cdw = -tf.reduce_sum(weights * tf.math.log(1.0 - y_pred), axis=-1) / 3.0
        return tf.reduce_mean(ce + mix * cdw)

    return loss


def augment_image(image):
    rows, cols = image.shape[:2]
    angle = np.random.uniform(-30, 30)
    tx = np.random.uniform(-0.12, 0.12) * cols
    ty = np.random.uniform(-0.12, 0.12) * rows
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    result = cv2.warpAffine(image, matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() > 0.5:
        result = cv2.flip(result, 1)
    if np.random.rand() > 0.5:
        result = cv2.flip(result, 0)
    if np.random.rand() > 0.5:
        gamma = np.random.uniform(0.85, 1.15)
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        result = cv2.LUT(result, table)
    return result


class HybridSequence(Sequence):
    def __init__(self, images, features, manifold, targets, indices, batch_size, augment=False, shuffle=False):
        super().__init__()
        self.images = images
        self.features = features
        self.manifold = manifold
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_index):
        batch_ids = self.indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        images = self.images[batch_ids].copy()
        if self.augment:
            for i in range(len(images)):
                images[i] = augment_image(images[i].astype(np.uint8)).astype(np.float32)
        return (
            (images / 255.0, self.features[batch_ids], self.manifold[batch_ids]),
            self.targets[batch_ids],
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.time()
        print(f"Epoch {epoch + 1}/{self.params['epochs']} started", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"Epoch {epoch + 1}/{self.params['epochs']} complete in {time.time() - self.started:.1f}s | "
            f"loss={logs.get('loss', float('nan')):.4f} | "
            f"accuracy={logs.get('accuracy', 0.0) * 100:.2f}% | "
            f"val_loss={logs.get('val_loss', float('nan')):.4f} | "
            f"val_accuracy={logs.get('val_accuracy', 0.0) * 100:.2f}%",
            flush=True,
        )

    def on_train_batch_end(self, batch, logs=None):
        steps = self.params.get("steps", 1)
        if (batch + 1) % 25 == 0 or batch + 1 == steps:
            logs = logs or {}
            print(
                f"  batch {batch + 1}/{steps} ({(batch + 1) / steps * 100:.1f}%) "
                f"loss={logs.get('loss', float('nan')):.4f}",
                flush=True,
            )


def metric_bundle(y_true, y_pred):
    return {
        "exact_accuracy": float(accuracy_score(y_true, y_pred)),
        "within_one_grade": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mes0_1_vs_mes2_3": float(accuracy_score(y_true >= 2, y_pred >= 2)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(4)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Patient-aware V20 Mod-SE(2) training")
    parser.add_argument("--npz", default="../Result/ColonoMind_v6/dataset_cache_v6.npz")
    parser.add_argument("--manifest", default="Result/ColonoMind_V20_DataAudit/v20_patient_safe_splits.csv")
    parser.add_argument("--output-dir", default="Result/ColonoMind_V20_PatientSafe")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=25,
        help="Validation epochs without improvement allowed before stopping.",
    )
    parser.add_argument(
        "--minimum-epochs", type=int, default=40,
        help="Do not let early stopping monitor validation before this epoch.",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable image-only augmentation for a separate ablation run.",
    )
    args = parser.parse_args()
    if args.minimum_epochs < 0 or args.minimum_epochs >= args.epochs:
        parser.error("--minimum-epochs must be >= 0 and less than --epochs")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be at least 1")

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading V6 cache and V20 split manifest...", flush=True)
    with np.load(args.npz, allow_pickle=True) as data:
        images = data["imgs"]
        raw_features = data["feats"].astype(np.float32)
        label_names = list(map(str, data["labels"]))

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    manifest = manifest.sort_values("index").reset_index(drop=True)
    if len(manifest) != len(label_names) or manifest["index"].tolist() != list(range(len(label_names))):
        raise ValueError("Manifest row indices do not align with the NPZ")
    if manifest["label"].astype(str).tolist() != label_names:
        raise ValueError("Manifest label order does not match the NPZ")
    if images.shape[1:] != (256, 256, 3):
        raise ValueError(f"Expected V6 images at 256x256x3, got {images.shape[1:]}")
    if raw_features.shape[1] != 28:
        raise ValueError(f"Expected 28 features, got {raw_features.shape[1]}")

    for cohort in ("NTUH", "LIMUC"):
        subset = manifest[(manifest["cohort"] == cohort) & manifest["patient_id"].notna()]
        crossings = subset.groupby("patient_id")["split"].nunique()
        if (crossings > 1).any():
            raise ValueError(f"{cohort} patient IDs cross V20 splits: {(crossings > 1).sum()}")

    y = np.asarray([int(name[-1]) for name in label_names], dtype=np.int64)
    split_names = manifest["split"].astype(str).to_numpy()
    train_idx = np.flatnonzero(split_names == "train")
    val_idx = np.flatnonzero(split_names == "validation")
    test_idx = np.flatnonzero(split_names == "test")
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Train, validation, and test splits must all be nonempty")

    print("Split sizes:", {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)}, flush=True)
    for name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        print(f"{name} label counts:", dict(zip(CLASS_NAMES, np.bincount(y[indices], minlength=4))), flush=True)

    scaler = StandardScaler().fit(raw_features[train_idx])
    features = scaler.transform(raw_features).astype(np.float32)
    pca = PCA(n_components=2, random_state=args.seed)
    pca.fit(features[train_idx])
    manifold = pca.transform(features).astype(np.float32)
    joblib.dump(scaler, os.path.join(args.output_dir, "feature_scaler.pkl"))
    joblib.dump(pca, os.path.join(args.output_dir, "feature_pca.pkl"))

    train_targets = ordinal_soft_labels(y[train_idx])
    hard_targets = to_categorical(y, NUM_CLASSES).astype(np.float32)
    all_train_targets = hard_targets.copy()
    all_train_targets[train_idx] = train_targets
    train_sequence = HybridSequence(
        images, features, manifold, all_train_targets,
        train_idx, args.batch_size, augment=args.augment, shuffle=True,
    )
    val_sequence = HybridSequence(
        images, features, manifold, hard_targets, val_idx, args.batch_size, shuffle=False
    )

    model = build_modse2_hybrid()
    model.compile(
        optimizer=Adam(learning_rate=1e-4, use_ema=True, ema_momentum=0.999),
        loss=make_cdw_ce_loss(),
        metrics=["accuracy"],
    )
    weights_path = os.path.join(args.output_dir, "colonomind_v20_best.weights.h5")
    history_path = os.path.join(args.output_dir, "training_history.csv")
    callbacks = [
        EpochLogger(),
        CSVLogger(history_path),
        ModelCheckpoint(weights_path, monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=True),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.early_stopping_patience,
            start_from_epoch=args.minimum_epochs,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=4, min_lr=1e-6),
    ]

    print(
        "Training Mod-SE(2) from scratch; "
        f"max_epochs={args.epochs}, early_stopping_start={args.minimum_epochs}, "
        f"patience={args.early_stopping_patience}; test split held out until final evaluation.",
        flush=True,
    )
    started = time.time()
    model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
    )
    print(f"Training finished in {(time.time() - started) / 60:.1f} minutes", flush=True)
    model.load_weights(weights_path)

    test_sequence = HybridSequence(
        images, features, manifold, hard_targets, test_idx, args.batch_size, shuffle=False
    )
    probabilities = model.predict(test_sequence, verbose=1)
    predictions = probabilities.argmax(axis=1)
    results = {
        "model": "Mod-SE(2) hybrid trained from scratch",
        "split_manifest": args.manifest,
        "evaluation_scope": {
            "NTUH_and_LIMUC": "patient-group split",
            "TMC-UCM": "official image split; patient IDs not verified",
        },
        "split_counts": {name: int(len(idx)) for name, idx in (
            ("train", train_idx), ("validation", val_idx), ("test", test_idx)
        )},
        "test": metric_bundle(y[test_idx], predictions),
        "test_by_cohort": {},
    }
    for cohort in sorted(manifest["cohort"].dropna().unique()):
        local = np.flatnonzero((manifest["cohort"].astype(str).to_numpy() == cohort) & (split_names == "test"))
        if len(local):
            results["test_by_cohort"][cohort] = {
                "samples": int(len(local)),
                **metric_bundle(y[local], predictions[np.searchsorted(test_idx, local)]),
            }

    with open(os.path.join(args.output_dir, "v20_evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Test metrics:", json.dumps(results["test"], indent=2), flush=True)
    print("Test metrics by cohort:", json.dumps(results["test_by_cohort"], indent=2), flush=True)
    print(classification_report(y[test_idx], predictions, labels=np.arange(4), target_names=CLASS_NAMES, digits=4))
    print("Evaluation saved:", os.path.join(args.output_dir, "v20_evaluation.json"), flush=True)


if __name__ == "__main__":
    main()
