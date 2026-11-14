#!/usr/bin/env python3
"""Train Mod-SE(2) and a disjoint, validation-gated feedback agent."""

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from train_colonomind_v20_patient_safe import (
    EpochLogger,
    HybridSequence,
    ValidationMetrics,
    build_modse2_hybrid,
    make_balanced_ce_loss,
    metric_bundle,
)


def selection_score(y_true, predictions):
    return 0.7 * np.mean(y_true == predictions) + 0.3 * f1_score(
        y_true, predictions, average="macro", labels=np.arange(4), zero_division=0
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="../Result/ColonoMind_v6/dataset_cache_v6.npz")
    parser.add_argument("--manifest", default="Result/ColonoMind_V22/v22_feedback_splits.csv")
    parser.add_argument("--output-dir", default="Result/ColonoMind_V22/run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("Loading V6 cache and V22 manifest...", flush=True)
    with np.load(args.npz, allow_pickle=True) as data:
        images = data["imgs"]
        raw_features = data["feats"].astype(np.float32)
        names = np.asarray(data["labels"], dtype=str)
    image_shape = tuple(images.shape[1:])
    if image_shape not in ((256, 256, 3), (384, 384, 3)) or raw_features.shape != (len(images), 28):
        raise ValueError("V22 requires aligned 256x256 or 384x384 RGB images and 28 biomarkers")
    print("Image shape:", image_shape, flush=True)
    frame = pd.read_csv(args.manifest, dtype={"patient_id": str}).sort_values("index").reset_index(drop=True)
    if len(frame) != len(names) or frame["index"].tolist() != list(range(len(names))):
        raise ValueError("Manifest indices do not align with the NPZ")
    if not np.array_equal(frame["label"].astype(str).to_numpy(), names):
        raise ValueError("Manifest labels do not align with the NPZ")
    for cohort in ("NTUH", "LIMUC"):
        groups = frame[frame["cohort"] == cohort].groupby("patient_id")["split"].nunique()
        if (groups > 1).any():
            raise ValueError(f"Patient split crossing in {cohort}")
    y = np.asarray([int(name[-1]) for name in names], dtype=np.int64)
    split = frame["split"].astype(str).to_numpy()
    indices = {name: np.flatnonzero(split == name) for name in (
        "cnn_train", "agent_train", "cnn_validation", "calibration", "test"
    )}
    if any(len(part) == 0 for part in indices.values()):
        raise ValueError("Every V22 partition must have images")
    print("V22 partition sizes:", {key: len(value) for key, value in indices.items()}, flush=True)

    train = indices["cnn_train"]
    scaler = StandardScaler().fit(raw_features[train])
    features = scaler.transform(raw_features).astype(np.float32)
    pca = PCA(n_components=2, random_state=args.seed).fit(features[train])
    manifold = pca.transform(features).astype(np.float32)
    joblib.dump(scaler, output / "feature_scaler.pkl")
    joblib.dump(pca, output / "feature_pca.pkl")
    targets = to_categorical(y, 4).astype(np.float32)

    def sequence(part, shuffle=False):
        return HybridSequence(
            images, features, manifold, targets, indices[part], args.batch_size,
            shuffle=shuffle,
        )

    model = build_modse2_hybrid(img_shape=image_shape)
    model.compile(
        optimizer=Adam(learning_rate=1e-4, use_ema=True, ema_momentum=0.999),
        loss=make_balanced_ce_loss(y[train]),
        metrics=["accuracy"],
    )
    cnn_validation = sequence("cnn_validation")
    diagnostics = ValidationMetrics(
        cnn_validation, y, indices["cnn_validation"], frame["cohort"].astype(str).to_numpy()
    )
    checkpoint = output / "colonomind_v22_best.weights.h5"
    callbacks = [
        diagnostics,
        EpochLogger(),
        CSVLogger(str(output / "training_history.csv")),
        ModelCheckpoint(str(checkpoint), monitor="val_selection_score", mode="max", save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor="val_selection_score", mode="max", patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_selection_score", mode="max", patience=4, factor=0.5, min_lr=1e-6),
    ]
    print("Training Mod-SE(2) CNN from scratch; feedback and test partitions excluded.", flush=True)
    started = time.time()
    model.fit(
        sequence("cnn_train", shuffle=True), validation_data=cnn_validation,
        epochs=args.epochs, callbacks=callbacks, verbose=0,
    )
    print(f"CNN training finished in {(time.time() - started) / 60:.1f} minutes", flush=True)
    model.load_weights(str(checkpoint))
    cnn_val, cnn_val_cohorts, cnn_val_prob = diagnostics.evaluate()
    np.savez_compressed(output / "cnn_validation_probabilities.npz", index=indices["cnn_validation"], label=y[indices["cnn_validation"]], probability=cnn_val_prob)

    feature_model = Model(model.inputs, model.get_layer("Fusion").output)

    def profile(part):
        batch_sequence = sequence(part)
        probabilities = model.predict(batch_sequence, verbose=0)
        deep = feature_model.predict(batch_sequence, verbose=0)
        idx = indices[part]
        vectors = np.concatenate((probabilities, features[idx], manifold[idx], deep), axis=1)
        return vectors, probabilities, y[idx]

    print("Profiling disjoint feedback-agent training images...", flush=True)
    agent_vectors, agent_cnn_prob, agent_labels = profile("agent_train")
    agent_errors = agent_cnn_prob.argmax(axis=1) != agent_labels
    sample_weights = np.where(agent_errors, 2.0, 1.0)
    sample_weights *= np.where(np.isin(agent_labels, (1, 2)), 1.5, 1.0)
    agent = lgb.LGBMClassifier(
        n_estimators=250, learning_rate=0.03, num_leaves=15, max_depth=5,
        min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
        class_weight="balanced", random_state=args.seed, n_jobs=8, verbosity=-1,
    )
    agent.fit(agent_vectors, agent_labels, sample_weight=sample_weights)
    agent.booster_.save_model(str(output / "feedback_agent.txt"))

    print("Calibrating handover on disjoint calibration images...", flush=True)
    calibration_vectors, calibration_cnn_prob, calibration_labels = profile("calibration")
    agent_prob = agent.predict_proba(calibration_vectors)
    base_predictions = calibration_cnn_prob.argmax(axis=1)
    agent_predictions = agent_prob.argmax(axis=1)
    confidence = calibration_cnn_prob.max(axis=1)
    candidates = []
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        overrides = confidence < threshold
        predictions = np.where(overrides, agent_predictions, base_predictions)
        result = metric_bundle(calibration_labels, predictions)
        candidates.append({
            "threshold": threshold,
            "overrides": int(overrides.sum()),
            "selection_score": float(selection_score(calibration_labels, predictions)),
            "metrics": result,
        })
        print(
            f"threshold={threshold:.2f}, overrides={overrides.sum()}, "
            f"exact={result['exact_accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}",
            flush=True,
        )
    chosen = max(candidates, key=lambda row: row["selection_score"])
    chosen_predictions = np.where(confidence < chosen["threshold"], agent_predictions, base_predictions)
    np.savez_compressed(
        output / "calibration_probabilities.npz",
        index=indices["calibration"], label=calibration_labels,
        cnn_probability=calibration_cnn_prob, agent_probability=agent_prob,
    )
    cohorts = frame.loc[indices["calibration"], "cohort"].astype(str).to_numpy()
    cohort_metrics = {
        cohort: metric_bundle(calibration_labels[cohorts == cohort], chosen_predictions[cohorts == cohort])
        for cohort in sorted(set(cohorts))
    }
    report = {
        "scope": "development_only_no_test_inference",
        "manifest": args.manifest,
        "cnn_validation": cnn_val,
        "cnn_validation_by_cohort": cnn_val_cohorts,
        "calibration_cnn": metric_bundle(calibration_labels, base_predictions),
        "calibration_agent": metric_bundle(calibration_labels, agent_predictions),
        "calibration_candidates": candidates,
        "selected_threshold": chosen["threshold"],
        "calibration_pipeline": chosen["metrics"],
        "calibration_pipeline_by_cohort": cohort_metrics,
    }
    (output / "development_evaluation.json").write_text(json.dumps(report, indent=2))
    (output / "inference_config.json").write_text(json.dumps({
        "cnn_weights": "colonomind_v22_best.weights.h5",
        "feature_scaler": "feature_scaler.pkl",
        "feature_pca": "feature_pca.pkl",
        "feedback_agent": "feedback_agent.txt",
        "agent_feature_order": ["cnn_probabilities_mes0_to_mes3", "standardized_biomarkers_28", "pca_2", "fusion_activation"],
        "handover_rule": "use_agent_when_cnn_max_probability_below_threshold",
        "threshold": chosen["threshold"],
    }, indent=2))
    print("Development evaluation:", json.dumps(report, indent=2), flush=True)
    print("Saved development report:", output / "development_evaluation.json", flush=True)


if __name__ == "__main__":
    main()
