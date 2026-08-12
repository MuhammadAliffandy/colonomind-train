"""
ColonoMind - Soft Voting Ensemble Evaluator with TTA
=====================================================
Combines all 5 DL models via probability averaging (soft voting)
with Test Time Augmentation (TTA) for boosted accuracy.
No retraining required.

Usage (run from colonomind-train/ root):
    python -m src.ensemble_eval --dataset TMC-UCM --models_dir Result/Intra_TMC-UCM
    python -m src.ensemble_eval --dataset TMC-UCM --models_dir Result/Intra_TMC-UCM --tta 5
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, cohen_kappa_score, classification_report)
from tabulate import tabulate
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MUST import dgx_models FIRST to register all @register_keras_serializable decorators
from src import dgx_models  # noqa: F401

from src.dgx_dataloader import load_all_images, load_tmc_ucm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────
# TTA Augmentation Functions (operate on batches)
# ─────────────────────────────────────────────────────────
def apply_tta_augmentations(images, n_augmentations=5):
    """Generate n augmented versions of each image batch.
    Returns list of augmented image batches."""
    augmented_batches = [images]  # Original is always included

    if n_augmentations >= 2:
        # Horizontal flip
        augmented_batches.append(images[:, :, ::-1, :])

    if n_augmentations >= 3:
        # Vertical flip
        augmented_batches.append(images[:, ::-1, :, :])

    if n_augmentations >= 4:
        # Slight brightness increase (+10%)
        bright = np.clip(images * 1.1, 0, 255).astype(np.float32)
        augmented_batches.append(bright)

    if n_augmentations >= 5:
        # Slight brightness decrease (-10%)
        dark = np.clip(images * 0.9, 0, 255).astype(np.float32)
        augmented_batches.append(dark)

    if n_augmentations >= 6:
        # Horizontal + Vertical flip
        augmented_batches.append(images[:, ::-1, ::-1, :])

    if n_augmentations >= 7:
        # Slight contrast increase
        mean = np.mean(images, axis=(1, 2, 3), keepdims=True)
        contrast = np.clip((images - mean) * 1.15 + mean, 0, 255).astype(np.float32)
        augmented_batches.append(contrast)

    return augmented_batches[:n_augmentations]


# ─────────────────────────────────────────────────────────
# Load test data identical to train_dgx.py split logic
# ─────────────────────────────────────────────────────────
def load_test_data(dataset_name, base_dir):
    DATASET_PATHS = {
        'NTUH': [
            f'{base_dir}/Dataset+Code/MES classification_20250313',
            f'{base_dir}/Dataset+Code/MES classification_20250724'
        ],
        'LIMUC': [
            f'{base_dir}/Dataset/LIMUC/train_and_validation_sets',
            f'{base_dir}/Dataset/LIMUC/test_set'
        ],
        'TMC-UCM': [f'{base_dir}/Dataset/TMC-UCM/images']
    }
    TMC_UCM_ROOT = f'{base_dir}/Dataset/TMC-UCM'

    if dataset_name == 'Unified':
        ti, tf_, tl, tp = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None)
        ni, nf, nl, np_ = load_all_images(DATASET_PATHS['NTUH'], 'NTUH')
        li, lf, ll, lp  = load_all_images(
            [DATASET_PATHS['LIMUC'][0], DATASET_PATHS['LIMUC'][1]], 'LIMUC')
        ai = ti+ni+li; af = tf_+nf+lf; al = tl+nl+ll; ap = tp+np_+lp
        _, Xti, _, Xtf, _, ytl, _, _ = train_test_split(
            ai, af, al, ap, test_size=0.2, random_state=42, stratify=al)
    elif dataset_name == 'LIMUC':
        Xti, Xtf, ytl, _ = load_all_images([DATASET_PATHS['LIMUC'][1]], dataset_name)
    elif dataset_name == 'TMC-UCM':
        Xti, Xtf, ytl, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Test')
    elif dataset_name == 'NTUH':
        ai, af, al, ap = load_all_images(DATASET_PATHS['NTUH'], dataset_name)
        _, Xti, _, Xtf, _, ytl, _, _ = train_test_split(
            ai, af, al, ap, test_size=0.2, random_state=42, stratify=al)
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        return None, None, None

    return np.array(Xti, dtype=np.float32), np.array(Xtf, dtype=np.float32), ytl


def compute_metrics(y_true, y_pred, label):
    return {
        "Label":  label,
        "Acc":    f"{accuracy_score(y_true, y_pred)*100:.2f}%",
        "F1":     f"{f1_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%",
        "Prec":   f"{precision_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%",
        "Recall": f"{recall_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%",
        "QWK":    f"{cohen_kappa_score(y_true, y_pred, weights='quadratic'):.4f}",
    }


def predict_with_tta(model, X_img, Xs, Xu, n_tta=5):
    """Run inference with TTA: average probabilities across augmented versions."""
    aug_batches = apply_tta_augmentations(X_img, n_tta)
    all_probas = []
    for aug_imgs in aug_batches:
        proba = model.predict([aug_imgs, Xs, Xu], verbose=0)
        all_probas.append(proba)
    return np.mean(all_probas, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, required=True,
                        help="TMC-UCM | NTUH | LIMUC | Unified")
    parser.add_argument("--models_dir", type=str, required=True,
                        help="Path to folder containing {Model}_Experiment/ subfolders")
    parser.add_argument("--base_dir",   type=str,
                        default="/home/D13K48009/raid/Clara/new_drive",
                        help="Root dir containing Dataset/ and Dataset+Code/")
    parser.add_argument("--tta",        type=int, default=5,
                        help="Number of TTA augmentations (1=no TTA, 5=default, 7=max)")
    args = parser.parse_args()

    tta_label = f"TTA×{args.tta}" if args.tta > 1 else "No TTA"

    print(f"\n{'='*60}")
    print(f"🔬 ColonoMind Soft-Voting Ensemble: {args.dataset}")
    print(f"   models_dir : {args.models_dir}")
    print(f"   TTA        : {tta_label}")
    print(f"{'='*60}\n")

    # Load data
    X_img, X_feat, y_labels = load_test_data(args.dataset, args.base_dir)
    if X_img is None:
        return

    le = LabelEncoder()
    le.fit(["MES0", "MES1", "MES2", "MES3"])
    y_true = le.transform(y_labels)
    print(f"✅ {len(y_true)} test samples loaded.")
    class_dist = {c: int((y_true == i).sum()) for i, c in enumerate(le.classes_)}
    print(f"   Class distribution: {class_dist}\n")

    # Custom objects for Keras model loading
    custom_objects = {
        "resnet50_preprocess": dgx_models.resnet50_preprocess,
        "densenet_preprocess": dgx_models.densenet_preprocess,
        "efficientnet_preprocess": dgx_models.efficientnet_preprocess,
        "convnext_preprocess": dgx_models.convnext_preprocess,
        "vit_preprocess": dgx_models.vit_preprocess
    }

    MODEL_NAMES = ["ResNet-50", "DenseNet-121", "EfficientNet-B4",
                   "ConvNeXt-Tiny", "ViT-B-16"]
    all_probas = {}
    individual_rows = []

    for mn in MODEL_NAMES:
        exp_dir = os.path.join(args.models_dir, f"{mn}_Experiment")
        kp = os.path.join(exp_dir, f"{mn}_hybrid.keras")
        sp = os.path.join(exp_dir, "base_scaler.pkl")
        up = os.path.join(exp_dir, "umap_model.pkl")

        missing_files = [p for p in [kp, sp, up] if not os.path.exists(p)]
        if missing_files:
            print(f"⏭️  {mn}: skipping — MISSING FILES:")
            for m in missing_files:
                print(f"      - {m}")
            continue

        print(f"🔍 Inferencing {mn} ({tta_label})...")
        try:
            model = tf.keras.models.load_model(kp, compile=False, custom_objects=custom_objects)
        except Exception as e:
            print(f"   ❌ Load failed: {e}")
            continue

        Xs = joblib.load(sp).transform(X_feat)
        Xu = joblib.load(up).transform(Xs)

        if args.tta > 1:
            proba = predict_with_tta(model, X_img, Xs, Xu, n_tta=args.tta)
        else:
            proba = model.predict([X_img, Xs, Xu], verbose=0)

        all_probas[mn] = proba
        row = compute_metrics(y_true, np.argmax(proba, axis=1), mn)
        individual_rows.append(row)
        print(f"   ✅ Acc={row['Acc']}  F1={row['F1']}  QWK={row['QWK']}")

    if not all_probas:
        print("❌ No models loaded successfully. Check --models_dir path.")
        return

    # Soft-voting ensemble
    print(f"\n🗳️  Soft-voting over {len(all_probas)} models ({tta_label})...")
    avg_proba = np.mean(list(all_probas.values()), axis=0)
    ens_preds = np.argmax(avg_proba, axis=1)
    ens_row   = compute_metrics(y_true, ens_preds, f"★ ENSEMBLE ({tta_label})")

    H = ["Model", "Acc", "F1", "Prec", "Recall", "QWK"]

    print(f"\n\n📊  INDIVIDUAL MODELS — {args.dataset} ({tta_label})")
    print(tabulate(
        [[r["Label"], r["Acc"], r["F1"], r["Prec"], r["Recall"], r["QWK"]]
         for r in individual_rows],
        headers=H, tablefmt="grid"))

    print(f"\n🏆  SOFT-VOTING ENSEMBLE — {args.dataset} ({tta_label})")
    print(tabulate(
        [[ens_row["Label"], ens_row["Acc"], ens_row["F1"],
          ens_row["Prec"], ens_row["Recall"], ens_row["QWK"]]],
        headers=H, tablefmt="grid"))

    print(f"\n📋  PER-CLASS BREAKDOWN (Ensemble):")
    print(classification_report(y_true, ens_preds,
                                target_names=le.classes_, zero_division=0))


if __name__ == "__main__":
    main()
