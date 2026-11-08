#!/usr/bin/env python3
"""
ColonoMind — Rapid Ensemble: Mod-SE(2) V18 + EndoViT Foundation Model
====================================================================
Combines the two best distinct paradigms:
  1. Mod-SE(2) V18 (In-house CNN + Wavelet + Color Biomarkers + UMAP): 80.71%
  2. EndoViT Foundation Model (700k-image Vision Transformer): 77.45%
Optimizes blending weights on the validation set, then evaluates on the untouched test set.
Takes ~1 minute to run on DGX GPU.
"""

import os
import json
import argparse
import numpy as np
import cv2
import joblib

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score

SEED = 42
NUM_CLASSES = 4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6_cache", default="../Result/ColonoMind_v6/dataset_cache_v6.npz")
    parser.add_argument("--v18_dir", default="../Result/ColonoMind_v18_Audit")
    parser.add_argument("--endovit_probs", default="../Result/ColonoMind_v19_teacher/teacher_probs_endovit.npz")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 COLONOMIND ENSEMBLE: Mod-SE(2) V18 + EndoViT Foundation Model")
    print("=" * 80)

    # 1. Load ground truth dataset
    print(f"📦 Loading dataset cache: {args.v6_cache}...")
    d = np.load(args.v6_cache, allow_pickle=True)
    imgs = d['imgs']
    feats = d['feats']
    labels = list(d['labels'])
    y_full = LabelEncoder().fit_transform(labels)

    # Identical stratified split as V18
    X_tv_i, X_te_i, X_tv_f, X_te_f, y_tv, y_te = train_test_split(
        imgs, feats, y_full, test_size=0.20, random_state=SEED, stratify=y_full
    )
    X_tr_i, X_va_i, X_tr_f, X_va_f, y_tr, y_va = train_test_split(
        X_tv_i, X_tv_f, y_tv, test_size=0.20, random_state=SEED, stratify=y_tv
    )

    # 2. Load V18 Mod-SE(2) Model & Predict
    print(f"⚡ Loading V18 Mod-SE(2) from: {args.v18_dir}...")
    scaler = joblib.load(os.path.join(args.v18_dir, "scaler_v17.pkl"))
    umap_mod = joblib.load(os.path.join(args.v18_dir, "umap_v17.pkl"))
    with tf.device("/CPU:0"):
        v18_model = load_model(os.path.join(args.v18_dir, "colonomind_v18_modse2.h5"), compile=False)

    print("   Predicting V18 Validation Set...")
    F_va_s = scaler.transform(X_va_f)
    U_va = umap_mod.transform(F_va_s)
    p_v18_va = v18_model.predict([X_va_i / 255.0, F_va_s, U_va], batch_size=args.batch_size, verbose=1)

    print("   Predicting V18 Test Set...")
    F_te_s = scaler.transform(X_te_f)
    U_te = umap_mod.transform(F_te_s)
    p_v18_te = v18_model.predict([X_te_i / 255.0, F_te_s, U_te], batch_size=args.batch_size, verbose=1)

    # 3. Load EndoViT Probabilities
    print(f"⚡ Loading EndoViT teacher probabilities from: {args.endovit_probs}...")
    endo_data = np.load(args.endovit_probs)
    p_endo_va = endo_data['p_va']
    p_endo_te = endo_data['p_te']

    acc_v18_va = accuracy_score(y_va, p_v18_va.argmax(1))
    acc_endo_va = accuracy_score(y_va, p_endo_va.argmax(1))
    print(f"\n📊 Standalone Validation Scores:")
    print(f"   • Mod-SE(2) V18  : {acc_v18_va * 100:.2f}%")
    print(f"   • EndoViT ViT    : {acc_endo_va * 100:.2f}%")

    # 4. Optimize Ensemble Blending Weight on Validation
    print("\n🔍 Optimizing Ensemble Weights (Grid Search on Validation)...")
    best_w = 0.5
    best_val_acc = 0.0

    for w in np.linspace(0.0, 1.0, 101):
        p_blend_va = w * p_v18_va + (1.0 - w) * p_endo_va
        score = accuracy_score(y_va, p_blend_va.argmax(1))
        if score > best_val_acc:
            best_val_acc = score
            best_w = w

    print(f"✅ Optimal Weight: {best_w:.2f} * Mod-SE(2) + {1.0 - best_w:.2f} * EndoViT")
    print(f"   Validation Ensemble Accuracy: {best_val_acc * 100:.2f}% (+{(best_val_acc - max(acc_v18_va, acc_endo_va))*100:.2f}%)")

    # 5. Evaluate on Unseen Test Set
    p_blend_te = best_w * p_v18_te + (1.0 - best_w) * p_endo_te
    final_preds = p_blend_te.argmax(1)

    test_exact = accuracy_score(y_te, final_preds)
    test_1off = np.mean(np.abs(y_te - final_preds) <= 1)
    test_qwk = cohen_kappa_score(y_te, final_preds, weights="quadratic")
    test_remission = accuracy_score(y_te >= 2, final_preds >= 2)

    print("\n" + "=" * 80)
    print("🏆 FINAL ENSEMBLE TEST SET RESULTS")
    print("=" * 80)
    print(f"   • Exact 4-Class Accuracy : {test_exact * 100:.2f}% ({np.sum(y_te == final_preds)} / {len(y_te)})")
    print(f"   • Clinical 1-Grade Off   : {test_1off * 100:.2f}% ({np.sum(np.abs(y_te - final_preds) <= 1)} / {len(y_te)})")
    print(f"   • Quadratic Weighted Kappa: {test_qwk:.4f}")
    print(f"   • Remission vs Active    : {test_remission * 100:.2f}% (MES 0-1 vs MES 2-3)")

    cm = confusion_matrix(y_te, final_preds)
    print("\n📋 4x4 Confusion Matrix:")
    header = f"{'':<8}" + "".join([f"MES{i:<7}" for i in range(NUM_CLASSES)])
    print(header)
    for i, row in enumerate(cm):
        print(f"MES{i}:".ljust(8) + "".join([f"{val:<10}" for val in row]))

    print("\n📄 Detailed Classification Report:")
    print(classification_report(y_te, final_preds, target_names=[f"MES{i}" for i in range(NUM_CLASSES)], digits=4))

    out_file = "../Result/ensemble_v18_endovit_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "test_exact_accuracy": float(test_exact),
            "test_1_grade_off": float(test_1off),
            "test_qwk": float(test_qwk),
            "test_remission_vs_active": float(test_remission),
            "best_w_v18": float(best_w),
            "confusion_matrix": cm.tolist()
        }, f, indent=2)
    print(f"💾 Results saved to {out_file}")

if __name__ == "__main__":
    main()

