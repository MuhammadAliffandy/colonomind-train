#!/usr/bin/env python3
"""
ColonoMind v19 — Noise-Robust Fine-Tuning of the V18 Mod-SE(2) Hybrid
=====================================================================
Starts from the V18 checkpoint (same architecture, same splits, same scaler/UMAP) and adds the
techniques the literature recommends for noisy ordinal medical labels:

  1. Confident Learning on the TRAIN split only: samples that the V18 model (still under-fit, so it
     has not memorised labels) confidently assigns to another grade are flagged and removed from
     fine-tuning. Validation and test labels are NEVER modified.
  2. Class-Distance-Weighted Cross-Entropy (CDW-CE) + soft ordinal targets: errors are penalised by
     their distance on the Mayo scale.
  3. Optimizer EMA (weight averaging) for a smoother, flatter solution.
  4. Checkpoint selection strictly on the validation split.
  5. Held-out-validation Feedback Agent (same as V18).
Test data is used once, at the very end.
"""
import os, sys, json, time, argparse, importlib.util, warnings
import numpy as np
import cv2, joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, classification_report

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v18", os.path.join(HERE, "train_colonomind_v18_audit_ordinal.py"))
v18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v18)

NUM_CLASSES, SEED = v18.NUM_CLASSES, v18.SEED


def ordinal_soft_labels(y, sigma=0.55):
    grades = np.arange(NUM_CLASSES)[None, :]
    w = np.exp(-((grades - y[:, None]) ** 2) / (2 * sigma ** 2))
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def make_cdw_ce_loss(alpha=2.0, mix=0.5):
    """soft-label CE + class-distance-weighted CE (penalises far-away grades more)."""
    dist = tf.constant([[abs(i - j) ** alpha for j in range(NUM_CLASSES)] for i in range(NUM_CLASSES)], tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        true_idx = tf.argmax(y_true, axis=-1)
        w = tf.gather(dist, true_idx)
        cdw = -tf.reduce_sum(w * tf.math.log(1.0 - y_pred), axis=-1) / (NUM_CLASSES - 1)
        return tf.reduce_mean(ce + mix * cdw)
    return loss


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    off1 = np.mean(np.abs(y_true - y_pred) <= 1)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    bin_acc = accuracy_score(y_true >= 2, y_pred >= 2)
    print(f"   [{name}] exact={acc*100:.2f}% | 1-grade-off={off1*100:.2f}% | QWK={qwk:.4f} | "
          f"remission(0-1) vs active(2-3)={bin_acc*100:.2f}%")
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="../Result/ColonoMind_v18_Audit", help="Dir with V18 checkpoint/scaler/umap")
    p.add_argument("--save_dir", default="../Result/ColonoMind_v19_NoiseRobust")
    p.add_argument("--v6_cache", default="../Result/ColonoMind_v6/dataset_cache_v6.npz")
    p.add_argument("--v15_cache", default="../Result/ColonoMind_v15_AutoBalanced/dataset_cache_v15_384.npz")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--noise_conf", type=float, default=0.75, help="min confidence of the contradicting prediction")
    p.add_argument("--noise_max_frac", type=float, default=0.15, help="cap on removed training fraction")
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    cache = args.v6_cache if os.path.exists(args.v6_cache) else args.v15_cache
    print(f"📦 Loading dataset cache from {cache}...", flush=True)
    d = np.load(cache, allow_pickle=True)
    imgs, feats, labels = d["imgs"], d["feats"], list(d["labels"])
    if imgs.shape[1] != args.img_size:
        print(f"🔄 Resizing {len(imgs)} images to {args.img_size}x{args.img_size}...", flush=True)
        imgs = np.array([cv2.resize(i.astype(np.uint8), (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
                         for i in imgs], dtype=np.float32)
    y_full = LabelEncoder().fit_transform(labels)

    print(f"🔄 Splitting 20,245 images into Train/Val/Test (Identical to V18)...", flush=True)
    # Identical splits to V18 -> same test set, no leakage between versions
    X_tv_i, X_te_i, X_tv_f, X_te_f, y_tv, y_te = train_test_split(
        imgs, feats, y_full, test_size=0.20, random_state=SEED, stratify=y_full)
    X_tr_i, X_va_i, X_tr_f, X_va_f, y_tr, y_va = train_test_split(
        X_tv_i, X_tv_f, y_tv, test_size=0.20, random_state=SEED, stratify=y_tv)

    print(f"📊 Dataset Splits: Train={len(y_tr)}, Val={len(y_va)}, Test={len(y_te)}", flush=True)
    print("🧬 Computing feature scaling and UMAP embeddings...", flush=True)
    scaler = joblib.load(os.path.join(args.base_dir, "scaler_v17.pkl"))
    umap_mod = joblib.load(os.path.join(args.base_dir, "umap_v17.pkl"))
    proc = lambda f: (scaler.transform(f), umap_mod.transform(scaler.transform(f)))
    Xtr_s, Utr = proc(X_tr_f); Xva_s, Uva = proc(X_va_f); Xte_s, Ute = proc(X_te_f)

    base_path = os.path.join(args.base_dir, "colonomind_v18_modse2.h5")
    print(f"⚡ Loading V18 checkpoint: {base_path}", flush=True)
    base = load_model(base_path, compile=False)

    pred = lambda m, I, F, U: m.predict([I / 255.0, F, U], batch_size=args.batch_size, verbose=0)

    print("\n🔍 Confident-learning pass over TRAIN split (scanning 12,956 samples)...", flush=True)
    P_tr = pred(base, X_tr_i, Xtr_s, Utr)
    p_given = P_tr[np.arange(len(y_tr)), y_tr]
    p_top, y_top = P_tr.max(1), P_tr.argmax(1)
    flagged = (y_top != y_tr) & (p_top >= args.noise_conf) & (p_given < 0.15)
    order = np.argsort(p_given)
    cap = int(args.noise_max_frac * len(y_tr))
    keep = np.ones(len(y_tr), bool)
    removed = [i for i in order if flagged[i]][:cap]
    keep[removed] = False
    print(f"   flagged={flagged.sum()} | removed={len(removed)} ({len(removed)/len(y_tr)*100:.1f}% of train)")
    for a in range(NUM_CLASSES):
        for b in range(NUM_CLASSES):
            n = int(np.sum(~keep & (y_tr == a) & (y_top == b)))
            if a != b and n:
                print(f"     given MES{a} but model confidently says MES{b}: {n}")
    json.dump({"removed_train_indices": [int(i) for i in removed]},
              open(os.path.join(args.save_dir, "removed_train_samples.json"), "w"))

    Xi, Xf, Xu, yk = X_tr_i[keep], Xtr_s[keep], Utr[keep], y_tr[keep]
    tr_gen = v18.HybridDataGenerator(Xi, Xf, Xu, ordinal_soft_labels(yk), batch_size=args.batch_size, augment=True)
    va_gen = v18.HybridDataGenerator(X_va_i, Xva_s, Uva, to_categorical(y_va, NUM_CLASSES),
                                     batch_size=args.batch_size, augment=False, shuffle=False)

    print("\n🔥 Fine-tuning with CDW-CE + EMA on the cleaned train split")
    model = load_model(base_path, compile=False)
    model.compile(optimizer=Adam(1e-4, use_ema=True, ema_momentum=0.999), loss=make_cdw_ce_loss(), metrics=["accuracy"])
    best_path = os.path.join(args.save_dir, "colonomind_v19_best.h5")
    t0 = time.time()
    model.fit(tr_gen, validation_data=va_gen, epochs=args.epochs, verbose=1, callbacks=[
        ModelCheckpoint(best_path, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=12, monitor="val_accuracy", restore_best_weights=False, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ])
    print(f"⏱️ fine-tune took {(time.time() - t0) / 60:.1f} min")
    ema_path = os.path.join(args.save_dir, "colonomind_v19_ema.h5")
    model.save(ema_path)

    # Model selection on VALIDATION only
    cands = {"v18": base, "v19_best": load_model(best_path, compile=False), "v19_ema": load_model(ema_path, compile=False)}
    val_scores = {k: accuracy_score(y_va, pred(m, X_va_i, Xva_s, Uva).argmax(1)) for k, m in cands.items()}
    print("\n📋 Validation accuracy by candidate:", {k: f"{v*100:.2f}%" for k, v in val_scores.items()})
    chosen = max(val_scores, key=val_scores.get)
    print(f"✅ Selected on validation: {chosen}")
    final_model = cands[chosen]

    final_preds, cnn_preds, _ = v18.run_active_feedback_loop(
        final_model, scaler, umap_mod,
        X_tr_i, Xtr_s, Utr, y_tr, X_va_i, Xva_s, Uva, y_va, X_te_i, Xte_s, Ute, y_te,
        save_dir=args.save_dir, batch_size=args.batch_size)

    print("\n" + "=" * 80 + "\n🏆 FINAL TEST EVALUATION (test split untouched until now)\n" + "=" * 80)
    evaluate("CNN only      ", y_te, cnn_preds)
    evaluate("CNN + Agent   ", y_te, final_preds)
    print("\nConfusion matrix (CNN + Agent):\n", confusion_matrix(y_te, final_preds))
    print(classification_report(y_te, final_preds, target_names=[f"MES{i}" for i in range(4)], digits=4))
    json.dump({"chosen": chosen, "val_scores": val_scores,
               "test_exact": float(accuracy_score(y_te, final_preds)),
               "test_qwk": float(cohen_kappa_score(y_te, final_preds, weights="quadratic"))},
              open(os.path.join(args.save_dir, "v19_evaluation.json"), "w"), indent=2)
    print("✅ COLONOMIND v19 COMPLETED")


if __name__ == "__main__":
    main()

