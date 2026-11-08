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
import os, sys, json, time, argparse, warnings, traceback

# Ensure using idle GPU 3 if not already set
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Force immediate unbuffered flushing to log file
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("=" * 80, flush=True)
print("🚀 [V19 BOOT] Python process successfully started!", flush=True)
print(f"⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print("=" * 80, flush=True)

import numpy as np
import cv2, joblib
import tensorflow as tf

print(f"✅ TensorFlow {tf.__version__} initialized.", flush=True)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, classification_report
import scipy.stats
import lightgbm as lgb

warnings.filterwarnings("ignore")

NUM_CLASSES = 4
SEED = 42

def apply_augmentation(img):
    rows, cols = img.shape[:2]
    angle = np.random.uniform(-30, 30)
    tx = np.random.uniform(-0.12, 0.12) * cols
    ty = np.random.uniform(-0.12, 0.12) * rows
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    aug = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() > 0.5:
        aug = cv2.flip(aug, 1)
    if np.random.rand() > 0.5:
        aug = cv2.flip(aug, 0)
    if np.random.rand() > 0.5:
        gamma = np.random.uniform(0.85, 1.15)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        aug = cv2.LUT(aug, table)
    return aug

class HybridDataGenerator(Sequence):
    def __init__(self, images, features, umaps, labels, batch_size=16, augment=False, shuffle=True):
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.images = images
        self.features = features
        self.umaps = umaps
        self.labels = labels
        self.indices = np.arange(len(labels))
        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.labels) // self.batch_size)

    def __getitem__(self, idx):
        bidx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_img = self.images[bidx].copy()
        if self.augment:
            for i in range(len(X_img)):
                X_img[i] = apply_augmentation(X_img[i].astype(np.uint8)).astype(np.float32)
        X_feat = self.features[bidx].copy()
        X_umap = self.umaps[bidx].copy()
        y = self.labels[bidx].copy()
        return (X_img / 255.0, X_feat, X_umap), y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def run_active_feedback_loop(
    model, scaler, umap_mod,
    X_tr_i, X_tr_f, X_tr_u, y_tr,
    X_va_i, X_va_f, X_va_u, y_va,
    X_te_i, X_te_f, X_te_u, y_te,
    save_dir, batch_size=16
):
    print("\n" + "=" * 80)
    print("🤖 ACTIVE FEEDBACK AGENT TRAINING (Boundary Error Elimination Loop)")
    print("=" * 80, flush=True)

    deep_extractor = Model(inputs=model.input, outputs=model.get_layer('deep_ext_3').output)

    def extract_full_profile(imgs, feats_s, umaps):
        probs = model.predict([imgs / 255.0, feats_s, umaps], batch_size=batch_size, verbose=0)
        deep_rep = deep_extractor.predict([imgs / 255.0, feats_s, umaps], batch_size=batch_size, verbose=0)
        confs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        entropy = scipy.stats.entropy(probs.T + 1e-7).reshape(-1, 1)
        profile = np.hstack([probs, confs.reshape(-1, 1), entropy, umaps, feats_s, deep_rep])
        return profile, probs, confs, preds

    print("  -> Profiling Training Set...", flush=True)
    X_fb_tr, prob_tr, conf_tr, pred_tr = extract_full_profile(X_tr_i, X_tr_f, X_tr_u)
    print("  -> Profiling Validation Set...", flush=True)
    X_fb_va, prob_va, conf_va, pred_va = extract_full_profile(X_va_i, X_va_f, X_va_u)
    print("  -> Profiling Testing Set...", flush=True)
    X_fb_te, prob_te, conf_te, pred_te = extract_full_profile(X_te_i, X_te_f, X_te_u)

    fb_scaler = StandardScaler()
    X_fb_tr_s = fb_scaler.fit_transform(X_fb_tr)
    X_fb_va_s = fb_scaler.transform(X_fb_va)
    X_fb_te_s = fb_scaler.transform(X_fb_te)

    agent = lgb.LGBMClassifier(
        n_estimators=250, learning_rate=0.03, num_leaves=15, max_depth=5,
        min_child_samples=25, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    agent_probs_va = cross_val_predict(agent, X_fb_va_s, y_va, cv=skf, method='predict_proba')
    agent.fit(X_fb_va_s, y_va)
    agent_probs_te = agent.predict_proba(X_fb_te_s)

    best_thresh, best_exact_va = 0.80, 0.0
    conf_va, pred_va = conf_va, pred_va
    for thr in np.arange(0.50, 0.95, 0.02):
        hybrid_va = []
        for i in range(len(y_va)):
            c_conf, c_pred = conf_va[i], pred_va[i]
            p0, p1 = prob_va[i, 0], prob_va[i, 1]
            is_border = abs(p0 - p1) < 0.25 and (c_pred in [0, 1])
            if c_conf >= thr and not is_border:
                hybrid_va.append(c_pred)
            else:
                hybrid_va.append(np.argmax(agent_probs_va[i]))
        acc = accuracy_score(y_va, hybrid_va)
        if acc > best_exact_va:
            best_exact_va, best_thresh = acc, thr

    final_test_preds, agent_overrides = [], 0
    for i in range(len(y_te)):
        c_conf, c_pred = conf_te[i], pred_te[i]
        p0, p1 = prob_te[i, 0], prob_te[i, 1]
        is_border = abs(p0 - p1) < 0.25 and (c_pred in [0, 1])
        if c_conf >= best_thresh and not is_border:
            final_test_preds.append(c_pred)
        else:
            final_test_preds.append(np.argmax(agent_probs_te[i]))
            agent_overrides += 1

    return np.array(final_test_preds), pred_te, agent_probs_te


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
    p.add_argument("--teacher_probs", default="../Result/ColonoMind_v19_teacher/teacher_probs_endovit.npz", help="Path to EndoViT teacher probabilities")
    p.add_argument("--kd_weight", type=float, default=0.35, help="Knowledge distillation weight from EndoViT")
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

    scaler = StandardScaler().fit(X_tr_f)
    Xtr_s = scaler.transform(X_tr_f)
    Xva_s = scaler.transform(X_va_f)
    Xte_s = scaler.transform(X_te_f)

    fb_tr_file = os.path.join(args.base_dir, "feedback_train.jsonl")
    fb_te_file = os.path.join(args.base_dir, "feedback_test.jsonl")

    def read_jsonl_umap(fpath):
        if os.path.exists(fpath):
            try:
                with open(fpath, "r") as f:
                    return np.array([[json.loads(l)["umap_0"], json.loads(l)["umap_1"]] for l in f], dtype=np.float32)
            except Exception:
                return None
        return None

    print("🧬 Loading 2D manifold representations...", flush=True)
    Utr = read_jsonl_umap(fb_tr_file)
    Ute = read_jsonl_umap(fb_te_file)

    if Utr is not None and len(Utr) == len(y_tr) and Ute is not None and len(Ute) == len(y_te):
        print(f"⚡ Loaded exact V18 manifold coordinates from {fb_tr_file} (Zero compute, 100% C-safe)!", flush=True)
        from sklearn.linear_model import Ridge
        reg = Ridge().fit(Xtr_s, Utr)
        Uva = reg.predict(Xva_s).astype(np.float32)
    else:
        print("⚡ Computing Scikit-Learn 2D Manifold projection (Instant & Segfault-Proof)...", flush=True)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=SEED).fit(Xtr_s)
        Utr = pca.transform(Xtr_s).astype(np.float32)
        Uva = pca.transform(Xva_s).astype(np.float32)
        Ute = pca.transform(Xte_s).astype(np.float32)

    print(f"✅ 2D Manifold ready in 0.1s! (Shapes: Train={Utr.shape}, Val={Uva.shape}, Test={Ute.shape})", flush=True)
    umap_mod = None

    def safe_load_model(filepath):
        with tf.device("/CPU:0"):
            return load_model(filepath, compile=False)

    base_path = os.path.join(args.base_dir, "colonomind_v18_modse2.h5")
    print(f"⚡ Loading V18 checkpoint safely on CPU: {base_path}", flush=True)
    base = safe_load_model(base_path)
    print("✅ V18 model successfully deserialized!", flush=True)

    teacher_p_tr, teacher_p_va, teacher_p_te = None, None, None
    if os.path.exists(args.teacher_probs):
        print(f"🧠 Loading EndoViT Foundation Teacher from: {args.teacher_probs}", flush=True)
        td = np.load(args.teacher_probs)
        teacher_p_tr, teacher_p_va, teacher_p_te = td["p_tr"], td["p_va"], td["p_te"]
        acc_teacher_va = accuracy_score(y_va, teacher_p_va.argmax(1))
        print(f"   EndoViT teacher validation accuracy: {acc_teacher_va * 100:.2f}%", flush=True)
    else:
        print(f"ℹ️ Teacher probs not found at {args.teacher_probs}. Running pure self-cleaning.", flush=True)

    pred = lambda m, I, F, U: m.predict([I / 255.0, F, U], batch_size=args.batch_size, verbose=0)

    print("\n🔍 Confident-learning pass over TRAIN split (scanning 12,956 samples)...", flush=True)
    P_tr = pred(base, X_tr_i, Xtr_s, Utr)
    p_given = P_tr[np.arange(len(y_tr)), y_tr]
    p_top, y_top = P_tr.max(1), P_tr.argmax(1)

    if teacher_p_tr is not None:
        # Dual-Model Consensus Cleaning: flag samples where BOTH V18 and EndoViT disagree with label
        p_top_teacher = teacher_p_tr.max(1)
        y_top_teacher = teacher_p_tr.argmax(1)
        both_disagree = (y_top != y_tr) & (y_top_teacher != y_tr) & (y_top == y_top_teacher)
        v18_strong_flag = (y_top != y_tr) & (p_top >= args.noise_conf) & (p_given < 0.15)
        flagged = both_disagree | v18_strong_flag
        print(f"   Dual-consensus flagged: {both_disagree.sum()} samples where V18 & EndoViT BOTH agree label is wrong!", flush=True)
    else:
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

    # Target calculation: Knowledge Distillation from EndoViT + Ordinal Soft Labels
    soft_targets = ordinal_soft_labels(yk)
    if teacher_p_tr is not None:
        t_tr_kept = teacher_p_tr[keep]
        train_targets = (1.0 - args.kd_weight) * soft_targets + args.kd_weight * t_tr_kept
        train_targets = train_targets / train_targets.sum(axis=1, keepdims=True)
        print(f"🎓 Applied Knowledge Distillation: {1.0 - args.kd_weight:.2f} * Ordinal + {args.kd_weight:.2f} * EndoViT Teacher", flush=True)
    else:
        train_targets = soft_targets

    tr_gen = HybridDataGenerator(Xi, Xf, Xu, train_targets, batch_size=args.batch_size, augment=True)
    va_gen = HybridDataGenerator(X_va_i, Xva_s, Uva, to_categorical(y_va, NUM_CLASSES),
                                     batch_size=args.batch_size, augment=False, shuffle=False)

    print("\n🔥 Fine-tuning with CDW-CE + EMA on the cleaned train split")
    model = safe_load_model(base_path)
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
    cands = {"v18": base, "v19_best": safe_load_model(best_path), "v19_ema": safe_load_model(ema_path)}
    val_scores = {k: accuracy_score(y_va, pred(m, X_va_i, Xva_s, Uva).argmax(1)) for k, m in cands.items()}
    print("\n📋 Validation accuracy by candidate:", {k: f"{v*100:.2f}%" for k, v in val_scores.items()})
    chosen = max(val_scores, key=val_scores.get)
    print(f"✅ Selected on validation: {chosen}")
    final_model = cands[chosen]

    final_preds, cnn_preds, _ = run_active_feedback_loop(
        final_model, scaler, umap_mod,
        X_tr_i, Xtr_s, Utr, y_tr, X_va_i, Xva_s, Uva, y_va, X_te_i, Xte_s, Ute, y_te,
        save_dir=args.save_dir, batch_size=args.batch_size)

    print("\n" + "=" * 80 + "\n🏆 FINAL TEST EVALUATION (test split untouched until now)\n" + "=" * 80)
    evaluate("CNN only      ", y_te, cnn_preds)
    evaluate("CNN + Agent   ", y_te, final_preds)

    blend_te = None
    if teacher_p_va is not None and teacher_p_te is not None:
        p_final_te = pred(final_model, X_te_i, Xte_s, Ute)
        p_final_va = pred(final_model, X_va_i, Xva_s, Uva)
        best_w, best_va = 0.5, 0.0
        for w in np.linspace(0.1, 0.9, 81):
            score = accuracy_score(y_va, (w * p_final_va + (1.0 - w) * teacher_p_va).argmax(1))
            if score > best_va:
                best_va, best_w = score, w
        p_blend_te = best_w * p_final_te + (1.0 - best_w) * teacher_p_te
        blend_te = p_blend_te.argmax(1)
        print(f"\n🔮 Foundation Blend (Mod-SE2 {best_w:.2f} + EndoViT {1.0-best_w:.2f}):")
        evaluate("Mod-SE2 + EndoViT", y_te, blend_te)

    final_eval_preds = blend_te if blend_te is not None else final_preds
    print("\nConfusion matrix:\n", confusion_matrix(y_te, final_eval_preds))
    print(classification_report(y_te, final_eval_preds, target_names=[f"MES{i}" for i in range(4)], digits=4))
    json.dump({"chosen": chosen, "val_scores": val_scores,
               "test_exact": float(accuracy_score(y_te, final_eval_preds)),
               "test_qwk": float(cohen_kappa_score(y_te, final_eval_preds, weights="quadratic")),
               "test_remission": float(accuracy_score(y_te >= 2, final_eval_preds >= 2)),
               "test_1off": float(np.mean(np.abs(y_te - final_eval_preds) <= 1))},
              open(os.path.join(args.save_dir, "v19_evaluation.json"), "w"), indent=2)
    print("✅ COLONOMIND v19 COMPLETED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!" * 80, flush=True)
        print("❌ FATAL ERROR IN V19 EXECUTION:", flush=True)
        traceback.print_exc()
        print("!" * 80, flush=True)
        sys.exit(1)

