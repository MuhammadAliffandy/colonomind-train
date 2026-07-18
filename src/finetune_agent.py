"""
Colonomind - Fine-Tune + Feedback Agent
========================================
Loads the previously saved best_hybrid_model.h5, fine-tunes
it with a very low learning rate, then runs the TMC Feedback
Loop agent to climb toward a target accuracy (default 97%).

Usage:
    python src/finetune_agent.py \\
        --model_dir  ./results/all_datasets \\
        --output_dir ./results/finetuned \\
        --epochs     30 \\
        --target_acc 0.97 \\
        --max_loops  25
"""

import os
import sys
import json
import random
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.config import IMG_SIZE, DATASETS
from src.data_loader import load_dataset
from src.train import focal_loss


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def get_hash(row):
    return hashlib.sha1(
        json.dumps(row.astype(str).to_dict(), sort_keys=True).encode()
    ).hexdigest()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(args):
    # Set target GPU before initializing TensorFlow
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"🔒 Restricting execution to GPU: {args.gpu}")
        
    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seed(args.seed)

    # ── GPU setup ──────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nTensorFlow {tf.__version__}")
    if gpus:
        print(f"✅ GPU: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  No GPU detected. Running on CPU.")

    # ══════════════════════════════════════════════
    # 1. LOAD SAVED ARTEFACTS FROM PREVIOUS TRAINING
    # ══════════════════════════════════════════════
    print(f"\n{'='*55}")
    print(f" LOADING ARTEFACTS FROM: {args.model_dir}")
    print(f"{'='*55}")

    model_path   = os.path.join(args.model_dir, "best_hybrid_model.h5")
    scaler_path  = os.path.join(args.model_dir, "scaler.pkl")
    le_path      = os.path.join(args.model_dir, "label_encoder.pkl")
    umap_path    = os.path.join(args.model_dir, "umap_model.pkl")
    split_path   = os.path.join(args.model_dir, "split_info.pkl")

    for p in [model_path, scaler_path, le_path, umap_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"❌ Missing artefact: {p}\n"
                                    f"   Run train_all.py first to create it.")

    print("  Loading artefacts …")
    scaler       = joblib.load(scaler_path)
    le           = joblib.load(le_path)
    num_classes  = len(le.classes_)

    rebuild_umap = False
    try:
        umap_reducer = joblib.load(umap_path)
    except Exception as e:
        print(f"  ⚠️  Failed to load UMAP model ({e}). Will re-fit dynamically to bypass Numba mismatch.")
        rebuild_umap = True

    print("  Building model architecture and loading weights (bypassing Keras 3 from_config issues) …")
    from src.model import build_hybrid_model
    model = build_hybrid_model(
        image_input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        feat_input_shape=(scaler.n_features_in_,),
        umap_feat_shape=(2,),
        num_classes=num_classes,
        dropout_rate=0.4
    )
    # Load weights completely bypasses from_config errors for Dense, BatchNormalization, etc.
    model.load_weights(model_path)
    
    print(f"  ✅ Loaded — {num_classes} classes: {list(le.classes_)}")

    # ══════════════════════════════════════════════
    # 2. RELOAD & REBUILD DATA (same as train_all)
    # ══════════════════════════════════════════════
    print(f"\n{'='*55}")
    print(" RELOADING DATASETS")
    print(f"{'='*55}")

    all_img, all_feat, all_label, all_paths, all_source = [], [], [], [], []
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"  ⚠️  Skipping '{name}': {path}")
            continue
        print(f"  Loading '{name}' …")
        Xi, Xf, yl, ip = load_dataset(path)
        all_img.append(Xi);  all_feat.append(Xf)
        all_label.append(yl); all_paths.append(ip)
        all_source.append(np.full(len(yl), name))
        print(f"    → {len(yl)} images")

    if not all_label:
        raise ValueError("No datasets loaded — check DATASETS paths in src/config.py.")

    X_img_all   = np.concatenate(all_img,   axis=0)
    X_feat_all  = np.concatenate(all_feat,  axis=0)
    y_label_all = np.concatenate(all_label, axis=0)
    source_all  = np.concatenate(all_source,axis=0)

    print(f"\n  Total: {len(y_label_all)} images from {np.unique(source_all)}")

    # Encode using the SAME LabelEncoder from training
    y_encoded = le.transform(y_label_all)

    # ── Stratified split (70/15/15) ───────────────
    split = args.split
    val_pct  = split[1] / 100
    test_pct = split[2] / 100

    X_img_train, X_img_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp, _, _, src_tr, src_tmp = \
        train_test_split(X_img_all, X_feat_all, y_encoded, source_all, source_all,
                         test_size=val_pct + test_pct,
                         stratify=y_encoded, random_state=args.seed)

    rel_test = test_pct / (val_pct + test_pct)
    X_img_val, X_img_test, X_feat_val, X_feat_test, y_val, y_test = \
        train_test_split(X_img_tmp, X_feat_tmp, y_tmp,
                         test_size=rel_test, stratify=y_tmp, random_state=args.seed)

    print(f"\n  Split: Train={len(y_train)} | Val={len(y_val)} | Test={len(y_test)}")

    # ── Normalise ─────────────────────────────────
    X_img_train = X_img_train.astype(np.float32) / 255.0
    X_img_val   = X_img_val.astype(np.float32)   / 255.0
    X_img_test  = X_img_test.astype(np.float32)  / 255.0

    # Use the SAME scaler fitted during original training
    X_feat_train_s = scaler.transform(X_feat_train)
    X_feat_val_s   = scaler.transform(X_feat_val)
    X_feat_test_s  = scaler.transform(X_feat_test)

    # ── SMOTE ─────────────────────────────────────
    print("  Applying SMOTE …")
    smote = SMOTE(random_state=args.seed)
    X_feat_bal, y_train_bal = smote.fit_resample(X_feat_train_s, y_train)

    X_img_bal = []
    for feat, label in zip(X_feat_bal, y_train_bal):
        dists = np.linalg.norm(X_feat_train_s[y_train == label] - feat, axis=1)
        idx   = np.where(y_train == label)[0][np.argmin(dists)]
        X_img_bal.append(X_img_train[idx])
    X_img_bal = np.array(X_img_bal, dtype=np.float32)

    y_train_cat = to_categorical(y_train_bal, num_classes)
    y_val_cat   = to_categorical(y_val,       num_classes)
    y_test_cat  = to_categorical(y_test,      num_classes)

    # ── UMAP (use existing reducer or re-fit if failed) ────────────────
    if rebuild_umap:
        print("  Re-fitting UMAP dynamically to ensure exact feature projection …")
        import umap
        umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, metric='euclidean', random_state=args.seed)
        umap_reducer.fit(X_feat_bal)
    else:
        print("  Projecting via saved UMAP …")
        
    X_umap_train = umap_reducer.transform(X_feat_bal)
    X_umap_val   = umap_reducer.transform(X_feat_val_s)
    X_umap_test  = umap_reducer.transform(X_feat_test_s)

    # ══════════════════════════════════════════════
    # 3. FINE-TUNE THE KERAS MODEL
    #    Very low LR so we adjust weights, not reset
    # ══════════════════════════════════════════════
    print(f"\n{'='*55}")
    print(f" FINE-TUNING KERAS MODEL  (lr={args.finetune_lr})")
    print(f"{'='*55}")

    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_train_bal),
                                         y=y_train_bal)
    cw_dict = {i: w for i, w in enumerate(class_weights)}

    model.compile(
        optimizer=Adam(args.finetune_lr),
        loss=focal_loss(gamma=2.0, alpha=1.0),
        metrics=['accuracy']
    )

    ft_callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=args.patience,
                      restore_best_weights=True, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, verbose=1, mode='min')
    ]

    ft_history = None
    if args.epochs > 0:
        ft_history = model.fit(
            [X_img_bal, X_feat_bal, X_umap_train], y_train_cat,
            validation_data=([X_img_val, X_feat_val_s, X_umap_val], y_val_cat),
            batch_size=args.batch_size,
            epochs=args.epochs,
            class_weight=cw_dict,
            callbacks=ft_callbacks,
            verbose=1
        )
    else:
        print("  Skipping Keras fine-tuning (--epochs 0). Using existing weights …")

    ft_loss, ft_acc = model.evaluate(
        [X_img_test, X_feat_test_s, X_umap_test], y_test_cat, verbose=0)
    print(f"\n  Fine-tuned Keras Test Accuracy: {ft_acc:.4f}")

    # ══════════════════════════════════════════════
    # 4. SUPER AGENT — ONE SHOT TRAINING
    #    Builds on the fine-tuned Keras model output
    #    using confidence-based hybrid routing.
    # ══════════════════════════════════════════════
    print(f"\\n{'='*55}")
    print(f" SUPER AGENT — HYBRID ROUTING")
    print(f"{'='*55}")

    # Build agent feature vectors:
    # [Confidence, UMAP_x, UMAP_y, HybridPred, 20 handcrafted] = 24 cols
    y_proba_tr = model.predict([X_img_bal, X_feat_bal, X_umap_train], verbose=0, batch_size=64)
    y_proba_te = model.predict([X_img_test, X_feat_test_s, X_umap_test], verbose=0, batch_size=64)

    X_agent_train = np.column_stack([
        np.max(y_proba_tr, axis=1),
        X_umap_train,
        np.argmax(y_proba_tr, axis=1).astype(float),
        X_feat_bal
    ])
    X_agent_test = np.column_stack([
        np.max(y_proba_te, axis=1),
        X_umap_test,
        np.argmax(y_proba_te, axis=1).astype(float),
        X_feat_test_s
    ])

    y_test_ints = np.argmax(y_test_cat, axis=1)
    feat_cols   = [f'feature_{i}' for i in range(X_agent_train.shape[1])]

    df_train_agent = pd.DataFrame(X_agent_train, columns=feat_cols)
    df_train_agent['label'] = y_train_bal

    df_test_orig = pd.DataFrame(X_agent_test, columns=feat_cols)
    df_test_orig['label'] = y_test_ints

    # Train the Agent ONCE, on TRAIN only
    print(f"\\n  🤖 Training LightGBM Super Agent (One-Shot)...")
    X_tr_sc  = agent_scaler.fit_transform(df_train_agent[feat_cols].values)
    y_tr     = df_train_agent['label'].values
    X_te_sc  = agent_scaler.transform(df_test_orig[feat_cols].values)

    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        n_estimators=200,
        min_child_samples=5,
        class_weight='balanced',
        random_state=args.seed,
        verbosity=-1
    )
    clf.fit(X_tr_sc, y_tr)

    # ── Final report ───────────────────────────────
    y_pred_deep = np.argmax(y_proba_te, axis=1)
    base_acc = accuracy_score(y_test_ints, y_pred_deep)

    conf_test = np.max(y_proba_te, axis=1)
    low_conf_mask = conf_test < args.threshold

    y_pred_agent = clf.predict(X_te_sc)
    
    y_pred_hybrid = np.where(low_conf_mask, y_pred_agent, y_pred_deep)
    hybrid_acc = accuracy_score(y_test_ints, y_pred_hybrid)

    print("\n" + "=" * 55)
    print(f"  📊 BASE DEEP LEARNING ACCURACY: {base_acc:.4f}  ({base_acc*100:.2f}%)")
    print(f"  ⚙️  HYBRID SELECTOR (Threshold = {args.threshold})")
    print(f"  🔍 Delegated {low_conf_mask.sum()} / {len(low_conf_mask)} low-confidence cases to Agent")
    print(f"  🚀 FINAL HYBRID ACCURACY:       {hybrid_acc:.4f}  ({hybrid_acc*100:.2f}%)")
    print("=" * 55)
    print("\\n✅ Hybrid Classification Report:")
    print(classification_report(
        y_test_ints, y_pred_hybrid,
        target_names=[f"MES{i}" for i in range(num_classes)], digits=4))

    # ══════════════════════════════════════════════
    # 5. SAVE EVERYTHING
    # ══════════════════════════════════════════════
    model.save(os.path.join(args.output_dir, "finetuned_hybrid_model.h5"))
    clf.booster_.save_model(os.path.join(args.output_dir, "super_agent_lgbm.txt"))
    joblib.dump(agent_scaler, os.path.join(args.output_dir, "agent_scaler.pkl"))
    joblib.dump(scaler,       os.path.join(args.output_dir, "scaler.pkl"))
    joblib.dump(le,           os.path.join(args.output_dir, "label_encoder.pkl"))
    joblib.dump(umap_reducer, os.path.join(args.output_dir, "umap_model.pkl"))

    # ── Training plot ──────────────────────────────
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    if ft_history is not None and 'accuracy' in ft_history.history:
        plt.plot(ft_history.history['accuracy'],     label='Train')
        plt.plot(ft_history.history['val_accuracy'], label='Val')
    plt.title('Fine-Tune Accuracy')
    plt.xlabel('Epoch'); plt.legend()

    plt.subplot(1, 2, 2)
    if ft_history is not None and 'loss' in ft_history.history:
        plt.plot(ft_history.history['loss'],     label='Train')
        plt.plot(ft_history.history['val_loss'], label='Val')
    plt.title('Fine-Tune Loss')
    plt.xlabel('Epoch'); plt.legend()

    # We can just leave the first two subplots
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "finetune_history.png"), dpi=300)

    print(f"\\n✅ All artefacts saved to: {args.output_dir}")
    print(f"   - finetuned_hybrid_model.h5")
    print(f"   - super_agent_lgbm.txt  +  agent_scaler.pkl")
    print(f"   - scaler.pkl, label_encoder.pkl, umap_model.pkl")
    print(f"   - finetune_history.png")


# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fine-tune saved Keras model then run TMC Feedback Loop agent"
    )
    parser.add_argument('--model_dir',    required=True,
                        help="Folder containing artefacts from train_all.py")
    parser.add_argument('--output_dir',   required=True,
                        help="Folder to save fine-tuned artefacts")
    parser.add_argument('--split',  type=int, nargs=3, default=[70, 15, 15],
                        help="Same Train/Val/Test split used in train_all (default: 70 15 15)")
    parser.add_argument('--gpu',          type=str,   default=None,
                        help="Specify GPU device ID (e.g., '1' or '0')")
    parser.add_argument('--epochs',       type=int,   default=30,
                        help="Fine-tuning epochs (default: 30)")
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--finetune_lr',  type=float, default=1e-5,
                        help="Learning rate for fine-tuning (default: 1e-5)")
    parser.add_argument('--patience',     type=int,   default=10,
                        help="EarlyStopping patience (default: 10)")
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--threshold',    type=float, default=0.70,
                        help="Confidence threshold for hybrid routing (default: 0.70)")

    args = parser.parse_args()
    assert sum(args.split) == 100, f"Split must sum to 100, got {sum(args.split)}"
    main(args)
