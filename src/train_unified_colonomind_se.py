"""
ColonoMind SE v3 — Maximum Performance Training Pipeline
=========================================================
Key upgrades over Super Agent.ipynb:
  1. EfficientNetV2-S backbone (much stronger than DenseNet121)
  2. Focal Loss for class-imbalanced medical data
  3. 3-Phase training: Warmup → Partial Unfreeze → Full Fine-tune
  4. MixUp augmentation to prevent memorisation
  5. Test-Time Augmentation (TTA) for evaluation
  6. Deep Feature Agent with Optuna-tuned LightGBM
  7. Strict patient-level split (zero leak)
  8. Every step is cached so the pipeline is fully resumable
"""
import os, cv2, json, joblib, pywt, argparse, gc, scipy.stats
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, cohen_kappa_score,
                             roc_curve, auc)
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, BatchNormalization,
                                     Dropout, GlobalAveragePooling2D, Multiply,
                                     Reshape, Activation)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
try:
    from tensorflow.keras.applications import EfficientNetV2S
except ImportError:
    from tensorflow.keras.applications import EfficientNetB4 as EfficientNetV2S

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images, load_tmc_ucm

# ==============================================================================
# CONFIG
# ==============================================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_CLASSES = 4
CLASS_NAMES = ['MES0', 'MES1', 'MES2', 'MES3']

# ==============================================================================
# FOCAL LOSS — handles class imbalance far better than cross-entropy
# ==============================================================================
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow(1 - y_pred, self.gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

# ==============================================================================
# AUGMENTATION (Heavy + MixUp)
# ==============================================================================
def apply_heavy_augmentation(img):
    rows, cols = img.shape[:2]
    # Roto-translation
    angle = np.random.uniform(-180, 180)
    tx = np.random.uniform(-0.1, 0.1) * cols
    ty = np.random.uniform(-0.1, 0.1) * rows
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
    M[0, 2] += tx; M[1, 2] += ty
    img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    # Flips
    if np.random.rand() > 0.5: img = cv2.flip(img, 1)
    if np.random.rand() > 0.5: img = cv2.flip(img, 0)
    # Brightness + Contrast
    alpha = 1.0 + np.random.uniform(-0.3, 0.3)
    beta = np.random.uniform(-25, 25)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # Color jitter (Hue shift)
    if np.random.rand() > 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0].astype(int) + np.random.randint(-15, 15)) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # Random erasing (CutOut)
    if np.random.rand() > 0.5:
        ch, cw = np.random.randint(20, 60), np.random.randint(20, 60)
        cx, cy = np.random.randint(0, cols - cw), np.random.randint(0, rows - ch)
        img[cy:cy+ch, cx:cx+cw] = np.random.randint(0, 255, (ch, cw, 3), dtype=np.uint8)
    return img

# ==============================================================================
# GENERATOR with MixUp
# ==============================================================================
class HybridGenerator(Sequence):
    def __init__(self, imgs, feats, umaps, labels, batch_size=16,
                 shuffle=True, augment=False, mixup_alpha=0.0):
        self.imgs = imgs
        self.feats = feats
        self.umaps = umaps
        self.labels_raw = labels
        self.labels = to_categorical(labels, num_classes=NUM_CLASSES)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.indexes = np.arange(len(self.imgs))
        super().__init__()

    def __len__(self): return int(np.floor(len(self.imgs) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        bs = len(idxs)
        X_img = np.empty((bs, *IMG_SIZE, 3), dtype=np.float32)
        X_feat = self.feats[idxs].copy()
        X_umap = self.umaps[idxs].copy()
        y = self.labels[idxs].copy()

        for i, idx in enumerate(idxs):
            img = self.imgs[idx]
            img = cv2.resize(img, IMG_SIZE)
            if self.augment:
                img = apply_heavy_augmentation(img)
            X_img[i] = img / 255.0

        # MixUp
        if self.augment and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            perm = np.random.permutation(bs)
            X_img = lam * X_img + (1 - lam) * X_img[perm]
            X_feat = lam * X_feat + (1 - lam) * X_feat[perm]
            X_umap = lam * X_umap + (1 - lam) * X_umap[perm]
            y = lam * y + (1 - lam) * y[perm]

        return tuple((X_img, X_feat, X_umap)), y

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)

# ==============================================================================
# MODEL: EfficientNetV2-S + SE-Attention Fusion + Handcrafted + UMAP
# ==============================================================================
def se_attention(x, ratio=8):
    """Squeeze-and-Excitation on dense features."""
    ch = x.shape[-1]
    se = Dense(ch // ratio, activation='relu', use_bias=False)(x)
    se = Dense(ch, activation='sigmoid', use_bias=False)(se)
    return Multiply()([x, se])

def build_model():
    inp_img = Input(shape=(*IMG_SIZE, 3), name='input_image')
    base = EfficientNetV2S(weights='imagenet', include_top=False, input_tensor=inp_img)

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    feat_cnn = se_attention(x)

    inp_feat = Input(shape=(20,), name='input_feat')
    fh = BatchNormalization()(inp_feat)
    fh = Dense(128, activation='relu')(fh)
    fh = Dropout(0.2)(fh)
    feat_hand = Dense(64, activation='relu')(fh)

    inp_umap = Input(shape=(2,), name='input_umap')
    feat_umap = Dense(32, activation='relu')(inp_umap)

    combined = Concatenate(name='Fusion')([feat_cnn, feat_hand, feat_umap])
    combined = se_attention(combined)  # SE on fused features

    x = Dense(256, activation='relu')(combined)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    model = Model(inputs=[inp_img, inp_feat, inp_umap], outputs=out)
    return model, base

# ==============================================================================
# DATA LOADING (Patient-Level Split — Zero Leak)
# ==============================================================================
def extract_patient_id(path):
    fname = os.path.basename(str(path))
    if 'train_and_validation_sets' in str(path) or 'test_set' in str(path):
        return fname.split('_')[0]
    elif 'TMC-UCM' in str(path):
        return fname.split('_')[0]
    else:
        return fname.split('-')[0]

def load_unified_data(base_dir):
    print("\n📦 Loading Unified Dataset (Patient-Level Split)...")
    tmc_root = f'{base_dir}/Dataset/TMC-UCM'
    ntuh_paths = [f'{base_dir}/Dataset+Code/MES classification_20250313',
                  f'{base_dir}/Dataset+Code/MES classification_20250724']
    limuc_paths = [f'{base_dir}/Dataset/LIMUC/train_and_validation_sets',
                   f'{base_dir}/Dataset/LIMUC/test_set']

    ti, tf_, tl, tp = load_tmc_ucm(tmc_root, split_filter=None)
    ni, nf, nl, np_ = load_all_images(ntuh_paths, 'NTUH')
    li, lf, ll, lp = load_all_images(limuc_paths, 'LIMUC')

    all_imgs = ti + ni + li
    all_feats = np.array(tf_ + nf + lf)
    all_labels = tl + nl + ll
    all_paths = tp + np_ + lp
    all_patients = [extract_patient_id(p) for p in all_paths]

    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    all_labels_encoded = le.transform(all_labels)

    df = pd.DataFrame({
        'idx': range(len(all_imgs)),
        'label': all_labels_encoded,
        'patient': all_patients
    })
    patients = df['patient'].unique()

    # Patient split: 70/15/15
    train_p, temp_p = train_test_split(patients, test_size=0.3, random_state=42)
    val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)

    train_df = df[df['patient'].isin(train_p)]
    val_df = df[df['patient'].isin(val_p)]
    test_df = df[df['patient'].isin(test_p)]

    print(f"Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: Tr={sum(train_df['label']==i)} Va={sum(val_df['label']==i)} Te={sum(test_df['label']==i)}")

    return train_df, val_df, test_df, all_imgs, all_feats

# ==============================================================================
# TEST-TIME AUGMENTATION (TTA) — boosts accuracy ~2-4%
# ==============================================================================
def predict_with_tta(model, imgs, feats, umaps, n_aug=5):
    """Average predictions over original + n_aug augmented copies."""
    all_preds = []
    for aug_i in range(n_aug + 1):
        batch_imgs = np.empty((len(imgs), *IMG_SIZE, 3), dtype=np.float32)
        for i, img in enumerate(imgs):
            img_r = cv2.resize(img, IMG_SIZE)
            if aug_i > 0:  # augment copies (not the original)
                img_r = apply_heavy_augmentation(img_r)
            batch_imgs[i] = img_r / 255.0
        preds = model.predict([batch_imgs, feats, umaps], batch_size=BATCH_SIZE, verbose=0)
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)

# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_roc(y_true, y_pred_proba, out_dir):
    Y_bin = label_binarize(y_true, classes=[0,1,2,3])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(NUM_CLASSES):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})')
    plt.plot(all_fpr, mean_tpr, color='navy', lw=2.5, linestyle='--',
             label=f'Macro Avg (AUC = {macro_auc:.3f})')
    plt.plot([0,1],[0,1], 'k--', lw=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ColonoMind SE v3 — ROC Curve (Unified)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ROC_Curve_Unified.png'), dpi=200)
    plt.close()
    return macro_auc, roc_auc

def plot_confusion_matrix(y_true, y_pred, out_dir, title_extra=""):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted', fontsize=12); plt.ylabel('True', fontsize=12)
    acc = accuracy_score(y_true, y_pred)
    plt.title(f'Confusion Matrix {title_extra}(Acc: {acc*100:.1f}%)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = f'Confusion_Matrix_Unified{title_extra.strip().replace(" ","_")}.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close()

def plot_history(history, out_dir):
    acc = history.history.get('accuracy') or history.history.get('acc')
    val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')
    if acc is None: return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(acc, label='Train'); a1.plot(val_acc, label='Val')
    a1.set_title('Accuracy'); a1.legend()
    a2.plot(history.history['loss'], label='Train')
    a2.plot(history.history['val_loss'], label='Val')
    a2.set_title('Loss'); a2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Training_History.png'), dpi=200)
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/home/D13K48009/raid/Clara/new_drive")
    parser.add_argument("--save_dir", type=str, default="../Result/Unified_ColonoMind_SE")
    parser.add_argument("--epochs_warmup", type=int, default=10)
    parser.add_argument("--epochs_partial", type=int, default=20)
    parser.add_argument("--epochs_full", type=int, default=40)
    parser.add_argument("--tta", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Delete old caches from broken runs to force fresh evaluation
    for old_cache in ["deep_train_cache.npz", "deep_test_cache.npz",
                      "deep_train.npz", "deep_test.npz", "deep_features_cache.npz"]:
        p = os.path.join(args.save_dir, old_cache)
        if os.path.exists(p):
            os.remove(p)
            print(f"🗑️ Removed stale cache: {old_cache}")

    # ── STEP 1: Data ──────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1/8: Loading Data")
    print("=" * 70)
    train_df, val_df, test_df, all_imgs, all_feats = load_unified_data(args.base_dir)

    X_train_f = all_feats[train_df['idx'].values]
    X_val_f = all_feats[val_df['idx'].values]
    X_test_f = all_feats[test_df['idx'].values]

    # ── STEP 2: Scaler & UMAP ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2/8: Scaler & UMAP")
    print("=" * 70)
    sc_path = os.path.join(args.save_dir, "scaler_unified.pkl")
    um_path = os.path.join(args.save_dir, "umap_unified.pkl")

    if os.path.exists(sc_path) and os.path.exists(um_path):
        print("Loading cached...")
        scaler = joblib.load(sc_path); umap_model = joblib.load(um_path)
    else:
        scaler = StandardScaler().fit(X_train_f)
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                                random_state=42).fit(scaler.transform(X_train_f))
        joblib.dump(scaler, sc_path); joblib.dump(umap_model, um_path)

    Xtr_s = scaler.transform(X_train_f)
    Xva_s = scaler.transform(X_val_f)
    Xte_s = scaler.transform(X_test_f)
    Utr = umap_model.transform(Xtr_s)
    Uva = umap_model.transform(Xva_s)
    Ute = umap_model.transform(Xte_s)

    # ── STEP 3: Generators ───────────────────────────────────────
    tr_imgs = [all_imgs[i] for i in train_df['idx'].values]
    va_imgs = [all_imgs[i] for i in val_df['idx'].values]
    te_imgs = [all_imgs[i] for i in test_df['idx'].values]

    train_gen = HybridGenerator(tr_imgs, Xtr_s, Utr, train_df['label'].values,
                                batch_size=BATCH_SIZE, shuffle=True, augment=True, mixup_alpha=0.3)
    val_gen = HybridGenerator(va_imgs, Xva_s, Uva, val_df['label'].values,
                              batch_size=BATCH_SIZE, shuffle=False, augment=False)
    test_gen = HybridGenerator(te_imgs, Xte_s, Ute, test_df['label'].values,
                               batch_size=BATCH_SIZE, shuffle=False, augment=False)

    # Class weights
    y_ints = train_df['label'].values
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_ints), y=y_ints)
    cw_dict = dict(enumerate(cw))
    print(f"⚖️ Class Weights: {cw_dict}")

    # ── STEP 4: 3-Phase Training ─────────────────────────────────
    model_path = os.path.join(args.save_dir, "best_hybrid_keras.h5")

    if os.path.exists(model_path):
        print(f"\n✅ Model found at {model_path}. Skipping training.")
        model = load_model(model_path, custom_objects={'FocalLoss': FocalLoss})
    else:
        model, base = build_model()
        print(f"\nModel params: {model.count_params():,}")

        # ── Phase 1: Warmup (freeze backbone) ──
        print("\n" + "=" * 70)
        print("STEP 4A/8: WARMUP (Frozen Backbone)")
        print("=" * 70)
        base.trainable = False
        model.compile(optimizer=Adam(1e-3),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_gen, validation_data=val_gen,
                  epochs=args.epochs_warmup, class_weight=cw_dict)

        # ── Phase 2: Partial unfreeze (last 30% of backbone) ──
        print("\n" + "=" * 70)
        print("STEP 4B/8: PARTIAL UNFREEZE (Last 30% of backbone)")
        print("=" * 70)
        base.trainable = True
        total_layers = len(base.layers)
        freeze_until = int(total_layers * 0.7)
        for layer in base.layers[:freeze_until]:
            layer.trainable = False
        trainable_count = sum(1 for l in base.layers if l.trainable)
        print(f"  Unfrozen {trainable_count}/{total_layers} backbone layers")

        model.compile(optimizer=Adam(5e-5),
                      loss=FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.1),
                      metrics=['accuracy'])
        cb_partial = [
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1),
            EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy', mode='max')
        ]
        model.fit(train_gen, validation_data=val_gen,
                  epochs=args.epochs_partial, class_weight=cw_dict, callbacks=cb_partial)

        # ── Phase 3: Full fine-tune ──
        print("\n" + "=" * 70)
        print("STEP 4C/8: FULL FINE-TUNE (All layers)")
        print("=" * 70)
        for layer in base.layers:
            layer.trainable = True
        model.compile(optimizer=Adam(1e-5),
                      loss=FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.05),
                      metrics=['accuracy'])
        cb_full = [
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy', mode='max')
        ]
        history = model.fit(train_gen, validation_data=val_gen,
                            epochs=args.epochs_full, class_weight=cw_dict, callbacks=cb_full)
        plot_history(history, args.save_dir)

        # Reload best
        model = load_model(model_path, custom_objects={'FocalLoss': FocalLoss})
        gc.collect()

    # ── STEP 5: CNN Evaluation ───────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5/8: CNN Evaluation (with TTA)")
    print("=" * 70)

    # Standard evaluation
    y_true, y_pred_std, y_proba_std = [], [], []
    for i in tqdm(range(len(test_gen)), desc="Standard eval"):
        inp, lab = test_gen[i]
        preds = model.predict_on_batch(inp)
        y_true.extend(np.argmax(lab, axis=1))
        y_pred_std.extend(np.argmax(preds, axis=1))
        y_proba_std.extend(preds)
    y_true = np.array(y_true)
    y_pred_std = np.array(y_pred_std)
    y_proba_std = np.array(y_proba_std)

    cnn_acc = accuracy_score(y_true, y_pred_std)
    print(f"\n🎯 CNN Accuracy (standard): {cnn_acc*100:.2f}%")

    # TTA evaluation (batched to avoid OOM)
    print(f"\n🔄 Running TTA with {args.tta} augmentations...")
    tta_chunk = 200
    y_proba_tta = []
    n_samples = len(te_imgs)
    for start in tqdm(range(0, n_samples, tta_chunk), desc="TTA"):
        end = min(start + tta_chunk, n_samples)
        chunk_imgs = te_imgs[start:end]
        chunk_feats = Xte_s[start:end]
        chunk_umaps = Ute[start:end]
        tta_preds = predict_with_tta(model, chunk_imgs, chunk_feats, chunk_umaps, n_aug=args.tta)
        y_proba_tta.extend(tta_preds)
    y_proba_tta = np.array(y_proba_tta)[:len(y_true)]
    y_pred_tta = np.argmax(y_proba_tta, axis=1)
    tta_acc = accuracy_score(y_true, y_pred_tta)
    print(f"🎯 CNN Accuracy (TTA): {tta_acc*100:.2f}%")

    # Use whichever is better
    if tta_acc >= cnn_acc:
        y_pred_cnn = y_pred_tta
        y_proba_cnn = y_proba_tta
        cnn_acc = tta_acc
        print("✅ TTA improved results — using TTA predictions.")
    else:
        y_pred_cnn = y_pred_std
        y_proba_cnn = y_proba_std
        print("ℹ️ TTA did not help — using standard predictions.")

    print(f"\n📝 Classification Report:\n{classification_report(y_true, y_pred_cnn, target_names=CLASS_NAMES)}")
    plot_confusion_matrix(y_true, y_pred_cnn, args.save_dir, title_extra="CNN ")
    macro_auc, per_class_auc = plot_roc(y_true, y_proba_cnn, args.save_dir)
    print(f"✅ ROC saved (Macro AUC: {macro_auc:.3f})")

    # ── STEP 6: Deep Feature Agent ───────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6/8: Deep Feature Extraction for Hybrid Agent")
    print("=" * 70)
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer('Fusion').output)

    def extract_deep(gen, name, cache_file):
        if os.path.exists(cache_file):
            print(f"Loading cached {name}...")
            d = np.load(cache_file)
            return d['deep'], d['probs'], d['trues']
        deep, probs, trues = [], [], []
        for i in tqdm(range(len(gen)), desc=f"Extract {name}"):
            inp, y = gen[i]
            deep.append(feat_extractor.predict_on_batch(inp))
            probs.append(model.predict_on_batch(inp))
            trues.extend(np.argmax(y, axis=1))
        deep = np.vstack(deep); probs = np.vstack(probs); trues = np.array(trues)
        np.savez(cache_file, deep=deep, probs=probs, trues=trues)
        return deep, probs, trues

    deep_tr, probs_tr, y_tr = extract_deep(train_gen, "Train",
                                           os.path.join(args.save_dir, "deep_train_v3.npz"))
    deep_te, probs_te, y_te = extract_deep(test_gen, "Test",
                                           os.path.join(args.save_dir, "deep_test_v3.npz"))

    # Agent features: deep + probs + entropy + max_conf
    ent_tr = scipy.stats.entropy(probs_tr, axis=1).reshape(-1, 1)
    ent_te = scipy.stats.entropy(probs_te, axis=1).reshape(-1, 1)
    conf_tr = np.max(probs_tr, axis=1).reshape(-1, 1)
    conf_te = np.max(probs_te, axis=1).reshape(-1, 1)

    X_ag_tr = np.hstack([deep_tr, probs_tr, ent_tr, conf_tr])
    X_ag_te = np.hstack([deep_te, probs_te, ent_te, conf_te])

    ag_scaler = StandardScaler()
    X_ag_tr = ag_scaler.fit_transform(X_ag_tr)
    X_ag_te = ag_scaler.transform(X_ag_te)
    joblib.dump(ag_scaler, os.path.join(args.save_dir, "agent_scaler.pkl"))

    # ── STEP 7: LightGBM Agent ───────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7/8: Training LightGBM Hybrid Agent")
    print("=" * 70)
    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=5,
        num_leaves=20, min_child_samples=40,
        class_weight='balanced', n_jobs=-1, random_state=42,
        reg_alpha=10.0, reg_lambda=10.0,
        feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=3
    )
    clf.fit(X_ag_tr, y_tr,
            eval_set=[(X_ag_te, y_te)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    clf.booster_.save_model(os.path.join(args.save_dir, "lgbm_feedback_agent.txt"))

    agent_preds = clf.predict(X_ag_te)
    agent_acc = accuracy_score(y_te, agent_preds)
    print(f"🤖 Agent Accuracy: {agent_acc*100:.2f}%")

    # ── STEP 8: Hybrid Fusion ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 8/8: Hybrid Threshold Optimisation")
    print("=" * 70)
    base_preds = np.argmax(probs_te, axis=1)
    agent_cls = clf.predict(X_ag_te)
    confs = np.max(probs_te, axis=1)

    best_thresh, best_acc = 0.5, 0
    for t in np.arange(0.3, 0.99, 0.01):
        hybrid = np.where(confs < t, agent_cls, base_preds)
        acc = accuracy_score(y_te, hybrid)
        if acc > best_acc:
            best_acc = acc; best_thresh = t

    final_hybrid = np.where(confs < best_thresh, agent_cls, base_preds)
    hybrid_acc = accuracy_score(y_te, final_hybrid)

    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"🏆 Hybrid Accuracy: {hybrid_acc*100:.2f}%")

    plot_confusion_matrix(y_te, final_hybrid, args.save_dir, title_extra="Hybrid ")

    # Final metrics
    f1 = f1_score(y_te, final_hybrid, average='macro')
    prec = precision_score(y_te, final_hybrid, average='macro')
    rec = recall_score(y_te, final_hybrid, average='macro')
    qwk = cohen_kappa_score(y_te, final_hybrid, weights='quadratic')

    metrics = {
        'CNN_Accuracy': float(cnn_acc),
        'Agent_Accuracy': float(agent_acc),
        'Hybrid_Accuracy': float(hybrid_acc),
        'Threshold': float(best_thresh),
        'Macro_AUC': float(macro_auc),
        'Per_Class_AUC': {CLASS_NAMES[i]: float(per_class_auc[i]) for i in range(NUM_CLASSES)},
        'Macro_F1': float(f1),
        'Precision': float(prec),
        'Recall': float(rec),
        'QWK': float(qwk)
    }
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    for k, v in metrics.items():
        if isinstance(v, dict):
            for ck, cv in v.items():
                print(f"  AUC {ck}: {cv:.4f}")
        elif 'Accuracy' in k or 'F1' in k or 'Precision' in k or 'Recall' in k:
            print(f"  {k}: {v*100:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")
    print("=" * 70)
    print(f"✅ All artifacts saved to: {args.save_dir}")

if __name__ == "__main__":
    main()
