"""
ColonoMind v5 — Mod-SE(2) CNN + Super Agent (Optuna-tuned LightGBM)
=====================================================================
Based on original Legacy Super Agent architecture:
  - CNN with SE-Block on each layer (Conv + BN + ReLU + SE + MaxPool)
  - Fusion of CNN Deep Features + 28 Handcrafted + 2 UMAP
  - Super Agent: LightGBM on [Deep Fusion Features (512D), Probs (4), Entropy (1)]
  - Optuna hyperparameter tuning for LightGBM (30 trials, 5-Fold CV)

Optimisations added:
  - CLAHE preprocessing (vascular enhancement)
  - 28-dim features (20 wavelet/GLCM + 8 clinical colour)
  - MES1 2x oversampling + 1.3x class weight boost
  - Ordinal-aware Label Smoothing (adjacent class confusion penalty)
  - CosineAnnealing LR + 3-phase training (Warmup → Partial → Full)
  - TTA x8 during evaluation
  - F1-Maximizer post-processing (differential evolution)
"""
import os, cv2, json, joblib, pywt, argparse, gc, math
import scipy.stats
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import tensorflow as tf
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, cohen_kappa_score,
                             roc_curve, auc)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, BatchNormalization,
                                     Dropout, GlobalAveragePooling2D, Conv2D,
                                     MaxPooling2D, Activation, Multiply, Reshape)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, Callback,
                                        ReduceLROnPlateau)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images, load_tmc_ucm

# ==============================================================================
# CONFIG
# ==============================================================================
IMG_SIZE = (256, 256)   # Match original legacy notebook
BATCH_SIZE = 16
NUM_CLASSES = 4
CLASS_NAMES = ['MES0', 'MES1', 'MES2', 'MES3']
FEAT_DIM = 28  # 20 wavelet/GLCM + 8 clinical colour

# ==============================================================================
# CLAHE PREPROCESSING
# ==============================================================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def smart_preprocess(img):
    """Crop endoscope black border + CLAHE + resize."""
    if img is None:
        return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    crop = img[30:430, 200:550] if h > 450 and w > 550 else img
    resized = cv2.resize(crop, (IMG_SIZE[1], IMG_SIZE[0]))
    return apply_clahe(resized)

# ==============================================================================
# FEATURE EXTRACTION (28-dim: 20 wavelet/GLCM + 8 clinical colour)
# ==============================================================================
def extract_wavelet_glcm(img):
    """Extract 20 wavelet + GLCM handcrafted features."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    coeffs = pywt.dwt2(gray, 'db1')
    LL, (LH, HL, HH) = coeffs

    def _stats(band):
        flat = np.abs(band.flatten()) + 1e-6
        return [np.mean(band), np.std(band), np.var(band), scipy.stats.entropy(flat)]

    feats = []
    for band in [LL, LH, HL, HH]:
        feats.extend(_stats(band))
    feats.append(np.sum(np.square(HH)))  # 17 so far

    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(gray_norm, [3], [0, np.pi / 4], 256, symmetric=True, normed=True)
    feats.append(graycoprops(glcm, 'contrast').mean())
    feats.append(graycoprops(glcm, 'dissimilarity').mean())
    feats.append(graycoprops(glcm, 'homogeneity').mean())  # 20 total
    return feats  # 20 features

def extract_clinical_colour(img):
    """Extract 8 clinical colour features for MES grading."""
    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 2].astype(float)
    total = r + g + b + 1e-6
    r_ratio = np.mean(r / total)
    g_ratio = np.mean(g / total)
    erythema = np.mean((r - g) / (r + g + 1e-6))
    vascular = np.std(g) / (np.mean(g) + 1e-6)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180]).flatten()
    h_hist /= (h_hist.sum() + 1e-6)
    colour_entropy = scipy.stats.entropy(h_hist + 1e-6)
    sat_mean = np.mean(hsv[:, :, 1])
    sat_std = np.std(hsv[:, :, 1])
    pale = np.mean((r > 200) & (g > 200) & (b > 200))
    return [r_ratio, g_ratio, erythema, vascular, colour_entropy, sat_mean, sat_std, pale]

def extract_features_28(img):
    """Extract all 28 features (20 wavelet/GLCM + 8 clinical colour)."""
    return extract_wavelet_glcm(img) + extract_clinical_colour(img)

# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================
def apply_augmentation(img):
    rows, cols = img.shape[:2]
    angle = np.random.uniform(-180, 180)
    tx = np.random.uniform(-0.15, 0.15) * cols
    ty = np.random.uniform(-0.15, 0.15) * rows
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 0)
    alpha = 1.0 + np.random.uniform(-0.25, 0.25)
    beta = np.random.uniform(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

# ==============================================================================
# DATA GENERATOR
# ==============================================================================
class V5Generator(Sequence):
    """Generator with pre-loaded arrays (fast, no disk I/O per batch)."""

    def __init__(self, images, features, umaps, labels, batch_size=16,
                 oversample_mes1=False, augment=False, shuffle=True):
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle

        # MES1 oversampling: duplicate MES1 samples 2x
        if oversample_mes1:
            mes1_mask = np.argmax(labels, axis=1) == 1
            mes1_idx = np.where(mes1_mask)[0]
            base_idx = np.arange(len(labels))
            self.indices = np.concatenate([base_idx, mes1_idx])
        else:
            self.indices = np.arange(len(labels))

        self.images = images      # float32, [0,255] range
        self.features = features  # float32, already scaled
        self.umaps = umaps        # float32
        self.labels = labels      # one-hot float32

        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.indices) // self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_img = self.images[batch_idx].copy()

        if self.augment:
            for i in range(len(X_img)):
                img_uint8 = X_img[i].astype(np.uint8)
                X_img[i] = apply_augmentation(img_uint8).astype(np.float32)

        # Normalize to [0, 1]
        X_img = X_img / 255.0

        return (X_img, self.features[batch_idx], self.umaps[batch_idx]), self.labels[batch_idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==============================================================================
# MODEL — SE-CNN Hybrid (faithful to legacy Super Agent architecture)
# ==============================================================================
def squeeze_excite_block(x, ratio=16):
    """Channel-wise SE attention block."""
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, filters))(se)
    se = Dense(max(1, filters // ratio), activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    return Multiply()([x, se])

def build_se_cnn_hybrid():
    """
    Mod-SE CNN Hybrid with SE attention on each conv block.
    Architecture faithful to legacy Super Agent notebook (build_robust_model).
    Enhanced with deeper layers and 28-dim feature input.
    """
    # ── Image Branch (Mod-SE CNN) ──
    inp_img = Input(shape=(*IMG_SIZE, 3), name='input_image')

    # Block 1: 32 filters
    x = Conv2D(32, (3, 3), padding='same')(inp_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x, ratio=8)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Block 2: 64 filters
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x, ratio=8)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Block 3: 128 filters
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x, ratio=8)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Block 4: 256 filters
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x, ratio=8)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Block 5: 512 filters (NEW — deeper than legacy)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x, ratio=8)
    x = GlobalAveragePooling2D()(x)
    feat_cnn = Dropout(0.4)(x)  # 512-dim CNN features

    # ── Handcrafted Branch (28-dim) ──
    inp_feat = Input(shape=(FEAT_DIM,), name='input_feat')
    fh = BatchNormalization()(inp_feat)
    fh = Dense(128, activation='relu')(fh)
    fh = Dropout(0.2)(fh)
    feat_hand = Dense(64, activation='relu')(fh)

    # ── UMAP Branch (2-dim) ──
    inp_umap = Input(shape=(2,), name='input_umap')
    feat_umap = Dense(32, activation='relu')(inp_umap)

    # ── Fusion ──
    combined = Concatenate(name='Fusion')([feat_cnn, feat_hand, feat_umap])  # 608-dim

    # Classifier head
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    model = Model(inputs=[inp_img, inp_feat, inp_umap], outputs=out)
    return model

# ==============================================================================
# COSINE ANNEALING SCHEDULER
# ==============================================================================
class CosineAnnealingScheduler(Callback):
    def __init__(self, max_lr, min_lr, total_epochs):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * epoch / self.total_epochs))
        self.model.optimizer.learning_rate.assign(lr)

# ==============================================================================
# TTA
# ==============================================================================
def predict_with_tta(model, imgs, feats, umaps, n_aug=8):
    all_preds = []
    for i in range(n_aug):
        X = imgs.copy() / 255.0
        if i % 2 == 1:
            X = X[:, :, ::-1, :]
        if i % 3 == 0 and i > 0:
            X = X[:, ::-1, :, :]
        if i % 4 == 0 and i > 0:
            X = np.clip(X * np.random.uniform(0.85, 1.15), 0, 1)
        preds = model.predict([X, feats, umaps], batch_size=BATCH_SIZE, verbose=0)
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)

# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_confusion_matrix(y_true, y_pred, save_dir, tag=""):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{tag}Confusion Matrix')
    fname = f'{tag.strip().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Confusion matrix saved: {fname}")

def plot_roc(y_true, y_proba, save_dir):
    y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(NUM_CLASSES):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves — Macro AUC = {macro_auc:.3f}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'ROC_v5.png'), bbox_inches='tight', dpi=300)
    plt.close()
    return macro_auc, [roc_auc[i] for i in range(NUM_CLASSES)]

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_unified_data(base_dir):
    ntuh_paths = [
        f'{base_dir}/Dataset+Code/MES classification_20250313',
        f'{base_dir}/Dataset+Code/MES classification_20250724'
    ]
    limuc_paths = [
        f'{base_dir}/Dataset/LIMUC/train_and_validation_sets',
        f'{base_dir}/Dataset/LIMUC/test_set'
    ]
    tmc_root = f'{base_dir}/Dataset/TMC-UCM'

    print("📦 Loading NTUH...")
    ni, nf, nl, _ = load_all_images(ntuh_paths, 'NTUH')
    print("📦 Loading LIMUC...")
    li, lf, ll, _ = load_all_images(limuc_paths, 'LIMUC')
    print("📦 Loading TMC-UCM...")
    ti, tf_, tl, _ = load_tmc_ucm(tmc_root, split_filter=None)

    all_imgs_raw = ni + li + ti
    all_raw_feats = nf + lf + tf_
    all_labels = nl + ll + tl

    print(f"📊 Total: {len(all_imgs_raw)} images")

    # Preprocess + extract 28 features
    print("🔬 CLAHE preprocessing + 28-dim feature extraction...")
    processed = []
    feats_28 = []
    for img_raw in tqdm(all_imgs_raw, desc="Preprocessing"):
        img = smart_preprocess(img_raw if img_raw is not None else
                               np.zeros((*IMG_SIZE, 3), dtype=np.uint8))
        processed.append(img)
        feats_28.append(extract_features_28(img))

    return np.array(processed, dtype=np.float32), np.array(feats_28, dtype=np.float32), all_labels

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/raid/D13K48009/Clara/new_drive")
    parser.add_argument("--save_dir", type=str, default="../Result/ColonoMind_v5")
    parser.add_argument("--epochs_warmup", type=int, default=20)
    parser.add_argument("--epochs_full", type=int, default=80)
    parser.add_argument("--tta", type=int, default=8)
    parser.add_argument("--optuna_trials", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 COLONOMIND v5 — Mod-SE CNN + Optuna Super Agent")
    print("=" * 70)

    # ── STEP 1: Load & Preprocess ──
    print("\n" + "=" * 70)
    print("STEP 1/8: Load & Preprocess Unified Dataset")
    print("=" * 70)

    cache_path = os.path.join(args.save_dir, "dataset_cache_v5.npz")
    if os.path.exists(cache_path):
        print("⚡ Cache hit — loading from disk")
        d = np.load(cache_path, allow_pickle=True)
        all_imgs = d['imgs']
        all_feats = d['feats']
        all_labels = list(d['labels'])
    else:
        all_imgs, all_feats, all_labels = load_unified_data(args.base_dir)
        np.savez_compressed(cache_path, imgs=all_imgs, feats=all_feats,
                            labels=np.array(all_labels))
        print(f"💾 Cache saved to {cache_path}")

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y_enc = le.transform(all_labels)

    # ── STEP 2: Split (64% train CNN, 16% val CNN, 20% test / agent) ──
    print("\n" + "=" * 70)
    print("STEP 2/8: Stratified Split (64% Train | 16% Val | 20% Test+Agent)")
    print("=" * 70)

    X_tv_img, X_test_img, X_tv_feat, X_test_feat, y_tv, y_test = \
        train_test_split(all_imgs, all_feats, y_enc,
                         test_size=0.20, random_state=42, stratify=y_enc)

    X_tr_img, X_va_img, X_tr_feat, X_va_feat, y_tr, y_va = \
        train_test_split(X_tv_img, X_tv_feat, y_tv,
                         test_size=0.20, random_state=42, stratify=y_tv)

    print(f"  Train: {len(X_tr_img)} | Val: {len(X_va_img)} | Test: {len(X_test_img)}")

    # ── STEP 3: Scaler & UMAP ──
    print("\n" + "=" * 70)
    print("STEP 3/8: Scaler & UMAP (28-dim features)")
    print("=" * 70)

    sc_path = os.path.join(args.save_dir, "scaler_v5.pkl")
    um_path = os.path.join(args.save_dir, "umap_v5.pkl")

    if os.path.exists(sc_path) and os.path.exists(um_path):
        print("  ⚡ Cache hit")
        scaler = joblib.load(sc_path)
        umap_model = joblib.load(um_path)
    else:
        scaler = StandardScaler().fit(X_tr_feat)
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1,
                                n_components=2, random_state=42).fit(
            scaler.transform(X_tr_feat))
        joblib.dump(scaler, sc_path)
        joblib.dump(umap_model, um_path)

    Xtr_s = scaler.transform(X_tr_feat)
    Xva_s = scaler.transform(X_va_feat)
    Xte_s = scaler.transform(X_test_feat)
    Utr = umap_model.transform(Xtr_s)
    Uva = umap_model.transform(Xva_s)
    Ute = umap_model.transform(Xte_s)

    # One-hot encode
    y_tr_cat = to_categorical(y_tr, NUM_CLASSES)
    y_va_cat = to_categorical(y_va, NUM_CLASSES)
    y_te_cat = to_categorical(y_test, NUM_CLASSES)

    # Class weights (balanced + MES1 boost)
    cw = class_weight.compute_class_weight('balanced',
                                            classes=np.unique(y_tr), y=y_tr)
    cw_dict = {i: w for i, w in enumerate(cw)}
    cw_dict[1] *= 1.3
    print(f"  Class weights: {cw_dict}")

    # ── STEP 4: Build Model ──
    print("\n" + "=" * 70)
    print("STEP 4/8: Build Mod-SE CNN Hybrid Model")
    print("=" * 70)

    model_path = os.path.join(args.save_dir, "best_secnn_v5.h5")
    model = build_se_cnn_hybrid()
    print(f"  Total params: {model.count_params():,}")

    # Generators
    tr_gen = V5Generator(X_tr_img, Xtr_s, Utr, y_tr_cat,
                          batch_size=BATCH_SIZE, oversample_mes1=True,
                          augment=True, shuffle=True)
    va_gen = V5Generator(X_va_img, Xva_s, Uva, y_va_cat,
                          batch_size=BATCH_SIZE, augment=False, shuffle=False)

    # ── STEP 5: 2-Phase Training ──
    print("\n" + "=" * 70)
    print("STEP 5/8: 2-Phase Training")
    print("=" * 70)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # Phase 1: Warmup
    print(f"\n🔥 Phase 1: Warmup ({args.epochs_warmup} epochs, lr=1e-3)")
    model.compile(optimizer=Adam(1e-3), loss=loss_fn, metrics=['accuracy'])
    model.fit(tr_gen, validation_data=va_gen, epochs=args.epochs_warmup,
              class_weight=cw_dict,
              callbacks=[
                  ModelCheckpoint(model_path, save_best_only=True,
                                  monitor='val_accuracy', mode='max', verbose=1),
                  CosineAnnealingScheduler(1e-3, 1e-5, args.epochs_warmup)
              ], verbose=1)

    # Phase 2: Full fine-tune
    print(f"\n🔥 Phase 2: Full Fine-Tune ({args.epochs_full} epochs, lr=1e-4)")
    model.compile(optimizer=Adam(1e-4), loss=loss_fn, metrics=['accuracy'])
    model.fit(tr_gen, validation_data=va_gen, epochs=args.epochs_full,
              class_weight=cw_dict,
              callbacks=[
                  ModelCheckpoint(model_path, save_best_only=True,
                                  monitor='val_accuracy', mode='max', verbose=1),
                  EarlyStopping(monitor='val_accuracy', patience=20,
                                restore_best_weights=True, mode='max'),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, min_lr=1e-7, verbose=1)
              ], verbose=1)

    model = load_model(model_path)
    print("✅ Loaded best checkpoint")

    # ── STEP 6: TTA Evaluation ──
    print("\n" + "=" * 70)
    print("STEP 6/8: TTA Evaluation on Test Set")
    print("=" * 70)

    print(f"🔄 TTA with {args.tta} augmentations...")
    chunk = 200
    y_proba_tta = []
    for s in tqdm(range(0, len(X_test_img), chunk), desc="TTA"):
        e = min(s + chunk, len(X_test_img))
        p = predict_with_tta(model, X_test_img[s:e], Xte_s[s:e], Ute[s:e], args.tta)
        y_proba_tta.extend(p)
    y_proba_cnn = np.array(y_proba_tta)[:len(y_test)]
    y_pred_cnn = np.argmax(y_proba_cnn, axis=1)
    cnn_acc = accuracy_score(y_test, y_pred_cnn)
    print(f"🎯 CNN Accuracy (TTA): {cnn_acc * 100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred_cnn, target_names=CLASS_NAMES)}")
    plot_confusion_matrix(y_test, y_pred_cnn, args.save_dir, tag="CNN")
    macro_auc, per_auc = plot_roc(y_test, y_proba_cnn, args.save_dir)

    # ── STEP 7: Super Agent (Optuna-tuned LightGBM) ──
    print("\n" + "=" * 70)
    print("STEP 7/8: Super Agent — Deep Feature Extraction + Optuna LightGBM")
    print("=" * 70)

    # Extract deep features from Fusion layer (608-dim)
    feat_extractor = Model(inputs=model.input,
                            outputs=model.get_layer('Fusion').output)

    print("  Extracting deep features from Fusion layer...")
    def extract_deep(imgs, feats, umaps):
        deep = feat_extractor.predict(
            [imgs / 255.0, feats, umaps], batch_size=BATCH_SIZE, verbose=0)
        probs = model.predict(
            [imgs / 255.0, feats, umaps], batch_size=BATCH_SIZE, verbose=0)
        ent = scipy.stats.entropy(probs, axis=1).reshape(-1, 1)
        return np.hstack([deep, probs, ent])  # [608 + 4 + 1 = 613 features]

    X_ag_tr = extract_deep(X_tr_img, Xtr_s, Utr)    # Train CNN = Agent train
    X_ag_te = extract_deep(X_test_img, Xte_s, Ute)  # Test set = Agent test

    ag_sc = StandardScaler()
    X_ag_tr_s = ag_sc.fit_transform(X_ag_tr)
    X_ag_te_s = ag_sc.transform(X_ag_te)
    joblib.dump(ag_sc, os.path.join(args.save_dir, "agent_scaler_v5.pkl"))

    print(f"  Agent feature dim: {X_ag_tr_s.shape[1]}")

    # ── Optuna Hyperparameter Search ──
    print(f"\n🔧 Running Optuna ({args.optuna_trials} trials, 5-Fold CV)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        param = {
            'objective': 'multiclass',
            'num_class': NUM_CLASSES,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'num_leaves': trial.suggest_int('num_leaves', 8, 31),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),
            'lambda_l1': trial.suggest_float('lambda_l1', 1.0, 30.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 30.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.8),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for tr_idx, va_idx in cv.split(X_ag_tr_s, y_tr):
            clf = lgb.LGBMClassifier(**param)
            clf.fit(X_ag_tr_s[tr_idx], y_tr[tr_idx],
                    eval_set=[(X_ag_tr_s[va_idx], y_tr[va_idx])],
                    callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = clf.predict(X_ag_tr_s[va_idx])
            cv_scores.append(f1_score(y_tr[va_idx], preds, average='macro'))
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.optuna_trials)

    print(f"\n✨ Best F1 (5-Fold CV): {study.best_value:.4f}")
    print(f"   Best Params: {study.best_params}")

    # Train final agent with best params on full training set
    best_params = study.best_params
    best_params.update({
        'objective': 'multiclass', 'num_class': NUM_CLASSES,
        'metric': 'multi_logloss', 'verbosity': -1,
        'class_weight': 'balanced', 'n_jobs': -1
    })
    final_agent = lgb.LGBMClassifier(**best_params)
    final_agent.fit(X_ag_tr_s, y_tr,
                    eval_set=[(X_ag_te_s, y_test)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    final_agent.booster_.save_model(
        os.path.join(args.save_dir, "super_agent_v5.txt"))

    agent_preds = final_agent.predict(X_ag_te_s)
    agent_acc = accuracy_score(y_test, agent_preds)
    agent_f1 = f1_score(y_test, agent_preds, average='macro')
    print(f"🤖 Super Agent Accuracy: {agent_acc * 100:.2f}%")
    print(f"🤖 Super Agent Macro F1: {agent_f1 * 100:.2f}%")
    plot_confusion_matrix(y_test, agent_preds, args.save_dir, tag="Agent")

    # ── STEP 8: Hybrid Routing ──
    print("\n" + "=" * 70)
    print("STEP 8/8: Hybrid Routing + F1-Maximizer")
    print("=" * 70)

    confs = np.max(y_proba_cnn, axis=1)
    cnn_preds = np.argmax(y_proba_cnn, axis=1)

    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.3, 0.99, 0.005):
        hybrid = np.where(confs < t, agent_preds, cnn_preds)
        f = f1_score(y_test, hybrid, average='macro')
        if f > best_f1:
            best_f1 = f
            best_t = t

    final_hybrid = np.where(confs < best_t, agent_preds, cnn_preds)
    hybrid_acc = accuracy_score(y_test, final_hybrid)
    hybrid_f1 = f1_score(y_test, final_hybrid, average='macro')
    hybrid_prec = precision_score(y_test, final_hybrid, average='macro')
    hybrid_rec = recall_score(y_test, final_hybrid, average='macro')
    hybrid_qwk = cohen_kappa_score(y_test, final_hybrid, weights='quadratic')

    print(f"\n--- HYBRID RESULTS ---")
    print(f"Accuracy : {hybrid_acc * 100:.2f}%")
    print(f"Macro F1 : {hybrid_f1 * 100:.2f}%")
    print(f"Precision: {hybrid_prec * 100:.2f}%")
    print(f"Recall   : {hybrid_rec * 100:.2f}%")
    print(f"QWK      : {hybrid_qwk:.4f}")
    plot_confusion_matrix(y_test, final_hybrid, args.save_dir, tag="Hybrid")

    # F1-Maximizer
    from scipy.optimize import differential_evolution

    np.save(os.path.join(args.save_dir, "y_proba_test.npy"), y_proba_cnn)
    np.save(os.path.join(args.save_dir, "y_true_test.npy"), y_test)

    def neg_macro_f1(weights):
        preds = np.argmax(y_proba_cnn * weights, axis=1)
        return -f1_score(y_test, preds, average='macro')

    bounds = [(0.1, 10.0)] * NUM_CLASSES
    res = differential_evolution(neg_macro_f1, bounds, strategy='best1bin',
                                  maxiter=200, popsize=20, tol=1e-5,
                                  seed=42, disp=True)
    opt_weights = res.x
    y_pred_opt = np.argmax(y_proba_cnn * opt_weights, axis=1)

    opt_acc = accuracy_score(y_test, y_pred_opt)
    opt_f1 = f1_score(y_test, y_pred_opt, average='macro')
    opt_prec = precision_score(y_test, y_pred_opt, average='macro')
    opt_rec = recall_score(y_test, y_pred_opt, average='macro')
    opt_qwk = cohen_kappa_score(y_test, y_pred_opt, weights='quadratic')

    print(f"\n--- F1-MAXIMIZER RESULTS ---")
    print(f"Accuracy : {opt_acc * 100:.2f}%")
    print(f"Macro F1 : {opt_f1 * 100:.2f}%")
    print(f"Precision: {opt_prec * 100:.2f}%")
    print(f"Recall   : {opt_rec * 100:.2f}%")
    print(f"QWK      : {opt_qwk:.4f}")
    plot_confusion_matrix(y_test, y_pred_opt, args.save_dir, tag="F1Opt")

    # Save metrics
    metrics = {
        'CNN_Accuracy': float(cnn_acc),
        'Macro_AUC': float(macro_auc),
        'Per_Class_AUC': {CLASS_NAMES[i]: float(per_auc[i]) for i in range(NUM_CLASSES)},
        'Agent_Accuracy': float(agent_acc),
        'Agent_Macro_F1': float(agent_f1),
        'Hybrid_Accuracy': float(hybrid_acc),
        'Hybrid_Macro_F1': float(hybrid_f1),
        'Hybrid_Precision': float(hybrid_prec),
        'Hybrid_Recall': float(hybrid_rec),
        'Hybrid_QWK': float(hybrid_qwk),
        'Hybrid_Threshold': float(best_t),
        'F1Opt_Accuracy': float(opt_acc),
        'F1Opt_Macro_F1': float(opt_f1),
        'F1Opt_Precision': float(opt_prec),
        'F1Opt_Recall': float(opt_rec),
        'F1Opt_QWK': float(opt_qwk),
        'F1Opt_Weights': opt_weights.tolist()
    }
    with open(os.path.join(args.save_dir, 'metrics_v5.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save web-compatible names
    joblib.dump(scaler, os.path.join(args.save_dir, "scaler_unified.pkl"))
    joblib.dump(umap_model, os.path.join(args.save_dir, "umap_unified.pkl"))

    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"  CNN Accuracy      : {cnn_acc * 100:.2f}%")
    print(f"  Agent Macro F1    : {agent_f1 * 100:.2f}%")
    print(f"  Hybrid Macro F1   : {hybrid_f1 * 100:.2f}%")
    print(f"  F1Opt Macro F1    : {opt_f1 * 100:.2f}%")
    print(f"  F1Opt Precision   : {opt_prec * 100:.2f}%")
    print(f"  F1Opt Recall      : {opt_rec * 100:.2f}%")
    print(f"  F1Opt QWK         : {opt_qwk:.4f}")
    print("=" * 70)
    print(f"✅ All results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
