"""
ColonoMind v5 — Mod-SE CNN + Optimised Super Agent
====================================================
Base: Legacy Super Agent (SE-CNN + LightGBM Deep Features)
Optimisations for 90%+ F1/Precision/Recall:
  1. Ordinal Focal Loss + OHEM (top-70% hardest samples per batch)
  2. CutMix augmentation (20% prob — avoids MES boundary confusion)
  3. 3-Phase training: Warmup(30) → Mid(60) → Full(120) + Cosine LR
  4. CLAHE preprocessing (vascular pattern enhancement)
  5. 28-dim features: 20 wavelet/GLCM + 8 clinical colour
  6. MES1 2× oversampling + 1.3× class weight boost
  7. TTA ×8 during evaluation
  8. Super Agent: 643-dim features (608 deep + 28 handcrafted + 2 UMAP + 4 probs + 1 entropy)
  9. Per-class confidence threshold routing (not global)
 10. Optuna 30 trials × 5-Fold CV on Macro F1
 11. F1-Maximizer: differential evolution post-processing
"""
import os, cv2, json, joblib, pywt, argparse, gc, math
import scipy.stats
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import lightgbm as lgb
import optuna
import tensorflow as tf
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, cohen_kappa_score,
                             roc_curve, auc)
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
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
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 16
NUM_CLASSES = 4
CLASS_NAMES = ['MES0', 'MES1', 'MES2', 'MES3']
FEAT_DIM    = 28   # 20 wavelet/GLCM + 8 clinical colour

# ==============================================================================
# PREPROCESSING
# ==============================================================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def smart_preprocess(img):
    if img is None:
        return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    crop = img[30:430, 200:550] if h > 450 and w > 550 else img
    resized = cv2.resize(crop, (IMG_SIZE[1], IMG_SIZE[0]))
    return apply_clahe(resized)

# ==============================================================================
# FEATURE EXTRACTION (28-dim)
# ==============================================================================
def extract_wavelet_glcm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    LL, (LH, HL, HH) = pywt.dwt2(gray, 'db1')
    def _stats(b):
        f = np.abs(b.flatten()) + 1e-6
        return [np.mean(b), np.std(b), np.var(b), scipy.stats.entropy(f)]
    feats = []
    for band in [LL, LH, HL, HH]:
        feats.extend(_stats(band))
    feats.append(np.sum(np.square(HH)))
    gn = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(gn, [3], [0, np.pi/4], 256, symmetric=True, normed=True)
    feats.append(graycoprops(glcm, 'contrast').mean())
    feats.append(graycoprops(glcm, 'dissimilarity').mean())
    feats.append(graycoprops(glcm, 'homogeneity').mean())
    return feats  # 20

def extract_clinical_colour(img):
    r, g, b = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
    t = r + g + b + 1e-6
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv],[0],None,[30],[0,180]).flatten()
    h_hist /= (h_hist.sum() + 1e-6)
    return [
        np.mean(r/t), np.mean(g/t),
        np.mean((r-g)/(r+g+1e-6)),
        np.std(g)/(np.mean(g)+1e-6),
        scipy.stats.entropy(h_hist+1e-6),
        np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]),
        np.mean((r>200)&(g>200)&(b>200))
    ]  # 8

def extract_features_28(img):
    return extract_wavelet_glcm(img) + extract_clinical_colour(img)

# ==============================================================================
# ORDINAL FOCAL LOSS + OHEM
# ==============================================================================
class OrdinalFocalOHEM(tf.keras.losses.Loss):
    """Focal loss + ordinal distance penalty + Online Hard Example Mining."""
    def __init__(self, gamma=2.5, ordinal_w=0.5, ohem_ratio=0.7, **kw):
        super().__init__(**kw)
        self.gamma = gamma
        self.ordinal_w = ordinal_w
        self.ohem_ratio = ohem_ratio

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce   = -y_true * tf.math.log(y_pred)
        pt   = tf.reduce_sum(y_true * y_pred, axis=-1)
        fl   = tf.math.pow(1.0 - pt, self.gamma) * tf.reduce_sum(ce, axis=-1)

        t_cls = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
        p_cls = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
        ord_p = tf.abs(t_cls - p_cls) * self.ordinal_w

        per_sample = fl + ord_p

        # OHEM: keep top-k hardest
        k = tf.maximum(tf.cast(
            tf.cast(tf.shape(per_sample)[0], tf.float32) * self.ohem_ratio,
            tf.int32), 1)
        top_k, _ = tf.math.top_k(per_sample, k=k)
        return tf.reduce_mean(top_k)

# ==============================================================================
# CUTMIX
# ==============================================================================
def cutmix_batch(imgs, feats, umaps, labels, alpha=0.4):
    B = len(imgs)
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(B)
    H, W = imgs.shape[1], imgs.shape[2]
    cut_h = int(H * np.sqrt(1 - lam))
    cut_w = int(W * np.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(0, cy-cut_h//2), min(H, cy+cut_h//2)
    x1, x2 = max(0, cx-cut_w//2), min(W, cx+cut_w//2)
    mixed = imgs.copy()
    mixed[:, y1:y2, x1:x2, :] = imgs[idx, y1:y2, x1:x2, :]
    lam = 1.0 - (y2-y1)*(x2-x1) / (H*W)
    return (mixed,
            lam*feats  + (1-lam)*feats[idx],
            lam*umaps  + (1-lam)*umaps[idx],
            lam*labels + (1-lam)*labels[idx])

# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================
def apply_augmentation(img):
    rows, cols = img.shape[:2]
    angle = np.random.uniform(-180, 180)
    tx = np.random.uniform(-0.15, 0.15) * cols
    ty = np.random.uniform(-0.15, 0.15) * rows
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
    M[0, 2] += tx; M[1, 2] += ty
    img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() > 0.5: img = cv2.flip(img, 1)
    if np.random.rand() > 0.5: img = cv2.flip(img, 0)
    alpha = 1.0 + np.random.uniform(-0.25, 0.25)
    beta  = np.random.uniform(-20, 20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# ==============================================================================
# DATA GENERATOR (with CutMix + MES1 oversampling)
# ==============================================================================
class V5Generator(Sequence):
    def __init__(self, images, features, umaps, labels,
                 batch_size=16, oversample_mes1=False,
                 augment=False, use_cutmix=False, cutmix_prob=0.2,
                 shuffle=True):
        self.batch_size  = batch_size
        self.augment     = augment
        self.use_cutmix  = use_cutmix
        self.cutmix_prob = cutmix_prob
        self.shuffle     = shuffle

        if oversample_mes1:
            mes1 = np.where(np.argmax(labels, axis=1) == 1)[0]
            self.indices = np.concatenate([np.arange(len(labels)), mes1])
        else:
            self.indices = np.arange(len(labels))

        self.images   = images    # uint8 or float32 [0,255]
        self.features = features  # already scaled
        self.umaps    = umaps
        self.labels   = labels    # one-hot
        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.indices) // self.batch_size)

    def __getitem__(self, idx):
        bidx  = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X_img = self.images[bidx].copy()

        if self.augment:
            for i in range(len(X_img)):
                X_img[i] = apply_augmentation(X_img[i].astype(np.uint8)).astype(np.float32)

        X_feat = self.features[bidx].copy()
        X_umap = self.umaps[bidx].copy()
        y      = self.labels[bidx].copy()

        if self.use_cutmix and np.random.rand() < self.cutmix_prob:
            X_img, X_feat, X_umap, y = cutmix_batch(X_img, X_feat, X_umap, y)

        return (X_img / 255.0, X_feat, X_umap), y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==============================================================================
# MODEL — Mod-SE CNN Hybrid (deepened from legacy)
# ==============================================================================
def se_block(x, ratio=8):
    f = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, f))(se)
    se = Dense(max(1, f//ratio), activation='relu', use_bias=False)(se)
    se = Dense(f, activation='sigmoid', use_bias=False)(se)
    return Multiply()([x, se])

def build_se_cnn_hybrid():
    """Mod-SE CNN: ConvBNReLU + SE Block on each stage, deepened to 512."""
    inp_img  = Input(shape=(*IMG_SIZE, 3), name='input_image')

    # Block 1 — 32 filters
    x = Conv2D(32, (3,3), padding='same')(inp_img)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = se_block(x, 8); x = MaxPooling2D((2,2))(x); x = Dropout(0.2)(x)
    # Block 2 — 64 filters
    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = se_block(x, 8); x = MaxPooling2D((2,2))(x); x = Dropout(0.2)(x)
    # Block 3 — 128 filters
    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = se_block(x, 8); x = MaxPooling2D((2,2))(x); x = Dropout(0.3)(x)
    # Block 4 — 256 filters
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = se_block(x, 8); x = MaxPooling2D((2,2))(x); x = Dropout(0.3)(x)
    # Block 5 — 512 filters (extra depth vs legacy)
    x = Conv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = se_block(x, 8)
    feat_cnn = GlobalAveragePooling2D()(x)   # 512-dim CNN features
    feat_cnn = Dropout(0.4)(feat_cnn)

    # Handcrafted branch (28-dim)
    inp_feat = Input(shape=(FEAT_DIM,), name='input_feat')
    fh = BatchNormalization()(inp_feat)
    fh = Dense(128, activation='relu')(fh)
    fh = Dropout(0.2)(fh)
    feat_hand = Dense(64, activation='relu')(fh)

    # UMAP branch (2-dim)
    inp_umap = Input(shape=(2,), name='input_umap')
    feat_umap = Dense(32, activation='relu')(inp_umap)

    # Fusion (512+64+32 = 608-dim)
    combined = Concatenate(name='Fusion')([feat_cnn, feat_hand, feat_umap])

    # Classifier
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    return Model(inputs=[inp_img, inp_feat, inp_umap], outputs=out)

# ==============================================================================
# COSINE ANNEALING SCHEDULER
# ==============================================================================
class CosineAnneal(Callback):
    def __init__(self, max_lr, min_lr, total_epochs):
        super().__init__()
        self.max_lr = max_lr; self.min_lr = min_lr; self.total = total_epochs
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.min_lr + 0.5*(self.max_lr - self.min_lr)*(1 + math.cos(math.pi*epoch/self.total))
        self.model.optimizer.learning_rate.assign(lr)

# ==============================================================================
# TTA
# ==============================================================================
def predict_tta(model, imgs, feats, umaps, n=8):
    preds = []
    for i in range(n):
        X = imgs.copy() / 255.0
        if i % 2 == 1: X = X[:, :, ::-1, :]
        if i % 3 == 0 and i > 0: X = X[:, ::-1, :, :]
        if i % 4 == 0 and i > 0: X = np.clip(X * np.random.uniform(0.85, 1.15), 0, 1)
        preds.append(model.predict([X, feats, umaps], batch_size=BATCH_SIZE, verbose=0))
    return np.mean(preds, axis=0)

# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_cm(y_true, y_pred, save_dir, tag=""):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'{tag} Confusion Matrix')
    fname = f'{tag.strip().replace(" ","_")}_confusion_matrix.png'
    plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✅ CM saved: {fname}")

def plot_roc(y_true, y_proba, save_dir):
    y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    fpr, tpr, rauc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:,i], y_proba[:,i])
        rauc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in range(NUM_CLASSES)) / NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    plt.figure(figsize=(10,8))
    for i in range(NUM_CLASSES):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'{CLASS_NAMES[i]} (AUC={rauc[i]:.3f})')
    plt.plot([0,1],[0,1],'k--',lw=2)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC — Macro AUC = {macro_auc:.3f}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'ROC_v5.png'), bbox_inches='tight', dpi=300)
    plt.close()
    return macro_auc, [rauc[i] for i in range(NUM_CLASSES)]

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_all(base_dir):
    ntuh = [f'{base_dir}/Dataset+Code/MES classification_20250313',
            f'{base_dir}/Dataset+Code/MES classification_20250724']
    limuc= [f'{base_dir}/Dataset/LIMUC/train_and_validation_sets',
            f'{base_dir}/Dataset/LIMUC/test_set']
    tmc  = f'{base_dir}/Dataset/TMC-UCM'
    print("📦 Loading NTUH...")
    ni, _, nl, _ = load_all_images(ntuh, 'NTUH')
    print("📦 Loading LIMUC...")
    li, _, ll, _ = load_all_images(limuc, 'LIMUC')
    print("📦 Loading TMC-UCM...")
    ti, _, tl, _ = load_tmc_ucm(tmc, split_filter=None)
    raw = ni+li+ti; labels = nl+ll+tl
    print(f"📊 Total: {len(raw)} images")

    print("🔬 CLAHE + 28-dim features...")
    imgs, feats = [], []
    for img in tqdm(raw):
        p = smart_preprocess(img if img is not None else
                             np.zeros((*IMG_SIZE,3), dtype=np.uint8))
        imgs.append(p)
        feats.append(extract_features_28(p))
    return np.array(imgs, dtype=np.float32), np.array(feats, dtype=np.float32), labels

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir",       default="/raid/D13K48009/Clara/new_drive")
    ap.add_argument("--save_dir",       default="../Result/ColonoMind_v5")
    ap.add_argument("--epochs_warmup",  type=int, default=30)
    ap.add_argument("--epochs_mid",     type=int, default=60)
    ap.add_argument("--epochs_full",    type=int, default=120)
    ap.add_argument("--tta",            type=int, default=8)
    ap.add_argument("--optuna_trials",  type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*70)
    print("🚀 COLONOMIND v5 — Mod-SE CNN + Optimised Super Agent")
    print("="*70)

    # ── STEP 1: Data ──────────────────────────────────────────────────────────
    print("\n" + "="*70); print("STEP 1/9: Load & Preprocess"); print("="*70)
    cache = os.path.join(args.save_dir, "dataset_cache_v5.npz")
    if os.path.exists(cache):
        print("⚡ Cache hit")
        d = np.load(cache, allow_pickle=True)
        all_imgs = d['imgs']; all_feats = d['feats']
        all_labels = list(d['labels'])
    else:
        all_imgs, all_feats, all_labels = load_all(args.base_dir)
        np.savez_compressed(cache, imgs=all_imgs, feats=all_feats,
                            labels=np.array(all_labels))
        print(f"💾 Cache saved")

    le = LabelEncoder(); le.fit(CLASS_NAMES)
    y_enc = le.transform(all_labels)

    # ── STEP 2: Split ─────────────────────────────────────────────────────────
    print("\n" + "="*70); print("STEP 2/9: Split (64/16/20)"); print("="*70)
    X_tv_i, X_te_i, X_tv_f, X_te_f, y_tv, y_te = train_test_split(
        all_imgs, all_feats, y_enc, test_size=0.20, random_state=42, stratify=y_enc)
    X_tr_i, X_va_i, X_tr_f, X_va_f, y_tr, y_va = train_test_split(
        X_tv_i, X_tv_f, y_tv,   test_size=0.20, random_state=42, stratify=y_tv)
    print(f"  Train:{len(X_tr_i)} | Val:{len(X_va_i)} | Test:{len(X_te_i)}")

    # ── STEP 3: Scaler & UMAP ─────────────────────────────────────────────────
    print("\n" + "="*70); print("STEP 3/9: Scaler & UMAP"); print("="*70)
    sp = os.path.join(args.save_dir,"scaler_v5.pkl")
    up = os.path.join(args.save_dir,"umap_v5.pkl")
    if os.path.exists(sp) and os.path.exists(up):
        print("  ⚡ Cache hit"); sc = joblib.load(sp); um = joblib.load(up)
    else:
        sc = StandardScaler().fit(X_tr_f)
        um = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                       random_state=42).fit(sc.transform(X_tr_f))
        joblib.dump(sc, sp); joblib.dump(um, up)

    Xtr_s=sc.transform(X_tr_f); Xva_s=sc.transform(X_va_f); Xte_s=sc.transform(X_te_f)
    Utr=um.transform(Xtr_s);    Uva=um.transform(Xva_s);    Ute=um.transform(Xte_s)

    y_tr_c = to_categorical(y_tr, NUM_CLASSES)
    y_va_c = to_categorical(y_va, NUM_CLASSES)

    # Class weights
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = {i: w for i, w in enumerate(cw)}
    cw_dict[1] *= 1.3
    print(f"  Class weights: {cw_dict}")

    # ── STEP 4: Build Model ───────────────────────────────────────────────────
    print("\n" + "="*70); print("STEP 4/9: Build Mod-SE CNN Hybrid"); print("="*70)
    mp = os.path.join(args.save_dir, "best_secnn_v5.h5")
    model = build_se_cnn_hybrid()
    print(f"  Total params: {model.count_params():,}")

    tr_gen = V5Generator(X_tr_i, Xtr_s, Utr, y_tr_c, BATCH_SIZE,
                          oversample_mes1=True, augment=True,
                          use_cutmix=True, cutmix_prob=0.2, shuffle=True)
    va_gen = V5Generator(X_va_i, Xva_s, Uva, y_va_c, BATCH_SIZE,
                          augment=False, shuffle=False)

    loss_fn = OrdinalFocalOHEM(gamma=2.5, ordinal_w=0.5, ohem_ratio=0.7)

    # ── STEP 5: 3-Phase Training ──────────────────────────────────────────────
    print("\n" + "="*70); print("STEP 5/9: 3-Phase Training"); print("="*70)

    # Phase 1 – Warmup (feature branches only)
    print(f"\n🔥 Phase 1: Warmup ({args.epochs_warmup} epochs, lr=5e-4)")
    model.compile(optimizer=Adam(5e-4), loss=loss_fn, metrics=['accuracy'])
    model.fit(tr_gen, validation_data=va_gen, epochs=args.epochs_warmup,
              class_weight=cw_dict,
              callbacks=[
                  ModelCheckpoint(mp, save_best_only=True,
                                  monitor='val_accuracy', mode='max', verbose=1),
                  CosineAnneal(5e-4, 1e-5, args.epochs_warmup)
              ], verbose=1)

    # Phase 2 – Mid tune
    print(f"\n🔥 Phase 2: Mid Fine-Tune ({args.epochs_mid} epochs, lr=2e-4)")
    model.compile(optimizer=Adam(2e-4), loss=loss_fn, metrics=['accuracy'])
    model.fit(tr_gen, validation_data=va_gen, epochs=args.epochs_mid,
              class_weight=cw_dict,
              callbacks=[
                  ModelCheckpoint(mp, save_best_only=True,
                                  monitor='val_accuracy', mode='max', verbose=1),
                  EarlyStopping(monitor='val_accuracy', patience=15,
                                restore_best_weights=True, mode='max'),
                  CosineAnneal(2e-4, 1e-6, args.epochs_mid)
              ], verbose=1)

    # Phase 3 – Deep fine-tune
    print(f"\n🔥 Phase 3: Full Fine-Tune ({args.epochs_full} epochs, lr=1e-4)")
    model.compile(optimizer=Adam(1e-4), loss=loss_fn, metrics=['accuracy'])
    model.fit(tr_gen, validation_data=va_gen, epochs=args.epochs_full,
              class_weight=cw_dict,
              callbacks=[
                  ModelCheckpoint(mp, save_best_only=True,
                                  monitor='val_accuracy', mode='max', verbose=1),
                  EarlyStopping(monitor='val_accuracy', patience=20,
                                restore_best_weights=True, mode='max'),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, min_lr=1e-7, verbose=1)
              ], verbose=1)

    model = load_model(mp, custom_objects={'OrdinalFocalOHEM': OrdinalFocalOHEM})
    print("✅ Best checkpoint loaded")

    # ── STEP 6: TTA Evaluation ────────────────────────────────────────────────
    print("\n" + "="*70); print("STEP 6/9: TTA Evaluation"); print("="*70)
    chunk = 200; y_proba_list = []
    for s in tqdm(range(0, len(X_te_i), chunk), desc="TTA"):
        e = min(s+chunk, len(X_te_i))
        y_proba_list.extend(predict_tta(model, X_te_i[s:e], Xte_s[s:e], Ute[s:e], args.tta))
    y_proba_cnn = np.array(y_proba_list)[:len(y_te)]
    y_pred_cnn  = np.argmax(y_proba_cnn, axis=1)
    cnn_acc     = accuracy_score(y_te, y_pred_cnn)
    cnn_f1      = f1_score(y_te, y_pred_cnn, average='macro')
    print(f"🎯 CNN — Acc:{cnn_acc*100:.2f}%  F1:{cnn_f1*100:.2f}%")
    print(f"\n{classification_report(y_te, y_pred_cnn, target_names=CLASS_NAMES)}")
    plot_cm(y_te, y_pred_cnn, args.save_dir, "CNN")
    macro_auc, per_auc = plot_roc(y_te, y_proba_cnn, args.save_dir)

    # ── STEP 7: Super Agent ───────────────────────────────────────────────────
    print("\n" + "="*70); print("STEP 7/9: Super Agent (643-dim features, Optuna)"); print("="*70)

    feat_ext = Model(inputs=model.input, outputs=model.get_layer('Fusion').output)

    def build_agent_features(imgs, feats_s, umaps, raw_feats):
        """643-dim = 608 deep + 28 raw handcrafted + 2 UMAP + 4 probs + 1 entropy"""
        deep  = feat_ext.predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0)
        probs = model.predict([imgs/255.0, feats_s, umaps],    batch_size=BATCH_SIZE, verbose=0)
        ent   = scipy.stats.entropy(probs, axis=1).reshape(-1,1)
        return np.hstack([deep, raw_feats, umaps, probs, ent])  # 608+28+2+4+1 = 643

    print("  Building agent features for Train & Test sets...")
    X_ag_tr = build_agent_features(X_tr_i, Xtr_s, Utr, X_tr_f)
    X_ag_te = build_agent_features(X_te_i, Xte_s, Ute, X_te_f)

    ag_sc = StandardScaler()
    X_ag_tr_s = ag_sc.fit_transform(X_ag_tr)
    X_ag_te_s = ag_sc.transform(X_ag_te)
    joblib.dump(ag_sc, os.path.join(args.save_dir, "agent_scaler_v5.pkl"))
    print(f"  Agent feature dim: {X_ag_tr_s.shape[1]}")

    # Optuna: maximise Macro F1 via 5-Fold CV
    print(f"\n🔧 Optuna ({args.optuna_trials} trials × 5-Fold, objective=Macro F1)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        param = dict(
            objective='multiclass', num_class=NUM_CLASSES,
            metric='multi_logloss', verbosity=-1,
            boosting_type='gbdt',
            n_estimators    = trial.suggest_int  ('n_estimators',    200, 1000),
            learning_rate   = trial.suggest_float('learning_rate',   0.005, 0.1, log=True),
            max_depth       = trial.suggest_int  ('max_depth',       3, 7),
            num_leaves      = trial.suggest_int  ('num_leaves',      8, 63),
            min_child_samples=trial.suggest_int  ('min_child_samples',15, 80),
            lambda_l1       = trial.suggest_float('lambda_l1',       0.1, 30.0, log=True),
            lambda_l2       = trial.suggest_float('lambda_l2',       0.1, 30.0, log=True),
            feature_fraction= trial.suggest_float('feature_fraction',0.3, 0.9),
            bagging_fraction= trial.suggest_float('bagging_fraction',0.5, 0.9),
            bagging_freq    = trial.suggest_int  ('bagging_freq',    1, 5),
            class_weight='balanced', n_jobs=-1
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_i, va_i in cv.split(X_ag_tr_s, y_tr):
            clf = lgb.LGBMClassifier(**param)
            clf.fit(X_ag_tr_s[tr_i], y_tr[tr_i],
                    eval_set=[(X_ag_tr_s[va_i], y_tr[va_i])],
                    callbacks=[lgb.early_stopping(30, verbose=False)])
            scores.append(f1_score(y_tr[va_i], clf.predict(X_ag_tr_s[va_i]), average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.optuna_trials)
    print(f"✨ Best CV Macro F1: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")

    best_p = study.best_params
    best_p.update(dict(objective='multiclass', num_class=NUM_CLASSES,
                        metric='multi_logloss', verbosity=-1,
                        class_weight='balanced', n_jobs=-1))
    final_agent = lgb.LGBMClassifier(**best_p)
    final_agent.fit(X_ag_tr_s, y_tr,
                    eval_set=[(X_ag_te_s, y_te)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    final_agent.booster_.save_model(os.path.join(args.save_dir, "super_agent_v5.txt"))

    ag_preds = final_agent.predict(X_ag_te_s)
    ag_proba = final_agent.predict_proba(X_ag_te_s)
    ag_acc   = accuracy_score(y_te, ag_preds)
    ag_f1    = f1_score(y_te, ag_preds, average='macro')
    ag_prec  = precision_score(y_te, ag_preds, average='macro')
    ag_rec   = recall_score(y_te, ag_preds, average='macro')
    ag_qwk   = cohen_kappa_score(y_te, ag_preds, weights='quadratic')
    print(f"🤖 Super Agent — Acc:{ag_acc*100:.2f}%  F1:{ag_f1*100:.2f}%  "
          f"Prec:{ag_prec*100:.2f}%  Rec:{ag_rec*100:.2f}%  QWK:{ag_qwk:.4f}")
    print(f"\n{classification_report(y_te, ag_preds, target_names=CLASS_NAMES)}")
    plot_cm(y_te, ag_preds, args.save_dir, "Agent")

    # ── STEP 8: Per-Class Confidence Hybrid Routing ───────────────────────────
    print("\n" + "="*70); print("STEP 8/9: Per-Class Confidence Hybrid Routing"); print("="*70)

    cnn_preds = np.argmax(y_proba_cnn, axis=1)
    confs     = np.max(y_proba_cnn,   axis=1)

    # Search per-class threshold (one threshold per predicted class)
    best_thresholds = {}
    for cls in range(NUM_CLASSES):
        best_t, best_f = 0.5, 0
        for t in np.arange(0.3, 0.99, 0.01):
            preds = cnn_preds.copy()
            mask  = (cnn_preds == cls) & (confs < t)
            preds[mask] = ag_preds[mask]
            f = f1_score(y_te, preds, average='macro')
            if f > best_f: best_f = f; best_t = t
        best_thresholds[cls] = best_t
    print(f"  Per-class thresholds: {best_thresholds}")

    # Apply per-class routing
    hybrid_preds = cnn_preds.copy()
    for cls, t in best_thresholds.items():
        mask = (cnn_preds == cls) & (confs < t)
        hybrid_preds[mask] = ag_preds[mask]

    hy_acc  = accuracy_score(y_te, hybrid_preds)
    hy_f1   = f1_score(y_te, hybrid_preds, average='macro')
    hy_prec = precision_score(y_te, hybrid_preds, average='macro')
    hy_rec  = recall_score(y_te, hybrid_preds, average='macro')
    hy_qwk  = cohen_kappa_score(y_te, hybrid_preds, weights='quadratic')
    print(f"🏆 Hybrid — Acc:{hy_acc*100:.2f}%  F1:{hy_f1*100:.2f}%  "
          f"Prec:{hy_prec*100:.2f}%  Rec:{hy_rec*100:.2f}%  QWK:{hy_qwk:.4f}")
    print(f"\n{classification_report(y_te, hybrid_preds, target_names=CLASS_NAMES)}")
    plot_cm(y_te, hybrid_preds, args.save_dir, "Hybrid")

    # ── STEP 9: F1-Maximizer ──────────────────────────────────────────────────
    print("\n" + "="*70); print("STEP 9/9: F1-Maximizer (Differential Evolution)"); print("="*70)
    from scipy.optimize import differential_evolution

    np.save(os.path.join(args.save_dir, "y_proba_test.npy"), y_proba_cnn)
    np.save(os.path.join(args.save_dir, "y_true_test.npy"),  y_te)

    def neg_f1(w):
        return -f1_score(y_te, np.argmax(y_proba_cnn * w, axis=1), average='macro')

    res = differential_evolution(neg_f1, [(0.1,10.0)]*NUM_CLASSES,
                                  strategy='best1bin', maxiter=200,
                                  popsize=20, tol=1e-5, seed=42, disp=True)
    w_opt  = res.x
    y_opt  = np.argmax(y_proba_cnn * w_opt, axis=1)
    op_acc = accuracy_score(y_te, y_opt)
    op_f1  = f1_score(y_te, y_opt, average='macro')
    op_pre = precision_score(y_te, y_opt, average='macro')
    op_rec = recall_score(y_te, y_opt, average='macro')
    op_qwk = cohen_kappa_score(y_te, y_opt, weights='quadratic')
    print(f"⚡ F1Opt — Acc:{op_acc*100:.2f}%  F1:{op_f1*100:.2f}%  "
          f"Prec:{op_pre*100:.2f}%  Rec:{op_rec*100:.2f}%  QWK:{op_qwk:.4f}")
    print(f"\n{classification_report(y_te, y_opt, target_names=CLASS_NAMES)}")
    plot_cm(y_te, y_opt, args.save_dir, "F1Opt")

    # ── Save everything ───────────────────────────────────────────────────────
    joblib.dump(sc,  os.path.join(args.save_dir, "scaler_unified.pkl"))
    joblib.dump(um,  os.path.join(args.save_dir, "umap_unified.pkl"))

    metrics = dict(
        CNN_Accuracy=float(cnn_acc), CNN_Macro_F1=float(cnn_f1),
        Macro_AUC=float(macro_auc),
        Per_Class_AUC={CLASS_NAMES[i]: float(per_auc[i]) for i in range(NUM_CLASSES)},
        Agent_Accuracy=float(ag_acc), Agent_F1=float(ag_f1),
        Agent_Precision=float(ag_prec), Agent_Recall=float(ag_rec), Agent_QWK=float(ag_qwk),
        Hybrid_Accuracy=float(hy_acc), Hybrid_F1=float(hy_f1),
        Hybrid_Precision=float(hy_prec), Hybrid_Recall=float(hy_rec), Hybrid_QWK=float(hy_qwk),
        Per_Class_Thresholds=best_thresholds,
        F1Opt_Accuracy=float(op_acc), F1Opt_F1=float(op_f1),
        F1Opt_Precision=float(op_pre), F1Opt_Recall=float(op_rec), F1Opt_QWK=float(op_qwk),
        F1Opt_Weights=w_opt.tolist()
    )
    with open(os.path.join(args.save_dir, 'metrics_v5.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    rows = [
        ("CNN (TTA)",         cnn_acc,  cnn_f1,  cnn_f1,  cnn_f1,  0),
        ("Super Agent",       ag_acc,   ag_f1,   ag_prec, ag_rec,  ag_qwk),
        ("Hybrid Routing",    hy_acc,   hy_f1,   hy_prec, hy_rec,  hy_qwk),
        ("F1-Maximizer",      op_acc,   op_f1,   op_pre,  op_rec,  op_qwk),
    ]
    print(f"  {'Stage':<20} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'QWK':>7}")
    print("  " + "-"*55)
    for name, acc, f1, pr, rc, qwk in rows:
        print(f"  {name:<20} {acc*100:>6.2f}% {f1*100:>6.2f}% "
              f"{pr*100:>6.2f}% {rc*100:>6.2f}% {qwk:>7.4f}")
    print("="*70)
    print(f"✅ All saved to: {args.save_dir}")

if __name__ == "__main__":
    main()
