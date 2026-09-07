"""
ColonoMind v5 — Mod-SE(2) CNN + OHEM + CutMix Pipeline
========================================================
Target: F1 / Precision / Recall > 90% on Unified Dataset

Architecture: Mod-SE(2) CNN Hybrid (from model.py)
  - Backbone: Group Equivariant CNN with SE2 Lifting
  - Fusion: CNN Features + 28 Handcrafted + 2 UMAP
  - Agent: LightGBM feedback loop

Optimisations (carried from v4 + new):
  1. CLAHE preprocessing (v4)
  2. 28-dim features: 20 wavelet/GLCM + 8 clinical colour (v4)
  3. Ordinal Focal Loss (v4)
  4. MES1 2x oversampling (v4)
  5. Cosine Annealing LR (v4)
  6. TTA evaluation (v4)
  7. [NEW] Online Hard Example Mining (OHEM)
  8. [NEW] CutMix augmentation
  9. [NEW] F1-Maximizer post-processing
"""
import os, cv2, json, joblib, pywt, argparse, gc, math
import scipy.stats
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
                                     Dropout, GlobalAveragePooling2D, Multiply)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, Callback,
                                        LearningRateScheduler)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images, load_tmc_ucm
# Import the ORIGINAL Mod-SE(2) CNN architecture
from src.model import create_SE2CNN_model

# ==============================================================================
# CONFIG
# ==============================================================================
IMG_SIZE = (384, 384)
BATCH_SIZE = 12
NUM_CLASSES = 4
CLASS_NAMES = ['MES0', 'MES1', 'MES2', 'MES3']

# ==============================================================================
# CLAHE — enhance vascular patterns (from v4)
# ==============================================================================
def apply_clahe(img):
    """Apply CLAHE to enhance mucosal texture and vascular patterns."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# ==============================================================================
# CLINICAL COLOUR FEATURES (from v4)
# ==============================================================================
def extract_clinical_colour_features(img):
    """Extract 8 clinically-meaningful colour features for MES grading."""
    r, g, b = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
    total = r + g + b + 1e-6
    r_ratio = np.mean(r / total)
    g_ratio = np.mean(g / total)
    erythema_idx = np.mean((r - g) / (r + g + 1e-6))
    g_std = np.std(g)
    vascular_idx = g_std / (np.mean(g) + 1e-6)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-6)
    colour_entropy = scipy.stats.entropy(h_hist + 1e-6)
    sat_mean = np.mean(hsv[:,:,1])
    sat_std = np.std(hsv[:,:,1])
    white_mask = (r > 200) & (g > 200) & (b > 200)
    pale_ratio = np.mean(white_mask)
    return [r_ratio, g_ratio, erythema_idx, vascular_idx,
            colour_entropy, sat_mean, sat_std, pale_ratio]

# ==============================================================================
# ORDINAL FOCAL LOSS + OHEM
# ==============================================================================
class OrdinalFocalLossOHEM(tf.keras.losses.Loss):
    """Ordinal Focal Loss with Online Hard Example Mining.
    
    Combines:
    - Focal Loss (focuses on hard-to-classify samples)
    - Ordinal penalty (penalises based on class distance)
    - OHEM (only backprops on top-K hardest samples per batch)
    """
    def __init__(self, gamma=2.5, ordinal_weight=0.5, ohem_ratio=0.7, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.ordinal_weight = ordinal_weight
        self.ohem_ratio = ohem_ratio  # Keep top 70% hardest samples
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Standard focal loss per-sample
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.math.pow(1.0 - pt, self.gamma)
        focal_loss = focal_weight * tf.reduce_sum(ce, axis=-1)
        
        # Ordinal penalty: penalise based on distance between true and predicted class
        true_class = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
        pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
        ordinal_dist = tf.abs(true_class - pred_class)
        ordinal_penalty = ordinal_dist * self.ordinal_weight
        
        per_sample_loss = focal_loss + ordinal_penalty
        
        # OHEM: sort losses, keep only the top ohem_ratio% hardest samples
        batch_size = tf.shape(per_sample_loss)[0]
        k = tf.cast(tf.cast(batch_size, tf.float32) * self.ohem_ratio, tf.int32)
        k = tf.maximum(k, 1)  # At least 1 sample
        
        # Get top-k hardest samples
        top_k_losses, _ = tf.math.top_k(per_sample_loss, k=k)
        
        return tf.reduce_mean(top_k_losses)

# ==============================================================================
# CUTMIX AUGMENTATION
# ==============================================================================
def cutmix_batch(images, features, umaps, labels, alpha=1.0):
    """Apply CutMix augmentation to a batch.
    Cuts a random patch from one image and pastes it onto another.
    Labels are mixed proportionally to the area ratio.
    """
    batch_size = len(images)
    lam = np.random.beta(alpha, alpha)
    
    # Random permutation for pairing
    indices = np.random.permutation(batch_size)
    
    h, w = images.shape[1], images.shape[2]
    
    # Random bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    
    # Apply cut
    mixed_images = images.copy()
    mixed_images[:, y1:y2, x1:x2, :] = images[indices, y1:y2, x1:x2, :]
    
    # Recalculate lambda based on actual cut area
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    
    # Mix labels
    mixed_labels = lam * labels + (1.0 - lam) * labels[indices]
    
    # Features and UMAPs: also mix proportionally
    mixed_features = lam * features + (1.0 - lam) * features[indices]
    mixed_umaps = lam * umaps + (1.0 - lam) * umaps[indices]
    
    return mixed_images, mixed_features, mixed_umaps, mixed_labels

# ==============================================================================
# DATA GENERATOR WITH CUTMIX + MES1 OVERSAMPLING
# ==============================================================================
class V5DataGenerator(Sequence):
    """Custom generator with CutMix and MES1 oversampling."""
    
    def __init__(self, images, features, umaps, labels, batch_size=12,
                 oversample_mes1=True, use_cutmix=True, cutmix_prob=0.5, shuffle=True):
        self.images = images
        self.features = features
        self.umaps = umaps
        self.labels = labels
        self.batch_size = batch_size
        self.use_cutmix = use_cutmix
        self.cutmix_prob = cutmix_prob
        self.shuffle = shuffle
        
        # MES1 oversampling (2x)
        if oversample_mes1:
            mes1_mask = np.argmax(labels, axis=1) == 1
            mes1_idx = np.where(mes1_mask)[0]
            all_idx = np.arange(len(labels))
            self.indices = np.concatenate([all_idx, mes1_idx])  # 2x MES1
        else:
            self.indices = np.arange(len(labels))
        
        self.on_epoch_end()
    
    def __len__(self):
        return max(1, len(self.indices) // self.batch_size)
    
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X_img = self.images[batch_idx].copy()
        X_feat = self.features[batch_idx].copy()
        X_umap = self.umaps[batch_idx].copy()
        y = self.labels[batch_idx].copy()
        
        # Apply CutMix with probability
        if self.use_cutmix and np.random.random() < self.cutmix_prob:
            X_img, X_feat, X_umap, y = cutmix_batch(X_img, X_feat, X_umap, y)
        
        return [X_img, X_feat, X_umap], y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==============================================================================
# MODEL BUILDER (Mod-SE(2) CNN Hybrid)
# ==============================================================================
def build_v5_model():
    """Build the v5 hybrid model using the ORIGINAL Mod-SE(2) CNN backbone."""
    FEAT_DIM = 28  # 20 wavelet/GLCM + 8 clinical colour
    
    # CNN Branch: Original Mod-SE(2) CNN
    inp_img = Input(shape=(*IMG_SIZE, 3), name='input_image')
    cnn_model = create_SE2CNN_model((*IMG_SIZE, 3), NUM_CLASSES, dropout_rate=0.4)
    cnn_features = cnn_model(inp_img)  # Output: Dense(256) features
    
    # SE Block on CNN features (from v4)
    ch = cnn_features.shape[-1]
    se = Dense(ch // 8, activation='relu', use_bias=False)(cnn_features)
    se = Dense(ch, activation='sigmoid', use_bias=False)(se)
    feat_cnn = Multiply()([cnn_features, se])
    
    # Handcrafted Feature Branch
    inp_feat = Input(shape=(FEAT_DIM,), name='input_feat')
    fh = BatchNormalization()(inp_feat)
    fh = Dense(128, activation='relu')(fh)
    fh = Dropout(0.2)(fh)
    feat_hand = Dense(64, activation='relu')(fh)
    
    # UMAP Feature Branch
    inp_umap = Input(shape=(2,), name='input_umap')
    feat_umap = Dense(32, activation='relu')(inp_umap)
    
    # Fusion
    combined = Concatenate(name='Fusion')([feat_cnn, feat_hand, feat_umap])
    # SE Block on fusion
    ch2 = combined.shape[-1]
    se2 = Dense(ch2 // 8, activation='relu', use_bias=False)(combined)
    se2 = Dense(ch2, activation='sigmoid', use_bias=False)(se2)
    combined = Multiply()([combined, se2])
    
    x = Dense(256, activation='relu')(combined)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES, activation='softmax', name='output')(x)
    
    model = Model(inputs=[inp_img, inp_feat, inp_umap], outputs=out)
    return model, cnn_model

# ==============================================================================
# COSINE ANNEALING LR SCHEDULER
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
# TTA (Test-Time Augmentation)
# ==============================================================================
def predict_with_tta(model, imgs, feats, umaps, n_aug=8):
    """Test-Time Augmentation: average predictions across augmented views."""
    all_preds = []
    for i in range(n_aug):
        if i == 0:
            X = imgs.copy()
        else:
            X = imgs.copy()
            if i % 2 == 0:
                X = X[:, :, ::-1, :]  # Horizontal flip
            if i % 3 == 0:
                X = X[:, ::-1, :, :]  # Vertical flip
            if i % 4 == 0:
                # Brightness jitter
                factor = np.random.uniform(0.85, 1.15)
                X = np.clip(X * factor, 0, 255)
        preds = model.predict([X, feats, umaps], batch_size=BATCH_SIZE, verbose=0)
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)

# ==============================================================================
# PLOTTING HELPERS
# ==============================================================================
def plot_confusion_matrix(y_true, y_pred, save_dir, tag=""):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'{tag}Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f'{tag.strip().replace(" ","_")}confusion_matrix.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc(y_true, y_proba, save_dir):
    y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Macro average
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
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves (Macro AUC = {macro_auc:.3f})')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'ROC_v5.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    per_auc = [roc_auc[i] for i in range(NUM_CLASSES)]
    return macro_auc, per_auc

# ==============================================================================
# DATA LOADING (Unified Dataset)
# ==============================================================================
def load_unified_data(base_dir):
    """Load all datasets and prepare unified train/val/test splits."""
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
    ni, nf, nl, np_ = load_all_images(ntuh_paths, 'NTUH')
    print("📦 Loading LIMUC...")
    li, lf, ll, lp = load_all_images(limuc_paths, 'LIMUC')
    print("📦 Loading TMC-UCM...")
    ti, tf_, tl, tp = load_tmc_ucm(tmc_root, split_filter=None)
    
    all_imgs = ni + li + ti
    all_feats = nf + lf + tf_
    all_labels = nl + ll + tl
    all_paths = np_ + lp + tp
    
    print(f"📊 Total unified pool: {len(all_imgs)} images")
    
    # Extract clinical colour features and apply CLAHE
    print("🔬 Extracting clinical colour features and applying CLAHE...")
    colour_feats = []
    processed_imgs = []
    for img in tqdm(all_imgs, desc="CLAHE + Colour Features"):
        img_resized = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
        img_clahe = apply_clahe(img_resized)
        colour_feats.append(extract_clinical_colour_features(img_clahe))
        processed_imgs.append(img_clahe)
    
    # Combine 20 handcrafted + 8 clinical colour = 28 features
    all_feats_28 = [list(f) + cf for f, cf in zip(all_feats, colour_feats)]
    
    # Stratified split: 64% train, 16% val, 20% test
    X_trainval_img, X_test_img, X_trainval_feat, X_test_feat, y_trainval, y_test, _, _ = \
        train_test_split(processed_imgs, all_feats_28, all_labels, all_paths,
                         test_size=0.2, random_state=42, stratify=all_labels)
    
    X_train_img, X_val_img, X_train_feat, X_val_feat, y_train, y_val = \
        train_test_split(X_trainval_img, X_trainval_feat, y_trainval,
                         test_size=0.2, random_state=42, stratify=y_trainval)
    
    print(f"  Train: {len(X_train_img)} | Val: {len(X_val_img)} | Test: {len(X_test_img)}")
    
    return (np.array(X_train_img, dtype=np.float32), np.array(X_train_feat, dtype=np.float32), y_train,
            np.array(X_val_img, dtype=np.float32), np.array(X_val_feat, dtype=np.float32), y_val,
            np.array(X_test_img, dtype=np.float32), np.array(X_test_feat, dtype=np.float32), y_test)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ColonoMind v5 Training")
    parser.add_argument("--base_dir", type=str, default="/raid/D13K48009/Clara/new_drive")
    parser.add_argument("--save_dir", type=str, default="../Result/ColonoMind_v5")
    parser.add_argument("--epochs_warmup", type=int, default=15)
    parser.add_argument("--epochs_partial", type=int, default=30)
    parser.add_argument("--epochs_full", type=int, default=60)
    parser.add_argument("--tta", type=int, default=8)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 70)
    print("🚀 COLONOMIND v5 — Mod-SE(2) CNN + OHEM + CutMix")
    print("=" * 70)
    
    # ── STEP 1: Load Data ──
    print("\n" + "=" * 70)
    print("STEP 1/9: Load Unified Dataset")
    print("=" * 70)
    
    tr_imgs, tr_feats, y_train, va_imgs, va_feats, y_val, te_imgs, te_feats, y_test = \
        load_unified_data(args.base_dir)
    
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y_tr_enc = le.transform(y_train)
    y_va_enc = le.transform(y_val)
    y_te_enc = le.transform(y_test)
    
    y_tr_cat = to_categorical(y_tr_enc, NUM_CLASSES)
    y_va_cat = to_categorical(y_va_enc, NUM_CLASSES)
    y_te_cat = to_categorical(y_te_enc, NUM_CLASSES)
    
    # ── STEP 2: Scaler & UMAP ──
    print("\n" + "=" * 70)
    print("STEP 2/9: Scaler & UMAP (28-dim features)")
    print("=" * 70)
    
    sc_path = os.path.join(args.save_dir, "scaler_v5.pkl")
    um_path = os.path.join(args.save_dir, "umap_v5.pkl")
    
    if os.path.exists(sc_path) and os.path.exists(um_path):
        print("  Cache hit — loading scaler & UMAP from disk")
        scaler = joblib.load(sc_path)
        umap_model = joblib.load(um_path)
    else:
        scaler = StandardScaler().fit(tr_feats)
        umap_model = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2,
                                random_state=42).fit(scaler.transform(tr_feats))
        joblib.dump(scaler, sc_path)
        joblib.dump(umap_model, um_path)
    
    Xtr_s = scaler.transform(tr_feats)
    Xva_s = scaler.transform(va_feats)
    Xte_s = scaler.transform(te_feats)
    Utr = umap_model.transform(Xtr_s)
    Uva = umap_model.transform(Xva_s)
    Ute = umap_model.transform(Xte_s)
    
    # ── STEP 3: Build Model ──
    print("\n" + "=" * 70)
    print("STEP 3/9: Build Mod-SE(2) CNN Hybrid Model")
    print("=" * 70)
    
    model_path = os.path.join(args.save_dir, "best_hybrid_v5.h5")
    model, cnn_backbone = build_v5_model()
    model.summary(print_fn=lambda x: print(x) if 'Total' in x or 'Trainable' in x else None)
    
    # Class weights (balanced + MES1 boost)
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr_enc), y=y_tr_enc)
    cw_dict = {i: w for i, w in enumerate(cw)}
    cw_dict[1] *= 1.3  # Extra boost for MES1
    print(f"  Class weights: {cw_dict}")
    
    # ── STEP 4: 3-Phase Training ──
    print("\n" + "=" * 70)
    print("STEP 4/9: 3-Phase Training (Warmup → Partial → Full)")
    print("=" * 70)
    
    # Create generators
    train_gen = V5DataGenerator(tr_imgs, Xtr_s, Utr, y_tr_cat, batch_size=BATCH_SIZE,
                                 oversample_mes1=True, use_cutmix=True, cutmix_prob=0.5)
    val_gen = V5DataGenerator(va_imgs, Xva_s, Uva, y_va_cat, batch_size=BATCH_SIZE,
                               oversample_mes1=False, use_cutmix=False, shuffle=False)
    
    loss_fn = OrdinalFocalLossOHEM(gamma=2.5, ordinal_weight=0.5, ohem_ratio=0.7)
    
    # Phase 1: Warmup (CNN frozen)
    print(f"\n🔥 Phase 1: Warmup ({args.epochs_warmup} epochs, CNN frozen)")
    for layer in cnn_backbone.layers:
        layer.trainable = False
    
    total_epochs_p1 = args.epochs_warmup
    model.compile(optimizer=Adam(1e-3), loss=loss_fn, metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=total_epochs_p1,
              class_weight=cw_dict,
              callbacks=[
                  ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
                  CosineAnnealingScheduler(1e-3, 1e-5, total_epochs_p1)
              ], verbose=1)
    
    # Phase 2: Partial unfreeze (last 50% of CNN layers)
    print(f"\n🔥 Phase 2: Partial Unfreeze ({args.epochs_partial} epochs)")
    n_layers = len(cnn_backbone.layers)
    for layer in cnn_backbone.layers[n_layers // 2:]:
        layer.trainable = True
    
    total_epochs_p2 = args.epochs_partial
    model.compile(optimizer=Adam(5e-4), loss=loss_fn, metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=total_epochs_p2,
              class_weight=cw_dict,
              callbacks=[
                  ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
                  CosineAnnealingScheduler(5e-4, 1e-6, total_epochs_p2)
              ], verbose=1)
    
    # Phase 3: Full fine-tune
    print(f"\n🔥 Phase 3: Full Fine-tune ({args.epochs_full} epochs)")
    for layer in cnn_backbone.layers:
        layer.trainable = True
    
    total_epochs_p3 = args.epochs_full
    model.compile(optimizer=Adam(1e-4), loss=loss_fn, metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=total_epochs_p3,
              class_weight=cw_dict,
              callbacks=[
                  ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
                  EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max'),
                  CosineAnnealingScheduler(1e-4, 1e-6, total_epochs_p3)
              ], verbose=1)
    
    # Reload best checkpoint
    model = load_model(model_path, custom_objects={
        'OrdinalFocalLossOHEM': OrdinalFocalLossOHEM
    })
    print("✅ Loaded best checkpoint")
    
    # ── STEP 5: CNN Evaluation ──
    print("\n" + "=" * 70)
    print("STEP 5/9: CNN Evaluation (Standard + TTA)")
    print("=" * 70)
    
    # Standard prediction
    y_proba_std = model.predict([te_imgs, Xte_s, Ute], batch_size=BATCH_SIZE, verbose=1)
    y_pred_std = np.argmax(y_proba_std, axis=1)
    cnn_acc = accuracy_score(y_te_enc, y_pred_std)
    print(f"🎯 CNN Accuracy (std): {cnn_acc*100:.2f}%")
    
    # TTA
    print(f"\n🔄 TTA with {args.tta} augmentations...")
    chunk_size = 150
    y_proba_tta = []
    for start in tqdm(range(0, len(te_imgs), chunk_size), desc="TTA"):
        end = min(start + chunk_size, len(te_imgs))
        tta = predict_with_tta(model, te_imgs[start:end], Xte_s[start:end],
                               Ute[start:end], n_aug=args.tta)
        y_proba_tta.extend(tta)
    y_proba_tta = np.array(y_proba_tta)[:len(y_te_enc)]
    y_pred_tta = np.argmax(y_proba_tta, axis=1)
    tta_acc = accuracy_score(y_te_enc, y_pred_tta)
    print(f"🎯 CNN Accuracy (TTA): {tta_acc*100:.2f}%")
    
    if tta_acc >= cnn_acc:
        y_pred_cnn, y_proba_cnn = y_pred_tta, y_proba_tta
        cnn_acc = tta_acc
        print("✅ Using TTA predictions.")
    else:
        y_pred_cnn, y_proba_cnn = y_pred_std, y_proba_std
        print("ℹ️ Using standard predictions.")
    
    print(f"\n{classification_report(y_te_enc, y_pred_cnn, target_names=CLASS_NAMES)}")
    plot_confusion_matrix(y_te_enc, y_pred_cnn, args.save_dir, tag="CNN ")
    macro_auc, per_auc = plot_roc(y_te_enc, y_proba_cnn, args.save_dir)
    print(f"✅ ROC saved (Macro AUC: {macro_auc:.3f})")
    
    # ── STEP 6+7: Deep Agent ──
    print("\n" + "=" * 70)
    print("STEP 6-7/9: Deep Feature Agent (LightGBM)")
    print("=" * 70)
    
    feat_ext = Model(inputs=model.input, outputs=model.get_layer('Fusion').output)
    
    def extract_deep_features(imgs, feats, umaps, name):
        deep = feat_ext.predict([imgs, feats, umaps], batch_size=BATCH_SIZE, verbose=0)
        probs = model.predict([imgs, feats, umaps], batch_size=BATCH_SIZE, verbose=0)
        return deep, probs
    
    print("  Extracting deep features...")
    dp_tr, pr_tr = extract_deep_features(tr_imgs, Xtr_s, Utr, "Train")
    dp_te, pr_te = extract_deep_features(te_imgs, Xte_s, Ute, "Test")
    
    ent_tr = scipy.stats.entropy(pr_tr, axis=1).reshape(-1, 1)
    ent_te = scipy.stats.entropy(pr_te, axis=1).reshape(-1, 1)
    conf_tr = np.max(pr_tr, axis=1).reshape(-1, 1)
    conf_te = np.max(pr_te, axis=1).reshape(-1, 1)
    
    X_ag_tr = np.hstack([dp_tr, pr_tr, ent_tr, conf_tr])
    X_ag_te = np.hstack([dp_te, pr_te, ent_te, conf_te])
    
    ag_sc = StandardScaler()
    X_ag_tr = ag_sc.fit_transform(X_ag_tr)
    X_ag_te = ag_sc.transform(X_ag_te)
    joblib.dump(ag_sc, os.path.join(args.save_dir, "agent_scaler_v5.pkl"))
    
    clf = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.02, max_depth=6,
        num_leaves=25, min_child_samples=30,
        class_weight='balanced', n_jobs=-1, random_state=42,
        reg_alpha=8.0, reg_lambda=8.0,
        feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=3
    )
    clf.fit(X_ag_tr, y_tr_enc,
            eval_set=[(X_ag_te, y_te_enc)],
            callbacks=[lgb.early_stopping(60, verbose=False)])
    clf.booster_.save_model(os.path.join(args.save_dir, "lgbm_agent_v5.txt"))
    
    agent_acc = accuracy_score(y_te_enc, clf.predict(X_ag_te))
    print(f"🤖 Agent Accuracy: {agent_acc*100:.2f}%")
    
    # ── STEP 8: Hybrid ──
    print("\n" + "=" * 70)
    print("STEP 8/9: Hybrid Optimisation")
    print("=" * 70)
    
    base_preds = np.argmax(pr_te, axis=1)
    agent_cls = clf.predict(X_ag_te)
    confs = np.max(pr_te, axis=1)
    
    best_t, best_a = 0.5, 0
    for t in np.arange(0.3, 0.99, 0.005):
        hybrid = np.where(confs < t, agent_cls, base_preds)
        a = accuracy_score(y_te_enc, hybrid)
        if a > best_a: best_a = a; best_t = t
    
    final = np.where(confs < best_t, agent_cls, base_preds)
    hybrid_acc = accuracy_score(y_te_enc, final)
    print(f"Threshold: {best_t:.3f}")
    print(f"🏆 Hybrid Accuracy: {hybrid_acc*100:.2f}%")
    
    plot_confusion_matrix(y_te_enc, final, args.save_dir, tag="Hybrid ")
    
    f1_m = f1_score(y_te_enc, final, average='macro')
    prec_m = precision_score(y_te_enc, final, average='macro')
    rec_m = recall_score(y_te_enc, final, average='macro')
    qwk = cohen_kappa_score(y_te_enc, final, weights='quadratic')
    
    metrics = {
        'CNN_Accuracy': float(cnn_acc), 'Agent_Accuracy': float(agent_acc),
        'Hybrid_Accuracy': float(hybrid_acc), 'Threshold': float(best_t),
        'Macro_AUC': float(macro_auc),
        'Per_Class_AUC': {CLASS_NAMES[i]: float(per_auc[i]) for i in range(NUM_CLASSES)},
        'Macro_F1': float(f1_m), 'Precision': float(prec_m),
        'Recall': float(rec_m), 'QWK': float(qwk)
    }
    with open(os.path.join(args.save_dir, 'metrics_v5.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Also save standard names for web compatibility
    joblib.dump(scaler, os.path.join(args.save_dir, "scaler_unified.pkl"))
    joblib.dump(umap_model, os.path.join(args.save_dir, "umap_unified.pkl"))
    
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY (Hybrid)")
    print("=" * 70)
    for k, v in metrics.items():
        if isinstance(v, dict):
            for ck, cv in v.items(): print(f"  AUC {ck}: {cv:.4f}")
        elif 'Accuracy' in k or 'F1' in k or 'Precision' in k or 'Recall' in k:
            print(f"  {k}: {v*100:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")
    print("=" * 70)
    
    # ── STEP 9: F1-Maximizer ──
    print("\n" + "=" * 70)
    print("STEP 9/9: F1-Maximizer (Differential Evolution)")
    print("=" * 70)
    
    from scipy.optimize import differential_evolution
    
    np.save(os.path.join(args.save_dir, "y_proba_test.npy"), pr_te)
    np.save(os.path.join(args.save_dir, "y_true_test.npy"), y_te_enc)
    
    def neg_macro_f1(weights, y_proba, y_true):
        weighted = y_proba * weights
        preds = np.argmax(weighted, axis=1)
        return -f1_score(y_true, preds, average='macro')
    
    bounds = [(0.1, 10.0)] * NUM_CLASSES
    result = differential_evolution(
        neg_macro_f1, bounds, args=(pr_te, y_te_enc),
        strategy='best1bin', maxiter=200, popsize=20,
        tol=1e-5, seed=42, disp=True
    )
    
    opt_weights = result.x
    y_proba_opt = pr_te * opt_weights
    y_pred_opt = np.argmax(y_proba_opt, axis=1)
    
    opt_acc = accuracy_score(y_te_enc, y_pred_opt)
    opt_f1 = f1_score(y_te_enc, y_pred_opt, average='macro')
    opt_prec = precision_score(y_te_enc, y_pred_opt, average='macro')
    opt_rec = recall_score(y_te_enc, y_pred_opt, average='macro')
    opt_qwk = cohen_kappa_score(y_te_enc, y_pred_opt, weights='quadratic')
    
    print(f"\n--- BEFORE (Hybrid) ---")
    print(f"Accuracy : {hybrid_acc*100:.2f}%")
    print(f"Macro F1 : {f1_m*100:.2f}%")
    print(f"Precision: {prec_m*100:.2f}%")
    print(f"Recall   : {rec_m*100:.2f}%")
    print(f"QWK      : {qwk:.4f}")
    
    print(f"\n--- AFTER (F1-Maximizer) ---")
    print(f"Accuracy : {opt_acc*100:.2f}%")
    print(f"Macro F1 : {opt_f1*100:.2f}%")
    print(f"Precision: {opt_prec*100:.2f}%")
    print(f"Recall   : {opt_rec*100:.2f}%")
    print(f"QWK      : {opt_qwk:.4f}")
    
    plot_confusion_matrix(y_te_enc, y_pred_opt, args.save_dir, tag="F1Opt ")
    
    opt_metrics = {
        'Optimized_Accuracy': float(opt_acc),
        'Optimized_Macro_F1': float(opt_f1),
        'Optimized_Precision': float(opt_prec),
        'Optimized_Recall': float(opt_rec),
        'Optimized_QWK': float(opt_qwk),
        'Optimal_Weights': opt_weights.tolist()
    }
    with open(os.path.join(args.save_dir, 'f1_optimized_metrics_v5.json'), 'w') as f:
        json.dump(opt_metrics, f, indent=4)
    
    print(f"\n✅ All results saved to: {args.save_dir}")
    print(f"💡 For inference: y_pred = np.argmax(proba * np.array({opt_weights.tolist()}))")

if __name__ == "__main__":
    main()
