"""
ColonoMind v8 — Hierarchical Classification Pipeline (Pure Mod-SE CNN)
======================================================================
Base: ColonoMind v6 Checkpoints (Fine-Tuning)
Optimisations for >80% Accuracy (Tackling Extreme Class Imbalance):
  1. Two-Stage Pipeline:
     - Stage 1: Detector (Normal vs Active Disease) -> Binary Classification.
     - Stage 2: Grader (MES1 vs MES2 vs MES3) -> 3-Class Classification.
  2. Fine-Tuning: Load V6 checkpoints, transfer weights, replace classification head,
     and fine-tune with specific subsets of data.
  3. Hierarchical Super Agents: Two separate LightGBM models.
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
                             f1_score, cohen_kappa_score)
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap.umap_ as umap

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, BatchNormalization,
                                     Dropout, GlobalAveragePooling2D, Conv2D,
                                     MaxPooling2D, Activation, Multiply, Reshape)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images, load_tmc_ucm

# ==============================================================================
# CONFIG
# ==============================================================================
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 16
FEAT_DIM    = 28   # 20 wavelet/GLCM + 8 clinical colour
ENSEMBLE_SEEDS = [42, 123, 999]

# ==============================================================================
# PREPROCESSING & AUGMENTATION (Same as V6/V7)
# ==============================================================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def smart_preprocess(img):
    if img is None: return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    crop = img[30:430, 200:550] if h > 450 and w > 550 else img
    resized = cv2.resize(crop, (IMG_SIZE[1], IMG_SIZE[0]))
    return apply_clahe(resized)

def apply_augmentation(img):
    rows, cols = img.shape[:2]
    angle = np.random.uniform(-180, 180)
    tx, ty = np.random.uniform(-0.15, 0.15) * cols, np.random.uniform(-0.15, 0.15) * rows
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
    M[0, 2] += tx; M[1, 2] += ty
    img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() > 0.5: img = cv2.flip(img, 1)
    if np.random.rand() > 0.5: img = cv2.flip(img, 0)
    return cv2.convertScaleAbs(img, alpha=1.0+np.random.uniform(-0.25, 0.25), beta=np.random.uniform(-20, 20))

class V8Generator(Sequence):
    def __init__(self, images, features, umaps, labels, batch_size=16, augment=False, shuffle=True):
        self.batch_size, self.augment, self.shuffle = batch_size, augment, shuffle
        self.indices = np.arange(len(labels))
        self.images, self.features, self.umaps, self.labels = images, features, umaps, labels
        self.on_epoch_end()

    def __len__(self): return max(1, len(self.indices) // self.batch_size)
    def __getitem__(self, idx):
        bidx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X_img = self.images[bidx].copy()
        if self.augment:
            for i in range(len(X_img)): X_img[i] = apply_augmentation(X_img[i].astype(np.uint8)).astype(np.float32)
        X_feat, X_umap, y = self.features[bidx].copy(), self.umaps[bidx].copy(), self.labels[bidx].copy()
        return (X_img / 255.0, X_feat, X_umap), y
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

# ==============================================================================
# MODEL BUILDER
# ==============================================================================
def se_block(x, ratio=8):
    f = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, f))(se)
    se = Dense(max(1, f//ratio), activation='relu', use_bias=False)(se)
    se = Dense(f, activation='sigmoid', use_bias=False)(se)
    return Multiply()([x, se])

def build_se_cnn_hybrid(seed, num_classes):
    tf.random.set_seed(seed); np.random.seed(seed)
    inp_img = Input(shape=(*IMG_SIZE, 3), name='input_image')
    x = Conv2D(32, (3,3), padding='same')(inp_img)
    x = BatchNormalization()(x); x = Activation('relu')(x); x = se_block(x, 8); x = MaxPooling2D((2,2))(x); x = Dropout(0.2)(x)
    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x); x = se_block(x, 8); x = MaxPooling2D((2,2))(x); x = Dropout(0.2)(x)
    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x); x = se_block(x, 8); x = MaxPooling2D((2,2))(x); x = Dropout(0.3)(x)
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x); x = se_block(x, 8); x = MaxPooling2D((2,2))(x); x = Dropout(0.3)(x)
    x = Conv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x); x = se_block(x, 8)
    feat_cnn = GlobalAveragePooling2D()(x); feat_cnn = Dropout(0.4)(feat_cnn)

    inp_feat = Input(shape=(FEAT_DIM,), name='input_feat')
    fh = BatchNormalization()(inp_feat); fh = Dense(128, activation='relu')(fh); fh = Dropout(0.2)(fh)
    feat_hand = Dense(64, activation='relu')(fh)
    
    inp_umap = Input(shape=(2,), name='input_umap')
    feat_umap = Dense(32, activation='relu')(inp_umap)

    combined = Concatenate(name='Fusion')([feat_cnn, feat_hand, feat_umap])
    x = Dense(512, activation='relu')(combined); x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x); x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inputs=[inp_img, inp_feat, inp_umap], outputs=out)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="../Result/ColonoMind_v8_Hierarchical")
    ap.add_argument("--v6_checkpoint_dir", default="../Result/ColonoMind_v6")
    ap.add_argument("--epochs_ft", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*70); print("🚀 COLONOMIND v8 — Hierarchical Two-Stage Pipeline"); print("="*70)
    
    # ── 1. Load Pre-cached Data ───────────────────────────────────────────────
    cache = os.path.join(args.v6_checkpoint_dir, "dataset_cache_v6.npz")
    if not os.path.exists(cache):
        raise FileNotFoundError(f"Cache not found: {cache}. Run V6 first.")
    
    d = np.load(cache, allow_pickle=True)
    all_imgs, all_feats, all_labels = d['imgs'], d['feats'], list(d['labels'])
    y_enc_full = LabelEncoder().fit_transform(all_labels) # 0, 1, 2, 3
    
    # Create Hierarchical Labels
    # Stage 1: 0 (Normal) vs 1 (Active Disease, which is MES1,2,3)
    y_stage1 = np.where(y_enc_full == 0, 0, 1)
    
    # Stage 2: Only Active Disease subset. Mapped to 0 (MES1), 1 (MES2), 2 (MES3)
    active_idx = np.where(y_enc_full > 0)[0]
    X_imgs_stg2 = all_imgs[active_idx]
    X_feats_stg2 = all_feats[active_idx]
    y_stage2 = y_enc_full[active_idx] - 1
    
    # ── 2. Data Splitting ──────────────────────────────────────────────────────
    # Splits for Stage 1 (All data)
    X_tv_i1, X_te_i1, X_tv_f1, X_te_f1, y_tv1, y_te1 = train_test_split(all_imgs, all_feats, y_stage1, test_size=0.20, random_state=42, stratify=y_stage1)
    X_tr_i1, X_va_i1, X_tr_f1, X_va_f1, y_tr1, y_va1 = train_test_split(X_tv_i1, X_tv_f1, y_tv1, test_size=0.20, random_state=42, stratify=y_tv1)
    
    # Splits for Stage 2 (Active only)
    X_tv_i2, X_te_i2, X_tv_f2, X_te_f2, y_tv2, y_te2 = train_test_split(X_imgs_stg2, X_feats_stg2, y_stage2, test_size=0.20, random_state=42, stratify=y_stage2)
    X_tr_i2, X_va_i2, X_tr_f2, X_va_f2, y_tr2, y_va2 = train_test_split(X_tv_i2, X_tv_f2, y_tv2, test_size=0.20, random_state=42, stratify=y_tv2)
    
    # Full evaluation ground truth (test set)
    # We must keep the same original test set mapping to compare properly
    _, _, _, _, _, y_te_full = train_test_split(all_imgs, all_feats, y_enc_full, test_size=0.20, random_state=42, stratify=y_enc_full)

    # ── 3. UMAP & Scaling ──────────────────────────────────────────────────────
    sc = StandardScaler().fit(X_tr_f1)
    um = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(sc.transform(X_tr_f1))
    
    def process_feats(features):
        scaled = sc.transform(features)
        return scaled, um.transform(scaled)
        
    Xtr_s1, Utr1 = process_feats(X_tr_f1); Xva_s1, Uva1 = process_feats(X_va_f1); Xte_s1, Ute1 = process_feats(X_te_f1)
    Xtr_s2, Utr2 = process_feats(X_tr_f2); Xva_s2, Uva2 = process_feats(X_va_f2); Xte_s2, Ute2 = process_feats(X_te_f2)

    # One-hot encoding
    y_tr1_c, y_va1_c = to_categorical(y_tr1, 2), to_categorical(y_va1, 2)
    y_tr2_c, y_va2_c = to_categorical(y_tr2, 3), to_categorical(y_va2, 3)
    
    cw1 = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr1), y=y_tr1)
    cw2 = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr2), y=y_tr2)

    # ── 4. STAGE 1 (Binary Detector) Fine-Tuning ──────────────────────────────
    print("\n" + "="*70); print("STAGE 1/4: Fine-Tuning CNN Detector (Normal vs Active)"); print("="*70)
    stage1_models = []
    loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
    
    tr_gen1 = V8Generator(X_tr_i1, Xtr_s1, Utr1, y_tr1_c, BATCH_SIZE, augment=True)
    va_gen1 = V8Generator(X_va_i1, Xva_s1, Uva1, y_va1_c, BATCH_SIZE, augment=False)
    
    for m_idx, seed in enumerate(ENSEMBLE_SEEDS):
        v6_path = os.path.join(args.v6_checkpoint_dir, f"secnn_v6_seed{seed}.h5")
        stg1_path = os.path.join(args.save_dir, f"stage1_cnn_seed{seed}.h5")
        
        if os.path.exists(stg1_path):
            print(f"⚡ Loading Stage 1 Model {seed}")
            model = load_model(stg1_path, compile=False)
        else:
            print(f"🔥 Fine-tuning Stage 1 Model {seed}")
            v6_model = load_model(v6_path, compile=False)
            model = build_se_cnn_hybrid(seed, num_classes=2)
            
            # Transfer weights bypassing Keras layer naming conflicts
            v8_weights = model.get_weights()
            v6_weights = v6_model.get_weights()
            # Copy all weights except the last layer's kernel and bias (last 2 elements)
            v8_weights[:-2] = v6_weights[:-2]
            model.set_weights(v8_weights)
                    
            model.compile(optimizer=Adam(5e-5), loss=loss_fn, metrics=['accuracy'])
            model.fit(tr_gen1, validation_data=va_gen1, epochs=args.epochs_ft, class_weight=dict(enumerate(cw1)),
                      callbacks=[ModelCheckpoint(stg1_path, save_best_only=True, monitor='val_accuracy'), EarlyStopping(patience=5)], verbose=1)
            model = load_model(stg1_path, compile=False)
        stage1_models.append(model)

    # ── 5. STAGE 2 (Ternary Grader) Fine-Tuning ───────────────────────────────
    print("\n" + "="*70); print("STAGE 2/4: Fine-Tuning CNN Grader (MES1 vs 2 vs 3)"); print("="*70)
    stage2_models = []
    
    tr_gen2 = V8Generator(X_tr_i2, Xtr_s2, Utr2, y_tr2_c, BATCH_SIZE, augment=True)
    va_gen2 = V8Generator(X_va_i2, Xva_s2, Uva2, y_va2_c, BATCH_SIZE, augment=False)
    
    for m_idx, seed in enumerate(ENSEMBLE_SEEDS):
        v6_path = os.path.join(args.v6_checkpoint_dir, f"secnn_v6_seed{seed}.h5")
        stg2_path = os.path.join(args.save_dir, f"stage2_cnn_seed{seed}.h5")
        
        if os.path.exists(stg2_path):
            print(f"⚡ Loading Stage 2 Model {seed}")
            model = load_model(stg2_path, compile=False)
        else:
            print(f"🔥 Fine-tuning Stage 2 Model {seed}")
            v6_model = load_model(v6_path, compile=False)
            model = build_se_cnn_hybrid(seed, num_classes=3)
            
            # Transfer weights bypassing Keras layer naming conflicts
            v8_weights = model.get_weights()
            v6_weights = v6_model.get_weights()
            # Copy all weights except the last layer's kernel and bias (last 2 elements)
            v8_weights[:-2] = v6_weights[:-2]
            model.set_weights(v8_weights)
                    
            model.compile(optimizer=Adam(5e-5), loss=loss_fn, metrics=['accuracy'])
            model.fit(tr_gen2, validation_data=va_gen2, epochs=args.epochs_ft, class_weight=dict(enumerate(cw2)),
                      callbacks=[ModelCheckpoint(stg2_path, save_best_only=True, monitor='val_accuracy'), EarlyStopping(patience=5)], verbose=1)
            model = load_model(stg2_path, compile=False)
        stage2_models.append(model)

    # ── 6. Super Agents (LightGBM for Stage 1 and Stage 2) ─────────────────────
    print("\n" + "="*70); print("STAGE 3/4: Training Hierarchical Super Agents"); print("="*70)
    def extract_deep(models, imgs, feats_s, umaps, num_c):
        deeps, probs = [], []
        extractors = [Model(inputs=m.input, outputs=m.get_layer('Fusion').output) for m in models]
        for i in range(len(models)):
            deeps.append(extractors[i].predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0))
            probs.append(models[i].predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0))
        return np.hstack([np.mean(deeps, axis=0), feats_s, umaps, np.mean(probs, axis=0)]), np.mean(probs, axis=0)

    # Stage 1 Agent
    X_ag_tr1, _ = extract_deep(stage1_models, X_tr_i1, Xtr_s1, Utr1, 2)
    X_ag_te1, prob_stg1 = extract_deep(stage1_models, X_te_i1, Xte_s1, Ute1, 2)
    lgb1 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.01, class_weight='balanced', max_depth=5, num_leaves=31)
    lgb1.fit(X_ag_tr1, y_tr1)
    ag_pred_1 = lgb1.predict(X_ag_te1)
    print(f"🎯 Stage 1 Test Acc: {accuracy_score(y_te1, ag_pred_1)*100:.2f}%")

    # Stage 2 Agent
    X_ag_tr2, _ = extract_deep(stage2_models, X_tr_i2, Xtr_s2, Utr2, 3)
    X_ag_te2_full, prob_stg2_full = extract_deep(stage2_models, X_te_i1, Xte_s1, Ute1, 3) # Predict on FULL test set to route later
    lgb2 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.01, class_weight='balanced', max_depth=6, num_leaves=40)
    lgb2.fit(X_ag_tr2, y_tr2)
    ag_pred_2_full = lgb2.predict(X_ag_te2_full)

    # ── 7. Evaluate End-to-End Pipeline ────────────────────────────────────────
    print("\n" + "="*70); print("STAGE 4/4: End-to-End Pipeline Inference"); print("="*70)
    final_preds = []
    
    # Inference Routing:
    for i in range(len(y_te_full)):
        is_active = ag_pred_1[i] == 1
        if not is_active:
            final_preds.append(0) # Output MES0
        else:
            final_preds.append(ag_pred_2_full[i] + 1) # Output MES1, 2, or 3
            
    final_preds = np.array(final_preds)
    
    print("\n🏆 V8 Hierarchical Final Metrics:")
    print(classification_report(y_te_full, final_preds, target_names=['MES0', 'MES1', 'MES2', 'MES3'], digits=4))
    
    print(f"\n✅ Pipeline Complete! Overall Accuracy: {accuracy_score(y_te_full, final_preds)*100:.2f}%")

if __name__ == "__main__":
    main()
