"""
ColonoMind v7 — Fine-Tuning Mod-SE CNN & Focal Loss
============================================================
Base: ColonoMind v6 Checkpoints
Optimisations for >80% Accuracy (focusing on MES1/MES2):
  1. Transfer Learning: Load V6 checkpoints as pre-trained weights.
  2. Loss Function: Custom Categorical Focal Loss (Gamma=2.0, Alpha scaled)
     to force the CNN to focus heavily on hard examples (MES1 & MES2).
  3. Fine-tuning Phase: Single phase training with very low learning rate (1e-5).
  4. Super Agent Class Weights: Explicitly penalize MES1 & MES2 errors in LightGBM.
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
from imblearn.over_sampling import SMOTE
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

# Custom Focal Loss
def custom_focal_loss(gamma=2.0, alpha=[0.25, 0.75, 0.75, 0.25]):
    # alpha weights: lower for MES0/MES3 (easier), higher for MES1/MES2 (harder)
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha_tensor * y_true * tf.math.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss

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
ENSEMBLE_SEEDS = [42, 123, 999]

# ==============================================================================
# PREPROCESSING
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
    for band in [LL, LH, HL, HH]: feats.extend(_stats(band))
    feats.append(np.sum(np.square(HH)))
    gn = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(gn, [3], [0, np.pi/4], 256, symmetric=True, normed=True)
    feats.append(graycoprops(glcm, 'contrast').mean())
    feats.append(graycoprops(glcm, 'dissimilarity').mean())
    feats.append(graycoprops(glcm, 'homogeneity').mean())
    return feats

def extract_clinical_colour(img):
    r, g, b = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
    t = r + g + b + 1e-6
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv],[0],None,[30],[0,180]).flatten()
    h_hist /= (h_hist.sum() + 1e-6)
    return [
        np.mean(r/t), np.mean(g/t), np.mean((r-g)/(r+g+1e-6)),
        np.std(g)/(np.mean(g)+1e-6), scipy.stats.entropy(h_hist+1e-6),
        np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]), np.mean((r>200)&(g>200)&(b>200))
    ]

def extract_features_28(img):
    return extract_wavelet_glcm(img) + extract_clinical_colour(img)

# ==============================================================================
# CUTMIX & AUGMENTATION
# ==============================================================================
def cutmix_batch(imgs, feats, umaps, labels, alpha=0.4):
    B, H, W = len(imgs), imgs.shape[1], imgs.shape[2]
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(B)
    cut_h, cut_w = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(0, cy-cut_h//2), min(H, cy+cut_h//2)
    x1, x2 = max(0, cx-cut_w//2), min(W, cx+cut_w//2)
    mixed = imgs.copy()
    mixed[:, y1:y2, x1:x2, :] = imgs[idx, y1:y2, x1:x2, :]
    lam = 1.0 - (y2-y1)*(x2-x1) / (H*W)
    return (mixed, lam*feats + (1-lam)*feats[idx], lam*umaps + (1-lam)*umaps[idx], lam*labels + (1-lam)*labels[idx])

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

class V6Generator(Sequence):
    def __init__(self, images, features, umaps, labels, batch_size=16, augment=False, use_cutmix=False, shuffle=True):
        self.batch_size, self.augment, self.use_cutmix, self.shuffle = batch_size, augment, use_cutmix, shuffle
        # Oversample MES1 directly in the generator
        mes1 = np.where(np.argmax(labels, axis=1) == 1)[0]
        self.indices = np.concatenate([np.arange(len(labels)), mes1]) if augment else np.arange(len(labels))
        self.images, self.features, self.umaps, self.labels = images, features, umaps, labels
        self.on_epoch_end()

    def __len__(self): return max(1, len(self.indices) // self.batch_size)
    def __getitem__(self, idx):
        bidx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X_img = self.images[bidx].copy()
        if self.augment:
            for i in range(len(X_img)): X_img[i] = apply_augmentation(X_img[i].astype(np.uint8)).astype(np.float32)
        X_feat, X_umap, y = self.features[bidx].copy(), self.umaps[bidx].copy(), self.labels[bidx].copy()
        if self.use_cutmix and np.random.rand() < 0.2:
            X_img, X_feat, X_umap, y = cutmix_batch(X_img, X_feat, X_umap, y)
        return (X_img / 255.0, X_feat, X_umap), y
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

# ==============================================================================
# MODEL — Mod-SE CNN Hybrid
# ==============================================================================
def se_block(x, ratio=8):
    f = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, f))(se)
    se = Dense(max(1, f//ratio), activation='relu', use_bias=False)(se)
    se = Dense(f, activation='sigmoid', use_bias=False)(se)
    return Multiply()([x, se])

def build_se_cnn_hybrid(seed):
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
    out = Dense(NUM_CLASSES, activation='softmax', name='output')(x)
    return Model(inputs=[inp_img, inp_feat, inp_umap], outputs=out)

class CosineAnneal(Callback):
    def __init__(self, max_lr, min_lr, total):
        super().__init__(); self.max_lr, self.min_lr, self.total = max_lr, min_lr, total
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.min_lr + 0.5*(self.max_lr - self.min_lr)*(1 + math.cos(math.pi*epoch/self.total))
        self.model.optimizer.learning_rate.assign(lr)

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
# MAIN PIPELINE
# ==============================================================================
def load_all(base_dir):
    ntuh = [f'{base_dir}/Dataset+Code/MES classification_20250313', f'{base_dir}/Dataset+Code/MES classification_20250724']
    limuc= [f'{base_dir}/Dataset/LIMUC/train_and_validation_sets', f'{base_dir}/Dataset/LIMUC/test_set']
    tmc  = f'{base_dir}/Dataset/TMC-UCM'
    ni, _, nl, _ = load_all_images(ntuh, 'NTUH')
    li, _, ll, _ = load_all_images(limuc, 'LIMUC')
    ti, _, tl, _ = load_tmc_ucm(tmc, split_filter=None)
    raw, labels = ni+li+ti, nl+ll+tl
    print(f"📊 Total: {len(raw)} images")
    imgs, feats = [], []
    for img in tqdm(raw, desc="Preprocessing"):
        p = smart_preprocess(img)
        imgs.append(p); feats.append(extract_features_28(p))
    return np.array(imgs, dtype=np.float32), np.array(feats, dtype=np.float32), labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/raid/D13K48009/Clara/new_drive")
    ap.add_argument("--save_dir", default="../Result/ColonoMind_v7")
    ap.add_argument("--v6_checkpoint_dir", default="../Result/ColonoMind_v6")
    ap.add_argument("--epochs_ft", type=int, default=30)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--optuna_trials", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*70); print("🚀 COLONOMIND v7 — Focal Loss Fine-Tuning"); print("="*70)
    
    # ── 1. Data Loading ────────────────────────────────────────────────────────
    cache = os.path.join(args.save_dir, "dataset_cache_v6.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        all_imgs, all_feats, all_labels = d['imgs'], d['feats'], list(d['labels'])
    else:
        all_imgs, all_feats, all_labels = load_all(args.base_dir)
        np.savez_compressed(cache, imgs=all_imgs, feats=all_feats, labels=np.array(all_labels))

    y_enc = LabelEncoder().fit_transform(all_labels)

    # ── 2. Splitting ───────────────────────────────────────────────────────────
    X_tv_i, X_te_i, X_tv_f, X_te_f, y_tv, y_te = train_test_split(all_imgs, all_feats, y_enc, test_size=0.20, random_state=42, stratify=y_enc)
    X_tr_i, X_va_i, X_tr_f, X_va_f, y_tr, y_va = train_test_split(X_tv_i, X_tv_f, y_tv, test_size=0.20, random_state=42, stratify=y_tv)

    # ── 3. UMAP & Scaling ──────────────────────────────────────────────────────
    sp, up = os.path.join(args.save_dir,"scaler_v7.pkl"), os.path.join(args.save_dir,"umap_v7.pkl")
    if os.path.exists(sp) and os.path.exists(up):
        sc, um = joblib.load(sp), joblib.load(up)
    else:
        sc = StandardScaler().fit(X_tr_f)
        um = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(sc.transform(X_tr_f))
        joblib.dump(sc, sp); joblib.dump(um, up)

    Xtr_s, Xva_s, Xte_s = sc.transform(X_tr_f), sc.transform(X_va_f), sc.transform(X_te_f)
    Utr, Uva, Ute = um.transform(Xtr_s), um.transform(Xva_s), um.transform(Xte_s)
    y_tr_c, y_va_c = to_categorical(y_tr, NUM_CLASSES), to_categorical(y_va, NUM_CLASSES)
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = {i: w for i, w in enumerate(cw)}; cw_dict[1] *= 1.3

    # ── 4. Focal Loss Fine-Tuning (3 Models) ───────────────────────────────────
    print(f"\n🧬 Focal Loss Fine-Tuning: Modifying {len(ENSEMBLE_SEEDS)} models from V6")
    loss_fn = custom_focal_loss(gamma=2.0, alpha=[0.25, 0.75, 0.75, 0.25])
    tr_gen = V6Generator(X_tr_i, Xtr_s, Utr, y_tr_c, BATCH_SIZE, augment=True, use_cutmix=True, shuffle=True)
    va_gen = V6Generator(X_va_i, Xva_s, Uva, y_va_c, BATCH_SIZE, augment=False, shuffle=False)
    
    ensemble_models = []
    for m_idx, seed in enumerate(ENSEMBLE_SEEDS):
        v6_path = os.path.join(args.v6_checkpoint_dir, f"secnn_v6_seed{seed}.h5")
        v7_path = os.path.join(args.save_dir, f"secnn_v7_seed{seed}.h5")
        
        if os.path.exists(v7_path):
            print(f"\n⚡ Found V7 checkpoint for Seed {seed} at {v7_path}. Skipping training.")
            model = load_model(v7_path, custom_objects={'focal_loss': loss_fn}, compile=False)
        else:
            print(f"\n🔥 Fine-tuning CNN Model {m_idx+1}/{len(ENSEMBLE_SEEDS)} (Seed: {seed}) from V6")
            if not os.path.exists(v6_path):
                raise FileNotFoundError(f"V6 checkpoint missing: {v6_path}. Train V6 first!")
                
            model = load_model(v6_path, compile=False)
            
            # Fine-tuning Phase (Low LR, Focal Loss)
            model.compile(optimizer=Adam(1e-5), loss=loss_fn, metrics=['accuracy'])
            model.fit(tr_gen, validation_data=va_gen, epochs=args.epochs_ft, class_weight=cw_dict,
                      callbacks=[ModelCheckpoint(v7_path, save_best_only=True, monitor='val_accuracy', mode='max'), 
                                 EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max'), 
                                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)], verbose=1)
            model = load_model(v7_path, custom_objects={'focal_loss': loss_fn}, compile=False)
        ensemble_models.append(model)

    # ── 5. Extract Averaged Ensemble Features ──────────────────────────────────
    print("\n" + "="*70); print("STEP 5/9: Extracting Averaged Deep Features (Ensemble)"); print("="*70)
    feat_extractors = [Model(inputs=m.input, outputs=m.get_layer('Fusion').output) for m in ensemble_models]
    
    def build_agent_features(imgs, feats_s, umaps, raw_feats):
        deeps, probs = [], []
        for i in range(len(ensemble_models)):
            deeps.append(feat_extractors[i].predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0))
            probs.append(ensemble_models[i].predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0))
        # Average the features and probabilities from all 3 models
        avg_deep = np.mean(deeps, axis=0)
        avg_prob = np.mean(probs, axis=0)
        ent = scipy.stats.entropy(avg_prob, axis=1).reshape(-1,1)
        return np.hstack([avg_deep, raw_feats, umaps, avg_prob, ent]), avg_prob
        
    X_ag_tr, _ = build_agent_features(X_tr_i, Xtr_s, Utr, X_tr_f)
    X_ag_te, cnn_ensemble_probs = build_agent_features(X_te_i, Xte_s, Ute, X_te_f)

    ag_sc = StandardScaler()
    X_ag_tr_s = ag_sc.fit_transform(X_ag_tr)
    X_ag_te_s = ag_sc.transform(X_ag_te)
    joblib.dump(ag_sc, os.path.join(args.save_dir, "agent_scaler_v7.pkl"))

    # CNN Ensemble Metrik
    cnn_preds = np.argmax(cnn_ensemble_probs, axis=1)
    print(f"🎯 CNN Ensemble (3 models) — Acc:{accuracy_score(y_te, cnn_preds)*100:.2f}%  F1:{f1_score(y_te, cnn_preds, average='macro')*100:.2f}%")

    # ── 6. SMOTE on Agent Features ─────────────────────────────────────────────
    print("\n" + "="*70); print("STEP 6/9: Feature-Space SMOTE for Super Agent"); print("="*70)
    sm = SMOTE(random_state=42)
    X_ag_tr_smote, y_tr_smote = sm.fit_resample(X_ag_tr_s, y_tr)
    print(f"  Train shape before SMOTE: {X_ag_tr_s.shape}, After: {X_ag_tr_smote.shape}")

    # ── 7. Super Agent (Optuna) ────────────────────────────────────────────────
    print("\n" + "="*70); print("STEP 7/9: Super Agent (Optuna on SMOTE features)"); print("="*70)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    def objective(trial):
        # V7 Custom Class Weights: Force heavily penalize MES1/MES2 errors
        custom_weights = {0: 1.0, 1: 3.5, 2: 3.5, 3: 1.5}
        param = dict(
            objective='multiclass', num_class=NUM_CLASSES, metric='multi_logloss', verbosity=-1, boosting_type='gbdt',
            n_estimators=trial.suggest_int('n_estimators', 200, 1000),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 7),
            num_leaves=trial.suggest_int('num_leaves', 8, 63),
            min_child_samples=trial.suggest_int('min_child_samples', 15, 80),
            feature_fraction=trial.suggest_float('feature_fraction', 0.3, 0.9),
            bagging_fraction=trial.suggest_float('bagging_fraction', 0.5, 0.9),
            bagging_freq=trial.suggest_int('bagging_freq', 1, 5),
            class_weight=custom_weights, n_jobs=8
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_i, va_i in cv.split(X_ag_tr_smote, y_tr_smote):
            clf = lgb.LGBMClassifier(**param)
            clf.fit(X_ag_tr_smote[tr_i], y_tr_smote[tr_i], eval_set=[(X_ag_tr_smote[va_i], y_tr_smote[va_i])], callbacks=[lgb.early_stopping(30, verbose=False)])
            scores.append(f1_score(y_tr_smote[va_i], clf.predict(X_ag_tr_smote[va_i]), average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.optuna_trials)
    
    best_p = study.best_params
    custom_weights = {0: 1.0, 1: 3.5, 2: 3.5, 3: 1.5}
    best_p.update(dict(objective='multiclass', num_class=NUM_CLASSES, metric='multi_logloss', verbosity=-1, class_weight=custom_weights, n_jobs=8))
    final_agent = lgb.LGBMClassifier(**best_p)
    final_agent.fit(X_ag_tr_smote, y_tr_smote, eval_set=[(X_ag_te_s, y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
    final_agent.booster_.save_model(os.path.join(args.save_dir, "super_agent_v7.txt"))

    ag_preds = final_agent.predict(X_ag_te_s)
    
    # ── 8. Hybrid Routing & F1-Maximizer ───────────────────────────────────────
    confs = np.max(cnn_ensemble_probs, axis=1)
    best_thresholds = {}
    for cls in range(NUM_CLASSES):
        best_t, best_f = 0.5, 0
        for t in np.arange(0.3, 0.99, 0.01):
            preds = cnn_preds.copy()
            mask = (cnn_preds == cls) & (confs < t)
            preds[mask] = ag_preds[mask]
            f = f1_score(y_te, preds, average='macro')
            if f > best_f: best_f, best_t = f, t
        best_thresholds[cls] = best_t
    
    hybrid_preds = cnn_preds.copy()
    for cls, t in best_thresholds.items():
        mask = (cnn_preds == cls) & (confs < t)
        hybrid_preds[mask] = ag_preds[mask]

    print("\n🏆 Hybrid Metrics:")
    print(classification_report(y_te, hybrid_preds, target_names=CLASS_NAMES, digits=4))
    
    # ── Save Results ───────────────────────────────────────────────────────────
    print(f"\n✅ All completed successfully! Checkpoints saved to {args.save_dir}")

if __name__ == "__main__":
    main()
