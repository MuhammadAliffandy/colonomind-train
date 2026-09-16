"""
ColonoMind v9 — Mod-SE V2 Architecture (CBAM + Residuals)
=========================================================
Optimisations for High Accuracy AND High F1 on MES1/MES2:
  1. Mod-SE V2 Architecture: 
     - Replaced basic SE (Squeeze-and-Excitation) with CBAM (Convolutional Block Attention Module)
       for Spatial + Channel Attention (allows the CNN to pinpoint WHERE the tiny ulcers are).
     - Added Residual Connections (Add shortcuts) to allow deeper feature extraction without vanishing gradients.
  2. Focal Loss Integration to aggressively penalize errors on MES1/MES2.
  3. LightGBM Super Agent Fusion at the end.
"""
import os, cv2, argparse, gc
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import lightgbm as lgb
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import umap.umap_ as umap
import joblib

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, BatchNormalization,
                                     Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D,
                                     MaxPooling2D, Activation, Multiply, Reshape, Add, Lambda)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)
import tensorflow.keras.backend as K

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# CONFIG
# ==============================================================================
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 16
FEAT_DIM    = 28   
ENSEMBLE_SEEDS = [42, 123, 999]

# ==============================================================================
# PREPROCESSING & AUGMENTATION
# ==============================================================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

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

class V9Generator(Sequence):
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
# LOSS FUNCTION
# ==============================================================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# ==============================================================================
# MOD-SE V2 (CBAM + Residuals) ARCHITECTURE
# ==============================================================================
def cbam_block(cbam_feature, ratio=8):
    # Channel attention
    channels = cbam_feature.shape[-1]
    shared_l1 = Dense(max(1, channels//ratio), activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_l2 = Dense(channels, kernel_initializer='he_normal', use_bias=True)
    
    avg_pool = GlobalAveragePooling2D()(cbam_feature)    
    avg_pool = Reshape((1,1,channels))(avg_pool)
    avg_pool = shared_l2(shared_l1(avg_pool))
    
    max_pool = GlobalMaxPooling2D()(cbam_feature)
    max_pool = Reshape((1,1,channels))(max_pool)
    max_pool = shared_l2(shared_l1(max_pool))
    
    cbam_feature_c = Add()([avg_pool, max_pool])
    cbam_feature_c = Activation('sigmoid')(cbam_feature_c)
    channel_attention = Multiply()([cbam_feature, cbam_feature_c])
    
    # Spatial attention
    avg_pool_s = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(channel_attention)
    max_pool_s = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(channel_attention)
    concat = Concatenate(axis=-1)([avg_pool_s, max_pool_s])
    cbam_feature_s = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    spatial_attention = Multiply()([channel_attention, cbam_feature_s])
    
    return spatial_attention

def res_cbam_layer(x, filters, kernel_size=(3,3), strides=(1,1)):
    # Shortcut path
    if x.shape[-1] != filters or strides != (1,1):
        shortcut = Conv2D(filters, (1,1), strides=strides, padding='same')(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x
        
    # Main path
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Apply CBAM Attention
    x = cbam_block(x)
    
    # Residual addition
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    
    # Reduce spatial dimension like pooling (since we use strides=2 in the first Conv2D)
    # Wait, the first Conv2D already handled stride=2. No need to pool.
    return x

def build_modse_v2_hybrid(seed, num_classes=4):
    tf.random.set_seed(seed); np.random.seed(seed)
    
    inp_img = Input(shape=(*IMG_SIZE, 3), name='input_image')
    x = Conv2D(32, (3,3), padding='same')(inp_img)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    
    x = res_cbam_layer(x, 64, strides=(2,2)); x = Dropout(0.2)(x)
    x = res_cbam_layer(x, 128, strides=(2,2)); x = Dropout(0.3)(x)
    x = res_cbam_layer(x, 256, strides=(2,2)); x = Dropout(0.3)(x)
    x = res_cbam_layer(x, 512, strides=(2,2))
    
    feat_cnn = GlobalAveragePooling2D()(x)
    feat_cnn = Dropout(0.4)(feat_cnn)

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
    ap.add_argument("--save_dir", default="../Result/ColonoMind_v9_ModSEV2")
    ap.add_argument("--v6_checkpoint_dir", default="../Result/ColonoMind_v6")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*70); print("🚀 COLONOMIND v9 — Mod-SE V2 (CBAM + Residual)"); print("="*70)
    
    # ── 1. Load Pre-cached Data ──
    cache = os.path.join(args.v6_checkpoint_dir, "dataset_cache_v6.npz")
    d = np.load(cache, allow_pickle=True)
    all_imgs, all_feats, all_labels = d['imgs'], d['feats'], list(d['labels'])
    y_enc_full = LabelEncoder().fit_transform(all_labels) # 0, 1, 2, 3
    
    # ── 2. Data Splitting ──
    X_tv_i, X_te_i, X_tv_f, X_te_f, y_tv, y_te_full = train_test_split(all_imgs, all_feats, y_enc_full, test_size=0.20, random_state=42, stratify=y_enc_full)
    X_tr_i, X_va_i, X_tr_f, X_va_f, y_tr, y_va = train_test_split(X_tv_i, X_tv_f, y_tv, test_size=0.20, random_state=42, stratify=y_tv)

    # ── 3. UMAP & Scaling ──
    sp, up = os.path.join(args.save_dir,"scaler_v9.pkl"), os.path.join(args.save_dir,"umap_v9.pkl")
    if os.path.exists(sp) and os.path.exists(up):
        sc, um = joblib.load(sp), joblib.load(up)
    else:
        sc = StandardScaler().fit(X_tr_f)
        um = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(sc.transform(X_tr_f))
        joblib.dump(sc, sp); joblib.dump(um, up)
    
    def process_feats(features):
        scaled = sc.transform(features)
        return scaled, um.transform(scaled)
        
    Xtr_s, Utr = process_feats(X_tr_f); Xva_s, Uva = process_feats(X_va_f); Xte_s, Ute = process_feats(X_te_f)
    
    y_tr_c, y_va_c = to_categorical(y_tr, 4), to_categorical(y_va, 4)
    # Focal loss alpha balancing
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    alpha = cw / np.sum(cw)
    focal_loss = categorical_focal_loss(gamma=2.0, alpha=alpha)

    # ── 4. Train Mod-SE V2 CNN Ensembles ──
    models = []
    tr_gen = V9Generator(X_tr_i, Xtr_s, Utr, y_tr_c, BATCH_SIZE, augment=True)
    va_gen = V9Generator(X_va_i, Xva_s, Uva, y_va_c, BATCH_SIZE, augment=False)
    
    for seed in ENSEMBLE_SEEDS:
        model_path = os.path.join(args.save_dir, f"modsev2_cnn_seed{seed}.h5")
        if os.path.exists(model_path):
            print(f"⚡ Loading Mod-SE V2 {seed}")
            model = load_model(model_path, compile=False)
        else:
            print(f"🔥 Training Mod-SE V2 from scratch {seed}")
            model = build_modse_v2_hybrid(seed, num_classes=4)
            model.compile(optimizer=Adam(1e-4), loss=focal_loss, metrics=['accuracy'])
            model.fit(tr_gen, validation_data=va_gen, epochs=args.epochs,
                      callbacks=[ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy'), EarlyStopping(patience=8)], verbose=1)
            model = load_model(model_path, compile=False)
        models.append(model)

    # ── 5. Train Super Agent ──
    print("\n" + "="*70); print("Training LightGBM Super Agent"); print("="*70)
    def extract_deep(models, imgs, feats_s, umaps):
        deeps, probs = [], []
        extractors = [Model(inputs=m.input, outputs=m.get_layer('Fusion').output) for m in models]
        for i in range(len(models)):
            deeps.append(extractors[i].predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0))
            probs.append(models[i].predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0))
        return np.hstack([np.mean(deeps, axis=0), feats_s, umaps, np.mean(probs, axis=0)]), np.mean(probs, axis=0)

    X_ag_tr, _ = extract_deep(models, X_tr_i, Xtr_s, Utr)
    X_ag_te, _ = extract_deep(models, X_te_i, Xte_s, Ute)
    
    # Strong LightGBM with custom weights focusing on MES1 and MES2
    sample_weights = np.array([3.5 if y in [1, 2] else 1.0 for y in y_tr])
    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.01, class_weight='balanced', max_depth=6, num_leaves=40)
    lgbm.fit(X_ag_tr, y_tr, sample_weight=sample_weights)
    lgbm.booster_.save_model(os.path.join(args.save_dir, "super_agent_v9.txt"))
    
    final_preds = lgbm.predict(X_ag_te)
    
    print("\n🏆 V9 Mod-SE V2 Final Metrics:")
    print(classification_report(y_te_full, final_preds, target_names=['MES0', 'MES1', 'MES2', 'MES3'], digits=4))
    print(f"\n✅ Pipeline Complete! Overall Accuracy: {accuracy_score(y_te_full, final_preds)*100:.2f}%")

if __name__ == "__main__":
    main()
