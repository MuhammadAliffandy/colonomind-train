"""
ColonoMind v11 — Final Fusion (Hybrid Loss)
===========================================
Combines the precision of the Two-Stage Hierarchical Classification (V8)
with the powerful Convolutional Block Attention Module architecture (V9).

Optimisations for >80% Accuracy AND Balanced F1:
  1. Stage 1 (Detector): Mod-SE V2 with Categorical Crossentropy (Protects MES0).
  2. Stage 2 (Grader): Mod-SE V2 with Focal Loss (Boosts MES1/MES2).
  3. Trains everything from scratch (no V6 checkpoints required).
"""
import os, cv2, json, joblib, argparse, gc
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

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, BatchNormalization,
                                     Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D,
                                     MaxPooling2D, Activation, Multiply, Reshape, Add, Layer)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images, load_tmc_ucm

# ==============================================================================
# CONFIG
# ==============================================================================
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 16
FEAT_DIM    = 28   
ENSEMBLE_SEEDS = [42, 123, 999]

# ==============================================================================
# LOSS FUNCTION
# ==============================================================================
import tensorflow.keras.backend as K

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
@tf.keras.utils.register_keras_serializable()
class ChannelPooling(Layer):
    def __init__(self, pool_type='mean', **kwargs):
        super(ChannelPooling, self).__init__(**kwargs)
        self.pool_type = pool_type
        
    def call(self, inputs):
        if self.pool_type == 'mean':
            return tf.reduce_mean(inputs, axis=-1, keepdims=True)
        else:
            return tf.reduce_max(inputs, axis=-1, keepdims=True)
            
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)
        
    def get_config(self):
        config = super(ChannelPooling, self).get_config()
        config.update({'pool_type': self.pool_type})
        return config

def cbam_block(cbam_feature, ratio=8):
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
    
    avg_pool_s = ChannelPooling('mean')(channel_attention)
    max_pool_s = ChannelPooling('max')(channel_attention)
    concat = Concatenate(axis=-1)([avg_pool_s, max_pool_s])
    cbam_feature_s = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    spatial_attention = Multiply()([channel_attention, cbam_feature_s])
    
    return spatial_attention

def res_cbam_layer(x, filters, kernel_size=(3,3), strides=(1,1)):
    if x.shape[-1] != filters or strides != (1,1):
        shortcut = Conv2D(filters, (1,1), strides=strides, padding='same')(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x
        
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def build_modse_v2_hybrid(seed, num_classes):
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
# DATA GENERATOR (WITH AUGMENTATION)
# ==============================================================================
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

class V10Generator(Sequence):
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
        return (X_img / 255.0, self.features[bidx], self.umaps[bidx]), self.labels[bidx]
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='../Dataset')
    parser.add_argument('--save_dir', type=str, default='../Result/ColonoMind_v11_FinalFusion')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    print("="*70); print("🚀 COLONOMIND v11 — Final Fusion (Hybrid Loss)"); print("="*70)
    
    # ── 1. Load Pre-cached Data ───────────────────────────────────────────────
    # V10 relies on the same base dataset cache as V6/V8 for valid comparison
    cache = "../Result/ColonoMind_v6/dataset_cache_v6.npz"
    if not os.path.exists(cache):
        raise FileNotFoundError(f"Cache not found: {cache}. Run V6 first.")
    
    d = np.load(cache, allow_pickle=True)
    all_imgs, all_feats, all_labels = d['imgs'], d['feats'], list(d['labels'])
    y_enc_full = LabelEncoder().fit_transform(all_labels) # 0, 1, 2, 3
    
    # ── 2. Universal Data Splitting (Matches V6/V8 EXACTLY) ───────────────────
    X_tv_i, X_te_i, X_tv_f, X_te_f, y_tv, y_te_full = train_test_split(all_imgs, all_feats, y_enc_full, test_size=0.20, random_state=42, stratify=y_enc_full)
    X_tr_i, X_va_i, X_tr_f, X_va_f, y_tr, y_va = train_test_split(X_tv_i, X_tv_f, y_tv, test_size=0.20, random_state=42, stratify=y_tv)
    
    # ── Create Hierarchical Subsets ──
    # STAGE 1: 0 (Normal) vs 1 (Active Disease)
    y_tr1 = np.where(y_tr == 0, 0, 1); y_va1 = np.where(y_va == 0, 0, 1); y_te1 = np.where(y_te_full == 0, 0, 1)
    
    # STAGE 2: Only Active Disease subset. (MES1 -> 0, MES2 -> 1, MES3 -> 2)
    active_tr = np.where(y_tr > 0)[0]; active_va = np.where(y_va > 0)[0]
    X_tr_i2, X_tr_f2, y_tr2 = X_tr_i[active_tr], X_tr_f[active_tr], y_tr[active_tr] - 1
    X_va_i2, X_va_f2, y_va2 = X_va_i[active_va], X_va_f[active_va], y_va[active_va] - 1

    # ── 3. UMAP & Scaling ──────────────────────────────────────────────────────
    sp, up = os.path.join(args.save_dir,"scaler_v10.pkl"), os.path.join(args.save_dir,"umap_v10.pkl")
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
    Xtr_s2, Utr2 = process_feats(X_tr_f2); Xva_s2, Uva2 = process_feats(X_va_f2)

    # One-hot encoding & Class Weights
    y_tr1_c, y_va1_c = to_categorical(y_tr1, 2), to_categorical(y_va1, 2)
    y_tr2_c, y_va2_c = to_categorical(y_tr2, 3), to_categorical(y_va2, 3)
    
    cw1 = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr1), y=y_tr1)
    cw2 = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr2), y=y_tr2)

    # ── 4. STAGE 1 (Binary Detector) Training ──────────────────────────────
    print("\n" + "="*70); print("STAGE 1/4: Training Mod-SE V2 Detector (Crossentropy)"); print("="*70)
    stage1_models = []
    loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
    
    tr_gen1 = V10Generator(X_tr_i, Xtr_s, Utr, y_tr1_c, BATCH_SIZE, augment=True)
    va_gen1 = V10Generator(X_va_i, Xva_s, Uva, y_va1_c, BATCH_SIZE, augment=False)
    
    for seed in ENSEMBLE_SEEDS:
        stg1_path = os.path.join(args.save_dir, f"stage1_modsev2_seed{seed}.h5")
        if os.path.exists(stg1_path):
            print(f"⚡ Loading Stage 1 Mod-SE V2 {seed}")
            model = load_model(stg1_path, compile=False, safe_mode=False, custom_objects={'ChannelPooling': ChannelPooling})
        else:
            print(f"🔥 Training Stage 1 Mod-SE V2 from scratch {seed}")
            model = build_modse_v2_hybrid(seed, num_classes=2)
            model.compile(optimizer=Adam(1e-4), loss=loss_fn, metrics=['accuracy'])
            model.fit(tr_gen1, validation_data=va_gen1, epochs=args.epochs, class_weight=dict(enumerate(cw1)),
                      callbacks=[ModelCheckpoint(stg1_path, save_best_only=True, monitor='val_accuracy'), EarlyStopping(patience=8)], verbose=1)
            model = load_model(stg1_path, compile=False, safe_mode=False, custom_objects={'ChannelPooling': ChannelPooling})
        stage1_models.append(model)

    # ── 5. STAGE 2 (Ternary Grader) Training ───────────────────────────────
    print("\n" + "="*70); print("STAGE 2/4: Training Mod-SE V2 Grader (Focal Loss)"); print("="*70)
    stage2_models = []
    
    tr_gen2 = V10Generator(X_tr_i2, Xtr_s2, Utr2, y_tr2_c, BATCH_SIZE, augment=True)
    va_gen2 = V10Generator(X_va_i2, Xva_s2, Uva2, y_va2_c, BATCH_SIZE, augment=False)
    
    for seed in ENSEMBLE_SEEDS:
        stg2_path = os.path.join(args.save_dir, f"stage2_modsev2_seed{seed}.h5")
        if os.path.exists(stg2_path):
            print(f"⚡ Loading Stage 2 Mod-SE V2 {seed}")
            model = load_model(stg2_path, compile=False, safe_mode=False, custom_objects={'ChannelPooling': ChannelPooling})
        else:
            print(f"🔥 Training Stage 2 Mod-SE V2 from scratch {seed}")
            model = build_modse_v2_hybrid(seed, num_classes=3)
            model.compile(optimizer=Adam(1e-4), loss=categorical_focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])
            model.fit(tr_gen2, validation_data=va_gen2, epochs=args.epochs, class_weight=dict(enumerate(cw2)),
                      callbacks=[ModelCheckpoint(stg2_path, save_best_only=True, monitor='val_accuracy'), EarlyStopping(patience=8)], verbose=1)
            model = load_model(stg2_path, compile=False, safe_mode=False, custom_objects={'ChannelPooling': ChannelPooling})
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
    X_ag_tr1, _ = extract_deep(stage1_models, X_tr_i, Xtr_s, Utr, 2)
    X_ag_te1, prob_stg1 = extract_deep(stage1_models, X_te_i, Xte_s, Ute, 2)
    lgb1 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.01, class_weight='balanced', max_depth=5, num_leaves=31)
    lgb1.fit(X_ag_tr1, y_tr1)
    ag_pred_1 = lgb1.predict(X_ag_te1)
    
    # Stage 2 Agent
    X_ag_tr2, _ = extract_deep(stage2_models, X_tr_i2, Xtr_s2, Utr2, 3)
    X_ag_te2_full, prob_stg2_full = extract_deep(stage2_models, X_te_i, Xte_s, Ute, 3) # Predict on FULL test set to route later
    lgb2 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.01, class_weight='balanced', max_depth=6, num_leaves=40)
    lgb2.fit(X_ag_tr2, y_tr2)
    ag_pred_2_full = lgb2.predict(X_ag_te2_full)
    
    # Save Super Agents
    lgb1.booster_.save_model(os.path.join(args.save_dir, "super_agent_stage1.txt"))
    lgb2.booster_.save_model(os.path.join(args.save_dir, "super_agent_stage2.txt"))

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
    
    print("\n🏆 V11 Final Fusion (Hybrid Loss) Final Metrics:")
    print(classification_report(y_te_full, final_preds, target_names=['MES0', 'MES1', 'MES2', 'MES3'], digits=4))
    print(f"\n✅ Pipeline Complete! Overall Accuracy: {accuracy_score(y_te_full, final_preds)*100:.2f}%")

if __name__ == "__main__":
    main()
