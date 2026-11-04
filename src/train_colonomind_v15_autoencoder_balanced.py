"""
ColonoMind v15 — Autoencoder-Pretrained, Cohort-Balanced, High-Res Hierarchical Pipeline
========================================================================================
Puncak Evolusi ColonoMind:
  1. Unsupervised Convolutional Autoencoder (CAE) Pretraining from Scratch:
     - 100% compliant dengan batas riset (tanpa ImageNet pre-trained weights).
     - Mempelajari representasi tekstur mikroskopis mukosa usus, pola vaskular, dan ulkus
       secara unsupervised sebelum bobot diwariskan ke classifier.
     - Mengotomatiskan pergeseran bobot (weight transfer) tanpa representation bottleneck.
  2. Cohort & Domain Balancing (Harmonisasi Senter):
     - Mengatasi ketimpangan masif: NTUH (199 gambar / ~5%) vs LIMUC (1686 gambar) vs TMC-UCM (1596 gambar).
     - Domain-stratified resampling: NTUH diaugmentasi dan diseimbangkan kuotanya dalam batch,
       mencegah LIMUC dan TMC mendominasi representasi fitur model.
  3. Spatial Upsampling (High-Resolution 384x384):
     - Mempertahankan eritema halus dan pola vaskular mikroskopis (kunci pembeda MES1 vs MES2)
       yang selama ini hilang akibat downscale agresif ke 256x256.
  4. Hierarchical Two-Stage Mod-SE Architecture (Warisan Kejayaan V8):
     - Stage 1: Detector (Normal vs Inflamed) -> Mengamankan F1 MES0 ~89-90%.
     - Stage 2: Ternary Grader (MES1 vs MES2 vs MES3) -> Spesialis derajat keparahan radang.
     - Deep Ensembling (3 Seeds) + Super Agents LightGBM (Deep + 28 Biomarker + UMAP).
  5. Automated End-to-End Evaluation & Dual-Metric Reporting:
     - Exact Match 4-Class Accuracy (>80%).
     - Clinical 1-Grade Off / Top-2 Tolerance Accuracy (>90%).
     - Per-Cohort Accuracy Breakdown (NTUH, LIMUC, TMC-UCM).
"""

import os, cv2, json, joblib, pywt, argparse, gc, math, time
import scipy.stats
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import lightgbm as lgb
import tensorflow as tf
from tqdm import tqdm
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, cohen_kappa_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import umap.umap_ as umap

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, BatchNormalization,
                                     Dropout, GlobalAveragePooling2D, Conv2D,
                                     Conv2DTranspose, UpSampling2D, MaxPooling2D,
                                     Activation, Multiply, Reshape)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, Callback)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images, load_tmc_ucm

# ==============================================================================
# CONFIG & HYPERPARAMETERS
# ==============================================================================
DEFAULT_IMG_SIZE = (384, 384)
BATCH_SIZE       = 16
FEAT_DIM         = 28   # 20 Wavelet/GLCM + 8 Clinical Colour
DEFAULT_SEEDS    = [42, 123, 999]
CLASS_NAMES      = ['MES0', 'MES1', 'MES2', 'MES3']

# ==============================================================================
# PREPROCESSING & HANDCRAFTED BIOMARKERS (28-DIM)
# ==============================================================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def smart_preprocess(img, target_size=DEFAULT_IMG_SIZE):
    if img is None:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    # Crop black margin / patient info overlay if frame is large
    crop = img[30:430, 200:550] if h > 450 and w > 550 else img
    resized = cv2.resize(crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)
    return apply_clahe(resized)

def extract_wavelet_glcm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    LL, (LH, HL, HH) = pywt.dwt2(gray, 'db1')
    def _stats(b):
        f = np.abs(b.flatten()) + 1e-6
        return [float(np.mean(b)), float(np.std(b)), float(np.var(b)), float(scipy.stats.entropy(f))]
    feats = []
    for band in [LL, LH, HL, HH]:
        feats.extend(_stats(band))
    feats.append(float(np.sum(np.square(HH))))
    gn = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(gn, [3], [0, np.pi/4], 256, symmetric=True, normed=True)
    feats.append(float(graycoprops(glcm, 'contrast').mean()))
    feats.append(float(graycoprops(glcm, 'dissimilarity').mean()))
    feats.append(float(graycoprops(glcm, 'homogeneity').mean()))
    return feats

def extract_clinical_colour(img):
    r, g, b = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
    t = r + g + b + 1e-6
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180]).flatten()
    h_hist /= (h_hist.sum() + 1e-6)
    
    erythema_idx = float(np.mean((r - g) / (r + g + 1e-6)))
    vascular_idx = float(np.std(g) / (np.mean(g) + 1e-6))
    colour_entropy = float(scipy.stats.entropy(h_hist + 1e-6))
    
    return [
        float(np.mean(r / t)), float(np.mean(g / t)),
        erythema_idx, vascular_idx,
        colour_entropy,
        float(np.mean(hsv[:,:,1])), float(np.std(hsv[:,:,1])),
        float(np.mean((r > 200) & (g > 200) & (b > 200)))
    ]

def extract_features_28(img):
    return extract_wavelet_glcm(img) + extract_clinical_colour(img)

# ==============================================================================
# AUGMENTATION
# ==============================================================================
def apply_augmentation(img, is_minority_cohort=False):
    rows, cols = img.shape[:2]
    # Extra rich rotations for minority cohort (NTUH)
    angle_range = (-180, 180) if is_minority_cohort else (-90, 90)
    angle = np.random.uniform(*angle_range)
    tx, ty = np.random.uniform(-0.15, 0.15) * cols, np.random.uniform(-0.15, 0.15) * rows
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
    M[0, 2] += tx; M[1, 2] += ty
    img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() > 0.5: img = cv2.flip(img, 1)
    if np.random.rand() > 0.5: img = cv2.flip(img, 0)
    alpha = 1.0 + np.random.uniform(-0.25, 0.25)
    beta = np.random.uniform(-20, 20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# ==============================================================================
# COHORT-BALANCED BATCH GENERATOR
# ==============================================================================
class CohortBalancedGenerator(Sequence):
    """
    Ensures batches are balanced across cohorts (NTUH : LIMUC : TMC-UCM)
    and classes, preventing LIMUC/TMC from overwhelming NTUH during training.
    """
    def __init__(self, images, features, umaps, labels, cohorts, batch_size=16, augment=False, shuffle=True):
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        
        self.images = images
        self.features = features
        self.umaps = umaps
        self.labels = labels
        self.cohorts = cohorts
        
        # Group indices by cohort
        self.cohort_indices = {}
        for c in np.unique(cohorts):
            self.cohort_indices[c] = np.where(cohorts == c)[0]
        
        self.unique_cohorts = list(self.cohort_indices.keys())
        self.samples_per_cohort = self.batch_size // max(1, len(self.unique_cohorts))
        self.num_batches = max(1, len(self.labels) // self.batch_size)
        self.on_epoch_end()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        batch_indices = []
        if self.augment and len(self.unique_cohorts) > 1:
            # Balanced cohort sampling: take equal quota from each cohort
            for c in self.unique_cohorts:
                pool = self.cohort_indices[c]
                chosen = np.random.choice(pool, size=self.samples_per_cohort, replace=True)
                batch_indices.extend(chosen)
            # Fill remainder if any
            if len(batch_indices) < self.batch_size:
                extra = np.random.choice(len(self.labels), size=self.batch_size - len(batch_indices), replace=False)
                batch_indices.extend(extra)
        else:
            # Standard sequential sampling for validation/test
            start = idx * self.batch_size
            end = min(len(self.labels), (idx + 1) * self.batch_size)
            batch_indices = self.indices[start:end]

        X_img = self.images[batch_indices].copy()
        if self.augment:
            for i in range(len(X_img)):
                c = self.cohorts[batch_indices[i]]
                is_minority = (c == 'NTUH')
                X_img[i] = apply_augmentation(X_img[i].astype(np.uint8), is_minority_cohort=is_minority).astype(np.float32)

        X_feat = self.features[batch_indices].copy()
        X_umap = self.umaps[batch_indices].copy()
        y = self.labels[batch_indices].copy()
        
        return (X_img / 255.0, X_feat, X_umap), y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indices)

# Generator for Autoencoder (Unsupervised: input=X_img, output=X_img)
class AutoencoderGenerator(Sequence):
    def __init__(self, images, batch_size=16, augment=True, shuffle=True):
        self.images = images
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(images))
        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.images) // self.batch_size)

    def __getitem__(self, idx):
        bidx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = self.images[bidx].copy()
        if self.augment:
            for i in range(len(X)):
                X[i] = apply_augmentation(X[i].astype(np.uint8)).astype(np.float32)
        X_norm = X / 255.0
        return X_norm, X_norm

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==============================================================================
# ARCHITECTURE BUILDERS: SE-BLOCK, CONVOLUTIONAL AUTOENCODER, & HYBRID CLASSIFIER
# ==============================================================================
def se_block(x, ratio=8):
    f = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, f))(se)
    se = Dense(max(1, f // ratio), activation='relu', use_bias=False)(se)
    se = Dense(f, activation='sigmoid', use_bias=False)(se)
    return Multiply()([x, se])

def build_modse_backbone(input_shape):
    """
    Pure in-house Mod-SE CNN Feature Extractor (Encoder).
    5 Conv2D + SE stages from scratch.
    """
    inp = Input(shape=input_shape, name='backbone_input')
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same', name='conv1')(inp)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='act1')(x)
    x = se_block(x, 8)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    x = Dropout(0.2, name='drop1')(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu', name='act2')(x)
    x = se_block(x, 8)
    x = MaxPooling2D((2, 2), name='pool2')(x)
    x = Dropout(0.2, name='drop2')(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu', name='act3')(x)
    x = se_block(x, 8)
    x = MaxPooling2D((2, 2), name='pool3')(x)
    x = Dropout(0.3, name='drop3')(x)
    
    # Block 4
    x = Conv2D(256, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu', name='act4')(x)
    x = se_block(x, 8)
    x = MaxPooling2D((2, 2), name='pool4')(x)
    x = Dropout(0.3, name='drop4')(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Activation('relu', name='act5')(x)
    x = se_block(x, 8)
    
    return Model(inputs=inp, outputs=x, name='modse_encoder')

def build_autoencoder(input_shape):
    """
    Self-Supervised Convolutional Autoencoder (CAE).
    Pretrained from scratch without labels to learn mucosal representation.
    """
    encoder = build_modse_backbone(input_shape)
    inp = encoder.input
    enc_out = encoder.output  # Shape: (H//16, W//16, 512)
    
    # Decoder: Mirror upsampling back to input_shape
    d = Conv2D(256, (3, 3), padding='same', activation='relu')(enc_out)
    d = UpSampling2D((2, 2))(d)
    d = BatchNormalization()(d)
    
    d = Conv2D(128, (3, 3), padding='same', activation='relu')(d)
    d = UpSampling2D((2, 2))(d)
    d = BatchNormalization()(d)
    
    d = Conv2D(64, (3, 3), padding='same', activation='relu')(d)
    d = UpSampling2D((2, 2))(d)
    d = BatchNormalization()(d)
    
    d = Conv2D(32, (3, 3), padding='same', activation='relu')(d)
    d = UpSampling2D((2, 2))(d)
    d = BatchNormalization()(d)
    
    out = Conv2D(3, (3, 3), padding='same', activation='sigmoid', name='reconstruction_output')(d)
    
    return Model(inputs=inp, outputs=out, name='colonoscopy_autoencoder'), encoder

def build_se_cnn_hybrid(seed, num_classes, input_shape, pretrained_encoder=None):
    """
    Builds the ColonoMind Hybrid Model:
    Deep Mod-SE CNN + 28 Handcrafted Features + 2D UMAP -> Fusion Layer -> Softmax
    """
    tf.random.set_seed(seed); np.random.seed(seed)
    inp_img = Input(shape=input_shape, name='input_image')
    
    if pretrained_encoder is not None:
        # Transfer pretrained encoder weights seamlessly
        encoder = build_modse_backbone(input_shape)
        encoder.set_weights(pretrained_encoder.get_weights())
        enc_feat_map = encoder(inp_img)
    else:
        encoder = build_modse_backbone(input_shape)
        enc_feat_map = encoder(inp_img)
        
    feat_cnn = GlobalAveragePooling2D(name='gap_cnn')(enc_feat_map)
    feat_cnn = Dropout(0.4)(feat_cnn)

    # Handcrafted feature branch
    inp_feat = Input(shape=(FEAT_DIM,), name='input_feat')
    fh = BatchNormalization()(inp_feat)
    fh = Dense(128, activation='relu')(fh)
    fh = Dropout(0.2)(fh)
    feat_hand = Dense(64, activation='relu')(fh)
    
    # UMAP branch
    inp_umap = Input(shape=(2,), name='input_umap')
    feat_umap = Dense(32, activation='relu')(inp_umap)

    # Multimodal Fusion Layer
    combined = Concatenate(name='Fusion')([feat_cnn, feat_hand, feat_umap])
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    
    return Model(inputs=[inp_img, inp_feat, inp_umap], outputs=out)

# ==============================================================================
# DATA INGESTION & CACHING
# ==============================================================================
def load_and_preprocess_all(base_dir, target_size=DEFAULT_IMG_SIZE):
    """
    Loads NTUH, LIMUC, and TMC-UCM datasets, tracks cohort metadata,
    and extracts 28 quantitative biomarkers.
    """
    print(f"\n📂 Loading datasets from base directory: {base_dir}")
    ntuh_dirs = [
        f'{base_dir}/Dataset+Code/MES classification_20250313',
        f'{base_dir}/Dataset+Code/MES classification_20250724'
    ]
    limuc_dirs = [
        f'{base_dir}/Dataset/LIMUC/train_and_validation_sets',
        f'{base_dir}/Dataset/LIMUC/test_set'
    ]
    tmc_dir = f'{base_dir}/Dataset/TMC-UCM'

    print("  -> Ingesting NTUH (Hospital Cohort)...")
    ni, _, nl, _ = load_all_images(ntuh_dirs, 'NTUH')
    print("  -> Ingesting LIMUC (Public Benchmark Cohort)...")
    li, _, ll, _ = load_all_images(limuc_dirs, 'LIMUC')
    print("  -> Ingesting TMC-UCM (Multicenter Cohort)...")
    ti, _, tl, _ = load_tmc_ucm(tmc_dir, split_filter=None)

    raw_images = ni + li + ti
    raw_labels = nl + ll + tl
    cohort_tags = ['NTUH'] * len(ni) + ['LIMUC'] * len(li) + ['TMC-UCM'] * len(ti)

    print(f"\n📊 Cohort Breakdown:")
    print(f"   • NTUH    : {len(ni)} images ({len(ni)/len(raw_images)*100:.1f}%)")
    print(f"   • LIMUC   : {len(li)} images ({len(li)/len(raw_images)*100:.1f}%)")
    print(f"   • TMC-UCM : {len(ti)} images ({len(ti)/len(raw_images)*100:.1f}%)")
    print(f"   • TOTAL   : {len(raw_images)} images")

    print(f"\n⚙️ Preprocessing to {target_size[0]}x{target_size[1]} & Extracting 28 Biomarkers...")
    imgs, feats = [], []
    for img in tqdm(raw_images, desc="Extracting Features"):
        p = smart_preprocess(img, target_size=target_size)
        imgs.append(p)
        feats.append(extract_features_28(p))

    return (np.array(imgs, dtype=np.float32),
            np.array(feats, dtype=np.float32),
            np.array(raw_labels),
            np.array(cohort_tags))

# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ColonoMind v15 — Autoencoder-Pretrained, Cohort-Balanced, High-Res Hierarchical Pipeline")
    parser.add_argument("--base_dir", default="/home/D13K48009/raid/Clara/new_drive", help="Root directory of datasets")
    parser.add_argument("--save_dir", default="../Result/ColonoMind_v15_AutoBalanced", help="Output directory for all artifacts")
    parser.add_argument("--img_size", type=int, nargs=2, default=[384, 384], help="Image resolution (height width)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs_ae", type=int, default=25, help="Epochs for Autoencoder pretraining")
    parser.add_argument("--epochs_stg1", type=int, default=30, help="Epochs for Stage 1 (Detector)")
    parser.add_argument("--epochs_stg2", type=int, default=35, help="Epochs for Stage 2 (Grader)")
    parser.add_argument("--skip_ae", action="store_true", help="Skip Autoencoder training if weights already exist")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    target_shape = (args.img_size[0], args.img_size[1], 3)
    
    print("=" * 80)
    print("🚀 COLONOMIND v15 — THE ULTIMATE BALANCED HIERARCHICAL PIPELINE")
    print("=" * 80)
    print(f"📁 Output Artifacts : {os.path.abspath(args.save_dir)}")
    print(f"🖼️ Resolution       : {target_shape[0]}x{target_shape[1]} (High-Sensitivity Upsampling)")
    print(f"⚖️ Cohort Balancing : Active (NTUH : LIMUC : TMC-UCM)")
    print(f"🧠 Feature Encoding : Convolutional Autoencoder (CAE) + Mod-SE + Super Agent")
    print("=" * 80)

    # ── 1. Load / Cache Data ──────────────────────────────────────────────────
    cache_path = os.path.join(args.save_dir, f"dataset_cache_v15_{target_shape[0]}.npz")
    if os.path.exists(cache_path):
        print(f"\n⚡ Loading preprocessed cache from: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        all_imgs, all_feats = d['imgs'], d['feats']
        all_labels, all_cohorts = d['labels'], d['cohorts']
    else:
        all_imgs, all_feats, all_labels, all_cohorts = load_and_preprocess_all(args.base_dir, target_size=(target_shape[0], target_shape[1]))
        print(f"💾 Saving processed cache to: {cache_path}")
        np.savez_compressed(cache_path, imgs=all_imgs, feats=all_feats, labels=all_labels, cohorts=all_cohorts)

    label_enc = LabelEncoder()
    y_full = label_enc.fit_transform(all_labels)  # 0, 1, 2, 3

    # ── 2. Stratified Data Split (80% Train+Val, 20% Test) ───────────────────
    # We stratify on labels to guarantee clinical representation
    X_tv_i, X_te_i, X_tv_f, X_te_f, y_tv, y_te, c_tv, c_te = train_test_split(
        all_imgs, all_feats, y_full, all_cohorts,
        test_size=0.20, random_state=42, stratify=y_full
    )
    X_tr_i, X_va_i, X_tr_f, X_va_f, y_tr, y_va, c_tr, c_va = train_test_split(
        X_tv_i, X_tv_f, y_tv, c_tv,
        test_size=0.20, random_state=42, stratify=y_tv
    )

    print(f"\n📊 Universal Splits: Train={len(y_tr)}, Val={len(y_va)}, Test={len(y_te)}")

    # ── 3. UMAP & Scaling ────────────────────────────────────────────────────
    scaler_path = os.path.join(args.save_dir, "scaler_v15.pkl")
    umap_path = os.path.join(args.save_dir, "umap_v15.pkl")
    
    if os.path.exists(scaler_path) and os.path.exists(umap_path):
        scaler = joblib.load(scaler_path)
        umap_mod = joblib.load(umap_path)
    else:
        scaler = StandardScaler().fit(X_tr_f)
        umap_mod = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(scaler.transform(X_tr_f))
        joblib.dump(scaler, scaler_path)
        joblib.dump(umap_mod, umap_path)

    def process_feats(f_arr):
        sc = scaler.transform(f_arr)
        um = umap_mod.transform(sc)
        return sc, um

    Xtr_s, Utr = process_feats(X_tr_f)
    Xva_s, Uva = process_feats(X_va_f)
    Xte_s, Ute = process_feats(X_te_f)

    # ── 4. PHASE 1: Convolutional Autoencoder (CAE) Pretraining ─────────────
    print("\n" + "="*80)
    print("PHASE 1/4: Unsupervised Convolutional Autoencoder (CAE) Pretraining")
    print("="*80)
    
    ae_model_path = os.path.join(args.save_dir, "modse_autoencoder.h5")
    autoencoder, encoder_backbone = build_autoencoder(target_shape)

    if os.path.exists(ae_model_path) and args.skip_ae:
        print(f"⚡ Autoencoder checkpoint found at {ae_model_path}. Loading weights...")
        autoencoder.load_weights(ae_model_path)
    else:
        print(f"🔥 Training Autoencoder on {len(X_tv_i)} images (Reconstruction loss: MSE)...")
        ae_tr_gen = AutoencoderGenerator(X_tr_i, batch_size=args.batch_size, augment=True)
        ae_va_gen = AutoencoderGenerator(X_va_i, batch_size=args.batch_size, augment=False)
        
        autoencoder.compile(optimizer=Adam(1e-4), loss='mse')
        autoencoder.fit(
            ae_tr_gen,
            validation_data=ae_va_gen,
            epochs=args.epochs_ae,
            callbacks=[
                ModelCheckpoint(ae_model_path, save_best_only=True, monitor='val_loss', verbose=1),
                EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=1
        )
        print(f"✅ Autoencoder training completed. Best model saved to: {ae_model_path}")

    # ── 5. PHASE 2: Stage 1 Binary Detector (Normal vs Active Flare) ──────────
    print("\n" + "="*80)
    print("PHASE 2/4: Training Stage 1 Detector (MES0 Normal vs MES1/2/3 Active Disease)")
    print("="*80)
    
    # Binary targets: 0 = Normal (MES0), 1 = Active Disease (MES 1, 2, 3)
    y_tr1 = np.where(y_tr == 0, 0, 1)
    y_va1 = np.where(y_va == 0, 0, 1)
    y_te1 = np.where(y_te == 0, 0, 1)
    y_tr1_c = to_categorical(y_tr1, 2)
    y_va1_c = to_categorical(y_va1, 2)
    
    cw1 = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr1), y=y_tr1)
    
    tr_gen1 = CohortBalancedGenerator(X_tr_i, Xtr_s, Utr, y_tr1_c, c_tr, batch_size=args.batch_size, augment=True)
    va_gen1 = CohortBalancedGenerator(X_va_i, Xva_s, Uva, y_va1_c, c_va, batch_size=args.batch_size, augment=False)
    
    stage1_models = []
    for seed in DEFAULT_SEEDS:
        stg1_path = os.path.join(args.save_dir, f"stage1_cnn_seed{seed}.h5")
        if os.path.exists(stg1_path):
            print(f"⚡ Loading Stage 1 Model (Seed {seed})")
            m = load_model(stg1_path, compile=False)
        else:
            print(f"🔥 Fine-tuning Stage 1 Model with Pretrained CAE Encoder (Seed {seed})")
            m = build_se_cnn_hybrid(seed, num_classes=2, input_shape=target_shape, pretrained_encoder=encoder_backbone)
            m.compile(optimizer=Adam(5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
            m.fit(
                tr_gen1,
                validation_data=va_gen1,
                epochs=args.epochs_stg1,
                class_weight=dict(enumerate(cw1)),
                callbacks=[
                    ModelCheckpoint(stg1_path, save_best_only=True, monitor='val_accuracy', verbose=1),
                    EarlyStopping(patience=6, restore_best_weights=True)
                ],
                verbose=1
            )
            m = load_model(stg1_path, compile=False)
        stage1_models.append(m)

    # ── 6. PHASE 3: Stage 2 Ternary Grader (MES1 vs MES2 vs MES3) ────────────
    print("\n" + "="*80)
    print("PHASE 3/4: Training Stage 2 Severity Grader (MES1 vs MES2 vs MES3)")
    print("="*80)
    
    active_tr = np.where(y_tr > 0)[0]
    X_tr_i2, X_tr_f2, y_tr2, c_tr2 = X_tr_i[active_tr], X_tr_f[active_tr], y_tr[active_tr] - 1, c_tr[active_tr]
    Xtr_s2, Utr2 = process_feats(X_tr_f2)

    active_va = np.where(y_va > 0)[0]
    X_va_i2, X_va_f2, y_va2, c_va2 = X_va_i[active_va], X_va_f[active_va], y_va[active_va] - 1, c_va[active_va]
    Xva_s2, Uva2 = process_feats(X_va_f2)
    
    y_tr2_c = to_categorical(y_tr2, 3)
    y_va2_c = to_categorical(y_va2, 3)
    cw2 = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr2), y=y_tr2)
    
    tr_gen2 = CohortBalancedGenerator(X_tr_i2, Xtr_s2, Utr2, y_tr2_c, c_tr2, batch_size=args.batch_size, augment=True)
    va_gen2 = CohortBalancedGenerator(X_va_i2, Xva_s2, Uva2, y_va2_c, c_va2, batch_size=args.batch_size, augment=False)
    
    stage2_models = []
    for seed in DEFAULT_SEEDS:
        stg2_path = os.path.join(args.save_dir, f"stage2_cnn_seed{seed}.h5")
        if os.path.exists(stg2_path):
            print(f"⚡ Loading Stage 2 Model (Seed {seed})")
            m = load_model(stg2_path, compile=False)
        else:
            print(f"🔥 Fine-tuning Stage 2 Model with Pretrained CAE Encoder (Seed {seed})")
            m = build_se_cnn_hybrid(seed, num_classes=3, input_shape=target_shape, pretrained_encoder=encoder_backbone)
            m.compile(optimizer=Adam(5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
            m.fit(
                tr_gen2,
                validation_data=va_gen2,
                epochs=args.epochs_stg2,
                class_weight=dict(enumerate(cw2)),
                callbacks=[
                    ModelCheckpoint(stg2_path, save_best_only=True, monitor='val_accuracy', verbose=1),
                    EarlyStopping(patience=6, restore_best_weights=True)
                ],
                verbose=1
            )
            m = load_model(stg2_path, compile=False)
        stage2_models.append(m)

    # ── 7. PHASE 4: Hierarchical Super Agents (LightGBM) ─────────────────────
    print("\n" + "="*80)
    print("PHASE 4/4: Training Hierarchical Super Agents (Multimodal LightGBM)")
    print("="*80)
    
    def extract_deep_features(models, imgs, feats_s, umaps):
        deeps, probs = [], []
        extractors = [Model(inputs=m.input, outputs=m.get_layer('Fusion').output) for m in models]
        for i in range(len(models)):
            deeps.append(extractors[i].predict([imgs/255.0, feats_s, umaps], batch_size=args.batch_size, verbose=0))
            probs.append(models[i].predict([imgs/255.0, feats_s, umaps], batch_size=args.batch_size, verbose=0))
        mean_deep = np.mean(deeps, axis=0)
        mean_prob = np.mean(probs, axis=0)
        return np.hstack([mean_deep, feats_s, umaps, mean_prob]), mean_prob

    print("  -> Extracting Deep Fusion Features for Stage 1 Agent...")
    X_ag_tr1, _ = extract_deep_features(stage1_models, X_tr_i, Xtr_s, Utr)
    X_ag_te1, prob_stg1 = extract_deep_features(stage1_models, X_te_i, Xte_s, Ute)
    
    lgb1 = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.015, class_weight='balanced', max_depth=5, num_leaves=31, random_state=42)
    lgb1.fit(X_ag_tr1, y_tr1)
    ag_pred_1 = lgb1.predict(X_ag_te1)
    ag1_acc = accuracy_score(y_te1, ag_pred_1) * 100
    print(f"🎯 Stage 1 Super Agent Test Accuracy: {ag1_acc:.2f}%")

    print("  -> Extracting Deep Fusion Features for Stage 2 Agent...")
    X_ag_tr2, _ = extract_deep_features(stage2_models, X_tr_i2, Xtr_s2, Utr2)
    X_ag_te2_full, prob_stg2_full = extract_deep_features(stage2_models, X_te_i, Xte_s, Ute)
    
    lgb2 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.015, class_weight='balanced', max_depth=6, num_leaves=40, random_state=42)
    lgb2.fit(X_ag_tr2, y_tr2)
    ag_pred_2_full = lgb2.predict(X_ag_te2_full)
    
    lgb1.booster_.save_model(os.path.join(args.save_dir, "super_agent_stage1.txt"))
    lgb2.booster_.save_model(os.path.join(args.save_dir, "super_agent_stage2.txt"))
    print(f"💾 Super Agents saved to: {args.save_dir}")

    # ── 8. COMPREHENSIVE END-TO-END EVALUATION ────────────────────────────────
    print("\n" + "="*80)
    print("🏆 COMPREHENSIVE PIPELINE EVALUATION (TEST SET)")
    print("=" * 80)
    
    final_preds = []
    for i in range(len(y_te)):
        is_active = (ag_pred_1[i] == 1)
        if not is_active:
            final_preds.append(0)  # Output MES0
        else:
            final_preds.append(ag_pred_2_full[i] + 1)  # Output MES1, 2, or 3
            
    final_preds = np.array(final_preds)
    
    # Calculate Primary & Clinical Metrics
    exact_acc = accuracy_score(y_te, final_preds) * 100
    top2_clinical_acc = np.mean(np.abs(y_te - final_preds) <= 1) * 100
    qwk = cohen_kappa_score(y_te, final_preds, weights='quadratic')
    macro_f1 = f1_score(y_te, final_preds, average='macro') * 100
    weighted_f1 = f1_score(y_te, final_preds, average='weighted') * 100
    
    print("\n📊 1. OVERALL EVALUATION METRICS:")
    print(f"   • Exact Match Accuracy (4-Class)       : {exact_acc:.2f}%")
    print(f"   • Clinical 1-Grade Off Accuracy (Top-2): {top2_clinical_acc:.2f}% (Clinical Tolerance)")
    print(f"   • Quadratic Weighted Kappa (QWK)       : {qwk:.4f}")
    print(f"   • Macro F1-Score                       : {macro_f1:.2f}%")
    print(f"   • Weighted F1-Score                    : {weighted_f1:.2f}%")
    
    print("\n📋 2. PER-CLASS CLASSIFICATION REPORT:")
    rep = classification_report(y_te, final_preds, target_names=CLASS_NAMES, digits=4)
    print(rep)
    
    print("🧩 3. CONFUSION MATRIX (True Rows vs Predicted Cols):")
    cm = confusion_matrix(y_te, final_preds)
    cm_header = "       " + "".join([f"{c:>8}" for c in CLASS_NAMES])
    print(cm_header)
    for idx, row in enumerate(cm):
        row_str = f"{CLASS_NAMES[idx]:>6}:" + "".join([f"{val:>8}" for val in row])
        print(row_str)

    print("\n🏥 4. PER-COHORT ACCURACY BREAKDOWN (Checking for Domain Bias):")
    cohort_results = {}
    for c in np.unique(c_te):
        idx_c = np.where(c_te == c)[0]
        acc_c = accuracy_score(y_te[idx_c], final_preds[idx_c]) * 100
        top2_c = np.mean(np.abs(y_te[idx_c] - final_preds[idx_c]) <= 1) * 100
        cohort_results[c] = {'Exact_Accuracy': float(acc_c), 'Top2_Accuracy': float(top2_c), 'Samples': int(len(idx_c))}
        print(f"   • {c:<8} (N={len(idx_c):>4}): Exact Acc = {acc_c:.2f}% | 1-Grade Off = {top2_c:.2f}%")

    # Export structured metrics JSON
    summary_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'exact_accuracy': float(exact_acc),
        'clinical_1_grade_off_accuracy': float(top2_clinical_acc),
        'quadratic_weighted_kappa': float(qwk),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'confusion_matrix': cm.tolist(),
        'per_cohort_breakdown': cohort_results,
        'image_resolution': [target_shape[0], target_shape[1]],
        'seeds': DEFAULT_SEEDS
    }
    
    summary_file = os.path.join(args.save_dir, "evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n💾 Full evaluation summary exported to: {summary_file}")
    print("=" * 80)
    print("✅ COLONOMIND v15 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()

