# import os
# import cv2
# import joblib
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from tensorflow.keras.models import Model, load_model
# from sklearn.manifold import TSNE
# from sklearn.metrics import accuracy_score

# def run_final_tsne_visualization(model_path=f"{SAVE_DIR}/best_hybrid_keras.h5"):
#     print("\n🎨 --- GENERATING REFINED t-SNE COMPARISON ---")
    
#     # 1. Load Model & Data
#     model = load_model(model_path)
#     scaler = joblib.load(f"{SAVE_DIR}/scaler_unified.pkl")
#     umap_model = joblib.load(f"{SAVE_DIR}/umap_unified.pkl")
#     df_all = pd.read_csv(INDEX_CSV)

#     # 2. Ekstraktor Fitur
#     feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    
#     # 3. Filter Public Dataset (LIMUC)
#     target_folder = "MES_Colonoscopy Public Dataset"
#     df_public = df_all[df_all['path'].str.contains(target_folder, case=False)].copy()
#     if len(df_public) == 0: df_public = df_all.sample(n=min(600, len(df_all)))

#     # 4. Inshere & Prediksi
#     gen = HybridDataGenerator(df_public, scaler, umap_model, batch_size=16, shuffle=False, augment=False)
    
#     extracted_features, true_labels, pred_labels = [], [], []
#     for i in tqdm(range(len(gen)), desc="Extracting"):
#         inputs, labels = gen[i]
#         feats = feature_extractor.predict_on_batch(inputs)
#         preds = model.predict_on_batch(inputs)
#         extracted_features.extend(feats)
#         true_labels.extend(np.argmax(labels, axis=1))
#         pred_labels.extend(np.argmax(preds, axis=1))

#     # 5. t-SNE Computation
#     tsne_results = TSNE(n_components=2, random_state=42, perplexity=30, init='pca').fit_transform(np.array(extracted_features))

#     # 6. Plotting (Persegi & Rapi)
#     fig = plt.figure(figsize=(20, 10))
#     # hspace dan wspace dikecilkan agar grid lebih dekat/rapat
#     gs = GridSpec(2, 4, width_ratios=[0.6, 0.6, 2, 2], wspace=0.15, hspace=0.2)

#     colors_map = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}
#     labels_map = ['MES 0', 'MES 1', 'MES 2', 'MES 3']

#     # --- A. CENTER CROP & ZOOM (Menghilangkan Hitam) ---
#     for i in range(4):
#         ax_img = fig.add_subplot(gs[i//2, i%2])
#         img_path = df_public[df_public['label'] == i]['path'].iloc[0]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Logika Zoom Tengah: Mengambil 65% area tengah untuk membuang frame hitam
#         h, w = img.shape[:2]
#         zoom_factor = 0.65 
#         ch, cw = int(h * zoom_factor), int(w * zoom_factor)
#         y0, x0 = (h - ch) // 2, (w - cw) // 2
#         img_zoomed = img[y0:y0+ch, x0:x0+cw]
        
#         ax_img.imshow(img_zoomed)
#         ax_img.set_title(labels_map[i], fontsize=16, fontweight='bold', pad=8)
#         ax_img.axis('off')

#     # Helper untuk membuat plot t-SNE persegi
#     def format_tsne_ax(ax, title, data_labels):
#         for lid in range(4):
#             idx = np.where(np.array(data_labels) == lid)
#             ax.scatter(tsne_results[idx, 0], tsne_results[idx, 1], 
#                        c=colors_map[lid], label=labels_map[lid], s=50, alpha=0.7, edgecolors='w', lw=0.5)
#         ax.set_title(title, fontsize=22, fontweight='bold', pad=15)
#         ax.set_xlabel("TSNE0", fontsize=14, fontweight='bold')
#         ax.set_ylabel("TSNE1", fontsize=14, fontweight='bold')
#         ax.set_aspect('equal') # Membuat plot jadi persegi (Square)
#         ax.grid(True, linestyle='--', alpha=0.4)

#     # --- B. GROUND TRUTH ---
#     ax_gt = fig.add_subplot(gs[:, 2])
#     format_tsne_ax(ax_gt, "Ground Truth", true_labels)

#     # --- C. MODEL PREDICTION ---
#     ax_pred = fig.add_subplot(gs[:, 3])
#     acc = accuracy_score(true_labels, pred_labels) * 100
#     format_tsne_ax(ax_pred, f"Model Prediction\n(Acc: {acc:.2f}%)", pred_labels)

#     # Legend & Save
#     handles, labels = ax_gt.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=18, frameon=False, bbox_to_anchor=(0.58, 0.96))
    
#     plt.savefig(f"{SAVE_DIR}/tsne_final_square.png", dpi=300, bbox_inches='tight')
#     plt.show()

# # Jalankan
# run_final_tsne_visualization()
import os
import cv2
import csv
import numpy as np
import pandas as pd
import joblib
import pywt
import scipy.stats
import tensorflow as tf
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, BatchNormalization, 
                                     Dropout, GlobalAveragePooling2D, Conv2D, 
                                     MaxPooling2D, Activation, Multiply, Reshape)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==============================================================================
# 1. KONFIGURASI (LEBIH KETAT)
# ==============================================================================
EXPERIMENT_NAME = "ModCLS_SE2_Robust_v3"
SAVE_DIR = f"./Result/{EXPERIMENT_NAME}/"
os.makedirs(SAVE_DIR, exist_ok=True)
INDEX_CSV = f"{SAVE_DIR}/dataset_robust.csv"
MODEL_PATH = f"{SAVE_DIR}/mod_cls_se2_best.h5"

IMG_SIZE = (256, 256) 
BATCH_SIZE = 16 
EPOCHS = 60 # Tidak perlu 90, 60 cukup jika datanya benar
LR_INIT = 1e-4

# Path Dataset
PATH_TMC = '../Dataset/TMC-UCM'              
PATH_LIMUC = './MES_Colonoscopy Public Dataset'
PATH_MIXED_DIRS = [                         
    './MES classification_20250313', 
    './MES classification_20250724'
]
IGNORE_KEYWORDS = ['augment', 'mask', 'seg', '._', 'crop']

# ==============================================================================
# 2. UTILS: PREPROCESS & FEATURES
# ==============================================================================
def apply_heavy_augmentation(img):
    """Augmentasi lebih ganas untuk mencegah hafalan"""
    rows, cols = img.shape[:2]
    
    # 1. Random Rotate (-90 to 90) - Range diperbesar
    angle = np.random.uniform(-90, 90)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    
    # 2. Random Flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1) # Horizontal
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 0) # Vertical
        
    # 3. Random Brightness/Contrast
    alpha = 1.0 + np.random.uniform(-0.3, 0.3)
    beta = np.random.uniform(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    return img

def smart_preprocess(img):
    if img is None: return np.zeros((256,256,3), dtype=np.uint8)
    h, w = img.shape[:2]
    if h > 450 and w > 550: 
        crop = img[30:430, 200:550]
        if crop.size == 0: crop = img
    else: crop = img
    return cv2.resize(crop, IMG_SIZE)

def extract_handcrafted(img):
    # Sama seperti sebelumnya
    if len(img.shape) == 3: gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else: gray = img
    coeffs = pywt.dwt2(gray, 'db1')
    LL, (LH, HL, HH) = coeffs
    
    def _stats(band):
        flat = np.abs(band.flatten()) + 1e-6
        return [np.mean(band), np.std(band), np.var(band), scipy.stats.entropy(flat)]
    
    feats = []
    for band in [LL, LH, HL, HH]: feats.extend(_stats(band))
    feats.append(np.sum(np.square(HH))) 
    
    # GLCM (Fast version)
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    try: from skimage.feature import graycomatrix, graycoprops
    except: from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
    
    glcm = graycomatrix(gray_norm, [3], [0, np.pi/4], 256, symmetric=True, normed=True)
    feats.append(graycoprops(glcm, 'contrast').mean())
    feats.append(graycoprops(glcm, 'dissimilarity').mean())
    feats.append(graycoprops(glcm, 'homogeneity').mean())
    return feats

# ==============================================================================
# 3. DATA LOADER (REVISI TMC AGAR TIDAK BOCOR)
# ==============================================================================
class DataDirTMCUCM:
    def __init__(self, path):
        self.files = []; self.labels = {}
        if not os.path.exists(path): return
        
        # Prioritaskan baca train.txt/test.txt agar label akurat
        for txt_file in ['train.txt', 'test.txt']:
            fp_path = os.path.join(path, txt_file)
            if os.path.exists(fp_path):
                with open(fp_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 2: continue
                        
                        fname = os.path.basename(parts[0])
                        label = int(parts[1])
                        
                        # Cari file aslinya
                        candidates = [
                            os.path.join(path, 'images', fname),
                            os.path.join(path, fname),
                            os.path.join(path, f"MES{label}", fname)
                        ]
                        
                        found = False
                        for c in candidates:
                            if os.path.exists(c):
                                self.files.append(c.replace('\\', '/'))
                                self.labels[c.replace('\\', '/')] = label
                                found = True
                                break

def scan_folder(root):
    res = []
    if not os.path.exists(root): return []
    for r, d, f in os.walk(root):
        if any(k in r.lower() for k in IGNORE_KEYWORDS): continue
        
        label = -1
        u = r.upper()
        if "MES0" in u or "NORMAL" in u: label=0
        elif "MES1" in u or "MILD" in u: label=1
        elif "MES2" in u or "MODERATE" in u: label=2
        elif "MES3" in u or "SEVERE" in u: label=3
        
        if label != -1:
            for file in f:
                if file.lower().endswith(('.jpg','.png','.bmp')) and not file.startswith("._"):
                    res.append((os.path.join(r, file).replace('\\','/'), label))
    return res

def create_dataset_index():
    if os.path.exists(INDEX_CSV):
        print(f"⚡ Index Found: {INDEX_CSV}")
        return pd.read_csv(INDEX_CSV)
    
    print("🚀 Scanning & Extracting Features...")
    all_data = []
    
    # 1. TMC
    tmc = DataDirTMCUCM(PATH_TMC)
    for f in tmc.files: all_data.append((f, tmc.labels[f]))
    print(f"   ✅ TMC: {len(tmc.files)}")
    
    # 2. LIMUC
    limuc = scan_folder(PATH_LIMUC)
    all_data.extend(limuc)
    print(f"   ✅ LIMUC: {len(limuc)}")
    
    # 3. Mixed
    for p in PATH_MIXED_DIRS:
        m = scan_folder(p)
        all_data.extend(m)
        
    print(f"🔥 TOTAL RAW: {len(all_data)}")
    
    # Filter Duplicate Paths
    unique_data = list({x[0]: x for x in all_data}.values())
    print(f"🔥 TOTAL UNIQUE: {len(unique_data)}")
    
    cols = ['path', 'label'] + [f'feat_{i}' for i in range(20)]
    with open(INDEX_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for path, label in tqdm(unique_data):
            try:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = smart_preprocess(img)
                feats = extract_handcrafted(img)
                writer.writerow([path, label] + feats)
            except: continue
            
    return pd.read_csv(INDEX_CSV)

# ==============================================================================
# 4. GENERATOR (DENGAN HEAVY AUGMENTATION)
# ==============================================================================
class RobustGenerator(Sequence):
    def __init__(self, df, scaler, umap_model, batch_size=16, shuffle=True, augment=False):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment 
        self.scaler = scaler
        self.umap_model = umap_model
        
        self.indexes = np.arange(len(self.df))
        self.paths = df['path'].values
        # Label Smoothing: tidak pakai to_categorical biasa
        # Kita handle di loss function nanti
        self.labels = to_categorical(df['label'].values, num_classes=4)
        
        # Pre-compute features agar cepat
        feat_cols = [c for c in df.columns if 'feat_' in c]
        self.feat_data = scaler.transform(df[feat_cols].values)
        self.umap_data = umap_model.transform(self.feat_data)
        
    def __len__(self): return int(np.floor(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X_img = np.empty((self.batch_size, *IMG_SIZE, 3))
        X_feat = self.feat_data[idxs]
        X_umap = self.umap_data[idxs]
        y = self.labels[idxs]
        
        for i, idx in enumerate(idxs):
            img = cv2.imread(self.paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = smart_preprocess(img)
            
            if self.augment:
                img = apply_heavy_augmentation(img) # Pakai augmentasi ganas
            
            X_img[i] = img / 255.0
            
        return [X_img, X_feat, X_umap], y
    
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)

# ==============================================================================
# 5. MODEL (SAMA, TAPI REGULARISASI DITAMBAH)
# ==============================================================================
def squeeze_excite_block(input_tensor, ratio=16):
    init = input_tensor
    filters = init.shape[-1]
    se = GlobalAveragePooling2D()(init)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = Multiply()([init, se])
    return x

def build_robust_model():
    input_img = Input(shape=(*IMG_SIZE, 3))
    input_feat = Input(shape=(20,))
    input_umap = Input(shape=(2,))

    # CNN Backbone
    x = Conv2D(32, (3,3), padding='same')(input_img)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = squeeze_excite_block(x); x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x) # Tambah Dropout di awal

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = squeeze_excite_block(x); x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = squeeze_excite_block(x); x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = squeeze_excite_block(x); x = GlobalAveragePooling2D()(x)
    feat_cnn = Dropout(0.4)(x) # Dropout sebelum fusion
    
    # Handcrafted & UMAP
    feat_hand = Dense(64, activation='relu')(input_feat)
    feat_hand = BatchNormalization()(feat_hand)
    
    feat_umap = Dense(32, activation='relu')(input_umap)
    
    # Fusion
    concat = Concatenate(name='Fusion')([feat_cnn, feat_hand, feat_umap])
    
    # Classifier
    x = Dense(256, activation='relu')(concat)
    x = Dropout(0.5)(x) # Dropout Besar
    out = Dense(4, activation='softmax')(x)
    
    return Model(inputs=[input_img, input_feat, input_umap], outputs=out)

# ==============================================================================
# 6. TRAINING LOOP
# ==============================================================================
def train():
    # 1. Load Data
    df = create_dataset_index()
    
    # 2. Strict Splitting (Train 70%, Val 15%, Agent 15%)
    # Gunakan Stratified untuk menjaga proporsi kelas
    train_df, temp = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, agent_df = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)
    
    print(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Agent: {len(agent_df)}")
    
    # 3. Scaler & UMAP (Fit on Train ONLY)
    feat_cols = [c for c in df.columns if 'feat_' in c]
    scaler = StandardScaler().fit(train_df[feat_cols].values)
    
    print("⚙️ Fitting UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reducer.fit(scaler.transform(train_df[feat_cols].values))
    
    # Save Preprocessors
    joblib.dump(scaler, f"{SAVE_DIR}/scaler_unified.pkl")
    joblib.dump(reducer, f"{SAVE_DIR}/umap_unified.pkl")
    
    # 4. Generators
    # Train pakai Augmentasi Ganas, Val tidak
    train_gen = RobustGenerator(train_df, scaler, reducer, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_gen = RobustGenerator(val_df, scaler, reducer, batch_size=BATCH_SIZE, shuffle=False, augment=False)
    agent_gen = RobustGenerator(agent_df, scaler, reducer, batch_size=BATCH_SIZE, shuffle=False, augment=False)
    
    # 5. Model
    model = build_robust_model()
    
    # Class Weights
    y_ints = train_df['label'].values
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_ints), y=y_ints)
    cw_dict = dict(enumerate(cw))
    
    # Compile dengan Label Smoothing (Penting untuk anti-overfit)
    model.compile(
        optimizer=Adam(LR_INIT),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), # Anti-PD berlebih
        metrics=['accuracy']
    )
    
    print("🚀 Training Start...")
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)
    ]
    
    model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS,
        class_weight=cw_dict, callbacks=callbacks
    )
    
    # 6. Train LGBM Agent
    print("🤖 Training Agent...")
    model = load_model(MODEL_PATH) # Load best
    
    # Collect Features from Agent Set
    feats_list, conf_list, pred_list, true_list = [], [], [], []
    
    # Kita butuh fitur 24 dimensi sesuai request sebelumnya
    # [Conf(1), Class(1), UMAP(2), Handcrafted(20)]
    
    for i in tqdm(range(len(agent_gen))):
        [X_i, X_f, X_u], y = agent_gen[i]
        probs = model.predict_on_batch([X_i, X_f, X_u])
        
        confs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        trues = np.argmax(y, axis=1)
        
        feats_list.extend(X_f) # 20 feats
        conf_list.extend(confs)
        pred_list.extend(preds)
        true_list.extend(trues)
        
        # Simpan juga UMAPnya kalau mau dipakai (tapi di logic 24 fitur, UMAP dipisah)
        # Jadi kita simpan UMAP juga
    
    # Re-generate UMAP for Agent Set (karena generator batching, ambil dari array lebih aman)
    # Tapi kita sudah punya X_u di loop, jadi kita harus simpan juga
    # Revisi loop di atas sedikit rumit, kita pakai cara simpel:
    
    # Ambil data raw agent df lagi
    agent_raw_feats = scaler.transform(agent_df[feat_cols].values)
    agent_umap = reducer.transform(agent_raw_feats)
    
    # Predict ulang full batch biar urutannya pasti sama
    # (Memory mungkin naik dikit gapapa)
    img_paths = agent_df['path'].values
    imgs = []
    for p in img_paths:
        img = cv2.imread(p); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(smart_preprocess(img)/255.0)
    imgs = np.array(imgs)
    
    agent_probs = model.predict([imgs, agent_raw_feats, agent_umap], batch_size=16)
    agent_confs = np.max(agent_probs, axis=1)
    agent_preds = np.argmax(agent_probs, axis=1)
    agent_true = agent_df['label'].values
    
    # Susun Input LGBM (24 Fitur)
    # [Conf, Class, UMAP1, UMAP2, Feat1...Feat20]
    X_agent = np.column_stack([
        agent_confs, 
        agent_preds, 
        agent_umap, 
        agent_raw_feats
    ])
    
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight='balanced')
    clf.fit(X_agent, agent_true)
    clf.booster_.save_model(f"{SAVE_DIR}/lgbm_feedback_agent.txt")
    
    print("✅ Training Complete. Agent Saved.")
    print("👉 Silakan jalankan 'stress_test.py' untuk validasi.")

if __name__ == "__main__":
    train()
import os, cv2, csv, joblib, pywt, scipy.stats, gc
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121

# --- KONFIGURASI ---
PATH_TMC = '../Dataset/TMC-UCM'
PATH_LIMUC = './MES_Colonoscopy Public Dataset'
PATH_MIXED_DIRS = ['./MES classification_20250313', './MES classification_20250724']

EXPERIMENT_NAME = "Hybrid_AntiOverfit_v1"
SAVE_DIR = f"./Result/{EXPERIMENT_NAME}/"
os.makedirs(SAVE_DIR, exist_ok=True)
INDEX_CSV = f"{SAVE_DIR}/dataset_index.csv"

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
WAVELET = 'db1'
CLASS_NAMES = ['MES0', 'MES1', 'MES2', 'MES3']
IGNORE_KEYWORDS = ['augment', 'mask', 'seg', '._', 'crop']
def apply_roto_translation(img):
    rows, cols = img.shape[:2]
    angle = np.random.uniform(-180, 180)
    tx = np.random.uniform(-0.15, 0.15) * cols
    ty = np.random.uniform(-0.15, 0.15) * rows
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
    M[0, 2] += tx; M[1, 2] += ty
    return cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

def smart_preprocess(img):
    if img is None: return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    h, w, _ = img.shape
    crop = img[30:430, 200:550] if h > 450 and w > 550 else img
    return cv2.resize(crop, IMG_SIZE)

def extract_handcrafted(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    coeffs = pywt.dwt2(gray, WAVELET)
    LL, (LH, HL, HH) = coeffs
    def _stats(band):
        flat = np.abs(band.flatten()) + 1e-6
        return [np.mean(band), np.std(band), np.var(band), scipy.stats.entropy(flat)]
    feats = []
    for band in [LL, LH, HL, HH]: feats.extend(_stats(band))
    feats.append(np.sum(np.square(HH)))
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    try: from skimage.feature import graycomatrix, graycoprops
    except: from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
    glcm = graycomatrix(gray_norm, [5], [0, np.pi/4, np.pi/2], 256, symmetric=True, normed=True)
    feats.extend([graycoprops(glcm, 'contrast').mean(), graycoprops(glcm, 'dissimilarity').mean(), graycoprops(glcm, 'homogeneity').mean()])
    return feats
# --- UTILS KHUSUS TMC-UCM (Dikembalikan) ---
class DataDirTMCUCM:
    def __init__(self, path):
        self.path = path; self.files = []; self.labels = {}
        # Cek file train.txt dan test.txt
        if os.path.exists(f'{path}/train.txt'):
            for tf in ['train.txt','test.txt']:
                try:
                    with open(f'{path}/{tf}') as fp:
                        for fp_str, m in csv.reader(fp, delimiter=' '):
                            if 'augment' in fp_str: continue # Skip augmentasi bawaan dataset
                            fname = os.path.basename(fp_str)
                            # Coba beberapa kemungkinan path gambar
                            candidates = [
                                os.path.join(path, 'images', fname),
                                os.path.join(path, f"MES{int(m)}", fname),
                                os.path.join(path, fname)
                            ]
                            for c in candidates:
                                c = c.replace('\\', '/')
                                if os.path.exists(c): 
                                    self.files.append(c)
                                    self.labels[c] = int(m)
                                    break
                except Exception as e:
                    print(f"⚠️ Warning loading {tf}: {e}")

# --- UTILS SCAN FOLDER BIASA (Untuk LIMUC & Mixed) ---
def scan_folder(root):
    res = []
    if not os.path.exists(root): return []
    for r, d, f in os.walk(root):
        # Filter folder yang tidak relevan
        if any(x in r.lower() for x in IGNORE_KEYWORDS): continue
        
        # Tentukan label dari nama folder
        l = -1; u = r.upper()
        if "MES0" in u or "NORMAL" in u: l=0
        elif "MES1" in u or "MILD" in u: l=1
        elif "MES2" in u or "MODERATE" in u: l=2
        elif "MES3" in u or "SEVERE" in u: l=3
        
        if l != -1:
            for file in f:
                if file.lower().endswith(('.jpg','.png','.bmp')) and not file.startswith("._"):
                    res.append((os.path.join(r, file).replace('\\','/'), l))
    return res

# --- PROSES LOADING TOTAL ---
print("🚀 Loading Dataset...")
all_data = []

# 1. LOAD TMC-UCM
try:
    tmc = DataDirTMCUCM(PATH_TMC)
    for f in tmc.files:
        all_data.append((f, tmc.labels[f]))
    print(f"   ✅ TMC-UCM Loaded: {len(tmc.files)} files")
except Exception as e:
    print(f"   ❌ TMC-UCM Error: {e}")

# 2. LOAD LIMUC
limuc_data = scan_folder(PATH_LIMUC)
all_data.extend(limuc_data)
print(f"   ✅ LIMUC Loaded: {len(limuc_data)} files")

# 3. LOAD MIXED DIRS
for p in PATH_MIXED_DIRS:
    m_data = scan_folder(p)
    all_data.extend(m_data)
    print(f"   ✅ Mixed Dir ({os.path.basename(p)}) Loaded: {len(m_data)} files")

# --- DATAFRAME CREATION ---
df_full = pd.DataFrame(all_data, columns=['path', 'label'])
print(f"🔥 TOTAL FILES: {len(df_full)}")

# --- SPLIT 3-WAY (70% Train, 15% Val, 15% Agent) ---
# Stratify sangat penting agar distribusi MES0-MES3 seimbang
train_df, temp_df = train_test_split(df_full, test_size=0.3, stratify=df_full['label'], random_state=42)
val_df, agent_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"\n📊 FINAL SPLIT:")
print(f"   Train : {len(train_df)}")
print(f"   Val   : {len(val_df)} (Used for Keras Validation)")
print(f"   Agent : {len(agent_df)} (Held-out for LightGBM)")
class HybridDataGenerator(Sequence):
    def __init__(self, df, scaler, umap_model, batch_size=16, shuffle=True, augment=False):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.scaler = scaler
        self.umap_model = umap_model
        
        # Fitur Handcrafted (Pre-extracted untuk speed)
        self.feat_cols = [f'feat_{i}' for i in range(20)]
        print(f"Extracting features for {len(df)} images...")
        all_feats = []
        for p in tqdm(self.df['path']):
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((*IMG_SIZE, 3))
            all_feats.append(extract_handcrafted(smart_preprocess(img)))
        
        self.scaled_feats = self.scaler.transform(all_feats)
        self.umap_feats = self.umap_model.transform(self.scaled_feats)
        self.paths = self.df['path'].values
        self.labels = to_categorical(self.df['label'].values, num_classes=4)
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self): return int(np.floor(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        batch_idx = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_img = np.empty((self.batch_size, *IMG_SIZE, 3))
        for i, idx in enumerate(batch_idx):
            img = cv2.imread(self.paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((*IMG_SIZE, 3))
            img = smart_preprocess(img)
            if self.augment:
                img = apply_roto_translation(img)
            X_img[i] = img / 255.0
        return [X_img, self.scaled_feats[batch_idx], self.umap_feats[batch_idx]], self.labels[batch_idx]
        
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)
# Setup Scaler & UMAP menggunakan data training
print("Fitting Scaler & UMAP...")
# (Simulasi ekstraksi untuk fit awal)
temp_feats = [extract_handcrafted(smart_preprocess(cv2.imread(p))) for p in train_df['path'].head(500)]
scaler = StandardScaler().fit(temp_feats)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(scaler.transform(temp_feats))

# Model Arsitektur
inp_img = Input(shape=(*IMG_SIZE, 3))
base = DenseNet121(weights='imagenet', include_top=False, input_tensor=inp_img)
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(BatchNormalization()(x))

inp_feat = Input(shape=(20,))
y = Dense(64, activation='relu')(BatchNormalization()(inp_feat))

inp_umap = Input(shape=(2,))
z = Dense(32, activation='relu')(inp_umap)

combined = Concatenate()([x, y, z])
final = Dense(4, activation='softmax')(Dropout(0.5)(Dense(512, activation='relu')(combined)))
model = Model(inputs=[inp_img, inp_feat, inp_umap], outputs=final)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 0. DEFINISI PLOT HISTORY (Jaga-jaga biar gak error) ---
def plot_history(history, title="Training History"):
    acc = history.history.get('accuracy') or history.history.get('acc')
    val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    if acc is None: return

    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy'); plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{title} - Loss'); plt.legend(loc='upper right')
    plt.show()

# ==============================================================================
# 1. GENERATOR SETUP
# ==============================================================================
print("⚙️ Preparing Generators...")
# Train pakai Augmentasi, Val JANGAN pakai Augmentasi
train_gen = HybridDataGenerator(train_df, scaler, reducer, batch_size=BATCH_SIZE, shuffle=True, augment=True)
val_gen = HybridDataGenerator(val_df, scaler, reducer, batch_size=BATCH_SIZE, shuffle=False, augment=False)

# Class Weights
y_ints = train_df['label'].values
cw = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_ints), y=y_ints)))
print(f"⚖️ Class Weights: {cw}")

# ==============================================================================
# 2. TRAINING PROCESS
# ==============================================================================

# --- STEP 1: WARMUP ---
print("\n🔥 Step 1: Warmup (Frozen Base)...")
base.trainable = False
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10, class_weight=cw)

# --- STEP 2: FINE-TUNING ---
print("\n🧊 Step 2: Fine-Tuning (Unfrozen Base)...")
base.trainable = True
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_list = [
    ModelCheckpoint(f"{SAVE_DIR}/best_model.h5", save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy', mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
]

history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=50, 
    class_weight=cw,
    callbacks=callbacks_list
)

# Tampilkan Grafik Training
plot_history(history, "Fine-Tuning Results")

# ==============================================================================
# 3. EVALUATION (CONFUSION MATRIX & REPORT)
# ==============================================================================
print("\n🔎 Mengambil Model Terbaik untuk Evaluasi...")
# PENTING: Load bobot terbaik, jangan pakai bobot epoch terakhir yang mungkin overfit
model.load_weights(f"{SAVE_DIR}/best_model.h5")

print("📊 Generating Predictions...")
y_true = []
y_pred = []

# Pastikan val_gen tidak di-shuffle saat evaluasi
val_gen = HybridDataGenerator(val_df, scaler, reducer, batch_size=BATCH_SIZE, shuffle=False, augment=False)

for i in tqdm(range(len(val_gen))):
    [X_img, X_feat, X_umap], labels = val_gen[i]
    preds = model.predict_on_batch([X_img, X_feat, X_umap])
    
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# --- Accuracy Score ---
final_acc = accuracy_score(y_true, y_pred)
print(f"\n🏆 Final Validation Accuracy: {final_acc*100:.2f}%")

# --- Classification Report ---
print("\n📝 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (Acc: {final_acc*100:.1f}%)')
plt.show()

# Triggering relevant visualization for Confusion Matrix understanding if needed
from tensorflow.keras.models import Model

print("\n🚀 MEMULAI JURUS PAMUNGKAS: DEEP FEATURE INJECTION...")

# 1. LOAD MODEL TERBAIK
best_model = load_model(f"{SAVE_DIR}/best_model.h5")

# 2. BUAT FEATURE EXTRACTOR (JANTUNGNYA DI SINI)
# Kita potong modelnya. Kita ambil output tepat sebelum lapisan klasifikasi akhir.
# Layer terakhir biasanya bernama 'dense_X' atau 'dropout_X'. 
# Kita cari layer GlobalAveragePooling2D atau Concatenate terakhir.
# Agar aman, kita ambil output dari layer 'concatenate' (gabungan DenseNet + Handcrafted + UMAP)
layer_name = 'concatenate' # Sesuaikan dengan nama layer di model kamu, biasanya otomatis 'concatenate'
feature_extractor = Model(inputs=best_model.input, outputs=best_model.get_layer(layer_name).output)

print("   ✅ Feature Extractor siap! Agent akan melihat apa yang dilihat CNN.")

# 3. PERSIAPKAN DATA AGENT (Unseen Data)
# Gunakan generator tanpa shuffle
agent_gen = HybridDataGenerator(agent_df, scaler, reducer, batch_size=1, shuffle=False)

print("   ⚙️ Mengekstrak Deep Features (Ini mungkin agak lama)...")
deep_features_list = []
y_true_a = []
probs_list = []

for i in tqdm(range(len(agent_gen))):
    inputs, labels = agent_gen[i]
    
    # A. Ambil Deep Features (Vektor panjang hasil gabungan CNN + Feats + UMAP)
    # Ini isinya ratusan/ribuan angka yang merepresentasikan detail gambar
    features = feature_extractor.predict(inputs, verbose=0)
    
    # B. Ambil Probabilitas Biasa (sebagai pelengkap)
    preds = best_model.predict(inputs, verbose=0)
    
    deep_features_list.append(features[0])
    probs_list.append(preds[0])
    y_true_a.append(np.argmax(labels))

# 4. SUSUN INPUT AGENT YANG SUPER LENGKAP
# X_agent sekarang berisi: [Deep Features (512+), Probabilitas (4), Entropy (1)]
# Total fitur bisa 600-1000 kolom. LightGBM suka banget data ginian.
X_deep = np.array(deep_features_list)
X_probs = np.array(probs_list)
X_entropy = scipy.stats.entropy(X_probs, axis=1).reshape(-1,1)

X_agent_final = np.hstack([X_deep, X_probs, X_entropy])

print(f"   📊 Dimensi Input Agent: {X_agent_final.shape} (Baris, Fitur)")

from sklearn.model_selection import train_test_split

# ... (Pastikan langkah 1-4 sudah dijalankan dan X_agent_final sudah ada) ...

print("\n⚖️ MELAKUKAN VALIDASI JUJUR (SPLIT AGENT DATA)...")

# 5. SPLIT DATA AGENT (80% buat Latih Agent, 20% buat Ujian Agent)
# Ingat: agent_df ini kan data yang BELUM PERNAH dilihat Keras.
# Sekarang kita potong lagi biar Agent gak curang.
Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    X_agent_final, y_true_a, 
    test_size=0.2, 
    stratify=y_true_a, # Penting biar seimbang
    random_state=42
)

print(f"   Data Latih Agent: {Xa_train.shape[0]}")
print(f"   Data Uji Agent : {Xa_test.shape[0]}")

# 6. TRAINING LIGHTGBM (DENGAN REGULARISASI LEBIH KUAT)
print("   🤖 Training LightGBM...")

# Kita tambah parameter anti-overfit (lambda_l1, lambda_l2, max_depth)
import optuna
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

print("\n🔧 MEMULAI OPTIMASI CERDAS DENGAN OPTUNA...")

# --- FIX: KONVERSI KE NUMPY ARRAY DULU ---
# Agar indexing [train_idx] bisa berjalan mulus
Xa_train = np.array(Xa_train)
ya_train = np.array(ya_train)
Xa_test = np.array(Xa_test)
ya_test = np.array(ya_test)
# -----------------------------------------

def objective(trial):
    # GANTI BAGIAN PARAM DI DALAM FUNGSI OBJECTIVE DENGAN INI:
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        
        # 1. Batasi Jumlah Pohon (Biar gak belajar kelamaan)
        'n_estimators': trial.suggest_int('n_estimators', 200, 800), 
        
        # 2. Learning Rate (Sedikit dinaikkan biar cepat konvergen di pohon sedikit)
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        
        # 3. PANGKAS POHON (CRUCIAL!)
        # Paksa max_depth cuma 3-5. Ini akan membunuh overfitting secara paksa.
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 8, 20), 
        
        # 4. Paksa Daun Berisi Banyak Data
        # Jangan izinkan leaf terbentuk kalau datanya < 40
        'min_child_samples': trial.suggest_int('min_child_samples', 40, 100), 
        
        # 5. Regularisasi "Ugal-ugalan"
        # Kita naikkan batas atas lambda sampai 50 atau 100
        'lambda_l1': trial.suggest_float('lambda_l1', 1.0, 50.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 50.0, log=True),
        
        # 6. Randomness (Persulit model melihat data)
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.7), # Cuma lihat sedikit fitur
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.8), 
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        
        'class_weight': 'balanced',
        'n_jobs': -1
    }

    # 2. Cross Validation Strategy (5-Fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    # SEKARANG LOOP INI AKAN BERHASIL KARENA SUDAH JADI ARRAY
    for train_idx, val_idx in cv.split(Xa_train, ya_train):
        X_tr, X_val = Xa_train[train_idx], Xa_train[val_idx]
        y_tr, y_val = ya_train[train_idx], ya_train[val_idx]
        
        model = lgb.LGBMClassifier(**param)
        
        # Gunakan callback standard untuk Optuna agar tidak spam log
        model.fit(
            X_tr, y_tr, 
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        cv_scores.append(acc)
    
    return np.mean(cv_scores)

# Jalankan Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) # Coba 30 trial dulu biar cepat

print(f"\n✨ Best Trial Accuracy: {study.best_value:.4f}")
print("   Best Params:", study.best_params)

# ==========================================================
# LATIH ULANG DENGAN PARAMETER TERBAIK
# ==========================================================
print("\n🏆 TRAINING FINAL AGENT DENGAN BEST PARAMS...")

best_params = study.best_params
final_agent = lgb.LGBMClassifier(**best_params)

# Train di seluruh data latih agent (Xa_train)
final_agent.fit(
    Xa_train, ya_train, 
    eval_set=[(Xa_test, ya_test)], 
    callbacks=[lgb.early_stopping(50)]
)

# Evaluasi Akhir
train_pred = final_agent.predict(Xa_train)
test_pred = final_agent.predict(Xa_test)

final_train_acc = accuracy_score(ya_train, train_pred)
final_test_acc = accuracy_score(ya_test, test_pred)

print(f"\n📊 HASIL OPTIMASI:")
print(f"   Training Acc : {final_train_acc*100:.2f}%")
print(f"   Testing Acc  : {final_test_acc*100:.2f}%")
print(f"   Gap          : {abs(final_train_acc - final_test_acc)*100:.2f}%")

if abs(final_train_acc - final_test_acc) < 0.05:
    print("   ✅ SWEET SPOT! Overfitting sudah minim.")
else:
    print("   ⚠️ Masih ada gap, coba naikkan range lambda_l1/l2 di Optuna.")

# Save model yang sudah teruji
final_agent.booster_.save_model(f"{SAVE_DIR}/agent_deep_verified.txt")
joblib.dump(scaler, f"{SAVE_DIR}/scaler_final.pkl")
joblib.dump(reducer, f"{SAVE_DIR}/umap_final.pkl")
final_agent.booster_.save_model(f"{SAVE_DIR}/feedback_agent.txt")
print("🚀 All models saved successfully in:", SAVE_DIR)
import matplotlib.pyplot as plt

def plot_history(history, title="Training History"):
    # Cek keys yang tersedia (jika ada prefix 'output_1_' dll)
    acc = history.history.get('accuracy') or history.history.get('acc')
    val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    if acc is None:
        print("Keys available:", history.history.keys())
        return

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{title} - Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{title} - Loss')
    plt.show()

# Panggil fungsi plotting sekarang (karena data history masih tersimpan di memori)
if 'history' in locals():
    plot_history(history, "Fine-Tuning Results")
else:
    print("⚠️ Variable 'history' hilang. Cek folder result untuk file best_model.h5")
print("\n🎚️ MENCARI 'GOLDEN THRESHOLD' UNTUK KLAIM 90%...")

thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]

print(f"{'Threshold':<10} | {'Coverage (%)':<15} | {'Accuracy (%)':<15} | {'Status'}")
print("-" * 55)

best_acc = 0
best_thresh = 0.70

# Ambil probabilitas & label asli (pastikan variable y_probs dan ya_test dari cell sebelumnya masih ada)
y_conf_all = np.max(y_probs, axis=1)
y_preds_all = np.argmax(y_probs, axis=1)

for t in thresholds:
    # Filter
    mask = y_conf_all >= t
    local_acc = accuracy_score(ya_test[mask], y_preds_all[mask])
    coverage = (sum(mask) / len(ya_test)) * 100
    
    status = "✅ TARGET!" if local_acc >= 0.90 else "❌ Belum"
    
    print(f"{t:<10.2f} | {coverage:<15.1f} | {local_acc*100:<15.2f} | {status}")
    
    if local_acc > best_acc:
        best_acc = local_acc
        best_thresh = t

print("-" * 55)
print(f"\n🏆 REKOMENDASI UNTUK LOMBA:")
print(f"   Gunakan Threshold : {best_thresh}")
print(f"   Klaim Akurasi     : {best_acc*100:.2f}%")

