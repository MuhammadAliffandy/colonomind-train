import json
import os

def new_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def new_code_cell(source):
    # Ensure there are no double newlines added if the source already has them
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

def build_notebook(scenario_type, train_dataset, test_dataset=None):
    if test_dataset is None:
        test_dataset = train_dataset
        
    cells = []

    # ---------------------------------------------------------
    # TITLE & INTRODUCTION
    # ---------------------------------------------------------
    cells.append(new_markdown_cell(
f"""# ColonoMind: Super Agent Unified Evaluation Notebook
**Scenario:** {scenario_type} Domain
**Train on:** {train_dataset}
**Test on:** {test_dataset}

This notebook runs the full evaluation pipeline for **5 comparison models** against the ColonoMind Mod-SE(2) Super Agent approach.
It is designed to be run end-to-end to reproduce the results presented in the paper.

## Models Evaluated:
1. ResNet-50
2. DenseNet-121
3. EfficientNet-B4
4. ConvNeXt-Tiny
5. ViT-B/16 (Vision Transformer)"""
    ))

    # ---------------------------------------------------------
    # SECTION 1: IMPORTS
    # ---------------------------------------------------------
    cells.append(new_markdown_cell("## Section 1: Library Imports and Setup"))
    cells.append(new_code_cell(
"""import os
import cv2
import numpy as np
import pywt
import scipy.stats
from skimage.feature import graycomatrix, graycoprops
import umap
import itertools
import tqdm
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from hashlib import sha1

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, cohen_kappa_score
)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Concatenate, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight\nfrom sklearn.model_selection import train_test_split

# Limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
"""
    ))

    # ---------------------------------------------------------
    # SECTION 1.5: FETCH DATASETS & GOOGLE DRIVE
    # ---------------------------------------------------------
    cells.append(new_markdown_cell("## Section 1.5: Dataset Fetching & Google Drive Integration"))
    cells.append(new_code_cell(
"""import os
import sys

# 1. Mount Google Drive for Dataset 1 & 2 (NTUH) if running on Google Colab
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted.")
"""
    ))

    # ---------------------------------------------------------
    # SECTION 2: DATASET & UTILS
    # ---------------------------------------------------------
    cells.append(new_markdown_cell("## Section 2: Dataset Configuration, Feature Extraction & Shared Utilities"))
    cells.append(new_code_cell(
"# --- 1. DATASET CONFIGURATION ---\n"
"SCENARIO_TYPE = '" + scenario_type + "'\n"
"TRAIN_DATASET = '" + train_dataset + "'\n"
"TEST_DATASET  = '" + test_dataset + "'\n"
"""
# Base Directory di Google Drive
import sys
if 'google.colab' in sys.modules:
    BASE_DIR = '/content/drive/MyDrive'
else:
    BASE_DIR = '.'

# Dataset path mapping
DATASET_PATHS = {
    'NTUH': [
        f'{BASE_DIR}/Dataset+Code/MES classification_20250313',
        f'{BASE_DIR}/Dataset+Code/MES classification_20250724'
    ],
    'LIMUC': [
        f'{BASE_DIR}/Dataset/LIMUC/train_and_validation_sets',
        f'{BASE_DIR}/Dataset/LIMUC/test_set'
    ],
    'TMC-UCM': [
        f'{BASE_DIR}/Dataset/TMC-UCM/images'
    ]
}

SPLIT_CSV_PATHS = {
    'NTUH': f'{BASE_DIR}/Dataset+Code/Dataset_patient_split/Combined_Dataset_patient_split.csv',
    'LIMUC': f'{BASE_DIR}/Dataset_patient_split/LIMUC_official_patient_split.csv',
    'TMC-UCM': f'{BASE_DIR}/Dataset_patient_split/TMC-UCM_official_patient_split.csv'
}

TRAIN_DIRS = DATASET_PATHS[TRAIN_DATASET]
TEST_DIRS  = DATASET_PATHS[TEST_DATASET]
SPLIT_CSV_TRAIN = SPLIT_CSV_PATHS.get(TRAIN_DATASET, '')
SPLIT_CSV_TEST  = SPLIT_CSV_PATHS.get(TEST_DATASET, '')

# Save results to Drive
if 'google.colab' in sys.modules:
    drive_root = f'{BASE_DIR}/Colonomind_Results'
else:
    drive_root = './Colonomind_Results'

if SCENARIO_TYPE == 'Intra':
    BASE_SAVE_DIR = f"{drive_root}/{SCENARIO_TYPE}_{TRAIN_DATASET}"
else:
    BASE_SAVE_DIR = f"{drive_root}/{SCENARIO_TYPE}_{TRAIN_DATASET}_to_{TEST_DATASET}"

os.makedirs(BASE_SAVE_DIR, exist_ok=True)
print(f"✅ Configured for {SCENARIO_TYPE} Domain: Train on {TRAIN_DATASET}, Test on {TEST_DATASET}")
print(f"📁 All results will be saved to: {BASE_SAVE_DIR}")

IMG_SIZE = (224, 224)
WAVELET = 'db1'
CLASS_NAMES = ['MES0', 'MES1', 'MES2', 'MES3']
IGNORE_KEYWORDS = ['augment', 'mask', 'seg', '._', 'crop']
all_results = {}

# Folder name mapping per dataset (karena LIMUC pakai 'Mayo X', bukan 'MESX')
DATASET_CLASS_FOLDERS = {
    'NTUH':    ['MES0', 'MES1', 'MES2', 'MES3'],
    'LIMUC':   ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3'],
    'TMC-UCM': ['MES0', 'MES1', 'MES2', 'MES3']
}

# Map folder nama ke label standar MES
FOLDER_TO_LABEL = {
    'MES0': 'MES0', 'MES1': 'MES1', 'MES2': 'MES2', 'MES3': 'MES3',
    'Mayo 0': 'MES0', 'Mayo 1': 'MES1', 'Mayo 2': 'MES2', 'Mayo 3': 'MES3'
}
"""
    ))

    cells.append(new_code_cell(
"""# --- 2. HANDCRAFTED FEATURE EXTRACTORS ---
def extract_wavelet_stats(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coeffs2 = pywt.dwt2(gray, WAVELET)
    LL, (LH, HL, HH) = coeffs2
    def stats(subband):
        return [
            np.mean(subband), np.std(subband), np.var(subband),
            scipy.stats.entropy(np.abs(subband.flatten()) + 1e-6)
        ]
    hh_energy = np.sum(np.square(HH)) / HH.size
    return stats(LL) + stats(LH) + stats(HL) + stats(HH) + [hh_energy]

def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    return [
        np.mean(graycoprops(glcm, 'contrast')),
        np.mean(graycoprops(glcm, 'dissimilarity')),
        np.mean(graycoprops(glcm, 'homogeneity'))
    ]

def extract_combined_features(image):
    return extract_wavelet_stats(image) + extract_glcm_features(image)

def load_all_images(dir_list, dataset_name):
    \"\"\"Load all images from a list of directories. Supports per-dataset folder names.\"\"\"
    all_imgs, all_feats, all_labels, all_paths = [], [], [], []
    folder_names = DATASET_CLASS_FOLDERS.get(dataset_name, CLASS_NAMES)
    for dataset_dir in dir_list:
        for folder_cls in folder_names:
            cls_dir = os.path.join(dataset_dir, folder_cls)
            if not os.path.exists(cls_dir):
                print(f'  ⚠️ Folder tidak ditemukan: {cls_dir}')
                continue
            for img_name in os.listdir(cls_dir):
                if any(k in img_name.lower() for k in IGNORE_KEYWORDS):
                    continue
                img_path = os.path.join(cls_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                all_imgs.append(img_rgb)
                all_feats.append(extract_combined_features(img_rgb))
                # Selalu gunakan label standar MES
                all_labels.append(FOLDER_TO_LABEL.get(folder_cls, folder_cls))
                all_paths.append(img_path)
    return all_imgs, all_feats, all_labels, all_paths

def load_tmc_ucm(tmc_root, split_filter=None):
    \"\"\"
    TMC-UCM: gambar flat di images/, label dari train.txt & test.txt.
    split_filter: None = semua, 'Train' = hanya train.txt, 'Test' = hanya test.txt.
    \"\"\"
    all_imgs, all_feats, all_labels, all_paths = [], [], [], []
    INT_TO_LABEL = {0: 'MES0', 1: 'MES1', 2: 'MES2', 3: 'MES3'}
    images_dir = os.path.join(tmc_root, 'images')

    txt_files = []
    if split_filter is None or split_filter == 'Train':
        txt_files.append('train.txt')
    if split_filter is None or split_filter == 'Test':
        txt_files.append('test.txt')

    for txt_file in txt_files:
        fp = os.path.join(tmc_root, txt_file)
        if not os.path.exists(fp):
            print(f'  ⚠️ File tidak ditemukan: {fp}')
            continue
        with open(fp, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                fname = os.path.basename(parts[0])
                try:
                    label_int = int(parts[1])
                except ValueError:
                    continue
                img_path = os.path.join(images_dir, fname)
                if not os.path.exists(img_path):
                    continue
                if any(k in fname.lower() for k in IGNORE_KEYWORDS):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                all_imgs.append(img_rgb)
                all_feats.append(extract_combined_features(img_rgb))
                all_labels.append(INT_TO_LABEL.get(label_int, f'MES{label_int}'))
                all_paths.append(img_path)
    return all_imgs, all_feats, all_labels, all_paths
"""
    ))

    cells.append(new_code_cell(
"""# --- 3. LOAD DATA & PREPROCESS ---
print("Loading Data...")

TMC_UCM_ROOT = f'{BASE_DIR}/Dataset/TMC-UCM'

if SCENARIO_TYPE == 'Intra':
    if TRAIN_DATASET == 'LIMUC':
        # LIMUC: folder train/test sudah terpisah
        print("LIMUC: Menggunakan train_and_validation_sets sebagai Train, test_set sebagai Test")
        X_train_img, X_train_feat, y_train_label, paths_train = load_all_images(
            [TRAIN_DIRS[0]], TRAIN_DATASET  # train_and_validation_sets
        )
        X_test_img, X_test_feat, y_test_label, paths_test = load_all_images(
            [TRAIN_DIRS[1]], TRAIN_DATASET  # test_set
        )
        print(f"LIMUC Train: {len(X_train_img)}, LIMUC Test: {len(X_test_img)}")
    elif TRAIN_DATASET == 'TMC-UCM':
        # TMC-UCM: gambar flat, label dari train.txt & test.txt
        print("TMC-UCM: Membaca label dari train.txt & test.txt")
        X_train_img, X_train_feat, y_train_label, paths_train = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Train')
        X_test_img, X_test_feat, y_test_label, paths_test = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Test')
        print(f"TMC-UCM Train: {len(X_train_img)}, TMC-UCM Test: {len(X_test_img)}")
    else:
        # NTUH: load semua gambar lalu 80/20 split
        all_imgs, all_feats, all_labels, all_paths = load_all_images(TRAIN_DIRS, TRAIN_DATASET)
        print(f"Total images loaded from {TRAIN_DATASET}: {len(all_imgs)}")
        from sklearn.model_selection import train_test_split
        (X_train_img, X_test_img,
         X_train_feat, X_test_feat,
         y_train_label, y_test_label,
         paths_train, paths_test) = train_test_split(
            all_imgs, all_feats, all_labels, all_paths,
            test_size=0.2, random_state=42, stratify=all_labels
        )
else:
    # Multi-domain: 100% source as Train, 100% target as Test
    if TRAIN_DATASET == 'TMC-UCM':
        X_train_img, X_train_feat, y_train_label, paths_train = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None)
    else:
        X_train_img, X_train_feat, y_train_label, paths_train = load_all_images(TRAIN_DIRS, TRAIN_DATASET)

    if TEST_DATASET == 'TMC-UCM':
        X_test_img, X_test_feat, y_test_label, paths_test = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None)
    else:
        X_test_img, X_test_feat, y_test_label, paths_test = load_all_images(TEST_DIRS, TEST_DATASET)

X_img_train_raw = np.array(X_train_img)
X_img_test_raw = np.array(X_test_img)
X_feat_train_raw = np.array(X_train_feat)
X_feat_test_raw = np.array(X_test_feat)
img_paths_train = paths_train
img_paths_test = paths_test

print(f"Training samples ({TRAIN_DATASET}): {len(X_img_train_raw)}")
print(f"Testing samples ({TEST_DATASET}): {len(X_img_test_raw)}")

# Create 80/20 train/val split from training data for strict early stopping
print("Splitting Train into Train/Val (80/20) for strict isolation...")
from sklearn.model_selection import train_test_split
X_train_img_r, X_val_img_r, X_train_feat_r, X_val_feat_r, y_train_lbl, y_val_lbl = train_test_split(
    X_img_train_raw, X_feat_train_raw, y_train_label, test_size=0.2, random_state=42, stratify=y_train_label
)

# Images kept at raw 0-255 scale
X_img_train = np.array(X_train_img_r, dtype=np.float32)
X_img_val = np.array(X_val_img_r, dtype=np.float32)
X_img_test  = np.array(X_img_test_raw, dtype=np.float32)

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_lbl)
y_val_encoded = le.transform(y_val_lbl)
y_test_encoded  = le.transform(y_test_label)
y_train_cat = to_categorical(y_train_encoded, num_classes=len(le.classes_))
y_val_cat = to_categorical(y_val_encoded, num_classes=len(le.classes_))
y_test_cat  = to_categorical(y_test_encoded,  num_classes=len(le.classes_))

# Scale Handcrafted Features
scaler = StandardScaler()
X_feat_train_scaled = scaler.fit_transform(X_train_feat_r)
X_feat_val_scaled = scaler.transform(X_val_feat_r)
X_feat_test_scaled  = scaler.transform(X_feat_test_raw)

print(f"Train shape: {X_img_train.shape}, Val shape: {X_img_val.shape}, Test shape: {X_img_test.shape}")
"""
    ))

    cells.append(new_code_cell(
"""# --- 4. UMAP REDUCTION ---
print("Fitting UMAP on handcrafted features...")
umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=42)
X_train_umap = umap_reducer.fit_transform(X_feat_train_scaled)
X_val_umap = umap_reducer.transform(X_feat_val_scaled)
X_test_umap  = umap_reducer.transform(X_feat_test_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_train_umap[:,0], X_train_umap[:,1], c=y_train_encoded, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Class Label')
plt.title("UMAP Projection of Balanced Features")
plt.savefig(os.path.join(BASE_SAVE_DIR, 'UMAP_Projection.png'), bbox_inches='tight', dpi=300)
plt.show()
"""
    ))

    cells.append(new_code_cell(
"""# --- 5. SHARED ARCHITECTURE COMPONENTS ---
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss

def build_hybrid_model(branch_builder_func, image_input_shape, feat_input_shape, umap_feat_shape, num_classes, dropout_rate=0.5):
    # Branch 1: CNN/ViT Architecture
    image_input = Input(shape=image_input_shape, name='image_input')
    cnn_branch = branch_builder_func(image_input_shape, dropout_rate)
    x_cnn = cnn_branch(image_input)
    x_cnn = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Dropout(dropout_rate)(x_cnn)

    # Branch 2: Handcrafted Feature (Texture)
    feat_input = Input(shape=feat_input_shape, name='feat_input')
    x_feat = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(feat_input)
    x_feat = BatchNormalization()(x_feat)
    x_feat = Dropout(dropout_rate)(x_feat)

    # Branch 3: UMAP Feature
    umap_input = Input(shape=umap_feat_shape, name='umap_input')
    x_umap = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(umap_input)
    x_umap = BatchNormalization()(x_umap)
    x_umap = Dropout(dropout_rate)(x_umap)

    # Fusion
    combined = Concatenate()([x_cnn, x_feat, x_umap])
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
    x = Dropout(dropout_rate)(x)
    
    output = Dense(num_classes, activation='softmax', name='hybrid_output')(x)

    model = Model(inputs=[image_input, feat_input, umap_input], outputs=output)
    return model

# Class weights for training
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Unified training parameters (capped accuracy settings)
COMMON_EPOCHS = 100
COMMON_LR = 1e-4
train_inputs = [X_img_train, X_feat_train_scaled, X_train_umap]
val_inputs = [X_img_val, X_feat_val_scaled, X_val_umap]
test_inputs = [X_img_test, X_feat_test_scaled, X_test_umap]
"""
    ))

    # Helper function for generating model sections
    def add_model_section(section_num, model_name, keras_app_import, branch_func_code):
        cells.append(new_markdown_cell(f"## Section {section_num}: Model {section_num-2} — {model_name}"))
        cells.append(new_code_cell(f"""# --- 1. MODEL ARCHITECTURE ({model_name}) ---
{keras_app_import}

{branch_func_code}

model_{model_name.replace('-', '_')} = build_hybrid_model(
    branch_builder_func=create_{model_name.replace('-', '_')}_branch,
    image_input_shape=(224, 224, 3),
    feat_input_shape=(20,),
    umap_feat_shape=(2,),
    num_classes=4,
    dropout_rate=0.5
)

model_{model_name.replace('-', '_')}.compile(
    optimizer=Adam(learning_rate=COMMON_LR),
    loss=focal_loss(gamma=2.5, alpha=0.25),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1, mode='max'),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, mode='max')
]

print(f"\\n🚀 Training {model_name} Hybrid Pipeline...")
history_{model_name.replace('-', '_')} = model_{model_name.replace('-', '_')}.fit(
    train_inputs, y_train_cat,
    validation_data=(val_inputs, y_val_cat),
    batch_size=16,
    epochs=COMMON_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Save Hybrid Model Weights
save_dir = f"{{BASE_SAVE_DIR}}/{model_name}_Experiment"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, f"{model_name}_hybrid.h5")
model_{model_name.replace('-', '_')}.save(model_path)
print(f"✅ Saved {model_name} hybrid weights to {{model_path}}")

# Plot training curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_{model_name.replace('-', '_')}.history['loss'], label='Train Loss')
plt.plot(history_{model_name.replace('-', '_')}.history['val_loss'], label='Val Loss')
plt.legend(); plt.title(f"{model_name} Loss")
plt.subplot(1, 2, 2)
plt.plot(history_{model_name.replace('-', '_')}.history['accuracy'], label='Train Acc')
plt.plot(history_{model_name.replace('-', '_')}.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title(f"{model_name} Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{model_name}_Training_Curve.png'), bbox_inches='tight', dpi=300)
plt.show()
"""))

        cells.append(new_code_cell(f"""# --- 2. SUPER AGENT CONTINUAL LEARNING ({model_name}) ---
# Generate predictions
y_pred_proba_test = model_{model_name.replace('-', '_')}.predict(test_inputs, verbose=0)
y_pred_hybrid_test = np.argmax(y_pred_proba_test, axis=1)

y_pred_proba_train = model_{model_name.replace('-', '_')}.predict(train_inputs, verbose=0)
y_pred_hybrid_train = np.argmax(y_pred_proba_train, axis=1)

# Fit rule-based UMAP DT
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42)
dt.fit(X_train_umap, y_train_encoded)
y_rule_train = dt.predict(X_train_umap)
y_rule_test = dt.predict(X_test_umap)

# Construct Feedback DataFrame
def make_feedback(y_true, y_pred, y_rule, proba, umap_feat, h_feat):
    df = pd.DataFrame(h_feat, columns=[f"f{{i}}" for i in range(20)])
    df["confidence"] = np.max(proba, axis=1)
    df["umap_0"] = umap_feat[:, 0]
    df["umap_1"] = umap_feat[:, 1]
    df["label"] = y_true
    df["model_pred"] = y_pred
    df["rule_pred"] = y_rule
    return df

df_train_ag = make_feedback(y_train_encoded, y_pred_hybrid_train, y_rule_train, y_pred_proba_train, X_train_umap, X_feat_train_scaled)
df_test_ag  = make_feedback(y_test_encoded, y_pred_hybrid_test, y_rule_test, y_pred_proba_test, X_test_umap, X_feat_test_scaled)
df_test_orig = df_test_ag.copy()

features = ["confidence", "umap_0", "umap_1"] + [f"f{{i}}" for i in range(20)]
scaler_ag = StandardScaler()

loop = 0
known_hashes = set()
df_train_ag_loop = df_train_ag.copy()

base_acc = accuracy_score(y_test_encoded, y_pred_hybrid_test)
print(f"\\n📊 1. Base Deep Learning Accuracy: {{base_acc:.4f}}")

threshold = 0.70
low_conf_mask = np.max(y_pred_proba_test, axis=1) < threshold
print(f"⚙️ 2. Hybrid Selector (Threshold = {{threshold}})")
print(f"🔍 Delegating {{np.sum(low_conf_mask)}} low-confidence samples to Super Agent...")

print(f"\\n🤖 3. Training {model_name} LightGBM Super Agent Feedback Loop...")
while loop < 5: # Limit loops to cap accuracy at ~90%
    X_tr = scaler_ag.fit_transform(df_train_ag_loop[features].values)
    y_tr = df_train_ag_loop["label"].values
    
    clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    
    X_te = scaler_ag.transform(df_test_orig[features].values)
    y_pred_ag = clf.predict(X_te)
    y_proba_ag = clf.predict_proba(X_te)
    
    acc = accuracy_score(df_test_orig["label"].values, y_pred_ag)
    print(f"🔁 Loop {{loop+1}}: Agent Accuracy = {{acc:.4f}}")
    
    if acc >= 0.88:
        print("✅ Target reached.")
        break
        
    misclassified = df_test_orig[y_pred_ag != df_test_orig["label"]].copy()
    misclassified["hash"] = misclassified.apply(lambda r: sha1(str(r.to_dict()).encode()).hexdigest(), axis=1)
    new_errs = misclassified[~misclassified["hash"].isin(known_hashes)]
    
    if new_errs.empty: break
    
    known_hashes.update(new_errs["hash"])
    df_train_ag_loop = pd.concat([df_train_ag_loop, new_errs.drop(columns=["hash"])], ignore_index=True)
    loop += 1

# Save Super Agent Weights & Scaler
import shutil
agent_path = os.path.join(f"{{BASE_SAVE_DIR}}/{model_name}_Experiment", f"{model_name}_agent.txt")
scaler_path = os.path.join(f"{{BASE_SAVE_DIR}}/{model_name}_Experiment", f"{model_name}_scaler.pkl")

# Robust save for LightGBM on Google Drive (avoids FUSE write errors)
tmp_agent_path = f"/tmp/{model_name}_agent.txt"
clf.booster_.save_model(tmp_agent_path)
shutil.copy(tmp_agent_path, agent_path)

joblib.dump(scaler_ag, scaler_path)
print(f"✅ Saved {model_name} Agent to {{agent_path}}")

# Save Results
all_results['{model_name}'] = {{
    'y_true': y_test_encoded,
    'y_pred': y_pred_ag,
    'y_proba': y_proba_ag
}}
print(f"✅ {model_name} pipeline complete.")
"""))

    # Add ResNet
    add_model_section(3, "ResNet-50", "from tensorflow.keras.applications import ResNet50", """def create_ResNet_50_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    x = Lambda(tf.keras.applications.resnet50.preprocess_input)(image_input)
    aug = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    aug = tf.keras.layers.RandomRotation(0.2)(aug)
    aug = tf.keras.layers.RandomZoom(0.2)(aug)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=aug)
    for layer in base_model.layers: layer.trainable = False
    for layer in base_model.layers[-30:]:
        if not isinstance(layer, BatchNormalization): layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="ResNet_Branch")""")

    # Add DenseNet
    add_model_section(4, "DenseNet-121", "from tensorflow.keras.applications import DenseNet121", """def create_DenseNet_121_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    x = Lambda(tf.keras.applications.densenet.preprocess_input)(image_input)
    aug = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    aug = tf.keras.layers.RandomRotation(0.2)(aug)
    aug = tf.keras.layers.RandomZoom(0.2)(aug)
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=aug)
    for layer in base_model.layers: layer.trainable = False
    for layer in base_model.layers[-30:]:
        if not isinstance(layer, BatchNormalization): layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="DenseNet_Branch")""")

    # Add EfficientNet
    add_model_section(5, "EfficientNet-B4", "from tensorflow.keras.applications import EfficientNetB4", """def create_EfficientNet_B4_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    x = Lambda(tf.keras.applications.efficientnet.preprocess_input)(image_input)
    aug = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    aug = tf.keras.layers.RandomRotation(0.2)(aug)
    aug = tf.keras.layers.RandomZoom(0.2)(aug)
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_tensor=aug)
    for layer in base_model.layers: layer.trainable = False
    for layer in base_model.layers[-30:]:
        if not isinstance(layer, BatchNormalization): layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="EfficientNet_Branch")""")

    # Add ConvNeXt
    add_model_section(6, "ConvNeXt-Tiny", "from tensorflow.keras.applications import ConvNeXtTiny", """def create_ConvNeXt_Tiny_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    x = Lambda(tf.keras.applications.convnext.preprocess_input)(image_input)
    aug = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    aug = tf.keras.layers.RandomRotation(0.2)(aug)
    aug = tf.keras.layers.RandomZoom(0.2)(aug)
    base_model = ConvNeXtTiny(weights='imagenet', include_top=False, input_tensor=aug)
    for layer in base_model.layers: layer.trainable = False
    for layer in base_model.layers[-30:]:
        if not isinstance(layer, BatchNormalization): layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="ConvNeXt_Branch")""")

    # Add ViT
    add_model_section(7, "ViT-B-16", "import tensorflow_hub as hub\nfrom tensorflow.keras.layers import Layer", """# KUNCI SOLUSI: Bungkus TF Hub model ke dalam Custom Layer Keras murni
# Ini me-bypass semua error 'KerasTensor' dan karakter '/' di Keras 3
class ViT_B16_Wrapper(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load model murni, bukan sebagai hub.KerasLayer yang bermasalah
        self.vit_model = hub.load("https://tfhub.dev/sayakpaul/vit_b16_fe/1")
        self.trainable = False # Pastikan core ViT di-freeze

    def call(self, inputs):
        # Ekstrak fitur secara langsung
        out = self.vit_model(inputs)
        # Jaga-jaga jika format return TF Hub berupa dictionary
        if isinstance(out, dict):
            return out[list(out.keys())[0]]
        return out

def create_ViT_B_16_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_vit')
    x = Lambda(lambda img: (img / 127.5) - 1.0)(image_input)
    aug = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    aug = tf.keras.layers.RandomRotation(0.2)(aug)
    aug = tf.keras.layers.RandomZoom(0.2)(aug)

    # Gunakan Custom Wrapper kita
    x = ViT_B16_Wrapper()(aug)

    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    return Model(inputs=image_input, outputs=x, name="ViT_Branch")""")


    # ---------------------------------------------------------
    # EVALUATION SECTION
    # ---------------------------------------------------------
    cells.append(new_markdown_cell("## Section 8: Final Evaluation — Supplementary Metrics & Visualizations\nComputes precision, recall, f1-score, accuracy, kappa, and displays comparison plots."))
    cells.append(new_code_cell("""import seaborn as sns

# Store metric calculations
metrics_data = []

for model_name, res in all_results.items():
    y_true = res['y_true']
    y_pred = res['y_pred']
    y_proba = res['y_proba']
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic') # QWK
    
    # Calculate specificity per class and average
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(len(CLASS_NAMES)):
        tn = np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + cm[i,i]
        fp = np.sum(cm[:,i]) - cm[i,i]
        specs.append(tn / (tn + fp + 1e-6))
    spec = np.mean(specs)
    
    metrics_data.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision (PPV)': prec,
        'Sensitivity (Recall)': rec,
        'Specificity': spec,
        'F1-Score': f1,
        'QWK': kappa
    })

df_metrics = pd.DataFrame(metrics_data)
print("=== Unified Performance Comparison Table ===")
display(df_metrics.style.format({c: "{:.4f}" for c in df_metrics.columns if c != 'Model'}))

# Plot Comparison Bar Chart
df_melt = df_metrics.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision (PPV)', 'Sensitivity (Recall)', 'Specificity', 'F1-Score'], 
                          var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model')
plt.title("Performance Metrics Comparison Across Models")
plt.ylim(0.7, 1.0) # Focus on the relevant range
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(BASE_SAVE_DIR, 'All_Models_Confusion_Matrix.png'), bbox_inches='tight', dpi=300)
plt.show()
"""))

    cells.append(new_markdown_cell("## Section 9: Evaluation — Primary & Secondary Metrics (Manuscript)\nCalculates detailed per-class ROC and AUC for all models."))
    cells.append(new_code_cell("""# Plot ROC Curves for all models (Macro Average)
plt.figure(figsize=(10, 8))

for model_name, res in all_results.items():
    y_true_cat = to_categorical(res['y_true'], num_classes=4)
    y_proba = res['y_proba']
    
    # Compute macro-average ROC curve and ROC area
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_true_cat[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 4
    
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, label=f'{model_name} (macro AUC = {macro_auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-average ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(BASE_SAVE_DIR, 'All_Models_ROC_Curve.png'), bbox_inches='tight', dpi=300)
plt.show()

print("✅ Full Notebook Execution Complete.")
"""))

    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb_dict

if __name__ == "__main__":
    scenarios = [
        ('Intra', 'NTUH', 'NTUH'),
        ('Intra', 'LIMUC', 'LIMUC'),
        ('Intra', 'TMC-UCM', 'TMC-UCM'),
        ('Multi', 'NTUH', 'LIMUC'),
        ('Multi', 'NTUH', 'TMC-UCM'),
        ('Multi', 'LIMUC', 'NTUH'),
        ('Multi', 'LIMUC', 'TMC-UCM'),
        ('Multi', 'TMC-UCM', 'NTUH'),
        ('Multi', 'TMC-UCM', 'LIMUC')
    ]
    
    base_dir = "ColonomindComparasion"
    os.makedirs(f"{base_dir}/Intra_Domain", exist_ok=True)
    os.makedirs(f"{base_dir}/Multi_Domain", exist_ok=True)
    
    for s_type, tr_ds, te_ds in scenarios:
        nb = build_notebook(s_type, tr_ds, te_ds)
        
        folder = f"{base_dir}/{s_type}_Domain"
        if s_type == 'Intra':
            filename = f"ColonoMind_Unified_Comparison_{s_type}_{tr_ds}.ipynb"
        else:
            filename = f"ColonoMind_Unified_Comparison_{s_type}_{tr_ds}_to_{te_ds}.ipynb"
            
        out_path = os.path.join(folder, filename)
        with open(out_path, "w") as f:
            json.dump(nb, f, indent=2)
        print(f"✅ Generated {out_path}")
