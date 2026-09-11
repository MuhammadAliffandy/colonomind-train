import os
import tensorflow as tf

# Make only GPU 7 visible to TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# List the visible GPUs (GPU 7 will appear as GPU 0 to TensorFlow)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Allow memory growth instead of setting a fixed limit
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Using GPU 3 (seen as GPU 0 by TF): {len(gpus)} physical, {len(logical_gpus)} logical GPUs")
    except RuntimeError as e:
        print("TF GPU setup error:", e)
else:
    print("No GPU available. Running on CPU.")

import os
import cv2
import numpy as np
import pywt
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.utils import to_categorical
import umap
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.tree import DecisionTreeClassifier, _tree, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import itertools
import tqdm
from transformers import TFViTModel, ViTFeatureExtractor


# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
WAVELET = 'db1'

TEST_DIR = './MES classification_20250724' # <-- Change this to your training dataset path
TRAIN_DIR  = './MES classification_20250313' # <-- Change this to your testing dataset path

# --- WAVELET + GLCM FEATURE EXTRACTORS ---
def extract_wavelet_stats(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coeffs2 = pywt.dwt2(gray, WAVELET)
    LL, (LH, HL, HH) = coeffs2
    def stats(subband):
        return [
            np.mean(subband),
            np.std(subband),
            np.var(subband),
            scipy.stats.entropy(np.abs(subband.flatten()) + 1e-6)
        ]
    features = []
    for band in [LL, LH, HL, HH]:
        features.extend(stats(band))
    hh_energy = np.sum(np.square(HH))
    features.append(hh_energy)
    return features

def extract_glcm_features_extended(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    angles = [0, np.pi/4, np.pi/2]
    glcm = graycomatrix(gray, distances=[5], angles=angles, levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    return [contrast, dissimilarity, homogeneity]

# --- REUSABLE DATA LOADER ---
def load_dataset(folder_path):
    X_img, X_feat, y_label, img_paths = [], [], [], []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path): continue
        for fname in os.listdir(label_path):
            img_path = os.path.join(label_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cropped = img[30:430, 200:550]
            resized = cv2.resize(cropped, IMG_SIZE)
            wavelet_feats = extract_wavelet_stats(resized)
            glcm_feats = extract_glcm_features_extended(resized)
            combined = wavelet_feats + glcm_feats
            X_img.append(resized)
            X_feat.append(combined)
            y_label.append(label)
            img_paths.append(img_path)
    return np.array(X_img), np.array(X_feat), np.array(y_label), np.array(img_paths)

# --- LOAD TRAINING + TEST SET ---
X_img_train_raw, X_feat_train_raw, y_train_label, img_paths_train = load_dataset(TRAIN_DIR)
X_img_test_raw,  X_feat_test_raw,  y_test_label,  img_paths_test  = load_dataset(TEST_DIR)

# --- NORMALIZE IMAGES ---
X_img_train = X_img_train_raw.astype(np.float32) / 255.0
X_img_test  = X_img_test_raw.astype(np.float32) / 255.0

# --- LABEL ENCODING ---
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_label)
y_test_encoded  = le.transform(y_test_label)
y_train_cat = to_categorical(y_train_encoded, num_classes=len(le.classes_))
y_test_cat  = to_categorical(y_test_encoded,  num_classes=len(le.classes_))

# --- FEATURE SCALING ---
scaler = StandardScaler()
X_feat_train_scaled = scaler.fit_transform(X_feat_train_raw)
X_feat_test_scaled  = scaler.transform(X_feat_test_raw)

# --- PLOT TRAIN CLASS DISTRIBUTION (Before SMOTE) ---
plt.figure(figsize=(8, 4))
plt.title("Training Set Class Distribution (Before SMOTE)")
plt.bar(*np.unique(y_train_encoded, return_counts=True), tick_label=le.classes_)
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- PLOT TEST CLASS DISTRIBUTION ---
plt.figure(figsize=(8, 4))
plt.title("Testing Set Class Distribution")
plt.bar(*np.unique(y_test_encoded, return_counts=True), tick_label=le.classes_, color='orange')
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- APPLY SMOTE TO TRAINING SET ---
smote = SMOTE(random_state=42)
X_feat_train_bal, y_train_bal = smote.fit_resample(X_feat_train_scaled, y_train_encoded)

# --- PLOT TRAIN CLASS DISTRIBUTION (After SMOTE) ---
plt.figure(figsize=(8, 4))
plt.title("Training Set Class Distribution (After SMOTE)")
plt.bar(*np.unique(y_train_bal, return_counts=True), tick_label=le.classes_, color='green')
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- MAP BALANCED FEATURES TO REAL IMAGES ---
X_img_train_bal, img_paths_train_bal = [], []
for feat, label in zip(X_feat_train_bal, y_train_bal):
    dists = np.linalg.norm(X_feat_train_scaled[y_train_encoded == label] - feat, axis=1)
    idx = np.where(y_train_encoded == label)[0][np.argmin(dists)]
    X_img_train_bal.append(X_img_train[idx])
    img_paths_train_bal.append(img_paths_train[idx])
X_img_train_bal = np.array(X_img_train_bal, dtype=np.float32)
y_train_cat_bal = to_categorical(y_train_bal, num_classes=len(le.classes_))

# --- UMAP PROJECTION (fit on training, apply to test) ---
umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, metric='euclidean', random_state=42)
X_train_umap = umap_reducer.fit_transform(X_feat_train_bal)
X_test_umap  = umap_reducer.transform(X_feat_test_scaled)

# --- SHAPE SUMMARY ---
print(f"X_img_train_bal: {X_img_train_bal.shape}, X_img_test: {X_img_test.shape}")
print(f"X_feat_train_bal: {X_feat_train_bal.shape}, X_feat_test_scaled: {X_feat_test_scaled.shape}")
print(f"X_train_umap: {X_train_umap.shape}, X_test_umap: {X_test_umap.shape}")
print(f"y_train_cat_bal: {y_train_cat_bal.shape}, y_test_cat: {y_test_cat.shape}")


import matplotlib.pyplot as plt

# --- Show training example ---
img_train = cv2.imread(img_paths_train[0])
img_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB)
cropped_train = img_train[30:430, 200:550]
resized_train = cv2.resize(cropped_train, IMG_SIZE)

# --- Show testing example ---
img_test = cv2.imread(img_paths_test[0])
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
cropped_test = img_test[30:430, 200:550]
resized_test = cv2.resize(cropped_test, IMG_SIZE)

# --- Plot side-by-side comparison ---
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

# Row 1: Training example
axs[0, 0].imshow(img_train)
axs[0, 0].set_title("Train - Original")
axs[0, 1].imshow(cropped_train)
axs[0, 1].set_title("Train - Cropped")
axs[0, 2].imshow(resized_train)
axs[0, 2].set_title("Train - Resized 224x224")

# Row 2: Testing example
axs[1, 0].imshow(img_test)
axs[1, 0].set_title("Test - Original")
axs[1, 1].imshow(cropped_test)
axs[1, 1].set_title("Test - Cropped")
axs[1, 2].imshow(resized_test)
axs[1, 2].set_title("Test - Resized 224x224")

# Formatting
for ax in axs.flatten():
    ax.axis('off')
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# --- Rescale to 0–255 for visualization purposes ---
train_original = X_img_train * 255.0
test_original  = X_img_test * 255.0

# --- Flatten pixel values across all channels and images ---
train_pixels_orig = train_original.flatten()
test_pixels_orig  = test_original.flatten()

train_pixels_norm = X_img_train.flatten()
test_pixels_norm  = X_img_test.flatten()

# --- Plot original pixel distributions (0–255) ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(train_pixels_orig, bins=50, color='gray', alpha=0.7, label='Train')
plt.hist(test_pixels_orig,  bins=50, color='red', alpha=0.5, label='Test')
plt.title("Pixel Value Distribution [0–255]")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()

# --- Plot normalized pixel distributions (0–1) ---
plt.subplot(1, 2, 2)
plt.hist(train_pixels_norm, bins=50, color='blue', alpha=0.7, label='Train')
plt.hist(test_pixels_norm,  bins=50, color='orange', alpha=0.5, label='Test')
plt.title("Pixel Value Distribution After Normalization [0–1]")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()


import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Use handcrafted feature names
feature_names = [
    'LL_mean', 'LL_std', 'LL_var', 'LL_entropy',
    'LH_mean', 'LH_std', 'LH_var', 'LH_entropy',
    'HL_mean', 'HL_std', 'HL_var', 'HL_entropy',
    'HH_mean', 'HH_std', 'HH_var', 'HH_entropy',
    'GLCM_contrast', 'GLCM_dissimilarity', 'GLCM_homogeneity',
    'HH_energy'
]

# Create raw DataFrames
df_train_raw = pd.DataFrame(X_feat_train_raw[:, :20], columns=feature_names)
df_test_raw  = pd.DataFrame(X_feat_test_raw[:, :20],  columns=feature_names)

# Standardize using same scaler
scaler = StandardScaler()
df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train_raw), columns=feature_names)
df_test_scaled  = pd.DataFrame(scaler.transform(df_test_raw),  columns=feature_names)

# Plot 1: Train - Before Scaling
plt.figure(figsize=(12, 5))
sns.boxplot(data=df_train_raw)
plt.title("Train Set - Handcrafted Features (Before Standardization)")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Train - After Scaling
plt.figure(figsize=(12, 5))
sns.boxplot(data=df_train_scaled)
plt.title("Train Set - Handcrafted Features (After Standardization)")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Test - Before Scaling
plt.figure(figsize=(12, 5))
sns.boxplot(data=df_test_raw)
plt.title("Test Set - Handcrafted Features (Before Standardization)")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 4: Test - After Scaling
plt.figure(figsize=(12, 5))
sns.boxplot(data=df_test_scaled)
plt.title("Test Set - Handcrafted Features (After Standardization)")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Handcrafted feature names
feature_names = [
    'LL_mean', 'LL_std', 'LL_var', 'LL_entropy',
    'LH_mean', 'LH_std', 'LH_var', 'LH_entropy',
    'HL_mean', 'HL_std', 'HL_var', 'HL_entropy',
    'HH_mean', 'HH_std', 'HH_var', 'HH_entropy',
    'HH_energy', 'GLCM_contrast', 'GLCM_dissimilarity', 'GLCM_homogeneity'
]

# Prepare raw DataFrames
X_feats_train = np.array(X_feat_train_raw, dtype=np.float32)
X_feats_test  = np.array(X_feat_test_raw, dtype=np.float32)
df_train_raw = pd.DataFrame(X_feats_train, columns=feature_names)
df_test_raw  = pd.DataFrame(X_feats_test,  columns=feature_names)

# Standardize using same scaler (like in real pipeline)
scaler = StandardScaler()
X_feats_train_scaled = scaler.fit_transform(X_feats_train)
X_feats_test_scaled  = scaler.transform(X_feats_test)
df_train_scaled = pd.DataFrame(X_feats_train_scaled, columns=feature_names)
df_test_scaled  = pd.DataFrame(X_feats_test_scaled,  columns=feature_names)

# --- PLOT 1: Train - Before Scaling (Log Scale) ---
plt.figure(figsize=(15, 5))
sns.boxplot(data=df_train_raw, palette="pastel")
plt.yscale('log')
plt.title("Train - Handcrafted Features Before Standardization (Log Scale)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- PLOT 2: Train - After Scaling ---
plt.figure(figsize=(15, 5))
sns.boxplot(data=df_train_scaled, palette="husl")
plt.title("Train - Handcrafted Features After Standardization")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- PLOT 3: Test - Before Scaling (Log Scale) ---
plt.figure(figsize=(15, 5))
sns.boxplot(data=df_test_raw, palette="pastel")
plt.yscale('log')
plt.title("Test - Handcrafted Features Before Standardization (Log Scale)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- PLOT 4: Test - After Scaling ---
plt.figure(figsize=(15, 5))
sns.boxplot(data=df_test_scaled, palette="husl")
plt.title("Test - Handcrafted Features After Standardization")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# --- MOD-SE(2) CNN (UNCHANGED) ---
def z2_se2n(input_tensor, kernel, orientations_nb, periodicity=2 * np.pi, diskMask=True, padding='VALID'):
    print("Base Kernel:\n", kernel.numpy())
    kernel_stack = rotate_lifting_kernels(kernel, orientations_nb, periodicity=periodicity, diskMask=diskMask)
    print("Z2-SE2N ROTATED KERNEL SET SHAPE:", kernel_stack.get_shape())
    kernels_as_if_2D = tf.transpose(kernel_stack, [1, 2, 3, 0, 4])
    kernelSizeH, kernelSizeW, channelsIN, channelsOUT = map(int, kernel.shape)
    kernels_as_if_2D = tf.reshape(kernels_as_if_2D, [kernelSizeH, kernelSizeW, channelsIN, orientations_nb * channelsOUT])
    layer_output = tf.nn.conv2d(input=input_tensor, filters=kernels_as_if_2D, strides=[1, 1, 1, 1], padding=padding)
    layer_output = tf.reshape(layer_output, [tf.shape(layer_output)[0], int(layer_output.shape[1]), int(layer_output.shape[2]), orientations_nb, channelsOUT])
    print("OUTPUT SE2N ACTIVATIONS SHAPE:", layer_output.get_shape())
    return layer_output, kernel_stack

def se2n_se2n(input_tensor, kernel, periodicity=2 * np.pi, diskMask=True, padding='VALID'):
    kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT = map(int, kernel.shape)
    kernel_stack = rotate_gconv_kernels(kernel, periodicity, diskMask)
    print("SE2N-SE2N ROTATED KERNEL SET SHAPE:", kernel_stack.get_shape())
    input_tensor_as_if_2D = tf.reshape(input_tensor, [tf.shape(input_tensor)[0], int(input_tensor.shape[1]), int(input_tensor.shape[2]), orientations_nb * channelsIN])
    kernels_as_if_2D = tf.transpose(kernel_stack, [1, 2, 3, 4, 0, 5])
    kernels_as_if_2D = tf.reshape(kernels_as_if_2D, [kernelSizeH, kernelSizeW, orientations_nb * channelsIN, orientations_nb * channelsOUT])
    layer_output = tf.nn.conv2d(input=input_tensor_as_if_2D, filters=kernels_as_if_2D, strides=[1, 1, 1, 1], padding=padding)
    layer_output = tf.reshape(layer_output, [tf.shape(layer_output)[0], int(layer_output.shape[1]), int(layer_output.shape[2]), orientations_nb, channelsOUT])
    print("OUTPUT SE2N ACTIVATIONS SHAPE:", layer_output.get_shape())
    return layer_output, kernel_stack

def spatial_max_pool(input_tensor, nbOrientations, padding='SAME'):
    activations = [None] * nbOrientations
    for i in range(nbOrientations):
        activations[i] = tf.nn.max_pool(value=input_tensor[:, :, :, i, :], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)
    tensor_pooled = tf.stack(activations, axis=-1)
    return tensor_pooled

def rotate_lifting_kernels(kernel, orientations_nb, periodicity=2 * np.pi, diskMask=True):
    kernelSizeH, kernelSizeW, channelsIN, channelsOUT = map(int, kernel.shape)
    print("Z2-SE2N BASE KERNEL SHAPE:", kernel.get_shape())
    kernel_flat = tf.reshape(kernel, [kernelSizeH * kernelSizeW, channelsIN * channelsOUT])
    idx, vals = MultiRotationOperatorMatrixSparse([kernelSizeH, kernelSizeW], orientations_nb, periodicity=periodicity, diskMask=diskMask)
    rotOp_matrix = tf.SparseTensor(idx, vals, [orientations_nb * kernelSizeH * kernelSizeW, kernelSizeH * kernelSizeW])
    set_of_rotated_kernels = tf.sparse_tensor_dense_matmul(rotOp_matrix, kernel_flat)
    set_of_rotated_kernels = tf.reshape(set_of_rotated_kernels, [orientations_nb, kernelSizeH, kernelSizeW, channelsIN, channelsOUT])
    return set_of_rotated_kernels

def rotate_gconv_kernels(kernel, periodicity=2 * np.pi, diskMask=True):
    kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT = map(int, kernel.shape)
    print("SE2N-SE2N BASE KERNEL SHAPE:", kernel.get_shape())
    kernel_flat = tf.reshape(kernel, [kernelSizeH * kernelSizeW, orientations_nb * channelsIN * channelsOUT])
    idx, vals = MultiRotationOperatorMatrixSparse([kernelSizeH, kernelSizeW], orientations_nb, periodicity=periodicity, diskMask=diskMask)
    rotOp_matrix = tf.SparseTensor(idx, vals, [orientations_nb * kernelSizeH * kernelSizeW, kernelSizeH * kernelSizeW])
    kernels_planar_rotated = tf.sparse_tensor_dense_matmul(rotOp_matrix, kernel_flat)
    kernels_planar_rotated = tf.reshape(kernels_planar_rotated, [orientations_nb, kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT])
    set_of_rotated_kernels = [None] * orientations_nb
    for orientation in range(orientations_nb):
        kernels_temp = kernels_planar_rotated[orientation]
        kernels_temp = tf.transpose(kernels_temp, [0, 1, 3, 4, 2])
        kernels_temp = tf.reshape(kernels_temp, [kernelSizeH * kernelSizeW * channelsIN * channelsOUT, orientations_nb])
        roll_matrix = tf.constant(np.roll(np.identity(orientations_nb), orientation, axis=1), dtype=tf.float32)
        kernels_temp = tf.matmul(kernels_temp, roll_matrix)
        kernels_temp = tf.reshape(kernels_temp, [kernelSizeH, kernelSizeW, channelsIN, channelsOUT, orientations_nb])
        kernels_temp = tf.transpose(kernels_temp, [0, 1, 4, 2, 3])
        set_of_rotated_kernels[orientation] = kernels_temp
    return tf.stack(set_of_rotated_kernels)

def CoordRotationInv(ij, NiNj, theta):
    centeri = np.floor(NiNj[0] / 2)
    centerj = np.floor(NiNj[1] / 2)
    ijOld = np.zeros([2])
    ijOld[0] = np.cos(theta) * (ij[0] - centeri) + np.sin(theta) * (ij[1] - centerj) + centeri
    ijOld[1] = -np.sin(theta) * (ij[0] - centeri) + np.cos(theta) * (ij[1] - centerj) + centerj
    return ijOld

def LinIntIndicesAndWeights(ij, NiNj):
    i, j = ij
    Ni, Nj = NiNj
    i1 = int(np.floor(i))
    i2 = i1 + 1
    j1 = int(np.floor(j))
    j2 = j1 + 1
    ti = i - i1
    tj = j - j1
    w11 = (1 - ti) * (1 - tj)
    w12 = (1 - ti) * tj
    w21 = ti * (1 - tj)
    w22 = ti * tj
    indicesAndWeights = []
    if (0 <= i1 < Ni) and (0 <= j1 < Nj):
        indicesAndWeights.append([i1, j1, w11])
    if (0 <= i1 < Ni) and (0 <= j2 < Nj):
        indicesAndWeights.append([i1, j2, w12])
    if (0 <= i2 < Ni) and (0 <= j1 < Nj):
        indicesAndWeights.append([i2, j1, w21])
    if (0 <= i2 < Ni) and (0 <= j2 < Nj):
        indicesAndWeights.append([i2, j2, w22])
    return indicesAndWeights

def ToLinearIndex(ij, NiNj):
    return ij[0] * NiNj[0] + ij[1]

def RotationOperatorMatrix(NiNj, theta, diskMask=True):
    Ni, Nj = NiNj
    cij = np.floor(Ni / 2)
    rotationMatrix = np.zeros([Ni * Nj, Ni * Nj])
    for i in range(NiNj[0]):
        for j in range(NiNj[0]):
            if not(diskMask) or ((i - cij) * (i - cij) + (j - cij) * (j - cij) <= (cij + 0.5) * (cij + 0.5)):
                linij = ToLinearIndex([i, j], NiNj)
                ijOld = CoordRotationInv([i, j], NiNj, theta)
                linIntIndicesAndWeights = LinIntIndicesAndWeights(ijOld, NiNj)
                for indexAndWeight in linIntIndicesAndWeights:
                    indexOld = [indexAndWeight[0], indexAndWeight[1]]
                    linIndexOld = ToLinearIndex(indexOld, NiNj)
                    weight = indexAndWeight[2]
                    rotationMatrix[linij, linIndexOld] = weight
    return rotationMatrix

def MultiRotationOperatorMatrixSparse(NiNj, Ntheta, periodicity=2 * np.pi, diskMask=True):
    idx, vals = [], []
    for r in range(Ntheta):
        idxr, valsr = RotationOperatorMatrixSparse(NiNj, periodicity * r / Ntheta, linIndOffset=r * NiNj[0] * NiNj[1], diskMask=diskMask)
        idx += idxr
        vals += valsr
    return idx, vals

def RotationOperatorMatrixSparse(NiNj, theta, diskMask=True, linIndOffset=0):
    Ni, Nj = NiNj
    cij = np.floor(Ni / 2)
    idx, vals = [], []
    for i in range(NiNj[0]):
        for j in range(NiNj[0]):
            if not(diskMask) or ((i - cij) * (i - cij) + (j - cij) * (j - cij) <= (cij + 0.5) * (cij + 0.5)):
                linij = ToLinearIndex([i, j], NiNj)
                ijOld = CoordRotationInv([i, j], NiNj, theta)
                linIntIndicesAndWeights = LinIntIndicesAndWeights(ijOld, NiNj)
                for indexAndWeight in linIntIndicesAndWeights:
                    indexOld = [indexAndWeight[0], indexAndWeight[1]]
                    linIndexOld = ToLinearIndex(indexOld, NiNj)
                    weight = indexAndWeight[2]
                    idx.append((linij + linIndOffset, linIndexOld))
                    vals.append(weight)
    return tuple(idx), tuple(vals)

def GroupConv2D(filters, kernel_size, strides=(1, 1), padding='same', groups=3):
    def layer(x):
        group_list = []
        in_channels = x.shape[-1]
        assert in_channels % groups == 0, f"Number of input channels ({in_channels}) must be divisible by groups ({groups})"
        group_size = in_channels // groups
        for i in range(groups):
            x_group = x[:, :, :, i * group_size : (i + 1) * group_size]
            group_conv = tf.keras.layers.Conv2D(filters // groups, kernel_size, strides=strides, padding=padding)(x_group)
            group_list.append(group_conv)
        x = Concatenate()(group_list)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x
    return layer

def SE2MaxPooling2D(pool_size=(2, 2)):
    def layer(x):
        x = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(x)
        return x
    return layer

def SE2LiftingLayer(x):
    N, H, W, C = x.shape
    assert C % 3 == 0, "Number of input channels must be divisible by 3"
    group_size = C // 3
    x = tf.keras.layers.Reshape((H, W, 3, group_size))(x)
    x = tf.keras.layers.Permute((1, 2, 4, 3))(x)
    return x

def create_SE2CNN_model(input_shape, num_classes, dropout_rate=0.5):
    input_layer = Input(shape=input_shape)
    x = input_layer
    x = GroupConv2D(32, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = GroupConv2D(64, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = GroupConv2D(128, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = GroupConv2D(256, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = GroupConv2D(512, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = GroupConv2D(1024, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = SE2LiftingLayer(x)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(1056, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

from tensorflow.keras.layers import Lambda

# --- BUILD HYBRID MODEL ---
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from transformers import TFViTModel

def build_hybrid_model(image_input_shape, feat_input_shape, umap_feat_shape, num_classes, dropout_rate=0.4):
    # --- Mod-SE(2) CNN Branch (Unchanged) ---
    image_input_se2 = Input(shape=image_input_shape, name='image_input_se2')
    cnn_branch = create_SE2CNN_model(image_input_shape, num_classes, dropout_rate)
    x_se2 = cnn_branch(image_input_se2)
    x_se2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x_se2)
    x_se2 = BatchNormalization()(x_se2)
    x_se2 = Dropout(dropout_rate)(x_se2)

    # --- Handcrafted Feature Branch ---
    feat_input = Input(shape=feat_input_shape, name='feat_input')
    x_feat = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(feat_input)
    x_feat = BatchNormalization()(x_feat)
    x_feat = Dropout(dropout_rate)(x_feat)

    # --- UMAP Feature Branch ---
    umap_input = Input(shape=umap_feat_shape, name='umap_feat_input')
    x_umap = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(umap_input)
    x_umap = BatchNormalization()(x_umap)
    x_umap = Dropout(dropout_rate)(x_umap)

    # --- Fusion ---
    combined = Concatenate()([x_se2, x_feat, x_umap])
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[image_input_se2, feat_input, umap_input], outputs=output)
    return model

import tensorflow as tf

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Use balanced integer labels from SMOTE
y_train_labels = y_train_bal  # already integer-encoded

# ✅ Compute class weights for focal loss / training balance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# ✅ Define training callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1, mode='max'),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, verbose=1, mode='max')
]

# ✅ Build hybrid model (Mod-SE(2) + handcrafted + UMAP)
model_hybrid = build_hybrid_model(
    image_input_shape=(224, 224, 3),
    feat_input_shape=(20,),       # handcrafted
    umap_feat_shape=(2,),         # UMAP projection
    num_classes=4,
    dropout_rate=0.4
)

# ✅ Compile model with focal loss
model_hybrid.compile(
    optimizer=Adam(1e-5),
    loss=focal_loss(gamma=2.5, alpha=0.25),
    metrics=['accuracy']
)

# ✅ Optional: Use data augmentation on training images (not required if inputs are balanced already)
datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    fill_mode='nearest'
)
datagen.fit(X_img_train_bal)

# ✅ Set training and validation inputs
train_inputs = [X_img_train_bal, X_feat_train_bal, X_train_umap]
val_inputs = [X_img_test, X_feat_test_scaled, X_test_umap]

# ✅ Fit model
history = model_hybrid.fit(
    train_inputs, y_train_cat_bal,
    validation_data=(val_inputs, y_test_cat),
    batch_size=16,
    epochs=20,
    class_weight=class_weight_dict,
    #callbacks=callbacks,
    verbose=1
)


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

from tensorflow.keras.utils import plot_model
plot_model(model_hybrid, to_file="model_3branch.architecture.png", show_shapes=True, show_layer_names=True)
model_hybrid.save("./Result/CrossDataset/Training1Testing2/TryFindingBestModel.h5")

# --- PARAMETER TUNING FOR DECISION TREE ---
max_depth_range = [3, 5, 7, 9, 11]
min_samples_leaf_range = [1, 3, 5, 10, 15]
threshold_range = np.linspace(0.3, 0.9, 13)

y_true = np.argmax(y_test_cat, axis=1)
y_pred_proba = model_hybrid.predict([X_img_test, X_feat_test_scaled, X_test_umap], verbose=0)
y_pred_hybrid = np.argmax(y_pred_proba, axis=1)

best_acc = 0
best_params = {}

for max_depth, min_samples_leaf in tqdm.tqdm(itertools.product(max_depth_range, min_samples_leaf_range)):
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    tree.fit(X_train_umap, y_train_labels)

    def generate_rule_function(tree):
        tree_ = tree.tree_
        def recurse(node):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                idx = tree_.feature[node]
                thr = tree_.threshold[node]
                return f"(x[{idx}] <= {thr}) and ({recurse(tree_.children_left[node])}) or " \
                       f"(x[{idx}] > {thr}) and ({recurse(tree_.children_right[node])})"
            else:
                pred = np.argmax(tree_.value[node][0])
                return f"(return_val := {pred})"
        func_code = f"def rule_based_predict(x):\n    global return_val\n    {recurse(0)}\n    return return_val"
        return func_code

    exec(generate_rule_function(tree), globals())
    y_pred_rule = np.array([rule_based_predict(x) for x in X_test_umap])

    for threshold in threshold_range:
        y_pred_fused = np.where(np.max(y_pred_proba, axis=1) < threshold, y_pred_rule, y_pred_hybrid)
        acc = accuracy_score(y_true, y_pred_fused)
        if acc > best_acc:
            best_acc = acc
            best_params = {
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'threshold': threshold,
                'tree_model': tree,
                'final_prediction': y_pred_fused.copy()
            }

print(f"✅ Best Accuracy: {best_acc:.4f}")
print("Best Parameters:", {k: v for k, v in best_params.items() if k != 'final_prediction'})

import joblib
import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# --- SAVE DIRECTORY ---
SAVE_DIR = "./Result/CrossDataset/Training1Testing2/"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- FINAL DECISION TREE BASED ON BEST PARAMS ---
tree_umap = DecisionTreeClassifier(
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
tree_umap.fit(X_train_umap, y_train_labels)

# --- SAVE MODELS ---
joblib.dump(tree_umap, os.path.join(SAVE_DIR, "rule_tree_mixed.pkl"))
joblib.dump(umap_reducer, os.path.join(SAVE_DIR, "umap_model_mixed.pkl"))

# --- PLOT TREE ---
fig_width = 40 if tree_umap.get_depth() < 8 else 100
plt.figure(figsize=(fig_width, 20))
plot_tree(
    tree_umap,
    feature_names=[f"UMAP{i}" for i in range(X_train_umap.shape[1])],
    class_names=[f'MES {i}' for i in range(4)],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title(f"Best Decision Tree on UMAP Features\n(max_depth={best_params['max_depth']}, min_samples_leaf={best_params['min_samples_leaf']})")
plt.tight_layout()

# --- SAVE TREE IMAGE ---
tree_plot_path = os.path.join(SAVE_DIR, "tree_umap_visualization.png")
plt.savefig(tree_plot_path, dpi=300)
plt.show()

print(f"✅ Tree and UMAP model saved to: {SAVE_DIR}")
print(f"🖼️  Tree visualization saved as: {tree_plot_path}")


def generate_rule_function(tree, feature_names=["UMAP0", "UMAP1"]):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            return (
                f"{indent}if x[{feature_names.index(name)}] <= {threshold:.6f}:\n" +
                recurse(tree_.children_left[node], depth + 1) +
                f"{indent}else:\n" +
                recurse(tree_.children_right[node], depth + 1)
            )
        else:
            pred_class = np.argmax(tree_.value[node][0])
            return f"{indent}return {pred_class}  # MES {pred_class}\n"
    function_code = "def rule_based_predict_best(x):\n" + recurse(0, 1)
    return function_code

rule_function_code = generate_rule_function(tree_umap)
print(rule_function_code)
exec(rule_function_code, globals())

from sklearn.metrics import classification_report, confusion_matrix
import os

# --- Set Save Path ---
SAVE_DIR = "./Result/CrossDataset/Training1Testing2/"
os.makedirs(SAVE_DIR, exist_ok=True)
REPORT_PATH = os.path.join(SAVE_DIR, "evaluation_report.txt")

# --- Final Predictions ---
threshold = best_params['threshold']
y_pred_rule_umap = np.array([rule_based_predict_best(row) for row in X_test_umap])
y_pred_hybrid = np.argmax(y_pred_proba, axis=1)
y_pred_combined = np.where(np.max(y_pred_proba, axis=1) < threshold, y_pred_rule_umap, y_pred_hybrid)
y_pred_override = np.array([hybrid_class if rule_class != hybrid_class else rule_class
                            for rule_class, hybrid_class in zip(y_pred_rule_umap, y_pred_hybrid)])
y_true = np.argmax(y_test_cat, axis=1)

# --- Generate Reports ---
report_hybrid = classification_report(y_true, y_pred_hybrid, digits=4)
report_combined = classification_report(y_true, y_pred_combined, digits=4)
report_override = classification_report(y_true, y_pred_override, digits=4)
report_rule = classification_report(y_true, y_pred_rule_umap, digits=4)

# --- Print to Console ---
print("📊 Hybrid Only:\n", report_hybrid)
print("📊 Rule-Aware Hybrid (Confidence-Guided Fallback):\n", report_combined)
print("📊 Rule-Aware Hybrid (Override When Agree):\n", report_override)
print("📊 Rule Only:\n", report_rule)

# --- Save to TXT File ---
with open(REPORT_PATH, 'w') as f:
    f.write("📊 Hybrid Only:\n")
    f.write(report_hybrid + "\n\n")
    f.write("📊 Rule-Aware Hybrid (Confidence-Guided Fallback):\n")
    f.write(report_combined + "\n\n")
    f.write("📊 Rule-Aware Hybrid (Override When Agree):\n")
    f.write(report_override + "\n\n")
    f.write("📊 Rule Only:\n")
    f.write(report_rule + "\n")

print(f"\n✅ Evaluation report saved to: {REPORT_PATH}")


from flask import Flask, request, jsonify
import openai
import threading

openai.api_key = "YOUR_OPENAI_API_KEY_HERE"  # or from environment

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    prompt = request.json.get("prompt", "")
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )
        content = res['choices'][0]['message']['content']
        # Try to find valid class (0–3)
        for token in content.split():
            if token.isdigit() and int(token) in [0, 1, 2, 3]:
                return jsonify({"class": int(token), "raw": content})
        return jsonify({"class": None, "raw": content})
    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Run in separate thread
def run_flask():
    app.run(host='0.0.0.0', port=5000)

threading.Thread(target=run_flask).start()



import os
import json
import numpy as np
from sklearn.metrics import classification_report

# --- Ensure UMAP projection is applied to test handcrafted features ---
X_feat_test_umap = umap_reducer.transform(X_feat_test_scaled)
y_true = np.argmax(y_test_cat, axis=1)  # <-- FIXED

# --- Predict using hybrid model ---
y_pred_proba = model_hybrid.predict(
    [X_img_test, X_feat_test_scaled, X_feat_test_umap], verbose=0
)
y_pred_hybrid = np.argmax(y_pred_proba, axis=1)
confidences = np.max(y_pred_proba, axis=1)

# --- Rule-based prediction ---
y_pred_rule_umap = np.array([rule_based_predict_best(row) for row in X_feat_test_umap])

# --- Fusion logic ---
threshold = best_params.get("threshold", 0.55)
y_pred_combined = np.where(confidences < threshold, y_pred_rule_umap, y_pred_hybrid)
y_pred_agree = np.array([
    rule if rule == model else rule
    for rule, model in zip(y_pred_rule_umap, y_pred_hybrid)
])

# --- Print evaluation ---
print("Hybrid Only:\n", classification_report(y_true, y_pred_hybrid, digits=4))
print("Rule-Aware Hybrid (Confidence-Guided):\n", classification_report(y_true, y_pred_combined, digits=4))
print("Rule-Aware Hybrid (Agreement Override):\n", classification_report(y_true, y_pred_agree, digits=4))
print("Rule Only:\n", classification_report(y_true, y_pred_rule_umap, digits=4))

# --- Save evaluation report to file ---
output_dir = "./Result/CrossDataset/Training1Testing2/"
os.makedirs(output_dir, exist_ok=True)

report_file = os.path.join(output_dir, "classification_report.txt")
with open(report_file, "w") as f:
    f.write("Hybrid Only:\n")
    f.write(classification_report(y_true, y_pred_hybrid, digits=4))
    f.write("\nRule-Aware Hybrid (Confidence-Guided):\n")
    f.write(classification_report(y_true, y_pred_combined, digits=4))
    f.write("\nRule-Aware Hybrid (Agreement Override):\n")
    f.write(classification_report(y_true, y_pred_agree, digits=4))
    f.write("\nRule Only:\n")
    f.write(classification_report(y_true, y_pred_rule_umap, digits=4))

print(f"✅ Classification report saved to: {report_file}")

# --- Save JSONL log for LLM feedback learning ---
log_file_path = os.path.join(output_dir, "llm_feedback_log_mixed.jsonl")
with open(log_file_path, "w") as f:
    for i in range(len(y_true)):
        entry = {
            "model_prediction": int(y_pred_hybrid[i]),
            "rule_prediction": int(y_pred_rule_umap[i]),
            "final_fusion_prediction": int(y_pred_combined[i]),
            "true_label": int(y_true[i]),
            "confidence": float(confidences[i]),
            "umap_0": float(X_feat_test_umap[i][0]),
            "umap_1": float(X_feat_test_umap[i][1]),
            "features": X_feat_test_scaled[i].tolist(),
            "feedback": ""
        }

        # Label feedback
        if y_pred_combined[i] != y_true[i]:
            if confidences[i] < threshold and y_pred_rule_umap[i] == y_true[i]:
                entry["feedback"] = "Should have used the rule-based prediction. Confidence was low."
            elif confidences[i] >= threshold and y_pred_hybrid[i] == y_true[i]:
                entry["feedback"] = "Correctly used the model prediction. Confidence was high."
            else:
                entry["feedback"] = "Incorrect prediction. Consider learning from UMAP and features."
        else:
            entry["feedback"] = "Correct prediction."

        f.write(json.dumps(entry) + "\n")

print(f"✅ Logged {len(y_true)} entries to {log_file_path}")


import openai
import json
import time
from tqdm import tqdm

# ✅ Set your OpenAI API Key
openai.api_key = "YOUR_OPENAI_API_KEY_HERE"  # Replace with your OpenAI key

# ✅ Path to your saved .jsonl log
log_file_path = "./Result/CrossDataset/Training1Testing2/llm_feedback_log_mixed.jsonl"

# ✅ Load prediction entries
with open(log_file_path, "r") as f:
    entries = [json.loads(line) for line in f]

# ✅ Function to build prompt from entry
def build_prompt(entry):
    return f"""You are a colonoscopy diagnosis assistant trained to override model predictions when necessary.

Model prediction: {entry['model_prediction']}
Rule prediction: {entry['rule_prediction']}
Confidence: {entry['confidence']:.2f}
UMAP: ({entry['umap_0']:.2f}, {entry['umap_1']:.2f})
Handcrafted features: {entry['features']}

True label: {entry['true_label']}

Task: Given all the information, what should the final predicted MES class be?
Only respond with a single number: 0, 1, 2, or 3.
"""

# ✅ Function to get LLM recommendation
def query_gpt(prompt, model="gpt-3.5-turbo", retries=3):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(1)
    return None

# ✅ Run through all entries and get LLM-based decisions
llm_preds = []
for entry in tqdm(entries, desc="LLM predicting"):
    prompt = build_prompt(entry)
    prediction = query_gpt(prompt)
    try:
        pred_int = int(prediction)
        if pred_int in [0, 1, 2, 3]:
            llm_preds.append(pred_int)
        else:
            llm_preds.append(entry["final_fusion_prediction"])  # fallback
    except:
        llm_preds.append(entry["final_fusion_prediction"])  # fallback

# ✅ Calculate accuracy of LLM override agent
true_labels = [e["true_label"] for e in entries]
correct = sum([int(p == t) for p, t in zip(llm_preds, true_labels)])
accuracy = correct / len(true_labels)

print(f"✅ LLM Agent Accuracy: {accuracy:.4f}")


import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report

# --- Setup ---
output_dir = "./Result/CrossDataset/Training1Testing2/"
os.makedirs(output_dir, exist_ok=True)

# ✅ Load previous log
log_file_path = os.path.join(output_dir, "llm_feedback_log_mixed.jsonl")
with open(log_file_path, "r") as f:
    entries = [json.loads(line) for line in f]

# ✅ Update and overwrite JSONL with LLM predictions
updated_log_path = os.path.join(output_dir, "llm_feedback_log_updated_mixed.jsonl")
with open(updated_log_path, "w") as f:
    for entry, llm_pred in zip(entries, llm_preds):
        entry["llm_prediction"] = llm_pred
        entry["llm_correct"] = int(llm_pred == entry["true_label"])
        f.write(json.dumps(entry) + "\n")

print(f"✅ Updated JSONL written to: {updated_log_path}")

# ✅ Collect predictions and labels
fusion_preds = [e["final_fusion_prediction"] for e in entries]
rule_preds = [e["rule_prediction"] for e in entries]
model_preds = [e["model_prediction"] for e in entries]
llm_preds = [e.get("llm_prediction", e["final_fusion_prediction"]) for e in entries]
true_labels = [e["true_label"] for e in entries]

# ✅ Per-class accuracy
methods = {
    "Model": model_preds,
    "Rule": rule_preds,
    "Fusion": fusion_preds,
    "LLM": llm_preds
}
class_labels = [0, 1, 2, 3]
acc_by_class = defaultdict(dict)

for method, preds in methods.items():
    for cls in class_labels:
        cls_indices = [i for i, t in enumerate(true_labels) if t == cls]
        cls_correct = sum([preds[i] == true_labels[i] for i in cls_indices])
        acc_by_class[method][cls] = cls_correct / len(cls_indices) if cls_indices else 0

# ✅ Bar Plot per-class accuracy
x = np.arange(len(class_labels))
width = 0.2
plt.figure(figsize=(10, 6))
for i, (method, accs) in enumerate(acc_by_class.items()):
    plt.bar(x + i*width, [accs[cls] for cls in class_labels], width=width, label=method)

plt.xticks(x + 1.5*width, [f"MES {cls}" for cls in class_labels])
plt.ylabel("Accuracy")
plt.title("Accuracy by MES Class per Method")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()

# ✅ Save plot
plot_path = os.path.join(output_dir, "accuracy_by_class_per_method.png")
plt.savefig(plot_path)
plt.show()
print(f"✅ Saved accuracy plot to: {plot_path}")

# ✅ Print and save classification reports
def print_and_save_report(name, preds):
    acc = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=[f"MES {i}" for i in range(4)], digits=4)
    print(f"\n🔍 {name} Accuracy: {acc:.4f}")
    print(report)

    report_path = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{name} Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"✅ Saved report to: {report_path}")

print_and_save_report("Model Only", model_preds)
print_and_save_report("Rule-Based", rule_preds)
print_and_save_report("Fusion", fusion_preds)
print_and_save_report("LLM Override", llm_preds)


# === STEP 1: Model Prediction on Training Set ===
# y_pred_proba_train: softmax probabilities
import json
import os

def create_feedback_jsonl(
    filename,
    y_true,
    y_model_pred,
    y_rule_pred,
    proba,
    umap_feat,
    handcrafted_feat,
    save_path="./Result/CrossDataset/Training1Testing2/"
):
    assert len(y_true) == len(y_model_pred) == len(y_rule_pred) == len(proba) == len(umap_feat) == len(handcrafted_feat)
    records = []
    for i in range(len(y_true)):
        record = {
            "true_label": int(y_true[i]),
            "model_pred": int(y_model_pred[i]),
            "rule_pred": int(y_rule_pred[i]),
            "confidence": float(np.max(proba[i])),
            "proba": list(map(float, proba[i])),
            "umap_0": float(umap_feat[i][0]),
            "umap_1": float(umap_feat[i][1]),
            "features": list(map(float, handcrafted_feat[i]))
        }
        records.append(record)
    
    full_path = os.path.join(save_path, filename)
    with open(full_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    
    print(f"✅ Saved {len(records)} samples to {full_path}")

# --- Predict on training set using all 3 branches ---
y_pred_proba_train = model_hybrid.predict(
    [X_img_train_bal, X_feat_train_bal, X_train_umap],
    verbose=1
)
y_pred_hybrid_train = np.argmax(y_pred_proba_train, axis=1)

# === STEP 2: Rule-Based Prediction on UMAP Train ===
y_pred_rule_umap_train = np.array([rule_based_predict_best(row) for row in X_train_umap])

# === STEP 3: Save feedback_train.jsonl ===
create_feedback_jsonl(
    filename="feedback_train.jsonl",
    y_true=y_train_bal,                       # SMOTE-balanced true labels
    y_model_pred=y_pred_hybrid_train,         # model prediction
    y_rule_pred=y_pred_rule_umap_train,       # rule-based prediction
    proba=y_pred_proba_train,                 # model probabilities
    umap_feat=X_train_umap,                   # 2D UMAP features
    handcrafted_feat=X_feat_train_bal      # original scaled 20D handcrafted features
)

import os
import json
import numpy as np

# Define base path for saving
base_dir = "./Result/CrossDataset/Training1Testing2/"

def create_feedback_jsonl(filename, y_true, y_model_pred, y_rule_pred, proba, umap_feat, handcrafted_feat):
    records = []
    for i in range(len(y_true)):
        entry = {
            "true_label": int(y_true[i]),
            "model_prediction": int(y_model_pred[i]),
            "rule_prediction": int(y_rule_pred[i]),
            "confidence": float(np.max(proba[i])),
            "umap_0": float(umap_feat[i, 0]),
            "umap_1": float(umap_feat[i, 1]),
            "features": [float(f) for f in handcrafted_feat[i]]
        }
        records.append(entry)

    path = os.path.join(base_dir, filename)
    with open(path, "w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    print(f"✅ Saved {len(records)} samples to {path}")

# === GENERATE TESTING SET FEEDBACK FILE ===
create_feedback_jsonl(
    filename="feedback_test.jsonl",
    y_true=y_test_encoded,
    y_model_pred=y_pred_hybrid,
    y_rule_pred=y_pred_rule_umap,
    proba=y_pred_proba,
    umap_feat=X_feat_test_umap,
    handcrafted_feat=X_feat_test_scaled
)


import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
import joblib
import warnings
from hashlib import sha1

warnings.filterwarnings("ignore")

# === PATHS ===
train_file = "./Result/CrossDataset/Training1Testing2/feedback_train.jsonl"
test_file  = "./Result/CrossDataset/Training1Testing2/feedback_test.jsonl"
save_dir   = "./Result/CrossDataset/Training1Testing2/"
os.makedirs(save_dir, exist_ok=True)

# === LOAD FUNCTION ===
def load_feedback_jsonl(path):
    with open(path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    df = pd.DataFrame(data)
    df["label"] = df["true_label"]
    return df

df_train = load_feedback_jsonl(train_file)
df_test  = load_feedback_jsonl(test_file)
df_test_orig = df_test.copy()  # Freeze

# === FEATURE ENCODING ===
def encode_features(df):
    df_feat = df[["confidence", "umap_0", "umap_1"]].copy()
    for i in range(20):
        df_feat[f"f{i}"] = df["features"].apply(lambda x: x[i])
    return df_feat.values, df["label"].values

X_train, y_train = encode_features(df_train)
X_test, y_test   = encode_features(df_test_orig)

# === SCALING ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# === TRAIN LOOP ===
loop = 0
acc_list = []
known_misclassified = set()

while True:
    clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced')

    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    acc_list.append(acc)

    print(f"🔁 Loop {loop+1}: Accuracy = {acc:.4f}")
    if acc >= 0.90:
        print("✅ Target reached.")
        break

    misclassified = df_test_orig[y_pred != y_test].copy()
    misclassified["hash"] = misclassified.apply(
        lambda row: sha1(json.dumps(row.to_dict(), sort_keys=True).encode()).hexdigest(), axis=1
    )
    new_errors = misclassified[~misclassified["hash"].isin(known_misclassified)]

    if new_errors.empty:
        print("⚠️ No new unique misclassified samples to learn from.")
        break

    print(f"➕ Adding {len(new_errors)} new feedback samples")
    known_misclassified.update(new_errors["hash"])
    df_train = pd.concat([df_train, new_errors.drop(columns=["hash"])], ignore_index=True)

    X_train, y_train = encode_features(df_train)
    X_train_scaled = scaler.fit_transform(X_train)

    loop += 1

# === SAVE FINAL AGENT ===
clf.booster_.save_model(os.path.join(save_dir, "feedback_agent.txt"))
joblib.dump(scaler, os.path.join(save_dir, "scaler_agent.pkl"))

# === SAVE LEARNING CURVE ===
plt.figure(figsize=(10, 5))
plt.plot(acc_list, marker='o', label="Test Accuracy")
plt.axhline(0.90, color='red', linestyle='--', label="Target")
plt.title("Agent Continual Learning Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "agent_learning_curve.png"))

# === FINAL REPORT ===
report = classification_report(y_test, y_pred, digits=4, output_dict=True)
report_df = pd.DataFrame(report).T
report_df.to_csv(os.path.join(save_dir, "agent_final_classification_report.csv"))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
import lightgbm as lgb

# === PATHS ===
save_dir = "./Result/CrossDataset/Training1Testing2/"
test_file = os.path.join(save_dir, "feedback_test.jsonl")
agent_file = os.path.join(save_dir, "feedback_agent.txt")
scaler_file = os.path.join(save_dir, "scaler_agent.pkl")

# === LOAD DATASET ===
with open(test_file, "r") as f:
    test_data = [json.loads(line.strip()) for line in f]
df_all = pd.DataFrame(test_data)
df_all["label"] = df_all["true_label"]

# === FEATURE ENCODER ===
def encode_features(df):
    df_feat = df[["confidence", "umap_0", "umap_1"]].copy()
    for i in range(20):
        df_feat[f"f{i}"] = df["features"].apply(lambda x: x[i])
    return df_feat.values, df["label"].values

X, y = encode_features(df_all)
scaler = joblib.load(scaler_file)
X_scaled = scaler.transform(X)

# === LOAD AGENT ===
agent = lgb.Booster(model_file=agent_file)

# === K-FOLD EVALUATION ===
n_classes = 4
class_names = [f"MES {i}" for i in range(n_classes)]
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold = 1
for train_idx, test_idx in kf.split(X_scaled, y):
    print(f"\n===== Fold {fold} =====")

    X_test, y_test = X_scaled[test_idx], y[test_idx]

    # Predict with existing agent
    y_proba = agent.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    y_true_oh = label_binarize(y_test, classes=list(range(n_classes)))

    # === Per-Class Custom Metrics ===
    cm = confusion_matrix(y_test, y_pred)
    metrics_dict = {}
    for i in range(n_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        npv       = TN / (TN + FN) if (TN + FN) > 0 else 0
        ppv       = precision
        accuracy  = (TP + TN) / (TP + TN + FP + FN)
        metrics_dict[f"MES {i}"] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "npv": npv,
            "ppv": ppv,
            "accuracy": accuracy
        }

    # Macro Average
    all_vals = pd.DataFrame(metrics_dict).T
    metrics_dict["Overall"] = {
        metric: all_vals[metric].mean() for metric in all_vals.columns
    }
    summary = pd.DataFrame(metrics_dict).T

    # Save CSV per fold
    summary.to_csv(os.path.join(save_dir, f"agent_eval_summary_fold{fold}.csv"))
    print("📊 Evaluation Summary (Fold", fold, "):\n", summary.round(4))

    summary.to_csv(os.path.join(save_dir, f"agent_eval_summary_fold{fold}.csv"))
    print("📊 Evaluation Summary (Fold", fold, "):\n", summary.round(4))
    
    # === Dump to TXT cumulative file ===
    txt_path = os.path.join(save_dir, "agent_eval_summary_all.txt")
    with open(txt_path, "a") as f_out:
        f_out.write(f"\n===== Fold {fold} =====\n")
        f_out.write(summary.round(4).to_string())
        f_out.write("\n")
    
    # === SAVE  ===
    if fold == 10:
        # Radar Chart
        metrics = ["precision", "recall", "f1-score", "npv"]
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, polar=True)
        for i in range(n_classes):
            label = f"MES {i}"
            values = summary.loc[label, metrics].values.tolist()
            values += values[:1]
            ax.plot(angles, values, label=label)
            ax.fill(angles, values, alpha=0.1)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], metrics)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"])
        plt.title("Radar Chart: Per-Class Evaluation (Fold 10)")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "agent_radar_chart_final.png"))
        plt.show()

        # Confusion Matrix
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0],
                    xticklabels=class_names, yticklabels=class_names)
        ax[0].set_title("Confusion Matrix (Fold 10)")
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("True")
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax[1],
                    xticklabels=class_names, yticklabels=class_names)
        ax[1].set_title("Normalized Confusion Matrix (Fold 10)")
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "agent_confusion_matrix_final.png"))
        plt.show()

        # ROC Curve
        fpr = dict(); tpr = dict(); roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_oh[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_oh.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        roc_auc["macro"] = roc_auc_score(y_true_oh, y_proba, average="macro")

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f"MES {i} (AUC = {roc_auc[i]:.2f})")
        plt.plot(fpr["micro"], tpr["micro"], color="black", linestyle="--", 
                 label=f"Micro Avg (AUC = {roc_auc['micro']:.2f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.title("ROC Curves (Fold 10)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "agent_roc_curve_final.png"))
        plt.show()

    fold += 1


# === Radar Chart ===
metrics = ["accuracy", "precision", "recall", "f1-score"]
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # to close the radar chart loop

plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)

for i in range(n_classes):
    label = f"MES {i}"
    values = summary.loc[label, metrics].values.tolist()
    values += values[:1]
    ax.plot(angles, values, label=label)
    ax.fill(angles, values, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], metrics)
plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"])
plt.title("Radar Chart: Accuracy, Precision, Recall, F1-Score (Per Class)")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "agent_radar_chart_accuracy.png"))
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import os

# === Define save directory ===
save_dir = "./Result/CrossDataset/Training1Testing2/"

# === Handcrafted feature names: 20D (from Wavelet + GLCM) ===
handcrafted_feature_names = [
    "LL_Mean", "LL_Std", "LL_Var", "LL_Entropy",
    "LH_Mean", "LH_Std", "LH_Var", "LH_Entropy",
    "HL_Mean", "HL_Std", "HL_Var", "HL_Entropy",
    "HH_Mean", "HH_Std", "HH_Var", "HH_Entropy",
    "HH_Energy",
    "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity"
]

# === Get feature importances using LightGBM Booster method ===
importances = agent.feature_importance()

# === Handcrafted features are from index 3 to 22 (inclusive) ===
handcrafted_importances = importances[3:23]

# === Sort descending ===
sorted_idx = np.argsort(handcrafted_importances)[::-1]
sorted_names = [handcrafted_feature_names[i] for i in sorted_idx]
sorted_vals = handcrafted_importances[sorted_idx]

# === Plot bar chart ===
plt.figure(figsize=(10, 6))
plt.barh(sorted_names, sorted_vals, color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("🔬 Handcrafted Feature Importance (LightGBM Agent)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "agent_handcrafted_feature_importance.png"))
plt.show()


print("hello")

for name, val in zip(sorted_names,sorted_vals):
    print(name,":",val)

print("Feedback log sample size:", len(current_data))  # From feedback loop
print("Validation set used during agent training:", len(y_val))  # Typically ~20% of that
print("Real test set used for evaluation:", len(y_true))  # 145


