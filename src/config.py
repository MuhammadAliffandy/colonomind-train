import os

# === Image & Feature Config ===
IMG_SIZE = (224, 224)
WAVELET = 'db1'

# === Server Paths (NTU DGX) ===
# Project root: ~/Clara/colono_train/
# Dataset root: ~/Clara/new_drive/Dataset_Extracted/Dataset+Code/
DATASET_BASE_DIR = os.path.expanduser("~/Clara/new_drive/Dataset_Extracted/Dataset+Code")

# === Dataset Registry ===
# Maps a short name to the actual folder name inside DATASET_BASE_DIR
DATASETS = {
    "dataset_1": os.path.join(DATASET_BASE_DIR, "MES classification_20250313"),
    "dataset_2": os.path.join(DATASET_BASE_DIR, "MES classification_20250724"),
    "public":    os.path.join(DATASET_BASE_DIR, "MES_Colonoscopy Public Dataset"),
    "mixed":     os.path.join(DATASET_BASE_DIR, "MES Mixed Data"),
}
