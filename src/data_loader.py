import os
import sys
import cv2
import numpy as np
import joblib
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import IMG_SIZE
from src.features import extract_wavelet_stats, extract_glcm_features_extended

# ====================================================
# Cache helpers — skip re-extracting features if
# the dataset has not changed since last run.
# Cache is stored next to the dataset folder as a .pkl
# ====================================================

def _get_cache_path(folder_path):
    """Create a unique cache filename based on dataset path."""
    key = hashlib.md5(folder_path.encode()).hexdigest()[:8]
    cache_dir = os.path.join(os.path.dirname(folder_path), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"features_{key}.pkl")

def _load_cache(folder_path):
    cache_path = _get_cache_path(folder_path)
    if os.path.exists(cache_path):
        print(f"  ⚡ Cache found! Loading from {cache_path} (skipping feature extraction)...")
        return joblib.load(cache_path)
    return None

def _save_cache(folder_path, data):
    cache_path = _get_cache_path(folder_path)
    joblib.dump(data, cache_path)
    print(f"  💾 Cache saved to {cache_path}")

# ====================================================
# Single image processing worker
# ====================================================

def _process_image(args):
    """Process a single image: resize, crop, extract features. Used in parallel pool."""
    img_path, label = args
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped = img[30:430, 200:550]
        resized = cv2.resize(cropped, IMG_SIZE)

        wavelet_feats = extract_wavelet_stats(resized)
        glcm_feats    = extract_glcm_features_extended(resized)
        combined      = wavelet_feats + glcm_feats

        return (resized, combined, label, img_path)
    except Exception as e:
        return None


# ====================================================
# Main loader — parallel + chunked + cached
# ====================================================

def load_dataset(folder_path, num_workers=8, chunk_size=500, use_cache=True):
    """
    Load a dataset folder with parallel processing and disk caching.

    Args:
        folder_path (str): Root folder containing class subfolders.
        num_workers (int): Number of parallel threads for image loading.
        chunk_size (int): Number of images processed per chunk (controls memory).
        use_cache (bool): If True, cache features to disk and reuse on next run.

    Returns:
        Tuple: (X_img, X_feat, y_label, img_paths) as numpy arrays.
    """
    # ---- Try loading from cache first ----
    if use_cache:
        cached = _load_cache(folder_path)
        if cached is not None:
            print(f"    -> {len(cached[2])} images loaded from cache")
            return cached

    # ---- Collect all image paths ----
    all_tasks = []
    for label in sorted(os.listdir(folder_path)):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for fname in sorted(os.listdir(label_path)):
            img_path = os.path.join(label_path, fname)
            all_tasks.append((img_path, label))

    print(f"  Found {len(all_tasks)} images. Processing in chunks of {chunk_size} with {num_workers} workers...")

    X_img, X_feat, y_label, img_paths = [], [], [], []

    # ---- Process in chunks to avoid OOM ----
    for chunk_start in range(0, len(all_tasks), chunk_size):
        chunk = all_tasks[chunk_start : chunk_start + chunk_size]
        chunk_end = min(chunk_start + chunk_size, len(all_tasks))
        print(f"  Processing chunk [{chunk_start+1} - {chunk_end}] / {len(all_tasks)}...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_image, task): task for task in chunk}
            for future in tqdm(as_completed(futures), total=len(chunk), desc="  Loading", leave=False):
                result = future.result()
                if result is None:
                    continue
                img, feat, label, path = result
                X_img.append(img)
                X_feat.append(feat)
                y_label.append(label)
                img_paths.append(path)

    X_img_arr    = np.array(X_img,    dtype=np.uint8)
    X_feat_arr   = np.array(X_feat,   dtype=np.float32)
    y_label_arr  = np.array(y_label)
    img_paths_arr = np.array(img_paths)

    print(f"  -> {len(y_label_arr)} images loaded")

    # ---- Save to cache for next run ----
    if use_cache:
        _save_cache(folder_path, (X_img_arr, X_feat_arr, y_label_arr, img_paths_arr))

    return X_img_arr, X_feat_arr, y_label_arr, img_paths_arr
