from joblib import Parallel, delayed
import os
import time
import hashlib
import pickle
import cv2
cv2.setNumThreads(0) # Prevent OpenCV deadlock in multiprocessing
import numpy as np
import pywt
import scipy.stats
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

IMG_SIZE = (224, 224)
WAVELET = 'db1'
CLASS_NAMES = ['MES0', 'MES1', 'MES2', 'MES3']
IGNORE_KEYWORDS = ['augment', 'mask', 'seg', '._', 'crop']

DATASET_CLASS_FOLDERS = {
    'NTUH':    ['MES0', 'MES1', 'MES2', 'MES3'],
    'LIMUC':   ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3'],
    'TMC-UCM': ['MES0', 'MES1', 'MES2', 'MES3']
}

FOLDER_TO_LABEL = {
    'MES0': 'MES0', 'MES1': 'MES1', 'MES2': 'MES2', 'MES3': 'MES3',
    'Mayo 0': 'MES0', 'Mayo 1': 'MES1', 'Mayo 2': 'MES2', 'Mayo 3': 'MES3'
}

# ── Cache directory (saved alongside the script) ──────────────────
_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Dataset_Cache')


def _safe_listdir(path, max_retries=5, delay=10):
    """os.listdir with retry logic for flaky NFS/network drives."""
    for attempt in range(1, max_retries + 1):
        try:
            return os.listdir(path)
        except (OSError, IOError) as e:
            if attempt < max_retries:
                print(f"  ⚠️  listdir error (attempt {attempt}/{max_retries}): {e} — retrying in {delay}s")
                time.sleep(delay)
            else:
                print(f"  ❌ listdir failed after {max_retries} attempts: {e}")
                raise


def _cache_key(identifier: str) -> str:
    """Generate a short hash key from an identifier string."""
    return hashlib.md5(identifier.encode()).hexdigest()[:12]


def _cache_path(cache_dir: str, key: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.pkl")


def _load_cache(cache_dir: str, key: str):
    """Return cached data or None if not found or if cache is empty (corrupt)."""
    path = _cache_path(cache_dir, key)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                # Check if cache returned empty lists (corrupt state)
                if isinstance(data, tuple) and len(data) > 0:
                    if len(data[0]) == 0:
                        print(f"  ⚠️ Cache {path} is empty (0 images). Ignoring.")
                        return None
                print(f"  ✅ Cache hit — loading from {path}")
                return data
        except Exception as e:
            print(f"  ⚠️ Failed to read cache {path}: {e}")
            return None
    return None


def _save_cache(cache_dir: str, key: str, data):
    """Persist data to cache. Non-fatal: if disk write fails, warn and continue."""
    path = _cache_path(cache_dir, key)
    for attempt in range(1, 4):
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  💾 Cache saved to {path}")
            return
        except (OSError, IOError) as e:
            if attempt < 3:
                print(f"  ⚠️  Cache save error (attempt {attempt}/3): {e} — retrying in 10s")
                time.sleep(10)
            else:
                print(f"  ⚠️  Cache save failed after 3 attempts: {e}")
                print(f"  ℹ️  Training will continue using in-memory data (cache won't be available next run)")


# ─────────────────────────────────────────────────────────────────
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

def process_single_image(img_path, folder_cls):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feats = extract_combined_features(img_rgb)
    label = FOLDER_TO_LABEL.get(folder_cls, folder_cls)
    return (img_rgb, feats, label, img_path)


def load_all_images(dir_list, dataset_name, cache_dir=None):
    """
    Load and preprocess all images from dir_list for a given dataset.
    Results are cached to disk so subsequent calls are instant.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    # Build a deterministic cache key from the inputs
    cache_id = f"load_all_images__{dataset_name}__{':'.join(sorted(dir_list))}"
    key = _cache_key(cache_id)

    cached = _load_cache(cache_dir, key)
    if cached is not None:
        return cached

    # ── Fresh load ──
    all_imgs, all_feats, all_labels, all_paths = [], [], [], []
    folder_names = DATASET_CLASS_FOLDERS.get(dataset_name, CLASS_NAMES)

    tasks = []
    for dataset_dir in dir_list:
        for folder_cls in folder_names:
            cls_dir = os.path.join(dataset_dir, folder_cls)
            if not os.path.exists(cls_dir):
                print(f'  ⚠️ Folder tidak ditemukan: {cls_dir}')
                continue
            for img_name in _safe_listdir(cls_dir):
                if any(k in img_name.lower() for k in IGNORE_KEYWORDS):
                    continue
                img_path = os.path.join(cls_dir, img_name)
                tasks.append((img_path, folder_cls))

    print(f"  Memproses {len(tasks)} gambar secara paralel menggunakan thread CPU...")
    results = Parallel(n_jobs=16, batch_size=32, verbose=10, backend="threading")(delayed(process_single_image)(p, c) for p, c in tasks)

    for r in results:
        if r is not None:
            all_imgs.append(r[0])
            all_feats.append(r[1])
            all_labels.append(r[2])
            all_paths.append(r[3])

    result = (all_imgs, all_feats, all_labels, all_paths)
    _save_cache(cache_dir, key, result)
    return result


def load_tmc_ucm(tmc_root, split_filter=None, cache_dir=None):
    """
    Load and preprocess TMC-UCM images.
    Results are cached to disk so subsequent calls are instant.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    cache_id = f"load_tmc_ucm__{tmc_root}__{split_filter}"
    key = _cache_key(cache_id)

    cached = _load_cache(cache_dir, key)
    if cached is not None:
        return cached

    # ── Fresh load ──
    all_imgs, all_feats, all_labels, all_paths = [], [], [], []
    INT_TO_LABEL = {0: 'MES0', 1: 'MES1', 2: 'MES2', 3: 'MES3'}
    images_dir = os.path.join(tmc_root, 'images')

    txt_files = []
    if split_filter is None or split_filter == 'Train':
        txt_files.append('train.txt')
    if split_filter is None or split_filter == 'Test':
        txt_files.append('test.txt')

    tasks = []

    # Preload existing files to avoid slow os.path.exists calls on network drives
    existing_images = set()
    if os.path.exists(images_dir):
        existing_images = set(_safe_listdir(images_dir))

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

                if fname not in existing_images:
                    continue
                img_path = os.path.join(images_dir, fname)
                if any(k in fname.lower() for k in IGNORE_KEYWORDS):
                    continue

                folder_cls_str = INT_TO_LABEL.get(label_int, f'MES{label_int}')
                tasks.append((img_path, folder_cls_str))

    print(f"  Memproses {len(tasks)} gambar TMC-UCM secara paralel menggunakan thread CPU...")
    results = Parallel(n_jobs=16, batch_size=32, verbose=10, backend="threading")(delayed(process_single_image)(p, c) for p, c in tasks)

    for r in results:
        if r is not None:
            all_imgs.append(r[0])
            all_feats.append(r[1])
            all_labels.append(r[2])
            all_paths.append(r[3])

    result = (all_imgs, all_feats, all_labels, all_paths)
    _save_cache(cache_dir, key, result)
    return result
