from joblib import Parallel, delayed
import os
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

def load_all_images(dir_list, dataset_name):
    all_imgs, all_feats, all_labels, all_paths = [], [], [], []
    folder_names = DATASET_CLASS_FOLDERS.get(dataset_name, CLASS_NAMES)
    
    tasks = []
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
                tasks.append((img_path, folder_cls))
                
    print(f"  Memproses {len(tasks)} gambar secara paralel menggunakan semua core CPU...")
    results = Parallel(n_jobs=16, batch_size=32, verbose=10)(delayed(process_single_image)(p, c) for p, c in tasks)
    
    for r in results:
        if r is not None:
            all_imgs.append(r[0])
            all_feats.append(r[1])
            all_labels.append(r[2])
            all_paths.append(r[3])
            
    return all_imgs, all_feats, all_labels, all_paths

def load_tmc_ucm(tmc_root, split_filter=None):
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
        existing_images = set(os.listdir(images_dir))
        
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
                
    print(f"  Memproses {len(tasks)} gambar TMC-UCM secara paralel...")
    results = Parallel(n_jobs=16, batch_size=32, verbose=10)(delayed(process_single_image)(p, c) for p, c in tasks)
    
    for r in results:
        if r is not None:
            all_imgs.append(r[0])
            all_feats.append(r[1])
            all_labels.append(r[2])
            all_paths.append(r[3])
            
    return all_imgs, all_feats, all_labels, all_paths
