import os
import cv2
import numpy as np
from src.config import IMG_SIZE
from src.features import extract_wavelet_stats, extract_glcm_features_extended

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
