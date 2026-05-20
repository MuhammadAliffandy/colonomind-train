import cv2
import numpy as np
import pywt
import scipy.stats
from skimage.feature import graycomatrix, graycoprops
from src.config import WAVELET

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
