#!/usr/bin/env python3
"""
ColonoMind v8 — Export & Packaging Script for DGX Server
=========================================================
Gathers all trained artifacts from ColonoMind v8 (Hierarchical Two-Stage Mod-SE):
  - Stage 1 CNN models (3 seeds)
  - Stage 2 CNN models (3 seeds)
  - Scaler & UMAP dimensionality reduction models
  - LightGBM Super Agent decision trees (Stage 1 & Stage 2)
  - Threshold configurations (v14 if available)
  - Self-contained inference engine code (v8_pipeline.py)

Packages everything into ~/xxdrive/ColonoMind_V8_Package/ and creates a single
ready-to-download ZIP archive: ~/xxdrive/colonomind_v8_package.zip.
"""

import os
import sys
import shutil
import glob
import zipfile
import argparse

def find_first_existing(candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

def main():
    parser = argparse.ArgumentParser(description="Package ColonoMind v8 models to ~/xxdrive")
    parser.add_argument("--dest", default=os.path.expanduser("~/xxdrive"),
                        help="Target directory (default: ~/xxdrive)")
    parser.add_argument("--v8_dir", default=None,
                        help="Explicit directory containing v8 result artifacts (optional)")
    args = parser.parse_args()

    dest_base = os.path.abspath(os.path.expanduser(args.dest))
    package_dir = os.path.join(dest_base, "ColonoMind_V8_Package")
    models_dir = os.path.join(package_dir, "models")

    try:
        os.makedirs(models_dir, exist_ok=True)
    except OSError as e:
        print(f"\n⚠️ PERINGATAN: Gagal membuat folder di '{dest_base}': {e}")
        print("💡 Folder network/drive mount mungkin sedang putus koneksi (stale mount) atau butuh mount ulang.")
        print("🔄 Mengalihkan penyimpanan otomatis ke folder lokal aman: ./ColonoMind_V8_Package ...")
        dest_base = os.path.abspath("./ColonoMind_V8_Export")
        package_dir = os.path.join(dest_base, "ColonoMind_V8_Package")
        models_dir = os.path.join(package_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

    print("=" * 75)
    print("🚀 COLONOMIND v8 — PACKAGING ARTIFACTS FOR LOCAL & WEB INTEGRATION")
    print("=" * 75)
    print(f"📁 Target Package Directory: {package_dir}")

    # Search paths for ColonoMind v8 / v12 / v14
    search_dirs = [
        args.v8_dir,
        "../Result/ColonoMind_v8_Hierarchical",
        "./Result/ColonoMind_v8_Hierarchical",
        "/raid/D13K48009/Clara/Result/ColonoMind_v8_Hierarchical",
        "/raid/D13K48009/Clara/colonomind-train/Result/ColonoMind_v8_Hierarchical",
        "../Result/ColonoMind_v12_V8FinetuneFocal",
        "./Result/ColonoMind_v12_V8FinetuneFocal",
        "/raid/D13K48009/Clara/Result/ColonoMind_v12_V8FinetuneFocal",
        "~/Clara/colono_train/Result/ColonoMind_v8_Hierarchical",
        "~/raid/Clara/colonomind-train/Result/ColonoMind_v8_Hierarchical"
    ]
    search_dirs = [os.path.abspath(os.path.expanduser(d)) for d in search_dirs if d]

    lgb_search_dirs = [
        "../Result/ColonoMind_v13_LGBMBalancer",
        "./Result/ColonoMind_v13_LGBMBalancer",
        "/raid/D13K48009/Clara/Result/ColonoMind_v13_LGBMBalancer",
        "../Result/ColonoMind_v8_Hierarchical",
        "./Result/ColonoMind_v8_Hierarchical",
        "/raid/D13K48009/Clara/Result/ColonoMind_v8_Hierarchical"
    ]
    lgb_search_dirs = [os.path.abspath(os.path.expanduser(d)) for d in lgb_search_dirs]

    v14_search_dirs = [
        "../Result/ColonoMind_v14_ThresholdOpt",
        "./Result/ColonoMind_v14_ThresholdOpt",
        "/raid/D13K48009/Clara/Result/ColonoMind_v14_ThresholdOpt"
    ]
    v14_search_dirs = [os.path.abspath(os.path.expanduser(d)) for d in v14_search_dirs]

    # Required files map: {filename: list_of_possible_source_paths}
    seeds = [42, 123, 999]
    files_to_find = {}

    # Stage 1 CNNs
    for s in seeds:
        files_to_find[f"stage1_cnn_seed{s}.h5"] = [os.path.join(d, f"stage1_cnn_seed{s}.h5") for d in search_dirs]

    # Stage 2 CNNs
    for s in seeds:
        files_to_find[f"stage2_cnn_seed{s}.h5"] = [os.path.join(d, f"stage2_cnn_seed{s}.h5") for d in search_dirs]

    # Scaler & UMAP
    files_to_find["scaler_v8.pkl"] = [os.path.join(d, "scaler_v8.pkl") for d in search_dirs]
    files_to_find["umap_v8.pkl"] = [os.path.join(d, "umap_v8.pkl") for d in search_dirs]

    # Super Agents (Hierarchical LightGBM)
    files_to_find["super_agent_stage1.txt"] = [os.path.join(d, "super_agent_stage1.txt") for d in search_dirs] + \
                                              [os.path.join(d, "balanced_agent_stage1.txt") for d in lgb_search_dirs]
    files_to_find["super_agent_stage2.txt"] = [os.path.join(d, "super_agent_stage2.txt") for d in search_dirs] + \
                                              [os.path.join(d, "balanced_agent_stage2.txt") for d in lgb_search_dirs]

    # Optional: v14 thresholds
    files_to_find["best_thresholds.json"] = [os.path.join(d, "best_thresholds.json") for d in v14_search_dirs]

    # Copy files
    print("\n🔍 Scanning and copying artifacts...")
    copied_count = 0
    missing_files = []

    for fname, candidates in files_to_find.items():
        found = find_first_existing(candidates)
        target_path = os.path.join(models_dir, fname)
        if found:
            shutil.copy2(found, target_path)
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            print(f"  ✅ [FOUND] {fname:<25} ({size_mb:.2f} MB) <- {found}")
            copied_count += 1
        else:
            if fname == "best_thresholds.json":
                print(f"  ℹ️ [OPTIONAL] {fname:<25} (Not found, default thresholds will be used)")
            else:
                print(f"  ❌ [MISSING] {fname:<25} (Searched in {len(candidates)} locations)")
                missing_files.append(fname)

    # Write self-contained v8_pipeline.py into package
    pipeline_code = '''import os
import cv2
import joblib
import json
import numpy as np
import scipy.stats
import pywt
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

IMG_SIZE = (256, 256)
ENSEMBLE_SEEDS = [42, 123, 999]

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def smart_preprocess(img):
    if img is None:
        return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    crop = img[30:430, 200:550] if h > 450 and w > 550 else img
    resized = cv2.resize(crop, (IMG_SIZE[1], IMG_SIZE[0]))
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
    r, g, b = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)
    t = r + g + b + 1e-6
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180]).flatten()
    h_hist /= (h_hist.sum() + 1e-6)
    return [
        float(np.mean(r / t)), float(np.mean(g / t)), float(np.mean((r - g) / (r + g + 1e-6))),
        float(np.std(g) / (np.mean(g) + 1e-6)), float(scipy.stats.entropy(h_hist + 1e-6)),
        float(np.mean(hsv[:, :, 1])), float(np.std(hsv[:, :, 1])), float(np.mean((r > 200) & (g > 200) & (b > 200)))
    ]

def extract_features_28(img):
    return np.array(extract_wavelet_glcm(img) + extract_clinical_colour(img), dtype=np.float32)

class ColonoMindV8Predictor:
    """End-to-End Hierarchical Predictor for Mod-SE CNN v8."""
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.models_dir = models_dir
        
        # Load Scaler & UMAP
        self.scaler = joblib.load(os.path.join(models_dir, "scaler_v8.pkl"))
        self.umap = joblib.load(os.path.join(models_dir, "umap_v8.pkl"))
        
        # Load Stage 1 CNNs (Binary Detector)
        self.stg1_models = []
        for s in ENSEMBLE_SEEDS:
            p = os.path.join(models_dir, f"stage1_cnn_seed{s}.h5")
            self.stg1_models.append(load_model(p, compile=False))
            
        # Load Stage 2 CNNs (Ternary Severity Grader)
        self.stg2_models = []
        for s in ENSEMBLE_SEEDS:
            p = os.path.join(models_dir, f"stage2_cnn_seed{s}.h5")
            self.stg2_models.append(load_model(p, compile=False))
            
        # Feature extractors
        self.stg1_extractors = [Model(inputs=m.input, outputs=m.get_layer('Fusion').output) for m in self.stg1_models]
        self.stg2_extractors = [Model(inputs=m.input, outputs=m.get_layer('Fusion').output) for m in self.stg2_models]
        
        # Load Super Agents
        self.agent1 = lgb.Booster(model_file=os.path.join(models_dir, "super_agent_stage1.txt"))
        self.agent2 = lgb.Booster(model_file=os.path.join(models_dir, "super_agent_stage2.txt"))
        
        # Thresholds (default or loaded from v14)
        thr_file = os.path.join(models_dir, "best_thresholds.json")
        if os.path.exists(thr_file):
            with open(thr_file, "r") as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = {"thr_stg1": 0.50, "w_mes1": 1.0, "w_mes2": 1.0, "w_mes3": 1.0}

    def predict(self, image_input):
        """
        Accepts: image file path or numpy RGB image (H, W, 3).
        Returns: dict with predicted_mes, probabilities, and stage details.
        """
        if isinstance(image_input, str):
            raw = cv2.imread(image_input)
            if raw is None:
                raise ValueError(f"Could not read image from {image_input}")
            img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input.copy()

        # Preprocessing
        clean_img = smart_preprocess(img_rgb)
        inp_img = np.expand_dims(clean_img.astype(np.float32) / 255.0, axis=0)
        
        # Handcrafted features
        raw_feat = extract_features_28(clean_img).reshape(1, -1)
        feat_scaled = self.scaler.transform(raw_feat)
        feat_umap = self.umap.transform(feat_scaled)
        
        # Stage 1: Detector (Normal vs Active)
        deeps1, probs1 = [], []
        for i in range(len(self.stg1_models)):
            deeps1.append(self.stg1_extractors[i].predict([inp_img, feat_scaled, feat_umap], verbose=0))
            probs1.append(self.stg1_models[i].predict([inp_img, feat_scaled, feat_umap], verbose=0))
        mean_deep1 = np.mean(deeps1, axis=0)
        mean_prob1 = np.mean(probs1, axis=0)
        
        x_agent1 = np.hstack([mean_deep1, feat_scaled, feat_umap, mean_prob1])
        prob_active = float(self.agent1.predict(x_agent1)[0])
        
        # Decision Routing
        thr_stg1 = self.thresholds.get("thr_stg1", 0.50)
        is_active = prob_active >= thr_stg1
        
        if not is_active:
            final_mes = 0
            confidence = 1.0 - prob_active
            active_breakdown = {"MES1": 0.0, "MES2": 0.0, "MES3": 0.0}
        else:
            # Stage 2: Severity Grader (MES1 vs MES2 vs MES3)
            deeps2, probs2 = [], []
            for i in range(len(self.stg2_models)):
                deeps2.append(self.stg2_extractors[i].predict([inp_img, feat_scaled, feat_umap], verbose=0))
                probs2.append(self.stg2_models[i].predict([inp_img, feat_scaled, feat_umap], verbose=0))
            mean_deep2 = np.mean(deeps2, axis=0)
            mean_prob2 = np.mean(probs2, axis=0)
            
            x_agent2 = np.hstack([mean_deep2, feat_scaled, feat_umap, mean_prob2])
            raw_stage2_probs = self.agent2.predict(x_agent2)[0] # Shape (3,)
            
            weights = np.array([
                self.thresholds.get("w_mes1", 1.0),
                self.thresholds.get("w_mes2", 1.0),
                self.thresholds.get("w_mes3", 1.0)
            ])
            weighted_probs = raw_stage2_probs * weights
            stg2_idx = int(np.argmax(weighted_probs))
            final_mes = stg2_idx + 1
            confidence = float(raw_stage2_probs[stg2_idx])
            active_breakdown = {
                "MES1": float(raw_stage2_probs[0]),
                "MES2": float(raw_stage2_probs[1]),
                "MES3": float(raw_stage2_probs[2])
            }

        return {
            "predicted_mes": final_mes,
            "label": f"MES {final_mes}",
            "confidence": round(confidence, 4),
            "prob_disease_active": round(prob_active, 4),
            "stage1_decision": "Active Inflammation" if is_active else "Normal / Quiescent",
            "active_breakdown": active_breakdown
        }

if __name__ == "__main__":
    import sys
    print("ColonoMind v8 Predictor Module Loaded Successfully.")
'''
    with open(os.path.join(package_dir, "v8_pipeline.py"), "w") as f:
        f.write(pipeline_code)
    print("  ✅ [GENERATED] v8_pipeline.py (Self-contained inference class)")

    # Write quick test runner test_v8_inference.py
    test_code = '''import os, sys, argparse, json
from v8_pipeline import ColonoMindV8Predictor

def main():
    parser = argparse.ArgumentParser(description="Test ColonoMind v8 inference")
    parser.add_argument("--image", required=True, help="Path to input colonoscopy image")
    args = parser.parse_args()

    print("=" * 60)
    print("ColonoMind v8 Hierarchical — Test Run")
    print("=" * 60)
    predictor = ColonoMindV8Predictor()
    result = predictor.predict(args.image)
    print(json.dumps(result, indent=4))
    print(f"\\n🎯 Result: {result['label']} (Confidence: {result['confidence']*100:.1f}%)")

if __name__ == "__main__":
    main()
'''
    with open(os.path.join(package_dir, "test_v8_inference.py"), "w") as f:
        f.write(test_code)
    print("  ✅ [GENERATED] test_v8_inference.py (CLI test runner)")

    # Write README
    readme_content = """# ColonoMind v8 Hierarchical Deployment Package

This package contains the complete, ready-to-run artifacts of **ColonoMind v8 (Hierarchical Two-Stage Mod-SE CNN + LightGBM Super Agents)**, which achieved **>80% accuracy** on colonoscopy inflammation grading.

## Package Contents:
- `models/`:
  - `stage1_cnn_seed42.h5`, `stage1_cnn_seed123.h5`, `stage1_cnn_seed999.h5`: Binary Detector CNNs (Normal vs Active)
  - `stage2_cnn_seed42.h5`, `stage2_cnn_seed123.h5`, `stage2_cnn_seed999.h5`: Ternary Severity Grader CNNs (MES1 vs 2 vs 3)
  - `scaler_v8.pkl`: StandardScaler for 28-dimensional handcrafted features
  - `umap_v8.pkl`: UMAP 2D projection model
  - `super_agent_stage1.txt`: LightGBM Stage 1 Booster
  - `super_agent_stage2.txt`: LightGBM Stage 2 Booster
  - `best_thresholds.json`: Tuned probability thresholds (v14 optimizer)
- `v8_pipeline.py`: Pure Python inference class `ColonoMindV8Predictor`
- `test_v8_inference.py`: Command line test script

## Usage Example:
```python
from v8_pipeline import ColonoMindV8Predictor

predictor = ColonoMindV8Predictor()
res = predictor.predict("path/to/endoscopy.jpg")

print(res["label"])          # e.g. "MES 1"
print(res["confidence"])     # e.g. 0.842
print(res["active_breakdown"])
```
"""
    with open(os.path.join(package_dir, "README_V8.md"), "w") as f:
        f.write(readme_content)
    print("  ✅ [GENERATED] README_V8.md")

    # Create ZIP archive
    zip_path = os.path.join(dest_base, "colonomind_v8_package.zip")
    print(f"\n📦 Zipping package into: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, package_dir)
                zipf.write(full_path, arcname=os.path.join("ColonoMind_V8_Package", rel_path))
    
    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"🎉 Complete! Package archive created successfully ({zip_mb:.2f} MB).")
    print(f"📍 Location on DGX: {zip_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
