import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import umap
from tqdm import tqdm

from dgx_dataloader import load_all_images, load_tmc_ucm
import dgx_models

def main():
    parser = argparse.ArgumentParser(description='ColonoMind: Generate Teacher Probabilities')
    parser.add_argument('--scenario', type=str, default='Unified', choices=['Intra', 'Multi', 'Unified'])
    parser.add_argument('--train_dataset', type=str, default='Unified')
    parser.add_argument('--test_dataset', type=str, default='Unified')
    parser.add_argument('--models_dir', type=str, required=True, help="Path to models (e.g., Result/Intra_Unified)")
    parser.add_argument('--base_dir', type=str, default='/home/D13K48009/raid/Clara/new_drive')
    parser.add_argument('--cache_dir', type=str, default=None)
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    DATASET_PATHS = {
        'NTUH': [
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
    TMC_UCM_ROOT = f'{BASE_DIR}/Dataset/TMC-UCM'

    # LOAD DATA (Exactly matching train_dgx.py split logic)
    print("Loading Data for Teacher...")
    if args.scenario == 'Unified':
        tmc_imgs, tmc_feats, tmc_labels, tmc_paths = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None, cache_dir=args.cache_dir)
        ntuh_imgs, ntuh_feats, ntuh_labels, ntuh_paths = load_all_images(DATASET_PATHS['NTUH'], 'NTUH', cache_dir=args.cache_dir)
        limuc_imgs, limuc_feats, limuc_labels, limuc_paths = load_all_images(
            [DATASET_PATHS['LIMUC'][0], DATASET_PATHS['LIMUC'][1]], 'LIMUC', cache_dir=args.cache_dir
        )
        all_imgs   = tmc_imgs   + ntuh_imgs   + limuc_imgs
        all_feats  = tmc_feats  + ntuh_feats  + limuc_feats
        all_labels = tmc_labels + ntuh_labels + limuc_labels
        all_paths  = tmc_paths  + ntuh_paths  + limuc_paths
        
        X_train_img_raw, X_test_img, X_train_feat_raw, X_test_feat, y_train_label_raw, y_test_label, _, _ = train_test_split(
            all_imgs, all_feats, all_labels, all_paths, test_size=0.2, random_state=42, stratify=all_labels
        )
    else:
        raise NotImplementedError("KD is currently optimized for Unified scenario.")

    print("Splitting Train into Train/Val (80/20)...")
    X_train_img, X_val_img, X_train_feat, X_val_feat, y_train_label, y_val_label = train_test_split(
        X_train_img_raw, X_train_feat_raw, y_train_label_raw, test_size=0.2, random_state=42, stratify=y_train_label_raw
    )

    X_img_train = np.array(X_train_img, dtype=np.float32)
    X_img_val = np.array(X_val_img, dtype=np.float32)

    scaler = StandardScaler()
    X_feat_train_scaled = scaler.fit_transform(np.array(X_train_feat))
    X_feat_val_scaled = scaler.transform(np.array(X_val_feat))

    print("Fitting UMAP on Train...")
    umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=42)
    X_train_umap = umap_reducer.fit_transform(X_feat_train_scaled)
    X_val_umap = umap_reducer.transform(X_feat_val_scaled)

    MODEL_NAMES = ["ResNet-50", "DenseNet-121", "EfficientNet-B4", "ConvNeXt-Tiny", "ViT-B-16"]
    
    custom_objects = {
        "resnet50_preprocess": dgx_models.resnet50_preprocess,
        "densenet_preprocess": dgx_models.densenet_preprocess,
        "efficientnet_preprocess": dgx_models.efficientnet_preprocess,
        "convnext_preprocess": dgx_models.convnext_preprocess,
        "vit_preprocess": dgx_models.vit_preprocess,
    }

    print("\\n[1] Generating Probabilities from Teachers...")
    
    all_train_probs = []
    all_val_probs = []
    
    # Batch prediction to prevent memory issues
    batch_size = 64
    
    for mn in MODEL_NAMES:
        model_path = os.path.join(args.models_dir, f"{mn}_Experiment", f"{mn}_hybrid.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing Teacher Model: {model_path}")
            
        print(f"  -> Loading Teacher: {mn}")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        print(f"     Predicting Train...")
        preds_train = model.predict([X_img_train, X_feat_train_scaled, X_train_umap], batch_size=batch_size, verbose=1)
        print(f"     Predicting Val...")
        preds_val = model.predict([X_img_val, X_feat_val_scaled, X_val_umap], batch_size=batch_size, verbose=1)
        
        all_train_probs.append(preds_train)
        all_val_probs.append(preds_val)

    # Average the probabilities (Soft-Voting Ensemble Logic)
    print("\\n[2] Averaging Teacher Predictions...")
    ensemble_train_probs = np.mean(all_train_probs, axis=0)
    ensemble_val_probs = np.mean(all_val_probs, axis=0)
    
    save_dir = os.path.join(args.models_dir, "Teacher_Probs")
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "teacher_train_probs.npy"), ensemble_train_probs)
    np.save(os.path.join(save_dir, "teacher_val_probs.npy"), ensemble_val_probs)
    
    print(f"✅ Teacher soft probabilities saved successfully to {save_dir}")

if __name__ == "__main__":
    main()
