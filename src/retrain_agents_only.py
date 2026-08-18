import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import joblib
import json

from dgx_dataloader import load_all_images, load_tmc_ucm
import dgx_models

def main():
    print("======================================================")
    print("🚀 FAST RETRAIN: AGENT ONLY (META-CLASSIFIER)")
    print("======================================================")

    SCENARIOS = ['Intra_Unified', 'Intra_TMC-UCM', 'Intra_NTUH', 'Intra_LIMUC']
    MODELS = ["ResNet-50", "DenseNet-121", "EfficientNet-B4", "ConvNeXt-Tiny", "ViT-B-16"]
    
    BASE_DIR = '/home/D13K48009/raid/Clara/new_drive'
    
    custom_objects = {
        "resnet50_preprocess": dgx_models.resnet50_preprocess,
        "densenet_preprocess": dgx_models.densenet_preprocess,
        "efficientnet_preprocess": dgx_models.efficientnet_preprocess,
        "convnext_preprocess": dgx_models.convnext_preprocess,
        "vit_preprocess": dgx_models.vit_preprocess,
    }

    for scenario_full in SCENARIOS:
        scenario = scenario_full.split('_')[0]
        dataset_name = scenario_full.split('_')[1]
        
        print(f"\n📂 Loading Data for {scenario_full}...")
        if dataset_name == 'Unified':
            tmc_imgs, tmc_feats, tmc_labels, _ = load_tmc_ucm(f'{BASE_DIR}/Dataset/TMC-UCM')
            ntuh_imgs, ntuh_feats, ntuh_labels, _ = load_all_images([f'{BASE_DIR}/Dataset+Code/MES classification_20250724'], 'NTUH')
            limuc_imgs, limuc_feats, limuc_labels, _ = load_all_images([f'{BASE_DIR}/Dataset/LIMUC/train_and_validation_sets', f'{BASE_DIR}/Dataset/LIMUC/test_set'], 'LIMUC')
            all_imgs = tmc_imgs + ntuh_imgs + limuc_imgs
            all_feats = tmc_feats + ntuh_feats + limuc_feats
            all_labels = tmc_labels + ntuh_labels + limuc_labels
            X_train_img_raw, X_test_img, X_train_feat_raw, X_test_feat, y_train_label_raw, y_test_label = train_test_split(all_imgs, all_feats, all_labels, test_size=0.2, random_state=42, stratify=all_labels)[:6]
        else:
            # For simplicity in this script, we skip loading the actual multi-dataset here and focus on the concept
            # We will just write a wrapper script around train_dgx.py passing a new flag --agent_only
            pass
