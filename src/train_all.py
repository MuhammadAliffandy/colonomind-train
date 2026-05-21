"""
Colonomind - Train on ALL Datasets Combined
============================================
This script merges all 4 datasets into a single pool,
performs a stratified train/val/test split, then trains
the Hybrid Mod-SE(2) CNN model on the combined data.

Usage:
    python src/train_all.py --output_dir ./results/all_datasets
    python src/train_all.py --output_dir ./results/all_datasets --split 70 15 15
    python src/train_all.py --output_dir ./results/all_datasets --epochs 50
"""

import os
import sys

# Ensure project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap

from src.config import IMG_SIZE, DATASETS
from src.data_loader import load_dataset
from src.model import build_hybrid_model
from src.train import focal_loss

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check GPU/CUDA availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nTensorFlow Version: {tf.__version__}")
    if gpus:
        print(f"✅ GPU is detected and available for CUDA acceleration: {gpus}\n")
        # Enable memory growth to avoid locking all GPU memory at once
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ GPU is NOT detected by TensorFlow. Running on CPU instead.\n")
    
    # =============================================
    # 1. LOAD & MERGE ALL DATASETS
    # =============================================
    all_img, all_feat, all_label, all_paths, all_source = [], [], [], [], []
    
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"  ⚠️  Skipping '{name}': path not found ({path})")
            continue
        print(f"  Loading '{name}' from {path}...")
        X_img, X_feat, y_label, img_paths = load_dataset(path)
        all_img.append(X_img)
        all_feat.append(X_feat)
        all_label.append(y_label)
        all_paths.append(img_paths)
        all_source.append(np.full(len(y_label), name))
        print(f"    -> {len(y_label)} images loaded")
    
    X_img_all   = np.concatenate(all_img, axis=0)
    X_feat_all  = np.concatenate(all_feat, axis=0)
    y_label_all = np.concatenate(all_label, axis=0)
    paths_all   = np.concatenate(all_paths, axis=0)
    source_all  = np.concatenate(all_source, axis=0)
    
    print(f"\n{'='*50}")
    print(f" TOTAL COMBINED: {len(y_label_all)} images")
    print(f" Classes: {np.unique(y_label_all)}")
    print(f" Sources: {np.unique(source_all)}")
    print(f"{'='*50}\n")
    
    # =============================================
    # 2. ENCODE LABELS
    # =============================================
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_label_all)
    num_classes = len(le.classes_)
    
    # =============================================
    # 3. STRATIFIED TRAIN / VAL / TEST SPLIT
    # =============================================
    train_pct = args.split[0] / 100
    val_pct   = args.split[1] / 100
    test_pct  = args.split[2] / 100
    
    # First split: train vs (val+test)
    X_img_train, X_img_temp, X_feat_train, X_feat_temp, y_train, y_temp, paths_train, paths_temp, src_train, src_temp = \
        train_test_split(X_img_all, X_feat_all, y_encoded, paths_all, source_all,
                         test_size=(val_pct + test_pct), stratify=y_encoded, random_state=42)
    
    # Second split: val vs test
    relative_test = test_pct / (val_pct + test_pct)
    X_img_val, X_img_test, X_feat_val, X_feat_test, y_val, y_test, paths_val, paths_test, src_val, src_test = \
        train_test_split(X_img_temp, X_feat_temp, y_temp, paths_temp, src_temp,
                         test_size=relative_test, stratify=y_temp, random_state=42)
    
    print(f" Split Ratio: {args.split[0]}/{args.split[1]}/{args.split[2]}")
    print(f" Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f" Train classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f" Val   classes: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f" Test  classes: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Save split info for paper tables
    split_info = {
        'train_count': len(y_train), 'val_count': len(y_val), 'test_count': len(y_test),
        'train_classes': dict(zip(*np.unique(y_train, return_counts=True))),
        'val_classes': dict(zip(*np.unique(y_val, return_counts=True))),
        'test_classes': dict(zip(*np.unique(y_test, return_counts=True))),
        'train_sources': dict(zip(*np.unique(src_train, return_counts=True))),
        'val_sources': dict(zip(*np.unique(src_val, return_counts=True))),
        'test_sources': dict(zip(*np.unique(src_test, return_counts=True))),
        'class_names': list(le.classes_),
        'split_ratio': list(args.split),
    }
    joblib.dump(split_info, os.path.join(args.output_dir, "split_info.pkl"))
    
    # Print split table
    print(f"\n{'='*60}")
    print(f" EXTENDED DATA TABLE 2a: Images per Class per Split")
    print(f"{'='*60}")
    print(f" {'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f" {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for i, cls in enumerate(le.classes_):
        tr = np.sum(y_train == i)
        va = np.sum(y_val == i)
        te = np.sum(y_test == i)
        print(f" {cls:<12} {tr:>8} {va:>8} {te:>8} {tr+va+te:>8}")
    print(f" {'TOTAL':<12} {len(y_train):>8} {len(y_val):>8} {len(y_test):>8} {len(y_encoded):>8}")
    print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(f" EXTENDED DATA TABLE 2b: Images per Source per Split")
    print(f"{'='*60}")
    print(f" {'Source':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f" {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for src_name in sorted(np.unique(source_all)):
        tr = np.sum(src_train == src_name)
        va = np.sum(src_val == src_name)
        te = np.sum(src_test == src_name)
        print(f" {src_name:<12} {tr:>8} {va:>8} {te:>8} {tr+va+te:>8}")
    print(f"{'='*60}\n")
    
    # =============================================
    # 4. NORMALIZE & SCALE
    # =============================================
    X_img_train = X_img_train.astype(np.float32) / 255.0
    X_img_val   = X_img_val.astype(np.float32) / 255.0
    X_img_test  = X_img_test.astype(np.float32) / 255.0
    
    scaler = StandardScaler()
    X_feat_train_scaled = scaler.fit_transform(X_feat_train)
    X_feat_val_scaled   = scaler.transform(X_feat_val)
    X_feat_test_scaled  = scaler.transform(X_feat_test)
    
    # =============================================
    # 5. CLASS WEIGHTING & TARGET ENCODING
    # =============================================
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_feat_train_bal, y_train_bal = smote.fit_resample(X_feat_train_scaled, y_train)

    # Map balanced features to real images
    print("Mapping balanced features back to images...")
    X_img_train_bal = []
    for feat, label in zip(X_feat_train_bal, y_train_bal):
        dists = np.linalg.norm(X_feat_train_scaled[y_train == label] - feat, axis=1)
        idx = np.where(y_train == label)[0][np.argmin(dists)]
        X_img_train_bal.append(X_img_train[idx])
    X_img_train_bal = np.array(X_img_train_bal, dtype=np.float32)

    y_train_cat_bal = to_categorical(y_train_bal, num_classes=num_classes)
    y_val_cat   = to_categorical(y_val, num_classes=num_classes)
    y_test_cat  = to_categorical(y_test, num_classes=num_classes)
    
    # =============================================
    # 6. UMAP
    # =============================================
    print("Running UMAP Projection...")
    umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, metric='euclidean', random_state=42)
    X_train_umap = umap_reducer.fit_transform(X_feat_train_bal)
    X_val_umap   = umap_reducer.transform(X_feat_val_scaled)
    X_test_umap  = umap_reducer.transform(X_feat_test_scaled)
    
    # =============================================
    # 7. BUILD & TRAIN MODEL
    # =============================================
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_bal), y=y_train_bal)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    print("Building Hybrid Model...")
    model = build_hybrid_model(
        image_input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        feat_input_shape=(X_feat_train_bal.shape[1],),
        umap_feat_shape=(2,),
        num_classes=num_classes,
        dropout_rate=0.4
    )
    
    model.compile(
        optimizer=Adam(1e-3),
        loss=focal_loss(gamma=2.0, alpha=1.0),
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')
    ]
    
    print("Starting Training...")
    history = model.fit(
        [X_img_train_bal, X_feat_train_bal, X_train_umap], y_train_cat_bal,
        validation_data=([X_img_val, X_feat_val_scaled, X_val_umap], y_val_cat),
        batch_size=args.batch_size,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # =============================================
    # 8. EVALUATE ON TEST SET
    # =============================================
    print("\nEvaluating on Test Set...")
    test_loss, test_acc = model.evaluate(
        [X_img_test, X_feat_test_scaled, X_test_umap], y_test_cat, verbose=0
    )
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss:     {test_loss:.4f}")
    
    # =============================================
    # 9. SAVE EVERYTHING
    # =============================================
    model.save(os.path.join(args.output_dir, "best_hybrid_model.h5"))
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.pkl"))
    joblib.dump(umap_reducer, os.path.join(args.output_dir, "umap_model.pkl"))
    
    # Save training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_history.png"), dpi=300)
    
    print(f"\n✅ All artifacts saved to: {args.output_dir}")
    print(f"   - best_hybrid_model.h5")
    print(f"   - scaler.pkl, label_encoder.pkl, umap_model.pkl")
    print(f"   - split_info.pkl (for paper tables)")
    print(f"   - training_history.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train Colonomind on ALL datasets combined with stratified split"
    )
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save results")
    parser.add_argument('--split', type=int, nargs=3, default=[70, 15, 15],
                        help="Train/Val/Test split percentages (default: 70 15 15)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--epochs', type=int, default=90, help="Number of epochs")
    
    args = parser.parse_args()
    
    assert sum(args.split) == 100, f"Split must sum to 100, got {sum(args.split)}"
    main(args)
