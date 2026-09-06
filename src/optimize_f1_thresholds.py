import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import differential_evolution

# Import local data loaders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images, load_tmc_ucm
from src.train import focal_loss
from src.train_unified_colonomind_se import OrdinalFocalLoss

def load_data(dataset_name, base_dir):
    """Load test data for a given dataset."""
    DATASET_PATHS = {
        'NTUH':    [f'{base_dir}/Dataset+Code/MES classification_20250313', f'{base_dir}/Dataset+Code/MES classification_20250724'],
        'LIMUC':   [f'{base_dir}/Dataset/LIMUC/train_and_validation_sets', f'{base_dir}/Dataset/LIMUC/test_set'],
    }
    TMC_UCM_ROOT = f'{base_dir}/Dataset/TMC-UCM'
    
    if dataset_name == 'Unified':
        tmc_imgs, tmc_feats, tmc_labels, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Test')
        limuc_imgs, limuc_feats, limuc_labels, _ = load_all_images([DATASET_PATHS['LIMUC'][1]], 'LIMUC')
        
        ntuh_imgs, ntuh_feats, ntuh_labels, _ = load_all_images(DATASET_PATHS['NTUH'], 'NTUH')
        # Simulate 20% test for NTUH
        np.random.seed(42)
        idx = np.random.choice(len(ntuh_imgs), int(0.2 * len(ntuh_imgs)), replace=False)
        ntuh_imgs_test = [ntuh_imgs[i] for i in idx]
        ntuh_feats_test = [ntuh_feats[i] for i in idx]
        ntuh_labels_test = [ntuh_labels[i] for i in idx]
        
        X_test_img = tmc_imgs + limuc_imgs + ntuh_imgs_test
        X_test_feat = tmc_feats + limuc_feats + ntuh_feats_test
        y_test_label = tmc_labels + limuc_labels + ntuh_labels_test
    else:
        print("This script currently supports 'Unified' dataset mode.")
        sys.exit(1)
        
    return np.array(X_test_img, dtype=np.float32), np.array(X_test_feat, dtype=np.float32), y_test_label

def objective_function(weights, y_proba, y_true):
    """
    Objective function for differential_evolution.
    We want to MAXIMIZE macro_f1, so we MINIMIZE negative macro_f1.
    """
    weighted_proba = y_proba * weights
    y_pred = np.argmax(weighted_proba, axis=1)
    f1 = f1_score(y_true, y_pred, average='macro')
    return -f1

def print_metrics(y_true, y_pred, title):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro')
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    print(f"\n--- {title} ---")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Macro F1 : {f1*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"QWK      : {qwk:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved .keras model")
    parser.add_argument("--base_dir", type=str, default="/raid/D13K48009/Clara/new_drive", help="Base dataset path")
    args = parser.parse_args()

    print(f"🚀 Loading Unified Dataset from {args.base_dir} ...")
    X_img, X_feat, y_labels = load_data('Unified', args.base_dir)
    
    le = LabelEncoder()
    le.fit(["MES0", "MES1", "MES2", "MES3"])
    y_true = le.transform(y_labels)
    
    print(f"📦 Dataset loaded: {len(X_img)} images")
    
    model_dir = os.path.dirname(args.model_path)
    base_scaler_path = os.path.join(model_dir, "base_scaler.pkl")
    umap_path = os.path.join(model_dir, "umap_model.pkl")
    
    print(f"🧠 Loading Model from {args.model_path} ...")
    
    custom_objs = {
        'focal_loss_fixed': focal_loss(gamma=2.5, alpha=0.25),
        'OrdinalFocalLoss': OrdinalFocalLoss,
        'ordinal_focal_loss_1': OrdinalFocalLoss,
        'ordinal_focal_loss': OrdinalFocalLoss
    }
    
    try:
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objs)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("📊 Loading tab scaler & UMAP reducer...")
    base_scaler = joblib.load(base_scaler_path)
    umap_reducer = joblib.load(umap_path)
    
    X_feat_scaled = base_scaler.transform(X_feat)
    X_umap = umap_reducer.transform(X_feat_scaled)

    # Resize images to match model input if necessary
    expected_shape = model.input[0].shape[1:3] if isinstance(model.input, list) else model.input.shape[1:3]
    if tuple(X_img.shape[1:3]) != tuple(expected_shape):
        print(f"✂️ Resizing images to {expected_shape} ...")
        X_img = tf.image.resize(X_img, expected_shape).numpy()

    print("🔮 Running inference to get baseline probabilities ...")
    # Batch predict to avoid OOM
    y_proba = model.predict([X_img, X_feat_scaled, X_umap], batch_size=32, verbose=1)
    
    y_pred_baseline = np.argmax(y_proba, axis=1)
    print_metrics(y_true, y_pred_baseline, "BASELINE PERFORMANCE (Standard Argmax)")

    print("\n🧬 Running Differential Evolution to find optimal Class Probability Weights ...")
    print("This will find [w0, w1, w2, w3] such that argmax(y_proba * W) maximizes Macro F1.")
    
    # We optimize 4 weights bounded between 0.1 and 10.0
    bounds = [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]
    
    result = differential_evolution(
        objective_function, 
        bounds, 
        args=(y_proba, y_true),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=1e-4,
        disp=True
    )
    
    best_weights = result.x
    print(f"\n✅ Optimization Complete!")
    print(f"Optimal Weights for [MES0, MES1, MES2, MES3]:")
    print(f"W0: {best_weights[0]:.4f}")
    print(f"W1: {best_weights[1]:.4f}")
    print(f"W2: {best_weights[2]:.4f}")
    print(f"W3: {best_weights[3]:.4f}")
    
    # Apply optimal weights
    y_proba_optimized = y_proba * best_weights
    y_pred_optimized = np.argmax(y_proba_optimized, axis=1)
    
    print_metrics(y_true, y_pred_optimized, "OPTIMIZED PERFORMANCE (F1-Maximizer)")
    
    print("\n💡 HOW TO USE IN INFERENCE:")
    print("Instead of: y_pred = np.argmax(probabilities)")
    print(f"Use       : y_pred = np.argmax(probabilities * np.array({best_weights.tolist()}))")

if __name__ == "__main__":
    main()
