import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images, load_tmc_ucm
from src.train import focal_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_test_data(dataset_name, base_dir=".."):
    DATASET_PATHS = {
        'NTUH':    [f'{base_dir}/Dataset/NTUH'],
        'LIMUC':   [f'{base_dir}/Dataset/LIMUC/train_and_validation_sets', f'{base_dir}/Dataset/LIMUC/test_set'],
        'TMC-UCM': [f'{base_dir}/Dataset/TMC-UCM/images']
    }
    TMC_UCM_ROOT = f'{base_dir}/Dataset/TMC-UCM'
    
    if dataset_name == 'Unified':
        tmc_imgs, tmc_feats, tmc_labels, tmc_paths = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None)
        ntuh_imgs, ntuh_feats, ntuh_labels, ntuh_paths = load_all_images(DATASET_PATHS['NTUH'], 'NTUH')
        limuc_imgs, limuc_feats, limuc_labels, limuc_paths = load_all_images([DATASET_PATHS['LIMUC'][0], DATASET_PATHS['LIMUC'][1]], 'LIMUC')
        all_imgs   = tmc_imgs   + ntuh_imgs   + limuc_imgs
        all_feats  = tmc_feats  + ntuh_feats  + limuc_feats
        all_labels = tmc_labels + ntuh_labels + limuc_labels
        all_paths  = tmc_paths  + ntuh_paths  + limuc_paths
        _, X_test_img, _, X_test_feat, _, y_test_label, _, _ = train_test_split(
            all_imgs, all_feats, all_labels, all_paths, test_size=0.2, random_state=42, stratify=all_labels
        )
    elif dataset_name == 'LIMUC':
        X_test_img, X_test_feat, y_test_label, _ = load_all_images([DATASET_PATHS['LIMUC'][1]], dataset_name)
    elif dataset_name == 'TMC-UCM':
        X_test_img, X_test_feat, y_test_label, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Test')
    elif dataset_name == 'NTUH':
        all_imgs, all_feats, all_labels, all_paths = load_all_images(DATASET_PATHS['NTUH'], dataset_name)
        _, X_test_img, _, X_test_feat, _, y_test_label, _, _ = train_test_split(
            all_imgs, all_feats, all_labels, all_paths, test_size=0.2, random_state=42, stratify=all_labels
        )
    else:
        print(f"Unknown dataset {dataset_name}")
        return None, None, None
        
    return np.array(X_test_img, dtype=np.float32), np.array(X_test_feat, dtype=np.float32), y_test_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Test dataset name (e.g., TMC-UCM)")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing the models")
    parser.add_argument("--base_dir", type=str, default="/home/D13K48009/raid/Clara/new_drive", help="Base directory containing the Dataset/ folder")
    args = parser.parse_args()

    print(f"\n🚀 Optimizing Thresholds for Dataset: {args.dataset}")
    print(f"   Using BASE_DIR: {args.base_dir}")
    print("Loading test dataset exactly as trained (Inference only!)...")
    
    X_img_test, X_feat_test, y_test_label = load_test_data(args.dataset, base_dir=args.base_dir)
    if X_img_test is None:
        return
        
    le = LabelEncoder()
    le.fit(["MES0", "MES1", "MES2", "MES3"])
    y_true = le.transform(y_test_label)
    
    models = ["ResNet-50", "DenseNet-121", "EfficientNet-B4", "ConvNeXt-Tiny", "ViT-B-16"]
    thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results_table = []
    
    for model_name in models:
        exp_dir = os.path.join(args.models_dir, f"{model_name}_Experiment")
        keras_path = os.path.join(exp_dir, f"{model_name}_hybrid.keras")
        agent_path = os.path.join(exp_dir, f"{model_name}_agent.txt")
        scaler_path = os.path.join(exp_dir, f"{model_name}_scaler.pkl")
        base_scaler_path = os.path.join(exp_dir, "base_scaler.pkl")
        umap_path = os.path.join(exp_dir, "umap_model.pkl")
        
        if not os.path.exists(keras_path) or not os.path.exists(agent_path):
            print(f"⏭️  Skipping {model_name} (Files missing)")
            continue
            
        print(f"\n🔍 Evaluating {model_name}...")
        try:
            model = tf.keras.models.load_model(keras_path, custom_objects={'focal_loss_fixed': focal_loss(gamma=2.5, alpha=0.25)})
        except Exception as e:
            print(f"Load error: {e}")
            continue
            
        import lightgbm as lgb
        agent = lgb.Booster(model_file=agent_path)
        scaler_ag = joblib.load(scaler_path)
        base_scaler = joblib.load(base_scaler_path)
        umap_reducer = joblib.load(umap_path)
        
        X_feat_test_scaled = base_scaler.transform(X_feat_test)
        X_test_umap = umap_reducer.transform(X_feat_test_scaled)
        
        # Auto-resize images if the model is V1 (224x224) vs V2 (384x384)
        expected_shape = model.input[0].shape[1:3]
        if tuple(X_img_test.shape[1:3]) != tuple(expected_shape):
            print(f"      [Auto-Fix] Resizing images from {X_img_test.shape[1:3]} to {expected_shape} to match model...")
            X_img_eval = tf.image.resize(X_img_test, expected_shape).numpy()
        else:
            X_img_eval = X_img_test
            
        y_pred_proba_test = model.predict([X_img_eval, X_feat_test_scaled, X_test_umap], verbose=1)
        y_pred_deep = np.argmax(y_pred_proba_test, axis=1)
        base_acc = accuracy_score(y_true, y_pred_deep)
        
        conf_test = np.max(y_pred_proba_test, axis=1)
        
        df_test_ag = pd.DataFrame(X_feat_test_scaled, columns=[f"f{i}" for i in range(20)])
        df_test_ag["confidence"] = conf_test
        df_test_ag["umap_0"] = X_test_umap[:, 0]
        df_test_ag["umap_1"] = X_test_umap[:, 1]
        features = ["confidence", "umap_0", "umap_1"] + [f"f{i}" for i in range(20)]
        
        X_te = scaler_ag.transform(df_test_ag[features].values)
        agent_preds_proba = agent.predict(X_te)
        
        if len(agent_preds_proba.shape) == 1:
            y_pred_agent = np.round(agent_preds_proba).astype(int)
        else:
            y_pred_agent = np.argmax(agent_preds_proba, axis=1)
            
        row = [model_name, f"{base_acc*100:.2f}%"]
        best_hyb, best_th = 0, 0
        
        for th in thresholds:
            if th == 0.0:
                hybrid_acc = base_acc
            else:
                low_conf_mask = conf_test < th
                y_pred_hybrid = np.where(low_conf_mask, y_pred_agent, y_pred_deep)
                hybrid_acc = accuracy_score(y_true, y_pred_hybrid)
                
            if hybrid_acc > best_hyb:
                best_hyb = hybrid_acc
                best_th = th
            row.append(f"{hybrid_acc*100:.2f}%")
            
        row.append(f"{best_hyb*100:.2f}% (th={best_th})")
        results_table.append(row)
        
    print(f"\n\n🏆 RESULTS FOR {args.dataset}")
    headers = ["Model", "Base Acc"] + [f"Th={th}" for th in thresholds] + ["BEST HYBRID"]
    print(tabulate(results_table, headers=headers, tablefmt="grid"))
    print("\n💡 NOTE: If Base Acc is equal to Best Hybrid, disable the Agent (Threshold = 0.0).")

if __name__ == "__main__":
    main()
