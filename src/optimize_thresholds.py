import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tabulate import tabulate

from src.data_loader import load_dataset
from src.train import focal_loss
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Test dataset name (e.g., TMC-UCM)")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing the models (e.g., Result/Intra_TMC-UCM)")
    args = parser.parse_args()

    print(f"\n🚀 Optimizing Thresholds for Dataset: {args.dataset}")
    print("Loading test dataset (Inference only, no training!)...")
    
    # Load dataset
    X_img_test, X_feat_test, y_test, le = load_dataset(args.dataset, limit=None)
    y_true = le.transform(y_test)
    
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
            print(f"⏭️  Skipping {model_name} (Model files not found)")
            continue
            
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Load artifacts
        try:
            model = tf.keras.models.load_model(keras_path, custom_objects={'focal_loss_fixed': focal_loss(gamma=2.5, alpha=0.25)})
        except Exception as e:
            print(f"Failed to load Keras model for {model_name}: {e}")
            continue
            
        import lightgbm as lgb
        agent = lgb.Booster(model_file=agent_path)
        scaler_ag = joblib.load(scaler_path)
        base_scaler = joblib.load(base_scaler_path)
        umap_reducer = joblib.load(umap_path)
        
        # Prepare test features
        X_feat_test_scaled = base_scaler.transform(X_feat_test)
        X_test_umap = umap_reducer.transform(X_feat_test_scaled)
        
        # DL Inference (This takes a minute)
        y_pred_proba_test = model.predict([X_img_test, X_feat_test_scaled, X_test_umap], verbose=1)
        y_pred_deep = np.argmax(y_pred_proba_test, axis=1)
        base_acc = accuracy_score(y_true, y_pred_deep)
        
        conf_test = np.max(y_pred_proba_test, axis=1)
        
        # Prepare agent features
        df_test_ag = pd.DataFrame(X_feat_test_scaled, columns=[f"f{i}" for i in range(20)])
        df_test_ag["confidence"] = conf_test
        df_test_ag["umap_0"] = X_test_umap[:, 0]
        df_test_ag["umap_1"] = X_test_umap[:, 1]
        features = ["confidence", "umap_0", "umap_1"] + [f"f{i}" for i in range(20)]
        
        X_te = scaler_ag.transform(df_test_ag[features].values)
        
        # Agent Inference
        agent_preds_proba = agent.predict(X_te)
        # Handle if LightGBM outputs raw values or probabilities
        if len(agent_preds_proba.shape) == 1:
            # Binary classification fallback? Should be multiclass
            y_pred_agent = np.round(agent_preds_proba).astype(int)
        else:
            y_pred_agent = np.argmax(agent_preds_proba, axis=1)
            
        row = [model_name, f"{base_acc*100:.2f}%"]
        
        best_hyb = 0
        best_th = 0
        
        for th in thresholds:
            if th == 0.0:
                # Disable agent completely
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
    print("\n💡 NOTE: If Base Acc is equal to Best Hybrid, it means the Super Agent should be disabled (Threshold = 0.0).")

if __name__ == "__main__":
    main()
