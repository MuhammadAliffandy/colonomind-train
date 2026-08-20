import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Monkey patch the dataloader to use 224x224 for V1 models
import dgx_dataloader
dgx_dataloader.IMG_SIZE = (224, 224)
from dgx_dataloader import load_all_images, load_tmc_ucm
import dgx_models

def main():
    print("======================================================")
    print("🕰️ RECOVERING V1 (224x224) METRICS")
    print("======================================================")
    
    SCENARIOS = ['Intra_NTUH', 'Intra_LIMUC', 'Intra_Unified']
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
        scenario_prefix = scenario_full.split('_')[0]
        dataset_name = scenario_full.split('_')[1]
        
        # Load Test Data for this scenario
        print(f"\n📂 Loading Test Data for {scenario_full} at 224x224...")
        if dataset_name == 'Unified':
            tmc_imgs, tmc_feats, tmc_labels, _ = load_tmc_ucm(f'{BASE_DIR}/Dataset/TMC-UCM')
            ntuh_imgs, ntuh_feats, ntuh_labels, _ = load_all_images([f'{BASE_DIR}/Dataset+Code/MES classification_20250724'], 'NTUH')
            limuc_imgs, limuc_feats, limuc_labels, _ = load_all_images([f'{BASE_DIR}/Dataset/LIMUC/train_and_validation_sets', f'{BASE_DIR}/Dataset/LIMUC/test_set'], 'LIMUC')
            all_imgs = tmc_imgs + ntuh_imgs + limuc_imgs
            all_feats = tmc_feats + ntuh_feats + limuc_feats
            all_labels = tmc_labels + ntuh_labels + limuc_labels
            
            from sklearn.model_selection import train_test_split
            _, X_test_img, _, X_test_feat, _, y_test_label = train_test_split(
                all_imgs, all_feats, all_labels, test_size=0.2, random_state=42, stratify=all_labels
            )
        else:
            if dataset_name == 'NTUH':
                all_imgs, all_feats, all_labels, _ = load_all_images([f'{BASE_DIR}/Dataset+Code/MES classification_20250724'], 'NTUH')
            elif dataset_name == 'LIMUC':
                all_imgs, all_feats, all_labels, _ = load_all_images([f'{BASE_DIR}/Dataset/LIMUC/train_and_validation_sets', f'{BASE_DIR}/Dataset/LIMUC/test_set'], 'LIMUC')
                
            from sklearn.model_selection import train_test_split
            _, X_test_img, _, X_test_feat, _, y_test_label = train_test_split(
                all_imgs, all_feats, all_labels, test_size=0.2, random_state=42, stratify=all_labels
            )

        le = LabelEncoder()
        le.fit(all_labels) # Fit on all to ensure consistency
        y_test_encoded = le.transform(y_test_label)
        X_img_test = np.array(X_test_img)
        X_test_feat_arr = np.array(X_test_feat)

        for model_name in MODELS:
            save_dir = f"../../Result/{scenario_full}/{model_name}_Experiment"
            model_path = os.path.join(save_dir, f"{model_name}_hybrid.keras")
            agent_path = os.path.join(save_dir, f"{model_name}_agent.txt")
            scaler_path = os.path.join(save_dir, f"{model_name}_scaler.pkl")
            base_scaler_path = os.path.join(save_dir, "base_scaler.pkl")
            umap_model_path = os.path.join(save_dir, "umap_model.pkl")
            
            if not os.path.exists(model_path):
                print(f"  ⏭️ Skipping {model_name} (No .keras model found)")
                continue
                
            print(f"  🚀 Processing {model_name}...")
            
            try:
                # 1. Load deep model
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                
                # 2. Scale features and UMAP
                base_scaler = joblib.load(base_scaler_path)
                umap_reducer = joblib.load(umap_model_path)
                
                X_feat_test_scaled = base_scaler.transform(X_test_feat_arr)
                X_test_umap = umap_reducer.transform(X_feat_test_scaled)
                
                # 3. Predict Deep
                y_pred_proba_test = model.predict([X_img_test, X_feat_test_scaled, X_test_umap], verbose=0)
                y_pred_deep = np.argmax(y_pred_proba_test, axis=1)
                
                # 4. Agent Prediction
                clf = lgb.Booster(model_file=agent_path)
                agent_scaler = joblib.load(scaler_path)
                
                # Build agent features
                df_test_ag = pd.DataFrame(X_feat_test_scaled, columns=[f"f{i}" for i in range(20)])
                for i in range(y_pred_proba_test.shape[1]):
                    df_test_ag[f"prob_class_{i}"] = y_pred_proba_test[:, i]
                df_test_ag["confidence"] = np.max(y_pred_proba_test, axis=1)
                df_test_ag["umap_0"] = X_test_umap[:, 0]
                df_test_ag["umap_1"] = X_test_umap[:, 1]
                
                prob_cols = [f"prob_class_{i}" for i in range(y_pred_proba_test.shape[1])]
                features = prob_cols + ["confidence", "umap_0", "umap_1"] + [f"f{i}" for i in range(20)]
                
                X_te = agent_scaler.transform(df_test_ag[features].values)
                y_pred_proba_agent = clf.predict(X_te) # Booster predict outputs probas for multiclass
                
                # 5. Smart Delegation
                conf_test = np.max(y_pred_proba_test, axis=1)
                conf_agent = np.max(y_pred_proba_agent, axis=1)
                y_pred_agent = np.argmax(y_pred_proba_agent, axis=1)
                
                THRESHOLD = 0.55
                low_conf_mask = (conf_test < THRESHOLD) & (conf_agent > conf_test)
                y_pred_hybrid = np.where(low_conf_mask, y_pred_agent, y_pred_deep)
                
                # 6. Calculate Metrics
                y_true = y_test_encoded
                metrics = {
                    'Model': model_name,
                    'Base_Accuracy': float(accuracy_score(y_true, y_pred_deep)),
                    'Base_Precision': float(precision_score(y_true, y_pred_deep, average='macro', zero_division=0)),
                    'Base_Recall': float(recall_score(y_true, y_pred_deep, average='macro', zero_division=0)),
                    'Base_F1-Score': float(f1_score(y_true, y_pred_deep, average='macro', zero_division=0)),
                    'Base_QWK': float(cohen_kappa_score(y_true, y_pred_deep, weights='quadratic')),
                    'Base_ConfusionMatrix': confusion_matrix(y_true, y_pred_deep).tolist(),
                    
                    'Hybrid_Accuracy': float(accuracy_score(y_true, y_pred_hybrid)),
                    'Hybrid_Precision': float(precision_score(y_true, y_pred_hybrid, average='macro', zero_division=0)),
                    'Hybrid_Recall': float(recall_score(y_true, y_pred_hybrid, average='macro', zero_division=0)),
                    'Hybrid_F1-Score': float(f1_score(y_true, y_pred_hybrid, average='macro', zero_division=0)),
                    'Hybrid_QWK': float(cohen_kappa_score(y_true, y_pred_hybrid, weights='quadratic')),
                    'Hybrid_ConfusionMatrix': confusion_matrix(y_true, y_pred_hybrid).tolist()
                }
                
                # Save
                metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                    
                print(f"    ✅ Recovered! Base: {metrics['Base_Accuracy']:.4f} | Hybrid: {metrics['Hybrid_Accuracy']:.4f}")
                
            except Exception as e:
                print(f"    ⚠️ Failed to recover {model_name}: {e}")

if __name__ == "__main__":
    main()
