import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from tabulate import tabulate
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import joblib
import lightgbm as lgb
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images, load_tmc_ucm
from src.train import focal_loss

def load_test_data(dataset_name, base_dir=".."):
    DATASET_PATHS = {
        'NTUH':    [f'{base_dir}/Dataset+Code/MES classification_20250313', f'{base_dir}/Dataset+Code/MES classification_20250724'],
        'LIMUC':   [f'{base_dir}/Dataset/LIMUC/train_and_validation_sets', f'{base_dir}/Dataset/LIMUC/test_set'],
        'TMC-UCM': [f'{base_dir}/Dataset/TMC-UCM/images']
    }
    TMC_UCM_ROOT = f'{base_dir}/Dataset/TMC-UCM'
    
    if dataset_name == 'Unified':
        tmc_imgs, tmc_feats, tmc_labels, tmc_paths = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None)
        ntuh_imgs, ntuh_feats, ntuh_labels, ntuh_paths = load_all_images(DATASET_PATHS['NTUH'], 'NTUH')
        limuc_imgs, limuc_feats, limuc_labels, limuc_paths = load_all_images(DATASET_PATHS['LIMUC'], 'LIMUC')
        all_imgs   = tmc_imgs   + ntuh_imgs   + limuc_imgs
        all_feats  = tmc_feats  + ntuh_feats  + limuc_feats
        all_labels = tmc_labels + ntuh_labels + limuc_labels
        
        # Test split logic must match training
        from sklearn.model_selection import train_test_split
        _, X_img, _, X_feat, _, y_label = train_test_split(all_imgs, all_feats, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
        
    elif dataset_name == 'LIMUC':
        X_img, X_feat, y_label, _ = load_all_images([DATASET_PATHS['LIMUC'][1]], dataset_name)
    elif dataset_name == 'TMC-UCM':
        X_img, X_feat, y_label, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Test')
    elif dataset_name == 'NTUH':
        all_imgs, all_feats, all_labels, _ = load_all_images(DATASET_PATHS['NTUH'], dataset_name)
        from sklearn.model_selection import train_test_split
        _, X_img, _, X_feat, _, y_label = train_test_split(all_imgs, all_feats, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    else:
        return None, None, None
        
    return np.array(X_img, dtype=np.float32), np.array(X_feat, dtype=np.float32), y_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="../Result", help="Directory containing the Intra folders")
    parser.add_argument("--base_dir", type=str, default="/home/D13K48009/raid/Clara/new_drive", help="Base directory containing the Dataset/ folder")
    args = parser.parse_args()

    models_list = ["ResNet-50", "DenseNet-121", "EfficientNet-B4", "ConvNeXt-Tiny", "ViT-B-16"]
    datasets = ["TMC-UCM", "LIMUC", "NTUH", "Unified"]
    
    le = LabelEncoder()
    le.fit(["MES0", "MES1", "MES2", "MES3"])
    
    out_dir = os.path.join(args.models_dir, "Ensemble_ROC_Results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n🚀 Fast ROC & Ensemble Evaluator")
    print(f"Results will be saved to: {out_dir}\n")
    
    ensemble_results = []
    
    for dataset in datasets:
        print(f"==================================================")
        print(f"📊 Processing Dataset: {dataset}")
        
        if dataset == "Unified":
            d_dir = os.path.join(args.models_dir, "Unified")
        else:
            d_dir = os.path.join(args.models_dir, f"Intra_{dataset}")
            
        if not os.path.exists(d_dir):
            print(f"⏭️  Skipping {dataset} (Directory {d_dir} not found)")
            continue
            
        print("Loading test data...")
        X_img_test, X_feat_test, y_test_label = load_test_data(dataset, base_dir=args.base_dir)
        if X_img_test is None:
            continue
            
        y_true = le.transform(y_test_label)
        Y_bin = label_binarize(y_true, classes=[0,1,2,3])
        
        plt.figure(figsize=(8, 6))
        
        all_y_preds = [] # For majority voting
        
        for model_name in models_list:
            exp_dir = os.path.join(d_dir, f"{model_name}_Experiment")
            keras_path = os.path.join(exp_dir, f"{model_name}_hybrid.keras")
            if not os.path.exists(keras_path):
                keras_path = os.path.join(exp_dir, f"{model_name}_hybrid.h5")
                
            if not os.path.exists(keras_path):
                print(f"  [!] Missing {model_name}, skipping...")
                continue
                
            print(f"  🧠 Inferencing {model_name}...")
            
            # Dynamic prep
            if model_name == 'ResNet-50': from tensorflow.keras.applications.resnet50 import preprocess_input as prep
            elif model_name == 'DenseNet-121': from tensorflow.keras.applications.densenet import preprocess_input as prep
            elif model_name == 'EfficientNet-B4': from tensorflow.keras.applications.efficientnet import preprocess_input as prep
            elif model_name == 'ConvNeXt-Tiny': from tensorflow.keras.applications.convnext import preprocess_input as prep
            else: prep = lambda img: (img / 127.5) - 1.0

            custom_objs = {
                'KerasLayer': hub.KerasLayer, 'preprocess_input': prep, '<lambda>': prep,
                'resnet50_preprocess': prep, 'densenet_preprocess': prep, 'efficientnet_preprocess': prep,
                'convnext_preprocess': prep, 'vit_preprocess': prep, 'focal_loss_fixed': focal_loss(gamma=2.5, alpha=0.25)
            }
            
            try:
                model = load_model(keras_path, compile=False, custom_objects=custom_objs)
            except Exception as e:
                print(f"  [!] Error loading {model_name}: {e}")
                continue
                
            # UMAP & Scaler
            base_scaler = joblib.load(os.path.join(exp_dir, "base_scaler.pkl"))
            umap_reducer = joblib.load(os.path.join(exp_dir, "umap_model.pkl"))
            
            X_feat_test_scaled = base_scaler.transform(X_feat_test)
            X_test_umap = umap_reducer.transform(X_feat_test_scaled)
            
            # Auto-resize
            expected_shape = model.input[0].shape[1:3]
            if tuple(X_img_test.shape[1:3]) != tuple(expected_shape):
                X_img_eval = tf.image.resize(X_img_test, expected_shape).numpy()
            else:
                X_img_eval = X_img_test
                
            y_pred_proba = model.predict([X_img_eval, X_feat_test_scaled, X_test_umap], verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            all_y_preds.append(y_pred)
            
            # Macro-Average ROC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(4):
                fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(4):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 4
            
            macro_auc = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, lw=2, label=f"{model_name} (AUC = {macro_auc:.3f})")

        if len(all_y_preds) == 0:
            print("  [!] No models loaded for dataset. Skipping ROC.")
            continue
            
        # Draw ROC
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Macro-Average ROC Curve - {dataset}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(out_dir, f"ROC_Curve_{dataset}.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"  ✅ Saved ROC Curve to {roc_path}")
        
        # Majority Voting Ensemble
        print(f"  🗳️ Calculating Majority Voting...")
        all_y_preds = np.array(all_y_preds) # Shape: (Models, Samples)
        ensemble_pred, _ = stats.mode(all_y_preds, axis=0)
        ensemble_pred = ensemble_pred.flatten()
        
        acc = accuracy_score(y_true, ensemble_pred)
        prec = precision_score(y_true, ensemble_pred, average='macro')
        rec = recall_score(y_true, ensemble_pred, average='macro')
        f1 = f1_score(y_true, ensemble_pred, average='macro')
        qwk = cohen_kappa_score(y_true, ensemble_pred, weights='quadratic')
        
        ensemble_results.append([dataset, f"{acc*100:.2f}%", f"{prec*100:.2f}%", f"{rec*100:.2f}%", f"{f1*100:.2f}%", f"{qwk:.4f}"])
        print(f"  🏆 Ensemble Accuracy: {acc*100:.2f}%")

    print(f"\n==================================================")
    print("🏆 FINAL MAJORITY VOTING ENSEMBLE RESULTS")
    headers = ["Dataset", "Accuracy", "Precision", "Recall", "F1", "QWK"]
    print(tabulate(ensemble_results, headers=headers, tablefmt="grid"))
    
    # Save ensemble results to CSV
    df = pd.DataFrame(ensemble_results, columns=headers)
    csv_path = os.path.join(out_dir, "Ensemble_Voting_Summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved Ensemble Summary to {csv_path}")

if __name__ == "__main__":
    main()
