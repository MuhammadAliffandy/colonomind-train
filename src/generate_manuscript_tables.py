import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

from dgx_dataloader import load_all_images, load_tmc_ucm
import tensorflow_hub as hub
import dgx_models  # For custom keras functions

np.random.seed(42)

# --- Helper Classes and Functions ---
class LGBMWrapper:
    def __init__(self, booster):
        self.booster = booster
    def predict_proba(self, X):
        return self.booster.predict(X)

def get_hybrid_proba(deep_proba, agent_proba, threshold=0.50):
    final_proba = deep_proba.copy()
    conf = np.max(deep_proba, axis=1)
    low_conf = conf < threshold
    final_proba[low_conf] = agent_proba[low_conf]
    return final_proba

def calc_ece(y_true, y_proba, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = predictions == y_true
    
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(float).mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece[0]

def calc_secondary_metrics(y_true, y_pred, num_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    metrics = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0
        
        metrics[i] = {
            'Sensitivity': sens,
            'Specificity': spec,
            'PPV': ppv,
            'NPV': npv,
            'F1': f1
        }
    return metrics

def bootstrap_metric(y_true, y_pred, metric_func, n_iterations=1000):
    scores = []
    n_size = int(len(y_true))
    for i in range(n_iterations):
        indices = np.random.randint(0, n_size, n_size)
        try:
            score = metric_func(y_true[indices], y_pred[indices])
            if not np.isnan(score):
                scores.append(score)
        except:
            continue
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    scores.sort()
    lower = scores[int(0.025 * len(scores))]
    upper = scores[int(0.975 * len(scores))]
    mean_score = np.mean(scores)
    return mean_score, lower, upper

def draw_confusion_matrix(y_true, y_pred, dataset_name, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['MES 0', 'MES 1', 'MES 2', 'MES 3'],
                yticklabels=['MES 0', 'MES 1', 'MES 2', 'MES 3'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {dataset_name} (Score-weighted)', fontsize=14)
    out_path = os.path.join(save_dir, f'Fig_CM_{dataset_name}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved {out_path}")

def draw_table6_plot(table6_data, save_dir):
    # Plotting for Table 6: classes detected successfully at different agreement thresholds
    # table6_data format: list of dicts: [{'Dataset': d, 'Threshold': t, 'Class': c, 'Accuracy': acc}, ...]
    df = pd.DataFrame(table6_data)
    if len(df) == 0:
        return
        
    g = sns.catplot(data=df, x='Accuracy', y='Class', hue='Threshold', col='Dataset', 
                    kind='bar', height=5, aspect=1.2, palette='Set2')
    
    # Add a baseline/target line (e.g., 80% accuracy threshold for "successful detection")
    target_accuracy = 0.80
    for ax in g.axes.flat:
        ax.axvline(target_accuracy, color='r', linestyle='--', label=f'Target ({target_accuracy*100}%)')
        ax.set_xlim(0, 1.05)
    
    plt.suptitle("Statistical Analysis for Model Agreement", y=1.05, fontsize=16)
    out_path = os.path.join(save_dir, 'Fig_4_Agreement_Stats.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="..")
    parser.add_argument("--models_dir", type=str, default="../Result/Intra_TMC-UCM")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    
    save_dir = os.path.join(args.base_dir, "Manuscript_Results")
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = ['ResNet-50', 'DenseNet-121', 'EfficientNet-B4', 'ConvNeXt-Tiny', 'ViT-B-16']
    
    models = {}
    scalers = {}
    agents = {}
    
    print("Loading models and agents...")
    for model_name in model_names:
        exp_dir = os.path.join(args.models_dir, f"{model_name}_Experiment")
        model_path = os.path.join(exp_dir, f"{model_name}_hybrid.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(exp_dir, f"{model_name}_hybrid.h5")
            
        if model_name == 'ResNet-50':
            from tensorflow.keras.applications.resnet50 import preprocess_input as prep
        elif model_name == 'DenseNet-121':
            from tensorflow.keras.applications.densenet import preprocess_input as prep
        elif model_name == 'EfficientNet-B4':
            from tensorflow.keras.applications.efficientnet import preprocess_input as prep
        elif model_name == 'ConvNeXt-Tiny':
            from tensorflow.keras.applications.convnext import preprocess_input as prep
        else:
            prep = lambda img: (img / 127.5) - 1.0

        custom_objs = {
            'KerasLayer': hub.KerasLayer,
            'preprocess_input': prep,
            '<lambda>': prep,
            'resnet50_preprocess': prep,
            'densenet_preprocess': prep,
            'efficientnet_preprocess': prep,
            'convnext_preprocess': prep,
            'vit_preprocess': prep,
            'Custom>resnet50_preprocess': prep,
            'Custom>densenet_preprocess': prep,
            'Custom>efficientnet_preprocess': prep,
            'Custom>convnext_preprocess': prep,
            'Custom>vit_preprocess': prep
        }
        
        keras_model = load_model(model_path, compile=False, custom_objects=custom_objs)
        scaler_ag = joblib.load(os.path.join(exp_dir, f"{model_name}_scaler.pkl"))
        agent_model = lgb.Booster(model_file=os.path.join(exp_dir, f"{model_name}_agent.txt"))
        
        models[model_name] = keras_model
        scalers[model_name] = scaler_ag
        agents[model_name] = LGBMWrapper(agent_model)
        
    le = LabelEncoder()
    le.fit(['MES0', 'MES1', 'MES2', 'MES3'])
    
    # Global UMAP and Scaler for feature extraction
    global_umap = joblib.load(os.path.join(args.models_dir, "ResNet-50_Experiment", "umap_model.pkl"))
    global_scaler = joblib.load(os.path.join(args.models_dir, "ResNet-50_Experiment", "base_scaler.pkl"))
    
    datasets = {}
    print("Loading TMC-UCM Test dataset...")
    tmc_imgs, tmc_feats, tmc_labels, _ = load_tmc_ucm(f'{args.base_dir}/Dataset/TMC-UCM', split_filter='Test')
    datasets['TMC-UCM'] = (tmc_imgs, tmc_feats, tmc_labels)
    
    print("Loading NTUH dataset...")
    ntuh_paths = [f'{args.base_dir}/Dataset+Code/MES classification_20250313', f'{args.base_dir}/Dataset+Code/MES classification_20250724']
    ntuh_imgs, ntuh_feats, ntuh_labels, _ = load_all_images(ntuh_paths, 'NTUH')
    datasets['NTUH'] = (ntuh_imgs, ntuh_feats, ntuh_labels)
    
    print("Loading LIMUC dataset...")
    limuc_paths = [f'{args.base_dir}/Dataset/LIMUC/train_and_validation_sets', f'{args.base_dir}/Dataset/LIMUC/test_set']
    limuc_imgs, limuc_feats, limuc_labels, _ = load_all_images(limuc_paths, 'LIMUC')
    datasets['LIMUC'] = (limuc_imgs, limuc_feats, limuc_labels)
    
    # Initialize DataFrames for Tables
    table1_rows = []
    table2_rows = []
    table3_rows = [] # NTUH
    table4_rows = [] # LIMUC
    table5_rows = [] # TMC-UCM
    table6_data = [] # Agreement

    for dataset_name, (imgs, feats, labels) in datasets.items():
        print(f"\\nProcessing {dataset_name} for Tables & Figures...")
        X_img = np.array(imgs, dtype=np.float32)
        X_feat = np.array(feats)
        y_true = le.transform(labels)
        
        X_feat_scaled = global_scaler.transform(X_feat)
        X_umap = global_umap.transform(X_feat_scaled)
        
        all_hybrid_probas = []
        all_hybrid_preds = []
        
        for model_name in model_names:
            print(f"  Running inference for {model_name}...")
            keras_model = models[model_name]
            agent_wrapper = agents[model_name]
            scaler_ag = scalers[model_name]
            
            deep_proba = keras_model.predict([X_img, X_feat_scaled, X_umap], verbose=0)
            
            df = pd.DataFrame(X_feat_scaled, columns=[f"f{i}" for i in range(20)])
            df["confidence"] = np.max(deep_proba, axis=1)
            df["umap_0"] = X_umap[:, 0]
            df["umap_1"] = X_umap[:, 1]
            features = ["confidence", "umap_0", "umap_1"] + [f"f{i}" for i in range(20)]
            X_ag = scaler_ag.transform(df[features].values)
            agent_proba = agent_wrapper.predict_proba(X_ag)
            
            hybrid_proba = get_hybrid_proba(deep_proba, agent_proba, args.threshold)
            hybrid_preds = np.argmax(hybrid_proba, axis=1)
            
            all_hybrid_probas.append(hybrid_proba)
            all_hybrid_preds.append(hybrid_preds)
            
            # --- Table 1: Primary Outcome Average (Acc, QWK) ---
            acc_func = lambda yt, yp: accuracy_score(yt, yp)
            acc_mean, acc_low, acc_high = bootstrap_metric(y_true, hybrid_preds, acc_func)
            
            qwk_func = lambda yt, yp: cohen_kappa_score(yt, yp, weights='quadratic')
            qwk_mean, qwk_low, qwk_high = bootstrap_metric(y_true, hybrid_preds, qwk_func)
            
            table1_rows.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy (95% CI)': f"{acc_mean:.3f} ({acc_low:.3f}-{acc_high:.3f})",
                'QWK (95% CI)': f"{qwk_mean:.3f} ({qwk_low:.3f}-{qwk_high:.3f})"
            })
            
            # --- Table 2: Primary Outcome Per Class (Acc) ---
            for cls_idx, cls_name in enumerate(['MES 0', 'MES 1', 'MES 2', 'MES 3']):
                cls_mask = y_true == cls_idx
                if np.sum(cls_mask) > 0:
                    acc_cls_func = lambda yt, yp: accuracy_score(yt, yp)
                    acc_cls_mean, acc_cls_low, acc_cls_high = bootstrap_metric(y_true[cls_mask], hybrid_preds[cls_mask], acc_cls_func)
                    
                    table2_rows.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Class': cls_name,
                        'Accuracy (95% CI)': f"{acc_cls_mean:.3f} ({acc_cls_low:.3f}-{acc_cls_high:.3f})"
                    })
                    
            # --- Table 3, 4, 5: Secondary Outcomes (Sens, Spec, PPV, NPV, F1, ECE) ---
            sec_metrics = calc_secondary_metrics(y_true, hybrid_preds)
            ece_val = calc_ece(y_true, hybrid_proba)
            
            for cls_idx, cls_name in enumerate(['MES 0', 'MES 1', 'MES 2', 'MES 3']):
                row = {
                    'Model': model_name,
                    'Class': cls_name,
                    'Sensitivity': f"{sec_metrics[cls_idx]['Sensitivity']:.3f}",
                    'Specificity': f"{sec_metrics[cls_idx]['Specificity']:.3f}",
                    'PPV': f"{sec_metrics[cls_idx]['PPV']:.3f}",
                    'NPV': f"{sec_metrics[cls_idx]['NPV']:.3f}",
                    'F1 Score': f"{sec_metrics[cls_idx]['F1']:.3f}",
                    'ECE': f"{ece_val:.3f}"
                }
                if dataset_name == 'NTUH':
                    table3_rows.append(row)
                elif dataset_name == 'LIMUC':
                    table4_rows.append(row)
                elif dataset_name == 'TMC-UCM':
                    table5_rows.append(row)
        
        # --- Table 6: Model Agreement Thresholds (3/5, 4/5, 5/5) ---
        all_hybrid_preds_arr = np.array(all_hybrid_preds) # shape: (5, N)
        for cls_idx, cls_name in enumerate(['MES 0', 'MES 1', 'MES 2', 'MES 3']):
            cls_mask = y_true == cls_idx
            if np.sum(cls_mask) > 0:
                y_true_cls = y_true[cls_mask]
                preds_cls = all_hybrid_preds_arr[:, cls_mask] # shape: (5, N_cls)
                
                # Check agreement: sum of correctly predicting models for each instance
                correct_votes = np.sum(preds_cls == cls_idx, axis=0) # shape: (N_cls,)
                
                for thresh, thresh_name in zip([3, 4, 5], ['3/5', '4/5', '5/5']):
                    successful_detects = np.sum(correct_votes >= thresh)
                    total_instances = len(y_true_cls)
                    accuracy = successful_detects / total_instances
                    
                    table6_data.append({
                        'Dataset': dataset_name,
                        'Class': cls_name,
                        'Threshold': thresh_name,
                        'Successful_Detects': successful_detects,
                        'Total_Instances': total_instances,
                        'Accuracy': accuracy
                    })
        
        # --- Figure 1, 2, 3: Confusion Matrix (Score-weighted ensemble) ---
        avg_probas = np.mean(all_hybrid_probas, axis=0)
        sw_preds = np.argmax(avg_probas, axis=1)
        draw_confusion_matrix(y_true, sw_preds, dataset_name, save_dir)
        
    print("\\nSaving all tables to CSV...")
    pd.DataFrame(table1_rows).to_csv(os.path.join(save_dir, 'Table_1_Primary_Average.csv'), index=False)
    pd.DataFrame(table2_rows).to_csv(os.path.join(save_dir, 'Table_2_Primary_PerClass.csv'), index=False)
    pd.DataFrame(table3_rows).to_csv(os.path.join(save_dir, 'Table_3_Secondary_NTUH.csv'), index=False)
    pd.DataFrame(table4_rows).to_csv(os.path.join(save_dir, 'Table_4_Secondary_LIMUC.csv'), index=False)
    pd.DataFrame(table5_rows).to_csv(os.path.join(save_dir, 'Table_5_Secondary_TMC-UCM.csv'), index=False)
    
    df_t6 = pd.DataFrame(table6_data)
    df_t6.to_csv(os.path.join(save_dir, 'Table_6_Agreement_Thresholds.csv'), index=False)
    
    print("Generating Figure 4 (Statistical Analysis for Table 6)...")
    draw_table6_plot(table6_data, save_dir)
    
    print("\\n✅ All 6 Tables and 4 Figures generated successfully in Manuscript_Results!")

if __name__ == "__main__":
    main()
