import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import tensorflow as tf
import lightgbm as lgb
from collections import Counter
from tqdm import tqdm

from dgx_dataloader import load_all_images, load_tmc_ucm
import dgx_models  # Import to register all custom keras functions

# Seed for reproducibility
np.random.seed(42)

def compute_macro_roc(y_true, y_proba, num_classes=4):
    y_true_cat = to_categorical(y_true, num_classes=num_classes)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(num_classes):
        fpr[i], t, _ = roc_curve(y_true_cat[:, i], y_proba[:, i])
        tpr[i] = t
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc

def bootstrap_qwk(y_true, y_pred, n_iterations=1000):
    scores = []
    n_size = int(len(y_true))
    for i in range(n_iterations):
        indices = np.random.randint(0, n_size, n_size)
        score = cohen_kappa_score(y_true[indices], y_pred[indices], weights='quadratic')
        if not np.isnan(score):
            scores.append(score)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    scores.sort()
    lower = scores[int(0.025 * len(scores))]
    upper = scores[int(0.975 * len(scores))]
    mean_score = np.mean(scores)
    return mean_score, lower, upper

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

def plot_roc_figure(dataset_name, y_true, model_probas, save_dir):
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(model_probas))
    
    for (model_name, y_proba), color in zip(model_probas.items(), colors):
        fpr, tpr, auc_score = compute_macro_roc(y_true, y_proba)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'Macro-average ROC Curve on {dataset_name}', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    
    out_path = os.path.join(save_dir, f'ROC_{dataset_name}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved {out_path}")

def draw_forest_plot(forest_data, save_dir):
    """
    forest_data format:
    {
       'NTUH': [
           {'method': 'Best single', 'mean': 0.8, 'lower': 0.75, 'upper': 0.85},
           ...
       ],
       'LIMUC': [...]
    }
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = []
    y_labels = []
    means = []
    errors_lower = []
    errors_upper = []
    colors = []
    
    current_y = 0
    # Spacing parameters
    group_spacing = 1.5
    item_spacing = 0.5
    
    # Define colors
    color_map = {
        'Best single': 'gray',
        'Majority voting': '#1f77b4', # blue
        'Score-weighted': '#ff7f0e'   # orange
    }
    
    datasets = list(forest_data.keys())
    for i, dataset in enumerate(datasets):
        # Add dataset label as text
        ax.text(0.50, current_y + 0.5, dataset, fontsize=12, fontweight='bold', va='center')
        
        items = forest_data[dataset]
        for item in items:
            method = item['method']
            y_pos.append(current_y)
            y_labels.append(method)
            means.append(item['mean'])
            errors_lower.append(item['mean'] - item['lower'])
            errors_upper.append(item['upper'] - item['mean'])
            colors.append(color_map.get(method, 'black'))
            current_y -= item_spacing
            
        current_y -= group_spacing
        
    # Plot error bars and points
    for i in range(len(y_pos)):
        ax.errorbar(means[i], y_pos[i], xerr=[[errors_lower[i]], [errors_upper[i]]], 
                    fmt='o' if y_labels[i] != 'Score-weighted' else 's', 
                    color=colors[i], ecolor=colors[i], elinewidth=2, capsize=0, markersize=10)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=11)
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    
    ax.set_xlabel('Quadratic-weighted kappa (95% CI)', fontsize=12)
    ax.set_title('A Adjudication vs best single model — agreement (QWK)', loc='left', fontsize=14, fontweight='bold', pad=20)
    
    # Optional: adjust x-axis limits if needed based on data
    valid_means = [m for m in means if not np.isnan(m)]
    if valid_means:
        min_x = min([m - el for m, el in zip(means, errors_lower) if not np.isnan(m)]) - 0.05
        max_x = max([m + eu for m, eu in zip(means, errors_upper) if not np.isnan(m)]) + 0.05
        if not np.isnan(min_x) and not np.isnan(max_x):
            ax.set_xlim(min_x, max_x)
    
    out_path = os.path.join(save_dir, 'Forest_Plot_QWK.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="..")
    parser.add_argument("--models_dir", type=str, default="../Result/Intra_TMC-UCM")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--only_qwk", action="store_true", help="Only generate QWK Forest Plot (skips ROC and TMC-UCM)")
    args = parser.parse_args()
    
    save_dir = os.path.join(args.base_dir, "Manuscript_Figures")
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = ['ResNet-50', 'DenseNet-121', 'EfficientNet-B4', 'ConvNeXt-Tiny', 'ViT-B-16']
    
    models = {}
    scalers = {}
    agents = {}
    
    print("Loading models...")
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
        
    # Global UMAP and Scaler for feature extraction
    global_umap = joblib.load(os.path.join(args.models_dir, "ResNet-50_Experiment", "umap_model.pkl"))
    global_scaler = joblib.load(os.path.join(args.models_dir, "ResNet-50_Experiment", "base_scaler.pkl"))
    
    le = LabelEncoder()
    le.fit(['MES0', 'MES1', 'MES2', 'MES3'])
    
    # Load Datasets
    datasets = {}
    
    if not args.only_qwk:
        print("Loading TMC-UCM Test dataset...")
        tmc_imgs, tmc_feats, tmc_labels, _ = load_tmc_ucm(f'{args.base_dir}/Dataset/TMC-UCM', split_filter='Test')
        datasets['TMC-UCM'] = (tmc_imgs, tmc_feats, tmc_labels)
    
    print("Loading NTUH dataset...")
    ntuh_paths = [f'{args.base_dir}/Dataset+Code/MES classification_20250313', f'{args.base_dir}/Dataset+Code/MES classification_20250724']
    ntuh_imgs, ntuh_feats, ntuh_labels, _ = load_all_images(ntuh_paths, 'NTUH')
    
    print("Loading LIMUC dataset...")
    limuc_paths = [f'{args.base_dir}/Dataset/LIMUC/train_and_validation_sets', f'{args.base_dir}/Dataset/LIMUC/test_set']
    limuc_imgs, limuc_feats, limuc_labels, _ = load_all_images(limuc_paths, 'LIMUC')
    
    datasets['NTUH'] = (ntuh_imgs, ntuh_feats, ntuh_labels)
    datasets['LIMUC'] = (limuc_imgs, limuc_feats, limuc_labels)
    
    forest_data = {}
    
    for dataset_name, (imgs, feats, labels) in datasets.items():
        print(f"\\nProcessing {dataset_name}...")
        X_img = np.array(imgs, dtype=np.float32)
        X_feat = np.array(feats)
        y_true = le.transform(labels)
        
        X_feat_scaled = global_scaler.transform(X_feat)
        X_umap = global_umap.transform(X_feat_scaled)
        
        model_probas = {}
        all_hybrid_probas = []
        all_hybrid_preds = []
        
        for model_name in model_names:
            keras_model = models[model_name]
            agent_wrapper = agents[model_name]
            scaler_ag = scalers[model_name]
            
            # Deep Probas
            deep_proba = keras_model.predict([X_img, X_feat_scaled, X_umap], verbose=0)
            
            # Agent Probas
            df = pd.DataFrame(X_feat_scaled, columns=[f"f{i}" for i in range(20)])
            df["confidence"] = np.max(deep_proba, axis=1)
            df["umap_0"] = X_umap[:, 0]
            df["umap_1"] = X_umap[:, 1]
            features = ["confidence", "umap_0", "umap_1"] + [f"f{i}" for i in range(20)]
            X_ag = scaler_ag.transform(df[features].values)
            agent_proba = agent_wrapper.predict_proba(X_ag)
            
            # Hybrid
            hybrid_proba = get_hybrid_proba(deep_proba, agent_proba, args.threshold)
            model_probas[model_name] = hybrid_proba
            all_hybrid_probas.append(hybrid_proba)
            all_hybrid_preds.append(np.argmax(hybrid_proba, axis=1))
            
        # Plot ROC for this dataset if not only_qwk
        if not args.only_qwk:
            plot_roc_figure(dataset_name, y_true, model_probas, save_dir)
        
        # Calculate QWK metrics for Forest Plot (only for NTUH and LIMUC usually, but let's do NTUH and LIMUC specifically as requested)
        if dataset_name in ['NTUH', 'LIMUC']:
            print(f"Calculating QWK CI via bootstrap for {dataset_name}...")
            # 1. Best single (find the one with highest QWK)
            best_qwk = -1
            best_single_preds = None
            for preds in all_hybrid_preds:
                qwk = cohen_kappa_score(y_true, preds, weights='quadratic')
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_single_preds = preds
            
            best_mean, best_low, best_high = bootstrap_qwk(y_true, best_single_preds)
            
            # 2. Majority Voting (threshold=3)
            majority_preds = []
            for i in range(len(y_true)):
                votes = [preds[i] for preds in all_hybrid_preds]
                counter = Counter(votes)
                most_common, count = counter.most_common(1)[0]
                majority_preds.append(most_common)
            majority_preds = np.array(majority_preds)
            maj_mean, maj_low, maj_high = bootstrap_qwk(y_true, majority_preds)
            
            # 3. Score-weighted (Soft Voting - average probas)
            avg_probas = np.mean(all_hybrid_probas, axis=0)
            score_weighted_preds = np.argmax(avg_probas, axis=1)
            sw_mean, sw_low, sw_high = bootstrap_qwk(y_true, score_weighted_preds)
            
            forest_data[dataset_name] = [
                {'method': 'Best single', 'mean': best_mean, 'lower': best_low, 'upper': best_high},
                {'method': 'Majority voting', 'mean': maj_mean, 'lower': maj_low, 'upper': maj_high},
                {'method': 'Score-weighted', 'mean': sw_mean, 'lower': sw_low, 'upper': sw_high}
            ]
            
    # Generate Forest Plot
    draw_forest_plot(forest_data, save_dir)
    print("✅ All manuscript figures generated successfully!")

if __name__ == "__main__":
    main()
