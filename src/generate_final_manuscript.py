import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from dgx_dataloader import load_all_images, load_tmc_ucm
import dgx_models  # For custom keras functions

np.random.seed(42)

# ── Retry wrapper for I/O operations (DGX storage can be flaky) ──
def retry_io(func, max_retries=5, delay=10):
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except (OSError, IOError) as e:
            if attempt < max_retries:
                print(f"⚠️  I/O Error (attempt {attempt}/{max_retries}): {e} — Retrying in {delay}s")
                time.sleep(delay)
            else:
                raise

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
        
        metrics[i] = {'Sensitivity': sens, 'Specificity': spec, 'PPV': ppv, 'NPV': npv, 'F1': f1}
    return metrics

def bootstrap_all_metrics(y_true, y_pred, n_iterations=1000):
    n_size = int(len(y_true))
    metrics_list = []
    for i in range(n_iterations):
        indices = np.random.randint(0, n_size, n_size)
        yt, yp = y_true[indices], y_pred[indices]
        try:
            acc = accuracy_score(yt, yp) * 100
            prec = precision_score(yt, yp, average='macro', zero_division=0) * 100
            rec = recall_score(yt, yp, average='macro', zero_division=0) * 100
            f1 = f1_score(yt, yp, average='macro', zero_division=0) * 100
            qwk = cohen_kappa_score(yt, yp, weights='quadratic')
            if not np.isnan(qwk):
                metrics_list.append((acc, prec, rec, f1, qwk))
        except:
            continue
    if len(metrics_list) == 0:
        return [0]*5, [0]*5, [0]*5
    
    metrics_arr = np.array(metrics_list)
    means = np.mean(metrics_arr, axis=0)
    lowers = []
    uppers = []
    for col in range(5):
        sorted_col = np.sort(metrics_arr[:, col])
        lowers.append(sorted_col[int(0.025 * len(sorted_col))])
        uppers.append(sorted_col[int(0.975 * len(sorted_col))])
    return means, lowers, uppers

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

# --- Plotting Functions ---
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
    retry_io(lambda: plt.savefig(out_path, bbox_inches='tight', dpi=300))
    plt.close()
    print(f"  ✅ Saved CM for {dataset_name}")

def plot_combined_roc_panels(all_roc_data, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    dataset_order = ['NTUH', 'TMC-UCM', 'LIMUC', 'Unified']
    
    for i, dataset_name in enumerate(dataset_order):
        ax = axes[i]
        if dataset_name not in all_roc_data:
            ax.set_title(f"No Data for {dataset_name}")
            continue
            
        model_probas = all_roc_data[dataset_name]['probas']
        y_true = all_roc_data[dataset_name]['y_true']
        
        colors = sns.color_palette("husl", len(model_probas))
        
        for (model_name, y_proba), color in zip(model_probas.items(), colors):
            fpr, tpr, auc_score = compute_macro_roc(y_true, y_proba)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {auc_score:.3f})')
            
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(dataset_name, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    out_path = os.path.join(save_dir, 'Fig_2_ROC_Combined_4Panels.png')
    retry_io(lambda: plt.savefig(out_path, bbox_inches='tight', dpi=300))
    plt.close()
    print(f"✅ Saved Combined ROC Grid to {out_path}")

def get_coverage_curve_data(y_true, y_pred, confidences, n_iterations=100, step=0.05):
    sorted_indices = np.argsort(confidences)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    coverages = np.arange(step, 1.0 + step, step)
    if coverages[-1] != 1.0:
        coverages = np.append(coverages, 1.0)
        
    n_total = len(y_true)
    results = []
    
    for cov in coverages:
        k = max(1, int(cov * n_total))
        y_t_k = y_true_sorted[:k]
        y_p_k = y_pred_sorted[:k]
        
        # Bootstrap QWK
        scores = []
        for _ in range(n_iterations):
            idx = np.random.randint(0, k, k)
            try:
                score = cohen_kappa_score(y_t_k[idx], y_p_k[idx], weights='quadratic')
                if not np.isnan(score):
                    scores.append(score)
            except:
                pass
        
        if len(scores) == 0:
            mean, lower, upper = 0.0, 0.0, 0.0
        else:
            scores.sort()
            mean = np.mean(scores)
            lower = scores[int(0.025 * len(scores))]
            upper = scores[int(0.975 * len(scores))]
            
        results.append((cov, mean, lower, upper))
    return results

def plot_coverage_curves_combined(coverage_data_dict, save_dir):
    datasets_to_plot = ['TMC-UCM', 'NTUH', 'LIMUC']
    valid_datasets = [d for d in datasets_to_plot if d in coverage_data_dict]
    
    if not valid_datasets:
        return
        
    fig, axes = plt.subplots(1, len(valid_datasets), figsize=(6 * len(valid_datasets), 5))
    if len(valid_datasets) == 1:
        axes = [axes]
        
    colors = {'Majority Voting': '#1f77b4', 'Score-weighted': '#ff7f0e'}
    
    for ax, dataset_name in zip(axes, valid_datasets):
        data = coverage_data_dict[dataset_name]
        for method, results in data.items():
            if method not in colors: continue
            
            covs = [r[0]*100 for r in results]
            means = [r[1] for r in results]
            lows = [r[2] for r in results]
            ups = [r[3] for r in results]
            
            ax.plot(covs, means, label=method, color=colors[method], lw=2)
            ax.fill_between(covs, lows, ups, color=colors[method], alpha=0.2)
            
        ax.set_title(dataset_name, fontsize=14)
        ax.set_xlabel('Coverage (%)', fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel('Quadratic-weighted kappa', fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
    plt.tight_layout()
    out_path = os.path.join(save_dir, 'Fig_5b_Coverage_Accuracy.png')
    retry_io(lambda: plt.savefig(out_path, bbox_inches='tight', dpi=300))
    plt.close()
    print(f"✅ Saved Fig 5b Coverage Curves")

# --- DataLoader Test Set Split Helper ---
def get_test_set(dataset_name, base_dir, cache_dir):
    if dataset_name == 'TMC-UCM':
        imgs, feats, labels, _ = load_tmc_ucm(f'{base_dir}/Dataset/TMC-UCM', split_filter='Test', cache_dir=cache_dir)
        return imgs, feats, labels
    elif dataset_name == 'LIMUC':
        imgs, feats, labels, _ = load_all_images([f'{base_dir}/Dataset/LIMUC/test_set'], 'LIMUC', cache_dir=cache_dir)
        return imgs, feats, labels
    elif dataset_name == 'NTUH':
        ntuh_paths = [f'{base_dir}/Dataset+Code/MES classification_20250313', f'{base_dir}/Dataset+Code/MES classification_20250724']
        imgs, feats, labels, paths = load_all_images(ntuh_paths, 'NTUH', cache_dir=cache_dir)
        _, X_test_img, _, X_test_feat, _, y_test_label, _, _ = train_test_split(
            imgs, feats, labels, paths, test_size=0.2, random_state=42, stratify=labels
        )
        return X_test_img, X_test_feat, y_test_label
    elif dataset_name == 'Unified':
        tmc_imgs, tmc_feats, tmc_labels, _ = load_tmc_ucm(f'{base_dir}/Dataset/TMC-UCM', split_filter=None, cache_dir=cache_dir)
        
        ntuh_paths = [f'{base_dir}/Dataset+Code/MES classification_20250313', f'{base_dir}/Dataset+Code/MES classification_20250724']
        ntuh_imgs, ntuh_feats, ntuh_labels, _ = load_all_images(ntuh_paths, 'NTUH', cache_dir=cache_dir)
        
        limuc_paths = [f'{base_dir}/Dataset/LIMUC/train_and_validation_sets', f'{base_dir}/Dataset/LIMUC/test_set']
        limuc_imgs, limuc_feats, limuc_labels, _ = load_all_images(limuc_paths, 'LIMUC', cache_dir=cache_dir)
        
        all_imgs = tmc_imgs + ntuh_imgs + limuc_imgs
        all_feats = tmc_feats + ntuh_feats + limuc_feats
        all_labels = tmc_labels + ntuh_labels + limuc_labels
        
        _, X_test_img, _, X_test_feat, _, y_test_label, _, _ = train_test_split(
            all_imgs, all_feats, all_labels, range(len(all_labels)), test_size=0.2, random_state=42, stratify=all_labels
        )
        return X_test_img, X_test_feat, y_test_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="..")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    
    save_dir = os.path.join(args.base_dir, "Manuscript_Final_Results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Final results will be saved to: {save_dir}")
    
    model_names = ['ResNet-50', 'DenseNet-121', 'EfficientNet-B4', 'ConvNeXt-Tiny', 'ViT-B-16']
    datasets = ['NTUH', 'TMC-UCM', 'LIMUC', 'Unified']
    
    # Global structure for the 4 manuscript tables
    manuscript_tables = {d: [] for d in datasets}
    
    # For Combined ROC Panels
    all_roc_data = {}
    
    # Store results to compile tables at the end
    results_primary = {m: {} for m in model_names}
    results_per_class = {m: {c: {} for c in ['MES 0', 'MES 1', 'MES 2', 'MES 3']} for m in model_names}
    
    table3_rows = [] # NTUH Secondary
    table4_rows = [] # LIMUC Secondary
    table5_rows = [] # TMC-UCM Secondary
    table6_rows = [] # Unified Secondary
    table7_data = [] # Agreement Thresholds
    
    coverage_data_dict = {}
    
    le = LabelEncoder()
    le.fit(['MES0', 'MES1', 'MES2', 'MES3'])
    
    for d in datasets:
        print(f"\n📦 Processing {d} (Intra-Domain)...")
        model_dir = f"{args.base_dir}/Result/Intra_{d}"
        if not os.path.exists(model_dir):
            print(f"  ⚠️ Skipping {d} because {model_dir} does not exist.")
            continue
            
        imgs, feats, labels = get_test_set(d, args.base_dir, args.cache_dir)
        X_img = np.array(imgs, dtype=np.float32)
        X_feat = np.array(feats)
        y_true = le.transform(labels)
        
        print(f"  Loaded {len(y_true)} test images.")
        
        # Load the global scaler & umap for this specific dataset training run
        try:
            global_scaler = joblib.load(os.path.join(model_dir, "base_scaler.pkl"))
            global_umap = joblib.load(os.path.join(model_dir, "umap_model.pkl"))
        except Exception as e:
            print(f"  ⚠️ Could not load scaler/umap for {d}: {e}")
            continue
            
        X_feat_scaled = global_scaler.transform(X_feat)
        X_umap = global_umap.transform(X_feat_scaled)
        
        all_hybrid_probas = []
        all_hybrid_preds = []
        model_probas_dict = {}
        
        for model_name in model_names:
            print(f"  Running inference for {model_name}...")
            exp_dir = os.path.join(model_dir, f"{model_name}_Experiment")
            model_path = os.path.join(exp_dir, f"{model_name}_hybrid.keras")
            if not os.path.exists(model_path):
                model_path = os.path.join(exp_dir, f"{model_name}_hybrid.h5")
            
            if not os.path.exists(model_path):
                print(f"  ⚠️ Model {model_path} not found.")
                continue
                
            # Preprocessing func
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
                'preprocess_input': prep, '<lambda>': prep,
                'resnet50_preprocess': prep, 'densenet_preprocess': prep,
                'efficientnet_preprocess': prep, 'convnext_preprocess': prep, 'vit_preprocess': prep,
                'Custom>resnet50_preprocess': prep, 'Custom>densenet_preprocess': prep,
                'Custom>efficientnet_preprocess': prep, 'Custom>convnext_preprocess': prep, 'Custom>vit_preprocess': prep
            }
            
            keras_model = load_model(model_path, compile=False, custom_objects=custom_objs)
            scaler_ag = joblib.load(os.path.join(exp_dir, f"{model_name}_scaler.pkl"))
            agent_model = lgb.Booster(model_file=os.path.join(exp_dir, f"{model_name}_agent.txt"))
            agent_wrapper = LGBMWrapper(agent_model)
            
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
            model_probas_dict[model_name] = hybrid_proba
            
            # -- Primary Metrics (Manuscript Tables) --
            means, lows, highs = bootstrap_all_metrics(y_true, hybrid_preds)
            
            manuscript_tables[d].append({
                'Model': model_name,
                'Accuracy (%)': f"{means[0]:.2f} ({lows[0]:.2f}-{highs[0]:.2f})",
                'Precision (%)': f"{means[1]:.2f} ({lows[1]:.2f}-{highs[1]:.2f})",
                'Recall (%)': f"{means[2]:.2f} ({lows[2]:.2f}-{highs[2]:.2f})",
                'F1 (%)': f"{means[3]:.2f} ({lows[3]:.2f}-{highs[3]:.2f})",
                'QWK': f"{means[4]:.4f} ({lows[4]:.4f}-{highs[4]:.4f})"
            })
                    
            # -- Secondary Metrics (Tables 3-6) --
            sec_metrics = calc_secondary_metrics(y_true, hybrid_preds)
            ece_val = calc_ece(y_true, hybrid_proba)
            
            for cls_idx, cls_name in enumerate(['MES 0', 'MES 1', 'MES 2', 'MES 3']):
                row = {
                    'Model': model_name, 'Class': cls_name,
                    'Sensitivity': f"{sec_metrics[cls_idx]['Sensitivity']:.3f}",
                    'Specificity': f"{sec_metrics[cls_idx]['Specificity']:.3f}",
                    'PPV': f"{sec_metrics[cls_idx]['PPV']:.3f}",
                    'NPV': f"{sec_metrics[cls_idx]['NPV']:.3f}",
                    'F1 Score': f"{sec_metrics[cls_idx]['F1']:.3f}",
                    'ECE': f"{ece_val:.3f}"
                }
                if d == 'NTUH': table3_rows.append(row)
                elif d == 'LIMUC': table4_rows.append(row)
                elif d == 'TMC-UCM': table5_rows.append(row)
                elif d == 'Unified': table6_rows.append(row)
                
        # -- Table 7: Agreement Thresholds --
        if len(all_hybrid_preds) > 0:
            all_hybrid_preds_arr = np.array(all_hybrid_preds)
            for cls_idx, cls_name in enumerate(['MES 0', 'MES 1', 'MES 2', 'MES 3']):
                cls_mask = y_true == cls_idx
                if np.sum(cls_mask) > 0:
                    y_true_cls = y_true[cls_mask]
                    preds_cls = all_hybrid_preds_arr[:, cls_mask]
                    correct_votes = np.sum(preds_cls == cls_idx, axis=0)
                    
                    for thresh, thresh_name in zip([3, 4, 5], ['3/5', '4/5', '5/5']):
                        successful_detects = np.sum(correct_votes >= thresh)
                        total_instances = len(y_true_cls)
                        table7_data.append({
                            'Dataset': d, 'Class': cls_name, 'Threshold': thresh_name,
                            'Successful_Detects': successful_detects, 'Total_Instances': total_instances,
                            'Accuracy': successful_detects / total_instances
                        })
            
            # -- Plot Fig 1a-d: CM & Collect ROC --
            avg_probas = np.mean(all_hybrid_probas, axis=0)
            sw_preds = np.argmax(avg_probas, axis=1)
            draw_confusion_matrix(y_true, sw_preds, d, save_dir)
            all_roc_data[d] = {'y_true': y_true, 'probas': model_probas_dict}
            
            # -- Fig 5b: Coverage curves --
            if d in ['TMC-UCM', 'NTUH', 'LIMUC']:
                print(f"  Calculating Coverage curve data for {d}...")
                
                # Majority Voting confidence (we use proportion of agreement as pseudo-confidence)
                maj_preds = []
                maj_confs = []
                for i in range(len(y_true)):
                    votes = [preds[i] for preds in all_hybrid_preds]
                    most_common, count = Counter(votes).most_common(1)[0]
                    maj_preds.append(most_common)
                    maj_confs.append(count / len(all_hybrid_preds))
                maj_preds = np.array(maj_preds)
                maj_confs = np.array(maj_confs)
                
                sw_confs = np.max(avg_probas, axis=1)
                
                cov_maj = get_coverage_curve_data(y_true, maj_preds, maj_confs, n_iterations=100)
                cov_sw = get_coverage_curve_data(y_true, sw_preds, sw_confs, n_iterations=100)
                
                coverage_data_dict[d] = {
                    'Majority Voting': cov_maj,
                    'Score-weighted': cov_sw
                }

    print("\n📝 Compiling Tables...")
    
    for i, d in enumerate(['TMC-UCM', 'LIMUC', 'NTUH', 'Unified']):
        # Actually the user requested them in order Table 1 (TMC), Table 2 (LIMUC), Table 3 (NTUH), Table 4 (Unified) based on screenshots?
        # The screenshots show Table 1 TMC, Table 2 LIMUC, Table 3 NTUH, Table 4 Unified. I'll save them as such.
        pass
        
    retry_io(lambda: pd.DataFrame(manuscript_tables['TMC-UCM']).to_csv(os.path.join(save_dir, 'Table_1_Performance_TMC-UCM.csv'), index=False))
    retry_io(lambda: pd.DataFrame(manuscript_tables['LIMUC']).to_csv(os.path.join(save_dir, 'Table_2_Performance_LIMUC.csv'), index=False))
    retry_io(lambda: pd.DataFrame(manuscript_tables['NTUH']).to_csv(os.path.join(save_dir, 'Table_3_Performance_NTUH.csv'), index=False))
    retry_io(lambda: pd.DataFrame(manuscript_tables['Unified']).to_csv(os.path.join(save_dir, 'Table_4_Performance_Unified.csv'), index=False))
    
    print("  ✅ Manuscript Tables 1-4 saved")
    
    print("📈 Generating Combined 2x2 ROC Panels...")
    plot_combined_roc_panels(all_roc_data, save_dir)
    
    print("📈 Generating Coverage Plot...")
    plot_coverage_curves_combined(coverage_data_dict, save_dir)
    
    print(f"\n🎉 SUCCESS! All 7 Tables and 6 Figures have been saved to: {save_dir}")

if __name__ == "__main__":
    main()
