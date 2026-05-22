import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# eClinicalMedicine Style Guidelines (Arial font, clean aesthetics)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils import resample

# Add src to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import IMG_SIZE, DATASETS
from src.data_loader import load_dataset
from src.model import build_hybrid_model

def compute_95_ci(y_true, y_pred, n_bootstraps=1000):
    accuracies = []
    for _ in range(n_bootstraps):
        idx = resample(np.arange(len(y_true)))
        acc = accuracy_score(y_true[idx], y_pred[idx])
        accuracies.append(acc)
    lower = np.percentile(accuracies, 2.5)
    upper = np.percentile(accuracies, 97.5)
    return np.mean(accuracies), lower, upper

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results/finetuned', help="Path to finetuned artefacts")
    parser.add_argument('--output_dir', type=str, default='./paper_results', help="Path to save paper evaluations")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=========================================")
    print(" PAPER EVALUATIONS: SUPER AGENT PIPELINE")
    print("=========================================")
    
    # 1. Load Artefacts
    print("\n1. Loading Artefacts...")
    model_path   = os.path.join(args.results_dir, "finetuned_hybrid_model.h5")
    scaler_path  = os.path.join(args.results_dir, "scaler.pkl")
    le_path      = os.path.join(args.results_dir, "label_encoder.pkl")
    umap_path    = os.path.join(args.results_dir, "umap_model.pkl")
    agent_sc_path= os.path.join(args.results_dir, "agent_scaler.pkl")
    agent_path   = os.path.join(args.results_dir, "super_agent_lgbm.txt")

    scaler       = joblib.load(scaler_path)
    le           = joblib.load(le_path)
    num_classes  = len(le.classes_)

    rebuild_umap = False
    try:
        umap_reducer = joblib.load(umap_path)
    except Exception as e:
        print(f"  ⚠️  Failed to load UMAP model ({e}). Will re-fit dynamically.")
        rebuild_umap = True
    agent_scaler = joblib.load(agent_sc_path)
    super_agent  = lgb.Booster(model_file=agent_path)
    num_classes  = len(le.classes_)

    print("   Building Keras Architecture...")
    model = build_hybrid_model(
        image_input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        feat_input_shape=(scaler.n_features_in_,),
        umap_feat_shape=(2,),
        num_classes=num_classes,
        dropout_rate=0.4
    )
    model.load_weights(model_path)

    # 2. Load Data and get Test Set
    print("\n2. Loading Test Set...")
    all_img, all_feat, all_label, all_source = [], [], [], []
    for name, path in DATASETS.items():
        if os.path.exists(path):
            Xi, Xf, yl, _ = load_dataset(path)
            all_img.append(Xi); all_feat.append(Xf)
            all_label.append(yl); all_source.append(np.full(len(yl), name))

    X_img_all   = np.concatenate(all_img, axis=0)
    X_feat_all  = np.concatenate(all_feat, axis=0)
    y_encoded   = le.transform(np.concatenate(all_label, axis=0))
    source_all  = np.concatenate(all_source, axis=0)

    # Stratified split to match training exactly
    X_img_train, X_img_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp = train_test_split(
        X_img_all, X_feat_all, y_encoded, test_size=0.30, stratify=y_encoded, random_state=args.seed
    )
    
    X_img_val, X_img_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
        X_img_tmp, X_feat_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=args.seed
    )

    X_img_test = X_img_test.astype(np.float32) / 255.0
    X_feat_test_s = scaler.transform(X_feat_test)

    if rebuild_umap:
        print("  Re-fitting UMAP dynamically to ensure exact feature projection …")
        from imblearn.over_sampling import SMOTE
        import umap
        
        X_feat_train_s = scaler.transform(X_feat_train)
        smote = SMOTE(random_state=args.seed)
        X_feat_bal, _ = smote.fit_resample(X_feat_train_s, y_train)
        
        umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, metric='euclidean', random_state=args.seed)
        umap_reducer.fit(X_feat_bal)
        
    X_umap_test = umap_reducer.transform(X_feat_test_s)
    
    # 3. Agent Predictions
    print("\n3. Generating Predictions...")
    start_time = time.time()
    y_proba_te = model.predict([X_img_test, X_feat_test_s, X_umap_test], verbose=0)
    
    X_agent_test = np.column_stack([
        np.max(y_proba_te, axis=1),
        X_umap_test,
        np.argmax(y_proba_te, axis=1).astype(float),
        X_feat_test_s
    ])
    
    X_test_scaled_agent = agent_scaler.transform(X_agent_test)
    y_pred_proba = super_agent.predict(X_test_scaled_agent)
    y_pred_final = np.argmax(y_pred_proba, axis=1)
    latency = (time.time() - start_time) / len(X_img_test) * 1000

    # 4. Compute Metrics
    print("\n4. Computing Paper Metrics...")
    acc = accuracy_score(y_test, y_pred_final)
    mean_acc, lower_ci, upper_ci = compute_95_ci(y_test, y_pred_final)
    
    cm = confusion_matrix(y_test, y_pred_final)
    
    # Save Confusion Matrix (eClinicalMedicine Style)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_,
                cbar_kws={'label': 'Number of Samples'}, linewidths=0.5, linecolor='white')
    plt.title(f'TMC Feedback Agent Confusion Matrix\nAccuracy: {acc*100:.2f}%', pad=15)
    plt.ylabel('True Label', weight='bold')
    plt.xlabel('Predicted Label', weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix_eClinicalMed.png'), dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

    # Save Agent Feature Importance (eClinicalMedicine Style)
    plt.figure(figsize=(10,6))
    ax = lgb.plot_importance(super_agent, max_num_features=15, title='Top 15 Agent Decision Features', 
                             xlabel='Feature Importance (Split)', ylabel='Features', grid=False)
    plt.title('Top 15 Agent Decision Features', pad=15, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'agent_feature_importance_eClinicalMed.png'), dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

    # Major 7: Handcrafted Feature Importance Analysis
    from sklearn.ensemble import RandomForestClassifier
    print("  Calculating Handcrafted Feature Importance (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    rf.fit(X_feat_train, y_train)
    importances = rf.feature_importances_
    features = [
        'LL_mean', 'LL_std', 'LL_var', 'LL_entropy',
        'LH_mean', 'LH_std', 'LH_var', 'LH_entropy',
        'HL_mean', 'HL_std', 'HL_var', 'HL_entropy',
        'HH_mean', 'HH_std', 'HH_var', 'HH_entropy',
        'HH_energy', 'GLCM_contrast', 'GLCM_dissimilarity', 'GLCM_homogeneity'
    ]
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(features)), importances[indices], align="center", color='#2B5B84')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
    plt.title("Random Forest Feature Importance (Handcrafted Modalities)", pad=15, weight='bold')
    plt.ylabel("Gini Importance", weight='bold')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'handcrafted_feature_importance_eClinicalMed.png'), dpi=600, bbox_inches='tight', facecolor='white')

    # Sensitivity / Specificity / eClinicalMedicine Table Export
    from sklearn.metrics import precision_recall_fscore_support
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_test, y_pred_final)
    
    table_data = []
    metrics_str = ""
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        table_data.append({
            'Class': le.classes_[i],
            'Sensitivity (Recall)': f"{sens:.4f}",
            'Specificity': f"{spec:.4f}",
            'Precision': f"{precisions[i]:.4f}",
            'F1-Score': f"{f1s[i]:.4f}",
            'Support (N)': int(supports[i])
        })
        metrics_str += f"Class {le.classes_[i]}: Sensitivity = {sens:.4f}, Specificity = {spec:.4f}\n"
        
    df_results = pd.DataFrame(table_data)
    
    # Save professional tables for manuscript
    df_results.to_csv(os.path.join(args.output_dir, 'eClinicalMedicine_Table.csv'), index=False)
    try:
        df_results.to_excel(os.path.join(args.output_dir, 'eClinicalMedicine_Table.xlsx'), index=False)
    except ImportError:
        print("  (openpyxl not installed, skipping .xlsx export)")

    # Save Report
    report = f"""=========================================
PAPER EVALUATION REPORT
=========================================
Final Agent Accuracy : {acc:.4f} (95% CI: {lower_ci:.4f} - {upper_ci:.4f})
Inference Latency    : {latency:.2f} ms per sample

--- Sensitivity & Specificity ---
{metrics_str}
--- Classification Report ---
{classification_report(y_test, y_pred_final, target_names=le.classes_, digits=4)}

=========================================
Model Size Details:
- Keras Model: {os.path.getsize(model_path) / (1024*1024):.2f} MB
- Agent Model: {os.path.getsize(agent_path) / 1024:.2f} KB
"""
    with open(os.path.join(args.output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(report)
        
    print(report)
    print(f"\n✅ All paper figures and reports saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
