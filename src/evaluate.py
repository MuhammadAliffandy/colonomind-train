import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix

def evaluate_predictions(y_true, y_pred, y_proba=None, class_names=None):
    """
    Standard evaluation logic for model performance.
    """
    if class_names is None:
        class_names = ['MES0', 'MES1', 'MES2', 'MES3']
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Calculate Specificity
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(len(class_names)):
        tn = np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + cm[i,i]
        fp = np.sum(cm[:,i]) - cm[i,i]
        specs.append(tn / (tn + fp + 1e-6))
    spec = np.mean(specs)
    
    metrics = {
        'Accuracy (%)': acc * 100,
        'Sensitivity (%)': rec * 100,
        'Specificity (%)': spec * 100,
        'Precision (PPV)': prec * 100,
        'F1-Score': f1,
        'QWK': kappa
    }
    
    if y_proba is not None:
        # One-vs-rest macro AUC
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['AUC'] = auc
        except ValueError:
            pass
            
    return metrics

def build_results_csv(output_path, model_names, results_dict):
    """
    Compiles results into a formal CSV table.
    """
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    df.index.name = 'Model'
    df.reset_index(inplace=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated results table at {output_path}")

if __name__ == "__main__":
    print("Evaluating models...")
    # This script is intended to be called with actual model predictions
    # Example: evaluate_predictions(y_test, y_pred_resnet)
    print("Evaluation logic configured successfully.")
