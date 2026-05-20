import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Saved confusion_matrix.png")

def calculate_sensitivity_specificity(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    metrics = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics.append((sens, spec))
    return metrics

def compute_95_ci(y_true, y_pred, n_bootstraps=1000):
    """Computes 95% Confidence Interval for Accuracy using bootstrapping."""
    from sklearn.utils import resample
    from sklearn.metrics import accuracy_score
    accuracies = []
    for _ in range(n_bootstraps):
        idx = resample(np.arange(len(y_true)))
        acc = accuracy_score(y_true[idx], y_pred[idx])
        accuracies.append(acc)
    lower = np.percentile(accuracies, 2.5)
    upper = np.percentile(accuracies, 97.5)
    return np.mean(accuracies), lower, upper

def feature_importance_analysis(X_feat, y):
    """Uses a Random Forest to calculate feature importance of the 20 handcrafted features."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_feat, y)
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
    plt.title("Feature Importance Analysis (Handcrafted Features)")
    plt.bar(range(X_feat.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_feat.shape[1]), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("Saved feature_importance.png")

if __name__ == "__main__":
    print("Run this script by passing trained model predictions and true labels.")
