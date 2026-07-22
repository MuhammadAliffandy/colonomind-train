import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from dgx_dataloader import load_all_images, load_tmc_ucm
from dgx_models import build_hybrid_model, MODEL_BUILDERS, focal_loss
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from collections import Counter
import tensorflow_hub as hub

def get_majority_vote(predictions, threshold):
    """
    Given a list of predictions for a single image from M models,
    returns the agreed class if the count >= threshold, else -1 (Refer to Doctor).
    """
    counter = Counter(predictions)
    most_common_pred, count = counter.most_common(1)[0]
    if count >= threshold:
        return most_common_pred
    else:
        return -1 # -1 denotes "Refer to Doctor"

def evaluate_ensemble(X_img, X_feat, X_umap, y_true, models, scalers, agents, threshold_confidence, voting_threshold):
    print(f"\n--- Evaluating Ensemble with Voting Threshold {voting_threshold}/{len(models)} ---")
    all_preds = [] # Shape: (M_models, N_samples)
    
    for i, model_name in enumerate(models.keys()):
        keras_model = models[model_name]
        agent_model = agents[model_name]
        scaler_ag = scalers[model_name]
        
        # Deep Predict
        y_pred_proba = keras_model.predict([X_img, X_feat, X_umap], verbose=0)
        y_pred_deep = np.argmax(y_pred_proba, axis=1)
        conf_deep = np.max(y_pred_proba, axis=1)
        
        # Agent Features
        df_ag = pd.DataFrame(X_feat, columns=[f"f{j}" for j in range(20)])
        df_ag["confidence"] = conf_deep
        df_ag["umap_0"] = X_umap[:, 0]
        df_ag["umap_1"] = X_umap[:, 1]
        features = ["confidence", "umap_0", "umap_1"] + [f"f{j}" for j in range(20)]
        
        X_te_ag = scaler_ag.transform(df_ag[features].values)
        y_pred_agent = agent_model.predict(X_te_ag)
        
        # Hybrid Routing
        low_conf_mask = conf_deep < threshold_confidence
        y_pred_hybrid = np.where(low_conf_mask, y_pred_agent, y_pred_deep)
        
        all_preds.append(y_pred_hybrid)
        
    all_preds = np.array(all_preds).T # Shape: (N_samples, M_models)
    
    final_preds = []
    for preds in all_preds:
        final_preds.append(get_majority_vote(preds, voting_threshold))
        
    final_preds = np.array(final_preds)
    
    # Calculate metrics
    processed_mask = final_preds != -1
    referred_count = np.sum(~processed_mask)
    total_count = len(y_true)
    
    if np.sum(processed_mask) > 0:
        framework_acc = accuracy_score(y_true[processed_mask], final_preds[processed_mask])
    else:
        framework_acc = 0.0
        
    print(f"Total Images: {total_count}")
    print(f"Referred to Doctor: {referred_count} ({(referred_count/total_count)*100:.2f}%)")
    print(f"Framework Accuracy (on processed): {framework_acc*100:.2f}%")
    
    # Per-class accuracy
    print("Per-class Accuracy (on processed):")
    per_class_acc = {}
    for cls in np.unique(y_true):
        cls_mask = (y_true == cls) & processed_mask
        cls_total = np.sum((y_true == cls) & processed_mask) # only count processed ones of this class
        if cls_total > 0:
            cls_acc = np.sum(final_preds[cls_mask] == y_true[cls_mask]) / cls_total
            print(f"  Class {cls}: {cls_acc*100:.2f}%")
            per_class_acc[f"Class_{cls}"] = float(cls_acc)
        else:
            print(f"  Class {cls}: N/A (all referred)")
            per_class_acc[f"Class_{cls}"] = None
            
    metrics = {
        "Voting_Threshold": f"{voting_threshold}/{len(models)}",
        "Total_Images": int(total_count),
        "Referred_to_Doctor_Count": int(referred_count),
        "Referred_to_Doctor_Percentage": float(referred_count/total_count) * 100,
        "Framework_Accuracy": float(framework_acc),
        "Per_Class_Accuracy": per_class_acc
    }
            
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Ensemble Voting Evaluation (TMC-UCM -> NTUH/LIMUC)")
    parser.add_argument("--base_dir", type=str, default="..")
    parser.add_argument("--models_dir", type=str, default="../Result/Intra_TMC-UCM", help="Path where TMC-UCM models are saved")
    parser.add_argument("--threshold_confidence", type=float, default=0.50, help="Hybrid routing confidence threshold")
    args = parser.parse_args()
    
    BASE_DIR = args.base_dir
    model_names = ['ResNet-50', 'DenseNet-121', 'EfficientNet-B4', 'ConvNeXt-Tiny', 'ViT-B-16']
    
    # Load Models
    models = {}
    scalers = {}
    agents = {}
    
    import subprocess
    print("Checking and Auto-Training missing TMC-UCM models...")
    for model_name in model_names:
        exp_dir = os.path.join(args.models_dir, f"{model_name}_Experiment")
        model_path = os.path.join(exp_dir, f"{model_name}_hybrid.keras")
        # Also check for legacy .h5 file for backward compatibility
        if not os.path.exists(model_path):
            legacy_path = os.path.join(exp_dir, f"{model_name}_hybrid.h5")
            if os.path.exists(legacy_path):
                model_path = legacy_path
        if not os.path.exists(model_path):
            print(f"⚠️ Model file {model_path} not found. Automatically training {model_name} on TMC-UCM...")
            cmd = f"python -u src/train_dgx.py --scenario Intra --train_dataset TMC-UCM --test_dataset TMC-UCM --model {model_name} --base_dir {args.base_dir}"
            subprocess.run(cmd, shell=True, check=True)
            print(f"✅ Auto-training for {model_name} completed.")

    print("\\nLoading pre-trained TMC-UCM models into memory...")
    for model_name in model_names:
        exp_dir = os.path.join(args.models_dir, f"{model_name}_Experiment")
        model_path = os.path.join(exp_dir, f"{model_name}_hybrid.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(exp_dir, f"{model_name}_hybrid.h5")
            
        # Dynamically map the correct preprocess_input to fix legacy keras saving bug
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
            'vit_preprocess': prep
        }
        
        keras_model = load_model(model_path, compile=False, custom_objects=custom_objs)
        scaler_ag = joblib.load(os.path.join(exp_dir, f"{model_name}_scaler.pkl"))
        agent_model = lgb.Booster(model_file=os.path.join(exp_dir, f"{model_name}_agent.txt"))
        
        # Wrap LGBM booster in a classifier-like interface
        class LGBMWrapper:
            def __init__(self, booster):
                self.booster = booster
            def predict(self, X):
                probs = self.booster.predict(X)
                return np.argmax(probs, axis=1)
                
        models[model_name] = keras_model
        scalers[model_name] = scaler_ag
        agents[model_name] = LGBMWrapper(agent_model)
        print(f"✅ Loaded {model_name}")
        
    # Test on NTUH, LIMUC, and Combined
    le = LabelEncoder()
    le.fit(['MES0', 'MES1', 'MES2', 'MES3'])
    
    ntuh_paths = [f'{BASE_DIR}/Dataset+Code/MES classification_20250313', f'{BASE_DIR}/Dataset+Code/MES classification_20250724']
    limuc_paths = [f'{BASE_DIR}/Dataset/LIMUC/train_and_validation_sets', f'{BASE_DIR}/Dataset/LIMUC/test_set']
    
    print("Loading NTUH dataset...")
    ntuh_imgs, ntuh_feats, ntuh_labels, _ = load_all_images(ntuh_paths, 'NTUH')
    print("Loading LIMUC dataset...")
    limuc_imgs, limuc_feats, limuc_labels, _ = load_all_images(limuc_paths, 'LIMUC')
    
    DATASETS_TO_EVALUATE = {
        'NTUH': (ntuh_imgs, ntuh_feats, ntuh_labels),
        'LIMUC': (limuc_imgs, limuc_feats, limuc_labels),
        'Combined_NTUH_LIMUC': (
            ntuh_imgs + limuc_imgs,
            ntuh_feats + limuc_feats,
            ntuh_labels + limuc_labels
        )
    }
    
    for test_dataset, (all_imgs, all_feats, all_labels) in DATASETS_TO_EVALUATE.items():
        print(f"\n{'='*50}")
        print(f"🧪 Testing Ensemble on {test_dataset}")
        print(f"{'='*50}")
        
        X_img = np.array(all_imgs, dtype=np.float32)
        X_feat = np.array(all_feats)
        y_encoded = le.transform(all_labels)
        
        umap_path = os.path.join(args.models_dir, f"ResNet-50_Experiment", "umap_model.pkl")
        feat_scaler_path = os.path.join(args.models_dir, f"ResNet-50_Experiment", "base_scaler.pkl") 
        
        feat_scaler = joblib.load(feat_scaler_path)
        umap_reducer = joblib.load(umap_path)
        
        X_feat_scaled = feat_scaler.transform(X_feat)
        X_umap = umap_reducer.transform(X_feat_scaled)
        
        dataset_results = []
        for v_thresh in [3, 4, 5]:
            metrics = evaluate_ensemble(
                X_img, X_feat_scaled, X_umap, y_encoded,
                models, scalers, agents,
                threshold_confidence=args.threshold_confidence,
                voting_threshold=v_thresh
            )
            dataset_results.append(metrics)
            
        # Save results to a folder
        save_dir = f"{BASE_DIR}/Result/Ensemble_Voting_Experiment"
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"voting_metrics_{test_dataset}.json")
        with open(out_path, 'w') as f:
            json.dump(dataset_results, f, indent=4)
        print(f"✅ Saved results to {out_path}")

if __name__ == "__main__":
    main()
