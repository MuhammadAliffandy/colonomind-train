import os
import argparse
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from dgx_dataloader import load_all_images, load_tmc_ucm
from dgx_models import build_hybrid_model, MODEL_BUILDERS, focal_loss

def main():
    parser = argparse.ArgumentParser(description="ColonoMind DGX Training Script")
    parser.add_argument("--scenario", type=str, required=True, choices=['Intra', 'Multi'])
    parser.add_argument("--train_dataset", type=str, required=True, choices=['NTUH', 'LIMUC', 'TMC-UCM'])
    parser.add_argument("--test_dataset", type=str, required=True, choices=['NTUH', 'LIMUC', 'TMC-UCM'])
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--base_dir", type=str, default="..", help="Base directory where Dataset and Dataset+Code folders are located")
    args = parser.parse_args()

    print(f"\\n{'='*50}")
    print(f"🚀 Starting Training on DGX")
    print(f"Scenario: {args.scenario}")
    print(f"Train Dataset: {args.train_dataset}")
    print(f"Test Dataset: {args.test_dataset}")
    print(f"Model: {args.model}")
    print(f"Base Dir: {args.base_dir}")
    print(f"{'='*50}\\n")

    BASE_DIR = args.base_dir
    DATASET_PATHS = {
        'NTUH': [
            f'{BASE_DIR}/Dataset+Code/MES classification_20250313',
            f'{BASE_DIR}/Dataset+Code/MES classification_20250724'
        ],
        'LIMUC': [
            f'{BASE_DIR}/Dataset/LIMUC/train_and_validation_sets',
            f'{BASE_DIR}/Dataset/LIMUC/test_set'
        ],
        'TMC-UCM': [
            f'{BASE_DIR}/Dataset/TMC-UCM/images'
        ]
    }
    TMC_UCM_ROOT = f'{BASE_DIR}/Dataset/TMC-UCM'
    TRAIN_DIRS = DATASET_PATHS[args.train_dataset]
    TEST_DIRS  = DATASET_PATHS[args.test_dataset]

    if args.scenario == 'Intra':
        BASE_SAVE_DIR = f"../Result/Intra_{args.train_dataset}/{args.model}_Experiment"
    else:
        BASE_SAVE_DIR = f"../Result/Multi_{args.train_dataset}_to_{args.test_dataset}/{args.model}_Experiment"
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # 1. LOAD DATA
    print("Loading Data...")
    if args.scenario == 'Intra':
        if args.train_dataset == 'LIMUC':
            X_train_img_raw, X_train_feat_raw, y_train_label_raw, _ = load_all_images([TRAIN_DIRS[0]], args.train_dataset)
            X_test_img, X_test_feat, y_test_label, _ = load_all_images([TRAIN_DIRS[1]], args.train_dataset)
        elif args.train_dataset == 'TMC-UCM':
            X_train_img_raw, X_train_feat_raw, y_train_label_raw, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Train')
            X_test_img, X_test_feat, y_test_label, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Test')
        else:
            all_imgs, all_feats, all_labels, all_paths = load_all_images(TRAIN_DIRS, args.train_dataset)
            X_train_img_raw, X_test_img, X_train_feat_raw, X_test_feat, y_train_label_raw, y_test_label, _, _ = train_test_split(
                all_imgs, all_feats, all_labels, all_paths, test_size=0.2, random_state=42, stratify=all_labels
            )
    else:
        if args.train_dataset == 'TMC-UCM':
            X_train_img_raw, X_train_feat_raw, y_train_label_raw, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None)
        else:
            X_train_img_raw, X_train_feat_raw, y_train_label_raw, _ = load_all_images(TRAIN_DIRS, args.train_dataset)

        if args.test_dataset == 'TMC-UCM':
            X_test_img, X_test_feat, y_test_label, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None)
        else:
            X_test_img, X_test_feat, y_test_label, _ = load_all_images(TEST_DIRS, args.test_dataset)

    # We split 20% of training data for Validation (Early Stopping)
    print("Splitting Train into Train/Val (80/20) for strict isolation...")
    X_train_img, X_val_img, X_train_feat, X_val_feat, y_train_label, y_val_label = train_test_split(
        X_train_img_raw, X_train_feat_raw, y_train_label_raw, test_size=0.2, random_state=42, stratify=y_train_label_raw
    )

    print(f"Training samples: {len(X_train_img)}")
    print(f"Validation samples: {len(X_val_img)}")
    print(f"Testing samples (Untouched): {len(X_test_img)}")

    # Images kept at raw 0-255 scale (preprocessing handled in dgx_models.py branch definition)
    X_img_train = np.array(X_train_img, dtype=np.float32)
    X_img_val = np.array(X_val_img, dtype=np.float32)
    X_img_test  = np.array(X_test_img, dtype=np.float32)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_label)
    y_val_encoded = le.transform(y_val_label)
    y_test_encoded  = le.transform(y_test_label)
    
    y_train_cat = to_categorical(y_train_encoded, num_classes=len(le.classes_))
    y_val_cat = to_categorical(y_val_encoded, num_classes=len(le.classes_))
    y_test_cat  = to_categorical(y_test_encoded,  num_classes=len(le.classes_))

    # Scale Handcrafted Features
    scaler = StandardScaler()
    X_feat_train_scaled = scaler.fit_transform(np.array(X_train_feat))
    X_feat_val_scaled = scaler.transform(np.array(X_val_feat))
    X_feat_test_scaled  = scaler.transform(np.array(X_test_feat))

    # SMOTE is removed as per reviewer feedback. Class imbalance handled via class_weights/loss

    # UMAP
    print("Fitting UMAP on Train...")
    umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=42)
    X_train_umap = umap_reducer.fit_transform(X_feat_train_scaled)
    X_val_umap = umap_reducer.transform(X_feat_val_scaled)
    X_test_umap  = umap_reducer.transform(X_feat_test_scaled)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_train_umap[:,0], X_train_umap[:,1], c=y_train_encoded, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    plt.title("UMAP Projection (Train Set)")
    plt.savefig(os.path.join(BASE_SAVE_DIR, 'UMAP_Projection.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Model Training
    print(f"\\n[1] Training Base Hybrid Model: {args.model}")
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    model = build_hybrid_model(
        branch_builder_func=MODEL_BUILDERS[args.model],
        image_input_shape=(224, 224, 3),
        feat_input_shape=(20,),
        umap_feat_shape=(2,),
        num_classes=len(le.classes_),
        dropout_rate=0.5
    )

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=focal_loss(gamma=2.5, alpha=0.25), metrics=['accuracy'])
    
    # Validation strictly uses val set, avoiding test set leakage
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, mode='max')
    ]

    history = model.fit(
        [X_img_train, X_feat_train_scaled, X_train_umap], y_train_cat,
        validation_data=([X_img_val, X_feat_val_scaled, X_val_umap], y_val_cat),
        batch_size=16, epochs=100, class_weight=class_weight_dict, callbacks=callbacks, verbose=1
    )

    model_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_hybrid.h5")
    model.save(model_path)
    print(f"✅ Saved base model to {model_path}")

    # Agent Training (Removed Test set leakage loop)
    print(f"\\n[2] Training Super Agent (LightGBM on Train)")
    y_pred_proba_train = model.predict([X_img_train, X_feat_train_scaled, X_train_umap], verbose=0)
    y_pred_hybrid_train = np.argmax(y_pred_proba_train, axis=1)

    y_pred_proba_test = model.predict([X_img_test, X_feat_test_scaled, X_test_umap], verbose=0)
    y_pred_hybrid_test = np.argmax(y_pred_proba_test, axis=1)

    def make_features(proba, umap_feat, h_feat):
        df = pd.DataFrame(h_feat, columns=[f"f{i}" for i in range(20)])
        df["confidence"] = np.max(proba, axis=1)
        df["umap_0"] = umap_feat[:, 0]
        df["umap_1"] = umap_feat[:, 1]
        return df

    df_train_ag = make_features(y_pred_proba_train, X_train_umap, X_feat_train_scaled)
    df_test_ag  = make_features(y_pred_proba_test, X_test_umap, X_feat_test_scaled)
    
    features = ["confidence", "umap_0", "umap_1"] + [f"f{i}" for i in range(20)]
    scaler_ag = StandardScaler()
    X_tr = scaler_ag.fit_transform(df_train_ag[features].values)
    y_tr = y_train_encoded
    
    clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    
    X_te = scaler_ag.transform(df_test_ag[features].values)
    y_pred_ag = clf.predict(X_te)
    
    # Save Super Agent
    agent_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_agent.txt")
    scaler_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_scaler.pkl")
    
    tmp_agent_path = f"/tmp/{args.model}_agent_{np.random.randint(1000)}.txt"
    clf.booster_.save_model(tmp_agent_path)
    shutil.copy(tmp_agent_path, agent_path)
    joblib.dump(scaler_ag, scaler_path)

    # 3. FINAL EVALUATION ON UNTOUCHED TEST SET
    print(f"\\n[3] Final Evaluation on Test Set")
    y_true = y_test_encoded
    acc = accuracy_score(y_true, y_pred_ag)
    prec = precision_score(y_true, y_pred_ag, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred_ag, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred_ag, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred_ag, weights='quadratic')
    
    cm = confusion_matrix(y_true, y_pred_ag)
    specs = []
    for i in range(len(le.classes_)):
        tn = np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + cm[i,i]
        fp = np.sum(cm[:,i]) - cm[i,i]
        specs.append(tn / (tn + fp + 1e-6))
    spec = np.mean(specs)

    metrics = {
        'Model': args.model,
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'Specificity': float(spec),
        'F1-Score': float(f1),
        'QWK': float(kappa)
    }

    metrics_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Evaluation Complete. Test Accuracy: {acc:.4f}, Test QWK: {kappa:.4f}")
    print(f"📁 Results saved to {BASE_SAVE_DIR}")

if __name__ == "__main__":
    main()
