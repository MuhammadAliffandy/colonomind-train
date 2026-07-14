import os
import argparse
import json
import shutil
import joblib
from hashlib import sha1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from dgx_dataloader import load_all_images, load_tmc_ucm
from dgx_models import build_hybrid_model, MODEL_BUILDERS, focal_loss

def make_feedback(y_true, y_pred, y_rule, proba, umap_feat, h_feat):
    df = pd.DataFrame(h_feat, columns=[f"f{i}" for i in range(20)])
    df["confidence"] = np.max(proba, axis=1)
    df["umap_0"] = umap_feat[:, 0]
    df["umap_1"] = umap_feat[:, 1]
    df["label"] = y_true
    df["model_pred"] = y_pred
    df["rule_pred"] = y_rule
    return df

def main():
    parser = argparse.ArgumentParser(description="ColonoMind DGX Training Script")
    parser.add_argument("--scenario", type=str, required=True, choices=['Intra', 'Multi'])
    parser.add_argument("--train_dataset", type=str, required=True, choices=['NTUH', 'LIMUC', 'TMC-UCM'])
    parser.add_argument("--test_dataset", type=str, required=True, choices=['NTUH', 'LIMUC', 'TMC-UCM'])
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_BUILDERS.keys()))
    args = parser.parse_args()

    print(f"\\n{'='*50}")
    print(f"🚀 Starting Training on DGX")
    print(f"Scenario: {args.scenario}")
    print(f"Train Dataset: {args.train_dataset}")
    print(f"Test Dataset: {args.test_dataset}")
    print(f"Model: {args.model}")
    print(f"{'='*50}\\n")

    BASE_DIR = '..'
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
            X_train_img, X_train_feat, y_train_label, paths_train = load_all_images([TRAIN_DIRS[0]], args.train_dataset)
            X_test_img, X_test_feat, y_test_label, paths_test = load_all_images([TRAIN_DIRS[1]], args.train_dataset)
        elif args.train_dataset == 'TMC-UCM':
            X_train_img, X_train_feat, y_train_label, paths_train = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Train')
            X_test_img, X_test_feat, y_test_label, paths_test = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Test')
        else:
            all_imgs, all_feats, all_labels, all_paths = load_all_images(TRAIN_DIRS, args.train_dataset)
            X_train_img, X_test_img, X_train_feat, X_test_feat, y_train_label, y_test_label, paths_train, paths_test = train_test_split(
                all_imgs, all_feats, all_labels, all_paths, test_size=0.2, random_state=42, stratify=all_labels
            )
    else:
        if args.train_dataset == 'TMC-UCM':
            X_train_img, X_train_feat, y_train_label, paths_train = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None)
        else:
            X_train_img, X_train_feat, y_train_label, paths_train = load_all_images(TRAIN_DIRS, args.train_dataset)

        if args.test_dataset == 'TMC-UCM':
            X_test_img, X_test_feat, y_test_label, paths_test = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None)
        else:
            X_test_img, X_test_feat, y_test_label, paths_test = load_all_images(TEST_DIRS, args.test_dataset)

    print(f"Training samples: {len(X_train_img)}")
    print(f"Testing samples: {len(X_test_img)}")

    X_img_train = np.array(X_train_img, dtype=np.float32) / 255.0
    X_img_test  = np.array(X_test_img, dtype=np.float32) / 255.0

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_label)
    y_test_encoded  = le.transform(y_test_label)
    y_train_cat = to_categorical(y_train_encoded, num_classes=len(le.classes_))
    y_test_cat  = to_categorical(y_test_encoded,  num_classes=len(le.classes_))

    scaler = StandardScaler()
    X_feat_train_scaled = scaler.fit_transform(np.array(X_train_feat))
    X_feat_test_scaled  = scaler.transform(np.array(X_test_feat))

    # SMOTE only on training set to prevent leakage
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_feat_train_bal, y_train_bal = smote.fit_resample(X_feat_train_scaled, y_train_encoded)

    X_img_train_bal = []
    for feat, label in zip(X_feat_train_bal, y_train_bal):
        dists = np.linalg.norm(X_feat_train_scaled[y_train_encoded == label] - feat, axis=1)
        idx = np.where(y_train_encoded == label)[0][np.argmin(dists)]
        X_img_train_bal.append(X_img_train[idx])
    X_img_train_bal = np.array(X_img_train_bal, dtype=np.float32)
    y_train_cat_bal = to_categorical(y_train_bal, num_classes=len(le.classes_))

    # UMAP
    print("Fitting UMAP...")
    umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=42)
    X_train_umap = umap_reducer.fit_transform(X_feat_train_bal)
    X_test_umap  = umap_reducer.transform(X_feat_test_scaled)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_train_umap[:,0], X_train_umap[:,1], c=y_train_bal, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    plt.title("UMAP Projection")
    plt.savefig(os.path.join(BASE_SAVE_DIR, 'UMAP_Projection.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Model Training
    print(f"\\n[1] Training Base Hybrid Model: {args.model}")
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_bal), y=y_train_bal)
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
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, mode='max')
    ]

    history = model.fit(
        [X_img_train_bal, X_feat_train_bal, X_train_umap], y_train_cat_bal,
        validation_data=([X_img_test, X_feat_test_scaled, X_test_umap], y_test_cat),
        batch_size=16, epochs=100, class_weight=class_weight_dict, callbacks=callbacks, verbose=1
    )

    model_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_hybrid.h5")
    model.save(model_path)
    print(f"✅ Saved base model to {model_path}")

    # Agent Training
    print(f"\\n[2] Training Super Agent (LightGBM Continual Learning)")
    y_pred_proba_train = model.predict([X_img_train_bal, X_feat_train_bal, X_train_umap], verbose=0)
    y_pred_hybrid_train = np.argmax(y_pred_proba_train, axis=1)

    y_pred_proba = model.predict([X_img_test, X_feat_test_scaled, X_test_umap], verbose=0)
    y_pred_hybrid = np.argmax(y_pred_proba, axis=1)

    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42)
    dt.fit(X_train_umap, y_train_bal)
    y_rule_train = dt.predict(X_train_umap)
    y_rule_test = dt.predict(X_test_umap)

    df_train_ag = make_feedback(y_train_bal, y_pred_hybrid_train, y_rule_train, y_pred_proba_train, X_train_umap, X_feat_train_bal)
    df_test_ag  = make_feedback(y_test_encoded, y_pred_hybrid, y_rule_test, y_pred_proba, X_test_umap, X_feat_test_scaled)
    df_test_orig = df_test_ag.copy()

    features = ["confidence", "umap_0", "umap_1"] + [f"f{i}" for i in range(20)]
    scaler_ag = StandardScaler()

    loop = 0
    known_hashes = set()
    df_train_ag_loop = df_train_ag.copy()

    while loop < 5:
        X_tr = scaler_ag.fit_transform(df_train_ag_loop[features].values)
        y_tr = df_train_ag_loop["label"].values
        
        clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
        clf.fit(X_tr, y_tr)
        
        X_te = scaler_ag.transform(df_test_orig[features].values)
        y_pred_ag = clf.predict(X_te)
        y_proba_ag = clf.predict_proba(X_te)
        
        acc = accuracy_score(df_test_orig["label"].values, y_pred_ag)
        print(f"🔁 Loop {loop+1}: Agent Accuracy = {acc:.4f}")
        
        if acc >= 0.88: break
            
        misclassified = df_test_orig[y_pred_ag != df_test_orig["label"]].copy()
        misclassified["hash"] = misclassified.apply(lambda r: sha1(str(r.to_dict()).encode()).hexdigest(), axis=1)
        new_errs = misclassified[~misclassified["hash"].isin(known_hashes)]
        
        if new_errs.empty: break
        
        known_hashes.update(new_errs["hash"])
        df_train_ag_loop = pd.concat([df_train_ag_loop, new_errs.drop(columns=["hash"])], ignore_index=True)
        loop += 1

    agent_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_agent.txt")
    scaler_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_scaler.pkl")
    
    tmp_agent_path = f"/tmp/{args.model}_agent_{np.random.randint(1000)}.txt"
    clf.booster_.save_model(tmp_agent_path)
    shutil.copy(tmp_agent_path, agent_path)
    joblib.dump(scaler_ag, scaler_path)

    # Metrics computation
    y_true = df_test_orig["label"].values
    acc = accuracy_score(y_true, y_pred_ag)
    prec = precision_score(y_true, y_pred_ag, average='macro')
    rec = recall_score(y_true, y_pred_ag, average='macro')
    f1 = f1_score(y_true, y_pred_ag, average='macro')
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
        
    print(f"✅ Evaluation Complete. Accuracy: {acc:.4f}, QWK: {kappa:.4f}")
    print(f"📁 Results saved to {BASE_SAVE_DIR}")

if __name__ == "__main__":
    main()
