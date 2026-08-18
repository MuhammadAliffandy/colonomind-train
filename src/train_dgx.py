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
import dgx_models
from dgx_models import build_hybrid_model, MODEL_BUILDERS, focal_loss

def focal_loss_with_label_smoothing(gamma=2.5, alpha=0.25, label_smoothing=0.1):
    """Focal loss with label smoothing for better generalization."""
    def loss(y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smooth = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        cross_entropy = -y_true_smooth * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss

def main():
    parser = argparse.ArgumentParser(description="ColonoMind DGX Training Script")
    parser.add_argument("--scenario", type=str, required=True, choices=['Intra', 'Multi', 'Unified'])
    parser.add_argument("--train_dataset", type=str, required=True, choices=['NTUH', 'LIMUC', 'TMC-UCM', 'Unified'])
    parser.add_argument("--test_dataset", type=str, required=True, choices=['NTUH', 'LIMUC', 'TMC-UCM', 'Unified'])
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--base_dir", type=str, default="..", help="Base directory where Dataset and Dataset+Code folders are located")
    parser.add_argument('--threshold', type=float, default=0.75, help='Confidence threshold for passing to Agent')
    parser.add_argument('--agent_only', action='store_true', help='Skip deep learning train and only retrain the Agent')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to cache preprocessed datasets. Defaults to ../Dataset_Cache/')
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"🚀 Starting Training on DGX")
    print(f"Scenario: {args.scenario}")
    print(f"Train Dataset: {args.train_dataset}")
    print(f"Test Dataset: {args.test_dataset}")
    print(f"Model: {args.model}")
    print(f"Base Dir: {args.base_dir}")
    print(f"{'='*50}\n")

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
    TRAIN_DIRS = DATASET_PATHS.get(args.train_dataset, [])
    TEST_DIRS  = DATASET_PATHS.get(args.test_dataset, [])

    if args.scenario == 'Unified':
        BASE_SAVE_DIR = f"../Result/Intra_Unified/{args.model}_Experiment"
    elif args.scenario == 'Intra':
        BASE_SAVE_DIR = f"../Result/Intra_{args.train_dataset}/{args.model}_Experiment"
    else:
        BASE_SAVE_DIR = f"../Result/Multi_{args.train_dataset}_to_{args.test_dataset}/{args.model}_Experiment"
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # 1. LOAD DATA
    print("Loading Data...")

    if args.scenario == 'Unified':
        # ── Unified: merge all 3 datasets, then split 80/20 stratified ──
        print("  Loading TMC-UCM for Unified...")
        tmc_imgs, tmc_feats, tmc_labels, tmc_paths = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None, cache_dir=args.cache_dir)
        print("  Loading NTUH for Unified...")
        ntuh_imgs, ntuh_feats, ntuh_labels, ntuh_paths = load_all_images(DATASET_PATHS['NTUH'], 'NTUH', cache_dir=args.cache_dir)
        print("  Loading LIMUC for Unified...")
        limuc_imgs, limuc_feats, limuc_labels, limuc_paths = load_all_images(
            [DATASET_PATHS['LIMUC'][0], DATASET_PATHS['LIMUC'][1]], 'LIMUC', cache_dir=args.cache_dir
        )
        all_imgs   = tmc_imgs   + ntuh_imgs   + limuc_imgs
        all_feats  = tmc_feats  + ntuh_feats  + limuc_feats
        all_labels = tmc_labels + ntuh_labels + limuc_labels
        all_paths  = tmc_paths  + ntuh_paths  + limuc_paths
        print(f"  Unified pool: {len(all_imgs)} images total")
        X_train_img_raw, X_test_img, X_train_feat_raw, X_test_feat, y_train_label_raw, y_test_label, _, _ = train_test_split(
            all_imgs, all_feats, all_labels, all_paths, test_size=0.2, random_state=42, stratify=all_labels
        )
    elif args.scenario == 'Intra':
        if args.train_dataset == 'LIMUC':
            X_train_img_raw, X_train_feat_raw, y_train_label_raw, _ = load_all_images([TRAIN_DIRS[0]], args.train_dataset, cache_dir=args.cache_dir)
            X_test_img, X_test_feat, y_test_label, _ = load_all_images([TRAIN_DIRS[1]], args.train_dataset, cache_dir=args.cache_dir)
        elif args.train_dataset == 'TMC-UCM':
            X_train_img_raw, X_train_feat_raw, y_train_label_raw, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Train', cache_dir=args.cache_dir)
            X_test_img, X_test_feat, y_test_label, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Test', cache_dir=args.cache_dir)
        else:
            all_imgs, all_feats, all_labels, all_paths = load_all_images(TRAIN_DIRS, args.train_dataset, cache_dir=args.cache_dir)
            X_train_img_raw, X_test_img, X_train_feat_raw, X_test_feat, y_train_label_raw, y_test_label, _, _ = train_test_split(
                all_imgs, all_feats, all_labels, all_paths, test_size=0.2, random_state=42, stratify=all_labels
            )
    else:
        if args.train_dataset == 'TMC-UCM':
            X_train_img_raw, X_train_feat_raw, y_train_label_raw, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None, cache_dir=args.cache_dir)
        else:
            X_train_img_raw, X_train_feat_raw, y_train_label_raw, _ = load_all_images(TRAIN_DIRS, args.train_dataset, cache_dir=args.cache_dir)

        if args.test_dataset == 'TMC-UCM':
            X_test_img, X_test_feat, y_test_label, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None, cache_dir=args.cache_dir)
        else:
            X_test_img, X_test_feat, y_test_label, _ = load_all_images(TEST_DIRS, args.test_dataset, cache_dir=args.cache_dir)

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

    # Save the base scaler and UMAP model
    joblib.dump(scaler, os.path.join(BASE_SAVE_DIR, 'base_scaler.pkl'))
    joblib.dump(umap_reducer, os.path.join(BASE_SAVE_DIR, 'umap_model.pkl'))

    # Model Training
    model_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_hybrid.keras")
    
    if args.agent_only:
        print(f"\n[1] --agent_only flag detected. Skipping Deep Learning training.")
        if os.path.exists(model_path):
            print(f"  -> Loading existing model from {model_path}")
            model = tf.keras.models.load_model(model_path, custom_objects={
                "resnet50_preprocess": dgx_models.resnet50_preprocess,
                "densenet_preprocess": dgx_models.densenet_preprocess,
                "efficientnet_preprocess": dgx_models.efficientnet_preprocess,
                "convnext_preprocess": dgx_models.convnext_preprocess,
                "vit_preprocess": dgx_models.vit_preprocess,
            }, compile=False)
        else:
            raise FileNotFoundError(f"Cannot run --agent_only because {model_path} does not exist!")
    else:
        print(f"\n[1] Training Base Hybrid Model: {args.model}")
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        model = build_hybrid_model(
            branch_builder_func=MODEL_BUILDERS[args.model],
            image_input_shape=(224, 224, 3),
            feat_input_shape=(20,),
            umap_feat_shape=(2,),
            num_classes=len(le.classes_),
            dropout_rate=0.3
        )

        # Cosine Decay LR schedule for smoother convergence
        EPOCHS = 150
        BATCH_SIZE = 32
        steps_per_epoch = max(1, len(X_img_train) // BATCH_SIZE)
        cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=steps_per_epoch * EPOCHS,
            alpha=1e-6  # minimum LR
        )

        model.compile(
            optimizer=Adam(learning_rate=cosine_decay),
            loss=focal_loss_with_label_smoothing(gamma=2.5, alpha=0.25, label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        # Validation strictly uses val set, avoiding test set leakage
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1, mode='max'),
        ]

        history = model.fit(
            [X_img_train, X_feat_train_scaled, X_train_umap], y_train_cat,
            validation_data=([X_img_val, X_feat_val_scaled, X_val_umap], y_val_cat),
            batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=class_weight_dict, callbacks=callbacks, verbose=1
        )

        model.save(model_path)
        print(f"✅ Saved base model to {model_path}")

    # Agent Training (Trained on Validation Set to prevent overfitting)
    print(f"\n[2] Training Super Agent (LightGBM on Validation Set)")
    y_pred_proba_val = model.predict([X_img_val, X_feat_val_scaled, X_val_umap], verbose=0)
    
    y_pred_proba_test = model.predict([X_img_test, X_feat_test_scaled, X_test_umap], verbose=0)

    def make_features(proba, umap_feat, h_feat):
        df = pd.DataFrame(h_feat, columns=[f"f{i}" for i in range(20)])
        for i in range(proba.shape[1]):
            df[f"prob_class_{i}"] = proba[:, i]
        df["confidence"] = np.max(proba, axis=1)
        df["umap_0"] = umap_feat[:, 0]
        df["umap_1"] = umap_feat[:, 1]
        return df

    df_val_ag = make_features(y_pred_proba_val, X_val_umap, X_feat_val_scaled)
    df_test_ag  = make_features(y_pred_proba_test, X_test_umap, X_feat_test_scaled)
    
    num_classes = y_pred_proba_val.shape[1]
    prob_cols = [f"prob_class_{i}" for i in range(num_classes)]
    features = prob_cols + ["confidence", "umap_0", "umap_1"] + [f"f{i}" for i in range(20)]
    
    scaler_ag = StandardScaler()
    
    print(f"  -> Training Agent on all {len(df_val_ag)} validation cases to learn realistic confidence bounds")
    X_tr = scaler_ag.fit_transform(df_val_ag[features].values)
    y_tr = y_val_encoded
        
    clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    
    X_te = scaler_ag.transform(df_test_ag[features].values)
    y_pred_agent = clf.predict(X_te)
    
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
    
    y_pred_deep = np.argmax(y_pred_proba_test, axis=1)
    base_acc = accuracy_score(y_true, y_pred_deep)
    
    conf_test = np.max(y_pred_proba_test, axis=1)
    low_conf_mask = conf_test < args.threshold
    
    y_pred_hybrid = np.where(low_conf_mask, y_pred_agent, y_pred_deep)
    hybrid_acc = accuracy_score(y_true, y_pred_hybrid)
    
    print(f"  📊 BASE DEEP LEARNING ACCURACY: {base_acc:.4f}  ({base_acc*100:.2f}%)")
    print(f"  ⚙️  HYBRID SELECTOR (Threshold = {args.threshold})")
    print(f"  🔍 Delegated {low_conf_mask.sum()} / {len(low_conf_mask)} low-confidence cases to Agent")
    print(f"  🚀 FINAL HYBRID ACCURACY:       {hybrid_acc:.4f}  ({hybrid_acc*100:.2f}%)")

    acc = accuracy_score(y_true, y_pred_hybrid)
    prec = precision_score(y_true, y_pred_hybrid, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred_hybrid, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred_hybrid, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred_hybrid, weights='quadratic')
    
    cm = confusion_matrix(y_true, y_pred_hybrid)
    specs = []
    for i in range(len(le.classes_)):
        tn = np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + cm[i,i]
        fp = np.sum(cm[:,i]) - cm[i,i]
        specs.append(tn / (tn + fp + 1e-6))
    spec = np.mean(specs)

    metrics = {
        'Model': args.model,
        'Base_Accuracy': float(base_acc),
        'Hybrid_Accuracy': float(hybrid_acc),
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
