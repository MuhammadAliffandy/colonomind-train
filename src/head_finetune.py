"""
ColonoMind - Head Fine-Tuning Script
=====================================
Fine-tunes ONLY the classification head (Dense layers after GAP)
while keeping the entire backbone frozen. Much faster than full retraining.

Usage (run from colonomind-train/ root):
    python -m src.head_finetune --dataset TMC-UCM --models_dir Result/Intra_TMC-UCM
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tabulate import tabulate
import joblib
import umap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import dgx_models
from src.dgx_dataloader import load_all_images, load_tmc_ucm


def load_train_test_data(dataset_name, base_dir):
    """Load train+test data using the same split logic as train_dgx.py."""
    DATASET_PATHS = {
        'NTUH': [
            f'{base_dir}/Dataset+Code/MES classification_20250313',
            f'{base_dir}/Dataset+Code/MES classification_20250724'
        ],
        'LIMUC': [
            f'{base_dir}/Dataset/LIMUC/train_and_validation_sets',
            f'{base_dir}/Dataset/LIMUC/test_set'
        ],
        'TMC-UCM': [f'{base_dir}/Dataset/TMC-UCM/images']
    }
    TMC_UCM_ROOT = f'{base_dir}/Dataset/TMC-UCM'

    if dataset_name == 'Unified':
        ti, tf_, tl, tp = load_tmc_ucm(TMC_UCM_ROOT, split_filter=None)
        ni, nf, nl, np_ = load_all_images(DATASET_PATHS['NTUH'], 'NTUH')
        li, lf, ll, lp  = load_all_images(
            [DATASET_PATHS['LIMUC'][0], DATASET_PATHS['LIMUC'][1]], 'LIMUC')
        ai = ti+ni+li; af = tf_+nf+lf; al = tl+nl+ll; ap = tp+np_+lp
        X_train_img, X_test_img, X_train_feat, X_test_feat, y_train, y_test, _, _ = \
            train_test_split(ai, af, al, ap, test_size=0.2, random_state=42, stratify=al)
    elif dataset_name == 'LIMUC':
        X_train_img, X_train_feat, y_train, _ = load_all_images(
            [DATASET_PATHS['LIMUC'][0]], dataset_name)
        X_test_img, X_test_feat, y_test, _ = load_all_images(
            [DATASET_PATHS['LIMUC'][1]], dataset_name)
    elif dataset_name == 'TMC-UCM':
        X_train_img, X_train_feat, y_train, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Train')
        X_test_img, X_test_feat, y_test, _ = load_tmc_ucm(TMC_UCM_ROOT, split_filter='Test')
    elif dataset_name == 'NTUH':
        ai, af, al, ap = load_all_images(DATASET_PATHS['NTUH'], dataset_name)
        X_train_img, X_test_img, X_train_feat, X_test_feat, y_train, y_test, _, _ = \
            train_test_split(ai, af, al, ap, test_size=0.2, random_state=42, stratify=al)
    else:
        return None

    return (np.array(X_train_img, dtype=np.float32), np.array(X_test_img, dtype=np.float32),
            np.array(X_train_feat), np.array(X_test_feat),
            y_train, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--base_dir", default="/home/D13K48009/raid/Clara/new_drive")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"⚡ Head Fine-Tuning: {args.dataset}")
    print(f"   models_dir : {args.models_dir}")
    print(f"   epochs     : {args.epochs}")
    print(f"   lr         : {args.lr}")
    print(f"{'='*60}\n")

    # Load data
    data = load_train_test_data(args.dataset, args.base_dir)
    if data is None:
        print("❌ Unknown dataset")
        return

    X_train_img, X_test_img, X_train_feat, X_test_feat, y_train_labels, y_test_labels = data

    le = LabelEncoder()
    le.fit(["MES0", "MES1", "MES2", "MES3"])
    y_train_enc = le.transform(y_train_labels)
    y_test_enc = le.transform(y_test_labels)

    # Split train into train/val
    X_tr_img, X_val_img, X_tr_feat, X_val_feat, y_tr, y_val = train_test_split(
        X_train_img, X_train_feat, y_train_enc, test_size=0.2, random_state=42, stratify=y_train_enc)

    y_tr_cat = to_categorical(y_tr, num_classes=4)
    y_val_cat = to_categorical(y_val, num_classes=4)

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    custom_objects = {
        "resnet50_preprocess": dgx_models.resnet50_preprocess,
        "densenet_preprocess": dgx_models.densenet_preprocess,
        "efficientnet_preprocess": dgx_models.efficientnet_preprocess,
        "convnext_preprocess": dgx_models.convnext_preprocess,
        "vit_preprocess": dgx_models.vit_preprocess
    }

    MODEL_NAMES = ["ResNet-50", "DenseNet-121", "EfficientNet-B4", "ConvNeXt-Tiny", "ViT-B-16"]
    results_before = []
    results_after = []

    for mn in MODEL_NAMES:
        exp_dir = os.path.join(args.models_dir, f"{mn}_Experiment")
        kp = os.path.join(exp_dir, f"{mn}_hybrid.keras")
        sp = os.path.join(exp_dir, "base_scaler.pkl")
        up = os.path.join(exp_dir, "umap_model.pkl")

        if not all(os.path.exists(p) for p in [kp, sp, up]):
            print(f"⏭️  {mn}: skipping (files missing)")
            continue

        print(f"\n{'─'*50}")
        print(f"⚡ Fine-tuning head: {mn}")

        model = tf.keras.models.load_model(kp, compile=False, custom_objects=custom_objects)
        scaler = joblib.load(sp)
        umap_model = joblib.load(up)

        # Scale features
        X_tr_s = scaler.transform(X_tr_feat)
        X_val_s = scaler.transform(X_val_feat)
        X_test_s = scaler.transform(X_test_feat)
        X_tr_u = umap_model.transform(X_tr_s)
        X_val_u = umap_model.transform(X_val_s)
        X_test_u = umap_model.transform(X_test_s)

        # Before accuracy
        proba_before = model.predict([X_test_img, X_test_s, X_test_u], verbose=0)
        acc_before = accuracy_score(y_test_enc, np.argmax(proba_before, axis=1))
        f1_before = f1_score(y_test_enc, np.argmax(proba_before, axis=1), average='macro', zero_division=0)
        print(f"   📊 BEFORE: Acc={acc_before*100:.2f}%, F1={f1_before*100:.2f}%")
        results_before.append({"Model": mn, "Acc": f"{acc_before*100:.2f}%", "F1": f"{f1_before*100:.2f}%"})

        # FREEZE everything except the last few Dense/BN/Dropout layers (classification head)
        # The hybrid model structure: CNN_branch → Dense(128) → BN → Dropout → Concat → Dense(128) → Dropout → Dense(4)
        # We unfreeze only the layers AFTER the backbone branches
        for layer in model.layers:
            layer.trainable = False

        # Unfreeze the classification head layers (last ~8 layers of the hybrid model)
        head_layer_types = (tf.keras.layers.Dense, tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)
        unfrozen = 0
        for layer in reversed(model.layers):
            if isinstance(layer, head_layer_types):
                layer.trainable = True
                unfrozen += 1
            elif isinstance(layer, tf.keras.layers.Concatenate):
                break  # Stop at the concatenation point

        # Also unfreeze layers right before concatenation (the per-branch Dense/BN/Dropout)
        # by continuing backwards through all Dense/BN/Dropout until we hit a Model or Lambda layer
        for layer in reversed(model.layers):
            if isinstance(layer, head_layer_types):
                layer.trainable = True
                unfrozen += 1
            elif isinstance(layer, (tf.keras.Model, tf.keras.layers.Lambda)):
                break

        trainable_count = sum(1 for l in model.layers if l.trainable)
        print(f"   🔓 Unfrozen {trainable_count} layers (head only)")

        # Compile with low LR
        model.compile(
            optimizer=Adam(learning_rate=args.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, mode='max')
        ]

        model.fit(
            [X_tr_img, X_tr_s, X_tr_u], y_tr_cat,
            validation_data=([X_val_img, X_val_s, X_val_u], y_val_cat),
            batch_size=32, epochs=args.epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks, verbose=1
        )

        # After accuracy
        proba_after = model.predict([X_test_img, X_test_s, X_test_u], verbose=0)
        acc_after = accuracy_score(y_test_enc, np.argmax(proba_after, axis=1))
        f1_after = f1_score(y_test_enc, np.argmax(proba_after, axis=1), average='macro', zero_division=0)
        diff = (acc_after - acc_before) * 100
        print(f"   📊 AFTER:  Acc={acc_after*100:.2f}%, F1={f1_after*100:.2f}% ({'+'if diff>=0 else ''}{diff:.2f}%)")
        results_after.append({"Model": mn, "Acc": f"{acc_after*100:.2f}%", "F1": f"{f1_after*100:.2f}%", "Δ": f"{'+' if diff>=0 else ''}{diff:.2f}%"})

        # Save fine-tuned model (overwrite)
        if acc_after > acc_before:
            model.save(kp)
            print(f"   ✅ Saved improved model to {kp}")
        else:
            print(f"   ⚠️  No improvement, keeping original model")

    # Summary table
    print(f"\n\n{'='*60}")
    print(f"📊 HEAD FINE-TUNING RESULTS — {args.dataset}")
    print(f"{'='*60}")

    H = ["Model", "Before Acc", "After Acc", "After F1", "Change"]
    rows = []
    for b, a in zip(results_before, results_after):
        rows.append([b["Model"], b["Acc"], a["Acc"], a["F1"], a["Δ"]])
    print(tabulate(rows, headers=H, tablefmt="grid"))


if __name__ == "__main__":
    main()
