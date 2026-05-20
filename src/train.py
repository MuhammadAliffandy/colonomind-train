import os
import argparse
import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import umap
import itertools
import tqdm

from src.config import IMG_SIZE, DATASETS
from src.data_loader import load_dataset
from src.model import build_hybrid_model

def resolve_dataset_path(name_or_path):
    """Resolves a dataset short name (e.g. 'dataset_1') to its full server path,
    or returns the path as-is if it's already a valid directory."""
    if name_or_path in DATASETS:
        resolved = DATASETS[name_or_path]
        print(f"  Resolved '{name_or_path}' -> {resolved}")
        return resolved
    return name_or_path

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = resolve_dataset_path(args.train_dir)
    test_path  = resolve_dataset_path(args.test_dir)
    
    print(f"Loading training data from {train_path}...")
    X_img_train_raw, X_feat_train_raw, y_train_label, img_paths_train = load_dataset(train_path)
    print(f"Loading testing data from {test_path}...")
    X_img_test_raw,  X_feat_test_raw,  y_test_label,  img_paths_test  = load_dataset(test_path)
    
    X_img_train = X_img_train_raw.astype(np.float32) / 255.0
    X_img_test  = X_img_test_raw.astype(np.float32) / 255.0

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_label)
    y_test_encoded  = le.transform(y_test_label)
    y_train_cat = to_categorical(y_train_encoded, num_classes=len(le.classes_))
    y_test_cat  = to_categorical(y_test_encoded,  num_classes=len(le.classes_))

    scaler = StandardScaler()
    X_feat_train_scaled = scaler.fit_transform(X_feat_train_raw)
    X_feat_test_scaled  = scaler.transform(X_feat_test_raw)

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_feat_train_bal, y_train_bal = smote.fit_resample(X_feat_train_scaled, y_train_encoded)

    # Map balanced features to real images
    X_img_train_bal, img_paths_train_bal = [], []
    for feat, label in zip(X_feat_train_bal, y_train_bal):
        dists = np.linalg.norm(X_feat_train_scaled[y_train_encoded == label] - feat, axis=1)
        idx = np.where(y_train_encoded == label)[0][np.argmin(dists)]
        X_img_train_bal.append(X_img_train[idx])
        img_paths_train_bal.append(img_paths_train[idx])
    X_img_train_bal = np.array(X_img_train_bal, dtype=np.float32)
    y_train_cat_bal = to_categorical(y_train_bal, num_classes=len(le.classes_))

    print("Running UMAP Projection...")
    umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, metric='euclidean', random_state=42)
    X_train_umap = umap_reducer.fit_transform(X_feat_train_bal)
    X_test_umap  = umap_reducer.transform(X_feat_test_scaled)

    num_classes = len(le.classes_)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_bal), y=y_train_bal)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    print("Building and Compiling Model...")
    model_hybrid = build_hybrid_model(
        image_input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        feat_input_shape=(X_feat_train_bal.shape[1],),
        umap_feat_shape=(2,),
        num_classes=num_classes,
        dropout_rate=0.4
    )

    model_hybrid.compile(
        optimizer=Adam(1e-5),
        loss=focal_loss(gamma=2.5, alpha=0.25),
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, verbose=1, mode='max')
    ]

    print("Starting Training...")
    train_inputs = [X_img_train_bal, X_feat_train_bal, X_train_umap]
    val_inputs = [X_img_test, X_feat_test_scaled, X_test_umap]

    history = model_hybrid.fit(
        train_inputs, y_train_cat_bal,
        validation_data=(val_inputs, y_test_cat),
        batch_size=args.batch_size,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    model_save_path = os.path.join(args.output_dir, "best_hybrid_model.h5")
    model_hybrid.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save components
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.pkl"))
    joblib.dump(umap_reducer, os.path.join(args.output_dir, "umap_model.pkl"))
    
    print("Optimization finished. Artifacts saved.")

if __name__ == '__main__':
    dataset_keys = ", ".join(DATASETS.keys())
    parser = argparse.ArgumentParser(
        description="Colonomind Model Training Script",
        epilog=f"Available dataset short names: {dataset_keys}"
    )
    parser.add_argument('--train_dir', type=str, required=True,
                        help=f"Dataset short name or full path. Short names: {dataset_keys}")
    parser.add_argument('--test_dir', type=str, required=True,
                        help=f"Dataset short name or full path. Short names: {dataset_keys}")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Path to save model and artifacts")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    
    args = parser.parse_args()
    main(args)
