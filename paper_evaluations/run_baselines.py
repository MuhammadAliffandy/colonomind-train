import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0, ConvNeXtTiny
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import IMG_SIZE, DATASETS
from src.data_loader import load_dataset

def get_baseline_model(model_name, input_shape, num_classes):
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    
    if model_name == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'densenet':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'convnext':
        base_model = ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'vit':
        try:
            from vit_keras import vit
            base_model = vit.vit_b16(image_size=input_shape[0], activation='softmax', pretrained=True, include_top=False, pretrained_top=False, classes=num_classes)
        except ImportError:
            raise ImportError("Please run `pip install vit-keras` to use the ViT baseline model.")
    else:
        raise ValueError(f"Model {model_name} not supported natively in this skeleton.")
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet', 'densenet', 'efficientnet', 'convnext', 'vit'], default='resnet')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default=None)
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        
    print(f"=== Baseline CNN Model Training (Major 13) ===")
    print(f"Model: {args.model.upper()}")
    print("==============================================\n")
    
    # 1. Load Data
    print("1. Loading Datasets...")
    all_img, all_label = [], []
    for name, path in DATASETS.items():
        if os.path.exists(path):
            Xi, _, yl, _ = load_dataset(path)
            all_img.append(Xi)
            all_label.append(yl)

    X_img_all = np.concatenate(all_img, axis=0)
    y_all = np.concatenate(all_label, axis=0)
    
    # Load Label Encoder if exists to ensure same mapping
    le_path = './results/finetuned/label_encoder.pkl'
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        y_encoded = le.transform(y_all)
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_all)

    # EXACT Same Split as Hybrid Training (Seed=42)
    X_img_train, X_img_tmp, y_train, y_tmp = train_test_split(
        X_img_all, y_encoded, test_size=0.30, stratify=y_encoded, random_state=args.seed
    )
    X_img_val, X_img_test, y_val, y_test = train_test_split(
        X_img_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=args.seed
    )
    
    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat   = to_categorical(y_val, num_classes)
    y_test_cat  = to_categorical(y_test, num_classes)

    X_img_train = X_img_train.astype(np.float32) / 255.0
    X_img_val   = X_img_val.astype(np.float32) / 255.0
    X_img_test  = X_img_test.astype(np.float32) / 255.0

    print("2. Building Model...")
    model = get_baseline_model(args.model, (IMG_SIZE[0], IMG_SIZE[1], 3), num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("3. Training Baseline...")
    es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(
        X_img_train, y_train_cat,
        validation_data=(X_img_val, y_val_cat),
        epochs=args.epochs, batch_size=16,
        callbacks=[es]
    )
    
    print("\n4. Evaluating on Test Set...")
    loss, acc = model.evaluate(X_img_test, y_test_cat, verbose=0)
    print(f"==============================================")
    print(f"✅ {args.model.upper()} Baseline Final Test Accuracy: {acc*100:.2f}%")
    print(f"==============================================")
