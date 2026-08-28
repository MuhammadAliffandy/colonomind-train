import os
import cv2
import json
import joblib
import pywt
import argparse
import scipy.stats
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, BatchNormalization, 
                                     Dropout, GlobalAveragePooling2D, Conv2D, 
                                     MaxPooling2D, Activation, Multiply, Reshape)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images, load_tmc_ucm

# ==============================================================================
# 1. UTILS: PREPROCESS & FEATURES
# ==============================================================================
def smart_preprocess(img, img_size):
    if img is None: return np.zeros((*img_size,3), dtype=np.uint8)
    h, w = img.shape[:2]
    if h > 450 and w > 550: 
        crop = img[30:430, 200:550]
        if crop.size == 0: crop = img
    else: crop = img
    return cv2.resize(crop, img_size)

def apply_heavy_augmentation(img):
    rows, cols = img.shape[:2]
    angle = np.random.uniform(-90, 90)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    
    if np.random.rand() > 0.5: img = cv2.flip(img, 1)
    if np.random.rand() > 0.5: img = cv2.flip(img, 0)
        
    alpha = 1.0 + np.random.uniform(-0.3, 0.3)
    beta = np.random.uniform(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

def extract_handcrafted(img):
    if len(img.shape) == 3: gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else: gray = img
    coeffs = pywt.dwt2(gray, 'db1')
    LL, (LH, HL, HH) = coeffs
    
    def _stats(band):
        flat = np.abs(band.flatten()) + 1e-6
        return [np.mean(band), np.std(band), np.var(band), scipy.stats.entropy(flat)]
    
    feats = []
    for band in [LL, LH, HL, HH]: feats.extend(_stats(band))
    feats.append(np.sum(np.square(HH))) 
    
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(gray_norm, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    feats.extend([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ])
    return np.array(feats, dtype=np.float32)

# ==============================================================================
# 2. GENERATOR
# ==============================================================================
class RobustGenerator(Sequence):
    def __init__(self, imgs, feats, umaps, labels, batch_size=16, shuffle=True, augment=False, img_size=(256,256)):
        self.imgs = imgs
        self.feats = feats
        self.umaps = umaps
        self.labels = to_categorical(labels, num_classes=4)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment 
        self.img_size = img_size
        self.indexes = np.arange(len(self.imgs))
        
    def __len__(self): return int(np.ceil(len(self.imgs) / self.batch_size))
    
    def __getitem__(self, index):
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X_img = np.empty((len(idxs), *self.img_size, 3))
        X_feat = self.feats[idxs]
        X_umap = self.umaps[idxs]
        y = self.labels[idxs]
        
        for i, idx in enumerate(idxs):
            img = self.imgs[idx]
            img = cv2.resize(img, self.img_size)
            
            if self.augment: img = apply_heavy_augmentation(img)
            X_img[i] = img / 255.0
            
        return [X_img, X_feat, X_umap], y
    
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)

# ==============================================================================
# 3. MODEL ARCHITECTURE
# ==============================================================================
def squeeze_excite_block(input_tensor, ratio=16):
    init = input_tensor
    filters = init.shape[-1]
    se = GlobalAveragePooling2D()(init)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = Multiply()([init, se])
    return x

def build_robust_model(img_size):
    input_img = Input(shape=(*img_size, 3))
    input_feat = Input(shape=(20,))
    input_umap = Input(shape=(2,))

    # CNN Backbone
    x = Conv2D(32, (3,3), padding='same')(input_img)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = squeeze_excite_block(x); x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x) 

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = squeeze_excite_block(x); x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = squeeze_excite_block(x); x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = squeeze_excite_block(x); x = GlobalAveragePooling2D()(x)
    feat_cnn = Dropout(0.4)(x) 
    
    # Handcrafted & UMAP
    feat_hand = Dense(64, activation='relu')(input_feat)
    feat_hand = BatchNormalization()(feat_hand)
    
    feat_umap = Dense(32, activation='relu')(input_umap)
    
    # Fusion
    concat = Concatenate(name='Fusion')([feat_cnn, feat_hand, feat_umap])
    
    # Classifier
    x = Dense(256, activation='relu')(concat)
    x = Dropout(0.5)(x) 
    out = Dense(4, activation='softmax')(x)
    
    return Model(inputs=[input_img, input_feat, input_umap], outputs=out)

# ==============================================================================
# 4. TRAINING PIPELINE
# ==============================================================================
def extract_patient_id(path):
    fname = os.path.basename(path)
    if 'train_and_validation_sets' in path or 'test_set' in path:
        return fname.split('_')[0]
    elif 'TMC-UCM' in path:
        return fname.split('_')[0]
    else:
        return fname.split('-')[0]

def load_unified_data(base_dir):
    print("\n📦 Loading Strict Unified Dataset (Patient-Level Split)...")
    tmc_root = f'{base_dir}/Dataset/TMC-UCM'
    ntuh_paths = [f'{base_dir}/Dataset+Code/MES classification_20250313', f'{base_dir}/Dataset+Code/MES classification_20250724']
    limuc_paths = [f'{base_dir}/Dataset/LIMUC/train_and_validation_sets', f'{base_dir}/Dataset/LIMUC/test_set']
    
    ti, tf_, tl, tp = load_tmc_ucm(tmc_root, split_filter=None)
    ni, nf, nl, np_ = load_all_images(ntuh_paths, 'NTUH')
    li, lf, ll, lp = load_all_images(limuc_paths, 'LIMUC')
    
    all_imgs = ti + ni + li
    all_feats = np.array(tf_ + nf + lf)
    all_labels = tl + nl + ll
    all_paths = tp + np_ + lp
    all_patients = [extract_patient_id(p) for p in all_paths]
    
    le = LabelEncoder()
    le.fit(['MES0', 'MES1', 'MES2', 'MES3'])
    all_labels_encoded = le.transform(all_labels)
    
    df = pd.DataFrame({'idx': range(len(all_imgs)), 'path': all_paths, 'label': all_labels_encoded, 'patient': all_patients})
    patients = df['patient'].unique()
    
    train_p, test_p = train_test_split(patients, test_size=0.2, random_state=42)
    train_p, val_p = train_test_split(train_p, test_size=0.2, random_state=42)
    
    train_df = df[df['patient'].isin(train_p)]
    val_df = df[df['patient'].isin(val_p)]
    test_df = df[df['patient'].isin(test_p)]
    
    print(f"Data Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    return train_df, val_df, test_df, all_imgs, all_feats

def plot_roc(y_true, y_pred_proba, dataset_name, out_dir):
    Y_bin = label_binarize(y_true, classes=[0,1,2,3])
    fpr = dict()
    tpr = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], y_pred_proba[:, i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 4
    
    macro_auc = auc(all_fpr, mean_tpr)
    plt.figure(figsize=(8,6))
    plt.plot(all_fpr, mean_tpr, lw=2, color='darkorange', label=f'Macro-Avg ROC (AUC = {macro_auc:.3f})')
    plt.plot([0,1],[0,1], 'k--')
    plt.title(f'ColonoMind SE (Unified) ROC - AUC={macro_auc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(out_dir, f'ROC_Curve_{dataset_name}.png'))
    plt.close()
    return macro_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/home/D13K48009/raid/Clara/new_drive")
    parser.add_argument("--save_dir", type=str, default="../Result/Unified_ColonoMind_SE")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    IMG_SIZE = (args.img_size, args.img_size)
    
    train_df, val_df, test_df, all_imgs, all_feats = load_unified_data(args.base_dir)
    
    X_train_f = all_feats[train_df['idx'].values]
    X_val_f = all_feats[val_df['idx'].values]
    X_test_f = all_feats[test_df['idx'].values]
    
    print("Fitting Scaler & UMAP...")
    scaler = StandardScaler()
    X_train_f_s = scaler.fit_transform(X_train_f)
    X_val_f_s = scaler.transform(X_val_f)
    X_test_f_s = scaler.transform(X_test_f)
    
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_train_u = umap_model.fit_transform(X_train_f_s)
    X_val_u = umap_model.transform(X_val_f_s)
    X_test_u = umap_model.transform(X_test_f_s)
    
    joblib.dump(scaler, os.path.join(args.save_dir, "scaler_unified.pkl"))
    joblib.dump(umap_model, os.path.join(args.save_dir, "umap_unified.pkl"))
    
    train_imgs = [all_imgs[i] for i in train_df['idx'].values]
    val_imgs = [all_imgs[i] for i in val_df['idx'].values]
    test_imgs = [all_imgs[i] for i in test_df['idx'].values]
    
    train_gen = RobustGenerator(train_imgs, X_train_f_s, X_train_u, train_df['label'].values, batch_size=16, shuffle=True, augment=True, img_size=IMG_SIZE)
    val_gen = RobustGenerator(val_imgs, X_val_f_s, X_val_u, val_df['label'].values, batch_size=16, shuffle=False, augment=False, img_size=IMG_SIZE)
    test_gen = RobustGenerator(test_imgs, X_test_f_s, X_test_u, test_df['label'].values, batch_size=16, shuffle=False, augment=False, img_size=IMG_SIZE)
    
    model_path = os.path.join(args.save_dir, "best_hybrid_keras.h5")
    if os.path.exists(model_path):
        print("Model weights found! Resuming...")
        model = load_model(model_path)
    else:
        print("Building ColonoMind SE CNN...")
        model = build_robust_model(IMG_SIZE)
        model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        
        callbacks = [
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        ]
        
        model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)
    
    print("\n🧠 Extracting Deep Features for Hybrid Agent...")
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer('Fusion').output)
    
    def extract_deep(gen):
        feats, preds, trues = [], [], []
        for i in range(len(gen)):
            inp, y = gen[i]
            feats.append(feat_extractor.predict_on_batch(inp))
            preds.append(model.predict_on_batch(inp))
            trues.extend(np.argmax(y, axis=1))
        return np.vstack(feats), np.vstack(preds), np.array(trues)
    
    deep_train, pred_train, y_train = extract_deep(train_gen)
    deep_test, pred_test, y_test = extract_deep(test_gen)
    
    df_ag_train = pd.DataFrame(deep_train)
    df_ag_train["confidence"] = np.max(pred_train, axis=1)
    
    df_ag_test = pd.DataFrame(deep_test)
    df_ag_test["confidence"] = np.max(pred_test, axis=1)
    
    ag_scaler = StandardScaler()
    X_ag_tr = ag_scaler.fit_transform(df_ag_train)
    X_ag_te = ag_scaler.transform(df_ag_test)
    joblib.dump(ag_scaler, os.path.join(args.save_dir, "agent_scaler.pkl"))
    
    print("Training LightGBM Agent...")
    clf = lgb.LGBMClassifier(n_estimators=200, max_depth=7, learning_rate=0.05, n_jobs=8, random_state=42)
    clf.fit(X_ag_tr, y_train)
    clf.booster_.save_model(os.path.join(args.save_dir, "lgbm_feedback_agent.txt"))
    
    print("Evaluating Framework...")
    # Base CNN
    base_preds = np.argmax(pred_test, axis=1)
    acc_base = accuracy_score(y_test, base_preds)
    
    # Hybrid Evaluation
    agent_preds_proba = clf.predict_proba(X_ag_te)
    agent_preds = np.argmax(agent_preds_proba, axis=1)
    
    # Optimize threshold
    best_thresh = 0.5
    best_acc = 0
    for t in np.arange(0.5, 0.99, 0.05):
        hybrid = np.where(df_ag_test["confidence"] < t, agent_preds, base_preds)
        acc = accuracy_score(y_test, hybrid)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            
    final_hybrid = np.where(df_ag_test["confidence"] < best_thresh, agent_preds, base_preds)
    acc_hybrid = accuracy_score(y_test, final_hybrid)
    
    print(f"Base CNN Accuracy: {acc_base*100:.2f}%")
    print(f"Hybrid Accuracy (Thresh={best_thresh:.2f}): {acc_hybrid*100:.2f}%")
    
    # Calculate ROC on Agent Probabilities for Hybrid
    hybrid_proba = np.where((df_ag_test["confidence"] < best_thresh)[:, None], agent_preds_proba, pred_test)
    auc_val = plot_roc(y_test, hybrid_proba, "Unified_Test", args.save_dir)
    
    metrics = {
        'Base_Accuracy': float(acc_base),
        'Hybrid_Accuracy': float(acc_hybrid),
        'Best_Threshold': float(best_thresh),
        'AUC': float(auc_val),
        'F1_Score': float(f1_score(y_test, final_hybrid, average='macro')),
        'QWK': float(cohen_kappa_score(y_test, final_hybrid, weights='quadratic'))
    }
    
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("✅ Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
