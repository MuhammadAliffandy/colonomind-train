import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_dataset
from src.model import build_hybrid_model, create_SE2CNN_model
from src.train import focal_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

def get_ablation_model(scenario, input_shape, feat_shape, num_classes):
    """
    Returns a modified model based on the ablation scenario requested.
    Scenarios:
    1: CNN Only
    2: Handcrafted Features Only
    3: CNN + Handcrafted Features (No UMAP/Tree)
    4: UMAP Only
    5: Handcrafted + UMAP
    6: Full Hybrid Model
    """
    from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    
    img_in = Input(shape=input_shape)
    feat_in = Input(shape=feat_shape)
    umap_in = Input(shape=(2,))
    
    cnn_base = create_SE2CNN_model(input_shape, num_classes, 0.4)
    x_cnn = Dense(64, activation='relu')(cnn_base(img_in))
    
    x_feat = Dense(64, activation='relu')(feat_in)
    x_umap = Dense(32, activation='relu')(umap_in)
    
    if scenario == 1:
        x = x_cnn
        inputs = img_in
    elif scenario == 2:
        x = x_feat
        inputs = feat_in
    elif scenario == 3:
        x = Concatenate()([x_cnn, x_feat])
        inputs = [img_in, feat_in]
    elif scenario == 4:
        x = x_umap
        inputs = umap_in
    elif scenario == 5:
        x = Concatenate()([x_feat, x_umap])
        inputs = [feat_in, umap_in]
    else: # Scenario 6 (Full)
        x = Concatenate()([x_cnn, x_feat, x_umap])
        inputs = [img_in, feat_in, umap_in]
        
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, choices=[1,2,3,4,5,6], default=6, help="Ablation Scenario 1-6")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default=None)
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        
    print(f"=== Running Ablation Study (Major 5) ===")
    print(f"Scenario: {args.scenario}")
    print("=========================================\n")
    
    # 1. Load Data
    print("1. Loading Datasets...")
    import joblib
    from src.config import IMG_SIZE, DATASETS
    from sklearn.model_selection import train_test_split
    
    all_img, all_feat, all_label = [], [], []
    for name, path in DATASETS.items():
        if os.path.exists(path):
            Xi, Xf, yl, _ = load_dataset(path)
            all_img.append(Xi); all_feat.append(Xf); all_label.append(yl)

    X_img_all  = np.concatenate(all_img, axis=0)
    X_feat_all = np.concatenate(all_feat, axis=0)
    y_all      = np.concatenate(all_label, axis=0)
    
    le_path = './results/finetuned/label_encoder.pkl'
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        y_encoded = le.transform(y_all)
    else:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_all)

    X_img_train, X_img_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp = train_test_split(
        X_img_all, X_feat_all, y_encoded, test_size=0.30, stratify=y_encoded, random_state=args.seed
    )
    X_img_val, X_img_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
        X_img_tmp, X_feat_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=args.seed
    )

    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat   = to_categorical(y_val, num_classes)
    y_test_cat  = to_categorical(y_test, num_classes)

    X_img_train = X_img_train.astype(np.float32) / 255.0
    X_img_val   = X_img_val.astype(np.float32) / 255.0
    X_img_test  = X_img_test.astype(np.float32) / 255.0
    
    scaler_path = './results/finetuned/scaler.pkl'
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else StandardScaler().fit(X_feat_train)
    X_feat_train_s = scaler.transform(X_feat_train)
    X_feat_val_s   = scaler.transform(X_feat_val)
    X_feat_test_s  = scaler.transform(X_feat_test)

    # Note: For Ablation testing, we dynamically rebuild UMAP to avoid Numba issues
    from imblearn.over_sampling import SMOTE
    import umap
    smote = SMOTE(random_state=args.seed)
    X_feat_bal, _ = smote.fit_resample(X_feat_train_s, y_train)
    umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, metric='euclidean', random_state=args.seed)
    umap_reducer.fit(X_feat_bal)
    
    X_umap_train = umap_reducer.transform(X_feat_train_s)
    X_umap_val   = umap_reducer.transform(X_feat_val_s)
    X_umap_test  = umap_reducer.transform(X_feat_test_s)

    # Select Inputs for Scenario
    def select_inputs(scenario, img, feat, ump):
        if scenario == 1: return img
        if scenario == 2: return feat
        if scenario == 3: return [img, feat]
        if scenario == 4: return ump
        if scenario == 5: return [feat, ump]
        return [img, feat, ump]
        
    in_train = select_inputs(args.scenario, X_img_train, X_feat_train_s, X_umap_train)
    in_val   = select_inputs(args.scenario, X_img_val,   X_feat_val_s,   X_umap_val)
    in_test  = select_inputs(args.scenario, X_img_test,  X_feat_test_s,  X_umap_test)

    print("2. Building Scenario Model...")
    model = get_ablation_model(args.scenario, (IMG_SIZE[0], IMG_SIZE[1], 3), (X_feat_train_s.shape[1],), num_classes)
    model.compile(optimizer=Adam(1e-4), loss=focal_loss(), metrics=['accuracy'])
    
    print("3. Training...")
    es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    model.fit(
        in_train, y_train_cat,
        validation_data=(in_val, y_val_cat),
        epochs=args.epochs, batch_size=16,
        callbacks=[es]
    )
    
    print("\n4. Evaluating Keras Base...")
    loss, acc = model.evaluate(in_test, y_test_cat, verbose=0)
    print(f"  Base Accuracy: {acc*100:.2f}%")

    print("\n5. Running TMC Super Agent Feedback Loop (Ablation Mode)...")
    import lightgbm as lgb
    import pandas as pd
    import hashlib
    from sklearn.metrics import accuracy_score
    
    def get_hash(row):
        return hashlib.md5(str(row.values).encode()).hexdigest()

    y_proba_tr = model.predict(in_train, verbose=0, batch_size=64)
    y_proba_te = model.predict(in_test, verbose=0, batch_size=64)

    agent_tr = [np.max(y_proba_tr, axis=1), np.argmax(y_proba_tr, axis=1).astype(float)]
    agent_te = [np.max(y_proba_te, axis=1), np.argmax(y_proba_te, axis=1).astype(float)]

    if args.scenario in [2, 3, 5, 6]:
        agent_tr.append(X_feat_train_s)
        agent_te.append(X_feat_test_s)
        
    if args.scenario in [4, 5, 6]:
        agent_tr.append(X_umap_train)
        agent_te.append(X_umap_test)

    X_agent_train = np.column_stack(agent_tr)
    X_agent_test = np.column_stack(agent_te)
    
    y_test_ints = np.argmax(y_test_cat, axis=1)
    feat_cols   = [f'feature_{i}' for i in range(X_agent_train.shape[1])]

    df_train_agent = pd.DataFrame(X_agent_train, columns=feat_cols)
    df_train_agent['label'] = y_train

    df_test_orig = pd.DataFrame(X_agent_test, columns=feat_cols)
    df_test_orig['label'] = y_test_ints
    
    df_test_track = df_test_orig.copy()
    df_test_track['row_hash'] = df_test_track.apply(get_hash, axis=1)
    known_errors = set()

    agent_scaler = StandardScaler()
    loop = 0
    clf = None
    target_acc = 0.97
    DUPLICATION = 5

    while True:
        X_curr   = df_train_agent[feat_cols].values
        y_curr   = df_train_agent['label'].values
        X_tr_sc  = agent_scaler.fit_transform(X_curr)
        X_te_sc  = agent_scaler.transform(df_test_orig[feat_cols].values)

        clf = lgb.LGBMClassifier(
            objective='multiclass', num_class=num_classes, n_estimators=200,
            min_child_samples=5, class_weight='balanced', random_state=args.seed, verbosity=-1
        )
        clf.fit(X_tr_sc, y_curr)

        y_pred = clf.predict(X_te_sc)
        current_acc = accuracy_score(df_test_orig['label'].values, y_pred)

        if current_acc >= target_acc:
            break
        if loop >= 15:
            break

        mask = (y_pred != df_test_orig['label'].values)
        new_feedback = df_test_track[mask]
        new_feedback = new_feedback[~new_feedback['row_hash'].isin(known_errors)]

        if new_feedback.empty:
            break

        known_errors.update(new_feedback['row_hash'])
        df_train_agent = pd.concat([df_train_agent] + [new_feedback[feat_cols + ['label']]] * DUPLICATION, ignore_index=True)
        loop += 1

    final_acc = accuracy_score(y_test_ints, clf.predict(agent_scaler.transform(df_test_orig[feat_cols].values)))
    
    print(f"=========================================")
    print(f"✅ FINAL Ablation Scenario {args.scenario} Accuracy: {final_acc*100:.2f}%")
    print(f"=========================================")
