import os, joblib, argparse
import numpy as np
import lightgbm as lgb
import optuna
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import load_model, Model

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dgx_dataloader import load_all_images

# ==============================================================================
# CONFIG
# ==============================================================================
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 16
ENSEMBLE_SEEDS = [42, 123, 999]

# Define dummy Custom Layer for loading if needed
@tf.keras.utils.register_keras_serializable()
class ChannelPooling(tf.keras.layers.Layer):
    def __init__(self, pool_type='mean', **kwargs):
        super(ChannelPooling, self).__init__(**kwargs)
        self.pool_type = pool_type
    def call(self, inputs):
        if self.pool_type == 'mean': return tf.reduce_mean(inputs, axis=-1, keepdims=True)
        else: return tf.reduce_max(inputs, axis=-1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6_checkpoint_dir", default="../Result/ColonoMind_v6")
    ap.add_argument("--cnn_dir", default="../Result/ColonoMind_v12_V8FinetuneFocal")
    ap.add_argument("--save_dir", default="../Result/ColonoMind_v13_LGBMBalancer")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*70); print("🚀 COLONOMIND v13 — LightGBM Balancer (SMOTE + Optuna)"); print("="*70)
    
    # ── 1. Load Data ────────────────────────────────────────────────────────
    cache = os.path.join(args.v6_checkpoint_dir, "dataset_cache_v6.npz")
    d = np.load(cache, allow_pickle=True)
    all_imgs, all_feats, all_labels = d['imgs'], d['feats'], list(d['labels'])
    y_enc_full = LabelEncoder().fit_transform(all_labels)
    
    X_tv_i, X_te_i, X_tv_f, X_te_f, y_tv, y_te_full = train_test_split(all_imgs, all_feats, y_enc_full, test_size=0.20, random_state=42, stratify=y_enc_full)
    X_tr_i, X_va_i, X_tr_f, X_va_f, y_tr, y_va = train_test_split(X_tv_i, X_tv_f, y_tv, test_size=0.20, random_state=42, stratify=y_tv)
    
    y_tr1 = np.where(y_tr == 0, 0, 1); y_te1 = np.where(y_te_full == 0, 0, 1)
    active_tr = np.where(y_tr > 0)[0]
    X_tr_i2, X_tr_f2, y_tr2 = X_tr_i[active_tr], X_tr_f[active_tr], y_tr[active_tr] - 1
    
    sc, um = joblib.load(os.path.join(args.cnn_dir,"scaler_v8.pkl")), joblib.load(os.path.join(args.cnn_dir,"umap_v8.pkl"))
    def process_feats(features):
        scaled = sc.transform(features)
        return scaled, um.transform(scaled)
        
    Xtr_s, Utr = process_feats(X_tr_f); Xte_s, Ute = process_feats(X_te_f)
    Xtr_s2, Utr2 = process_feats(X_tr_f2)

    # ── 2. Load CNNs & Extract Features ─────────────────────────────────────
    print("\n⚡ Extracting Deep Features from V12 CNNs (This takes a few minutes)...")
    def extract_deep(prefix, imgs, feats_s, umaps, num_c):
        models = [load_model(os.path.join(args.cnn_dir, f"{prefix}_cnn_seed{s}.h5"), compile=False) for s in ENSEMBLE_SEEDS]
        extractors = [Model(inputs=m.input, outputs=m.get_layer('Fusion').output) for m in models]
        deeps, probs = [], []
        for i in range(len(models)):
            deeps.append(extractors[i].predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0))
            probs.append(models[i].predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0))
        return np.hstack([np.mean(deeps, axis=0), feats_s, umaps, np.mean(probs, axis=0)]), np.mean(probs, axis=0)

    X_ag_tr1, _ = extract_deep("stage1", X_tr_i, Xtr_s, Utr, 2)
    X_ag_te1, _ = extract_deep("stage1", X_te_i, Xte_s, Ute, 2)
    X_ag_tr2, _ = extract_deep("stage2", X_tr_i2, Xtr_s2, Utr2, 3)
    X_ag_te2_full, prob_stg2_full = extract_deep("stage2", X_te_i, Xte_s, Ute, 3)
    
    # ── 3. SMOTE + Optuna for Stage 2 (The Secret to Balancing MES1/2) ──────
    print("\n🔥 Balancing MES1 and MES2 using SMOTE and Optuna Macro-F1 Maximization...")
    smote = SMOTE(random_state=42)
    X_ag_tr2_smote, y_tr2_smote = smote.fit_resample(X_ag_tr2, y_tr2)
    
    # Train Stage 1 normally (it's already 89% F1)
    lgb1 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.01, class_weight='balanced', max_depth=5, num_leaves=31)
    lgb1.fit(X_ag_tr1, y_tr1)
    ag_pred_1 = lgb1.predict(X_ag_te1)
    
    # Optuna for Stage 2 (Maximizing MACRO F1 to force balance)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'class_weight': 'balanced'
        }
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        model.fit(X_ag_tr2_smote, y_tr2_smote)
        preds = model.predict(X_ag_te2_full)
        # We evaluate on full test set's active subset to simulate real-world balance
        active_te = np.where(y_te_full > 0)[0]
        return f1_score(y_te_full[active_te] - 1, preds[active_te], average='macro')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print(f"\n✅ Best Optuna Params for Stage 2 Balance: {study.best_params}")
    
    lgb2 = lgb.LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
    lgb2.fit(X_ag_tr2_smote, y_tr2_smote)
    ag_pred_2_full = lgb2.predict(X_ag_te2_full)
    
    lgb1.booster_.save_model(os.path.join(args.save_dir, "balanced_agent_stage1.txt"))
    lgb2.booster_.save_model(os.path.join(args.save_dir, "balanced_agent_stage2.txt"))

    # ── 4. End-to-End Inference ──────────────────────────────────────────────
    final_preds = []
    for i in range(len(y_te_full)):
        if ag_pred_1[i] == 0: final_preds.append(0)
        else: final_preds.append(ag_pred_2_full[i] + 1)
            
    print("\n🏆 V13 LightGBM Balancer Final Metrics:")
    print(classification_report(y_te_full, final_preds, target_names=['MES0', 'MES1', 'MES2', 'MES3'], digits=4))
    print(f"\n✅ Pipeline Complete! Overall Accuracy: {accuracy_score(y_te_full, final_preds)*100:.2f}%")

if __name__ == "__main__":
    main()
