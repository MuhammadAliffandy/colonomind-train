import os, joblib, argparse, json
import numpy as np
import lightgbm as lgb
import optuna
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
    ap.add_argument("--lgb_dir", default="../Result/ColonoMind_v13_LGBMBalancer")
    ap.add_argument("--save_dir", default="../Result/ColonoMind_v14_ThresholdOpt")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*70); print("🚀 COLONOMIND v14 — Probability Threshold Optimizer"); print("="*70)
    
    # ── 1. Load Data ────────────────────────────────────────────────────────
    cache = os.path.join(args.v6_checkpoint_dir, "dataset_cache_v6.npz")
    d = np.load(cache, allow_pickle=True)
    all_imgs, all_feats, all_labels = d['imgs'], d['feats'], list(d['labels'])
    y_enc_full = LabelEncoder().fit_transform(all_labels)
    
    X_tv_i, X_te_i, X_tv_f, X_te_f, y_tv, y_te_full = train_test_split(all_imgs, all_feats, y_enc_full, test_size=0.20, random_state=42, stratify=y_enc_full)
    X_tr_i, X_va_i, X_tr_f, X_va_f, y_tr, y_va = train_test_split(X_tv_i, X_tv_f, y_tv, test_size=0.20, random_state=42, stratify=y_tv)
    
    sc, um = joblib.load(os.path.join(args.cnn_dir,"scaler_v8.pkl")), joblib.load(os.path.join(args.cnn_dir,"umap_v8.pkl"))
    def process_feats(features):
        scaled = sc.transform(features)
        return scaled, um.transform(scaled)
        
    Xva_s, Uva = process_feats(X_va_f); Xte_s, Ute = process_feats(X_te_f)

    # ── 2. Load CNNs & Extract Validation/Test Features ──────────────────────
    print("\n⚡ Extracting Deep Features (Validation & Test Sets)...")
    def extract_deep(prefix, imgs, feats_s, umaps):
        models = [load_model(os.path.join(args.cnn_dir, f"{prefix}_cnn_seed{s}.h5"), compile=False) for s in ENSEMBLE_SEEDS]
        extractors = [Model(inputs=m.input, outputs=m.get_layer('Fusion').output) for m in models]
        deeps, probs = [], []
        for i in range(len(models)):
            deeps.append(extractors[i].predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0))
            probs.append(models[i].predict([imgs/255.0, feats_s, umaps], batch_size=BATCH_SIZE, verbose=0))
        return np.hstack([np.mean(deeps, axis=0), feats_s, umaps, np.mean(probs, axis=0)])

    # Validation Set Extraction
    X_ag_va1 = extract_deep("stage1", X_va_i, Xva_s, Uva)
    X_ag_va2_full = extract_deep("stage2", X_va_i, Xva_s, Uva)
    
    # Test Set Extraction
    X_ag_te1 = extract_deep("stage1", X_te_i, Xte_s, Ute)
    X_ag_te2_full = extract_deep("stage2", X_te_i, Xte_s, Ute)
    
    # ── 3. Load LightGBM Models & Get Probabilities ─────────────────────────
    lgb1 = lgb.Booster(model_file=os.path.join(args.lgb_dir, "balanced_agent_stage1.txt"))
    lgb2 = lgb.Booster(model_file=os.path.join(args.lgb_dir, "balanced_agent_stage2.txt"))
    
    # Stage 1 Probabilities (Probability of being Active)
    prob_va1 = lgb1.predict(X_ag_va1) 
    prob_te1 = lgb1.predict(X_ag_te1)
    
    # Stage 2 Probabilities [MES1, MES2, MES3]
    prob_va2_full = lgb2.predict(X_ag_va2_full)
    prob_te2_full = lgb2.predict(X_ag_te2_full)
    
    # ── 4. Optuna Threshold Optimization on VALIDATION SET ──────────────────
    print("\n🔥 Running Probability Threshold Optimization (Targeting >80% Acc & Balanced F1)...")
    
    def simulate_pipeline(p1, p2, thr_stg1, w_mes1, w_mes2, w_mes3, y_true):
        preds = []
        weights = np.array([w_mes1, w_mes2, w_mes3])
        for i in range(len(y_true)):
            if p1[i] < thr_stg1: # Predict Normal
                preds.append(0)
            else: # Predict Active
                # Multiply raw probabilities by our custom weights before argmax
                weighted_probs = p2[i] * weights
                preds.append(np.argmax(weighted_probs) + 1)
        return np.array(preds)

    def objective(trial):
        thr_stg1 = trial.suggest_float('thr_stg1', 0.3, 0.7)
        w_mes1 = trial.suggest_float('w_mes1', 0.8, 2.5)
        w_mes2 = trial.suggest_float('w_mes2', 0.8, 2.5)
        w_mes3 = trial.suggest_float('w_mes3', 0.5, 1.5)
        
        preds_va = simulate_pipeline(prob_va1, prob_va2_full, thr_stg1, w_mes1, w_mes2, w_mes3, y_va)
        acc = accuracy_score(y_va, preds_va)
        f1s = f1_score(y_va, preds_va, average=None)
        
        # Objective: Maximize Accuracy + Heavily reward MES1 and MES2 balance
        return acc + (0.5 * min(f1s[1], f1s[2]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=500) # 500 trials takes less than 10 seconds!
    best = study.best_params
    print(f"\n✅ Best Thresholds Found on Validation Set: {best}")
    
    # ── 5. Apply to TEST SET ────────────────────────────────────────────────
    final_preds_te = simulate_pipeline(prob_te1, prob_te2_full, best['thr_stg1'], best['w_mes1'], best['w_mes2'], best['w_mes3'], y_te_full)
    
    print("\n🏆 V14 Threshold Optimizer Final Metrics (Test Set):")
    print(classification_report(y_te_full, final_preds_te, target_names=['MES0', 'MES1', 'MES2', 'MES3'], digits=4))
    print(f"\n✅ Pipeline Complete! Overall Accuracy: {accuracy_score(y_te_full, final_preds_te)*100:.2f}%")
    
    # Save best thresholds
    with open(os.path.join(args.save_dir, "best_thresholds.json"), "w") as f:
        json.dump(best, f, indent=4)

if __name__ == "__main__":
    main()
