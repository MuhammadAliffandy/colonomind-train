import os
import argparse
import json
import shutil
import joblib
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
import lightgbm as lgb

def build_arg_parser(model_choices):
    parser = argparse.ArgumentParser(description="ColonoMind DGX Training Script")
    parser.add_argument("--scenario", type=str, required=True, choices=['Intra', 'Multi'])
    parser.add_argument("--train_dataset", type=str, required=True, choices=['NTUH', 'LIMUC', 'TMC-UCM'])
    parser.add_argument("--test_dataset", type=str, required=True, choices=['NTUH', 'LIMUC', 'TMC-UCM'])
    parser.add_argument("--model", type=str, required=True, choices=list(model_choices))
    parser.add_argument("--base_dir", type=str, default="..", help="Base directory where Dataset and Dataset+Code folders are located")
    parser.add_argument('--threshold', type=float, default=None,
                        help="Fixed routing threshold. If omitted, tuned on val_cal.")
    parser.add_argument('--tune_threshold', action='store_true', default=True,
                        help="Select routing threshold on val_cal (default).")
    parser.add_argument('--no_tune_threshold', dest='tune_threshold', action='store_false',
                        help="Disable tuning; requires --threshold.")
    parser.add_argument('--val_es_frac', type=float, default=0.15)
    parser.add_argument('--val_cal_frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def validate_args(args, parser):
    if not args.tune_threshold and args.threshold is None:
        parser.error("--no_tune_threshold requires --threshold.")
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be in [0, 1].")
    if args.val_es_frac <= 0 or args.val_cal_frac <= 0:
        parser.error("--val_es_frac and --val_cal_frac must both be > 0.")
    if args.val_es_frac + args.val_cal_frac >= 1.0:
        parser.error("--val_es_frac + --val_cal_frac must be < 1.0.")


def make_features(proba, umap_feat, h_feat):
    df = pd.DataFrame(h_feat, columns=[f"f{i}" for i in range(20)])
    df["confidence"] = np.max(proba, axis=1)
    df["umap_0"] = umap_feat[:, 0]
    df["umap_1"] = umap_feat[:, 1]
    return df


def _bootstrap_qwk_se(y_true, y_pred, seed, n_bootstrap=1000):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score = cohen_kappa_score(y_true[idx], y_pred[idx], weights='quadratic')
            if np.isfinite(score):
                scores.append(float(score))
        except Exception:
            continue
    if not scores:
        return 0.0
    se = float(np.std(scores, ddof=1))
    if not np.isfinite(se):
        return 0.0
    return se


def oof_agent_predictions(X, y, seed, n_splits=5):
    """Stratified k-fold cross-fit LightGBM. Returns OOF predicted labels and used splits."""
    X = np.asarray(X)
    y = np.asarray(y)
    _, class_counts = np.unique(y, return_counts=True)
    min_class_count = int(np.min(class_counts))
    used_splits = max(2, min(n_splits, min_class_count))

    skf = StratifiedKFold(n_splits=used_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in skf.split(X, y):
        lgbm_kwargs = {'random_state': seed, 'class_weight': 'balanced'}
        if len(tr_idx) < 150:
            lgbm_kwargs['min_child_samples'] = 5
        clf_fold = lgb.LGBMClassifier(**lgbm_kwargs)
        clf_fold.fit(X[tr_idx], y[tr_idx])
        oof_pred[va_idx] = clf_fold.predict(X[va_idx])

    return oof_pred, used_splits


def select_threshold(y_true, y_pred_deep, conf_deep, y_pred_agent_oof, grid, seed):
    """Sweep grid, score by QWK, apply 1-SE least-delegation tie-break.
    Returns (chosen_threshold, sweep_table, selection_metadata)."""
    y_true = np.asarray(y_true)
    y_pred_deep = np.asarray(y_pred_deep)
    conf_deep = np.asarray(conf_deep)
    y_pred_agent_oof = np.asarray(y_pred_agent_oof)
    grid = np.asarray(grid)

    deep_only_qwk = cohen_kappa_score(y_true, y_pred_deep, weights='quadratic')
    sweep = []
    for thr in grid:
        delegated = conf_deep < thr
        y_pred_hybrid = np.where(delegated, y_pred_agent_oof, y_pred_deep)
        sweep.append({
            'threshold': float(np.round(thr, 3)),
            'qwk': float(cohen_kappa_score(y_true, y_pred_hybrid, weights='quadratic')),
            'accuracy': float(accuracy_score(y_true, y_pred_hybrid)),
            'macro_f1': float(f1_score(y_true, y_pred_hybrid, average='macro', zero_division=0)),
            'delegation_rate': float(np.mean(delegated))
        })

    best_qwk = max(row['qwk'] for row in sweep)
    best_rows = [row for row in sweep if np.isclose(row['qwk'], best_qwk)]
    best_threshold_for_se = min(row['threshold'] for row in best_rows)
    best_pred = np.where(conf_deep < best_threshold_for_se, y_pred_agent_oof, y_pred_deep)
    qwk_se = _bootstrap_qwk_se(y_true, best_pred, seed=seed, n_bootstrap=1000)

    one_se_cutoff = best_qwk - qwk_se
    within_one_se = [row for row in sweep if row['qwk'] >= one_se_cutoff]

    if best_qwk <= deep_only_qwk:
        chosen_threshold = 0.0
        threshold_source = 'degenerate_no_benefit'
    else:
        chosen_threshold = float(min(row['threshold'] for row in within_one_se))
        threshold_source = 'tuned_val_cal'

    chosen_delegation = float(np.mean(conf_deep < chosen_threshold))
    metadata = {
        'threshold_source': threshold_source,
        'selection_metric': 'quadratic_weighted_kappa',
        'best_qwk_val_cal': float(best_qwk),
        'qwk_se_bootstrap': float(qwk_se),
        'n_within_1se': int(len(within_one_se)),
        'delegation_rate_val_cal': chosen_delegation,
        'deep_only_qwk_val_cal': float(deep_only_qwk)
    }

    return float(chosen_threshold), sweep, metadata


def _subset_by_index(arr, idx):
    if isinstance(arr, np.ndarray):
        return arr[idx]
    return [arr[i] for i in idx]


def _plot_threshold_sweep(sweep_rows, chosen_threshold, qwk_se, out_path, title, annotate_posthoc=False):
    df_sweep = pd.DataFrame(sweep_rows)
    if df_sweep.empty:
        return

    best_qwk = float(df_sweep['qwk'].max())
    cutoff = best_qwk - float(qwk_se)
    x = df_sweep['threshold'].values

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(x, df_sweep['qwk'].values, marker='o', color='tab:blue', label='QWK')
    ax1.fill_between(x, cutoff, best_qwk, color='tab:blue', alpha=0.12, label='1-SE band')
    ax1.axvline(chosen_threshold, color='tab:green', linestyle='--', linewidth=1.5, label=f'Chosen={chosen_threshold:.3f}')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('QWK', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(x, df_sweep['delegation_rate'].values, marker='s', color='tab:orange', label='Delegation Rate')
    ax2.set_ylabel('Delegation Rate', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    if annotate_posthoc:
        ax1.set_title('POST-HOC SENSITIVITY - NOT USED FOR SELECTION')
    else:
        ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    from dgx_dataloader import load_all_images, load_tmc_ucm
    from dgx_models import build_hybrid_model, MODEL_BUILDERS, focal_loss
    import umap
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    parser = build_arg_parser(MODEL_BUILDERS.keys())
    args = parser.parse_args()
    validate_args(args, parser)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(f"\\n{'='*50}")
    print(f"🚀 Starting Training on DGX")
    print(f"Scenario: {args.scenario}")
    print(f"Train Dataset: {args.train_dataset}")
    print(f"Test Dataset: {args.test_dataset}")
    print(f"Model: {args.model}")
    print(f"Base Dir: {args.base_dir}")
    print(f"Seed: {args.seed}")
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
                all_imgs, all_feats, all_labels, all_paths, test_size=0.2, random_state=args.seed, stratify=all_labels
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

    # Three-way split on train pool: train / val_es / val_cal
    print("Splitting Train Pool into Train/Val_ES/Val_Cal...")
    idx_all = np.arange(len(y_train_label_raw))
    holdout_frac = args.val_es_frac + args.val_cal_frac
    idx_train, idx_holdout = train_test_split(
        idx_all,
        test_size=holdout_frac,
        random_state=args.seed,
        stratify=np.array(y_train_label_raw)
    )
    holdout_labels = np.array(y_train_label_raw)[idx_holdout]
    val_cal_ratio = args.val_cal_frac / holdout_frac
    idx_val_es, idx_val_cal = train_test_split(
        idx_holdout,
        test_size=val_cal_ratio,
        random_state=args.seed,
        stratify=holdout_labels
    )

    X_train_img = _subset_by_index(X_train_img_raw, idx_train)
    X_train_feat = _subset_by_index(X_train_feat_raw, idx_train)
    y_train_label = _subset_by_index(y_train_label_raw, idx_train)

    X_val_es_img = _subset_by_index(X_train_img_raw, idx_val_es)
    X_val_es_feat = _subset_by_index(X_train_feat_raw, idx_val_es)
    y_val_es_label = _subset_by_index(y_train_label_raw, idx_val_es)

    X_val_cal_img = _subset_by_index(X_train_img_raw, idx_val_cal)
    X_val_cal_feat = _subset_by_index(X_train_feat_raw, idx_val_cal)
    y_val_cal_label = _subset_by_index(y_train_label_raw, idx_val_cal)

    print(f"Training samples: {len(X_train_img)}")
    print(f"Validation (ES) samples: {len(X_val_es_img)}")
    print(f"Validation (Cal) samples: {len(X_val_cal_img)}")
    print(f"Testing samples (Untouched): {len(X_test_img)}")

    # Images kept at raw 0-255 scale (preprocessing handled in dgx_models.py branch definition)
    X_img_train = np.array(X_train_img, dtype=np.float32)
    X_img_val_es = np.array(X_val_es_img, dtype=np.float32)
    X_img_val_cal = np.array(X_val_cal_img, dtype=np.float32)
    X_img_test  = np.array(X_test_img, dtype=np.float32)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_label)
    y_val_es_encoded = le.transform(y_val_es_label)
    y_val_cal_encoded = le.transform(y_val_cal_label)
    y_test_encoded  = le.transform(y_test_label)
    
    y_train_cat = to_categorical(y_train_encoded, num_classes=len(le.classes_))
    y_val_es_cat = to_categorical(y_val_es_encoded, num_classes=len(le.classes_))

    # Scale Handcrafted Features
    scaler = StandardScaler()
    X_feat_train_scaled = scaler.fit_transform(np.array(X_train_feat))
    X_feat_val_es_scaled = scaler.transform(np.array(X_val_es_feat))
    X_feat_val_cal_scaled = scaler.transform(np.array(X_val_cal_feat))
    X_feat_test_scaled  = scaler.transform(np.array(X_test_feat))

    # SMOTE is removed as per reviewer feedback. Class imbalance handled via class_weights/loss

    # UMAP
    print("Fitting UMAP on Train...")
    umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=args.seed)
    X_train_umap = umap_reducer.fit_transform(X_feat_train_scaled)
    X_val_es_umap = umap_reducer.transform(X_feat_val_es_scaled)
    X_val_cal_umap = umap_reducer.transform(X_feat_val_cal_scaled)
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
        validation_data=([X_img_val_es, X_feat_val_es_scaled, X_val_es_umap], y_val_es_cat),
        batch_size=16, epochs=100, class_weight=class_weight_dict, callbacks=callbacks, verbose=1
    )

    model_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_hybrid.h5")
    model.save(model_path)
    print(f"✅ Saved base model to {model_path}")

    # Agent Calibration on val_cal with OOF threshold tuning
    print(f"\\n[2] Calibrating Super Agent on val_cal")
    proba_val_cal = model.predict([X_img_val_cal, X_feat_val_cal_scaled, X_val_cal_umap], verbose=0)
    y_pred_deep_val_cal = np.argmax(proba_val_cal, axis=1)
    conf_val_cal = np.max(proba_val_cal, axis=1)

    df_val_cal_ag = make_features(proba_val_cal, X_val_cal_umap, X_feat_val_cal_scaled)

    features = ["confidence", "umap_0", "umap_1"] + [f"f{i}" for i in range(20)]
    scaler_ag = StandardScaler()
    X_cal = scaler_ag.fit_transform(df_val_cal_ag[features].values)
    y_cal = y_val_cal_encoded

    y_pred_agent_oof, n_splits_used = oof_agent_predictions(X_cal, y_cal, seed=args.seed, n_splits=5)

    grid = np.round(np.arange(0.30, 0.951, 0.025), 3)
    tuned_threshold, val_sweep_rows, selection_metadata = select_threshold(
        y_true=y_cal,
        y_pred_deep=y_pred_deep_val_cal,
        conf_deep=conf_val_cal,
        y_pred_agent_oof=y_pred_agent_oof,
        grid=grid,
        seed=args.seed
    )

    if args.threshold is not None:
        chosen_threshold = float(args.threshold)
        selection_metadata['threshold_source'] = 'cli_override' if args.tune_threshold else 'cli_fixed'
    else:
        chosen_threshold = float(tuned_threshold)

    low_confidence_selection = len(y_cal) < 150
    if low_confidence_selection:
        print("⚠️  WARNING: val_cal has fewer than 150 samples; threshold selection may be unstable.")

    lgbm_kwargs_final = {'random_state': args.seed, 'class_weight': 'balanced'}
    if len(y_cal) < 150:
        lgbm_kwargs_final['min_child_samples'] = 5

    clf_final = lgb.LGBMClassifier(**lgbm_kwargs_final)
    clf_final.fit(X_cal, y_cal)
    
    # Save Super Agent
    agent_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_agent.txt")
    scaler_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_scaler.pkl")
    
    tmp_agent_path = f"/tmp/{args.model}_agent_{np.random.randint(1000)}.txt"
    clf_final.booster_.save_model(tmp_agent_path)
    shutil.copy(tmp_agent_path, agent_path)
    joblib.dump(scaler_ag, scaler_path)

    chosen_row = next((r for r in val_sweep_rows if np.isclose(r['threshold'], chosen_threshold)), None)
    if chosen_row is None:
        val_delegate_rate = float(np.mean(conf_val_cal < chosen_threshold))
    else:
        val_delegate_rate = float(chosen_row['delegation_rate'])

    git_commit = 'unknown'
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        pass

    threshold_selection_payload = {
        'chosen_threshold': float(chosen_threshold),
        'threshold_source': selection_metadata.get('threshold_source', 'tuned_val_cal'),
        'selection_metric': selection_metadata.get('selection_metric', 'quadratic_weighted_kappa'),
        'n_val_cal': int(len(y_cal)),
        'n_splits_oof': int(n_splits_used),
        'best_qwk_val_cal': float(selection_metadata.get('best_qwk_val_cal', 0.0)),
        'qwk_se_bootstrap': float(selection_metadata.get('qwk_se_bootstrap', 0.0)),
        'n_within_1se': int(selection_metadata.get('n_within_1se', 0)),
        'delegation_rate_val_cal': float(val_delegate_rate),
        'deep_only_qwk_val_cal': float(selection_metadata.get('deep_only_qwk_val_cal', 0.0)),
        'low_confidence_selection': bool(low_confidence_selection),
        'sweep': val_sweep_rows,
        'seed': int(args.seed),
        'git_commit': git_commit
    }

    threshold_json_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_threshold_selection.json")
    with open(threshold_json_path, 'w') as f:
        json.dump(threshold_selection_payload, f, indent=4)

    threshold_val_plot_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_threshold_sweep_val.png")
    _plot_threshold_sweep(
        val_sweep_rows,
        chosen_threshold=chosen_threshold,
        qwk_se=selection_metadata.get('qwk_se_bootstrap', 0.0),
        out_path=threshold_val_plot_path,
        title='Validation Threshold Sweep (val_cal)',
        annotate_posthoc=False
    )

    # 3. FINAL EVALUATION ON UNTOUCHED TEST SET
    print(f"\\n[3] Final Evaluation on Test Set")
    proba_test = model.predict([X_img_test, X_feat_test_scaled, X_test_umap], verbose=0)
    df_test_ag = make_features(proba_test, X_test_umap, X_feat_test_scaled)
    X_test_ag = scaler_ag.transform(df_test_ag[features].values)

    y_true = y_test_encoded

    y_pred_deep = np.argmax(proba_test, axis=1)
    base_acc = accuracy_score(y_true, y_pred_deep)

    conf_test = np.max(proba_test, axis=1)
    y_pred_agent = clf_final.predict(X_test_ag)
    low_conf_mask = conf_test < chosen_threshold

    y_pred_hybrid = np.where(low_conf_mask, y_pred_agent, y_pred_deep)
    hybrid_acc = accuracy_score(y_true, y_pred_hybrid)
    
    print(f"  📊 BASE DEEP LEARNING ACCURACY: {base_acc:.4f}  ({base_acc*100:.2f}%)")
    print(f"  ⚙️  HYBRID SELECTOR (Threshold = {chosen_threshold:.3f})")
    print(f"  🔍 Delegated {low_conf_mask.sum()} / {len(low_conf_mask)} low-confidence cases to Agent")
    print(f"  🔍 Delegation Rate (Test): {float(np.mean(low_conf_mask)):.3f}")
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
        'QWK': float(kappa),
        'Chosen_Threshold': float(chosen_threshold),
        'Threshold_Source': threshold_selection_payload['threshold_source'],
        'Delegation_Rate_Test': float(np.mean(low_conf_mask)),
        'Agent_Fit_Set': 'val_cal',
        'N_Train': int(len(X_train_img)),
        'N_Val_ES': int(len(X_val_es_img)),
        'N_Val_Cal': int(len(X_val_cal_img)),
        'N_Test': int(len(X_test_img)),
        'Seed': int(args.seed),
        'Git_Commit': git_commit
    }

    test_sweep_rows = []
    for thr in grid:
        delegated = conf_test < thr
        y_h = np.where(delegated, y_pred_agent, y_pred_deep)
        test_sweep_rows.append({
            'threshold': float(np.round(thr, 3)),
            'qwk': float(cohen_kappa_score(y_true, y_h, weights='quadratic')),
            'accuracy': float(accuracy_score(y_true, y_h)),
            'macro_f1': float(f1_score(y_true, y_h, average='macro', zero_division=0)),
            'delegation_rate': float(np.mean(delegated))
        })

    threshold_test_plot_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_threshold_sensitivity_test.png")
    _plot_threshold_sweep(
        test_sweep_rows,
        chosen_threshold=chosen_threshold,
        qwk_se=selection_metadata.get('qwk_se_bootstrap', 0.0),
        out_path=threshold_test_plot_path,
        title='POST-HOC SENSITIVITY - NOT USED FOR SELECTION',
        annotate_posthoc=True
    )

    metrics_path = os.path.join(BASE_SAVE_DIR, f"{args.model}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Evaluation Complete. Test Accuracy: {acc:.4f}, Test QWK: {kappa:.4f}")
    print(f"📁 Results saved to {BASE_SAVE_DIR}")

if __name__ == "__main__":
    main()
