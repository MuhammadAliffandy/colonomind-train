# Implementation Spec — Validation-Based Threshold Tuning and Agent Calibration

**Target file:** `src/train_dgx.py` (only)
**Repo:** `MuhammadAliffandy/colonomind-train`, baseline commit `3b788a0`
**Scope:** replace the fixed routing threshold with a value selected on held-out data, and move the Super Agent fit off in-sample CNN predictions.

---

## 1. Problem being solved

Two defects in the current agent stage, both independent of the already-fixed feedback loop.

**Defect A — the agent is fit on in-sample confidences.**
`train_dgx.py:172` computes `y_pred_proba_train` by predicting on `X_img_train`, the exact images the CNN was just trained on. Those confidences are saturated near 1.0. The agent (line 194) therefore learns a decision boundary in a confidence region that barely exists at inference time. The router at line 217 then selects on that same feature, whose distribution shifts between fit and deployment.

**Defect B — the routing threshold is a fixed constant.**
`args.threshold` defaults to `0.70` (line 31) and is never tuned. The CNN trains with `focal_loss(gamma=2.5, alpha=0.25)` (line 152), which deliberately reshapes the confidence distribution, and the five backbones (ResNet-50, DenseNet-121, EfficientNet-B4, ConvNeXt-Tiny, ViT-B-16) each land in different calibration regimes. A single constant means the delegation rate is an uncontrolled variable across the comparison table.

Tuning the threshold on the test set is not an acceptable fix — that reintroduces leakage in a subtler form. Everything below is selected on held-out data only.

---

## 2. Design

### 2.1 Three-way split of the training pool

The current split (line 94) produces `train` / `val`, where `val` drives `EarlyStopping(restore_best_weights=True)`. Reusing that same `val` to fit the agent *and* select the threshold would be triple-dipping on one split.

Split the training pool three ways instead:

| Split | Fraction of pool | Used for |
|---|---|---|
| `train` | 70% | CNN weights, UMAP fit, scaler fit |
| `val_es` | 15% | early stopping / LR schedule only |
| `val_cal` | 15% | agent fit + threshold selection only |

`test` remains completely untouched until the single final evaluation. One CNN training run — no extra compute.

### 2.2 Cross-fitting inside `val_cal`

If the agent is fit on all of `val_cal` and the threshold is then swept on `val_cal`, the agent's predictions there are in-sample and optimistic, biasing the sweep toward over-delegation. Avoid this with 5-fold cross-fitting:

1. Split `val_cal` into 5 stratified folds.
2. For each fold *k*: fit LightGBM on the other 4 folds, predict fold *k*. Collect out-of-fold predictions covering all of `val_cal`.
3. Sweep the threshold against those OOF predictions.
4. After the threshold is frozen, refit LightGBM once on **all** of `val_cal` — that is the deployed agent.

LightGBM is fast; this adds seconds, not minutes.

### 2.3 Selection rule

- **Grid:** `np.round(np.arange(0.30, 0.951, 0.025), 3)` (26 points).
- **Objective:** quadratic-weighted kappa (`cohen_kappa_score(..., weights='quadratic')`). MES grading is ordinal; QWK is the clinically meaningful metric. Log accuracy and macro-F1 alongside, but select on QWK.
- **Tie-breaking (mandatory, do not skip):** compute the standard error of the best QWK via 1000-iteration bootstrap over `val_cal`. Among all thresholds whose QWK is within 1 SE of the maximum, select the **lowest** threshold — i.e. the least delegation. This prevents the agent absorbing cases it does not actually improve, and makes the choice reproducible rather than a knife-edge argmax.
- **Degenerate case:** if no threshold beats the deep model alone on `val_cal` QWK, select `threshold = 0.0` (route nothing to the agent) and set `"threshold_source": "degenerate_no_benefit"` in the output. Report that honestly; it is a legitimate finding, not a failure.
- **Small-sample guard:** if `len(val_cal) < 150`, emit a loud warning that the selection is unstable and record `"low_confidence_selection": true` in the JSON.

---

## 3. Changes to `src/train_dgx.py`

### 3.1 CLI (near line 31)

```python
parser.add_argument('--threshold', type=float, default=None,
                    help="Fixed routing threshold. If omitted, tuned on val_cal.")
parser.add_argument('--tune_threshold', action='store_true', default=True,
                    help="Select routing threshold on val_cal (default).")
parser.add_argument('--no_tune_threshold', dest='tune_threshold', action='store_false',
                    help="Disable tuning; requires --threshold.")
parser.add_argument('--val_es_frac', type=float, default=0.15)
parser.add_argument('--val_cal_frac', type=float, default=0.15)
parser.add_argument('--seed', type=int, default=42)
```

Validation at startup: if `--no_tune_threshold` is passed without `--threshold`, exit with a clear error. If both `--threshold` and tuning are active, `--threshold` wins and `"threshold_source"` is recorded as `"cli_override"`.

### 3.2 Replace the split (line 94)

Two sequential stratified splits off `*_raw`. Preserve `random_state=args.seed` and `stratify=` on both. Keep the existing variable names `X_train_img`, `X_train_feat`, `y_train_label` for the 70% portion so downstream code needs no renaming; add `*_val_es_*` and `*_val_cal_*` alongside. Print all four set sizes.

Do **not** change how `X_train_img_raw` / `X_test_img` are loaded (lines 68–90). The Intra-LIMUC and Intra-TMC-UCM branches use the datasets' own predefined splits; that behaviour stays.

### 3.3 Fit transforms on `train` only (lines 107–126)

`LabelEncoder`, `StandardScaler` and UMAP must remain `fit`/`fit_transform` on the 70% `train` portion, then `transform` for `val_es`, `val_cal`, and `test`. Add the two new sets to each `transform` call. **No transform may see `val_cal` or `test` during fitting.**

### 3.4 Model fit (line 161)

`validation_data` uses `val_es` only. Callbacks unchanged.

### 3.5 Agent stage (replaces lines 171–200)

```python
# Deep-model predictions on every split
proba_val_cal = model.predict([X_img_val_cal, X_feat_val_cal_scaled, X_val_cal_umap], verbose=0)
proba_test    = model.predict([X_img_test,    X_feat_test_scaled,    X_test_umap],    verbose=0)

# Agent features (unchanged layout: confidence + umap_0 + umap_1 + f0..f19)
df_val_cal_ag = make_features(proba_val_cal, X_val_cal_umap, X_feat_val_cal_scaled)
df_test_ag    = make_features(proba_test,    X_test_umap,    X_feat_test_scaled)

# Scaler for the agent: fit on val_cal (the agent's training set), not train
scaler_ag = StandardScaler()
X_cal = scaler_ag.fit_transform(df_val_cal_ag[features].values)
y_cal = y_val_cal_encoded
```

Delete `y_pred_proba_train` / `y_pred_hybrid_train` and every use of them. The agent must no longer touch train-set predictions.

Keep `LGBMClassifier(random_state=args.seed, class_weight='balanced')` for continuity with prior results; if `val_cal` is small, also pass `min_child_samples=5`.

### 3.6 New helper functions

Add two module-level functions above `main()`:

```python
def oof_agent_predictions(X, y, seed, n_splits=5):
    """Stratified k-fold cross-fit LightGBM. Returns OOF predicted labels for all of X."""

def select_threshold(y_true, y_pred_deep, conf_deep, y_pred_agent_oof, grid, seed):
    """Sweep grid, score by QWK, apply 1-SE least-delegation tie-break.
    Returns (chosen_threshold, sweep_table, selection_metadata)."""
```

`select_threshold` must be pure — no file I/O, no globals — so it can be unit-tested.

`n_splits` must degrade gracefully: if the rarest class in `val_cal` has fewer than 5 members, reduce `n_splits` to that count (minimum 2) and record the value used.

### 3.7 Final evaluation (lines 213–232)

Apply the frozen threshold to test **once**:

```python
conf_test     = np.max(proba_test, axis=1)
low_conf_mask = conf_test < chosen_threshold
y_pred_agent  = clf_final.predict(scaler_ag.transform(df_test_ag[features].values))
y_pred_hybrid = np.where(low_conf_mask, y_pred_agent, y_pred_deep)
```

No sweeping, no re-selection, no conditional logic on test metrics anywhere in this block.

---

## 4. New outputs

### 4.1 `{model}_threshold_selection.json`

```json
{
  "chosen_threshold": 0.575,
  "threshold_source": "tuned_val_cal",
  "selection_metric": "quadratic_weighted_kappa",
  "n_val_cal": 412,
  "n_splits_oof": 5,
  "best_qwk_val_cal": 0.7412,
  "qwk_se_bootstrap": 0.0231,
  "n_within_1se": 6,
  "delegation_rate_val_cal": 0.183,
  "deep_only_qwk_val_cal": 0.7218,
  "low_confidence_selection": false,
  "sweep": [
    {"threshold": 0.300, "qwk": 0.7218, "accuracy": 0.8107, "macro_f1": 0.6944, "delegation_rate": 0.000}
  ],
  "seed": 42,
  "git_commit": "3b788a0"
}
```

The full sweep array goes in the file — that table is what answers "why this value" in review.

### 4.2 `{model}_threshold_sweep_val.png`

QWK vs threshold on `val_cal`, with the chosen point marked, the 1-SE band shaded, and delegation rate on a secondary y-axis. Title must say **validation**.

### 4.3 `{model}_threshold_sensitivity_test.png` (post-hoc)

The same sweep computed on test, for the sensitivity ablation. Title must read `POST-HOC SENSITIVITY — NOT USED FOR SELECTION`, and the figure must be written *after* the frozen-threshold evaluation. This exists so reviewers can see the curve was not cherry-picked; it must never feed back into any choice.

### 4.4 Extend `{model}_metrics.json`

Add: `Chosen_Threshold`, `Threshold_Source`, `Delegation_Rate_Test`, `Agent_Fit_Set` (`"val_cal"`), `N_Train`, `N_Val_ES`, `N_Val_Cal`, `N_Test`, `Seed`, `Git_Commit`. Keep all existing keys and their names — downstream table scripts depend on them.

---

## 5. Constraints

**Must not:**
- reference `y_test_encoded`, `y_test_cat`, `y_test_label` or any test array anywhere before the final evaluation block
- fit the agent, scaler, UMAP, or LightGBM on test data
- select or adjust the threshold using any test metric
- reuse `val_es` for agent fitting or threshold selection
- introduce any loop that retrains based on an accuracy target
- import from or reference `scratch_code.py`
- modify `src/train.py`, `src/train_all.py`, `src/finetune_agent.py`, or `generate_unified_notebook.py` (separate tasks)
- change existing keys in `metrics.json`

**Must:**
- seed everything from `args.seed`
- keep the run resumable — `run_intra_experiments_dgx.sh` skips on the presence of `{model}_metrics.json`, so that file stays the last artifact written
- print the chosen threshold and delegation rate to stdout so they appear in DGX logs

---

## 6. Acceptance checks

The agent should add `tests/test_threshold_selection.py` and confirm all of the following pass.

1. **Split disjointness** — carry index arrays through both splits; assert the three index sets are pairwise disjoint and their union is the full pool.
2. **No test leakage, static** — `grep -n "test" src/train_dgx.py`, confirm every hit above the final evaluation block is a path string, a CLI arg, or a comment. Paste the result in the PR.
3. **Determinism** — two runs with `--seed 42` on the same data produce an identical `chosen_threshold`.
4. **Selection purity** — call `select_threshold` on synthetic data where the optimal threshold is known by construction; assert it is recovered.
5. **Tie-break correctness** — synthetic case with a flat QWK plateau; assert the *lowest* threshold in the plateau is chosen.
6. **Degenerate path** — synthetic case where the agent is worse everywhere; assert `chosen_threshold == 0.0` and `threshold_source == "degenerate_no_benefit"`.
7. **CLI override** — `--no_tune_threshold --threshold 0.7` reproduces the current fixed-threshold behaviour exactly; `--no_tune_threshold` without `--threshold` exits non-zero with a clear message.
8. **Smoke run** — one full run on the smallest scenario (`Intra TMC-UCM ResNet-50`); confirm all four artifacts appear and the JSON validates against the schema in §4.1.

---

## 7. Out of scope — flag, do not fix

Note these in the PR description; do not attempt them in this change.

- **Patient-level grouping.** `dgx_dataloader.py` returns only file paths — no patient, case, or video ID. All splits (`train_dgx.py:78` and `:94`) are per-image with `stratify=labels`, so frames from the same colonoscopy can land on both sides. This is a separate class of leakage and needs a data-provenance decision before `StratifiedGroupKFold` can be used.
- **`src/train.py:129`** still sets `val_inputs` to the test set with `restore_best_weights=True`. Separate fix.
- **Stale comments** in `train_all.py:285-288` referencing a feedback loop that no longer exists.

---

## 8. Deliverable

One PR against the baseline commit containing: the modified `src/train_dgx.py`, the new test file, the static-grep output from check 2, and the smoke-run JSON from check 8. No changes to any other pipeline file.
