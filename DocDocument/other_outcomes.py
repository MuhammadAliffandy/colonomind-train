# ============================================================
# EVALUATION CELL — add to end of each backbone notebook
# All outcomes reported with 95% CI (method chosen per metric).
#
# CI method chosen per metric:
#   - Wilson CI   → binomial proportions:
#                   accuracy, per-class precision/recall/NPV, >=2-grade error
#   - Bootstrap CI → non-proportional / composite metrics:
#                    QWK, macro AUROC, ECE, per-class F1
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, f1_score
)
from statsmodels.stats.proportion import proportion_confint

# ------------------------------------------------------------
# Utility: Wilson CI for a count/total proportion
# ------------------------------------------------------------
def wilson_ci(count, total, alpha=0.05):
    """Return (point_estimate, lo, hi). Handles total=0 gracefully."""
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = proportion_confint(count, total, alpha=alpha, method="wilson")
    return count / total, lo, hi

# ------------------------------------------------------------
# Primary — accuracy (Wilson CI)
# ------------------------------------------------------------
def accuracy_with_wilson_ci(y_true, y_pred):
    n_correct = int(np.sum(np.asarray(y_true) == np.asarray(y_pred)))
    return wilson_ci(n_correct, len(y_true))

# ------------------------------------------------------------
# Primary — QWK (Bootstrap CI)
# ------------------------------------------------------------
def quadratic_weighted_kappa(y_true, y_pred, labels=(0, 1, 2, 3)):
    return cohen_kappa_score(y_true, y_pred, labels=list(labels), weights="quadratic")

# ------------------------------------------------------------
# Per-class metrics with CIs:
#   precision / recall / NPV → Wilson (they are proportions)
#   F1                       → Bootstrap (harmonic mean, no closed form)
# ------------------------------------------------------------
def per_class_metrics_with_ci(y_true, y_pred, labels=(0, 1, 2, 3),
                              n_boot=1000, seed=42):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    total = cm.sum()

    # -- Wilson CIs on precision / recall / NPV from confusion matrix counts --
    rows = []
    for i, c in enumerate(labels):
        TP = int(cm[i, i])
        FN = int(cm[i, :].sum() - TP)
        FP = int(cm[:, i].sum() - TP)
        TN = int(total - TP - FN - FP)

        prec, prec_lo, prec_hi = wilson_ci(TP, TP + FP)   # of predicted positives, how many correct
        rec,  rec_lo,  rec_hi  = wilson_ci(TP, TP + FN)   # of true positives, how many caught
        npv,  npv_lo,  npv_hi  = wilson_ci(TN, TN + FN)   # of predicted negatives, how many correct

        rows.append({
            "MES": int(c),
            "support": int((y_true == c).sum()),
            "precision": prec, "precision_lo": prec_lo, "precision_hi": prec_hi,
            "recall":    rec,  "recall_lo":    rec_lo,  "recall_hi":    rec_hi,
            "npv":       npv,  "npv_lo":       npv_lo,  "npv_hi":       npv_hi,
        })

    # -- Bootstrap CI for per-class F1 --
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1_boot = {c: [] for c in labels}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            f1_vals = f1_score(y_true[idx], y_pred[idx],
                               labels=list(labels), average=None, zero_division=0)
            for c, v in zip(labels, f1_vals):
                f1_boot[c].append(v)
        except Exception:
            continue

    # Point-estimate F1 (not from bootstrap)
    f1_point = f1_score(y_true, y_pred, labels=list(labels),
                        average=None, zero_division=0)
    for i, c in enumerate(labels):
        boots = np.asarray(f1_boot[c])
        f1_lo = float(np.percentile(boots, 2.5)) if len(boots) >= 10 else float("nan")
        f1_hi = float(np.percentile(boots, 97.5)) if len(boots) >= 10 else float("nan")
        rows[i]["f1"] = float(f1_point[i])
        rows[i]["f1_lo"] = f1_lo
        rows[i]["f1_hi"] = f1_hi

    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Macro AUROC (Bootstrap CI)
# ------------------------------------------------------------
def macro_auroc_ovr(y_true, y_probs, labels=(0, 1, 2, 3)):
    return roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro",
                         labels=list(labels))

# ------------------------------------------------------------
# ECE + reliability bins (Bootstrap CI for point ECE)
# ------------------------------------------------------------
def expected_calibration_error(y_true, y_probs, n_bins=10):
    y_true = np.asarray(y_true)
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    accuracies  = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = ((confidences > lo) & (confidences <= hi)) if i > 0 \
                 else ((confidences >= lo) & (confidences <= hi))
        n_in = int(in_bin.sum())
        if n_in == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "avg_conf": None, "accuracy": None})
            continue
        avg_conf = float(confidences[in_bin].mean())
        acc = float(accuracies[in_bin].mean())
        ece += (n_in / n) * abs(avg_conf - acc)
        bins.append({"lo": float(lo), "hi": float(hi), "n": n_in,
                     "avg_conf": avg_conf, "accuracy": acc})
    return float(ece), bins

# ------------------------------------------------------------
# >=2-grade error rate (Wilson CI — it IS a proportion)
# ------------------------------------------------------------
def ge2_grade_error_with_ci(y_true, y_pred):
    diffs = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    n_severe = int((diffs >= 2).sum())
    return wilson_ci(n_severe, len(y_true))

# ------------------------------------------------------------
# Generic paired bootstrap CI (for QWK, AUROC, ECE)
# ------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, y_probs, metric_fn, n_boot=1000,
                 seed=42, use_probs=False):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_probs = np.asarray(y_probs)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            v = metric_fn(y_true[idx], y_probs[idx]) if use_probs \
                else metric_fn(y_true[idx], y_pred[idx])
            if v is not None and not np.isnan(v):
                vals.append(v)
        except Exception:
            continue
    if len(vals) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

# ------------------------------------------------------------
# Reliability diagram
# ------------------------------------------------------------
def plot_reliability_diagram(bins, ece, model_name, cohort_name, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ys = [b["accuracy"] for b in bins if b["n"] > 0]
    conf = [b["avg_conf"] for b in bins if b["n"] > 0]
    sizes = [b["n"] for b in bins if b["n"] > 0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.scatter(conf, ys, s=[max(20, 3 * s) for s in sizes],
               alpha=0.7, edgecolors="k", label="Bin (size ∝ n)")
    for c, y in zip(conf, ys):
        ax.plot([c, c], [c, y], "gray", alpha=0.3, linewidth=1)
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_title(f"{model_name} — {cohort_name}\nECE = {ece:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------
# One-shot: evaluate + save
# ------------------------------------------------------------
def _fmt(p, lo, hi):
    if np.isnan(p): return "  n/a"
    return f"{p:.3f} ({lo:.3f}-{hi:.3f})"

def evaluate_and_save(y_true, y_pred, y_probs, model_name, cohort_name,
                      save_dir="./Result/Evaluation/", n_boot=1000):
    """Every reported number carries a 95% CI (Wilson for proportions,
    bootstrap for QWK, macro AUROC, ECE, per-class F1)."""
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_probs = np.asarray(y_probs)

    # -------- Primary --------
    acc, acc_lo, acc_hi = accuracy_with_wilson_ci(y_true, y_pred)
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    qwk_lo, qwk_hi = bootstrap_ci(
        y_true, y_pred, y_probs,
        lambda yt, yp: cohen_kappa_score(yt, yp, labels=[0,1,2,3], weights="quadratic"),
        n_boot=n_boot,
    )

    # -------- Secondary --------
    per_cls_df = per_class_metrics_with_ci(y_true, y_pred, n_boot=n_boot)

    auroc = macro_auroc_ovr(y_true, y_probs)
    auroc_lo, auroc_hi = bootstrap_ci(
        y_true, y_pred, y_probs,
        lambda yt, yprob: roc_auc_score(yt, yprob, multi_class="ovr",
                                        average="macro", labels=[0,1,2,3]),
        n_boot=n_boot, use_probs=True,
    )
    ece, bins = expected_calibration_error(y_true, y_probs, n_bins=10)
    ece_lo, ece_hi = bootstrap_ci(
        y_true, y_pred, y_probs,
        lambda yt, yprob: expected_calibration_error(yt, yprob, n_bins=10)[0],
        n_boot=n_boot, use_probs=True,
    )

    # -------- Extras --------
    ge2, ge2_lo, ge2_hi = ge2_grade_error_with_ci(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    # -------- Console --------
    print("=" * 68)
    print(f"  {model_name}  |  cohort: {cohort_name}  |  N = {len(y_true)}")
    print("=" * 68)

    print("\n[PRIMARY OUTCOMES]")
    print(f"  Accuracy (Wilson):   {acc:.4f}  (95% CI {acc_lo:.4f}-{acc_hi:.4f})")
    print(f"  QWK (bootstrap):     {qwk:.4f}  (95% CI {qwk_lo:.4f}-{qwk_hi:.4f})")

    print("\n[SECONDARY OUTCOMES]")
    print(f"  Per-class (Wilson CI on precision/recall/NPV, bootstrap CI on F1):")
    header = f"  {'MES':<4}{'N':>5}  {'Precision':<20}  {'Recall':<20}  {'F1':<20}  {'NPV':<20}"
    print(header); print("  " + "-" * (len(header) - 2))
    for _, r in per_cls_df.iterrows():
        print(f"  {int(r.MES):<4}{int(r.support):>5}  "
              f"{_fmt(r.precision, r.precision_lo, r.precision_hi):<20}  "
              f"{_fmt(r.recall,    r.recall_lo,    r.recall_hi):<20}  "
              f"{_fmt(r.f1,        r.f1_lo,        r.f1_hi):<20}  "
              f"{_fmt(r.npv,       r.npv_lo,       r.npv_hi):<20}")

    print(f"\n  Macro AUROC OvR (bootstrap): {auroc:.4f}  (95% CI {auroc_lo:.4f}-{auroc_hi:.4f})")
    print(f"  ECE 10-bin (bootstrap):      {ece:.4f}  (95% CI {ece_lo:.4f}-{ece_hi:.4f})")

    print(f"\n[EXTRAS]")
    print(f"  >=2-grade error (Wilson): {ge2:.4f}  (95% CI {ge2_lo:.4f}-{ge2_hi:.4f})")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    for row in cm:
        print("    " + "  ".join(f"{v:>4d}" for v in row))
    print()

    # -------- Save --------
    result_dict = {
        "model": model_name, "cohort": cohort_name,
        "n_samples": int(len(y_true)),
        "primary": {
            "accuracy":    {"point": acc,  "ci95": [acc_lo, acc_hi],  "method": "Wilson"},
            "QWK":         {"point": qwk,  "ci95": [qwk_lo, qwk_hi],  "method": "bootstrap"},
        },
        "secondary": {
            "per_class":       per_cls_df.to_dict(orient="records"),
            "macro_AUROC_ovr": {"point": auroc, "ci95": [auroc_lo, auroc_hi], "method": "bootstrap"},
            "ECE":             {"point": ece,   "ci95": [ece_lo, ece_hi],     "method": "bootstrap"},
            "reliability_bins": bins,
        },
        "extras": {
            "ge2_grade_error_rate": {"point": ge2, "ci95": [ge2_lo, ge2_hi], "method": "Wilson"},
            "confusion_matrix": cm.tolist(),
        },
    }
    json_path = os.path.join(save_dir, f"{model_name}_{cohort_name}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"  ✅ JSON:  {json_path}")

    csv_path = os.path.join(save_dir, f"{model_name}_{cohort_name}_perclass.csv")
    per_cls_df.to_csv(csv_path, index=False)
    print(f"  ✅ CSV:   {csv_path}")

    fig_path = os.path.join(save_dir, f"{model_name}_{cohort_name}_reliability.png")
    plot_reliability_diagram(bins, ece, model_name, cohort_name, fig_path)
    print(f"  ✅ Fig:   {fig_path}")

    return result_dict


# ============================================================
# CALL EVALUATION — internal test cohort
# ------------------------------------------------------------
# Inputs already produced by earlier cells:
#   y_test_encoded  = ground truth (N,)
#   y_pred_hybrid   = predicted class (N,)
#   y_pred_proba    = softmax probabilities (N, 4)
# ------------------------------------------------------------
# ⚠️ CHANGE model_name PER NOTEBOOK:
#     "ResNet50" / "DenseNet121" / "EfficientNetB4" /
#     "ConvNeXtTiny" / "ViT_B16"
# ============================================================
results = evaluate_and_save(
    y_true=y_test_encoded,
    y_pred=y_pred_hybrid,
    y_probs=y_pred_proba,
    model_name="ResNet50",           # ← change per notebook
    cohort_name="internal_test",     # ← later: "LIMUC" / "TMC_UCM"
    save_dir="./Result/Evaluation/",
    n_boot=1000,
)
