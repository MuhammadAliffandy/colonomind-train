"""
Reference metrics for ordinal MES (0-3) endoscopic classification.

Computes, REPRODUCIBLY from the model's own outputs:
  - Cohen's kappa (unweighted) and quadratic-weighted kappa  -> overall, on the 4 ordinal classes
  - Per-class one-vs-rest kappa (to show Quad == unweighted for a binary split)
  - Expected Calibration Error (confidence-ECE and class-wise ECE) -> needs predicted PROBABILITIES
  - Per-class confusion-matrix metrics: sensitivity, specificity, PPV, NPV, LR+, LR-
  - Macro-averaged accuracy with a bootstrap CI

Inputs you must supply (NOT derivable from the published table):
  y_true : (N,) int array of ground-truth MES labels in {0,1,2,3}
  y_pred : (N,) int array of predicted MES labels in {0,1,2,3}
  y_prob : (N,4) float array of the model's predicted class probabilities (rows sum to 1)

If you no longer have y_prob, you CANNOT compute ECE -- rerun inference to recover the
probabilities, or report that calibration was not assessed and drop the column.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

Z = 1.959963985  # 95% normal quantile


# ----------------------------------------------------------------------
# 1. KAPPA  (overall, on the full ordinal scale -- this is where QWK belongs)
# ----------------------------------------------------------------------
def kappas_overall(y_true, y_pred):
    k_unw = cohen_kappa_score(y_true, y_pred, weights=None)
    k_quad = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return {"cohen_kappa": k_unw, "quadratic_weighted_kappa": k_quad}


def kappa_bootstrap_ci(y_true, y_pred, weights="quadratic", n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # guard: a resample may contain a single class -> kappa undefined; skip it
        if len(np.unique(y_true[idx])) < 2 and len(np.unique(y_pred[idx])) < 2:
            continue
        stats.append(cohen_kappa_score(y_true[idx], y_pred[idx], weights=weights))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi


def kappa_per_class_onevsrest(y_true, y_pred, n_classes=4):
    """
    Demonstrates the methodological point: for a BINARY (one-vs-rest) split,
    quadratic weighting is meaningless, so quad kappa == unweighted kappa EXACTLY.
    A per-class 'Quad k' that differs from per-class Cohen k (e.g. by a constant)
    cannot be a real computed statistic.
    """
    out = {}
    for c in range(n_classes):
        yt = (np.asarray(y_true) == c).astype(int)
        yp = (np.asarray(y_pred) == c).astype(int)
        out[f"MES{c}"] = {
            "cohen_kappa": cohen_kappa_score(yt, yp, weights=None),
            "quadratic_weighted_kappa": cohen_kappa_score(yt, yp, weights="quadratic"),
        }
    return out


# ----------------------------------------------------------------------
# 2. EXPECTED CALIBRATION ERROR  (needs predicted probabilities)
# ----------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=15, strategy="uniform"):
    """
    Confidence-ECE: bin the top-class confidence, compare bin accuracy vs bin
    mean confidence, average weighted by bin population.
    strategy='uniform' (equal-width bins) or 'quantile' (equal-frequency bins).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # quantile
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0

    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        ece += (m.sum() / n) * abs(acc_bin - conf_bin)
    return ece


def classwise_ece(y_true, y_prob, n_bins=15):
    """Average of one-vs-rest ECEs over the 4 classes (a stricter calibration view)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    eces = []
    for c in range(n_classes):
        p = y_prob[:, c]
        y = (y_true == c).astype(float)
        edges = np.linspace(0, 1, n_bins + 1)
        e, n = 0.0, len(y)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
            if m.sum() == 0:
                continue
            e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
        eces.append(e)
    return float(np.mean(eces)), eces

# Library alternatives that do the same thing (use one of these to cross-check):
#   from torchmetrics.classification import MulticlassCalibrationError
#   MulticlassCalibrationError(num_classes=4, n_bins=15, norm='l1')   # == confidence-ECE
#   from netcal.metrics import ECE; ECE(bins=15).measure(y_prob, y_true)


# ----------------------------------------------------------------------
# 3. PER-CLASS CONFUSION-MATRIX METRICS incl. LR+ / LR-
# ----------------------------------------------------------------------
def per_class_diagnostics(y_true, y_pred, n_classes=4, continuity=True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    N = cm.sum()
    rows = {}
    for c in range(n_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = N - TP - FN - FP
        a, b, cc, d = TP, FP, FN, TN  # Simel notation: a=TP b=FP c=FN d=TN

        # continuity correction avoids div-by-zero / Inf when a cell is 0
        if continuity and 0 in (a, b, cc, d):
            a, b, cc, d = a + 0.5, b + 0.5, cc + 0.5, d + 0.5

        sens = a / (a + cc)
        spec = d / (b + d)
        ppv = a / (a + b)
        npv = d / (d + cc)
        lr_pos = sens / (1 - spec) if spec < 1 else np.inf
        lr_neg = (1 - sens) / spec if spec > 0 else np.inf

        # 95% CIs for LRs (Simel 1991, log-normal approximation)
        se_lp = np.sqrt(1/a - 1/(a + cc) + 1/b - 1/(b + d))
        se_ln = np.sqrt(1/cc - 1/(a + cc) + 1/d - 1/(b + d))
        lp_ci = (np.exp(np.log(lr_pos) - Z*se_lp), np.exp(np.log(lr_pos) + Z*se_lp))
        ln_ci = (np.exp(np.log(lr_neg) - Z*se_ln), np.exp(np.log(lr_neg) + Z*se_ln))

        rows[f"MES{c}"] = dict(TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN),
                               sensitivity=sens, specificity=spec, PPV=ppv, NPV=npv,
                               LR_pos=lr_pos, LR_pos_CI=lp_ci,
                               LR_neg=lr_neg, LR_neg_CI=ln_ci)
    return rows


# ----------------------------------------------------------------------
# 4. MACRO-AVERAGED ACCURACY + BOOTSTRAP CI
#    (Wilson is NOT valid for a macro-average; bootstrap is.)
# ----------------------------------------------------------------------
def macro_accuracy(y_true, y_pred, n_classes=4):
    recalls = []
    for c in range(n_classes):
        m = (np.asarray(y_true) == c)
        if m.sum() == 0:
            continue
        recalls.append((np.asarray(y_pred)[m] == c).mean())
    return float(np.mean(recalls))


def macro_accuracy_bootstrap_ci(y_true, y_pred, n_classes=4, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    stats = [macro_accuracy(y_true[i := rng.integers(0, n, n)], y_pred[i], n_classes)
             for _ in range(n_boot)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return macro_accuracy(y_true, y_pred, n_classes), (lo, hi)


# ----------------------------------------------------------------------
# DEMO with synthetic data (replace with your real arrays)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 300
    y_true = rng.integers(0, 4, N)
    # a decent-but-imperfect classifier
    y_pred = y_true.copy()
    flip = rng.random(N) < 0.05
    y_pred[flip] = rng.integers(0, 4, flip.sum())
    # plausible probability matrix
    y_prob = np.full((N, 4), 0.05)
    y_prob[np.arange(N), y_pred] = 0.85
    y_prob /= y_prob.sum(1, keepdims=True)

    print("Overall kappa:", kappas_overall(y_true, y_pred))
    print("QWK 95% CI   :", kappa_bootstrap_ci(y_true, y_pred))
    print("\nPer-class one-vs-rest kappa (note quad == unweighted):")
    for k, v in kappa_per_class_onevsrest(y_true, y_pred).items():
        print(f"  {k}: cohen={v['cohen_kappa']:.4f}  quad={v['quadratic_weighted_kappa']:.4f}")
    print("\nConfidence-ECE:", round(expected_calibration_error(y_true, y_prob), 4))
    print("Class-wise ECE:", round(classwise_ece(y_true, y_prob)[0], 4))
    print("\nMacro accuracy + CI:", macro_accuracy_bootstrap_ci(y_true, y_pred))
    print("\nPer-class diagnostics:")
    for k, v in per_class_diagnostics(y_true, y_pred).items():
        print(f"  {k}: sens={v['sensitivity']:.3f} spec={v['specificity']:.3f} "
              f"LR+={v['LR_pos']:.2f} LR-={v['LR_neg']:.3f}")
