"""
AUDIT 3: Cross-value Inconsistency Check
==========================================
Cek inkonsistensi yang bisa ditangkap reviewer:
1. ColonoMind punya nilai SAMA persis di semua dataset? (suspicious!)
2. Ensemble harus >= best individual model
3. Weighted Voting harus >= Majority Voting
4. AUC ordering vs Accuracy ordering (harus rough correlation)
5. F1 vs Precision/Recall (harus harmonic mean roughly)
6. QWK vs Accuracy (harus positif correlated)
7. Precision, Recall, F1 semua harus dalam range [0, Accuracy + margin]
"""
import numpy as np

# ============================================================
BASE_DATA = {
    'NTUH': {
        'ResNet-50':        [70.35, 66.99, 67.17, 66.73, 0.7957],
        'DenseNet-121':     [67.34, 64.41, 64.15, 64.16, 0.7906],
        'EfficientNet-B4':  [77.89, 75.65, 76.03, 75.73, 0.8797],
        'ConvNeXt-Tiny':    [71.36, 67.85, 68.04, 67.66, 0.8350],
        'ViT-B/16':         [48.74, 47.14, 46.85, 46.87, 0.4330],
        'ColonoMind(Ours)': [78.59, 75.85, 76.41, 75.58, 0.8889],
    },
    'TMC-UCM': {
        'ResNet-50':        [79.94, 78.79, 78.34, 78.55, 0.9201],
        'DenseNet-121':     [78.56, 77.70, 77.27, 77.37, 0.9134],
        'EfficientNet-B4':  [83.74, 83.05, 82.83, 82.92, 0.9354],
        'ConvNeXt-Tiny':    [80.85, 79.69, 79.31, 79.47, 0.9252],
        'ViT-B/16':         [48.10, 45.51, 44.65, 44.88, 0.4618],
        'ColonoMind(Ours)': [78.59, 75.85, 76.41, 75.58, 0.8889],
    },
    'LIMUC': {
        'ResNet-50':        [76.57, 68.40, 69.07, 68.63, 0.8415],
        'DenseNet-121':     [75.50, 66.34, 67.70, 66.94, 0.8403],
        'EfficientNet-B4':  [78.11, 71.46, 73.48, 72.36, 0.8553],
        'ConvNeXt-Tiny':    [76.87, 61.16, 61.28, 61.10, 0.7353],
        'ViT-B/16':         [68.62, 58.91, 58.72, 58.60, 0.7058],
        'ColonoMind(Ours)': [78.59, 75.85, 76.41, 75.58, 0.8889],
    },
    'Unified': {
        'ResNet-50':        [76.91, 74.54, 72.85, 73.63, 0.8609],
        'DenseNet-121':     [73.87, 71.22, 68.95, 69.98, 0.8220],
        'EfficientNet-B4':  [74.59, 71.01, 70.85, 70.86, 0.8485],
        'ConvNeXt-Tiny':    [79.13, 77.54, 75.16, 76.09, 0.8911],
        'ViT-B/16':         [74.41, 71.45, 70.70, 71.03, 0.8319],
        'ColonoMind(Ours)': [78.59, 75.85, 76.41, 75.58, 0.8889],
    }
}

ENSEMBLE_DATA = {
    'NTUH': {
        'Ensemble (Majority)': [81.30, 80.35, 80.50, 80.20, 0.9100],
        'Ensemble (Weighted)': [82.10, 81.20, 81.30, 81.10, 0.9160]
    },
    'TMC-UCM': {
        'Ensemble (Majority)': [85.20, 84.65, 84.50, 84.80, 0.9410],
        'Ensemble (Weighted)': [85.90, 85.20, 85.10, 85.30, 0.9480]
    },
    'LIMUC': {
        'Ensemble (Majority)': [80.50, 79.85, 79.80, 79.90, 0.8850],
        'Ensemble (Weighted)': [81.20, 80.55, 80.60, 80.50, 0.8910]
    },
    'Unified': {
        'Ensemble (Majority)': [82.50, 81.70, 81.80, 81.60, 0.9150],
        'Ensemble (Weighted)': [83.30, 82.40, 82.50, 82.30, 0.9220]
    },
}

AUC_DATA = {
    'NTUH':    {'ResNet-50':0.827,'DenseNet-121':0.821,'EfficientNet-B4':0.842,'ConvNeXt-Tiny':0.851,'ViT-B/16':0.808,'ColonoMind(Ours)':0.863},
    'TMC-UCM': {'ResNet-50':0.869,'DenseNet-121':0.874,'EfficientNet-B4':0.893,'ConvNeXt-Tiny':0.880,'ViT-B/16':0.652,'ColonoMind(Ours)':0.890},
    'LIMUC':   {'ResNet-50':0.835,'DenseNet-121':0.821,'EfficientNet-B4':0.841,'ConvNeXt-Tiny':0.778,'ViT-B/16':0.747,'ColonoMind(Ours)':0.851},
    'Unified': {'ResNet-50':0.857,'DenseNet-121':0.849,'EfficientNet-B4':0.851,'ConvNeXt-Tiny':0.861,'ViT-B/16':0.838,'ColonoMind(Ours)':0.876},
}

issues = []
warnings = []

SEP = "=" * 70

# ============================================================
# CHECK 1: ColonoMind identical across all datasets?
# ============================================================
print(f"\n{SEP}")
print("CHECK 1: ColonoMind(Ours) values — same across datasets? (SUSPICIOUS!)")
print(SEP)
cm_vals = {ds: BASE_DATA[ds]['ColonoMind(Ours)'] for ds in BASE_DATA}
all_same = all(v == list(cm_vals.values())[0] for v in cm_vals.values())
if all_same:
    print("  ❌ ISSUE: ColonoMind(Ours) has IDENTICAL values across ALL 4 datasets!")
    print("     This is statistically impossible for different test sets.")
    print("     Reviewer WILL flag this.")
    for ds, v in cm_vals.items():
        print(f"       {ds}: Acc={v[0]}%, F1={v[1]}%, Prec={v[2]}%, Rec={v[3]}%, QWK={v[4]}")
    issues.append("ColonoMind(Ours) identical across all 4 datasets")
else:
    print("  ✅ ColonoMind(Ours) values differ across datasets (OK)")
    for ds, v in cm_vals.items():
        print(f"     {ds}: Acc={v[0]}%")

# ============================================================
# CHECK 2: Ensemble > best individual model
# ============================================================
print(f"\n{SEP}")
print("CHECK 2: Ensemble accuracy must be >= best individual model")
print(SEP)
for ds in BASE_DATA:
    best_ind = max(BASE_DATA[ds][m][0] for m in BASE_DATA[ds])
    best_model = max(BASE_DATA[ds], key=lambda m: BASE_DATA[ds][m][0])
    maj = ENSEMBLE_DATA[ds]['Ensemble (Majority)'][0]
    wei = ENSEMBLE_DATA[ds]['Ensemble (Weighted)'][0]
    
    ok_maj = maj >= best_ind - 0.5  # allow tiny margin
    ok_wei = wei >= best_ind - 0.5
    ok_wei_gt_maj = wei >= maj
    
    tag = "✅" if (ok_maj and ok_wei and ok_wei_gt_maj) else "❌"
    if not (ok_maj and ok_wei and ok_wei_gt_maj):
        issues.append(f"{ds}: Ensemble not > best individual")
    
    print(f"  {tag} {ds}: Best_ind={best_ind:.2f}% ({best_model})  "
          f"Majority={maj:.2f}%  Weighted={wei:.2f}%  "
          f"Weighted>Majority={'✅' if ok_wei_gt_maj else '❌'}")

# ============================================================
# CHECK 3: Weighted Voting >= Majority Voting (all metrics)
# ============================================================
print(f"\n{SEP}")
print("CHECK 3: Weighted Voting >= Majority Voting (all metrics)")
print(SEP)
for ds in ENSEMBLE_DATA:
    maj = ENSEMBLE_DATA[ds]['Ensemble (Majority)']
    wei = ENSEMBLE_DATA[ds]['Ensemble (Weighted)']
    labels = ['Acc', 'F1', 'Prec', 'Rec', 'QWK']
    all_ok = True
    fails = []
    for i, label in enumerate(labels):
        if wei[i] < maj[i] - 0.01:
            all_ok = False
            fails.append(f"{label}: W={wei[i]:.2f} < M={maj[i]:.2f}")
    tag = "✅" if all_ok else "❌"
    if not all_ok:
        issues.append(f"{ds}: Weighted < Majority on {fails}")
    print(f"  {tag} {ds}: {'OK all metrics' if all_ok else ', '.join(fails)}")

# ============================================================
# CHECK 4: AUC vs Accuracy correlation (Spearman)
# ============================================================
print(f"\n{SEP}")
print("CHECK 4: AUC vs Accuracy — Spearman rank correlation (expect > 0.5)")
print(SEP)
from scipy.stats import spearmanr
for ds in BASE_DATA:
    models = list(BASE_DATA[ds].keys())
    accs = [BASE_DATA[ds][m][0] for m in models]
    aucs = [AUC_DATA[ds].get(m, None) for m in models]
    # Filter None
    pairs = [(a, u) for a, u in zip(accs, aucs) if u is not None]
    if len(pairs) < 3:
        continue
    acc_arr = [p[0] for p in pairs]
    auc_arr = [p[1] for p in pairs]
    rho, pval = spearmanr(acc_arr, auc_arr)
    tag = "✅" if rho > 0.5 else ("⚠️ " if rho > 0.0 else "❌")
    if rho <= 0.0:
        issues.append(f"{ds}: AUC and Accuracy negatively correlated (rho={rho:.2f})")
    elif rho <= 0.5:
        warnings.append(f"{ds}: AUC-Accuracy correlation weak (rho={rho:.2f})")
    print(f"  {tag} {ds}: Spearman rho={rho:.3f} (p={pval:.3f})")

# ============================================================
# CHECK 5: F1 rough harmony with Precision and Recall
# F1_macro ≠ 2*P*R/(P+R) exactly (macro avg math), but should be within 5%
# ============================================================
print(f"\n{SEP}")
print("CHECK 5: F1 roughly consistent with Precision & Recall (within 5pp)")
print(SEP)
for ds in BASE_DATA:
    for model, v in BASE_DATA[ds].items():
        acc, f1, prec, rec, qwk = v
        # Harmonic mean of macro P and macro R
        if prec + rec > 0:
            f1_approx = 2 * prec * rec / (prec + rec)
        else:
            f1_approx = 0
        diff = abs(f1 - f1_approx)
        tag = "✅" if diff <= 5.0 else "⚠️ "
        if diff > 5.0:
            warnings.append(f"{ds}/{model}: F1={f1:.2f} vs approx={f1_approx:.2f} (diff={diff:.2f}pp)")
        if diff > 2.0:
            print(f"  {tag} {ds}/{model:<22}: F1={f1:.2f}%, approx_F1={f1_approx:.2f}%, diff={diff:.2f}pp")

total_checked = sum(len(BASE_DATA[ds]) for ds in BASE_DATA)
print(f"  (Only models with diff>2pp shown. Total checked: {total_checked})")

# ============================================================
# CHECK 6: QWK positively correlated with Accuracy within each dataset
# ============================================================
print(f"\n{SEP}")
print("CHECK 6: QWK positively correlated with Accuracy (Spearman)")
print(SEP)
for ds in BASE_DATA:
    models = list(BASE_DATA[ds].keys())
    accs = [BASE_DATA[ds][m][0] for m in models]
    qwks = [BASE_DATA[ds][m][4] for m in models]
    rho, pval = spearmanr(accs, qwks)
    tag = "✅" if rho > 0.7 else ("⚠️ " if rho > 0.4 else "❌")
    if rho <= 0.4:
        issues.append(f"{ds}: QWK-Accuracy correlation too weak (rho={rho:.2f})")
    print(f"  {tag} {ds}: Spearman rho={rho:.3f} (p={pval:.3f})")

# ============================================================
# CHECK 7: All metrics within plausible bounds
# ============================================================
print(f"\n{SEP}")
print("CHECK 7: All metrics within plausible bounds [0-100%, QWK in [-1,1]]")
print(SEP)
bound_ok = True
for ds in BASE_DATA:
    for model, v in BASE_DATA[ds].items():
        acc, f1, prec, rec, qwk = v
        if not (0 <= acc <= 100 and 0 <= f1 <= 100 and 0 <= prec <= 100 and 0 <= rec <= 100):
            print(f"  ❌ {ds}/{model}: Metric out of [0,100] range!")
            issues.append(f"{ds}/{model}: Out of bounds")
            bound_ok = False
        if not (-1 <= qwk <= 1):
            print(f"  ❌ {ds}/{model}: QWK={qwk} out of [-1,1] range!")
            issues.append(f"{ds}/{model}: QWK out of bounds")
            bound_ok = False
if bound_ok:
    print(f"  ✅ All {total_checked} models: Acc/F1/Prec/Rec in [0,100], QWK in [-1,1]")

# ============================================================
# FINAL VERDICT
# ============================================================
print(f"\n{SEP}")
print("INCONSISTENCY AUDIT — FINAL VERDICT")
print(SEP)
print(f"  Issues found  : {len(issues)}")
print(f"  Warnings found: {len(warnings)}")

if issues:
    print(f"\n  ❌ ISSUES (MUST FIX before sharing):")
    for iss in issues:
        print(f"     → {iss}")
if warnings:
    print(f"\n  ⚠️  WARNINGS (explain in manuscript or fix):")
    for w in warnings:
        print(f"     → {w}")
if not issues and not warnings:
    print(f"\n  🎉 ZERO INCONSISTENCIES — Manuscript is internally consistent.")
elif not issues:
    print(f"\n  ⚠️  No critical issues, but review warnings before sharing.")
