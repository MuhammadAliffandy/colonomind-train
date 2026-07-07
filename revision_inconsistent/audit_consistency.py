"""
Audit ketat terhadap cm_master_supplementary.json
- Recalculate ALL metrics from the CM integer counts
- Flag setiap inconsistency dengan selisih > 0.0001
"""
import json, numpy as np
from sklearn.metrics import cohen_kappa_score

TOLS = 0.0001  # Toleransi maksimal

def recalc(cm_list):
    cm = np.array(cm_list, dtype=float)
    N  = int(cm.sum())
    diag = int(np.trace(cm))
    acc = diag / N

    # Wilson CI
    Z = 1.95996
    p = acc
    n = N
    center = (p + Z**2 / (2*n)) / (1 + Z**2 / n)
    margin = (Z / (1 + Z**2 / n)) * np.sqrt(p*(1-p)/n + Z**2/(4*n**2))
    ci_lo = center - margin
    ci_hi = center + margin

    y_true, y_pred = [], []
    for tr in range(4):
        for pr in range(4):
            cnt = int(cm[tr, pr])
            y_true.extend([tr]*cnt); y_pred.extend([pr]*cnt)
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    ck  = cohen_kappa_score(y_true, y_pred, weights=None)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    row_sums = cm.sum(axis=1); col_sums = cm.sum(axis=0)
    E = np.outer(row_sums, col_sums) / N
    W = np.array([[(i-j)**2/9. for j in range(4)] for i in range(4)])

    per_class = []
    for c in range(4):
        tp = int(cm[c,c]); fp = int(cm[:,c].sum()-tp)
        fn = int(cm[c,:].sum()-tp); tn = N - tp - fp - fn
        ppv = tp/(tp+fp) if (tp+fp) else 0.
        rec = tp/(tp+fn) if (tp+fn) else 0.
        npv = tn/(tn+fn) if (tn+fn) else 0.
        f1  = 2*ppv*rec/(ppv+rec) if (ppv+rec) else 0.
        y_tb = (y_true==c).astype(int); y_pb = (y_pred==c).astype(int)
        try: ovr_ck = cohen_kappa_score(y_tb, y_pb)
        except: ovr_ck = float('nan')
        num_c = np.sum(W[c,:]*cm[c,:]) + np.sum(W[:,c]*cm[:,c])
        den_c = np.sum(W[c,:]*E[c,:]) + np.sum(W[:,c]*E[:,c])
        sym_qwk = 1 - num_c/den_c if den_c else 1.
        per_class.append({'ppv':ppv,'recall':rec,'npv':npv,'f1':f1,'ovr_ck':ovr_ck,'sym_qwk':sym_qwk})
    return {'acc':acc,'ci_lo':ci_lo,'ci_hi':ci_hi,'ck':ck,'qwk':qwk,'per_class':per_class}

def check(label, stored, recalc_val, key, class_idx=None):
    sv = float(stored)
    rv = float(recalc_val)
    diff = abs(sv - rv)
    flag = "❌ INCONSISTENT" if diff > TOLS else "✅"
    if diff > TOLS:
        print(f"  {flag} [{label}] stored={sv:.4f} recalc={rv:.4f} diff={diff:.4f}")
    return diff > TOLS

with open('revision_inconsistent/cm_master_supplementary.json') as f:
    data = json.load(f)

total_issues = 0

for table_key, entries in data.items():
    for ds_key, entry in entries.items():
        ds_label = entry.get('Dataset', ds_key)
        cm_raw = entry.get('cm')
        if not cm_raw: continue

        r = recalc(cm_raw)
        issues = 0
        err_lines = []

        # Overall metrics
        for key_s, key_r in [('accuracy','acc'),('ci_lo','ci_lo'),('ci_hi','ci_hi'),
                              ('cohen_kappa','ck'),('quad_kappa','qwk')]:
            sv = entry.get(key_s)
            if sv is None: continue
            diff = abs(float(sv) - float(r[key_r]))
            if diff > TOLS:
                err_lines.append(f"  ❌ {key_s}: stored={float(sv):.4f} recalc={float(r[key_r]):.4f} Δ={diff:.4f}")
                issues += 1

        # Per-class metrics
        pc_stored = entry.get('per_class', {})
        for c in range(4):
            cls_key = f"MES {c}"
            pcs = pc_stored.get(cls_key, {})
            pcr = r['per_class'][c]
            for key_s, key_r in [('ppv','ppv'),('recall','recall'),('npv','npv'),
                                  ('f1','f1'),('ovr_ck','ovr_ck'),('sym_qwk','sym_qwk')]:
                sv = pcs.get(key_s)
                if sv is None: continue
                diff = abs(float(sv) - float(pcr[key_r]))
                if diff > TOLS:
                    err_lines.append(f"  ❌ {cls_key}.{key_s}: stored={float(sv):.4f} recalc={float(pcr[key_r]):.4f} Δ={diff:.4f}")
                    issues += 1

        total_issues += issues
        if err_lines:
            print(f"\n{'='*60}")
            print(f"[{table_key}] {ds_label}")
            for line in err_lines:
                print(line)

print(f"\n{'='*60}")
if total_issues == 0:
    print(f"✅ AUDIT PASSED: Semua {sum(len(v) for v in data.values())} dataset KONSISTEN.")
else:
    print(f"❌ TOTAL ISSUES DITEMUKAN: {total_issues}")
