#!/usr/bin/env python3
"""
Endoscopy foundation-model teacher probe (decision gate for knowledge distillation)
==================================================================================
1. Extract FROZEN features from a GI-endoscopy pretrained encoder for every image in the V6 cache.
2. Fit a probe on the TRAIN split only (hyper-parameters chosen on VALIDATION only).
3. Report val/test exact accuracy, 1-grade-off, QWK, remission-vs-active accuracy.
4. Save teacher probabilities (train = 5-fold out-of-fold, val/test = full-train probe) for distillation.

Splits are identical to V18 (test_size 0.20 then 0.20, random_state 42, stratified).

Usage (DGX):
  pip install torch torchvision timm transformers huggingface_hub scikit-learn
  python src/extract_endo_foundation_features.py --model gastronet
  python src/extract_endo_foundation_features.py --model endovit
"""
import os, json, argparse
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

MODELS = {
    # BONS-AI-TUE-AMC/GastroNetDinov2 only contains a README (no weights) -> use the weights repo
    "gastronet_weights": {"repo": "tgwboers/GastroNet-5M_Pretrained_Weights", "kind": "list_only"},
    "gastronet": {"repo": "BONS-AI-TUE-AMC/GastroNetDinov2", "kind": "hf_auto"},
    "endovit": {"repo": "egeozsoy/EndoViT", "kind": "timm_vit"},
}


def load_encoder(name, device):
    import torch
    cfg = MODELS[name]
    repo = cfg["repo"]
    try:
        if cfg["kind"] == "hf_auto":
            from transformers import AutoModel
            model = AutoModel.from_pretrained(repo, trust_remote_code=True)
            def fwd(x):
                out = model(pixel_values=x) if "pixel_values" in model.forward.__code__.co_varnames else model(x)
                h = getattr(out, "last_hidden_state", None)
                if h is None:
                    h = out[0] if isinstance(out, (tuple, list)) else out
                return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1) if h.ndim == 3 else h
            return model.to(device).eval(), fwd
        if cfg["kind"] == "timm_vit":
            import timm
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download(repo, "pytorch_model.bin")
            model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="avg")
            # The checkpoint pickles an argparse.Namespace. Allow-list only that class instead of
            # disabling weights_only, so no arbitrary code can run while loading.
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            sd = sd.get("model", sd)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"   missing keys: {len(missing)} | unexpected: {len(unexpected)}")
            if len(missing) > 10:
                raise RuntimeError(f"Encoder keys do not match timm ViT-B/16 ({len(missing)} missing): {missing[:5]}")
            return model.to(device).eval(), (lambda x: model(x))
    except Exception as e:
        print(f"❌ Could not load '{name}' from {repo}: {e}")
    try:
        from huggingface_hub import list_repo_files
        print(f"📂 Files in {repo}:")
        for f in list_repo_files(repo):
            print("   ", f)
    except Exception as e:
        print(f"   (cannot list repo files: {e})")
    raise SystemExit("Fix the model id/loader above (see files list) and re-run.")


def extract(imgs, fwd, device, size, bs=64):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            b = imgs[i:i + bs].astype(np.uint8)
            if b.shape[1] != size:
                b = np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_CUBIC) for x in b])
            x = torch.from_numpy(b).permute(0, 3, 1, 2).float().div(255.0).to(device)
            x = (x - mean) / std
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                f = fwd(x)
            feats.append(f.float().cpu().numpy())
            if (i // bs) % 50 == 0:
                print(f"   extracted {i}/{len(imgs)}")
    return np.concatenate(feats)


def report(tag, y, p):
    pred = p.argmax(1)
    print(f"   [{tag}] exact={accuracy_score(y, pred)*100:.2f}% | 1-off={np.mean(np.abs(y-pred)<=1)*100:.2f}% | "
          f"QWK={cohen_kappa_score(y, pred, weights='quadratic'):.4f} | "
          f"remission(0-1) vs active(2-3)={accuracy_score(y>=2, pred>=2)*100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gastronet", choices=list(MODELS))
    ap.add_argument("--v6_cache", default="../Result/ColonoMind_v6/dataset_cache_v6.npz")
    ap.add_argument("--save_dir", default="../Result/ColonoMind_v19_teacher")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ CUDA not usable from this PyTorch build (driver/wheel mismatch). Running on CPU: slow.\n"
              "   Install a cu124 wheel in a separate env, see instructions.")
        torch.set_num_threads(min(32, os.cpu_count() or 8))

    d = np.load(args.v6_cache, allow_pickle=True)
    imgs, y_full = d["imgs"], LabelEncoder().fit_transform(list(d["labels"]))
    idx = np.arange(len(y_full))
    tv, te = train_test_split(idx, test_size=0.20, random_state=42, stratify=y_full)
    tr, va = train_test_split(tv, test_size=0.20, random_state=42, stratify=y_full[tv])
    print(f"Split: train={len(tr)} val={len(va)} test={len(te)}")

    feat_path = os.path.join(args.save_dir, f"feats_{args.model}.npy")
    if os.path.exists(feat_path):
        F = np.load(feat_path)
    else:
        model, fwd = load_encoder(args.model, device)
        F = extract(imgs, fwd, device, args.size)
        np.save(feat_path, F)
    print(f"Feature matrix: {F.shape}")

    sc = StandardScaler().fit(F[tr])
    Ftr, Fva, Fte = sc.transform(F[tr]), sc.transform(F[va]), sc.transform(F[te])
    ytr, yva, yte = y_full[tr], y_full[va], y_full[te]

    best = (-1, None)
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        clf = LogisticRegression(C=C, max_iter=2000, n_jobs=-1)
        clf.fit(Ftr, ytr)
        acc = accuracy_score(yva, clf.predict(Fva))
        print(f"   C={C}: val exact={acc*100:.2f}%")
        if acc > best[0]:
            best = (acc, C)
    C = best[1]
    print(f"✅ Selected C={C} on validation")

    clf = LogisticRegression(C=C, max_iter=2000, n_jobs=-1).fit(Ftr, ytr)
    p_va, p_te = clf.predict_proba(Fva), clf.predict_proba(Fte)
    p_tr = np.zeros((len(tr), 4), np.float32)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(Ftr, ytr):
        p_tr[b] = LogisticRegression(C=C, max_iter=2000, n_jobs=-1).fit(Ftr[a], ytr[a]).predict_proba(Ftr[b])

    print("\n📊 Frozen-feature probe results")
    report("train OOF", ytr, p_tr); report("val", yva, p_va); report("test", yte, p_te)
    print("Confusion (test):\n", confusion_matrix(yte, p_te.argmax(1)))
    np.savez(os.path.join(args.save_dir, f"teacher_probs_{args.model}.npz"),
             p_tr=p_tr, p_va=p_va, p_te=p_te, idx_tr=tr, idx_va=va, idx_te=te)
    json.dump({"model": args.model, "C": C, "val_exact": float(best[0]),
               "test_exact": float(accuracy_score(yte, p_te.argmax(1)))},
              open(os.path.join(args.save_dir, f"probe_{args.model}.json"), "w"), indent=2)
    print("\nDecision gate: if val/test exact is clearly above ~83%, distil into Mod-SE(2); otherwise skip KD.")


if __name__ == "__main__":
    main()

