# ColonoMind: Agent Context Awareness & Project Knowledge Base (AGENTS.md)

This document provides persistent context awareness, system architecture reference, and historical continuity for AI agents collaborating on the **ColonoMind** project.

---

## 1. Executive Summary & Best Model Heritage

* **Core Project Goal**: AI-assisted endoscopic severity grading for Ulcerative Colitis according to the **Mayo Endoscopic Score (MES 0–3)**.
* **Strict Research Constraint**: Must use pure in-house **Mod-SE CNN** (Convolutional Neural Network with Squeeze-and-Excitation blocks) trained from scratch without external pre-trained ImageNet weights.
* **Breakthrough Model**: **ColonoMind V8 — Hierarchical Two-Stage Mod-SE CNN**
  * *Code*: `src/train_colonomind_v8_hierarchical.py`
  * *Akurasi Exact Match (4 Kelas)*: **>80%** (memecahkan bottleneck V5–V7 yang tertahan di ~78–79%).
  * *Akurasi Clinical Tolerance (1-Grade Off / Top-2)*: **>90%** (standar toleransi klinis antar-endoskopis).

---

## 2. ColonoMind V8 Architecture Breakdown

### A. Two-Stage Decision Pipeline
1. **Stage 1 — Binary Detector (Normal vs Active Disease)**:
   * Ensemble of 3 Mod-SE CNNs (`stage1_cnn_seed42.h5`, `seed123.h5`, `seed999.h5`).
   * Specifically trained to separate **MES0 (Remission/Normal)** from **Active Inflammation (MES 1, 2, 3)**.
   * Protects MES0 F1-Score at **~89%** and eliminates false positive flare diagnoses.
2. **Stage 2 — Ternary Severity Grader (MES1 vs MES2 vs MES3)**:
   * Ensemble of 3 Mod-SE CNNs (`stage2_cnn_seed42.h5`, `seed123.h5`, `seed999.h5`).
   * Specifically fine-tuned only on inflamed mucosa subsets to adjudicate mild, moderate, and severe inflammation.
3. **Hierarchical Super Agents (LightGBM)**:
   * `super_agent_stage1.txt`: Evaluates concatenated Deep Features (Fusion layer) + 28 Handcrafted Biomarkers + 2D UMAP + Model Probabilities to produce disease active probability.
   * `super_agent_stage2.txt`: Evaluates same multimodal feature vector on the ternary subset to adjudicate severity.
4. **Decision Routing Logic**:
   * If `prob_active < thr_stg1 (default 0.50)` $\rightarrow$ Output **MES0**.
   * Else $\rightarrow$ Run Stage 2 and output `argmax(probs * weights) + 1` (yielding **MES1, MES2, or MES3**).

### B. 28 Quantitative Biomarkers (Tissue Features)
* **20 Wavelet & Texture Features**:
  * 16 Subband statistics (Mean, Std, Var, Entropy for LL, LH, HL, HH via Daubechies `db1`).
  * 1 High-Frequency Energy (`hh_energy`).
  * 3 GLCM features (Contrast, Dissimilarity, Homogeneity with distance=5, angles=[0, 45, 90]).
* **8 Clinical Color Moments**:
  * R-Mean, G-Mean, Normalized Red-Green ratio `(R - G)/(R + G)`.
  * Green channel variance, Hue histogram entropy.
  * HSV Saturation mean & standard deviation.
  * Highlight / mucosal glare reflection ratio (`R>200 & G>200 & B>200`).

---

## 3. Web & Production Architecture (`colonomind-rework`)

The clinical web application is located in the sibling repository `../colonomind-rework`.

### A. Decoupled Core Contract
The entire frontend, SMART on FHIR integration, and AI chat co-pilot interact with the model via a **single decoupled contract** in `app/services/pred_services.py`:

```python
result = predict_from_image(image_path)
```

**Returned JSON Schema (Mandatory for Frontend & FHIR compatibility):**
```json
{
  "prediction_label": "MES1",
  "prediction": 1,
  "confidence": 0.7873,
  "disease_active_probability": 0.9198,
  "active_breakdown": {
    "MES1": 0.7873,
    "MES2": 0.1267,
    "MES3": 0.0858
  },
  "status": "High Confidence Analysis",
  "is_referral": false,
  "all_features": {
    "LL_Mean": 206.0,
    ...
    "Col_HighLight_Ratio": 0.0078
  }
}
```

* **Frontend UI (`templates/index.html`)**: Never touches ML code directly. Consumes the above JSON via `/predict` endpoint.
* **SMART on FHIR (`app/smart/fhir_bundle.py`)**: Transforms `result` into an HL7 FHIR R4 Bundle (`Patient`, `Observation`, `DiagnosticReport`) with SNOMED and LOINC codings.
* **ColonoTalk Chat (`app/services/ai_services.py`)**: Uses `prediction` and `all_features` to fuel clinical dialogue with clinicians via OpenAI.

---

## 4. Containerization & GCP CI/CD Pipeline

* **Docker Base**: `python:3.10-slim`, `gunicorn --bind :$PORT --workers 1 --threads 8 run:app`.
* **Model Ingestion Rule**: Model weights (`.h5`, `.pkl`, `.txt`) are **NEVER committed to Git**.
* **Cloud Storage**: Stored in Google Cloud Storage bucket `gs://colonoscan-models/`.
* **Google Cloud Build (`cloudbuild.yaml`)**:
  1. Downloads V8 model files from `gs://colonoscan-models/ColonoMind_V8_Package/models/*` (or `colonomind_v8/*`) into `./new_model/`.
  2. Builds Docker image and copies `COPY new_model/ /app/new_model/`.
  3. Pushes to Google Artifact Registry (`asia-southeast1`).
  4. Deploys container to **Google Cloud Run** (4Gi RAM, 2 vCPU, Port 8501).

---

## 5. Server Environments & Paths Quick Reference

| Environment | Item | Path / Value |
|---|---|---|
| **NTU DGX GPU Server** | Project Directory | `~/raid/Clara/colonomind-train/` |
| **NTU DGX GPU Server** | Cloud Drive Mount | `~/xxdrive` |
| **NTU DGX GPU Server** | Datasets | `~/Clara/new_drive/Dataset_Extracted/Dataset+Code/` |
| **NTU DGX GPU Server** | Packaging Script | `python src/package_v8_to_xxdrive.py --dest ~/xxdrive` |
| **Local Development** | Training Repo | `Colonomind Training Resource/` |
| **Local Development** | Web Deployment Repo | `colonomind-rework/` |
| **Google Cloud (GCP)** | Cloud Storage Bucket | `gs://colonoscan-models/` |
| **Google Cloud (GCP)** | Cloud Run Service | `colonomind` (Region: `asia-southeast1`) |
