# Colonomind Training Architecture

Welcome to the **Colonomind Training Repository**. This repository has been recently refactored to streamline the training and evaluation of the Mod-SE(2) CNN Hybrid model across various colonoscopy datasets (e.g., Dataset 1, Dataset 2, Public Dataset, TMC).

---

## 📁 Repository Structure

```text
.
├── Legacy_Notebooks/      # Archived Jupyter Notebooks from previous experiments.
├── paper_evaluations/     # Scripts for generating paper-ready results
├── src/                   # Core modular Python code
│   ├── config.py          # Global parameters, server paths, dataset registry
│   ├── features.py        # Handcrafted feature extraction (Wavelet, GLCM)
│   ├── data_loader.py     # Image loading, preprocessing, and feature binding
│   ├── model.py           # Mod-SE(2) CNN layers and Hybrid Model architecture
│   └── train.py           # Master script for executing training runs
├── antigravity.md         # AI assistant instruction log
├── requirements.txt       # Python package dependencies
└── README.md              # This file
```

---

## 🚀 Getting Started

### 1. Installation
To ensure all collaborators are using the exact same environment, please install dependencies from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 2. Server Environment (NTU DGX)
On the NTU DGX server, the following paths are pre-configured:

| Item | Path |
|---|---|
| **Project Root** | `~/Clara/colono_train/` |
| **Dataset Root** | `~/Clara/new_drive/Dataset_Extracted/Dataset+Code/` |

**Available Datasets (registered in `src/config.py`):**
| Short Name | Folder |
|---|---|
| `dataset_1` | `MES classification_20250313` |
| `dataset_2` | `MES classification_20250724` |
| `public` | `MES_Colonoscopy Public Dataset` |
| `mixed` | `MES Mixed Data` |

### 3. How to Train a Model
You no longer need to copy/paste Jupyter Notebooks for new experiments! Use the unified `train.py` script. It automatically handles SMOTE balancing, UMAP projections, CNN training, and artifact saving.

**Option A — Using short dataset names (recommended on DGX):**
```bash
python src/train.py \
  --train_dir dataset_1 \
  --test_dir dataset_2 \
  --output_dir ./results/train1_test2
```

**Option B — Using full custom paths:**
```bash
python src/train.py \
  --train_dir ~/Clara/new_drive/Dataset_Extracted/Dataset+Code/"MES classification_20250313" \
  --test_dir ~/Clara/new_drive/Dataset_Extracted/Dataset+Code/"MES classification_20250724" \
  --output_dir ./results/train1_test2 \
  --batch_size 16 \
  --epochs 20
```

#### Command Arguments:
- `--train_dir`: A registered short name (`dataset_1`, `dataset_2`, `public`, `mixed`) or an absolute/relative path.
- `--test_dir`: Same as above.
- `--output_dir`: Path where the trained models and artifacts will be saved.
- `--batch_size`: (Optional) Default is 16.
- `--epochs`: (Optional) Default is 20.

### 4. Output Artifacts
After training completes, the `--output_dir` will contain:
- `best_hybrid_model.h5`: The trained Mod-SE(2) CNN weights.
- `scaler.pkl`: The StandardScaler fitted on your training handcrafted features.
- `label_encoder.pkl`: The encoder used for your dataset classes.
- `umap_model.pkl`: The fitted UMAP reducer for hybrid decision tree inference.

---

## 📊 Paper Evaluations Framework

To satisfy reviewer requirements and generate paper-ready figures and tables, we have a dedicated `paper_evaluations/` directory.

- **`data_stats.py`**: Generates exclusion flowcharts (N=2536 -> N=997) and Dataset Distribution tables (Extended Data Tables 2a & 2b).
- **`run_ablation.py`**: Runs the 6 ablation scenarios across intra, cross, and multi-domain settings.
- **`run_baselines.py`**: Compares our Hybrid model against standard architectures (ResNet, DenseNet, EfficientNet, ConvNeXt, ViT).
- **`run_analysis.py`**: Calculates 95% Confidence Intervals, Sensitivity/Specificity per class, plots Confusion Matrices, and generates Handcrafted Feature Importance charts.
- **`run_benchmark.py`**: Measures computational performance (Model Size in MB, Total Parameters, Hardware specs, and Inference Latency in ms).
- **`architecture.md`**: Contains a Mermaid.js diagram of the end-to-end system architecture.

*To run any of these, simply execute them via Python from the root directory.*

---

## 🤝 Team Collaboration Guidelines

To keep this repository clean and lightweight for all team members, please adhere to the following rules:

1. **Do not push Datasets**: Never commit `.jpg`, `.png`, or `.csv` dataset files to Git. Keep them local.
2. **Do not push Large Model Weights**: Files like `.h5` or `.pkl` generated in your `results/` folder should remain local unless requested. If you must share them, use Google Drive, AWS S3, or Git LFS.
3. **Avoid Jupyter Notebook Outputs**: If you must create a `.ipynb` file for testing, **clear the outputs** (Cell > All Output > Clear) before committing. Notebook outputs inflate the repository size and break git history.
4. **Modify `src/`, don't duplicate**: If you want to change the CNN architecture or add a new handcrafted feature, edit `src/model.py` or `src/features.py`. Do not create a duplicate `src_v2/` folder. Keep the code centralized.

*Happy Coding!* 🚀
