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

### 5. Batch Training (All Dataset Combinations)
To train across **every** dataset combination at once (intra-domain, cross-domain, and multi-domain), use the batch script:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

This will automatically run **all experiments** and organize results into:
```text
results/
├── intra/                          # Same dataset train & test
│   ├── dataset_1/
│   ├── dataset_2/
│   ├── public/
│   └── mixed/
├── cross/                          # Different dataset train & test
│   ├── train_dataset_1_test_dataset_2/
│   ├── train_dataset_1_test_public/
│   ├── train_dataset_2_test_dataset_1/
│   └── ... (12 combinations total)
└── multi/                          # Mixed-domain training
    ├── train_mixed_test_dataset_1/
    ├── train_mixed_test_dataset_2/
    └── train_mixed_test_public/
```

Each folder contains: `best_hybrid_model.h5`, `scaler.pkl`, `label_encoder.pkl`, `umap_model.pkl`.

---

## 📊 Paper Evaluations Workflow

Follow this order to generate all paper-ready figures and tables:

### Step 1: Train Models
```bash
# Train a single experiment
python src/train.py --train_dir dataset_1 --test_dir dataset_2 --output_dir ./results/cross/train_dataset_1_test_dataset_2

# Or train ALL experiments at once
./run_all_experiments.sh
```

### Step 2: Run Evaluations (no training needed)
```bash
# Data exclusion flowchart & distribution tables
python paper_evaluations/data_stats.py
```

### Step 3: Run Evaluations (needs trained model)
```bash
# Confusion Matrix, 95% CI, Sensitivity/Specificity, Feature Importance
python paper_evaluations/run_analysis.py --model_dir ./results/cross/train_dataset_1_test_dataset_2

# Computational performance (params, latency, model size)
python paper_evaluations/run_benchmark.py --model_path ./results/cross/train_dataset_1_test_dataset_2/best_hybrid_model.h5
```

### Step 4: Baselines & Ablation (trains its own models)
```bash
# Compare against ResNet, DenseNet, EfficientNet, ViT
python paper_evaluations/run_baselines.py --model resnet

# Run ablation scenarios (1-6)
python paper_evaluations/run_ablation.py --scenario 1 --domain cross
```

### Step 5: Architecture Diagram
Open `paper_evaluations/architecture.md` in any Mermaid-compatible viewer (GitHub, VS Code, etc.).

**Available Scripts:**
| Script | Needs Training First? | Description |
|---|---|---|
| `data_stats.py` | ❌ No | Exclusion flowchart & data split tables |
| `run_baselines.py` | ❌ No (trains its own) | ResNet, DenseNet, EfficientNet, ViT comparison |
| `run_ablation.py` | ❌ No (trains its own) | 6 ablation scenarios |
| `run_analysis.py` | ✅ Yes | CM, 95% CI, Sensitivity/Specificity, Feature Importance |
| `run_benchmark.py` | ✅ Yes | Model size, params, inference latency |
| `architecture.md` | ❌ No | End-to-end system diagram |

*Run all scripts from the project root directory (`~/Clara/colono_train/`).*

---

## 🤝 Team Collaboration Guidelines

To keep this repository clean and lightweight for all team members, please adhere to the following rules:

1. **Do not push Datasets**: Never commit `.jpg`, `.png`, or `.csv` dataset files to Git. Keep them local.
2. **Do not push Large Model Weights**: Files like `.h5` or `.pkl` generated in your `results/` folder should remain local unless requested. If you must share them, use Google Drive, AWS S3, or Git LFS.
3. **Avoid Jupyter Notebook Outputs**: If you must create a `.ipynb` file for testing, **clear the outputs** (Cell > All Output > Clear) before committing. Notebook outputs inflate the repository size and break git history.
4. **Modify `src/`, don't duplicate**: If you want to change the CNN architecture or add a new handcrafted feature, edit `src/model.py` or `src/features.py`. Do not create a duplicate `src_v2/` folder. Keep the code centralized.

*Happy Coding!* 🚀
