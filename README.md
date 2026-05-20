# Colonomind Training Architecture

Welcome to the **Colonomind Training Repository**. This repository has been recently refactored to streamline the training and evaluation of the Mod-SE(2) CNN Hybrid model across various colonoscopy datasets (e.g., Dataset 1, Dataset 2, Public Dataset, TMC).

---

## 📁 Repository Structure

```text
.
├── Legacy_Notebooks/      # Archived Jupyter Notebooks from previous experiments.
├── src/                   # Core modular Python code
│   ├── config.py          # Global parameters (IMG_SIZE, WAVELET)
│   ├── features.py        # Handcrafted feature extraction (Wavelet, GLCM)
│   ├── data_loader.py     # Image loading, preprocessing, and feature binding
│   ├── model.py           # Mod-SE(2) CNN layers and Hybrid Model architecture
│   └── train.py           # Master script for executing training runs
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

### 2. Dataset Preparation
Ensure your datasets are unzipped and organized locally. **Do not commit datasets to GitHub.**
Expected dataset folder structure:
```text
Dataset_Name/
├── class_0/
│   ├── img1.jpg
│   └── img2.jpg
├── class_1/
└── class_2/
```

### 3. How to Train a Model
You no longer need to copy/paste Jupyter Notebooks for new experiments! Use the unified `train.py` script. It automatically handles SMOTE balancing, UMAP projections, CNN training, and artifact saving.

**Example Command:**
```bash
python src/train.py \
  --train_dir "./path/to/Train_Dataset" \
  --test_dir "./path/to/Test_Dataset" \
  --output_dir "./results/Experiment_Name" \
  --batch_size 16 \
  --epochs 20
```

#### Command Arguments:
- `--train_dir`: Absolute or relative path to the training dataset folder.
- `--test_dir`: Absolute or relative path to the testing dataset folder.
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
