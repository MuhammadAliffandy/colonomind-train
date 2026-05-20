# Antigravity AI Assistant Guidelines for Colonomind Project

This file serves as a memory and instruction log for the Antigravity AI assistant regarding the Colonomind Training project. When performing tasks in this repository, the AI must adhere to the following rules:

## 1. Project Architecture Rules
- **Core Training Code**: All core logic for the Hybrid Mod-SE(2) CNN architecture must remain in the `src/` directory (`config.py`, `features.py`, `data_loader.py`, `model.py`, `train.py`). Do not duplicate this folder.
- **Paper Evaluations**: All scripts requested by reviewers (Ablation studies, Baselines, Data Statistics, SHAP Feature Importance, Benchmarks) must be placed in the `paper_evaluations/` directory.
- **Legacy Code**: Do not delete old Jupyter Notebooks. If new legacy notebooks appear, they must be moved to `Legacy_Notebooks/`.

## 2. Git & Repository Hygiene
- **Jupyter Notebooks**: Always run `jupyter nbconvert --clear-output --inplace <file>` or equivalent Python scripts to clear all embedded outputs (images, logs) before committing to Git. This prevents the repository size from blowing up.
- **Secrets**: Ensure no OpenAI API keys (`sk-...`) or other secrets are hardcoded in any script or notebook. Replace them with `os.getenv("OPENAI_API_KEY")`.
- **Large Files**: Do not track datasets (`.jpg`, `.csv`) or large model weights (`.h5`, `.pkl`) in Git unless explicitly asked by the user via Git LFS.

## 3. Collaboration & Documentation
- Whenever a major structural change is made (like adding a new evaluation metric), **always update the `README.md`** to reflect how the team should run the new code.
- Ensure any python scripts intended for team use include `argparse` for easy CLI execution.
