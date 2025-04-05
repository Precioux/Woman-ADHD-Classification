# ADHD-fMRI Classification Pipeline

This repository contains a complete machine learning pipeline for predicting ADHD diagnosis and participant sex from fMRI-based brain connectome data and associated metadata. The model uses dimensionality reduction (PCA), feature selection (SelectKBest), and gradient boosting (XGBoost), with careful preprocessing and cross-dataset evaluation.

## 🧠 Problem Statement

Accurately identifying ADHD, especially in female participants, is challenging due to class imbalance and clinical variability. This pipeline aims to detect:

- **ADHD diagnosis** (binary)
- **Sex** (Female = 1)
- **ADHD-in-female** as a focused subgroup prediction

## 📁 Data Sources

Data is provided in three splits:
- **TRAIN_NEW** – Primary dataset for model development.
- **TRAIN_OLD** – Used for generalization testing.
- **TEST** – Blind evaluation set.

Each set includes:
- Functional connectome matrices (fMRI)
- Categorical metadata
- Quantitative metadata

## 🔧 Pipeline Overview

1. **Data Merging**: Combine fMRI, categorical, and quantitative metadata.
2. **Imputation**: Median for numerical, mode for categorical values.
3. **One-Hot Encoding**: For categorical metadata.
4. **Dimensionality Reduction**:
   - PCA on fMRI features (first 100 components).
   - SelectKBest using mutual information to select top 100 features from combined feature set.
5. **Model Training**:
   - Separate XGBoost classifiers for ADHD and Sex_F.
   - Threshold tuning for ADHD-in-female subgroup.
6. **Evaluation**:
   - On TRAIN_NEW (train-test split)
   - On TRAIN_OLD (external validation)
   - On TEST (final predictions)

## 📦 Files Included

- `train_pipeline.py`: Full training pipeline using TRAIN_NEW
- `evaluate_old.py`: Applies trained models to TRAIN_OLD
- `predict_test.py`: Runs inference on TEST and generates submission
- Saved models:
  - `quant_imputer.pkl`
  - `cat_imputer.pkl`
  - `scaler.pkl`
  - `pca.pkl`
  - `selectkbest.pkl`
  - `xgb_af_model.pkl`
  - `fmri_cols.pkl`
  - `meta_cols.pkl`
  - `combo_column_names.pkl`

## 📤 Submission Format

The final output CSV has:
| participant_id | ADHD_Outcome | Sex_F |
|----------------|--------------|-------|
| ABC123         | 0            | 1     |
| DEF456         | 1            | 0     |
| ...            | ...          | ...   |

## 🚀 How to Run

1. Place input files in a `/data` directory.
2. Run `train_pipeline.py` to train models and save preprocessing artifacts.
3. Use `evaluate_old.py` to test generalization on TRAIN_OLD.
4. Run `predict_test.py` to generate the final submission for TEST.

## 🧪 Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, xgboost, joblib

Install via:

```bash
pip install -r requirements.txt
```

## 🧠 Acknowledgments

Data provided by [WiDS Datathon 2025](https://www.widsconference.org/datathon.html)