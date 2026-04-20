# 🏥 Triagegeist: SOTA Emergency Triage Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-orange.svg)](https://lightgbm.readthedocs.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/triagegeist)

A high-performance, zero-leakage predictive pipeline designed to predict the **Emergency Severity Index (ESI)** from clinical data. This repository contains the solution tailored for the Kagan Triagegeist competition, combining advanced Deep Learning Semantics (`Bio_ClinicalBERT`) with high-speed Gradient Boosting (`LightGBM`).

---

## 🎯 Approach: The Two-Stage Hierarchical Classifier
Predicting emergency severity suffers from severe **class imbalance**, where ESI-1 (life-saving interventions) occurs in less than `~4%` of patients. Treating this as a flat 5-class problem dilutes critical cases.

This engine utilizes a **Hierarchical Two-Stage Pipeline**:
1. **The ESI-1 Hunter (Binary Level):** Explicitly hunts for the most critical ESI-1 patients against all else. Trained with intense asymmetric sample weighting (`scale_pos_weight=20.0`).
2. **The Acuity Sorter (Multiclass Level):** If the patient survives the first gate (not ESI-1), this 4-class classifier carefully sorts ESI 2 through 5 using balanced boundaries.

### 🧠 Semantic Understanding via ClinicalBERT
Raw TF-IDF fails to catch the nuanced intent behind medical jargon, spelling errors, and chief complaints. We offload textual understanding to **HuggingFace's `emilyalsentzer/Bio_ClinicalBERT`**. The 768-dimensional embeddings are heavily compressed to 100 components using **PCA**.

---

## 🛡️ Zero Data Leakage Architecture
In clinical predictions, data leakage is a silent killer. This repository features an aggressive, paranoid Cross-Validation loop:
* **PCA inside the K-Fold:** The Principal Component Analysis acts purely *inside* the K-Fold iteration. Eigenvectors are `.fit()` strictly on `X_train` and blindly transformed across `X_val`, preventing future variance bleed.
* **Native Category Alignment:** `LabelEncoder` usage is stripped. Real categorical variables map through dynamic Pandas `categories` learned from the Training fold structure, passing natively to LightGBM.
* **Native Missingness (NaNs):** Heartrate not captured? This absence of information is clinically valuable. LightGBM handles `NaN` structures natively instead of destroying patterns with median imputation.

---

## 📥 Setup and Reproducibility 

### 1. Requirements Installation
To rebuild this environment locally, simply use `pip`:
```bash
pip install -r requirements.txt
```

### 2. Generating Bio_ClinicalBERT Embeddings
To comply with processing limits (like Kaggle's 9-hour limit), semantic extraction is executed offline. Before running the main notebook, you must compile the embeddings from the raw input `chief_complaints.csv`.
```bash
python extract_bert.py
```
> **Note:** This will output `cc_bert_features.pkl` (~300MB). Keep it in the root directory.

### 3. Training & Evaluation
Open the primary notebook to evaluate the entire system end-to-end:
```bash
jupyter notebook triagegeist_baseline.ipynb
```
Select **Restart & Run All**. The pipeline will:
1. Merge Tabular Vitals with ClinicalBERT vectors.
2. Formulate explicitly derived Clinical Flags (`is_hypotensive`, `is_tachypneic`).
3. Optimize the Binary Threshold dynamically via $F_2$-Score Grid Search on the Validation fold.
4. Output Out-of-Fold QWK Metrics and generate `submission.csv`.

---

## 📊 Interpretability
Because predicting triage requires high accountability, the Notebook inherently outputs **SHAP (SHapley Additive exPlanations)** visualization curves, specifically targeted on isolated cross-validation fragments out-of-fold, demonstrating exactly *why* a patient was accelerated to ESI-1.

---
*Developed for the Kaggle Triagegeist Challenge*
