# EdgeSynth: Synthetic Data Generation for Edge AI

EdgeSynth is an end-to-end framework for generating, balancing, and deploying synthetic tabular data models optimized for **edge environments**.  
It addresses key challenges such as **data scarcity**, **class imbalance**, and **resource-constrained inference** by integrating **CTGAN** for synthetic data generation, **SMOTENC** for balancing, and **ONNX** for lightweight edge deployment.

---

## **Project Overview**
- Generate realistic **synthetic tabular data** using Conditional Tabular GAN (CTGAN).
- Balance the dataset using **SMOTENC** to handle minority-class underrepresentation.
- Train and evaluate **Logistic Regression** and **XGBoost** classifiers using **TSTR** (Train on Synthetic, Test on Real).
- Export models to **ONNX** format for optimized **edge inference**.
- Benchmark inference latency and model size for deployment readiness.

---

## **Project Structure**
```
edge_synth_submission/
├── main.py                     # Entry point for the full pipeline
├── ctgan_model.py             # CTGAN model training
├── convert_to_csv.py          # Converts raw dataset to clean CSV
├── check_cuda.py              # Checks GPU availability
├── analysis/                  # Evaluation and visualization
│   ├── logistic_regression_classifier.py
│   ├── xg_boost_classifier.py
│   ├── heatmap.py
|   |-- data_distribution_comparison.py
│   ├── ks_test.py
│   ├── compare_model_size.py
├── optimization/             # ONNX conversion & inference
│   ├── export_logistic_to_onnx.py
│   ├── export_xgboost_to_onnx.py
│   ├── xgboost_onnx_inference.py
├── data/
│   ├── raw/                  # Original dataset
│   │   ├── adult.csv
│   ├── synthetic/            # Generated synthetic datasets
│   │   ├── synthetic_adult.csv
│   │   ├── smote_synthetic_adult.csv
├── models/                   # Saved models & encoders
│   ├── ctgan_model.pkl
│   ├── logistic_model.joblib
│   ├── logistic_model.onnx
│   ├── xgboost_model.joblib
│   ├── xgboost_model.onnx
│   ├── xgb_label_encoder.joblib
└── README.md                # Project documentation
```

---

## **Setup Instructions**

### **1. Clone or Download**
```bash
git clone https://github.com/your-repo/edge_synth_submission.git
cd edge_synth_submission
```

### **2. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate    # On Mac/Linux
venv\Scripts\activate     # On Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **Step 1 — Train CTGAN**
```bash
python ctgan_model.py and main.py
```

### **Step 2 — Balance Classes and Train Classifiers**
```bash
# Generates SMOTENC-balanced dataset and trains the classifiers
python analysis/logistic_regression_classifier.py
python analysis/xg_boost_classifier.py
```

### **Step 3 — Export Models to ONNX**
```bash
python optimization/export_logistic_to_onnx.py
python optimization/export_xgboost_to_onnx.py
```

### **Step 4 — Simulate Edge Inference**
```bash
python optimization/xgboost_onnx_inference.py
```

---

## **Results Summary**
| Model              | Accuracy | Minority Recall | Model Size | Inference Latency |
|--------------------|---------|-----------------|------------|--------------------|
| Logistic Regression | 71%     | 78%             | 1 KB       | 0.029s             |
| XGBoost            | 71%     | 76%             | 170 KB     | 0.070s             |

---

## **Dataset**
The project uses the **UCI Adult Income dataset**, publicly available at:  
[https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)

---

## **License**
This project is for **academic and research purposes** only.  
© 2025 Nauman Haroon, Ulster University.
