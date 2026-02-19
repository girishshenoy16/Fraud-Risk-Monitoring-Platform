# 💳 Fraud Detection Analytics (Rule + ML Based)

> An enterprise-style fraud detection system combining statistical rules, anomaly detection, and supervised machine learning to identify high-risk financial transactions and monitor fraud exposure through an interactive dashboard.

---

## 🧠 Problem Statement

Financial institutions process millions of transactions daily.
Fraudulent activity represents a small fraction of transactions — but causes disproportionately high financial losses.

This project builds a **multi-layer fraud detection analytics system** that:

* Detects anomalous transactions using statistical methods (Z-Score)
* Identifies outliers using Isolation Forest
* Predicts fraud probability using supervised ML models
* Categorizes transactions into risk tiers
* Provides executive-level monitoring via Streamlit dashboard

---

## 🏦 Real-World Relevance

This architecture mirrors fraud systems used in:

* Banks
* Fintech startups
* Payment gateways
* Risk analytics teams

It simulates a layered detection pipeline combining:

* Rule-based alerts
* Unsupervised anomaly detection
* Supervised classification models
* Risk scoring and operational dashboards

---

# 📊 Dataset Used

### 🔹 Credit Card Fraud Detection Dataset

* Source: Kaggle
* Transactions: 284,807
* Fraud Cases: 492
* Fraud Rate: ~0.17%
* Highly imbalanced (realistic financial fraud scenario)

### Features

* `Time` — Seconds between transactions
* `Amount` — Transaction value
* `V1 – V28` — PCA-transformed anonymized features
* `Class` — Target variable (0 = Legit, 1 = Fraud)

⚠ Dataset not included in repository due to size.

---

# 📥 Dataset Setup

1️⃣ Download dataset from Kaggle: [DataSet](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

2️⃣ Create folder:

```
data/raw/
```

3️⃣ Place file:

```
data/raw/creditcard.csv
```

---

# 🏗️ System Architecture

```
Raw Data
   ↓
EDA
   ↓
Preprocessing (Scaling + Feature Engineering)
   ↓
--------------------------------
Layer 1: Rule-Based Detection (Z-Score)
Layer 2: Isolation Forest (Anomaly Detection)
Layer 3: Supervised ML (Logistic + XGBoost)
--------------------------------
Risk Scoring Engine
   ↓
Severity Classification
   ↓
Enterprise Monitoring Dashboard
```

---

# 🛠️ Tech Stack

| Layer             | Tools                        |
| ----------------- | ---------------------------- |
| Data Processing   | Pandas, NumPy                |
| Statistical Rules | Z-Score                      |
| Unsupervised ML   | Isolation Forest             |
| Supervised ML     | Logistic Regression, XGBoost |
| Explainability    | SHAP                         |
| Visualization     | Matplotlib                   |
| Dashboard         | Streamlit                    |
| Model Persistence | Joblib                       |

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/girishshenoy16/Fraud-Risk-Monitoring-Platform.git
cd Fraud-Detection-Analytics
```

---

## 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🚀 Running the Project

## Step 1 — Run Full ML Pipeline

```bash
python src/main.py
```

This will:

* Perform EDA
* Preprocess data
* Engineer features
* Train models
* Tune hyperparameters
* Generate evaluation plots
* Save trained models

---

## Step 2 — Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard opens in browser.

---

# 📊 Dashboard Modules

### 📌 Overview

* Total transactions
* Fraud detected
* Detection rate
* Threshold used

### 📌 Model Performance

* ROC Curve
* Precision-Recall Curve
* Confusion Matrix

### 📌 Feature Insights

* Top fraud-driving features

### 📌 Transaction Explorer

* Filter by Risk Category
* Color-coded severity
* Clean financial formatting

### 📌 Explainability

* SHAP transaction-level breakdown
* Risk score interpretation
* Action recommendation

### 📌 Fraud Trend Monitoring

* Fraud trend by hour

### 📌 Financial Exposure

* Estimated blocked fraud amount

### 📌 Executive Summary

* Business-ready fraud insights

---

# 🚨 Risk Classification Logic

| Risk Score  | Category    | Action            |
| ----------- | ----------- | ----------------- |
| ≥ 0.75      | High Risk   | Block transaction |
| 0.40 – 0.74 | Medium Risk | Manual review     |
| < 0.40      | Low Risk    | Approve           |

Fraud threshold optimized for high recall in imbalanced environment.

---

# 📂 Project Structure

```
Fraud-Detection-Analytics/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── hyperparameter_tuning.py
│   ├── evaluation.py
│   ├── feature_importance.py
│   ├── shap_explain.py
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# 📈 Key Results

* ROC-AUC ≈ 0.97+
* Strong fraud recall in imbalanced dataset
* Clear SHAP explainability
* Enterprise-grade risk monitoring dashboard

---

# 🎯 Resume Highlights

* Built multi-layer fraud detection system (Rule + Isolation Forest + XGBoost)
* Achieved high ROC-AUC on extreme class imbalance dataset
* Implemented SHAP for transaction-level interpretability
* Designed enterprise monitoring dashboard using Streamlit
* Applied hyperparameter tuning and risk threshold optimization

---

# 🏆 Why This Project Stands Out

✔ Combines statistical + unsupervised + supervised detection

✔ Handles extreme imbalance properly

✔ Includes explainable AI

✔ Production-style pipeline

✔ Executive-level dashboard