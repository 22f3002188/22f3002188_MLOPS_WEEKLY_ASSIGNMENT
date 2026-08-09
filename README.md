# Week 8 – MLSecOps: Data Poisoning Attack on IRIS

## Overview

This assignment explores **MLSecOps (Machine Learning Security Operations)** and its role in securing the machine learning lifecycle.

The assignment focuses on:

- ML security threat vectors
- Data poisoning attacks
- Poisoning the IRIS training data at different severity levels
- MLflow experiment tracking
- Comparing model performance under data corruption
- Data poisoning detection and mitigation
- Data quantity versus data quality

---

## Objectives

The objectives of this assignment are to:

1. Identify major security threat vectors in ML systems.
2. Simulate data poisoning on the IRIS dataset.
3. Create 5%, 10%, and 50% poisoned variants.
4. Train an IRIS classification model on clean and poisoned data.
5. Track experiments using MLflow.
6. Compare accuracy, precision, recall, and F1 score.
7. Analyze the effect of data poisoning.
8. Discuss production-level MLSecOps mitigation strategies.
9. Understand the relationship between data quality and data quantity.

---

## Repository Structure

```text
22f3002188_MLOPS_WEEKLY_ASSIGNMENT/
│
├── iris.csv
├── iris.csv.dvc
├── iris_poisoned_5.csv
├── iris_poisoned_10.csv
├── iris_poisoned_50.csv
│
├── scripts/
│   ├── poison_data.py
│   └── train_mlflow.py
│
├── train.py
├── requirements.txt
├── mlflow.db
├── mlruns/
├── models/
├── model.joblib
├── tests/
├── .dvc/
├── .github/
├── .gitignore
└── README.md