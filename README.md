# IRIS ML Pipeline with DVC and Google Cloud Storage

**Course:** MLOps Weekly Assignment - Week 2
**Student Name:** Harsh Jayswal
**IITM BS Roll Number:** 22f3002188

---

# Project Overview

This project demonstrates the integration of **Data Version Control (DVC)** into an end-to-end Machine Learning pipeline for the **IRIS Classification** dataset.

The objective is to build a reproducible ML workflow by combining:

* Git for source code versioning
* DVC for dataset and model versioning
* Google Cloud Storage (GCS) as the DVC remote storage
* Vertex AI Workbench as the development environment

The project tracks multiple versions of the dataset and corresponding trained models while maintaining a lightweight Git repository.

---

# Objectives

* Initialize DVC in an existing Git repository.
* Configure Google Cloud Storage as the default DVC remote.
* Version datasets and trained models.
* Simulate multiple data iterations.
* Switch between historical versions using Git and DVC.
* Maintain a reproducible ML workflow.

---

# Technologies Used

* Python 3.12
* DVC
* Git & GitHub
* Google Cloud Storage (GCS)
* Google Vertex AI Workbench
* Pandas
* Scikit-learn
* Joblib

---

# Repository Structure

```text
22f3002188_MLOPS_WEEKLY_ASSIGNMENT
│
├── .dvc/                  # DVC configuration
├── .dvcignore             # DVC ignore rules
├── .gitignore             # Git ignore rules
├── README.md
├── train.py               # Model training script
├── iris.csv.dvc           # DVC pointer for dataset
├── models.dvc             # DVC pointer for trained model
├── metrics/
│   └── accuracy.txt       # Model accuracy
```

---

# DVC Remote

Remote Storage:

Google Cloud Storage (GCS)

```
gs://22f3002188-mlops-week2
```

The actual datasets and trained model artifacts are stored in Google Cloud Storage, while Git stores only lightweight `.dvc` pointer files.

---

# Workflow

## 1. Initialize DVC

```bash
dvc init
```

---

## 2. Configure GCS Remote

```bash
dvc remote add -d gcsremote gs://22f3002188-mlops-week2
```

---

## 3. Track Dataset

```bash
dvc add iris.csv
dvc push
```

---

## 4. Train Model

```bash
python train.py
```

The training script:

* Loads the IRIS dataset
* Splits the data into train/test sets
* Trains a Decision Tree Classifier
* Saves the trained model
* Stores model accuracy

---

## 5. Track Model

```bash
dvc add models
dvc push
```

---

## 6. Commit Changes

```bash
git add .
git commit -m "Version update"
git push
```

---

# Dataset Versions

Two different dataset versions were used:

* Version 1
* Version 2

Each dataset version has its own corresponding trained model tracked by DVC.

---

# Switching Between Versions

To restore an older version:

```bash
git checkout <commit_hash>
dvc checkout
```

To return to the latest version:

```bash
git checkout week_2
dvc checkout
```

This restores both the dataset and trained model associated with the selected Git commit.

---

# Training Output

The training process generates:

* Trained model (`models/`)
* Accuracy report (`metrics/accuracy.txt`)

These outputs are version-controlled using DVC.

---

# Files Included

* `train.py` – Training pipeline
* `.dvc/` – DVC configuration
* `iris.csv.dvc` – Dataset pointer
* `models.dvc` – Model pointer
* `.gitignore`
* `.dvcignore`
* `README.md`

---

# Files Excluded

The following files are intentionally **not stored in Git**:

* Original dataset
* Trained model files
* DVC cache
* Large binary artifacts

These files are stored in the configured Google Cloud Storage remote through DVC.

---

# Reproducing the Project

Clone the repository:

```bash
git clone <repository_url>
cd 22f3002188_MLOPS_WEEKLY_ASSIGNMENT
```

Pull data from the DVC remote:

```bash
dvc pull
```

Train the model:

```bash
python train.py
```

---

# Learning Outcomes

This assignment helped in understanding:

* Data Version Control (DVC)
* Versioning datasets and models
* Google Cloud Storage integration
* Reproducible Machine Learning workflows
* Managing multiple data/model versions using Git and DVC

---

# Author

**Harsh Jayswal**

IIT Madras BS Degree Programme

Roll Number: **22f3002188**
