# Week 3: Integrating Feast Feature Store into IRIS Pipeline

## Student Information

- **Name:** Harsh Jayswal
- **IITM BS Roll Number:** 22f3002188
- **Course:** MLOps
- **Week:** 3
- **Assignment:** Integrating Feast Feature Store into the IRIS Pipeline

---

# Project Overview

This assignment demonstrates how to integrate the **Feast Feature Store** into a Machine Learning pipeline using the IRIS dataset.

The objective is to separate feature engineering from model training so that both **training** and **inference** use the same feature definitions, preventing training-serving skew.

The complete workflow was executed on **Google Cloud Platform (GCP)**.

---

# Objectives

- Initialize a Feast Feature Repository
- Define Entity, Data Source and Feature View
- Materialize features into the Online Store
- Retrieve Historical Features for Training
- Retrieve Online Features for Inference
- Train an IRIS Classification Model using Feast

---

# Project Structure

```
22f3002188_MLOPS_WEEKLY_ASSIGNMENT
│
├── feature_repo/
│   └── feature_repo/
│       ├── feature_store.yaml
│       ├── feature_definitions.py
│       └── data/
│           ├── iris.parquet
│           ├── registry.db
│           └── online_store.db
│
├── iris.csv
├── prepare_feast_data.py
├── train.py
├── inference.py
├── metrics/
│   └── accuracy.txt
├── README.md
└── .gitignore
```

---

# Technologies Used

- Python
- Feast 0.64
- Pandas
- Scikit-learn
- SQLite
- Google Cloud Platform (GCP)

---

# Feature Store Architecture

```
Raw Dataset
      │
      ▼
prepare_feast_data.py
      │
      ▼
iris.parquet
      │
      ▼
Feast Feature Store
      │
 ┌──────────────┐
 │ Offline Store│
 └──────────────┘
      │
      ▼
Historical Features
      │
      ▼
Model Training
      │
      ▼
Trained Model

-----------------------------

Online Store
      │
      ▼
Feature Retrieval
      │
      ▼
Real-time Inference
```

---

# Files Description

## prepare_feast_data.py

- Reads the IRIS dataset
- Creates Entity IDs
- Adds timestamps required by Feast
- Converts CSV to Parquet format

---

## feature_definitions.py

Defines:

- Entity
- File Source
- Feature View

The following features are registered:

- sepal_length
- sepal_width
- petal_length
- petal_width

---

## train.py

This script:

- Connects to Feast
- Retrieves Historical Features using the Offline Store
- Trains a Decision Tree Classifier
- Saves the trained model
- Stores model accuracy

---

## inference.py

This script:

- Loads the trained model
- Retrieves Online Features from Feast
- Performs predictions
- Displays inference results

---

# Feast Components

## Entity

```
iris
```

Unique identifier used for retrieving feature values.

---

## Data Source

```
iris.parquet
```

Stores historical feature values.

---

## Feature View

```
iris_features
```

Contains the following features:

- sepal_length
- sepal_width
- petal_length
- petal_width

---

# Execution Steps

## Step 1

Prepare Feast Dataset

```
python prepare_feast_data.py
```

---

## Step 2

Register Feature Definitions

```
cd feature_repo/feature_repo

feast apply
```

---

## Step 3

Materialize Features

```
feast materialize \
2025-09-01T00:00:00 \
2030-01-01T00:00:00
```

---

## Step 4

Train Model

```
python train.py
```

---

## Step 5

Run Inference

```
python inference.py
```

---

# Sample Training Output

```
Training Complete

Accuracy : 0.4444
```

---

# Sample Inference Output

```
Features fetched from Feast Online Store

Prediction

setosa
setosa
setosa
```

---

# Learning Outcomes

Through this assignment, I learned:

- What a Feature Store is
- Difference between Offline and Online Stores
- How Feast prevents Training-Serving Skew
- Creating Entities and Feature Views
- Materializing features into the Online Store
- Fetching Historical Features for Training
- Fetching Online Features for Real-time Inference
- Integrating Feast with an ML Pipeline

---

# Conclusion

This project successfully integrates the Feast Feature Store into an IRIS Machine Learning pipeline. Historical features are retrieved from the Offline Store for model training, while Online Features are served for real-time inference, ensuring consistency between training and production environments.
