# IRIS ML Pipeline on Vertex AI

## Student Information

* IITM BS Roll Number: 22f3002188
* Course: MLOps
* Assignment: Week 1 – IRIS ML Pipeline on Vertex AI

---

## Project Objective

The objective of this assignment is to build an end-to-end machine learning pipeline on Google Cloud Platform (GCP) using Vertex AI Workbench and Google Cloud Storage (GCS).

The pipeline performs the following tasks:

1. Fetches the IRIS dataset from a GCS bucket.
2. Trains a machine learning model for IRIS classification.
3. Stores training artifacts in GCS using timestamp-based folder organization.
4. Loads the trained model from GCS and performs inference on evaluation data.
5. Executes the pipeline multiple times to generate separate artifact folders.
6. (Optional) Trains and compares results using multiple dataset versions.

---

## Repository Structure

```text
.
├── train.py
├── inference.py
├── train_versioned.py
├── README.md
└── screenshots/
```

---

## File Description

### train.py

Downloads the dataset from Google Cloud Storage, performs train-test splitting, trains a Decision Tree classifier, evaluates model performance, and uploads generated artifacts to GCS.

Generated artifacts:

* model.pkl
* metrics.json
* eval.csv

Artifacts are stored under timestamp-based folders in the GCS bucket.

---

### inference.py

Downloads the trained model and evaluation dataset from GCS and performs inference on the evaluation set.

Outputs:

* Predictions
* Accuracy score

---

### train_versioned.py

(Optional Task 6)

Runs the training pipeline on different dataset versions (v1 and v2) and stores results in separate artifact folders for comparison.

---

### reference_notebook.ipynb

Starter notebook provided as part of the assignment resources.

---

### requirements.txt

Contains all required Python dependencies.

---

## Google Cloud Storage Structure

```text
gs://22f3002188-mlops-week1/

data/
│
├── iris.csv
├── v1/
└── v2/

artifacts/
│
├── <timestamp_1>/
└── <timestamp_2>/
```

---

## Tasks Completed

### Task 1

* Activated Google Cloud account
* Created Vertex AI Workbench instance
* Enabled required APIs

### Task 2

* Created Google Cloud Storage bucket
* Uploaded IRIS dataset to GCS

### Task 3

* Downloaded data from GCS
* Trained IRIS classification model
* Stored training artifacts in timestamp-based folders

### Task 4

* Created separate inference pipeline
* Loaded trained model from GCS
* Performed inference on evaluation dataset

### Task 5

* Executed training and inference pipeline twice
* Generated multiple timestamp-based artifact folders

### Task 6 (Optional)

* Executed training on multiple dataset versions
* Compared model performance across versions

---

## Technologies Used

* Python
* Google Cloud Storage (GCS)
* Vertex AI Workbench
* Scikit-learn
* Pandas
* Joblib

---

## Notes

The trained model files and generated artifacts are intentionally not included in this repository, as per assignment guidelines. All execution outputs are stored in Google Cloud Storage and demonstrated through execution screenshots.
