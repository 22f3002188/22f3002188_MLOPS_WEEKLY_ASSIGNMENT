# Week 9 MLOps Assignment — Explainability, Fairness, Drift and Governance

## Overview

This assignment extends the IRIS machine learning pipeline with responsible
ML practices including:

- Sensitive attribute introduction
- Fairness analysis using Fairlearn
- Model explainability using SHAP
- Data drift detection
- ML governance through a Model Card

The work was completed on the `week_9` branch.

---

## Dataset

The Week 9 dataset is based on the IRIS dataset containing 101 samples.

### Class Distribution

| Species | Samples |
|---|---:|
| Setosa | 36 |
| Versicolor | 30 |
| Virginica | 35 |
| **Total** | **101** |

The four model features are:

- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`

A randomly assigned `location` attribute with values `0` and `1` was
introduced for fairness analysis.

The `location` attribute was **not used as a model feature**.

---

# Task 1 — Introduce Location Attribute

A `location` column was randomly assigned to each sample.

Location distribution:

| Location | Samples |
|---|---:|
| 0 | 45 |
| 1 | 56 |


Model Training

A Decision Tree classifier was trained using only the original four Iris
features.

The sensitive location attribute was excluded from training.

Training Configuration
Algorithm: Decision Tree Classifier
Criterion: Gini
Maximum depth: 3
Random state: 42
Train/test split: 80/20
Stratification: Enabled

Training script:

scripts/train_week9.py

Model artifact:

models/model_week9.joblib
Model Performance
Metric	Score
Accuracy	0.9524
Precision	0.9583
Recall	0.9524
F1 Score	0.9518
Classification Report
Class	Precision	Recall	F1 Score
Setosa	1.00	1.00	1.00
Versicolor	1.00	0.83	0.91
Virginica	0.88	1.00	0.93
Task 2 — Fairness Analysis with Fairlearn

Fairlearn's MetricFrame was used to evaluate model performance
separately for location groups 0 and 1.

The following metrics were calculated:

Accuracy
Precision
Recall
Overall Metrics
Metric	Overall
Accuracy	0.9524
Precision	0.9583
Recall	0.9524
Metrics by Location
Metric	Location 0	Location 1
Accuracy	0.8750	1.0000
Precision	0.9167	1.0000
Recall	0.8750	1.0000
Metric Differences
Metric	Difference
Accuracy	0.1250
Precision	0.0833
Recall	0.1250

The location attribute was randomly assigned and was not used as a model
feature.

The observed differences between the two groups can occur because the
test set is relatively small. Therefore, these results should not be
interpreted as definitive evidence of discrimination.

In a production system, fairness metrics should be evaluated using
larger and representative datasets.

Fairness Script
scripts/fairness_analysis.py

Fairlearn version used:

0.14.0
Task 3 — SHAP Explainability

SHAP was used to explain the predictions of the Decision Tree classifier.

A TreeExplainer was used to generate SHAP values for the complete
Week 9 dataset.

Features
sepal_length
sepal_width
petal_length
petal_width
Classes
setosa
versicolor
virginica

The SHAP output has the shape:

(101, 4, 3)

where:

101 = number of samples
4 = number of features
3 = number of classes
Generated SHAP Plots
shap_plots/shap_summary_setosa.png
shap_plots/shap_summary_versicolor.png
shap_plots/shap_summary_virginica.png
Virginica SHAP Interpretation

The Virginica summary plot explains how the four features influence the
model's prediction of the Virginica class.

A positive SHAP value means the feature pushes the prediction toward
Virginica.
A negative SHAP value means the feature pushes the prediction away
from Virginica.
A point on the right side of the plot has a positive contribution.
A point on the left side has a negative contribution.
A red point represents a high value of that feature.
A blue point represents a low value of that feature.

Therefore, a group of red points on the right indicates that high values
of that feature provide evidence toward predicting Virginica.

SHAP Script
scripts/shap_analysis.py

SHAP version used:

0.52.0
Task 4 — Data Drift Detection

A simulated production dataset was created by shifting the distributions
of selected Iris features.

The reference dataset was compared with the simulated production dataset
using the Kolmogorov-Smirnov (KS) statistical test.

The significance threshold was:

p-value < 0.05

A p-value below 0.05 indicates statistically significant distribution
drift.

Drift Results
Feature	KS Statistic	p-value	Drift Detected
Sepal length	0.0594	0.994595	No
Sepal width	0.0891	0.820090	No
Petal length	0.3366	0.000018	Yes
Petal width	0.3366	0.000018	Yes
Interpretation

No significant drift was detected in:

sepal_length
sepal_width

Significant drift was detected in:

petal_length
petal_width

This indicates that the simulated production data has a different
distribution for petal length and petal width compared with the
reference training data.

If such drift occurred in a real production environment, the deployed
model should be monitored because its performance could degrade when
production inputs differ substantially from the training distribution.

Data drift does not automatically mean that model performance has
degraded. Model predictions and performance should also be monitored
when ground-truth labels become available.

Generated Drift Files
drift_results/production_data.csv
drift_results/drift_results.csv
drift_results/sepal_length_distribution.png
drift_results/sepal_width_distribution.png
drift_results/petal_length_distribution.png
drift_results/petal_width_distribution.png
Drift Detection Script
scripts/drift_analysis.py
Task 5 — Model Card

A Model Card was created to document the model and its governance
considerations.

The Model Card contains:

Intended use
Training data description
Model configuration
Overall performance
Performance by species
Fairness results
SHAP explainability information
Drift monitoring results
Limitations
Fairness considerations
Governance recommendations

Model Card:

MODEL_CARD.md
Data Drift vs Concept Drift
Data Drift

Data drift occurs when the distribution of model input features changes
over time.

For example:

Training petal_length distribution
                ↓
Production petal_length distribution changes
                ↓
Data drift

The Week 9 assignment detects data drift using the KS test.

Concept Drift

Concept drift occurs when the relationship between input features and the
target variable changes.

For example, the same feature values may correspond to different target
classes in production compared with training.

Concept drift cannot be reliably detected from input distributions alone.
It requires production predictions to be compared with ground-truth
labels and monitoring of model performance over time.

Explainability vs Interpretability

Interpretability means that the internal logic of a model can be directly
understood.

For example, a simple decision tree can be inspected directly.

Explainability uses a separate method to explain predictions of a model,
including models that may be difficult to interpret directly.

SHAP provides post-hoc explanations by assigning contribution values to
features.

Fairness Considerations

The location attribute was introduced as a sensitive attribute and was
randomly assigned.

It was intentionally excluded from model training.

The purpose of the attribute is to demonstrate fairness auditing rather
than to represent a real demographic characteristic.

Even when a sensitive feature is excluded from training, correlated
features can potentially act as proxy variables. Therefore, fairness
should still be evaluated in production systems.

Model Limitations

The following limitations should be considered:

The dataset contains only 101 samples.
The location attribute was randomly generated and does not represent
a real-world demographic group.
The fairness evaluation uses a relatively small test set.
The production dataset used for drift analysis is simulated.
Data drift does not necessarily imply model performance degradation.
Concept drift cannot be evaluated without production ground-truth
labels.
SHAP explains model behavior but does not establish causal
relationships.
The model should not be used for high-stakes decisions without
additional validation.
Production Monitoring and Governance

A production deployment should monitor:

Data quality
Feature distributions
Data drift
Prediction distributions
Model accuracy
Precision and recall
Fairness metrics
Concept drift
Model version
Training dataset version

Model Cards should be maintained for deployed models to provide
accountability and documentation.

Project Structure
22f3002188_MLOPS_WEEKLY_ASSIGNMENT/
│
├── README.md
├── MODEL_CARD.md
├── iris_week9.csv
├── requirements.txt
│
├── models/
│   └── model_week9.joblib
│
├── scripts/
│   ├── add_location.py
│   ├── train_week9.py
│   ├── fairness_analysis.py
│   ├── shap_analysis.py
│   └── drift_analysis.py
│
├── shap_plots/
│   ├── shap_summary_setosa.png
│   ├── shap_summary_versicolor.png
│   └── shap_summary_virginica.png
│
└── drift_results/
    ├── production_data.csv
    ├── drift_results.csv
    ├── sepal_length_distribution.png
    ├── sepal_width_distribution.png
    ├── petal_length_distribution.png
    └── petal_width_distribution.png
Technologies Used
Python 3
Pandas
NumPy
Scikit-learn
Fairlearn
SHAP
Matplotlib
Joblib
Git
GitHub
DVC
Summary

The Week 9 IRIS MLOps pipeline demonstrates responsible machine learning
practices across the model lifecycle.

Explainability

SHAP was used to understand feature contributions to model predictions.

Fairness

Fairlearn MetricFrame was used to evaluate accuracy, precision, and
recall across location groups.

Drift Monitoring

The Kolmogorov-Smirnov test was used to identify changes in feature
distributions between reference and simulated production data.

Governance

A Model Card was created to document the model's purpose, training data,
performance, fairness results, limitations, and monitoring requirements.

The classifier achieved approximately 95% accuracy on the evaluation
dataset. Fairness analysis showed subgroup performance differences,
while SHAP provided feature-level explanations. Drift analysis identified
significant distribution changes in petal length and petal width in the
simulated production dataset.

These components together demonstrate how explainability, fairness,
monitoring, and governance can be integrated into an MLOps pipeline.



Then save it with:


```bash
nano README.md

Paste everything above, then:

Ctrl + O
Enter
Ctrl + X

Finally run:

git add README.md
git status

Then commit and push:

git commit -m "Week 9: Add explainability fairness drift and governance"
git push origin week_9

The sensitive attribute is used only for fairness evaluation.

Script:

```text
scripts/add_location.py
