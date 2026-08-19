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

The sensitive attribute is used only for fairness evaluation.

Script:

```text
scripts/add_location.py