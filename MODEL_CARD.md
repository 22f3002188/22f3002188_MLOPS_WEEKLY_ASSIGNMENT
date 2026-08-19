# Model Card — IRIS Classifier

## 1. Model Overview

**Model:** Decision Tree Classifier

**Version:** Week 9

**Purpose:**  
Classify Iris flowers into one of three species:

- Setosa
- Versicolor
- Virginica

The model uses four numerical measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The `location` attribute is used only for fairness evaluation and is
explicitly excluded from model training.

---

## 2. Intended Use

The model is intended for educational and demonstration purposes as
part of an MLOps pipeline.

It demonstrates:

- Classification
- Explainability using SHAP
- Fairness auditing using Fairlearn
- Data drift monitoring
- ML governance

The model should not be used for high-stakes or production decisions
without additional validation and monitoring.

---

## 3. Training Data

The Week 9 classifier was trained using a 101-row Iris dataset.

Class distribution:

| Species | Samples |
|---|---:|
| Setosa | 36 |
| Versicolor | 30 |
| Virginica | 35 |
| **Total** | **101** |

Four numerical features were used for training:

- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`

The sensitive attribute `location` was randomly assigned values of
0 or 1 and was not included as a model feature.

---

## 4. Model Architecture

The classifier is a Decision Tree.

Configuration:

- Criterion: Gini
- Maximum depth: 3
- Random state: 42

---

## 5. Overall Performance

The model was evaluated on a stratified 20% test split.

| Metric | Score |
|---|---:|
| Accuracy | 0.9524 |
| Precision | 0.9583 |
| Recall | 0.9524 |
| F1 Score | 0.9518 |

---

## 6. Performance by Species

| Species | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Setosa | 1.00 | 1.00 | 1.00 |
| Versicolor | 1.00 | 0.83 | 0.91 |
| Virginica | 0.88 | 1.00 | 0.93 |

The model performed very well overall. Versicolor had lower recall
than the other classes, while Virginica had slightly lower precision.

---

## 7. Fairness Evaluation

Fairness was evaluated using Fairlearn's `MetricFrame` with `location`
as the sensitive attribute.

Location was randomly assigned and was not used during model training.

| Metric | Overall | Location 0 | Location 1 |
|---|---:|---:|---:|
| Accuracy | 0.9524 | 0.8750 | 1.0000 |
| Precision | 0.9583 | 0.9167 | 1.0000 |
| Recall | 0.9524 | 0.8750 | 1.0000 |

The observed accuracy difference between the two groups is 0.125.

Because the test set contains only 21 samples, the subgroup results
are sensitive to sampling variation. The observed difference should
therefore not be interpreted as definitive evidence of discrimination.

Fairness metrics should be monitored over larger production samples.

---

## 8. Explainability

SHAP was used to explain model predictions for all three classes.

Generated explanations:

- `shap_summary_setosa.png`
- `shap_summary_versicolor.png`
- `shap_summary_virginica.png`

SHAP values indicate how each feature contributes to a class
prediction.

For the Virginica class:

- Positive SHAP values push the prediction toward Virginica.
- Negative SHAP values push the prediction away from Virginica.
- Red points represent high feature values.
- Blue points represent low feature values.

The SHAP analysis helps identify which Iris measurements most strongly
influence the model's predictions.

---

## 9. Data Drift Monitoring

Production data was simulated by shifting the distributions of
`petal_length` and `petal_width`.

A Kolmogorov-Smirnov test was used to compare the reference and
production distributions.

| Feature | KS Statistic | p-value | Drift |
|---|---:|---:|---|
| Sepal length | 0.0594 | 0.994595 | No |
| Sepal width | 0.0891 | 0.820090 | No |
| Petal length | 0.3366 | 0.000018 | Yes |
| Petal width | 0.3366 | 0.000018 | Yes |

A significance threshold of 0.05 was used.

The results indicate statistically significant data drift in petal
length and petal width.

---

## 10. Limitations

- The dataset contains only 101 samples.
- The dataset is relatively small for estimating subgroup fairness.
- The location attribute was randomly assigned and does not represent
  a real demographic group.
- The production dataset is simulated rather than collected from a
  real production environment.
- Data drift does not necessarily imply model performance degradation.
- Concept drift cannot be assessed without ground-truth production
  labels.
- SHAP explains model behavior but does not guarantee that the model
  is causally correct or fair.

---

## 11. Fairness Considerations

The sensitive `location` attribute was excluded from model training
but retained for fairness auditing.

This demonstrates that excluding a sensitive attribute from training
does not eliminate the need for fairness evaluation.

In a real production system, fairness should be monitored using
representative demographic data, sufficiently large samples, and
appropriate fairness metrics.

---

## 12. Monitoring and Governance

A production deployment should monitor:

1. Feature/data drift
2. Model performance
3. Fairness metrics
4. Prediction distributions
5. Data quality
6. Concept drift when ground-truth labels become available

Model versions, training data, evaluation results, fairness findings,
and known limitations should be documented to support accountability.

---

## 13. Summary

The Week 9 IRIS pipeline demonstrates responsible ML practices by
combining:

- **Fairlearn** for fairness auditing
- **SHAP** for model explainability
- **Statistical tests** for data drift detection
- **Model documentation** for governance

The model performs well on the evaluation dataset, but its small
dataset size and simulated production environment mean that additional
validation would be required before real-world deployment.
