import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from fairlearn.metrics import MetricFrame


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_FILE = "iris_week9.csv"
MODEL_FILE = "models/model_week9.joblib"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"
SENSITIVE_COLUMN = "location"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

sensitive_features = df[SENSITIVE_COLUMN]


# --------------------------------------------------
# Same Train/Test Split Used During Training
# --------------------------------------------------

X_train, X_test, y_train, y_test, location_train, location_test = train_test_split(
    X,
    y,
    sensitive_features,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# Load Week 9 Model
# --------------------------------------------------

model = joblib.load(MODEL_FILE)

predictions = model.predict(X_test)


# --------------------------------------------------
# Define Metrics
# --------------------------------------------------

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred):
    return precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )


def recall(y_true, y_pred):
    return recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )


metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
}


# --------------------------------------------------
# Fairlearn MetricFrame
# --------------------------------------------------

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=predictions,
    sensitive_features=location_test,
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("=" * 60)
print("TASK 2 - FAIRNESS ANALYSIS")
print("FAIRLEARN METRICFRAME")
print("=" * 60)

print("\nOverall Metrics:")
print(metric_frame.overall)

print("\nMetrics by Location:")
print(metric_frame.by_group)

print("\nMetric Differences:")
print(metric_frame.difference())

print("\nLocation Distribution in Test Set:")
print(location_test.value_counts().sort_index())

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(
    """
Location was randomly assigned and was NOT used as a model feature.

The metrics are evaluated separately for location groups 0 and 1.
Differences between groups can occur because the dataset and test
set are relatively small.

A large persistent performance gap could indicate a fairness concern,
while smaller differences are expected from random sampling.
"""
)
