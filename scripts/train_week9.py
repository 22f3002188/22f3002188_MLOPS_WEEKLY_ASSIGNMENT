import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


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


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("WEEK 9 IRIS CLASSIFIER")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df[TARGET_COLUMN].value_counts())


# --------------------------------------------------
# Prepare Features
# --------------------------------------------------

# IMPORTANT:
# location is deliberately excluded.
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# --------------------------------------------------
# Train Decision Tree
# --------------------------------------------------

model = DecisionTreeClassifier(
    max_depth=3,
    criterion="gini",
    random_state=42,
)

model.fit(X_train, y_train)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# --------------------------------------------------
# Verify Classes
# --------------------------------------------------

print("Model classes:")
print(model.classes_)


# --------------------------------------------------
# Save Model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("\nModel saved successfully:")
print(MODEL_FILE)

print("\n" + "=" * 60)
print("WEEK 9 TRAINING COMPLETED")
print("=" * 60)
