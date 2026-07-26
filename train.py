import os
import feast
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# Connect to Feast Feature Store
# --------------------------------------------------

store = feast.FeatureStore(
    repo_path="feature_repo/feature_repo"
)

# --------------------------------------------------
# Load Labels
# --------------------------------------------------

labels = pd.read_csv("iris.csv")
labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"])

# --------------------------------------------------
# Fetch Historical Features from Feast
# --------------------------------------------------

training_df = store.get_historical_features(
    entity_df=labels[["iris_id", "event_timestamp"]],
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
).to_df()

training_df["species"] = labels["species"]

print("\nTraining Data Preview")
print(training_df.head())

# --------------------------------------------------
# Prepare Features and Labels
# --------------------------------------------------

X = training_df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

y = training_df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# Configure MLflow
# --------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Classification")

# --------------------------------------------------
# Hyperparameter Search
# --------------------------------------------------

max_depth_values = [2, 3, 5]
criterion_values = ["gini", "entropy"]

best_accuracy = 0.0
best_model = None

for depth in max_depth_values:

    for criterion in criterion_values:

        with mlflow.start_run():

            print("=" * 60)
            print("Training Model")
            print(f"max_depth = {depth}")
            print(f"criterion = {criterion}")

            # Build Model
            model = DecisionTreeClassifier(
                max_depth=depth,
                criterion=criterion,
                random_state=42,
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
            )

            # Log Parameters
            mlflow.log_param("max_depth", depth)
            mlflow.log_param("criterion", criterion)

            # Log Metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log Model to MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="IrisClassifier",
            )

            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")

            # Keep Best Model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

# --------------------------------------------------
# Save Best Model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/model.joblib")
joblib.dump(best_model, "model.joblib")

print("\nBest model saved successfully!")
print("Saved to:")
print("  models/model.joblib")
print("  model.joblib")

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("Training Completed Successfully")
print(f"Best Accuracy : {best_accuracy:.4f}")
print("MLflow Tracking : sqlite:///mlflow.db")
print("=" * 60)