import feast
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def test_model_accuracy():

    # Connect to Feast
    store = feast.FeatureStore(
        repo_path="feature_repo/feature_repo"
    )

    # Read labels
    labels = pd.read_csv("iris.csv")
    labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"])

    # Get historical features
    training_df = store.get_historical_features(
        entity_df=labels[["iris_id", "event_timestamp"]],
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
        ],
    ).to_df()

    # Add labels
    training_df["species"] = labels["species"]

    X = training_df[
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    ]

    y = training_df["species"]

    # Same split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Load trained model
    model = joblib.load("models/model.pkl")

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel Accuracy: {accuracy:.4f}")

    # Minimum acceptable accuracy
    assert accuracy >= 0.43