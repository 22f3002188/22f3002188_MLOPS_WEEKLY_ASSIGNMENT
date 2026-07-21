import feast
import pandas as pd
import mlflow
import mlflow.pyfunc

# --------------------------------------------------
# Configure MLflow
# --------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load latest registered model from MLflow Registry
model = mlflow.pyfunc.load_model(
    model_uri="models:/IrisClassifier/latest"
)

# --------------------------------------------------
# Connect to Feast
# --------------------------------------------------

store = feast.FeatureStore(
    repo_path="feature_repo/feature_repo"
)

# --------------------------------------------------
# Fetch Online Features
# --------------------------------------------------

online_features = store.get_online_features(
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
    entity_rows=[
        {"iris_id": 1001},
        {"iris_id": 1002},
        {"iris_id": 1003},
    ],
).to_dict()

df = pd.DataFrame(online_features)

print("\nFeatures fetched from Feast Online Store:\n")
print(df)

# --------------------------------------------------
# Prepare Features
# --------------------------------------------------

X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

# --------------------------------------------------
# Predict
# --------------------------------------------------

predictions = model.predict(X)

df["prediction"] = predictions

print("\nPredictions\n")
print(df)