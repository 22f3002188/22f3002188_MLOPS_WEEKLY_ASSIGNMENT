import feast
import pandas as pd
import joblib

# Connect to Feast
store = feast.FeatureStore(
    repo_path="feature_repo/feature_repo"
)

# Load trained model
model = joblib.load("models/model.pkl")

# Retrieve online features
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

# Convert to DataFrame
df = pd.DataFrame(online_features)

print("Features fetched from Feast Online Store:\n")
print(df)

# Prepare features
X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

# Predict
predictions = model.predict(X)

df["prediction"] = predictions

print("\nPredictions:\n")
print(df)