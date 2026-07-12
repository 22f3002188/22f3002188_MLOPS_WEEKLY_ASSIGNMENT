import feast
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Connect to Feast
store = feast.FeatureStore(
    repo_path="feature_repo/feature_repo"
)

# Read labels from CSV
labels = pd.read_csv("iris.csv")

# Convert timestamp
labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"])

# Fetch historical features
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

print(training_df.head())

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

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

joblib.dump(model, "models/model.pkl")

with open("metrics/accuracy.txt", "w") as f:
    f.write(f"{accuracy:.4f}")

print("\n==============================")
print("Training Complete")
print("Accuracy:", accuracy)
print("==============================")