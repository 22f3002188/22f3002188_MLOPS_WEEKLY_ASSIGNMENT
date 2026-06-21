from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from datetime import datetime
import pandas as pd
import joblib
import json
import os

BUCKET_NAME = "22f3002188-mlops-week1"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# Download dataset
bucket.blob("data/iris.csv").download_to_filename("iris.csv")

# Load data
df = pd.read_csv("iris.csv")

# Last column = target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

# Timestamp folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs("artifacts", exist_ok=True)

# Save model
joblib.dump(model, "artifacts/model.pkl")

# Save metrics
with open("artifacts/metrics.json", "w") as f:
    json.dump({"accuracy": float(acc)}, f)

# Save evaluation data
eval_df = X_test.copy()
eval_df["target"] = y_test

eval_df.to_csv(
    "artifacts/eval.csv",
    index=False
)

# Upload artifacts
for file in [
    "model.pkl",
    "metrics.json",
    "eval.csv"
]:
    bucket.blob(
        f"artifacts/{timestamp}/{file}"
    ).upload_from_filename(
        f"artifacts/{file}"
    )

print("\nTraining Completed")
print("Accuracy:", acc)
print("Timestamp:", timestamp)