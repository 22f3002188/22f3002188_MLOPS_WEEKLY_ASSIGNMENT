from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from datetime import datetime
import pandas as pd
import joblib
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--version", required=True)
args = parser.parse_args()

VERSION = args.version

BUCKET_NAME = "22f3002188-mlops-week1"

if VERSION == "v1":
    DATA_FILE = "data/v1/data.csv"
elif VERSION == "v2":
    DATA_FILE = "data/v2/data (1).csv"
else:
    raise Exception("Use v1 or v2")

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

bucket.blob(DATA_FILE).download_to_filename("dataset.csv")

df = pd.read_csv("dataset.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

artifact_dir = f"artifacts/{VERSION}_{timestamp}"

os.makedirs("tmp", exist_ok=True)

joblib.dump(model, "tmp/model.pkl")

with open("tmp/metrics.json", "w") as f:
    json.dump({"accuracy": float(acc)}, f)

for file in ["model.pkl", "metrics.json"]:
    bucket.blob(
        f"{artifact_dir}/{file}"
    ).upload_from_filename(
        f"tmp/{file}"
    )

print("Version:", VERSION)
print("Accuracy:", acc)
print("Artifacts:", artifact_dir)