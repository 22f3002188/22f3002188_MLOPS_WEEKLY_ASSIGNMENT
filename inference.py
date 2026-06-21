from google.cloud import storage
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

BUCKET_NAME = "22f3002188-mlops-week1"

timestamp = input("Enter artifact timestamp: ")

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# Download model
bucket.blob(
    f"artifacts/{timestamp}/model.pkl"
).download_to_filename("model.pkl")

# Download evaluation set
bucket.blob(
    f"artifacts/{timestamp}/eval.csv"
).download_to_filename("eval.csv")

# Load model
model = joblib.load("model.pkl")

# Load evaluation data
df = pd.read_csv("eval.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Run inference
predictions = model.predict(X)

# Accuracy
acc = accuracy_score(y, predictions)

print("\nInference Complete")
print("Accuracy:", acc)

print("\nFirst 10 Predictions:")
print(predictions[:10])