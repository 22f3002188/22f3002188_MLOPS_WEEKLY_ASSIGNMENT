import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Create output directories
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("iris.csv")

# Features and Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Model
# -----------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "models/model.pkl")

# -----------------------------
# Save Metrics
# -----------------------------
with open("metrics/accuracy.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}")

print("=" * 40)
print("Training Completed Successfully")
print(f"Accuracy : {accuracy:.4f}")
print("Model saved to : models/model.pkl")
print("Metrics saved to : metrics/accuracy.txt")
print("=" * 40)