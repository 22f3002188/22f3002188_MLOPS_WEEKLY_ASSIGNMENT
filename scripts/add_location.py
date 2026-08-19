import pandas as pd
import numpy as np

INPUT_FILE = "/tmp/iris_v1.csv"
OUTPUT_FILE = "iris_week9.csv"

# Reproducible random assignment
np.random.seed(42)

# Load the three-class Iris dataset
df = pd.read_csv(INPUT_FILE)

# Randomly assign location group 0 or 1
df["location"] = np.random.randint(0, 2, size=len(df))

# Save Week 9 dataset
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("WEEK 9 DATASET CREATED")
print("=" * 60)

print(f"Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["species"].value_counts())

print("\nLocation distribution:")
print(df["location"].value_counts().sort_index())

print("\nClass distribution by location:")
print(pd.crosstab(df["location"], df["species"]))

print("\nFirst 5 rows:")
print(df.head())
