import os
import pandas as pd

# Read CSV
df = pd.read_csv("iris.csv")

# Convert timestamp columns to datetime
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
df["created_timestamp"] = pd.to_datetime(df["created_timestamp"])

# Save to parquet
os.makedirs("feature_repo/feature_repo/data", exist_ok=True)

df.to_parquet(
    "feature_repo/feature_repo/data/iris.parquet",
    index=False,
)

print(df.dtypes)
print("\nParquet created successfully!")