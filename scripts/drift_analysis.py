import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_FILE = "iris_week9.csv"
OUTPUT_DIR = "drift_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


# --------------------------------------------------
# Load Reference / Training Data
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

reference_data = df[FEATURE_COLUMNS].copy()


# --------------------------------------------------
# Simulate Production Data
# --------------------------------------------------

np.random.seed(42)

production_data = reference_data.copy()

# No meaningful shift in sepal features
production_data["sepal_length"] += np.random.normal(
    loc=0.0,
    scale=0.05,
    size=len(production_data),
)

production_data["sepal_width"] += np.random.normal(
    loc=0.0,
    scale=0.05,
    size=len(production_data),
)

# Intentional distribution shift
production_data["petal_length"] += np.random.normal(
    loc=0.8,
    scale=0.10,
    size=len(production_data),
)

production_data["petal_width"] += np.random.normal(
    loc=0.3,
    scale=0.05,
    size=len(production_data),
)


# --------------------------------------------------
# Save Simulated Production Dataset
# --------------------------------------------------

production_file = os.path.join(
    OUTPUT_DIR,
    "production_data.csv",
)

production_data.to_csv(
    production_file,
    index=False,
)


# --------------------------------------------------
# KS Test
# --------------------------------------------------

results = []

for feature in FEATURE_COLUMNS:

    statistic, p_value = ks_2samp(
        reference_data[feature],
        production_data[feature],
    )

    drift_detected = p_value < 0.05

    results.append(
        {
            "feature": feature,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": drift_detected,
        }
    )


results_df = pd.DataFrame(results)


# --------------------------------------------------
# Save Results
# --------------------------------------------------

results_file = os.path.join(
    OUTPUT_DIR,
    "drift_results.csv",
)

results_df.to_csv(
    results_file,
    index=False,
)


# --------------------------------------------------
# Print Results
# --------------------------------------------------

print("=" * 70)
print("TASK 4 - DATA DRIFT ANALYSIS")
print("=" * 70)

print("\nReference dataset shape:")
print(reference_data.shape)

print("\nProduction dataset shape:")
print(production_data.shape)

print("\nKS Test Results:")
print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("DRIFT INTERPRETATION")
print("=" * 70)

for _, row in results_df.iterrows():

    feature = row["feature"]
    p_value = row["p_value"]

    if row["drift_detected"]:
        print(
            f"{feature}: DRIFT DETECTED "
            f"(p-value = {p_value:.6f})"
        )
    else:
        print(
            f"{feature}: No significant drift "
            f"(p-value = {p_value:.6f})"
        )


# --------------------------------------------------
# Generate Distribution Plots
# --------------------------------------------------

for feature in FEATURE_COLUMNS:

    plt.figure(figsize=(8, 5))

    plt.hist(
        reference_data[feature],
        bins=15,
        alpha=0.6,
        label="Reference / Training",
    )

    plt.hist(
        production_data[feature],
        bins=15,
        alpha=0.6,
        label="Production",
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution Comparison - {feature}")
    plt.legend()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{feature}_distribution.png",
    )

    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nGenerated files:")

print(production_file)
print(results_file)

for feature in FEATURE_COLUMNS:
    print(
        os.path.join(
            OUTPUT_DIR,
            f"{feature}_distribution.png",
        )
    )

print("\n" + "=" * 70)
print("DRIFT ANALYSIS COMPLETED")
print("=" * 70)
