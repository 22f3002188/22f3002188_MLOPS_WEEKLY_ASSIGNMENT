import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_FILE = "iris_week9.csv"
MODEL_FILE = "models/model_week9.joblib"
OUTPUT_DIR = "shap_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

feature_columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

X = df[feature_columns]


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# Create SHAP Explainer
# --------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)


print("=" * 60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 60)

print("\nFeatures:")
print(feature_columns)

print("\nClasses:")
print(list(model.classes_))

print("\nSHAP output type:")
print(type(shap_values))

print("\nSHAP output shape:")
print(getattr(shap_values, "shape", "list"))


# --------------------------------------------------
# Generate Summary Plot for Each Class
# --------------------------------------------------

for class_index, class_name in enumerate(model.classes_):

    print("\nGenerating plot for:", class_name)

    plt.figure()

    # SHAP >= 0.50 returns a 3D array:
    # samples x features x classes
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:

        class_shap_values = shap_values[:, :, class_index]

    else:
        # Compatibility with older SHAP versions
        class_shap_values = shap_values[class_index]

    shap.summary_plot(
        class_shap_values,
        X,
        show=False,
    )

    plt.title(f"SHAP Summary Plot - {class_name}")

    output_file = os.path.join(
        OUTPUT_DIR,
        f"shap_summary_{class_name}.png",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", output_file)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED")
print("=" * 60)

print("\nGenerated files:")

for class_name in model.classes_:
    print(
        os.path.join(
            OUTPUT_DIR,
            f"shap_summary_{class_name}.png",
        )
    )
