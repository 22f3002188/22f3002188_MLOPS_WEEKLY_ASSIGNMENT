import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"

POISONING_LEVELS = {
    "0%": 0.00,
    "5%": 0.05,
    "10%": 0.10,
    "50%": 0.50,
}


# ------------------------------------------------------------
# Load clean dataset
# ------------------------------------------------------------

df = pd.read_csv("iris.csv")

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()


# ------------------------------------------------------------
# Create ONE fixed clean train/test split
# ------------------------------------------------------------

train_indices, test_indices = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

clean_test = df.iloc[test_indices].copy()


print("=" * 70)
print("DATASET SPLIT")
print("=" * 70)
print(f"Total samples : {len(df)}")
print(f"Training size : {len(train_indices)}")
print(f"Test size     : {len(test_indices)}")
print()


# ------------------------------------------------------------
# MLflow
# ------------------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "Iris_MLSecOps_Poisoning_v2"
)


# ------------------------------------------------------------
# Function to poison training data
# ------------------------------------------------------------

def poison_training_data(
    training_df,
    poisoning_rate,
    rng,
):
    poisoned_df = training_df.copy()

    n_samples = len(poisoned_df)

    n_poisoned = int(
        n_samples * poisoning_rate
    )

    if n_poisoned == 0:
        return poisoned_df, 0

    poisoned_indices = rng.choice(
        poisoned_df.index,
        size=n_poisoned,
        replace=False,
    )

    # Replace all four features
    for column in FEATURE_COLUMNS:

        min_value = df[column].min()
        max_value = df[column].max()

        poisoned_df.loc[
            poisoned_indices,
            column
        ] = rng.uniform(
            min_value,
            max_value,
            size=n_poisoned,
        )

    # Replace labels with random labels
    classes = df[TARGET_COLUMN].unique()

    poisoned_df.loc[
        poisoned_indices,
        TARGET_COLUMN
    ] = rng.choice(
        classes,
        size=n_poisoned,
    )

    return poisoned_df, n_poisoned


# ------------------------------------------------------------
# Run four experiments
# ------------------------------------------------------------

for poisoning_level, poisoning_rate in POISONING_LEVELS.items():

    print("=" * 70)
    print(f"EXPERIMENT: {poisoning_level} POISONING")
    print("=" * 70)

    # New RNG for reproducibility
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    # Get clean training data
    training_df = df.iloc[
        train_indices
    ].copy()

    # Apply poisoning ONLY to training data
    poisoned_training_df, n_poisoned = (
        poison_training_data(
            training_df,
            poisoning_rate,
            rng,
        )
    )

    X_train = poisoned_training_df[
        FEATURE_COLUMNS
    ]

    y_train = poisoned_training_df[
        TARGET_COLUMN
    ]

    # Test data ALWAYS remains clean
    X_test = clean_test[
        FEATURE_COLUMNS
    ]

    y_test = clean_test[
        TARGET_COLUMN
    ]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DecisionTreeClassifier(
        max_depth=3,
        criterion="gini",
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=f"Poisoning_{poisoning_level}"
    ):

        mlflow.log_param(
            "poisoning_level",
            poisoning_level,
        )

        mlflow.log_param(
            "poisoning_rate",
            poisoning_rate,
        )

        mlflow.log_param(
            "poisoned_samples",
            n_poisoned,
        )

        mlflow.log_param(
            "training_samples",
            len(X_train),
        )

        mlflow.log_param(
            "test_samples",
            len(X_test),
        )

        mlflow.log_param(
            "model",
            "DecisionTreeClassifier",
        )

        mlflow.log_param(
            "max_depth",
            3,
        )

        mlflow.log_param(
            "criterion",
            "gini",
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Poisoned samples : {n_poisoned}"
    )

    print(
        f"Clean test samples : {len(X_test)}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print()


print("=" * 70)
print("ALL MLSecOps EXPERIMENTS COMPLETED")
print("=" * 70)
print(
    "Experiment: Iris_MLSecOps_Poisoning"
)
print(
    "Tracking: sqlite:///mlflow.db"
)
print("=" * 70)