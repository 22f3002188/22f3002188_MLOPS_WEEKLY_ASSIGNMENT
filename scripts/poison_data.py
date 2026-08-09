import argparse

import numpy as np
import pandas as pd


RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"


def poison_dataset(input_file, output_file, poisoning_rate):
    df = pd.read_csv(input_file)

    # Validate columns
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    poisoned_df = df.copy()

    n_samples = len(poisoned_df)
    n_poisoned = int(n_samples * poisoning_rate)

    # Reproducible random generator
    rng = np.random.default_rng(RANDOM_SEED)

    # Select rows to poison
    poisoned_indices = rng.choice(
        n_samples,
        size=n_poisoned,
        replace=False
    )

    # Replace all four features with random values
    # within the original feature ranges.
    for column in FEATURE_COLUMNS:
        min_value = df[column].min()
        max_value = df[column].max()

        poisoned_df.loc[
            poisoned_indices, column
        ] = rng.uniform(
            min_value,
            max_value,
            size=n_poisoned
        )

    # Replace labels with random class labels
    classes = df[TARGET_COLUMN].unique()

    poisoned_df.loc[
        poisoned_indices, TARGET_COLUMN
    ] = rng.choice(
        classes,
        size=n_poisoned
    )

    # Save poisoned dataset
    poisoned_df.to_csv(
        output_file,
        index=False
    )

    print("=" * 50)
    print("DATA POISONING COMPLETE")
    print("=" * 50)
    print(f"Input dataset    : {input_file}")
    print(f"Output dataset   : {output_file}")
    print(f"Poisoning rate   : {poisoning_rate * 100:.0f}%")
    print(f"Total samples    : {n_samples}")
    print(f"Poisoned samples : {n_poisoned}")
    print(f"Clean samples    : {n_samples - n_poisoned}")
    print(f"Random seed      : {RANDOM_SEED}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate data poisoning on the IRIS dataset."
    )

    parser.add_argument(
        "--input",
        default="iris.csv",
        help="Clean IRIS dataset"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output poisoned dataset"
    )

    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Poisoning rate between 0 and 1"
    )

    args = parser.parse_args()

    if not 0 <= args.rate <= 1:
        raise ValueError(
            "Poisoning rate must be between 0 and 1."
        )

    poison_dataset(
        args.input,
        args.output,
        args.rate
    )
