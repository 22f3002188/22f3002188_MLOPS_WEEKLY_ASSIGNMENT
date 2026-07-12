import pandas as pd


def test_dataset_exists():
    """Check dataset is not empty"""
    df = pd.read_csv("iris.csv")
    assert len(df) > 0


def test_no_missing_values():
    """Check for missing values"""
    df = pd.read_csv("iris.csv")
    assert df.isnull().sum().sum() == 0


def test_expected_columns():
    """Check required columns exist"""

    df = pd.read_csv("iris.csv")

    expected_columns = [
        "event_timestamp",
        "iris_id",
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
        "created_timestamp",
    ]

    assert list(df.columns) == expected_columns


def test_numeric_features():
    """Check feature columns are numeric"""

    df = pd.read_csv("iris.csv")

    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    for col in feature_columns:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_feature_ranges():
    """Check feature values are within reasonable ranges"""

    df = pd.read_csv("iris.csv")

    assert df["sepal_length"].between(4.5, 6.0).all()
    assert df["sepal_width"].between(2.0, 4.0).all()
    assert df["petal_length"].between(1.0, 4.5).all()
    assert df["petal_width"].between(-0.2, 1.5).all()