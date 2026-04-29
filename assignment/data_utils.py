"""Shared loaders and preprocessor for the Adult dataset."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income",
]

DROP_COLS = ["fnlwgt", "education-num"]
NUMERIC = ["age", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


def load_split(name: str) -> pd.DataFrame:
    """Load the 'train' (adult.data) or 'test' (adult.test) split.

    Handles the UCI quirks: ', '-separated values, '?' for missing,
    a one-line banner at the top of adult.test, and trailing '.' on
    test-set labels.
    """
    if name == "train":
        path, skip = DATA_DIR / "adult.data", 0
    elif name == "test":
        path, skip = DATA_DIR / "adult.test", 1
    else:
        raise ValueError(f"unknown split: {name}")

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skiprows=skip,
        skipinitialspace=True,
        na_values="?",
    )
    df["income"] = (
        df["income"].str.rstrip(".").map({"<=50K": 0, ">50K": 1}).astype(int)
    )
    return df


def split_xy(df: pd.DataFrame):
    df = df.drop(columns=DROP_COLS)
    y = df["income"].astype(int)
    X = df.drop(columns=["income"])
    return X, y


def build_preprocessor(n_bins: int = 5) -> ColumnTransformer:
    """Discretize numerics (quantile bins, one-hot) and one-hot categoricals.

    Missing categoricals are imputed with the most frequent value;
    numerics are imputed with the median before binning.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("disc", KBinsDiscretizer(
            n_bins=n_bins, encode="onehot", strategy="quantile",
        )),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])
