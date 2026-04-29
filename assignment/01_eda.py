"""Exploratory analysis: schema, missing values, class balance, distributions.

Usage: python 01_eda.py | tee results/01_eda.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import CATEGORICAL, NUMERIC, load_split

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def main() -> None:
    train = load_split("train")
    test = load_split("test")

    print(f"train shape: {train.shape}")
    print(f"test shape:  {test.shape}")

    print("\n== dtypes ==")
    print(train.dtypes.to_string())

    print("\n== class balance (train) ==")
    bal = train["income"].value_counts(normalize=True).rename(
        {0: "<=50K", 1: ">50K"}
    )
    print(bal.to_string())

    print("\n== missing values per column (train) ==")
    miss = train.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print(miss.to_string() if len(miss) else "  none")
    any_missing = train.isna().any(axis=1)
    print(
        f"rows with any missing: {any_missing.sum()} "
        f"({any_missing.mean():.2%})"
    )

    print("\n== numeric summary (train) ==")
    print(train[NUMERIC].describe().round(2).to_string())

    print("\n== categorical cardinality (train) ==")
    for c in CATEGORICAL:
        print(f"  {c:18s} {train[c].nunique():3d} unique")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.ravel(), NUMERIC):
        train[col].plot.hist(bins=40, ax=ax)
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(FIGURES / "numeric_hist.png", dpi=120)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    train["income"].value_counts().rename(
        {0: "<=50K", 1: ">50K"}
    ).plot.bar(ax=ax)
    ax.set_title("Income class balance (train)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGURES / "class_balance.png", dpi=120)

    print(f"\nsaved figures to {FIGURES}")


if __name__ == "__main__":
    main()
