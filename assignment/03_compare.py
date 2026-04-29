"""Compare kNN against Logistic Regression and Random Forest on the holdout.

Reuses the best kNN hyperparameters from 02_knn.py if available.

Usage: python 03_compare.py | tee results/03_compare.txt
"""
from pathlib import Path
import json
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from data_utils import build_preprocessor, load_split, split_xy

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = [
            f"{v:.4f}" if isinstance(v, float) else str(v) for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    train = load_split("train")
    test = load_split("test")
    Xtr, ytr = split_xy(train)
    Xte, yte = split_xy(test)

    knn_params = {"n_neighbors": 31, "weights": "distance"}
    bp_path = RESULTS / "knn_holdout.json"
    if bp_path.exists():
        d = json.loads(bp_path.read_text())
        knn_params = {
            k.replace("clf__", ""): v for k, v in d["best_params"].items()
        }
        print(f"loaded best kNN params from 02_knn.py: {knn_params}")
    else:
        print(f"WARN: {bp_path} not found, using fallback {knn_params}")

    models = {
        "kNN": KNeighborsClassifier(n_jobs=-1, **knn_params),
        "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=-1),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1,
        ),
    }

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([
            ("pre", build_preprocessor(n_bins=5)),
            ("clf", clf),
        ])
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        fit_t = time.time() - t0
        yhat = pipe.predict(Xte)
        rows.append({
            "model": name,
            "accuracy":  float(accuracy_score(yte, yhat)),
            "precision": float(precision_score(yte, yhat)),
            "recall":    float(recall_score(yte, yhat)),
            "f1":        float(f1_score(yte, yhat)),
            "fit_seconds": round(fit_t, 1),
        })
        print(
            f"{name:20s} fit {fit_t:6.1f}s  "
            f"acc={rows[-1]['accuracy']:.4f}  "
            f"prec={rows[-1]['precision']:.4f}  "
            f"rec={rows[-1]['recall']:.4f}  "
            f"f1={rows[-1]['f1']:.4f}"
        )

    df = pd.DataFrame(rows)
    print("\n== holdout comparison ==")
    print(df.to_string(index=False))
    df.to_csv(RESULTS / "comparison.csv", index=False)
    write_markdown_table(df, RESULTS / "comparison.md")
    print(f"\nsaved {RESULTS / 'comparison.csv'} and {RESULTS / 'comparison.md'}")


if __name__ == "__main__":
    main()
