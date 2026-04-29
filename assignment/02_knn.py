"""kNN classifier: GridSearchCV on the train split, evaluate on adult.test.

Usage: python 02_knn.py | tee results/02_knn.txt
"""
from pathlib import Path
import json
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from data_utils import build_preprocessor, load_split, split_xy

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> None:
    train = load_split("train")
    test = load_split("test")
    Xtr, ytr = split_xy(train)
    Xte, yte = split_xy(test)

    pipe = Pipeline([
        ("pre", build_preprocessor(n_bins=5)),
        ("clf", KNeighborsClassifier()),
    ])

    grid = {
        "clf__n_neighbors": [5, 11, 21, 31, 51],
        "clf__weights": ["uniform", "distance"],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    # n_jobs=1: avoids forking multiple processes (each would clone the full
    # dataset in memory), which kills WSL under its memory cap.
    gs = GridSearchCV(
        pipe, grid, cv=cv,
        scoring="f1", n_jobs=1, verbose=1, refit=True,
    )

    t0 = time.time()
    gs.fit(Xtr, ytr)
    print(f"\nfit time: {time.time() - t0:.1f}s")
    print(f"best params: {gs.best_params_}")
    print(f"best CV f1:  {gs.best_score_:.4f}")

    cv_df = (
        pd.DataFrame(gs.cv_results_)[[
            "param_clf__n_neighbors", "param_clf__weights",
            "mean_test_score", "std_test_score",
        ]]
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )
    cv_df.to_csv(RESULTS / "knn_cv_scores.csv", index=False)
    print("\n== top 5 CV configurations ==")
    print(cv_df.head().to_string(index=False))

    yhat = gs.predict(Xte)
    metrics = {
        "accuracy":  float(accuracy_score(yte, yhat)),
        "precision": float(precision_score(yte, yhat)),
        "recall":    float(recall_score(yte, yhat)),
        "f1":        float(f1_score(yte, yhat)),
    }
    print("\n== holdout metrics (kNN, on adult.test) ==")
    for k, v in metrics.items():
        print(f"  {k:9s} {v:.4f}")

    cm = confusion_matrix(yte, yhat)
    print("\n== confusion matrix (rows=true, cols=pred; 0=<=50K, 1=>50K) ==")
    print(cm)

    print("\n== classification report ==")
    print(classification_report(
        yte, yhat, target_names=["<=50K", ">50K"], digits=4,
    ))

    out = {
        "best_params": gs.best_params_,
        "best_cv_f1": float(gs.best_score_),
        "holdout": metrics,
        "confusion_matrix": cm.tolist(),
    }
    (RESULTS / "knn_holdout.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
