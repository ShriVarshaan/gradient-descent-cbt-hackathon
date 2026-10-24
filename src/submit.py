"""Submitted predictions: LightGBM + XGBoost on all 164 features (81 pipeline + 83 history).

Usage: python src/submit.py <train_dir> <test_dir> <submission.csv> [--no-cv]

Each model is trained on all of train/ and the score is the mean of the two predicted
probabilities. Hyperparameters are the ones in ensemble.py. Time-CV mean AP: 0.408.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from cv import time_folds
from ensemble import models
from features import ALL_FEATURES, build_features
from prepare import load_tables

MODELS = ["lgb", "xgb"]


def fit_predict(train, test):
    preds = {}
    for name in MODELS:
        model = models()[name]
        model.fit(train[ALL_FEATURES], train["label"])
        preds[name] = model.predict_proba(test[ALL_FEATURES])[:, 1]
    preds["blend"] = np.mean([preds[name] for name in MODELS], axis=0)
    return preds


def run_cv(df):
    rows = []
    for i, (tr, va) in enumerate(time_folds(df["close_time"].to_numpy())):
        val = df.iloc[va]
        preds = fit_predict(df.iloc[tr], val)
        rows.append({"fold": i, "val_suspicious": int(val["label"].sum()),
                     **{k: average_precision_score(val["label"], p) for k, p in preds.items()}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir", type=Path)
    ap.add_argument("test_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-cv", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    tables = load_tables(args.train_dir)
    labelled = build_features(tables).merge(tables["labels"], on="TRANSACTION_HASH")
    labelled = labelled.assign(label=labelled["is_suspicious"]).sort_values("close_time").reset_index(drop=True)

    if not args.no_cv:
        cv = run_cv(labelled)
        print(cv.round(4).to_string(index=False))
        print("mean AP", cv.drop(columns=["fold", "val_suspicious"]).mean().round(4).to_dict())

    test = build_features(load_tables(args.test_dir)).reset_index(drop=True)
    test["score"] = fit_predict(labelled, test)["blend"]
    test[["TRANSACTION_HASH", "score"]].to_csv(args.out, index=False)
    print(f"wrote {len(test)} rows to {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
