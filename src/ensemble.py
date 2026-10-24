"""Step 4: LightGBM + XGBoost + CatBoost ensemble on the 81 pipeline features plus the burst features.

Usage: python src/ensemble.py <train_dir> <test_dir> <submission.csv> [--no-cv] [--models lgb,xgb,cat] [--seeds N]

Each model is trained on all of train/ and the final score is the mean of the three
predicted probabilities. Hyperparameters were picked by mean AP over the time folds in cv.py.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from burst_features import build_burst_features
from cv import time_folds
from features import FEATURES as BASE_FEATURES, build_features as build_base_features
from prepare import load_tables


def build_features(tables):
    base = build_base_features(tables)
    return base.merge(build_burst_features(tables), on="TRANSACTION_HASH", how="left", validate="one_to_one")


def feature_list(df):
    """The 81 base features plus every b_* burst feature."""
    return BASE_FEATURES + [c for c in df.columns if c.startswith("b_")]


def models(seed=0):
    return {
        "lgb": LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=63, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                              random_state=seed, verbose=-1),
        "xgb": XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=5, subsample=0.8,
                             colsample_bytree=0.4, min_child_weight=10, reg_lambda=10.0,
                             tree_method="hist", random_state=seed),
        "cat": CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                                  random_seed=seed, verbose=0),
    }


def fit_predict(train, test, features, names=("lgb", "xgb", "cat"), seeds=(0,)):
    """Per-model probabilities on `test` (averaged over seeds) and their mean."""
    preds = {}
    for name in names:
        runs = []
        for seed in seeds:
            model = models(seed)[name].fit(train[features], train["label"])
            runs.append(model.predict_proba(test[features])[:, 1])
        preds[name] = np.mean(runs, axis=0)
    preds["ensemble"] = np.mean([preds[n] for n in names], axis=0)
    return preds


def run_cv(df, features):
    rows = []
    for i, (tr, va) in enumerate(time_folds(df["close_time"].to_numpy())):
        val = df.iloc[va]
        preds = fit_predict(df.iloc[tr], val, features)
        rows.append({"fold": i, "val_suspicious": int(val["label"].sum()),
                     **{k: average_precision_score(val["label"], p) for k, p in preds.items()}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir", type=Path)
    ap.add_argument("test_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--models", default="lgb,xgb,cat", help="comma-separated models to average")
    ap.add_argument("--seeds", type=int, default=1, help="average each model over this many seeds")
    args = ap.parse_args()
    t0 = time.time()

    tables = load_tables(args.train_dir)
    labelled = build_features(tables).merge(tables["labels"], on="TRANSACTION_HASH")
    labelled = labelled.assign(label=labelled["is_suspicious"]).sort_values("close_time").reset_index(drop=True)
    features = feature_list(labelled)
    print(f"{len(features)} features")

    if not args.no_cv:
        cv = run_cv(labelled, features)
        print(cv.round(4).to_string(index=False))
        print("mean AP", cv.drop(columns=["fold", "val_suspicious"]).mean().round(4).to_dict())

    test = build_features(load_tables(args.test_dir)).reset_index(drop=True)
    test["score"] = fit_predict(labelled, test, features, args.models.split(","), range(args.seeds))["ensemble"]
    test[["TRANSACTION_HASH", "score"]].to_csv(args.out, index=False)
    print(f"wrote {len(test)} rows to {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
