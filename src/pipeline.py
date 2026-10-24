"""Step 2: time-based CV, baseline LightGBM, first submission.

Usage: python src/pipeline.py <train_dir> <test_dir> <submission.csv>
"""
import sys
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from cv import time_folds
from features import FEATURES, build_features
from prepare import load_tables

# Tuned on the 4 time folds: every fold improves over the 300-tree, 31-leaf default.
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=63, min_child_samples=50,
              random_state=0, verbose=-1)


def run_cv(df, features=FEATURES):
    rows = []
    for i, (tr, va) in enumerate(time_folds(df["close_time"].to_numpy())):
        train, val = df.iloc[tr], df.iloc[va]
        model = LGBMClassifier(**PARAMS).fit(train[features], train["label"])
        scores = model.predict_proba(val[features])[:, 1]
        rows.append({
            "fold": i,
            "n_train": len(train),
            "n_val": len(val),
            "val_suspicious": int(val["label"].sum()),
            "prevalence": val["label"].mean(),
            "ap": average_precision_score(val["label"], scores),
        })
    return pd.DataFrame(rows)


def fit_and_score(train, test):
    model = LGBMClassifier(**PARAMS).fit(train[FEATURES], train["label"])
    return model.predict_proba(test[FEATURES])[:, 1]


def main(train_dir, test_dir, out_path):
    train_tables = load_tables(train_dir)
    labelled = build_features(train_tables).merge(train_tables["labels"], on="TRANSACTION_HASH")
    labelled = labelled.assign(label=labelled["is_suspicious"]).sort_values("close_time").reset_index(drop=True)

    cv = run_cv(labelled)
    print(cv.to_string(index=False))
    print(f"mean AP {cv['ap'].mean():.4f}")

    test = build_features(load_tables(test_dir)).reset_index(drop=True)
    test["score"] = fit_and_score(labelled, test)
    test[["TRANSACTION_HASH", "score"]].to_csv(out_path, index=False)
    print(f"wrote {len(test)} rows to {out_path}")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:4]))
