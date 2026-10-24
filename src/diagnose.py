"""Where the model misses: out-of-fold AP and top-1% recall per difficulty and family.

Usage: python src/diagnose.py <train_dir>
The family and difficulty labels are only used to report results, never as features.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from cv import time_folds
from features import FEATURES, build_features
from pipeline import PARAMS
from prepare import load_tables


def oof_scores(df, features=FEATURES, params=PARAMS):
    """Out-of-fold scores for every validated row; NaN for rows never validated (bin 0)."""
    scores = np.full(len(df), np.nan)
    fold = np.full(len(df), -1)
    for i, (tr, va) in enumerate(time_folds(df["close_time"].to_numpy())):
        model = LGBMClassifier(**params).fit(df.iloc[tr][features], df.iloc[tr]["label"])
        scores[va] = model.predict_proba(df.iloc[va][features])[:, 1]
        fold[va] = i
    return scores, fold


def report(df, scores, fold):
    d = df.assign(score=scores, fold=fold)[fold >= 0]
    # top 1% within each fold, as the site's review budget
    d["top1"] = d.groupby("fold")["score"].rank(ascending=False, pct=True) <= 0.01
    neg = d[d.label == 0]
    rows = []
    for (diff, fam), pos in d[d.label == 1].groupby(["difficulty", "family"]):
        sub = pd.concat([pos, neg])
        rows.append({"difficulty": diff, "family": fam, "n": len(pos),
                     "ap_vs_all_neg": average_precision_score(sub.label, sub.score),
                     "recall_top1": pos.top1.mean(),
                     "median_rank_pct": d.score.rank(ascending=False, pct=True)[pos.index].median()})
    out = pd.DataFrame(rows).sort_values("ap_vs_all_neg")
    print(out.to_string(index=False, float_format="%.3f"))
    print(f"overall OOF AP {average_precision_score(d.label, d.score):.4f}")
    return out


def main(train_dir):
    tables = load_tables(train_dir)
    df = build_features(tables).merge(tables["labels"], on="TRANSACTION_HASH")
    df = df.assign(label=df["is_suspicious"]).sort_values("close_time").reset_index(drop=True)
    report(df, *oof_scores(df))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
