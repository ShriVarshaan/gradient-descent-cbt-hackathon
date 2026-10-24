"""Drop one feature group at a time and compare time-CV mean AP on the dev split only.

The latest 20% (holdout) is not used here. Use it once, at the end, with pipeline.py.

Usage: python src/drop_group.py <train_dir>
"""
import sys
from pathlib import Path

from cv import holdout_mask
from features import BASIC_FEATURES, FEATURES, FX_FEATURES, build_features
from graph_features import GRAPH_FEATURES
from history_features import HISTORY_FEATURES
from motif_features import MOTIF_FEATURES, RARE_FEATURES
from pipeline import run_cv
from prepare import load_tables

GROUPS = {
    "basic": BASIC_FEATURES,
    "fx": FX_FEATURES,
    "history": HISTORY_FEATURES,
    "graph_tx": [c for c in GRAPH_FEATURES if c.startswith("g_")],
    "graph_acc": [c for c in GRAPH_FEATURES if not c.startswith("g_")],
    "motif": MOTIF_FEATURES,
    "rare": RARE_FEATURES,
}


def main(train_dir):
    assert sorted(c for g in GROUPS.values() for c in g) == sorted(FEATURES), "groups must cover FEATURES exactly"
    tables = load_tables(train_dir)
    labelled = build_features(tables).merge(tables["labels"], on="TRANSACTION_HASH")
    labelled = labelled.assign(label=labelled["is_suspicious"]).sort_values("close_time").reset_index(drop=True)
    dev = labelled[~holdout_mask(labelled["close_time"].to_numpy())].reset_index(drop=True)

    base = run_cv(dev, FEATURES)["ap"].mean()
    print(f"{'all features':18s} mean AP {base:.4f}  ({len(FEATURES)} features)")
    for name, cols in GROUPS.items():
        rest = [c for c in FEATURES if c not in set(cols)]
        ap = run_cv(dev, rest)["ap"].mean()
        print(f"{'drop ' + name:18s} mean AP {ap:.4f}  delta {ap - base:+.4f}  ({len(rest)} features)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
