"""Add feature groups one at a time and compare time-CV mean AP.

Usage: python src/ablation.py <train_dir>
"""
import sys
from pathlib import Path

from features import BASIC_FEATURES, FX_FEATURES, build_features
from graph_features import GRAPH_FEATURES
from motif_features import MOTIF_FEATURES, RARE_FEATURES
from pipeline import run_cv
from prepare import load_tables

LEG = FX_FEATURES[:10]
PAYMENT = FX_FEATURES[10:13]
ACCOUNT = FX_FEATURES[13:]
GRAPH_TX = [c for c in GRAPH_FEATURES if c.startswith("g_")]
GRAPH_ACC = [c for c in GRAPH_FEATURES if not c.startswith("g_")]

STEPS = [
    ("basic", BASIC_FEATURES),
    ("+ leg", BASIC_FEATURES + LEG),
    ("+ payment", BASIC_FEATURES + LEG + PAYMENT),
    ("+ account", BASIC_FEATURES + LEG + PAYMENT + ACCOUNT),
    ("+ graph tx", BASIC_FEATURES + FX_FEATURES + GRAPH_TX),
    ("+ graph acc", BASIC_FEATURES + FX_FEATURES + GRAPH_FEATURES),
    ("+ motif", BASIC_FEATURES + FX_FEATURES + GRAPH_FEATURES + MOTIF_FEATURES),
    ("+ new links", BASIC_FEATURES + FX_FEATURES + GRAPH_FEATURES + MOTIF_FEATURES + RARE_FEATURES),
]


def main(train_dir):
    tables = load_tables(train_dir)
    labelled = build_features(tables).merge(tables["labels"], on="TRANSACTION_HASH")
    labelled = labelled.assign(label=labelled["is_suspicious"]).sort_values("close_time").reset_index(drop=True)

    for name, features in STEPS:
        cv = run_cv(labelled, features)
        print(f"{name:12s} mean AP {cv['ap'].mean():.4f}  ({len(features)} features)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
