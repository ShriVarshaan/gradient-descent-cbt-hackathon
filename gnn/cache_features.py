"""Builds the 340 leak-free features (same as train_best_ensemble.py Step 1) and caches them
to `outputs/cache/` so that model experiments can iterate quickly.

Usage:
    python gnn/cache_features.py --train-dir train-data --test-dir test
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from build_graph import build  # noqa: E402
from features_advanced import extract_all_domain_features  # noqa: E402
from model import augment_hetero_graph  # noqa: E402
from train_best_ensemble import compute_clean_graph_features  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=Path, default=Path("train-data"))
    p.add_argument("--test-dir", type=Path, default=Path("test"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/cache"))
    a = p.parse_args()
    a.cache_dir.mkdir(parents=True, exist_ok=True)

    tr_pt, te_pt = Path("graphs/train.pt"), Path("graphs/test_set.pt")
    if tr_pt.exists():
        b_tr = torch.load(tr_pt, weights_only=False)
    else:
        d, s, nm = build(a.train_dir, None)
        b_tr = {"data": d, "stats": s, "names": nm}
        torch.save(b_tr, tr_pt)
    if te_pt.exists():
        b_te = torch.load(te_pt, weights_only=False)
    else:
        d, s, nm = build(a.test_dir, b_tr["stats"])
        b_te = {"data": d, "stats": s, "names": nm}
        torch.save(b_te, te_pt)

    aug_tr, aug_stats = augment_hetero_graph(b_tr["data"], aug_stats=None)
    aug_te, _ = augment_hetero_graph(b_te["data"], aug_stats=aug_stats)
    names = [f"tx_{c}" for c in (b_tr["names"]["tx"] + aug_stats["names"])]

    out = {}
    for tag, aug, sdir in [("train", aug_tr, a.train_dir), ("test", aug_te, a.test_dir)]:
        base = pd.DataFrame(aug["tx"].x.cpu().numpy(), columns=names)
        g = compute_clean_graph_features(aug, base)
        hashes = list(aug["tx"].hash)
        adv, cids = extract_all_domain_features(sdir, hashes)
        X = pd.concat([base, g, adv], axis=1)
        X.insert(0, "TRANSACTION_HASH", hashes)
        X["__ts"] = aug["tx"].time.cpu().numpy()
        if tag == "train":
            X["__y"] = aug["tx"].y.cpu().numpy().astype(int)
        X.to_parquet(a.cache_dir / f"{tag}_X.parquet", index=False)
        with open(a.cache_dir / f"{tag}_cids.pkl", "wb") as f:
            pickle.dump(cids, f)
        out[tag] = X.shape
    print(out)


if __name__ == "__main__":
    main()
