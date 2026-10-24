"""Step 3: combine the LightGBM pipeline with the GraphSAGE GNN from gnn/.

Usage: python src/combo.py <train_dir> <test_dir> <submission.csv> [--method stack|blend] [--no-cv]

The GNN's score for each transaction is used in one of two ways:
  stack  one more LightGBM feature. Train rows get out-of-fold GNN scores (each time block is scored
         by GNNs trained on the other blocks), so LightGBM never sees a score fitted on that row's label.
  blend  weighted average of the two models' rank percentiles.
The time-CV report scores both honestly: every GNN used for a fold is trained on that fold's
training rows only, and stacking's training rows get inner out-of-fold scores.
GNN fits run in parallel worker processes; each trains for a fixed number of epochs.
"""
import argparse
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))

from build_graph import build  # noqa: E402
from train import SAGE, scores  # noqa: E402

from cv import time_folds  # noqa: E402
from features import FEATURES, build_features  # noqa: E402
from pipeline import PARAMS  # noqa: E402
from prepare import load_tables  # noqa: E402

# gnn/train.py defaults; pos_weight=5 was best on validation (gnn/README.md)
GNN = dict(hidden=64, layers=2, dropout=0.2, lr=0.005, weight_decay=1e-4, pos_weight=5.0)
STACK_FEATURES = FEATURES + ["gnn_score"]
N_BINS, EMBARGO_S = 5, 3600     # same time bins and embargo as cv.time_folds


def fit_gnn(data, mask, seed, epochs):
    """Train on the labelled tx in `mask` for a fixed number of epochs (no early stopping)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = to_hetero(SAGE(GNN["hidden"], GNN["layers"], GNN["dropout"]), data.metadata(), aggr="sum")
    with torch.no_grad():                       # initialise the lazy layers
        model(data.x_dict, data.edge_index_dict)
    opt = torch.optim.Adam(model.parameters(), lr=GNN["lr"], weight_decay=GNN["weight_decay"])
    y = data["tx"].y
    pw = torch.tensor(GNN["pos_weight"])
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)["tx"].squeeze(-1)
        F.binary_cross_entropy_with_logits(logits[mask], y[mask], pos_weight=pw).backward()
        opt.step()
    return model


_GRAPHS = {}


def _init_worker(paths, threads):
    torch.set_num_threads(threads)
    for name, path in paths.items():
        _GRAPHS[name] = torch.load(path, weights_only=False)


def _run_job(job):
    """job = (train graph rows to fit on, seed, epochs, graph to score) -> sigmoid score per tx."""
    rows, seed, epochs, target = job
    train = _GRAPHS["train"]
    mask = torch.zeros(train["tx"].num_nodes, dtype=torch.bool)
    mask[torch.as_tensor(rows)] = True
    model = fit_gnn(train, mask, seed, epochs)
    return torch.sigmoid(scores(model, _GRAPHS[target])).numpy()


def time_bins(close_time):
    t = np.asarray(close_time, dtype="int64")
    edges = np.linspace(t.min(), t.max() + 1, N_BINS + 1)
    return np.searchsorted(edges, t, side="right") - 1, edges


def block_oof(rows, bin_id, t, edges):
    """Split `rows` by time bin: each bin is predicted by a GNN fitted on the other bins in `rows`,
    minus an embargo around the predicted bin. Returns [(fit_rows, predict_rows)]."""
    splits = []
    for b in np.unique(bin_id[rows]):
        pred = rows[bin_id[rows] == b]
        fit = rows[(bin_id[rows] != b) & ((t[rows] < edges[b] - EMBARGO_S) | (t[rows] >= edges[b + 1] + EMBARGO_S))]
        if len(fit):
            splits.append((fit, pred))
    return splits


def rank_pct(x):
    return rankdata(x) / len(x)


def blend(lgbm, gnn, w):
    return w * rank_pct(lgbm) + (1 - w) * rank_pct(gnn)


def lgbm(train, test, features):
    model = LGBMClassifier(**PARAMS).fit(train[features], train["label"])
    return model.predict_proba(test[features])[:, 1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train_dir", type=Path)
    p.add_argument("test_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--method", choices=["stack", "blend"], default="stack")
    p.add_argument("--blend-weight", type=float, default=0.5, help="LightGBM's weight in the rank blend")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="GNN seeds averaged per fit")
    p.add_argument("--gnn-epochs", type=int, default=70, help="median best epoch was ~62 on an 80%% train split")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=3, help="torch threads per worker")
    p.add_argument("--no-cv", action="store_true", help="skip the time-CV report (about 3x faster)")
    p.add_argument("--gnn-out", type=Path, help="directory to also write the GNN scores (train OOF and test)")
    args = p.parse_args()
    start = time.time()

    train_tables = load_tables(args.train_dir)
    labelled = build_features(train_tables).merge(train_tables["labels"], on="TRANSACTION_HASH")
    labelled = labelled.assign(label=labelled["is_suspicious"]).sort_values("close_time").reset_index(drop=True)
    test = build_features(load_tables(args.test_dir)).reset_index(drop=True)

    train_g, stats, _ = build(args.train_dir)
    test_g, _, _ = build(args.test_dir, stats)
    g_row = pd.Index(train_g["tx"].hash).get_indexer(labelled["TRANSACTION_HASH"])
    g_test = pd.Index(test_g["tx"].hash).get_indexer(test["TRANSACTION_HASH"])
    assert (g_row >= 0).all() and (g_test >= 0).all(), "graph and feature table disagree on transactions"
    print(f"features and graphs built ({time.time() - start:.0f}s)")

    t = labelled["close_time"].to_numpy()
    bin_id, edges = time_bins(t)
    everyone = np.arange(len(labelled))
    folds = time_folds(t, N_BINS, EMBARGO_S)

    # every GNN score we need, as (key, fit rows, target graph, rows to keep); keys group seeds together
    specs = [(("test",), everyone, "test", None)]
    need_oof = args.method == "stack" or args.gnn_out
    if need_oof:
        specs += [(("oof", i), fit, "train", pred) for i, (fit, pred) in enumerate(block_oof(everyone, bin_id, t, edges))]
    if not args.no_cv:
        for k, (tr, va) in enumerate(folds):
            specs.append((("fold", k), tr, "train", va))
            specs += [(("inner", k, i), fit, "train", pred)
                      for i, (fit, pred) in enumerate(block_oof(tr, bin_id, t, edges))]

    jobs = [(g_row[fit], seed, args.gnn_epochs, target) for _, fit, target, _ in specs for seed in args.seeds]
    print(f"training {len(jobs)} GNNs ({len(specs)} fits x {len(args.seeds)} seeds) on {args.workers} workers")
    with tempfile.TemporaryDirectory() as tmp:
        paths = {"train": f"{tmp}/train.pt", "test": f"{tmp}/test.pt"}
        torch.save(train_g, paths["train"])
        torch.save(test_g, paths["test"])
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(paths, args.threads)) as pool:
            out = list(pool.map(_run_job, jobs))
    n = len(args.seeds)
    gnn = {}
    for i, (key, _, target, keep) in enumerate(specs):
        s = np.mean(out[i * n:(i + 1) * n], axis=0)
        gnn[key] = (keep, s[g_row[keep]]) if target == "train" else s[g_test]
    print(f"GNNs done ({time.time() - start:.0f}s)")

    oof = np.full(len(labelled), np.nan)
    for key, val in gnn.items():
        if key[0] == "oof":
            oof[val[0]] = val[1]
    assert not need_oof or not np.isnan(oof).any(), "some train rows have no out-of-fold GNN score"

    if not args.no_cv:
        rows, weights = [], np.round(np.linspace(0, 1, 11), 1)
        for k, (tr, va) in enumerate(folds):
            y = labelled["label"].to_numpy()[va]
            base = lgbm(labelled.iloc[tr], labelled.iloc[va], FEATURES)
            g = gnn[("fold", k)][1]
            row = {"fold": k, "val_suspicious": int(y.sum()), "lgbm": average_precision_score(y, base),
                   "gnn": average_precision_score(y, g)}
            row.update({f"blend_{w}": average_precision_score(y, blend(base, g, w)) for w in weights})
            inner = np.full(len(labelled), np.nan)
            for key, val in gnn.items():
                if key[:2] == ("inner", k):
                    inner[val[0]] = val[1]
            fit = tr[~np.isnan(inner[tr])]      # the first fold has a single time bin, so no inner OOF
            if len(np.unique(bin_id[fit])) >= 2:
                train = labelled.iloc[fit].assign(gnn_score=inner[fit])
                stacked = lgbm(train, labelled.iloc[va].assign(gnn_score=g), STACK_FEATURES)
                row["stack"] = average_precision_score(y, stacked)
            rows.append(row)
        cv = pd.DataFrame(rows)
        cols = ["fold", "val_suspicious", "lgbm", "gnn", f"blend_{args.blend_weight}", "stack"]
        print(cv[cols].round(4).to_string(index=False))
        has_stack = cv["stack"].notna()
        print(f"mean AP, all folds:   lgbm {cv.lgbm.mean():.4f}  gnn {cv.gnn.mean():.4f}  "
              f"blend {cv[f'blend_{args.blend_weight}'].mean():.4f}")
        print(f"mean AP, folds {cv.fold[has_stack].min()}-{cv.fold[has_stack].max()}: "
              f"lgbm {cv.lgbm[has_stack].mean():.4f}  stack {cv['stack'][has_stack].mean():.4f}  "
              f"blend {cv[f'blend_{args.blend_weight}'][has_stack].mean():.4f}")
        sweep = cv[[f"blend_{w}" for w in weights]].mean()
        print("blend weight sweep (LightGBM weight: mean AP)  "
              + "  ".join(f"{w}: {a:.4f}" for w, a in zip(weights, sweep)))

    if args.gnn_out:
        args.gnn_out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"TRANSACTION_HASH": labelled["TRANSACTION_HASH"], "gnn_score": oof}).to_csv(
            args.gnn_out / "gnn_train_oof.csv", index=False)
        pd.DataFrame({"TRANSACTION_HASH": test["TRANSACTION_HASH"], "gnn_score": gnn[("test",)]}).to_csv(
            args.gnn_out / "gnn_test.csv", index=False)

    if args.method == "stack":
        score = lgbm(labelled.assign(gnn_score=oof), test.assign(gnn_score=gnn[("test",)]), STACK_FEATURES)
    else:
        score = blend(lgbm(labelled, test, FEATURES), gnn[("test",)], args.blend_weight)
    test.assign(score=score)[["TRANSACTION_HASH", "score"]].to_csv(args.out, index=False)
    print(f"wrote {len(test)} rows to {args.out} with method={args.method} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
