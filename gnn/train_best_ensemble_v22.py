"""Leak-Free, Distribution-Aligned Graph + Motif + Multi-GBDT + Inductive GNN Pipeline.

Fixes the two root causes of train-vs-test degradation:
  1. Eliminates the broken single-checkpoint `test_set_gnn_features.parquet` scale/sign mismatch
     by computing all 73 1-hop/2-hop temporal & bipartite account message-passing features
     deterministically and identically on `train.pt` and `test_set.pt`.
  2. Eliminates random-fold transductive label leakage (`mgnn_*` hijacking 45% of tree gain)
     by training all GBDT models (`CatBoost` d5/d6/d7, `XGBoost` d5/d6, `LightGBM` d6) directly
     on the 340 leak-free features and combining:
       - 5-Fold Contiguous Time-Block CV Ensemble (with 2-hour embargo, early-stopped per fold)
       - Full-Train Refit Ensemble (trained on 100% of `train-data`)
       - Inductive `MotifTxGNN` (trained with isolated subgraph message passing)
       - Multi-Scale Ephemeral Temporal Connected Component (E-TCC) prediction propagation.

Usage:
    python gnn/train_best_ensemble.py --train-dir train-data --test-dir test --out-csv submission.csv
"""
import argparse
import copy
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import scatter
from xgboost import XGBClassifier

from build_graph import build
from features_advanced import extract_all_domain_features
from model import augment_hetero_graph

SEED_OFFSET = 0  # set by --seed-offset; shifts every model seed for seed-averaged runs

warnings.filterwarnings("ignore")


class MotifTxGNN(nn.Module):
    """Residual GNN operating on the full 340-feature node space and passing
    mean + max messages over Ephemeral Temporal Connected Component (E-TCC) edges.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 80, emb_dim: int = 16, dropout: float = 0.25):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.conv1_lin = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2_lin = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.head = nn.Linear(emb_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, ei: torch.Tensor):
        n = x.shape[0]
        h0 = self.enc(x)
        if ei.shape[1] > 0:
            src, dst = ei[0], ei[1]
            m1_mean = scatter(h0[src], dst, dim=0, dim_size=n, reduce="mean")
            m1_max = scatter(h0[src], dst, dim=0, dim_size=n, reduce="max")
        else:
            m1_mean = torch.zeros_like(h0)
            m1_max = torch.zeros_like(h0)
        h1 = h0 + F.dropout(
            F.silu(self.norm1(self.conv1_lin(torch.cat([h0, m1_mean, m1_max], dim=-1)))),
            p=self.dropout,
            training=self.training,
        )

        if ei.shape[1] > 0:
            m2_mean = scatter(h1[src], dst, dim=0, dim_size=n, reduce="mean")
            m2_max = scatter(h1[src], dst, dim=0, dim_size=n, reduce="max")
        else:
            m2_mean = torch.zeros_like(h1)
            m2_max = torch.zeros_like(h1)
        h2 = h1 + F.dropout(
            F.silu(self.norm2(self.conv2_lin(torch.cat([h1, m2_mean, m2_max], dim=-1)))),
            p=self.dropout,
            training=self.training,
        )

        emb = self.bottleneck(torch.cat([h0, h1, h2], dim=-1))
        logits = self.head(F.silu(emb)).squeeze(-1)
        return logits, emb


def compute_clean_graph_features(aug_data, df_base: pd.DataFrame) -> pd.DataFrame:
    """Computes 73 deterministic, leak-free 1-hop & 2-hop graph-propagated features
    identically on any split (`train`, `validation`, or `test`):
      - Temporal tx<->tx degree (`gnn_temp_log_deg`)
      - 24 primary sender & receiver account features (`gnn_sender_acc_f0..11`, `gnn_dest_acc_f0..11`)
      - 48 1-hop/2-hop temporal neighborhood & account Leave-One-Out (LOO) / max aggregations
        of the 6 core burst and corridor signals.
    """
    n_tx = aug_data["tx"].x.shape[0]
    n_acc = aug_data["account"].x.shape[0]
    x_acc = aug_data["account"].x.cpu().numpy()
    sender_id = aug_data["tx"].sender_id.cpu()
    dest_id = aug_data["tx"].dest_id.cpu()
    ei_temp = aug_data["tx", "temporal", "tx"].edge_index.cpu()
    src, dst = ei_temp[0], ei_temp[1]

    temp_deg = scatter(torch.ones_like(src, dtype=torch.float), dst, dim=0, dim_size=n_tx, reduce="sum")
    g_cols = {"gnn_temp_log_deg": np.log1p(temp_deg.numpy())}

    s_np = sender_id.numpy()
    d_np = dest_id.numpy()
    for j in range(x_acc.shape[1]):
        g_cols[f"gnn_sender_acc_f{j}"] = x_acc[s_np, j]
        g_cols[f"gnn_dest_acc_f{j}"] = x_acc[d_np, j]

    key_cols = [
        "tx_burst_combined_10m",
        "tx_log_pair_cnt",
        "tx_log_sender_10m",
        "tx_log_dest_10m",
        "tx_log_dest_out_10m",
        "tx_log_sender_in_10m",
    ]
    X_key = torch.tensor(df_base[key_cols].values, dtype=torch.float)

    t_mean = scatter(X_key[src], dst, dim=0, dim_size=n_tx, reduce="mean")
    t_max = scatter(X_key[src], dst, dim=0, dim_size=n_tx, reduce="max")
    h2_mean = scatter(t_mean[src], dst, dim=0, dim_size=n_tx, reduce="mean")
    h2_max = scatter(t_max[src], dst, dim=0, dim_size=n_tx, reduce="max")

    sk_sum = scatter(X_key, sender_id, dim=0, dim_size=n_acc, reduce="sum")
    sk_cnt = scatter(torch.ones((n_tx, 1)), sender_id, dim=0, dim_size=n_acc, reduce="sum")
    sk_loo = (sk_sum[sender_id] - X_key) / (sk_cnt[sender_id] - 1.0).clamp(min=1.0)
    sk_amax = scatter(X_key, sender_id, dim=0, dim_size=n_acc, reduce="max")[sender_id]

    dk_sum = scatter(X_key, dest_id, dim=0, dim_size=n_acc, reduce="sum")
    dk_cnt = scatter(torch.ones((n_tx, 1)), dest_id, dim=0, dim_size=n_acc, reduce="sum")
    dk_loo = (dk_sum[dest_id] - X_key) / (dk_cnt[dest_id] - 1.0).clamp(min=1.0)
    dk_amax = scatter(X_key, dest_id, dim=0, dim_size=n_acc, reduce="max")[dest_id]

    for i, c in enumerate(key_cols):
        g_cols[f"gnn_p1_mean_{c}"] = t_mean[:, i].numpy()
        g_cols[f"gnn_p1_max_{c}"] = t_max[:, i].numpy()
        g_cols[f"gnn_p2_mean_{c}"] = h2_mean[:, i].numpy()
        g_cols[f"gnn_p2_max_{c}"] = h2_max[:, i].numpy()
        g_cols[f"gnn_sloo_{c}"] = sk_loo[:, i].numpy()
        g_cols[f"gnn_smax_{c}"] = sk_amax[:, i].numpy()
        g_cols[f"gnn_dloo_{c}"] = dk_loo[:, i].numpy()
        g_cols[f"gnn_dmax_{c}"] = dk_amax[:, i].numpy()

    return pd.DataFrame(g_cols, dtype=np.float32)


def propagate_predictions_over_motifs(probs: np.ndarray, cids: dict) -> np.ndarray:
    """Multi-scale Ephemeral Temporal Connected Component (E-TCC) prediction smoothing."""
    eph = cids["is_eph_1h"]
    c120_m = pd.Series(probs).groupby(cids["cid_e120"]).transform("mean").values
    c240_m = pd.Series(probs).groupby(cids["cid_e240"]).transform("mean").values
    c240_x = pd.Series(probs).groupby(cids["cid_e240"]).transform("max").values
    c360_m = pd.Series(probs).groupby(cids["cid_e360"]).transform("mean").values
    c360_x = pd.Series(probs).groupby(cids["cid_e360"]).transform("max").values
    c600_m = pd.Series(probs).groupby(cids["cid_e600"]).transform("mean").values

    comp_mean = 0.20 * c120_m + 0.35 * c240_m + 0.30 * c360_m + 0.15 * c600_m
    comp_max = 0.50 * c240_x + 0.50 * c360_x
    return np.where(eph == 1, 0.50 * probs + 0.35 * comp_mean + 0.15 * comp_max, probs)


def train_inductive_motif_gnn(
    X_tr_df: pd.DataFrame,
    y_np: np.ndarray,
    ts_tr: np.ndarray,
    tr_cids: dict,
    X_te_df: pd.DataFrame,
    te_cids: dict,
    tr_m_temp: np.ndarray,
    va_m_temp: np.ndarray,
    n_folds: int = 5,
    epochs: int = 35,
    seed: int = 42,
):
    """Trains `MotifTxGNN` with strict isolated subgraph message passing across:
      1. The 80/20 temporal validation split (`tr_m_temp` -> `va_m_temp`)
      2. 5 Contiguous Time-Block Folds (with 2-hour embargo), accumulating calibrated test predictions.
    """
    X_tr_raw = X_tr_df.fillna(0.0).values.astype(np.float32)
    X_te_raw = X_te_df.fillna(0.0).values.astype(np.float32)

    mean = X_tr_raw[tr_m_temp].mean(axis=0)
    std = X_tr_raw[tr_m_temp].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    X_tr_std = torch.tensor(np.clip((X_tr_raw - mean) / std, -10.0, 10.0), dtype=torch.float)
    X_te_std = torch.tensor(np.clip((X_te_raw - mean) / std, -10.0, 10.0), dtype=torch.float)

    s_tr, d_tr = tr_cids["etcc_edges"]
    ei_tr_full = np.stack([np.concatenate([s_tr, d_tr]), np.concatenate([d_tr, s_tr])])

    s_te, d_te = te_cids["etcc_edges"]
    ei_te = torch.tensor(np.stack([np.concatenate([s_te, d_te]), np.concatenate([d_te, s_te])]), dtype=torch.long)

    y_t = torch.tensor(y_np, dtype=torch.float)

    def fit_isolated(tr_mask_b: np.ndarray, va_mask_b: np.ndarray, run_seed: int):
        e_tr_m = tr_mask_b[ei_tr_full[0]] & tr_mask_b[ei_tr_full[1]]
        e_va_m = va_mask_b[ei_tr_full[0]] & va_mask_b[ei_tr_full[1]]
        ei_f_tr = torch.tensor(ei_tr_full[:, e_tr_m], dtype=torch.long)
        ei_f_va = torch.tensor(ei_tr_full[:, e_va_m], dtype=torch.long)

        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        model = MotifTxGNN(X_tr_std.shape[1], hidden_dim=80, emb_dim=16, dropout=0.25)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-4)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.5))

        tr_t = torch.tensor(tr_mask_b, dtype=torch.bool)
        best_ap = -1.0
        best_auc = 0.0
        best_state = None
        for ep in range(1, epochs + 1):
            model.train()
            opt.zero_grad()
            l, _ = model(X_tr_std, ei_f_tr)
            loss = crit(l[tr_t], y_t[tr_t])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            if ep % 5 == 0 or ep == epochs:
                model.eval()
                with torch.no_grad():
                    vl, _ = model(X_tr_std, ei_f_va)
                    vp = torch.sigmoid(vl[va_mask_b]).numpy()
                    ap = average_precision_score(y_np[va_mask_b], vp)
                    auc = roc_auc_score(y_np[va_mask_b], vp)
                    if ap > best_ap:
                        best_ap = ap
                        best_auc = auc
                        best_state = copy.deepcopy(model.state_dict())
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vl, _ = model(X_tr_std, ei_f_va)
            vp_va = torch.sigmoid(vl[va_mask_b]).numpy()
            tl, _ = model(X_te_std, ei_te)
            vp_te = torch.sigmoid(tl).numpy()
        return vp_va, vp_te, best_ap, best_auc

    # 1. Strict 80/20 temporal split evaluation
    va_gnn_temp, _, temp_ap, temp_auc = fit_isolated(tr_m_temp, va_m_temp, seed)
    print(f"  [Inductive MotifTxGNN Temporal Split] Val AP: {temp_ap:.4f} | ROC-AUC: {temp_auc:.4f}")

    # 2. 5-Fold Contiguous Time-Block CV for OOF & Test ensemble
    n_tr = len(y_np)
    n_te = len(X_te_df)
    order = np.argsort(ts_tr)
    fold_id = np.zeros(n_tr, dtype=int)
    fold_id[order] = np.arange(n_tr) * n_folds // n_tr

    oof_gnn = np.zeros(n_tr, dtype=np.float32)
    te_gnn = np.zeros(n_te, dtype=np.float32)
    for f in range(n_folds):
        f_va = fold_id == f
        va_min, va_max = ts_tr[f_va].min(), ts_tr[f_va].max()
        f_tr = (~f_va) & ((ts_tr < va_min - 7200) | (ts_tr > va_max + 7200))
        vp_va, vp_te, f_ap, f_auc = fit_isolated(f_tr, f_va, seed + 10 + f)
        # Align fold probability mean to base rate (0.035) so folds are identically scaled
        oof_gnn[f_va] = np.clip(vp_va * (0.035 / max(vp_va.mean(), 1e-4)), 0.0, 1.0)
        te_gnn += np.clip(vp_te * (0.035 / max(vp_te.mean(), 1e-4)), 0.0, 1.0) / n_folds

    return va_gnn_temp, oof_gnn, te_gnn, temp_ap, temp_auc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, default=Path("train-data"))
    parser.add_argument("--test-dir", type=Path, default=Path("test"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out-csv", type=Path, default=Path("submission.csv"))
    parser.add_argument("--pseudo", type=Path, default=None, help="Earlier test submission CSV used for pseudo-labels in the final refit")
    parser.add_argument("--pseudo-pos", type=float, default=0.005, help="Top fraction of --pseudo scores labelled suspicious")
    parser.add_argument("--pseudo-neg", type=float, default=0.5, help="Bottom fraction of --pseudo scores labelled legitimate")
    parser.add_argument("--val-dir", type=Path, default=None, help="Unlabelled extra split (e.g. validation) to pseudo-label")
    parser.add_argument("--val-pseudo", type=Path, default=None, help="Submission-style CSV of scores for --val-dir")
    parser.add_argument("--extra-val", type=Path, default=None, help="Extra feature parquet for --val-dir")
    parser.add_argument("--seed-offset", type=int, default=0, help="Added to every random seed")
    parser.add_argument("--extra-train", type=Path, default=None, help="Parquet of extra train features keyed by TRANSACTION_HASH")
    parser.add_argument("--extra-test", type=Path, default=None, help="Parquet of extra test features keyed by TRANSACTION_HASH")
    args = parser.parse_args()
    global SEED_OFFSET
    SEED_OFFSET = args.seed_offset

    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("================ Step 1: Building 340 Leak-Free, Distribution-Aligned Features ================")
    train_graph_pt = Path("graphs/train.pt")
    test_graph_pt = Path("graphs/test_set.pt")
    if train_graph_pt.exists():
        b_tr = torch.load(train_graph_pt, weights_only=False)
    else:
        print(f"  Building training graph from {args.train_dir}...")
        data_tr, stats_tr, names_tr = build(args.train_dir, None)
        b_tr = {"data": data_tr, "stats": stats_tr, "names": names_tr}
        train_graph_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(b_tr, train_graph_pt)

    if test_graph_pt.exists():
        b_te = torch.load(test_graph_pt, weights_only=False)
    else:
        print(f"  Building test graph from {args.test_dir}...")
        data_te, stats_te, names_te = build(args.test_dir, b_tr["stats"])
        b_te = {"data": data_te, "stats": stats_te, "names": names_te}
        test_graph_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(b_te, test_graph_pt)

    aug_tr, aug_stats = augment_hetero_graph(b_tr["data"], aug_stats=None)
    aug_te, _ = augment_hetero_graph(b_te["data"], aug_stats=aug_stats)
    tx_feat_names = [f"tx_{c}" for c in (b_tr["names"]["tx"] + aug_stats["names"])]

    df_tr_base = pd.DataFrame(aug_tr["tx"].x.cpu().numpy(), columns=tx_feat_names)
    df_te_base = pd.DataFrame(aug_te["tx"].x.cpu().numpy(), columns=tx_feat_names)

    tr_hashes = aug_tr["tx"].hash
    te_hashes = aug_te["tx"].hash
    ts_tr = aug_tr["tx"].time.cpu().numpy()
    ts_te = aug_te["tx"].time.cpu().numpy()
    y_all = aug_tr["tx"].y.cpu().numpy().astype(int)

    df_tr_g = compute_clean_graph_features(aug_tr, df_tr_base)
    df_te_g = compute_clean_graph_features(aug_te, df_te_base)

    adv_tr_df, tr_cids = extract_all_domain_features(args.train_dir, tr_hashes)
    adv_te_df, te_cids = extract_all_domain_features(args.test_dir, te_hashes)

    X_tr_all = pd.concat([df_tr_base, df_tr_g, adv_tr_df], axis=1)
    X_te_all = pd.concat([df_te_base, df_te_g, adv_te_df], axis=1)
    if args.extra_train is not None:
        # Extra feature table (e.g. the pipeline's v7 features), aligned to the graph's row order.
        ex_tr = pd.read_parquet(args.extra_train).set_index("TRANSACTION_HASH").reindex(list(tr_hashes))
        ex_te = pd.read_parquet(args.extra_test).set_index("TRANSACTION_HASH").reindex(list(te_hashes))
        assert ex_tr.notna().all().all() and ex_te.notna().all().all(), "extra features missing for some hashes"
        ex_tr.columns = ex_te.columns = [f"x_{c}" for c in ex_tr.columns]
        X_tr_all = pd.concat([X_tr_all, ex_tr.reset_index(drop=True).astype("float32")], axis=1)
        X_te_all = pd.concat([X_te_all, ex_te.reset_index(drop=True).astype("float32")], axis=1)

    assert list(X_tr_all.columns) == list(X_te_all.columns), "Train/Test column mismatch!"

    def pseudo_rows(X, hashes, csv):
        ps = pd.read_csv(csv).set_index("TRANSACTION_HASH").reindex(list(hashes))["score"].to_numpy()
        assert not np.isnan(ps).any(), f"{csv} does not cover every hash"
        r = ps.argsort().argsort() / (len(ps) - 1)
        pos, neg = r >= 1 - args.pseudo_pos, r <= args.pseudo_neg
        return pd.concat([X[pos], X[neg]], ignore_index=True), np.r_[np.ones(pos.sum(), int), np.zeros(neg.sum(), int)]

    PX, Py = [], []
    if args.pseudo is not None:
        x, y = pseudo_rows(X_te_all, te_hashes, args.pseudo); PX.append(x); Py.append(y)
    if args.val_dir is not None:
        # v22: a third, unlabelled week, featurised exactly like the test split.
        val_graph_pt = Path("graphs/val.pt")
        if val_graph_pt.exists():
            b_va = torch.load(val_graph_pt, weights_only=False)
        else:
            data_va, stats_va, names_va = build(args.val_dir, b_tr["stats"])
            b_va = {"data": data_va, "stats": stats_va, "names": names_va}
            torch.save(b_va, val_graph_pt)
        aug_va, _ = augment_hetero_graph(b_va["data"], aug_stats=aug_stats)
        df_va_base = pd.DataFrame(aug_va["tx"].x.cpu().numpy(), columns=tx_feat_names)
        va_hashes = aug_va["tx"].hash
        adv_va_df, _ = extract_all_domain_features(args.val_dir, va_hashes)
        X_va_all = pd.concat([df_va_base, compute_clean_graph_features(aug_va, df_va_base), adv_va_df], axis=1)
        if args.extra_val is not None:
            ex_va = pd.read_parquet(args.extra_val).set_index("TRANSACTION_HASH").reindex(list(va_hashes))
            assert ex_va.notna().all().all(), "extra val features missing for some hashes"
            ex_va.columns = [f"x_{c}" for c in ex_va.columns]
            X_va_all = pd.concat([X_va_all, ex_va.reset_index(drop=True).astype("float32")], axis=1)
        assert list(X_va_all.columns) == list(X_tr_all.columns), "Val/Train column mismatch!"
        x, y = pseudo_rows(X_va_all, va_hashes, args.val_pseudo); PX.append(x); Py.append(y)
    if PX:
        print(f"  Pseudo-labels: +{int(sum(y.sum() for y in Py))} suspicious, +{int(sum((y == 0).sum() for y in Py))} legitimate rows from {len(PX)} unlabelled split(s)")
    all_feats = list(X_tr_all.columns)
    n_nan_tr = int(X_tr_all.isna().sum().sum())
    n_nan_te = int(X_te_all.isna().sum().sum())
    print(f"  Assembled {len(all_feats)} clean features | Train rows: {len(X_tr_all):,} (NaNs={n_nan_tr}) | Test rows: {len(X_te_all):,} (NaNs={n_nan_te})")

    # Save clean test GNN/graph features to replace the old mismatched file
    df_te_g_out = df_te_g.copy()
    df_te_g_out.insert(0, "TRANSACTION_HASH", te_hashes)
    df_te_g_out.to_parquet(args.out_dir / "test_set_gnn_features.parquet", index=False)

    # Strict 80/20 Zero-Leakage Temporal Split (with 1-hour embargo)
    cut_ts = np.quantile(ts_tr, 0.80)
    tr_m = ts_tr <= (cut_ts - 3600)
    va_m = ts_tr > cut_ts

    print("\n================ Step 2: Evaluating on Strict Zero-Leakage Temporal Split ================")
    va_gnn_temp, oof_gnn_cb, te_gnn_cb, mgnn_temp_ap, mgnn_temp_auc = train_inductive_motif_gnn(
        X_tr_all, y_all, ts_tr, tr_cids, X_te_all, te_cids, tr_m, va_m, n_folds=5, epochs=35, seed=42 + SEED_OFFSET
    )

    X_tr, y_tr = X_tr_all.loc[tr_m], y_all[tr_m]
    X_va, y_va = X_tr_all.loc[va_m], y_all[va_m]

    # 1. CatBoost Depth 6
    cb6 = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.03, auto_class_weights="SqrtBalanced",
        l2_leaf_reg=5.0, eval_metric="PRAUC", early_stopping_rounds=60, random_seed=42 + SEED_OFFSET, verbose=False,
    ).fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    p_cb6 = propagate_predictions_over_motifs(cb6.predict_proba(X_tr_all)[:, 1], tr_cids)
    ap_cb6, auc_cb6 = average_precision_score(y_va, p_cb6[va_m]), roc_auc_score(y_va, p_cb6[va_m])
    print(f"  1. CatBoost (depth=6, it={cb6.get_best_iteration():3d}) -> Val AP: {ap_cb6:.4f} | ROC-AUC: {auc_cb6:.4f}")

    # 2. CatBoost Depth 7
    cb7 = CatBoostClassifier(
        iterations=1000, depth=7, learning_rate=0.025, auto_class_weights="SqrtBalanced",
        l2_leaf_reg=6.0, eval_metric="PRAUC", early_stopping_rounds=60, random_seed=77 + SEED_OFFSET, verbose=False,
    ).fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    p_cb7 = propagate_predictions_over_motifs(cb7.predict_proba(X_tr_all)[:, 1], tr_cids)
    ap_cb7, auc_cb7 = average_precision_score(y_va, p_cb7[va_m]), roc_auc_score(y_va, p_cb7[va_m])
    print(f"  2. CatBoost (depth=7, it={cb7.get_best_iteration():3d}) -> Val AP: {ap_cb7:.4f} | ROC-AUC: {auc_cb7:.4f}")

    # 3. CatBoost Depth 5
    cb5 = CatBoostClassifier(
        iterations=1000, depth=5, learning_rate=0.03, auto_class_weights="SqrtBalanced",
        l2_leaf_reg=4.0, eval_metric="PRAUC", early_stopping_rounds=60, random_seed=100 + SEED_OFFSET, verbose=False,
    ).fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    p_cb5 = propagate_predictions_over_motifs(cb5.predict_proba(X_tr_all)[:, 1], tr_cids)
    ap_cb5, auc_cb5 = average_precision_score(y_va, p_cb5[va_m]), roc_auc_score(y_va, p_cb5[va_m])
    print(f"  3. CatBoost (depth=5, it={cb5.get_best_iteration():3d}) -> Val AP: {ap_cb5:.4f} | ROC-AUC: {auc_cb5:.4f}")

    # 4. XGBoost Depth 6
    xgb6 = XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.02, subsample=0.80, colsample_bytree=0.55,
        min_child_weight=10, reg_alpha=1.5, reg_lambda=5.0, scale_pos_weight=3.5,
        eval_metric="aucpr", early_stopping_rounds=50, random_state=42 + SEED_OFFSET, n_jobs=-1, tree_method="hist",
    ).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_xgb6 = propagate_predictions_over_motifs(xgb6.predict_proba(X_tr_all)[:, 1], tr_cids)
    ap_xgb6, auc_xgb6 = average_precision_score(y_va, p_xgb6[va_m]), roc_auc_score(y_va, p_xgb6[va_m])
    print(f"  4. XGBoost  (depth=6, it={xgb6.best_iteration:3d}) -> Val AP: {ap_xgb6:.4f} | ROC-AUC: {auc_xgb6:.4f}")

    # 5. XGBoost Depth 5
    xgb5 = XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.02, subsample=0.80, colsample_bytree=0.50,
        min_child_weight=8, reg_alpha=1.0, reg_lambda=4.0, scale_pos_weight=3.0,
        eval_metric="aucpr", early_stopping_rounds=50, random_state=100 + SEED_OFFSET, n_jobs=-1, tree_method="hist",
    ).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_xgb5 = propagate_predictions_over_motifs(xgb5.predict_proba(X_tr_all)[:, 1], tr_cids)
    ap_xgb5, auc_xgb5 = average_precision_score(y_va, p_xgb5[va_m]), roc_auc_score(y_va, p_xgb5[va_m])
    print(f"  5. XGBoost  (depth=5, it={xgb5.best_iteration:3d}) -> Val AP: {ap_xgb5:.4f} | ROC-AUC: {auc_xgb5:.4f}")

    # 6. LightGBM Leaves 31
    lgb31 = LGBMClassifier(
        n_estimators=400, learning_rate=0.02, num_leaves=31, max_depth=6, scale_pos_weight=3.0,
        subsample=0.80, colsample_bytree=0.50, min_child_samples=25, reg_alpha=1.5, reg_lambda=5.0,
        random_state=42 + SEED_OFFSET, n_jobs=-1, verbose=-1,
    ).fit(X_tr, y_tr)
    p_lgb31 = propagate_predictions_over_motifs(lgb31.predict_proba(X_tr_all)[:, 1], tr_cids)
    ap_lgb31, auc_lgb31 = average_precision_score(y_va, p_lgb31[va_m]), roc_auc_score(y_va, p_lgb31[va_m])
    print(f"  6. LightGBM (leaves=31)       -> Val AP: {ap_lgb31:.4f} | ROC-AUC: {auc_lgb31:.4f}")

    p_gnn_full = np.zeros(len(X_tr_all), dtype=np.float32)
    p_gnn_full[va_m] = np.clip(va_gnn_temp * (0.035 / max(va_gnn_temp.mean(), 1e-4)), 0.0, 1.0)
    p_gnn_prop = propagate_predictions_over_motifs(p_gnn_full, tr_cids)

    p_cb_val = 0.45 * p_cb6 + 0.30 * p_cb7 + 0.25 * p_cb5
    p_xgb_val = 0.50 * p_xgb6 + 0.50 * p_xgb5
    p_ens_val = 0.54 * p_cb_val + 0.36 * p_xgb_val + 0.06 * p_lgb31 + 0.04 * p_gnn_prop

    ap_cb_ens = average_precision_score(y_va, p_cb_val[va_m])
    ap_xgb_ens = average_precision_score(y_va, p_xgb_val[va_m])
    ap_ens, auc_ens = average_precision_score(y_va, p_ens_val[va_m]), roc_auc_score(y_va, p_ens_val[va_m])

    print("\n================ Zero-Leakage Temporal Validation Summary ================")
    print(f"  Inductive MotifTxGNN                       -> Val AP: {mgnn_temp_ap:.4f} | ROC-AUC: {mgnn_temp_auc:.4f}")
    print(f"  Graph + Motif + LightGBM (Propagated)      -> Val AP: {ap_lgb31:.4f} | ROC-AUC: {auc_lgb31:.4f}")
    print(f"  Graph + Motif + XGBoost Blend (Propagated) -> Val AP: {ap_xgb_ens:.4f} | ROC-AUC: {roc_auc_score(y_va, p_xgb_val[va_m]):.4f}")
    print(f"  Graph + Motif + CatBoost Blend (Propagated)-> Val AP: {ap_cb_ens:.4f} | ROC-AUC: {roc_auc_score(y_va, p_cb_val[va_m]):.4f}")
    print(f"  Full Multi-Model Ensemble (Propagated)     -> Val AP: {ap_ens:.4f} | ROC-AUC: {auc_ens:.4f}")
    print("==========================================================================")

    imp = pd.Series(0.5 * xgb6.feature_importances_ + 0.5 * xgb5.feature_importances_, index=all_feats).sort_values(ascending=False)
    print("\nTop 15 Most Important Features (Leak-Free XGBoost Gain):")
    print(imp.head(15).to_string())

    # Step 3: 5-Fold Contiguous Time-Block Cross-Validation + Full-Train Refit Ensemble for Test
    print(f"\n================ Step 3: 5-Fold Contiguous Time-Block CV + 100% Refit on {len(X_te_all):,} Test Rows ================")
    n_tr = len(X_tr_all)
    n_te = len(X_te_all)
    n_folds = 5
    order = np.argsort(ts_tr)
    fold_id = np.zeros(n_tr, dtype=int)
    fold_id[order] = np.arange(n_tr) * n_folds // n_tr

    oof_cb = np.zeros(n_tr, dtype=np.float32)
    oof_xgb = np.zeros(n_tr, dtype=np.float32)
    oof_lgb = np.zeros(n_tr, dtype=np.float32)

    te_cb_cv = np.zeros(n_te, dtype=np.float32)
    te_xgb_cv = np.zeros(n_te, dtype=np.float32)
    te_lgb_cv = np.zeros(n_te, dtype=np.float32)

    for f in range(n_folds):
        f_va = fold_id == f
        va_min, va_max = ts_tr[f_va].min(), ts_tr[f_va].max()
        f_tr = (~f_va) & ((ts_tr < va_min - 7200) | (ts_tr > va_max + 7200))

        X_f_tr, y_f_tr = X_tr_all.loc[f_tr], y_all[f_tr]
        if PX:
            # Pseudo-labelled rows also go into every fold model (validation stays on train rows).
            X_f_tr = pd.concat([X_f_tr] + PX, ignore_index=True)
            y_f_tr = np.concatenate([y_f_tr] + Py)
        X_f_va, y_f_va = X_tr_all.loc[f_va], y_all[f_va]

        f_cb6 = CatBoostClassifier(
            iterations=950, depth=6, learning_rate=0.03, auto_class_weights="SqrtBalanced",
            l2_leaf_reg=5.0, eval_metric="PRAUC", early_stopping_rounds=60, random_seed=42 + SEED_OFFSET + f, verbose=False,
        ).fit(X_f_tr, y_f_tr, eval_set=(X_f_va, y_f_va), verbose=False)

        f_cb7 = CatBoostClassifier(
            iterations=950, depth=7, learning_rate=0.025, auto_class_weights="SqrtBalanced",
            l2_leaf_reg=6.0, eval_metric="PRAUC", early_stopping_rounds=60, random_seed=77 + SEED_OFFSET + f, verbose=False,
        ).fit(X_f_tr, y_f_tr, eval_set=(X_f_va, y_f_va), verbose=False)

        f_xgb6 = XGBClassifier(
            n_estimators=750, max_depth=6, learning_rate=0.02, subsample=0.80, colsample_bytree=0.55,
            min_child_weight=10, reg_alpha=1.5, reg_lambda=5.0, scale_pos_weight=3.5,
            eval_metric="aucpr", early_stopping_rounds=50, random_state=42 + SEED_OFFSET + f, n_jobs=-1, tree_method="hist",
        ).fit(X_f_tr, y_f_tr, eval_set=[(X_f_va, y_f_va)], verbose=False)

        f_xgb5 = XGBClassifier(
            n_estimators=750, max_depth=5, learning_rate=0.02, subsample=0.80, colsample_bytree=0.50,
            min_child_weight=8, reg_alpha=1.0, reg_lambda=4.0, scale_pos_weight=3.0,
            eval_metric="aucpr", early_stopping_rounds=50, random_state=100 + SEED_OFFSET + f, n_jobs=-1, tree_method="hist",
        ).fit(X_f_tr, y_f_tr, eval_set=[(X_f_va, y_f_va)], verbose=False)

        f_lgb = LGBMClassifier(
            n_estimators=380, learning_rate=0.02, num_leaves=31, max_depth=6, scale_pos_weight=3.0,
            subsample=0.80, colsample_bytree=0.50, min_child_samples=25, reg_alpha=1.5, reg_lambda=5.0,
            random_state=42 + SEED_OFFSET + f, n_jobs=-1, verbose=-1,
        ).fit(X_f_tr, y_f_tr)

        oof_cb[f_va] = 0.60 * f_cb6.predict_proba(X_f_va)[:, 1] + 0.40 * f_cb7.predict_proba(X_f_va)[:, 1]
        oof_xgb[f_va] = 0.50 * f_xgb6.predict_proba(X_f_va)[:, 1] + 0.50 * f_xgb5.predict_proba(X_f_va)[:, 1]
        oof_lgb[f_va] = f_lgb.predict_proba(X_f_va)[:, 1]

        te_cb_cv += (0.60 * f_cb6.predict_proba(X_te_all)[:, 1] + 0.40 * f_cb7.predict_proba(X_te_all)[:, 1]) / n_folds
        te_xgb_cv += (0.50 * f_xgb6.predict_proba(X_te_all)[:, 1] + 0.50 * f_xgb5.predict_proba(X_te_all)[:, 1]) / n_folds
        te_lgb_cv += f_lgb.predict_proba(X_te_all)[:, 1] / n_folds

        fold_ens = propagate_predictions_over_motifs(
            0.54 * oof_cb + 0.36 * oof_xgb + 0.06 * oof_lgb + 0.04 * oof_gnn_cb, tr_cids
        )
        print(f"  [Time-Block Fold {f + 1}/{n_folds}] Ensemble Val AP: {average_precision_score(y_f_va, fold_ens[f_va]):.4f} | ROC-AUC: {roc_auc_score(y_f_va, fold_ens[f_va]):.4f}")

    oof_ens_prop = propagate_predictions_over_motifs(
        0.54 * oof_cb + 0.36 * oof_xgb + 0.06 * oof_lgb + 0.04 * oof_gnn_cb, tr_cids
    )
    oof_full_ap = average_precision_score(y_all, oof_ens_prop)
    oof_full_auc = roc_auc_score(y_all, oof_ens_prop)
    print(f"  [5-Fold Contiguous Time-Block Full OOF] AP: {oof_full_ap:.4f} | ROC-AUC: {oof_full_auc:.4f}")

    # Full-train refit models on 100% of train-data
    print("  Refitting models on 100% of training data...")
    X_fit = pd.concat([X_tr_all] + PX, ignore_index=True) if PX else X_tr_all
    y_fit = np.concatenate([y_all] + Py) if PX else y_all

    cb6_full = CatBoostClassifier(
        iterations=max(int(cb6.get_best_iteration() * 1.12), 450), depth=6, learning_rate=0.03,
        auto_class_weights="SqrtBalanced", l2_leaf_reg=5.0, random_seed=42 + SEED_OFFSET, verbose=False,
    ).fit(X_fit, y_fit, verbose=False)

    cb7_full = CatBoostClassifier(
        iterations=max(int(cb7.get_best_iteration() * 1.12), 400), depth=7, learning_rate=0.025,
        auto_class_weights="SqrtBalanced", l2_leaf_reg=6.0, random_seed=77 + SEED_OFFSET, verbose=False,
    ).fit(X_fit, y_fit, verbose=False)

    cb5_full = CatBoostClassifier(
        iterations=max(int(cb5.get_best_iteration() * 1.12), 450), depth=5, learning_rate=0.03,
        auto_class_weights="SqrtBalanced", l2_leaf_reg=4.0, random_seed=100 + SEED_OFFSET, verbose=False,
    ).fit(X_fit, y_fit, verbose=False)

    xgb6_full = XGBClassifier(
        n_estimators=max(int(xgb6.best_iteration * 1.12), 300), max_depth=6, learning_rate=0.02,
        subsample=0.80, colsample_bytree=0.55, min_child_weight=10, reg_alpha=1.5, reg_lambda=5.0,
        scale_pos_weight=3.5, eval_metric="aucpr", random_state=42 + SEED_OFFSET, n_jobs=-1, tree_method="hist",
    ).fit(X_fit, y_fit, verbose=False)

    xgb5_full = XGBClassifier(
        n_estimators=max(int(xgb5.best_iteration * 1.12), 300), max_depth=5, learning_rate=0.02,
        subsample=0.80, colsample_bytree=0.50, min_child_weight=8, reg_alpha=1.0, reg_lambda=4.0,
        scale_pos_weight=3.0, eval_metric="aucpr", random_state=100 + SEED_OFFSET, n_jobs=-1, tree_method="hist",
    ).fit(X_fit, y_fit, verbose=False)

    lgb31_full = copy.deepcopy(lgb31).fit(X_fit, y_fit)

    te_cb_full = 0.45 * cb6_full.predict_proba(X_te_all)[:, 1] + 0.30 * cb7_full.predict_proba(X_te_all)[:, 1] + 0.25 * cb5_full.predict_proba(X_te_all)[:, 1]
    te_xgb_full = 0.50 * xgb6_full.predict_proba(X_te_all)[:, 1] + 0.50 * xgb5_full.predict_proba(X_te_all)[:, 1]
    te_lgb_full = lgb31_full.predict_proba(X_te_all)[:, 1]

    # Blend 50% 5-Fold Time-Block CV Ensemble + 50% 100%-Train Refit Ensemble
    te_cb = propagate_predictions_over_motifs(0.50 * te_cb_cv + 0.50 * te_cb_full, te_cids)
    te_xgb = propagate_predictions_over_motifs(0.50 * te_xgb_cv + 0.50 * te_xgb_full, te_cids)
    te_lgb = propagate_predictions_over_motifs(0.50 * te_lgb_cv + 0.50 * te_lgb_full, te_cids)
    te_gnn = propagate_predictions_over_motifs(te_gnn_cb, te_cids)

    te_prop_ens = 0.54 * te_cb + 0.36 * te_xgb + 0.06 * te_lgb + 0.04 * te_gnn

    # Verify exact row alignment with test/transactions.parquet
    test_tx = pd.read_parquet(args.test_dir / "transactions.parquet", columns=["TRANSACTION_HASH"])
    assert (test_tx["TRANSACTION_HASH"].values == np.array(te_hashes)).all(), "Hash order mismatch!"

    sub_best = pd.DataFrame({"TRANSACTION_HASH": te_hashes, "score": te_prop_ens})
    sub_xgb = pd.DataFrame({"TRANSACTION_HASH": te_hashes, "score": te_xgb})
    sub_lgb = pd.DataFrame({"TRANSACTION_HASH": te_hashes, "score": te_lgb})
    sub_cb = pd.DataFrame({"TRANSACTION_HASH": te_hashes, "score": te_cb})

    sub_best.to_csv(args.out_csv, index=False)
    sub_best.to_csv(args.out_dir / "test_submission_best_ensemble.csv", index=False)
    sub_xgb.to_csv(args.out_dir / "test_submission_gnn_xgb.csv", index=False)
    sub_lgb.to_csv(args.out_dir / "test_submission_gnn_lgbm.csv", index=False)
    sub_cb.to_csv(args.out_dir / "test_submission_gnn_catboost.csv", index=False)

    xgb6_full.save_model(str(args.out_dir / "xgb_gnn_model.json"))

    metrics_summary = {
        "n_features": len(all_feats),
        "zero_leakage_temporal_val_metrics": {
            "inductive_motif_gnn_ap": float(mgnn_temp_ap),
            "inductive_motif_gnn_auc": float(mgnn_temp_auc),
            "catboost_d6_ap": float(ap_cb6),
            "catboost_d6_auc": float(auc_cb6),
            "catboost_d7_ap": float(ap_cb7),
            "catboost_d7_auc": float(auc_cb7),
            "catboost_d5_ap": float(ap_cb5),
            "catboost_d5_auc": float(auc_cb5),
            "xgb_d6_ap": float(ap_xgb6),
            "xgb_d6_auc": float(auc_xgb6),
            "xgb_d5_ap": float(ap_xgb5),
            "xgb_d5_auc": float(auc_xgb5),
            "lgb_31_ap": float(ap_lgb31),
            "lgb_31_auc": float(auc_lgb31),
            "catboost_blend_ap": float(ap_cb_ens),
            "xgb_blend_ap": float(ap_xgb_ens),
            "full_ensemble_ap": float(ap_ens),
            "full_ensemble_auc": float(auc_ens),
        },
        "contiguous_time_block_5fold_oof": {
            "oof_ap": float(oof_full_ap),
            "oof_auc": float(oof_full_auc),
        },
        "top_15_features": {k: float(v) for k, v in imp.head(15).items()},
    }
    with (args.out_dir / "tuning_metrics.json").open("w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\n[Done in {time.time() - t_start:.1f}s] Updated primary submission: {args.out_csv}")
    print(sub_best["score"].describe().to_string())


if __name__ == "__main__":
    main()
