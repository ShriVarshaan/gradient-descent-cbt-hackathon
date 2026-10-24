"""End-to-end Pipeline:
1. Build / Load Bipartite + Temporal PyG Graphs (`train.pt`, `test.pt`).
2. Tune Heterogeneous GNN (`HeteroTxGNN`) architecture & training hyperparameters on the temporal split.
3. Train 5-Fold Tuned GNN to produce leak-free Out-Of-Fold (OOF) predictions, structural embeddings,
   and 1-hop / 2-hop graph-propagated features.
4. Create & Tune an XGBoost model (`XGBClassifier`) and LightGBM model (`LGBMClassifier`) using the
   GNN outputs as features, compare performance, and export models + test predictions.

Usage:
    # Full run (GNN tuning + 5-fold OOF + XGBoost/LightGBM tuning):
    python gnn/train_gnn_xgb.py --train-dir train-data --test-dir validation --out-dir outputs

    # Fast re-tuning of XGBoost/LightGBM using cached 5-fold GNN embeddings:
    python gnn/train_gnn_xgb.py --train-dir train-data --test-dir validation --out-dir outputs --reuse-gnn
"""
import argparse
import copy
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import scatter
from xgboost import XGBClassifier

from build_graph import build as build_graph_from_dir
from model import HeteroTxGNN, augment_hetero_graph

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_build_graphs(train_dir: Path, test_dir: Path | None, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_pt = cache_dir / "train.pt"
    if train_pt.exists():
        print(f"[Graph] Loading cached train graph from {train_pt}")
        train_bundle = torch.load(train_pt, weights_only=False)
    else:
        print(f"[Graph] Building train graph from {train_dir}...")
        data, stats, names = build_graph_from_dir(train_dir, stats=None)
        train_bundle = {"data": data, "stats": stats, "names": names}
        torch.save(train_bundle, train_pt)

    test_bundle = None
    if test_dir is not None and test_dir.exists():
        test_pt = cache_dir / "test.pt"
        if test_pt.exists():
            print(f"[Graph] Loading cached test graph from {test_pt}")
            test_bundle = torch.load(test_pt, weights_only=False)
        else:
            print(f"[Graph] Building test graph from {test_dir}...")
            data_t, _, names_t = build_graph_from_dir(test_dir, stats=train_bundle["stats"])
            test_bundle = {"data": data_t, "stats": train_bundle["stats"], "names": names_t}
            torch.save(test_bundle, test_pt)

    # Augment graphs with temporal burst/fan-in/fan-out node features & temporal tx<->tx edges
    t0 = time.time()
    aug_train_data, aug_stats = augment_hetero_graph(train_bundle["data"], aug_stats=None)
    train_bundle["data"] = aug_train_data
    train_bundle["names"]["tx"] = train_bundle["names"]["tx"] + aug_stats["names"]

    if test_bundle is not None:
        aug_test_data, _ = augment_hetero_graph(test_bundle["data"], aug_stats=aug_stats)
        test_bundle["data"] = aug_test_data
        test_bundle["names"]["tx"] = test_bundle["names"]["tx"] + aug_stats["names"]

    n_temp_edges = aug_train_data["tx", "temporal", "tx"].edge_index.shape[1]
    print(
        f"[Graph] Augmented graphs in {time.time() - t0:.2f}s | "
        f"tx features: {aug_train_data['tx'].x.shape[1]} | "
        f"temporal tx<->tx edges: {n_temp_edges:,}"
    )
    return train_bundle, test_bundle


def train_single_gnn(
    data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    hidden_dim: int = 64,
    emb_dim: int = 16,
    num_layers: int = 2,
    dropout: float = 0.25,
    lr: float = 3e-3,
    weight_decay: float = 2e-4,
    pos_weight_power: float = 0.5,
    epochs: int = 36,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
):
    set_seed(seed)
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    model = HeteroTxGNN(
        tx_in_dim=data["tx"].x.shape[1],
        acc_in_dim=data["account"].x.shape[1],
        bipartite_edge_dim=data["account", "sends", "tx"].edge_attr.shape[1],
        temporal_edge_dim=data["tx", "temporal", "tx"].edge_attr.shape[1],
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    y = data["tx"].y
    n_pos = y[train_mask].sum().clamp(min=1.0)
    n_neg = (train_mask.sum() - n_pos).clamp(min=1.0)
    pos_weight = ((n_neg / n_pos) ** pos_weight_power).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-4)

    best_ap = -1.0
    best_auc = 0.0
    best_state = None
    y_val_np = y[val_mask].detach().cpu().numpy()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        if epoch % 3 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(data)
                val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
                ap = average_precision_score(y_val_np, val_probs)
                auc = roc_auc_score(y_val_np, val_probs)
                if ap > best_ap:
                    best_ap = ap
                    best_auc = auc
                    best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ap, best_auc


@torch.no_grad()
def infer_gnn(model: HeteroTxGNN, data, device: torch.device):
    model.eval()
    data = data.to(device)
    logits, emb = model(data)
    probs = torch.sigmoid(logits)
    return (
        logits.cpu().numpy(),
        probs.cpu().numpy(),
        emb.cpu().numpy(),
    )


def tune_gnn_hyperparameters(data, device: torch.device, epochs: int = 30, seed: int = 42):
    """Grid/candidate search over GNN hyperparameters on the temporal validation split."""
    print("\n================ Step 1: Tuning GNN Hyperparameters ================")
    train_mask = data["tx"].train_mask
    val_mask = data["tx"].val_mask

    candidates = [
        {"hidden_dim": 48, "emb_dim": 16, "num_layers": 2, "dropout": 0.15, "lr": 4e-3, "pos_weight_power": 0.4},
        {"hidden_dim": 64, "emb_dim": 16, "num_layers": 2, "dropout": 0.25, "lr": 3e-3, "pos_weight_power": 0.5},
        {"hidden_dim": 64, "emb_dim": 24, "num_layers": 3, "dropout": 0.20, "lr": 3e-3, "pos_weight_power": 0.35},
    ]

    best_cfg = candidates[0]
    best_ap = -1.0
    best_model = None

    for idx, cfg in enumerate(candidates, 1):
        t0 = time.time()
        model, ap, auc = train_single_gnn(
            data=data,
            train_mask=train_mask,
            val_mask=val_mask,
            hidden_dim=cfg["hidden_dim"],
            emb_dim=cfg["emb_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            lr=cfg["lr"],
            pos_weight_power=cfg["pos_weight_power"],
            epochs=epochs,
            device=device,
            seed=seed + idx,
        )
        elapsed = time.time() - t0
        print(
            f"  [Trial {idx}/{len(candidates)}] layers={cfg['num_layers']} hidden={cfg['hidden_dim']} "
            f"emb={cfg['emb_dim']} drop={cfg['dropout']} pw={cfg['pos_weight_power']} "
            f"-> Temporal Val AP: {ap:.4f} | ROC-AUC: {auc:.4f} ({elapsed:.1f}s)"
        )
        if ap > best_ap:
            best_ap = ap
            best_cfg = cfg
            best_model = model

    print(f"[GNN Tuning] Best Config: {best_cfg} with Temporal Val AP = {best_ap:.4f}")
    return best_cfg, best_model


def compute_graph_and_gnn_neighborhood_features(
    data, df_base: pd.DataFrame, gnn_probs: np.ndarray, gnn_logits: np.ndarray
) -> pd.DataFrame:
    """Propagates out-of-fold GNN probabilities AND key structural burst/corridor signals across:
      1. 1-hop and 2-hop Temporal tx<->tx neighborhoods (mean & max)
      2. Primary sender and receiver account neighborhoods (leave-one-out mean & max)
      3. Explicit 1-hop sender and receiver account feature vectors
    """
    n_tx = data["tx"].x.shape[0]
    n_acc = data["account"].x.shape[0]
    x_acc = data["account"].x.cpu().numpy()
    sender_id = data["tx"].sender_id.cpu()
    dest_id = data["tx"].dest_id.cpu()

    prob_t = torch.tensor(gnn_probs, dtype=torch.float)
    logit_t = torch.tensor(gnn_logits, dtype=torch.float)

    # 1. Temporal tx<->tx neighborhood GNN score aggregations
    ei_temp = data["tx", "temporal", "tx"].edge_index.cpu()
    ea_temp = data["tx", "temporal", "tx"].edge_attr.cpu()
    src, dst = ei_temp[0], ei_temp[1]

    time_w = (1.0 - 0.5 * ea_temp[:, 0]).clamp(min=0.2)
    temp_prob_mean = scatter(prob_t[src], dst, dim=0, dim_size=n_tx, reduce="mean")
    temp_prob_wmean = scatter(prob_t[src] * time_w, dst, dim=0, dim_size=n_tx, reduce="sum") / scatter(
        time_w, dst, dim=0, dim_size=n_tx, reduce="sum"
    ).clamp(min=1e-6)
    temp_prob_max = scatter(prob_t[src], dst, dim=0, dim_size=n_tx, reduce="max")
    temp_logit_mean = scatter(logit_t[src], dst, dim=0, dim_size=n_tx, reduce="mean")
    temp_logit_max = scatter(logit_t[src], dst, dim=0, dim_size=n_tx, reduce="max")
    temp_deg = scatter(torch.ones_like(src, dtype=torch.float), dst, dim=0, dim_size=n_tx, reduce="sum")

    # 2-hop temporal smoothing of GNN probability
    hop2_prob_mean = scatter(temp_prob_mean[src], dst, dim=0, dim_size=n_tx, reduce="mean")
    hop2_prob_max = scatter(temp_prob_max[src], dst, dim=0, dim_size=n_tx, reduce="max")

    # 2. Leave-one-out (LOO) account-level GNN probability aggregations
    s_sum = scatter(prob_t, sender_id, dim=0, dim_size=n_acc, reduce="sum")
    s_cnt = scatter(torch.ones_like(prob_t), sender_id, dim=0, dim_size=n_acc, reduce="sum")
    s_loo_mean = (s_sum[sender_id] - prob_t) / (s_cnt[sender_id] - 1.0).clamp(min=1.0)
    s_max = scatter(prob_t, sender_id, dim=0, dim_size=n_acc, reduce="max")[sender_id]

    d_sum = scatter(prob_t, dest_id, dim=0, dim_size=n_acc, reduce="sum")
    d_cnt = scatter(torch.ones_like(prob_t), dest_id, dim=0, dim_size=n_acc, reduce="sum")
    d_loo_mean = (d_sum[dest_id] - prob_t) / (d_cnt[dest_id] - 1.0).clamp(min=1.0)
    d_max = scatter(prob_t, dest_id, dim=0, dim_size=n_acc, reduce="max")[dest_id]

    dest_as_sender_mean = (s_sum / s_cnt.clamp(min=1.0))[dest_id]
    sender_as_dest_mean = (d_sum / d_cnt.clamp(min=1.0))[sender_id]

    cols = {
        "gnn_temp_prob_mean": temp_prob_mean.numpy(),
        "gnn_temp_prob_wmean": temp_prob_wmean.numpy(),
        "gnn_temp_prob_max": temp_prob_max.numpy(),
        "gnn_temp_logit_mean": temp_logit_mean.numpy(),
        "gnn_temp_logit_max": temp_logit_max.numpy(),
        "gnn_temp_log_deg": np.log1p(temp_deg.numpy()),
        "gnn_hop2_prob_mean": hop2_prob_mean.numpy(),
        "gnn_hop2_prob_max": hop2_prob_max.numpy(),
        "gnn_sender_loo_prob_mean": s_loo_mean.numpy(),
        "gnn_sender_prob_max": s_max.numpy(),
        "gnn_dest_loo_prob_mean": d_loo_mean.numpy(),
        "gnn_dest_prob_max": d_max.numpy(),
        "gnn_dest_as_sender_prob": dest_as_sender_mean.numpy(),
        "gnn_sender_as_dest_prob": sender_as_dest_mean.numpy(),
        "gnn_prob_x_temp_mean": gnn_probs * temp_prob_mean.numpy(),
        "gnn_prob_plus_temp_wmean": 0.5 * gnn_probs + 0.5 * temp_prob_wmean.numpy(),
    }

    # 3. Primary sender & receiver account features
    s_np = sender_id.numpy()
    d_np = dest_id.numpy()
    for j in range(x_acc.shape[1]):
        cols[f"gnn_sender_acc_f{j}"] = x_acc[s_np, j]
        cols[f"gnn_dest_acc_f{j}"] = x_acc[d_np, j]

    # 4. 1-hop & 2-hop graph propagation of key burst/corridor signals + GNN probability
    key_cols = [
        "tx_burst_combined_10m",
        "tx_log_pair_cnt",
        "tx_log_sender_10m",
        "tx_log_dest_10m",
        "tx_log_dest_out_10m",
        "tx_log_sender_in_10m",
    ]
    X_key = torch.tensor(
        np.column_stack([df_base[key_cols].values, gnn_probs[:, None]]),
        dtype=torch.float,
    )
    key_names = key_cols + ["gnn_prob"]

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

    for i, c in enumerate(key_names):
        cols[f"gnn_p1_mean_{c}"] = t_mean[:, i].numpy()
        cols[f"gnn_p1_max_{c}"] = t_max[:, i].numpy()
        cols[f"gnn_p2_mean_{c}"] = h2_mean[:, i].numpy()
        cols[f"gnn_p2_max_{c}"] = h2_max[:, i].numpy()
        cols[f"gnn_sloo_{c}"] = sk_loo[:, i].numpy()
        cols[f"gnn_smax_{c}"] = sk_amax[:, i].numpy()
        cols[f"gnn_dloo_{c}"] = dk_loo[:, i].numpy()
        cols[f"gnn_dmax_{c}"] = dk_amax[:, i].numpy()

    return pd.DataFrame(cols, dtype=np.float32)


def generate_oof_gnn_features(
    train_bundle: dict,
    test_bundle: dict | None,
    best_cfg: dict,
    n_folds: int = 5,
    epochs: int = 36,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
):
    """Generates leak-free Out-Of-Fold (OOF) GNN predictions & embeddings for train,
    and ensemble-averaged GNN predictions & embeddings for test/validation.
    """
    print(f"\n================ Step 2: {n_folds}-Fold OOF GNN Feature Extraction ================")
    data = train_bundle["data"]
    n_tx = data["tx"].x.shape[0]
    y_np = data["tx"].y.cpu().numpy().astype(int)
    emb_dim = best_cfg["emb_dim"]

    oof_logits = np.zeros(n_tx, dtype=np.float32)
    oof_probs = np.zeros(n_tx, dtype=np.float32)
    oof_emb = np.zeros((n_tx, emb_dim), dtype=np.float32)

    test_logits_accum = None
    test_probs_accum = None
    test_emb_accum = None
    if test_bundle is not None:
        n_test = test_bundle["data"]["tx"].x.shape[0]
        test_logits_accum = np.zeros(n_test, dtype=np.float32)
        test_probs_accum = np.zeros(n_test, dtype=np.float32)
        test_emb_accum = np.zeros((n_test, emb_dim), dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_tx), y_np)):
        t0 = time.time()
        tr_mask = torch.zeros(n_tx, dtype=torch.bool)
        va_mask = torch.zeros(n_tx, dtype=torch.bool)
        tr_mask[tr_idx] = True
        va_mask[va_idx] = True

        model, val_ap, val_auc = train_single_gnn(
            data=data,
            train_mask=tr_mask,
            val_mask=va_mask,
            hidden_dim=best_cfg["hidden_dim"],
            emb_dim=best_cfg["emb_dim"],
            num_layers=best_cfg["num_layers"],
            dropout=best_cfg["dropout"],
            lr=best_cfg["lr"],
            pos_weight_power=best_cfg["pos_weight_power"],
            epochs=epochs,
            device=device,
            seed=seed + 100 + fold,
        )
        print(f"  [GNN Fold {fold + 1}/{n_folds}] Val AP: {val_ap:.4f} | Val ROC-AUC: {val_auc:.4f} ({time.time() - t0:.1f}s)")

        logits_all, probs_all, emb_all = infer_gnn(model, data, device)
        oof_logits[va_idx] = logits_all[va_idx]
        oof_probs[va_idx] = probs_all[va_idx]
        oof_emb[va_idx] = emb_all[va_idx]

        if test_bundle is not None:
            t_logits, t_probs, t_emb = infer_gnn(model, test_bundle["data"], device)
            test_logits_accum += t_logits / n_folds
            test_probs_accum += t_probs / n_folds
            test_emb_accum += t_emb / n_folds

    oof_ap = average_precision_score(y_np, oof_probs)
    oof_auc = roc_auc_score(y_np, oof_probs)
    print(f"[GNN Full OOF] Overall AP: {oof_ap:.4f} | Overall ROC-AUC: {oof_auc:.4f}")

    def pack_dataframe(bundle, gnn_logits, gnn_probs, gnn_emb):
        d = bundle["data"]
        tx_feat_names = [f"tx_{c}" for c in bundle["names"]["tx"]]
        df_base = pd.DataFrame(d["tx"].x.cpu().numpy(), columns=tx_feat_names)
        df_base.insert(0, "TRANSACTION_HASH", d["tx"].hash)
        df_base["TIMESTAMP"] = d["tx"].time.cpu().numpy()

        df_gnn = pd.DataFrame(
            {
                "gnn_logit": gnn_logits,
                "gnn_prob": gnn_probs,
                **{f"gnn_emb_{i}": gnn_emb[:, i] for i in range(emb_dim)},
            },
            dtype=np.float32,
        )
        df_neigh = compute_graph_and_gnn_neighborhood_features(d, df_base, gnn_probs, gnn_logits)
        df_out = pd.concat([df_base, df_gnn, df_neigh], axis=1)
        if "y" in d["tx"]:
            df_out["is_suspicious"] = d["tx"].y.cpu().numpy().astype(int)
        return df_out

    train_df = pack_dataframe(train_bundle, oof_logits, oof_probs, oof_emb)
    test_df = (
        pack_dataframe(test_bundle, test_logits_accum, test_probs_accum, test_emb_accum)
        if test_bundle is not None
        else None
    )
    return train_df, test_df


def rebuild_propagated_from_cached(train_bundle: dict, test_bundle: dict | None, out_dir: Path):
    """Re-computes all 1-hop & 2-hop graph-propagated features from cached GNN OOF outputs."""
    tr_cache = pd.read_parquet(out_dir / "train_full_with_gnn.parquet")
    te_cache = pd.read_parquet(out_dir / "test_full_with_gnn.parquet") if (out_dir / "test_full_with_gnn.parquet").exists() else None

    emb_cols = [c for c in tr_cache.columns if c.startswith("gnn_emb_")]

    def repack(bundle, cached_df):
        d = bundle["data"]
        tx_feat_names = [f"tx_{c}" for c in bundle["names"]["tx"]]
        df_base = pd.DataFrame(d["tx"].x.cpu().numpy(), columns=tx_feat_names)
        df_base.insert(0, "TRANSACTION_HASH", d["tx"].hash)
        df_base["TIMESTAMP"] = d["tx"].time.cpu().numpy()

        gnn_logits = cached_df["gnn_logit"].values
        gnn_probs = cached_df["gnn_prob"].values
        df_gnn = cached_df[["gnn_logit", "gnn_prob"] + emb_cols].copy()
        df_neigh = compute_graph_and_gnn_neighborhood_features(d, df_base, gnn_probs, gnn_logits)
        df_out = pd.concat([df_base, df_gnn, df_neigh], axis=1)
        if "y" in d["tx"]:
            df_out["is_suspicious"] = d["tx"].y.cpu().numpy().astype(int)
        return df_out

    return repack(train_bundle, tr_cache), (repack(test_bundle, te_cache) if te_cache is not None and test_bundle is not None else None)


def tune_and_evaluate_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame | None, out_dir: Path):
    """Tunes XGBoost using the GNN features + tabular features on the temporal split,
    compares against Tabular-only XGBoost and GNN + LightGBM, and exports models & predictions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ignore_cols = {"TRANSACTION_HASH", "TIMESTAMP", "is_suspicious"}
    all_feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    tabular_only_cols = [c for c in all_feature_cols if not c.startswith("gnn_")]
    gnn_only_cols = [c for c in all_feature_cols if c.startswith("gnn_")]

    # Temporal split: first 80% train, last 20% val (with 1-hour embargo)
    cut_ts = np.quantile(train_df["TIMESTAMP"].values, 0.80)
    embargo = 3600
    tr_mask = train_df["TIMESTAMP"].values <= (cut_ts - embargo)
    va_mask = train_df["TIMESTAMP"].values > cut_ts

    X_tr_all, y_tr = train_df.loc[tr_mask, all_feature_cols], train_df.loc[tr_mask, "is_suspicious"].values
    X_va_all, y_va = train_df.loc[va_mask, all_feature_cols], train_df.loc[va_mask, "is_suspicious"].values

    X_tr_tab = train_df.loc[tr_mask, tabular_only_cols]
    X_va_tab = train_df.loc[va_mask, tabular_only_cols]

    # 1. Baseline Tabular-Only XGBoost (without GNN features)
    xgb_base = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb_base.fit(X_tr_tab, y_tr, eval_set=[(X_va_tab, y_va)], verbose=False)
    preds_xgb_tab = xgb_base.predict_proba(X_va_tab)[:, 1]
    ap_xgb_tab = average_precision_score(y_va, preds_xgb_tab)
    auc_xgb_tab = roc_auc_score(y_va, preds_xgb_tab)

    # 2. Tune XGBoost with GNN Features
    print("\n================ Step 3: Tuning XGBoost with GNN Features ================")
    xgb_candidates = [
        dict(max_depth=4, learning_rate=0.02, subsample=0.75, colsample_bytree=0.60, min_child_weight=8, reg_alpha=1.0, reg_lambda=4.0, scale_pos_weight=2.0),
        dict(max_depth=5, learning_rate=0.02, subsample=0.80, colsample_bytree=0.60, min_child_weight=10, reg_alpha=1.5, reg_lambda=5.0, scale_pos_weight=3.0),
        dict(max_depth=6, learning_rate=0.02, subsample=0.80, colsample_bytree=0.60, min_child_weight=10, reg_alpha=1.5, reg_lambda=5.0, scale_pos_weight=3.0),
        dict(max_depth=6, learning_rate=0.02, subsample=0.80, colsample_bytree=0.60, min_child_weight=10, reg_alpha=1.5, reg_lambda=5.0, scale_pos_weight=4.0),
        dict(max_depth=7, learning_rate=0.02, subsample=0.80, colsample_bytree=0.60, min_child_weight=10, reg_alpha=1.5, reg_lambda=5.0, scale_pos_weight=4.0),
        dict(max_depth=7, learning_rate=0.02, subsample=0.80, colsample_bytree=0.60, min_child_weight=10, reg_alpha=1.5, reg_lambda=5.0, scale_pos_weight=5.0),
    ]

    best_xgb_ap = -1.0
    best_xgb_auc = 0.0
    best_xgb_cfg = None
    best_xgb_model = None
    best_xgb_preds = None

    for idx, cfg in enumerate(xgb_candidates, 1):
        xgb = XGBClassifier(
            n_estimators=800,
            eval_metric="aucpr",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            **cfg,
        )
        xgb.fit(X_tr_all, y_tr, eval_set=[(X_va_all, y_va)], verbose=False)
        preds = xgb.predict_proba(X_va_all)[:, 1]
        ap = average_precision_score(y_va, preds)
        auc = roc_auc_score(y_va, preds)
        print(
            f"  [XGB Trial {idx}/{len(xgb_candidates)}] depth={cfg['max_depth']} lr={cfg['learning_rate']} "
            f"mcw={cfg['min_child_weight']} spw={cfg['scale_pos_weight']} (best_iter={xgb.best_iteration}) "
            f"-> Val AP: {ap:.4f} | ROC-AUC: {auc:.4f}"
        )
        if ap > best_xgb_ap:
            best_xgb_ap = ap
            best_xgb_auc = auc
            best_xgb_cfg = {**cfg, "best_n_estimators": int(xgb.best_iteration + 1)}
            best_xgb_model = xgb
            best_xgb_preds = preds

    print(f"[XGB Tuning] Best Config: {best_xgb_cfg}")

    # 3. Also fit Tuned Hybrid GNN + LightGBM for comparison & ensembling
    lgb_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02,
        num_leaves=47,
        max_depth=7,
        scale_pos_weight=3.5,
        subsample_for_bin=50000,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_samples=25,
        reg_alpha=1.5,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_tr_all, y_tr)
    preds_lgb = lgb_model.predict_proba(X_va_all)[:, 1]
    ap_lgb = average_precision_score(y_va, preds_lgb)
    auc_lgb = roc_auc_score(y_va, preds_lgb)

    # 4. Blend of Tuned XGBoost + Tuned LightGBM
    preds_blend = 0.60 * best_xgb_preds + 0.40 * preds_lgb
    ap_blend = average_precision_score(y_va, preds_blend)
    auc_blend = roc_auc_score(y_va, preds_blend)

    print("\n================ Final Model Comparison (Temporal Val Split) ================")
    print(f"1. Tabular-Only XGBoost ({len(tabular_only_cols):3d} feats)       -> Val AP: {ap_xgb_tab:.4f} | ROC-AUC: {auc_xgb_tab:.4f}")
    print(f"2. Tuned GNN + LightGBM ({len(all_feature_cols):3d} feats)      -> Val AP: {ap_lgb:.4f} | ROC-AUC: {auc_lgb:.4f}")
    print(f"3. Tuned GNN + Tuned XGBoost ({len(all_feature_cols):3d} feats) -> Val AP: {best_xgb_ap:.4f} | ROC-AUC: {best_xgb_auc:.4f}")
    print(f"4. GNN + XGBoost + LightGBM Ensemble     -> Val AP: {ap_blend:.4f} | ROC-AUC: {auc_blend:.4f}")
    print("=============================================================================")

    # Feature importance breakdown for Tuned XGBoost
    xgb_imp = pd.Series(best_xgb_model.feature_importances_, index=all_feature_cols).sort_values(ascending=False)
    print("\nTop 15 Features in Tuned XGBoost (Gain Importance):")
    print(xgb_imp.head(15).to_string())

    # Save GNN feature tables
    gnn_export_cols = ["TRANSACTION_HASH"] + gnn_only_cols
    train_gnn_path = out_dir / "train_gnn_features.parquet"
    train_df[gnn_export_cols].to_parquet(train_gnn_path, index=False)
    print(f"\n[Export] Saved train GNN features ({len(gnn_only_cols)} cols) to {train_gnn_path}")

    train_df.to_parquet(out_dir / "train_full_with_gnn.parquet", index=False)

    summary = {
        "metrics_temporal_val": {
            "tabular_only_xgboost_ap": float(ap_xgb_tab),
            "tabular_only_xgboost_auc": float(auc_xgb_tab),
            "gnn_lightgbm_ap": float(ap_lgb),
            "gnn_lightgbm_auc": float(auc_lgb),
            "tuned_gnn_xgboost_ap": float(best_xgb_ap),
            "tuned_gnn_xgboost_auc": float(best_xgb_auc),
            "ensemble_ap": float(ap_blend),
            "ensemble_auc": float(auc_blend),
        },
        "best_xgb_params": best_xgb_cfg,
        "top_15_xgb_features": {k: float(v) for k, v in xgb_imp.head(15).items()},
    }
    with (out_dir / "tuning_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if test_df is not None:
        test_gnn_path = out_dir / "test_gnn_features.parquet"
        test_df[gnn_export_cols].to_parquet(test_gnn_path, index=False)
        test_df.to_parquet(out_dir / "test_full_with_gnn.parquet", index=False)
        print(f"[Export] Saved test GNN features ({len(gnn_only_cols)} cols) to {test_gnn_path}")

        # Refit Tuned XGBoost on 100% of training data using optimal number of trees
        final_n_est = max(int(best_xgb_cfg["best_n_estimators"] * 1.15), 200)
        final_cfg = {k: v for k, v in best_xgb_cfg.items() if k != "best_n_estimators"}
        final_xgb = XGBClassifier(
            n_estimators=final_n_est,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            **final_cfg,
        )
        final_xgb.fit(train_df[all_feature_cols], train_df["is_suspicious"].values, verbose=False)
        final_xgb.save_model(str(out_dir / "xgb_gnn_model.json"))

        xgb_test_scores = final_xgb.predict_proba(test_df[all_feature_cols])[:, 1]
        sub_xgb = pd.DataFrame({
            "TRANSACTION_HASH": test_df["TRANSACTION_HASH"],
            "score": xgb_test_scores,
        })
        sub_xgb_path = out_dir / "submission_gnn_xgb.csv"
        sub_xgb.to_csv(sub_xgb_path, index=False)
        print(f"[Submission] Wrote {len(sub_xgb)} GNN+XGBoost predictions to {sub_xgb_path}")

        # Also refit LightGBM on 100% of training data
        final_lgb = copy.deepcopy(lgb_model)
        final_lgb.fit(train_df[all_feature_cols], train_df["is_suspicious"].values)
        lgb_test_scores = final_lgb.predict_proba(test_df[all_feature_cols])[:, 1]
        sub_lgb = pd.DataFrame({
            "TRANSACTION_HASH": test_df["TRANSACTION_HASH"],
            "score": lgb_test_scores,
        })
        sub_lgb_path = out_dir / "submission_gnn_lgbm.csv"
        sub_lgb.to_csv(sub_lgb_path, index=False)
        print(f"[Submission] Wrote {len(sub_lgb)} GNN+LightGBM predictions to {sub_lgb_path}")

        # Also write Ensemble submission
        ens_test_scores = 0.60 * xgb_test_scores + 0.40 * lgb_test_scores
        sub_ens = pd.DataFrame({
            "TRANSACTION_HASH": test_df["TRANSACTION_HASH"],
            "score": ens_test_scores,
        })
        sub_ens_path = out_dir / "submission_gnn_ensemble.csv"
        sub_ens.to_csv(sub_ens_path, index=False)
        print(f"[Submission] Wrote {len(sub_ens)} Ensemble predictions to {sub_ens_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, default=Path("train-data"), help="Path to labelled training split")
    parser.add_argument("--test-dir", type=Path, default=Path("validation"), help="Path to unlabelled validation/test split")
    parser.add_argument("--graph-cache", type=Path, default=Path("graphs"), help="Directory to cache built PyG .pt graphs")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for exported GNN features and submissions")
    parser.add_argument("--folds", type=int, default=5, help="Number of Stratified CV folds for OOF GNN features")
    parser.add_argument("--tune-epochs", type=int, default=30, help="Epochs per GNN tuning trial")
    parser.add_argument("--oof-epochs", type=int, default=36, help="Epochs per GNN CV fold")
    parser.add_argument("--reuse-gnn", action="store_true", help="Reuse cached OOF GNN embeddings if available")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Running on device: {device}")

    train_bundle, test_bundle = load_or_build_graphs(args.train_dir, args.test_dir, args.graph_cache)

    if args.reuse_gnn and (args.out_dir / "train_full_with_gnn.parquet").exists():
        print("[GNN] Reusing cached 5-fold GNN embeddings and computing full 1-hop/2-hop propagated features...")
        train_df, test_df = rebuild_propagated_from_cached(train_bundle, test_bundle, args.out_dir)
    else:
        best_gnn_cfg, best_gnn_model = tune_gnn_hyperparameters(
            train_bundle["data"], device=device, epochs=args.tune_epochs, seed=args.seed
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"config": best_gnn_cfg, "state_dict": best_gnn_model.state_dict()}, args.out_dir / "gnn_best_model.pt")

        train_df, test_df = generate_oof_gnn_features(
            train_bundle=train_bundle,
            test_bundle=test_bundle,
            best_cfg=best_gnn_cfg,
            n_folds=args.folds,
            epochs=args.oof_epochs,
            seed=args.seed,
            device=device,
        )

    tune_and_evaluate_xgboost(train_df, test_df, args.out_dir)


if __name__ == "__main__":
    main()
