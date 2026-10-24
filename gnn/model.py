"""Heterogeneous + Temporal GNN (Account <-> Transaction + Temporal Tx <-> Tx)
and graph augmentation for GNN -> XGBoost / LightGBM pipelines.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


def extract_primary_endpoints(data):
    """Recover primary customer sender and receiver account IDs for each tx node
    directly from the HeteroData transfer leg edges.
    """
    n_tx = data["tx"].x.shape[0]
    sends_ei = data["account", "sends", "tx"].edge_index.cpu().numpy()
    sends_ea = data["account", "sends", "tx"].edge_attr.cpu().numpy()
    recvs_ei = data["tx", "receives", "account"].edge_index.cpu().numpy()
    recvs_ea = data["tx", "receives", "account"].edge_attr.cpu().numpy()

    is_first = sends_ea[:, 0] == sends_ea[:, 0].min()
    is_last = recvs_ea[:, 1] > 0

    sender = np.zeros(n_tx, dtype=np.int64)
    sender[sends_ei[1, is_first]] = sends_ei[0, is_first]

    dest = np.zeros(n_tx, dtype=np.int64)
    dest[recvs_ei[0, is_last]] = recvs_ei[1, is_last]
    return sender, dest


def _fast_window_counts(query_acc: np.ndarray, query_ts: np.ndarray,
                        event_acc: np.ndarray, event_ts: np.ndarray,
                        half_widths: list[int], n_accounts: int) -> dict[int, np.ndarray]:
    """Fast vectorized window counts per account using sorted arrays."""
    order = np.argsort(event_ts, kind="mergesort")
    e_acc = event_acc[order]
    e_ts = event_ts[order]

    acc_order = np.argsort(e_acc, kind="mergesort")
    e_acc_sorted = e_acc[acc_order]
    e_ts_by_acc = e_ts[acc_order]
    splits = np.searchsorted(e_acc_sorted, np.arange(n_accounts + 1))

    q_order = np.argsort(query_acc, kind="mergesort")
    q_acc_sorted = query_acc[q_order]
    q_ts_sorted = query_ts[q_order]
    q_splits = np.searchsorted(q_acc_sorted, np.arange(n_accounts + 1))

    results = {w: np.zeros(len(query_ts), dtype=np.float32) for w in half_widths}
    for a in range(n_accounts):
        q_start, q_end = q_splits[a], q_splits[a + 1]
        if q_start == q_end:
            continue
        e_start, e_end = splits[a], splits[a + 1]
        if e_start == e_end:
            continue
        t_events = e_ts_by_acc[e_start:e_end]
        t_queries = q_ts_sorted[q_start:q_end]
        orig_idx = q_order[q_start:q_end]
        for w in half_widths:
            hi = np.searchsorted(t_events, t_queries + w, side="right")
            lo = np.searchsorted(t_events, t_queries - w, side="left")
            results[w][orig_idx] = hi - lo
    return results


def augment_hetero_graph(data, aug_stats: dict | None = None):
    """Augment HeteroData in-place (on a shallow clone) with:
    1. High-signal temporal burst, fan-in, fan-out, smurfing, and corridor features on `tx` nodes.
    2. Explicit temporal transaction-to-transaction edges `('tx', 'temporal', 'tx')`
       connecting same-sender bursts, same-receiver fan-ins, and pass-through relays.
    """
    data = data.clone()
    n_tx = data["tx"].x.shape[0]
    n_acc = data["account"].x.shape[0]
    ts = data["tx"].time.cpu().numpy()
    x_tx_np = data["tx"].x.cpu().numpy()
    usd = x_tx_np[:, 0]  # standardized log_usd_first

    sender, dest = extract_primary_endpoints(data)

    # 1. Compute fast temporal burst counts for sender->out, dest->in, sender<-in, dest->out
    s_out = _fast_window_counts(sender, ts, sender, ts, [60, 900], n_acc)
    d_in = _fast_window_counts(dest, ts, dest, ts, [60, 300, 900, 1800], n_acc)
    s_in = _fast_window_counts(sender, ts, dest, ts, [300, 1800], n_acc)
    d_out = _fast_window_counts(dest, ts, sender, ts, [300, 1800], n_acc)

    # Sort transactions chronologically to compute gaps & USD amount similarity
    df = pd.DataFrame({
        "tx": np.arange(n_tx),
        "ts": ts,
        "sender": sender,
        "dest": dest,
        "usd": usd,
        "pair": sender * n_acc + dest,
    }).sort_values("ts", kind="mergesort").reset_index(drop=True)

    g_s = df.groupby("sender", sort=False)
    s_prev_gap = (df["ts"] - g_s["ts"].shift(1)).fillna(604800).clip(0, 604800).values
    s_next_gap = (g_s["ts"].shift(-1) - df["ts"]).fillna(604800).clip(0, 604800).values
    s_prev_usd_diff = (df["usd"] - g_s["usd"].shift(1)).abs().fillna(2.0).clip(0, 5.0).values
    s_next_usd_diff = (df["usd"] - g_s["usd"].shift(-1)).abs().fillna(2.0).clip(0, 5.0).values

    g_d = df.groupby("dest", sort=False)
    d_prev_gap = (df["ts"] - g_d["ts"].shift(1)).fillna(604800).clip(0, 604800).values
    d_next_gap = (g_d["ts"].shift(-1) - df["ts"]).fillna(604800).clip(0, 604800).values
    d_prev_usd_diff = (df["usd"] - g_d["usd"].shift(1)).abs().fillna(2.0).clip(0, 5.0).values
    d_next_usd_diff = (df["usd"] - g_d["usd"].shift(-1)).abs().fillna(2.0).clip(0, 5.0).values

    pair_counts = df.groupby("pair", sort=False)["tx"].transform("size").values

    # Unsort back to original tx index order
    inv = np.empty(n_tx, dtype=np.int64)
    inv[df["tx"].values] = np.arange(n_tx)

    extra_dict = {
        "log_sender_2m": np.log1p(s_out[60]),
        "log_sender_30m": np.log1p(s_out[900]),
        "log_dest_2m": np.log1p(d_in[60]),
        "log_dest_10m": np.log1p(d_in[300]),
        "log_dest_30m": np.log1p(d_in[900]),
        "log_dest_1h": np.log1p(d_in[1800]),
        "log_sender_in_10m": np.log1p(s_in[300]),
        "log_sender_in_1h": np.log1p(s_in[1800]),
        "log_dest_out_10m": np.log1p(d_out[300]),
        "log_dest_out_1h": np.log1p(d_out[1800]),
        "log_s_prev_gap": np.log1p(s_prev_gap[inv]),
        "log_s_next_gap": np.log1p(s_next_gap[inv]),
        "log_d_prev_gap": np.log1p(d_prev_gap[inv]),
        "log_d_next_gap": np.log1p(d_next_gap[inv]),
        "s_min_usd_diff": np.minimum(s_prev_usd_diff[inv], s_next_usd_diff[inv]),
        "d_min_usd_diff": np.minimum(d_prev_usd_diff[inv], d_next_usd_diff[inv]),
        "log_pair_cnt": np.log1p(pair_counts[inv]),
        "burst_combined_10m": np.log1p(np.expm1(x_tx_np[:, 19]) + d_in[300] + s_in[300] + d_out[300]),
    }
    extra_df = pd.DataFrame(extra_dict).astype(np.float32)

    if aug_stats is None:
        if "train_mask" in data["tx"]:
            fit_rows = data["tx"].train_mask.cpu().numpy()
            fit_df = extra_df.loc[fit_rows]
        else:
            fit_df = extra_df
        aug_stats = {
            "mean": fit_df.mean().values.astype(np.float32),
            "std": fit_df.std().replace(0, 1).fillna(1).values.astype(np.float32),
            "names": list(extra_df.columns),
        }

    extra_std = (extra_df.values - aug_stats["mean"]) / aug_stats["std"]
    data["tx"].x = torch.cat([data["tx"].x, torch.tensor(extra_std, dtype=torch.float)], dim=1)

    # 2. Build temporal ('tx', 'temporal', 'tx') edges
    src_list, dst_list, attr_list = [], [], []

    # (a) Same sender within 1800s (up to k=4 temporal neighbors)
    for k in range(1, 5):
        nxt_tx = g_s["tx"].shift(-k)
        nxt_ts = g_s["ts"].shift(-k)
        nxt_usd = g_s["usd"].shift(-k)
        dt = nxt_ts - df["ts"]
        m = nxt_tx.notna() & (dt <= 1800)
        s = df["tx"][m].values.astype(np.int64)
        t = nxt_tx[m].values.astype(np.int64)
        dt_v = np.log1p(dt[m].values) / np.log1p(1800)
        du_v = np.abs(df["usd"][m].values - nxt_usd[m].values).clip(0, 5.0)
        n_e = len(s)
        src_list.extend([s, t])
        dst_list.extend([t, s])
        attr_list.append(np.column_stack([dt_v, np.ones(n_e), np.zeros(n_e), np.zeros(n_e), du_v, np.ones(n_e)]))
        attr_list.append(np.column_stack([dt_v, np.ones(n_e), np.zeros(n_e), np.zeros(n_e), du_v, -np.ones(n_e)]))

    # (b) Same receiver (dest) within 1800s (up to k=4 temporal neighbors)
    for k in range(1, 5):
        nxt_tx = g_d["tx"].shift(-k)
        nxt_ts = g_d["ts"].shift(-k)
        nxt_usd = g_d["usd"].shift(-k)
        dt = nxt_ts - df["ts"]
        m = nxt_tx.notna() & (dt <= 1800)
        s = df["tx"][m].values.astype(np.int64)
        t = nxt_tx[m].values.astype(np.int64)
        dt_v = np.log1p(dt[m].values) / np.log1p(1800)
        du_v = np.abs(df["usd"][m].values - nxt_usd[m].values).clip(0, 5.0)
        n_e = len(s)
        src_list.extend([s, t])
        dst_list.extend([t, s])
        attr_list.append(np.column_stack([dt_v, np.zeros(n_e), np.ones(n_e), np.zeros(n_e), du_v, np.ones(n_e)]))
        attr_list.append(np.column_stack([dt_v, np.zeros(n_e), np.ones(n_e), np.zeros(n_e), du_v, -np.ones(n_e)]))

    # (c) Pass-through relay: dest of tx1 == sender of tx2 within 3600s
    events = pd.concat([
        df[["tx", "ts", "dest", "usd"]].rename(columns={"dest": "acc"}).assign(role=0),
        df[["tx", "ts", "sender", "usd"]].rename(columns={"sender": "acc"}).assign(role=1),
    ]).sort_values("ts", kind="mergesort").reset_index(drop=True)
    g_ev = events.groupby("acc", sort=False)
    for k in range(1, 4):
        nxt_tx = g_ev["tx"].shift(-k)
        nxt_ts = g_ev["ts"].shift(-k)
        nxt_role = g_ev["role"].shift(-k)
        nxt_usd = g_ev["usd"].shift(-k)
        dt = nxt_ts - events["ts"]
        m = (events["role"] != nxt_role) & nxt_tx.notna() & (dt <= 3600) & (events["tx"] != nxt_tx)
        s = events["tx"][m].values.astype(np.int64)
        t = nxt_tx[m].values.astype(np.int64)
        dt_v = np.log1p(dt[m].values) / np.log1p(3600)
        du_v = np.abs(events["usd"][m].values - nxt_usd[m].values).clip(0, 5.0)
        n_e = len(s)
        src_list.extend([s, t])
        dst_list.extend([t, s])
        attr_list.append(np.column_stack([dt_v, np.zeros(n_e), np.zeros(n_e), np.ones(n_e), du_v, np.ones(n_e)]))
        attr_list.append(np.column_stack([dt_v, np.zeros(n_e), np.zeros(n_e), np.ones(n_e), du_v, -np.ones(n_e)]))

    src_all = np.concatenate(src_list)
    dst_all = np.concatenate(dst_list)
    attr_all = np.vstack(attr_list).astype(np.float32)

    data["tx", "temporal", "tx"].edge_index = torch.tensor(np.stack([src_all, dst_all]), dtype=torch.long)
    data["tx", "temporal", "tx"].edge_attr = torch.tensor(attr_all, dtype=torch.float)
    data["tx"].sender_id = torch.tensor(sender, dtype=torch.long)
    data["tx"].dest_id = torch.tensor(dest, dtype=torch.long)

    return data, aug_stats


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class GatedEdgeConv(nn.Module):
    """Fast edge-gated message passing layer.
    Projects node features BEFORE gathering onto edges (O(N * H^2) instead of O(E * H^2))
    and modulates messages by a learned edge-attribute gate in [0, 2].
    """

    def __init__(self, hidden_dim: int, edge_dim: int, use_max_aggr: bool = False):
        super().__init__()
        self.src_lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.use_max_aggr = use_max_aggr
        out_mult = 2 if use_max_aggr else 1
        self.out_lin = nn.Linear(hidden_dim * out_mult, hidden_dim)

    def forward(self, x_src: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_dst: int) -> torch.Tensor:
        z_src = self.src_lin(x_src)
        gate = 2.0 * self.edge_gate(edge_attr)  # [E, 1]
        msg = z_src[edge_index[0]] * gate        # [E, H]
        agg_mean = scatter(msg, edge_index[1], dim=0, dim_size=num_dst, reduce="mean")
        if self.use_max_aggr:
            agg_max = scatter(msg, edge_index[1], dim=0, dim_size=num_dst, reduce="max")
            return self.out_lin(torch.cat([agg_mean, agg_max], dim=-1))
        return self.out_lin(agg_mean)


class HeteroTxGNN(nn.Module):
    """Hybrid Bipartite + Temporal Heterogeneous GNN over ('account', 'tx') nodes.

    Propagates edge-conditioned messages across:
      - 4 bipartite account <-> tx transfer-leg edge types
      - 1 temporal tx <-> tx edge type (same-sender burst, same-dest fan-in, pass-through relay)
      - Explicit max + mean temporal neighborhood pooling so peak burst anomalies are preserved.
    """

    def __init__(
        self,
        tx_in_dim: int,
        acc_in_dim: int,
        bipartite_edge_dim: int = 7,
        temporal_edge_dim: int = 6,
        hidden_dim: int = 64,
        emb_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.num_layers = num_layers

        self.tx_enc = make_mlp(tx_in_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.acc_enc = make_mlp(acc_in_dim, hidden_dim, hidden_dim, dropout=dropout)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            layer_dict = nn.ModuleDict({
                "sends": GatedEdgeConv(hidden_dim, bipartite_edge_dim, use_max_aggr=False),
                "rev_receives": GatedEdgeConv(hidden_dim, bipartite_edge_dim, use_max_aggr=False),
                "receives": GatedEdgeConv(hidden_dim, bipartite_edge_dim, use_max_aggr=True),
                "rev_sends": GatedEdgeConv(hidden_dim, bipartite_edge_dim, use_max_aggr=True),
                "temporal": GatedEdgeConv(hidden_dim, temporal_edge_dim, use_max_aggr=True),
            })
            self.layers.append(layer_dict)
            self.norms.append(
                nn.ModuleDict({
                    "tx": nn.LayerNorm(hidden_dim),
                    "account": nn.LayerNorm(hidden_dim),
                })
            )

        # Fuse initial node encoding + final GNN state + temporal max-pooled neighbor state
        # + primary sender & receiver account embeddings
        fusion_in_dim = hidden_dim * 5
        self.bottleneck = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, data):
        x_tx_0 = self.tx_enc(data["tx"].x)
        x_acc_0 = self.acc_enc(data["account"].x)

        n_tx = x_tx_0.shape[0]
        n_acc = x_acc_0.shape[0]
        h_tx, h_acc = x_tx_0, x_acc_0

        ei_sends = data["account", "sends", "tx"].edge_index
        ea_sends = data["account", "sends", "tx"].edge_attr
        ei_rev_rec = data["account", "rev_receives", "tx"].edge_index
        ea_rev_rec = data["account", "rev_receives", "tx"].edge_attr

        ei_rec = data["tx", "receives", "account"].edge_index
        ea_rec = data["tx", "receives", "account"].edge_attr
        ei_rev_sends = data["tx", "rev_sends", "account"].edge_index
        ea_rev_sends = data["tx", "rev_sends", "account"].edge_attr

        ei_temp = data["tx", "temporal", "tx"].edge_index
        ea_temp = data["tx", "temporal", "tx"].edge_attr

        for layer, norm_dict in zip(self.layers, self.norms):
            # Messages to tx nodes (from sender accounts, receiver accounts, and temporal tx neighbors)
            m_tx = (
                layer["sends"](h_acc, ei_sends, ea_sends, n_tx)
                + layer["rev_receives"](h_acc, ei_rev_rec, ea_rev_rec, n_tx)
                + layer["temporal"](h_tx, ei_temp, ea_temp, n_tx)
            )
            # Messages to account nodes (from incoming and outgoing txs, using mean+max)
            m_acc = (
                layer["receives"](h_tx, ei_rec, ea_rec, n_acc)
                + layer["rev_sends"](h_tx, ei_rev_sends, ea_rev_sends, n_acc)
            )

            h_tx = h_tx + F.dropout(F.silu(norm_dict["tx"](m_tx)), p=self.dropout, training=self.training)
            h_acc = h_acc + F.dropout(F.silu(norm_dict["account"](m_acc)), p=self.dropout, training=self.training)

        # Max-pool over temporal neighbors to preserve peak burst signals
        if ei_temp.shape[1] > 0:
            temp_max = scatter(h_tx[ei_temp[0]], ei_temp[1], dim=0, dim_size=n_tx, reduce="max")
        else:
            temp_max = torch.zeros_like(x_tx_0)

        sender_acc_h = h_acc[data["tx"].sender_id]
        dest_acc_h = h_acc[data["tx"].dest_id]

        tx_cat = torch.cat([x_tx_0, h_tx, temp_max, sender_acc_h, dest_acc_h], dim=-1)
        emb = self.bottleneck(tx_cat)
        logits = self.classifier(F.silu(emb)).squeeze(-1)
        return logits, emb
