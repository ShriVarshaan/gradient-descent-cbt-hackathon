"""Comprehensive, vectorized domain + graph + temporal motif feature extraction
for the FX Suspicious Transaction Detection benchmark.

Extracts ~135 high-signal features per transaction in ~2.5 seconds:
  1. Ephemeral & All-Pair Temporal Connected Component (TCC) Motif Invariants
     across 6 time/amount scales (120s, 180s, 240s, 360s, 600s, 1200s):
     - Component size (E_C), unique accounts (V_C), cyclomatic number (E_C - V_C + 1)
     - Density (E_C / V_C), time span, transaction rate (E_C / (span + 60))
     - Algebraic motif detectors: Cycle, Bipartite, Fan-In, Fan-Out, Fan-Out-In, Chain, Stack
     - Max sender out-degree & max receiver in-degree inside the component
     - Component-level aggregations of burstiness, USD stats, slippage, leg counts
  2. Dynamic Programming Directed Causal Chains & Cycle Return Times:
     - Backward, forward, and total causal chain length at 180s, 300s, 600s, 1800s
     - Relative position in causal chain
     - Explicit 2-cycle (v -> u) and return-to-sender time gaps
  3. Multi-Hour Relay Wave Causality & Flow Conservation (for aggregate & distributed relays):
     - Causal vs. anti-causal incoming/outgoing counts and USD volumes in [30m, 7h] and [1h, 6h]
     - Causal asymmetry and hub flow-conservation log-ratio error
  4. Account Network Centrality & Corridor Statistics:
     - PageRank & HITS Hub/Authority scores on full & ephemeral graphs
     - Directed/undirected pair counts, spans, rates, and share of account traffic
     - Ledger block co-occurrence and sequence velocity
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

XRP = "0x1111222211112222111122221111222211112222"


def _interval_stats(
    q_acc: np.ndarray,
    q_ts: np.ndarray,
    e_acc: np.ndarray,
    e_ts: np.ndarray,
    e_usd: np.ndarray,
    lo_dt: int,
    hi_dt: int,
    n_acc: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-account interval count and USD sum in [q_ts + lo_dt, q_ts + hi_dt]."""
    order = np.lexsort((e_ts, e_acc))
    ea, et, eu = e_acc[order], e_ts[order], e_usd[order]
    eu_cs = np.r_[0.0, np.cumsum(eu)]
    splits = np.searchsorted(ea, np.arange(n_acc + 1))

    q_order = np.argsort(q_acc, kind="mergesort")
    qa, qt = q_acc[q_order], q_ts[q_order]
    q_splits = np.searchsorted(qa, np.arange(n_acc + 1))

    cnt = np.zeros(len(q_ts), dtype=np.float32)
    usum = np.zeros(len(q_ts), dtype=np.float32)
    for a in range(n_acc):
        qs, qe = q_splits[a], q_splits[a + 1]
        es, ee = splits[a], splits[a + 1]
        if qs == qe or es == ee:
            continue
        t_ev = et[es:ee]
        t_q = qt[qs:qe]
        idx_q = q_order[qs:qe]
        l_idx = np.searchsorted(t_ev, t_q + lo_dt, side="left")
        r_idx = np.searchsorted(t_ev, t_q + hi_dt, side="right")
        cnt[idx_q] = r_idx - l_idx
        usum[idx_q] = eu_cs[es + r_idx] - eu_cs[es + l_idx]
    return cnt, usum


def _compute_causal_chains(u: np.ndarray, v: np.ndarray, ts: np.ndarray, eph_mask: np.ndarray | None, window: int, n_acc: int, prefix: str) -> dict[str, np.ndarray]:
    """Exact O(N) dynamic programming for backward and forward directed causal chains."""
    n = len(ts)
    order = np.argsort(ts, kind="mergesort")
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)

    u_s, v_s, ts_s = u[order], v[order], ts[order]
    active = eph_mask[order] if eph_mask is not None else np.ones(n, dtype=np.float32)

    last_in_ts = np.full(n_acc, -10**12, dtype=np.int64)
    last_in_len = np.zeros(n_acc, dtype=np.int32)
    bwd_len = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if active[i] == 0:
            continue
        ui, vi, ti = u_s[i], v_s[i], ts_s[i]
        if ti - last_in_ts[ui] <= window:
            bwd_len[i] = last_in_len[ui] + 1
        else:
            bwd_len[i] = 1
        if ti - last_in_ts[vi] <= window:
            if bwd_len[i] > last_in_len[vi]:
                last_in_len[vi] = int(bwd_len[i])
        else:
            last_in_len[vi] = int(bwd_len[i])
        last_in_ts[vi] = ti

    next_out_ts = np.full(n_acc, 10**12, dtype=np.int64)
    next_out_len = np.zeros(n_acc, dtype=np.int32)
    fwd_len = np.zeros(n, dtype=np.float32)

    for i in range(n - 1, -1, -1):
        if active[i] == 0:
            continue
        ui, vi, ti = u_s[i], v_s[i], ts_s[i]
        if next_out_ts[vi] - ti <= window:
            fwd_len[i] = next_out_len[vi] + 1
        else:
            fwd_len[i] = 1
        if next_out_ts[ui] - ti <= window:
            if fwd_len[i] > next_out_len[ui]:
                next_out_len[ui] = int(fwd_len[i])
        else:
            next_out_len[ui] = int(fwd_len[i])
        next_out_ts[ui] = ti

    bwd_orig = bwd_len[inv]
    fwd_orig = fwd_len[inv]
    tot_orig = np.maximum(bwd_orig + fwd_orig - 1.0, 0.0)
    pos_ratio = bwd_orig / np.maximum(tot_orig, 1.0)
    return {
        f"{prefix}_bwd": bwd_orig,
        f"{prefix}_fwd": fwd_orig,
        f"{prefix}_tot": tot_orig,
        f"{prefix}_pos": pos_ratio,
    }


def _compute_amount_causal_chains(
    u: np.ndarray,
    v: np.ndarray,
    ts: np.ndarray,
    uf_log: np.ndarray,
    ul_log: np.ndarray,
    window: int,
    max_du: float,
    n_acc: int,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Directed causal chain detector requiring both temporal causality (0 <= dt <= window)
    and pass-through amount conservation (|uf_log_next - ul_log_prev| <= max_du).
    Tracks cumulative USD retention and total chain time span from first hop to last hop.
    """
    n = len(ts)
    order = np.argsort(ts, kind="mergesort")
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)

    u_s, v_s, ts_s, uf_s, ul_s = u[order], v[order], ts[order], uf_log[order], ul_log[order]

    in_hist = [[] for _ in range(n_acc)]
    bwd = np.ones(n, dtype=np.float32)
    start_uf = uf_s.copy()
    start_ts = ts_s.copy()

    for i in range(n):
        ui, vi, ti, ufi, uli = u_s[i], v_s[i], ts_s[i], uf_s[i], ul_s[i]
        best_l = 1
        s_uf = ufi
        s_t = ti
        for prev_t, prev_ul, prev_l, p_suf, p_st in in_hist[ui]:
            if 0 <= ti - prev_t <= window and abs(ufi - prev_ul) <= max_du:
                if prev_l + 1 > best_l:
                    best_l = prev_l + 1
                    s_uf = p_suf
                    s_t = p_st
        bwd[i] = best_l
        start_uf[i] = s_uf
        start_ts[i] = s_t
        in_hist[vi].append((ti, uli, best_l, s_uf, s_t))
        if len(in_hist[vi]) > 6:
            in_hist[vi].pop(0)

    out_hist = [[] for _ in range(n_acc)]
    fwd = np.ones(n, dtype=np.float32)
    end_ul = ul_s.copy()
    end_ts = ts_s.copy()

    for i in range(n - 1, -1, -1):
        ui, vi, ti, ufi, uli = u_s[i], v_s[i], ts_s[i], uf_s[i], ul_s[i]
        best_l = 1
        e_ul = uli
        e_t = ti
        for next_t, next_uf, next_l, n_eul, n_et in out_hist[vi]:
            if 0 <= next_t - ti <= window and abs(next_uf - uli) <= max_du:
                if next_l + 1 > best_l:
                    best_l = next_l + 1
                    e_ul = n_eul
                    e_t = n_et
        fwd[i] = best_l
        end_ul[i] = e_ul
        end_ts[i] = e_t
        out_hist[ui].append((ti, ufi, best_l, e_ul, e_t))
        if len(out_hist[ui]) > 6:
            out_hist[ui].pop(0)

    tot = (bwd + fwd - 1.0)[inv]
    ret = np.exp(np.clip((end_ul - start_uf)[inv], -5.0, 5.0))
    err = np.abs((end_ul - start_uf)[inv])
    span = np.log1p(np.maximum((end_ts - start_ts)[inv], 0))
    return {
        f"{prefix}_tot": tot,
        f"{prefix}_ret": ret,
        f"{prefix}_err": err,
        f"{prefix}_span": span,
    }


def _extract_tcc_scale(
    u: np.ndarray,
    v: np.ndarray,
    ts: np.ndarray,
    uf_raw: np.ndarray,
    uf_log: np.ndarray,
    burst_10m: np.ndarray,
    n_legs: np.ndarray,
    eph_filter: np.ndarray | None,
    tau: int,
    max_du: float,
    prefix: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Fast vectorized Temporal Connected Component (TCC) motif feature extractor.
    Returns (features_df, comp_id, edge_src, edge_dst).
    """
    n = len(ts)
    ev_acc = np.concatenate([u, v])
    ev_tx = np.concatenate([np.arange(n), np.arange(n)])
    ev_ts = np.concatenate([ts, ts])
    ev_uf = np.concatenate([uf_log, uf_log])
    ev_eph = np.concatenate([eph_filter, eph_filter]) if eph_filter is not None else np.ones(2 * n, dtype=np.float32)

    order = np.lexsort((ev_ts, ev_acc))
    ea, et, ets, euf, eep = ev_acc[order], ev_tx[order], ev_ts[order], ev_uf[order], ev_eph[order]

    src_l, dst_l = [], []
    for k in range(1, 10):
        same_a = ea[:-k] == ea[k:]
        dt = ets[k:] - ets[:-k]
        du = np.abs(euf[k:] - euf[:-k])
        both_ok = (eep[:-k] == 1) & (eep[k:] == 1) & (du <= max_du)
        m = same_a & (dt <= tau) & both_ok & (et[:-k] != et[k:])
        src_l.append(et[:-k][m])
        dst_l.append(et[k:][m])

    s_e = np.concatenate(src_l)
    d_e = np.concatenate(dst_l)
    adj = csr_matrix(
        (np.ones(len(s_e) * 2, dtype=np.float32), (np.concatenate([s_e, d_e]), np.concatenate([d_e, s_e]))),
        shape=(n, n),
    )
    _, comp_id = connected_components(adj, directed=False)

    d_pair = u * 10000 + v
    c_df = pd.DataFrame({
        "cid": comp_id,
        "ts": ts,
        "u": u,
        "v": v,
        "dp": d_pair,
        "uf": uf_raw,
        "uf_log": uf_log,
        "burst": burst_10m,
        "legs": n_legs,
    })
    g = c_df.groupby("cid", sort=False)

    c_ntx = g["ts"].transform("size").values.astype(np.float32)
    c_min_ts = g["ts"].transform("min").values.astype(np.float32)
    c_max_ts = g["ts"].transform("max").values.astype(np.float32)
    c_span = c_max_ts - c_min_ts
    c_usd_mean = g["uf"].transform("mean").values.astype(np.float32)
    c_usd_std = g["uf"].transform("std").fillna(0.0).values.astype(np.float32)
    c_usd_min = g["uf_log"].transform("min").values.astype(np.float32)
    c_usd_max = g["uf_log"].transform("max").values.astype(np.float32)
    c_burst_mean = g["burst"].transform("mean").values.astype(np.float32)
    c_burst_max = g["burst"].transform("max").values.astype(np.float32)
    c_legs_mean = g["legs"].transform("mean").values.astype(np.float32)

    c_nsend = g["u"].transform("nunique").values.astype(np.float32)
    c_ndest = g["v"].transform("nunique").values.astype(np.float32)
    c_ndpair = g["dp"].transform("nunique").values.astype(np.float32)

    # Unique accounts V_C per component
    c_acc = pd.concat([c_df[["cid", "u"]].rename(columns={"u": "a"}), c_df[["cid", "v"]].rename(columns={"v": "a"})])
    v_cnt = c_acc.groupby("cid", sort=False)["a"].nunique()
    c_nacc = v_cnt.reindex(comp_id).values.astype(np.float32)

    # Max sender out-degree & max receiver in-degree inside each component
    u_deg_in_c = c_df.groupby(["cid", "u"], sort=False)["ts"].size().groupby(level=0).max()
    v_deg_in_c = c_df.groupby(["cid", "v"], sort=False)["ts"].size().groupby(level=0).max()
    c_max_out_deg = u_deg_in_c.reindex(comp_id).values.astype(np.float32)
    c_max_in_deg = v_deg_in_c.reindex(comp_id).values.astype(np.float32)

    # Number of pass-through accounts inside C: |S_C \cap D_C| = S_C + D_C - V_C
    c_n_passthrough = np.maximum(c_nsend + c_ndest - c_nacc, 0.0)
    c_cyc = np.maximum(c_ntx - c_nacc + 1.0, 0.0)

    # Algebraic motif indicators
    active = (c_ntx >= 3).astype(np.float32)
    if eph_filter is not None:
        active = active * eph_filter

    is_cycle = active * ((c_nsend == c_nacc) & (c_ndest == c_nacc)).astype(np.float32)
    is_bipartite = active * ((c_n_passthrough == 0) & (c_nsend >= 2) & (c_ndest >= 2)).astype(np.float32)
    is_fan_in = active * ((c_ndest == 1) & (c_nsend >= 3)).astype(np.float32)
    is_fan_out = active * ((c_nsend == 1) & (c_ndest >= 3)).astype(np.float32)
    is_fan_out_in = active * ((c_max_out_deg >= 2) & (c_max_in_deg >= 2) & (c_n_passthrough >= 2)).astype(np.float32)
    is_chain = active * ((c_max_out_deg == 1) & (c_max_in_deg == 1) & (c_ntx == c_nacc - 1)).astype(np.float32)
    is_stack = active * (c_ntx > c_ndpair).astype(np.float32)

    mask = eph_filter if eph_filter is not None else np.ones(n, dtype=np.float32)
    out = pd.DataFrame({
        f"{prefix}_ntx": c_ntx * mask,
        f"{prefix}_nacc": c_nacc * mask,
        f"{prefix}_cyc": c_cyc * mask,
        f"{prefix}_density": (c_ntx / np.maximum(c_nacc, 1.0)) * mask,
        f"{prefix}_span": np.log1p(c_span) * mask,
        f"{prefix}_rate": (c_ntx / (c_span + 60.0)) * mask,
        f"{prefix}_step_dt": np.log1p(c_span / np.maximum(c_ntx - 1.0, 1.0)) * mask,
        f"{prefix}_usd_cv": (c_usd_std / (c_usd_mean + 1.0)) * mask,
        f"{prefix}_usd_range": (c_usd_max - c_usd_min) * mask,
        f"{prefix}_burst_mean": c_burst_mean * mask,
        f"{prefix}_burst_max": c_burst_max * mask,
        f"{prefix}_legs_mean": c_legs_mean * mask,
        f"{prefix}_send_ratio": (c_nsend / np.maximum(c_nacc, 1.0)) * mask,
        f"{prefix}_dest_ratio": (c_ndest / np.maximum(c_nacc, 1.0)) * mask,
        f"{prefix}_pt_ratio": (c_n_passthrough / np.maximum(c_nacc, 1.0)) * mask,
        f"{prefix}_max_out_deg": c_max_out_deg * mask,
        f"{prefix}_max_in_deg": c_max_in_deg * mask,
        f"{prefix}_repeat_edges": (c_ntx - c_ndpair) * mask,
        f"{prefix}_rel_pos": ((ts - c_min_ts) / np.maximum(c_span, 1.0)) * mask,
        f"{prefix}_is_cycle": is_cycle,
        f"{prefix}_is_bipartite": is_bipartite,
        f"{prefix}_is_fan_in": is_fan_in,
        f"{prefix}_is_fan_out": is_fan_out,
        f"{prefix}_is_fan_out_in": is_fan_out_in,
        f"{prefix}_is_chain": is_chain,
        f"{prefix}_is_stack": is_stack,
    }, dtype=np.float32)
    return out, comp_id, s_e, d_e


def _compute_pagerank_and_hits(u: np.ndarray, v: np.ndarray, weights: np.ndarray, n_acc: int, n_iter: int = 15):
    """Fast power-iteration PageRank and HITS Hub/Authority on the account graph."""
    adj = csr_matrix((weights, (u, v)), shape=(n_acc, n_acc))
    out_deg = np.asarray(adj.sum(axis=1)).squeeze().clip(min=1e-6)
    trans = adj.T.multiply(1.0 / out_deg)

    pr = np.full(n_acc, 1.0 / n_acc, dtype=np.float32)
    for _ in range(n_iter):
        pr = 0.15 / n_acc + 0.85 * (trans @ pr)
    pr = pr * n_acc

    hub = np.ones(n_acc, dtype=np.float32)
    auth = np.ones(n_acc, dtype=np.float32)
    for _ in range(n_iter):
        auth = adj.T @ hub
        auth = auth / (np.linalg.norm(auth) + 1e-8)
        hub = adj @ auth
        hub = hub / (np.linalg.norm(hub) + 1e-8)
    return pr, hub, auth


def extract_all_domain_features(split_dir: Path, tx_hash_order: list[str] | None = None):
    """Extract the full advanced domain, motif, causal-chain, relay-wave, and centrality
    feature table from a split directory (`train-data`, `validation`, or `test`).
    Also returns the component IDs for multi-scale E-TCC post-processing and the E-TCC edge list.
    """
    split_dir = Path(split_dir)
    tx = pd.read_parquet(split_dir / "transactions.parquet")
    tr = pd.read_parquet(split_dir / "transfers.parquet")
    rates = pd.read_csv(split_dir / "market_rates.csv")

    tr["TS"] = tr.TIMESTAMP.astype(int)
    tr["IDX"] = tr.TRANSFER_INDEX.astype(int)
    tr["ASSET"] = np.where(tr.TOKEN_ADDRESS == XRP, "XRP", tr.TOKEN_ADDRESS + ":" + tr.SYMBOL)
    tr["HOUR"] = tr.TS // 3600 * 3600
    tr = tr.merge(rates.rename(columns={"TIMESTAMP": "HOUR"}), on=["HOUR", "ASSET"], how="left")
    tr["USD"] = tr.VALUE * tr.PRICE_MICRO_USD / 1e6
    tr = tr.sort_values(["TRANSACTION_HASH", "IDX"])

    legs = tr.groupby("TRANSACTION_HASH").agg(
        n_legs=("IDX", "size"),
        usd_first=("USD", "first"),
        usd_last=("USD", "last"),
        usd_max=("USD", "max"),
        usd_min=("USD", "min"),
        usd_sum=("USD", "sum"),
    )
    df = tx[["TRANSACTION_HASH", "TIMESTAMP", "LEDGER_INDEX", "TRANSACTION_INDEX", "ACCOUNT", "SEQUENCE", "SOURCE_FEE", "EXTRA_TRANSACTION_FIELDS"]].merge(
        legs, left_on="TRANSACTION_HASH", right_index=True, how="left"
    )
    if tx_hash_order is not None:
        df = pd.DataFrame({"TRANSACTION_HASH": tx_hash_order}).merge(df, on="TRANSACTION_HASH", how="left")

    df["TS"] = df.TIMESTAMP.astype(int)
    extra = df.EXTRA_TRANSACTION_FIELDS.map(json.loads)
    df["DEST"] = extra.map(lambda f: f["Destination"])

    n = len(df)
    accs = pd.Index(np.sort(pd.unique(pd.concat([df.ACCOUNT, df.DEST]))))
    n_acc = len(accs)
    u = accs.get_indexer(df.ACCOUNT.values)
    v = accs.get_indexer(df.DEST.values)
    ts = df.TS.values
    uf_raw = df.usd_first.values.astype(np.float32)
    ul_raw = df.usd_last.values.astype(np.float32)
    uf_log = np.log1p(uf_raw)
    n_legs = df.n_legs.values.astype(np.float32)

    # 1. Corridor / Pair Ephemerality & Reciprocity Features
    d_pair = u * n_acc + v
    r_pair = v * n_acc + u
    u_pair = np.minimum(u, v) * n_acc + np.maximum(u, v)

    s_dp = pd.Series(ts).groupby(d_pair, sort=False)
    s_up = pd.Series(ts).groupby(u_pair, sort=False)

    dp_cnt = s_dp.transform("size").values.astype(np.float32)
    dp_span = (s_dp.transform("max") - s_dp.transform("min")).values.astype(np.float32)
    up_cnt = s_up.transform("size").values.astype(np.float32)
    up_span = (s_up.transform("max") - s_up.transform("min")).values.astype(np.float32)

    # Reciprocal directed count (how many times v -> u occurs)
    dp_val_cnt = pd.Series(d_pair).value_counts()
    rev_cnt = dp_val_cnt.reindex(r_pair).fillna(0.0).values.astype(np.float32)

    is_eph_30m = (up_span <= 1800).astype(np.float32)
    is_eph_1h = (up_span <= 3600).astype(np.float32)
    is_eph_6h = (up_span <= 21600).astype(np.float32)
    is_multiday_pair = (up_span > 36000).astype(np.float32)

    # Account-level ephemerality rate (what fraction of sender's/receiver's txs are on ephemeral pairs?)
    u_eph_rate = pd.Series(is_eph_1h).groupby(u, sort=False).transform("mean").values.astype(np.float32)
    v_eph_rate = pd.Series(is_eph_1h).groupby(v, sort=False).transform("mean").values.astype(np.float32)

    # Local 10-minute burstiness (on all edges AND on ephemeral-only edges)
    s_10m_cnt, s_10m_usd = _interval_stats(u, ts, u, ts, uf_raw, -300, 300, n_acc)
    d_10m_cnt, d_10m_usd = _interval_stats(v, ts, v, ts, ul_raw, -300, 300, n_acc)
    burst_10m = np.log1p(s_10m_cnt + d_10m_cnt)

    # Ephemeral-only 2m, 3m, 10m and 30m burst counts
    eph_idx = np.flatnonzero(is_eph_1h == 1)
    u_eph, v_eph, ts_eph, uf_eph = u[eph_idx], v[eph_idx], ts[eph_idx], uf_raw[eph_idx]
    s_eph_2m, _ = _interval_stats(u, ts, u_eph, ts_eph, uf_eph, -120, 120, n_acc)
    d_eph_2m, _ = _interval_stats(v, ts, v_eph, ts_eph, uf_eph, -120, 120, n_acc)
    s_eph_3m, _ = _interval_stats(u, ts, u_eph, ts_eph, uf_eph, -180, 180, n_acc)
    d_eph_3m, _ = _interval_stats(v, ts, v_eph, ts_eph, uf_eph, -180, 180, n_acc)
    s_eph_10m, _ = _interval_stats(u, ts, u_eph, ts_eph, uf_eph, -300, 300, n_acc)
    d_eph_10m, _ = _interval_stats(v, ts, v_eph, ts_eph, uf_eph, -300, 300, n_acc)
    s_eph_30m, _ = _interval_stats(u, ts, u_eph, ts_eph, uf_eph, -900, 900, n_acc)
    d_eph_30m, _ = _interval_stats(v, ts, v_eph, ts_eph, uf_eph, -900, 900, n_acc)

    # 2. Multi-Hour Relay Wave Causality & Flow Conservation Features
    u_in_c_cnt, u_in_c_usd = _interval_stats(u, ts, v, ts, ul_raw, -25000, -1800, n_acc)
    u_in_a_cnt, u_in_a_usd = _interval_stats(u, ts, v, ts, ul_raw, 1800, 25000, n_acc)
    v_out_c_cnt, v_out_c_usd = _interval_stats(v, ts, u, ts, uf_raw, 1800, 25000, n_acc)
    v_out_a_cnt, v_out_a_usd = _interval_stats(v, ts, u, ts, uf_raw, -25000, -1800, n_acc)

    s_1h_cnt, s_1h_usd = _interval_stats(u, ts, u, ts, uf_raw, -3600, 3600, n_acc)
    d_1h_cnt, d_1h_usd = _interval_stats(v, ts, v, ts, ul_raw, -3600, 3600, n_acc)

    causal_asym_cnt = (u_in_c_cnt + v_out_c_cnt) - (u_in_a_cnt + v_out_a_cnt)
    causal_asym_usd = np.log1p(u_in_c_usd + v_out_c_usd) - np.log1p(u_in_a_usd + v_out_a_usd)
    u_relay_ratio_err = np.abs(np.log((s_1h_usd + 10.0) / (u_in_c_usd + 10.0)))
    v_relay_ratio_err = np.abs(np.log((v_out_c_usd + 10.0) / (d_1h_usd + 10.0)))

    # Hub wave consistency across [7200s, 26000s] (2h - 7.2h relay delay)
    v_in_now_cnt, v_in_now_usd = _interval_stats(v, ts, v, ts, ul_raw, -4500, 4500, n_acc)
    v_out_caus_cnt, v_out_caus_usd = _interval_stats(v, ts, u, ts, uf_raw, 7200, 26000, n_acc)
    v_out_anti_cnt, _ = _interval_stats(v, ts, u, ts, uf_raw, -26000, -7200, n_acc)

    u_out_now_cnt, u_out_now_usd = _interval_stats(u, ts, u, ts, uf_raw, -4500, 4500, n_acc)
    u_in_caus_cnt, u_in_caus_usd = _interval_stats(u, ts, v, ts, ul_raw, -26000, -7200, n_acc)
    u_in_anti_cnt, _ = _interval_stats(u, ts, v, ts, ul_raw, 7200, 26000, n_acc)

    v_w_ratio = v_out_caus_usd / np.maximum(v_in_now_usd, 1.0)
    u_w_ratio = u_out_now_usd / np.maximum(u_in_caus_usd, 1.0)
    v_w_good = ((v_out_caus_cnt >= 1) & (v_out_anti_cnt == 0) & (v_w_ratio >= 0.55) & (v_w_ratio <= 1.50)).astype(np.float32)
    u_w_good = ((u_in_caus_cnt >= 1) & (u_in_anti_cnt == 0) & (u_w_ratio >= 0.55) & (u_w_ratio <= 1.50)).astype(np.float32)
    u_hub_good_mean = pd.Series(u_w_good).groupby(u, sort=False).transform("mean").values.astype(np.float32)
    v_hub_good_mean = pd.Series(v_w_good).groupby(v, sort=False).transform("mean").values.astype(np.float32)
    hub_good_score = np.maximum(u_hub_good_mean, v_hub_good_mean)

    # 3. Block density & sequence features
    block_tx_cnt = df.groupby("LEDGER_INDEX", sort=False)["TS"].transform("size").values.astype(np.float32)
    ts_sorted = np.sort(ts)
    global_20s = (np.searchsorted(ts_sorted, ts + 20, "right") - np.searchsorted(ts_sorted, ts - 20, "left")).astype(np.float32)

    # 4. PageRank & HITS Centrality on Full and Ephemeral Account Graphs
    pr_full, hub_full, auth_full = _compute_pagerank_and_hits(u, v, np.ones(n, dtype=np.float32), n_acc)
    pr_eph, hub_eph, auth_eph = _compute_pagerank_and_hits(u_eph, v_eph, np.ones(len(eph_idx), dtype=np.float32), n_acc)

    ul_log = np.log1p(ul_raw)
    base_adv = pd.DataFrame({
        "adv_dp_cnt": np.log1p(dp_cnt),
        "adv_dp_span": np.log1p(dp_span),
        "adv_up_cnt": np.log1p(up_cnt),
        "adv_up_span": np.log1p(up_span),
        "adv_rev_cnt": np.log1p(rev_cnt),
        "adv_is_eph_30m": is_eph_30m,
        "adv_is_eph_1h": is_eph_1h,
        "adv_is_eph_6h": is_eph_6h,
        "adv_is_multiday_pair": is_multiday_pair,
        "adv_dp_rate": dp_cnt / (dp_span + 60.0),
        "adv_up_rate": up_cnt / (up_span + 60.0),
        "adv_u_eph_rate": u_eph_rate,
        "adv_v_eph_rate": v_eph_rate,
        "adv_min_eph_rate": np.minimum(u_eph_rate, v_eph_rate),
        "adv_max_eph_rate": np.maximum(u_eph_rate, v_eph_rate),
        "adv_s_eph_2m": np.log1p(s_eph_2m),
        "adv_d_eph_2m": np.log1p(d_eph_2m),
        "adv_tot_eph_2m": np.log1p(s_eph_2m + d_eph_2m) * is_eph_1h,
        "adv_tot_eph_3m": np.log1p(s_eph_3m + d_eph_3m) * is_eph_1h,
        "adv_s_eph_10m": np.log1p(s_eph_10m),
        "adv_d_eph_10m": np.log1p(d_eph_10m),
        "adv_tot_eph_10m": np.log1p(s_eph_10m + d_eph_10m) * is_eph_1h,
        "adv_tot_eph_30m": np.log1p(s_eph_30m + d_eph_30m) * is_eph_1h,
        "adv_u_in_c_cnt": np.log1p(u_in_c_cnt),
        "adv_u_in_a_cnt": np.log1p(u_in_a_cnt),
        "adv_v_out_c_cnt": np.log1p(v_out_c_cnt),
        "adv_v_out_a_cnt": np.log1p(v_out_a_cnt),
        "adv_causal_asym_cnt": causal_asym_cnt,
        "adv_causal_asym_usd": causal_asym_usd,
        "adv_u_relay_err": u_relay_ratio_err,
        "adv_v_relay_err": v_relay_ratio_err,
        "adv_hub_good_score": hub_good_score,
        "adv_block_tx_cnt": block_tx_cnt,
        "adv_tx_index": df.TRANSACTION_INDEX.astype(np.float32).values,
        "adv_global_20s": np.log1p(global_20s),
        "adv_u_pr_full": pr_full[u],
        "adv_v_pr_full": pr_full[v],
        "adv_u_hub_full": hub_full[u],
        "adv_v_auth_full": auth_full[v],
        "adv_u_pr_eph": pr_eph[u],
        "adv_v_pr_eph": pr_eph[v],
        "adv_u_hub_eph": hub_eph[u],
        "adv_v_auth_eph": auth_eph[v],
    }, dtype=np.float32)

    # 5. Directed Causal Chains (standard & amount-conserving for medium-difficulty motifs)
    chain_cols = {}
    for w in [180, 300, 600]:
        chain_cols.update(_compute_causal_chains(u, v, ts, is_eph_1h, w, n_acc, f"adv_echain{w}"))
    chain_cols.update(_compute_causal_chains(u, v, ts, None, 300, n_acc, "adv_achain300"))
    chain_cols.update(_compute_amount_causal_chains(u, v, ts, uf_log, ul_log, 600, 0.25, n_acc, "adv_acchain600"))
    chain_cols.update(_compute_amount_causal_chains(u, v, ts, uf_log, ul_log, 2400, 0.28, n_acc, "adv_acchain2400"))
    chain_df = pd.DataFrame(chain_cols, dtype=np.float32)

    # 6. Multi-Scale Ephemeral & All-Pair Temporal Connected Component (TCC) Motif Invariants
    f_e120, cid_e120, _, _ = _extract_tcc_scale(u, v, ts, uf_raw, uf_log, burst_10m, n_legs, is_eph_1h, 120, 10.0, "adv_etcc120")
    f_e240, cid_e240, s_e240, d_e240 = _extract_tcc_scale(u, v, ts, uf_raw, uf_log, burst_10m, n_legs, is_eph_1h, 240, 10.0, "adv_etcc240")
    f_e360, cid_e360, s_e360, d_e360 = _extract_tcc_scale(u, v, ts, uf_raw, uf_log, burst_10m, n_legs, is_eph_1h, 360, 10.0, "adv_etcc360")
    f_e600, cid_e600, _, _ = _extract_tcc_scale(u, v, ts, uf_raw, uf_log, burst_10m, n_legs, is_eph_1h, 600, 0.35, "adv_etcc600du")
    f_a120, cid_a120, _, _ = _extract_tcc_scale(u, v, ts, uf_raw, uf_log, burst_10m, n_legs, None, 120, 10.0, "adv_atcc120")
    f_a300, cid_a300, _, _ = _extract_tcc_scale(u, v, ts, uf_raw, uf_log, burst_10m, n_legs, None, 300, 0.30, "adv_atcc300du")

    out_df = pd.concat([base_adv, chain_df, f_e120, f_e240, f_e360, f_e600, f_a120, f_a300], axis=1)
    cids = {
        "is_eph_1h": is_eph_1h,
        "cid_e120": cid_e120,
        "cid_e240": cid_e240,
        "cid_e360": cid_e360,
        "cid_e600": cid_e600,
        "etcc_edges": (s_e360, d_e360),
    }
    return out_df, cids
