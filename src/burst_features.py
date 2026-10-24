"""Burst features: how dense and how connected the activity around each transaction is in time.

Planted motifs (fan-in, fan-out, bipartite, chains, cycles, stacks) move money through a small set of
accounts within minutes, while their legitimate look-alikes are spread over longer periods. These
features describe each payment's time-local neighbourhood rather than the payment itself.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from graph_features import WEEK, tx_table, value_legs

CLUSTER_WINDOWS = (30, 60, 120, 300, 900, 1800)
COUNT_WINDOWS = (60, 300, 900, 3600)
HOP_HORIZONS = (600, 3600, 6 * 3600)
RELAY_HORIZONS = (3 * 3600, 12 * 3600)
MOTIF_WINDOWS = (3600, 3 * 3600)
MAX_NEXT = 8


def cluster_ids(tx, w):
    """Connected component of each tx, linking txs that share an account within `w` seconds."""
    inc = pd.DataFrame({"acc": np.concatenate([tx.ACCOUNT.values, tx.DEST.values]),
                        "i": np.tile(np.arange(len(tx)), 2),
                        "TS": np.tile(tx.TS.values, 2)}).sort_values(["acc", "TS"])
    link = (inc.acc.values[1:] == inc.acc.values[:-1]) & (np.diff(inc.TS.values) <= w)
    a, b = inc.i.values[:-1][link], inc.i.values[1:][link]
    return connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(tx), len(tx))),
                                directed=False)[1]


def cluster_features(tx, w):
    """Connected components of txs linked when they share an account within `w` seconds."""
    comp = cluster_ids(tx, w)
    c = pd.DataFrame({"c": comp, "s": tx.ACCOUNT.values, "d": tx.DEST.values, "TS": tx.TS.values,
                      "usd": tx.usd_first.values})
    g = c.groupby("c")
    accounts = pd.concat([c[["c", "s"]].rename(columns={"s": "a"}),
                          c[["c", "d"]].rename(columns={"d": "a"})]).groupby("c").a.nunique()
    size = g.size()
    f = pd.DataFrame(index=tx.index)
    k = f"{w}s"
    f[f"cl_size_{k}"] = np.log(size.reindex(comp).values)
    f[f"cl_acc_{k}"] = np.log(accounts.reindex(comp).values)
    f[f"cl_dur_{k}"] = np.log1p((g.TS.max() - g.TS.min()).reindex(comp).values)
    f[f"cl_senders_{k}"] = np.log(g.s.nunique().reindex(comp).values)
    f[f"cl_dests_{k}"] = np.log(g.d.nunique().reindex(comp).values)
    f[f"cl_density_{k}"] = f[f"cl_size_{k}"] - f[f"cl_acc_{k}"]
    f[f"cl_rate_{k}"] = f[f"cl_size_{k}"] - f[f"cl_dur_{k}"]
    usd_std = g.usd.std().reindex(comp).values
    usd_mean = g.usd.mean().reindex(comp).values
    f[f"cl_usd_cv_{k}"] = np.nan_to_num(usd_std / np.maximum(usd_mean, 1e-9))
    f[f"cl_log_usd_{k}"] = np.log1p(g.usd.sum().reindex(comp).values)
    return f


def counts(ts_sorted_by_acc, acc, ts, half):
    out = np.zeros(len(ts))
    for a, idx in pd.Series(np.arange(len(ts))).groupby(acc):
        t = ts_sorted_by_acc.get(a)
        if t is None:
            continue
        q = ts[idx.values]
        out[idx.values] = np.searchsorted(t, q + half, "right") - np.searchsorted(t, q - half, "left")
    return out


def gaps(ts_sorted_by_acc, acc, ts):
    """Gap to the account's previous and next event (excluding events at the same instant)."""
    prev, nxt = np.full(len(ts), WEEK, float), np.full(len(ts), WEEK, float)
    for a, idx in pd.Series(np.arange(len(ts))).groupby(acc):
        t = ts_sorted_by_acc.get(a)
        if t is None:
            continue
        q = ts[idx.values]
        lo = np.searchsorted(t, q, "left") - 1
        hi = np.searchsorted(t, q, "right")
        ok = lo >= 0
        prev[idx.values[ok]] = q[ok] - t[lo[ok]]
        ok = hi < len(t)
        nxt[idx.values[ok]] = t[hi[ok]] - q[ok]
    return np.log1p(prev), np.log1p(nxt)


def next_hops(tx, horizon):
    """Edges i -> j where j is one of the first MAX_NEXT payments sent by i's destination within horizon."""
    order = np.argsort(tx.TS.values, kind="stable")
    by_sender = {a: order[np.isin(order, idx)] for a, idx in tx.groupby("ACCOUNT").indices.items()}
    src, dst = [], []
    ts = tx.TS.values
    for d, idx in tx.groupby("DEST").indices.items():
        out = by_sender.get(d)
        if out is None:
            continue
        t_out = ts[out]
        lo = np.searchsorted(t_out, ts[idx], "right")
        hi = np.searchsorted(t_out, ts[idx] + horizon, "right")
        for i, l, h in zip(idx, lo, hi):
            h = min(h, l + MAX_NEXT)
            if h > l:
                src.extend([i] * (h - l))
                dst.extend(out[l:h])
    return np.array(src, dtype=int), np.array(dst, dtype=int)


def hop_features(tx, horizon):
    n = len(tx)
    src, dst = next_hops(tx, horizon)
    f = pd.DataFrame(index=tx.index)
    k = f"{horizon}s"
    f[f"hop_out_{k}"] = np.bincount(src, minlength=n)
    f[f"hop_in_{k}"] = np.bincount(dst, minlength=n)
    # longest forward and backward chains through this tx (DP in time order)
    order = np.argsort(tx.TS.values, kind="stable")
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    e = pd.DataFrame({"s": src, "d": dst})
    down = np.zeros(n)
    for s, grp in sorted(e.groupby("s").d, key=lambda x: -rank[x[0]]):
        down[s] = 1 + down[grp.values].max()
    up = np.zeros(n)
    for d, grp in sorted(e.groupby("d").s, key=lambda x: rank[x[0]]):
        up[d] = 1 + up[grp.values].max()
    f[f"chain_down_{k}"], f[f"chain_up_{k}"] = down, up
    f[f"chain_len_{k}"] = down + up
    # does money come back to the sender within 2 or 3 hops?
    acc = tx.ACCOUNT.values
    dest = tx.DEST.values
    cyc2 = np.zeros(n)
    m = dest[dst] == acc[src]
    np.maximum.at(cyc2, src[m], 1)
    e2 = e.merge(e.rename(columns={"s": "d", "d": "d2"}), on="d")
    cyc3 = np.zeros(n)
    m = dest[e2.d2.values] == acc[e2.s.values]
    np.maximum.at(cyc3, e2.s.values[m], 1)
    f[f"cycle2_{k}"], f[f"cycle3_{k}"] = cyc2, cyc3
    # forwarded share: how much of this payment's value moves on
    usd_next = pd.Series(tx.usd_first.values[dst]).groupby(src).sum()
    f[f"fwd_ratio_{k}"] = np.log((usd_next.reindex(range(n)).fillna(0).values + 1)
                                 / (tx.usd_last.values + 1)).clip(-5, 5)
    # fan of fans: distinct senders feeding this tx's sender, distinct receivers fed by this tx's dest
    f[f"hop2_fan_{k}"] = np.log1p(pd.Series(dest[e2.d2.values]).groupby(e2.s.values).nunique()
                                  .reindex(range(n)).fillna(0).values)
    return f


def relay_features(tx, horizon):
    """Week-long relay behaviour: accounts that keep passing on what they receive, along repeated paths."""
    n = len(tx)
    k = f"{horizon}s"
    src, dst = next_hops(tx, horizon)
    acc, dest, ts, usd = tx.ACCOUNT.values, tx.DEST.values, tx.TS.values, tx.usd_first.values
    f = pd.DataFrame(index=tx.index)
    forwarded = np.bincount(src, minlength=n) > 0
    # share of an account's incoming payments it forwards within the horizon, and of its outgoing ones that were fed
    fed = np.bincount(dst, minlength=n) > 0
    relay_in = pd.Series(forwarded).groupby(dest).mean()
    relay_out = pd.Series(fed).groupby(acc).mean()
    n_in = pd.Series(dest).value_counts()
    f[f"s_relay_in_{k}"] = relay_in.reindex(acc).fillna(0).values
    f[f"d_relay_in_{k}"] = relay_in.reindex(dest).fillna(0).values
    f[f"s_relay_out_{k}"] = relay_out.reindex(acc).fillna(0).values
    f[f"d_relay_out_{k}"] = relay_out.reindex(dest).fillna(0).values
    f[f"s_relay_n_{k}"] = np.log1p(f[f"s_relay_in_{k}"] * n_in.reindex(acc).fillna(0).values)
    f[f"d_relay_n_{k}"] = np.log1p(f[f"d_relay_in_{k}"] * n_in.reindex(dest).fillna(0).values)
    # repeated 2-hop paths (sender -> dest -> next receiver)
    triple = pd.Series(list(zip(acc[src], dest[src], dest[dst])))
    tcount = triple.map(triple.value_counts()).values
    f[f"path_rep_out_{k}"] = np.log1p(pd.Series(tcount).groupby(src).max().reindex(range(n)).fillna(0).values)
    f[f"path_rep_in_{k}"] = np.log1p(pd.Series(tcount).groupby(dst).max().reindex(range(n)).fillna(0).values)
    # what the sender received just before vs what it now sends (aggregation)
    ins = pd.DataFrame({"a": dest, "t": ts, "u": usd}).sort_values("t")
    cum = ins.groupby("a").u.cumsum()
    ins = ins.assign(c=cum.values)
    out = np.zeros(n)
    by_acc = {a: (g.t.values, g.c.values) for a, g in ins.groupby("a")}
    for a, idx in pd.Series(np.arange(n)).groupby(acc):
        if a not in by_acc:
            continue
        t, c = by_acc[a]
        q = ts[idx.values]
        hi = np.searchsorted(t, q, "left") - 1
        lo = np.searchsorted(t, q - horizon, "left") - 1
        c_hi = np.where(hi >= 0, c[np.maximum(hi, 0)], 0)
        c_lo = np.where(lo >= 0, c[np.maximum(lo, 0)], 0)
        out[idx.values] = c_hi - c_lo
    f[f"s_recv_before_{k}"] = np.log1p(out)
    f[f"s_recv_before_ratio_{k}"] = np.log((out + 1) / (usd + 1)).clip(-5, 5)
    # pair-level relay consistency: how often this sender->dest pair is fed just before and forwarded just after
    fwd = pd.Series(forwarded.astype(float))
    fd = pd.Series(fed.astype(float))
    pair = [acc, dest]
    f[f"pair_fed_frac_{k}"] = fd.groupby(pair).transform("mean").values
    f[f"pair_fwd_frac_{k}"] = fwd.groupby(pair).transform("mean").values
    f[f"pair_through_{k}"] = np.log1p((fd * fwd).groupby(pair).transform("sum").values)
    # for the sender: share of its outgoing payments that are both fed and forwarded
    f[f"s_through_frac_{k}"] = (fd * fwd).groupby(acc).transform("mean").values
    f[f"d_through_frac_{k}"] = pd.Series((fd * fwd).groupby(acc).mean()).reindex(dest).fillna(0).values
    return f


def motif_features(tx, w):
    """Slower motifs through busy accounts: first-time counterparties, bipartite blocks and converging paths."""
    n = len(tx)
    k = f"{w}s"
    acc, dest, ts = tx.ACCOUNT.values, tx.DEST.values, tx.TS.values.astype(np.int64)
    pair_n = tx.groupby(["ACCOUNT", "DEST"]).TS.transform("size").values
    order = np.argsort(ts, kind="stable")
    out_idx = {a: order[np.isin(order, i)] for a, i in tx.groupby("ACCOUNT").indices.items()}
    in_idx = {a: order[np.isin(order, i)] for a, i in tx.groupby("DEST").indices.items()}

    def window(idx_by_acc, a, t):
        i = idx_by_acc[a]
        tt = ts[i]
        return i[np.searchsorted(tt, t - w, "left"):np.searchsorted(tt, t + w, "right")]

    pairs = set(zip(acc, dest))
    pair_times = {p: np.sort(g) for p, g in pd.Series(ts).groupby([acc, dest])}
    f = {name: np.zeros(n) for name in ["new_dests", "new_senders", "dests", "senders", "biclique"]}
    for i in range(n):
        out_w = window(out_idx, acc[i], ts[i])
        in_w = window(in_idx, dest[i], ts[i])
        recv = set(dest[out_w])
        send = set(acc[in_w])
        f["dests"][i], f["senders"][i] = len(recv), len(send)
        f["new_dests"][i] = len(set(dest[out_w[pair_n[out_w] == 1]]))
        f["new_senders"][i] = len(set(acc[in_w[pair_n[in_w] == 1]]))
        # other senders of this receiver that also paid another receiver of this sender, within the window
        cnt = 0
        if len(recv) > 1 and len(send) > 1:
            for s2 in send - {acc[i]}:
                for r2 in recv - {dest[i]}:
                    if (s2, r2) in pairs:
                        t2 = pair_times[(s2, r2)]
                        if np.searchsorted(t2, ts[i] + w, "right") > np.searchsorted(t2, ts[i] - w, "left"):
                            cnt += 1
        f["biclique"][i] = cnt
    out = pd.DataFrame({f"{name}_{k}": np.log1p(v) for name, v in f.items()}, index=tx.index)
    # converging paths: sender -> several intermediaries -> the same sink
    src, dst = next_hops(tx, w)
    e = pd.DataFrame({"i": src, "j": dst, "s": acc[src], "mid": dest[src], "sink": dest[dst]})
    e = e[e.sink != e.s]
    conv = e.groupby(["s", "sink"]).mid.transform("nunique").values
    out[f"converge_out_{k}"] = np.log1p(pd.Series(conv).groupby(e.i.values).max().reindex(range(n)).fillna(0).values)
    out[f"converge_in_{k}"] = np.log1p(pd.Series(conv).groupby(e.j.values).max().reindex(range(n)).fillna(0).values)
    return out


def account_life(tx):
    both = pd.concat([pd.DataFrame({"a": tx.ACCOUNT, "TS": tx.TS}), pd.DataFrame({"a": tx.DEST, "TS": tx.TS})])
    g = both.groupby("a").TS
    first, last, deg = g.min(), g.max(), g.size()
    f = pd.DataFrame(index=tx.index)
    for side, col in [("s", "ACCOUNT"), ("d", "DEST")]:
        acc = tx[col]
        f[f"{side}_log_deg"] = np.log(deg.reindex(acc).values)
        f[f"{side}_since_first"] = np.log1p(tx.TS.values - first.reindex(acc).values)
        f[f"{side}_until_last"] = np.log1p(last.reindex(acc).values - tx.TS.values)
        f[f"{side}_life"] = np.log1p(last.reindex(acc).values - first.reindex(acc).values)
    return f


def build_burst_features(tables):
    tr = value_legs(tables["transfers"], tables["market_rates"])
    tx = tx_table(tables["transactions"], tr)
    ts = tx.TS.values.astype(np.int64)
    out_ts = {a: np.sort(ts[i]) for a, i in tx.groupby("ACCOUNT").indices.items()}
    in_ts = {a: np.sort(ts[i]) for a, i in tx.groupby("DEST").indices.items()}
    all_ts = np.sort(ts)

    parts = [cluster_features(tx, w) for w in CLUSTER_WINDOWS]
    f = pd.DataFrame(index=tx.index)
    for h in COUNT_WINDOWS:
        f[f"s_out_{h}"] = np.log1p(counts(out_ts, tx.ACCOUNT.values, ts, h))
        f[f"s_in_{h}"] = np.log1p(counts(in_ts, tx.ACCOUNT.values, ts, h))
        f[f"d_in_{h}"] = np.log1p(counts(in_ts, tx.DEST.values, ts, h))
        f[f"d_out_{h}"] = np.log1p(counts(out_ts, tx.DEST.values, ts, h))
        f[f"global_{h}"] = np.log1p(np.searchsorted(all_ts, ts + h, "right") - np.searchsorted(all_ts, ts - h, "left"))
        f[f"d_in_rel_{h}"] = f[f"d_in_{h}"] - f[f"global_{h}"]
        f[f"s_out_rel_{h}"] = f[f"s_out_{h}"] - f[f"global_{h}"]
    f["s_out_prev_gap"], f["s_out_next_gap"] = gaps(out_ts, tx.ACCOUNT.values, ts)
    f["d_in_prev_gap"], f["d_in_next_gap"] = gaps(in_ts, tx.DEST.values, ts)
    f["s_in_prev_gap"], f["s_in_next_gap"] = gaps(in_ts, tx.ACCOUNT.values, ts)
    f["d_out_prev_gap"], f["d_out_next_gap"] = gaps(out_ts, tx.DEST.values, ts)
    # amount relative to the sender's and receiver's usual payment
    usd = tx.usd_first
    f["usd_vs_sender_med"] = np.log1p(usd) - np.log1p(usd.groupby(tx.ACCOUNT).transform("median"))
    f["usd_vs_dest_med"] = np.log1p(usd) - np.log1p(usd.groupby(tx.DEST).transform("median"))
    f["pair_count"] = np.log(tx.groupby(["ACCOUNT", "DEST"]).TS.transform("size"))
    f["reverse_pair"] = tx.set_index(["ACCOUNT", "DEST"]).index.isin(
        tx.set_index(["DEST", "ACCOUNT"]).index).astype(float)
    parts += [f, account_life(tx)] + [hop_features(tx, h) for h in HOP_HORIZONS]
    parts += [relay_features(tx, h) for h in RELAY_HORIZONS]
    parts += [motif_features(tx, w) for w in MOTIF_WINDOWS]
    out = pd.concat(parts, axis=1).astype(float).add_prefix("b_")
    return pd.concat([tx[["TRANSACTION_HASH"]], out], axis=1)
