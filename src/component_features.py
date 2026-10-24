"""New-link component features: the shape of the motif a payment belongs to.

Planted motifs (fan-in/out, bipartite, stack, chain, cycle) are built from sender-destination
pairs that are rarely used, among accounts that are otherwise busy. Following every payment of
a busy account drowns the motif, so here two payments are linked only if both use a rare pair,
they share an account, and they are at most GAP seconds apart. Each payment gets the size,
account count, time span and density of its connected group.
"""
import json

import numpy as np
import pandas as pd

RARE_PAIR = 2
GAPS = {"1h": 3600, "3h": 3 * 3600}
STATS = ["txs", "accounts", "span_h", "density", "senders", "dests"]
COMPONENT_FEATURES = [f"q_{s}_{g}" for g in GAPS for s in STATS]


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _components(src, dst, ts, rare, gap):
    """Union-find over rare payments: link consecutive rare payments of an account within gap."""
    n = len(ts)
    parent = np.arange(n)
    idx = np.flatnonzero(rare)
    ev = pd.DataFrame({"acc": np.r_[src[idx], dst[idx]], "ts": np.r_[ts[idx], ts[idx]], "tx": np.r_[idx, idx]})
    ev = ev.sort_values(["acc", "ts", "tx"], kind="stable")
    same = (ev.acc.to_numpy()[1:] == ev.acc.to_numpy()[:-1]) & (np.diff(ev.ts.to_numpy()) <= gap)
    a, b = ev.tx.to_numpy()[:-1][same], ev.tx.to_numpy()[1:][same]
    for x, y in zip(a, b):
        rx, ry = _find(parent, x), _find(parent, y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)
    return np.array([_find(parent, i) for i in range(n)])


def build_component_features(tables):
    tx = tables["transactions"]
    ts = tx["TIMESTAMP"].astype("int64").to_numpy()
    src = tx["ACCOUNT"].to_numpy()
    dst = tx["EXTRA_TRANSACTION_FIELDS"].map(lambda s: json.loads(s)["Destination"]).to_numpy()
    pair = pd.Series(src).astype(str) + ">" + pd.Series(dst).astype(str)
    rare = (pair.map(pair.value_counts()) <= RARE_PAIR).to_numpy()

    f = pd.DataFrame({"TRANSACTION_HASH": tx["TRANSACTION_HASH"].to_numpy()})
    for g, gap in GAPS.items():
        comp = _components(src, dst, ts, rare, gap)
        d = pd.DataFrame({"comp": comp, "src": src, "dst": dst, "ts": ts})
        grp = d.groupby("comp")
        accounts = pd.concat([d[["comp", "src"]].rename(columns={"src": "acc"}),
                              d[["comp", "dst"]].rename(columns={"dst": "acc"})]).groupby("comp").acc.nunique()
        stats = pd.DataFrame({
            "txs": grp.size(),
            "accounts": accounts,
            "span_h": (grp.ts.max() - grp.ts.min()) / 3600,
            "senders": grp.src.nunique(),
            "dests": grp.dst.nunique(),
        })
        stats["density"] = stats.txs / stats.accounts
        joined = d[["comp"]].join(stats, on="comp")
        for s in STATS:
            f[f"q_{s}_{g}"] = np.where(rare, joined[s].to_numpy(), 0.0)  # non-rare payments: no motif
    return f
