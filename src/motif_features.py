"""Local motif features: who an account pays and is paid by within a short time window.

Planted fan-in, fan-out, bipartite, stack, chain and cycle motifs run within minutes to a
few hours, on accounts that are otherwise busy. Week-long totals dilute them, so these
features count payments and distinct counterparties within +-W seconds of each payment,
for the sender and the destination in both directions, plus how bursty that window is
compared with the account's weekly rate, and whether the sender-destination pair is new.
"""
import json

import numpy as np
import pandas as pd

WEEK = 7 * 24 * 3600
WINDOWS = {"10m": 600, "1h": 3600, "3h": 3 * 3600}
ROLES = ["s_out", "s_in", "d_in", "d_out"]

MOTIF_FEATURES = (
    [f"m_{role}_{w}" for w in WINDOWS for role in ROLES]
    + [f"m_{role}_cp_{w}" for w in WINDOWS for role in ROLES]
    + [f"m_{role}_burst_{w}" for w in WINDOWS for role in ["s_out", "d_in"]]
    + [f"m_pair_{w}" for w in WINDOWS]
    + ["m_pair_total", "m_pair_first", "m_pair_rank"]
)


def _index(acc, ts, cp):
    """Per account: event times (sorted) and the counterparty code of each event."""
    df = pd.DataFrame({"acc": acc, "ts": ts, "cp": cp}).sort_values("ts", kind="stable")
    return {a: (g.ts.to_numpy(), g.cp.to_numpy()) for a, g in df.groupby("acc")}


def _window(index, q_acc, q_ts, half):
    """Event count and distinct counterparties in each query account's +-half window."""
    count = np.zeros(len(q_acc))
    distinct = np.zeros(len(q_acc))
    for a, pos in pd.Series(q_acc).groupby(q_acc).indices.items():
        if a not in index:
            continue
        t, c = index[a]
        lo = np.searchsorted(t, q_ts[pos] - half, "left")
        hi = np.searchsorted(t, q_ts[pos] + half, "right")
        count[pos] = hi - lo
        distinct[pos] = [len(np.unique(c[l:h])) for l, h in zip(lo, hi)]
    return count, distinct


def build_motif_features(tables):
    tx = tables["transactions"]
    ts = tx["TIMESTAMP"].astype("int64").to_numpy()
    dest = tx["EXTRA_TRANSACTION_FIELDS"].map(lambda s: json.loads(s)["Destination"])
    codes, _ = pd.factorize(pd.concat([tx["ACCOUNT"], dest], ignore_index=True))
    src, dst = codes[: len(tx)], codes[len(tx):]

    out_idx = _index(src, ts, dst)  # account -> payments it sends, keyed by destination
    in_idx = _index(dst, ts, src)   # account -> payments it receives, keyed by sender
    queries = {"s_out": (out_idx, src), "s_in": (in_idx, src),
               "d_in": (in_idx, dst), "d_out": (out_idx, dst)}

    f = pd.DataFrame({"TRANSACTION_HASH": tx["TRANSACTION_HASH"].to_numpy()})
    weekly = {"s_out": np.bincount(src, minlength=codes.max() + 1)[src],
              "d_in": np.bincount(dst, minlength=codes.max() + 1)[dst]}
    for w, half in WINDOWS.items():
        for role, (index, acc) in queries.items():
            count, distinct = _window(index, acc, ts, half)
            f[f"m_{role}_{w}"] = np.log1p(count)
            f[f"m_{role}_cp_{w}"] = np.log1p(distinct)
            if role in weekly:
                expected = weekly[role] * 2 * half / WEEK
                f[f"m_{role}_burst_{w}"] = np.log1p(count) - np.log1p(expected)

    pair = pd.factorize(pd.Series(src).astype(str) + ">" + pd.Series(dst).astype(str))[0]
    pair_idx = _index(pair, ts, np.zeros(len(pair), dtype=int))
    for w, half in WINDOWS.items():
        f[f"m_pair_{w}"] = np.log1p(_window(pair_idx, pair, ts, half)[0])
    order = pd.DataFrame({"pair": pair, "ts": ts}).sort_values("ts", kind="stable")
    rank = order.groupby("pair").cumcount().reindex(f.index)
    f["m_pair_total"] = np.log1p(np.bincount(pair)[pair])
    f["m_pair_first"] = (rank == 0).astype(float)
    f["m_pair_rank"] = np.log1p(rank)
    return f
