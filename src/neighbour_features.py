"""Second-stage features: first-stage scores pooled over each payment's time-local neighbours.

Planted motifs are judged better as a whole than payment by payment, so stage 2 sees, for each
payment, how suspicious its neighbours looked to stage 1: the other members of its burst cluster,
the payments sharing an account with it within a window, and the payments one hop up or down.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from burst_features import CLUSTER_WINDOWS, cluster_ids, next_hops
from graph_features import tx_table, value_legs

SHARE_WINDOWS = (900, 3600, 6 * 3600)
HOP_HORIZONS = (3600, 6 * 3600)


def share_matrix(tx, w):
    """Payments i != j that share an account (either side) within w seconds."""
    inc = pd.DataFrame({"acc": np.concatenate([tx.ACCOUNT.values, tx.DEST.values]),
                        "i": np.tile(np.arange(len(tx)), 2),
                        "TS": np.tile(tx.TS.values.astype(np.int64), 2)}).sort_values(["acc", "TS"])
    acc, idx, ts = inc.acc.values, inc.i.values, inc.TS.values
    rows, cols = [], []
    starts = np.flatnonzero(np.r_[True, acc[1:] != acc[:-1]])
    ends = np.r_[starts[1:], len(acc)]
    for s, e in zip(starts, ends):
        t, ii = ts[s:e], idx[s:e]
        hi = np.searchsorted(t, t + w, "right")
        for k in range(len(t)):
            if hi[k] > k + 1:
                rows.append(np.full(hi[k] - k - 1, ii[k]))
                cols.append(ii[k + 1:hi[k]])
    r, c = np.concatenate(rows), np.concatenate(cols)
    keep = r != c
    m = sp.coo_matrix((np.ones(keep.sum()), (r[keep], c[keep])), shape=(len(tx),) * 2).tocsr()
    m = ((m + m.T) > 0).astype(float)
    return m


def hop_matrix(tx, horizon):
    src, dst = next_hops(tx, horizon)
    m = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(len(tx),) * 2).tocsr()
    return ((m + m.T) > 0).astype(float)


def build_neighbourhoods(tables):
    """Hash order plus the neighbour structures stage 2 pools over."""
    tr = value_legs(tables["transfers"], tables["market_rates"])
    tx = tx_table(tables["transactions"], tr)
    nb = {f"cl{w}": cluster_ids(tx, w) for w in CLUSTER_WINDOWS}
    nb.update({f"sh{w}": share_matrix(tx, w) for w in SHARE_WINDOWS})
    nb.update({f"hop{h}": hop_matrix(tx, h) for h in HOP_HORIZONS})
    return tx.TRANSACTION_HASH.values, nb


def _row_max(m, s):
    """Max of s over each row's neighbours (0 where a row has none)."""
    v = m.multiply(s[None, :]).tocsr()
    return np.asarray(v.max(axis=1).todense()).ravel()


def pool_scores(nb, s):
    """Aggregates of stage-1 scores s (aligned to the neighbourhood order), excluding each payment itself."""
    out = {}
    logit = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    for name, v in nb.items():
        if name.startswith("cl"):
            g = pd.Series(s).groupby(v)
            n = g.transform("size").values
            tot = g.transform("sum").values
            out[f"n_{name}_mean"] = np.where(n > 1, (tot - s) / np.maximum(n - 1, 1), 0)
            # max over the other members: the top score, or the second one if this payment is the top
            top = g.transform("max").values
            second = pd.Series(s).groupby(v).transform(lambda x: np.sort(x.values)[-2] if len(x) > 1 else 0).values
            out[f"n_{name}_max"] = np.where(s >= top, second, top)
        else:
            deg = np.asarray(v.sum(axis=1)).ravel()
            out[f"n_{name}_mean"] = np.where(deg > 0, (v @ s) / np.maximum(deg, 1), 0)
            out[f"n_{name}_logit"] = np.where(deg > 0, (v @ logit) / np.maximum(deg, 1), -14)
            out[f"n_{name}_max"] = _row_max(v, s)
            out[f"n_{name}_hi"] = v @ (s > 0.5).astype(float)
    return pd.DataFrame(out)
