"""Scenario reconstruction: link payments that belong to the same planted motif, then describe each motif.

Every planted scenario (suspicious or decoy) is a small set of payments that share accounts and sit close
in time, but the 700 accounts are shared by everything, so plain time-clustering percolates. Instead:

1. candidate_pairs: every pair of payments that share an account within PAIR_WINDOW seconds, with
   features of the pair (shared role, gap, intervening activity of the shared account, novelty of the two
   sender-receiver pairs, amount and currency continuity).
2. A link model (trained in pipeline code on train/ scenario IDs) scores each pair: same scenario or not.
3. scenario_features: connected components of confident links are the reconstructed motifs; each payment
   gets its motif's size, shape, timing, amount and currency statistics, plus soft link-degree features.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from graph_features import tx_table, value_legs

PAIR_WINDOW = 3 * 3600
LINK_THRESHOLDS = (0.3, 0.6)
ROLES = {("s", "s"): 0, ("s", "d"): 1, ("d", "s"): 2, ("d", "d"): 3}  # role of the shared account in i, j


def payment_table(tables):
    tr = value_legs(tables["transfers"], tables["market_rates"])
    tx = tx_table(tables["transactions"], tr).sort_values(["TS", "TRANSACTION_HASH"]).reset_index(drop=True)
    tx["pair_n"] = tx.groupby(["ACCOUNT", "DEST"]).TS.transform("size")
    a, b = tx.ACCOUNT.where(tx.ACCOUNT < tx.DEST, tx.DEST), tx.DEST.where(tx.ACCOUNT < tx.DEST, tx.ACCOUNT)
    tx["upair_n"] = tx.groupby([a, b]).TS.transform("size")
    deg = pd.concat([tx.ACCOUNT, tx.DEST]).value_counts()
    tx["s_deg"], tx["d_deg"] = deg.reindex(tx.ACCOUNT).values, deg.reindex(tx.DEST).values
    return tx


def candidate_pairs(tx, window=PAIR_WINDOW):
    """All (i, j) with i before j sharing an account within `window`; one row per (i, j, shared account)."""
    inc = pd.concat([
        pd.DataFrame({"i": tx.index, "a": tx.ACCOUNT, "r": "s", "t": tx.TS}),
        pd.DataFrame({"i": tx.index, "a": tx.DEST, "r": "d", "t": tx.TS}),
    ]).sort_values(["a", "t", "i"]).reset_index(drop=True)
    a, t, idx, role = inc.a.values, inc.t.values.astype(np.int64), inc.i.values, inc.r.values
    pos = inc.groupby("a").cumcount().values
    parts = []
    o = 1
    while True:
        ok = (a[o:] == a[:-o]) & (t[o:] - t[:-o] <= window)
        if not ok.any():
            break
        k = np.flatnonzero(ok)
        parts.append(pd.DataFrame({"i": idx[k], "j": idx[k + o], "ri": role[k], "rj": role[k + o],
                                   "inter": o - 1, "acc": a[k], "pos": pos[k]}))
        o += 1
    p = pd.concat(parts, ignore_index=True)
    p = p[p.i != p.j]
    swap = (p.i > p.j).values  # tx is sorted by time, so index order is time order
    i, j, ri, rj = p.i.values, p.j.values, p.ri.values, p.rj.values
    p["i"], p["j"] = np.where(swap, j, i), np.where(swap, i, j)
    p["ri"], p["rj"] = np.where(swap, rj, ri), np.where(swap, ri, rj)
    return p.sort_values("inter", kind="stable").drop_duplicates(["i", "j"]).reset_index(drop=True)


def pair_features(tx, p):
    i, j = p.i.values, p.j.values
    ts = tx.TS.values.astype(np.int64)
    s, d = tx.ACCOUNT.values, tx.DEST.values
    usd_in, usd_out = tx.usd_first.values, tx.usd_last.values
    f = pd.DataFrame(index=p.index)
    f["role"] = [ROLES[(x, y)] for x, y in zip(p.ri.values, p.rj.values)]
    f["log_dt"] = np.log1p(ts[j] - ts[i])
    f["log_inter"] = np.log1p(p.inter.values)
    # intervening events per hour of the shared account, relative to its overall rate
    deg = pd.concat([tx.ACCOUNT, tx.DEST]).value_counts()
    acc_deg = deg.reindex(p.acc.values).values
    f["acc_log_deg"] = np.log(acc_deg)
    f["inter_rel"] = np.log1p(p.inter.values) - np.log1p(acc_deg * (ts[j] - ts[i] + 60) / (7 * 86400))
    oi = np.where(p.ri.values == "s", d[i], s[i])
    oj = np.where(p.rj.values == "s", d[j], s[j])
    f["same_other"] = (oi == oj).astype(float)
    f["oi_log_deg"], f["oj_log_deg"] = np.log(deg.reindex(oi).values), np.log(deg.reindex(oj).values)
    f["pair_n_i"], f["pair_n_j"] = np.log(tx.pair_n.values[i]), np.log(tx.pair_n.values[j])
    f["upair_n_i"], f["upair_n_j"] = np.log(tx.upair_n.values[i]), np.log(tx.upair_n.values[j])
    f["both_novel"] = ((tx.pair_n.values[i] == 1) & (tx.pair_n.values[j] == 1)).astype(float)
    f["cycle"] = ((d[j] == s[i]) & (p.ri.values == "d")).astype(float)  # i: x->a, j: a->x
    f["amt_ratio"] = np.log(usd_in[j] / usd_out[i])  # j's cost vs i's delivered value
    f["amt_ratio_same"] = np.log(usd_out[j] / usd_out[i])
    f["log_usd_i"], f["log_usd_j"] = np.log1p(usd_out[i]), np.log1p(usd_out[j])
    sf, sl = tx.sym_first.values, tx.sym_last.values
    f["same_sym_last"] = (sl[i] == sl[j]).astype(float)
    f["flow_sym"] = (sl[i] == sf[j]).astype(float)
    f["legs_i"], f["legs_j"] = tx.n_legs.values[i], tx.n_legs.values[j]
    f["paths_i"], f["paths_j"] = tx.HAS_PATHS.values[i], tx.HAS_PATHS.values[j]
    f["same_fee"] = (tx.SOURCE_FEE.values[i] == tx.SOURCE_FEE.values[j]).astype(float)
    hour = ts[i] % 86400 / 3600
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    # how many candidate partners each endpoint has: crowded neighbourhoods make any one link less likely
    cnt = np.bincount(np.r_[i, j], minlength=len(tx))
    f["cand_i"], f["cand_j"] = np.log(cnt[i]), np.log(cnt[j])
    # rank of this pair's gap among i's later partners and j's earlier partners
    f["rank_i"] = np.log1p(pd.Series(ts[j]).groupby(i).rank(method="min").values)
    f["rank_j"] = np.log1p(pd.Series(-ts[i]).groupby(j).rank(method="min").values)
    return f


PAIR_FEATURES = ["role", "log_dt", "log_inter", "acc_log_deg", "inter_rel", "same_other", "oi_log_deg", "oj_log_deg",
                 "pair_n_i", "pair_n_j", "upair_n_i", "upair_n_j", "both_novel", "cycle", "amt_ratio",
                 "amt_ratio_same", "log_usd_i", "log_usd_j", "same_sym_last", "flow_sym", "legs_i", "legs_j",
                 "paths_i", "paths_j", "same_fee", "hour_sin", "hour_cos", "cand_i", "cand_j", "rank_i", "rank_j"]


def build_pairs(tables):
    tx = payment_table(tables)
    p = candidate_pairs(tx)
    return tx, pd.concat([p, pair_features(tx, p)], axis=1)


def component_stats(tx, p, comp, prefix):
    """Per-payment statistics of its reconstructed motif (connected component)."""
    n = len(tx)
    c = pd.DataFrame({"c": comp, "ts": tx.TS.values.astype(np.int64), "s": tx.ACCOUNT.values, "d": tx.DEST.values,
                      "usd": tx.usd_last.values, "cost": tx.usd_first.values, "sym": tx.sym_last.values,
                      "cross": (tx.n_assets.values > 1).astype(float), "novel": (tx.pair_n.values == 1).astype(float)})
    g = c.groupby("c")
    size = g.size()
    st = pd.DataFrame({
        "size": size,
        "span": np.log1p(g.ts.max() - g.ts.min()),
        "n_snd": g.s.nunique(), "n_rcv": g.d.nunique(),
        "usd_sum": np.log1p(g.usd.sum()), "usd_med": np.log1p(g.usd.median()),
        "usd_cv": (g.usd.std() / g.usd.mean()).fillna(0),
        "n_sym": g.sym.nunique(), "cross": g.cross.mean(), "novel": g.novel.mean(),
    })
    # accounts per component, and the share that both send and receive inside it (intermediaries)
    inc = pd.concat([c[["c", "s"]].rename(columns={"s": "a"}), c[["c", "d"]].rename(columns={"d": "a"})])
    st["n_acc"] = inc.groupby("c").a.nunique()
    both = c.groupby(["c", "s"]).size().rename("o").reset_index().merge(
        c.groupby(["c", "d"]).size().rename("i").reset_index().rename(columns={"d": "s"}), on=["c", "s"])
    st["n_mid"] = both.groupby("c").size().reindex(st.index).fillna(0)
    st["max_in"] = c.groupby(["c", "d"]).size().groupby("c").max()
    st["max_out"] = c.groupby(["c", "s"]).size().groupby("c").max()
    # value conservation at intermediaries: |log(out/in)| per (component, account), median over the component
    vin = c.groupby(["c", "d"]).usd.sum().rename("vin").reset_index().rename(columns={"d": "a"})
    vout = c.groupby(["c", "s"]).cost.sum().rename("vout").reset_index().rename(columns={"s": "a"})
    m = vin.merge(vout, on=["c", "a"])
    m["dev"] = np.abs(np.log(m.vout / m.vin))
    st["conserve"] = m.groupby("c").dev.median().reindex(st.index)
    gaps = c.sort_values("ts").groupby("c").ts.diff()
    st["gap_med"] = np.log1p(gaps.groupby(c.c).median()).reindex(st.index).fillna(0)
    st["gap_max"] = np.log1p(gaps.groupby(c.c).max()).reindex(st.index).fillna(0)
    st["acc_ratio"] = st.n_acc / st["size"]
    st["mid_ratio"] = st.n_mid / st.n_acc
    out = st.reindex(comp).reset_index(drop=True)
    single = out["size"].values == 1
    out.loc[single, out.columns.difference(["size"])] = np.nan
    out["size"] = np.log(out["size"])
    # position of this payment inside its motif
    out["t_frac"] = ((c.ts - g.ts.transform("min")) / (g.ts.transform("max") - g.ts.transform("min")).replace(0, np.nan)).values
    out["s_out_in_comp"] = np.log(c.groupby(["c", "s"]).ts.transform("size").values)
    out["d_in_in_comp"] = np.log(c.groupby(["c", "d"]).ts.transform("size").values)
    out.columns = [f"{prefix}_{k}" for k in out.columns]
    return out


def scenario_features(tx, p, prob):
    """Features of each payment's reconstructed motif, given link probabilities `prob` for the pairs `p`."""
    n = len(tx)
    i, j = p.i.values, p.j.values
    f = pd.DataFrame(index=tx.index)
    w = np.bincount(i, prob, n) + np.bincount(j, prob, n)
    f["sc_soft_deg"] = np.log1p(w)
    mx = np.zeros(n)
    np.maximum.at(mx, i, prob)
    np.maximum.at(mx, j, prob)
    f["sc_max_link"] = mx
    for th in LINK_THRESHOLDS:
        k = prob >= th
        f[f"sc_deg_{th}"] = np.log1p(np.bincount(i[k], minlength=n) + np.bincount(j[k], minlength=n))
        adj = coo_matrix((np.ones(k.sum()), (i[k], j[k])), shape=(n, n))
        _, comp = connected_components(adj, directed=False)
        f = f.join(component_stats(tx, p, comp, f"sc{int(th * 10)}"))
    f.insert(0, "TRANSACTION_HASH", tx.TRANSACTION_HASH.values)
    return f
