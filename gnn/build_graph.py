"""Build the account <-> transaction bipartite graph with node feature tensors.

    python gnn/build_graph.py <split_dir> graphs/train.pt
    python gnn/build_graph.py <test_dir>  graphs/test.pt --stats graphs/train.pt

<split_dir> holds transactions.parquet, transfers.parquet and market_rates.csv,
plus labels.csv for the labelled train split. Output is a dict saved with
torch.save: {'data': HeteroData, 'stats': normalisation stats, 'names': feature names}.
Load it with torch.load(path, weights_only=False).

Nodes: 'account' and 'tx'. Every transfer leg adds account -[sends]-> tx (its
FROM_ADDRESS) and tx -[receives]-> account (its TO_ADDRESS), so liquidity
providers on conversions join the graph too; reverse edges are added by
ToUndirected. Features use the whole supplied window, which the rules allow.
Assets are encoded by currency symbol only: issuer addresses differ between splits.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

XRP = "0x1111222211112222111122221111222211112222"
SYMBOLS = ["XRP", "USD", "GBP", "EUR"]
WEEK = 7 * 24 * 3600
VAL_FRACTION = 0.2      # last 20% of time is validation, as in the EDA


def load(split):
    split = Path(split)
    tx = pd.read_parquet(split / "transactions.parquet")
    tr = pd.read_parquet(split / "transfers.parquet")
    rates = pd.read_csv(split / "market_rates.csv")
    labels = pd.read_csv(split / "labels.csv") if (split / "labels.csv").exists() else None
    return tx, tr, rates, labels


def value_legs(tr, rates):
    """USD value of every leg at the hour's reference price."""
    tr = tr.copy()
    tr["TS"] = tr.TIMESTAMP.astype(int)
    tr["IDX"] = tr.TRANSFER_INDEX.astype(int)
    tr["ASSET"] = np.where(tr.TOKEN_ADDRESS == XRP, "XRP", tr.TOKEN_ADDRESS + ":" + tr.SYMBOL)
    tr["HOUR"] = tr.TS // 3600 * 3600
    rates = rates.rename(columns={"TIMESTAMP": "HOUR"})
    tr = tr.merge(rates, on=["HOUR", "ASSET"], how="left", validate="many_to_one")
    assert tr.PRICE_MICRO_USD.notna().all(), "missing market rate for some legs"
    tr["USD"] = tr.VALUE * tr.PRICE_MICRO_USD / 1e6
    return tr.sort_values(["TRANSACTION_HASH", "IDX"]).reset_index(drop=True)


def symbol_onehot(symbols, prefix):
    return pd.DataFrame({f"{prefix}_{s}": (symbols == s).astype(float).values for s in SYMBOLS},
                        index=symbols.index)


def next_event(queries, events, direction, value_col):
    """For each (account, TS) query, the gap to and value of that account's next/previous event."""
    q = queries[["ACC", "TS"]].reset_index().sort_values("TS")
    e = events[["ACC", "TS", value_col]].rename(columns={"TS": "TS_E"}).sort_values("TS_E")
    m = pd.merge_asof(q, e, left_on="TS", right_on="TS_E", by="ACC",
                      direction=direction, allow_exact_matches=False).set_index("index").sort_index()
    return (m.TS_E - m.TS).abs(), m[value_col]


def window_count(ts_by_acc, acc, ts, half_width):
    """Number of the account's events within +-half_width seconds of ts (including itself)."""
    out = np.zeros(len(ts), dtype=float)
    for a, idx in pd.Series(np.arange(len(ts))).groupby(acc.values):
        t = ts_by_acc[a]
        out[idx.values] = (np.searchsorted(t, ts.values[idx] + half_width, "right")
                           - np.searchsorted(t, ts.values[idx] - half_width, "left"))
    return out


def tx_table(tx, tr):
    tx = tx.copy()
    tx["TS"] = tx.TIMESTAMP.astype(int)
    extra = tx.EXTRA_TRANSACTION_FIELDS.map(json.loads)
    tx["DEST"] = extra.map(lambda f: f["Destination"])
    tx["HAS_SENDMAX"] = extra.map(lambda f: "SendMax" in f).astype(float)
    tx["HAS_PATHS"] = extra.map(lambda f: "Paths" in f).astype(float)
    legs = tr.groupby("TRANSACTION_HASH").agg(
        n_legs=("IDX", "size"), n_assets=("ASSET", "nunique"), usd_max=("USD", "max"),
        usd_first=("USD", "first"), usd_last=("USD", "last"),
        sym_first=("SYMBOL", "first"), sym_last=("SYMBOL", "last"))
    tx = tx.merge(legs, left_on="TRANSACTION_HASH", right_index=True, validate="one_to_one")
    return tx.reset_index(drop=True)


def tx_features(tx):
    f = pd.DataFrame(index=tx.index)
    f["log_usd_first"] = np.log1p(tx.usd_first)
    f["log_usd_last"] = np.log1p(tx.usd_last)
    f["log_usd_max"] = np.log1p(tx.usd_max)
    f["n_legs"] = tx.n_legs
    f["cross_currency"] = (tx.n_assets > 1).astype(float)
    f["slippage"] = 1 - tx.usd_last / tx.usd_first
    f["has_sendmax"] = tx.HAS_SENDMAX
    f["has_paths"] = tx.HAS_PATHS
    f["log_fee"] = np.log(tx.SOURCE_FEE.astype(float))
    hour = tx.TS % 86400 / 3600
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f = f.join(symbol_onehot(tx.sym_first, "first")).join(symbol_onehot(tx.sym_last, "last"))

    # bursts: the sender's payments around this one (strongest EDA signal)
    out_ts = tx.groupby("ACCOUNT").TS.apply(np.sort)
    for name, half in [("sender_10m", 300), ("sender_1h", 1800)]:
        f[f"log_{name}"] = np.log1p(window_count(out_ts, tx.ACCOUNT, tx.TS, half))

    # pass-through: does the receiver send on soon after, and how similar is the amount?
    sends = tx.assign(ACC=tx.ACCOUNT, USD_OUT=tx.usd_first)
    recvs = tx.assign(ACC=tx.DEST, USD_IN=tx.usd_last)
    gap, usd = next_event(tx.assign(ACC=tx.DEST), sends, "forward", "USD_OUT")
    f["log_dest_next_out_gap"] = np.log1p(gap.fillna(WEEK).clip(upper=WEEK))
    f["dest_next_out_ratio"] = np.log((usd / tx.usd_last).fillna(1)).clip(-5, 5)
    f["dest_forwards_1h"] = (gap <= 3600).astype(float)
    # ...and did the sender just receive what it is now sending?
    gap, usd = next_event(tx.assign(ACC=tx.ACCOUNT), recvs, "backward", "USD_IN")
    f["log_sender_prev_in_gap"] = np.log1p(gap.fillna(WEEK).clip(upper=WEEK))
    f["sender_prev_in_ratio"] = np.log((tx.usd_first / usd).fillna(1)).clip(-5, 5)
    return f.astype(float)


def account_features(tx, tr, accounts, sender_10m):
    a = pd.DataFrame(index=pd.Index(accounts, name="ACC"))
    out, inn = tx.groupby("ACCOUNT"), tx.groupby("DEST")
    a["log_n_out"] = np.log1p(out.size())
    a["log_n_in"] = np.log1p(inn.size())
    a["log_usd_out"] = np.log1p(out.usd_first.sum())
    a["log_usd_in"] = np.log1p(inn.usd_last.sum())
    a["log_out_counterparties"] = np.log1p(out.DEST.nunique())
    a["log_in_counterparties"] = np.log1p(inn.ACCOUNT.nunique())
    a = a.fillna(0)
    a["out_in_usd_ratio"] = (a.log_usd_out - a.log_usd_in).clip(-10, 10)   # pass-through accounts sit near 0
    both = pd.concat([tx[["ACCOUNT", "TS"]].rename(columns={"ACCOUNT": "ACC"}),
                      tx[["DEST", "TS"]].rename(columns={"DEST": "ACC"})])
    span = both.groupby("ACC").TS.agg(lambda t: t.max() - t.min())
    a["log_active_span_h"] = np.log1p(span.reindex(a.index).fillna(0) / 3600)
    gaps = tx.sort_values("TS").groupby("ACCOUNT").TS.diff()
    a["log_median_out_gap"] = np.log1p(gaps.groupby(tx.ACCOUNT).median().reindex(a.index).fillna(WEEK))
    a["frac_out_cross_currency"] = out.n_assets.apply(lambda n: (n > 1).mean()).reindex(a.index).fillna(0)
    a["log_max_sender_10m"] = np.log1p(sender_10m.groupby(tx.ACCOUNT).max().reindex(a.index).fillna(0))
    # legs an account touches without being the payment's sender or destination: liquidity provider role
    leg = tr.merge(tx[["TRANSACTION_HASH", "ACCOUNT", "DEST"]], on="TRANSACTION_HASH")
    lp = pd.concat([leg.FROM_ADDRESS[(leg.FROM_ADDRESS != leg.ACCOUNT) & (leg.FROM_ADDRESS != leg.DEST)],
                    leg.TO_ADDRESS[(leg.TO_ADDRESS != leg.ACCOUNT) & (leg.TO_ADDRESS != leg.DEST)]])
    a["log_lp_legs"] = np.log1p(lp.value_counts().reindex(a.index).fillna(0))
    return a.astype(float)


def standardise(df, rows, stats):
    if stats is None:
        fit = df.loc[rows] if rows is not None else df
        stats = {"mean": fit.mean().values, "std": fit.std().replace(0, 1).fillna(1).values}
    x = (df.values - stats["mean"]) / stats["std"]
    return torch.tensor(x, dtype=torch.float), stats


def build(split, stats=None):
    tx_raw, tr_raw, rates, labels = load(split)
    tr = value_legs(tr_raw, rates)
    tx = tx_table(tx_raw, tr)
    f_tx = tx_features(tx)

    accounts = np.sort(pd.unique(pd.concat([tx.ACCOUNT, tx.DEST, tr.FROM_ADDRESS, tr.TO_ADDRESS])))
    acc_id = pd.Series(np.arange(len(accounts)), index=accounts)
    tx_id = pd.Series(np.arange(len(tx)), index=tx.TRANSACTION_HASH)
    f_acc = account_features(tx, tr, accounts, np.expm1(f_tx.log_sender_10m))

    f_edge = pd.DataFrame({"leg_index": tr.IDX.astype(float),
                           "is_last_leg": (tr.IDX == tr.groupby("TRANSACTION_HASH").IDX.transform("max")).astype(float),
                           "log_usd": np.log1p(tr.USD)}).join(symbol_onehot(tr.SYMBOL, "sym"))

    data = HeteroData()
    data["tx"].time = torch.tensor(tx.TS.values, dtype=torch.long)
    data["tx"].hash = tx.TRANSACTION_HASH.tolist()
    data["account"].address = accounts.tolist()
    train_rows = None
    if labels is not None:
        y = tx[["TRANSACTION_HASH"]].merge(labels, on="TRANSACTION_HASH", how="left", validate="one_to_one")
        assert y.is_suspicious.notna().all(), "some transactions have no label"
        data["tx"].y = torch.tensor(y.is_suspicious.values, dtype=torch.float)
        cut = tx.TS.quantile(1 - VAL_FRACTION)
        train_rows = (tx.TS <= cut).values
        data["tx"].train_mask = torch.tensor(train_rows)
        data["tx"].val_mask = torch.tensor(~train_rows)

    fit = stats is None
    stats = stats or {}
    data["tx"].x, stats["tx"] = standardise(f_tx, train_rows, None if fit else stats["tx"])
    data["account"].x, stats["account"] = standardise(f_acc, None, None if fit else stats["account"])
    edge_attr, stats["edge"] = standardise(f_edge, None, None if fit else stats["edge"])

    src = torch.tensor(acc_id[tr.FROM_ADDRESS].values)
    dst = torch.tensor(acc_id[tr.TO_ADDRESS].values)
    leg_tx = torch.tensor(tx_id[tr.TRANSACTION_HASH].values)
    data["account", "sends", "tx"].edge_index = torch.stack([src, leg_tx])
    data["account", "sends", "tx"].edge_attr = edge_attr
    data["tx", "receives", "account"].edge_index = torch.stack([leg_tx, dst])
    data["tx", "receives", "account"].edge_attr = edge_attr
    data = T.ToUndirected()(data)

    names = {"tx": list(f_tx.columns), "account": list(f_acc.columns), "edge": list(f_edge.columns)}
    return data, stats, names


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("split_dir")
    p.add_argument("out")
    p.add_argument("--stats", help="graph .pt built from train; reuse its normalisation for val/test")
    args = p.parse_args()
    stats = torch.load(args.stats, weights_only=False)["stats"] if args.stats else None
    data, stats, names = build(args.split_dir, stats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "stats": stats, "names": names}, args.out)
    print(data)
    for t in ["tx", "account"]:
        x = data[t].x
        print(f"{t}: {x.shape[1]} features, NaN={torch.isnan(x).sum().item()}, inf={torch.isinf(x).sum().item()}")
    if "y" in data["tx"]:
        for m in ["train_mask", "val_mask"]:
            mask = data["tx"][m]
            print(f"{m}: n={mask.sum().item()} pos={data['tx'].y[mask].sum().item():.0f}")


if __name__ == "__main__":
    main()
