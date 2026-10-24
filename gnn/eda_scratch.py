import json, numpy as np, pandas as pd
d = "train-data/"
tx = pd.read_parquet(d + "transactions.parquet")
tr = pd.read_parquet(d + "transfers.parquet")
rates = pd.read_csv(d + "market_rates.csv")
lab = pd.read_csv(d + "labels.csv")
XRP = "0x1111222211112222111122221111222211112222"
tr["TS"] = tr.TIMESTAMP.astype(int); tr["IDX"] = tr.TRANSFER_INDEX.astype(int)
tr["ASSET"] = np.where(tr.TOKEN_ADDRESS == XRP, "XRP", tr.TOKEN_ADDRESS + ":" + tr.SYMBOL)
tr["HOUR"] = tr.TS // 3600 * 3600
tr = tr.merge(rates.rename(columns={"TIMESTAMP": "HOUR"}), on=["HOUR", "ASSET"], how="left")
tr["USD"] = tr.VALUE * tr.PRICE_MICRO_USD / 1e6
tr = tr.sort_values(["TRANSACTION_HASH", "IDX"])
g = tr.groupby("TRANSACTION_HASH").agg(usd=("USD", "first"), nlegs=("IDX", "size"), a0=("ASSET", "first"), a1=("ASSET", "last"))
df = tx[["TRANSACTION_HASH", "TIMESTAMP", "ACCOUNT", "EXTRA_TRANSACTION_FIELDS", "SEQUENCE", "MEMOS", "SOURCE_TAG", "FLAGS", "SOURCE_FEE", "LAST_LEDGER_SEQUENCE", "LEDGER_INDEX"]].copy()
df["TS"] = df.TIMESTAMP.astype(int)
df["DEST"] = df.EXTRA_TRANSACTION_FIELDS.map(lambda s: json.loads(s)["Destination"])
df["extra_keys"] = df.EXTRA_TRANSACTION_FIELDS.map(lambda s: ",".join(sorted(json.loads(s).keys())))
df = df.merge(g, left_on="TRANSACTION_HASH", right_index=True).merge(lab, on="TRANSACTION_HASH")
df.to_pickle("outputs/cache/eda_df.pkl")
print(pd.crosstab(df.extra_keys, df.difficulty))
print(pd.crosstab(df.SOURCE_FEE.astype(str), df.difficulty).head(20))
print(pd.crosstab(df.nlegs, df.difficulty))
df["lls"] = df.LAST_LEDGER_SEQUENCE.astype(float) - df.LEDGER_INDEX.astype(float)
print(df.groupby("difficulty").lls.describe())
print(pd.crosstab(df.FLAGS.astype(str), df.difficulty))
print(df.MEMOS.notna().groupby(df.difficulty).mean(), df.SOURCE_TAG.notna().groupby(df.difficulty).mean())
