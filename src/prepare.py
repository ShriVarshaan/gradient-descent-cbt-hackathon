"""Step 1: load the organiser tables, parse the JSON payment fields, add USD values."""
import json
from pathlib import Path

import pandas as pd


def load_tables(data_dir):
    d = Path(data_dir)
    return {
        "transactions": pd.read_parquet(d / "transactions.parquet"),
        "transfers": pd.read_parquet(d / "transfers.parquet"),
        "blocks": pd.read_parquet(d / "blocks.parquet"),
        "market_rates": pd.read_csv(d / "market_rates.csv"),
        "labels": pd.read_csv(d / "labels.csv"),
    }


def parse_payments(tx):
    extra = tx["EXTRA_TRANSACTION_FIELDS"].map(json.loads)
    delivered = tx["DELIVERED_AMOUNT"].map(json.loads)
    is_xrp = delivered.map(lambda d: isinstance(d, str))

    out = pd.DataFrame({"TRANSACTION_HASH": tx["TRANSACTION_HASH"]})
    out["destination"] = extra.map(lambda e: e["Destination"])
    out["has_send_max"] = extra.map(lambda e: "SendMax" in e)
    out["has_paths"] = extra.map(lambda e: "Paths" in e)
    out["delivered_currency"] = delivered.map(lambda d: "XRP" if isinstance(d, str) else d["currency"])
    out["delivered_issuer"] = delivered.map(lambda d: None if isinstance(d, str) else d["issuer"])
    out["delivered_value"] = [
        int(d) / 1e6 if x else float(d["value"]) for d, x in zip(delivered, is_xrp)
    ]
    return out


def add_usd(transfers, rates):
    """USD value per leg, using the latest hourly rate at or before the leg's time."""
    is_xrp = transfers["SYMBOL"].eq("XRP")
    asset = (transfers["TOKEN_ADDRESS"] + ":" + transfers["SYMBOL"]).where(~is_xrp, "XRP")

    legs = transfers.assign(asset=asset, ts=pd.to_numeric(transfers["TIMESTAMP"]).astype("int64"))
    prices = rates.assign(
        asset=rates["ASSET"],
        ts=rates["TIMESTAMP"].astype("int64"),
        price_usd=rates["PRICE_MICRO_USD"] / 1_000_000,
    )
    merged = pd.merge_asof(
        legs.sort_values("ts"),
        prices[["ts", "asset", "price_usd"]].sort_values("ts"),
        on="ts",
        by="asset",
        direction="backward",
    )
    merged["usd_value"] = merged["VALUE"] * merged["price_usd"]
    return merged.drop(columns=["ts"])
