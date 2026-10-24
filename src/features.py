"""Step 2 basic features: one row per transaction, built from the Step 1 tables."""
import pandas as pd

from prepare import add_usd, join_blocks, parse_payments

FEATURES = [
    "n_legs",
    "total_usd",
    "first_leg_usd",
    "last_leg_usd",
    "max_leg_usd",
    "delivered_value",
    "delivered_is_xrp",
    "has_send_max",
    "has_paths",
    "hour",
]


def build_features(tables):
    tx = join_blocks(tables["transactions"], tables["blocks"])
    parsed = parse_payments(tables["transactions"])
    legs = add_usd(tables["transfers"], tables["market_rates"])
    legs = legs.assign(leg=pd.to_numeric(legs["TRANSFER_INDEX"]).astype("int64"))
    legs = legs.sort_values(["TRANSACTION_HASH", "leg"])

    by_tx = legs.groupby("TRANSACTION_HASH")["usd_value"]
    per_tx = pd.DataFrame({
        "n_legs": legs.groupby("TRANSACTION_HASH").size(),
        "total_usd": by_tx.sum(),
        "first_leg_usd": by_tx.first(),
        "last_leg_usd": by_tx.last(),
        "max_leg_usd": by_tx.max(),
    })

    out = tx[["TRANSACTION_HASH", "close_time"]].merge(parsed, on="TRANSACTION_HASH")
    out = out.merge(per_tx, left_on="TRANSACTION_HASH", right_index=True)
    out["hour"] = pd.to_datetime(out["close_time"], unit="s", utc=True).dt.hour
    out["delivered_is_xrp"] = (out["delivered_currency"] == "XRP").astype(int)
    out["has_send_max"] = out["has_send_max"].astype(int)
    out["has_paths"] = out["has_paths"].astype(int)
    return out.drop(columns=["destination", "delivered_currency", "delivered_issuer"])
