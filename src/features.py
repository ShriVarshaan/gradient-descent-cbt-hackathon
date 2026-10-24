"""Features: one row per transaction, built from the Step 1 tables.

Basic features come from our Step 1 tables. FX features (leg, USD payment and
account behaviour) and HISTORY_FEATURES (history_features.py) come from
feature/feature-engineering. Graph-context features come
from the gnn branch (see graph_features.py). Short-window motif features are in
motif_features.py. All are merged on TRANSACTION_HASH.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import payments.py from the project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graph_features import GRAPH_FEATURES, build_graph_features
from history_features import (
    HISTORY_FEATURES,
    build_behavior_features,
    build_flow_features,
    build_historical_account_graph_features,
    build_laundering_pattern_features,
    build_time_features,
    build_xrpl_features,
)
from motif_features import MOTIF_FEATURES, RARE_FEATURES, build_motif_features
from payments import project
from prepare import add_usd, join_blocks, parse_payments

BASIC_FEATURES = [
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

FX_FEATURES = [
    "leg_count",
    "distinct_assets",
    "distinct_leg_senders",
    "distinct_leg_receivers",
    "max_leg_amount",
    "min_leg_amount",
    "mean_leg_amount",
    "total_leg_amount",
    "is_cross_currency",
    "leg_amount_ratio",
    "usd_value",
    "log_usd_value",
    "is_large_payment",
    "sender_tx_count",
    "sender_total_usd",
    "sender_unique_receivers",
    "receiver_tx_count",
    "receiver_total_usd",
    "receiver_unique_senders",
    "sender_avg_usd",
    "receiver_avg_usd",
]

FEATURES = BASIC_FEATURES + FX_FEATURES + HISTORY_FEATURES + GRAPH_FEATURES + MOTIF_FEATURES + RARE_FEATURES



# -----------------------------------------
# 1. TRANSACTION-LEVEL FEATURES
# -----------------------------------------

def build_leg_features(transfers):
    df = transfers.copy()

    # Identify assets by issuer + currency
    df["asset_id"] = (
        df["TOKEN_ADDRESS"].fillna("XRP").astype(str)
        + ":"
        + df["SYMBOL"].fillna("XRP").astype(str)
    )

    df["amount"] = pd.to_numeric(
        df["VALUE"], errors="coerce"
    )

    features = df.groupby("TRANSACTION_HASH").agg(
        leg_count=("TRANSACTION_HASH", "size"),
        distinct_assets=("asset_id", "nunique"),
        distinct_leg_senders=("FROM_ADDRESS", "nunique"),
        distinct_leg_receivers=("TO_ADDRESS", "nunique"),
        max_leg_amount=("amount", "max"),
        min_leg_amount=("amount", "min"),
        mean_leg_amount=("amount", "mean"),
        total_leg_amount=("amount", "sum"),
    ).reset_index()

    features["is_cross_currency"] = (
        features["distinct_assets"] > 1
    ).astype(int)

    features["leg_amount_ratio"] = (
        features["max_leg_amount"]
        / features["min_leg_amount"].clip(lower=1e-9)
    )

    return features


# -----------------------------------------
# 2. USD PAYMENT FEATURES
# -----------------------------------------

def build_payment_features(payments):
    features = payments[[
        "TRANSACTION_HASH", "usd_value"
    ]].copy()

    features["log_usd_value"] = np.log1p(
        features["usd_value"].clip(lower=0)
    )

    features["is_large_payment"] = (
        features["usd_value"] >= 10000
    ).astype(int)

    return features


# -----------------------------------------
# 4. COMBINE ALL FEATURES
# -----------------------------------------

# -----------------------------------------
# 5. BASIC AND COMBINED BUILDERS
# -----------------------------------------

def build_basic_features(tables):
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


def build_fx_features(tables):
    rates = {
        (int(r["TIMESTAMP"]), r["ASSET"]): int(r["PRICE_MICRO_USD"])
        for r in tables["market_rates"].to_dict("records")
    }
    payments = pd.DataFrame(project(tables["transactions"].to_dict("records"), rates))
    payments["usd_value"] = pd.to_numeric(payments["VALUE"], errors="coerce")

    leg_features = build_leg_features(tables["transfers"])
    payment_features = build_payment_features(payments)

    features = leg_features.merge(
        payment_features, on="TRANSACTION_HASH", how="left", validate="one_to_one"
    )
    for additional in [
        build_historical_account_graph_features(payments),
        build_time_features(payments),
        build_flow_features(payments),
        build_xrpl_features(tables["transactions"], tables["transfers"]),
        build_behavior_features(payments),
        build_laundering_pattern_features(payments),
    ]:
        features = features.merge(
            additional, on="TRANSACTION_HASH", how="left", validate="one_to_one"
        )
    assert features["TRANSACTION_HASH"].is_unique
    return features


def build_features(tables):
    basic = build_basic_features(tables)
    fx = build_fx_features(tables)
    graph = build_graph_features(tables)
    motif = build_motif_features(tables)
    out = basic.merge(fx, on="TRANSACTION_HASH", how="left", validate="one_to_one")
    out = out.merge(graph, on="TRANSACTION_HASH", how="left", validate="one_to_one")
    return out.merge(motif, on="TRANSACTION_HASH", how="left", validate="one_to_one")
