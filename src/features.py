
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import payments.py from the project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from payments import load_payments


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
# 3. ACCOUNT BEHAVIOUR FEATURES
# -----------------------------------------

def build_account_features(payments):

    # Outgoing account activity
    outgoing = payments.groupby("FROM_ADDRESS").agg(
        sender_tx_count=("TRANSACTION_HASH", "count"),
        sender_total_usd=("usd_value", "sum"),
        sender_unique_receivers=("TO_ADDRESS", "nunique"),
    )

    # Incoming account activity
    incoming = payments.groupby("TO_ADDRESS").agg(
        receiver_tx_count=("TRANSACTION_HASH", "count"),
        receiver_total_usd=("usd_value", "sum"),
        receiver_unique_senders=("FROM_ADDRESS", "nunique"),
    )

    features = payments[[
        "TRANSACTION_HASH",
        "FROM_ADDRESS",
        "TO_ADDRESS"
    ]].copy()

    features = features.join(
        outgoing, on="FROM_ADDRESS"
    )

    features = features.join(
        incoming, on="TO_ADDRESS"
    )

    features["sender_avg_usd"] = (
        features["sender_total_usd"]
        / features["sender_tx_count"].clip(lower=1)
    )

    features["receiver_avg_usd"] = (
        features["receiver_total_usd"]
        / features["receiver_tx_count"].clip(lower=1)
    )

    return features.drop(
        columns=["FROM_ADDRESS", "TO_ADDRESS"]
    )


# -----------------------------------------
# 4. COMBINE ALL FEATURES
# -----------------------------------------

def build_features(directory):
    directory = Path(directory)

    transfers = pd.read_parquet(
        directory / "transfers.parquet"
    )

    payments = pd.DataFrame(
        load_payments(directory)
    )

    payments["usd_value"] = pd.to_numeric(
        payments["VALUE"], errors="coerce"
    )

    leg_features = build_leg_features(transfers)
    payment_features = build_payment_features(payments)
    account_features = build_account_features(payments)

    features = leg_features.merge(
        payment_features,
        on="TRANSACTION_HASH",
        how="left",
        validate="one_to_one"
    )

    features = features.merge(
        account_features,
        on="TRANSACTION_HASH",
        how="left",
        validate="one_to_one"
    )

    assert features["TRANSACTION_HASH"].is_unique
    assert len(features) == len(leg_features)

    return features


# -----------------------------------------
# 5. RUN AND EXPORT
# -----------------------------------------

if __name__ == "__main__":

    for split in ["train", "validation"]:

        features = build_features(ROOT / split)

        output_path = ROOT / f"{split}_features.csv"
        features.to_csv(output_path, index=False)

        print(f"\n{split.upper()}")
        print("Feature shape:", features.shape)
        print("Missing USD values:",
              features["usd_value"].isna().sum())
        print("Saved to:", output_path)
        print(features.head().to_string(index=False))
