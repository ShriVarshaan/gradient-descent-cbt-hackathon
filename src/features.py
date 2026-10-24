
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from payments import load_payments


# ============================================================
# 1. TRANSACTION-LEVEL FEATURES
# ============================================================

def build_leg_features(transfers):
    df = transfers.copy()

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


# ============================================================
# 2. USD PAYMENT FEATURES
# ============================================================

def build_payment_features(payments):
    features = payments[
        ["TRANSACTION_HASH", "usd_value"]
    ].copy()

    features["log_usd_value"] = np.log1p(
        features["usd_value"].clip(lower=0)
    )

    features["is_large_payment"] = (
        features["usd_value"] >= 10000
    ).astype(int)

    return features


# ============================================================
# 3. ACCOUNT BEHAVIOUR FEATURES
# ============================================================

def build_account_features(payments):
    outgoing = payments.groupby("FROM_ADDRESS").agg(
        sender_tx_count=("TRANSACTION_HASH", "count"),
        sender_total_usd=("usd_value", "sum"),
        sender_unique_receivers=("TO_ADDRESS", "nunique"),
    )

    incoming = payments.groupby("TO_ADDRESS").agg(
        receiver_tx_count=("TRANSACTION_HASH", "count"),
        receiver_total_usd=("usd_value", "sum"),
        receiver_unique_senders=("FROM_ADDRESS", "nunique"),
    )

    features = payments[
        ["TRANSACTION_HASH", "FROM_ADDRESS", "TO_ADDRESS"]
    ].copy()

    features = features.join(outgoing, on="FROM_ADDRESS")
    features = features.join(incoming, on="TO_ADDRESS")

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


# ============================================================
# 4. HISTORICAL VELOCITY FEATURES
# ============================================================

def build_time_features(payments):
    df = payments[
        ["TRANSACTION_HASH", "FROM_ADDRESS",
         "TO_ADDRESS", "TIMESTAMP", "usd_value"]
    ].copy()

    df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"])
    result = df[["TRANSACTION_HASH"]].copy()

    for address_col, prefix in [
        ("FROM_ADDRESS", "sender"),
        ("TO_ADDRESS", "receiver")
    ]:
        activity = df.sort_values(
            [address_col, "TIMESTAMP", "TRANSACTION_HASH"]
        )

        for window, suffix in [
            (3600, "1h"),
            (86400, "24h")
        ]:
            counts = pd.Series(
                0, index=df.index, dtype="int64"
            )
            volumes = pd.Series(
                0.0, index=df.index
            )

            for _, group in activity.groupby(
                address_col, sort=False
            ):
                times = group["TIMESTAMP"].to_numpy()
                amounts = group["usd_value"].to_numpy()

                end = np.searchsorted(
                    times, times, side="left"
                )
                start = np.searchsorted(
                    times, times - window, side="left"
                )

                cumulative = np.concatenate(
                    ([0.0], np.cumsum(amounts))
                )

                counts.loc[group.index] = end - start
                volumes.loc[group.index] = (
                    cumulative[end] - cumulative[start]
                )

            result[f"{prefix}_tx_count_{suffix}"] = counts
            result[f"{prefix}_usd_{suffix}"] = volumes

    return result


# ============================================================
# 5. RAPID PASS-THROUGH / FLOW FEATURES
# ============================================================

def build_flow_features(payments):
    df = payments[
        ["TRANSACTION_HASH", "FROM_ADDRESS",
         "TO_ADDRESS", "TIMESTAMP", "usd_value"]
    ].copy()

    df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"])
    result = df[["TRANSACTION_HASH"]].copy()

    incoming_total = df.groupby(
        "TO_ADDRESS"
    )["usd_value"].sum()

    outgoing_total = df.groupby(
        "FROM_ADDRESS"
    )["usd_value"].sum()

    sender_in = df["FROM_ADDRESS"].map(
        incoming_total
    ).fillna(0)

    sender_out = df["FROM_ADDRESS"].map(
        outgoing_total
    ).fillna(0)

    receiver_in = df["TO_ADDRESS"].map(
        incoming_total
    ).fillna(0)

    receiver_out = df["TO_ADDRESS"].map(
        outgoing_total
    ).fillna(0)

    result["sender_flow_ratio"] = (
        sender_out / (sender_in + 1.0)
    )
    result["receiver_flow_ratio"] = (
        receiver_out / (receiver_in + 1.0)
    )

    result["sender_net_flow_usd"] = sender_in - sender_out
    result["receiver_net_flow_usd"] = receiver_in - receiver_out

    # Time since sender last received funds
    incoming_history = df.sort_values(
        ["TO_ADDRESS", "TIMESTAMP"]
    )

    incoming_times = {
        address: group["TIMESTAMP"].to_numpy()
        for address, group in incoming_history.groupby(
            "TO_ADDRESS"
        )
    }

    seconds_since_incoming = []

    for address, timestamp in zip(
        df["FROM_ADDRESS"], df["TIMESTAMP"]
    ):
        times = incoming_times.get(address)

        if times is None or len(times) == 0:
            seconds_since_incoming.append(-1)
            continue

        pos = np.searchsorted(
            times, timestamp, side="left"
        )

        if pos == 0:
            seconds_since_incoming.append(-1)
        else:
            seconds_since_incoming.append(
                timestamp - times[pos - 1]
            )

    result["seconds_since_sender_received"] = (
        seconds_since_incoming
    )

    result["sender_received_within_1h"] = (
        (result["seconds_since_sender_received"] >= 0)
        & (result["seconds_since_sender_received"] <= 3600)
    ).astype(int)

    history = df.sort_values(
        ["TIMESTAMP", "TRANSACTION_HASH"]
    )

    sender_in_24h = pd.Series(0.0, index=df.index)
    sender_out_24h = pd.Series(0.0, index=df.index)

    for address, group in history.groupby(
        "FROM_ADDRESS"
    ):
        times = group["TIMESTAMP"].to_numpy()
        amounts = group["usd_value"].to_numpy()

        end = np.searchsorted(
            times, times, side="left"
        )
        start = np.searchsorted(
            times, times - 86400, side="left"
        )

        cumulative = np.concatenate(
            ([0.0], np.cumsum(amounts))
        )

        sender_out_24h.loc[group.index] = (
            cumulative[end] - cumulative[start]
        )

    for address, group in history.groupby(
        "TO_ADDRESS"
    ):
        times = group["TIMESTAMP"].to_numpy()
        amounts = group["usd_value"].to_numpy()

        cumulative = np.concatenate(
            ([0.0], np.cumsum(amounts))
        )

        sent = history[
            history["FROM_ADDRESS"] == address
        ]

        if sent.empty:
            continue

        sent_times = sent["TIMESTAMP"].to_numpy()

        sent_end = np.searchsorted(
            times, sent_times, side="left"
        )
        sent_start = np.searchsorted(
            times, sent_times - 86400, side="left"
        )

        sender_in_24h.loc[sent.index] = (
            cumulative[sent_end] - cumulative[sent_start]
        )

    result["sender_prior_incoming_usd_24h"] = (
        sender_in_24h
    )
    result["sender_prior_outgoing_usd_24h"] = (
        sender_out_24h
    )
    result["sender_pass_through_ratio_24h"] = (
        sender_out_24h / (sender_in_24h + 1.0)
    )
    result["sender_net_inflow_24h"] = (
        sender_in_24h - sender_out_24h
    )

    return result


# ============================================================
# 6. GRAPH-BASED FEATURES
# ============================================================

def build_graph_features(payments):
    df = payments[
        ["TRANSACTION_HASH", "FROM_ADDRESS", "TO_ADDRESS"]
    ].copy()

    outgoing = df.groupby(
        "FROM_ADDRESS"
    )["TO_ADDRESS"].nunique()

    incoming = df.groupby(
        "TO_ADDRESS"
    )["FROM_ADDRESS"].nunique()

    result = df[["TRANSACTION_HASH"]].copy()

    result["sender_out_degree"] = (
        df["FROM_ADDRESS"]
        .map(outgoing).fillna(0).astype(int)
    )
    result["sender_in_degree"] = (
        df["FROM_ADDRESS"]
        .map(incoming).fillna(0).astype(int)
    )
    result["receiver_out_degree"] = (
        df["TO_ADDRESS"]
        .map(outgoing).fillna(0).astype(int)
    )
    result["receiver_in_degree"] = (
        df["TO_ADDRESS"]
        .map(incoming).fillna(0).astype(int)
    )

    result["sender_degree_ratio"] = (
        result["sender_out_degree"]
        / (result["sender_in_degree"] + 1)
    )
    result["receiver_degree_ratio"] = (
        result["receiver_out_degree"]
        / (result["receiver_in_degree"] + 1)
    )

    edges = set(
        zip(df["FROM_ADDRESS"], df["TO_ADDRESS"])
    )

    result["is_reciprocal"] = [
        int((receiver, sender) in edges)
        for sender, receiver in zip(
            df["FROM_ADDRESS"], df["TO_ADDRESS"]
        )
    ]

    pair_counts = df.groupby(
        ["FROM_ADDRESS", "TO_ADDRESS"]
    ).size()

    result["pair_transaction_count"] = [
        pair_counts.loc[(sender, receiver)]
        for sender, receiver in zip(
            df["FROM_ADDRESS"], df["TO_ADDRESS"]
        )
    ]

    return result


# ============================================================
# 7. XRPL-SPECIFIC FEATURES
# ============================================================

def build_xrpl_features(transactions, transfers):
    df = transfers.copy()

    df["is_path"] = (
        df["TRANSFER_TYPE"] == "path_payment"
    ).astype(int)

    df["is_direct"] = (
        df["TRANSFER_TYPE"] == "direct_payment"
    ).astype(int)

    df["is_xrp"] = (
        df["SYMBOL"] == "XRP"
    ).astype(int)

    leg_features = df.groupby("TRANSACTION_HASH").agg(
        path_leg_count=("is_path", "sum"),
        direct_leg_count=("is_direct", "sum"),
        xrp_leg_count=("is_xrp", "sum"),
        max_transfer_index=("TRANSFER_INDEX", "max"),
    ).reset_index()

    total_legs = (
        leg_features["path_leg_count"]
        + leg_features["direct_leg_count"]
    ).clip(lower=1)

    leg_features["path_leg_fraction"] = (
        leg_features["path_leg_count"] / total_legs
    )

    leg_features["xrp_leg_fraction"] = (
        leg_features["xrp_leg_count"] / total_legs
    )

    tx = transactions[
        ["TRANSACTION_HASH", "FEE", "SEQUENCE"]
    ].copy()

    tx["fee_xrp"] = pd.to_numeric(
        tx["FEE"], errors="coerce"
    )

    tx["account_sequence"] = pd.to_numeric(
        tx["SEQUENCE"], errors="coerce"
    )

    tx = tx[
        ["TRANSACTION_HASH", "fee_xrp", "account_sequence"]
    ]

    return leg_features.merge(
        tx,
        on="TRANSACTION_HASH",
        how="left",
        validate="one_to_one"
    )


# ============================================================
# 8. COMBINE ALL FEATURES
# ============================================================

def build_features(directory):
    directory = Path(directory)

    transactions = pd.read_parquet(
        directory / "transactions.parquet"
    )

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
    time_features = build_time_features(payments)
    flow_features = build_flow_features(payments)
    graph_features = build_graph_features(payments)

    xrpl_features = build_xrpl_features(
        transactions, transfers
    )

    features = leg_features.merge(
        payment_features,
        on="TRANSACTION_HASH",
        how="left",
        validate="one_to_one"
    )

    for additional in [
        account_features,
        time_features,
        flow_features,
        graph_features,
        xrpl_features
    ]:
        features = features.merge(
            additional,
            on="TRANSACTION_HASH",
            how="left",
            validate="one_to_one"
        )

    assert features["TRANSACTION_HASH"].is_unique
    assert len(features) == len(leg_features)

    assert set(features["TRANSACTION_HASH"]) == set(
        payments["TRANSACTION_HASH"]
    ), "Some payment transactions are missing from features"

    return features


# ============================================================
# 9. RUN AND EXPORT
# ============================================================

if __name__ == "__main__":

    for split in ["train", "validation"]:
        features = build_features(ROOT / split)

        output_path = ROOT / f"{split}_features.csv"

        features.to_csv(output_path, index=False)

        print(f"\n{split.upper()}")
        print("Feature shape:", features.shape)
        print(
            "Missing USD values:",
            features["usd_value"].isna().sum()
        )
        print("Saved to:", output_path)
        print(features.head().to_string(index=False))
