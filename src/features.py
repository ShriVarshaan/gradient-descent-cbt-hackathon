
import sys
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from payments import load_payments


# ============================================================
# 1. TRANSACTION LEG FEATURES
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
# 2. PAYMENT FEATURES
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
# 3. HISTORICAL ACCOUNT AND GRAPH FEATURES
# ============================================================

def build_historical_account_graph_features(payments):
    df = payments[
        ["TRANSACTION_HASH", "FROM_ADDRESS",
         "TO_ADDRESS", "TIMESTAMP", "usd_value"]
    ].copy()

    df["TIMESTAMP"] = pd.to_numeric(
        df["TIMESTAMP"], errors="raise"
    )

    df["usd_value"] = pd.to_numeric(
        df["usd_value"], errors="coerce"
    ).fillna(0)

    df = df.sort_values(
        ["TIMESTAMP", "TRANSACTION_HASH"]
    ).reset_index(drop=True)

    names = [
        "sender_tx_count",
        "sender_total_usd",
        "sender_unique_receivers",
        "receiver_tx_count",
        "receiver_total_usd",
        "receiver_unique_senders",
        "sender_avg_usd",
        "receiver_avg_usd",
        "sender_flow_ratio",
        "receiver_flow_ratio",
        "sender_net_flow_usd",
        "receiver_net_flow_usd",
        "sender_out_degree",
        "sender_in_degree",
        "receiver_out_degree",
        "receiver_in_degree",
        "sender_degree_ratio",
        "receiver_degree_ratio",
        "is_reciprocal",
        "pair_transaction_count",
    ]

    output = {
        name: np.zeros(len(df), dtype=float)
        for name in names
    }

    outgoing_count = defaultdict(int)
    incoming_count = defaultdict(int)

    outgoing_usd = defaultdict(float)
    incoming_usd = defaultdict(float)

    outgoing_neighbors = defaultdict(set)
    incoming_neighbors = defaultdict(set)

    pair_counts = defaultdict(int)

    times = df["TIMESTAMP"].to_numpy()
    senders = df["FROM_ADDRESS"].to_numpy()
    receivers = df["TO_ADDRESS"].to_numpy()
    values = df["usd_value"].to_numpy()

    i = 0
    n = len(df)

    while i < n:
        timestamp = times[i]

        j = i + 1
        while j < n and times[j] == timestamp:
            j += 1

        # Calculate features before updating current batch
        for k in range(i, j):
            sender = senders[k]
            receiver = receivers[k]

            s_out = outgoing_count[sender]
            s_in = incoming_count[sender]
            r_out = outgoing_count[receiver]
            r_in = incoming_count[receiver]

            s_out_usd = outgoing_usd[sender]
            s_in_usd = incoming_usd[sender]
            r_out_usd = outgoing_usd[receiver]
            r_in_usd = incoming_usd[receiver]

            output["sender_tx_count"][k] = s_out
            output["sender_total_usd"][k] = s_out_usd
            output["sender_unique_receivers"][k] = len(
                outgoing_neighbors[sender]
            )

            output["receiver_tx_count"][k] = r_in
            output["receiver_total_usd"][k] = r_in_usd
            output["receiver_unique_senders"][k] = len(
                incoming_neighbors[receiver]
            )

            output["sender_avg_usd"][k] = (
                s_out_usd / max(s_out, 1)
            )
            output["receiver_avg_usd"][k] = (
                r_in_usd / max(r_in, 1)
            )

            output["sender_flow_ratio"][k] = (
                s_out_usd / (s_in_usd + 1.0)
            )
            output["receiver_flow_ratio"][k] = (
                r_out_usd / (r_in_usd + 1.0)
            )

            output["sender_net_flow_usd"][k] = (
                s_in_usd - s_out_usd
            )
            output["receiver_net_flow_usd"][k] = (
                r_in_usd - r_out_usd
            )

            s_out_degree = len(outgoing_neighbors[sender])
            s_in_degree = len(incoming_neighbors[sender])
            r_out_degree = len(outgoing_neighbors[receiver])
            r_in_degree = len(incoming_neighbors[receiver])

            output["sender_out_degree"][k] = s_out_degree
            output["sender_in_degree"][k] = s_in_degree
            output["receiver_out_degree"][k] = r_out_degree
            output["receiver_in_degree"][k] = r_in_degree

            output["sender_degree_ratio"][k] = (
                s_out_degree / (s_in_degree + 1)
            )
            output["receiver_degree_ratio"][k] = (
                r_out_degree / (r_in_degree + 1)
            )

            output["is_reciprocal"][k] = int(
                (receiver, sender) in pair_counts
            )

            output["pair_transaction_count"][k] = (
                pair_counts[(sender, receiver)]
            )

        # Add current batch to historical state
        for k in range(i, j):
            sender = senders[k]
            receiver = receivers[k]
            value = values[k]

            outgoing_count[sender] += 1
            incoming_count[receiver] += 1

            outgoing_usd[sender] += value
            incoming_usd[receiver] += value

            outgoing_neighbors[sender].add(receiver)
            incoming_neighbors[receiver].add(sender)

            pair_counts[(sender, receiver)] += 1

        i = j

    result = pd.DataFrame(output)
    result.insert(
        0, "TRANSACTION_HASH",
        df["TRANSACTION_HASH"].to_numpy()
    )

    return result


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
# 5. HISTORICAL FLOW FEATURES
# ============================================================

def build_flow_features(payments):
    df = payments[
        ["TRANSACTION_HASH", "FROM_ADDRESS",
         "TO_ADDRESS", "TIMESTAMP", "usd_value"]
    ].copy()

    df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"])
    df["usd_value"] = pd.to_numeric(
        df["usd_value"], errors="coerce"
    ).fillna(0)

    result = df[["TRANSACTION_HASH"]].copy()

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

    for _, group in history.groupby(
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
# 6. XRPL-SPECIFIC FEATURES
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
# 7. 7-DAY ACTIVITY, BURSTINESS AND OBSERVED ACCOUNT AGE
# ============================================================

def build_behavior_features(payments):
    df = payments[
        ["TRANSACTION_HASH", "FROM_ADDRESS",
         "TO_ADDRESS", "TIMESTAMP", "usd_value"]
    ].copy()

    df["TIMESTAMP"] = pd.to_numeric(
        df["TIMESTAMP"], errors="raise"
    )
    df["usd_value"] = pd.to_numeric(
        df["usd_value"], errors="coerce"
    ).fillna(0)

    result = df[["TRANSACTION_HASH"]].copy()

    for address_col, prefix in [
        ("FROM_ADDRESS", "sender"),
        ("TO_ADDRESS", "receiver")
    ]:
        activity = df.sort_values(
            [address_col, "TIMESTAMP", "TRANSACTION_HASH"]
        )

        count_7d = pd.Series(0, index=df.index, dtype="int64")
        volume_7d = pd.Series(0.0, index=df.index)
        count_24h = pd.Series(0, index=df.index, dtype="int64")
        count_1h = pd.Series(0, index=df.index, dtype="int64")
        observed_age = pd.Series(-1.0, index=df.index)
        seconds_since_last = pd.Series(-1.0, index=df.index)

        for _, group in activity.groupby(
            address_col, sort=False
        ):
            times = group["TIMESTAMP"].to_numpy(
                dtype="float64"
            )
            amounts = group["usd_value"].to_numpy(
                dtype="float64"
            )

            end = np.searchsorted(
                times, times, side="left"
            )

            start_7d = np.searchsorted(
                times, times - 7 * 86400, side="left"
            )
            start_24h = np.searchsorted(
                times, times - 86400, side="left"
            )
            start_1h = np.searchsorted(
                times, times - 3600, side="left"
            )

            cumulative = np.concatenate(
                ([0.0], np.cumsum(amounts))
            )

            count_7d.loc[group.index] = end - start_7d
            volume_7d.loc[group.index] = (
                cumulative[end] - cumulative[start_7d]
            )

            count_24h.loc[group.index] = end - start_24h
            count_1h.loc[group.index] = end - start_1h

            has_history = end > 0

            ages = np.full(len(times), -1.0)
            ages[has_history] = (
                times[has_history] - times[0]
            )
            observed_age.loc[group.index] = ages

            gaps = np.full(len(times), -1.0)
            gaps[has_history] = (
                times[has_history]
                - times[end[has_history] - 1]
            )
            seconds_since_last.loc[group.index] = gaps

        result[f"{prefix}_tx_count_7d"] = count_7d
        result[f"{prefix}_usd_7d"] = volume_7d
        result[f"{prefix}_observed_age_seconds"] = observed_age
        result[f"{prefix}_seconds_since_last_tx"] = (
            seconds_since_last
        )

        result[f"{prefix}_burst_ratio_24h"] = (
            (count_24h + 1)
            / (count_7d / 7.0 + 1)
        )

        result[f"{prefix}_burst_ratio_1h"] = (
            (count_1h + 1)
            / (count_7d / 168.0 + 1)
        )

        result[f"{prefix}_activity_concentration_24h"] = (
            count_24h / count_7d.clip(lower=1)
        )

        result[f"{prefix}_activity_concentration_1h"] = (
            count_1h / count_7d.clip(lower=1)
        )

    return result


# ============================================================
# 8. MULTI-HOP, PASS-THROUGH AND STRUCTURING FEATURES
# ============================================================

def build_laundering_pattern_features(payments):
    df = payments[
        ["TRANSACTION_HASH", "FROM_ADDRESS",
         "TO_ADDRESS", "TIMESTAMP", "usd_value"]
    ].copy()

    df["TIMESTAMP"] = pd.to_numeric(
        df["TIMESTAMP"], errors="raise"
    )
    df["usd_value"] = pd.to_numeric(
        df["usd_value"], errors="coerce"
    ).fillna(0)

    df = df.sort_values(
        ["TIMESTAMP", "TRANSACTION_HASH"]
    ).reset_index(drop=True)

    n = len(df)

    feature_names = [
        "sender_recent_incoming_count_1h",
        "sender_recent_incoming_usd_1h",
        "sender_recent_incoming_count_24h",
        "sender_recent_incoming_usd_24h",
        "sender_upstream_unique_24h",
        "sender_incoming_outgoing_similarity",
        "sender_last_incoming_gap_seconds",
        "sender_rapid_forwarding",
        "sender_two_hop_chain",
        "sender_three_hop_chain",
        "sender_small_outgoing_count_24h",
        "sender_small_outgoing_usd_24h",
        "sender_small_payment_fraction_24h",
        "sender_repeated_amount_count_24h",
        "sender_outgoing_concentration_24h",
        "sender_incoming_to_current_ratio",
    ]

    output = {
        name: np.zeros(n, dtype=float)
        for name in feature_names
    }

    incoming = defaultdict(deque)
    outgoing = defaultdict(deque)

    window = 86400
    small_threshold = 1000.0

    times = df["TIMESTAMP"].to_numpy()
    senders = df["FROM_ADDRESS"].to_numpy()
    receivers = df["TO_ADDRESS"].to_numpy()
    values = df["usd_value"].to_numpy()

    i = 0

    while i < n:
        timestamp = times[i]

        j = i + 1
        while j < n and times[j] == timestamp:
            j += 1

        pending = []

        for k in range(i, j):
            sender = senders[k]
            receiver = receivers[k]
            value = values[k]

            sender_in = incoming[sender]
            sender_out = outgoing[sender]

            while (
                sender_in
                and sender_in[0][0] < timestamp - window
            ):
                sender_in.popleft()

            while (
                sender_out
                and sender_out[0][0] < timestamp - window
            ):
                sender_out.popleft()

            in_24h = list(sender_in)
            out_24h = list(sender_out)

            in_1h = [
                x for x in in_24h
                if x[0] >= timestamp - 3600
            ]

            output[
                "sender_recent_incoming_count_1h"
            ][k] = len(in_1h)

            output[
                "sender_recent_incoming_usd_1h"
            ][k] = sum(x[2] for x in in_1h)

            output[
                "sender_recent_incoming_count_24h"
            ][k] = len(in_24h)

            incoming_usd = sum(x[2] for x in in_24h)

            output[
                "sender_recent_incoming_usd_24h"
            ][k] = incoming_usd

            output[
                "sender_upstream_unique_24h"
            ][k] = len(set(x[1] for x in in_24h))

            if in_24h:
                last_incoming = in_24h[-1]
                gap = timestamp - last_incoming[0]

                output[
                    "sender_last_incoming_gap_seconds"
                ][k] = gap

                output[
                    "sender_rapid_forwarding"
                ][k] = int(gap <= 3600)

                incoming_amounts = np.array(
                    [x[2] for x in in_24h]
                )

                similarities = (
                    np.minimum(incoming_amounts, value)
                    / np.maximum(
                        np.maximum(incoming_amounts, value),
                        1e-9
                    )
                )

                output[
                    "sender_incoming_outgoing_similarity"
                ][k] = similarities.max()

                output[
                    "sender_incoming_to_current_ratio"
                ][k] = incoming_usd / (value + 1.0)

                depth = max(x[3] for x in in_24h)

                output[
                    "sender_two_hop_chain"
                ][k] = int(depth >= 1)

                output[
                    "sender_three_hop_chain"
                ][k] = int(depth >= 2)

                new_depth = min(depth + 1, 3)

            else:
                output[
                    "sender_last_incoming_gap_seconds"
                ][k] = -1

                new_depth = 0

            small_outgoing = [
                x for x in out_24h
                if x[2] < small_threshold
            ]

            output[
                "sender_small_outgoing_count_24h"
            ][k] = len(small_outgoing)

            output[
                "sender_small_outgoing_usd_24h"
            ][k] = sum(x[2] for x in small_outgoing)

            output[
                "sender_small_payment_fraction_24h"
            ][k] = (
                len(small_outgoing) / len(out_24h)
                if out_24h else 0
            )

            output[
                "sender_repeated_amount_count_24h"
            ][k] = sum(
                abs(x[2] - value)
                <= max(value * 0.01, 0.01)
                for x in out_24h
            )

            if out_24h:
                by_receiver = defaultdict(float)

                for _, counterparty, amount, _ in out_24h:
                    by_receiver[counterparty] += amount

                total = sum(by_receiver.values())

                output[
                    "sender_outgoing_concentration_24h"
                ][k] = (
                    max(by_receiver.values()) / total
                    if total > 0 else 0
                )

            pending.append(
                (timestamp, sender, receiver, value, new_depth)
            )

        for t, sender, receiver, value, depth in pending:
            outgoing[sender].append(
                (t, receiver, value, depth)
            )
            incoming[receiver].append(
                (t, sender, value, depth)
            )

        i = j

    result = pd.DataFrame(output)
    result.insert(
        0, "TRANSACTION_HASH",
        df["TRANSACTION_HASH"].to_numpy()
    )

    return result


# ============================================================
# 9. COMBINE ALL FEATURES
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

    historical_features = (
        build_historical_account_graph_features(payments)
    )

    time_features = build_time_features(payments)
    flow_features = build_flow_features(payments)

    xrpl_features = build_xrpl_features(
        transactions, transfers
    )

    behavior_features = build_behavior_features(payments)

    laundering_features = build_laundering_pattern_features(
        payments
    )

    features = leg_features.merge(
        payment_features,
        on="TRANSACTION_HASH",
        how="left",
        validate="one_to_one"
    )

    for additional in [
        historical_features,
        time_features,
        flow_features,
        xrpl_features,
        behavior_features,
        laundering_features
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
# 10. RUN AND EXPORT
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

        numeric = features.drop(
            columns=["TRANSACTION_HASH"]
        )

        print(
            "Missing feature values:",
            int(numeric.isna().sum().sum())
        )
        print(
            "Infinite feature values:",
            int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        )
