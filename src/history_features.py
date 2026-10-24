"""Past-only account, velocity, flow, XRPL, behaviour and laundering-pattern features.

Ported from feature/feature-engineering (e925529, 7519be8). Every account feature uses only
payments strictly before the current timestamp, so ties at the same second are
treated the same regardless of row order.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# Columns beyond FX_FEATURES; the first 8 account columns of
# build_historical_account_graph_features are already listed there.
HISTORY_FEATURES = [
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
    "sender_tx_count_1h",
    "sender_usd_1h",
    "sender_tx_count_24h",
    "sender_usd_24h",
    "receiver_tx_count_1h",
    "receiver_usd_1h",
    "receiver_tx_count_24h",
    "receiver_usd_24h",
    "seconds_since_sender_received",
    "sender_received_within_1h",
    "sender_prior_incoming_usd_24h",
    "sender_prior_outgoing_usd_24h",
    "sender_pass_through_ratio_24h",
    "sender_net_inflow_24h",
    "path_leg_count",
    "direct_leg_count",
    "xrp_leg_count",
    "max_transfer_index",
    "path_leg_fraction",
    "xrp_leg_fraction",
    "fee_xrp",
    "account_sequence",
    "sender_tx_count_7d",
    "sender_usd_7d",
    "sender_observed_age_seconds",
    "sender_seconds_since_last_tx",
    "sender_burst_ratio_24h",
    "sender_burst_ratio_1h",
    "sender_activity_concentration_24h",
    "sender_activity_concentration_1h",
    "receiver_tx_count_7d",
    "receiver_usd_7d",
    "receiver_observed_age_seconds",
    "receiver_seconds_since_last_tx",
    "receiver_burst_ratio_24h",
    "receiver_burst_ratio_1h",
    "receiver_activity_concentration_24h",
    "receiver_activity_concentration_1h",
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
    "matched_incoming_ratio_1h",
    "matched_incoming_gap_seconds",
    "two_hop_amount_similarity",
    "three_hop_amount_similarity",
    "fanin_unique_senders_1h",
    "fanout_unique_receivers_1h",
    "fanin_fanout_ratio_24h",
    "sender_cycle_2hop",
    "sender_cycle_3hop",
    "sender_near_identical_burst_1h",
    "fanin_usd_1h",
    "fanout_usd_1h",
    "fanin_to_current_ratio_1h",
    "sender_chain_match_count_24h",
    "amount_preserving_2hop_90",
    "amount_preserving_3hop_90",
    "rapid_amount_preserving_2hop_1h",
]

# Aggregation and neighbourhood columns from feature/feature-engineering (7519be8).
AGG_TOPO_FEATURES = [
    "aggregate_incoming_sum_1h",
    "aggregate_incoming_count_1h",
    "aggregate_incoming_unique_senders_1h",
    "aggregate_incoming_match_1h",
    "aggregate_incoming_match_24h",
    "best_two_incoming_match_24h",
    "best_three_incoming_match_24h",
    "aggregate_match_sender_count_1h",
    "aggregate_incoming_span_seconds_1h",
    "incoming_to_outgoing_volume_ratio_24h",
    "outgoing_to_incoming_volume_ratio_24h",
    "aggregate_outgoing_count_1h",
    "shared_sender_neighbors_24h",
    "shared_receiver_neighbors_24h",
    "shared_neighbor_count_24h",
    "sender_two_hop_reach_24h",
    "receiver_two_hop_reach_24h",
    "sender_common_destinations_24h",
    "receiver_common_sources_24h",
    "pair_reverse_count_24h",
    "sender_neighbor_activity_1h",
    "receiver_neighbor_activity_1h",
    "sender_neighbor_burst_1h",
    "receiver_neighbor_burst_1h",
]
HISTORY_FEATURES += AGG_TOPO_FEATURES

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

    # One lookup per address instead of a full-table scan per receiver
    sent_by_address = dict(tuple(history.groupby("FROM_ADDRESS")))

    for address, group in history.groupby(
        "TO_ADDRESS"
    ):
        times = group["TIMESTAMP"].to_numpy()
        amounts = group["usd_value"].to_numpy()

        cumulative = np.concatenate(
            ([0.0], np.cumsum(amounts))
        )

        sent = sent_by_address.get(address)

        if sent is None:
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

    # TRANSFER_INDEX is stored as text; compare it as a number
    df["TRANSFER_INDEX"] = pd.to_numeric(df["TRANSFER_INDEX"])

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
# 9. AMOUNT-MATCHED CHAINS, FAN-IN/OUT AND CYCLES
# ============================================================

def build_advanced_graph_features(payments):
    feature_names = [
        "matched_incoming_ratio_1h",
        "matched_incoming_gap_seconds",
        "two_hop_amount_similarity",
        "three_hop_amount_similarity",
        "fanin_unique_senders_1h",
        "fanout_unique_receivers_1h",
        "fanin_fanout_ratio_24h",
        "sender_cycle_2hop",
        "sender_cycle_3hop",
        "sender_near_identical_burst_1h",
        "fanin_usd_1h",
        "fanout_usd_1h",
        "fanin_to_current_ratio_1h",
        "sender_chain_match_count_24h",
        "amount_preserving_2hop_90",
        "amount_preserving_3hop_90",
        "rapid_amount_preserving_2hop_1h",
    ]

    HOUR = 3600
    DAY = 86400

    df = payments[
        [
            "TRANSACTION_HASH",
            "TIMESTAMP",
            "FROM_ADDRESS",
            "TO_ADDRESS",
            "usd_value",
        ]
    ].copy()

    df["TIMESTAMP"] = pd.to_numeric(
        df["TIMESTAMP"], errors="raise"
    )
    df["usd_value"] = pd.to_numeric(
        df["usd_value"], errors="coerce"
    ).fillna(0.0)

    df = df.sort_values(
        ["TIMESTAMP", "TRANSACTION_HASH"],
        kind="mergesort"
    ).reset_index(drop=True)

    n = len(df)

    output = {
        name: np.zeros(n, dtype=float)
        for name in feature_names
    }

    output["matched_incoming_gap_seconds"][:] = -1.0

    # Each incoming record:
    # (timestamp, sender, amount, previous_chain_similarity)
    incoming = defaultdict(deque)

    # Each outgoing record:
    # (timestamp, receiver, amount, previous_chain_similarity)
    outgoing = defaultdict(deque)

    def amount_similarity(a, b):
        a = abs(float(a))
        b = abs(float(b))

        if a == 0 and b == 0:
            return 1.0

        if a == 0 or b == 0:
            return 0.0

        return min(a, b) / max(a, b)

    def prune(history, current_time):
        cutoff = current_time - DAY

        while history and history[0][0] < cutoff:
            history.popleft()

    records = df.to_dict("records")

    start = 0

    while start < n:
        current_time = records[start]["TIMESTAMP"]

        end = start + 1

        while (
            end < n
            and records[end]["TIMESTAMP"] == current_time
        ):
            end += 1

        pending_updates = []

        # Calculate features before inserting transactions
        # with the same timestamp into history.
        for k in range(start, end):
            row = records[k]

            sender = row["FROM_ADDRESS"]
            receiver = row["TO_ADDRESS"]
            amount = max(float(row["usd_value"]), 0.0)

            sender_in = incoming[sender]
            sender_out = outgoing[sender]
            receiver_out = outgoing[receiver]

            prune(sender_in, current_time)
            prune(sender_out, current_time)
            prune(receiver_out, current_time)

            # -----------------------------------------
            # 1. Incoming amount matching
            # -----------------------------------------

            best_one_hour = 0.0
            best_two_hop = 0.0
            best_three_hop = 0.0

            best_match_gap = -1.0
            best_match_score = -1.0

            chain_match_count = 0

            incoming_1h = []
            incoming_24h = list(sender_in)

            for previous in incoming_24h:
                previous_time = previous[0]
                previous_amount = previous[2]
                previous_chain_similarity = previous[3]

                similarity = amount_similarity(
                    amount, previous_amount
                )

                gap = current_time - previous_time

                best_two_hop = max(
                    best_two_hop, similarity
                )

                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_gap = gap

                if similarity >= 0.90:
                    chain_match_count += 1

                if previous_chain_similarity > 0:
                    best_three_hop = max(
                        best_three_hop,
                        min(
                            similarity,
                            previous_chain_similarity
                        )
                    )

                if gap <= HOUR:
                    incoming_1h.append(previous)

                    best_one_hour = max(
                        best_one_hour, similarity
                    )

            output["matched_incoming_ratio_1h"][k] = (
                best_one_hour
            )

            output["matched_incoming_gap_seconds"][k] = (
                best_match_gap
            )

            output["two_hop_amount_similarity"][k] = (
                best_two_hop
            )

            output["three_hop_amount_similarity"][k] = (
                best_three_hop
            )

            output["sender_chain_match_count_24h"][k] = (
                chain_match_count
            )

            # -----------------------------------------
            # 2. New amount-preserving flow indicators
            # -----------------------------------------

            output["amount_preserving_2hop_90"][k] = int(
                best_two_hop >= 0.90
            )

            output["amount_preserving_3hop_90"][k] = int(
                best_three_hop >= 0.90
            )

            output[
                "rapid_amount_preserving_2hop_1h"
            ][k] = int(
                best_one_hour >= 0.90
            )

            # -----------------------------------------
            # 3. Fan-in and fan-out
            # -----------------------------------------

            outgoing_1h = [
                previous
                for previous in sender_out
                if current_time - previous[0] <= HOUR
            ]

            output["fanin_unique_senders_1h"][k] = len({
                previous[1]
                for previous in incoming_1h
            })

            output["fanout_unique_receivers_1h"][k] = len({
                previous[1]
                for previous in outgoing_1h
            })

            unique_incoming_24h = len({
                previous[1]
                for previous in sender_in
            })

            unique_outgoing_24h = len({
                previous[1]
                for previous in sender_out
            })

            output["fanin_fanout_ratio_24h"][k] = (
                unique_incoming_24h
                / (unique_outgoing_24h + 1)
            )

            # -----------------------------------------
            # 4. Circular transfers
            # -----------------------------------------

            # Current A -> B completes a two-edge cycle
            # if B -> A occurred previously.
            #
            # receiver_out stores earlier B -> ... transfers.

            output["sender_cycle_2hop"][k] = int(
                sender != receiver
                and any(
                    previous[1] == sender
                    for previous in receiver_out
                )
            )

            # Current A -> B completes a three-edge
            # cycle if earlier B -> C -> A occurred.

            three_hop_cycle = False

            for previous in sender_in:
                intermediary = previous[1]
                intermediary_to_sender_time = previous[0]

                if intermediary in (sender, receiver):
                    continue

                intermediary_in = incoming[intermediary]
                prune(intermediary_in, current_time)

                if any(
                    earlier[1] == receiver
                    and earlier[0] < intermediary_to_sender_time
                    for earlier in intermediary_in
                ):
                    three_hop_cycle = True
                    break

            output["sender_cycle_3hop"][k] = int(
                three_hop_cycle
            )

            # -----------------------------------------
            # 5. Repeated near-identical payments
            # -----------------------------------------

            output[
                "sender_near_identical_burst_1h"
            ][k] = sum(
                amount_similarity(
                    amount, previous[2]
                ) >= 0.99
                for previous in outgoing_1h
            )

            # -----------------------------------------
            # 6. Recent incoming and outgoing volume
            # -----------------------------------------

            fanin_usd = sum(
                previous[2]
                for previous in incoming_1h
            )

            fanout_usd = sum(
                previous[2]
                for previous in outgoing_1h
            )

            output["fanin_usd_1h"][k] = fanin_usd
            output["fanout_usd_1h"][k] = fanout_usd

            output["fanin_to_current_ratio_1h"][k] = (
                fanin_usd / (amount + 1.0)
            )

            # -----------------------------------------
            # 7. Prepare history update
            # -----------------------------------------

            # For a future payment sent by the receiver,
            # this transaction becomes its incoming edge.
            #
            # The chain similarity records whether the
            # sender previously received a similar amount.

            pending_updates.append(
                (
                    current_time,
                    sender,
                    receiver,
                    amount,
                    best_two_hop,
                )
            )

        # Only update history after every transaction
        # at the current timestamp has been evaluated.

        for (
            timestamp,
            sender,
            receiver,
            amount,
            chain_similarity,
        ) in pending_updates:

            outgoing[sender].append(
                (
                    timestamp,
                    receiver,
                    amount,
                    chain_similarity,
                )
            )

            incoming[receiver].append(
                (
                    timestamp,
                    sender,
                    amount,
                    chain_similarity,
                )
            )

        start = end

    result = pd.DataFrame(output)
    result.insert(
        0,
        "TRANSACTION_HASH",
        df["TRANSACTION_HASH"].values
    )

    return result


def build_aggregate_flow_features(payments):
    from collections import defaultdict, deque
    from itertools import combinations

    HOUR = 3600
    DAY = 86400
    EPS = 1e-9

    feature_names = [
        "aggregate_incoming_sum_1h",
        "aggregate_incoming_count_1h",
        "aggregate_incoming_unique_senders_1h",
        "aggregate_incoming_match_1h",
        "aggregate_incoming_match_24h",
        "best_two_incoming_match_24h",
        "best_three_incoming_match_24h",
        "aggregate_match_sender_count_1h",
        "aggregate_incoming_span_seconds_1h",
        "incoming_to_outgoing_volume_ratio_24h",
        "outgoing_to_incoming_volume_ratio_24h",
        "aggregate_outgoing_count_1h",
    ]

    df = payments[
        [
            "TRANSACTION_HASH",
            "TIMESTAMP",
            "FROM_ADDRESS",
            "TO_ADDRESS",
            "usd_value",
        ]
    ].copy()

    df["TIMESTAMP"] = pd.to_numeric(
        df["TIMESTAMP"], errors="raise"
    )

    df["usd_value"] = pd.to_numeric(
        df["usd_value"], errors="coerce"
    ).fillna(0.0)

    df = df.sort_values(
        ["TIMESTAMP", "TRANSACTION_HASH"],
        kind="mergesort"
    ).reset_index(drop=True)

    n = len(df)

    output = {
        name: np.zeros(n, dtype=float)
        for name in feature_names
    }

    incoming = defaultdict(deque)
    outgoing = defaultdict(deque)

    def prune(history, current_time):
        cutoff = current_time - DAY

        while history and history[0][0] < cutoff:
            history.popleft()

    def similarity(a, b):
        a = max(float(a), 0.0)
        b = max(float(b), 0.0)

        if a <= EPS or b <= EPS:
            return 0.0

        return min(a, b) / max(a, b)

    def best_combination_match(amounts, target, size):
        if len(amounts) < size or target <= EPS:
            return 0.0

        # Bound the combinatorial search.
        # Use the most recent 30 incoming payments.
        recent = amounts[-30:]

        best = 0.0

        for group in combinations(recent, size):
            score = similarity(sum(group), target)

            if score > best:
                best = score

            if best >= 0.999999:
                break

        return best

    records = df.to_dict("records")

    start = 0

    while start < n:
        current_time = records[start]["TIMESTAMP"]

        end = start + 1

        while (
            end < n
            and records[end]["TIMESTAMP"] == current_time
        ):
            end += 1

        pending = []

        # Evaluate the entire timestamp batch before
        # adding any transactions from that timestamp.
        for k in range(start, end):
            row = records[k]

            sender = row["FROM_ADDRESS"]
            receiver = row["TO_ADDRESS"]
            amount = max(float(row["usd_value"]), 0.0)

            sender_in = incoming[sender]
            sender_out = outgoing[sender]

            prune(sender_in, current_time)
            prune(sender_out, current_time)

            in_24h = list(sender_in)
            out_24h = list(sender_out)

            in_1h = [
                record for record in in_24h
                if current_time - record[0] <= HOUR
            ]

            out_1h = [
                record for record in out_24h
                if current_time - record[0] <= HOUR
            ]

            # Records are:
            # (timestamp, counterparty, amount)

            in_sum_1h = sum(x[2] for x in in_1h)
            in_sum_24h = sum(x[2] for x in in_24h)

            out_sum_24h = sum(x[2] for x in out_24h)

            output["aggregate_incoming_sum_1h"][k] = (
                in_sum_1h
            )

            output["aggregate_incoming_count_1h"][k] = (
                len(in_1h)
            )

            output[
                "aggregate_incoming_unique_senders_1h"
            ][k] = len({
                x[1] for x in in_1h
            })

            output["aggregate_incoming_match_1h"][k] = (
                similarity(in_sum_1h, amount)
            )

            output["aggregate_incoming_match_24h"][k] = (
                similarity(in_sum_24h, amount)
            )

            # Look for 2 or 3 incoming payments whose
            # combined value matches this outgoing payment.
            incoming_amounts = [
                x[2] for x in in_24h
            ]

            output[
                "best_two_incoming_match_24h"
            ][k] = best_combination_match(
                incoming_amounts,
                amount,
                2
            )

            output[
                "best_three_incoming_match_24h"
            ][k] = best_combination_match(
                incoming_amounts,
                amount,
                3
            )

            # Number of distinct incoming counterparties
            # contributing to recent accumulated funds.
            output[
                "aggregate_match_sender_count_1h"
            ][k] = (
                len({x[1] for x in in_1h})
                if similarity(in_sum_1h, amount) >= 0.90
                else 0
            )

            if len(in_1h) >= 2:
                output[
                    "aggregate_incoming_span_seconds_1h"
                ][k] = (
                    in_1h[-1][0] - in_1h[0][0]
                )

            output[
                "incoming_to_outgoing_volume_ratio_24h"
            ][k] = (
                in_sum_24h / (out_sum_24h + amount + 1.0)
            )

            output[
                "outgoing_to_incoming_volume_ratio_24h"
            ][k] = (
                (out_sum_24h + amount)
                / (in_sum_24h + 1.0)
            )

            output[
                "aggregate_outgoing_count_1h"
            ][k] = len(out_1h)

            pending.append(
                (
                    current_time,
                    sender,
                    receiver,
                    amount,
                )
            )

        for timestamp, sender, receiver, amount in pending:
            outgoing[sender].append(
                (timestamp, receiver, amount)
            )
            incoming[receiver].append(
                (timestamp, sender, amount)
            )

        start = end

    result = pd.DataFrame(output)

    result.insert(
        0,
        "TRANSACTION_HASH",
        df["TRANSACTION_HASH"].values
    )

    return result

def build_graph_topology_features(payments):
    from collections import defaultdict, deque

    HOUR = 3600
    DAY = 86400

    feature_names = [
        "shared_sender_neighbors_24h",
        "shared_receiver_neighbors_24h",
        "shared_neighbor_count_24h",
        "sender_two_hop_reach_24h",
        "receiver_two_hop_reach_24h",
        "sender_common_destinations_24h",
        "receiver_common_sources_24h",
        "pair_reverse_count_24h",
        "sender_neighbor_activity_1h",
        "receiver_neighbor_activity_1h",
        "sender_neighbor_burst_1h",
        "receiver_neighbor_burst_1h",
    ]

    df = payments[
        [
            "TRANSACTION_HASH",
            "TIMESTAMP",
            "FROM_ADDRESS",
            "TO_ADDRESS",
        ]
    ].copy()

    df["TIMESTAMP"] = pd.to_numeric(
        df["TIMESTAMP"], errors="raise"
    )

    df = df.sort_values(
        ["TIMESTAMP", "TRANSACTION_HASH"],
        kind="mergesort"
    ).reset_index(drop=True)

    n = len(df)

    output = {
        name: np.zeros(n, dtype=float)
        for name in feature_names
    }

    # Historical outgoing and incoming edges.
    # Records contain (timestamp, counterparty).
    outgoing = defaultdict(deque)
    incoming = defaultdict(deque)

    def prune(history, now, window=DAY):
        cutoff = now - window

        while history and history[0][0] < cutoff:
            history.popleft()

    def recent(history, now, window):
        prune(history, now)
        return [
            item
            for item in history
            if now - item[0] <= window
        ]

    def outgoing_neighbors(account, now):
        return {
            item[1]
            for item in recent(outgoing[account], now, DAY)
        }

    def incoming_neighbors(account, now):
        return {
            item[1]
            for item in recent(incoming[account], now, DAY)
        }

    records = df.to_dict("records")

    start = 0

    while start < n:
        now = records[start]["TIMESTAMP"]

        end = start + 1

        while (
            end < n
            and records[end]["TIMESTAMP"] == now
        ):
            end += 1

        pending = []

        for k in range(start, end):
            row = records[k]

            sender = row["FROM_ADDRESS"]
            receiver = row["TO_ADDRESS"]

            sender_out = recent(outgoing[sender], now, DAY)
            sender_in = recent(incoming[sender], now, DAY)

            receiver_out = recent(outgoing[receiver], now, DAY)
            receiver_in = recent(incoming[receiver], now, DAY)

            sender_destinations = {
                x[1] for x in sender_out
            }

            sender_sources = {
                x[1] for x in sender_in
            }

            receiver_destinations = {
                x[1] for x in receiver_out
            }

            receiver_sources = {
                x[1] for x in receiver_in
            }

            # ----------------------------------------
            # 1. Shared neighbourhood structure
            # ----------------------------------------

            shared_senders = (
                sender_sources & receiver_sources
            )

            shared_receivers = (
                sender_destinations
                & receiver_destinations
            )

            output[
                "shared_sender_neighbors_24h"
            ][k] = len(shared_senders)

            output[
                "shared_receiver_neighbors_24h"
            ][k] = len(shared_receivers)

            output[
                "shared_neighbor_count_24h"
            ][k] = (
                len(shared_senders)
                + len(shared_receivers)
            )

            # ----------------------------------------
            # 2. Two-hop network reach
            # ----------------------------------------

            sender_two_hop = set()

            for neighbor in sender_destinations:
                sender_two_hop.update(
                    outgoing_neighbors(neighbor, now)
                )

            receiver_two_hop = set()

            for neighbor in receiver_sources:
                receiver_two_hop.update(
                    incoming_neighbors(neighbor, now)
                )

            sender_two_hop.discard(sender)
            receiver_two_hop.discard(receiver)

            output[
                "sender_two_hop_reach_24h"
            ][k] = len(sender_two_hop)

            output[
                "receiver_two_hop_reach_24h"
            ][k] = len(receiver_two_hop)

            # ----------------------------------------
            # 3. Shared destinations / sources
            # ----------------------------------------

            common_destinations = 0

            for source in sender_sources:
                source_destinations = (
                    outgoing_neighbors(source, now)
                )

                common_destinations += len(
                    source_destinations
                    & sender_destinations
                )

            common_sources = 0

            for destination in receiver_destinations:
                destination_sources = (
                    incoming_neighbors(destination, now)
                )

                common_sources += len(
                    destination_sources
                    & receiver_sources
                )

            output[
                "sender_common_destinations_24h"
            ][k] = common_destinations

            output[
                "receiver_common_sources_24h"
            ][k] = common_sources

            # ----------------------------------------
            # 4. Reverse transactions
            # ----------------------------------------

            output["pair_reverse_count_24h"][k] = sum(
                1
                for previous in receiver_out
                if previous[1] == sender
            )

            # ----------------------------------------
            # 5. Neighbour activity
            # ----------------------------------------

            sender_neighbor_activity = 0
            receiver_neighbor_activity = 0

            sender_active_neighbors = 0
            receiver_active_neighbors = 0

            for neighbor in sender_destinations:
                activity = len(
                    recent(outgoing[neighbor], now, HOUR)
                )

                sender_neighbor_activity += activity

                if activity >= 2:
                    sender_active_neighbors += 1

            for neighbor in receiver_sources:
                activity = len(
                    recent(incoming[neighbor], now, HOUR)
                )

                receiver_neighbor_activity += activity

                if activity >= 2:
                    receiver_active_neighbors += 1

            output[
                "sender_neighbor_activity_1h"
            ][k] = sender_neighbor_activity

            output[
                "receiver_neighbor_activity_1h"
            ][k] = receiver_neighbor_activity

            output[
                "sender_neighbor_burst_1h"
            ][k] = sender_active_neighbors

            output[
                "receiver_neighbor_burst_1h"
            ][k] = receiver_active_neighbors

            pending.append((now, sender, receiver))

        # Insert transactions only after processing
        # every transaction at the same timestamp.
        for timestamp, sender, receiver in pending:
            outgoing[sender].append(
                (timestamp, receiver)
            )

            incoming[receiver].append(
                (timestamp, sender)
            )

        start = end

    result = pd.DataFrame(output)

    result.insert(
        0,
        "TRANSACTION_HASH",
        df["TRANSACTION_HASH"].values
    )

    return result
