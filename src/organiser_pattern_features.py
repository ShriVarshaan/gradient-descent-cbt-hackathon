
"""
Decode & Detect: organiser-inspired AML features.

Covers:
- Common funders / mule networks
- Fast hops and money trails
- Repeated near-threshold payments
- Cross-currency transfers
- Unusual conversion prices
- Circular payments
- Delayed forwarding
- Missing destination tags

Account deletion is not simulated in this dataset.

Run:
    python src/organiser_pattern_features.py

Requires existing:
    payments.py
    train/transactions.parquet
    train/transfers.parquet
    train/market_rates.csv
    validation/...
    train_features_advanced.csv
    validation_features_advanced.csv

Optional:
    test-data/...
    test_features_advanced.csv

Outputs:
    train_features_patterns.csv
    validation_features_patterns.csv
    test_features_patterns.csv (if inputs exist)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from payments import load_payments

WINDOWS = (5, 15, 60, 360, 1440)
THRESHOLDS = (1000, 5000, 10000)
EPS = 1e-9


def safe_number(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError, OverflowError):
        return default


def parse_json(value):
    if isinstance(value, dict):
        return value
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (TypeError, ValueError):
        return {}


def load_split(folder):
    tx = pd.read_parquet(folder / "transactions.parquet")
    transfers = pd.read_parquet(folder / "transfers.parquet")
    rates = pd.read_csv(folder / "market_rates.csv")

    payments = pd.DataFrame(load_payments(folder))

    needed = [
        "TRANSACTION_HASH",
        "TIMESTAMP",
        "FROM_ADDRESS",
        "TO_ADDRESS",
        "VALUE",
    ]

    missing = set(needed) - set(payments.columns)
    if missing:
        raise ValueError(f"Payment columns missing: {missing}")

    payments = payments.copy()

    payments["TIMESTAMP"] = pd.to_numeric(
        payments["TIMESTAMP"], errors="coerce"
    )

    # payments.py projects customer transfers into reference USD.
    payments["usd"] = pd.to_numeric(
        payments["VALUE"], errors="coerce"
    ).fillna(0).clip(lower=0)

    payments = payments.dropna(
        subset=[
            "TRANSACTION_HASH",
            "TIMESTAMP",
            "FROM_ADDRESS",
            "TO_ADDRESS",
        ]
    )

    payments = payments.sort_values(
        ["TIMESTAMP", "TRANSACTION_HASH"],
        kind="mergesort",
    ).reset_index(drop=True)

    return tx, transfers, rates, payments


def account_network_features(p):
    """
    Full-window account network and common-funder features.
    Uses only the supplied ledger, never labels.
    """

    out = p[[
        "TRANSACTION_HASH",
        "FROM_ADDRESS",
        "TO_ADDRESS",
        "TIMESTAMP",
        "usd",
    ]].copy()

    senders = defaultdict(set)
    receivers = defaultdict(set)
    incoming_total = defaultdict(float)
    outgoing_total = defaultdict(float)
    incoming_count = defaultdict(int)
    outgoing_count = defaultdict(int)

    for r in p.itertuples(index=False):
        s = r.FROM_ADDRESS
        d = r.TO_ADDRESS
        amount = float(r.usd)

        receivers[s].add(d)
        senders[d].add(s)

        outgoing_total[s] += amount
        incoming_total[d] += amount

        outgoing_count[s] += 1
        incoming_count[d] += 1

    all_accounts = set(senders) | set(receivers)

    common_funders = defaultdict(set)

    # Wallets that receive from the same source.
    for funder, funded_wallets in receivers.items():
        for wallet in funded_wallets:
            common_funders[wallet].add(funder)

    account_features = {}

    for account in all_accounts:
        parents = senders[account]
        children = receivers[account]

        siblings = set()

        for parent in parents:
            siblings.update(receivers[parent])

        siblings.discard(account)

        shared_parent_max = 0

        for sibling in siblings:
            overlap = len(
                common_funders[account]
                & common_funders[sibling]
            )
            shared_parent_max = max(shared_parent_max, overlap)

        total_in = incoming_total[account]
        total_out = outgoing_total[account]

        account_features[account] = {
            "fund_in_degree": len(parents),
            "fund_out_degree": len(children),
            "fund_common_parent_wallets": len(siblings),
            "fund_shared_parent_max": shared_parent_max,
            "fund_in_tx_count": incoming_count[account],
            "fund_out_tx_count": outgoing_count[account],
            "fund_in_usd": total_in,
            "fund_out_usd": total_out,
            "fund_turnover_ratio": (
                min(total_in, total_out)
                / max(total_in, total_out, EPS)
            ),
            "fund_net_usd": total_in - total_out,
            "fund_is_intermediary": int(
                bool(parents) and bool(children)
            ),
            "fund_reciprocal_neighbors": len(
                parents & children
            ),
        }

    names = list(next(iter(account_features.values())).keys())

    rows = []

    for r in p.itertuples(index=False):
        s = account_features[r.FROM_ADDRESS]
        d = account_features[r.TO_ADDRESS]

        record = {"TRANSACTION_HASH": r.TRANSACTION_HASH}

        for name in names:
            record["sender_" + name] = s[name]
            record["receiver_" + name] = d[name]

        record["network_shared_funders"] = len(
            senders[r.FROM_ADDRESS]
            & senders[r.TO_ADDRESS]
        )

        record["network_sender_receiver_reciprocal"] = int(
            r.FROM_ADDRESS in receivers[r.TO_ADDRESS]
        )

        rows.append(record)

    return pd.DataFrame(rows)


def money_trail_features(p):
    """
    Full-window temporal matching and fast-hop features.

    For each payment A -> B:
      - Look for B -> C payments after it.
      - Look for X -> A payments before it.
      - Match amounts in reference USD.
      - Identify fast forwarding and longer delays.

    Uses the full seven-day window, permitted by competition.
    """

    p = p.reset_index(drop=True)

    outgoing = defaultdict(list)
    incoming = defaultdict(list)

    for i, r in enumerate(p.itertuples(index=False)):
        outgoing[r.FROM_ADDRESS].append(i)
        incoming[r.TO_ADDRESS].append(i)

    times = p["TIMESTAMP"].to_numpy(dtype=float)
    amounts = p["usd"].to_numpy(dtype=float)
    src = p["FROM_ADDRESS"].to_numpy()
    dst = p["TO_ADDRESS"].to_numpy()

    outgoing_times = {
        a: times[idx].tolist()
        for a, idx in outgoing.items()
    }

    incoming_times = {
        a: times[idx].tolist()
        for a, idx in incoming.items()
    }

    rows = []

    for i in range(len(p)):
        t = times[i]
        amount = max(amounts[i], EPS)
        s = src[i]
        d = dst[i]

        record = {
            "TRANSACTION_HASH": p.at[i, "TRANSACTION_HASH"]
        }

        # Earlier incoming transfers to this sender.
        prior_indices = incoming.get(s, [])
        prior_times = incoming_times.get(s, [])

        start = bisect_left(prior_times, t - 86400)
        end = bisect_left(prior_times, t)

        prior = prior_indices[start:end]

        # Subsequent outgoing transfers from this receiver.
        next_indices = outgoing.get(d, [])
        next_times = outgoing_times.get(d, [])

        start = bisect_right(next_times, t)
        end = bisect_right(next_times, t + 86400)

        following = next_indices[start:end]

        record["trail_prior_in_count_24h"] = len(prior)
        record["trail_next_out_count_24h"] = len(following)

        if prior:
            errors = [
                abs(amounts[j] - amount) / amount
                for j in prior
            ]

            best_pos = int(np.argmin(errors))
            best_idx = prior[best_pos]

            record["trail_prior_best_error"] = errors[best_pos]
            record["trail_prior_best_delay_min"] = (
                t - times[best_idx]
            ) / 60
        else:
            record["trail_prior_best_error"] = 1.0
            record["trail_prior_best_delay_min"] = -1.0

        if following:
            errors = [
                abs(amounts[j] - amount) / amount
                for j in following
            ]

            best_pos = int(np.argmin(errors))
            best_idx = following[best_pos]

            record["trail_next_best_error"] = errors[best_pos]
            record["trail_next_best_delay_min"] = (
                times[best_idx] - t
            ) / 60

            record["trail_next_best_ratio"] = (
                amounts[best_idx] / amount
            )

            record["trail_next_distinct_receivers"] = len({
                dst[j] for j in following
            })
        else:
            record["trail_next_best_error"] = 1.0
            record["trail_next_best_delay_min"] = -1.0
            record["trail_next_best_ratio"] = 0.0
            record["trail_next_distinct_receivers"] = 0

        for minutes in WINDOWS:
            seconds = minutes * 60

            prior_window = [
                j for j in prior
                if t - times[j] <= seconds
            ]

            next_window = [
                j for j in following
                if times[j] - t <= seconds
            ]

            prior_usd = sum(amounts[j] for j in prior_window)
            next_usd = sum(amounts[j] for j in next_window)

            record[f"trail_prior_count_{minutes}m"] = (
                len(prior_window)
            )

            record[f"trail_next_count_{minutes}m"] = (
                len(next_window)
            )

            record[f"trail_prior_usd_ratio_{minutes}m"] = (
                prior_usd / amount
            )

            record[f"trail_next_usd_ratio_{minutes}m"] = (
                next_usd / amount
            )

            record[f"trail_prior_match_{minutes}m"] = sum(
                abs(amounts[j] - amount) / amount <= 0.10
                for j in prior_window
            )

            record[f"trail_next_match_{minutes}m"] = sum(
                abs(amounts[j] - amount) / amount <= 0.10
                for j in next_window
            )

        # Approximate 3-hop trail:
        # A -> B -> C -> D, with each hop after the preceding one.
        # Bound candidate expansion to avoid exploding at hubs.
        two_hop_count = 0
        three_hop_count = 0

        first_hops = [
            j for j in following
            if times[j] - t <= 3600
            and abs(amounts[j] - amount) / amount <= 0.15
            and dst[j] != s
        ][:30]

        two_hop_count = len(first_hops)

        for j in first_hops:
            next_account = dst[j]

            indices = outgoing.get(next_account, [])
            ts = outgoing_times.get(next_account, [])

            lo = bisect_right(ts, times[j])
            hi = bisect_right(ts, min(t + 7200, times[j] + 3600))

            for k in indices[lo:hi][:30]:
                if (
                    dst[k] not in {s, d, next_account}
                    and abs(amounts[k] - amounts[j])
                    / max(amounts[j], EPS) <= 0.15
                ):
                    three_hop_count += 1

        record["trail_fast_two_hop_count"] = two_hop_count
        record["trail_fast_three_hop_count"] = three_hop_count

        rows.append(record)

    return pd.DataFrame(rows)


def threshold_features(p):
    """
    Repeated payments near round-number thresholds.

    These are exploratory thresholds, NOT asserted regulatory
    limits in the synthetic competition.
    """

    p = p.copy()
    p["hour"] = (
        p["TIMESTAMP"] // 3600
    ).astype("int64")

    p["day"] = (
        p["TIMESTAMP"] // 86400
    ).astype("int64")

    for threshold in THRESHOLDS:
        near = (
            (p["usd"] >= threshold * 0.90)
            & (p["usd"] < threshold)
        ).astype(int)

        p[f"limit_below_{threshold}"] = near

        p[f"limit_sender_day_{threshold}"] = (
            near.groupby(
                [p["FROM_ADDRESS"], p["day"]]
            ).transform("sum")
        )

        p[f"limit_sender_hour_{threshold}"] = (
            near.groupby(
                [p["FROM_ADDRESS"], p["hour"]]
            ).transform("sum")
        )

        p[f"limit_receiver_day_{threshold}"] = (
            near.groupby(
                [p["TO_ADDRESS"], p["day"]]
            ).transform("sum")
        )

        p[f"limit_distance_{threshold}"] = (
            (threshold - p["usd"]) / threshold
        ).clip(lower=0, upper=1)

    p["limit_sender_day_count"] = p.groupby(
        ["FROM_ADDRESS", "day"]
    )["TRANSACTION_HASH"].transform("size")

    p["limit_sender_hour_count"] = p.groupby(
        ["FROM_ADDRESS", "hour"]
    )["TRANSACTION_HASH"].transform("size")

    p["limit_sender_day_usd"] = p.groupby(
        ["FROM_ADDRESS", "day"]
    )["usd"].transform("sum")

    p["limit_sender_hour_usd"] = p.groupby(
        ["FROM_ADDRESS", "hour"]
    )["usd"].transform("sum")

    columns = [
        c for c in p.columns
        if c.startswith("limit_")
    ]

    return p[["TRANSACTION_HASH"] + columns]


def transaction_metadata_features(tx, transfers):
    """
    Destination tags, conversion legs, repeated asset paths,
    and payment-level transaction metadata.
    """

    tx = tx.copy()
    transfers = transfers.copy()

    fields = tx["EXTRA_TRANSACTION_FIELDS"].apply(parse_json)

    tx["meta_has_destination_tag"] = fields.apply(
        lambda d: int(
            "DestinationTag" in d
            and d["DestinationTag"] is not None
        )
    )

    tx["meta_missing_destination_tag"] = (
        1 - tx["meta_has_destination_tag"]
    )

    tx["meta_has_sendmax"] = fields.apply(
        lambda d: int("SendMax" in d)
    )

    tx["meta_has_paths"] = fields.apply(
        lambda d: int(bool(d.get("Paths")))
    )

    transfers["asset_id"] = (
        transfers["TOKEN_ADDRESS"].fillna("__NATIVE__").astype(str)
        + ":"
        + transfers["SYMBOL"].fillna("XRP").astype(str)
    )

    transfers["raw_value"] = pd.to_numeric(
        transfers["VALUE"], errors="coerce"
    ).fillna(0).abs()

    grouped = transfers.groupby("TRANSACTION_HASH")

    legs = grouped.agg(
        pattern_leg_count=("TRANSFER_INDEX", "size"),
        pattern_distinct_assets=("asset_id", "nunique"),
        pattern_distinct_leg_senders=("FROM_ADDRESS", "nunique"),
        pattern_distinct_leg_receivers=("TO_ADDRESS", "nunique"),
        pattern_distinct_transfer_types=("TRANSFER_TYPE", "nunique"),
    ).reset_index()

    legs["pattern_cross_currency"] = (
        legs["pattern_distinct_assets"] > 1
    ).astype(int)

    legs["pattern_bridge_conversion"] = (
        legs["pattern_distinct_assets"] >= 3
    ).astype(int)

    columns = [
        "TRANSACTION_HASH",
        "meta_has_destination_tag",
        "meta_missing_destination_tag",
        "meta_has_sendmax",
        "meta_has_paths",
    ]

    result = tx[columns].merge(
        legs,
        on="TRANSACTION_HASH",
        how="left",
        validate="one_to_one",
    )

    return result


def conversion_price_features(tx, transfers, rates):
    """
    Estimate executed conversion cost relative to reference USD.

    Uses the first executed transfer leg as the source debit,
    as specified by the dataset guide.

    Reference prices are synthetic valuations, not executable
    market quotes. Deviations may reflect ordinary spreads.
    """

    rates = rates.copy()

    rates["TIMESTAMP"] = pd.to_numeric(
        rates["TIMESTAMP"], errors="coerce"
    )

    rates["PRICE_MICRO_USD"] = pd.to_numeric(
        rates["PRICE_MICRO_USD"], errors="coerce"
    )

    price_lookup = {
        (int(r.TIMESTAMP), str(r.ASSET)):
        float(r.PRICE_MICRO_USD) / 1_000_000
        for r in rates.itertuples(index=False)
        if pd.notna(r.TIMESTAMP)
        and pd.notna(r.PRICE_MICRO_USD)
    }

    tx_times = tx.set_index(
        "TRANSACTION_HASH"
    )["TIMESTAMP"].to_dict()

    transfers = transfers.copy()

    transfers["TRANSFER_INDEX"] = pd.to_numeric(
        transfers["TRANSFER_INDEX"],
        errors="coerce",
    )

    transfers = transfers.sort_values(
        ["TRANSACTION_HASH", "TRANSFER_INDEX"]
    )

    rows = []

    for tx_hash, group in transfers.groupby(
        "TRANSACTION_HASH", sort=False
    ):
        first = group.iloc[0]
        last = group.iloc[-1]

        timestamp = safe_number(tx_times.get(tx_hash), np.nan)

        record = {
            "TRANSACTION_HASH": tx_hash,
            "price_reference_available": 0,
            "price_effective_spread": 0.0,
            "price_abs_effective_spread": 0.0,
            "price_source_usd": 0.0,
            "price_destination_usd": 0.0,
        }

        if not np.isfinite(timestamp):
            rows.append(record)
            continue

        hour = int(timestamp // 3600 * 3600)

        def asset_and_value(leg):
            symbol = str(leg["SYMBOL"])
            token = leg["TOKEN_ADDRESS"]

            if symbol == "XRP":
                asset = "XRP"
            else:
                asset = f"{token}:{symbol}"

            return asset, safe_number(leg["VALUE"])

        source_asset, source_amount = asset_and_value(first)
        dest_asset, dest_amount = asset_and_value(last)

        source_price = price_lookup.get(
            (hour, source_asset)
        )

        dest_price = price_lookup.get(
            (hour, dest_asset)
        )

        if source_price is not None and dest_price is not None:
            source_usd = source_amount * source_price
            dest_usd = dest_amount * dest_price

            if dest_usd > EPS:
                spread = source_usd / dest_usd - 1.0

                record.update({
                    "price_reference_available": 1,
                    "price_effective_spread": spread,
                    "price_abs_effective_spread": abs(spread),
                    "price_source_usd": source_usd,
                    "price_destination_usd": dest_usd,
                })

        rows.append(record)

    return pd.DataFrame(rows)


def circular_flow_features(p):
    """
    Detect A -> B -> A and A -> B -> C -> A patterns.

    Full-window structural patterns, not proof of wash trading.
    """

    edges = defaultdict(set)
    pair_count = defaultdict(int)

    for r in p.itertuples(index=False):
        s = r.FROM_ADDRESS
        d = r.TO_ADDRESS

        edges[s].add(d)
        pair_count[(s, d)] += 1

    rows = []

    for r in p.itertuples(index=False):
        s = r.FROM_ADDRESS
        d = r.TO_ADDRESS

        reciprocal = d in edges and s in edges[d]

        # Count possible triangles containing this edge.
        triangle_count = sum(
            1
            for c in edges.get(d, ())
            if s in edges.get(c, ())
            and c not in {s, d}
        )

        rows.append({
            "TRANSACTION_HASH": r.TRANSACTION_HASH,
            "cycle_has_return_edge": int(reciprocal),
            "cycle_reverse_pair_count": pair_count[(d, s)],
            "cycle_triangle_count": triangle_count,
            "cycle_pair_repetition": pair_count[(s, d)],
        })

    return pd.DataFrame(rows)


def build_pattern_features(folder):
    tx, transfers, rates, p = load_split(folder)

    print(
        f"Transactions: {len(tx):,} | "
        f"Payments: {len(p):,} | "
        f"Transfer legs: {len(transfers):,}",
        flush=True,
    )

    groups = [
        ("networks", account_network_features(p)),
        ("money trails", money_trail_features(p)),
        ("thresholds", threshold_features(p)),
        (
            "transaction metadata",
            transaction_metadata_features(tx, transfers),
        ),
        (
            "conversion prices",
            conversion_price_features(tx, transfers, rates),
        ),
        ("circular flows", circular_flow_features(p)),
    ]

    result = tx[["TRANSACTION_HASH"]].copy()

    if result["TRANSACTION_HASH"].duplicated().any():
        raise ValueError("Duplicate transaction hashes")

    for name, group in groups:
        if group["TRANSACTION_HASH"].duplicated().any():
            raise ValueError(
                f"Duplicate hashes in feature group: {name}"
            )

        print(
            f"Adding {name}: {len(group.columns) - 1} features",
            flush=True,
        )

        result = result.merge(
            group,
            on="TRANSACTION_HASH",
            how="left",
            validate="one_to_one",
        )

    numeric = result.columns.drop("TRANSACTION_HASH")

    result[numeric] = result[numeric].replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    return result


def process_split(split, folder):
    print(f"\n{'=' * 60}", flush=True)
    print(f"BUILDING {split.upper()} FEATURES", flush=True)
    print(f"{'=' * 60}", flush=True)

    new = build_pattern_features(folder)

    base_path = ROOT / f"{split}_features_advanced.csv"

    if base_path.exists():
        base = pd.read_csv(base_path)

        # Do not silently overwrite existing feature names.
        overlap = (
            set(base.columns)
            & set(new.columns)
        ) - {"TRANSACTION_HASH"}

        if overlap:
            raise ValueError(
                f"Feature name collisions: {sorted(overlap)}"
            )

        result = base.merge(
            new,
            on="TRANSACTION_HASH",
            how="left",
            validate="one_to_one",
        )
    else:
        print(
            f"No baseline CSV for {split}; "
            "saving pattern features alone."
        )
        result = new

    numeric = result.columns.drop("TRANSACTION_HASH")

    result[numeric] = result[numeric].replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    output = ROOT / f"{split}_features_patterns.csv"
    result.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Shape: {result.shape}", flush=True)

    return result


def main():
    outputs = {}

    for split, candidates in [
        ("train", ["train"]),
        ("validation", ["validation"]),
        ("test", ["test-data", "test"]),
    ]:
        folder = None

        for candidate in candidates:
            path = ROOT / candidate

            if (
                (path / "transactions.parquet").exists()
                and (path / "transfers.parquet").exists()
                and (path / "market_rates.csv").exists()
            ):
                folder = path
                break

        if folder is None:
            if split == "test":
                print(
                    "\nTest data not extracted yet; "
                    "skipping test."
                )
                continue

            raise FileNotFoundError(
                f"Could not find {split} dataset"
            )

        outputs[split] = process_split(split, folder)

    print("\n" + "=" * 60)
    print("ALL PATTERN FEATURES GENERATED")
    print("=" * 60)

    for split, df in outputs.items():
        print(f"{split}: {df.shape}")


if __name__ == "__main__":
    main()
