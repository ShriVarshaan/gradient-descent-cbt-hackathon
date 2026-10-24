
"""
Token-aware temporal relay features for XRPL AML detection.

Run:
    python src/relay_matching_features.py

Outputs:
    train_features_relay.csv
    validation_features_relay.csv

Uses strictly earlier incoming transfers.
Matches only within the same TOKEN_ADDRESS.
Does not modify existing advanced feature files.
"""

from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

WINDOWS = (5, 15, 60, 1440)
MAX_HISTORY = 1440

REQUIRED_COLUMNS = [
    "TRANSACTION_HASH",
    "TIMESTAMP",
    "FROM_ADDRESS",
    "TO_ADDRESS",
    "TOKEN_ADDRESS",
    "VALUE",
]



def load_transfers(split):
    path = ROOT / split / "transfers.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{split} transfers missing columns: {sorted(missing)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    df = df.rename(columns={
        "FROM_ADDRESS": "sender",
        "TO_ADDRESS": "receiver",
        "TOKEN_ADDRESS": "token",
        "VALUE": "amount",
        "TIMESTAMP": "timestamp",
    })

    # TIMESTAMP may be numeric, while DATETIME is a date string.
    # Numeric timestamps must be interpreted with the correct unit.
    numeric = pd.to_numeric(df["timestamp"], errors="coerce")

    if numeric.notna().mean() > 0.95:
        median = numeric.dropna().abs().median()

        if median > 1e17:
            unit = "ns"
        elif median > 1e14:
            unit = "us"
        elif median > 1e11:
            unit = "ms"
        else:
            unit = "s"

        df["timestamp"] = pd.to_datetime(
            numeric,
            unit=unit,
            utc=True,
            errors="coerce",
        )
    else:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    ).abs()

    # Keep native-token transfers even when TOKEN_ADDRESS is null.
    df["token"] = df["token"].fillna("__NATIVE__").astype(str)

    df = df.dropna(
        subset=[
            "TRANSACTION_HASH",
            "sender",
            "receiver",
            "timestamp",
            "amount",
        ]
    )

    df = df[df["amount"] > 0].copy()

    df["sender"] = df["sender"].astype(str)
    df["receiver"] = df["receiver"].astype(str)

    df = df.sort_values(
        ["timestamp", "TRANSACTION_HASH"],
        kind="mergesort",
    ).reset_index(drop=True)

    print(f"{split}: {len(df):,} usable transfers")

    return df



def calculate_features(transfers):
    incoming = defaultdict(deque)
    results = []

    for timestamp, batch in transfers.groupby(
        "timestamp",
        sort=True,
    ):
        current_minute = timestamp.value / 60_000_000_000

        for row in batch.itertuples(index=False):
            amount = float(row.amount)
            key = (row.sender, row.token)
            history = incoming[key]

            while (
                history
                and current_minute - history[0][0] > MAX_HISTORY
            ):
                history.popleft()

            record = {
                "TRANSACTION_HASH": row.TRANSACTION_HASH,
                "relay_previous_in_count": len(history),
                "relay_previous_in_usd": sum(
                    item[1] for item in history
                ),
            }

            if history:
                closest = min(
                    history,
                    key=lambda item: abs(item[1] - amount),
                )

                record["relay_best_amount_ratio"] = (
                    closest[1] / amount
                )

                record["relay_best_relative_error"] = (
                    abs(closest[1] - amount) / amount
                )

                record["relay_best_delay_minutes"] = (
                    current_minute - closest[0]
                )

            else:
                record["relay_best_amount_ratio"] = 0.0
                record["relay_best_relative_error"] = 1.0
                record["relay_best_delay_minutes"] = -1.0

            for window in WINDOWS:
                recent = [
                    item
                    for item in history
                    if current_minute - item[0] <= window
                ]

                amounts = np.array(
                    [item[1] for item in recent],
                    dtype=np.float64,
                )

                total = float(amounts.sum())

                record[f"relay_in_count_{window}m"] = len(recent)
                record[f"relay_in_usd_{window}m"] = total

                record[f"relay_out_to_in_ratio_{window}m"] = (
                    amount / total if total > 0 else 0.0
                )

                if len(amounts):
                    errors = np.abs(amounts - amount) / amount

                    record[f"relay_min_error_{window}m"] = (
                        float(errors.min())
                    )

                    record[f"relay_matches_5pct_{window}m"] = (
                        int((errors <= 0.05).sum())
                    )

                    record[f"relay_matches_10pct_{window}m"] = (
                        int((errors <= 0.10).sum())
                    )

                    record[f"relay_coverage_{window}m"] = (
                        min(total / amount, 1.0)
                    )

                else:
                    record[f"relay_min_error_{window}m"] = 1.0
                    record[f"relay_matches_5pct_{window}m"] = 0
                    record[f"relay_matches_10pct_{window}m"] = 0
                    record[f"relay_coverage_{window}m"] = 0.0

            results.append(record)

        # Add incoming payments after evaluating all transfers
        # at this timestamp to avoid same-time matching.
        for row in batch.itertuples(index=False):
            key = (row.receiver, row.token)
            incoming[key].append(
                (current_minute, float(row.amount))
            )

    features = pd.DataFrame(results)

    if features.empty:
        raise ValueError("No relay features generated")

    numeric_columns = [
        col
        for col in features.columns
        if col != "TRANSACTION_HASH"
    ]

    features = (
        features
        .groupby("TRANSACTION_HASH", as_index=False)[numeric_columns]
        .max()
    )

    return features.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)


def process_split(split):
    print(f"\n{'=' * 60}")
    print(f"GENERATING {split.upper()} RELAY FEATURES")
    print(f"{'=' * 60}")

    base_path = ROOT / f"{split}_features_advanced.csv"

    base = pd.read_csv(base_path)
    transfers = load_transfers(split)
    relay = calculate_features(transfers)

    # The old column names are retained for consistency,
    # but amounts are now same-token units, NOT USD.
    rename = {
        "relay_previous_in_usd": "relay_previous_in_token_units",
    }

    for window in WINDOWS:
        rename[f"relay_in_usd_{window}m"] = (
            f"relay_in_token_units_{window}m"
        )

    relay = relay.rename(columns=rename)

    combined = base.merge(
        relay,
        on="TRANSACTION_HASH",
        how="left",
        validate="one_to_one",
    )

    new_cols = [
        c for c in relay.columns
        if c != "TRANSACTION_HASH"
    ]

    combined[new_cols] = combined[new_cols].fillna(0)

    output = ROOT / f"{split}_features_relay.csv"
    combined.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Rows: {len(combined):,}")
    print(f"Features: {len(combined.columns) - 1}")

    return combined


def main():
    train = process_split("train")
    validation = process_split("validation")

    if list(train.columns) != list(validation.columns):
        raise ValueError(
            "Training and validation feature columns differ"
        )

    print("\nSUCCESS")
    print(f"Train: {train.shape}")
    print(f"Validation: {validation.shape}")


if __name__ == "__main__":
    main()
