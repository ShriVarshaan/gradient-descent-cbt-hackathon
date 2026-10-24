import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare import RIPPLE_EPOCH_OFFSET, add_usd, join_blocks, parse_payments  # noqa: E402


def _tx(hash_, ledger, ts, extra, delivered):
    return {
        "TRANSACTION_HASH": hash_,
        "LEDGER_INDEX": ledger,
        "LEDGER_HASH": f"L{ledger}",
        "TIMESTAMP": str(ts),
        "EXTRA_TRANSACTION_FIELDS": json.dumps(extra),
        "DELIVERED_AMOUNT": delivered,
    }


def test_join_blocks_keeps_rows_and_matches_timestamp():
    tx = pd.DataFrame([
        _tx("A", "10", 1788349496, {"Destination": "rX"}, '"1000"'),
        _tx("B", "11", 1788745672, {"Destination": "rY"}, '"2000"'),
    ])
    blocks = pd.DataFrame({
        "LEDGER_INDEX": ["10", "11"],
        "LEDGER_HASH": ["L10", "L11"],
        "CLOSE_TIME": ["841664696", "842060872"],
    })
    out = join_blocks(tx, blocks)
    assert len(out) == len(tx)
    assert (out["close_time"] == out["TIMESTAMP"].astype("int64")).all()


def test_join_blocks_fails_on_unmatched_transaction():
    tx = pd.DataFrame([_tx("A", "99", 1788349496, {"Destination": "rX"}, '"1000"')])
    blocks = pd.DataFrame({"LEDGER_INDEX": ["10"], "LEDGER_HASH": ["L10"], "CLOSE_TIME": ["841664696"]})
    with pytest.raises(ValueError):
        join_blocks(tx, blocks)


def test_join_blocks_fails_when_close_time_disagrees():
    tx = pd.DataFrame([_tx("A", "10", 1788349496 + 60, {"Destination": "rX"}, '"1000"')])
    blocks = pd.DataFrame({"LEDGER_INDEX": ["10"], "LEDGER_HASH": ["L10"], "CLOSE_TIME": ["841664696"]})
    with pytest.raises(ValueError):
        join_blocks(tx, blocks)


def test_ripple_offset_is_2000_epoch():
    assert RIPPLE_EPOCH_OFFSET == 946684800


def test_parse_payments_xrp_and_iou():
    tx = pd.DataFrame([
        _tx("X", "1", 1, {"Destination": "rX", "SendMax": "5"}, '"1500000"'),
        _tx("I", "2", 2, {"Destination": "rI", "Paths": []}, json.dumps({"currency": "USD", "issuer": "rIss", "value": "12.5"})),
    ])
    out = parse_payments(tx).set_index("TRANSACTION_HASH")
    assert out.loc["X", "delivered_currency"] == "XRP"
    assert out.loc["X", "delivered_value"] == pytest.approx(1.5)
    assert bool(out.loc["X", "has_send_max"]) and not bool(out.loc["X", "has_paths"])
    assert out.loc["I", "delivered_currency"] == "USD"
    assert out.loc["I", "delivered_issuer"] == "rIss"
    assert out.loc["I", "delivered_value"] == pytest.approx(12.5)


def test_add_usd_uses_latest_rate_at_or_before_leg_time():
    transfers = pd.DataFrame({
        "TIMESTAMP": ["3700"],
        "SYMBOL": ["XRP"],
        "TOKEN_ADDRESS": ["0x1111"],
        "VALUE": [10.0],
    })
    rates = pd.DataFrame({
        "TIMESTAMP": [3600, 7200],
        "ASSET": ["XRP", "XRP"],
        "PRICE_MICRO_USD": [500_000, 900_000],
    })
    out = add_usd(transfers, rates)
    assert out["usd_value"].iloc[0] == pytest.approx(10.0 * 0.5)


@pytest.mark.skipif("DATA_DIR" not in os.environ, reason="set DATA_DIR to the train/ folder")
def test_real_train_data_counts():
    from prepare import load_tables
    t = load_tables(os.environ["DATA_DIR"])
    blocks_joined = join_blocks(t["transactions"], t["blocks"])
    assert len(blocks_joined) == 40131
    assert not t["transactions"]["TRANSACTION_HASH"].duplicated().any()
    assert add_usd(t["transfers"], t["market_rates"])["usd_value"].notna().all()
    assert len(add_usd(t["transfers"], t["market_rates"])) == 60557
