import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv import time_folds  # noqa: E402


def test_folds_train_only_on_earlier_time():
    t = np.arange(0, 10_000, 10)
    for train, val in time_folds(t, n_bins=5, embargo_s=0):
        assert len(val) > 0 and len(train) > 0
        assert t[train].max() < t[val].min()


def test_embargo_removes_rows_near_validation_start():
    t = np.arange(0, 10_000, 10)
    embargo = 500
    for train, val in time_folds(t, n_bins=5, embargo_s=embargo):
        assert (t[train] < t[val].min() - embargo).all()


def test_no_row_is_in_train_and_val():
    t = np.arange(0, 10_000, 10)
    for train, val in time_folds(t, n_bins=5, embargo_s=300):
        assert set(train).isdisjoint(set(val))


@pytest.mark.skipif("SUBMISSION" not in os.environ, reason="set SUBMISSION to a predictions CSV")
def test_submission_format():
    sub = pd.read_csv(os.environ["SUBMISSION"])
    assert list(sub.columns) == ["TRANSACTION_HASH", "score"]
    assert len(sub) == 40101
    assert sub["TRANSACTION_HASH"].is_unique
    assert sub["score"].between(0, 1).all()
    assert sub["score"].notna().all()
