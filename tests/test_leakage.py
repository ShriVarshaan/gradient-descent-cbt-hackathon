import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FEATURES, build_features  # noqa: E402
from prepare import load_tables  # noqa: E402

# Annotation and answer columns from labels.csv or the transactions. Never features.
FORBIDDEN = {"is_suspicious", "label", "family", "difficulty", "scenario_id", "cohort_id"}


def test_no_feature_name_is_a_label_or_annotation():
    assert FORBIDDEN.isdisjoint(FEATURES)


@pytest.mark.skipif("DATA_DIR" not in os.environ, reason="set DATA_DIR to the train/ folder")
def test_features_build_without_labels():
    # The test download has no labels.csv, so features must not need one.
    tables = load_tables(os.environ["DATA_DIR"])
    tables.pop("labels", None)
    feats = build_features(tables)
    assert FORBIDDEN.isdisjoint(feats.columns)
    assert set(FEATURES) <= set(feats.columns)


@pytest.mark.skipif("DATA_DIR" not in os.environ, reason="set DATA_DIR to the train/ folder")
def test_features_do_not_depend_on_row_order():
    # A feature that reads row position instead of the data would leak or shift on the test file.
    tables = load_tables(os.environ["DATA_DIR"])
    base = build_features(tables).set_index("TRANSACTION_HASH").sort_index()
    shuffled = dict(tables)
    shuffled["transactions"] = tables["transactions"].sample(frac=1, random_state=1)
    shuffled["transfers"] = tables["transfers"].sample(frac=1, random_state=2)
    again = build_features(shuffled).set_index("TRANSACTION_HASH").sort_index()
    pd.testing.assert_frame_equal(base[FEATURES], again[FEATURES], check_dtype=False, rtol=1e-9)
