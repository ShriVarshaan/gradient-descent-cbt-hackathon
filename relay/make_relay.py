"""Write relay-wave features for one split to parquet. Usage: python make_relay.py <split_dir> <out.parquet>"""
import sys

import pandas as pd

from payments import project
from prepare import load_tables
from relay_wave_features import RELAY_FEATURES, build_relay_wave_features

t = load_tables(sys.argv[1])
rates = {(int(r["TIMESTAMP"]), r["ASSET"]): int(r["PRICE_MICRO_USD"]) for r in t["market_rates"].to_dict("records")}
f = build_relay_wave_features(pd.DataFrame(project(t["transactions"].to_dict("records"), rates)))
assert f.TRANSACTION_HASH.is_unique and f[RELAY_FEATURES].notna().all().all()
f.to_parquet(sys.argv[2], index=False)
print(f.shape)
