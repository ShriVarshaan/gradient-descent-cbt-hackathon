"""Average the rank of several submission files. Usage: seedavg.py <out.csv> <in1.csv> <in2.csv> ..."""
import sys
import pandas as pd
from scipy.stats import rankdata
out, ins = sys.argv[1], sys.argv[2:]
base = pd.read_csv(ins[0])[["TRANSACTION_HASH"]]
ranks = []
for p in ins:
    d = base.merge(pd.read_csv(p), on="TRANSACTION_HASH", how="left", validate="one_to_one")
    assert d.score.notna().all(), p
    ranks.append(rankdata(d.score) / len(d))
base["score"] = sum(ranks) / len(ranks)
assert len(base) == base.TRANSACTION_HASH.nunique() and base.score.between(0, 1).all()
base.to_csv(out, index=False)
print(f"{out}: {len(base)} rows from {len(ins)} runs")
