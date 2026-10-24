# Gradient Descent — Decode & Detect

LightGBM detector that gives every test transaction a suspicion score from 0 to 1.

## Reproduce the predictions

Python 3.9.6. Unzip the training package and the test package anywhere, then:

```sh
pip install -r requirements.txt
python src/pipeline.py <training-data>/train <test-data> submission.csv
```

`<training-data>/train` holds `transactions.parquet`, `transfers.parquet`, `blocks.parquet`,
`market_rates.csv` and `labels.csv`. `<test-data>` holds the same files without labels.
The script prints time-based CV average precision per fold, then writes `submission.csv`
(`TRANSACTION_HASH,score`, 40,101 rows).

Hardware and time: about 25 seconds on an Apple M3 Pro laptop (CPU only).
LightGBM uses `random_state=0`, so reruns give the same scores.

## What it does

| File | Role |
|---|---|
| `src/prepare.py` | Loads the tables, parses the payment JSON, values each leg in USD at the hour's reference price |
| `src/features.py` | One row per transaction: basic features, FX features, graph-context features (81 in total) |
| `src/graph_features.py` | Sender bursts, pass-through timing (does the receiver forward the money soon after?), and sender/destination account roles |
| `src/cv.py` | Expanding-window time folds with a one-hour embargo |
| `src/pipeline.py` | Time-CV report, then trains on all of `train/` and scores the test data |
| `src/ablation.py` | Adds feature groups one at a time and prints CV AP: `python src/ablation.py <training-data>/train` |
| `payments.py` | Organiser-provided payment loader (unchanged) |

Feature-group ablation (4 time folds on `train/`, mean AP):

| Features | Count | Mean AP |
|---|---|---|
| basic | 10 | 0.0408 |
| + FX leg | 20 | 0.0401 |
| + FX payment | 23 | 0.0401 |
| + FX account | 31 | 0.0688 |
| + graph transaction | 57 | 0.3166 |
| + graph account (submitted) | 81 | 0.3159 |

## Tests

```sh
pytest
SUBMISSION=submission.csv DATA_DIR=<training-data>/train pytest   # also checks real data and the CSV
```

## What we did not write ourselves

- **Libraries:** pandas, NumPy, PyArrow, LightGBM, scikit-learn, pytest (see `requirements.txt`).
- **Organiser code:** `payments.py` is copied unchanged from the Decode & Detect training package.
- **AI assistance:** parts of the code, tests and this README were written with Claude Code (Anthropic).
- No pre-trained models, external data or outside services are used.

## License

MIT, see `LICENSE`.
