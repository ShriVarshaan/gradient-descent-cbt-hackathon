# Gradient Descent — Decode & Detect

An ensemble of LightGBM, XGBoost and CatBoost on 254 per-transaction features (81 pipeline features + 173 burst features). Every test transaction gets a suspicion score from 0 to 1.
(The repo also holds an earlier LightGBM + GraphSAGE GNN combination, `src/combo.py`, described below.)

## Reproduce the predictions

Python 3.14.2. Unzip the training package and the test package anywhere, then:

```sh
pip install -r requirements.txt
python src/ensemble.py <training-data>/train <test-data> predictions.csv            # submitted predictions
python src/ensemble.py <training-data>/train <test-data> predictions.csv --no-cv    # skip the time-CV report
python src/combo.py <training-data>/train <test-data> predictions.csv --method stack           # stacking
python src/combo.py <training-data>/train <test-data> predictions.csv --method blend --no-cv   # rank blend
```

`<training-data>/train` holds `transactions.parquet`, `transfers.parquet`, `blocks.parquet`,
`market_rates.csv` and `labels.csv`. `<test-data>` holds the same files without labels.
The output is `TRANSACTION_HASH,score` with 40,101 rows.

Hardware and time: Intel laptop CPU, no GPU. `src/ensemble.py` takes about 5 minutes with its CV report
(about 1–2 minutes with `--no-cv`); seeds are fixed (`random_state=0` / `random_seed=0`). The blend command takes about 5–9 minutes (3 GNN fits).
The stacking command with its time-CV report takes about 35 minutes (57 GNN fits on 4 worker processes);
add `--no-cv` to skip the report (18 fits). Seeds are fixed (LightGBM `random_state=0`, GNN seeds 0, 1 and 2).

## The gradient-boosting ensemble (`src/ensemble.py`)

LightGBM, XGBoost and CatBoost are each trained on all of `train/` and the submitted score is the mean of their
three predicted probabilities. Features:

- the 81 pipeline features from `src/features.py`;
- 173 burst features from `src/burst_features.py`, which describe each payment's time-local neighbourhood:
  connected clusters of payments sharing an account within 30 s to 30 min (size, accounts, duration, rate,
  amount spread), windowed sender/receiver counts relative to overall ledger activity, gaps to neighbouring
  payments, forward/backward chain length, 2- and 3-hop cycles, forwarded share of value, account lifetimes,
  week-long relay behaviour (how often an account passes on what it receives, repeated 2-hop paths), and slower
  motifs through busy accounts (first-time counterparties, bipartite blocks, sender -> intermediaries -> common sink).

Hyperparameters were picked by mean average precision over the expanding-window time folds in `src/cv.py`
(one-hour embargo). Time-CV mean AP:

| Model | 81 features | 230 features (no motifs) | 244 features |
|---|---|---|---|
| LightGBM | 0.350 | 0.605 | 0.626 |
| XGBoost | 0.324 | 0.578 | |
| CatBoost | 0.337 | 0.577 | |
| Mean of the three | 0.351 | 0.598 | |

`--models lgb` scores with LightGBM alone instead of the three-model mean; `--seeds N` averages each model over N seeds.
The latest submission is LightGBM alone over 5 seeds on 254 features (time-CV 0.628 with one seed):

```sh
python src/ensemble.py <training-data>/train <test-data> predictions.csv --no-cv --models lgb --seeds 5
```

## How the LightGBM and GNN models are combined (`src/combo.py`)

`src/pipeline.py` is the LightGBM model on 81 per-transaction features. `gnn/` is a heterogeneous GraphSAGE
over account and transaction nodes, with direct transaction-to-transaction flow and burst edges
(see `gnn/README.md`). `src/combo.py` builds both from the raw data and combines them:

- **`--method stack`:** the GNN's score is one more LightGBM feature (`gnn_score`). Training rows get
  out-of-fold scores: the week is cut into 5 time blocks, and each block is scored by GNNs trained on the
  other blocks, minus a one-hour embargo. LightGBM therefore never sees a GNN score fitted on that row's own label.
  Test rows are scored by GNNs trained on all of `train/`.
- **`--method blend`:** the average of the two models' rank percentiles (`--blend-weight`, default 0.5 for LightGBM).

Each GNN score is the mean of 3 seeds, each trained for 70 epochs with `pos_weight=5`.
The time-CV report evaluates both methods honestly: a GNN used for a fold is trained on that fold's training rows only.

| File | Role |
|---|---|
| `src/ensemble.py` | LightGBM + XGBoost + CatBoost ensemble, time-CV report, writes the submitted predictions |
| `src/burst_features.py` | Time-local cluster, chain, cycle and relay features |
| `src/combo.py` | GNN + LightGBM combination, time-CV report, writes the predictions |
| `src/pipeline.py` | LightGBM alone: `python src/pipeline.py <training-data>/train <test-data> submission.csv` |
| `src/prepare.py`, `src/features.py`, `src/graph_features.py` | Table loading, USD valuation, the 81 LightGBM features |
| `src/cv.py` | Expanding-window time folds with a one-hour embargo |
| `src/ablation.py` | LightGBM feature-group ablation |
| `gnn/build_graph.py`, `gnn/train.py` | Graph construction and the GraphSAGE model |
| `notebooks/eda_labelled.ipynb` | Exploratory analysis behind the GNN design |
| `payments.py` | Organiser-provided payment loader (unchanged) |

## Tests

```sh
pytest
SUBMISSION=predictions.csv DATA_DIR=<training-data>/train pytest   # also checks real data and the CSV
```

## What we did not write ourselves

- **Libraries:** pandas, NumPy, PyArrow, SciPy, LightGBM, XGBoost, CatBoost, scikit-learn, PyTorch, PyTorch Geometric, pytest
  (versions in `requirements.txt`). Matplotlib is used for plots in the EDA notebook only.
- **Organiser material:** `payments.py` is copied unchanged from the Decode & Detect training package. The dataset and its README are the organisers'.
- **Outside services:** the EDA notebook was run on Google Colab. Nothing else.
- **AI assistance:** parts of the code, tests and documentation were written with Claude Code (Anthropic).
  The team directed the work and reviewed the results.
- No pre-trained models, external data or test labels are used.

## License

MIT, see `LICENSE`.
