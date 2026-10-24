# Gradient Descent — Decode & Detect

LightGBM plus a GraphSAGE graph neural network. Every test transaction gets a suspicion score from 0 to 1.

## Reproduce the predictions

Python 3.14.2. Unzip the training package and the test package anywhere, then:

```sh
pip install -r requirements.txt
python src/combo.py <training-data>/train <test-data> predictions.csv --method stack           # stacking
python src/combo.py <training-data>/train <test-data> predictions.csv --method blend --no-cv   # rank blend
```

`<training-data>/train` holds `transactions.parquet`, `transfers.parquet`, `blocks.parquet`,
`market_rates.csv` and `labels.csv`. `<test-data>` holds the same files without labels.
The output is `TRANSACTION_HASH,score` with 40,101 rows.

Hardware and time: Intel laptop CPU, no GPU. The blend command takes about 5–9 minutes (3 GNN fits).
The stacking command with its time-CV report takes about 35 minutes (57 GNN fits on 4 worker processes);
add `--no-cv` to skip the report (18 fits). Seeds are fixed (LightGBM `random_state=0`, GNN seeds 0, 1 and 2).

## How the two models are combined

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

- **Libraries:** pandas, NumPy, PyArrow, SciPy, LightGBM, scikit-learn, PyTorch, PyTorch Geometric, pytest
  (versions in `requirements.txt`). Matplotlib is used for plots in the EDA notebook only.
- **Organiser material:** `payments.py` is copied unchanged from the Decode & Detect training package. The dataset and its README are the organisers'.
- **Outside services:** the EDA notebook was run on Google Colab. Nothing else.
- **AI assistance:** parts of the code, tests and documentation were written with Claude Code (Anthropic).
  The team directed the work and reviewed the results.
- No pre-trained models, external data or test labels are used.

## License

MIT, see `LICENSE`.
