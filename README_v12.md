# v12: Aidan's ensemble + 12 relay-wave features

Aidan's pipeline (`origin/aidan-stuff`, 3bec925: graph + GNN + CatBoost/XGBoost/LightGBM, motif
propagation) with 12 extra columns from `relay_wave_features.py` (rare pairs that recur in separate
waves across the week, crew size at the hub, arrivals and forwards around each payment).

## Run (Python 3.12, macOS arm64 tested; about 5 min on 12 cores)

    pip install -r gnn/requirements.txt
    python relay/make_relay.py <train_dir> relay_train.parquet
    python relay/make_relay.py <test_dir>  relay_test.parquet
    cd gnn
    # macOS: make xgboost/lightgbm share torch's OpenMP runtime, or XGBoost hangs or segfaults
    export DYLD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
    python run_torch_first.py train_best_ensemble.py --train-dir <train_dir> --test-dir <test_dir> \
        --out-dir outputs --out-csv submission_v12.csv \
        --extra-train ../relay_train.parquet --extra-test ../relay_test.parquet

## Changes to Aidan's code

- `train_best_ensemble.py`: optional `--extra-train/--extra-test` (feature tables keyed by TRANSACTION_HASH).
  Without them it runs exactly as before.
- `run_torch_first.py`: imports torch before lightgbm (the reverse order segfaults on macOS).

## Offline result (honest)

| Check | Aidan alone | v12 |
|---|---|---|
| 80/20 temporal split, full ensemble AP | 0.6543 | 0.6532 |
| 5-fold time-block OOF AP | 0.6369 | 0.6329 |

Test predictions: Spearman 0.986 with Aidan's 67-scoring submission; 389/401 of the top 1% shared.

## Not written by us

numpy, pandas, pyarrow, scipy, scikit-learn, xgboost, lightgbm, catboost, torch, torch_geometric;
`payments.py` is the organisers' file; `gnn/` is Aidan's code; AI assistance: Claude Code.
