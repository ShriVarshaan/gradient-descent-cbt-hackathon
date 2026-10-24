# GNN: GraphSAGE on transaction nodes

Scores every transaction for suspicion with a heterogeneous GraphSAGE over accounts and transactions.
The EDA behind the design is in `notebooks/eda_labelled.ipynb`.

## Run it

```sh
pip install -r gnn/requirements.txt        # CUDA build of torch first if you have an NVIDIA GPU

D=train-data/decode-and-detect-training-data
python gnn/build_graph.py $D/train      graphs/train.pt
python gnn/build_graph.py $D/validation graphs/validation.pt --stats graphs/train.pt
python gnn/build_graph.py test-data     graphs/test.pt       --stats graphs/train.pt

python gnn/train.py graphs/train.pt --seeds 0 1 2 --predict graphs/test.pt --out predictions.csv
```

- The train build should print `train_mask: n=32105 pos=993` and `val_mask: n=8026 pos=257`.
- Each graph builds in ~10 s. Each training run takes 1–3 min on a laptop CPU, and the predicting command above does 3 runs.
- `--stats` is required for validation and test, so they are normalised with the train statistics.
- `predictions.csv` averages the runs of the best `pos_weight` and has one row per transaction.

## Graph

| | What | Count (train) |
|---|---|---|
| `tx` nodes | one per payment; 26 features: USD values, legs, cross-currency, slippage, fee, hour, first/last currency, sender bursts (10 min, 1 h), pass-through gaps and amount ratios | 40,131 |
| `account` nodes | 12 features: in/out counts and USD, counterparties, out/in balance, active span, median gap, cross-currency share, max burst, liquidity-provider legs | 700 |
| `account -sends-> tx`, `tx -receives-> account` | one per transfer leg, 7 features (leg index, last leg, USD, currency) | 60,557 each |
| `tx -flows-> tx` | a's receiver sends b within 24 h; 5 nearest per tx | 158,382 |
| `tx -bursts-> tx` | same sender within 10 min | 17,532 |

Every relation also gets a reverse (`rev_*`). Labels (`y`), `train_mask` and `val_mask` are on `tx`; validation is the last 20% of time.

Design choices:

- **Labels are per transaction.** About two-thirds of senders have at least one suspicious payment, so classifying accounts would not work.
- **Currencies are encoded by symbol, not issuer.** Issuer addresses differ between splits.
- **Features use the whole week**, including what happens after each payment. The rules allow this, and the test split is also a full week.
- `family`, `difficulty`, `scenario_id` and `cohort_id` are annotation only and never used.

## Results (validation AP, mean ± std over seeds 0–2)

Random ranking scores 0.032 on the validation split. The best single feature, sender payments within ±5 min, scores 0.143.

| Model | `pos_weight=1` | `pos_weight=5` | `pos_weight=31` |
|---|---|---|---|
| Same model, no edges (MLP on tx features) | 0.163 ± 0.016 | | |
| GraphSAGE, bipartite edges only | 0.169 ± 0.005 | 0.176 ± 0.007 | 0.175 ± 0.012 |
| **GraphSAGE + flow and burst edges** | 0.184 ± 0.014 | **0.212 ± 0.008** | |
| ...same, dropout 0.5 instead of 0.2 | | 0.212 ± 0.015 | |

What we learned:

1. **Bipartite edges alone add almost nothing** (+0.006 over no edges). Averaging an account's whole week of payments washes out the local timing that marks laundering.
2. **Linking payments directly in time is what makes the graph useful** (+0.036). With flow and burst edges, two layers can see "this money was passed on 5 minutes later" and same-sender bursts.
3. **Moderate class weights help** (`pos_weight=5`). Fully balancing (31) is no better and is noisier.
4. **The model overfits.** Train AP reaches 0.5–0.85 while validation stays around 0.21. Raising dropout to 0.5 did not help, so the default stays at 0.2.

Reproduce the table:

```sh
python gnn/train.py graphs/train_bipartite.pt --no-graph --seeds 0 1 2
python gnn/train.py graphs/train_bipartite.pt --pos-weight 1 5 31 --seeds 0 1 2   # built with --no-tx-edges
python gnn/train.py graphs/train.pt           --pos-weight 1 5    --seeds 0 1 2
```

## Next ideas

- Use edge features with a convolution that accepts them (`GINEConv`, `GATConv(edge_dim=...)`), since SAGE ignores `edge_attr`.
- Tune the flow window and k (the README says hard cases span longer histories), and add a third layer to reach 3-hop chains.
- Feed the GNN's score into A's LightGBM as a feature, or blend the two rankings.
