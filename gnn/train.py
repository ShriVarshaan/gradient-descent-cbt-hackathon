"""Full-batch GraphSAGE on tx-node classification, scored with validation AP.

    python gnn/train.py graphs/train.pt                                   # one run, default settings
    python gnn/train.py graphs/train.pt --pos-weight 1 31 --seeds 0 1 2   # class-weight comparison
    python gnn/train.py graphs/train.pt --predict graphs/test.pt --out predictions.csv

Every (pos_weight, seed) run trains on train_mask and keeps the epoch with the best
validation AP. With --predict, the best pos_weight's runs are averaged into one score per tx.
Build the graphs first with gnn/build_graph.py.
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import Linear, SAGEConv, to_hetero


class SAGE(torch.nn.Module):
    """SAGE layers plus a skip from each node's own raw features, so strong tx features survive."""

    def __init__(self, hidden, layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList(SAGEConv((-1, -1), hidden) for _ in range(layers))
        self.skip = Linear(-1, hidden)
        self.out = Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = F.dropout(conv(h, edge_index).relu(), self.dropout, self.training)
        return self.out(h + self.skip(x))


def load(path, device, no_graph=False):
    data = torch.load(path, weights_only=False)["data"].to(device)
    if no_graph:                                # ablation: same model, every node sees only itself
        for store in data.edge_stores:
            store.edge_index = store.edge_index[:, :0]
    return data


def scores(model, data):
    model.eval()
    with torch.no_grad():
        return model(data.x_dict, data.edge_index_dict)["tx"].squeeze(-1)


def train_one(data, args, pos_weight, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = to_hetero(SAGE(args.hidden, args.layers, args.dropout), data.metadata(), aggr="sum").to(data["tx"].x.device)
    with torch.no_grad():                       # initialise the lazy layers
        model(data.x_dict, data.edge_index_dict)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y, train, val = data["tx"].y, data["tx"].train_mask, data["tx"].val_mask
    pw = torch.tensor(pos_weight, device=y.device)
    best_ap, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)["tx"].squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits[train], y[train], pos_weight=pw)
        loss.backward()
        opt.step()
        p = scores(model, data)
        ap = average_precision_score(y[val].cpu(), p[val].cpu())
        history.append((epoch, loss.item(), ap))
        if ap > best_ap:
            best_ap, best_epoch, best_state = ap, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    train_ap = average_precision_score(y[train].cpu(), scores(model, data)[train].cpu())
    return model, dict(pos_weight=pos_weight, seed=seed, best_epoch=best_epoch, epochs_run=epoch,
                       train_ap=train_ap, val_ap=best_ap), history


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train_graph")
    p.add_argument("--pos-weight", type=float, nargs="+", default=[1.0],
                   help="BCE pos_weight values to compare; ~31 balances the classes")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=50, help="stop after this many epochs without a better val AP")
    p.add_argument("--predict", help="graph .pt to score, e.g. graphs/test.pt")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--history", help="optional CSV of per-epoch loss and val AP for every run")
    p.add_argument("--no-graph", action="store_true", help="ablation: drop all edges (an MLP on tx features)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(args.train_graph, device, args.no_graph)
    y = data["tx"].y
    print(f"device={device}  train pos={y[data['tx'].train_mask].sum():.0f}  val pos={y[data['tx'].val_mask].sum():.0f}"
          f"  val AP floor={y[data['tx'].val_mask].mean():.4f}")

    rows, models, histories = [], {}, []
    for pw in args.pos_weight:
        for seed in args.seeds:
            start = time.time()
            model, row, hist = train_one(data, args, pw, seed)
            row["seconds"] = round(time.time() - start, 1)
            rows.append(row)
            models.setdefault(pw, []).append(model)
            histories += [dict(pos_weight=pw, seed=seed, epoch=e, loss=l, val_ap=a) for e, l, a in hist]
            print(f"pos_weight={pw:<5g} seed={seed}  best epoch {row['best_epoch']:>3}  "
                  f"train AP {row['train_ap']:.4f}  val AP {row['val_ap']:.4f}  ({row['seconds']}s)")

    results = pd.DataFrame(rows)
    summary = results.groupby("pos_weight").val_ap.agg(["mean", "std", "min", "max", "count"])
    print("\nval AP by pos_weight\n" + summary.round(4).to_string())
    if args.history:
        pd.DataFrame(histories).to_csv(args.history, index=False)

    if args.predict:
        best_pw = summary["mean"].idxmax()
        test = load(args.predict, device, args.no_graph)
        p = torch.stack([torch.sigmoid(scores(m, test)) for m in models[best_pw]]).mean(0).cpu().numpy()
        pd.DataFrame({"TRANSACTION_HASH": test["tx"].hash, "score": p}).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}: {len(p)} rows, averaged {len(models[best_pw])} runs with pos_weight={best_pw:g}")


if __name__ == "__main__":
    main()
