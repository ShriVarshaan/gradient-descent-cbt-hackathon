"""Run the trained GNN + Tuned XGBoost / LightGBM models on an unlabelled test directory
(e.g. `test/`) and output submission CSV files (`TRANSACTION_HASH,score`).

Usage:
    python gnn/predict_test.py --test-dir test --out-csv submission.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from build_graph import build as build_graph_from_dir
from model import HeteroTxGNN, augment_hetero_graph
from train_gnn_xgb import compute_graph_and_gnn_neighborhood_features, infer_gnn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-pt", type=Path, default=Path("graphs/train.pt"), help="Cached train.pt graph")
    parser.add_argument("--test-dir", type=Path, default=Path("test"), help="Directory containing test parquet files")
    parser.add_argument("--test-pt", type=Path, default=Path("graphs/test_set.pt"), help="Path to cache built test.pt graph")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory with trained models and outputs")
    parser.add_argument("--out-csv", type=Path, default=Path("submission.csv"), help="Primary output submission CSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Loading train graph from {args.train_pt}...")
    train_bundle = torch.load(args.train_pt, weights_only=False)

    if args.test_pt.exists():
        print(f"[Predict] Loading cached test graph from {args.test_pt}...")
        test_bundle = torch.load(args.test_pt, weights_only=False)
    else:
        print(f"[Predict] Building test graph from {args.test_dir}...")
        data_t, _, names_t = build_graph_from_dir(args.test_dir, stats=train_bundle["stats"])
        test_bundle = {"data": data_t, "stats": train_bundle["stats"], "names": names_t}
        args.test_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(test_bundle, args.test_pt)

    # Augment train & test graphs with identical normalization statistics
    aug_train_data, aug_stats = augment_hetero_graph(train_bundle["data"], aug_stats=None)
    aug_test_data, _ = augment_hetero_graph(test_bundle["data"], aug_stats=aug_stats)
    tx_feat_names = [f"tx_{c}" for c in (test_bundle["names"]["tx"] + aug_stats["names"])]

    # Load trained GNN checkpoint and run inference on test graph
    gnn_ckpt = torch.load(args.out_dir / "gnn_best_model.pt", map_location=device, weights_only=False)
    cfg = gnn_ckpt["config"]
    gnn_model = HeteroTxGNN(
        tx_in_dim=aug_test_data["tx"].x.shape[1],
        acc_in_dim=aug_test_data["account"].x.shape[1],
        bipartite_edge_dim=aug_test_data["account", "sends", "tx"].edge_attr.shape[1],
        temporal_edge_dim=aug_test_data["tx", "temporal", "tx"].edge_attr.shape[1],
        hidden_dim=cfg["hidden_dim"],
        emb_dim=cfg["emb_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    gnn_model.load_state_dict(gnn_ckpt["state_dict"])

    print(f"[Predict] Running GNN inference on {aug_test_data['tx'].x.shape[0]:,} test transactions...")
    t_logits, t_probs, t_emb = infer_gnn(gnn_model, aug_test_data, device)

    df_base = pd.DataFrame(aug_test_data["tx"].x.cpu().numpy(), columns=tx_feat_names)
    df_base.insert(0, "TRANSACTION_HASH", aug_test_data["tx"].hash)
    df_base["TIMESTAMP"] = aug_test_data["tx"].time.cpu().numpy()

    emb_dim = cfg["emb_dim"]
    df_gnn = pd.DataFrame(
        {
            "gnn_logit": t_logits,
            "gnn_prob": t_probs,
            **{f"gnn_emb_{i}": t_emb[:, i] for i in range(emb_dim)},
        },
        dtype=np.float32,
    )
    df_neigh = compute_graph_and_gnn_neighborhood_features(aug_test_data, df_base, t_probs, t_logits)
    test_df = pd.concat([df_base, df_gnn, df_neigh], axis=1)

    # Save test GNN features
    gnn_only_cols = [c for c in test_df.columns if c.startswith("gnn_")]
    test_gnn_feat_path = args.out_dir / "test_set_gnn_features.parquet"
    test_df[["TRANSACTION_HASH"] + gnn_only_cols].to_parquet(test_gnn_feat_path, index=False)
    print(f"[Export] Saved test GNN features ({len(gnn_only_cols)} cols) to {test_gnn_feat_path}")

    # Load full training feature table to align columns & train LightGBM companion
    train_df = pd.read_parquet(args.out_dir / "train_full_with_gnn.parquet")
    ignore_cols = {"TRANSACTION_HASH", "TIMESTAMP", "is_suspicious"}
    all_feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    # 1. Score with Tuned XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.load_model(str(args.out_dir / "xgb_gnn_model.json"))
    xgb_scores = xgb_model.predict_proba(test_df[all_feature_cols])[:, 1]

    # 2. Score with Tuned LightGBM model
    lgb_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02,
        num_leaves=47,
        max_depth=7,
        scale_pos_weight=3.5,
        subsample_for_bin=50000,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_samples=25,
        reg_alpha=1.5,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(train_df[all_feature_cols], train_df["is_suspicious"].values)
    lgb_scores = lgb_model.predict_proba(test_df[all_feature_cols])[:, 1]

    # 3. Ensemble (60% Tuned XGBoost + 40% Tuned LightGBM)
    ens_scores = 0.60 * xgb_scores + 0.40 * lgb_scores

    # Verify exact row alignment with raw test/transactions.parquet
    raw_tx_hashes = pd.read_parquet(args.test_dir / "transactions.parquet", columns=["TRANSACTION_HASH"])["TRANSACTION_HASH"]
    assert (raw_tx_hashes.values == test_df["TRANSACTION_HASH"].values).all(), "Hash order mismatch!"

    sub_xgb = pd.DataFrame({"TRANSACTION_HASH": test_df["TRANSACTION_HASH"], "score": xgb_scores})
    sub_lgb = pd.DataFrame({"TRANSACTION_HASH": test_df["TRANSACTION_HASH"], "score": lgb_scores})
    sub_ens = pd.DataFrame({"TRANSACTION_HASH": test_df["TRANSACTION_HASH"], "score": ens_scores})

    # Write primary submission CSV (Tuned GNN + Tuned XGBoost) and all 3 variants in outputs/
    sub_xgb.to_csv(args.out_csv, index=False)
    sub_xgb.to_csv(args.out_dir / "test_submission_gnn_xgb.csv", index=False)
    sub_lgb.to_csv(args.out_dir / "test_submission_gnn_lgbm.csv", index=False)
    sub_ens.to_csv(args.out_dir / "test_submission_gnn_ensemble.csv", index=False)

    print(f"\n[Done] Wrote {len(sub_xgb):,} rows to:")
    print(f"  1. {args.out_csv} (Primary: Tuned GNN + Tuned XGBoost)")
    print(f"  2. {args.out_dir / 'test_submission_gnn_xgb.csv'}")
    print(f"  3. {args.out_dir / 'test_submission_gnn_lgbm.csv'}")
    print(f"  4. {args.out_dir / 'test_submission_gnn_ensemble.csv'}")
    print("\nSummary statistics of primary submission scores:")
    print(sub_xgb["score"].describe().to_string())


if __name__ == "__main__":
    main()
