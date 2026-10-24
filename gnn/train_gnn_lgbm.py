"""Train the HeteroTxGNN, extract Out-Of-Fold (OOF) predictions + structural embeddings,
and feed them into LightGBM and XGBoost alongside tabular + graph features.

Usage:
    python gnn/train_gnn_lgbm.py --train-dir train-data --test-dir validation --out-dir outputs
"""
from train_gnn_xgb import main


if __name__ == "__main__":
    main()
