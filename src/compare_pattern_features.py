
"""
Compare original 211 features against expanded 316 features.

Uses CatBoost only. No LightGBM or XGBoost.

Run:
    python src/compare_pattern_features.py

Outputs:
    outputs/pattern_comparison_scores.csv
    outputs/pattern_comparison_oof.csv
    outputs/pattern_comparison_importance.csv
    outputs/pattern_comparison_predictions.csv
    outputs/pattern_comparison_summary.txt
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
FOLDS = 5

LABEL_PATH = ROOT / "train" / "labels.csv"

DATASETS = {
    "baseline_211": {
        "train": ROOT / "train_features_advanced.csv",
        "validation": ROOT / "validation_features_advanced.csv",
    },
    "patterns_316": {
        "train": ROOT / "train_features_patterns.csv",
        "validation": ROOT / "validation_features_patterns.csv",
    },
}


def prepare_data(paths, labels):
    train = pd.read_csv(paths["train"])
    validation = pd.read_csv(paths["validation"])

    if train["TRANSACTION_HASH"].duplicated().any():
        raise ValueError("Duplicate train hashes")

    if validation["TRANSACTION_HASH"].duplicated().any():
        raise ValueError("Duplicate validation hashes")

    merged = train.merge(
        labels[["TRANSACTION_HASH", "is_suspicious"]],
        on="TRANSACTION_HASH",
        how="left",
        validate="one_to_one",
    )

    if merged["is_suspicious"].isna().any():
        raise ValueError("Missing labels")

    features = [
        c for c in train.columns
        if c not in {
            "TRANSACTION_HASH",
            "is_suspicious",
            "family",
            "difficulty",
            "scenario_id",
            "cohort_id",
        }
    ]

    if not set(features).issubset(validation.columns):
        raise ValueError("Validation features don't match training")

    X = merged[features].apply(
        pd.to_numeric, errors="coerce"
    )
    X_val = validation[features].apply(
        pd.to_numeric, errors="coerce"
    )

    X = X.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    # CatBoost handles NaNs in numeric features.
    X = X.astype(np.float32)
    X_val = X_val.astype(np.float32)

    y = merged["is_suspicious"].astype(int).to_numpy()

    return (
        X, y, X_val,
        merged["TRANSACTION_HASH"].to_numpy(),
        validation["TRANSACTION_HASH"].to_numpy(),
        features,
    )


def make_model(seed):
    return CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.035,
        l2_leaf_reg=5,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )


def main():
    labels = pd.read_csv(LABEL_PATH)

    if labels["TRANSACTION_HASH"].duplicated().any():
        raise ValueError("Duplicate label hashes")

    results = {}
    fold_metrics = []
    importance_rows = []
    prediction_frames = []

    reference_hashes = None
    reference_y = None

    for name, paths in DATASETS.items():
        print("\n" + "=" * 65, flush=True)
        print(f"EXPERIMENT: {name}", flush=True)
        print("=" * 65, flush=True)

        (
            X, y, X_val, hashes, val_hashes, features
        ) = prepare_data(paths, labels)

        print(f"Train: {X.shape}", flush=True)
        print(f"Validation: {X_val.shape}", flush=True)
        print(f"Suspicious: {int(y.sum())}", flush=True)

        if reference_hashes is None:
            reference_hashes = hashes
            reference_y = y
        else:
            if not np.array_equal(hashes, reference_hashes):
                raise ValueError(
                    "Training transaction order differs between datasets"
                )
            if not np.array_equal(y, reference_y):
                raise ValueError("Label order differs")

        cv = StratifiedKFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=SEED,
        )

        oof = np.zeros(len(y), dtype=float)
        val_predictions = np.zeros(len(X_val), dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(
            cv.split(X, y), start=1
        ):
            model = make_model(SEED + fold)

            model.fit(X.iloc[tr_idx], y[tr_idx])

            predictions = model.predict_proba(
                X.iloc[va_idx]
            )[:, 1]

            oof[va_idx] = predictions

            val_predictions += (
                model.predict_proba(X_val)[:, 1] / FOLDS
            )

            ap = average_precision_score(
                y[va_idx], predictions
            )

            print(
                f"Fold {fold}: AP = {ap * 100:.2f}",
                flush=True,
            )

            fold_metrics.append({
                "experiment": name,
                "fold": fold,
                "ap": ap,
                "ap_100": ap * 100,
            })

            importance_rows.extend(
                {
                    "experiment": name,
                    "fold": fold,
                    "feature": feature,
                    "importance": float(importance),
                }
                for feature, importance in zip(
                    features,
                    model.feature_importances_,
                )
            )

        total_ap = average_precision_score(y, oof)

        results[name] = {
            "ap": total_ap,
            "oof": oof,
            "features": len(features),
        }

        print(
            f"\n{name} OVERALL OOF AP: {total_ap * 100:.2f}",
            flush=True,
        )

        prediction_frames.append(
            pd.DataFrame({
                "TRANSACTION_HASH": val_hashes,
                "experiment": name,
                "score": val_predictions,
            })
        )

    baseline = results["baseline_211"]["ap"]
    expanded = results["patterns_316"]["ap"]

    delta = (expanded - baseline) * 100

    print("\n" + "=" * 65, flush=True)
    print("FINAL COMPARISON", flush=True)
    print("=" * 65, flush=True)

    print(f"Baseline (211 features): {baseline * 100:.2f}")
    print(f"Expanded (316 features): {expanded * 100:.2f}")
    print(f"Difference: {delta:+.2f} AP points")

    if delta > 0:
        print("\nRESULT: Expanded features improved local AP.")
    elif delta < 0:
        print("\nRESULT: Baseline performed better.")
    else:
        print("\nRESULT: No measured AP difference.")

    pd.DataFrame(fold_metrics).to_csv(
        OUT / "pattern_comparison_scores.csv",
        index=False,
    )

    pd.DataFrame({
        "TRANSACTION_HASH": reference_hashes,
        "is_suspicious": reference_y,
        "baseline_oof": results["baseline_211"]["oof"],
        "patterns_oof": results["patterns_316"]["oof"],
    }).to_csv(
        OUT / "pattern_comparison_oof.csv",
        index=False,
    )

    importance = pd.DataFrame(importance_rows)

    importance = (
        importance.groupby(
            ["experiment", "feature"],
            as_index=False,
        )["importance"]
        .mean()
        .sort_values(
            ["experiment", "importance"],
            ascending=[True, False],
        )
    )

    importance.to_csv(
        OUT / "pattern_comparison_importance.csv",
        index=False,
    )

    pd.concat(
        prediction_frames,
        ignore_index=True,
    ).to_csv(
        OUT / "pattern_comparison_predictions.csv",
        index=False,
    )

    summary = (
        f"Baseline AP: {baseline * 100:.4f}\n"
        f"Expanded AP: {expanded * 100:.4f}\n"
        f"Difference: {delta:+.4f}\n"
        f"Features: 211 vs 316\n"
        f"Evaluation: {FOLDS}-fold stratified OOF\n"
    )

    (OUT / "pattern_comparison_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\nResults saved to outputs/", flush=True)


if __name__ == "__main__":
    main()
