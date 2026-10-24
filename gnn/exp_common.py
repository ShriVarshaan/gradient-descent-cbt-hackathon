import pickle, sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, "gnn")
from train_best_ensemble import propagate_predictions_over_motifs

C = "outputs/cache/"
X = pd.read_parquet(C + "train_X.parquet")
cids = pickle.load(open(C + "train_cids.pkl", "rb"))
lab = pd.read_csv("train-data/labels.csv").set_index("TRANSACTION_HASH").loc[X.TRANSACTION_HASH].reset_index()
y = X.pop("__y").values; ts = X.pop("__ts").values; X = X.drop(columns="TRANSACTION_HASH")

def folds(ts, k=5):
    o = np.argsort(ts); f = np.zeros(len(ts), int); f[o] = np.arange(len(ts)) * k // len(ts); return f

def oof_xgb(X, y, ts, params=None, prop=True):
    fid = folds(ts); oof = np.zeros(len(y))
    for f in range(5):
        va = fid == f; lo, hi = ts[va].min(), ts[va].max()
        tr = (~va) & ((ts < lo - 7200) | (ts > hi + 7200))
        m = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                          min_child_weight=5, reg_lambda=5, n_jobs=-1, tree_method="hist", **(params or {}))
        m.fit(X[tr], y[tr]); oof[va] = m.predict_proba(X[va])[:, 1]
    return propagate_predictions_over_motifs(oof, cids) if prop else oof

def report(p, name):
    ap = average_precision_score(y, p)
    rk = pd.Series(p).rank(ascending=False).values
    out = {"AP": round(ap, 4)}
    for d in ["easy", "medium", "hard"]:
        m = (lab.difficulty == d).values
        out[d + "_medrank"] = int(np.median(rk[m]))
        out[d + "_top1250"] = round((rk[m] <= 1250).mean(), 2)
    print(name, out)
    return ap

if __name__ == "__main__":
    p = oof_xgb(X, y, ts)
    report(p, "baseline_xgb_prop")
    rk = pd.Series(p).rank(ascending=False).values
    lab["rank"] = rk
    print(lab[lab.is_suspicious == 1].groupby(["difficulty", "family"])["rank"].describe()[["count", "25%", "50%", "75%"]])
    # top false positives by family
    print(lab[(lab.is_suspicious == 0) & (rk <= 1500)].family.value_counts())
    np.save(C + "oof_base.npy", p)
