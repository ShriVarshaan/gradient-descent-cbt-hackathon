"""Step 2a: time-ordered validation folds with an embargo gap."""
import numpy as np


def time_folds(close_time, n_bins=5, embargo_s=3600):
    """Expanding-window folds. Fold k validates on time bin k and trains on bins before it,
    dropping any training row within embargo_s seconds of the validation start."""
    t = np.asarray(close_time, dtype="int64")
    edges = np.linspace(t.min(), t.max() + 1, n_bins + 1)
    bin_id = np.searchsorted(edges, t, side="right") - 1
    folds = []
    for k in range(1, n_bins):
        val = np.flatnonzero(bin_id == k)
        train = np.flatnonzero((bin_id < k) & (t < edges[k] - embargo_s))
        folds.append((train, val))
    return folds


def holdout_mask(close_time, frac=0.8):
    """True for the latest (1 - frac) of transactions by time. Used once, to report a final score;
    never for tuning or feature selection."""
    t = np.asarray(close_time, dtype="int64")
    return t > np.quantile(t, frac)
