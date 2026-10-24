
"""Full-window wallet and temporal flow features for Decode & Detect.

Use independently on each dataset split. No labels or identifiers are learned.
The competition permits offline full-window transaction analysis.
"""

from collections import defaultdict
import numpy as np
import pandas as pd

# Output columns of build_full_window_features, in order. These use the whole split,
# future payments included, so they are not past-only like history_features.
FULL_WINDOW_FEATURES = [
    'sender_fw_out_count',
    'sender_fw_out_usd',
    'sender_fw_out_unique_counterparties',
    'sender_fw_in_count',
    'sender_fw_in_usd',
    'sender_fw_in_unique_counterparties',
    'sender_fw_total_usd',
    'sender_fw_net_usd',
    'sender_fw_flow_balance',
    'sender_fw_out_in_ratio',
    'sender_fw_in_out_ratio',
    'sender_fw_counterparty_ratio',
    'sender_fw_turnover_per_tx',
    'sender_fw_remaining_active_seconds',
    'receiver_fw_out_count',
    'receiver_fw_out_usd',
    'receiver_fw_out_unique_counterparties',
    'receiver_fw_in_count',
    'receiver_fw_in_usd',
    'receiver_fw_in_unique_counterparties',
    'receiver_fw_total_usd',
    'receiver_fw_net_usd',
    'receiver_fw_flow_balance',
    'receiver_fw_out_in_ratio',
    'receiver_fw_in_out_ratio',
    'receiver_fw_counterparty_ratio',
    'receiver_fw_turnover_per_tx',
    'receiver_fw_remaining_active_seconds',
    'sender_fw_incoming_prior_count_1h',
    'sender_fw_incoming_prior_usd_1h',
    'sender_fw_incoming_future_count_1h',
    'sender_fw_incoming_future_usd_1h',
    'sender_fw_incoming_prior_count_24h',
    'sender_fw_incoming_prior_usd_24h',
    'sender_fw_incoming_future_count_24h',
    'sender_fw_incoming_future_usd_24h',
    'sender_fw_outgoing_prior_count_1h',
    'sender_fw_outgoing_prior_usd_1h',
    'sender_fw_outgoing_future_count_1h',
    'sender_fw_outgoing_future_usd_1h',
    'sender_fw_outgoing_prior_count_24h',
    'sender_fw_outgoing_prior_usd_24h',
    'sender_fw_outgoing_future_count_24h',
    'sender_fw_outgoing_future_usd_24h',
    'receiver_fw_incoming_prior_count_1h',
    'receiver_fw_incoming_prior_usd_1h',
    'receiver_fw_incoming_future_count_1h',
    'receiver_fw_incoming_future_usd_1h',
    'receiver_fw_incoming_prior_count_24h',
    'receiver_fw_incoming_prior_usd_24h',
    'receiver_fw_incoming_future_count_24h',
    'receiver_fw_incoming_future_usd_24h',
    'receiver_fw_outgoing_prior_count_1h',
    'receiver_fw_outgoing_prior_usd_1h',
    'receiver_fw_outgoing_future_count_1h',
    'receiver_fw_outgoing_future_usd_1h',
    'receiver_fw_outgoing_prior_count_24h',
    'receiver_fw_outgoing_prior_usd_24h',
    'receiver_fw_outgoing_future_count_24h',
    'receiver_fw_outgoing_future_usd_24h',
    'fw_upstream_volume_match_1h',
    'fw_downstream_volume_match_1h',
    'fw_bridge_volume_match_1h',
    'fw_downstream_fraction_1h',
    'fw_upstream_fraction_1h',
    'fw_upstream_volume_match_24h',
    'fw_downstream_volume_match_24h',
    'fw_bridge_volume_match_24h',
    'fw_downstream_fraction_24h',
    'fw_upstream_fraction_24h',
    'fw_upstream_best_single_match_1h',
    'fw_upstream_nearest_gap_seconds_1h',
    'fw_upstream_best_single_match_24h',
    'fw_upstream_nearest_gap_seconds_24h',
    'fw_downstream_best_single_match_1h',
    'fw_downstream_nearest_gap_seconds_1h',
    'fw_downstream_best_single_match_24h',
    'fw_downstream_nearest_gap_seconds_24h',
    'fw_shared_sources',
    'fw_shared_destinations',
    'fw_sender_neighbor_out_degree_mean',
    'fw_receiver_neighbor_in_degree_mean',
    'fw_reverse_edge_exists',
]


def build_full_window_features(payments: pd.DataFrame) -> pd.DataFrame:
    required = ['TRANSACTION_HASH', 'TIMESTAMP', 'FROM_ADDRESS', 'TO_ADDRESS', 'usd_value']
    missing = set(required) - set(payments.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    df = payments[required].copy().reset_index(drop=True)
    if not df.TRANSACTION_HASH.is_unique:
        raise ValueError('Expected one payment per transaction hash')
    df['TIMESTAMP'] = pd.to_numeric(df['TIMESTAMP'], errors='raise').astype('int64')
    df['usd_value'] = pd.to_numeric(df['usd_value'], errors='raise').astype('float64')
    if not np.isfinite(df['usd_value']).all():
        raise ValueError('Non-finite USD amounts')

    result = df[['TRANSACTION_HASH']].copy()
    amounts = df.usd_value.to_numpy()
    times = df.TIMESTAMP.to_numpy()
    senders = df.FROM_ADDRESS.to_numpy()
    receivers = df.TO_ADDRESS.to_numpy()
    n = len(df)

    # Whole-week account totals and counterparties. Current edge included.
    outgoing = df.groupby('FROM_ADDRESS', sort=False).agg(
        fw_out_count=('usd_value', 'size'),
        fw_out_usd=('usd_value', 'sum'),
        fw_out_unique_counterparties=('TO_ADDRESS', 'nunique'),
        fw_out_first=('TIMESTAMP', 'min'),
        fw_out_last=('TIMESTAMP', 'max'),
    )
    incoming = df.groupby('TO_ADDRESS', sort=False).agg(
        fw_in_count=('usd_value', 'size'),
        fw_in_usd=('usd_value', 'sum'),
        fw_in_unique_counterparties=('FROM_ADDRESS', 'nunique'),
        fw_in_first=('TIMESTAMP', 'min'),
        fw_in_last=('TIMESTAMP', 'max'),
    )
    account = outgoing.join(incoming, how='outer')
    numeric = ['fw_out_count', 'fw_out_usd', 'fw_out_unique_counterparties',
               'fw_in_count', 'fw_in_usd', 'fw_in_unique_counterparties']
    account[numeric] = account[numeric].fillna(0)
    account['fw_total_usd'] = account.fw_in_usd + account.fw_out_usd
    account['fw_net_usd'] = account.fw_in_usd - account.fw_out_usd
    account['fw_flow_balance'] = 1 - np.abs(account.fw_net_usd) / (account.fw_total_usd + 1)
    account['fw_out_in_ratio'] = account.fw_out_usd / (account.fw_in_usd + 1)
    account['fw_in_out_ratio'] = account.fw_in_usd / (account.fw_out_usd + 1)
    account['fw_counterparty_ratio'] = account.fw_out_unique_counterparties / (account.fw_in_unique_counterparties + 1)
    account['fw_turnover_per_tx'] = account.fw_total_usd / (account.fw_in_count + account.fw_out_count + 1)

    for prefix, addresses in [('sender', senders), ('receiver', receivers)]:
        for col in [*numeric, 'fw_total_usd', 'fw_net_usd', 'fw_flow_balance',
                    'fw_out_in_ratio', 'fw_in_out_ratio', 'fw_counterparty_ratio',
                    'fw_turnover_per_tx']:
            result[f'{prefix}_{col}'] = pd.Series(addresses).map(account[col]).fillna(0).to_numpy()
        result[f'{prefix}_fw_remaining_active_seconds'] = (
            pd.Series(addresses).map(account[['fw_out_last', 'fw_in_last']].max(axis=1))
            .fillna(0).to_numpy() - times
        ).clip(min=0)

    # Per-wallet sorted arrays; windows exclude same-timestamp transfers to avoid
    # inventing an ordering when multiple payments share the same timestamp.
    by_in = defaultdict(list)
    by_out = defaultdict(list)
    for i, (s, r) in enumerate(zip(senders, receivers)):
        by_out[s].append(i)
        by_in[r].append(i)

    def arrays(groups):
        out = {}
        for wallet, indices in groups.items():
            ids = np.array(indices, dtype=np.int64)
            ids = ids[np.lexsort((ids, times[ids]))]
            ts = times[ids]
            vs = amounts[ids]
            out[wallet] = (ts, vs, np.r_[0., np.cumsum(vs)])
        return out

    incoming_arrays = arrays(by_in)
    outgoing_arrays = arrays(by_out)
    windows = [(3600, '1h'), (86400, '24h')]
    temporal = {}
    for side in ['sender', 'receiver']:
        for direction in ['incoming', 'outgoing']:
            for window, suffix in windows:
                for kind in ['prior_count', 'prior_usd', 'future_count', 'future_usd']:
                    temporal[f'{side}_fw_{direction}_{kind}_{suffix}'] = np.zeros(n)

    for side, wallets in [('sender', senders), ('receiver', receivers)]:
        for direction, store in [('incoming', incoming_arrays), ('outgoing', outgoing_arrays)]:
            for wallet in np.unique(wallets):
                idx = np.flatnonzero(wallets == wallet)
                record = store.get(wallet)
                if record is None:
                    continue
                ts, vals, cumulative = record
                current_times = times[idx]
                left = np.searchsorted(ts, current_times, side='left')
                right = np.searchsorted(ts, current_times, side='right')
                for window, suffix in windows:
                    prior_start = np.searchsorted(ts, current_times - window, side='left')
                    future_end = np.searchsorted(ts, current_times + window, side='right')
                    stem = f'{side}_fw_{direction}_'
                    temporal[stem + 'prior_count_' + suffix][idx] = left - prior_start
                    temporal[stem + 'prior_usd_' + suffix][idx] = cumulative[left] - cumulative[prior_start]
                    temporal[stem + 'future_count_' + suffix][idx] = future_end - right
                    temporal[stem + 'future_usd_' + suffix][idx] = cumulative[future_end] - cumulative[right]

    for name, values in temporal.items():
        result[name] = values

    # Edge-level forwarding: current sender previously received; current receiver
    # subsequently sends. Compare individual payments and aggregate USD in windows.
    for window, suffix in windows:
        before = temporal[f'sender_fw_incoming_prior_usd_{suffix}']
        after = temporal[f'receiver_fw_outgoing_future_usd_{suffix}']
        result[f'fw_upstream_volume_match_{suffix}'] = np.minimum(before, amounts) / np.maximum(np.maximum(before, amounts), 1e-9)
        result[f'fw_downstream_volume_match_{suffix}'] = np.minimum(after, amounts) / np.maximum(np.maximum(after, amounts), 1e-9)
        result[f'fw_bridge_volume_match_{suffix}'] = np.minimum(before, after) / np.maximum(np.maximum(before, after), 1e-9)
        result[f'fw_downstream_fraction_{suffix}'] = after / (amounts + 1)
        result[f'fw_upstream_fraction_{suffix}'] = before / (amounts + 1)

    # Best single-edge matches before and after, plus closest time gap.
    for side, wallets, store, temporal_direction in [
        ('upstream', senders, incoming_arrays, 'prior'),
        ('downstream', receivers, outgoing_arrays, 'future'),
    ]:
        for window, suffix in windows:
            best = np.zeros(n)
            closest_gap = np.full(n, -1.)
            for wallet in np.unique(wallets):
                idx = np.flatnonzero(wallets == wallet)
                record = store.get(wallet)
                if record is None:
                    continue
                ts, vals, _ = record
                for i in idx:
                    if temporal_direction == 'prior':
                        lo = np.searchsorted(ts, times[i] - window, side='left')
                        hi = np.searchsorted(ts, times[i], side='left')
                    else:
                        lo = np.searchsorted(ts, times[i], side='right')
                        hi = np.searchsorted(ts, times[i] + window, side='right')
                    if lo >= hi:
                        continue
                    candidates = vals[lo:hi]
                    denom = np.maximum(np.maximum(np.abs(candidates), abs(amounts[i])), 1e-9)
                    best[i] = np.max(np.minimum(np.abs(candidates), abs(amounts[i])) / denom)
                    closest_gap[i] = (times[i] - ts[hi-1]) if temporal_direction == 'prior' else (ts[lo] - times[i])
            result[f'fw_{side}_best_single_match_{suffix}'] = best
            result[f'fw_{side}_nearest_gap_seconds_{suffix}'] = closest_gap

    # Neighborhood totals across the entire observation window (not labels).
    out_neighbors = {wallet: set(df.loc[ids, 'TO_ADDRESS']) for wallet, ids in by_out.items()}
    in_neighbors = {wallet: set(df.loc[ids, 'FROM_ADDRESS']) for wallet, ids in by_in.items()}
    out_degree = {wallet: len(neighbors) for wallet, neighbors in out_neighbors.items()}
    in_degree = {wallet: len(neighbors) for wallet, neighbors in in_neighbors.items()}
    result['fw_shared_sources'] = [len(in_neighbors.get(s, set()) & in_neighbors.get(r, set())) for s, r in zip(senders, receivers)]
    result['fw_shared_destinations'] = [len(out_neighbors.get(s, set()) & out_neighbors.get(r, set())) for s, r in zip(senders, receivers)]
    result['fw_sender_neighbor_out_degree_mean'] = [np.mean([out_degree.get(w, 0) for w in out_neighbors.get(s, ())]) if out_neighbors.get(s) else 0 for s in senders]
    result['fw_receiver_neighbor_in_degree_mean'] = [np.mean([in_degree.get(w, 0) for w in in_neighbors.get(r, ())]) if in_neighbors.get(r) else 0 for r in receivers]
    result['fw_reverse_edge_exists'] = [int(s in out_neighbors.get(r, set())) for s, r in zip(senders, receivers)]

    # Stable finite outputs for LightGBM.
    feature_cols = result.columns.difference(['TRANSACTION_HASH'])
    result[feature_cols] = result[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    assert result.TRANSACTION_HASH.is_unique
    assert len(result) == n
    return result
