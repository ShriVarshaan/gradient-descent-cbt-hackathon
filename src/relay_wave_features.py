"""Relay-wave features: rare sender-receiver pairs that recur in separate waves across the week.

Relay scenarios fire every ~40-48h: a small crew pays one hub within an hour or two, then the
hub forwards within a few hours. Each crew pair is used only 2-6 times, on different days, while
the accounts themselves are busy with legitimate traffic. These columns describe the pair's
recurrence and how many recurring-rare pairs gather at the sender and receiver. They use the
whole split (allowed: the README calls this an offline investigation).
"""
import numpy as np
import pandas as pd

WAVE_GAP = 12 * 3600  # pair uses further apart than this belong to different waves
MAX_PAIR = 6          # pairs used more often than this are routine relationships
CO_WINDOW = 2 * 3600
FWD_WINDOW = 6 * 3600

RELAY_FEATURES = [
    "rw_pair_n",
    "rw_pair_waves",
    "rw_pair_span_h",
    "rw_pair_recurring",
    "rw_recv_rr_in_senders",
    "rw_recv_rr_in_waves",
    "rw_send_rr_out_receivers",
    "rw_send_rr_in_senders",
    "rw_recv_rr_out_receivers",
    "rw_co_arrivals_2h",
    "rw_recv_forwards_6h",
    "rw_send_rr_inflow_6h",
]


def _waves(ts):
    ts = np.sort(ts)
    return 1 + int((np.diff(ts) > WAVE_GAP).sum())


def build_relay_wave_features(payments):
    df = payments[["TRANSACTION_HASH", "TIMESTAMP", "FROM_ADDRESS", "TO_ADDRESS"]].copy()
    df["T"] = pd.to_numeric(df["TIMESTAMP"], errors="raise").astype("int64")
    pair = df.groupby(["FROM_ADDRESS", "TO_ADDRESS"])["T"]
    df["rw_pair_n"] = pair.transform("size")
    df["rw_pair_waves"] = pair.transform(lambda s: _waves(s.to_numpy()))
    df["rw_pair_span_h"] = (pair.transform("max") - pair.transform("min")) / 3600
    rr = (df["rw_pair_n"].between(2, MAX_PAIR) & (df["rw_pair_waves"] >= 2)).astype(int)
    df["rw_pair_recurring"] = rr

    r = df[rr == 1]
    in_senders = r.groupby("TO_ADDRESS")["FROM_ADDRESS"].nunique()
    in_waves = r.groupby("TO_ADDRESS")["T"].apply(lambda s: _waves(s.to_numpy()))
    out_receivers = r.groupby("FROM_ADDRESS")["TO_ADDRESS"].nunique()
    df["rw_recv_rr_in_senders"] = df["TO_ADDRESS"].map(in_senders).fillna(0)
    df["rw_recv_rr_in_waves"] = df["TO_ADDRESS"].map(in_waves).fillna(0)
    df["rw_send_rr_out_receivers"] = df["FROM_ADDRESS"].map(out_receivers).fillna(0)
    df["rw_send_rr_in_senders"] = df["FROM_ADDRESS"].map(in_senders).fillna(0)
    df["rw_recv_rr_out_receivers"] = df["TO_ADDRESS"].map(out_receivers).fillna(0)

    # Timing around the current payment, using only recurring-rare payments of the hub.
    rr_in = {a: np.sort(g.to_numpy()) for a, g in r.groupby("TO_ADDRESS")["T"]}
    out_all = {a: np.sort(g.to_numpy()) for a, g in df.groupby("FROM_ADDRESS")["T"]}
    empty = np.array([], dtype="int64")
    t, src, dst = df["T"].to_numpy(), df["FROM_ADDRESS"].to_numpy(), df["TO_ADDRESS"].to_numpy()
    co, fwd, inflow = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    for i in range(len(df)):
        a = rr_in.get(dst[i], empty)
        co[i] = np.searchsorted(a, t[i] + CO_WINDOW, "right") - np.searchsorted(a, t[i] - CO_WINDOW, "left")
        o = out_all.get(dst[i], empty)
        fwd[i] = np.searchsorted(o, t[i] + FWD_WINDOW, "right") - np.searchsorted(o, t[i], "right")
        s = rr_in.get(src[i], empty)
        inflow[i] = np.searchsorted(s, t[i], "left") - np.searchsorted(s, t[i] - FWD_WINDOW, "left")
    df["rw_co_arrivals_2h"] = co - rr.to_numpy()  # exclude the payment itself
    df["rw_recv_forwards_6h"] = fwd
    df["rw_send_rr_inflow_6h"] = inflow
    return df[["TRANSACTION_HASH"] + RELAY_FEATURES]


# v2: wave spacing (relays repeat every ~40-48h) and how much of a hub's traffic is the crew.
PERIOD_LO, PERIOD_HI = 30 * 3600, 60 * 3600

RELAY_V2_FEATURES = RELAY_FEATURES + [
    "rw_recv_periodic_gaps",
    "rw_send_periodic_gaps",
    "rw_pair_periodic_gaps",
    "rw_recv_crew_share",
    "rw_send_crew_share",
]


def _wave_starts(ts):
    ts = np.sort(ts)
    return ts[np.r_[True, np.diff(ts) > WAVE_GAP]]


def _periodic_gaps(ts):
    g = np.diff(_wave_starts(ts))
    return int(((g >= PERIOD_LO) & (g <= PERIOD_HI)).sum())


def build_relay_wave_v2_features(payments):
    out = build_relay_wave_features(payments)
    df = payments[["TRANSACTION_HASH", "TIMESTAMP", "FROM_ADDRESS", "TO_ADDRESS"]].copy()
    df["T"] = pd.to_numeric(df["TIMESTAMP"], errors="raise").astype("int64")
    df = df.merge(out[["TRANSACTION_HASH", "rw_pair_recurring"]], on="TRANSACTION_HASH")
    r = df[df["rw_pair_recurring"] == 1]
    recv_gaps = r.groupby("TO_ADDRESS")["T"].apply(lambda s: _periodic_gaps(s.to_numpy()))
    send_gaps = r.groupby("FROM_ADDRESS")["T"].apply(lambda s: _periodic_gaps(s.to_numpy()))
    pair_gaps = df.groupby(["FROM_ADDRESS", "TO_ADDRESS"])["T"].transform(lambda s: _periodic_gaps(s.to_numpy()))
    crew_in = r.groupby("TO_ADDRESS")["FROM_ADDRESS"].nunique() / df.groupby("TO_ADDRESS")["FROM_ADDRESS"].nunique()
    crew_out = r.groupby("FROM_ADDRESS")["TO_ADDRESS"].nunique() / df.groupby("FROM_ADDRESS")["TO_ADDRESS"].nunique()
    df["rw_recv_periodic_gaps"] = df["TO_ADDRESS"].map(recv_gaps).fillna(0)
    df["rw_send_periodic_gaps"] = df["FROM_ADDRESS"].map(send_gaps).fillna(0)
    df["rw_pair_periodic_gaps"] = pair_gaps
    df["rw_recv_crew_share"] = df["TO_ADDRESS"].map(crew_in).fillna(0)
    df["rw_send_crew_share"] = df["FROM_ADDRESS"].map(crew_out).fillna(0)
    return out.merge(df[["TRANSACTION_HASH"] + RELAY_V2_FEATURES[len(RELAY_FEATURES):]], on="TRANSACTION_HASH")
