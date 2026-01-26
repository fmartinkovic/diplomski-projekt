#src/features.py
from __future__ import annotations

import pandas as pd
import numpy as np

from .config import DT_SECONDS, resolve_col


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts_col = resolve_col(df, "ts")

    # Dataset contains HH:MM (possibly with seconds). Use first 5 chars.
    ts = pd.to_datetime(df[ts_col].astype(str).str.slice(0, 5), format="%H:%M", errors="coerce")
    df = df.loc[ts.notna()].copy()

    df["Timestamp_dt"] = ts
    df["Hour"] = df["Timestamp_dt"].dt.hour.astype(int)
    df["Minute"] = df["Timestamp_dt"].dt.minute.astype(int)
    df["tod_bin"] = (df["Hour"] * 2 + (df["Minute"] >= 30).astype(int)).astype(int)

    tod_order = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    df["TimeOfDay"] = df["tod_bin"].map({i: tod_order[i] for i in range(48)})
    df["TimeOfDay"] = pd.Categorical(df["TimeOfDay"], categories=tod_order, ordered=True)
    return df


def add_units_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    traffic_col = resolve_col(df, "traffic_kb") if "Traffic Volume (KByte)" in df.columns else None
    if traffic_col and traffic_col in df.columns:
        df["Traffic_MB"] = pd.to_numeric(df[traffic_col], errors="coerce") / 1024.0

    ds_col = resolve_col(df, "ds_ms") if "Deep Sleep Time (Millisecond)" in df.columns else None
    if ds_col and ds_col in df.columns:
        ds_s = pd.to_numeric(df[ds_col], errors="coerce") / 1000.0
        df["DeepSleep_s"] = ds_s
        df["sleep_on"] = ds_s > 0
        df["sleep_frac"] = (ds_s.clip(lower=0) / DT_SECONDS).clip(upper=1.0)
    else:
        df["sleep_on"] = False
        df["sleep_frac"] = 0.0

    return df


def add_prev_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bs_col = resolve_col(df, "bs")
    cell_col = resolve_col(df, "cell")
    keys = [bs_col, cell_col]

    df = df.sort_values(keys + ["Timestamp_dt", "tod_bin"])

    if "Traffic_MB" in df.columns:
        df["Prev_Traffic_MB"] = df.groupby(keys)["Traffic_MB"].shift(1)

    users_col = resolve_col(df, "users") if "Number of Users" in df.columns else None
    if users_col and users_col in df.columns:
        df["Prev_Users"] = df.groupby(keys)[users_col].shift(1)

    return df


def add_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset has HH:MM but no explicit date. If multiple days are concatenated,
    tod_bin will wrap from 47 -> 0. We infer 'DayIndex' per (BS, Cell) whenever
    tod_bin decreases relative to previous row.
    """
    df = df.copy()

    bs_col = resolve_col(df, "bs")
    cell_col = resolve_col(df, "cell")
    keys = [bs_col, cell_col]

    df = df.sort_values(keys + ["Timestamp_dt", "tod_bin"])

    def infer_day(g: pd.DataFrame) -> pd.Series:
        d = g["tod_bin"].diff()
        return (d.fillna(0) < 0).cumsum().astype(int)

    df["DayIndex"] = df.groupby(keys, group_keys=False).apply(infer_day)
    return df


def prepare_5g(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_units_and_labels(df)
    df = add_prev_features(df)
    df = add_day_index(df)
    return df
