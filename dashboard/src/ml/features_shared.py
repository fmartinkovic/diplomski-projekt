# src/ml/features_shared.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import resolve_col


@dataclass(frozen=True)
class FeatureSpec:
    """
    Transfer-safe feature specification.

    Notes:
      - Uses only columns present in BOTH 5G and 4G CSVs (per your schemas).
      - Uses engineered columns added by prepare_5g(): tod_bin, Traffic_MB, Prev_Traffic_MB, Prev_Users.
      - Does NOT use any 5G-only supervision columns (Deep Sleep, sleep_on, sleep_frac) as features.

    Important:
      - use_time_features controls whether ANY time-of-day features are included.
        If False, both tod_bin and cyclical features are removed.
      - use_time_cyclical controls whether we add sin/cos encodings (only if time features enabled).
    """
    use_energy_features: bool = True
    use_prev_features: bool = True
    use_time_features: bool = True
    use_time_cyclical: bool = True  # only applied if use_time_features=True


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _time_cyclical(tod_bin: pd.Series, period: int = 48) -> Tuple[pd.Series, pd.Series]:
    x = _safe_numeric(tod_bin).fillna(0.0).astype(float)
    ang = 2.0 * np.pi * (x / float(period))
    return (np.sin(ang), np.cos(ang))


def make_X(
    df: pd.DataFrame,
    spec: FeatureSpec = FeatureSpec(),
    *,
    return_feature_names: bool = True,
) -> Tuple[pd.DataFrame, List[str]] | pd.DataFrame:
    """
    Build a transfer-safe feature matrix X from a prepared dataframe.

    Requirements on df:
      - Must contain canonical columns:
          Base Station ID, Cell ID, Timestamp, PRB Usage Ratio (%),
          Traffic Volume (KByte), Number of Users, BBU Energy (W), RRU Energy (W)
      - If you want time features: should contain 'tod_bin' (from prepare_5g()).
      - If you want prev/dynamics: should contain Prev_Traffic_MB / Prev_Users (from prepare_5g()).

    Returns:
      - X: pandas DataFrame of numeric features
      - feature_names: list[str] (optional)
    """
    prb_col = resolve_col(df, "prb")
    traffic_col = resolve_col(df, "traffic_kb")
    users_col = resolve_col(df, "users")
    bbu_col = resolve_col(df, "bbu_w")
    rru_col = resolve_col(df, "rru_w")

    out = pd.DataFrame(index=df.index)

    # --- Core load features (robust across 4G/5G)
    out["prb"] = _safe_numeric(df[prb_col])
    out["traffic_kb"] = _safe_numeric(df[traffic_col])
    out["users"] = _safe_numeric(df[users_col])

    # --- Optional energy features
    if spec.use_energy_features:
        out["bbu_w"] = _safe_numeric(df[bbu_col])
        out["rru_w"] = _safe_numeric(df[rru_col])

    # --- Time-of-day features
    if spec.use_time_features:
        if "tod_bin" in df.columns:
            out["tod_bin"] = _safe_numeric(df["tod_bin"])
            if spec.use_time_cyclical:
                s, c = _time_cyclical(df["tod_bin"])
                out["tod_sin"] = s
                out["tod_cos"] = c
        else:
            # Allow degraded behavior if caller forgot prepare_5g()
            out["tod_bin"] = np.nan
            if spec.use_time_cyclical:
                out["tod_sin"] = np.nan
                out["tod_cos"] = np.nan

    # --- Units feature if present (interpretability)
    if "Traffic_MB" in df.columns:
        out["traffic_mb"] = _safe_numeric(df["Traffic_MB"])

    # --- Prev-step dynamics
    if spec.use_prev_features:
        out["prev_traffic_mb"] = _safe_numeric(df["Prev_Traffic_MB"]) if "Prev_Traffic_MB" in df.columns else np.nan
        out["prev_users"] = _safe_numeric(df["Prev_Users"]) if "Prev_Users" in df.columns else np.nan

    feature_names = list(out.columns)
    return (out, feature_names) if return_feature_names else out


def make_y_sleep_on(df: pd.DataFrame) -> pd.Series:
    """
    Target vector for supervised learning on 5G:
      y = sleep_on (boolean) => convert to int {0,1}.
    """
    if "sleep_on" not in df.columns:
        raise ValueError("make_y_sleep_on: expected column 'sleep_on' in df (5G only).")
    return df["sleep_on"].astype(bool).astype(int)
