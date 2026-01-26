#src/policy.py
from __future__ import annotations

import pandas as pd
import numpy as np
from .config import DT_HOURS


def compute_thresholds(prb_series: pd.Series, q1=0.3, q2=0.7) -> tuple[float, float]:
    s = pd.to_numeric(prb_series, errors="coerce").dropna()
    if len(s) < 10:
        return (np.nan, np.nan)
    return (float(s.quantile(q1)), float(s.quantile(q2)))


def decide_state(prb: float, p30: float, p70: float) -> str:
    if pd.isna(prb) or pd.isna(p30) or pd.isna(p70):
        return "UNKNOWN"
    if prb < p30:
        return "SLEEP"
    if prb < p70:
        return "ECO"
    return "FULL"


def decide_state_hysteresis(
    prb: float,
    p30: float,
    p70: float,
    prev_state: str,
    h_sleep: float,
    h_eco: float,
) -> str:
    if pd.isna(prb) or pd.isna(p30) or pd.isna(p70):
        return "UNKNOWN"

    p30_enter = p30 - h_sleep
    p30_exit = p30 + h_sleep
    p70_enter = p70 - h_eco
    p70_exit = p70 + h_eco

    # Prevent overlap pathologies
    p70_enter = max(p70_enter, p30_exit)

    ps = prev_state if prev_state in {"FULL", "ECO", "SLEEP"} else "FULL"

    if ps == "FULL":
        if prb < p30_enter:
            return "SLEEP"
        if prb < p70_enter:
            return "ECO"
        return "FULL"

    if ps == "ECO":
        if prb < p30_enter:
            return "SLEEP"
        if prb >= p70_exit:
            return "FULL"
        return "ECO"

    # ps == "SLEEP"
    if prb >= p30_exit:
        if prb < p70_enter:
            return "ECO"
        return "FULL"
    return "SLEEP"


def eco_saved_wh(rru_w: float, state: str, alpha: float) -> float:
    """
    Economy Mode savings expressed in Wh for the current sampling interval.
    """
    if pd.isna(rru_w):
        return 0.0
    return float(alpha) * float(rru_w) * DT_HOURS if state == "ECO" else 0.0


def baseline_wh(rru_w: float) -> float:
    """
    Baseline energy consumption in Wh for the current sampling interval.
    """
    if pd.isna(rru_w):
        return 0.0
    return float(rru_w) * DT_HOURS
