# src/controller_ml.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from src.config import DT_HOURS, resolve_col
from src.ml.features_shared import FeatureSpec, make_X


@dataclass(frozen=True)
class MLControllerSpec:
    tau_on: float = 0.80
    tau_off: float = 0.70
    hysteresis_enabled: bool = True

    def resolved_tau_off(self) -> float:
        if not self.hysteresis_enabled:
            return float(self.tau_on)
        return float(min(self.tau_on, self.tau_off))


def load_model(model_path: str | Path):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"ML model not found: {p}")
    return joblib.load(str(p))


def predict_p_sleep_on(df_prepared: pd.DataFrame, model, feature_spec: FeatureSpec) -> np.ndarray:
    X = make_X(df_prepared, spec=feature_spec, return_feature_names=False)
    p = model.predict_proba(X)[:, 1].astype(float)
    return np.clip(p, 0.0, 1.0)


def decide_state_prob_hysteresis(p: float, prev: str, tau_on: float, tau_off: float) -> str:
    ps = prev if prev in {"FULL", "ECO"} else "FULL"
    if np.isnan(p):
        return ps
    if ps == "FULL":
        return "ECO" if p >= tau_on else "FULL"
    return "FULL" if p <= tau_off else "ECO"


def apply_simulation_only(
    df_with_probs: pd.DataFrame,
    controller: MLControllerSpec,
    alpha: float,
) -> pd.DataFrame:
    """
    Fast path: Applies hysteresis and energy calc assuming 'p_sleep_on' exists.
    Ensures output schema matches dashboard expectations (p30, p70).
    """
    df_out = df_with_probs.copy()
    
    bs_col = resolve_col(df_out, "bs")
    cell_col = resolve_col(df_out, "cell")
    rru_col = resolve_col(df_out, "rru_w")

    if "p_sleep_on" not in df_out.columns:
        raise ValueError("apply_simulation_only: 'p_sleep_on' column missing.")
    
    if "DayIndex" not in df_out.columns or "tod_bin" not in df_out.columns:
        raise ValueError("apply_simulation_only: Required time features (DayIndex, tod_bin) missing.")

    tau_on = float(controller.tau_on)
    tau_off = float(controller.resolved_tau_off())

    df_out = df_out.sort_values([bs_col, cell_col, "DayIndex", "tod_bin"]).copy()

    def apply_group(g: pd.DataFrame) -> pd.DataFrame:
        prev = "FULL"
        out_states: list[str] = []
        pvals = pd.to_numeric(g["p_sleep_on"], errors="coerce").to_numpy()

        for pv in pvals:
            s = decide_state_prob_hysteresis(float(pv) if not np.isnan(pv) else np.nan, prev, tau_on, tau_off)
            out_states.append(s)
            prev = s

        g = g.copy()
        g["State"] = out_states
        return g

    df_out = (
        df_out.groupby([bs_col, cell_col, "DayIndex"], dropna=False, group_keys=False)
        .apply(apply_group)
    )

    # Energy simulation
    rru = pd.to_numeric(df_out[rru_col], errors="coerce").fillna(0.0)
    df_out["baseline_Wh"] = rru * DT_HOURS
    is_eco = df_out["State"].astype(str).eq("ECO")
    df_out["eco_saved_Wh"] = (float(alpha) * rru * DT_HOURS) * is_eco.astype(float)
    
    # --- SCHEMA COMPATIBILITY ---
    # Downstream UI expects these columns even if they are empty
    if "p30" not in df_out.columns:
        df_out["p30"] = np.nan
    if "p70" not in df_out.columns:
        df_out["p70"] = np.nan
    
    return df_out


def apply_ml_controller_and_simulation(
    df: pd.DataFrame,
    *,
    model,
    feature_spec: FeatureSpec,
    controller: MLControllerSpec,
    alpha: float,
) -> pd.DataFrame:
    """
    Full path: Prediction -> Hysteresis -> Energy -> Schema Alignment.
    """
    df_out = df.copy()
    
    # 1. Prediction
    p = predict_p_sleep_on(df_out, model=model, feature_spec=feature_spec)
    df_out["p_sleep_on"] = p

    # 2. Simulation (Hysteresis + Energy + Placeholders)
    return apply_simulation_only(df_out, controller=controller, alpha=alpha)