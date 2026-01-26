# src/optimization.py
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

from src.controller_ml import MLControllerSpec, apply_simulation_only
from src.kpis import calculate_risk_metrics

def run_pareto_sweep(
    df_with_probs: pd.DataFrame,
    alpha: float,
    tau_on_range: list[float],
    tau_off_offset: float = 0.10,
    prb_threshold: float = 20.0,
) -> pd.DataFrame:
    """
    Sweeps tau_on values. 
    tau_off is set to tau_on - tau_off_offset (constrained to >= 0).
    Returns a summary DataFrame of (Savings vs Risk).
    """
    results = []

    # Use a progress bar if running in Streamlit
    prog = st.progress(0.0)
    n_steps = len(tau_on_range)

    for i, t_on in enumerate(tau_on_range):
        t_off = max(0.0, t_on - tau_off_offset)
        
        ctrl = MLControllerSpec(
            tau_on=t_on,
            tau_off=t_off,
            hysteresis_enabled=True
        )
        
        # Fast simulation (reuses existing p_sleep_on)
        sim_df = apply_simulation_only(df_with_probs, controller=ctrl, alpha=alpha)
        
        # Metrics
        total_kwh = sim_df["eco_saved_Wh"].sum() / 1000.0
        risk_meta = calculate_risk_metrics(sim_df, prb_threshold=prb_threshold)
        
        results.append({
            "tau_on": t_on,
            "tau_off": t_off,
            "saved_kwh": total_kwh,
            "risk_pct": risk_meta["risk_percent_total"],
            "risk_eco_pct": risk_meta["risk_percent_eco"],
            "eco_coverage_pct": (100.0 * risk_meta["eco_intervals"] / len(sim_df)) if len(sim_df) else 0.0
        })
        
        prog.progress((i + 1) / n_steps)
    
    prog.empty()
    return pd.DataFrame(results)