# app.py
from __future__ import annotations

import streamlit as st
from pathlib import Path

from src.cache import (
    get_raw_df,
    get_policy_df,
    get_view_df,
)
from src.ui_tabs import (
    render_tab_overview,
    render_tab_topn,
    render_tab_drilldown,
    render_tab_heterogeneity,
    render_tab_risk_optimization, # Import the new function
    render_tab_distribution_check,
)

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Network Energy Optimization", layout="wide")
st.title("📡 Network Energy Optimization Dashboard")

# -------------------------
# PATHS
# -------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"

# We assume the user has placed the 4G file here. 
# Adjust the filename key if yours differs.
FILE_MAP = {
    "5G Weekday (Training)": DATA_DIR / "Performance_5G_Weekday.csv",
    "5G Weekend (Evaluation)": DATA_DIR / "Performance_5G_Weekend.csv",
    "4G Weekday (Inference)": DATA_DIR / "Performance_4G_Weekday.csv",
    "4G Weekend (Inference)": DATA_DIR / "Performance_4G_Weekend.csv",
}

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("Dataset & Policy")

dataset_label = st.sidebar.selectbox("Select dataset", list(FILE_MAP.keys()))
path = FILE_MAP[dataset_label]
path_str = str(path)

if not path.exists():
    st.error(f"File not found: {path}\n\nPlease place the CSV file in: {DATA_DIR}")
    st.stop()

# -------------------------
# AUTO-DETECT GROUND TRUTH
# -------------------------
# We peek at the raw columns to see if 'ds_ms' or equivalent exists.
# (The pipeline fills missing cols with 0, so we must check RAW headers).
raw_preview = get_raw_df(path_str)
gt_cols = ["ds_ms", "Deep Sleep Time (Millisecond)", "Deep Sleep Time"]
has_gt = any(c in raw_preview.columns for c in gt_cols)

if has_gt:
    st.sidebar.success("✅ Ground Truth detected (5G)")
else:
    st.sidebar.info("ℹ️ No Ground Truth (4G Mode). Running inference only.")

# -------------------------
# WINDOW & PARAMS
# -------------------------
window_mode = st.sidebar.radio("Analytics window", ["Selected time-of-day", "Full day"], index=0)

tod_order = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
tod = st.sidebar.select_slider("Time of day (30-min bins)", options=tod_order, value="12:00")
tod_bin = tod_order.index(tod)

st.sidebar.divider()
st.sidebar.subheader("Controller")

alpha = st.sidebar.slider("Economy saving fraction α (RRU)", 0.1, 0.9, 0.45, 0.05)
controller_type = st.sidebar.radio("Type", ["Heuristic", "ML"], index=1)

# Defaults
threshold_scope = "Global"
use_hysteresis = True
h_sleep = 2.0
h_eco = 2.0
ml_tau_on = 0.80
ml_tau_off = 0.70
ml_hyst_en = True

# ML Feature defaults (Must match training!)
ml_use_energy = False
ml_use_prev = True
ml_use_time = True
ml_use_cyc = True

if controller_type == "Heuristic":
    threshold_scope = st.sidebar.selectbox("Threshold scope", ["Global", "Per-BaseStation", "Per-Cell"])
    use_hysteresis = st.sidebar.checkbox("Hysteresis", value=True)
    if use_hysteresis:
        h_sleep = st.sidebar.slider("Hysteresis (Sleep) [pp]", 0.0, 5.0, 2.0)
        h_eco = st.sidebar.slider("Hysteresis (Eco) [pp]", 0.0, 5.0, 2.0)
else:
    # ML Controller
    ml_model_path = st.sidebar.text_input("Model path", str(MODELS_DIR / "sleep_on_5g_weekday.joblib"))
    ml_hyst_en = st.sidebar.checkbox("Hysteresis", value=True)
    ml_tau_on = st.sidebar.slider("Enter ECO (p >= )", 0.0, 1.0, 0.80)
    ml_tau_off = st.sidebar.slider("Exit ECO (p <= )", 0.0, 1.0, 0.70, disabled=not ml_hyst_en)
    
    with st.sidebar.expander("Feature Spec (Advanced)"):
        st.caption("Must match trained model configuration")
        ml_use_energy = st.checkbox("Energy Features", False)
        ml_use_prev = st.checkbox("Prev Features", True)
        ml_use_time = st.checkbox("Time Features", True)
        ml_use_cyc = st.checkbox("Cyclical Time", True)

# -------------------------
# COMPUTATION
# -------------------------
try:
    # Get Policy DF (Full Day)
    df_policy = get_policy_df(
        path_str,
        controller_type=controller_type,
        alpha=float(alpha),
        threshold_scope=threshold_scope,
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        ml_model_path=ml_model_path if controller_type=="ML" else "",
        ml_tau_on=float(ml_tau_on),
        ml_tau_off=float(ml_tau_off),
        ml_hysteresis_enabled=bool(ml_hyst_en),
        ml_use_energy_features=bool(ml_use_energy),
        ml_use_prev_features=bool(ml_use_prev),
        ml_use_time_features=bool(ml_use_time),
        ml_use_time_cyclical=bool(ml_use_cyc),
    )
    
    # Get View DF (Windowed)
    df_view = get_view_df(
        path_str,
        controller_type=controller_type,
        alpha=float(alpha),
        threshold_scope=threshold_scope,
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        ml_model_path=ml_model_path if controller_type=="ML" else "",
        ml_tau_on=float(ml_tau_on),
        ml_tau_off=float(ml_tau_off),
        ml_hysteresis_enabled=bool(ml_hyst_en),
        ml_use_energy_features=bool(ml_use_energy),
        ml_use_prev_features=bool(ml_use_prev),
        ml_use_time_features=bool(ml_use_time),
        ml_use_time_cyclical=bool(ml_use_cyc),
        window_mode=window_mode,
        tod_bin=int(tod_bin),
    )
except Exception as e:
    st.error(f"Pipeline computation failed. Details: {e}")
    st.stop()

window_label = f"{window_mode} ({tod})" if "Selected" in window_mode else "Full Day"

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🌍 Overview", "🏭 Top-N Savings", "🔍 Drill-Down", "📊 Heterogeneity", "⚖️ Risk & Optimization", "📉 Drift Detection"]
)

with tab1:
    render_tab_overview(
        df_view=df_view,
        df_policy_full=df_policy,
        tod=tod,
        dataset=dataset_label,
        window_label=window_label,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        ml_tau_on=float(ml_tau_on),
        ml_tau_off=float(ml_tau_off),
        ml_hysteresis_enabled=bool(ml_hyst_en),
        show_gt_metrics=has_gt,
    )

with tab2:
    render_tab_topn(
        df_view=df_view,
        tod=tod,
        dataset=dataset_label,
        window_label=window_label,
        path_str=path_str,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        ml_model_path=ml_model_path if controller_type=="ML" else "",
        ml_tau_on=float(ml_tau_on),
        ml_tau_off=float(ml_tau_off),
        ml_hysteresis_enabled=bool(ml_hyst_en),
        ml_use_energy_features=bool(ml_use_energy),
        ml_use_prev_features=bool(ml_use_prev),
        ml_use_time_features=bool(ml_use_time),
        ml_use_time_cyclical=bool(ml_use_cyc),
        window_mode=window_mode,
        tod_bin=int(tod_bin),
        show_gt_metrics=has_gt,
    )

with tab3:
    render_tab_drilldown(
        df_view=df_view,
        tod=tod,
        dataset=dataset_label,
        window_label=window_label,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        show_gt_metrics=has_gt,
    )

with tab4:
    render_tab_heterogeneity(
        df_view=df_view,
        tod=tod,
        dataset=dataset_label,
        window_label=window_label,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        path_str=path_str,
        alpha=float(alpha),
        hysteresis_enabled=bool(use_hysteresis),
        h_sleep=float(h_sleep),
        h_eco=float(h_eco),
        ml_model_path=ml_model_path if controller_type=="ML" else "",
        ml_tau_on=float(ml_tau_on),
        ml_tau_off=float(ml_tau_off),
        ml_hysteresis_enabled=bool(ml_hyst_en),
        ml_use_energy_features=bool(ml_use_energy),
        ml_use_prev_features=bool(ml_use_prev),
        ml_use_time_features=bool(ml_use_time),
        ml_use_time_cyclical=bool(ml_use_cyc),
        window_mode=window_mode,
        tod_bin=int(tod_bin),
        show_gt_metrics=has_gt,
    )

with tab5:
    if controller_type != "ML":
        st.warning("Optimization is only available for the ML Controller.")
    else:
        render_tab_risk_optimization(
            df_view=df_view,
            df_policy_full=df_policy,
            alpha=float(alpha),
            current_tau_on=float(ml_tau_on),
            current_tau_off=float(ml_tau_off)
            # prb_threshold is now handled inside the function via slider
        )


with tab6:
    render_tab_distribution_check(
        df_view=df_view, 
        path_5g_train=str(FILE_MAP["5G Weekday (Training)"])
    )