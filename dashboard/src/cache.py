# src/cache.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.io import load_csv
from src.config import canonicalize_columns, validate_schema
from src.features import prepare_5g
from src.aggregations import (
    apply_policy_and_simulation,
    slice_time_of_day,
    bs_summary,
    compute_threshold_table_for_scope,
)
from src.kpis import per_cell_kpis

from src.controller_ml import (
    load_model as load_ml_model,
    apply_ml_controller_and_simulation,
    MLControllerSpec,
)
from src.ml.features_shared import FeatureSpec


def _cache_dir_for_source(source_path: str) -> Path:
    p = Path(source_path)
    d = p.parent / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parquet_path_for_csv(csv_path: str) -> Path:
    p = Path(csv_path)
    cache_dir = _cache_dir_for_source(csv_path)
    return cache_dir / f"{p.stem}.parquet"


def _thr_parquet_path(csv_or_parquet_path: str, threshold_scope: str) -> Path:
    p = Path(csv_or_parquet_path)
    cache_dir = _cache_dir_for_source(str(p))
    stem = p.stem.replace(".parquet", "")
    safe_scope = threshold_scope.replace("/", "_")
    return cache_dir / f"{stem}.thresholds.{safe_scope}.parquet"


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def ensure_parquet_dataset(csv_path: str) -> str:
    pq = _parquet_path_for_csv(csv_path)
    if pq.exists():
        return str(pq)

    raw = load_csv(csv_path)
    raw = canonicalize_columns(raw)
    validate_schema(
        raw,
        required_keys=["ts", "bs", "cell", "prb", "traffic_kb", "rru_w"],
        where="ensure_parquet_dataset:raw",
    )

    _write_parquet(raw, pq)
    return str(pq)


@st.cache_data(show_spinner=False)
def get_raw_df(source_path: str) -> pd.DataFrame:
    p = Path(source_path)
    if p.suffix.lower() == ".csv":
        pq_path = ensure_parquet_dataset(source_path)
        df = _read_parquet(Path(pq_path))
    elif p.suffix.lower() == ".parquet":
        df = _read_parquet(p)
    else:
        df = load_csv(source_path)
        df = canonicalize_columns(df)

    validate_schema(df, required_keys=["ts", "bs", "cell", "prb", "traffic_kb", "rru_w"], where="cache:get_raw_df")
    return df


@st.cache_data(show_spinner=False)
def get_prepared_df(source_path: str) -> pd.DataFrame:
    raw = get_raw_df(source_path)
    df = prepare_5g(raw)
    validate_schema(
        df,
        required_keys=["ts", "bs", "cell", "prb", "traffic_kb", "rru_w"],
        required_cols=["tod_bin", "DayIndex", "sleep_on", "sleep_frac"],
        where="cache:get_prepared_df",
    )
    return df


def ensure_thresholds_parquet(source_path: str, threshold_scope: str) -> str:
    thr_path = _thr_parquet_path(source_path, threshold_scope)
    if thr_path.exists():
        return str(thr_path)

    df = get_prepared_df(source_path)
    thr = compute_threshold_table_for_scope(df, threshold_scope=threshold_scope)
    _write_parquet(thr, thr_path)
    return str(thr_path)


@st.cache_data(show_spinner=False)
def get_thresholds_df(source_path: str, threshold_scope: str) -> pd.DataFrame:
    thr_pq = ensure_thresholds_parquet(source_path, threshold_scope)
    return _read_parquet(Path(thr_pq))


@st.cache_resource(show_spinner=False)
def get_ml_model_cached(model_path: str):
    return load_ml_model(model_path)


@st.cache_data(show_spinner=False)
def get_policy_df(
    source_path: str,
    *,
    controller_type: str,
    alpha: float,
    # Heuristic params
    threshold_scope: str,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    # ML params
    ml_model_path: str,
    ml_tau_on: float,
    ml_tau_off: float,
    ml_hysteresis_enabled: bool,
    # ML feature spec (must match the trained model)
    ml_use_energy_features: bool,
    ml_use_prev_features: bool,
    ml_use_time_features: bool,
    ml_use_time_cyclical: bool,
) -> pd.DataFrame:
    df = get_prepared_df(source_path)

    if controller_type == "Heuristic":
        thr = get_thresholds_df(source_path, threshold_scope)

        out = apply_policy_and_simulation(
            df,
            threshold_scope=threshold_scope,
            alpha=alpha,
            hysteresis_enabled=hysteresis_enabled,
            h_sleep=h_sleep,
            h_eco=h_eco,
            thresholds_table=thr,
        )

        validate_schema(
            out,
            required_keys=["bs", "cell", "prb", "rru_w"],
            required_cols=["tod_bin", "DayIndex", "p30", "p70", "State", "baseline_Wh", "eco_saved_Wh"],
            where="cache:get_policy_df(Heuristic)",
        )
        return out

    if controller_type == "ML":
        model = get_ml_model_cached(ml_model_path)
        fsp = FeatureSpec(
            use_energy_features=bool(ml_use_energy_features),
            use_prev_features=bool(ml_use_prev_features),
            use_time_features=bool(ml_use_time_features),
            use_time_cyclical=bool(ml_use_time_cyclical),
        )
        ctrl = MLControllerSpec(
            tau_on=float(ml_tau_on),
            tau_off=float(ml_tau_off),
            hysteresis_enabled=bool(ml_hysteresis_enabled),
        )

        out = apply_ml_controller_and_simulation(
            df,
            model=model,
            feature_spec=fsp,
            controller=ctrl,
            alpha=alpha,
        )

        validate_schema(
            out,
            required_keys=["bs", "cell", "prb", "rru_w"],
            required_cols=["tod_bin", "DayIndex", "State", "baseline_Wh", "eco_saved_Wh", "p_sleep_on", "p30", "p70"],
            where="cache:get_policy_df(ML)",
        )
        return out

    raise ValueError(f"Unknown controller_type: {controller_type!r}. Use 'Heuristic' or 'ML'.")


@st.cache_data(show_spinner=False)
def get_view_df(
    source_path: str,
    *,
    controller_type: str,
    alpha: float,
    # Heuristic
    threshold_scope: str,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    # ML
    ml_model_path: str,
    ml_tau_on: float,
    ml_tau_off: float,
    ml_hysteresis_enabled: bool,
    ml_use_energy_features: bool,
    ml_use_prev_features: bool,
    ml_use_time_features: bool,
    ml_use_time_cyclical: bool,
    # Window
    window_mode: str,
    tod_bin: int,
) -> pd.DataFrame:
    df_policy = get_policy_df(
        source_path,
        controller_type=controller_type,
        alpha=alpha,
        threshold_scope=threshold_scope,
        hysteresis_enabled=hysteresis_enabled,
        h_sleep=h_sleep,
        h_eco=h_eco,
        ml_model_path=ml_model_path,
        ml_tau_on=ml_tau_on,
        ml_tau_off=ml_tau_off,
        ml_hysteresis_enabled=ml_hysteresis_enabled,
        ml_use_energy_features=ml_use_energy_features,
        ml_use_prev_features=ml_use_prev_features,
        ml_use_time_features=ml_use_time_features,
        ml_use_time_cyclical=ml_use_time_cyclical,
    )

    if window_mode == "Selected time-of-day":
        return slice_time_of_day(df_policy, tod_bin=tod_bin)
    return df_policy


@st.cache_data(show_spinner=False)
def get_bs_summary_cached(
    source_path: str,
    *,
    controller_type: str,
    alpha: float,
    threshold_scope: str,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    ml_model_path: str,
    ml_tau_on: float,
    ml_tau_off: float,
    ml_hysteresis_enabled: bool,
    ml_use_energy_features: bool,
    ml_use_prev_features: bool,
    ml_use_time_features: bool,
    ml_use_time_cyclical: bool,
    window_mode: str,
    tod_bin: int,
) -> pd.DataFrame:
    df_view = get_view_df(
        source_path,
        controller_type=controller_type,
        alpha=alpha,
        threshold_scope=threshold_scope,
        hysteresis_enabled=hysteresis_enabled,
        h_sleep=h_sleep,
        h_eco=h_eco,
        ml_model_path=ml_model_path,
        ml_tau_on=ml_tau_on,
        ml_tau_off=ml_tau_off,
        ml_hysteresis_enabled=ml_hysteresis_enabled,
        ml_use_energy_features=ml_use_energy_features,
        ml_use_prev_features=ml_use_prev_features,
        ml_use_time_features=ml_use_time_features,
        ml_use_time_cyclical=ml_use_time_cyclical,
        window_mode=window_mode,
        tod_bin=tod_bin,
    )
    return bs_summary(df_view)


@st.cache_data(show_spinner=False)
def get_per_cell_kpis_cached(
    source_path: str,
    *,
    controller_type: str,
    alpha: float,
    threshold_scope: str,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    ml_model_path: str,
    ml_tau_on: float,
    ml_tau_off: float,
    ml_hysteresis_enabled: bool,
    ml_use_energy_features: bool,
    ml_use_prev_features: bool,
    ml_use_time_features: bool,
    ml_use_time_cyclical: bool,
    window_mode: str,
    tod_bin: int,
) -> pd.DataFrame:
    df_view = get_view_df(
        source_path,
        controller_type=controller_type,
        alpha=alpha,
        threshold_scope=threshold_scope,
        hysteresis_enabled=hysteresis_enabled,
        h_sleep=h_sleep,
        h_eco=h_eco,
        ml_model_path=ml_model_path,
        ml_tau_on=ml_tau_on,
        ml_tau_off=ml_tau_off,
        ml_hysteresis_enabled=ml_hysteresis_enabled,
        ml_use_energy_features=ml_use_energy_features,
        ml_use_prev_features=ml_use_prev_features,
        ml_use_time_features=ml_use_time_features,
        ml_use_time_cyclical=ml_use_time_cyclical,
        window_mode=window_mode,
        tod_bin=tod_bin,
    )
    return per_cell_kpis(df_view)
