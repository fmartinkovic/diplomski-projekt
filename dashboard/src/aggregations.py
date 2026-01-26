#aggregations.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import resolve_col, DT_HOURS
from src.policy import decide_state, decide_state_hysteresis


def slice_time_of_day(df: pd.DataFrame, tod_bin: int) -> pd.DataFrame:
    return df.loc[df["tod_bin"] == tod_bin].copy()


def _require_columns(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: Missing required columns: {missing}")


def _ensure_threshold_columns_exist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "p30" not in df.columns:
        df["p30"] = np.nan
    if "p70" not in df.columns:
        df["p70"] = np.nan
    return df


def _repair_prb_suffixes(df: pd.DataFrame, prb_name: str) -> pd.DataFrame:
    df = df.copy()
    if prb_name in df.columns:
        return df

    px = f"{prb_name}_x"
    py = f"{prb_name}_y"

    if px in df.columns and py in df.columns:
        df[prb_name] = df[px]
        df.loc[df[prb_name].isna(), prb_name] = df.loc[df[prb_name].isna(), py]
        df = df.drop(columns=[px, py], errors="ignore")
        return df

    if px in df.columns:
        df[prb_name] = df[px]
        df = df.drop(columns=[px], errors="ignore")
        return df

    if py in df.columns:
        df[prb_name] = df[py]
        df = df.drop(columns=[py], errors="ignore")
        return df

    return df


def _get_threshold_keys(df: pd.DataFrame, threshold_scope: str) -> list[str]:
    bs_col = resolve_col(df, "bs")
    cell_col = resolve_col(df, "cell")

    if threshold_scope == "Global":
        return ["tod_bin"]
    if threshold_scope == "Per-BaseStation":
        return [bs_col, "tod_bin"]
    if threshold_scope == "Per-Cell":
        return [bs_col, cell_col, "tod_bin"]

    raise ValueError(f"Unknown threshold_scope: {threshold_scope}")


def _compute_threshold_table(df: pd.DataFrame, keys: list[str], prb_col: str, min_n: int = 10) -> pd.DataFrame:
    """
    Stable threshold computation that always returns columns p30 and p70.
    """
    d = df[keys + [prb_col]].copy()
    d[prb_col] = pd.to_numeric(d[prb_col], errors="coerce")

    n_valid = d.groupby(keys, dropna=False)[prb_col].count().rename("n_valid").reset_index()

    q = (
        d.groupby(keys, dropna=False)[prb_col]
        .quantile([0.3, 0.7])
        .unstack(level=-1)
        .reset_index()
        .rename(columns={0.3: "p30", 0.7: "p70"})
    )

    thr = n_valid.merge(q, on=keys, how="left")
    thr.loc[thr["n_valid"] < min_n, ["p30", "p70"]] = np.nan
    thr = thr.drop(columns=["n_valid"])

    if "p30" not in thr.columns:
        thr["p30"] = np.nan
    if "p70" not in thr.columns:
        thr["p70"] = np.nan

    return thr


def compute_threshold_table_for_scope(df: pd.DataFrame, threshold_scope: str, min_n: int = 10) -> pd.DataFrame:
    """
    Public helper used for persistence: compute thresholds for the dataset once per scope.

    Returns a table with keys(scope) + ['p30','p70'].
    """
    prb_col = resolve_col(df, "prb")
    df2 = _repair_prb_suffixes(df, prb_col)
    _require_columns(df2, ["tod_bin", prb_col], where="compute_threshold_table_for_scope")

    keys = _get_threshold_keys(df2, threshold_scope)
    thr = _compute_threshold_table(df2, keys=keys, prb_col=prb_col, min_n=min_n)
    thr = thr[keys + ["p30", "p70"]]
    return thr


def _attach_thresholds_per_bin(
    df: pd.DataFrame,
    threshold_scope: str,
    thresholds_table: pd.DataFrame | None = None,
    min_n: int = 10,
) -> pd.DataFrame:
    """
    Attach p30/p70 thresholds.
    If thresholds_table is provided, it is merged directly (fast, no quantiles).
    Otherwise, thresholds are computed on the fly.
    """
    df = df.copy()

    prb_col = resolve_col(df, "prb")
    df = _repair_prb_suffixes(df, prb_col)
    _require_columns(df, ["tod_bin", prb_col], where="_attach_thresholds_per_bin")

    keys = _get_threshold_keys(df, threshold_scope)

    if thresholds_table is None:
        thr = _compute_threshold_table(df, keys=keys, prb_col=prb_col, min_n=min_n)
        thr = thr[keys + ["p30", "p70"]]
    else:
        thr = thresholds_table.copy()
        # Basic contract check
        _require_columns(thr, keys + ["p30", "p70"], where="_attach_thresholds_per_bin(thresholds_table)")
        thr = thr[keys + ["p30", "p70"]]

    df = df.merge(thr, on=keys, how="left", validate="m:1")
    df = _ensure_threshold_columns_exist(df)
    df = _repair_prb_suffixes(df, prb_col)
    return df


def _fallback_global_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prb_col = resolve_col(df, "prb")
    df = _repair_prb_suffixes(df, prb_col)
    _require_columns(df, [prb_col], where="_fallback_global_thresholds")

    s = pd.to_numeric(df[prb_col], errors="coerce").dropna()
    if len(s) < 10:
        df["p30"] = np.nan
        df["p70"] = np.nan
        return df

    df["p30"] = float(s.quantile(0.3))
    df["p70"] = float(s.quantile(0.7))
    return df


def _apply_hysteresis_states(df: pd.DataFrame, h_sleep: float, h_eco: float) -> pd.DataFrame:
    df = df.copy()

    prb_col = resolve_col(df, "prb")
    bs_col = resolve_col(df, "bs")
    cell_col = resolve_col(df, "cell")

    df = _repair_prb_suffixes(df, prb_col)
    _require_columns(
        df,
        [bs_col, cell_col, "DayIndex", "tod_bin", prb_col, "p30", "p70"],
        where="_apply_hysteresis_states",
    )

    df = df.sort_values([bs_col, cell_col, "DayIndex", "tod_bin"]).copy()

    def apply_group(g: pd.DataFrame) -> pd.DataFrame:
        prev = "FULL"
        out_states: list[str] = []

        prb_vals = pd.to_numeric(g[prb_col], errors="coerce").to_numpy()
        p30_vals = pd.to_numeric(g["p30"], errors="coerce").to_numpy()
        p70_vals = pd.to_numeric(g["p70"], errors="coerce").to_numpy()

        for prb, p30, p70 in zip(prb_vals, p30_vals, p70_vals):
            s = decide_state_hysteresis(
                prb=float(prb) if not np.isnan(prb) else np.nan,
                p30=float(p30) if not np.isnan(p30) else np.nan,
                p70=float(p70) if not np.isnan(p70) else np.nan,
                prev_state=prev,
                h_sleep=h_sleep,
                h_eco=h_eco,
            )
            out_states.append(s)
            if s in {"FULL", "ECO", "SLEEP"}:
                prev = s

        g = g.copy()
        g["State"] = out_states
        return g

    return df.groupby([bs_col, cell_col, "DayIndex"], dropna=False, group_keys=False).apply(apply_group)


def apply_policy_and_simulation(
    df: pd.DataFrame,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool = True,
    h_sleep: float = 2.0,
    h_eco: float = 2.0,
    thresholds_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df_out = df.copy()

    prb_col = resolve_col(df_out, "prb")
    rru_col = resolve_col(df_out, "rru_w")
    df_out = _repair_prb_suffixes(df_out, prb_col)

    _require_columns(df_out, ["tod_bin", prb_col, rru_col], where="apply_policy_and_simulation")

    # Attach thresholds (precomputed if provided)
    df_out = _attach_thresholds_per_bin(
        df_out,
        threshold_scope=threshold_scope,
        thresholds_table=thresholds_table,
        min_n=10,
    )

    # If thresholds are NaN everywhere, fallback globally (rare but safe)
    if df_out["p30"].isna().all() or df_out["p70"].isna().all():
        df_out = _fallback_global_thresholds(df_out)

    # --- State
    if hysteresis_enabled:
        if "DayIndex" not in df_out.columns:
            raise ValueError("DayIndex missing. Ensure prepare_5g() adds DayIndex before calling policy.")
        df_out = _apply_hysteresis_states(df_out, h_sleep=h_sleep, h_eco=h_eco)
    else:
        prb = pd.to_numeric(df_out[prb_col], errors="coerce")
        p30 = pd.to_numeric(df_out["p30"], errors="coerce")
        p70 = pd.to_numeric(df_out["p70"], errors="coerce")

        unknown = prb.isna() | p30.isna() | p70.isna()
        sleep = prb < p30
        eco = (prb >= p30) & (prb < p70)

        df_out["State"] = np.select(
            [unknown, sleep, eco],
            ["UNKNOWN", "SLEEP", "ECO"],
            default="FULL",
        )

    # --- Energy (Wh) vectorized
    rru = pd.to_numeric(df_out[rru_col], errors="coerce").fillna(0.0)
    df_out["baseline_Wh"] = rru * DT_HOURS
    is_eco = df_out["State"].astype(str).eq("ECO")
    df_out["eco_saved_Wh"] = (alpha * rru * DT_HOURS) * is_eco.astype(float)

    return df_out


def bs_summary(df_view: pd.DataFrame) -> pd.DataFrame:
    bs_col = resolve_col(df_view, "bs")
    prb_col = resolve_col(df_view, "prb")
    traffic_col = resolve_col(df_view, "traffic_kb")

    _require_columns(
        df_view,
        [bs_col, prb_col, traffic_col, "baseline_Wh", "eco_saved_Wh", "sleep_on"],
        where="bs_summary",
    )

    out = (
        df_view.groupby(bs_col, dropna=False)
        .agg(
            traffic_kbyte=(traffic_col, "sum"),
            mean_prb=(prb_col, "mean"),
            baseline_Wh=("baseline_Wh", "sum"),
            eco_saved_Wh=("eco_saved_Wh", "sum"),
            p_sleep=("sleep_on", "mean"),
        )
        .reset_index()
    )

    out["eco_saved_pct"] = np.where(out["baseline_Wh"] > 0, 100.0 * out["eco_saved_Wh"] / out["baseline_Wh"], 0.0)
    return out


def bs_mean_prb(df_view: pd.DataFrame) -> pd.DataFrame:
    bs_col = resolve_col(df_view, "bs")
    prb_col = resolve_col(df_view, "prb")
    return (
        df_view.groupby(bs_col, dropna=False)[prb_col]
        .mean()
        .reset_index()
        .rename(columns={prb_col: "mean_prb"})
    )


def state_distribution(states: pd.Series) -> pd.DataFrame:
    dist = (states.value_counts(normalize=True) * 100).reset_index()
    dist.columns = ["State", "Percent"]
    return dist
