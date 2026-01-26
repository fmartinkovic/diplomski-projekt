# scripts/infer_4g_profile.py
from __future__ import annotations

import sys
from pathlib import Path
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_prepared_df
from src.ml.features_shared import FeatureSpec, make_X
from src.config import resolve_col


def _agg_profile(df_pred: pd.DataFrame, prob_col: str = "p_sleep_on") -> pd.DataFrame:
    d = df_pred.copy()
    d[prob_col] = pd.to_numeric(d[prob_col], errors="coerce")
    d = d.dropna(subset=[prob_col])

    if "tod_bin" not in d.columns:
        raise ValueError("Expected 'tod_bin' in prepared dataframe.")

    g = d.groupby("tod_bin", dropna=False)[prob_col]
    out = pd.DataFrame({
        "tod_bin": g.mean().index.astype(int),
        "p_mean": g.mean().values,
        "p_median": g.median().values,
        "p10": g.quantile(0.10).values,
        "p90": g.quantile(0.90).values,
        "n": g.size().values,
    }).sort_values("tod_bin")

    # Add TimeOfDay label if present
    if "TimeOfDay" in d.columns:
        tod_map = d.drop_duplicates("tod_bin")[["tod_bin", "TimeOfDay"]].set_index("tod_bin")["TimeOfDay"].astype(str)
        out["TimeOfDay"] = out["tod_bin"].map(tod_map).astype(str)

    return out


def _infer(df_prepared: pd.DataFrame, model, spec: FeatureSpec, out_col: str = "p_sleep_on") -> pd.DataFrame:
    X = make_X(df_prepared, spec=spec, return_feature_names=False)
    p = model.predict_proba(X)[:, 1].astype(float)
    out = df_prepared.copy()
    out[out_col] = np.clip(p, 0.0, 1.0)
    return out


def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    MODELS_DIR = ROOT / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "sleep_on_5g_weekday.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} (train first).")

    p4g_weekday = DATA_DIR / "Performance_4G_Weekday.csv"
    p4g_weekend = DATA_DIR / "Performance_4G_Weekend.csv"
    if not p4g_weekday.exists():
        raise FileNotFoundError(f"4G weekday CSV not found: {p4g_weekday}")
    if not p4g_weekend.exists():
        raise FileNotFoundError(f"4G weekend CSV not found: {p4g_weekend}")

    print("[INFO] Loading model...")
    model = joblib.load(str(model_path))

    # IMPORTANT: this must match training features.
    # You trained with energy OFF (per your recent run), so keep it OFF here.
    feature_spec = FeatureSpec(
        use_energy_features=False,
        use_prev_features=True,
        use_time_features=True,
        use_time_cyclical=True,
    )

    print("[INFO] Loading + preparing 4G weekday...")
    df_wd = load_prepared_df(str(p4g_weekday), require_sleep=False)
    print("[INFO] Inferring probabilities (weekday)...")
    df_wd_p = _infer(df_wd, model, feature_spec, out_col="p_sleep_on")
    prof_wd = _agg_profile(df_wd_p)

    print("[INFO] Loading + preparing 4G weekend...")
    df_we = load_prepared_df(str(p4g_weekend), require_sleep=False)
    print("[INFO] Inferring probabilities (weekend)...")
    df_we_p = _infer(df_we, model, feature_spec, out_col="p_sleep_on")
    prof_we = _agg_profile(df_we_p)

    out_wd = MODELS_DIR / "4g_profile_weekday.csv"
    out_we = MODELS_DIR / "4g_profile_weekend.csv"
    prof_wd.to_csv(out_wd, index=False)
    prof_we.to_csv(out_we, index=False)

    summary: Dict[str, Any] = {
        "model": str(model_path),
        "feature_spec": {
            "use_energy_features": bool(feature_spec.use_energy_features),
            "use_prev_features": bool(feature_spec.use_prev_features),
            "use_time_features": bool(feature_spec.use_time_features),
            "use_time_cyclical": bool(feature_spec.use_time_cyclical),
        },
        "weekday": {
            "rows": int(len(df_wd)),
            "bs": int(df_wd[resolve_col(df_wd, "bs")].nunique()),
            "cells": int(df_wd[resolve_col(df_wd, "cell")].nunique()),
            "p_sleep_on_mean": float(np.mean(df_wd_p["p_sleep_on"])),
        },
        "weekend": {
            "rows": int(len(df_we)),
            "bs": int(df_we[resolve_col(df_we, "bs")].nunique()),
            "cells": int(df_we[resolve_col(df_we, "cell")].nunique()),
            "p_sleep_on_mean": float(np.mean(df_we_p["p_sleep_on"])),
        },
        "outputs": {
            "weekday_profile_csv": str(out_wd),
            "weekend_profile_csv": str(out_we),
        },
    }

    out_json = MODELS_DIR / "4g_profile_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[RESULT] 4G inference profiles written")
    print(f"  Weekday CSV: {out_wd}")
    print(f"  Weekend CSV: {out_we}")
    print(f"  Summary JSON: {out_json}")
    print("\n[Quick check]")
    print("  Mean p_sleep_on weekday:", summary["weekday"]["p_sleep_on_mean"])
    print("  Mean p_sleep_on weekend:", summary["weekend"]["p_sleep_on_mean"])


if __name__ == "__main__":
    main()
