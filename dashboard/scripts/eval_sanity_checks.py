# scripts/eval_sanity_checks.py
from __future__ import annotations

import sys
from pathlib import Path
import json
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_prepared_df
from src.config import resolve_col
from src.ml.features_shared import FeatureSpec, make_X, make_y_sleep_on


def _eval_probs(y_true: np.ndarray, p1: np.ndarray) -> Dict[str, float]:
    p1 = np.asarray(p1, dtype=float)
    p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
    return {
        "auroc": float(roc_auc_score(y_true, p1)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, p1)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, p1)),
        "p_mean": float(np.mean(p1)),
        "p_std": float(np.std(p1)),
    }


def prb_only_score_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    PRB-only baseline using a monotonic score:
      score = -PRB (lower PRB => higher sleep probability)

    This produces AUROC/AUPRC without fitting any model.
    """
    prb_col = resolve_col(df, "prb")
    y = make_y_sleep_on(df).to_numpy().astype(int)
    prb = pd.to_numeric(df[prb_col], errors="coerce").to_numpy()
    mask = np.isfinite(prb)
    y = y[mask]
    prb = prb[mask]

    # Convert score -> "probability-like" via rank normalization (for Brier we need [0,1])
    # Ranking is stable and monotonic.
    order = np.argsort(prb)  # low prb first
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(1.0, 0.0, num=len(prb))  # low prb => near 1
    p = np.clip(ranks, 1e-6, 1.0 - 1e-6)

    return {
        "n_used": int(len(y)),
        "label_prevalence": float(np.mean(y)),
        "metrics": _eval_probs(y, p),
        "note": "score=-PRB with rank-based normalization to [0,1]",
    }


def best_prb_threshold(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple threshold baseline:
      predict sleep_on = 1 if PRB < t

    Choose t on TRAIN by maximizing F1, then evaluate on TEST.
    """
    prb_col = resolve_col(df_train, "prb")

    y_tr = make_y_sleep_on(df_train).to_numpy().astype(int)
    prb_tr = pd.to_numeric(df_train[prb_col], errors="coerce").to_numpy()
    tr_mask = np.isfinite(prb_tr)
    y_tr = y_tr[tr_mask]
    prb_tr = prb_tr[tr_mask]

    y_te = make_y_sleep_on(df_test).to_numpy().astype(int)
    prb_te = pd.to_numeric(df_test[prb_col], errors="coerce").to_numpy()
    te_mask = np.isfinite(prb_te)
    y_te = y_te[te_mask]
    prb_te = prb_te[te_mask]

    # Sweep thresholds on a grid (0..100 in 0.5 increments)
    thresholds = np.linspace(0.0, 100.0, 201)
    best = {"t": None, "f1": -1.0, "precision": 0.0, "recall": 0.0}

    for t in thresholds:
        pred = (prb_tr < t).astype(int)
        tp = int(((pred == 1) & (y_tr == 1)).sum())
        fp = int(((pred == 1) & (y_tr == 0)).sum())
        fn = int(((pred == 0) & (y_tr == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        if f1 > best["f1"]:
            best = {"t": float(t), "f1": float(f1), "precision": float(precision), "recall": float(recall)}

    # Evaluate on TEST with chosen threshold
    t = float(best["t"])
    pred_te = (prb_te < t).astype(int)
    tp = int(((pred_te == 1) & (y_te == 1)).sum())
    fp = int(((pred_te == 1) & (y_te == 0)).sum())
    fn = int(((pred_te == 0) & (y_te == 1)).sum())
    tn = int(((pred_te == 0) & (y_te == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "train_select": best,
        "test_at_t": {
            "t": t,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "test_prevalence": float(np.mean(y_te)),
        },
        "threshold_grid": {"min": 0.0, "max": 100.0, "n": int(len(thresholds)), "step": 0.5},
    }


def bs_holdout_train_eval(
    df: pd.DataFrame,
    *,
    feature_spec: FeatureSpec,
    test_frac: float = 0.2,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Harder split: hold out entire Base Stations (all their cells) as test.
    Train the same HGB + calibration pipeline, evaluate on held-out BS.
    """
    bs_col = resolve_col(df, "bs")
    groups = df[bs_col].astype(str).to_numpy()
    idx = np.arange(len(df))

    gss = GroupShuffleSplit(n_splits=1, test_size=float(test_frac), random_state=int(random_state))
    train_idx, test_idx = next(gss.split(idx, groups=groups))

    df_tr = df.iloc[train_idx]
    df_te = df.iloc[test_idx]

    X_tr = make_X(df_tr, spec=feature_spec, return_feature_names=False)
    y_tr = make_y_sleep_on(df_tr).to_numpy().astype(int)

    X_te = make_X(df_te, spec=feature_spec, return_feature_names=False)
    y_te = make_y_sleep_on(df_te).to_numpy().astype(int)

    base = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                random_state=int(random_state),
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
            )),
        ]
    )
    base.fit(X_tr, y_tr)

    calib = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    calib.fit(X_tr, y_tr)

    p_te = calib.predict_proba(X_te)[:, 1]
    metrics = _eval_probs(y_te, p_te)

    return {
        "split": {
            "strategy": "base_station_holdout",
            "test_frac": float(test_frac),
            "random_state": int(random_state),
            "train_rows": int(len(df_tr)),
            "test_rows": int(len(df_te)),
            "train_bs": int(df_tr[bs_col].nunique()),
            "test_bs": int(df_te[bs_col].nunique()),
            "train_cells": int(df_tr[resolve_col(df_tr, 'cell')].nunique()),
            "test_cells": int(df_te[resolve_col(df_te, 'cell')].nunique()),
        },
        "test_label_prevalence": float(np.mean(y_te)),
        "test_metrics": metrics,
    }


def duplicate_daily_profile_check(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks whether many cells share identical 48-bin PRB daily profiles,
    which can make holdout evaluation easier.

    Method:
      - For each (BS,Cell), sort by tod_bin and take PRB vector
      - Round PRB to 2 decimals (noise-tolerant)
      - Hash tuple, count duplicates
    """
    bs_col = resolve_col(df, "bs")
    cell_col = resolve_col(df, "cell")
    prb_col = resolve_col(df, "prb")

    if "tod_bin" not in df.columns:
        return {"ok": False, "reason": "tod_bin missing; run prepare_5g() first."}

    d = df[[bs_col, cell_col, "tod_bin", prb_col]].copy()
    d[prb_col] = pd.to_numeric(d[prb_col], errors="coerce")
    d = d.dropna(subset=[prb_col])

    # Group and build signature
    d = d.sort_values([bs_col, cell_col, "tod_bin"])
    key_cols = [bs_col, cell_col]

    sigs = []
    for _, g in d.groupby(key_cols, dropna=False):
        # Expect ~48 bins; but be robust
        v = g[prb_col].to_numpy()
        v = np.round(v.astype(float), 2)
        sigs.append(tuple(v.tolist()))

    if not sigs:
        return {"ok": False, "reason": "No signatures constructed."}

    # Count duplicates
    from collections import Counter
    c = Counter(sigs)
    counts = np.array(list(c.values()), dtype=int)

    n_cells = int(len(sigs))
    n_unique = int(len(c))
    max_dupe = int(counts.max())
    frac_unique = float(n_unique / n_cells)

    # How many cells belong to a signature that appears >=2 times?
    n_in_dupe_groups = int(sum(v for v in c.values() if v >= 2))
    frac_in_dupe_groups = float(n_in_dupe_groups / n_cells)

    return {
        "ok": True,
        "n_cells": n_cells,
        "n_unique_profiles": n_unique,
        "frac_unique_profiles": frac_unique,
        "max_duplicate_count_for_a_profile": max_dupe,
        "frac_cells_in_duplicate_profile_groups": frac_in_dupe_groups,
        "note": "PRB rounded to 2 decimals; signature length may vary if bins missing.",
    }


def calibration_deciles(model, df: pd.DataFrame, feature_spec: FeatureSpec) -> Dict[str, Any]:
    """
    Simple calibration table:
      - compute p = model.predict_proba(X)[:,1]
      - bin into deciles by predicted probability
      - report mean(p) and observed prevalence per bin
    """
    X = make_X(df, spec=feature_spec, return_feature_names=False)
    y = make_y_sleep_on(df).to_numpy().astype(int)
    p = model.predict_proba(X)[:, 1].astype(float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)

    # Deciles by quantiles
    qs = np.quantile(p, np.linspace(0, 1, 11))
    # Make bins strictly increasing to avoid issues when p has many ties
    qs = np.unique(qs)
    if len(qs) < 3:
        return {"ok": False, "reason": "Too many probability ties for deciles."}

    bins = pd.cut(p, bins=qs, include_lowest=True, duplicates="drop")
    tab = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin", dropna=False).agg(
        n=("y", "size"),
        p_mean=("p", "mean"),
        y_rate=("y", "mean"),
    ).reset_index()

    # Convert bin labels to strings for JSON
    tab["bin"] = tab["bin"].astype(str)

    return {
        "ok": True,
        "n_rows": int(len(df)),
        "table": tab.to_dict(orient="records"),
    }


def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    MODELS_DIR = ROOT / "models"

    data_path = DATA_DIR / "Performance_5G_Weekday.csv"
    model_path = MODELS_DIR / "sleep_on_5g_weekday.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"5G CSV not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path} (run train script first)")

    print("[INFO] Loading prepared 5G dataset...")
    df = load_prepared_df(str(data_path))

    print("[INFO] Loading trained model...")
    model = joblib.load(str(model_path))

    # Feature spec should match how you trained (set these explicitly)
    feature_spec = FeatureSpec(
        use_energy_features=False,   # you just trained with this ablation; set True if you retrain with energy
        use_prev_features=True,
        use_time_cyclical=True,
    )

    # ------------------------------------------------------------------
    # A) PRB-only score baseline (no fit)
    # ------------------------------------------------------------------
    print("[INFO] Running PRB-only score baseline...")
    prb_score = prb_only_score_baseline(df)

    # ------------------------------------------------------------------
    # B) Best PRB threshold baseline using the SAME split as your training report:
    #    cell-group holdout. We'll recreate a similar split here:
    #    group by (BS||Cell) and hold out 20% of cells.
    # ------------------------------------------------------------------
    print("[INFO] Running PRB-threshold baseline with cell-group split...")
    bs_col = resolve_col(df, "bs")
    cell_col = resolve_col(df, "cell")
    cell_groups = (df[bs_col].astype(str) + "||" + df[cell_col].astype(str)).to_numpy()
    idx = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(gss.split(idx, groups=cell_groups))
    df_tr = df.iloc[train_idx]
    df_te = df.iloc[test_idx]
    prb_thr = best_prb_threshold(df_tr, df_te)

    # ------------------------------------------------------------------
    # C) Base-station holdout evaluation (harder generalization)
    # ------------------------------------------------------------------
    print("[INFO] Running base-station holdout train/eval...")
    bs_holdout = bs_holdout_train_eval(df, feature_spec=feature_spec, test_frac=0.2, random_state=0)

    # ------------------------------------------------------------------
    # D) Duplicate daily profile check
    # ------------------------------------------------------------------
    print("[INFO] Running duplicate-profile check...")
    dup_check = duplicate_daily_profile_check(df)

    # ------------------------------------------------------------------
    # E) Calibration deciles on a held-out cell split (same as threshold split above)
    # ------------------------------------------------------------------
    print("[INFO] Running calibration deciles on held-out cells...")
    calib = calibration_deciles(model, df_te, feature_spec=feature_spec)

    # Also compute model metrics on the same df_te for easy comparison
    print("[INFO] Computing model metrics on held-out cells (same split as PRB threshold)...")
    X_te = make_X(df_te, spec=feature_spec, return_feature_names=False)
    y_te = make_y_sleep_on(df_te).to_numpy().astype(int)
    p_te = model.predict_proba(X_te)[:, 1]
    model_on_cell_holdout = {
        "split": {
            "strategy": "cell_group_holdout (recreated)",
            "test_frac": 0.2,
            "random_state": 0,
            "test_rows": int(len(df_te)),
            "test_cells": int(df_te[cell_col].nunique()),
            "test_bs": int(df_te[bs_col].nunique()),
        },
        "test_label_prevalence": float(np.mean(y_te)),
        "test_metrics": _eval_probs(y_te, p_te),
    }

    report: Dict[str, Any] = {
        "dataset": "Performance_5G_Weekday.csv",
        "n_rows": int(len(df)),
        "n_bs": int(df[bs_col].nunique()),
        "n_cells": int(df[cell_col].nunique()),
        "label_prevalence": float(make_y_sleep_on(df).mean()),
        "feature_spec_used_for_eval": {
            "use_energy_features": bool(feature_spec.use_energy_features),
            "use_prev_features": bool(feature_spec.use_prev_features),
            "use_time_cyclical": bool(feature_spec.use_time_cyclical),
        },
        "sanity_checks": {
            "prb_only_score_baseline": prb_score,
            "prb_threshold_baseline": prb_thr,
            "model_on_cell_holdout": model_on_cell_holdout,
            "base_station_holdout_eval": bs_holdout,
            "duplicate_daily_profile_check": dup_check,
            "calibration_deciles_on_cell_holdout_test": calib,
        },
    }

    out_path = MODELS_DIR / "eval_sanity_5g_weekday.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[RESULT] Sanity evaluation complete")
    print(f"  Report: {out_path}")
    print("\n[Key metrics]")
    print("  PRB-only AUROC:", prb_score["metrics"]["auroc"])
    print("  PRB-only AUPRC:", prb_score["metrics"]["auprc"])
    print("  Model(cell-holdout) AUROC:", model_on_cell_holdout["test_metrics"]["auroc"])
    print("  Model(cell-holdout) AUPRC:", model_on_cell_holdout["test_metrics"]["auprc"])
    print("  Model(BS-holdout) AUROC:", bs_holdout["test_metrics"]["auroc"])
    print("  Model(BS-holdout) AUPRC:", bs_holdout["test_metrics"]["auprc"])


if __name__ == "__main__":
    main()
