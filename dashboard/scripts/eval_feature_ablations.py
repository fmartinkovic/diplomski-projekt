# scripts/eval_feature_ablations.py
from __future__ import annotations

import sys
from pathlib import Path
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

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


def train_eval_on_bs_holdout(
    df: pd.DataFrame,
    feature_spec: FeatureSpec,
    *,
    test_frac: float = 0.2,
    random_state: int = 0,
) -> Dict[str, Any]:
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
        },
        "test_label_prevalence": float(np.mean(y_te)),
        "test_metrics": metrics,
    }


def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    MODELS_DIR = ROOT / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_path = DATA_DIR / "Performance_5G_Weekday.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"5G CSV not found: {data_path}")

    print("[INFO] Loading prepared 5G dataset...")
    df = load_prepared_df(str(data_path), require_sleep=True)

    bs_col = resolve_col(df, "bs")
    cell_col = resolve_col(df, "cell")
    prevalence = float(make_y_sleep_on(df).mean())

    # Ablation configurations
    variants: List[tuple[str, FeatureSpec]] = [
        ("full", FeatureSpec(use_energy_features=True, use_prev_features=True, use_time_features=True, use_time_cyclical=True)),
        ("no_energy", FeatureSpec(use_energy_features=False, use_prev_features=True, use_time_features=True, use_time_cyclical=True)),
        ("no_time", FeatureSpec(use_energy_features=False, use_prev_features=True, use_time_features=False, use_time_cyclical=False)),
        ("no_prev", FeatureSpec(use_energy_features=False, use_prev_features=False, use_time_features=True, use_time_cyclical=True)),
        ("load_only", FeatureSpec(use_energy_features=False, use_prev_features=False, use_time_features=False, use_time_cyclical=False)),
    ]

    print("[INFO] Running ablations (BS holdout)...")
    results: Dict[str, Any] = {}
    rows = []

    for name, spec in variants:
        print(f"  - {name}")
        out = train_eval_on_bs_holdout(df, feature_spec=spec, test_frac=0.2, random_state=0)
        results[name] = {
            "feature_spec": {
                "use_energy_features": bool(spec.use_energy_features),
                "use_prev_features": bool(spec.use_prev_features),
                "use_time_features": bool(spec.use_time_features),
                "use_time_cyclical": bool(spec.use_time_cyclical),
            },
            **out,
        }

        m = out["test_metrics"]
        rows.append({
            "variant": name,
            "auroc": m["auroc"],
            "auprc": m["auprc"],
            "brier": m["brier"],
            "p_mean": m["p_mean"],
            "p_std": m["p_std"],
            "test_label_prevalence": out["test_label_prevalence"],
        })

    report = {
        "dataset": "Performance_5G_Weekday.csv",
        "n_rows": int(len(df)),
        "n_bs": int(df[bs_col].nunique()),
        "n_cells": int(df[cell_col].nunique()),
        "label_prevalence": prevalence,
        "ablations": results,
        "note": "All evaluations use BS-holdout split + HGB + isotonic calibration.",
    }

    out_json = MODELS_DIR / "feature_ablations_5g_weekday.json"
    out_csv = MODELS_DIR / "feature_ablations_5g_weekday.csv"

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).sort_values("auprc", ascending=False).to_csv(out_csv, index=False)

    print("\n[RESULT] Ablation evaluation complete")
    print(f"  JSON: {out_json}")
    print(f"  CSV : {out_csv}")
    print("\n[Top-line]")
    for r in sorted(rows, key=lambda x: x["auprc"], reverse=True):
        print(f"  {r['variant']:<10}  AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  Brier={r['brier']:.4f}")


if __name__ == "__main__":
    main()
