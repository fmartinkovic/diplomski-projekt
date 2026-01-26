# src/ml/train.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

import joblib

from src.ml.features_shared import FeatureSpec, make_X, make_y_sleep_on
from src.ml.split import SplitSpec, make_train_test_indices, describe_split


@dataclass(frozen=True)
class TrainSpec:
    """
    Training configuration.
    """
    model_type: str = "hgb"  # "logreg" or "hgb"
    calibrate: bool = True
    calibration_method: str = "isotonic"  # "sigmoid" or "isotonic"
    random_state: int = 0
    split: SplitSpec = SplitSpec(test_days=1, fallback_test_frac=0.2, random_state=0)
    features: FeatureSpec = FeatureSpec(
        use_energy_features=True,
        use_prev_features=True,
        use_time_cyclical=True,
    )


def _build_base_model(spec: TrainSpec) -> Any:
    """
    Returns an sklearn estimator that supports predict_proba.
    """
    if spec.model_type == "logreg":
        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=500, solver="lbfgs", random_state=spec.random_state)),
            ]
        )

    if spec.model_type == "hgb":
        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("clf", HistGradientBoostingClassifier(
                    random_state=spec.random_state,
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=300,
                )),
            ]
        )

    raise ValueError(f"Unknown model_type: {spec.model_type!r}. Use 'logreg' or 'hgb'.")


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


def train_sleep_on_model(
    df_5g_prepared: pd.DataFrame,
    spec: TrainSpec = TrainSpec(),
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a low-risk estimator on 5G:
      y = sleep_on

    Uses an adaptive, leakage-safe split:
      - Prefer last-days-per-cell
      - Fall back to group holdout by cell when DayIndex is effectively single-day
    """
    X, feature_names = make_X(df_5g_prepared, spec=spec.features, return_feature_names=True)
    y = make_y_sleep_on(df_5g_prepared).to_numpy().astype(int)

    # Adaptive split
    train_idx, test_idx, split_meta = make_train_test_indices(df_5g_prepared, spec=spec.split)

    X_train = X.iloc[train_idx]
    y_train = y[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y[test_idx]

    base = _build_base_model(spec)
    base.fit(X_train, y_train)

    model = base
    if spec.calibrate:
        calib = CalibratedClassifierCV(
            estimator=base,
            method=spec.calibration_method,
            cv=3,
        )
        calib.fit(X_train, y_train)
        model = calib

    p_test = model.predict_proba(X_test)[:, 1]
    metrics = _eval_probs(y_test, p_test)

    report: Dict[str, Any] = {
        "train_spec": {
            "model_type": spec.model_type,
            "calibrate": bool(spec.calibrate),
            "calibration_method": spec.calibration_method,
            "random_state": int(spec.random_state),
            "split": {
                "test_days": int(spec.split.test_days),
                "fallback_test_frac": float(spec.split.fallback_test_frac),
                "random_state": int(spec.split.random_state),
            },
            "features": {
                "use_energy_features": bool(spec.features.use_energy_features),
                "use_prev_features": bool(spec.features.use_prev_features),
                "use_time_cyclical": bool(spec.features.use_time_cyclical),
            },
        },
        "split": describe_split(df_5g_prepared, train_idx=train_idx, test_idx=test_idx, meta=split_meta),
        "features": feature_names,
        "test_metrics": metrics,
        "test_label_prevalence": float(np.mean(y_test)),
    }
    return model, report


def save_model_bundle(
    model: Any,
    report: Dict[str, Any],
    out_dir: str | Path,
    *,
    name: str = "sleep_on_model",
) -> Dict[str, str]:
    """
    Save model + report into a directory.

    Outputs:
      - {name}.joblib
      - {name}.report.json
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / f"{name}.joblib"
    report_path = out / f"{name}.report.json"

    joblib.dump(model, model_path)

    import json

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)

    report_path.write_text(json.dumps(report, indent=2, default=_default), encoding="utf-8")
    return {"model_path": str(model_path), "report_path": str(report_path)}
