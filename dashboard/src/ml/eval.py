# src/ml/eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

from src.ml.features_shared import FeatureSpec, make_X, make_y_sleep_on


@dataclass(frozen=True)
class EvalSpec:
    """
    Evaluation configuration.
    """
    features: FeatureSpec = FeatureSpec(
        use_energy_features=True,
        use_prev_features=True,
        use_time_cyclical=True,
    )
    threshold: float = 0.5  # for converting probabilities to a hard decision (optional)


def evaluate_sleep_on_model(
    model: Any,
    df_5g_prepared: pd.DataFrame,
    spec: EvalSpec = EvalSpec(),
) -> Dict[str, Any]:
    """
    Evaluate a fitted model on a provided 5G prepared dataframe.

    Returns a metrics dict; caller controls which subset is passed (train/val/test).
    """
    X, feat_names = make_X(df_5g_prepared, spec=spec.features, return_feature_names=True)
    y = make_y_sleep_on(df_5g_prepared).to_numpy().astype(int)

    p = model.predict_proba(X)[:, 1]
    p = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)

    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    auprc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(brier_score_loss(y, p))

    yhat = (p >= float(spec.threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()

    out: Dict[str, Any] = {
        "n_rows": int(len(df_5g_prepared)),
        "label_prevalence": float(np.mean(y)),
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "threshold": float(spec.threshold),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "features_used": feat_names,
        "prob_summary": {"mean": float(np.mean(p)), "std": float(np.std(p))},
    }
    return out


def infer_sleep_on_probability(
    model: Any,
    df_prepared: pd.DataFrame,
    *,
    features: FeatureSpec = FeatureSpec(),
    out_col: str = "p_sleep_on",
) -> pd.DataFrame:
    """
    Add an inference column p_sleep_on to any prepared dataframe (5G or 4G).

    This is the deployable artifact for your controller stage:
      p_sleep_on ≈ confidence of low-risk regime

    Returns df copy with new column.
    """
    X = make_X(df_prepared, spec=features, return_feature_names=False)
    p = model.predict_proba(X)[:, 1]
    df_out = df_prepared.copy()
    df_out[out_col] = np.clip(p.astype(float), 0.0, 1.0)
    return df_out
