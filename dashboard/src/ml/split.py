# src/ml/split.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

from src.config import resolve_col


@dataclass(frozen=True)
class SplitSpec:
    """
    Leakage-safe split spec for your setting.

    Strategy:
      1) Prefer time-based split: hold out the last K DayIndex values per (BS, Cell).
         This is ideal when you truly have multiple days per cell.
      2) If the dataset is effectively "single-day-per-cell" (DayIndex max=0 for most groups),
         time holdout degenerates (train=0). In that case, fall back to a GROUP holdout:
         hold out a fraction of CELLS as test (GroupShuffleSplit by Cell ID).

    This keeps:
      - no within-cell leakage across train/test
      - a non-degenerate split in single-day datasets
    """
    test_days: int = 1
    fallback_test_frac: float = 0.2
    random_state: int = 0


def _cell_group_labels(df: pd.DataFrame) -> pd.Series:
    """
    Build a stable grouping label for splitting by cell.
    """
    bs_col = resolve_col(df, "bs")
    cell_col = resolve_col(df, "cell")
    # Combine BS+Cell to avoid collisions if Cell ID is not globally unique
    return df[bs_col].astype(str) + "||" + df[cell_col].astype(str)


def last_days_holdout_indices(df: pd.DataFrame, spec: SplitSpec = SplitSpec()) -> Tuple[np.ndarray, np.ndarray]:
    """
    Primary split: within each cell, hold out last K DayIndex values as test.

    Returns (train_idx, test_idx).
    Raises ValueError if the split degenerates.
    """
    if "DayIndex" not in df.columns:
        raise ValueError("last_days_holdout_indices: expected 'DayIndex' in df. Run prepare_5g() first.")

    groups = _cell_group_labels(df)
    d = pd.DataFrame({"group": groups, "DayIndex": df["DayIndex"]}).copy()
    d["DayIndex"] = pd.to_numeric(d["DayIndex"], errors="coerce")
    if d["DayIndex"].isna().any():
        raise ValueError("last_days_holdout_indices: DayIndex contains NaN after coercion.")

    # For each cell, define test set as last K DayIndex values
    max_day = d.groupby("group", dropna=False)["DayIndex"].transform("max")
    test_mask = d["DayIndex"] >= (max_day - (spec.test_days - 1))

    train_idx = np.flatnonzero(~test_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())

    if len(test_idx) == 0 or len(train_idx) == 0:
        raise ValueError(
            f"last_days_holdout_indices: degenerate split. "
            f"train={len(train_idx)}, test={len(test_idx)}. "
            f"Likely single-day-per-cell data (DayIndex max=0)."
        )

    return train_idx, test_idx


def cell_group_holdout_indices(df: pd.DataFrame, spec: SplitSpec = SplitSpec()) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback split: hold out a fraction of CELLS (groups) as test.

    This is leakage-safe in your setting because no (BS,Cell) appears in both sets.
    """
    groups = _cell_group_labels(df).to_numpy()
    idx = np.arange(len(df))

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=float(spec.fallback_test_frac),
        random_state=int(spec.random_state),
    )
    train_idx, test_idx = next(gss.split(idx, groups=groups))
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    if len(test_idx) == 0 or len(train_idx) == 0:
        raise ValueError(
            f"cell_group_holdout_indices: degenerate split. train={len(train_idx)}, test={len(test_idx)}."
        )

    return train_idx, test_idx


def make_train_test_indices(df: pd.DataFrame, spec: SplitSpec = SplitSpec()) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Adaptive split:
      - Try last-days holdout
      - If it degenerates, fall back to group holdout by cell
    """
    strategy = "last_days_per_cell"
    try:
        train_idx, test_idx = last_days_holdout_indices(df, spec=spec)
    except ValueError:
        strategy = "cell_group_holdout"
        train_idx, test_idx = cell_group_holdout_indices(df, spec=spec)

    meta = {"strategy": strategy}
    return train_idx, test_idx, meta


def describe_split(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, meta: Dict[str, Any] | None = None) -> dict:
    """
    Log split characteristics (thesis reproducibility).
    """
    bs_col = resolve_col(df, "bs")
    cell_col = resolve_col(df, "cell")

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    out = {
        "strategy": (meta or {}).get("strategy", "<unknown>"),
        "n_rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_bs": int(train[bs_col].nunique()),
        "test_bs": int(test[bs_col].nunique()),
        "train_cells": int(train[cell_col].nunique()),
        "test_cells": int(test[cell_col].nunique()),
    }

    if "DayIndex" in df.columns:
        out["train_day_minmax"] = (int(train["DayIndex"].min()), int(train["DayIndex"].max()))
        out["test_day_minmax"] = (int(test["DayIndex"].min()), int(test["DayIndex"].max()))

    return out
