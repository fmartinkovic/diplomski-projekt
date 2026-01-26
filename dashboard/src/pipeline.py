# src/pipeline.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.io_core import load_csv_core
from src.config import canonicalize_columns, validate_schema
from src.features import prepare_5g


def load_raw_df(source_path: str) -> pd.DataFrame:
    """
    Pure (non-Streamlit) loader for raw datasets.

    - CSV: read + canonicalize column names
    - Parquet: read directly
    """
    p = Path(source_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {source_path}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        df = canonicalize_columns(df)
    else:
        df = load_csv_core(str(p))
        df = canonicalize_columns(df)

    validate_schema(
        df,
        required_keys=["ts", "bs", "cell", "prb", "traffic_kb", "rru_w"],
        where="pipeline:load_raw_df",
    )
    return df


def load_prepared_df(source_path: str, *, require_sleep: bool = True) -> pd.DataFrame:
    """
    Pure (non-Streamlit) loader for prepared datasets.

    require_sleep=True  -> for 5G training/eval (expects sleep_on/sleep_frac)
    require_sleep=False -> for 4G inference (no sleep columns)
    """
    raw = load_raw_df(source_path)
    df = prepare_5g(raw)

    required_cols = ["tod_bin", "DayIndex"]
    if require_sleep:
        required_cols += ["sleep_on", "sleep_frac"]

    validate_schema(
        df,
        required_keys=["ts", "bs", "cell", "prb", "traffic_kb", "rru_w"],
        required_cols=required_cols,
        where="pipeline:load_prepared_df",
    )
    return df
