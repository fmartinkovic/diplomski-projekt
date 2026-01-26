# src/io_core.py
from __future__ import annotations

import pandas as pd


def load_csv_core(path: str) -> pd.DataFrame:
    """
    Streamlit-free CSV loader.

    Mirrors src.io.load_csv behavior (encoding + column normalization)
    but has no Streamlit dependency, so it can be used in scripts/tests/CI.
    """
    # encoding="utf-8-sig" strips BOM if present
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Normalize column names:
    # - remove BOM if still present
    # - normalize non-breaking spaces
    # - strip leading/trailing whitespace
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )
    return df
