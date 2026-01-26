# scripts/train_sleep_on_5g.py
from __future__ import annotations

import sys
from pathlib import Path
import json

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_prepared_df
from src.ml.train import train_sleep_on_model, save_model_bundle
from src.ml.train import TrainSpec
from src.ml.split import SplitSpec
from src.ml.features_shared import FeatureSpec


def main() -> None:
    """
    Offline training script for the 5G low-risk (sleep_on) estimator.

    Run from project root:
        python3 scripts/train_sleep_on_5g.py
    """

    # ------------------------------------------------------------------
    # 1) Paths
    # ------------------------------------------------------------------
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    MODELS_DIR = ROOT / "models"

    data_path = DATA_DIR / "Performance_5G_Weekday.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"5G CSV not found: {data_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Load prepared 5G dataframe (NO streamlit dependency)
    # ------------------------------------------------------------------
    print("[INFO] Loading prepared 5G dataset...")
    df_5g = load_prepared_df(str(data_path))

    # These column names are canonicalized by your pipeline
    print(f"[INFO] Loaded rows: {len(df_5g):,}")
    print(f"[INFO] Cells: {df_5g['Cell ID'].nunique():,}")
    print(f"[INFO] Base stations: {df_5g['Base Station ID'].nunique():,}")

    # ------------------------------------------------------------------
    # 3) Training configuration
    # ------------------------------------------------------------------
    train_spec = TrainSpec(
        model_type="hgb",
        calibrate=True,
        calibration_method="isotonic",
        random_state=0,
        split=SplitSpec(test_days=1),
        features=FeatureSpec(
            use_energy_features=False,
            use_prev_features=True,
            use_time_cyclical=True,
        ),
    )

    # ------------------------------------------------------------------
    # 4) Train model
    # ------------------------------------------------------------------
    print("[INFO] Training model...")
    model, report = train_sleep_on_model(df_5g, spec=train_spec)

    # ------------------------------------------------------------------
    # 5) Persist model + report
    # ------------------------------------------------------------------
    bundle_name = "sleep_on_5g_weekday"
    paths = save_model_bundle(model, report, out_dir=MODELS_DIR, name=bundle_name)

    # ------------------------------------------------------------------
    # 6) Print summary
    # ------------------------------------------------------------------
    print("\n[RESULT] Training complete")
    print(f"  Model path : {paths['model_path']}")
    print(f"  Report path: {paths['report_path']}")
    print("\n[Test metrics]")
    for k, v in report["test_metrics"].items():
        print(f"  {k:>8}: {v:.4f}")

    print("\n[Split]")
    print(json.dumps(report["split"], indent=2))


if __name__ == "__main__":
    main()
