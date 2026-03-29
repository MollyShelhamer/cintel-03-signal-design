"""
signal_design_shelhamer_aqi.py - AQI Signal Pipeline

Author: Molly Shelhamer
Date: 2026-03

Dataset:
EPA Annual AQI by County (2025)

Purpose:
- Read AQI data from CSV
- Create meaningful air quality signals
- Save enhanced dataset for analysis
"""

# === IMPORTS ===
import logging
from pathlib import Path
from typing import Final

import polars as pl
from datafun_toolkit.logger import get_logger, log_header, log_path

# === LOGGER ===
LOG: logging.Logger = get_logger("AQI_PIPELINE", level="DEBUG")

# === PATHS ===
ROOT_DIR: Final[Path] = Path.cwd()
DATA_DIR: Final[Path] = ROOT_DIR / "data"
ARTIFACTS_DIR: Final[Path] = ROOT_DIR / "artifacts"

DATA_FILE: Final[Path] = DATA_DIR / "annual_aqi_by_county_2025.csv"
OUTPUT_FILE: Final[Path] = ARTIFACTS_DIR / "signals_shelhamer_aqi.csv"


def main() -> None:
    """Run AQI signal pipeline."""
    log_header(LOG, "AQI SIGNAL PIPELINE")
    LOG.info("START main()")
    log_path(LOG, "DATA_FILE", DATA_FILE)
    log_path(LOG, "OUTPUT_FILE", OUTPUT_FILE)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # STEP 1: READ CSV
    # -------------------------
    # Ensure proper UTF-8 encoding and strip whitespace from column names
    # STEP 1: READ CSV
    df: pl.DataFrame = pl.read_csv(DATA_FILE, separator="\t", encoding="utf8")
    df = df.rename({col: col.strip() for col in df.columns})
    LOG.info(f"Loaded {df.height} AQI records")
    LOG.info(f"Columns detected: {df.columns}")
    # -------------------------
    # STEP 2: DESIGN SIGNALS
    # -------------------------
    total_days = pl.col("Days with AQI")

    unhealthy_days = (
        pl.col("Unhealthy Days")
        + pl.col("Very Unhealthy Days")
        + pl.col("Hazardous Days")
    )

    unhealthy_ratio = (
        pl.when(total_days > 0)
        .then(unhealthy_days / total_days)
        .otherwise(0.0)
        .alias("unhealthy_ratio")
    )

    good_day_ratio = (
        pl.when(total_days > 0)
        .then(pl.col("Good Days") / total_days)
        .otherwise(0.0)
        .alias("good_day_ratio")
    )

    pollution_severity = (
        pl.col("Moderate Days") * 1
        + pl.col("Unhealthy for Sensitive Groups Days") * 2
        + pl.col("Unhealthy Days") * 3
        + pl.col("Very Unhealthy Days") * 4
        + pl.col("Hazardous Days") * 5
    ).alias("pollution_severity_index")

    aqi_spread = (pl.col("Max AQI") - pl.col("Median AQI")).alias("aqi_spread")

    pm_days = pl.col("Days PM2.5") + pl.col("Days PM10")
    pm_ratio = (
        pl.when(total_days > 0)
        .then(pm_days / total_days)
        .otherwise(0.0)
        .alias("pm_pollution_ratio")
    )

    # -------------------------
    # STEP 3: APPLY SIGNALS
    # -------------------------
    df_with_signals = df.with_columns(
        [
            unhealthy_ratio,
            good_day_ratio,
            pollution_severity,
            aqi_spread,
            pm_ratio,
        ]
    )

    LOG.info("Signal columns created successfully")

    # -------------------------
    # STEP 4: SELECT OUTPUT
    # -------------------------
    signals_df = df_with_signals.select(
        [
            "unhealthy_ratio",
            "good_day_ratio",
            "pollution_severity_index",
            "aqi_spread",
            "pm_pollution_ratio",
        ]
    )

    LOG.info(f"Final dataset has {signals_df.height} rows")

    # -------------------------
    # STEP 5: SAVE OUTPUT
    # -------------------------
    signals_df.write_csv(OUTPUT_FILE)
    LOG.info(f"Saved output to {OUTPUT_FILE}")
    LOG.info("Pipeline executed successfully!")
    LOG.info("END main()")


if __name__ == "__main__":
    main()
