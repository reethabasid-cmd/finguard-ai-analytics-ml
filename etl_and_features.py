"""Data cleaning and feature engineering for FinGuard AI."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

RAW_COLUMNS = [
    "transaction_id", "timestamp", "customer_id", "merchant_category",
    "region", "payment_method", "amount", "account_age_days",
    "tx_frequency_24h", "device_risk_score",
]
CATEGORICAL_COLUMNS = ["merchant_category", "region", "payment_method"]
NUMERIC_COLUMNS = [
    "amount", "account_age_days", "tx_frequency_24h", "device_risk_score",
    "amount_to_avg_ratio", "is_high_velocity", "risk_weighted_amount",
    "hour_of_day",
]
FEATURE_COLUMNS = [
    "merchant_category", "region", "payment_method", *NUMERIC_COLUMNS,
]


def clean_transactions(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw records and return a JSON-friendly data-quality report."""
    if raw is None or raw.empty:
        raise ValueError("The input dataset is empty.")

    frame = raw.copy()
    rows_ingested = len(frame)
    missing_columns = [column for column in RAW_COLUMNS if column not in frame.columns]
    for column in missing_columns:
        frame[column] = np.nan

    # Stable synthetic values make incomplete uploads usable while preserving
    # a visible record of which source columns were absent.
    frame["transaction_id"] = frame["transaction_id"].fillna(
        pd.Series([f"UPLOAD-{i:08d}" for i in range(len(frame))], index=frame.index)
    )
    frame["customer_id"] = frame["customer_id"].fillna("UNKNOWN")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    valid_times = frame["timestamp"].dropna()
    fallback_time = valid_times.median() if not valid_times.empty else pd.Timestamp.now(tz="UTC")
    frame["timestamp"] = frame["timestamp"].fillna(fallback_time)

    before_dedup = len(frame)
    frame = frame.drop_duplicates().copy()
    duplicates_removed = before_dedup - len(frame)

    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    negative_amounts = int((frame["amount"] < 0).sum())
    frame = frame.loc[frame["amount"].isna() | (frame["amount"] >= 0)].copy()

    numeric_defaults = {
        "amount": float(frame["amount"].median()) if frame["amount"].notna().any() else 50.0,
        "account_age_days": 365.0,
        "tx_frequency_24h": 1.0,
        "device_risk_score": 0.10,
    }
    numeric_missing = 0
    for column, default in numeric_defaults.items():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        numeric_missing += int(frame[column].isna().sum())
        median = frame[column].median()
        frame[column] = frame[column].fillna(median if pd.notna(median) else default)

    frame["account_age_days"] = frame["account_age_days"].clip(lower=0)
    frame["tx_frequency_24h"] = frame["tx_frequency_24h"].clip(lower=0)
    frame["device_risk_score"] = frame["device_risk_score"].clip(0, 1)
    frame["amount"] = frame["amount"].clip(lower=0)

    categorical_missing = 0
    for column in CATEGORICAL_COLUMNS:
        categorical_missing += int(frame[column].isna().sum())
        frame[column] = (
            frame[column].fillna("unknown").astype(str).str.strip()
            .replace("", "unknown")
        )

    if "is_fraud" in frame.columns:
        frame["is_fraud"] = pd.to_numeric(frame["is_fraud"], errors="coerce")
        # A missing or malformed target cannot be used for supervised training.
        frame = frame.loc[frame["is_fraud"].isin([0, 1])].copy()
        frame["is_fraud"] = frame["is_fraud"].astype(int)

    frame = frame.reset_index(drop=True)
    rows_cleaned = len(frame)
    report = {
        "rows_ingested": int(rows_ingested),
        "rows_after_deduplication": int(before_dedup - duplicates_removed),
        "rows_cleaned": int(rows_cleaned),
        "duplicates_removed": int(duplicates_removed),
        "negative_amounts_removed": int(negative_amounts),
        "numeric_values_imputed": int(numeric_missing),
        "categorical_values_imputed": int(categorical_missing),
        "missing_source_columns": missing_columns,
        "anomalies_fixed": int(duplicates_removed + negative_amounts + numeric_missing + categorical_missing),
        "cleaned_record_pct": round(100 * rows_cleaned / max(rows_ingested, 1), 2),
    }
    LOGGER.info("ETL data quality report: %s", report)
    return frame, report


def engineer_features(
    clean: pd.DataFrame,
    baseline_amount: float | None = None,
) -> pd.DataFrame:
    """Add transaction behavior features without requiring training state."""
    frame = clean.copy()
    global_baseline = float(baseline_amount or frame["amount"].median() or 1.0)
    global_baseline = max(global_baseline, 1.0)

    if "customer_id" in frame and frame["customer_id"].nunique() > 1:
        customer_medians = frame.groupby("customer_id")["amount"].transform("median")
        reference = customer_medians.where(customer_medians > 0, global_baseline)
    else:
        reference = global_baseline

    frame["amount_to_avg_ratio"] = (
        frame["amount"] / np.maximum(reference, 1.0)
    ).clip(0, 100)
    frame["is_high_velocity"] = (frame["tx_frequency_24h"] >= 5).astype(int)
    frame["risk_weighted_amount"] = (
        frame["amount"] * (1.0 + frame["device_risk_score"])
    )
    frame["hour_of_day"] = frame["timestamp"].dt.hour.astype(int)
    return frame