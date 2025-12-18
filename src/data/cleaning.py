"""
Data cleaning utilities for fraud detection.

This module provides functions to clean and preprocess raw data,
including datetime parsing, missing value handling, and duplicate removal.
"""

from typing import Dict, Tuple

import pandas as pd
import numpy as np


def clean_fraud_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean the e-commerce fraud dataset.

    Performs the following operations:
    1. Parse signup_time and purchase_time as datetime
    2. Handle missing values (report and handle appropriately)
    3. Remove exact duplicate rows
    4. Validate and fix data types

    Parameters
    ----------
    df : pd.DataFrame
        Raw fraud data from load_fraud_data().

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        - Cleaned DataFrame
        - Cleaning report dictionary with statistics about changes made

    Notes
    -----
    The cleaning report includes:
    - original_rows: number of rows before cleaning
    - duplicates_removed: number of duplicate rows removed
    - missing_values: dict of column -> count of missing values
    - final_rows: number of rows after cleaning
    """
    report = {
        "original_rows": len(df),
        "duplicates_removed": 0,
        "missing_values": {},
        "final_rows": 0,
    }

    # Create a copy to avoid modifying original
    df_clean = df.copy()

    # 1. Check and report missing values before handling
    missing_before = df_clean.isnull().sum()
    report["missing_values"] = missing_before[missing_before > 0].to_dict()

    # 2. Parse datetime columns
    df_clean["signup_time"] = pd.to_datetime(df_clean["signup_time"], errors="coerce")
    df_clean["purchase_time"] = pd.to_datetime(df_clean["purchase_time"], errors="coerce")

    # 3. Handle missing values
    # For categorical columns: fill with 'unknown'
    categorical_cols = ["source", "browser", "sex"]
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("unknown")

    # For numeric columns: we keep NaN for now and report
    # (age, purchase_value, ip_address should not have missing in clean data)

    # 4. Remove exact duplicates
    rows_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    report["duplicates_removed"] = rows_before_dedup - len(df_clean)

    # 5. Validate data types
    df_clean["user_id"] = df_clean["user_id"].astype(int)
    df_clean["age"] = pd.to_numeric(df_clean["age"], errors="coerce")
    df_clean["purchase_value"] = pd.to_numeric(df_clean["purchase_value"], errors="coerce")
    df_clean["class"] = df_clean["class"].astype(int)

    # 6. Basic validation: remove rows with invalid timestamps
    invalid_time_mask = (
        df_clean["signup_time"].isna() | 
        df_clean["purchase_time"].isna()
    )
    if invalid_time_mask.any():
        report["invalid_timestamps_removed"] = invalid_time_mask.sum()
        df_clean = df_clean[~invalid_time_mask]

    report["final_rows"] = len(df_clean)

    return df_clean, report


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        Summary with columns: column, missing_count, missing_percent
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    summary = pd.DataFrame({
        "column": missing_count.index,
        "missing_count": missing_count.values,
        "missing_percent": missing_percent.values
    })

    # Only show columns with missing values, sorted by count
    summary = summary[summary["missing_count"] > 0]
    summary = summary.sort_values("missing_count", ascending=False)

    return summary.reset_index(drop=True)


def get_duplicate_summary(df: pd.DataFrame) -> Dict:
    """
    Analyze duplicates in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    Dict
        Summary with total_rows, duplicate_rows, unique_rows, duplicate_percent
    """
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates())
    duplicate_rows = total_rows - unique_rows

    return {
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "unique_rows": unique_rows,
        "duplicate_percent": (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
    }
