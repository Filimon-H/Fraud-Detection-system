"""
Velocity and frequency-based feature engineering.

This module provides functions to compute transaction velocity
and frequency features, which are strong fraud indicators.
High-frequency transactions from a single user or device
often signal fraudulent activity.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_velocity_features(
    df: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "purchase_time",
    windows_hours: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Add transaction velocity features per user.

    Computes the number of transactions per user within specified
    time windows (rolling counts).

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with user ID and timestamp columns.
    user_col : str, optional
        Name of user identifier column. Default is "user_id".
    time_col : str, optional
        Name of timestamp column. Default is "purchase_time".
    windows_hours : List[int], optional
        Time windows in hours to compute velocity over.
        Default is [1, 24] (1 hour and 24 hours).

    Returns
    -------
    pd.DataFrame
        DataFrame with added velocity feature columns:
        - tx_count_{user_col}_{window}h: transactions in last {window} hours

    Notes
    -----
    Fraud patterns often involve rapid-fire transactions.
    These features help detect velocity abuse.
    """
    if windows_hours is None:
        windows_hours = [1, 24]

    df_out = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df_out[time_col]):
        df_out[time_col] = pd.to_datetime(df_out[time_col])

    # Sort by user and time for rolling computation
    df_out = df_out.sort_values([user_col, time_col]).reset_index(drop=True)

    for window in windows_hours:
        col_name = f"tx_count_{user_col}_{window}h"
        window_seconds = window * 3600

        # Compute count of transactions in the window for each user
        # Using a groupby + rolling approach
        counts = []
        for _, group in df_out.groupby(user_col):
            group_counts = _count_in_window(
                group[time_col].values,
                window_seconds
            )
            counts.extend(group_counts)

        df_out[col_name] = counts

    return df_out


def _count_in_window(timestamps: np.ndarray, window_seconds: int) -> List[int]:
    """
    Count transactions within a time window for a sorted array of timestamps.

    For each transaction, count how many previous transactions
    (including itself) fall within the window.

    Parameters
    ----------
    timestamps : np.ndarray
        Sorted array of timestamps (datetime64).
    window_seconds : int
        Window size in seconds.

    Returns
    -------
    List[int]
        Count of transactions in window for each position.
    """
    n = len(timestamps)
    counts = []

    # Convert to numeric for faster comparison
    ts_numeric = timestamps.astype("datetime64[s]").astype(np.int64)

    for i in range(n):
        current_ts = ts_numeric[i]
        window_start = current_ts - window_seconds

        # Count transactions in [window_start, current_ts]
        count = np.sum((ts_numeric[:i+1] >= window_start))
        counts.append(int(count))

    return counts


def add_device_velocity_features(
    df: pd.DataFrame,
    device_col: str = "device_id",
    time_col: str = "purchase_time",
    windows_hours: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Add transaction velocity features per device.

    Similar to add_velocity_features but groups by device ID.
    Useful for detecting device-based fraud patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with device ID and timestamp columns.
    device_col : str, optional
        Name of device identifier column. Default is "device_id".
    time_col : str, optional
        Name of timestamp column. Default is "purchase_time".
    windows_hours : List[int], optional
        Time windows in hours. Default is [1, 24].

    Returns
    -------
    pd.DataFrame
        DataFrame with added device velocity feature columns.
    """
    if windows_hours is None:
        windows_hours = [1, 24]

    df_out = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df_out[time_col]):
        df_out[time_col] = pd.to_datetime(df_out[time_col])

    df_out = df_out.sort_values([device_col, time_col]).reset_index(drop=True)

    for window in windows_hours:
        col_name = f"tx_count_{device_col}_{window}h"
        window_seconds = window * 3600

        counts = []
        for _, group in df_out.groupby(device_col):
            group_counts = _count_in_window(
                group[time_col].values,
                window_seconds
            )
            counts.extend(group_counts)

        df_out[col_name] = counts

    return df_out


def add_user_transaction_count(df: pd.DataFrame, user_col: str = "user_id") -> pd.DataFrame:
    """
    Add total transaction count per user across entire dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data.
    user_col : str, optional
        User identifier column. Default is "user_id".

    Returns
    -------
    pd.DataFrame
        DataFrame with 'user_total_transactions' column.
    """
    df_out = df.copy()

    user_counts = df_out.groupby(user_col).size().reset_index(name="user_total_transactions")
    df_out = df_out.merge(user_counts, on=user_col, how="left")

    return df_out
