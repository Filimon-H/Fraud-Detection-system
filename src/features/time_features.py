"""
Time-based feature engineering for fraud detection.

This module provides functions to extract temporal features from
transaction timestamps, which are often strong fraud indicators.
"""

import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the transaction DataFrame.

    Creates the following features:
    - hour_of_day: Hour when purchase was made (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - is_weekend: Binary flag for weekend transactions
    - time_since_signup: Seconds between signup and purchase

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with 'signup_time' and 'purchase_time' columns
        already parsed as datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame with added time feature columns.

    Notes
    -----
    Fraudulent transactions often occur at unusual hours or
    very quickly after account signup. These features help
    capture such patterns.
    """
    df_out = df.copy()

    # Extract hour of day from purchase time
    df_out["hour_of_day"] = df_out["purchase_time"].dt.hour

    # Extract day of week (0 = Monday, 6 = Sunday)
    df_out["day_of_week"] = df_out["purchase_time"].dt.dayofweek

    # Weekend flag (Saturday = 5, Sunday = 6)
    df_out["is_weekend"] = df_out["day_of_week"].isin([5, 6]).astype(int)

    # Time since signup in seconds
    time_diff = df_out["purchase_time"] - df_out["signup_time"]
    df_out["time_since_signup"] = time_diff.dt.total_seconds()

    # Handle negative values (purchase before signup - likely data error)
    negative_mask = df_out["time_since_signup"] < 0
    if negative_mask.any():
        # Set to NaN for investigation, or could set to 0
        df_out.loc[negative_mask, "time_since_signup"] = np.nan

    return df_out


def add_signup_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features extracted from signup timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with 'signup_time' column as datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame with added signup time features.
    """
    df_out = df.copy()

    df_out["signup_hour"] = df_out["signup_time"].dt.hour
    df_out["signup_day_of_week"] = df_out["signup_time"].dt.dayofweek
    df_out["signup_month"] = df_out["signup_time"].dt.month

    return df_out


def categorize_hour(hour: int) -> str:
    """
    Categorize hour into time-of-day periods.

    Parameters
    ----------
    hour : int
        Hour of day (0-23).

    Returns
    -------
    str
        One of: 'night' (0-5), 'morning' (6-11), 
        'afternoon' (12-17), 'evening' (18-23).
    """
    if 0 <= hour < 6:
        return "night"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"


def add_time_period_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add categorical time period feature based on purchase hour.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with 'hour_of_day' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'purchase_time_period' column.
    """
    df_out = df.copy()

    if "hour_of_day" not in df_out.columns:
        df_out["hour_of_day"] = df_out["purchase_time"].dt.hour

    df_out["purchase_time_period"] = df_out["hour_of_day"].apply(categorize_hour)

    return df_out
