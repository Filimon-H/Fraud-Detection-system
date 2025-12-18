"""
Geolocation utilities for IP address to country mapping.

This module provides functions to convert IP addresses to integers
and perform range-based lookups to map transactions to countries.
"""

from typing import Union

import pandas as pd
import numpy as np


def ip_to_int(ip: Union[str, float, int]) -> int:
    """
    Convert an IP address to its integer representation.

    IP addresses are stored as 32-bit integers where each octet
    represents 8 bits. For example: 192.168.1.1 = 192*256^3 + 168*256^2 + 1*256 + 1

    Parameters
    ----------
    ip : str, float, or int
        IP address as string (e.g., "192.168.1.1") or as a float/int
        representing the numeric form.

    Returns
    -------
    int
        Integer representation of the IP address.
        Returns -1 if the IP cannot be parsed.

    Examples
    --------
    >>> ip_to_int("192.168.1.1")
    3232235777
    >>> ip_to_int(3232235777.0)
    3232235777
    """
    # If already numeric, convert directly
    if isinstance(ip, (int, float)):
        if pd.isna(ip):
            return -1
        return int(ip)

    # If string, parse the dotted format
    if isinstance(ip, str):
        try:
            parts = ip.strip().split(".")
            if len(parts) != 4:
                return -1
            result = 0
            for part in parts:
                result = result * 256 + int(part)
            return result
        except (ValueError, AttributeError):
            return -1

    return -1


def ip_series_to_int(ip_series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series of IP addresses to integers.

    Handles both string format (dotted notation) and numeric format.

    Parameters
    ----------
    ip_series : pd.Series
        Series of IP addresses.

    Returns
    -------
    pd.Series
        Series of integer IP addresses. Invalid IPs are set to -1.
    """
    # Check if already numeric
    if pd.api.types.is_numeric_dtype(ip_series):
        return ip_series.fillna(-1).astype(np.int64)

    # Otherwise, apply conversion function
    return ip_series.apply(ip_to_int)


def prepare_ip_ranges(ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare IP range DataFrame for efficient range lookup.

    Ensures IP bounds are integers and the DataFrame is sorted
    by lower_bound for merge_asof compatibility.

    Parameters
    ----------
    ip_df : pd.DataFrame
        Raw IP country data with columns: lower_bound_ip_address,
        upper_bound_ip_address, country.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame sorted by lower_bound_ip_address.
    """
    df = ip_df.copy()

    # Ensure IP bounds are integers
    df["lower_bound_ip_address"] = pd.to_numeric(
        df["lower_bound_ip_address"], errors="coerce"
    ).fillna(0).astype(np.int64)

    df["upper_bound_ip_address"] = pd.to_numeric(
        df["upper_bound_ip_address"], errors="coerce"
    ).fillna(0).astype(np.int64)

    # Sort by lower bound for merge_asof
    df = df.sort_values("lower_bound_ip_address").reset_index(drop=True)

    return df


def merge_ip_to_country(
    transactions_df: pd.DataFrame,
    ip_df: pd.DataFrame,
    ip_column: str = "ip_address"
) -> pd.DataFrame:
    """
    Merge transaction data with country based on IP address range lookup.

    Uses pandas merge_asof to efficiently find the IP range that contains
    each transaction's IP address, then validates the upper bound.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Transaction data with an IP address column.
    ip_df : pd.DataFrame
        IP range to country mapping (prepared by prepare_ip_ranges).
    ip_column : str, optional
        Name of the IP address column in transactions_df.
        Default is "ip_address".

    Returns
    -------
    pd.DataFrame
        Transaction data with an added "country" column.
        Unmatched IPs will have country = "Unknown".

    Notes
    -----
    The merge_asof finds the closest lower_bound <= ip_address,
    then we validate that ip_address <= upper_bound.
    """
    df = transactions_df.copy()

    # Convert IP addresses to integers
    df["ip_int"] = ip_series_to_int(df[ip_column])

    # Prepare IP ranges
    ip_ranges = prepare_ip_ranges(ip_df)

    # Sort transactions by IP for merge_asof
    df_sorted = df.sort_values("ip_int").reset_index(drop=True)

    # Perform range-based merge using merge_asof
    # This finds the largest lower_bound that is <= ip_int
    merged = pd.merge_asof(
        df_sorted,
        ip_ranges[["lower_bound_ip_address", "upper_bound_ip_address", "country"]],
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    # Validate upper bound: IP must be <= upper_bound
    invalid_mask = (
        merged["ip_int"] > merged["upper_bound_ip_address"]
    ) | (
        merged["lower_bound_ip_address"].isna()
    ) | (
        merged["ip_int"] < 0
    )

    merged.loc[invalid_mask, "country"] = "Unknown"

    # Clean up temporary columns
    merged = merged.drop(columns=[
        "ip_int", "lower_bound_ip_address", "upper_bound_ip_address"
    ])

    # Restore original order
    merged = merged.sort_index().reset_index(drop=True)

    return merged


def get_country_fraud_stats(df: pd.DataFrame, target_col: str = "class") -> pd.DataFrame:
    """
    Calculate fraud statistics by country.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with 'country' and target columns.
    target_col : str, optional
        Name of the fraud indicator column. Default is "class".

    Returns
    -------
    pd.DataFrame
        Statistics per country: total_transactions, fraud_count,
        fraud_rate, sorted by fraud_count descending.
    """
    stats = df.groupby("country").agg(
        total_transactions=(target_col, "count"),
        fraud_count=(target_col, "sum")
    ).reset_index()

    stats["fraud_rate"] = stats["fraud_count"] / stats["total_transactions"]
    stats = stats.sort_values("fraud_count", ascending=False).reset_index(drop=True)

    return stats
