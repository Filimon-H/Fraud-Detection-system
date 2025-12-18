"""
Data loading utilities for fraud detection datasets.

This module provides functions to load the raw CSV files with proper
data types and basic validation.
"""

from pathlib import Path
from typing import Union

import pandas as pd


def load_fraud_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the e-commerce fraud dataset (Fraud_Data.csv).

    Parameters
    ----------
    filepath : str or Path
        Path to the Fraud_Data.csv file.

    Returns
    -------
    pd.DataFrame
        Raw fraud data with columns: user_id, signup_time, purchase_time,
        purchase_value, device_id, source, browser, sex, age, ip_address, class.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fraud data file not found: {filepath}")

    df = pd.read_csv(filepath)

    required_columns = [
        "user_id", "signup_time", "purchase_time", "purchase_value",
        "device_id", "source", "browser", "sex", "age", "ip_address", "class"
    ]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_ip_country_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the IP address to country mapping dataset.

    Parameters
    ----------
    filepath : str or Path
        Path to the IpAddress_to_Country.csv file.

    Returns
    -------
    pd.DataFrame
        IP range data with columns: lower_bound_ip_address,
        upper_bound_ip_address, country.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"IP country file not found: {filepath}")

    df = pd.read_csv(filepath)

    required_columns = ["lower_bound_ip_address", "upper_bound_ip_address", "country"]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_creditcard_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the credit card fraud dataset (creditcard.csv).

    Parameters
    ----------
    filepath : str or Path
        Path to the creditcard.csv file.

    Returns
    -------
    pd.DataFrame
        Credit card transaction data with columns: Time, V1-V28, Amount, Class.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Credit card data file not found: {filepath}")

    df = pd.read_csv(filepath)

    required_columns = ["Time", "Amount", "Class"]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df
