"""
Unit tests for data loading and processing modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.geo import ip_to_int, ip_series_to_int


class TestIpConversion:
    """Tests for IP address conversion functions."""

    def test_ip_to_int_dotted_string(self):
        """Test conversion of dotted IP string to integer."""
        assert ip_to_int("192.168.1.1") == 3232235777
        assert ip_to_int("0.0.0.0") == 0
        assert ip_to_int("255.255.255.255") == 4294967295

    def test_ip_to_int_numeric(self):
        """Test conversion when IP is already numeric."""
        assert ip_to_int(3232235777) == 3232235777
        assert ip_to_int(3232235777.0) == 3232235777

    def test_ip_to_int_invalid(self):
        """Test handling of invalid IP addresses."""
        assert ip_to_int("invalid") == -1
        assert ip_to_int("192.168.1") == -1  # Missing octet
        assert ip_to_int(None) == -1
        assert ip_to_int(np.nan) == -1

    def test_ip_series_to_int(self):
        """Test conversion of pandas Series of IPs."""
        series = pd.Series([3232235777.0, 167772161.0, np.nan])
        result = ip_series_to_int(series)
        
        assert result.iloc[0] == 3232235777
        assert result.iloc[1] == 167772161
        assert result.iloc[2] == -1


class TestDataLoader:
    """Tests for data loading functions."""

    def test_load_fraud_data_missing_file(self):
        """Test that FileNotFoundError is raised for missing file."""
        from src.data.loader import load_fraud_data
        
        with pytest.raises(FileNotFoundError):
            load_fraud_data("nonexistent_file.csv")


class TestDataCleaning:
    """Tests for data cleaning functions."""

    def test_get_missing_value_summary(self):
        """Test missing value summary generation."""
        from src.data.cleaning import get_missing_value_summary
        
        df = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [1, None, None, 4],
            'c': [1, 2, 3, 4]
        })
        
        summary = get_missing_value_summary(df)
        
        # Should have 2 rows (columns a and b have missing)
        assert len(summary) == 2
        # Column b should be first (more missing)
        assert summary.iloc[0]['column'] == 'b'
        assert summary.iloc[0]['missing_count'] == 2

    def test_get_duplicate_summary(self):
        """Test duplicate summary generation."""
        from src.data.cleaning import get_duplicate_summary
        
        df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': [1, 2, 2, 3]
        })
        
        summary = get_duplicate_summary(df)
        
        assert summary['total_rows'] == 4
        assert summary['duplicate_rows'] == 1
        assert summary['unique_rows'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
