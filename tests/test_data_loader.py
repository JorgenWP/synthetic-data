"""
Tests for data loading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from src.data import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader(data_dir="data/raw")
        assert loader.data_dir == "data/raw"
    
    def test_split_data(self):
        """Test data splitting functionality."""
        # Create sample data
        df = pd.DataFrame({
            'col1': np.arange(100),
            'col2': np.random.rand(100)
        })
        
        loader = DataLoader()
        train_df, val_df, test_df = loader.split_data(
            df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
        
        # Check sizes
        assert len(train_df) == 70
        assert len(val_df) == 15
        assert len(test_df) == 15
        
        # Check no data loss
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
    
    def test_validate_data_empty(self):
        """Test validation with empty dataframe."""
        loader = DataLoader()
        df = pd.DataFrame()
        
        is_valid, errors = loader.validate_data(df)
        assert not is_valid
        assert len(errors) > 0
        assert "empty" in errors[0].lower()
