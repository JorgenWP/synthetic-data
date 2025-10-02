"""
Data loading utilities for transactional data.
"""

import os
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np


class DataLoader:
    """
    Load and manage transactional data from various sources.
    
    This class provides utilities to load raw transactional data
    that will be converted into graph representations.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = data_dir
        
    def load_transactions(
        self, 
        filename: str,
        file_format: str = "csv"
    ) -> pd.DataFrame:
        """
        Load transactional data from file.
        
        Args:
            filename: Name of the file to load
            file_format: Format of the file ('csv', 'json', 'parquet')
            
        Returns:
            DataFrame containing the transactional data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if file_format == "csv":
            return pd.read_csv(filepath)
        elif file_format == "json":
            return pd.read_json(filepath)
        elif file_format == "parquet":
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that the loaded data meets requirements.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Add your validation logic here
        if df.empty:
            errors.append("DataFrame is empty")
            
        # Example: Check for required columns
        # required_cols = ['transaction_id', 'source', 'target', 'timestamp']
        # missing = set(required_cols) - set(df.columns)
        # if missing:
        #     errors.append(f"Missing required columns: {missing}")
        
        return len(errors) == 0, errors
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: Optional[int] = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        n = len(df_shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df_shuffled[:train_end]
        val_df = df_shuffled[train_end:val_end]
        test_df = df_shuffled[val_end:]
        
        return train_df, val_df, test_df
