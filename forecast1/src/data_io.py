"""
Data IO module for PV forecasting system.
Simple data loading and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "src/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw CSV data with proper time handling."""
    logger.info(f"Loading raw data from {file_path}")
    
    # Skip header lines and load data
    df = pd.read_csv(file_path, skiprows=3)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    
    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning."""
    logger.info("Cleaning data")
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Handle negative electricity values
    if (df['electricity'] < 0).any():
        df.loc[df['electricity'] < 0, 'electricity'] = 0
    
    # Forward fill missing values
    df = df.ffill().bfill()
    
    return df


def create_time_splits(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    splits = config['splits']
    
    train_mask = (df.index >= splits['train_start']) & (df.index <= splits['train_end'])
    val_mask = (df.index >= splits['val_start']) & (df.index <= splits['val_end'])
    test_mask = (df.index >= splits['test_start']) & (df.index <= splits['test_end'])
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def load_and_process_data(config: Dict) -> pd.DataFrame:
    """Main data processing pipeline."""
    # Load and process raw data directly
    df = load_raw_data(config['data']['raw_file'])
    df = clean_data(df)
    
    return df 