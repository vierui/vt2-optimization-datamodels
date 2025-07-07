"""
Feature engineering module for PV forecasting system.
Creates lag features, cyclical features, and handles scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Simple feature engineering for time-series forecasting."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.target_column = config['data']['target_column']
        self.weather_columns = config['data']['weather_columns']
        self.target_lags = config['features']['target_lags']
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.fitted = False
        
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for the target variable."""
        feature_df = df.copy()
        
        for lag in self.target_lags:
            feature_df[f'{self.target_column}_lag_{lag}'] = df[self.target_column].shift(lag)
            
        return feature_df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical time-based features."""
        feature_df = df.copy()
        
        # Hour of day
        feature_df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        feature_df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # Day of week
        feature_df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        feature_df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Day of year
        feature_df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        feature_df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        
        return feature_df
    
    def make_features(self, df: pd.DataFrame, use_weather: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Main feature engineering pipeline."""
        logger.info(f"Creating features (use_weather={use_weather})")
        
        # Create features
        feature_df = self.create_lag_features(df)
        feature_df = self.create_cyclical_features(feature_df)
        
        # Drop rows with NaN values (due to lags)
        max_lag = max(self.target_lags)
        feature_df = feature_df.iloc[max_lag:].copy()
        
        # Separate features and target
        target = feature_df[self.target_column].copy()
        
        # Select feature columns
        feature_cols = [col for col in feature_df.columns if col != self.target_column]
        
        if not use_weather:
            # Remove weather columns
            feature_cols = [col for col in feature_cols if col not in self.weather_columns]
        
        features = feature_df[feature_cols].copy()
        features = features.fillna(features.mean())
        
        # Fit scalers if not already fitted
        if not self.fitted:
            self.feature_scaler.fit(features)
            self.target_scaler.fit(target.values.reshape(-1, 1))
            self.fitted = True
        
        # Scale features and target
        features_scaled = pd.DataFrame(
            self.feature_scaler.transform(features),
            index=features.index,
            columns=features.columns
        )
        
        target_scaled = pd.Series(
            self.target_scaler.transform(target.values.reshape(-1, 1)).flatten(),
            index=target.index,
            name=target.name
        )
        
        logger.info(f"Created {len(features_scaled.columns)} features for {len(features_scaled)} samples")
        
        return features_scaled, target_scaled
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled target values."""
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten() 