# forecast2/features.py
import numpy as np
import pandas as pd

def time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encodings for hour-of-day and day-of-year (Europe/Zurich)."""
    df = df.copy()
    # assume datetime index in Europe/Zurich and already tz-aware
    dt = df.index
    # hour
    hours = dt.hour + dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    # day-of-year
    doy = dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df

def lag_roll_features(df: pd.DataFrame, target_col: str,
                          lags=(24,), rolling_mean_hrs: int = 3) -> pd.DataFrame:
    """Append lag and rolling-mean features derived from the target."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    df[f"{target_col}_roll_mean"] = (
        df[target_col].shift(1).rolling(window=rolling_mean_hrs).mean()
    )
    return df

def features(raw_df: pd.DataFrame,
                        target_col: str = "electricity_pu",
                        lags=(24,), rolling_mean_hrs: int = 3) -> pd.DataFrame:
    """All feature engineering steps, in order."""
    df = time_features(raw_df)
    df = lag_roll_features(df, target_col, lags, rolling_mean_hrs)
    return df