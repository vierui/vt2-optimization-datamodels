import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    df = df.copy()
    df["hour"]  = df[time_col].dt.hour
    df["month"] = df[time_col].dt.month
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df

def add_lags(df: pd.DataFrame, cols, lags) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        for l in lags:
            df[f"{c}_lag{l}"] = df[c].shift(l)
    return df

def add_rolling_means(df: pd.DataFrame, cols, windows=(24, 168)) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        for w in windows:
            df[f"{c}_roll{w}"] = df[c].rolling(w, min_periods=w).mean()
    return df