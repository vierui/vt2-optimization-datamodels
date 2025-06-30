# forecast2/features.py
import numpy as np
import pandas as pd
from pvlib.solarposition import get_solarposition
from pvlib.location import Location

# site coordinates
LOC = Location(46.2312, 7.3589, tz="Europe/Zurich")

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
                          lags=(1, 2, 24, 48, 168), rolling_mean_hrs: int = 3) -> pd.DataFrame:
    """Append lag and rolling-mean features derived from the target and irradiance."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    df[f"{target_col}_roll_mean"] = (
        df[target_col].rolling(window=rolling_mean_hrs).mean().shift(1)
    )
    
    # Add irradiance lag features (1h and 24h as suggested in T-4)
    if "irradiance_direct" in df.columns:
        df["irradiance_direct_lag1"] = df["irradiance_direct"].shift(1)
        df["irradiance_direct_lag24"] = df["irradiance_direct"].shift(24)
    
    if "irradiance_diffuse" in df.columns:
        df["irradiance_diffuse_lag1"] = df["irradiance_diffuse"].shift(1)
        df["irradiance_diffuse_lag24"] = df["irradiance_diffuse"].shift(24)
    
    return df

def weather_features(df: pd.DataFrame, raw_df: pd.DataFrame, 
                    lags_weather=(1, 24), rolling_mean_hrs: int = 3) -> pd.DataFrame:
    """Add weather features including lags, rolling means, and derived features."""
    df = df.copy()
    
    # --- weather features -------------------------------------------------
    wx_cols = ["swgdn", "cldtot", "t2m", "prectotland"]
    df[wx_cols] = raw_df[wx_cols]      # pass-through originals

    # binary rain flag
    df["is_rain"] = (df["prectotland"] > 0.1).astype(int)

    # lags and rolling means
    w_lags = lags_weather if isinstance(lags_weather, (list, tuple)) else [1, 24]
    for col in ["swgdn", "cldtot", "t2m", "prectotland"]:
        for lg in w_lags:
            df[f"{col}_lag{lg}"] = df[col].shift(lg)

    roll = rolling_mean_hrs
    df["swgdn_roll_mean"]  = df["swgdn"].shift(1).rolling(roll).mean()
    df["cldtot_roll_mean"] = df["cldtot"].shift(1).rolling(roll).mean()
    
    return df

def features(raw_df: pd.DataFrame,
                        target_col: str = "electricity_pu",
                        lags=(1,2,24,48,168), 
                        lags_weather=(1,24),
                        rolling_mean_hrs: int = 3) -> pd.DataFrame:
    """All feature engineering steps, in order."""
    df = time_features(raw_df)
    df = lag_roll_features(df, target_col, lags, rolling_mean_hrs)
    df = weather_features(df, raw_df, lags_weather, rolling_mean_hrs)
    
    # solar-zenith & daylight
    solpos = get_solarposition(
        time=df.index, latitude=46.2312, longitude=7.3589, altitude=0
    )
    df["solar_zenith"] = solpos["zenith"]
    df["is_day"] = (df["solar_zenith"] < 89.9).astype(int)
    
    # ensure returned df excludes obsolete clearsky features if still present
    return df.drop(columns=[c for c in ("ghi_cs","cs_ratio") if c in df.columns])