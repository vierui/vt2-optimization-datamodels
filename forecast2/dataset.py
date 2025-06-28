# forecast2/dataset.py
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml

from .features import features

def load_config() -> Dict:
    with open(Path(__file__).parent / "config.yaml") as f:
        return yaml.safe_load(f)

CFG = load_config()

# forecast2/dataset.py
def load_raw() -> pd.DataFrame:
    csv_path = Path(__file__).parents[1] / CFG["data_csv"]

    # 1) read; ignore header comments that start with "#"
    df = pd.read_csv(csv_path, comment="#", parse_dates=["time"])

    # 2) tag the timestamps as UTC
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # 3) set index and convert into Europe/Zurich
    df = df.set_index("time").tz_convert("Europe/Zurich")

    # 4) per-unit target
    df["electricity_pu"] = df["electricity"] / CFG["capacity_kw"]
    return df

def make_sets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw()
    df_feat = features(
        df,
        target_col="electricity_pu",
        lags=CFG["features"]["lags"],
        rolling_mean_hrs=CFG["features"]["rolling_mean_hrs"],
    ).dropna()                      # drop rows with undefined lags

    train = df_feat.loc[:CFG["split"]["train_end"]]
    val   = df_feat.loc[
        CFG["split"]["train_end"]: CFG["split"]["val_end"]
    ]
    test  = df_feat.loc[
        CFG["split"]["val_end"]: CFG["split"]["test_end"]
    ]
    return train, val, test

def split(df: pd.DataFrame, horizon: int = 24):
    """
    X = features at time t
    y = vector [t+1, t+2, ..., t+24] of per-unit electricity
    """
    X = df.drop(columns=["electricity_pu"])
    # build matrix of shifted targets
    y_cols = []
    for h in range(1, horizon + 1):
        col = f"electricity_pu_t+{h}"
        df[col] = df["electricity_pu"].shift(-h)
        y_cols.append(col)
    y = df[y_cols]
    # align shapes
    X, y = X.iloc[:-horizon], y.dropna()
    return X, y.iloc[: X.shape[0]]