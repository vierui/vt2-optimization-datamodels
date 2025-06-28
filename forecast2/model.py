# forecast2/model.py
import joblib
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from .dataset import CFG

def build_model():
    base = XGBRegressor(
        **CFG["xgb_params"]
    )
    return MultiOutputRegressor(base)

def save_model(mod, name="xgb_day_ahead"):
    Path(CFG["model_dir"]).mkdir(parents=True, exist_ok=True)
    joblib.dump(mod, Path(CFG["model_dir"]) / f"{name}.joblib")

def load_model(name="xgb_day_ahead"):
    return joblib.load(Path(CFG["model_dir"]) / f"{name}.joblib")