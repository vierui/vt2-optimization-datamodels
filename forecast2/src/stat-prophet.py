# forecast2/src/stat-prophet.py
import joblib
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("prophet not available, Prophet functionality will be limited", UserWarning)

from .dataset import load_raw

MODEL_PATH = Path(__file__).parents[1] / "models" / "prophet.pkl"

def train(cfg: dict = None, reload: bool = False):
    """Train Prophet model"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if MODEL_PATH.exists() and not reload:
        return joblib.load(MODEL_PATH)

    if not PROPHET_AVAILABLE:
        warnings.warn("prophet not available, cannot train Prophet", UserWarning)
        return None

    try:
        df = load_raw().reset_index()
        df_prophet = pd.DataFrame({
            'ds': df['time'],
            'y': df['electricity_pu']
        })
        
        # Configure Prophet with reasonable defaults for electricity data
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95,
            changepoint_prior_scale=0.05
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df_prophet)
        
        joblib.dump(model, MODEL_PATH)
        return model
    except Exception as e:
        warnings.warn(f"Prophet training failed: {e}", UserWarning)
        return None

def predict(model, day: str, horizon: int = 24):
    """Make predictions using Prophet model"""
    if model is None:
        warnings.warn("No trained Prophet model available", UserWarning)
        return np.zeros(horizon)
    
    try:
        start = pd.Timestamp(day).tz_localize("Europe/Zurich")
        future_dates = pd.date_range(start=start, periods=horizon, freq='H')
        
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        return forecast['yhat'].values
    except Exception as e:
        warnings.warn(f"Prophet prediction failed: {e}", UserWarning)
        return np.zeros(horizon)

def tune(cfg: dict = None, n_trials: int = 40):
    """Tune Prophet parameters - basic implementation"""
    # For now, just return the default trained model
    # Could be extended with proper hyperparameter tuning
    return train({}, reload=True) 