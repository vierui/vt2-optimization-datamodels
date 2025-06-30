# forecast2/src/ml-xgb.py
import joblib
import optuna
import warnings
from pathlib import Path
import numpy as np
import yaml
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from .dataset import make_sets, make_anchor_training_set, CFG

MODEL_PATH = Path(__file__).parents[1] / "models" / "xgb.joblib"
HORIZON = 24

def nmae(y_true, y_pred, mask=None):
    """Normalized Mean Absolute Error"""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return mean_absolute_error(y_true, y_pred) / np.mean(y_true)

def _anchor_row(df, day: str):
    """Return the last timestamp BEFORE 00:00 local time of <day>"""
    import pandas as pd
    tgt = pd.Timestamp(day).tz_localize("Europe/Zurich")
    return df[df.index < tgt].iloc[-1]

def train(cfg: dict = None, reload: bool = False):
    """Train XGBoost model"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if MODEL_PATH.exists() and not reload:
        return joblib.load(MODEL_PATH)

    train_df, val_df, _ = make_sets()
    X_train, y_train = make_anchor_training_set(train_df, HORIZON)
    
    # If no config provided, reload from file to get latest parameters
    if cfg is None:
        config_path = Path(__file__).parents[1] / "config.yaml"
        with open(config_path, 'r') as f:
            current_config = yaml.safe_load(f)
        params = current_config.get("xgb", {})
    else:
        params = cfg
    
    print(f"Training XGBoost with parameters: {params}")
    base = XGBRegressor(**params)
    model = MultiOutputRegressor(base).fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model

def predict(model, day: str, horizon: int = HORIZON):
    """Make day-ahead predictions for a specific day"""
    _, _, test_df = make_sets()
    anchor = _anchor_row(test_df, day)
    X_in = anchor.drop("electricity_pu").to_frame().T
    y_pred = model.predict(X_in)[0]  # shape (24,)
    return y_pred

def tune(cfg: dict = None, n_trials: int = 40):
    """Tune XGBoost hyperparameters using Optuna"""
    train_df, val_df, _ = make_sets()
    X_train, y_train = make_anchor_training_set(train_df, HORIZON)
    X_val, y_val = make_anchor_training_set(val_df, HORIZON)

    def objective(trial):
        ranges = CFG["xgb_tuning_ranges"]
        params = {
            "n_estimators": trial.suggest_int("n_estimators", ranges["n_estimators"]["min"], ranges["n_estimators"]["max"]),
            "max_depth": trial.suggest_int("max_depth", ranges["max_depth"]["min"], ranges["max_depth"]["max"]),
            "learning_rate": trial.suggest_float("learning_rate", ranges["learning_rate"]["min"], ranges["learning_rate"]["max"], log=ranges["learning_rate"].get("log", False)),
            "subsample": trial.suggest_float("subsample", ranges["subsample"]["min"], ranges["subsample"]["max"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", ranges["colsample_bytree"]["min"], ranges["colsample_bytree"]["max"]),
            "min_child_weight": trial.suggest_float("min_child_weight", ranges["min_child_weight"]["min"], ranges["min_child_weight"]["max"], log=ranges["min_child_weight"].get("log", False)),
            "gamma": trial.suggest_float("gamma", ranges["gamma"]["min"], ranges["gamma"]["max"]),
            "reg_lambda": trial.suggest_float("reg_lambda", ranges["reg_lambda"]["min"], ranges["reg_lambda"]["max"]),
            "objective": "reg:squarederror", "eval_metric": "mae", "n_jobs": -1,
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MultiOutputRegressor(XGBRegressor(**params))
            model.fit(X_train, y_train)
            y_hat = model.predict(X_val)
            return nmae(y_val.values.ravel(), y_hat.ravel())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("Best params:", study.best_params, "score:", study.best_value)

    # Update config file with best parameters
    config_path = Path(__file__).parents[1] / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update xgb section with best parameters
    config['xgb'].update(study.best_params)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config file with best parameters: {study.best_params}")
    
    return train(reload=True) 