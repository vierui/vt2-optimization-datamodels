import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred):
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":   r2_score(y_true, y_pred),
    }

def to_frame(metrics_dict):
    """Convenience → metrics dict ⇢ tidy DataFrame."""
    return pd.DataFrame(metrics_dict).T[['MAE', 'RMSE', 'R2']]