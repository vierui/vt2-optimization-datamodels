"""
Evaluation utilities for time-series forecasting models.
Provides rolling-origin evaluation, metrics, and model comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import ttest_rel
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RollingOriginEvaluator:
    """Rolling origin cross-validation for time series forecasting."""
    
    def __init__(self, forecast_horizon: int = 24, step_size: int = 24):
        """
        Initialize evaluator.
        
        Args:
            forecast_horizon: Number of steps to forecast ahead
            step_size: Step size between rolling origins
        """
        self.forecast_horizon = forecast_horizon
        self.step_size = step_size
        
    def generate_splits(self, data: pd.Series, start_ratio: float = 0.7,
                       min_train_size: int = 1000) -> List[Tuple[int, int, int]]:
        """
        Generate rolling origin train/forecast splits.
        
        Args:
            data: Time series data
            start_ratio: Where to start rolling evaluation (as fraction of total data)
            min_train_size: Minimum training size
            
        Returns:
            List of (train_end, forecast_start, forecast_end) indices
        """
        total_length = len(data)
        start_idx = max(min_train_size, int(total_length * start_ratio))
        
        splits = []
        for train_end in range(start_idx, total_length - self.forecast_horizon, self.step_size):
            forecast_start = train_end
            forecast_end = train_end + self.forecast_horizon
            
            if forecast_end <= total_length:
                splits.append((train_end, forecast_start, forecast_end))
        
        logger.info(f"Generated {len(splits)} rolling splits")
        return splits
    
    def evaluate_forecasts(self, actuals: np.ndarray, forecasts: np.ndarray) -> Dict[str, float]:
        """
        Evaluate forecast performance.
        
        Args:
            actuals: Actual values
            forecasts: Forecast values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Handle different shapes
        if actuals.ndim > 1:
            actuals = actuals.flatten()
        if forecasts.ndim > 1:
            forecasts = forecasts.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(actuals) | np.isnan(forecasts))
        actuals_clean = actuals[mask]
        forecasts_clean = forecasts[mask]
        
        if len(actuals_clean) == 0:
            return {'error': 'No valid forecast pairs'}
        
        # Calculate metrics
        mae = mean_absolute_error(actuals_clean, forecasts_clean)
        rmse = np.sqrt(mean_squared_error(actuals_clean, forecasts_clean))
        
        # Avoid division by zero in MAPE
        mape = np.mean(np.abs((actuals_clean - forecasts_clean) / 
                             np.maximum(actuals_clean, 1e-8))) * 100
        
        # Additional metrics
        bias = np.mean(forecasts_clean - actuals_clean)
        r2 = np.corrcoef(actuals_clean, forecasts_clean)[0, 1] ** 2 if len(actuals_clean) > 1 else 0
        
        # Normalized metrics
        mae_norm = mae / (np.mean(actuals_clean) + 1e-8)
        rmse_norm = rmse / (np.mean(actuals_clean) + 1e-8)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'bias': bias,
            'r2': r2,
            'mae_norm': mae_norm,
            'rmse_norm': rmse_norm,
            'n_samples': len(actuals_clean)
        }


def calculate_metrics_by_horizon(actuals: np.ndarray, forecasts: np.ndarray) -> Dict[str, List[float]]:
    """
    Calculate metrics by forecast horizon.
    
    Args:
        actuals: Actual values (samples, horizon)
        forecasts: Forecast values (samples, horizon)
        
    Returns:
        Dictionary with metrics by horizon
    """
    horizon_length = actuals.shape[1] if actuals.ndim > 1 else 1
    
    mae_by_horizon = []
    rmse_by_horizon = []
    
    for h in range(horizon_length):
        if actuals.ndim > 1:
            actual_h = actuals[:, h]
            forecast_h = forecasts[:, h]
        else:
            actual_h = actuals
            forecast_h = forecasts
        
        # Remove NaN values
        mask = ~(np.isnan(actual_h) | np.isnan(forecast_h))
        actual_clean = actual_h[mask]
        forecast_clean = forecast_h[mask]
        
        if len(actual_clean) > 0:
            mae_h = mean_absolute_error(actual_clean, forecast_clean)
            rmse_h = np.sqrt(mean_squared_error(actual_clean, forecast_clean))
        else:
            mae_h = np.nan
            rmse_h = np.nan
        
        mae_by_horizon.append(mae_h)
        rmse_by_horizon.append(rmse_h)
    
    return {
        'mae_by_horizon': mae_by_horizon,
        'rmse_by_horizon': rmse_by_horizon
    }


def calculate_metrics_by_time(forecast_dates: pd.DatetimeIndex, 
                             actuals: np.ndarray, 
                             forecasts: np.ndarray) -> pd.DataFrame:
    """
    Calculate metrics by time features (hour, day of week).
    
    Args:
        forecast_dates: Datetime index for forecasts
        actuals: Actual values
        forecasts: Forecast values
        
    Returns:
        DataFrame with time-based metrics
    """
    # Flatten arrays if multi-dimensional
    if actuals.ndim > 1:
        actuals = actuals[:, 0]  # First hour of each forecast
        forecasts = forecasts[:, 0]
        forecast_dates = forecast_dates[:len(actuals)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': forecast_dates,
        'actual': actuals,
        'forecast': forecasts,
        'error': actuals - forecasts,
        'abs_error': np.abs(actuals - forecasts)
    })
    
    # Add time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    return df


def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray) -> Dict[str, float]:
    """
    Diebold-Mariano test for forecast accuracy comparison.
    
    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        
    Returns:
        Dictionary with test statistics
    """
    # Calculate loss differential
    loss1 = errors1 ** 2
    loss2 = errors2 ** 2
    loss_diff = loss1 - loss2
    
    # Remove NaN values
    loss_diff_clean = loss_diff[~np.isnan(loss_diff)]
    
    if len(loss_diff_clean) == 0:
        return {'error': 'No valid loss differences'}
    
    # Simple t-test (assuming no autocorrelation for simplicity)
    mean_diff = np.mean(loss_diff_clean)
    std_diff = np.std(loss_diff_clean, ddof=1)
    
    if std_diff == 0:
        return {'dm_stat': 0, 'p_value': 1.0, 'interpretation': 'No difference'}
    
    t_stat = mean_diff / (std_diff / np.sqrt(len(loss_diff_clean)))
    
    # Approximate p-value (two-tailed)
    from scipy.stats import t
    p_value = 2 * (1 - t.cdf(abs(t_stat), len(loss_diff_clean) - 1))
    
    # Interpretation
    if p_value < 0.05:
        if mean_diff < 0:
            interpretation = "Model 1 significantly better"
        else:
            interpretation = "Model 2 significantly better"
    else:
        interpretation = "No significant difference"
    
    return {
        'dm_stat': t_stat,
        'p_value': p_value,
        'mean_loss_diff': mean_diff,
        'interpretation': interpretation
    }


def plot_forecast_comparison(actual: np.ndarray, 
                           forecasts: Dict[str, np.ndarray],
                           dates: pd.DatetimeIndex = None,
                           title: str = "Forecast Comparison",
                           save_path: str = None) -> None:
    """
    Plot comparison of multiple forecasts.
    
    Args:
        actual: Actual values
        forecasts: Dictionary of model_name -> forecast_values
        dates: Datetime index (optional)
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(15, 8))
    
    # Use index if no dates provided
    x_axis = dates if dates is not None else range(len(actual))
    
    # Plot actual
    plt.plot(x_axis, actual, label='Actual', alpha=0.8, linewidth=2, color='black')
    
    # Plot forecasts
    colors = plt.cm.Set1(np.linspace(0, 1, len(forecasts)))
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        plt.plot(x_axis, forecast, label=model_name, alpha=0.7, 
                linewidth=2, color=colors[i])
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_heatmap(time_df: pd.DataFrame, 
                      metric: str = 'abs_error',
                      title: str = "Error Heatmap",
                      save_path: str = None) -> None:
    """
    Plot error heatmap by day of week and hour.
    
    Args:
        time_df: DataFrame with time features and errors
        metric: Metric to plot ('abs_error', 'error', etc.)
        title: Plot title
        save_path: Path to save plot
    """
    # Create pivot table
    heatmap_data = time_df.pivot_table(
        values=metric, 
        index='day_of_week', 
        columns='hour', 
        aggfunc='mean'
    )
    
    # Rename index for better display
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    heatmap_data.index = dow_names
    
    plt.figure(figsize=(15, 6))
    sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd', cbar_kws={'label': metric})
    plt.title(title)
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_horizon_degradation(horizon_metrics: Dict[str, List[float]],
                           title: str = "Forecast Accuracy by Horizon",
                           save_path: str = None) -> None:
    """
    Plot forecast accuracy degradation by horizon.
    
    Args:
        horizon_metrics: Dictionary with metrics by horizon
        title: Plot title
        save_path: Path to save plot
    """
    horizons = range(1, len(horizon_metrics['mae_by_horizon']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE
    ax1.plot(horizons, horizon_metrics['mae_by_horizon'], 'bo-', linewidth=2, markersize=6)
    ax1.set_title('MAE by Forecast Horizon')
    ax1.set_xlabel('Hours Ahead')
    ax1.set_ylabel('MAE')
    ax1.grid(True, alpha=0.3)
    
    # RMSE
    ax2.plot(horizons, horizon_metrics['rmse_by_horizon'], 'ro-', linewidth=2, markersize=6)
    ax2.set_title('RMSE by Forecast Horizon')
    ax2.set_xlabel('Hours Ahead')
    ax2.set_ylabel('RMSE')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_model_comparison_report(results: Dict[str, Dict]) -> str:
    """
    Create a markdown report comparing model performance.
    
    Args:
        results: Dictionary of model_name -> results_dict
        
    Returns:
        Markdown report string
    """
    report = "# Model Comparison Report\n\n"
    
    # Summary table
    report += "## Performance Summary\n\n"
    report += "| Model | MAE | RMSE | MAPE | R² | Bias |\n"
    report += "|-------|-----|------|------|----|----- |\n"
    
    for model_name, model_results in results.items():
        metrics = model_results.get('metrics', {})
        mae = metrics.get('mae', 0)
        rmse = metrics.get('rmse', 0)
        mape = metrics.get('mape', 0)
        r2 = metrics.get('r2', 0)
        bias = metrics.get('bias', 0)
        
        report += f"| {model_name} | {mae:.4f} | {rmse:.4f} | {mape:.2f}% | {r2:.3f} | {bias:.4f} |\n"
    
    # Best model
    if results:
        best_model = min(results.keys(), key=lambda x: results[x].get('metrics', {}).get('mae', float('inf')))
        report += f"\n**Best Model (by MAE):** {best_model}\n\n"
    
    # Model details
    report += "## Model Details\n\n"
    for model_name, model_results in results.items():
        report += f"### {model_name}\n\n"
        
        # Parameters
        if 'params' in model_results:
            report += "**Parameters:**\n"
            for param, value in model_results['params'].items():
                report += f"- {param}: {value}\n"
            report += "\n"
        
        # Metrics
        if 'metrics' in model_results:
            report += "**Metrics:**\n"
            for metric, value in model_results['metrics'].items():
                if isinstance(value, float):
                    report += f"- {metric.upper()}: {value:.4f}\n"
                else:
                    report += f"- {metric.upper()}: {value}\n"
            report += "\n"
    
    return report 