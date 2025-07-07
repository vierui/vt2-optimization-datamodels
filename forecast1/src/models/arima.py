"""
SARIMA baseline model for PV forecasting.
Handles parameter tuning and rolling forecasts.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Tuple, List, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SARIMAForecaster:
    """SARIMA baseline forecaster for univariate time series."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sarima_config = config['sarima']
        self.best_params = None
        self.best_model = None
        self.best_aic = np.inf
        
    def grid_search_sarima(self, train_data: pd.Series, val_data: pd.Series) -> Dict:
        """
        Grid search for optimal SARIMA parameters.
        
        Args:
            train_data: Training time series
            val_data: Validation time series for parameter selection
            
        Returns:
            Dictionary with best parameters and validation metrics
        """
        logger.info("Starting SARIMA grid search")
        
        p_range = self.sarima_config['p_range']
        q_range = self.sarima_config['q_range']
        d = self.sarima_config['d']
        P_range = self.sarima_config['P_range']
        Q_range = self.sarima_config['Q_range']
        D = self.sarima_config['D']
        seasonal_period = self.sarima_config['seasonal_period']
        
        best_params = None
        best_aic = np.inf
        best_val_mae = np.inf
        results = []
        
        total_combinations = len(p_range) * len(q_range) * len(P_range) * len(Q_range)
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        for p in p_range:
            for q in q_range:
                for P in P_range:
                    for Q in Q_range:
                        try:
                            # Fit SARIMA model
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, seasonal_period)
                            
                            model = SARIMAX(
                                train_data,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            
                            fitted_model = model.fit(disp=False, maxiter=100)
                            
                            # Calculate AIC
                            aic = fitted_model.aic
                            
                            # Validate on validation set
                            val_forecast = fitted_model.forecast(steps=len(val_data))
                            val_mae = mean_absolute_error(val_data, val_forecast)
                            val_rmse = np.sqrt(mean_squared_error(val_data, val_forecast))
                            
                            results.append({
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'aic': aic,
                                'val_mae': val_mae,
                                'val_rmse': val_rmse
                            })
                            
                            # Select best based on validation MAE
                            if val_mae < best_val_mae:
                                best_params = (order, seasonal_order)
                                best_aic = aic
                                best_val_mae = val_mae
                                self.best_model = fitted_model
                                
                            logger.info(f"SARIMA{order}x{seasonal_order}: AIC={aic:.2f}, Val MAE={val_mae:.4f}")
                            
                        except Exception as e:
                            logger.warning(f"Failed SARIMA{order}x{seasonal_order}: {str(e)}")
                            continue
        
        if best_params is None:
            raise ValueError("No valid SARIMA models could be fitted")
        
        self.best_params = best_params
        logger.info(f"Best SARIMA parameters: {best_params[0]}x{best_params[1]}")
        logger.info(f"Best validation MAE: {best_val_mae:.4f}")
        
        return {
            'best_params': best_params,
            'best_aic': best_aic,
            'best_val_mae': best_val_mae,
            'all_results': results
        }
    
    def fit(self, train_data: pd.Series, val_data: pd.Series = None) -> 'SARIMAForecaster':
        """
        Fit SARIMA model with optimal parameters.
        
        Args:
            train_data: Training time series
            val_data: Optional validation data for parameter tuning
            
        Returns:
            Self for method chaining
        """
        if val_data is not None:
            # Perform grid search
            self.grid_search_sarima(train_data, val_data)
        else:
            # Use default parameters
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, self.sarima_config['seasonal_period'])
            
            model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.best_model = model.fit(disp=False)
            self.best_params = (order, seasonal_order)
        
        logger.info("SARIMA model fitted successfully")
        return self
    
    def fit_manual(self, train_data: pd.Series, order: Tuple = None, 
                  seasonal_order: Tuple = None) -> 'SARIMAForecaster':
        """
        Fit SARIMA model with manual parameters.
        
        Args:
            train_data: Training time series
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            
        Returns:
            Self for method chaining
        """
        if order is None:
            order = (1, 1, 1)
        if seasonal_order is None:
            seasonal_order = (1, 1, 1, self.sarima_config['seasonal_period'])
        
        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.best_model = model.fit(disp=False)
        self.best_params = (order, seasonal_order)
        
        logger.info(f"SARIMA{order}x{seasonal_order} fitted manually")
        return self
    
    def forecast(self, steps: int) -> np.ndarray:
        """
        Generate forecast for specified number of steps.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Forecast array
        """
        if self.best_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = self.best_model.forecast(steps=steps)
        return np.maximum(0, forecast)  # Ensure non-negative values
    
    def rolling_forecast(self, data: pd.Series, forecast_horizon: int = 24, 
                        start_date: str = None) -> pd.DataFrame:
        """
        Perform rolling origin forecasts.
        
        Args:
            data: Complete time series data
            forecast_horizon: Number of steps to forecast each time
            start_date: Start date for rolling forecasts
            
        Returns:
            DataFrame with forecasts and actuals
        """
        logger.info(f"Starting rolling forecast with horizon {forecast_horizon}")
        
        if start_date is None:
            # Start from 80% of the data length
            start_idx = int(len(data) * 0.8)
        else:
            start_idx = data.index.get_loc(pd.Timestamp(start_date))
        
        forecasts = []
        actuals = []
        forecast_dates = []
        
        # Rolling forecast loop
        for i in range(start_idx, len(data) - forecast_horizon, forecast_horizon):
            try:
                # Training data up to current point
                train_data = data.iloc[:i]
                
                # Actual values for the forecast period
                actual_values = data.iloc[i:i+forecast_horizon]
                
                # Fit model and forecast
                model = SARIMAX(
                    train_data,
                    order=self.best_params[0],
                    seasonal_order=self.best_params[1],
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                fitted_model = model.fit(disp=False, maxiter=50)
                forecast_values = fitted_model.forecast(steps=forecast_horizon)
                forecast_values = np.maximum(0, forecast_values)  # Non-negative
                
                # Store results
                forecasts.extend(forecast_values)
                actuals.extend(actual_values.values)
                forecast_dates.extend(actual_values.index)
                
            except Exception as e:
                logger.warning(f"Failed forecast at position {i}: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': forecast_dates,
            'actual': actuals,
            'forecast': forecasts
        })
        results_df['date'] = pd.to_datetime(results_df['date'])
        results_df = results_df.set_index('date')
        
        logger.info(f"Completed {len(results_df)} rolling forecasts")
        
        return results_df
    
    def evaluate_forecast(self, forecast_df: pd.DataFrame) -> Dict:
        """
        Evaluate forecast performance.
        
        Args:
            forecast_df: DataFrame with 'actual' and 'forecast' columns
            
        Returns:
            Dictionary with evaluation metrics
        """
        actual = forecast_df['actual'].values
        forecast = forecast_df['forecast'].values
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual = actual[mask]
        forecast = forecast[mask]
        
        if len(actual) == 0:
            return {'error': 'No valid forecast pairs'}
        
        mae = mean_absolute_error(actual, forecast)
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mape = np.mean(np.abs((actual - forecast) / (actual + 1e-8))) * 100
        
        # Additional metrics
        bias = np.mean(forecast - actual)
        r2 = np.corrcoef(actual, forecast)[0, 1] ** 2 if len(actual) > 1 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'bias': bias,
            'r2': r2,
            'n_samples': len(actual)
        }
    
    def get_model_summary(self) -> str:
        """Get summary of the fitted model."""
        if self.best_model is None:
            return "No model fitted"
        
        return str(self.best_model.summary())
    
    def diagnose_residuals(self, data: pd.Series) -> Dict:
        """
        Perform residual diagnostics.
        
        Args:
            data: Time series data used for fitting
            
        Returns:
            Dictionary with diagnostic results
        """
        if self.best_model is None:
            return {'error': 'No model fitted'}
        
        # Get residuals
        residuals = self.best_model.resid
        
        # Ljung-Box test for autocorrelation
        try:
            lb_result = acorr_ljungbox(residuals, lags=10, return_df=False)
            if isinstance(lb_result, tuple):
                lb_stat, lb_pvalue = lb_result
                lb_stat_val = lb_stat[-1] if hasattr(lb_stat, '__getitem__') else lb_stat
                lb_pvalue_val = lb_pvalue[-1] if hasattr(lb_pvalue, '__getitem__') else lb_pvalue
            else:
                lb_stat_val = lb_result
                lb_pvalue_val = None
        except Exception as e:
            logger.warning(f"Ljung-Box test failed: {e}")
            lb_stat_val = None
            lb_pvalue_val = None
        
        # Basic residual statistics
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        residual_skew = residuals.skew()
        residual_kurt = residuals.kurtosis()
        
        return {
            'ljung_box_stat': lb_stat_val,
            'ljung_box_pvalue': lb_pvalue_val,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residual_skewness': residual_skew,
            'residual_kurtosis': residual_kurt,
            'autocorrelation_test': 'PASS' if lb_pvalue_val and lb_pvalue_val > 0.05 else 'FAIL'
        }


def evaluate_sarima_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Evaluate SARIMA forecast performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'error': 'No valid forecast pairs'}
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
    
    # Additional metrics
    bias = np.mean(y_pred_clean - y_true_clean)
    r2 = np.corrcoef(y_true_clean, y_pred_clean)[0, 1] ** 2 if len(y_true_clean) > 1 else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'bias': bias,
        'r2': r2,
        'n_samples': len(y_true_clean)
    }


def create_sarima_forecaster(config: Dict) -> SARIMAForecaster:
    """Factory function to create SARIMA forecaster."""
    return SARIMAForecaster(config) 