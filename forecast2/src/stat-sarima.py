# forecast2/src/stat-sarima.py
import joblib
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import statsmodels components
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError as e:
    STATSMODELS_AVAILABLE = False
    STL = None
    seasonal_decompose = None
    adfuller = None
    kpss = None
    plot_acf = None
    plot_pacf = None
    SARIMAX = None
    warnings.warn(f"statsmodels not available: {e}", UserWarning)

# Import pmdarima (optional for auto_arima)
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError as e:
    PMDARIMA_AVAILABLE = False
    pm = None
    # Don't warn here - we'll handle it in training

from .dataset import load_raw

MODEL_PATH = Path(__file__).parents[1] / "models" / "sarima.pkl"
DIAGNOSTICS_PATH = Path(__file__).parents[1] / "results" / "sarima_diagnostics"

def seasonal_trend_decomposition(y: pd.Series, period: int = 24, save_plots: bool = True) -> dict:
    """
    Perform STL decomposition to analyze seasonality and trend.
    
    Returns:
        dict: Contains decomposition components and recommendations for differencing
    """
    print("=== SEASONAL-TREND DECOMPOSITION ===")
    
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels not available - install statsmodels")
    
    # Try STL first, fallback to seasonal_decompose
    try:
        if STL is not None:
            print("Using STL decomposition...")
            stl = STL(y, seasonal=period, robust=True)
            result = stl.fit()
        else:
            print("Using classical seasonal decomposition...")
            result = seasonal_decompose(y, model='additive', period=period)
    except Exception as e:
        print(f"STL failed, trying classical decomposition: {e}")
        if seasonal_decompose is not None:
            result = seasonal_decompose(y, model='additive', period=period)
        else:
            raise ImportError("No seasonal decomposition method available")
    
    # Analyze components
    trend_strength = 1 - np.var(result.resid) / np.var(result.trend + result.resid)
    seasonal_strength = 1 - np.var(result.resid) / np.var(result.seasonal + result.resid)
    
    print(f"Trend strength: {trend_strength:.4f}")
    print(f"Seasonal strength: {seasonal_strength:.4f}")
    
    # Recommendations based on decomposition
    recommend_seasonal_diff = seasonal_strength > 0.3
    recommend_trend_diff = trend_strength > 0.3
    
    print(f"Recommend seasonal differencing (D=1): {recommend_seasonal_diff}")
    print(f"Recommend trend differencing (d=1): {recommend_trend_diff}")
    
    if save_plots:
        DIAGNOSTICS_PATH.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        result.observed.plot(ax=axes[0], title='Original Series', color='blue')
        result.trend.plot(ax=axes[1], title='Trend', color='green')
        result.seasonal.plot(ax=axes[2], title='Seasonal (24h)', color='orange')
        result.resid.plot(ax=axes[3], title='Residual', color='red')
        
        plt.tight_layout()
        plt.savefig(DIAGNOSTICS_PATH / "stl_decomposition.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"STL decomposition plot saved to: {DIAGNOSTICS_PATH / 'stl_decomposition.png'}")
    
    return {
        'trend_strength': trend_strength,
        'seasonal_strength': seasonal_strength,
        'recommend_d': 1 if recommend_trend_diff else 0,
        'recommend_D': 1 if recommend_seasonal_diff else 0,
        'decomposition': result
    }

def unit_root_tests(y: pd.Series, max_lags: int = 24) -> dict:
    """
    Perform ADF and KPSS tests on raw, differenced, and seasonally differenced series.
    
    Returns:
        dict: Test results and stationarity recommendations
    """
    print("\n=== UNIT ROOT TESTS ===")
    
    if not STATSMODELS_AVAILABLE or adfuller is None or kpss is None:
        raise ImportError("statsmodels.tsa.stattools not available - install statsmodels")
    
    results = {}
    
    # Test different series
    series_to_test = {
        'raw': y,
        'diff': y.diff().dropna(),
        'seasonal_diff': y.diff(24).dropna(),
        'double_diff': y.diff().diff(24).dropna()
    }
    
    for name, series in series_to_test.items():
        print(f"\n--- {name.upper()} SERIES ---")
        
        # ADF Test (null hypothesis: unit root exists)
        try:
            adf_stat, adf_pvalue, adf_lags, adf_nobs, adf_critical, adf_icbest = adfuller(
                series.dropna(), maxlag=max_lags, autolag='AIC'
            )
            adf_stationary = adf_pvalue < 0.05
            print(f"ADF: statistic={adf_stat:.4f}, p-value={adf_pvalue:.4f}, stationary={adf_stationary}")
        except Exception as e:
            print(f"ADF test failed: {e}")
            adf_stationary = False
            adf_pvalue = 1.0
        
        # KPSS Test (null hypothesis: series is stationary)
        try:
            kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss(
                series.dropna(), regression='ct', nlags='auto'
            )
            kpss_stationary = kpss_pvalue > 0.05
            print(f"KPSS: statistic={kpss_stat:.4f}, p-value={kpss_pvalue:.4f}, stationary={kpss_stationary}")
        except Exception as e:
            print(f"KPSS test failed: {e}")
            kpss_stationary = False
            kpss_pvalue = 0.0
        
        # Combined assessment
        both_agree_stationary = adf_stationary and kpss_stationary
        print(f"Both tests agree stationary: {both_agree_stationary}")
        
        results[name] = {
            'adf_pvalue': adf_pvalue,
            'kpss_pvalue': kpss_pvalue,
            'adf_stationary': adf_stationary,
            'kpss_stationary': kpss_stationary,
            'both_stationary': both_agree_stationary
        }
    
    # Determine best differencing strategy
    if results['raw']['both_stationary']:
        best_diff = (0, 0)  # No differencing needed
    elif results['diff']['both_stationary']:
        best_diff = (1, 0)  # Only trend differencing
    elif results['seasonal_diff']['both_stationary']:
        best_diff = (0, 1)  # Only seasonal differencing
    elif results['double_diff']['both_stationary']:
        best_diff = (1, 1)  # Both differencing
    else:
        best_diff = (1, 1)  # Default fallback
    
    print(f"\nRecommended differencing (d, D): {best_diff}")
    
    return {
        'results': results,
        'recommended_d': best_diff[0],
        'recommended_D': best_diff[1]
    }

def acf_pacf_analysis(y: pd.Series, lags: int = 72, save_plots: bool = True) -> dict:
    """
    Generate ACF and PACF plots to identify potential ARIMA orders.
    
    Returns:
        dict: Suggested (p,q) and (P,Q) values based on significant lags
    """
    print("\n=== ACF/PACF ANALYSIS ===")
    
    if not STATSMODELS_AVAILABLE or plot_acf is None or plot_pacf is None:
        raise ImportError("statsmodels.graphics.tsaplots not available - install statsmodels")
    
    # Prepare stationary series (apply recommended differencing)
    unit_root_results = unit_root_tests(y, max_lags=24)
    d_rec = unit_root_results['recommended_d']
    D_rec = unit_root_results['recommended_D']
    
    y_diff = y.copy()
    if d_rec > 0:
        y_diff = y_diff.diff()
    if D_rec > 0:  
        y_diff = y_diff.diff(24)
    y_diff = y_diff.dropna()
    
    print(f"Using differenced series with d={d_rec}, D={D_rec}")
    
    if save_plots:
        DIAGNOSTICS_PATH.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # ACF plot
        plot_acf(y_diff, lags=lags, ax=axes[0], title="Autocorrelation Function (ACF)")
        axes[0].set_xlabel("Lag")
        axes[0].grid(True, alpha=0.3)
        
        # PACF plot  
        plot_pacf(y_diff, lags=lags, ax=axes[1], title="Partial Autocorrelation Function (PACF)")
        axes[1].set_xlabel("Lag")
        axes[1].grid(True, alpha=0.3)
        
        # Add vertical lines at seasonal lags
        for ax in axes:
            ax.axvline(x=24, color='red', linestyle='--', alpha=0.5, label='24h seasonal')
            ax.axvline(x=48, color='red', linestyle='--', alpha=0.3)
            ax.axvline(x=72, color='red', linestyle='--', alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(DIAGNOSTICS_PATH / "acf_pacf_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ACF/PACF plots saved to: {DIAGNOSTICS_PATH / 'acf_pacf_plots.png'}")
    
    # Simple heuristics for order selection (this could be more sophisticated)
    # Look for significant lags in first few positions
    suggestions = {
        'p_candidates': [0, 1, 2, 3],  # PACF cutoff suggests AR order
        'q_candidates': [0, 1, 2, 3],  # ACF cutoff suggests MA order  
        'P_candidates': [0, 1, 2],     # Seasonal AR based on lags 24, 48, etc.
        'Q_candidates': [0, 1, 2],     # Seasonal MA based on seasonal ACF
        'd_recommended': d_rec,
        'D_recommended': D_rec
    }
    
    print("Suggested parameter ranges:")
    print(f"  p (AR): {suggestions['p_candidates']}")
    print(f"  d (differencing): {suggestions['d_recommended']}")
    print(f"  q (MA): {suggestions['q_candidates']}")
    print(f"  P (seasonal AR): {suggestions['P_candidates']}")
    print(f"  D (seasonal differencing): {suggestions['D_recommended']}")
    print(f"  Q (seasonal MA): {suggestions['Q_candidates']}")
    
    return suggestions

def run_diagnostics(y: pd.Series) -> dict:
    """
    Run all pre-modeling diagnostics and return combined recommendations.
    """
    print("Starting SARIMA pre-modeling diagnostics...")
    
    # Step 1: Seasonal-Trend Decomposition
    stl_results = seasonal_trend_decomposition(y, period=24, save_plots=True)
    
    # Step 2: Unit Root Tests
    unit_root_results = unit_root_tests(y, max_lags=24)
    
    # Step 3: ACF/PACF Analysis  
    acf_pacf_results = acf_pacf_analysis(y, lags=72, save_plots=True)
    
    # Combine recommendations
    final_recommendations = {
        'd': max(stl_results['recommend_d'], unit_root_results['recommended_d']),
        'D': max(stl_results['recommend_D'], unit_root_results['recommended_D']),
        'p_range': acf_pacf_results['p_candidates'],
        'q_range': acf_pacf_results['q_candidates'],
        'P_range': acf_pacf_results['P_candidates'],
        'Q_range': acf_pacf_results['Q_candidates']
    }
    
    print("\n=== FINAL RECOMMENDATIONS ===")
    print(f"Differencing: d={final_recommendations['d']}, D={final_recommendations['D']}")
    print(f"AR orders to try: p in {final_recommendations['p_range']}")
    print(f"MA orders to try: q in {final_recommendations['q_range']}")
    print(f"Seasonal AR orders: P in {final_recommendations['P_range']}")
    print(f"Seasonal MA orders: Q in {final_recommendations['Q_range']}")
    
    return {
        'stl': stl_results,
        'unit_root': unit_root_results, 
        'acf_pacf': acf_pacf_results,
        'recommendations': final_recommendations
    }

def train(cfg: dict = None, reload: bool = False, run_diagnostics_flag: bool = None):
    """Train SARIMA model with comprehensive pre-modeling diagnostics"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if MODEL_PATH.exists() and not reload:
        try:
            import pickle
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            print("⚠️  Corrupted model file, retraining...")

    if not STATSMODELS_AVAILABLE:
        warnings.warn("statsmodels not available, cannot train SARIMA", UserWarning)
        return None

    y = load_raw()["electricity_pu"]
    warnings.filterwarnings("ignore")
    
    # Simple logic: if config has specific parameters (p,q,P,Q), use them. Otherwise run diagnostics.
    if cfg and all(key in cfg for key in ['p', 'q', 'P', 'Q']):
        print("🎯 Using specific SARIMA parameters from config...")
        order = (cfg['p'], cfg['d'], cfg['q'])
        seasonal_order = (cfg['P'], cfg['D'], cfg['Q'], 24)
        use_exact_params = True
    else:
        print("🔬 Running diagnostics to find parameters...")
        diagnostics_results = run_diagnostics(y)
        rec = diagnostics_results['recommendations']
        d_start = rec['d']
        D_start = rec['D']
        max_p = max(rec['p_range'])
        max_q = max(rec['q_range'])
        max_P = max(rec['P_range'])
        max_Q = max(rec['Q_range'])
        use_exact_params = False
    
    try:
        if use_exact_params:
            print(f"🎯 Training SARIMA with exact parameters: SARIMA{order}×{seasonal_order}")
        else:
            print(f"🔍 Using auto_arima to find best parameters...")
            if PMDARIMA_AVAILABLE and pm is not None:
                model = pm.auto_arima(
                    y, seasonal=True, m=24, 
                    d=d_start, D=D_start,
                    start_p=0, start_q=0, 
                    max_p=max_p, max_q=max_q,
                    max_P=max_P, max_Q=max_Q,
                    trace=True, error_action="ignore", 
                    stepwise=True, n_fits=50,
                    information_criterion="aicc"
                )
                order, seasonal_order = model.order, model.seasonal_order
                print(f"Selected SARIMA order: {order}")
                print(f"Selected seasonal order: {seasonal_order}")
            else:
                # Fallback if pmdarima not available
                order = (min(2, max_p), d_start, min(2, max_q))
                seasonal_order = (min(1, max_P), D_start, min(1, max_Q), 24)
                print(f"Using SARIMA order: {order}")
                print(f"Using seasonal order: {seasonal_order}")
        
        # Train the model
        sm_model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit()
        
        # Save model using pickle protocol 4 for better compression and compatibility
        import pickle
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(sm_model, f, protocol=4)
        
        print(f"✅ Model trained and saved to: {MODEL_PATH}")
        return sm_model
        
    except Exception as e:
        warnings.warn(f"SARIMA training failed: {e}", UserWarning)
        return None

def predict(model, day: str, horizon: int = 24):
    """Make predictions using SARIMA model"""
    if model is None:
        warnings.warn("No trained SARIMA model available", UserWarning)
        return np.zeros(horizon)
    
    try:
        start = pd.Timestamp(day).tz_localize("Europe/Zurich")
        idx0 = model.data.row_labels.get_loc(start)
        return model.get_prediction(idx0, idx0 + horizon - 1).predicted_mean.values
    except Exception as e:
        warnings.warn(f"SARIMA prediction failed: {e}", UserWarning)
        return np.zeros(horizon)

def tune(cfg: dict = None, n_trials: int = 40):
    """
    Proper SARIMA hyperparameter tuning using grid search over diagnostic-informed ranges.
    Searches for optimal (p,d,q)(P,D,Q,s) combination using AICc criterion.
    """
    print("🔧 SARIMA HYPERPARAMETER TUNING")
    print("=" * 50)
    
    if not STATSMODELS_AVAILABLE:
        warnings.warn("statsmodels not available, cannot tune SARIMA", UserWarning)
        return None
    
    y = load_raw()["electricity_pu"]
    warnings.filterwarnings("ignore")
    
    # Get parameter ranges from diagnostics/config - REDUCED for speed
    if cfg and 'diagnostics' in cfg:
        print("📋 Using diagnostic-informed parameter ranges from config...")
        d_fixed = cfg.get('d', 1)
        D_fixed = cfg.get('D', 1) 
        # Reduced ranges for faster tuning (most SARIMA models work well with lower orders)
        p_range = [0, 1, 2]  # Reduced from [0,1,2,3]
        q_range = [0, 1, 2]  # Reduced from [0,1,2,3]  
        P_range = [0, 1]     # Reduced from [0,1,2]
        Q_range = [0, 1]     # Reduced from [0,1,2]
    else:
        print("🔬 Running diagnostics to get parameter ranges...")
        diagnostics_results = run_diagnostics(y)
        rec = diagnostics_results['recommendations']
        d_fixed = rec['d']
        D_fixed = rec['D']
        # Use reduced ranges regardless of diagnostic recommendations for speed
        p_range = [0, 1, 2]
        q_range = [0, 1, 2]
        P_range = [0, 1]
        Q_range = [0, 1]
    
    print(f"🎯 Parameter search space:")
    print(f"   p (AR): {p_range}")
    print(f"   d (trend diff): {d_fixed} [FIXED from diagnostics]")
    print(f"   q (MA): {q_range}")
    print(f"   P (seasonal AR): {P_range}")
    print(f"   D (seasonal diff): {D_fixed} [FIXED from diagnostics]")
    print(f"   Q (seasonal MA): {Q_range}")
    print(f"   s (seasonal period): 24 [FIXED]")
    
    # Generate all combinations
    from itertools import product
    all_combinations = list(product(p_range, q_range, P_range, Q_range))
    total_models = len(all_combinations)
    print(f"🔍 Total models to evaluate: {total_models}")
    
    # Time series split for validation (use last 20% as validation)
    split_idx = int(len(y) * 0.8)
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:]
    
    print(f"📊 Train set: {len(y_train)} observations")
    print(f"📊 Validation set: {len(y_val)} observations")
    
    best_model = None
    best_aic = float('inf')
    best_params = None
    results = []
    
    print(f"\n🔍 Starting grid search...")
    
    for i, (p, q, P, Q) in enumerate(all_combinations):
        order = (p, d_fixed, q)
        seasonal_order = (P, D_fixed, Q, 24)
        
        try:
            print(f"[{i+1:2d}/{total_models}] Testing SARIMA{order}×{seasonal_order}...", end=" ")
            
            # Fit model on training data
            model = SARIMAX(
                y_train, 
                order=order, 
                seasonal_order=seasonal_order,
                enforce_stationarity=False, 
                enforce_invertibility=False
            ).fit(disp=False)
            
            # Calculate AICc (corrected AIC for finite sample sizes)
            n = len(y_train)
            k = len(model.params)  # number of parameters
            aic = model.aic
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else float('inf')
            
            results.append({
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': aic,
                'aicc': aicc,
                'bic': model.bic,
                'loglik': model.llf
            })
            
            print(f"AICc={aicc:.2f}")
            
            # Track best model
            if aicc < best_aic:
                best_aic = aicc
                best_model = model
                best_params = (order, seasonal_order)
                print(f"    ✅ NEW BEST! AICc={aicc:.2f}")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:50]}...")
            results.append({
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': float('inf'),
                'aicc': float('inf'),
                'bic': float('inf'),
                'loglik': float('-inf'),
                'error': str(e)
            })
            continue
    
    if best_model is None:
        print("❌ No valid models found during tuning!")
        return None
    
    print(f"\n🏆 BEST MODEL FOUND:")
    print(f"   SARIMA{best_params[0]}×{best_params[1]}")
    print(f"   AICc: {best_aic:.4f}")
    print(f"   Parameters: {len(best_model.params)}")
    
    # Retrain best model on full dataset
    print(f"\n🔄 Retraining best model on full dataset...")
    final_model = SARIMAX(
        y, 
        order=best_params[0], 
        seasonal_order=best_params[1],
        enforce_stationarity=False, 
        enforce_invertibility=False
    ).fit(disp=False)
    
    # Save results
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_model, f, protocol=4)
    
    # Save tuning results
    tuning_results = {
        'best_order': best_params[0],
        'best_seasonal_order': best_params[1], 
        'best_aicc': best_aic,
        'all_results': results,
        'search_space': {
            'p_range': p_range,
            'q_range': q_range,
            'P_range': P_range,
            'Q_range': Q_range,
            'd_fixed': d_fixed,
            'D_fixed': D_fixed
        }
    }
    
    joblib.dump(tuning_results, MODEL_PATH.parent / "sarima_tuning_results.pkl")
    
    print(f"💾 Best model saved to: {MODEL_PATH}")
    print(f"📊 Tuning results saved to: {MODEL_PATH.parent / 'sarima_tuning_results.pkl'}")
    
    return final_model 