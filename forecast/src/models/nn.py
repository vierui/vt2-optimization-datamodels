"""
Neural Network model for PV forecasting.
Supports hyperparameter tuning and manual configuration.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Tuple, Optional, Any
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class PVNeuralNet:
    """Neural Network forecaster for PV generation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.nn_config = config['neural_net']
        self.forecast_length = config['forecast']['forecast_length']
        # length of historical window for convolutional model
        self.history_length = config['forecast'].get('history_length', 24)
        
        self.model = None
        self.history = None
        self.best_params = None
        self.feature_scaler = None
        self.target_scaler = None
        
    def quantile_loss(self, tau: float = 0.5):
        """Quantile (pinball) loss for given tau."""

        def loss(y_true, y_pred):
            err = y_true - y_pred
            return tf.reduce_mean(tf.maximum(tau * err, (tau - 1) * err))

        return loss

    def build_model(self, input_shape: Tuple[int, int], hyperparams: Dict = None) -> keras.Model:
        """Build Temporal Convolutional Network."""

        if hyperparams is None:
            hyperparams = self.nn_config['manual']

        inputs = layers.Input(shape=input_shape)
        x = inputs
        for dilation in hyperparams.get('dilations', [1, 2, 4, 8]):
            residual = x
            x = layers.Conv1D(
                filters=hyperparams['filters'],
                kernel_size=hyperparams['kernel_size'],
                dilation_rate=dilation,
                padding='causal',
                activation='relu'
            )(x)
            x = layers.Dropout(hyperparams['dropout'])(x)
            if residual.shape[-1] == x.shape[-1]:
                x = layers.Add()([x, residual])

        x = layers.Flatten()(x)
        outputs = layers.Dense(self.forecast_length)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        loss_fn = 'mae'
        if self.nn_config.get('loss', 'mae') == 'quantile':
            loss_fn = self.quantile_loss(0.5)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
            loss=loss_fn,
            metrics=['mae']
        )

        return model
    
    def create_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create history and forecast windows for TCN training."""

        X_sequences: list[np.ndarray] = []
        y_sequences: list[np.ndarray] = []

        total_length = len(X)
        for i in range(total_length - self.history_length - self.forecast_length + 1):
            X_seq = X.iloc[i : i + self.history_length].values
            y_seq = y.iloc[i + self.history_length : i + self.history_length + self.forecast_length].values
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

        return np.array(X_sequences), np.array(y_sequences)
    
    def prepare_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> Tuple:
        """
        Prepare data for neural network training.
        
        Args:
            X_train, y_train: Training features and target
            X_val, y_val: Validation features and target
            
        Returns:
            Tuple of prepared training and validation data
        """
        logger.info("Preparing data for neural network")
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
        
        logger.info(f"Training sequences: {X_train_seq.shape}")
        logger.info(f"Validation sequences: {X_val_seq.shape}")
        
        return X_train_seq, y_train_seq, X_val_seq, y_val_seq
    
    def train_manual(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Train model with manual hyperparameters.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Training results dictionary
        """
        logger.info("Training neural network with manual parameters")
        
        # Prepare data
        X_train_seq, y_train_seq, X_val_seq, y_val_seq = self.prepare_data(
            X_train, y_train, X_val, y_val
        )
        
        # Build model
        self.model = self.build_model(X_train_seq.shape[1:])
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=self.nn_config['early_stopping']['monitor'],
                patience=self.nn_config['early_stopping']['patience'],
                restore_best_weights=self.nn_config['early_stopping']['restore_best_weights']
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.nn_config['manual']['epochs'],
            batch_size=self.nn_config['manual']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Store best parameters
        self.best_params = self.nn_config['manual'].copy()
        
        return {
            'model': self.model,
            'history': self.history.history,
            'best_params': self.best_params
        }


class HyperModel(kt.HyperModel):
    """Hypermodel for Keras Tuner."""

    def __init__(self, config: Dict, input_shape: Tuple[int, int]):
        self.config = config
        self.input_shape = input_shape
        self.forecast_length = config['forecast']['forecast_length']

    def build(self, hp):
        """Build TCN model for hyperparameter tuning."""

        filters = hp.Choice('filters', self.config['neural_net']['search_space']['filters'])
        kernel_size = hp.Choice('kernel_size', self.config['neural_net']['search_space']['kernel_size'])
        dropout = hp.Choice('dropout', self.config['neural_net']['search_space']['dropout'])
        learning_rate = hp.Choice('learning_rate', self.config['neural_net']['search_space']['learning_rate'])

        inputs = keras.layers.Input(shape=self.input_shape)
        x = inputs
        for dilation in self.config['neural_net']['search_space'].get('dilations', [1, 2, 4, 8]):
            residual = x
            x = keras.layers.Conv1D(filters=filters,
                                    kernel_size=kernel_size,
                                    dilation_rate=dilation,
                                    padding='causal',
                                    activation='relu')(x)
            x = keras.layers.Dropout(dropout)(x)
            if residual.shape[-1] == x.shape[-1]:
                x = keras.layers.Add()([x, residual])

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(self.forecast_length)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        loss_fn = 'mae'
        if self.config['neural_net'].get('loss', 'mae') == 'quantile':
            loss_fn = PVNeuralNet(self.config).quantile_loss(0.5)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=['mae']
        )

        return model


def train_nn(config: Dict, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series, search: bool = True) -> Dict:
    """
    Train neural network with or without hyperparameter search.
    
    Args:
        config: Configuration dictionary
        X_train, y_train: Training data
        X_val, y_val: Validation data
        search: Whether to perform hyperparameter search
        
    Returns:
        Training results dictionary
    """
    nn_forecaster = PVNeuralNet(config)
    
    if search:
        return train_with_search(config, nn_forecaster, X_train, y_train, X_val, y_val)
    else:
        return nn_forecaster.train_manual(X_train, y_train, X_val, y_val)


def train_with_search(config: Dict, nn_forecaster: PVNeuralNet,
                     X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
    """
    Train neural network with hyperparameter search.
    
    Args:
        config: Configuration dictionary
        nn_forecaster: Neural network instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Training results dictionary
    """
    logger.info("Training neural network with hyperparameter search")
    
    # Prepare data
    X_train_seq, y_train_seq, X_val_seq, y_val_seq = nn_forecaster.prepare_data(
        X_train, y_train, X_val, y_val
    )
    
    # Setup hyperparameter tuner
    tuner = kt.RandomSearch(
        HyperModel(config, X_train_seq.shape[1:]),
        objective='val_loss',
        max_trials=config['neural_net']['tuner']['max_trials'],
        executions_per_trial=config['neural_net']['tuner']['executions_per_trial'],
        directory='../models',
        project_name='pv_forecasting_tuner'
    )
    
    # Setup callbacks for tuning
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['neural_net']['early_stopping']['patience'],
            restore_best_weights=True
        )
    ]
    
    # Search for best hyperparameters
    logger.info("Starting hyperparameter search...")
    tuner.search(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,  # Reduced epochs for search
        callbacks=callbacks,
        verbose=1
    )
    
    # Get best model and parameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Store results
    nn_forecaster.model = best_model
    nn_forecaster.best_params = {
        'filters': best_hps.get('filters'),
        'kernel_size': best_hps.get('kernel_size'),
        'dropout': best_hps.get('dropout'),
        'learning_rate': best_hps.get('learning_rate')
    }
    
    logger.info(f"Best hyperparameters: {nn_forecaster.best_params}")
    
    return {
        'model': nn_forecaster.model,
        'tuner': tuner,
        'best_params': nn_forecaster.best_params
    }


def forecast_nn(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Generate forecasts using trained neural network.
    
    Args:
        model: Trained Keras model
        X: Input features
        
    Returns:
        Forecast array
    """
    predictions = model.predict(X, verbose=0)
    return np.maximum(0, predictions)  # Ensure non-negative


def evaluate_nn_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Evaluate neural network forecast performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Flatten arrays for evaluation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
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