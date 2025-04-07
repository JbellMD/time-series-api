import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import logging

from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMForecastingService:
    """Service for forecasting time series data using LSTM"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = settings.LSTM_LOOKBACK
        self.history = None
        self.last_sequence = None
        self.dates = None
        self.frequency = None
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback), 0])
            y.append(data[i + self.lookback, 0])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model"""
        model = Sequential()
        
        # First LSTM layer with Dropout
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Second LSTM layer with Dropout
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def _prepare_data(self, data: Union[List[float], List[Dict[str, Any]]]) -> Tuple[np.ndarray, Optional[List[datetime]]]:
        """Prepare data for LSTM model"""
        if isinstance(data[0], dict):
            # Data is in dict format with date and value
            df = pd.DataFrame(data)
            
            # Extract dates if available
            dates = None
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date']).tolist()
            
            # Extract values
            if 'value' in df.columns:
                values = df['value'].values
            elif 'close' in df.columns:
                values = df['close'].values
            else:
                # Use the first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    values = df[numeric_cols[0]].values
                else:
                    raise ValueError("No numeric data found in input")
        else:
            # Data is a list of values
            values = np.array(data)
            dates = None
        
        # Reshape and normalize
        values = values.reshape(-1, 1)
        normalized_values = self.scaler.fit_transform(values)
        
        return normalized_values, dates
    
    def train(self, data: Union[List[float], List[Dict[str, Any]]], frequency: str):
        """Train LSTM model on time series data"""
        logger.info("Training LSTM model")
        
        # Prepare data
        normalized_values, dates = self._prepare_data(data)
        self.dates = dates
        self.frequency = frequency
        
        # Create sequences
        X, y = self._create_sequences(normalized_values)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = self._build_model((X.shape[1], 1))
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        self.model.fit(
            X, y,
            epochs=settings.LSTM_EPOCHS,
            batch_size=settings.LSTM_BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save the last sequence for prediction
        self.last_sequence = normalized_values[-self.lookback:].reshape(1, self.lookback, 1)
        
        # Save the original data for confidence intervals
        self.history = normalized_values
        
        logger.info("LSTM model training completed")
    
    def predict(self, periods: int, confidence_interval: float = 0.95) -> Dict[str, Any]:
        """Generate forecasts using trained LSTM model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Generating {periods} forecasts with LSTM model")
        
        # Generate future dates
        future_dates = None
        if self.dates:
            last_date = pd.to_datetime(self.dates[-1])
            if self.frequency == 'D':
                future_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
            elif self.frequency == 'W':
                future_dates = [last_date + timedelta(weeks=i+1) for i in range(periods)]
            elif self.frequency == 'M':
                # Approximate months as 30 days
                future_dates = [last_date + timedelta(days=(i+1)*30) for i in range(periods)]
            elif self.frequency == 'Q':
                # Approximate quarters as 90 days
                future_dates = [last_date + timedelta(days=(i+1)*90) for i in range(periods)]
            elif self.frequency == 'Y':
                # Approximate years as 365 days
                future_dates = [last_date + timedelta(days=(i+1)*365) for i in range(periods)]
            elif self.frequency == 'H':
                future_dates = [last_date + timedelta(hours=i+1) for i in range(periods)]
            else:
                # Default to daily
                future_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        # Generate predictions
        predictions = []
        current_sequence = self.last_sequence.copy()
        
        for _ in range(periods):
            # Predict next value
            next_value = self.model.predict(current_sequence)[0, 0]
            predictions.append(next_value)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[:, 1:, :], [[next_value]], axis=1)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        # Calculate confidence intervals
        # For LSTM, we'll use a bootstrap approach to estimate uncertainty
        n_bootstraps = 100
        bootstrap_predictions = np.zeros((n_bootstraps, periods))
        
        for i in range(n_bootstraps):
            # Create a bootstrap sample of the history
            bootstrap_indices = np.random.choice(
                len(self.history) - self.lookback,
                size=len(self.history) - self.lookback,
                replace=True
            )
            
            # Create sequences from bootstrap sample
            bootstrap_X = np.array([self.history[j:j+self.lookback, 0] for j in bootstrap_indices])
            bootstrap_y = np.array([self.history[j+self.lookback, 0] for j in bootstrap_indices])
            
            # Reshape for LSTM
            bootstrap_X = np.reshape(bootstrap_X, (bootstrap_X.shape[0], bootstrap_X.shape[1], 1))
            
            # Train a bootstrap model
            bootstrap_model = self._build_model((bootstrap_X.shape[1], 1))
            bootstrap_model.fit(
                bootstrap_X, bootstrap_y,
                epochs=10,  # Fewer epochs for bootstrap models
                batch_size=32,
                verbose=0
            )
            
            # Generate predictions with bootstrap model
            current_sequence = self.last_sequence.copy()
            for j in range(periods):
                next_value = bootstrap_model.predict(current_sequence)[0, 0]
                bootstrap_predictions[i, j] = next_value
                current_sequence = np.append(current_sequence[:, 1:, :], [[next_value]], axis=1)
        
        # Calculate confidence intervals
        lower_percentile = (1 - confidence_interval) / 2
        upper_percentile = 1 - lower_percentile
        
        lower_bound = np.percentile(bootstrap_predictions, lower_percentile * 100, axis=0)
        upper_bound = np.percentile(bootstrap_predictions, upper_percentile * 100, axis=0)
        
        # Inverse transform bounds
        lower_bound = self.scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
        upper_bound = self.scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
        
        # Format results
        forecast_data = []
        for i in range(periods):
            forecast_point = {
                "value": float(predictions[i, 0]),
                "lower_bound": float(lower_bound[i]),
                "upper_bound": float(upper_bound[i])
            }
            
            if future_dates:
                forecast_point["date"] = future_dates[i].strftime("%Y-%m-%d")
            
            forecast_data.append(forecast_point)
        
        # Calculate metrics on training data
        train_predictions = self.model.predict(
            np.reshape(self._create_sequences(self.history)[0], (-1, self.lookback, 1))
        )
        train_predictions = self.scaler.inverse_transform(train_predictions)
        
        actual_values = self.scaler.inverse_transform(
            self.history[self.lookback:].reshape(-1, 1)
        )
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((train_predictions - actual_values) ** 2))
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual_values - train_predictions) / actual_values)) * 100
        
        return {
            "forecast": forecast_data,
            "model": "lstm",
            "metrics": {
                "rmse": float(rmse),
                "mape": float(mape)
            },
            "parameters": {
                "lookback": self.lookback,
                "confidence_interval": confidence_interval
            }
        }
