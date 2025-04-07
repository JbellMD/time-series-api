import pandas as pd
import numpy as np
from prophet import Prophet
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

class ProphetService:
    def __init__(self):
        self.model = None
        self.forecast = None
        self.metrics = {}
    
    def _prepare_data(self, data: Union[List[float], List[Dict[str, Any]]], frequency: str) -> pd.DataFrame:
        """Convert input data to Prophet's required format (ds, y)"""
        if isinstance(data[0], dict):
            # Data is already in dict format with date and value
            df = pd.DataFrame(data)
            df.columns = ['ds', 'y'] if list(df.columns) == ['date', 'value'] else df.columns
        else:
            # Data is a list of values, generate dates
            end_date = datetime.now()
            
            # Generate dates based on frequency
            if frequency == 'D':
                start_date = end_date - timedelta(days=len(data))
                dates = pd.date_range(start=start_date, periods=len(data), freq='D')
            elif frequency == 'W':
                start_date = end_date - timedelta(weeks=len(data))
                dates = pd.date_range(start=start_date, periods=len(data), freq='W')
            elif frequency == 'M':
                start_date = end_date - timedelta(days=30 * len(data))
                dates = pd.date_range(start=start_date, periods=len(data), freq='M')
            elif frequency == 'Q':
                start_date = end_date - timedelta(days=90 * len(data))
                dates = pd.date_range(start=start_date, periods=len(data), freq='Q')
            elif frequency == 'Y':
                start_date = end_date - timedelta(days=365 * len(data))
                dates = pd.date_range(start=start_date, periods=len(data), freq='Y')
            elif frequency == 'H':
                start_date = end_date - timedelta(hours=len(data))
                dates = pd.date_range(start=start_date, periods=len(data), freq='H')
            elif frequency == 'min':
                start_date = end_date - timedelta(minutes=len(data))
                dates = pd.date_range(start=start_date, periods=len(data), freq='min')
            else:
                # Default to daily
                start_date = end_date - timedelta(days=len(data))
                dates = pd.date_range(start=start_date, periods=len(data), freq='D')
            
            df = pd.DataFrame({
                'ds': dates,
                'y': data
            })
        
        return df
    
    def train(self, data: Union[List[float], List[Dict[str, Any]]], frequency: str, 
              seasonal_periods: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> None:
        """Train Prophet model on the provided data"""
        # Prepare data
        df = self._prepare_data(data, frequency)
        
        # Initialize model with default or custom parameters
        model_params = {
            'daily_seasonality': frequency == 'D' or frequency == 'H' or frequency == 'min',
            'weekly_seasonality': frequency == 'D' or frequency == 'W',
            'yearly_seasonality': frequency == 'M' or frequency == 'Q' or frequency == 'Y',
        }
        
        # Update with custom parameters if provided
        if params:
            model_params.update(params)
        
        self.model = Prophet(**model_params)
        
        # Add seasonality if specified
        if seasonal_periods:
            self.model.add_seasonality(
                name='custom_seasonal',
                period=seasonal_periods,
                fourier_order=5
            )
        
        # Fit model
        self.model.fit(df)
    
    def predict(self, periods: int, frequency: str, 
                confidence_interval: float = 0.95) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Generate forecast for the specified number of periods"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=frequency)
        
        # Make prediction
        self.forecast = self.model.predict(future)
        
        # Calculate metrics on training data
        train_data = future.iloc[:-periods]
        if not train_data.empty:
            train_actual = train_data.merge(self.forecast[['ds', 'yhat']], on='ds')
            if 'y' in train_actual.columns:
                # Calculate metrics
                mse = np.mean((train_actual['y'] - train_actual['yhat']) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(train_actual['y'] - train_actual['yhat']))
                mape = np.mean(np.abs((train_actual['y'] - train_actual['yhat']) / train_actual['y'])) * 100
                
                self.metrics = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mape': float(mape)
                }
        
        return self.forecast, self.metrics
    
    def get_components(self) -> Dict[str, List[float]]:
        """Extract trend and seasonality components from the forecast"""
        if self.forecast is None:
            raise ValueError("Forecast must be generated before extracting components")
        
        components = {}
        
        if 'trend' in self.forecast.columns:
            components['trend'] = self.forecast['trend'].tolist()
        
        # Extract seasonality components
        for col in self.forecast.columns:
            if col.startswith('seasonal') or col.endswith('_seasonal'):
                components[col] = self.forecast[col].tolist()
        
        return components
    
    def format_results(self, include_history: bool = False) -> Dict[str, Any]:
        """Format forecast results for API response"""
        if self.forecast is None:
            raise ValueError("Forecast must be generated before formatting results")
        
        # Get forecast period only
        forecast_data = self.forecast.iloc[-self.forecast.shape[0]:]
        
        # Format forecast
        formatted_forecast = []
        for _, row in forecast_data.iterrows():
            formatted_forecast.append({
                'date': row['ds'].strftime('%Y-%m-%d %H:%M:%S'),
                'value': float(row['yhat']),
                'lower_bound': float(row['yhat_lower']),
                'upper_bound': float(row['yhat_upper'])
            })
        
        # Format history if requested
        history = None
        if include_history:
            history_data = self.forecast.iloc[:-self.forecast.shape[0]]
            history = []
            for _, row in history_data.iterrows():
                history.append({
                    'date': row['ds'].strftime('%Y-%m-%d %H:%M:%S'),
                    'value': float(row['yhat']),
                    'lower_bound': float(row['yhat_lower']),
                    'upper_bound': float(row['yhat_upper'])
                })
        
        # Get components
        components = self.get_components()
        
        return {
            'forecast': formatted_forecast,
            'metrics': self.metrics,
            'lower_bound': self.forecast['yhat_lower'].tolist(),
            'upper_bound': self.forecast['yhat_upper'].tolist(),
            'history': history,
            'components': components
        }
