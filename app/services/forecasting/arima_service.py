import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

class ARIMAService:
    def __init__(self):
        self.model = None
        self.forecast = None
        self.results = None
        self.metrics = {}
        self.data_df = None
        self.is_seasonal = False
    
    def _prepare_data(self, data: Union[List[float], List[Dict[str, Any]]], frequency: str) -> pd.DataFrame:
        """Convert input data to time series format"""
        if isinstance(data[0], dict):
            # Data is already in dict format with date and value
            df = pd.DataFrame(data)
            df.columns = ['ds', 'y'] if list(df.columns) == ['date', 'value'] else df.columns
            df.set_index('ds', inplace=True)
            df.index = pd.to_datetime(df.index)
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
                'y': data
            }, index=dates)
        
        return df
    
    def _check_stationarity(self, data: pd.Series) -> Tuple[bool, int]:
        """Check if time series is stationary and suggest differencing order"""
        result = adfuller(data)
        p_value = result[1]
        
        # If p-value > 0.05, the series is non-stationary
        if p_value > 0.05:
            # Try differencing and check again
            for d in range(1, 3):
                diff_data = data.diff(d).dropna()
                diff_result = adfuller(diff_data)
                if diff_result[1] <= 0.05:
                    return False, d
            return False, 1  # Default to first order differencing if still not stationary
        
        return True, 0
    
    def _determine_order(self, data: pd.Series, seasonal_periods: Optional[int] = None) -> Dict[str, int]:
        """Determine optimal order for ARIMA/SARIMA model"""
        # Check stationarity and get differencing order
        is_stationary, d = self._check_stationarity(data)
        
        # Simple heuristic for p and q
        p = min(5, len(data) // 10)
        q = min(5, len(data) // 10)
        
        order = (p, d, q)
        
        # For seasonal model
        seasonal_order = None
        if seasonal_periods:
            self.is_seasonal = True
            # Simple heuristic for seasonal components
            P = min(2, len(data) // (seasonal_periods * 2))
            D = 1  # Usually 1 is sufficient for seasonal differencing
            Q = min(2, len(data) // (seasonal_periods * 2))
            seasonal_order = (P, D, Q, seasonal_periods)
        
        return {
            'order': order,
            'seasonal_order': seasonal_order
        }
    
    def train(self, data: Union[List[float], List[Dict[str, Any]]], frequency: str, 
              seasonal_periods: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> None:
        """Train ARIMA/SARIMA model on the provided data"""
        # Prepare data
        self.data_df = self._prepare_data(data, frequency)
        
        # Determine model order if not provided
        if params and 'order' in params:
            order = params['order']
            seasonal_order = params.get('seasonal_order')
            self.is_seasonal = seasonal_order is not None
        else:
            order_params = self._determine_order(self.data_df['y'], seasonal_periods)
            order = order_params['order']
            seasonal_order = order_params['seasonal_order']
        
        # Train model
        if self.is_seasonal:
            self.model = SARIMAX(
                self.data_df['y'],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            self.model = ARIMA(
                self.data_df['y'],
                order=order
            )
        
        self.results = self.model.fit()
        
        # Calculate metrics
        self.metrics = {
            'aic': float(self.results.aic),
            'bic': float(self.results.bic),
            'mse': float(np.mean(self.results.resid ** 2)),
            'rmse': float(np.sqrt(np.mean(self.results.resid ** 2))),
            'mae': float(np.mean(np.abs(self.results.resid))),
        }
        
        if not np.any(np.isnan(self.data_df['y'])):
            self.metrics['mape'] = float(np.mean(np.abs(self.results.resid / self.data_df['y'])) * 100)
    
    def predict(self, periods: int, frequency: str, 
                confidence_interval: float = 0.95) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Generate forecast for the specified number of periods"""
        if self.results is None:
            raise ValueError("Model must be trained before prediction")
        
        # Generate forecast
        forecast = self.results.get_forecast(steps=periods)
        
        # Get confidence intervals
        conf_int = forecast.conf_int(alpha=1-confidence_interval)
        
        # Create forecast dataframe
        last_date = self.data_df.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=self._get_freq_unit(frequency)),
            periods=periods,
            freq=self._get_freq_str(frequency)
        )
        
        forecast_mean = forecast.predicted_mean
        
        self.forecast = pd.DataFrame({
            'ds': forecast_index,
            'yhat': forecast_mean,
            'yhat_lower': conf_int.iloc[:, 0].values,
            'yhat_upper': conf_int.iloc[:, 1].values
        })
        
        return self.forecast, self.metrics
    
    def _get_freq_unit(self, frequency: str) -> str:
        """Convert API frequency to pandas timedelta unit"""
        freq_map = {
            'D': 'days',
            'W': 'weeks',
            'M': 'months',
            'Q': 'days',  # Approximate
            'Y': 'days',  # Approximate
            'H': 'hours',
            'min': 'minutes'
        }
        return freq_map.get(frequency, 'days')
    
    def _get_freq_str(self, frequency: str) -> str:
        """Convert API frequency to pandas frequency string"""
        freq_map = {
            'D': 'D',
            'W': 'W',
            'M': 'M',
            'Q': 'Q',
            'Y': 'Y',
            'H': 'H',
            'min': 'min'
        }
        return freq_map.get(frequency, 'D')
    
    def get_components(self) -> Dict[str, List[float]]:
        """Extract trend and seasonality components from the forecast"""
        if self.results is None:
            raise ValueError("Model must be trained before extracting components")
        
        components = {}
        
        # For SARIMA, we can decompose the time series
        if self.is_seasonal:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomposition = seasonal_decompose(self.data_df['y'], model='additive')
                
                components['trend'] = decomposition.trend.fillna(method='bfill').fillna(method='ffill').tolist()
                components['seasonal'] = decomposition.seasonal.fillna(method='bfill').fillna(method='ffill').tolist()
                components['residual'] = decomposition.resid.fillna(method='bfill').fillna(method='ffill').tolist()
            except:
                # If decomposition fails, return empty components
                pass
        
        return components
    
    def format_results(self, include_history: bool = False) -> Dict[str, Any]:
        """Format forecast results for API response"""
        if self.forecast is None:
            raise ValueError("Forecast must be generated before formatting results")
        
        # Format forecast
        formatted_forecast = []
        for _, row in self.forecast.iterrows():
            formatted_forecast.append({
                'date': row['ds'].strftime('%Y-%m-%d %H:%M:%S'),
                'value': float(row['yhat']),
                'lower_bound': float(row['yhat_lower']),
                'upper_bound': float(row['yhat_upper'])
            })
        
        # Format history if requested
        history = None
        if include_history and self.data_df is not None:
            history = []
            for date, row in self.data_df.iterrows():
                history.append({
                    'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'value': float(row['y'])
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
