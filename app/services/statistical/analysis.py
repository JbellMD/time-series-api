import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import scipy.stats as stats

class StatisticalAnalysisService:
    """Service for performing statistical analysis on time series data"""
    
    def __init__(self):
        self.data = None
    
    def _prepare_data(self, data: Union[List[float], List[Dict[str, Any]]]) -> pd.Series:
        """Convert input data to pandas Series"""
        if isinstance(data[0], dict):
            # Data is in dict format with date and value
            df = pd.DataFrame(data)
            if 'value' in df.columns:
                series = df['value']
            elif 'close' in df.columns:
                series = df['close']
            else:
                # Use the first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    series = df[numeric_cols[0]]
                else:
                    raise ValueError("No numeric data found in input")
        else:
            # Data is a list of values
            series = pd.Series(data)
        
        return series
    
    def test_stationarity(self, data: List[float], test: str = 'adf', alpha: float = 0.05) -> Dict[str, Any]:
        """Test stationarity of time series using specified test"""
        series = self._prepare_data(data)
        
        if test == 'adf':
            # Augmented Dickey-Fuller test
            result = adfuller(series, autolag='AIC')
            statistic, p_value, _, _, critical_values, _ = result
            
            # Determine if stationary
            is_stationary = p_value < alpha
            
            # Suggest differencing if non-stationary
            suggested_differencing = None
            if not is_stationary:
                # Try differencing and test again
                for d in range(1, 3):
                    diff_series = series.diff(d).dropna()
                    diff_result = adfuller(diff_series, autolag='AIC')
                    if diff_result[1] < alpha:
                        suggested_differencing = d
                        break
            
            return {
                'test': 'adf',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'critical_values': {k.replace('%', ''): float(v) for k, v in critical_values.items()},
                'is_stationary': is_stationary,
                'suggested_differencing': suggested_differencing
            }
        
        elif test == 'kpss':
            # KPSS test
            result = kpss(series, regression='c', nlags='auto')
            statistic, p_value, _, critical_values = result
            
            # For KPSS, null hypothesis is that the series is stationary
            # So p_value > alpha means stationary
            is_stationary = p_value > alpha
            
            # Suggest differencing if non-stationary
            suggested_differencing = None
            if not is_stationary:
                # Try differencing and test again
                for d in range(1, 3):
                    diff_series = series.diff(d).dropna()
                    diff_result = kpss(diff_series, regression='c', nlags='auto')
                    if diff_result[1] > alpha:
                        suggested_differencing = d
                        break
            
            return {
                'test': 'kpss',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'critical_values': {k.replace('%', ''): float(v) for k, v in critical_values.items()},
                'is_stationary': is_stationary,
                'suggested_differencing': suggested_differencing
            }
        
        elif test == 'pp':
            # Phillips-Perron test
            from statsmodels.tsa.stattools import adfuller
            # PP test is not directly available in statsmodels, use ADF as an approximation
            result = adfuller(series, autolag='AIC')
            statistic, p_value, _, _, critical_values, _ = result
            
            # Determine if stationary
            is_stationary = p_value < alpha
            
            # Suggest differencing if non-stationary
            suggested_differencing = None
            if not is_stationary:
                # Try differencing and test again
                for d in range(1, 3):
                    diff_series = series.diff(d).dropna()
                    diff_result = adfuller(diff_series, autolag='AIC')
                    if diff_result[1] < alpha:
                        suggested_differencing = d
                        break
            
            return {
                'test': 'pp',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'critical_values': {k.replace('%', ''): float(v) for k, v in critical_values.items()},
                'is_stationary': is_stationary,
                'suggested_differencing': suggested_differencing
            }
        
        else:
            raise ValueError(f"Unsupported stationarity test: {test}")
    
    def detect_anomalies(self, data: Union[List[float], List[Dict[str, Any]]], 
                        method: str = 'iqr', sensitivity: float = 1.5,
                        window_size: Optional[int] = None) -> Dict[str, Any]:
        """Detect anomalies in time series using specified method"""
        series = self._prepare_data(data)
        
        if method == 'iqr':
            # Interquartile Range method
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - sensitivity * iqr
            upper_bound = q3 + sensitivity * iqr
            
            # Detect anomalies
            anomalies = series[(series < lower_bound) | (series > upper_bound)]
            
            # Format results
            anomaly_list = [
                {
                    'index': int(i),
                    'value': float(v),
                    'type': 'low' if v < lower_bound else 'high'
                }
                for i, v in anomalies.items()
            ]
            
            return {
                'anomalies': anomaly_list,
                'count': len(anomaly_list),
                'method': 'iqr',
                'threshold': float(sensitivity),
                'bounds': {
                    'lower': float(lower_bound),
                    'upper': float(upper_bound)
                }
            }
        
        elif method == 'z_score':
            # Z-score method
            mean = series.mean()
            std = series.std()
            
            z_scores = (series - mean) / std
            
            # Detect anomalies
            anomalies = series[abs(z_scores) > sensitivity]
            
            # Format results
            anomaly_list = [
                {
                    'index': int(i),
                    'value': float(v),
                    'z_score': float(z_scores[i]),
                    'type': 'low' if v < mean else 'high'
                }
                for i, v in anomalies.items()
            ]
            
            return {
                'anomalies': anomaly_list,
                'count': len(anomaly_list),
                'method': 'z_score',
                'threshold': float(sensitivity),
                'stats': {
                    'mean': float(mean),
                    'std': float(std)
                }
            }
        
        elif method == 'isolation_forest':
            # Isolation Forest method
            X = series.values.reshape(-1, 1)
            
            # Train model
            model = IsolationForest(
                contamination=min(0.1, max(0.01, 1.0 / sensitivity)),
                random_state=42
            )
            
            # Predict anomalies
            predictions = model.fit_predict(X)
            scores = model.decision_function(X)
            
            # Anomalies have prediction -1
            anomaly_indices = np.where(predictions == -1)[0]
            
            # Format results
            anomaly_list = [
                {
                    'index': int(i),
                    'value': float(series.iloc[i]),
                    'score': float(scores[i]),
                    'type': 'outlier'
                }
                for i in anomaly_indices
            ]
            
            return {
                'anomalies': anomaly_list,
                'count': len(anomaly_list),
                'method': 'isolation_forest',
                'threshold': float(sensitivity)
            }
        
        elif method == 'local_outlier_factor':
            # Local Outlier Factor method
            X = series.values.reshape(-1, 1)
            
            # Train model
            model = LocalOutlierFactor(
                n_neighbors=min(20, len(X) // 2),
                contamination=min(0.1, max(0.01, 1.0 / sensitivity))
            )
            
            # Predict anomalies
            predictions = model.fit_predict(X)
            
            # Anomalies have prediction -1
            anomaly_indices = np.where(predictions == -1)[0]
            
            # Format results
            anomaly_list = [
                {
                    'index': int(i),
                    'value': float(series.iloc[i]),
                    'type': 'outlier'
                }
                for i in anomaly_indices
            ]
            
            return {
                'anomalies': anomaly_list,
                'count': len(anomaly_list),
                'method': 'local_outlier_factor',
                'threshold': float(sensitivity)
            }
        
        elif method == 'moving_average':
            # Moving Average method
            if window_size is None:
                window_size = max(5, len(series) // 10)
            
            # Calculate moving average
            ma = series.rolling(window=window_size).mean()
            
            # Calculate standard deviation of residuals
            residuals = series - ma
            std = residuals.std()
            
            # Detect anomalies
            anomalies = series[abs(residuals) > sensitivity * std]
            
            # Format results
            anomaly_list = [
                {
                    'index': int(i),
                    'value': float(v),
                    'ma_value': float(ma[i]),
                    'deviation': float(v - ma[i]),
                    'type': 'low' if v < ma[i] else 'high'
                }
                for i, v in anomalies.items() if not np.isnan(ma[i])
            ]
            
            return {
                'anomalies': anomaly_list,
                'count': len(anomaly_list),
                'method': 'moving_average',
                'threshold': float(sensitivity),
                'window_size': window_size
            }
        
        elif method == 'prophet':
            # Prophet method (placeholder)
            # In a real implementation, you would use Prophet to detect anomalies
            return {
                'anomalies': [],
                'count': 0,
                'method': 'prophet',
                'threshold': float(sensitivity),
                'error': 'Prophet method not implemented yet'
            }
        
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
    
    def calculate_correlation(self, data_x: List[float], data_y: List[float], 
                             method: str = 'pearson', max_lag: Optional[int] = None) -> Dict[str, Any]:
        """Calculate correlation between two time series"""
        # Convert to pandas Series
        series_x = pd.Series(data_x)
        series_y = pd.Series(data_y)
        
        if method == 'pearson':
            # Pearson correlation
            corr, p_value = stats.pearsonr(series_x, series_y)
            
            return {
                'coefficient': float(corr),
                'p_value': float(p_value),
                'method': 'pearson'
            }
        
        elif method == 'spearman':
            # Spearman correlation
            corr, p_value = stats.spearmanr(series_x, series_y)
            
            return {
                'coefficient': float(corr),
                'p_value': float(p_value),
                'method': 'spearman'
            }
        
        elif method == 'kendall':
            # Kendall correlation
            corr, p_value = stats.kendalltau(series_x, series_y)
            
            return {
                'coefficient': float(corr),
                'p_value': float(p_value),
                'method': 'kendall'
            }
        
        elif method == 'cross_correlation':
            # Cross-correlation
            if max_lag is None:
                max_lag = len(series_x) // 2
            
            # Calculate cross-correlation
            cross_corr = [
                float(np.corrcoef(series_x[lag:], series_y[:-lag if lag > 0 else len(series_y)])[0, 1])
                if lag > 0 else
                float(np.corrcoef(series_x, series_y)[0, 1])
                if lag == 0 else
                float(np.corrcoef(series_x[:lag], series_y[-lag:])[0, 1])
                for lag in range(-max_lag, max_lag + 1)
            ]
            
            # Find lag with highest correlation
            best_lag = range(-max_lag, max_lag + 1)[np.argmax(np.abs(cross_corr))]
            
            return {
                'coefficient': float(cross_corr[max_lag]),  # Correlation at lag 0
                'method': 'cross_correlation',
                'lags': list(range(-max_lag, max_lag + 1)),
                'correlations': [float(c) for c in cross_corr],
                'best_lag': int(best_lag)
            }
        
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
    
    def analyze_acf_pacf(self, data: List[float], max_lag: int = 40, alpha: float = 0.05) -> Dict[str, Any]:
        """Calculate Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)"""
        series = self._prepare_data(data)
        
        # Calculate ACF and PACF
        acf_values = acf(series, nlags=max_lag, alpha=alpha)
        pacf_values = pacf(series, nlags=max_lag, alpha=alpha)
        
        # Extract confidence intervals
        acf_ci = acf_values[1]
        pacf_ci = pacf_values[1]
        
        # Format results
        result = {
            'acf': {
                'values': [float(v) for v in acf_values[0]],
                'confidence_intervals': [
                    [float(ci[0]), float(ci[1])]
                    for ci in acf_ci
                ] if acf_ci is not None else None
            },
            'pacf': {
                'values': [float(v) for v in pacf_values[0]],
                'confidence_intervals': [
                    [float(ci[0]), float(ci[1])]
                    for ci in pacf_ci
                ] if pacf_ci is not None else None
            },
            'lags': list(range(max_lag + 1)),
            'suggested_ar_order': self._suggest_ar_order(pacf_values[0]),
            'suggested_ma_order': self._suggest_ma_order(acf_values[0])
        }
        
        return result
    
    def _suggest_ar_order(self, pacf_values: np.ndarray) -> int:
        """Suggest AR order based on PACF values"""
        # Use 95% confidence interval (approximately 2/sqrt(n))
        threshold = 2 / np.sqrt(len(pacf_values))
        
        # Find the lag where PACF drops below threshold
        for i in range(1, len(pacf_values)):
            if abs(pacf_values[i]) < threshold:
                return i - 1
        
        # If all values are significant, suggest a reasonable maximum
        return min(5, len(pacf_values) // 4)
    
    def _suggest_ma_order(self, acf_values: np.ndarray) -> int:
        """Suggest MA order based on ACF values"""
        # Use 95% confidence interval (approximately 2/sqrt(n))
        threshold = 2 / np.sqrt(len(acf_values))
        
        # Find the lag where ACF drops below threshold
        for i in range(1, len(acf_values)):
            if abs(acf_values[i]) < threshold:
                return i - 1
        
        # If all values are significant, suggest a reasonable maximum
        return min(5, len(acf_values) // 4)
