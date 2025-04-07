import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
import pandas_ta as ta

class TechnicalIndicatorService:
    """Service for calculating technical indicators on time series data"""
    
    def __init__(self):
        self.data = None
    
    def _prepare_data(self, data: Union[List[float], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Convert input data to pandas DataFrame"""
        if isinstance(data[0], dict):
            # Check if it's OHLCV data
            if all(key in data[0] for key in ['open', 'high', 'low', 'close']):
                df = pd.DataFrame(data)
                # Check if 'date' or 'timestamp' exists
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                elif 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
            else:
                # Simple time series with date and value
                df = pd.DataFrame(data)
                df.columns = ['date', 'close'] if list(df.columns) == ['date', 'value'] else df.columns
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
        else:
            # Simple list of values
            df = pd.DataFrame({'close': data})
        
        return df
    
    def calculate_sma(self, data: Union[List[float], List[Dict[str, Any]]], window: int) -> Dict[str, Any]:
        """Calculate Simple Moving Average"""
        df = self._prepare_data(data)
        
        # Calculate SMA
        df['sma'] = df['close'].rolling(window=window).mean()
        
        # Format results
        result = {
            'values': df['sma'].dropna().tolist(),
            'metadata': {
                'window': window,
                'description': f'Simple Moving Average with window {window}'
            }
        }
        
        return result
    
    def calculate_ema(self, data: Union[List[float], List[Dict[str, Any]]], window: int) -> Dict[str, Any]:
        """Calculate Exponential Moving Average"""
        df = self._prepare_data(data)
        
        # Calculate EMA
        df['ema'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Format results
        result = {
            'values': df['ema'].dropna().tolist(),
            'metadata': {
                'window': window,
                'description': f'Exponential Moving Average with window {window}'
            }
        }
        
        return result
    
    def calculate_macd(self, data: Union[List[float], List[Dict[str, Any]]], 
                      fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        df = self._prepare_data(data)
        
        # Calculate MACD using pandas_ta
        macd = ta.macd(df['close'], fast=fast_period, slow=slow_period, signal=signal_period)
        
        # Format results
        result = {
            'values': [
                {
                    'macd': float(macd['MACD_' + str(fast_period) + '_' + str(slow_period) + '_' + str(signal_period)][i]),
                    'signal': float(macd['MACDs_' + str(fast_period) + '_' + str(slow_period) + '_' + str(signal_period)][i]),
                    'histogram': float(macd['MACDh_' + str(fast_period) + '_' + str(slow_period) + '_' + str(signal_period)][i])
                }
                for i in range(len(macd)) if not np.isnan(macd['MACD_' + str(fast_period) + '_' + str(slow_period) + '_' + str(signal_period)][i])
            ],
            'metadata': {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period,
                'description': f'MACD with fast={fast_period}, slow={slow_period}, signal={signal_period}'
            }
        }
        
        return result
    
    def calculate_rsi(self, data: Union[List[float], List[Dict[str, Any]]], window: int = 14) -> Dict[str, Any]:
        """Calculate RSI (Relative Strength Index)"""
        df = self._prepare_data(data)
        
        # Calculate RSI using pandas_ta
        rsi = ta.rsi(df['close'], length=window)
        
        # Format results
        result = {
            'values': rsi.dropna().tolist(),
            'metadata': {
                'window': window,
                'description': f'Relative Strength Index with window {window}'
            }
        }
        
        return result
    
    def calculate_bbands(self, data: Union[List[float], List[Dict[str, Any]]], 
                        window: int = 20, std_dev: float = 2.0) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        df = self._prepare_data(data)
        
        # Calculate Bollinger Bands using pandas_ta
        bbands = ta.bbands(df['close'], length=window, std=std_dev)
        
        # Format results
        result = {
            'values': [
                {
                    'upper': float(bbands['BBU_' + str(window) + '_' + str(std_dev)][i]),
                    'middle': float(bbands['BBM_' + str(window) + '_' + str(std_dev)][i]),
                    'lower': float(bbands['BBL_' + str(window) + '_' + str(std_dev)][i])
                }
                for i in range(len(bbands)) if not np.isnan(bbands['BBM_' + str(window) + '_' + str(std_dev)][i])
            ],
            'metadata': {
                'window': window,
                'std_dev': std_dev,
                'description': f'Bollinger Bands with window {window} and std_dev {std_dev}'
            }
        }
        
        return result
    
    def calculate_atr(self, data: List[Dict[str, Any]], window: int = 14) -> Dict[str, Any]:
        """Calculate Average True Range"""
        df = self._prepare_data(data)
        
        # Calculate ATR using pandas_ta
        atr = ta.atr(df['high'], df['low'], df['close'], length=window)
        
        # Format results
        result = {
            'values': atr.dropna().tolist(),
            'metadata': {
                'window': window,
                'description': f'Average True Range with window {window}'
            }
        }
        
        return result
    
    def calculate_stoch(self, data: List[Dict[str, Any]], k_period: int = 14, 
                       d_period: int = 3, smooth_k: int = 3) -> Dict[str, Any]:
        """Calculate Stochastic Oscillator"""
        df = self._prepare_data(data)
        
        # Calculate Stochastic Oscillator using pandas_ta
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=k_period, d=d_period, smooth_k=smooth_k)
        
        # Format results
        result = {
            'values': [
                {
                    'k': float(stoch['STOCHk_' + str(k_period) + '_' + str(d_period) + '_' + str(smooth_k)][i]),
                    'd': float(stoch['STOCHd_' + str(k_period) + '_' + str(d_period) + '_' + str(smooth_k)][i])
                }
                for i in range(len(stoch)) if not np.isnan(stoch['STOCHk_' + str(k_period) + '_' + str(d_period) + '_' + str(smooth_k)][i])
            ],
            'metadata': {
                'k_period': k_period,
                'd_period': d_period,
                'smooth_k': smooth_k,
                'description': f'Stochastic Oscillator with k={k_period}, d={d_period}, smooth_k={smooth_k}'
            }
        }
        
        return result
    
    def calculate_adx(self, data: List[Dict[str, Any]], window: int = 14) -> Dict[str, Any]:
        """Calculate Average Directional Index"""
        df = self._prepare_data(data)
        
        # Calculate ADX using pandas_ta
        adx = ta.adx(df['high'], df['low'], df['close'], length=window)
        
        # Format results
        result = {
            'values': [
                {
                    'adx': float(adx['ADX_' + str(window)][i]),
                    'di_plus': float(adx['DMP_' + str(window)][i]),
                    'di_minus': float(adx['DMN_' + str(window)][i])
                }
                for i in range(len(adx)) if not np.isnan(adx['ADX_' + str(window)][i])
            ],
            'metadata': {
                'window': window,
                'description': f'Average Directional Index with window {window}'
            }
        }
        
        return result
    
    def calculate_ichimoku(self, data: List[Dict[str, Any]], tenkan: int = 9, 
                          kijun: int = 26, senkou: int = 52) -> Dict[str, Any]:
        """Calculate Ichimoku Cloud"""
        df = self._prepare_data(data)
        
        # Calculate Ichimoku Cloud using pandas_ta
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'], 
                              tenkan=tenkan, kijun=kijun, senkou=senkou)
        
        # Format results
        result = {
            'values': [
                {
                    'tenkan_sen': float(ichimoku['ITS_' + str(tenkan)][i]) if not np.isnan(ichimoku['ITS_' + str(tenkan)][i]) else None,
                    'kijun_sen': float(ichimoku['IKS_' + str(kijun)][i]) if not np.isnan(ichimoku['IKS_' + str(kijun)][i]) else None,
                    'senkou_span_a': float(ichimoku['ISA_' + str(tenkan) + '_' + str(kijun)][i]) if not np.isnan(ichimoku['ISA_' + str(tenkan) + '_' + str(kijun)][i]) else None,
                    'senkou_span_b': float(ichimoku['ISB_' + str(senkou)][i]) if not np.isnan(ichimoku['ISB_' + str(senkou)][i]) else None,
                    'chikou_span': float(ichimoku['ICS_' + str(kijun)][i]) if i + kijun < len(ichimoku) and not np.isnan(ichimoku['ICS_' + str(kijun)][i]) else None
                }
                for i in range(len(ichimoku))
            ],
            'metadata': {
                'tenkan': tenkan,
                'kijun': kijun,
                'senkou': senkou,
                'description': f'Ichimoku Cloud with tenkan={tenkan}, kijun={kijun}, senkou={senkou}'
            }
        }
        
        return result
    
    def calculate_vwap(self, data: List[Dict[str, Any]], window: int = None) -> Dict[str, Any]:
        """Calculate Volume Weighted Average Price"""
        df = self._prepare_data(data)
        
        # Check if volume is available
        if 'volume' not in df.columns:
            raise ValueError("Volume data is required for VWAP calculation")
        
        # Calculate VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        if window:
            # Calculate VWAP for a specific window
            df['vwap'] = (df['close'] * df['volume']).rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        
        # Format results
        result = {
            'values': df['vwap'].dropna().tolist(),
            'metadata': {
                'window': window,
                'description': f'Volume Weighted Average Price' + (f' with window {window}' if window else '')
            }
        }
        
        return result
    
    def calculate_custom(self, data: Union[List[float], List[Dict[str, Any]]], 
                        formula: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate custom indicator using a formula"""
        df = self._prepare_data(data)
        
        # Default params
        if params is None:
            params = {}
        
        try:
            # Use pandas eval to evaluate the formula
            # This is a simplified approach and may not work for complex formulas
            formula = formula.replace('close', "df['close']")
            for param, value in params.items():
                formula = formula.replace(param, str(value))
            
            df['custom'] = eval(formula)
            
            # Format results
            result = {
                'values': df['custom'].dropna().tolist(),
                'metadata': {
                    'formula': formula,
                    'params': params,
                    'description': f'Custom indicator with formula: {formula}'
                }
            }
            
            return result
        
        except Exception as e:
            raise ValueError(f"Error calculating custom indicator: {str(e)}")
    
    def calculate_indicator(self, indicator_type: str, data: Union[List[float], List[Dict[str, Any]]], 
                           window: int = 14, additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate the specified technical indicator"""
        if additional_params is None:
            additional_params = {}
        
        if indicator_type == 'sma':
            return self.calculate_sma(data, window)
        
        elif indicator_type == 'ema':
            return self.calculate_ema(data, window)
        
        elif indicator_type == 'macd':
            fast_period = additional_params.get('fast_period', 12)
            slow_period = additional_params.get('slow_period', 26)
            signal_period = additional_params.get('signal_period', 9)
            return self.calculate_macd(data, fast_period, slow_period, signal_period)
        
        elif indicator_type == 'rsi':
            return self.calculate_rsi(data, window)
        
        elif indicator_type == 'bbands':
            std_dev = additional_params.get('std_dev', 2.0)
            return self.calculate_bbands(data, window, std_dev)
        
        elif indicator_type == 'atr':
            return self.calculate_atr(data, window)
        
        elif indicator_type == 'stoch':
            k_period = additional_params.get('k_period', 14)
            d_period = additional_params.get('d_period', 3)
            smooth_k = additional_params.get('smooth_k', 3)
            return self.calculate_stoch(data, k_period, d_period, smooth_k)
        
        elif indicator_type == 'adx':
            return self.calculate_adx(data, window)
        
        elif indicator_type == 'ichimoku':
            tenkan = additional_params.get('tenkan', 9)
            kijun = additional_params.get('kijun', 26)
            senkou = additional_params.get('senkou', 52)
            return self.calculate_ichimoku(data, tenkan, kijun, senkou)
        
        elif indicator_type == 'vwap':
            return self.calculate_vwap(data, window)
        
        elif indicator_type == 'custom':
            formula = additional_params.get('formula')
            if not formula:
                raise ValueError("Formula is required for custom indicator")
            return self.calculate_custom(data, formula, additional_params)
        
        else:
            raise ValueError(f"Unsupported indicator type: {indicator_type}")
