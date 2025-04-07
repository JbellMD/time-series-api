from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class TechnicalIndicator(str, Enum):
    SMA = "sma"  # Simple Moving Average
    EMA = "ema"  # Exponential Moving Average
    MACD = "macd"  # Moving Average Convergence Divergence
    RSI = "rsi"  # Relative Strength Index
    BBANDS = "bbands"  # Bollinger Bands
    ATR = "atr"  # Average True Range
    OBV = "obv"  # On-Balance Volume
    STOCH = "stoch"  # Stochastic Oscillator
    ADX = "adx"  # Average Directional Index
    ICHIMOKU = "ichimoku"  # Ichimoku Cloud
    VWAP = "vwap"  # Volume Weighted Average Price
    CUSTOM = "custom"  # Custom indicator


class IndicatorRequest(BaseModel):
    data: Union[List[float], List[Dict[str, Any]]] = Field(
        ..., description="Time series data. Can be a list of values or a list of dictionaries with OHLCV data"
    )
    indicator: TechnicalIndicator = Field(
        ..., description="Technical indicator to calculate"
    )
    window: int = Field(
        14, description="Window size for the indicator calculation", ge=1, le=500
    )
    additional_params: Optional[Dict[str, Any]] = Field(
        None, description="Additional parameters for the indicator calculation"
    )

    @validator("data")
    def validate_data(cls, v, values):
        if not v:
            raise ValueError("Data cannot be empty")
        
        # Check if indicator requires OHLCV data
        ohlcv_indicators = [
            TechnicalIndicator.BBANDS, 
            TechnicalIndicator.ATR, 
            TechnicalIndicator.OBV,
            TechnicalIndicator.STOCH,
            TechnicalIndicator.ICHIMOKU,
            TechnicalIndicator.VWAP
        ]
        
        if "indicator" in values and values["indicator"] in ohlcv_indicators:
            # Check if data is a list of dictionaries with OHLCV data
            if isinstance(v[0], dict):
                required_keys = ["open", "high", "low", "close"]
                for item in v:
                    missing_keys = [key for key in required_keys if key not in item]
                    if missing_keys:
                        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
            else:
                raise ValueError(f"The {values['indicator']} indicator requires OHLCV data")
        
        return v


class IndicatorResult(BaseModel):
    indicator: TechnicalIndicator = Field(
        ..., description="Technical indicator calculated"
    )
    values: List[Union[float, Dict[str, float]]] = Field(
        ..., description="Calculated indicator values"
    )
    metadata: Dict[str, Any] = Field(
        ..., description="Metadata about the calculation"
    )


class PatternType(str, Enum):
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_and_shoulders"
    INV_HEAD_SHOULDERS = "inverse_head_and_shoulders"
    TRIANGLE_ASCENDING = "ascending_triangle"
    TRIANGLE_DESCENDING = "descending_triangle"
    TRIANGLE_SYMMETRICAL = "symmetrical_triangle"
    FLAG = "flag"
    PENNANT = "pennant"
    WEDGE = "wedge"
    CHANNEL = "channel"
    CUP_HANDLE = "cup_and_handle"
    CUSTOM = "custom"


class PatternRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ..., description="OHLCV data for pattern recognition"
    )
    pattern: Optional[PatternType] = Field(
        None, description="Specific pattern to look for. If None, all patterns will be checked"
    )
    sensitivity: float = Field(
        0.75, description="Sensitivity of pattern detection (0.0 to 1.0)", ge=0.0, le=1.0
    )
    min_points: int = Field(
        5, description="Minimum number of points required for a pattern", ge=3
    )

    @validator("data")
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        
        required_keys = ["open", "high", "low", "close"]
        for item in v:
            missing_keys = [key for key in required_keys if key not in item]
            if missing_keys:
                raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
        
        return v


class PatternResult(BaseModel):
    patterns: List[Dict[str, Any]] = Field(
        ..., description="Detected patterns with details"
    )
    count: int = Field(
        ..., description="Number of patterns detected"
    )
