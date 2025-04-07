from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import numpy as np


class ForecastingModel(str, Enum):
    PROPHET = "prophet"
    ARIMA = "arima"
    SARIMA = "sarima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LSTM = "lstm"


class Frequency(str, Enum):
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"
    HOURLY = "H"
    MINUTELY = "min"


class ForecastRequest(BaseModel):
    data: Union[List[float], List[Dict[str, Any]]] = Field(
        ..., description="Time series data. Can be a list of values or a list of dictionaries with 'date' and 'value' keys"
    )
    model: ForecastingModel = Field(
        ForecastingModel.PROPHET, description="Forecasting model to use"
    )
    periods: int = Field(
        10, description="Number of periods to forecast", ge=1, le=1000
    )
    frequency: Frequency = Field(
        Frequency.DAILY, description="Frequency of the time series data"
    )
    confidence_interval: float = Field(
        0.95, description="Confidence interval for the forecast", ge=0.5, le=0.99
    )
    seasonal_periods: Optional[int] = Field(
        None, description="Number of periods in a seasonal cycle"
    )
    include_history: bool = Field(
        False, description="Whether to include historical data in the response"
    )
    params: Optional[Dict[str, Any]] = Field(
        None, description="Additional parameters for the forecasting model"
    )

    @validator("data")
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        
        # Check if data is a list of dictionaries
        if isinstance(v[0], dict):
            for item in v:
                if "date" not in item or "value" not in item:
                    raise ValueError("Each dictionary must contain 'date' and 'value' keys")
                if not isinstance(item["value"], (int, float)):
                    raise ValueError("Value must be a number")
        # Check if data is a list of numbers
        else:
            for item in v:
                if not isinstance(item, (int, float)):
                    raise ValueError("Each item must be a number")
        
        return v


class ForecastResult(BaseModel):
    forecast: List[Dict[str, Any]] = Field(
        ..., description="Forecasted values with dates"
    )
    model: ForecastingModel = Field(
        ..., description="Model used for forecasting"
    )
    metrics: Dict[str, float] = Field(
        ..., description="Performance metrics of the forecast"
    )
    lower_bound: Optional[List[float]] = Field(
        None, description="Lower bound of the confidence interval"
    )
    upper_bound: Optional[List[float]] = Field(
        None, description="Upper bound of the confidence interval"
    )
    history: Optional[List[Dict[str, Any]]] = Field(
        None, description="Historical data used for forecasting"
    )
    components: Optional[Dict[str, List[float]]] = Field(
        None, description="Decomposed components of the time series (trend, seasonality, etc.)"
    )


class TimeSeriesDecomposition(BaseModel):
    trend: List[float] = Field(..., description="Trend component")
    seasonal: List[float] = Field(..., description="Seasonal component")
    residual: List[float] = Field(..., description="Residual component")
    dates: List[str] = Field(..., description="Dates corresponding to the components")


class DecompositionRequest(BaseModel):
    data: Union[List[float], List[Dict[str, Any]]] = Field(
        ..., description="Time series data. Can be a list of values or a list of dictionaries with 'date' and 'value' keys"
    )
    frequency: Frequency = Field(
        Frequency.DAILY, description="Frequency of the time series data"
    )
    model: str = Field(
        "additive", description="Decomposition model: 'additive' or 'multiplicative'"
    )
    seasonal_periods: Optional[int] = Field(
        None, description="Number of periods in a seasonal cycle"
    )

    @validator("model")
    def validate_model(cls, v):
        if v not in ["additive", "multiplicative"]:
            raise ValueError("Model must be either 'additive' or 'multiplicative'")
        return v
