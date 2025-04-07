from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class StationarityTest(str, Enum):
    ADF = "adf"  # Augmented Dickey-Fuller
    KPSS = "kpss"  # Kwiatkowski-Phillips-Schmidt-Shin
    PP = "pp"  # Phillips-Perron


class StationarityRequest(BaseModel):
    data: List[float] = Field(
        ..., description="Time series data for stationarity testing"
    )
    test: StationarityTest = Field(
        StationarityTest.ADF, description="Stationarity test to perform"
    )
    alpha: float = Field(
        0.05, description="Significance level for the test", ge=0.01, le=0.1
    )

    @validator("data")
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        if len(v) < 10:
            raise ValueError("Time series too short for stationarity testing (minimum 10 points)")
        return v


class StationarityResult(BaseModel):
    test: StationarityTest = Field(
        ..., description="Stationarity test performed"
    )
    statistic: float = Field(
        ..., description="Test statistic"
    )
    p_value: float = Field(
        ..., description="P-value of the test"
    )
    critical_values: Dict[str, float] = Field(
        ..., description="Critical values at different significance levels"
    )
    is_stationary: bool = Field(
        ..., description="Whether the time series is stationary according to the test"
    )
    suggested_differencing: Optional[int] = Field(
        None, description="Suggested differencing order to achieve stationarity"
    )


class AnomalyDetectionMethod(str, Enum):
    IQR = "iqr"  # Interquartile Range
    Z_SCORE = "z_score"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "local_outlier_factor"
    PROPHET = "prophet"
    MOVING_AVERAGE = "moving_average"


class AnomalyRequest(BaseModel):
    data: Union[List[float], List[Dict[str, Any]]] = Field(
        ..., description="Time series data for anomaly detection"
    )
    method: AnomalyDetectionMethod = Field(
        AnomalyDetectionMethod.IQR, description="Anomaly detection method to use"
    )
    sensitivity: float = Field(
        1.5, description="Sensitivity parameter for anomaly detection", ge=0.1
    )
    window_size: Optional[int] = Field(
        None, description="Window size for moving average method"
    )

    @validator("window_size")
    def validate_window_size(cls, v, values):
        if "method" in values and values["method"] == AnomalyDetectionMethod.MOVING_AVERAGE:
            if v is None or v < 2:
                raise ValueError("Window size must be at least 2 for moving average method")
        return v


class AnomalyResult(BaseModel):
    anomalies: List[Dict[str, Any]] = Field(
        ..., description="Detected anomalies with details"
    )
    count: int = Field(
        ..., description="Number of anomalies detected"
    )
    method: AnomalyDetectionMethod = Field(
        ..., description="Method used for anomaly detection"
    )
    threshold: float = Field(
        ..., description="Threshold used for anomaly detection"
    )


class CorrelationMethod(str, Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    CROSS_CORRELATION = "cross_correlation"


class CorrelationRequest(BaseModel):
    data_x: List[float] = Field(
        ..., description="First time series data"
    )
    data_y: List[float] = Field(
        ..., description="Second time series data"
    )
    method: CorrelationMethod = Field(
        CorrelationMethod.PEARSON, description="Correlation method to use"
    )
    max_lag: Optional[int] = Field(
        None, description="Maximum lag for cross-correlation"
    )

    @validator("data_y")
    def validate_data_length(cls, v, values):
        if "data_x" in values and len(v) != len(values["data_x"]):
            raise ValueError("Both time series must have the same length")
        return v

    @validator("max_lag")
    def validate_max_lag(cls, v, values):
        if "method" in values and values["method"] == CorrelationMethod.CROSS_CORRELATION:
            if v is None:
                return len(values.get("data_x", [])) // 2
            if v < 1:
                raise ValueError("Maximum lag must be at least 1")
        return v


class CorrelationResult(BaseModel):
    coefficient: float = Field(
        ..., description="Correlation coefficient"
    )
    p_value: Optional[float] = Field(
        None, description="P-value of the correlation test"
    )
    method: CorrelationMethod = Field(
        ..., description="Correlation method used"
    )
    lags: Optional[List[int]] = Field(
        None, description="Lags for cross-correlation"
    )
    correlations: Optional[List[float]] = Field(
        None, description="Correlation values for each lag"
    )
    best_lag: Optional[int] = Field(
        None, description="Lag with the highest correlation"
    )
