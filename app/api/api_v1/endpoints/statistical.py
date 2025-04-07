from fastapi import APIRouter, HTTPException, BackgroundTasks
import redis
import json
import hashlib
from typing import Dict, Any, List

from app.models.statistical import (
    StationarityRequest, StationarityResult,
    AnomalyRequest, AnomalyResult,
    CorrelationRequest, CorrelationResult,
    StationarityTest, AnomalyDetectionMethod, CorrelationMethod
)
from app.services.statistical.analysis import StatisticalAnalysisService
from app.core.config import settings

# Initialize Redis client for caching
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True
)

router = APIRouter()

def generate_cache_key(request_data: Dict[str, Any]) -> str:
    """Generate a cache key based on request data"""
    # Convert request data to a string and hash it
    request_str = json.dumps(request_data, sort_keys=True)
    return f"statistical:{hashlib.md5(request_str.encode()).hexdigest()}"

@router.post("/stationarity", response_model=StationarityResult)
async def test_stationarity(request: StationarityRequest, background_tasks: BackgroundTasks):
    """
    Test stationarity of time series data.
    
    - **data**: Time series data for stationarity testing
    - **test**: Stationarity test to perform (adf, kpss, pp)
    - **alpha**: Significance level for the test (0.01 to 0.1)
    """
    # Check if result is in cache
    cache_key = generate_cache_key(request.dict())
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    try:
        # Initialize service
        analysis_service = StatisticalAnalysisService()
        
        # Test stationarity
        result = analysis_service.test_stationarity(
            data=request.data,
            test=request.test,
            alpha=request.alpha
        )
        
        # Cache result
        background_tasks.add_task(
            redis_client.set,
            cache_key,
            json.dumps(result),
            ex=settings.CACHE_EXPIRY
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/anomalies", response_model=AnomalyResult)
async def detect_anomalies(request: AnomalyRequest, background_tasks: BackgroundTasks):
    """
    Detect anomalies in time series data.
    
    - **data**: Time series data for anomaly detection
    - **method**: Anomaly detection method to use
    - **sensitivity**: Sensitivity parameter for anomaly detection
    - **window_size**: Window size for moving average method (optional)
    """
    # Check if result is in cache
    cache_key = generate_cache_key(request.dict())
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    try:
        # Initialize service
        analysis_service = StatisticalAnalysisService()
        
        # Detect anomalies
        result = analysis_service.detect_anomalies(
            data=request.data,
            method=request.method,
            sensitivity=request.sensitivity,
            window_size=request.window_size
        )
        
        # Cache result
        background_tasks.add_task(
            redis_client.set,
            cache_key,
            json.dumps(result),
            ex=settings.CACHE_EXPIRY
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/correlation", response_model=CorrelationResult)
async def calculate_correlation(request: CorrelationRequest, background_tasks: BackgroundTasks):
    """
    Calculate correlation between two time series.
    
    - **data_x**: First time series data
    - **data_y**: Second time series data
    - **method**: Correlation method to use
    - **max_lag**: Maximum lag for cross-correlation (optional)
    """
    # Check if result is in cache
    cache_key = generate_cache_key(request.dict())
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    try:
        # Initialize service
        analysis_service = StatisticalAnalysisService()
        
        # Calculate correlation
        result = analysis_service.calculate_correlation(
            data_x=request.data_x,
            data_y=request.data_y,
            method=request.method,
            max_lag=request.max_lag
        )
        
        # Cache result
        background_tasks.add_task(
            redis_client.set,
            cache_key,
            json.dumps(result),
            ex=settings.CACHE_EXPIRY
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/acf-pacf")
async def analyze_acf_pacf(background_tasks: BackgroundTasks, data: List[float], max_lag: int = 40, alpha: float = 0.05):
    """
    Calculate Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).
    
    - **data**: Time series data
    - **max_lag**: Maximum lag to calculate
    - **alpha**: Significance level for confidence intervals
    """
    # Generate cache key
    request_data = {"data": data, "max_lag": max_lag, "alpha": alpha}
    cache_key = generate_cache_key(request_data)
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    try:
        # Initialize service
        analysis_service = StatisticalAnalysisService()
        
        # Calculate ACF and PACF
        result = analysis_service.analyze_acf_pacf(
            data=data,
            max_lag=max_lag,
            alpha=alpha
        )
        
        # Cache result
        background_tasks.add_task(
            redis_client.set,
            cache_key,
            json.dumps(result),
            ex=settings.CACHE_EXPIRY
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.get("/methods")
async def list_methods():
    """List available statistical analysis methods"""
    return {
        "stationarity_tests": [
            {
                "id": "adf",
                "name": "Augmented Dickey-Fuller",
                "description": "Tests the null hypothesis that a unit root is present in a time series sample"
            },
            {
                "id": "kpss",
                "name": "Kwiatkowski-Phillips-Schmidt-Shin",
                "description": "Tests the null hypothesis that a time series is stationary around a deterministic trend"
            },
            {
                "id": "pp",
                "name": "Phillips-Perron",
                "description": "Tests the null hypothesis that a time series has a unit root"
            }
        ],
        "anomaly_detection_methods": [
            {
                "id": "iqr",
                "name": "Interquartile Range",
                "description": "Detects anomalies based on the interquartile range"
            },
            {
                "id": "z_score",
                "name": "Z-Score",
                "description": "Detects anomalies based on the number of standard deviations from the mean"
            },
            {
                "id": "isolation_forest",
                "name": "Isolation Forest",
                "description": "Detects anomalies using an ensemble of isolation trees"
            },
            {
                "id": "local_outlier_factor",
                "name": "Local Outlier Factor",
                "description": "Detects anomalies based on local density deviation"
            },
            {
                "id": "moving_average",
                "name": "Moving Average",
                "description": "Detects anomalies based on deviation from moving average"
            },
            {
                "id": "prophet",
                "name": "Prophet",
                "description": "Detects anomalies using Facebook's Prophet forecasting model",
                "status": "coming_soon"
            }
        ],
        "correlation_methods": [
            {
                "id": "pearson",
                "name": "Pearson",
                "description": "Measures linear correlation between two variables"
            },
            {
                "id": "spearman",
                "name": "Spearman",
                "description": "Measures monotonic correlation between two variables"
            },
            {
                "id": "kendall",
                "name": "Kendall",
                "description": "Measures ordinal association between two variables"
            },
            {
                "id": "cross_correlation",
                "name": "Cross-Correlation",
                "description": "Measures similarity between two time series as a function of the lag"
            }
        ]
    }
