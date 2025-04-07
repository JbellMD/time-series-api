from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import redis
import json
import hashlib
from typing import Dict, Any

from app.models.forecasting import (
    ForecastRequest, ForecastResult, 
    DecompositionRequest, TimeSeriesDecomposition
)
from app.services.forecasting.factory import ForecastingFactory
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
    return f"forecast:{hashlib.md5(request_str.encode()).hexdigest()}"

@router.post("/", response_model=ForecastResult)
async def create_forecast(request: ForecastRequest, background_tasks: BackgroundTasks):
    """
    Generate a forecast based on the provided time series data.
    
    - **data**: Time series data as a list of values or a list of dictionaries with 'date' and 'value' keys
    - **model**: Forecasting model to use (prophet, arima, sarima, exponential_smoothing, lstm)
    - **periods**: Number of periods to forecast
    - **frequency**: Frequency of the time series data (D, W, M, Q, Y, H, min)
    - **confidence_interval**: Confidence interval for the forecast (0.5 to 0.99)
    - **seasonal_periods**: Number of periods in a seasonal cycle (optional)
    - **include_history**: Whether to include historical data in the response
    - **params**: Additional parameters for the forecasting model (optional)
    """
    # Check if result is in cache
    cache_key = generate_cache_key(request.dict())
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    try:
        # Generate forecast
        result = ForecastingFactory.forecast(
            data=request.data,
            model_type=request.model,
            periods=request.periods,
            frequency=request.frequency,
            confidence_interval=request.confidence_interval,
            seasonal_periods=request.seasonal_periods,
            include_history=request.include_history,
            params=request.params
        )
        
        # Cache result
        background_tasks.add_task(
            redis_client.set,
            cache_key,
            json.dumps(result),
            ex=settings.CACHE_EXPIRY
        )
        
        return result
    
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/decompose", response_model=TimeSeriesDecomposition)
async def decompose_time_series(request: DecompositionRequest):
    """
    Decompose a time series into trend, seasonal, and residual components.
    
    - **data**: Time series data as a list of values or a list of dictionaries with 'date' and 'value' keys
    - **frequency**: Frequency of the time series data (D, W, M, Q, Y, H, min)
    - **model**: Decomposition model: 'additive' or 'multiplicative'
    - **seasonal_periods**: Number of periods in a seasonal cycle (optional)
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Prepare data
        if isinstance(request.data[0], dict):
            # Data is already in dict format with date and value
            df = pd.DataFrame(request.data)
            df.columns = ['ds', 'y'] if list(df.columns) == ['date', 'value'] else df.columns
            df.set_index('ds', inplace=True)
            df.index = pd.to_datetime(df.index)
        else:
            # Data is a list of values, generate dates
            end_date = datetime.now()
            
            # Generate dates based on frequency
            if request.frequency == 'D':
                start_date = end_date - timedelta(days=len(request.data))
                dates = pd.date_range(start=start_date, periods=len(request.data), freq='D')
            elif request.frequency == 'W':
                start_date = end_date - timedelta(weeks=len(request.data))
                dates = pd.date_range(start=start_date, periods=len(request.data), freq='W')
            elif request.frequency == 'M':
                start_date = end_date - timedelta(days=30 * len(request.data))
                dates = pd.date_range(start=start_date, periods=len(request.data), freq='M')
            elif request.frequency == 'Q':
                start_date = end_date - timedelta(days=90 * len(request.data))
                dates = pd.date_range(start=start_date, periods=len(request.data), freq='Q')
            elif request.frequency == 'Y':
                start_date = end_date - timedelta(days=365 * len(request.data))
                dates = pd.date_range(start=start_date, periods=len(request.data), freq='Y')
            elif request.frequency == 'H':
                start_date = end_date - timedelta(hours=len(request.data))
                dates = pd.date_range(start=start_date, periods=len(request.data), freq='H')
            elif request.frequency == 'min':
                start_date = end_date - timedelta(minutes=len(request.data))
                dates = pd.date_range(start=start_date, periods=len(request.data), freq='min')
            else:
                # Default to daily
                start_date = end_date - timedelta(days=len(request.data))
                dates = pd.date_range(start=start_date, periods=len(request.data), freq='D')
            
            df = pd.DataFrame({
                'y': request.data
            }, index=dates)
        
        # Set seasonal periods
        periods = request.seasonal_periods
        if periods is None:
            # Set default seasonal periods based on frequency
            if request.frequency == 'D':
                periods = 7  # Weekly seasonality
            elif request.frequency == 'W':
                periods = 52  # Yearly seasonality
            elif request.frequency == 'M':
                periods = 12  # Yearly seasonality
            elif request.frequency == 'Q':
                periods = 4  # Yearly seasonality
            elif request.frequency == 'H':
                periods = 24  # Daily seasonality
            elif request.frequency == 'min':
                periods = 60  # Hourly seasonality
            else:
                periods = 7  # Default
        
        # Decompose time series
        decomposition = seasonal_decompose(
            df['y'], 
            model=request.model,
            period=periods
        )
        
        # Format results
        dates_str = [date.strftime('%Y-%m-%d %H:%M:%S') for date in df.index]
        
        return {
            'trend': decomposition.trend.fillna(method='bfill').fillna(method='ffill').tolist(),
            'seasonal': decomposition.seasonal.fillna(method='bfill').fillna(method='ffill').tolist(),
            'residual': decomposition.resid.fillna(method='bfill').fillna(method='ffill').tolist(),
            'dates': dates_str
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.get("/models")
async def list_models():
    """List available forecasting models"""
    return {
        "models": [
            {
                "id": "prophet",
                "name": "Prophet",
                "description": "Facebook's time series forecasting model",
                "parameters": [
                    {"name": "changepoint_prior_scale", "type": "float", "default": 0.05},
                    {"name": "seasonality_prior_scale", "type": "float", "default": 10.0},
                    {"name": "holidays_prior_scale", "type": "float", "default": 10.0},
                    {"name": "seasonality_mode", "type": "string", "default": "additive"}
                ]
            },
            {
                "id": "arima",
                "name": "ARIMA",
                "description": "AutoRegressive Integrated Moving Average",
                "parameters": [
                    {"name": "order", "type": "tuple", "default": "(1,1,1)"},
                    {"name": "trend", "type": "string", "default": "c"}
                ]
            },
            {
                "id": "sarima",
                "name": "SARIMA",
                "description": "Seasonal AutoRegressive Integrated Moving Average",
                "parameters": [
                    {"name": "order", "type": "tuple", "default": "(1,1,1)"},
                    {"name": "seasonal_order", "type": "tuple", "default": "(1,1,1,12)"},
                    {"name": "trend", "type": "string", "default": "c"}
                ]
            },
            {
                "id": "exponential_smoothing",
                "name": "Exponential Smoothing",
                "description": "Exponential Smoothing State Space Model",
                "status": "coming_soon"
            },
            {
                "id": "lstm",
                "name": "LSTM",
                "description": "Long Short-Term Memory Neural Network",
                "status": "coming_soon"
            }
        ]
    }
