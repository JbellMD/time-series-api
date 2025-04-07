from fastapi import APIRouter, HTTPException, BackgroundTasks
import redis
import json
import hashlib
from typing import Dict, Any, List

from app.models.technical import (
    IndicatorRequest, IndicatorResult, 
    PatternRequest, PatternResult,
    TechnicalIndicator
)
from app.services.technical.indicators import TechnicalIndicatorService
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
    return f"technical:{hashlib.md5(request_str.encode()).hexdigest()}"

@router.post("/indicator", response_model=IndicatorResult)
async def calculate_indicator(request: IndicatorRequest, background_tasks: BackgroundTasks):
    """
    Calculate a technical indicator on the provided time series data.
    
    - **data**: Time series data as a list of values or a list of dictionaries with OHLCV data
    - **indicator**: Technical indicator to calculate (sma, ema, macd, rsi, bbands, etc.)
    - **window**: Window size for the indicator calculation
    - **additional_params**: Additional parameters for the indicator calculation (optional)
    """
    # Check if result is in cache
    cache_key = generate_cache_key(request.dict())
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    try:
        # Initialize service
        indicator_service = TechnicalIndicatorService()
        
        # Calculate indicator
        result = indicator_service.calculate_indicator(
            indicator_type=request.indicator,
            data=request.data,
            window=request.window,
            additional_params=request.additional_params
        )
        
        # Prepare response
        response = {
            "indicator": request.indicator,
            "values": result["values"],
            "metadata": result["metadata"]
        }
        
        # Cache result
        background_tasks.add_task(
            redis_client.set,
            cache_key,
            json.dumps(response),
            ex=settings.CACHE_EXPIRY
        )
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/pattern", response_model=PatternResult)
async def detect_patterns(request: PatternRequest):
    """
    Detect chart patterns in the provided OHLCV data.
    
    - **data**: OHLCV data for pattern recognition
    - **pattern**: Specific pattern to look for (optional)
    - **sensitivity**: Sensitivity of pattern detection (0.0 to 1.0)
    - **min_points**: Minimum number of points required for a pattern
    """
    try:
        # This is a placeholder for pattern recognition
        # In a real implementation, you would use a pattern recognition library or algorithm
        
        # Mock response for demonstration
        patterns = []
        
        # Return mock patterns
        return {
            "patterns": patterns,
            "count": len(patterns)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.get("/indicators")
async def list_indicators():
    """List available technical indicators"""
    return {
        "indicators": [
            {
                "id": "sma",
                "name": "Simple Moving Average",
                "description": "Average of prices over a specified period",
                "parameters": [
                    {"name": "window", "type": "int", "default": 14, "description": "Window size"}
                ]
            },
            {
                "id": "ema",
                "name": "Exponential Moving Average",
                "description": "Weighted average giving more importance to recent prices",
                "parameters": [
                    {"name": "window", "type": "int", "default": 14, "description": "Window size"}
                ]
            },
            {
                "id": "macd",
                "name": "Moving Average Convergence Divergence",
                "description": "Trend-following momentum indicator",
                "parameters": [
                    {"name": "fast_period", "type": "int", "default": 12, "description": "Fast period"},
                    {"name": "slow_period", "type": "int", "default": 26, "description": "Slow period"},
                    {"name": "signal_period", "type": "int", "default": 9, "description": "Signal period"}
                ]
            },
            {
                "id": "rsi",
                "name": "Relative Strength Index",
                "description": "Momentum oscillator measuring speed and change of price movements",
                "parameters": [
                    {"name": "window", "type": "int", "default": 14, "description": "Window size"}
                ]
            },
            {
                "id": "bbands",
                "name": "Bollinger Bands",
                "description": "Volatility bands placed above and below a moving average",
                "parameters": [
                    {"name": "window", "type": "int", "default": 20, "description": "Window size"},
                    {"name": "std_dev", "type": "float", "default": 2.0, "description": "Standard deviation multiplier"}
                ]
            },
            {
                "id": "atr",
                "name": "Average True Range",
                "description": "Volatility indicator",
                "parameters": [
                    {"name": "window", "type": "int", "default": 14, "description": "Window size"}
                ]
            },
            {
                "id": "stoch",
                "name": "Stochastic Oscillator",
                "description": "Momentum indicator comparing closing price to price range",
                "parameters": [
                    {"name": "k_period", "type": "int", "default": 14, "description": "%K period"},
                    {"name": "d_period", "type": "int", "default": 3, "description": "%D period"},
                    {"name": "smooth_k", "type": "int", "default": 3, "description": "%K smoothing"}
                ]
            },
            {
                "id": "adx",
                "name": "Average Directional Index",
                "description": "Trend strength indicator",
                "parameters": [
                    {"name": "window", "type": "int", "default": 14, "description": "Window size"}
                ]
            },
            {
                "id": "ichimoku",
                "name": "Ichimoku Cloud",
                "description": "Multiple indicator system showing support, resistance, momentum, and trend",
                "parameters": [
                    {"name": "tenkan", "type": "int", "default": 9, "description": "Tenkan-sen period"},
                    {"name": "kijun", "type": "int", "default": 26, "description": "Kijun-sen period"},
                    {"name": "senkou", "type": "int", "default": 52, "description": "Senkou span B period"}
                ]
            },
            {
                "id": "vwap",
                "name": "Volume Weighted Average Price",
                "description": "Average price weighted by volume",
                "parameters": [
                    {"name": "window", "type": "int", "default": null, "description": "Window size (optional)"}
                ]
            },
            {
                "id": "custom",
                "name": "Custom Indicator",
                "description": "Custom indicator using a formula",
                "parameters": [
                    {"name": "formula", "type": "string", "description": "Formula to calculate the indicator"}
                ]
            }
        ]
    }
