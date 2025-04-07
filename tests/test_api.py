import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
HEADERS = {"Content-Type": "application/json"}

def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print("✅ Health check passed")

def test_forecasting_endpoints():
    """Test forecasting endpoints"""
    # Generate sample time series data
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    values = [50 + 10 * np.sin(i / 10) + np.random.normal(0, 2) for i in range(100)]
    
    # Format data for API
    data = [
        {"date": date.strftime("%Y-%m-%d"), "value": value}
        for date, value in zip(dates, values)
    ]
    
    # Test Prophet forecast
    payload = {
        "data": data,
        "model": "prophet",
        "periods": 30,
        "frequency": "D",
        "confidence_interval": 0.95,
        "include_history": True
    }
    
    response = requests.post(f"{BASE_URL}/forecast/", json=payload)
    assert response.status_code == 200
    forecast_result = response.json()
    assert "forecast" in forecast_result
    assert len(forecast_result["forecast"]) == 30
    print("✅ Prophet forecast endpoint passed")
    
    # Test ARIMA forecast
    payload["model"] = "arima"
    response = requests.post(f"{BASE_URL}/forecast/", json=payload)
    assert response.status_code == 200
    forecast_result = response.json()
    assert "forecast" in forecast_result
    assert len(forecast_result["forecast"]) == 30
    print("✅ ARIMA forecast endpoint passed")
    
    # Test decomposition
    payload = {
        "data": data,
        "frequency": "D",
        "model": "prophet"
    }
    response = requests.post(f"{BASE_URL}/forecast/decompose", json=payload)
    assert response.status_code == 200
    decomposition_result = response.json()
    assert "trend" in decomposition_result
    assert "seasonal" in decomposition_result
    print("✅ Decomposition endpoint passed")

def test_technical_endpoints():
    """Test technical analysis endpoints"""
    # Generate sample OHLCV data
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    close_values = [100 + 10 * np.sin(i / 10) + np.random.normal(0, 2) for i in range(100)]
    open_values = [close_values[i] - np.random.normal(0, 1) for i in range(100)]
    high_values = [max(open_values[i], close_values[i]) + np.random.normal(0, 1) for i in range(100)]
    low_values = [min(open_values[i], close_values[i]) - np.random.normal(0, 1) for i in range(100)]
    volume_values = [1000 + np.random.normal(0, 100) for _ in range(100)]
    
    # Format data for API
    data = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "volume": volume_val
        }
        for date, open_val, high_val, low_val, close_val, volume_val in zip(
            dates, open_values, high_values, low_values, close_values, volume_values
        )
    ]
    
    # Test SMA indicator
    payload = {
        "data": data,
        "indicator": "sma",
        "window": 14,
        "additional_params": {}
    }
    
    response = requests.post(f"{BASE_URL}/technical/indicator", json=payload)
    assert response.status_code == 200
    indicator_result = response.json()
    assert "values" in indicator_result
    assert len(indicator_result["values"]) > 0
    print("✅ SMA indicator endpoint passed")
    
    # Test RSI indicator
    payload["indicator"] = "rsi"
    response = requests.post(f"{BASE_URL}/technical/indicator", json=payload)
    assert response.status_code == 200
    indicator_result = response.json()
    assert "values" in indicator_result
    assert len(indicator_result["values"]) > 0
    print("✅ RSI indicator endpoint passed")
    
    # Test MACD indicator
    payload["indicator"] = "macd"
    payload["additional_params"] = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    }
    response = requests.post(f"{BASE_URL}/technical/indicator", json=payload)
    assert response.status_code == 200
    indicator_result = response.json()
    assert "values" in indicator_result
    assert len(indicator_result["values"]) > 0
    print("✅ MACD indicator endpoint passed")
    
    # Test Bollinger Bands
    payload["indicator"] = "bbands"
    payload["additional_params"] = {"std_dev": 2.0}
    response = requests.post(f"{BASE_URL}/technical/indicator", json=payload)
    assert response.status_code == 200
    indicator_result = response.json()
    assert "values" in indicator_result
    assert len(indicator_result["values"]) > 0
    print("✅ Bollinger Bands endpoint passed")
    
    # Test list indicators
    response = requests.get(f"{BASE_URL}/technical/indicators")
    assert response.status_code == 200
    indicators_list = response.json()
    assert "indicators" in indicators_list
    assert len(indicators_list["indicators"]) > 0
    print("✅ List indicators endpoint passed")

def test_statistical_endpoints():
    """Test statistical analysis endpoints"""
    # Generate sample time series data
    values = [50 + 10 * np.sin(i / 10) + np.random.normal(0, 2) + i * 0.1 for i in range(100)]
    
    # Test stationarity
    payload = {
        "data": values,
        "test": "adf",
        "alpha": 0.05
    }
    
    response = requests.post(f"{BASE_URL}/statistical/stationarity", json=payload)
    assert response.status_code == 200
    stationarity_result = response.json()
    assert "is_stationary" in stationarity_result
    print("✅ Stationarity test endpoint passed")
    
    # Test anomaly detection
    payload = {
        "data": values,
        "method": "iqr",
        "sensitivity": 1.5
    }
    
    response = requests.post(f"{BASE_URL}/statistical/anomalies", json=payload)
    assert response.status_code == 200
    anomaly_result = response.json()
    assert "anomalies" in anomaly_result
    print("✅ Anomaly detection endpoint passed")
    
    # Test correlation
    values2 = [values[i] + np.random.normal(0, 5) for i in range(100)]
    payload = {
        "data_x": values,
        "data_y": values2,
        "method": "pearson"
    }
    
    response = requests.post(f"{BASE_URL}/statistical/correlation", json=payload)
    assert response.status_code == 200
    correlation_result = response.json()
    assert "coefficient" in correlation_result
    print("✅ Correlation endpoint passed")
    
    # Test ACF/PACF
    response = requests.post(
        f"{BASE_URL}/statistical/acf-pacf",
        json={"data": values, "max_lag": 20, "alpha": 0.05}
    )
    assert response.status_code == 200
    acf_pacf_result = response.json()
    assert "acf" in acf_pacf_result
    assert "pacf" in acf_pacf_result
    print("✅ ACF/PACF endpoint passed")
    
    # Test list methods
    response = requests.get(f"{BASE_URL}/statistical/methods")
    assert response.status_code == 200
    methods_list = response.json()
    assert "stationarity_tests" in methods_list
    assert "anomaly_detection_methods" in methods_list
    assert "correlation_methods" in methods_list
    print("✅ List methods endpoint passed")

def run_all_tests():
    """Run all API tests"""
    print("Starting API tests...")
    
    try:
        test_health_check()
        test_forecasting_endpoints()
        test_technical_endpoints()
        test_statistical_endpoints()
        
        print("\n🎉 All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")

if __name__ == "__main__":
    run_all_tests()
