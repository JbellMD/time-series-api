# Time Series Analysis API Documentation

## Overview

The Time Series Analysis API provides robust endpoints for time series forecasting, technical analysis, and statistical analysis. This API is designed for data scientists, analysts, and developers who need to perform time series analysis and forecasting at scale.

## Authentication

The API supports two authentication methods:

### OAuth2 Password Flow

```
POST /api/v1/auth/token
```

**Request Body:**
```json
{
  "username": "user",
  "password": "password",
  "scope": "read:forecasts read:technical"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### API Key Authentication

API keys can be used by including the `X-API-Key` header in your requests:

```
X-API-Key: ts_api_1234567890abcdef
```

To create an API key (admin only):

```
POST /api/v1/auth/api-keys
```

**Request Body:**
```json
{
  "name": "My API Key",
  "scopes": ["read:forecasts", "read:technical"]
}
```

## Forecasting Endpoints

### Generate Forecast

```
POST /api/v1/forecast/
```

**Request Body:**
```json
{
  "data": [
    {"date": "2023-01-01", "value": 100},
    {"date": "2023-01-02", "value": 101},
    {"date": "2023-01-03", "value": 99}
  ],
  "model": "prophet",
  "periods": 30,
  "frequency": "D",
  "confidence_interval": 0.95,
  "include_history": false
}
```

**Response:**
```json
{
  "forecast": [
    {
      "date": "2023-01-04",
      "value": 102.5,
      "lower_bound": 98.2,
      "upper_bound": 106.8
    },
    ...
  ],
  "model": "prophet",
  "metrics": {
    "rmse": 2.3,
    "mape": 1.8
  },
  "parameters": {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10
  }
}
```

### Available Models

1. **Prophet** (`prophet`): Facebook's Prophet model for forecasting with multiple seasonality patterns.
2. **ARIMA** (`arima`): Auto-Regressive Integrated Moving Average model.
3. **SARIMA** (`sarima`): Seasonal ARIMA model.
4. **LSTM** (`lstm`): Long Short-Term Memory neural network for time series forecasting.

### Decompose Time Series

```
POST /api/v1/forecast/decompose
```

**Request Body:**
```json
{
  "data": [
    {"date": "2023-01-01", "value": 100},
    {"date": "2023-01-02", "value": 101},
    {"date": "2023-01-03", "value": 99}
  ],
  "frequency": "D",
  "model": "prophet"
}
```

**Response:**
```json
{
  "trend": [
    {"date": "2023-01-01", "value": 99.5},
    {"date": "2023-01-02", "value": 100.0},
    {"date": "2023-01-03", "value": 100.5}
  ],
  "seasonal": [
    {"date": "2023-01-01", "value": 0.5},
    {"date": "2023-01-02", "value": 1.0},
    {"date": "2023-01-03", "value": -1.5}
  ],
  "residual": [
    {"date": "2023-01-01", "value": 0.0},
    {"date": "2023-01-02", "value": 0.0},
    {"date": "2023-01-03", "value": 0.0}
  ],
  "model": "prophet"
}
```

## Technical Analysis Endpoints

### Calculate Technical Indicator

```
POST /api/v1/technical/indicator
```

**Request Body:**
```json
{
  "data": [
    {
      "date": "2023-01-01",
      "open": 100,
      "high": 105,
      "low": 98,
      "close": 102,
      "volume": 1000
    },
    ...
  ],
  "indicator": "sma",
  "window": 14,
  "additional_params": {}
}
```

**Response:**
```json
{
  "indicator": "sma",
  "values": [
    {"date": "2023-01-14", "value": 103.5},
    {"date": "2023-01-15", "value": 104.2},
    ...
  ],
  "metadata": {
    "window": 14
  }
}
```

### Available Indicators

```
GET /api/v1/technical/indicators
```

**Response:**
```json
{
  "indicators": [
    {
      "id": "sma",
      "name": "Simple Moving Average",
      "description": "Average of prices over a specified period",
      "parameters": [
        {"name": "window", "type": "int", "default": 14, "description": "Window size"}
      ]
    },
    ...
  ]
}
```

## Statistical Analysis Endpoints

### Test Stationarity

```
POST /api/v1/statistical/stationarity
```

**Request Body:**
```json
{
  "data": [100, 101, 99, 102, 103, 101, 104],
  "test": "adf",
  "alpha": 0.05
}
```

**Response:**
```json
{
  "test": "adf",
  "statistic": -2.5,
  "p_value": 0.04,
  "critical_values": {
    "1": -3.75,
    "5": -3.0,
    "10": -2.63
  },
  "is_stationary": true,
  "suggested_differencing": null
}
```

### Detect Anomalies

```
POST /api/v1/statistical/anomalies
```

**Request Body:**
```json
{
  "data": [100, 101, 99, 102, 103, 101, 150, 104],
  "method": "iqr",
  "sensitivity": 1.5
}
```

**Response:**
```json
{
  "anomalies": [
    {
      "index": 6,
      "value": 150,
      "type": "high"
    }
  ],
  "count": 1,
  "method": "iqr",
  "threshold": 1.5,
  "bounds": {
    "lower": 95.5,
    "upper": 108.5
  }
}
```

### Calculate Correlation

```
POST /api/v1/statistical/correlation
```

**Request Body:**
```json
{
  "data_x": [100, 101, 99, 102, 103, 101, 104],
  "data_y": [50, 52, 49, 53, 54, 51, 55],
  "method": "pearson"
}
```

**Response:**
```json
{
  "coefficient": 0.95,
  "p_value": 0.001,
  "method": "pearson"
}
```

### Analyze ACF/PACF

```
POST /api/v1/statistical/acf-pacf
```

**Request Body:**
```json
{
  "data": [100, 101, 99, 102, 103, 101, 104, 105, 103, 106],
  "max_lag": 5,
  "alpha": 0.05
}
```

**Response:**
```json
{
  "acf": {
    "values": [1.0, 0.7, 0.4, 0.2, 0.1, 0.0],
    "confidence_intervals": [
      [1.0, 1.0],
      [0.4, 1.0],
      [0.1, 0.7],
      [-0.1, 0.5],
      [-0.2, 0.4],
      [-0.3, 0.3]
    ]
  },
  "pacf": {
    "values": [1.0, 0.7, 0.1, 0.0, 0.0, 0.0],
    "confidence_intervals": [
      [1.0, 1.0],
      [0.4, 1.0],
      [-0.2, 0.4],
      [-0.3, 0.3],
      [-0.3, 0.3],
      [-0.3, 0.3]
    ]
  },
  "lags": [0, 1, 2, 3, 4, 5],
  "suggested_ar_order": 1,
  "suggested_ma_order": 0
}
```

### List Available Methods

```
GET /api/v1/statistical/methods
```

**Response:**
```json
{
  "stationarity_tests": [
    {
      "id": "adf",
      "name": "Augmented Dickey-Fuller",
      "description": "Tests the null hypothesis that a unit root is present in a time series sample"
    },
    ...
  ],
  "anomaly_detection_methods": [
    {
      "id": "iqr",
      "name": "Interquartile Range",
      "description": "Detects anomalies based on the interquartile range"
    },
    ...
  ],
  "correlation_methods": [
    {
      "id": "pearson",
      "name": "Pearson",
      "description": "Measures linear correlation between two variables"
    },
    ...
  ]
}
```

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of a request:

- `200 OK`: The request was successful.
- `400 Bad Request`: The request was invalid or cannot be processed.
- `401 Unauthorized`: Authentication failed or not provided.
- `403 Forbidden`: The authenticated user does not have permission to access the requested resource.
- `404 Not Found`: The requested resource was not found.
- `429 Too Many Requests`: Rate limit exceeded.
- `500 Internal Server Error`: An error occurred on the server.

Error responses include a JSON body with details:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse. By default, clients are limited to 60 requests per minute. Rate limit information is included in the response headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1617567600
```

## Caching

The API implements caching to improve performance. Responses include cache control headers:

```
Cache-Control: max-age=3600
ETag: "1234567890abcdef"
```

To use a cached response, include the `If-None-Match` header in your request:

```
If-None-Match: "1234567890abcdef"
```

If the resource has not changed, the API will return a `304 Not Modified` response with no body.
