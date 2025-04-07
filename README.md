# Time Series Analysis API

A high-performance REST API for time series analysis and forecasting, built with FastAPI and modern ML libraries.

## Features

- **Multiple Forecasting Models**:
  - ARIMA/SARIMA
  - Prophet
  - LSTM
  - Exponential Smoothing
  
- **Technical Analysis**:
  - Moving Averages
  - Bollinger Bands
  - RSI, MACD
  - Custom Indicators
  
- **Statistical Analysis**:
  - Trend Detection
  - Seasonality Analysis
  - Stationarity Tests
  - Anomaly Detection
  
- **Data Processing**:
  - Automatic Data Cleaning
  - Missing Value Handling
  - Resampling
  - Feature Engineering

- **Async Processing**:
  - Background Tasks
  - Celery Integration
  - Real-time Updates
  
- **API Features**:
  - FastAPI with OpenAPI Docs
  - JWT Authentication
  - Rate Limiting
  - Caching
  - Swagger Documentation

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

4. Start Redis (for caching and Celery):
```bash
docker-compose up -d redis
```

5. Run the API:
```bash
uvicorn app.main:app --reload
```

6. Start Celery worker (optional):
```bash
celery -A app.worker worker --loglevel=info
```

7. Start Flower monitoring (optional):
```bash
celery -A app.worker flower
```

## API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
time-series-api/
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── forecasting.py
│   │   │   ├── technical.py
│   │   │   └── statistical.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── models/
│   │   ├── forecasting.py
│   │   └── technical.py
│   ├── services/
│   │   ├── forecasting/
│   │   ├── technical/
│   │   └── statistical/
│   ├── utils/
│   │   ├── data_processing.py
│   │   └── validation.py
│   ├── worker.py
│   └── main.py
├── tests/
│   ├── api/
│   ├── services/
│   └── conftest.py
├── docker/
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

## Examples

### Forecasting Example

```python
import requests

data = {
    "data": [1, 2, 3, 4, 5],
    "model": "prophet",
    "periods": 5,
    "frequency": "D"
}

response = requests.post(
    "http://localhost:8000/api/v1/forecast",
    json=data
)

print(response.json())
```

### Technical Analysis Example

```python
import requests

data = {
    "data": [10, 12, 15, 14, 13],
    "indicator": "sma",
    "window": 3
}

response = requests.post(
    "http://localhost:8000/api/v1/technical/indicator",
    json=data
)

print(response.json())
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
