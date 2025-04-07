from typing import Dict, Any, List, Union, Optional
import logging

from app.services.forecasting.prophet_service import ProphetForecastingService
from app.services.forecasting.arima_service import ARIMAForecastingService
from app.services.forecasting.lstm_service import LSTMForecastingService
from app.models.forecasting import ForecastingModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingFactory:
    """Factory for creating forecasting services"""
    
    @staticmethod
    def create_forecasting_service(model_type: str):
        """Create a forecasting service based on model type"""
        if model_type.lower() == "prophet" or model_type == ForecastingModel.PROPHET:
            return ProphetForecastingService()
        elif model_type.lower() == "arima" or model_type in [ForecastingModel.ARIMA, ForecastingModel.SARIMA]:
            return ARIMAForecastingService()
        elif model_type.lower() == "lstm" or model_type == ForecastingModel.LSTM:
            return LSTMForecastingService()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def forecast(
        data: Union[List[float], List[Dict[str, Any]]],
        model_type: str,
        periods: int,
        frequency: str,
        confidence_interval: float = 0.95,
        seasonal_periods: Optional[int] = None,
        include_history: bool = False,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate forecasts using the specified model"""
        logger.info(f"Creating forecast with model {model_type} for {periods} periods")
        
        # Create forecasting service
        service = ForecastingFactory.create_forecasting_service(model_type)
        
        # Train model
        service.train(data, frequency)
        
        # Generate forecast
        forecast = service.predict(
            periods=periods,
            confidence_interval=confidence_interval
        )
        
        return forecast
    
    @staticmethod
    def decompose(
        data: Union[List[float], List[Dict[str, Any]]],
        model_type: str,
        frequency: str
    ) -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components"""
        logger.info(f"Decomposing time series with model {model_type}")
        
        # Currently only Prophet supports decomposition
        if model_type.lower() == "prophet" or model_type == ForecastingModel.PROPHET:
            service = ProphetForecastingService()
            service.train(data, frequency)
            return service.decompose()
        else:
            raise ValueError(f"Decomposition not supported for model type: {model_type}")
