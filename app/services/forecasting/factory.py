from typing import Dict, Any, Union, List, Optional
from app.models.forecasting import ForecastingModel, Frequency
from app.services.forecasting.prophet_service import ProphetService
from app.services.forecasting.arima_service import ARIMAService

class ForecastingFactory:
    """Factory class to create appropriate forecasting service based on model type"""
    
    @staticmethod
    def get_service(model_type: ForecastingModel):
        """Get the appropriate forecasting service"""
        if model_type == ForecastingModel.PROPHET:
            return ProphetService()
        elif model_type in [ForecastingModel.ARIMA, ForecastingModel.SARIMA]:
            return ARIMAService()
        elif model_type == ForecastingModel.EXPONENTIAL_SMOOTHING:
            # TODO: Implement Exponential Smoothing service
            raise NotImplementedError("Exponential Smoothing model not yet implemented")
        elif model_type == ForecastingModel.LSTM:
            # TODO: Implement LSTM service
            raise NotImplementedError("LSTM model not yet implemented")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def forecast(
        data: Union[List[float], List[Dict[str, Any]]],
        model_type: ForecastingModel,
        periods: int,
        frequency: Frequency,
        confidence_interval: float = 0.95,
        seasonal_periods: Optional[int] = None,
        include_history: bool = False,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate forecast using the specified model"""
        # Get appropriate service
        service = ForecastingFactory.get_service(model_type)
        
        # Train model
        service.train(data, frequency, seasonal_periods, params)
        
        # Generate forecast
        service.predict(periods, frequency, confidence_interval)
        
        # Format results
        result = service.format_results(include_history)
        result['model'] = model_type
        
        return result
