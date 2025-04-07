from celery import Celery
import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, Any, List, Union, Optional

from app.core.config import settings

# Initialize Celery
celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Configure Celery
celery_app.conf.task_routes = {
    "app.worker.forecast_task": "main-queue",
    "app.worker.analyze_task": "main-queue",
    "app.worker.batch_process_task": "batch-queue"
}

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=4,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=1800,  # 30 minutes
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="app.worker.forecast_task")
def forecast_task(
    self,
    data: Union[List[float], List[Dict[str, Any]]],
    model_type: str,
    periods: int,
    frequency: str,
    confidence_interval: float = 0.95,
    seasonal_periods: Optional[int] = None,
    include_history: bool = False,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Celery task for forecasting time series data.
    This runs as a background task for longer forecasts.
    """
    try:
        logger.info(f"Starting forecast task with model {model_type} for {periods} periods")
        
        # Import here to avoid circular imports
        from app.services.forecasting.factory import ForecastingFactory
        
        # Generate forecast
        result = ForecastingFactory.forecast(
            data=data,
            model_type=model_type,
            periods=periods,
            frequency=frequency,
            confidence_interval=confidence_interval,
            seasonal_periods=seasonal_periods,
            include_history=include_history,
            params=params
        )
        
        logger.info(f"Forecast task completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error in forecast task: {str(e)}")
        self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(bind=True, name="app.worker.analyze_task")
def analyze_task(
    self,
    data: Union[List[float], List[Dict[str, Any]]],
    analysis_type: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Celery task for analyzing time series data.
    This runs as a background task for complex analyses.
    """
    try:
        logger.info(f"Starting analysis task of type {analysis_type}")
        
        if params is None:
            params = {}
        
        if analysis_type == "stationarity":
            # Import here to avoid circular imports
            from app.services.statistical.analysis import StatisticalAnalysisService
            
            # Initialize service
            analysis_service = StatisticalAnalysisService()
            
            # Test stationarity
            result = analysis_service.test_stationarity(
                data=data,
                test=params.get("test", "adf"),
                alpha=params.get("alpha", 0.05)
            )
        
        elif analysis_type == "anomalies":
            # Import here to avoid circular imports
            from app.services.statistical.analysis import StatisticalAnalysisService
            
            # Initialize service
            analysis_service = StatisticalAnalysisService()
            
            # Detect anomalies
            result = analysis_service.detect_anomalies(
                data=data,
                method=params.get("method", "iqr"),
                sensitivity=params.get("sensitivity", 1.5),
                window_size=params.get("window_size")
            )
        
        elif analysis_type == "technical_indicator":
            # Import here to avoid circular imports
            from app.services.technical.indicators import TechnicalIndicatorService
            
            # Initialize service
            indicator_service = TechnicalIndicatorService()
            
            # Calculate indicator
            result = indicator_service.calculate_indicator(
                indicator_type=params.get("indicator", "sma"),
                data=data,
                window=params.get("window", 14),
                additional_params=params.get("additional_params")
            )
        
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        logger.info(f"Analysis task completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error in analysis task: {str(e)}")
        self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(bind=True, name="app.worker.batch_process_task")
def batch_process_task(
    self,
    file_path: str,
    task_type: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Celery task for batch processing time series data from a file.
    This runs as a background task for processing large datasets.
    """
    try:
        logger.info(f"Starting batch process task of type {task_type} for file {file_path}")
        
        if params is None:
            params = {}
        
        # Load data from file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Process data based on task type
        if task_type == "forecast":
            # Import here to avoid circular imports
            from app.services.forecasting.factory import ForecastingFactory
            
            # Extract time series data
            date_col = params.get("date_column", "date")
            value_col = params.get("value_column", "value")
            
            if date_col in df.columns and value_col in df.columns:
                # Convert to list of dicts
                data = df[[date_col, value_col]].rename(
                    columns={date_col: "date", value_col: "value"}
                ).to_dict('records')
                
                # Generate forecast
                result = ForecastingFactory.forecast(
                    data=data,
                    model_type=params.get("model", "prophet"),
                    periods=params.get("periods", 10),
                    frequency=params.get("frequency", "D"),
                    confidence_interval=params.get("confidence_interval", 0.95),
                    seasonal_periods=params.get("seasonal_periods"),
                    include_history=params.get("include_history", False),
                    params=params.get("model_params")
                )
            else:
                raise ValueError(f"Required columns {date_col} and {value_col} not found in file")
        
        elif task_type == "analyze":
            # Process multiple time series for analysis
            results = []
            
            # Group by series identifier if specified
            group_col = params.get("group_column")
            date_col = params.get("date_column", "date")
            value_col = params.get("value_column", "value")
            
            if group_col and group_col in df.columns:
                # Process each group separately
                for group_name, group_df in df.groupby(group_col):
                    if date_col in group_df.columns and value_col in group_df.columns:
                        # Sort by date
                        group_df = group_df.sort_values(date_col)
                        
                        # Extract values
                        values = group_df[value_col].tolist()
                        
                        # Run analysis task
                        result = analyze_task.delay(
                            data=values,
                            analysis_type=params.get("analysis_type", "stationarity"),
                            params=params.get("analysis_params", {})
                        ).get()
                        
                        results.append({
                            "group": group_name,
                            "result": result
                        })
                
                result = {"groups": results}
            
            else:
                # Process entire dataset as one series
                if date_col in df.columns and value_col in df.columns:
                    # Sort by date
                    df = df.sort_values(date_col)
                    
                    # Extract values
                    values = df[value_col].tolist()
                    
                    # Run analysis task
                    result = analyze_task.delay(
                        data=values,
                        analysis_type=params.get("analysis_type", "stationarity"),
                        params=params.get("analysis_params", {})
                    ).get()
                else:
                    raise ValueError(f"Required columns {date_col} and {value_col} not found in file")
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        logger.info(f"Batch process task completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error in batch process task: {str(e)}")
        self.retry(exc=e, countdown=60, max_retries=3)
