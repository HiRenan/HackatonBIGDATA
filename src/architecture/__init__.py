"""
Phase 6: Architecture Patterns Package
Production-ready design patterns for forecasting system
"""

__version__ = "1.0.0"
__author__ = "Hackathon Forecast Team"

from .factories import ModelFactory, FeatureFactory, EvaluationFactory
from .strategies import ForecastStrategy, ValidationStrategy, OptimizationStrategy
from .observers import TrainingObserver, ValidationObserver, BusinessRulesObserver
from .pipelines import DataProcessingPipeline, FeaturePipeline, ModelTrainingPipeline

__all__ = [
    # Factories
    'ModelFactory',
    'FeatureFactory',
    'EvaluationFactory',

    # Strategies
    'ForecastStrategy',
    'ValidationStrategy',
    'OptimizationStrategy',

    # Observers
    'TrainingObserver',
    'ValidationObserver',
    'BusinessRulesObserver',

    # Pipelines
    'DataProcessingPipeline',
    'FeaturePipeline',
    'ModelTrainingPipeline'
]