#!/usr/bin/env python3
"""
Phase 6: Factory Pattern Implementation
Enterprise-grade factories for dynamic component creation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Union
import importlib
import inspect
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class BaseFactory(ABC):
    """Abstract base factory with common functionality"""

    def __init__(self):
        self._registry: Dict[str, Type] = {}
        self._config_cache: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def create(self, component_type: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Create component instance"""
        pass

    def register(self, name: str, component_class: Type) -> None:
        """Register a component class"""
        if not inspect.isclass(component_class):
            raise ValueError(f"Expected class, got {type(component_class)}")

        self._registry[name] = component_class
        logger.info(f"Registered {component_class.__name__} as '{name}'")

    def list_registered(self) -> List[str]:
        """List all registered component names"""
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if component is registered"""
        return name in self._registry

class ModelFactory(BaseFactory):
    """Factory for creating forecast models dynamically"""

    def __init__(self):
        super().__init__()
        self._initialize_default_models()

    def _initialize_default_models(self) -> None:
        """Register default Phase 5 models"""
        try:
            # Prophet for seasonal patterns
            from ..models.prophet_seasonal import ProphetSeasonal
            self.register('prophet', ProphetSeasonal)

            # LightGBM for feature interactions
            from ..models.lightgbm_master import LightGBMMaster
            self.register('lightgbm', LightGBMMaster)

            # LSTM for temporal dependencies
            from ..models.lstm_temporal import LSTMTemporal
            self.register('lstm', LSTMTemporal)

            # ARIMA for time series structure
            from ..models.arima_temporal import ARIMARegressor
            self.register('arima', ARIMARegressor)

            # Advanced ensemble
            from ..models.advanced_ensemble import AdvancedEnsembleOrchestrator
            self.register('ensemble', AdvancedEnsembleOrchestrator)

        except ImportError as e:
            logger.warning(f"Could not register default model: {e}")

    def create(self, model_type: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Create model instance

        Args:
            model_type: Type of model ('prophet', 'lightgbm', 'lstm', 'arima', 'ensemble')
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Model instance

        Raises:
            ValueError: If model type not registered
            Exception: If model creation fails
        """
        if not self.is_registered(model_type):
            raise ValueError(
                f"Model type '{model_type}' not registered. "
                f"Available: {', '.join(self.list_registered())}"
            )

        model_class = self._registry[model_type]

        try:
            # Merge config with kwargs
            final_config = {**(config or {}), **kwargs}

            # Handle different model initialization patterns
            if model_type == 'prophet':
                return self._create_prophet(model_class, final_config)
            elif model_type == 'lightgbm':
                return self._create_lightgbm(model_class, final_config)
            elif model_type == 'lstm':
                return self._create_lstm(model_class, final_config)
            elif model_type == 'arima':
                return self._create_arima(model_class, final_config)
            elif model_type == 'ensemble':
                return self._create_ensemble(model_class, final_config)
            else:
                # Generic initialization
                return model_class(**final_config)

        except Exception as e:
            logger.error(f"Failed to create {model_type} model: {e}")
            raise RuntimeError(f"Model creation failed: {e}") from e

    def _create_prophet(self, model_class: Type, config: Dict[str, Any]) -> Any:
        """Create Prophet model with specific configuration"""
        prophet_params = {
            'seasonality_mode': config.get('seasonality_mode', 'multiplicative'),
            'yearly_seasonality': config.get('yearly_seasonality', True),
            'weekly_seasonality': config.get('weekly_seasonality', True),
            'daily_seasonality': config.get('daily_seasonality', False),
            'interval_width': config.get('interval_width', 0.95)
        }
        return model_class(**prophet_params)

    def _create_lightgbm(self, model_class: Type, config: Dict[str, Any]) -> Any:
        """Create LightGBM model with specific configuration"""
        lgb_params = {
            'n_estimators': config.get('n_estimators', 1000),
            'learning_rate': config.get('learning_rate', 0.1),
            'num_leaves': config.get('num_leaves', 31),
            'feature_fraction': config.get('feature_fraction', 0.8),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'random_state': config.get('random_state', 42)
        }
        return model_class(**lgb_params)

    def _create_lstm(self, model_class: Type, config: Dict[str, Any]) -> Any:
        """Create LSTM model with specific configuration"""
        lstm_params = {
            'sequence_length': config.get('sequence_length', 30),
            'hidden_size': config.get('hidden_size', 128),
            'num_layers': config.get('num_layers', 2),
            'dropout': config.get('dropout', 0.2),
            'learning_rate': config.get('learning_rate', 0.001)
        }
        return model_class(**lstm_params)

    def _create_arima(self, model_class: Type, config: Dict[str, Any]) -> Any:
        """Create ARIMA model with specific configuration"""
        arima_params = {
            'seasonal_periods': config.get('seasonal_periods', [7, 30]),
            'max_p': config.get('max_p', 3),
            'max_d': config.get('max_d', 2),
            'max_q': config.get('max_q', 3),
            'seasonal': config.get('seasonal', True)
        }
        return model_class(**arima_params)

    def _create_ensemble(self, model_class: Type, config: Dict[str, Any]) -> Any:
        """Create ensemble model with specific configuration"""
        ensemble_params = {
            'base_models': config.get('base_models', ['prophet', 'lightgbm', 'lstm', 'arima']),
            'meta_learner': config.get('meta_learner', 'ridge_regression'),
            'cv_folds': config.get('cv_folds', 5),
            'random_state': config.get('random_state', 42)
        }
        return model_class(**ensemble_params)

class FeatureFactory(BaseFactory):
    """Factory for creating feature engineering components"""

    def __init__(self):
        super().__init__()
        self._initialize_default_features()

    def _initialize_default_features(self) -> None:
        """Register default feature engines"""
        try:
            from ..features.temporal_features_engine import TemporalFeaturesEngine
            self.register('temporal', TemporalFeaturesEngine)

            from ..features.aggregation_features_engine import AggregationFeaturesEngine
            self.register('aggregation', AggregationFeaturesEngine)

            from ..features.behavioral_features_engine import BehavioralFeaturesEngine
            self.register('behavioral', BehavioralFeaturesEngine)

            from ..features.business_features_engine import BusinessFeaturesEngine
            self.register('business', BusinessFeaturesEngine)

        except ImportError as e:
            logger.warning(f"Could not register default features: {e}")

    def create(self, feature_type: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Create feature engine instance

        Args:
            feature_type: Type of features ('temporal', 'aggregation', 'behavioral', 'business')
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Feature engine instance
        """
        if not self.is_registered(feature_type):
            raise ValueError(
                f"Feature type '{feature_type}' not registered. "
                f"Available: {', '.join(self.list_registered())}"
            )

        feature_class = self._registry[feature_type]

        try:
            final_config = {**(config or {}), **kwargs}
            return feature_class(**final_config)

        except Exception as e:
            logger.error(f"Failed to create {feature_type} features: {e}")
            raise RuntimeError(f"Feature creation failed: {e}") from e

class EvaluationFactory(BaseFactory):
    """Factory for creating evaluation components"""

    def __init__(self):
        super().__init__()
        self._initialize_default_evaluators()

    def _initialize_default_evaluators(self) -> None:
        """Register default evaluators"""
        try:
            from ..evaluation.metrics import wmape, mape, mae, rmse
            from ..evaluation.error_analysis import ErrorDecomposer, ErrorVisualizationEngine
            from ..evaluation.model_diagnostics import ModelDiagnostics

            # Register metric functions
            self._registry['wmape'] = wmape
            self._registry['mape'] = mape
            self._registry['mae'] = mae
            self._registry['rmse'] = rmse

            # Register analysis classes
            self.register('error_decomposer', ErrorDecomposer)
            self.register('error_visualizer', ErrorVisualizationEngine)
            self.register('model_diagnostics', ModelDiagnostics)

        except ImportError as e:
            logger.warning(f"Could not register default evaluators: {e}")

    def create(self, evaluator_type: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Create evaluation component

        Args:
            evaluator_type: Type of evaluator
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Evaluator instance or function
        """
        if not self.is_registered(evaluator_type):
            raise ValueError(
                f"Evaluator type '{evaluator_type}' not registered. "
                f"Available: {', '.join(self.list_registered())}"
            )

        evaluator = self._registry[evaluator_type]

        try:
            # If it's a class, instantiate it
            if inspect.isclass(evaluator):
                final_config = {**(config or {}), **kwargs}
                return evaluator(**final_config)
            else:
                # It's a function, return it directly
                return evaluator

        except Exception as e:
            logger.error(f"Failed to create {evaluator_type} evaluator: {e}")
            raise RuntimeError(f"Evaluator creation failed: {e}") from e

# Global factory instances for easy access
model_factory = ModelFactory()
feature_factory = FeatureFactory()
evaluation_factory = EvaluationFactory()

def register_custom_component(factory_name: str, component_name: str, component_class: Type) -> None:
    """
    Register custom component with specified factory

    Args:
        factory_name: Name of factory ('model', 'feature', 'evaluation')
        component_name: Name to register component under
        component_class: Class to register
    """
    factories = {
        'model': model_factory,
        'feature': feature_factory,
        'evaluation': evaluation_factory
    }

    if factory_name not in factories:
        raise ValueError(f"Unknown factory: {factory_name}. Available: {', '.join(factories.keys())}")

    factories[factory_name].register(component_name, component_class)

if __name__ == "__main__":
    # Demo usage
    print("🏭 Phase 6 Factory Pattern Demo")
    print("=" * 50)

    # Model factory demo
    print("\n📊 Model Factory:")
    for model_type in model_factory.list_registered():
        print(f"  - {model_type}")
        try:
            model = model_factory.create(model_type)
            print(f"    ✅ Created: {model.__class__.__name__}")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    # Feature factory demo
    print("\n🔧 Feature Factory:")
    for feature_type in feature_factory.list_registered():
        print(f"  - {feature_type}")
        try:
            features = feature_factory.create(feature_type)
            print(f"    ✅ Created: {features.__class__.__name__}")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    # Evaluation factory demo
    print("\n📈 Evaluation Factory:")
    for eval_type in evaluation_factory.list_registered():
        print(f"  - {eval_type}")
        try:
            evaluator = evaluation_factory.create(eval_type)
            if callable(evaluator):
                print(f"    ✅ Function: {evaluator.__name__}")
            else:
                print(f"    ✅ Created: {evaluator.__class__.__name__}")
        except Exception as e:
            print(f"    ❌ Error: {e}")