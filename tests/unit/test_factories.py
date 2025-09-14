#!/usr/bin/env python3
"""
Phase 6: Unit Tests for Factory Pattern
Tests for all factory implementations
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from architecture.factories import (
    ModelFactory, FeatureFactory, EvaluationFactory,
    model_factory, feature_factory, evaluation_factory,
    register_custom_component
)


class TestBaseFactory:
    """Tests for base factory functionality"""

    def test_registry_initialization(self, clean_factories):
        """Test factory registry is initialized"""
        factory = ModelFactory()
        assert hasattr(factory, '_registry')
        assert isinstance(factory._registry, dict)

    def test_register_component(self, clean_factories):
        """Test registering a component"""
        factory = ModelFactory()

        class MockModel:
            pass

        factory.register('mock_model', MockModel)
        assert factory.is_registered('mock_model')
        assert 'mock_model' in factory.list_registered()

    def test_register_invalid_component(self, clean_factories):
        """Test registering invalid component raises error"""
        factory = ModelFactory()

        with pytest.raises(ValueError, match="Expected class"):
            factory.register('invalid', "not a class")

    def test_list_registered_components(self, clean_factories):
        """Test listing registered components"""
        factory = ModelFactory()

        class MockModel1:
            pass
        class MockModel2:
            pass

        factory.register('model1', MockModel1)
        factory.register('model2', MockModel2)

        registered = factory.list_registered()
        assert 'model1' in registered
        assert 'model2' in registered


class TestModelFactory:
    """Tests for ModelFactory"""

    def test_model_factory_initialization(self, clean_factories):
        """Test model factory initializes with default models"""
        factory = ModelFactory()
        assert factory.is_registered('prophet') or factory.is_registered('lightgbm')

    @patch('architecture.factories.LightGBMMaster')
    def test_create_lightgbm_model(self, mock_lightgbm, clean_factories):
        """Test creating LightGBM model"""
        factory = ModelFactory()

        # Mock the LightGBM class
        mock_instance = Mock()
        mock_lightgbm.return_value = mock_instance
        factory.register('lightgbm', mock_lightgbm)

        config = {'n_estimators': 100, 'learning_rate': 0.1}
        model = factory.create('lightgbm', config)

        assert model == mock_instance
        mock_lightgbm.assert_called_once()

    @patch('architecture.factories.ProphetSeasonal')
    def test_create_prophet_model(self, mock_prophet, clean_factories):
        """Test creating Prophet model"""
        factory = ModelFactory()

        mock_instance = Mock()
        mock_prophet.return_value = mock_instance
        factory.register('prophet', mock_prophet)

        config = {'seasonality_mode': 'additive'}
        model = factory.create('prophet', config)

        assert model == mock_instance
        mock_prophet.assert_called_once()

    def test_create_unregistered_model(self, clean_factories):
        """Test creating unregistered model raises error"""
        factory = ModelFactory()

        with pytest.raises(ValueError, match="not registered"):
            factory.create('nonexistent_model')

    @patch('architecture.factories.LightGBMMaster')
    def test_create_model_with_kwargs(self, mock_lightgbm, clean_factories):
        """Test creating model with additional kwargs"""
        factory = ModelFactory()

        mock_instance = Mock()
        mock_lightgbm.return_value = mock_instance
        factory.register('lightgbm', mock_lightgbm)

        config = {'n_estimators': 100}
        model = factory.create('lightgbm', config, learning_rate=0.05)

        # Should merge config with kwargs
        expected_config = {'n_estimators': 100, 'learning_rate': 0.05}
        mock_lightgbm.assert_called_once()

    def test_model_creation_failure(self, clean_factories):
        """Test model creation failure handling"""
        factory = ModelFactory()

        class FailingModel:
            def __init__(self, **kwargs):
                raise RuntimeError("Creation failed")

        factory.register('failing_model', FailingModel)

        with pytest.raises(RuntimeError, match="Model creation failed"):
            factory.create('failing_model')


class TestFeatureFactory:
    """Tests for FeatureFactory"""

    def test_feature_factory_initialization(self, clean_factories):
        """Test feature factory initializes"""
        factory = FeatureFactory()
        assert hasattr(factory, '_registry')

    @patch('architecture.factories.TemporalFeaturesEngine')
    def test_create_temporal_features(self, mock_temporal, clean_factories):
        """Test creating temporal features"""
        factory = FeatureFactory()

        mock_instance = Mock()
        mock_temporal.return_value = mock_instance
        factory.register('temporal', mock_temporal)

        features = factory.create('temporal', {'lag_periods': [1, 2, 3]})

        assert features == mock_instance
        mock_temporal.assert_called_once()

    def test_create_unregistered_features(self, clean_factories):
        """Test creating unregistered features raises error"""
        factory = FeatureFactory()

        with pytest.raises(ValueError, match="not registered"):
            factory.create('nonexistent_features')

    def test_feature_creation_failure(self, clean_factories):
        """Test feature creation failure handling"""
        factory = FeatureFactory()

        class FailingFeatures:
            def __init__(self, **kwargs):
                raise RuntimeError("Feature creation failed")

        factory.register('failing_features', FailingFeatures)

        with pytest.raises(RuntimeError, match="Feature creation failed"):
            factory.create('failing_features')


class TestEvaluationFactory:
    """Tests for EvaluationFactory"""

    def test_evaluation_factory_initialization(self, clean_factories):
        """Test evaluation factory initializes"""
        factory = EvaluationFactory()
        assert hasattr(factory, '_registry')

    @patch('architecture.factories.wmape')
    def test_create_metric_function(self, mock_wmape, clean_factories):
        """Test creating metric function"""
        factory = EvaluationFactory()

        def mock_wmape_func(y_true, y_pred):
            return 0.15

        factory._registry['wmape'] = mock_wmape_func

        metric = factory.create('wmape')
        assert metric == mock_wmape_func

    @patch('architecture.factories.ErrorDecomposer')
    def test_create_evaluation_class(self, mock_decomposer, clean_factories):
        """Test creating evaluation class"""
        factory = EvaluationFactory()

        mock_instance = Mock()
        mock_decomposer.return_value = mock_instance
        factory.register('error_decomposer', mock_decomposer)

        evaluator = factory.create('error_decomposer')

        assert evaluator == mock_instance
        mock_decomposer.assert_called_once()

    def test_create_unregistered_evaluator(self, clean_factories):
        """Test creating unregistered evaluator raises error"""
        factory = EvaluationFactory()

        with pytest.raises(ValueError, match="not registered"):
            factory.create('nonexistent_evaluator')


class TestGlobalFactoryInstances:
    """Tests for global factory instances"""

    def test_global_model_factory_exists(self):
        """Test global model factory instance exists"""
        assert model_factory is not None
        assert isinstance(model_factory, ModelFactory)

    def test_global_feature_factory_exists(self):
        """Test global feature factory instance exists"""
        assert feature_factory is not None
        assert isinstance(feature_factory, FeatureFactory)

    def test_global_evaluation_factory_exists(self):
        """Test global evaluation factory instance exists"""
        assert evaluation_factory is not None
        assert isinstance(evaluation_factory, EvaluationFactory)

    def test_register_custom_component_model(self, clean_factories):
        """Test registering custom component via global function"""
        class CustomModel:
            pass

        register_custom_component('model', 'custom_model', CustomModel)
        assert model_factory.is_registered('custom_model')

    def test_register_custom_component_feature(self, clean_factories):
        """Test registering custom feature component"""
        class CustomFeature:
            pass

        register_custom_component('feature', 'custom_feature', CustomFeature)
        assert feature_factory.is_registered('custom_feature')

    def test_register_custom_component_evaluation(self, clean_factories):
        """Test registering custom evaluation component"""
        class CustomEvaluator:
            pass

        register_custom_component('evaluation', 'custom_evaluator', CustomEvaluator)
        assert evaluation_factory.is_registered('custom_evaluator')

    def test_register_custom_component_invalid_factory(self, clean_factories):
        """Test registering to invalid factory raises error"""
        class CustomComponent:
            pass

        with pytest.raises(ValueError, match="Unknown factory"):
            register_custom_component('invalid_factory', 'component', CustomComponent)


class TestFactoryIntegration:
    """Integration tests for factory pattern"""

    @pytest.mark.integration
    def test_factory_pipeline_integration(self, clean_factories):
        """Test factories work together in a pipeline"""
        # Register mock components
        class MockModel:
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.zeros(len(X))

        class MockFeatures:
            def engineer_features(self, data):
                return data

        def mock_metric(y_true, y_pred):
            return 0.15

        model_factory.register('mock_model', MockModel)
        feature_factory.register('mock_features', MockFeatures)
        evaluation_factory._registry['mock_metric'] = mock_metric

        # Test pipeline
        features = feature_factory.create('mock_features')
        model = model_factory.create('mock_model')
        metric = evaluation_factory.create('mock_metric')

        # Simulate pipeline
        data = pd.DataFrame({'feature_1': [1, 2, 3], 'target': [1, 2, 3]})
        engineered_data = features.engineer_features(data)

        X = engineered_data[['feature_1']]
        y = engineered_data['target']

        model.fit(X, y)
        predictions = model.predict(X)
        score = metric(y, predictions)

        assert score == 0.15
        assert len(predictions) == len(y)

    @pytest.mark.performance
    def test_factory_creation_performance(self, clean_factories):
        """Test factory creation performance"""
        import time

        class FastModel:
            def __init__(self, **kwargs):
                pass

        model_factory.register('fast_model', FastModel)

        # Time multiple creations
        start_time = time.time()
        for _ in range(100):
            model_factory.create('fast_model')
        end_time = time.time()

        # Should be fast (less than 1 second for 100 creations)
        assert (end_time - start_time) < 1.0


@pytest.mark.parametrize("factory_name,factory_instance", [
    ("model", model_factory),
    ("feature", feature_factory),
    ("evaluation", evaluation_factory)
])
class TestFactoryParametrized:
    """Parametrized tests for all factories"""

    def test_factory_has_registry(self, factory_name, factory_instance):
        """Test all factories have registry"""
        assert hasattr(factory_instance, '_registry')
        assert isinstance(factory_instance._registry, dict)

    def test_factory_has_required_methods(self, factory_name, factory_instance):
        """Test all factories have required methods"""
        assert hasattr(factory_instance, 'create')
        assert hasattr(factory_instance, 'register')
        assert hasattr(factory_instance, 'list_registered')
        assert hasattr(factory_instance, 'is_registered')

    def test_factory_list_registered_returns_list(self, factory_name, factory_instance):
        """Test list_registered returns list"""
        registered = factory_instance.list_registered()
        assert isinstance(registered, list)

    def test_factory_is_registered_returns_bool(self, factory_name, factory_instance):
        """Test is_registered returns boolean"""
        result = factory_instance.is_registered('nonexistent')
        assert isinstance(result, bool)
        assert result is False