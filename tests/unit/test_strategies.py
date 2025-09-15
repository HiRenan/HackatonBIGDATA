#!/usr/bin/env python3
"""
Phase 6: Unit Tests for Strategy Pattern
Tests for all strategy implementations
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from architecture.strategies import (
    BaseStrategy, ForecastStrategy, ProphetForecastStrategy,
    EnsembleForecastStrategy, ValidationStrategy,
    TimeSeriesValidationStrategy, WalkForwardValidationStrategy,
    OptimizationStrategy, OptunaOptimizationStrategy,
    create_strategy, STRATEGY_REGISTRY
)


class TestBaseStrategy:
    """Tests for BaseStrategy abstract base class"""

    def test_base_strategy_initialization(self):
        """Test BaseStrategy initialization"""
        class ConcreteStrategy(BaseStrategy):
            def _setup(self):
                self.test_param = self.config.get('test_param', 'default')

            def execute(self, *args, **kwargs):
                return self.test_param

        config = {'test_param': 'custom_value'}
        strategy = ConcreteStrategy(config)

        assert strategy.config == config
        assert strategy.test_param == 'custom_value'

    def test_base_strategy_default_config(self):
        """Test BaseStrategy with default config"""
        class ConcreteStrategy(BaseStrategy):
            def _setup(self):
                pass

            def execute(self, *args, **kwargs):
                return "executed"

        strategy = ConcreteStrategy()
        assert strategy.config == {}
        assert strategy.execute() == "executed"


class TestProphetForecastStrategy:
    """Tests for ProphetForecastStrategy"""

    def test_prophet_initialization(self):
        """Test Prophet strategy initialization"""
        config = {
            'seasonality_mode': 'additive',
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'interval_width': 0.99
        }

        strategy = ProphetForecastStrategy(config)

        assert strategy.seasonality_mode == 'additive'
        assert strategy.yearly_seasonality is False
        assert strategy.weekly_seasonality is True
        assert strategy.interval_width == 0.99

    def test_prophet_default_config(self):
        """Test Prophet strategy with default config"""
        strategy = ProphetForecastStrategy()

        assert strategy.seasonality_mode == 'multiplicative'
        assert strategy.yearly_seasonality is True
        assert strategy.weekly_seasonality is True
        assert strategy.interval_width == 0.95

    @patch('architecture.strategies.Prophet')
    def test_prophet_execute(self, mock_prophet_class, time_series_data):
        """Test Prophet execution"""
        # Mock Prophet model
        mock_prophet = Mock()
        mock_prophet_class.return_value = mock_prophet

        # Mock forecast result
        mock_forecast = pd.DataFrame({
            'yhat': [100, 101, 102, 103, 104],
            'yhat_lower': [95, 96, 97, 98, 99],
            'yhat_upper': [105, 106, 107, 108, 109],
            'trend': [100, 100.5, 101, 101.5, 102],
            'yearly': [0, 0.5, 1, 0.5, 0],
            'weekly': [0, 0.2, -0.1, 0.3, 0]
        })
        mock_prophet.predict.return_value = mock_forecast
        mock_prophet.make_future_dataframe.return_value = pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=5)})

        strategy = ProphetForecastStrategy()

        # Prepare test data
        test_data = time_series_data[['date', 'value']].head(100)
        test_data.columns = ['date', 'target']

        result = strategy.execute(test_data, forecast_periods=5)

        # Verify results
        assert 'predictions' in result
        assert 'lower_bounds' in result
        assert 'upper_bounds' in result
        assert 'components' in result

        assert len(result['predictions']) == 5
        np.testing.assert_array_equal(result['predictions'], mock_forecast['yhat'].values)

    @patch('architecture.strategies.Prophet')
    def test_prophet_fit_predict(self, mock_prophet_class, time_series_data):
        """Test Prophet fit and predict methods"""
        mock_prophet = Mock()
        mock_prophet_class.return_value = mock_prophet

        # Mock predict result
        mock_forecast = pd.DataFrame({
            'yhat': [100, 101, 102],
            'yhat_lower': [95, 96, 97],
            'yhat_upper': [105, 106, 107]
        })
        mock_prophet.predict.return_value = mock_forecast

        strategy = ProphetForecastStrategy()

        # Test fit
        X = time_series_data[['date']].head(100)
        y = time_series_data['value'].head(100)

        fitted_strategy = strategy.fit(X, y)
        assert fitted_strategy == strategy

        # Test predict
        strategy.model = mock_prophet  # Set model directly for testing
        predictions = strategy.predict(X.head(3))

        np.testing.assert_array_equal(predictions, mock_forecast['yhat'].values)

    def test_prophet_predict_without_fit(self):
        """Test Prophet predict without fitting raises error"""
        strategy = ProphetForecastStrategy()
        X = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=3)})

        with pytest.raises(RuntimeError, match="Model not fitted"):
            strategy.predict(X)

    def test_prophet_import_error(self):
        """Test Prophet strategy when Prophet is not available"""
        with patch('architecture.strategies.Prophet', side_effect=ImportError):
            strategy = ProphetForecastStrategy()
            test_data = pd.DataFrame({'date': [datetime.now()], 'target': [100]})

            with pytest.raises(RuntimeError, match="Prophet not installed"):
                strategy.execute(test_data)


class TestEnsembleForecastStrategy:
    """Tests for EnsembleForecastStrategy"""

    def test_ensemble_initialization(self):
        """Test Ensemble strategy initialization"""
        config = {
            'base_strategies': ['prophet', 'lightgbm'],
            'meta_learner': 'ridge',
            'cv_folds': 3
        }

        strategy = EnsembleForecastStrategy(config)

        assert strategy.base_strategies == ['prophet', 'lightgbm']
        assert strategy.meta_learner == 'ridge'
        assert strategy.cv_folds == 3

    @patch.object(EnsembleForecastStrategy, '_create_strategy')
    def test_ensemble_execute_success(self, mock_create_strategy, time_series_data):
        """Test successful ensemble execution"""
        # Mock individual strategy results
        mock_strategy1 = Mock()
        mock_strategy1.execute.return_value = {
            'predictions': np.array([100, 101, 102]),
            'lower_bounds': np.array([95, 96, 97]),
            'upper_bounds': np.array([105, 106, 107])
        }

        mock_strategy2 = Mock()
        mock_strategy2.execute.return_value = {
            'predictions': np.array([102, 103, 104]),
            'lower_bounds': np.array([97, 98, 99]),
            'upper_bounds': np.array([107, 108, 109])
        }

        mock_create_strategy.side_effect = [mock_strategy1, mock_strategy2]

        config = {'base_strategies': ['strategy1', 'strategy2']}
        strategy = EnsembleForecastStrategy(config)

        test_data = time_series_data.head(10)
        result = strategy.execute(test_data)

        # Verify ensemble result
        assert 'predictions' in result
        assert 'individual_results' in result
        assert len(result['individual_results']) == 2

        # Should be average of individual predictions
        expected_predictions = np.array([101, 102, 103])
        np.testing.assert_array_equal(result['predictions'], expected_predictions)

    @patch.object(EnsembleForecastStrategy, '_create_strategy')
    def test_ensemble_execute_partial_failure(self, mock_create_strategy, time_series_data):
        """Test ensemble execution with some strategy failures"""
        # First strategy succeeds
        mock_strategy1 = Mock()
        mock_strategy1.execute.return_value = {
            'predictions': np.array([100, 101, 102]),
            'lower_bounds': np.array([95, 96, 97]),
            'upper_bounds': np.array([105, 106, 107])
        }

        # Second strategy fails
        mock_strategy2 = Mock()
        mock_strategy2.execute.side_effect = RuntimeError("Strategy failed")

        mock_create_strategy.side_effect = [mock_strategy1, mock_strategy2]

        config = {'base_strategies': ['strategy1', 'strategy2']}
        strategy = EnsembleForecastStrategy(config)

        test_data = time_series_data.head(10)
        result = strategy.execute(test_data)

        # Should still work with one successful strategy
        assert 'predictions' in result
        assert len(result['individual_results']) == 1

    @patch.object(EnsembleForecastStrategy, '_create_strategy')
    def test_ensemble_execute_all_fail(self, mock_create_strategy, time_series_data):
        """Test ensemble execution when all strategies fail"""
        mock_strategy = Mock()
        mock_strategy.execute.side_effect = RuntimeError("Strategy failed")
        mock_create_strategy.return_value = mock_strategy

        config = {'base_strategies': ['strategy1', 'strategy2']}
        strategy = EnsembleForecastStrategy(config)

        test_data = time_series_data.head(10)

        with pytest.raises(RuntimeError, match="All base strategies failed"):
            strategy.execute(test_data)


class TestTimeSeriesValidationStrategy:
    """Tests for TimeSeriesValidationStrategy"""

    def test_timeseries_validation_initialization(self):
        """Test TimeSeriesValidation initialization"""
        config = {
            'n_splits': 3,
            'test_size': 0.25,
            'gap': 5
        }

        strategy = TimeSeriesValidationStrategy(config)

        assert strategy.n_splits == 3
        assert strategy.test_size == 0.25
        assert strategy.gap == 5

    @patch('sklearn.model_selection.TimeSeriesSplit')
    def test_timeseries_split(self, mock_tscv, ml_dataset):
        """Test time series splitting"""
        # Mock TimeSeriesSplit
        mock_splits = [(np.array([0, 1, 2]), np.array([3, 4])),
                      (np.array([0, 1, 2, 3]), np.array([4, 5]))]
        mock_tscv_instance = Mock()
        mock_tscv_instance.split.return_value = mock_splits
        mock_tscv.return_value = mock_tscv_instance

        strategy = TimeSeriesValidationStrategy({'n_splits': 2})
        X, y = ml_dataset

        splits = strategy.split(X, y)

        assert len(splits) == 2
        assert splits == mock_splits

    def test_timeseries_validate(self, ml_dataset):
        """Test time series validation execution"""
        strategy = TimeSeriesValidationStrategy({'n_splits': 2})
        X, y = ml_dataset

        # Mock model
        mock_model = Mock()
        mock_model.predict.side_effect = [
            np.array([1.0, 1.1]),  # First fold predictions
            np.array([1.2, 1.3])   # Second fold predictions
        ]

        with patch.object(strategy, 'split') as mock_split:
            # Mock splits
            mock_split.return_value = [
                (np.array([0, 1, 2]), np.array([3, 4])),
                (np.array([0, 1, 2, 3]), np.array([4, 5]))
            ]

            result = strategy.validate(mock_model, X, y)

        # Verify results structure
        assert 'wmape_mean' in result
        assert 'wmape_std' in result
        assert 'mae_mean' in result
        assert 'mae_std' in result
        assert 'rmse_mean' in result
        assert 'rmse_std' in result

        # Model should be fitted twice
        assert mock_model.fit.call_count == 2


class TestWalkForwardValidationStrategy:
    """Tests for WalkForwardValidationStrategy"""

    def test_walk_forward_initialization(self):
        """Test WalkForward initialization"""
        config = {
            'window_size': 100,
            'step_size': 10,
            'forecast_horizon': 5
        }

        strategy = WalkForwardValidationStrategy(config)

        assert strategy.window_size == 100
        assert strategy.step_size == 10
        assert strategy.forecast_horizon == 5

    def test_walk_forward_split(self, ml_dataset):
        """Test walk-forward splitting logic"""
        config = {
            'window_size': 50,
            'step_size': 10,
            'forecast_horizon': 5
        }

        strategy = WalkForwardValidationStrategy(config)
        X, y = ml_dataset

        splits = strategy.split(X, y)

        # Should generate multiple splits
        assert len(splits) > 0

        # Each split should have correct format
        for train_idx, val_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(val_idx, np.ndarray)
            assert len(train_idx) <= 50  # Window size
            assert len(val_idx) <= 5     # Forecast horizon


class TestOptunaOptimizationStrategy:
    """Tests for OptunaOptimizationStrategy"""

    def test_optuna_initialization(self):
        """Test Optuna strategy initialization"""
        config = {
            'n_trials': 50,
            'timeout': 1800,
            'direction': 'maximize'
        }

        strategy = OptunaOptimizationStrategy(config)

        assert strategy.n_trials == 50
        assert strategy.timeout == 1800
        assert strategy.direction == 'maximize'

    @patch('architecture.strategies.optuna')
    def test_optuna_optimize(self, mock_optuna):
        """Test Optuna optimization execution"""
        # Mock optuna study
        mock_study = Mock()
        mock_study.best_params = {'param1': 0.5}
        mock_study.best_value = 0.85
        mock_study.trials = [Mock(), Mock()]  # 2 trials

        mock_optuna.create_study.return_value = mock_study

        strategy = OptunaOptimizationStrategy({'n_trials': 2})

        # Mock objective function
        def objective_func(param1):
            return param1 * 2

        search_space = {
            'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
        }

        result = strategy.optimize(objective_func, search_space)

        # Verify results
        assert result['best_params'] == {'param1': 0.5}
        assert result['best_value'] == 0.85
        assert result['n_trials'] == 2
        assert 'study' in result

    def test_optuna_import_error(self):
        """Test Optuna strategy when optuna is not available"""
        with patch('architecture.strategies.optuna', side_effect=ImportError):
            strategy = OptunaOptimizationStrategy()

            with pytest.raises(RuntimeError, match="Optuna not installed"):
                strategy.optimize(lambda x: x, {})


class TestStrategyRegistry:
    """Tests for strategy registry and factory function"""

    def test_strategy_registry_structure(self):
        """Test strategy registry has correct structure"""
        assert 'forecast' in STRATEGY_REGISTRY
        assert 'validation' in STRATEGY_REGISTRY
        assert 'optimization' in STRATEGY_REGISTRY

        assert 'prophet' in STRATEGY_REGISTRY['forecast']
        assert 'ensemble' in STRATEGY_REGISTRY['forecast']

        assert 'timeseries_cv' in STRATEGY_REGISTRY['validation']
        assert 'walk_forward' in STRATEGY_REGISTRY['validation']

        assert 'optuna' in STRATEGY_REGISTRY['optimization']

    def test_create_strategy_valid(self):
        """Test creating valid strategies"""
        prophet_strategy = create_strategy('forecast', 'prophet')
        assert isinstance(prophet_strategy, ProphetForecastStrategy)

        validation_strategy = create_strategy('validation', 'timeseries_cv')
        assert isinstance(validation_strategy, TimeSeriesValidationStrategy)

        optimization_strategy = create_strategy('optimization', 'optuna')
        assert isinstance(optimization_strategy, OptunaOptimizationStrategy)

    def test_create_strategy_with_config(self):
        """Test creating strategy with config"""
        config = {'n_splits': 3, 'test_size': 0.3}
        strategy = create_strategy('validation', 'timeseries_cv', config)

        assert strategy.n_splits == 3
        assert strategy.test_size == 0.3

    def test_create_strategy_invalid_type(self):
        """Test creating strategy with invalid type"""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            create_strategy('invalid_type', 'prophet')

    def test_create_strategy_invalid_name(self):
        """Test creating strategy with invalid name"""
        with pytest.raises(ValueError, match="Unknown forecast strategy"):
            create_strategy('forecast', 'invalid_strategy')


@pytest.mark.parametrize("strategy_type,strategy_name,expected_class", [
    ("forecast", "prophet", ProphetForecastStrategy),
    ("forecast", "ensemble", EnsembleForecastStrategy),
    ("validation", "timeseries_cv", TimeSeriesValidationStrategy),
    ("validation", "walk_forward", WalkForwardValidationStrategy),
    ("optimization", "optuna", OptunaOptimizationStrategy),
])
class TestStrategyParametrized:
    """Parametrized tests for all strategies"""

    def test_create_strategy_returns_correct_type(self, strategy_type, strategy_name, expected_class):
        """Test creating strategy returns correct type"""
        strategy = create_strategy(strategy_type, strategy_name)
        assert isinstance(strategy, expected_class)

    def test_strategy_has_required_methods(self, strategy_type, strategy_name, expected_class):
        """Test strategy has required methods"""
        strategy = create_strategy(strategy_type, strategy_name)

        assert hasattr(strategy, 'execute')
        assert callable(strategy.execute)

        if hasattr(strategy, '_setup'):
            assert callable(strategy._setup)

    def test_strategy_config_handling(self, strategy_type, strategy_name, expected_class):
        """Test strategy handles config correctly"""
        config = {'test_param': 'test_value'}
        strategy = create_strategy(strategy_type, strategy_name, config)

        assert strategy.config is not None
        assert isinstance(strategy.config, dict)