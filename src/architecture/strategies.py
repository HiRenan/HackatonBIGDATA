#!/usr/bin/env python3
"""
Phase 6: Strategy Pattern Implementation
Interchangeable algorithms for different forecasting strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Abstract base strategy"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Setup strategy-specific configuration"""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the strategy"""
        pass

class ForecastStrategy(BaseStrategy):
    """Strategy for different forecasting approaches"""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ForecastStrategy':
        """Fit the forecasting model"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions with uncertainty quantification"""
        pass

class ProphetForecastStrategy(ForecastStrategy):
    """Prophet-based forecasting strategy"""

    def _setup(self) -> None:
        self.seasonality_mode = self.config.get('seasonality_mode', 'multiplicative')
        self.yearly_seasonality = self.config.get('yearly_seasonality', True)
        self.weekly_seasonality = self.config.get('weekly_seasonality', True)
        self.interval_width = self.config.get('interval_width', 0.95)
        self.model = None

    def execute(self, data: pd.DataFrame, forecast_periods: int = 30) -> Dict[str, Any]:
        """Execute Prophet forecasting"""
        try:
            from prophet import Prophet

            # Prepare data for Prophet
            prophet_data = data[['date', 'target']].copy()
            prophet_data.columns = ['ds', 'y']

            # Initialize and fit Prophet
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                interval_width=self.interval_width
            )

            self.model.fit(prophet_data)

            # Make future predictions
            future = self.model.make_future_dataframe(periods=forecast_periods)
            forecast = self.model.predict(future)

            return {
                'predictions': forecast['yhat'].values,
                'lower_bounds': forecast['yhat_lower'].values,
                'upper_bounds': forecast['yhat_upper'].values,
                'components': {
                    'trend': forecast['trend'].values,
                    'seasonal': forecast.get('yearly', np.zeros(len(forecast))),
                    'weekly': forecast.get('weekly', np.zeros(len(forecast)))
                }
            }

        except ImportError:
            logger.error("Prophet not available")
            raise RuntimeError("Prophet not installed")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ProphetForecastStrategy':
        """Fit Prophet model"""
        data = pd.DataFrame({'date': X.index, 'target': y})
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Prophet"""
        if self.model is None:
            raise RuntimeError("Model not fitted")

        future = pd.DataFrame({'ds': X.index})
        forecast = self.model.predict(future)
        return forecast['yhat'].values

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions with uncertainty"""
        if self.model is None:
            raise RuntimeError("Model not fitted")

        future = pd.DataFrame({'ds': X.index})
        forecast = self.model.predict(future)

        return {
            'predictions': forecast['yhat'].values,
            'lower_bounds': forecast['yhat_lower'].values,
            'upper_bounds': forecast['yhat_upper'].values
        }

class EnsembleForecastStrategy(ForecastStrategy):
    """Ensemble-based forecasting strategy"""

    def _setup(self) -> None:
        self.base_strategies = self.config.get('base_strategies', ['prophet', 'lightgbm'])
        self.meta_learner = self.config.get('meta_learner', 'ridge')
        self.cv_folds = self.config.get('cv_folds', 5)
        self.models = {}
        self.meta_model = None

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute ensemble forecasting"""
        results = {}

        # Get predictions from each base strategy
        for strategy_name in self.base_strategies:
            try:
                strategy = self._create_strategy(strategy_name)
                strategy_result = strategy.execute(data, **kwargs)
                results[strategy_name] = strategy_result
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue

        # Combine predictions using meta-learner
        if results:
            combined_predictions = self._combine_predictions(results)
            return combined_predictions
        else:
            raise RuntimeError("All base strategies failed")

    def _create_strategy(self, strategy_name: str) -> ForecastStrategy:
        """Create individual strategy"""
        if strategy_name == 'prophet':
            return ProphetForecastStrategy(self.config.get('prophet_config', {}))
        else:
            # Placeholder for other strategies
            raise NotImplementedError(f"Strategy {strategy_name} not implemented")

    def _combine_predictions(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine predictions from multiple strategies"""
        predictions_list = []
        weights = []

        for strategy_name, result in results.items():
            predictions_list.append(result['predictions'])
            # Simple equal weighting for now
            weights.append(1.0)

        # Weighted average
        weights = np.array(weights) / sum(weights)
        combined_predictions = np.average(predictions_list, axis=0, weights=weights)

        # Combine uncertainty bounds
        lower_bounds = []
        upper_bounds = []

        for result in results.values():
            if 'lower_bounds' in result:
                lower_bounds.append(result['lower_bounds'])
            if 'upper_bounds' in result:
                upper_bounds.append(result['upper_bounds'])

        combined_lower = np.min(lower_bounds, axis=0) if lower_bounds else None
        combined_upper = np.max(upper_bounds, axis=0) if upper_bounds else None

        return {
            'predictions': combined_predictions,
            'lower_bounds': combined_lower,
            'upper_bounds': combined_upper,
            'individual_results': results
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleForecastStrategy':
        """Fit ensemble model"""
        # Implementation for fitting ensemble
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        # Implementation for ensemble prediction
        return np.zeros(len(X))

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make ensemble predictions with uncertainty"""
        return {
            'predictions': self.predict(X),
            'lower_bounds': np.zeros(len(X)),
            'upper_bounds': np.zeros(len(X))
        }

class ValidationStrategy(BaseStrategy):
    """Strategy for different validation approaches"""

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation splits"""
        pass

    @abstractmethod
    def validate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform validation"""
        pass

class TimeSeriesValidationStrategy(ValidationStrategy):
    """Time series cross-validation strategy"""

    def _setup(self) -> None:
        self.n_splits = self.config.get('n_splits', 5)
        self.test_size = self.config.get('test_size', None)
        self.gap = self.config.get('gap', 0)

    def execute(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Execute time series validation"""
        return self.validate(model, X, y)

    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series splits"""
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap=self.gap
        )

        return list(tscv.split(X))

    def validate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform time series cross-validation"""
        from ..evaluation.metrics import wmape, mae, rmse

        splits = self.split(X, y)
        scores = {'wmape': [], 'mae': [], 'rmse': []}

        for train_idx, val_idx in splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            try:
                # Fit and predict
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)

                if hasattr(model, 'predict'):
                    predictions = model.predict(X_val)
                else:
                    raise AttributeError("Model has no predict method")

                # Calculate metrics
                scores['wmape'].append(wmape(y_val, predictions))
                scores['mae'].append(mae(y_val, predictions))
                scores['rmse'].append(rmse(y_val, predictions))

            except Exception as e:
                logger.warning(f"Validation fold failed: {e}")
                continue

        # Average scores
        final_scores = {}
        for metric, values in scores.items():
            if values:
                final_scores[f'{metric}_mean'] = np.mean(values)
                final_scores[f'{metric}_std'] = np.std(values)
            else:
                final_scores[f'{metric}_mean'] = np.nan
                final_scores[f'{metric}_std'] = np.nan

        return final_scores

class WalkForwardValidationStrategy(ValidationStrategy):
    """Walk-forward validation strategy"""

    def _setup(self) -> None:
        self.window_size = self.config.get('window_size', 252)  # 1 year of daily data
        self.step_size = self.config.get('step_size', 30)  # 1 month step
        self.forecast_horizon = self.config.get('forecast_horizon', 30)

    def execute(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Execute walk-forward validation"""
        return self.validate(model, X, y)

    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits"""
        splits = []
        n_samples = len(X)

        for i in range(self.window_size, n_samples - self.forecast_horizon, self.step_size):
            train_start = max(0, i - self.window_size)
            train_end = i
            val_start = i
            val_end = min(n_samples, i + self.forecast_horizon)

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)

            splits.append((train_idx, val_idx))

        return splits

    def validate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform walk-forward validation"""
        from ..evaluation.metrics import wmape, mae, rmse

        splits = self.split(X, y)
        all_predictions = []
        all_actuals = []

        for train_idx, val_idx in splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            try:
                # Fit and predict
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)

                if hasattr(model, 'predict'):
                    predictions = model.predict(X_val)
                else:
                    raise AttributeError("Model has no predict method")

                all_predictions.extend(predictions)
                all_actuals.extend(y_val.values)

            except Exception as e:
                logger.warning(f"Walk-forward step failed: {e}")
                continue

        # Calculate overall metrics
        if all_predictions and all_actuals:
            all_predictions = np.array(all_predictions)
            all_actuals = np.array(all_actuals)

            return {
                'wmape': wmape(all_actuals, all_predictions),
                'mae': mae(all_actuals, all_predictions),
                'rmse': rmse(all_actuals, all_predictions),
                'n_predictions': len(all_predictions)
            }
        else:
            return {'error': 'No successful predictions made'}

class OptimizationStrategy(BaseStrategy):
    """Strategy for hyperparameter optimization"""

    @abstractmethod
    def optimize(self, objective_func, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters"""
        pass

class OptunaOptimizationStrategy(OptimizationStrategy):
    """Optuna-based hyperparameter optimization"""

    def _setup(self) -> None:
        self.n_trials = self.config.get('n_trials', 100)
        self.timeout = self.config.get('timeout', 3600)  # 1 hour
        self.direction = self.config.get('direction', 'minimize')

    def execute(self, objective_func, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Optuna optimization"""
        return self.optimize(objective_func, search_space)

    def optimize(self, objective_func, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize using Optuna"""
        try:
            import optuna

            study = optuna.create_study(direction=self.direction)

            def objective(trial):
                # Suggest parameters based on search space
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )

                return objective_func(**params)

            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

            return {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'study': study
            }

        except ImportError:
            logger.error("Optuna not available")
            raise RuntimeError("Optuna not installed")

# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'forecast': {
        'prophet': ProphetForecastStrategy,
        'ensemble': EnsembleForecastStrategy
    },
    'validation': {
        'timeseries_cv': TimeSeriesValidationStrategy,
        'walk_forward': WalkForwardValidationStrategy
    },
    'optimization': {
        'optuna': OptunaOptimizationStrategy
    }
}

def create_strategy(strategy_type: str, strategy_name: str, config: Optional[Dict[str, Any]] = None) -> BaseStrategy:
    """
    Create strategy instance

    Args:
        strategy_type: Type of strategy ('forecast', 'validation', 'optimization')
        strategy_name: Specific strategy name
        config: Configuration dictionary

    Returns:
        Strategy instance
    """
    if strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    if strategy_name not in STRATEGY_REGISTRY[strategy_type]:
        available = ', '.join(STRATEGY_REGISTRY[strategy_type].keys())
        raise ValueError(f"Unknown {strategy_type} strategy: {strategy_name}. Available: {available}")

    strategy_class = STRATEGY_REGISTRY[strategy_type][strategy_name]
    return strategy_class(config)

if __name__ == "__main__":
    # Demo usage
    print("🎯 Phase 6 Strategy Pattern Demo")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'target': np.random.randn(100).cumsum() + 100
    })

    # Test forecast strategies
    print("\n📈 Forecast Strategies:")
    try:
        prophet_strategy = create_strategy('forecast', 'prophet')
        print("✅ Prophet strategy created")

        ensemble_strategy = create_strategy('forecast', 'ensemble')
        print("✅ Ensemble strategy created")
    except Exception as e:
        print(f"❌ Error creating forecast strategies: {e}")

    # Test validation strategies
    print("\n📊 Validation Strategies:")
    try:
        ts_validation = create_strategy('validation', 'timeseries_cv')
        print("✅ Time series CV strategy created")

        wf_validation = create_strategy('validation', 'walk_forward')
        print("✅ Walk-forward validation strategy created")
    except Exception as e:
        print(f"❌ Error creating validation strategies: {e}")

    # Test optimization strategies
    print("\n🔧 Optimization Strategies:")
    try:
        optuna_optimization = create_strategy('optimization', 'optuna')
        print("✅ Optuna optimization strategy created")
    except Exception as e:
        print(f"❌ Error creating optimization strategies: {e}")