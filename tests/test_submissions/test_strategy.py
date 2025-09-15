"""
Tests for submission strategy module
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.submissions.strategy import (
    BaseSubmissionStrategy, BaselineSubmissionStrategy, SingleModelSubmissionStrategy,
    EnsembleSubmissionStrategy, OptimizedEnsembleSubmissionStrategy, FinalSubmissionStrategy,
    SubmissionStrategyFactory, SubmissionPhase
)


class TestSubmissionStrategy(unittest.TestCase):
    """Test cases for base SubmissionStrategy class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'model_config': {'test_param': 'test_value'},
            'risk_tolerance': 'medium',
            'post_processing_enabled': True,
            'validation_strategy': 'simple_holdout'
        }
        self.strategy = BaseSubmissionStrategy(self.config)

    def test_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.config, self.config)
        self.assertEqual(self.strategy.risk_tolerance, 'medium')
        self.assertTrue(self.strategy.post_processing_enabled)
        self.assertEqual(self.strategy.validation_strategy, 'simple_holdout')
        self.assertIsNotNone(self.strategy.submission_id)

    def test_get_model_config(self):
        """Test model configuration retrieval"""
        model_config = self.strategy.get_model_config()
        self.assertEqual(model_config, {'test_param': 'test_value'})

    def test_should_enable_post_processing(self):
        """Test post-processing enablement check"""
        self.assertTrue(self.strategy.should_enable_post_processing())

        # Test with disabled post-processing
        config_disabled = self.config.copy()
        config_disabled['post_processing_enabled'] = False
        strategy_disabled = BaseSubmissionStrategy(config_disabled)
        self.assertFalse(strategy_disabled.should_enable_post_processing())

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError"""
        with self.assertRaises(NotImplementedError):
            self.strategy.create_model()

        with self.assertRaises(NotImplementedError):
            self.strategy.train_model(None, None)

        with self.assertRaises(NotImplementedError):
            self.strategy.predict(None, None)


class TestBaselineStrategy(unittest.TestCase):
    """Test cases for BaselineStrategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'model_config': {
                'prophet_config': {
                    'daily_seasonality': True,
                    'weekly_seasonality': True,
                    'yearly_seasonality': True
                }
            },
            'risk_tolerance': 'low'
        }
        self.strategy = BaselineSubmissionStrategy(self.config)

        # Create sample data
        self.train_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'total_sales': np.random.randint(100, 1000, 100),
            'store_id': [1] * 100,
            'product_id': [1] * 100
        })

        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-04-11', periods=30),
            'store_id': [1] * 30,
            'product_id': [1] * 30
        })

    def test_initialization(self):
        """Test baseline strategy initialization"""
        self.assertEqual(self.strategy.risk_tolerance, 'low')
        self.assertEqual(self.strategy.model_type, 'prophet')

    @patch('src.submissions.strategy.Prophet')
    def test_create_model(self, mock_prophet):
        """Test model creation"""
        mock_model = Mock()
        mock_prophet.return_value = mock_model

        model = self.strategy.create_model()

        self.assertEqual(model, mock_model)
        mock_prophet.assert_called_once()

    @patch('src.submissions.strategy.Prophet')
    def test_train_model(self, mock_prophet):
        """Test model training"""
        mock_model = Mock()
        mock_prophet.return_value = mock_model

        result = self.strategy.train_model(self.train_data, self.test_data)

        self.assertIsNotNone(result)
        mock_model.fit.assert_called_once()

    @patch('src.submissions.strategy.Prophet')
    def test_predict(self, mock_prophet):
        """Test prediction generation"""
        mock_model = Mock()
        mock_model.predict.return_value = pd.DataFrame({
            'yhat': np.random.rand(30) * 100
        })
        mock_prophet.return_value = mock_model

        predictions = self.strategy.predict(mock_model, self.test_data)

        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), len(self.test_data))
        self.assertIn('prediction', predictions.columns)


class TestSingleModelStrategy(unittest.TestCase):
    """Test cases for SingleModelStrategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'model_config': {
                'lightgbm_config': {
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8
                }
            },
            'risk_tolerance': 'medium'
        }
        self.strategy = SingleModelSubmissionStrategy(self.config)

    def test_initialization(self):
        """Test single model strategy initialization"""
        self.assertEqual(self.strategy.risk_tolerance, 'medium')
        self.assertEqual(self.strategy.model_type, 'lightgbm')

    @patch('src.submissions.strategy.lgb.LGBMRegressor')
    def test_create_model(self, mock_lgbm):
        """Test model creation"""
        mock_model = Mock()
        mock_lgbm.return_value = mock_model

        model = self.strategy.create_model()

        self.assertEqual(model, mock_model)
        mock_lgbm.assert_called_once()


class TestEnsembleStrategy(unittest.TestCase):
    """Test cases for EnsembleStrategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'model_config': {
                'ensemble_config': {
                    'models': ['lightgbm', 'prophet'],
                    'ensemble_method': 'stacking',
                    'meta_learner': 'linear_regression'
                }
            },
            'risk_tolerance': 'medium'
        }
        self.strategy = EnsembleSubmissionStrategy(self.config)

    def test_initialization(self):
        """Test ensemble strategy initialization"""
        self.assertEqual(self.strategy.risk_tolerance, 'medium')
        self.assertEqual(self.strategy.model_type, 'ensemble')
        self.assertEqual(len(self.strategy.models), 2)

    def test_create_model(self):
        """Test ensemble model creation"""
        ensemble = self.strategy.create_model()

        self.assertIsNotNone(ensemble)
        self.assertEqual(len(ensemble['models']), 2)
        self.assertIn('meta_learner', ensemble)


class TestOptimizedEnsembleStrategy(unittest.TestCase):
    """Test cases for OptimizedEnsembleStrategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'model_config': {
                'optimized_ensemble_config': {
                    'models': ['lightgbm', 'prophet', 'tree_ensemble'],
                    'optimization_method': 'optuna',
                    'n_trials': 10,
                    'cv_folds': 3
                }
            },
            'risk_tolerance': 'high'
        }
        self.strategy = OptimizedEnsembleSubmissionStrategy(self.config)

    def test_initialization(self):
        """Test optimized ensemble strategy initialization"""
        self.assertEqual(self.strategy.risk_tolerance, 'high')
        self.assertEqual(self.strategy.model_type, 'optimized_ensemble')


class TestFinalStrategy(unittest.TestCase):
    """Test cases for FinalStrategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'model_config': {
                'final_config': {
                    'use_best_model': True,
                    'ensemble_all_previous': True,
                    'final_optimization': True,
                    'max_optimization_time_minutes': 30
                }
            },
            'risk_tolerance': 'high'
        }
        self.strategy = FinalSubmissionStrategy(self.config)

    def test_initialization(self):
        """Test final strategy initialization"""
        self.assertEqual(self.strategy.risk_tolerance, 'high')
        self.assertEqual(self.strategy.model_type, 'final')


class TestSubmissionStrategyFactory(unittest.TestCase):
    """Test cases for SubmissionStrategyFactory"""

    def test_create_baseline_strategy(self):
        """Test creation of baseline strategy"""
        config = {'risk_tolerance': 'low'}
        strategy = SubmissionStrategyFactory.create('baseline', config)

        self.assertIsInstance(strategy, BaselineSubmissionStrategy)
        self.assertEqual(strategy.risk_tolerance, 'low')

    def test_create_single_model_strategy(self):
        """Test creation of single model strategy"""
        config = {'risk_tolerance': 'medium'}
        strategy = SubmissionStrategyFactory.create('single_model', config)

        self.assertIsInstance(strategy, SingleModelSubmissionStrategy)
        self.assertEqual(strategy.risk_tolerance, 'medium')

    def test_create_ensemble_strategy(self):
        """Test creation of ensemble strategy"""
        config = {'risk_tolerance': 'medium'}
        strategy = SubmissionStrategyFactory.create('ensemble', config)

        self.assertIsInstance(strategy, EnsembleSubmissionStrategy)
        self.assertEqual(strategy.risk_tolerance, 'medium')

    def test_create_optimized_ensemble_strategy(self):
        """Test creation of optimized ensemble strategy"""
        config = {'risk_tolerance': 'high'}
        strategy = SubmissionStrategyFactory.create('optimized_ensemble', config)

        self.assertIsInstance(strategy, OptimizedEnsembleSubmissionStrategy)
        self.assertEqual(strategy.risk_tolerance, 'high')

    def test_create_final_strategy(self):
        """Test creation of final strategy"""
        config = {'risk_tolerance': 'high'}
        strategy = SubmissionStrategyFactory.create('final', config)

        self.assertIsInstance(strategy, FinalSubmissionStrategy)
        self.assertEqual(strategy.risk_tolerance, 'high')

    def test_create_invalid_strategy(self):
        """Test creation of invalid strategy raises error"""
        with self.assertRaises(ValueError):
            SubmissionStrategyFactory.create('invalid_strategy', {})

    def test_get_available_strategies(self):
        """Test getting list of available strategies"""
        strategies = SubmissionStrategyFactory.get_available_strategies()

        expected_strategies = ['baseline', 'single_model', 'ensemble', 'optimized_ensemble', 'final']
        self.assertEqual(strategies, expected_strategies)


class TestSubmissionPhase(unittest.TestCase):
    """Test cases for SubmissionPhase enum"""

    def test_submission_phase_values(self):
        """Test submission phase enum values"""
        self.assertEqual(SubmissionPhase.BASELINE.value, 'baseline')
        self.assertEqual(SubmissionPhase.SINGLE_MODEL.value, 'single_model')
        self.assertEqual(SubmissionPhase.ENSEMBLE.value, 'ensemble')
        self.assertEqual(SubmissionPhase.OPTIMIZED_ENSEMBLE.value, 'optimized_ensemble')
        self.assertEqual(SubmissionPhase.FINAL.value, 'final')

    def test_phase_progression(self):
        """Test phase progression logic"""
        phases = [phase.value for phase in SubmissionPhase]
        expected_order = ['baseline', 'single_model', 'ensemble', 'optimized_ensemble', 'final']

        self.assertEqual(phases, expected_order)


if __name__ == '__main__':
    unittest.main()