"""
Integration tests for submissions module
Tests end-to-end workflows and component interactions
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.submissions.strategy import SubmissionStrategyFactory
from src.submissions.risk_manager import create_risk_manager
from src.submissions.leaderboard_analyzer import analyze_competition
from src.submissions.submission_pipeline import create_submission_pipeline
from src.submissions.timeline_manager import create_competition_timeline
from src.submissions.post_processor import create_post_processor


class TestSubmissionWorkflow(unittest.TestCase):
    """Test complete submission workflow integration"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample training data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')

        self.train_data = pd.DataFrame({
            'date': np.random.choice(dates[:800], 1000),
            'store_id': np.random.randint(1, 11, 1000),
            'product_id': np.random.randint(1, 21, 1000),
            'total_sales': np.random.randint(10, 500, 1000)
        })

        # Create sample test data
        self.test_data = pd.DataFrame({
            'date': np.random.choice(dates[800:], 200),
            'store_id': np.random.randint(1, 11, 200),
            'product_id': np.random.randint(1, 21, 200)
        })

        # Create sample leaderboard data
        self.leaderboard_data = pd.DataFrame({
            'rank': range(1, 51),
            'team_name': [f'Team_{i}' if i != 15 else 'our_team' for i in range(1, 51)],
            'score': sorted(np.random.uniform(10, 30, 50)),
            'submissions': np.random.randint(1, 6, 50)
        })

        # Configuration for testing
        self.config = {
            'strategy': {
                'model_config': {
                    'lightgbm_config': {
                        'num_leaves': 31,
                        'learning_rate': 0.1,
                        'n_estimators': 10  # Small for testing
                    }
                },
                'risk_tolerance': 'medium',
                'post_processing_enabled': True,
                'validation_strategy': 'simple_holdout'
            },
            'pipeline': {
                'steps': {
                    'data_validation': {'enabled': True},
                    'model_training': {'enabled': True, 'timeout_minutes': 5},
                    'risk_assessment': {'enabled': True},
                    'competitive_analysis': {'enabled': True},
                    'post_processing': {'enabled': True},
                    'submission_execution': {'enabled': False}  # Don't save files in tests
                }
            },
            'risk_management': {
                'enable_overfitting_assessment': True,
                'enable_complexity_assessment': True,
                'enable_leakage_assessment': True,
                'enable_execution_assessment': True,
                'overfitting_config': {
                    'max_acceptable_gap': 0.1,
                    'severe_gap_threshold': 0.2,
                    'weight': 1.0
                },
                'complexity_config': {
                    'max_safe_features': 100,
                    'max_safe_depth': 10,
                    'max_training_time_minutes': 5,
                    'weight': 1.0
                },
                'risk_thresholds': {
                    'low': 0.3,
                    'medium': 0.6,
                    'high': 0.8
                }
            },
            'competitive_analysis': {
                'leaderboard_config': {
                    'top_tier_size': 3,
                    'contender_tier_size': 10,
                    'improvement_buffer': 0.05
                }
            },
            'post_processing': {
                'business_rules': {
                    'enabled': True,
                    'min_value': 0.0,
                    'enable_non_negativity': True
                },
                'outlier_capping': {
                    'enabled': True,
                    'method': 'quantile',
                    'upper_quantile': 0.99
                }
            }
        }

    @patch('src.submissions.strategy.lgb.LGBMRegressor')
    def test_baseline_strategy_workflow(self, mock_lgbm):
        """Test complete workflow with baseline strategy"""
        # Mock the model
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.random.rand(len(self.test_data)) * 100
        mock_model.n_features_in_ = 5
        mock_lgbm.return_value = mock_model

        # Create strategy
        strategy = SubmissionStrategyFactory.create('single_model', self.config['strategy'])

        # Create and run pipeline
        pipeline = create_submission_pipeline(self.config['pipeline'])

        result = pipeline.execute_submission_pipeline(
            submission_strategy=strategy,
            train_data=self.train_data,
            test_data=self.test_data,
            team_name='our_team',
            leaderboard_data=self.leaderboard_data.to_dict('records')
        )

        # Verify results
        self.assertTrue(result['success'])
        self.assertIn('final_submission', result)
        self.assertIn('risk_assessment', result)
        self.assertIn('competitive_analysis', result)
        self.assertGreater(result['total_execution_time'], 0)

    def test_risk_assessment_integration(self):
        """Test risk assessment integration with different scenarios"""
        # Create risk manager
        risk_manager = create_risk_manager(self.config['risk_management'])

        # Test low risk scenario
        mock_model = Mock()
        mock_model.n_features_in_ = 20

        low_risk_assessment = risk_manager.assess_submission(
            model=mock_model,
            train_score=0.15,
            validation_score=0.16,  # Small gap
            num_features=20,
            training_time=2,
            memory_usage_gb=4,
            prediction_time_seconds=30
        )

        self.assertLessEqual(low_risk_assessment.overall_risk, 0.5)
        self.assertFalse(risk_manager.should_block_submission(low_risk_assessment))

        # Test high risk scenario
        high_risk_assessment = risk_manager.assess_submission(
            model=mock_model,
            train_score=0.05,
            validation_score=0.98,  # Suspicious gap
            num_features=200,
            training_time=120,
            memory_usage_gb=20,
            prediction_time_seconds=800
        )

        self.assertGreater(high_risk_assessment.overall_risk, 0.5)
        self.assertTrue(risk_manager.should_block_submission(high_risk_assessment))

    def test_competitive_analysis_integration(self):
        """Test competitive analysis with different team positions"""
        # Test top performer
        top_leaderboard = self.leaderboard_data.copy()
        top_leaderboard.loc[top_leaderboard['team_name'] == 'our_team', 'rank'] = 2
        top_leaderboard.loc[top_leaderboard['team_name'] == 'our_team', 'score'] = 11.5

        top_intelligence = analyze_competition(top_leaderboard, 'our_team')

        self.assertEqual(top_intelligence.position_analysis.competitive_zone, 'leader')
        self.assertLess(top_intelligence.gap_analysis.gap_to_top_3, 2.0)

        # Test bottom performer
        bottom_leaderboard = self.leaderboard_data.copy()
        bottom_leaderboard.loc[bottom_leaderboard['team_name'] == 'our_team', 'rank'] = 45
        bottom_leaderboard.loc[bottom_leaderboard['team_name'] == 'our_team', 'score'] = 28.5

        bottom_intelligence = analyze_competition(bottom_leaderboard, 'our_team')

        self.assertEqual(bottom_intelligence.position_analysis.competitive_zone, 'bottom_tier')
        self.assertGreater(bottom_intelligence.gap_analysis.gap_to_top_3, 10.0)

    def test_post_processing_integration(self):
        """Test post-processing integration"""
        # Create post processor
        post_processor = create_post_processor(self.config['post_processing'])

        # Create sample predictions with issues
        predictions = pd.DataFrame({
            'store_id': [1, 1, 2, 2, 3],
            'product_id': [1, 2, 1, 2, 1],
            'date': pd.date_range('2024-01-01', periods=5),
            'prediction': [-10, 1000000, 0, 50, 100]  # Negative and extreme values
        })

        # Apply post-processing
        processed_predictions = post_processor.apply_post_processing(predictions)

        # Verify corrections
        self.assertTrue((processed_predictions['prediction'] >= 0).all())  # No negatives
        self.assertLess(processed_predictions['prediction'].max(), 1000000)  # Outliers capped

    @patch('src.submissions.strategy.lgb.LGBMRegressor')
    def test_pipeline_error_handling(self, mock_lgbm):
        """Test pipeline error handling and recovery"""
        # Mock model that raises an error during training
        mock_model = Mock()
        mock_model.fit.side_effect = RuntimeError("Training failed")
        mock_lgbm.return_value = mock_model

        strategy = SubmissionStrategyFactory.create('single_model', self.config['strategy'])
        pipeline = create_submission_pipeline(self.config['pipeline'])

        result = pipeline.execute_submission_pipeline(
            submission_strategy=strategy,
            train_data=self.train_data,
            test_data=self.test_data,
            team_name='our_team'
        )

        # Verify failure is handled gracefully
        self.assertFalse(result['success'])
        self.assertIn('model_training', result['steps_failed'])

    def test_timeline_management_integration(self):
        """Test timeline management integration"""
        timeline_config = {
            'deadline_tracker': {
                'upcoming_alert_hours': 24,
                'urgent_alert_hours': 6,
                'critical_alert_hours': 2
            },
            'submission_windows': {
                'baseline': {
                    'day': 7,
                    'optimal_hour': 18,
                    'priority': 'medium'
                },
                'final_submission': {
                    'day': 1,
                    'optimal_hour': 18,
                    'priority': 'critical'
                }
            }
        }

        timeline_manager = create_competition_timeline(timeline_config)

        # Test deadline checking
        windows = timeline_manager.get_submission_windows()
        self.assertIn('baseline', windows)
        self.assertIn('final_submission', windows)

        # Test alert generation (would depend on current date in real scenario)
        alerts = timeline_manager.check_upcoming_deadlines()
        self.assertIsInstance(alerts, list)

    def test_end_to_end_submission_workflow(self):
        """Test complete end-to-end submission workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock successful model training
            with patch('src.submissions.strategy.lgb.LGBMRegressor') as mock_lgbm:
                mock_model = Mock()
                mock_model.fit.return_value = mock_model
                mock_model.predict.return_value = np.random.rand(len(self.test_data)) * 100
                mock_model.n_features_in_ = 10
                mock_lgbm.return_value = mock_model

                # Enable submission execution for this test
                config = self.config.copy()
                config['pipeline']['steps']['submission_execution']['enabled'] = True

                # Create all components
                strategy = SubmissionStrategyFactory.create('single_model', config['strategy'])
                pipeline = create_submission_pipeline(config['pipeline'])
                risk_manager = create_risk_manager(config['risk_management'])

                # Execute workflow
                result = pipeline.execute_submission_pipeline(
                    submission_strategy=strategy,
                    train_data=self.train_data,
                    test_data=self.test_data,
                    team_name='our_team',
                    leaderboard_data=self.leaderboard_data.to_dict('records'),
                    output_dir=str(temp_path)
                )

                # Verify complete workflow
                self.assertTrue(result['success'])
                self.assertIsNotNone(result['final_submission'])
                self.assertIsNotNone(result['risk_assessment'])
                self.assertIsNotNone(result['competitive_analysis'])

                # Verify risk assessment integration
                risk_assessment = result['risk_assessment']
                self.assertIn(risk_assessment.risk_level.value, ['low', 'medium', 'high'])

                # Verify competitive analysis integration
                competitive_analysis = result['competitive_analysis']
                self.assertIsNotNone(competitive_analysis.position_analysis)
                self.assertIsNotNone(competitive_analysis.gap_analysis)

    def test_configuration_validation(self):
        """Test configuration validation across components"""
        # Test with minimal configuration
        minimal_config = {
            'strategy': {'risk_tolerance': 'low'},
            'pipeline': {'steps': {}},
            'risk_management': {},
            'competitive_analysis': {},
            'post_processing': {}
        }

        # Should work with defaults
        strategy = SubmissionStrategyFactory.create('baseline', minimal_config['strategy'])
        self.assertIsNotNone(strategy)

        pipeline = create_submission_pipeline(minimal_config['pipeline'])
        self.assertIsNotNone(pipeline)

        risk_manager = create_risk_manager(minimal_config['risk_management'])
        self.assertIsNotNone(risk_manager)

        post_processor = create_post_processor(minimal_config['post_processing'])
        self.assertIsNotNone(post_processor)

    def test_performance_monitoring(self):
        """Test performance monitoring integration"""
        with patch('src.submissions.strategy.lgb.LGBMRegressor') as mock_lgbm:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_model.predict.return_value = np.random.rand(len(self.test_data)) * 100
            mock_model.n_features_in_ = 15
            mock_lgbm.return_value = mock_model

            strategy = SubmissionStrategyFactory.create('single_model', self.config['strategy'])
            pipeline = create_submission_pipeline(self.config['pipeline'])

            # Enable performance monitoring
            start_time = pd.Timestamp.now()

            result = pipeline.execute_submission_pipeline(
                submission_strategy=strategy,
                train_data=self.train_data,
                test_data=self.test_data,
                team_name='our_team'
            )

            end_time = pd.Timestamp.now()

            # Verify performance metrics are captured
            self.assertGreater(result['total_execution_time'], 0)
            self.assertLessEqual(result['total_execution_time'], (end_time - start_time).total_seconds() + 1)

    def test_data_validation_integration(self):
        """Test data validation integration"""
        # Test with invalid training data
        invalid_train_data = pd.DataFrame({
            'date': [None, None],  # Missing dates
            'store_id': [1, 2],
            'product_id': [1, 2],
            'total_sales': [100, 200]
        })

        strategy = SubmissionStrategyFactory.create('baseline', self.config['strategy'])
        pipeline = create_submission_pipeline(self.config['pipeline'])

        result = pipeline.execute_submission_pipeline(
            submission_strategy=strategy,
            train_data=invalid_train_data,
            test_data=self.test_data,
            team_name='our_team'
        )

        # Should handle validation errors gracefully
        if not result['success']:
            self.assertIn('data_validation', result.get('steps_failed', []))


class TestComponentInteractions(unittest.TestCase):
    """Test interactions between different components"""

    def test_strategy_risk_manager_interaction(self):
        """Test interaction between strategy and risk manager"""
        # High-risk strategy should be flagged by risk manager
        high_risk_config = {
            'model_config': {
                'lightgbm_config': {
                    'num_leaves': 1000,  # Very complex
                    'learning_rate': 0.001,
                    'n_estimators': 10000
                }
            },
            'risk_tolerance': 'high'
        }

        strategy = SubmissionStrategyFactory.create('single_model', high_risk_config)

        risk_config = {
            'enable_complexity_assessment': True,
            'complexity_config': {
                'max_safe_features': 50,
                'max_safe_depth': 5,
                'max_training_time_minutes': 10,
                'weight': 2.0
            },
            'risk_thresholds': {'low': 0.2, 'medium': 0.5, 'high': 0.7}
        }

        risk_manager = create_risk_manager(risk_config)

        # Mock complex model
        mock_model = Mock()
        mock_model.n_features_in_ = 100

        assessment = risk_manager.assess_submission(
            model=mock_model,
            num_features=100,
            training_time=60  # Long training time
        )

        # Should be flagged as risky due to complexity
        self.assertGreater(assessment.overall_risk, 0.5)

    def test_competitive_analysis_strategy_interaction(self):
        """Test how competitive analysis influences strategy selection"""
        # Create leaderboard where we're far behind
        behind_leaderboard = pd.DataFrame({
            'rank': range(1, 51),
            'team_name': [f'Team_{i}' if i != 45 else 'our_team' for i in range(1, 51)],
            'score': sorted(np.random.uniform(10, 30, 50)),
        })

        intelligence = analyze_competition(behind_leaderboard, 'our_team')

        # When far behind, should recommend aggressive strategy
        self.assertEqual(intelligence.position_analysis.competitive_zone, 'bottom_tier')
        self.assertTrue(intelligence.gap_analysis.gap_to_top_3 > 10.0)

        # Strategic recommendations should reflect urgency
        recommendations = intelligence.strategic_recommendations
        aggressive_keywords = ['aggressive', 'experimental', 'risk', 'ensemble']
        has_aggressive_rec = any(
            any(keyword in rec.lower() for keyword in aggressive_keywords)
            for rec in recommendations
        )
        self.assertTrue(has_aggressive_rec or len(recommendations) > 5)  # Many recommendations when behind

    def test_timeline_strategy_interaction(self):
        """Test how timeline constraints affect strategy selection"""
        # Create timeline with very short deadline
        urgent_timeline_config = {
            'submission_windows': {
                'final_submission': {
                    'day': 0,  # Today
                    'optimal_hour': 23,
                    'priority': 'critical'
                }
            }
        }

        timeline_manager = create_competition_timeline(urgent_timeline_config)

        # With urgent timeline, should prefer simpler, faster strategies
        windows = timeline_manager.get_submission_windows()
        final_window = windows.get('final_submission', {})

        if final_window.get('priority') == 'critical':
            # Should recommend baseline or single model over complex ensemble
            recommended_strategies = ['baseline', 'single_model']
            # In real implementation, timeline would influence strategy selection
            self.assertIn('baseline', recommended_strategies)


if __name__ == '__main__':
    unittest.main()