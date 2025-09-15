"""
Tests for risk manager module
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

from src.submissions.risk_manager import (
    RiskAssessment, OverfittingRiskAssessor, ComplexityRiskAssessor,
    LeakageRiskAssessor, ExecutionRiskAssessor, RiskManager, create_risk_manager
)


class TestRiskAssessment(unittest.TestCase):
    """Test cases for RiskAssessment class"""

    def setUp(self):
        """Set up test fixtures"""
        self.risk_factors = {
            'overfitting': 0.3,
            'complexity': 0.2,
            'leakage': 0.1,
            'execution': 0.15
        }
        self.risk_assessment = RiskAssessment(
            overall_risk=0.25,
            risk_level="MEDIUM",
            confidence=0.8,
            risk_factors=self.risk_factors,
            recommendations=['Test recommendation 1', 'Test recommendation 2'],
            metadata={'test_meta': 'value'}
        )

    def test_initialization(self):
        """Test risk assessment initialization"""
        self.assertEqual(self.risk_assessment.overall_risk, 0.25)
        self.assertEqual(self.risk_assessment.risk_level, "MEDIUM")
        self.assertEqual(self.risk_assessment.confidence, 0.8)
        self.assertEqual(self.risk_assessment.risk_factors, self.risk_factors)
        self.assertEqual(len(self.risk_assessment.recommendations), 2)
        self.assertEqual(self.risk_assessment.metadata['test_meta'], 'value')

    def test_get_risk_factor(self):
        """Test retrieving specific risk factors"""
        self.assertEqual(self.risk_assessment.get_risk_factor('overfitting'), 0.3)
        self.assertEqual(self.risk_assessment.get_risk_factor('complexity'), 0.2)
        self.assertIsNone(self.risk_assessment.get_risk_factor('nonexistent'))

    def test_is_high_risk(self):
        """Test high risk detection"""
        self.assertFalse(self.risk_assessment.is_high_risk())

        high_risk = RiskAssessment(
            overall_risk=0.8,
            risk_level="HIGH",
            confidence=0.9,
            risk_factors={},
            recommendations=[],
            metadata={}
        )
        self.assertTrue(high_risk.is_high_risk())

    def test_get_top_risks(self):
        """Test getting top risk factors"""
        top_risks = self.risk_assessment.get_top_risks(2)

        self.assertEqual(len(top_risks), 2)
        self.assertEqual(top_risks[0][0], 'overfitting')  # Highest risk
        self.assertEqual(top_risks[1][0], 'complexity')   # Second highest

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result_dict = self.risk_assessment.to_dict()

        self.assertIn('overall_risk', result_dict)
        self.assertIn('risk_level', result_dict)
        self.assertIn('confidence', result_dict)
        self.assertIn('risk_factors', result_dict)
        self.assertIn('recommendations', result_dict)
        self.assertIn('metadata', result_dict)


class TestRiskLevels(unittest.TestCase):
    """Test cases for risk level strings"""

    def test_risk_level_values(self):
        """Test risk level string values"""
        valid_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        for level in valid_levels:
            self.assertIsInstance(level, str)

    def test_risk_level_ordering(self):
        """Test risk level availability"""
        levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        self.assertIn('LOW', levels)
        self.assertIn('MEDIUM', levels)
        self.assertIn('HIGH', levels)


class TestOverfittingRisk(unittest.TestCase):
    """Test cases for OverfittingRisk"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'max_acceptable_gap': 0.1,
            'severe_gap_threshold': 0.2,
            'weight': 1.5
        }
        self.overfitting_risk = OverfittingRisk(self.config)

    def test_initialization(self):
        """Test overfitting risk initialization"""
        self.assertEqual(self.overfitting_risk.max_acceptable_gap, 0.1)
        self.assertEqual(self.overfitting_risk.severe_gap_threshold, 0.2)
        self.assertEqual(self.overfitting_risk.weight, 1.5)

    def test_assess_low_risk(self):
        """Test assessment with low overfitting risk"""
        model = Mock()
        train_score = 0.15
        val_score = 0.16  # Small gap

        risk_score, recommendations = self.overfitting_risk.assess(model, train_score, val_score)

        self.assertLess(risk_score, 0.3)  # Should be low risk
        self.assertGreater(len(recommendations), 0)

    def test_assess_high_risk(self):
        """Test assessment with high overfitting risk"""
        model = Mock()
        train_score = 0.10
        val_score = 0.35  # Large gap

        risk_score, recommendations = self.overfitting_risk.assess(model, train_score, val_score)

        self.assertGreater(risk_score, 0.7)  # Should be high risk
        self.assertIn('severe overfitting', ' '.join(recommendations).lower())

    def test_assess_missing_scores(self):
        """Test assessment with missing scores"""
        model = Mock()

        risk_score, recommendations = self.overfitting_risk.assess(model, None, None)

        self.assertEqual(risk_score, 0.5)  # Default moderate risk
        self.assertIn('Unable to assess', recommendations[0])


class TestComplexityRisk(unittest.TestCase):
    """Test cases for ComplexityRisk"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'max_safe_features': 100,
            'max_safe_depth': 10,
            'max_training_time_minutes': 30,
            'weight': 1.0
        }
        self.complexity_risk = ComplexityRisk(self.config)

    def test_initialization(self):
        """Test complexity risk initialization"""
        self.assertEqual(self.complexity_risk.max_safe_features, 100)
        self.assertEqual(self.complexity_risk.max_safe_depth, 10)
        self.assertEqual(self.complexity_risk.max_training_time_minutes, 30)

    def test_assess_low_complexity(self):
        """Test assessment with low complexity model"""
        model = Mock()
        model.n_features_in_ = 50  # Below max_safe_features

        risk_score, recommendations = self.complexity_risk.assess(
            model, num_features=50, training_time=10
        )

        self.assertLess(risk_score, 0.5)  # Should be low to medium risk

    def test_assess_high_complexity(self):
        """Test assessment with high complexity model"""
        model = Mock()
        model.n_features_in_ = 200  # Above max_safe_features

        risk_score, recommendations = self.complexity_risk.assess(
            model, num_features=200, training_time=60
        )

        self.assertGreater(risk_score, 0.5)  # Should be medium to high risk


class TestLeakageRisk(unittest.TestCase):
    """Test cases for LeakageRisk"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'suspicious_score_threshold': 0.95,
            'weight': 2.0
        }
        self.leakage_risk = LeakageRisk(self.config)

    def test_initialization(self):
        """Test leakage risk initialization"""
        self.assertEqual(self.leakage_risk.suspicious_score_threshold, 0.95)
        self.assertEqual(self.leakage_risk.weight, 2.0)

    def test_assess_normal_score(self):
        """Test assessment with normal performance score"""
        model = Mock()
        validation_score = 0.85  # Good but not suspicious

        risk_score, recommendations = self.leakage_risk.assess(
            model, validation_score=validation_score
        )

        self.assertLess(risk_score, 0.3)  # Should be low risk

    def test_assess_suspicious_score(self):
        """Test assessment with suspiciously high score"""
        model = Mock()
        validation_score = 0.98  # Too good to be true

        risk_score, recommendations = self.leakage_risk.assess(
            model, validation_score=validation_score
        )

        self.assertGreater(risk_score, 0.7)  # Should be high risk
        self.assertIn('data leakage', ' '.join(recommendations).lower())


class TestExecutionRisk(unittest.TestCase):
    """Test cases for ExecutionRisk"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'max_memory_gb': 16,
            'max_prediction_time_seconds': 300,
            'weight': 1.0
        }
        self.execution_risk = ExecutionRisk(self.config)

    def test_initialization(self):
        """Test execution risk initialization"""
        self.assertEqual(self.execution_risk.max_memory_gb, 16)
        self.assertEqual(self.execution_risk.max_prediction_time_seconds, 300)

    def test_assess_normal_resources(self):
        """Test assessment with normal resource usage"""
        model = Mock()

        risk_score, recommendations = self.execution_risk.assess(
            model, memory_usage_gb=8, prediction_time_seconds=120
        )

        self.assertLess(risk_score, 0.5)  # Should be low to medium risk

    def test_assess_high_resources(self):
        """Test assessment with high resource usage"""
        model = Mock()

        risk_score, recommendations = self.execution_risk.assess(
            model, memory_usage_gb=20, prediction_time_seconds=600
        )

        self.assertGreater(risk_score, 0.5)  # Should be medium to high risk


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'enable_overfitting_assessment': True,
            'enable_complexity_assessment': True,
            'enable_leakage_assessment': True,
            'enable_execution_assessment': True,
            'overfitting_config': {
                'max_acceptable_gap': 0.1,
                'severe_gap_threshold': 0.2,
                'weight': 1.5
            },
            'complexity_config': {
                'max_safe_features': 100,
                'max_safe_depth': 10,
                'max_training_time_minutes': 30,
                'weight': 1.0
            },
            'leakage_config': {
                'suspicious_score_threshold': 0.95,
                'weight': 2.0
            },
            'execution_config': {
                'max_memory_gb': 16,
                'max_prediction_time_seconds': 300,
                'weight': 1.0
            },
            'risk_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8
            }
        }
        self.risk_manager = RiskManager(self.config)

    def test_initialization(self):
        """Test risk manager initialization"""
        self.assertTrue(self.risk_manager.enable_overfitting_assessment)
        self.assertTrue(self.risk_manager.enable_complexity_assessment)
        self.assertTrue(self.risk_manager.enable_leakage_assessment)
        self.assertTrue(self.risk_manager.enable_execution_assessment)
        self.assertIsNotNone(self.risk_manager.overfitting_risk)
        self.assertIsNotNone(self.risk_manager.complexity_risk)
        self.assertIsNotNone(self.risk_manager.leakage_risk)
        self.assertIsNotNone(self.risk_manager.execution_risk)

    def test_assess_submission_low_risk(self):
        """Test submission assessment with low risk"""
        model = Mock()
        model.n_features_in_ = 50

        assessment = self.risk_manager.assess_submission(
            model=model,
            train_score=0.15,
            validation_score=0.16,
            num_features=50,
            training_time=10,
            memory_usage_gb=8,
            prediction_time_seconds=120
        )

        self.assertIsInstance(assessment, RiskAssessment)
        self.assertLessEqual(assessment.overall_risk, 0.4)
        self.assertIn(assessment.risk_level, ["LOW", "MEDIUM"])

    def test_assess_submission_high_risk(self):
        """Test submission assessment with high risk"""
        model = Mock()
        model.n_features_in_ = 200

        assessment = self.risk_manager.assess_submission(
            model=model,
            train_score=0.10,
            validation_score=0.98,  # Suspicious score
            num_features=200,
            training_time=60,
            memory_usage_gb=20,
            prediction_time_seconds=600
        )

        self.assertIsInstance(assessment, RiskAssessment)
        self.assertGreaterEqual(assessment.overall_risk, 0.5)
        self.assertIn(assessment.risk_level, ["MEDIUM", "HIGH"])

    def test_determine_risk_level(self):
        """Test risk level determination"""
        # Test low risk
        self.assertEqual(self.risk_manager._determine_risk_level(0.2), "LOW")

        # Test medium risk
        self.assertEqual(self.risk_manager._determine_risk_level(0.5), "MEDIUM")

        # Test high risk
        self.assertEqual(self.risk_manager._determine_risk_level(0.9), "HIGH")

    def test_should_block_submission(self):
        """Test submission blocking logic"""
        # Low risk assessment
        low_risk = RiskAssessment(
            overall_risk=0.2,
            risk_level="LOW",
            confidence=0.8,
            risk_factors={},
            recommendations=[],
            metadata={}
        )
        self.assertFalse(self.risk_manager.should_block_submission(low_risk))

        # High risk assessment
        high_risk = RiskAssessment(
            overall_risk=0.9,
            risk_level="HIGH",
            confidence=0.9,
            risk_factors={},
            recommendations=[],
            metadata={}
        )
        self.assertTrue(self.risk_manager.should_block_submission(high_risk))

    def test_get_risk_summary(self):
        """Test risk summary generation"""
        model = Mock()
        model.n_features_in_ = 75

        assessment = self.risk_manager.assess_submission(
            model=model,
            train_score=0.15,
            validation_score=0.18,
            num_features=75,
            training_time=20
        )

        summary = self.risk_manager.get_risk_summary(assessment)

        self.assertIn('Overall Risk', summary)
        self.assertIn('Risk Level', summary)
        self.assertIn('Confidence', summary)


class TestCreateRiskManager(unittest.TestCase):
    """Test cases for create_risk_manager factory function"""

    def test_create_with_config(self):
        """Test creating risk manager with configuration"""
        config = {
            'enable_overfitting_assessment': True,
            'enable_complexity_assessment': False,
            'overfitting_config': {
                'max_acceptable_gap': 0.15,
                'weight': 1.0
            },
            'risk_thresholds': {
                'low': 0.25,
                'medium': 0.65,
                'high': 0.85
            }
        }

        risk_manager = create_risk_manager(config)

        self.assertIsInstance(risk_manager, RiskManager)
        self.assertTrue(risk_manager.enable_overfitting_assessment)
        self.assertFalse(risk_manager.enable_complexity_assessment)

    def test_create_with_defaults(self):
        """Test creating risk manager with default configuration"""
        risk_manager = create_risk_manager({})

        self.assertIsInstance(risk_manager, RiskManager)
        # Should have reasonable defaults
        self.assertTrue(hasattr(risk_manager, 'overfitting_risk'))
        self.assertTrue(hasattr(risk_manager, 'complexity_risk'))


if __name__ == '__main__':
    unittest.main()