#!/usr/bin/env python3
"""
Phase 7: Risk Management System
Advanced risk assessment for submission strategies with multi-dimensional analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result"""
    overall_risk: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_factors: Dict[str, float]
    recommendations: List[str]
    confidence: float
    timestamp: datetime
    details: Dict[str, Any]

class BaseRiskAssessor(ABC):
    """Abstract base class for risk assessors"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.weight = self.config.get('weight', 1.0)
        self.enabled = self.config.get('enabled', True)

    @abstractmethod
    def assess_risk(self,
                   model: Any,
                   validation_data: pd.DataFrame,
                   **kwargs) -> Tuple[float, Dict[str, Any]]:
        """
        Assess risk for a specific model

        Returns:
            Tuple of (risk_score, details)
            risk_score: 0.0 (no risk) to 1.0 (maximum risk)
            details: Additional information about the assessment
        """
        pass

    @abstractmethod
    def get_recommendations(self, risk_score: float, details: Dict[str, Any]) -> List[str]:
        """Get recommendations based on risk assessment"""
        pass

class OverfittingRiskAssessor(BaseRiskAssessor):
    """Assess overfitting risk based on train/validation performance gap"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_acceptable_gap = self.config.get('max_acceptable_gap', 0.1)
        self.severe_gap_threshold = self.config.get('severe_gap_threshold', 0.2)

    def assess_risk(self,
                   model: Any,
                   validation_data: pd.DataFrame,
                   **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Assess overfitting risk"""
        try:
            train_score = kwargs.get('train_score')
            validation_score = kwargs.get('validation_score')

            if train_score is None or validation_score is None:
                logger.warning("Train or validation score not provided for overfitting assessment")
                return 0.5, {'error': 'Missing scores', 'estimated': True}

            # Calculate performance gap
            performance_gap = abs(validation_score - train_score) / max(abs(train_score), 0.001)

            # Calculate risk score
            if performance_gap <= self.max_acceptable_gap:
                risk_score = performance_gap / self.max_acceptable_gap * 0.3
            elif performance_gap <= self.severe_gap_threshold:
                risk_score = 0.3 + (performance_gap - self.max_acceptable_gap) / (self.severe_gap_threshold - self.max_acceptable_gap) * 0.4
            else:
                risk_score = 0.7 + min((performance_gap - self.severe_gap_threshold) / self.severe_gap_threshold, 1.0) * 0.3

            # Additional checks
            details = {
                'performance_gap': performance_gap,
                'train_score': train_score,
                'validation_score': validation_score,
                'gap_threshold': self.max_acceptable_gap,
                'severe_threshold': self.severe_gap_threshold
            }

            # Check for cross-validation consistency
            cv_scores = kwargs.get('cv_scores')
            if cv_scores is not None:
                cv_std = np.std(cv_scores)
                cv_mean = np.mean(cv_scores)
                cv_consistency = cv_std / max(abs(cv_mean), 0.001)
                details['cv_consistency'] = cv_consistency

                if cv_consistency > 0.1:  # High variability
                    risk_score = min(risk_score + 0.2, 1.0)

            return risk_score, details

        except Exception as e:
            logger.error(f"Error in overfitting risk assessment: {str(e)}")
            return 0.8, {'error': str(e), 'estimated': True}

    def get_recommendations(self, risk_score: float, details: Dict[str, Any]) -> List[str]:
        """Get overfitting-related recommendations"""
        recommendations = []

        if risk_score > 0.7:
            recommendations.extend([
                "HIGH OVERFITTING RISK: Consider simpler model",
                "Reduce model complexity (fewer features, higher regularization)",
                "Increase training data or use data augmentation"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Moderate overfitting detected: Add regularization",
                "Consider feature selection or dimensionality reduction",
                "Validate with additional holdout set"
            ])
        elif risk_score > 0.2:
            recommendations.append("Minor overfitting: Monitor performance closely")

        if 'cv_consistency' in details and details['cv_consistency'] > 0.1:
            recommendations.append("High CV variability: Ensure stable validation strategy")

        return recommendations

class ComplexityRiskAssessor(BaseRiskAssessor):
    """Assess model complexity risk"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_features = self.config.get('max_safe_features', 1000)
        self.max_depth = self.config.get('max_safe_depth', 20)

    def assess_risk(self,
                   model: Any,
                   validation_data: pd.DataFrame,
                   **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Assess complexity risk"""
        try:
            risk_score = 0.0
            details = {}

            # Feature count risk
            feature_count = kwargs.get('feature_count', 0)
            if hasattr(model, 'n_features_in_'):
                feature_count = model.n_features_in_

            if feature_count > 0:
                feature_risk = min(feature_count / self.max_features, 1.0)
                risk_score += feature_risk * 0.4
                details['feature_count'] = feature_count
                details['feature_risk'] = feature_risk

            # Model-specific complexity
            model_complexity = self._assess_model_complexity(model)
            risk_score += model_complexity * 0.6
            details['model_complexity'] = model_complexity

            # Training time risk
            training_time = kwargs.get('training_time', 0)
            if training_time > 300:  # 5 minutes
                time_risk = min((training_time - 300) / 1200, 0.5)  # Max 0.5 risk from time
                risk_score = min(risk_score + time_risk, 1.0)
                details['training_time'] = training_time
                details['time_risk'] = time_risk

            return min(risk_score, 1.0), details

        except Exception as e:
            logger.error(f"Error in complexity risk assessment: {str(e)}")
            return 0.6, {'error': str(e), 'estimated': True}

    def _assess_model_complexity(self, model: Any) -> float:
        """Assess model-specific complexity"""
        complexity = 0.0

        try:
            # LightGBM/XGBoost complexity
            if hasattr(model, 'num_trees'):
                tree_count = model.num_trees()
                complexity = min(tree_count / 1000, 0.8)  # Risk increases with tree count

            elif hasattr(model, 'n_estimators'):
                estimators = model.n_estimators
                complexity = min(estimators / 500, 0.6)

            # Deep model complexity
            elif hasattr(model, 'get_params'):
                params = model.get_params()
                if 'max_depth' in params:
                    depth = params['max_depth'] or 6
                    complexity = min(depth / self.max_depth, 0.4)

            # Ensemble complexity
            elif hasattr(model, 'estimators_'):
                ensemble_size = len(model.estimators_)
                complexity = min(ensemble_size / 10, 0.7)

        except Exception as e:
            logger.debug(f"Could not assess model complexity: {str(e)}")
            complexity = 0.3  # Default medium complexity

        return complexity

    def get_recommendations(self, risk_score: float, details: Dict[str, Any]) -> List[str]:
        """Get complexity-related recommendations"""
        recommendations = []

        if risk_score > 0.7:
            recommendations.extend([
                "HIGH COMPLEXITY RISK: Simplify model architecture",
                "Reduce number of features or use feature selection",
                "Consider simpler model types"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Moderate complexity: Monitor training stability",
                "Consider early stopping or pruning"
            ])

        if 'feature_count' in details and details['feature_count'] > self.max_features:
            recommendations.append(f"Feature count ({details['feature_count']}) exceeds safe limit")

        if 'training_time' in details and details['training_time'] > 600:
            recommendations.append("Long training time may indicate excessive complexity")

        return recommendations

class LeakageRiskAssessor(BaseRiskAssessor):
    """Assess data leakage risk"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.suspicious_score_threshold = self.config.get('suspicious_score_threshold', 0.95)

    def assess_risk(self,
                   model: Any,
                   validation_data: pd.DataFrame,
                   **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Assess data leakage risk"""
        try:
            risk_score = 0.0
            details = {}

            # Perfect or near-perfect scores are suspicious
            validation_score = kwargs.get('validation_score')
            if validation_score is not None:
                # For WMAPE, lower is better, so very low scores are suspicious
                if validation_score < 5.0:  # WMAPE < 5% is very suspicious
                    score_risk = 1.0
                elif validation_score < 10.0:  # WMAPE < 10% is suspicious
                    score_risk = 0.7
                else:
                    score_risk = 0.0

                risk_score += score_risk * 0.6
                details['validation_score'] = validation_score
                details['score_risk'] = score_risk

            # Feature importance analysis
            feature_importance = kwargs.get('feature_importance')
            if feature_importance is not None:
                importance_risk = self._assess_feature_importance_risk(feature_importance)
                risk_score += importance_risk * 0.4
                details['importance_risk'] = importance_risk

            # Temporal consistency check
            temporal_consistency = kwargs.get('temporal_consistency')
            if temporal_consistency is not None and temporal_consistency > 0.99:
                risk_score = min(risk_score + 0.3, 1.0)
                details['temporal_consistency'] = temporal_consistency

            return min(risk_score, 1.0), details

        except Exception as e:
            logger.error(f"Error in leakage risk assessment: {str(e)}")
            return 0.5, {'error': str(e), 'estimated': True}

    def _assess_feature_importance_risk(self, feature_importance: Dict[str, float]) -> float:
        """Assess risk based on feature importance patterns"""
        if not feature_importance:
            return 0.0

        # Check for single dominant feature
        sorted_importance = sorted(feature_importance.values(), reverse=True)
        if len(sorted_importance) > 1:
            top_feature_ratio = sorted_importance[0] / max(sorted_importance[1], 0.001)
            if top_feature_ratio > 10:  # One feature dominates
                return 0.8
            elif top_feature_ratio > 5:
                return 0.4

        # Check for suspicious feature names
        suspicious_patterns = ['target', 'label', 'y_', 'future', 'next_', 'answer']
        for feature_name in feature_importance.keys():
            if any(pattern in feature_name.lower() for pattern in suspicious_patterns):
                return 0.9

        return 0.0

    def get_recommendations(self, risk_score: float, details: Dict[str, Any]) -> List[str]:
        """Get leakage-related recommendations"""
        recommendations = []

        if risk_score > 0.8:
            recommendations.extend([
                "CRITICAL LEAKAGE RISK: Investigate data pipeline",
                "Check for future information in features",
                "Verify temporal data splitting"
            ])
        elif risk_score > 0.5:
            recommendations.extend([
                "Potential leakage detected: Review feature engineering",
                "Validate data preprocessing steps"
            ])

        if 'score_risk' in details and details['score_risk'] > 0.5:
            recommendations.append("Unusually good performance may indicate leakage")

        return recommendations

class ExecutionRiskAssessor(BaseRiskAssessor):
    """Assess execution and deployment risk"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_memory_gb = self.config.get('max_memory_gb', 16)
        self.max_prediction_time = self.config.get('max_prediction_time_seconds', 300)

    def assess_risk(self,
                   model: Any,
                   validation_data: pd.DataFrame,
                   **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Assess execution risk"""
        try:
            risk_score = 0.0
            details = {}

            # Memory usage risk
            memory_usage = kwargs.get('memory_usage_gb', 0)
            if memory_usage > 0:
                memory_risk = min(memory_usage / self.max_memory_gb, 1.0)
                risk_score += memory_risk * 0.3
                details['memory_usage_gb'] = memory_usage
                details['memory_risk'] = memory_risk

            # Prediction time risk
            prediction_time = kwargs.get('prediction_time_seconds', 0)
            if prediction_time > 0:
                time_risk = min(prediction_time / self.max_prediction_time, 1.0)
                risk_score += time_risk * 0.3
                details['prediction_time_seconds'] = prediction_time
                details['time_risk'] = time_risk

            # Model stability risk
            stability_score = self._assess_model_stability(model, validation_data)
            risk_score += stability_score * 0.4
            details['stability_risk'] = stability_score

            return min(risk_score, 1.0), details

        except Exception as e:
            logger.error(f"Error in execution risk assessment: {str(e)}")
            return 0.6, {'error': str(e), 'estimated': True}

    def _assess_model_stability(self, model: Any, validation_data: pd.DataFrame) -> float:
        """Assess model stability"""
        try:
            # Test with small perturbations
            if len(validation_data) > 100:
                sample_data = validation_data.sample(100)

                # Add small noise and check prediction stability
                noise_scale = sample_data.select_dtypes(include=[np.number]).std().mean() * 0.01
                noisy_data = sample_data.copy()

                for col in sample_data.select_dtypes(include=[np.number]).columns:
                    noisy_data[col] += np.random.normal(0, noise_scale, len(noisy_data))

                # Compare predictions (this would need actual model interface)
                # For now, return moderate risk
                return 0.3

        except Exception:
            pass

        return 0.4  # Default moderate stability risk

    def get_recommendations(self, risk_score: float, details: Dict[str, Any]) -> List[str]:
        """Get execution-related recommendations"""
        recommendations = []

        if risk_score > 0.7:
            recommendations.extend([
                "HIGH EXECUTION RISK: Test deployment thoroughly",
                "Consider model simplification for reliability",
                "Implement fallback strategies"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Moderate execution risk: Monitor resource usage",
                "Test prediction pipeline end-to-end"
            ])

        if 'memory_risk' in details and details['memory_risk'] > 0.5:
            recommendations.append(f"High memory usage: {details['memory_usage_gb']:.1f}GB")

        if 'time_risk' in details and details['time_risk'] > 0.5:
            recommendations.append(f"Long prediction time: {details['prediction_time_seconds']:.1f}s")

        return recommendations

class RiskManager:
    """Comprehensive risk management system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.assessors = self._initialize_assessors()
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }

    def _initialize_assessors(self) -> Dict[str, BaseRiskAssessor]:
        """Initialize risk assessors"""
        assessors = {}

        # Overfitting risk
        if self.config.get('enable_overfitting_assessment', True):
            assessors['overfitting'] = OverfittingRiskAssessor(
                self.config.get('overfitting_config', {})
            )

        # Complexity risk
        if self.config.get('enable_complexity_assessment', True):
            assessors['complexity'] = ComplexityRiskAssessor(
                self.config.get('complexity_config', {})
            )

        # Leakage risk
        if self.config.get('enable_leakage_assessment', True):
            assessors['leakage'] = LeakageRiskAssessor(
                self.config.get('leakage_config', {})
            )

        # Execution risk
        if self.config.get('enable_execution_assessment', True):
            assessors['execution'] = ExecutionRiskAssessor(
                self.config.get('execution_config', {})
            )

        return assessors

    def assess_full_risk(self,
                        model: Any,
                        validation_data: pd.DataFrame,
                        **kwargs) -> RiskAssessment:
        """Perform comprehensive risk assessment"""
        logger.info("Performing comprehensive risk assessment")

        risk_factors = {}
        all_recommendations = []
        total_risk = 0.0
        total_weight = 0.0

        # Run all assessors
        for name, assessor in self.assessors.items():
            if assessor.enabled:
                try:
                    risk_score, details = assessor.assess_risk(model, validation_data, **kwargs)
                    recommendations = assessor.get_recommendations(risk_score, details)

                    risk_factors[name] = {
                        'score': risk_score,
                        'weight': assessor.weight,
                        'details': details
                    }

                    all_recommendations.extend([f"{name.title()}: {rec}" for rec in recommendations])

                    # Weighted average
                    total_risk += risk_score * assessor.weight
                    total_weight += assessor.weight

                except Exception as e:
                    logger.error(f"Error in {name} risk assessment: {str(e)}")
                    risk_factors[name] = {
                        'score': 0.8,
                        'weight': assessor.weight,
                        'details': {'error': str(e)}
                    }

        # Calculate overall risk
        overall_risk = total_risk / max(total_weight, 1.0)

        # Determine risk level
        if overall_risk <= self.risk_thresholds['low']:
            risk_level = 'LOW'
        elif overall_risk <= self.risk_thresholds['medium']:
            risk_level = 'MEDIUM'
        elif overall_risk <= self.risk_thresholds['high']:
            risk_level = 'HIGH'
        else:
            risk_level = 'CRITICAL'

        # Calculate confidence
        confidence = self._calculate_confidence(risk_factors)

        return RiskAssessment(
            overall_risk=overall_risk,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=all_recommendations,
            confidence=confidence,
            timestamp=datetime.now(),
            details={
                'assessors_used': list(self.assessors.keys()),
                'total_weight': total_weight,
                'risk_thresholds': self.risk_thresholds
            }
        )

    def _calculate_confidence(self, risk_factors: Dict[str, Dict[str, Any]]) -> float:
        """Calculate confidence in risk assessment"""
        # Higher confidence when more assessors agree
        scores = [factor['score'] for factor in risk_factors.values()]

        if len(scores) < 2:
            return 0.6  # Lower confidence with few assessors

        # Calculate agreement (inverse of standard deviation)
        score_std = np.std(scores)
        agreement = max(0.0, 1.0 - score_std)

        # Adjust for number of assessors
        assessor_bonus = min(len(scores) / 4, 0.2)

        confidence = 0.5 + agreement * 0.3 + assessor_bonus

        return min(confidence, 1.0)

    def get_risk_summary(self, risk_assessment: RiskAssessment) -> str:
        """Get human-readable risk summary"""
        summary_lines = [
            f"RISK LEVEL: {risk_assessment.risk_level}",
            f"Overall Risk Score: {risk_assessment.overall_risk:.2f}",
            f"Confidence: {risk_assessment.confidence:.2f}",
            ""
        ]

        # Add factor breakdown
        summary_lines.append("Risk Factors:")
        for factor_name, factor_data in risk_assessment.risk_factors.items():
            score = factor_data['score']
            summary_lines.append(f"  {factor_name.title()}: {score:.2f}")

        # Add recommendations
        if risk_assessment.recommendations:
            summary_lines.append("\nRecommendations:")
            for rec in risk_assessment.recommendations[:5]:  # Top 5
                summary_lines.append(f"  • {rec}")

        return "\n".join(summary_lines)

def weighted_average(risk_factors: Dict[str, float]) -> float:
    """
    Calculate weighted average of risk factors

    Args:
        risk_factors: Dictionary with risk factor names and scores (0.0 to 1.0)

    Returns:
        Weighted average risk score
    """
    if not risk_factors:
        return 0.0

    # Default weights for risk factors
    weights = {
        'overfitting_risk': 1.5,  # Most critical
        'complexity_risk': 1.0,
        'data_leakage_risk': 2.0,  # Very critical
        'execution_risk': 1.0
    }

    total_weighted_score = 0.0
    total_weight = 0.0

    for factor, score in risk_factors.items():
        weight = weights.get(factor, 1.0)
        total_weighted_score += score * weight
        total_weight += weight

    return total_weighted_score / total_weight if total_weight > 0 else 0.0


def calculate_train_val_gap(model: Any) -> float:
    """Calculate training vs validation gap for overfitting assessment"""
    # Handle both dict and object models
    def get_param(param_name, default=0):
        if isinstance(model, dict):
            return model.get(param_name, default)
        else:
            return getattr(model, param_name, default)

    # Check for suspiciously high validation score (possible overfitting)
    val_score = get_param('validation_score', 0)
    if val_score > 0.95:
        return 0.9  # Very high overfitting risk

    # Check actual train-val gap if available
    train_score = get_param('train_score', 0)
    if train_score > 0 and val_score > 0:
        return min(abs(train_score - val_score), 1.0)

    return 0.1  # Default moderate gap


def assess_model_complexity(model: Any) -> float:
    """Assess model complexity risk"""
    # Mock implementation - in real scenario would analyze model parameters
    complexity_indicators = 0

    # Handle both dict and object models
    def get_param(param_name, default=0):
        if isinstance(model, dict):
            return model.get(param_name, default)
        else:
            return getattr(model, param_name, default)

    if get_param('n_estimators', 0) > 1000:
        complexity_indicators += 0.3
    if get_param('max_depth', 0) > 20:
        complexity_indicators += 0.3
    if get_param('learning_rate', 1.0) < 0.01:
        complexity_indicators += 0.2

    return min(complexity_indicators, 1.0)


def check_for_leakage(model: Any) -> float:
    """Check for potential data leakage"""
    # Mock implementation - in real scenario would analyze feature importance and data
    # Handle both dict and object models
    def get_param(param_name, default=0):
        if isinstance(model, dict):
            return model.get(param_name, default)
        else:
            return getattr(model, param_name, default)

    if get_param('validation_score', 0) > 0.95:
        return 0.8  # Suspiciously high score
    return 0.1  # Low leakage risk


def test_prediction_pipeline(model: Any) -> float:
    """Test prediction pipeline execution risk"""
    try:
        # Mock test - in real scenario would run actual pipeline test
        if hasattr(model, 'predict'):
            return 0.1  # Low execution risk
        else:
            return 0.7  # High risk if no predict method
    except Exception:
        return 0.9  # Very high risk if pipeline fails


def assess_submission_risk(model: Any, validation_score: float) -> str:
    """
    Submission Risk Assessment

    Args:
        model: The trained model
        validation_score: Validation score (typically WMAPE)

    Returns:
        Risk assessment string with recommendation
    """
    risk_factors = {
        'overfitting_risk': calculate_train_val_gap(model),
        'complexity_risk': assess_model_complexity(model),
        'data_leakage_risk': check_for_leakage(model),
        'execution_risk': test_prediction_pipeline(model)
    }

    overall_risk = weighted_average(risk_factors)

    if overall_risk > 0.7:
        return 'HIGH_RISK - Consider simpler model'
    elif overall_risk > 0.4:
        return 'MEDIUM_RISK - Additional validation needed'
    else:
        return 'LOW_RISK - Safe to submit'


# Legacy function for backward compatibility
def assess_submission_risk_legacy(model: Any,
                                 validation_data: pd.DataFrame,
                                 config: Optional[Dict[str, Any]] = None,
                                 **kwargs) -> RiskAssessment:
    """Convenience function for risk assessment (legacy version)"""
    risk_manager = RiskManager(config)
    return risk_manager.assess_full_risk(model, validation_data, **kwargs)


def create_risk_manager(config: Optional[Dict[str, Any]] = None) -> RiskManager:
    """Factory function to create a configured RiskManager"""
    if config is None:
        config = {}
    return RiskManager(config)


if __name__ == "__main__":
    # Demo usage
    print("⚠️ Risk Management System Demo")
    print("=" * 50)

    # Create risk manager
    risk_manager = RiskManager()
    print(f"✅ Created risk manager with {len(risk_manager.assessors)} assessors")

    # Mock risk assessment
    mock_data = pd.DataFrame({'feature': range(100)})
    mock_kwargs = {
        'train_score': 15.0,
        'validation_score': 18.5,
        'feature_count': 150,
        'training_time': 180
    }

    assessment = risk_manager.assess_full_risk(None, mock_data, **mock_kwargs)

    print(f"\nRisk Assessment Results:")
    print(f"Overall Risk: {assessment.overall_risk:.2f}")
    print(f"Risk Level: {assessment.risk_level}")
    print(f"Confidence: {assessment.confidence:.2f}")

    print(f"\nRecommendations ({len(assessment.recommendations)}):")
    for rec in assessment.recommendations[:3]:
        print(f"  • {rec}")

    print("\n⚠️ Risk management system ready!")
    print("Ready to assess submission risks comprehensively.")