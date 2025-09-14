#!/usr/bin/env python3
"""
Phase 6: Model Validator
Comprehensive model validation system with automated checks and business rules
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import wmape, mape, mae, rmse
from architecture.observers import event_publisher
from config.phase6_config import get_config
from validation.kpi_system import kpi_manager

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Validation result status"""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"

class ValidationCategory(Enum):
    """Validation check categories"""
    STATISTICAL = "statistical"
    BUSINESS = "business"
    TECHNICAL = "technical"
    DATA_QUALITY = "data_quality"

@dataclass
class ValidationCheck:
    """Individual validation check result"""
    name: str
    category: ValidationCategory
    result: ValidationResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'category': self.category.value,
            'result': self.result.value,
            'message': self.message,
            'details': self.details,
            'severity': self.severity,
            'recommendation': self.recommendation
        }

@dataclass
class ModelValidationReport:
    """Complete model validation report"""
    model_name: str
    validation_timestamp: datetime
    overall_result: ValidationResult
    checks: List[ValidationCheck]
    summary_metrics: Dict[str, float]
    kpi_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_passed_checks(self) -> List[ValidationCheck]:
        """Get all passed checks"""
        return [check for check in self.checks if check.result == ValidationResult.PASSED]

    def get_warning_checks(self) -> List[ValidationCheck]:
        """Get all warning checks"""
        return [check for check in self.checks if check.result == ValidationResult.WARNING]

    def get_failed_checks(self) -> List[ValidationCheck]:
        """Get all failed checks"""
        return [check for check in self.checks if check.result == ValidationResult.FAILED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_name': self.model_name,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'overall_result': self.overall_result.value,
            'checks': [check.to_dict() for check in self.checks],
            'summary_metrics': self.summary_metrics,
            'kpi_results': self.kpi_results,
            'metadata': self.metadata,
            'summary': {
                'total_checks': len(self.checks),
                'passed': len(self.get_passed_checks()),
                'warnings': len(self.get_warning_checks()),
                'failed': len(self.get_failed_checks())
            }
        }

class BaseValidator:
    """Base class for validation checks"""

    def __init__(self, name: str, category: ValidationCategory, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.category = category
        self.config = config or {}

    def validate(self, actual: np.ndarray, predicted: np.ndarray,
                model: Any = None, metadata: Optional[Dict[str, Any]] = None) -> ValidationCheck:
        """Perform validation check"""
        try:
            return self._perform_validation(actual, predicted, model, metadata)
        except Exception as e:
            logger.error(f"Validation check {self.name} failed: {e}")
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.FAILED,
                message=f"Validation check failed: {str(e)}",
                severity="high"
            )

    def _perform_validation(self, actual: np.ndarray, predicted: np.ndarray,
                           model: Any = None, metadata: Optional[Dict[str, Any]] = None) -> ValidationCheck:
        """Override in subclasses"""
        raise NotImplementedError

class AccuracyValidator(BaseValidator):
    """Validates model accuracy against thresholds"""

    def __init__(self, wmape_threshold: float = 0.20, mape_threshold: float = 0.25):
        super().__init__("accuracy_check", ValidationCategory.STATISTICAL)
        self.wmape_threshold = wmape_threshold
        self.mape_threshold = mape_threshold

    def _perform_validation(self, actual: np.ndarray, predicted: np.ndarray,
                           model: Any = None, metadata: Optional[Dict[str, Any]] = None) -> ValidationCheck:
        """Validate accuracy metrics"""
        wmape_score = wmape(actual, predicted)
        mape_score = mape(actual, predicted)

        details = {
            'wmape': wmape_score,
            'mape': mape_score,
            'wmape_threshold': self.wmape_threshold,
            'mape_threshold': self.mape_threshold
        }

        if wmape_score <= self.wmape_threshold and mape_score <= self.mape_threshold:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.PASSED,
                message=f"Accuracy acceptable: WMAPE={wmape_score:.3f}, MAPE={mape_score:.3f}",
                details=details
            )
        elif wmape_score <= self.wmape_threshold * 1.2:  # 20% tolerance
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.WARNING,
                message=f"Accuracy marginal: WMAPE={wmape_score:.3f}, MAPE={mape_score:.3f}",
                details=details,
                recommendation="Consider feature engineering or hyperparameter tuning"
            )
        else:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.FAILED,
                message=f"Accuracy poor: WMAPE={wmape_score:.3f}, MAPE={mape_score:.3f}",
                details=details,
                severity="high",
                recommendation="Model requires significant improvement or replacement"
            )

class BiasValidator(BaseValidator):
    """Validates forecast bias"""

    def __init__(self, bias_threshold: float = 0.10):
        super().__init__("bias_check", ValidationCategory.STATISTICAL)
        self.bias_threshold = bias_threshold

    def _perform_validation(self, actual: np.ndarray, predicted: np.ndarray,
                           model: Any = None, metadata: Optional[Dict[str, Any]] = None) -> ValidationCheck:
        """Validate forecast bias"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bias = np.mean((predicted - actual) / actual)
            abs_bias = abs(bias)

        details = {
            'bias': bias,
            'absolute_bias': abs_bias,
            'bias_threshold': self.bias_threshold,
            'direction': 'overforecast' if bias > 0 else 'underforecast'
        }

        if abs_bias <= self.bias_threshold:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.PASSED,
                message=f"Bias acceptable: {bias:.3f} ({details['direction']})",
                details=details
            )
        elif abs_bias <= self.bias_threshold * 1.5:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.WARNING,
                message=f"Significant bias detected: {bias:.3f} ({details['direction']})",
                details=details,
                recommendation="Review model calibration and training data balance"
            )
        else:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.FAILED,
                message=f"Severe bias detected: {bias:.3f} ({details['direction']})",
                details=details,
                severity="high",
                recommendation="Critical bias issue - investigate data quality and model assumptions"
            )

class ResidualValidator(BaseValidator):
    """Validates residual patterns"""

    def __init__(self):
        super().__init__("residual_analysis", ValidationCategory.STATISTICAL)

    def _perform_validation(self, actual: np.ndarray, predicted: np.ndarray,
                           model: Any = None, metadata: Optional[Dict[str, Any]] = None) -> ValidationCheck:
        """Validate residual patterns"""
        residuals = actual - predicted

        # Calculate residual statistics
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        residual_skew = self._calculate_skewness(residuals)
        residual_kurt = self._calculate_kurtosis(residuals)

        # Test for autocorrelation in residuals
        autocorr = self._calculate_autocorrelation(residuals, lag=1)

        details = {
            'mean': residual_mean,
            'std': residual_std,
            'skewness': residual_skew,
            'kurtosis': residual_kurt,
            'autocorrelation_lag1': autocorr,
            'normality_test': abs(residual_skew) < 2 and abs(residual_kurt) < 7
        }

        issues = []
        if abs(residual_skew) > 2:
            issues.append("high skewness")
        if abs(residual_kurt) > 7:
            issues.append("high kurtosis")
        if abs(autocorr) > 0.3:
            issues.append("autocorrelation")

        if not issues:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.PASSED,
                message="Residuals show good statistical properties",
                details=details
            )
        elif len(issues) <= 1:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.WARNING,
                message=f"Residual issues detected: {', '.join(issues)}",
                details=details,
                recommendation="Consider model improvements or feature engineering"
            )
        else:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.FAILED,
                message=f"Multiple residual issues: {', '.join(issues)}",
                details=details,
                severity="high",
                recommendation="Significant model fit issues - review model architecture"
            )

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        if len(data) < 4:
            return 3.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 3.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_autocorrelation(self, data: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag"""
        if len(data) <= lag:
            return 0.0
        return np.corrcoef(data[:-lag], data[lag:])[0, 1]

class BusinessRulesValidator(BaseValidator):
    """Validates business rule compliance"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("business_rules", ValidationCategory.BUSINESS, config)

    def _perform_validation(self, actual: np.ndarray, predicted: np.ndarray,
                           model: Any = None, metadata: Optional[Dict[str, Any]] = None) -> ValidationCheck:
        """Validate business rules"""
        violations = []

        # Non-negativity check
        negative_predictions = np.sum(predicted < 0)
        if negative_predictions > 0:
            violations.append(f"{negative_predictions} negative predictions")

        # Extreme values check
        q99 = np.percentile(actual, 99)
        extreme_predictions = np.sum(predicted > q99 * 3)  # 3x the 99th percentile
        if extreme_predictions > len(predicted) * 0.01:  # More than 1%
            violations.append(f"{extreme_predictions} extreme predictions")

        # Reasonable forecast horizon check
        if metadata and 'forecast_horizon' in metadata:
            horizon = metadata['forecast_horizon']
            if horizon > 365:  # More than 1 year
                violations.append(f"forecast horizon too long ({horizon} days)")

        # Seasonal consistency (if applicable)
        if metadata and 'seasonality_expected' in metadata:
            if metadata['seasonality_expected'] and len(predicted) >= 52:
                seasonal_pattern = self._check_seasonal_pattern(predicted)
                if not seasonal_pattern:
                    violations.append("missing expected seasonal pattern")

        details = {
            'negative_predictions': negative_predictions,
            'extreme_predictions': extreme_predictions,
            'violations': violations
        }

        if not violations:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.PASSED,
                message="All business rules satisfied",
                details=details
            )
        elif len(violations) <= 2:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.WARNING,
                message=f"Business rule violations: {'; '.join(violations)}",
                details=details,
                recommendation="Review and apply business constraints"
            )
        else:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.FAILED,
                message=f"Multiple business rule violations: {'; '.join(violations)}",
                details=details,
                severity="high",
                recommendation="Critical business rule compliance issues"
            )

    def _check_seasonal_pattern(self, data: np.ndarray, period: int = 52) -> bool:
        """Check for seasonal pattern in data"""
        if len(data) < period * 2:
            return True  # Can't validate with insufficient data

        # Simple seasonality check using autocorrelation
        autocorr = np.corrcoef(data[:-period], data[period:])[0, 1]
        return autocorr > 0.3  # Moderate seasonal correlation

class DataQualityValidator(BaseValidator):
    """Validates data quality issues"""

    def __init__(self):
        super().__init__("data_quality", ValidationCategory.DATA_QUALITY)

    def _perform_validation(self, actual: np.ndarray, predicted: np.ndarray,
                           model: Any = None, metadata: Optional[Dict[str, Any]] = None) -> ValidationCheck:
        """Validate data quality"""
        issues = []

        # Check for missing values
        missing_actual = np.sum(np.isnan(actual))
        missing_predicted = np.sum(np.isnan(predicted))
        if missing_actual > 0:
            issues.append(f"{missing_actual} missing actual values")
        if missing_predicted > 0:
            issues.append(f"{missing_predicted} missing predictions")

        # Check for infinite values
        inf_actual = np.sum(np.isinf(actual))
        inf_predicted = np.sum(np.isinf(predicted))
        if inf_actual > 0:
            issues.append(f"{inf_actual} infinite actual values")
        if inf_predicted > 0:
            issues.append(f"{inf_predicted} infinite predictions")

        # Check data size
        if len(actual) < 30:
            issues.append(f"insufficient data size ({len(actual)} samples)")

        # Check for constant predictions
        if np.std(predicted) < 1e-6:
            issues.append("predictions are nearly constant")

        details = {
            'data_size': len(actual),
            'missing_actual': missing_actual,
            'missing_predicted': missing_predicted,
            'infinite_actual': inf_actual,
            'infinite_predicted': inf_predicted,
            'prediction_variance': np.var(predicted),
            'issues': issues
        }

        if not issues:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.PASSED,
                message="Data quality is good",
                details=details
            )
        elif len(issues) <= 1:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.WARNING,
                message=f"Data quality issues: {'; '.join(issues)}",
                details=details,
                recommendation="Address data quality issues before model deployment"
            )
        else:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.FAILED,
                message=f"Significant data quality problems: {'; '.join(issues)}",
                details=details,
                severity="high",
                recommendation="Critical data quality issues must be resolved"
            )

class CrossValidationValidator(BaseValidator):
    """Validates model using cross-validation"""

    def __init__(self, cv_folds: int = 5, min_score: float = 0.5):
        super().__init__("cross_validation", ValidationCategory.TECHNICAL)
        self.cv_folds = cv_folds
        self.min_score = min_score

    def _perform_validation(self, actual: np.ndarray, predicted: np.ndarray,
                           model: Any = None, metadata: Optional[Dict[str, Any]] = None) -> ValidationCheck:
        """Validate using cross-validation"""
        if model is None:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.WARNING,
                message="No model provided for cross-validation",
                recommendation="Provide trained model for comprehensive validation"
            )

        try:
            # Prepare features (simplified - assuming we have the original features)
            X = metadata.get('features', np.column_stack([actual, predicted]))  # Fallback
            y = actual

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')

            # Convert to positive RMSE scores
            rmse_scores = np.sqrt(-cv_scores)
            mean_rmse = np.mean(rmse_scores)
            std_rmse = np.std(rmse_scores)

            # Calculate R² for comparison
            try:
                r2_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
                mean_r2 = np.mean(r2_scores)
            except:
                mean_r2 = None

            details = {
                'cv_folds': self.cv_folds,
                'mean_rmse': mean_rmse,
                'std_rmse': std_rmse,
                'mean_r2': mean_r2,
                'rmse_scores': rmse_scores.tolist(),
                'min_score_threshold': self.min_score
            }

            if mean_r2 and mean_r2 >= self.min_score:
                return ValidationCheck(
                    name=self.name,
                    category=self.category,
                    result=ValidationResult.PASSED,
                    message=f"Cross-validation passed: R²={mean_r2:.3f}, RMSE={mean_rmse:.3f}±{std_rmse:.3f}",
                    details=details
                )
            elif mean_r2 and mean_r2 >= self.min_score * 0.8:
                return ValidationCheck(
                    name=self.name,
                    category=self.category,
                    result=ValidationResult.WARNING,
                    message=f"Cross-validation marginal: R²={mean_r2:.3f}, RMSE={mean_rmse:.3f}±{std_rmse:.3f}",
                    details=details,
                    recommendation="Consider model improvements for better cross-validation performance"
                )
            else:
                return ValidationCheck(
                    name=self.name,
                    category=self.category,
                    result=ValidationResult.FAILED,
                    message=f"Cross-validation failed: R²={mean_r2:.3f}, RMSE={mean_rmse:.3f}±{std_rmse:.3f}",
                    details=details,
                    severity="high",
                    recommendation="Model shows poor generalization performance"
                )

        except Exception as e:
            return ValidationCheck(
                name=self.name,
                category=self.category,
                result=ValidationResult.FAILED,
                message=f"Cross-validation error: {str(e)}",
                severity="medium",
                recommendation="Check model compatibility with cross-validation"
            )

class ModelValidator:
    """Comprehensive model validation system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().dict()
        self.validators: List[BaseValidator] = []
        self.validation_history: List[ModelValidationReport] = []
        self.max_history = 100

        self._initialize_validators()

    def _initialize_validators(self):
        """Initialize default validators"""
        validation_config = self.config.get('validation', {})

        self.validators = [
            AccuracyValidator(
                wmape_threshold=validation_config.get('wmape_threshold', 0.20),
                mape_threshold=validation_config.get('mape_threshold', 0.25)
            ),
            BiasValidator(validation_config.get('bias_threshold', 0.10)),
            ResidualValidator(),
            BusinessRulesValidator(validation_config.get('business_rules', {})),
            DataQualityValidator(),
            CrossValidationValidator(
                cv_folds=validation_config.get('cv_folds', 5),
                min_score=validation_config.get('min_r2_score', 0.5)
            )
        ]

    def add_validator(self, validator: BaseValidator):
        """Add custom validator"""
        self.validators.append(validator)

    def validate_model(self, model_name: str, actual: np.ndarray, predicted: np.ndarray,
                      model: Any = None, metadata: Optional[Dict[str, Any]] = None) -> ModelValidationReport:
        """Perform comprehensive model validation"""
        logger.info(f"Starting model validation: {model_name}")

        validation_checks = []
        for validator in self.validators:
            try:
                check = validator.validate(actual, predicted, model, metadata)
                validation_checks.append(check)
            except Exception as e:
                logger.error(f"Validator {validator.name} failed: {e}")
                validation_checks.append(ValidationCheck(
                    name=validator.name,
                    category=validator.category,
                    result=ValidationResult.FAILED,
                    message=f"Validator error: {str(e)}",
                    severity="medium"
                ))

        # Calculate summary metrics
        summary_metrics = {
            'wmape': wmape(actual, predicted),
            'mape': mape(actual, predicted),
            'mae': mae(actual, predicted),
            'rmse': rmse(actual, predicted),
            'r2': r2_score(actual, predicted) if len(actual) > 1 else 0.0
        }

        # Calculate KPIs
        kpi_results = None
        try:
            kpi_metadata = metadata.copy() if metadata else {}
            kpi_metadata.update(summary_metrics)
            kpi_results = kpi_manager.calculate_all_kpis(actual, predicted, kpi_metadata)
            kpi_results = {name: result.to_dict() for name, result in kpi_results.items()}
        except Exception as e:
            logger.error(f"KPI calculation failed: {e}")

        # Determine overall result
        overall_result = self._determine_overall_result(validation_checks)

        # Create validation report
        report = ModelValidationReport(
            model_name=model_name,
            validation_timestamp=datetime.now(),
            overall_result=overall_result,
            checks=validation_checks,
            summary_metrics=summary_metrics,
            kpi_results=kpi_results,
            metadata=metadata or {}
        )

        # Store in history
        self.validation_history.append(report)
        if len(self.validation_history) > self.max_history:
            self.validation_history.pop(0)

        # Publish validation event
        event_publisher.publish_event('model_validation_completed', {
            'model_name': model_name,
            'overall_result': overall_result.value,
            'wmape': summary_metrics['wmape'],
            'passed_checks': len(report.get_passed_checks()),
            'failed_checks': len(report.get_failed_checks())
        })

        logger.info(f"Model validation completed: {model_name} - {overall_result.value}")
        return report

    def _determine_overall_result(self, checks: List[ValidationCheck]) -> ValidationResult:
        """Determine overall validation result"""
        failed_checks = [c for c in checks if c.result == ValidationResult.FAILED]
        warning_checks = [c for c in checks if c.result == ValidationResult.WARNING]

        # Critical failures (high severity)
        critical_failures = [c for c in failed_checks if c.severity == "high"]

        if critical_failures:
            return ValidationResult.FAILED
        elif failed_checks:
            return ValidationResult.WARNING
        elif len(warning_checks) > len(checks) / 2:  # More than half are warnings
            return ValidationResult.WARNING
        else:
            return ValidationResult.PASSED

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation history summary"""
        if not self.validation_history:
            return {'status': 'No validation history'}

        recent_validations = self.validation_history[-10:]  # Last 10 validations

        # Overall statistics
        passed_count = sum(1 for v in recent_validations if v.overall_result == ValidationResult.PASSED)
        warning_count = sum(1 for v in recent_validations if v.overall_result == ValidationResult.WARNING)
        failed_count = sum(1 for v in recent_validations if v.overall_result == ValidationResult.FAILED)

        # Average metrics
        avg_metrics = {}
        for metric in ['wmape', 'mape', 'mae', 'rmse']:
            values = [v.summary_metrics.get(metric, 0) for v in recent_validations]
            avg_metrics[f'avg_{metric}'] = np.mean(values)

        return {
            'total_validations': len(self.validation_history),
            'recent_validations': len(recent_validations),
            'passed_count': passed_count,
            'warning_count': warning_count,
            'failed_count': failed_count,
            'success_rate': passed_count / len(recent_validations),
            'average_metrics': avg_metrics,
            'last_validation': self.validation_history[-1].validation_timestamp.isoformat()
        }

# Global model validator instance
model_validator = ModelValidator()

def validate_model(model_name: str, actual: np.ndarray, predicted: np.ndarray,
                  model: Any = None, metadata: Optional[Dict[str, Any]] = None) -> ModelValidationReport:
    """Validate model (convenience function)"""
    return model_validator.validate_model(model_name, actual, predicted, model, metadata)


if __name__ == "__main__":
    # Demo model validation
    print("✅ Model Validation System Demo")
    print("=" * 50)

    # Generate sample data
    np.random.seed(42)
    actual = np.random.exponential(100, 500) + 50
    predicted = actual + np.random.normal(0, 10, 500)

    # Add some systematic issues for testing
    predicted[:50] = predicted[:50] * 1.3  # Overforecast bias in early samples
    predicted[100:110] = -5  # Some negative predictions
    predicted[200:205] = np.inf  # Some infinite predictions

    # Sample metadata
    metadata = {
        'model_type': 'lightgbm',
        'training_samples': 1000,
        'features': np.random.randn(500, 5),
        'forecast_horizon': 30,
        'seasonality_expected': True
    }

    print(f"\n🧮 Validating model with {len(actual)} predictions...")

    # Run validation
    report = model_validator.validate_model("demo_model", actual, predicted, metadata=metadata)

    # Display results
    print(f"\n📊 Overall Result: {report.overall_result.value.upper()}")
    print(f"Validation Timestamp: {report.validation_timestamp}")

    # Summary metrics
    print("\n📈 Summary Metrics:")
    for metric, value in report.summary_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    # Validation checks
    print("\n✅ Validation Checks:")
    for check in report.checks:
        status_icons = {
            ValidationResult.PASSED: "🟢",
            ValidationResult.WARNING: "🟡",
            ValidationResult.FAILED: "🔴"
        }
        icon = status_icons.get(check.result, "⚪")
        print(f"  {icon} {check.name}: {check.message}")

        if check.recommendation:
            print(f"    💡 Recommendation: {check.recommendation}")

    # Summary statistics
    print(f"\n📋 Check Summary:")
    print(f"  Total Checks: {len(report.checks)}")
    print(f"  Passed: {len(report.get_passed_checks())}")
    print(f"  Warnings: {len(report.get_warning_checks())}")
    print(f"  Failed: {len(report.get_failed_checks())}")

    # Validation history
    print(f"\n📚 Validation History Summary:")
    summary = model_validator.get_validation_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n✅ Model validation demo completed!")