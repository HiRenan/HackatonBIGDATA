#!/usr/bin/env python3
"""
Phase 6: KPI System
Enterprise-grade KPI tracking and validation system for retail forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import warnings

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import wmape, mape, mae, rmse
from architecture.observers import event_publisher
from config.phase6_config import get_config

logger = logging.getLogger(__name__)

class KPIStatus(Enum):
    """KPI status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"

class KPICategory(Enum):
    """KPI categories"""
    ACCURACY = "accuracy"
    BUSINESS = "business"
    TECHNICAL = "technical"
    OPERATIONAL = "operational"

@dataclass
class KPIThreshold:
    """KPI threshold configuration"""
    excellent: float
    good: float
    warning: float
    critical: float

    def evaluate(self, value: float, higher_is_better: bool = True) -> KPIStatus:
        """Evaluate value against thresholds"""
        if higher_is_better:
            if value >= self.excellent:
                return KPIStatus.EXCELLENT
            elif value >= self.good:
                return KPIStatus.GOOD
            elif value >= self.warning:
                return KPIStatus.WARNING
            else:
                return KPIStatus.CRITICAL
        else:
            if value <= self.excellent:
                return KPIStatus.EXCELLENT
            elif value <= self.good:
                return KPIStatus.GOOD
            elif value <= self.warning:
                return KPIStatus.WARNING
            else:
                return KPIStatus.CRITICAL

@dataclass
class KPIResult:
    """KPI calculation result"""
    name: str
    category: KPICategory
    value: float
    status: KPIStatus
    threshold: KPIThreshold
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    segment: Optional[str] = None
    target_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'category': self.category.value,
            'value': self.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'segment': self.segment,
            'target_value': self.target_value
        }

class BaseKPI:
    """Base class for KPI calculations"""

    def __init__(self, name: str, category: KPICategory, threshold: KPIThreshold,
                 higher_is_better: bool = True, description: str = ""):
        self.name = name
        self.category = category
        self.threshold = threshold
        self.higher_is_better = higher_is_better
        self.description = description

    def calculate(self, actual: np.ndarray, predicted: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None,
                 segment: Optional[str] = None) -> KPIResult:
        """Calculate KPI value"""
        try:
            value = self._calculate_metric(actual, predicted, metadata)
            status = self.threshold.evaluate(value, self.higher_is_better)

            return KPIResult(
                name=self.name,
                category=self.category,
                value=value,
                status=status,
                threshold=self.threshold,
                timestamp=datetime.now(),
                metadata=metadata or {},
                segment=segment
            )

        except Exception as e:
            logger.error(f"KPI calculation failed for {self.name}: {e}")
            return KPIResult(
                name=self.name,
                category=self.category,
                value=float('nan'),
                status=KPIStatus.CRITICAL,
                threshold=self.threshold,
                timestamp=datetime.now(),
                metadata={'error': str(e)},
                segment=segment
            )

    def _calculate_metric(self, actual: np.ndarray, predicted: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> float:
        """Override in subclasses"""
        raise NotImplementedError

class WMAPEKpi(BaseKPI):
    """WMAPE KPI - Primary forecasting accuracy metric"""

    def __init__(self):
        super().__init__(
            name="WMAPE",
            category=KPICategory.ACCURACY,
            threshold=KPIThreshold(excellent=0.10, good=0.15, warning=0.20, critical=0.30),
            higher_is_better=False,
            description="Weighted Mean Absolute Percentage Error"
        )

    def _calculate_metric(self, actual: np.ndarray, predicted: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> float:
        return wmape(actual, predicted)

class MAPEKPI(BaseKPI):
    """MAPE KPI - Secondary accuracy metric"""

    def __init__(self):
        super().__init__(
            name="MAPE",
            category=KPICategory.ACCURACY,
            threshold=KPIThreshold(excellent=0.12, good=0.18, warning=0.25, critical=0.40),
            higher_is_better=False,
            description="Mean Absolute Percentage Error"
        )

    def _calculate_metric(self, actual: np.ndarray, predicted: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> float:
        return mape(actual, predicted)

class DirectionalAccuracyKPI(BaseKPI):
    """Directional Accuracy KPI - Trend prediction accuracy"""

    def __init__(self):
        super().__init__(
            name="Directional_Accuracy",
            category=KPICategory.BUSINESS,
            threshold=KPIThreshold(excellent=0.80, good=0.70, warning=0.60, critical=0.50),
            higher_is_better=True,
            description="Percentage of correct trend direction predictions"
        )

    def _calculate_metric(self, actual: np.ndarray, predicted: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> float:
        if len(actual) < 2:
            return 0.5  # No trend to evaluate

        # Calculate actual and predicted changes
        actual_changes = np.diff(actual)
        predicted_changes = np.diff(predicted)

        # Calculate directional accuracy
        correct_directions = (actual_changes * predicted_changes) > 0
        return np.mean(correct_directions)

class ForecastBiasKPI(BaseKPI):
    """Forecast Bias KPI - Systematic over/under forecasting"""

    def __init__(self):
        super().__init__(
            name="Forecast_Bias",
            category=KPICategory.ACCURACY,
            threshold=KPIThreshold(excellent=0.02, good=0.05, warning=0.10, critical=0.20),
            higher_is_better=False,
            description="Systematic bias in forecasts (0 = unbiased)"
        )

    def _calculate_metric(self, actual: np.ndarray, predicted: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> float:
        bias = np.mean(predicted - actual) / np.mean(actual)
        return abs(bias)

class CoverageKPI(BaseKPI):
    """Prediction Interval Coverage KPI"""

    def __init__(self):
        super().__init__(
            name="Prediction_Coverage_95",
            category=KPICategory.ACCURACY,
            threshold=KPIThreshold(excellent=0.95, good=0.90, warning=0.85, critical=0.80),
            higher_is_better=True,
            description="95% prediction interval coverage"
        )

    def _calculate_metric(self, actual: np.ndarray, predicted: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> float:
        # Extract prediction intervals from metadata
        lower_bounds = metadata.get('lower_bounds', predicted - 1.96 * np.std(predicted))
        upper_bounds = metadata.get('upper_bounds', predicted + 1.96 * np.std(predicted))

        # Calculate coverage
        coverage = np.mean((actual >= lower_bounds) & (actual <= upper_bounds))
        return coverage

class DataQualityKPI(BaseKPI):
    """Data Quality KPI - Input data completeness and quality"""

    def __init__(self):
        super().__init__(
            name="Data_Quality_Score",
            category=KPICategory.TECHNICAL,
            threshold=KPIThreshold(excellent=0.98, good=0.95, warning=0.90, critical=0.85),
            higher_is_better=True,
            description="Overall data quality score"
        )

    def _calculate_metric(self, actual: np.ndarray, predicted: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> float:
        if metadata is None:
            return 1.0

        # Calculate various data quality metrics
        missing_rate = metadata.get('missing_rate', 0.0)
        outlier_rate = metadata.get('outlier_rate', 0.0)
        duplicate_rate = metadata.get('duplicate_rate', 0.0)

        # Composite quality score
        quality_score = 1.0 - (missing_rate + outlier_rate + duplicate_rate) / 3.0
        return max(0.0, quality_score)

class ModelPerformanceKPI(BaseKPI):
    """Model Performance KPI - Training and inference efficiency"""

    def __init__(self):
        super().__init__(
            name="Model_Performance_Score",
            category=KPICategory.TECHNICAL,
            threshold=KPIThreshold(excellent=0.90, good=0.80, warning=0.70, critical=0.50),
            higher_is_better=True,
            description="Overall model performance score"
        )

    def _calculate_metric(self, actual: np.ndarray, predicted: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> float:
        if metadata is None:
            return 0.8  # Default score

        # Training time efficiency (inverse - lower is better)
        training_time = metadata.get('training_time_minutes', 60)
        training_score = max(0, 1 - (training_time / 120))  # 2 hours = score 0

        # Inference speed
        inference_time = metadata.get('inference_time_ms', 100)
        inference_score = max(0, 1 - (inference_time / 1000))  # 1 second = score 0

        # Memory efficiency
        memory_usage = metadata.get('memory_usage_gb', 8)
        memory_score = max(0, 1 - (memory_usage / 16))  # 16GB = score 0

        # Composite performance score
        return (training_score + inference_score + memory_score) / 3.0

class BusinessValueKPI(BaseKPI):
    """Business Value KPI - Revenue impact and business metrics"""

    def __init__(self):
        super().__init__(
            name="Business_Value_Score",
            category=KPICategory.BUSINESS,
            threshold=KPIThreshold(excellent=0.85, good=0.75, warning=0.65, critical=0.50),
            higher_is_better=True,
            description="Estimated business value generated"
        )

    def _calculate_metric(self, actual: np.ndarray, predicted: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> float:
        if metadata is None:
            return 0.7  # Default moderate value

        # Inventory optimization impact
        inventory_impact = metadata.get('inventory_optimization', 0.1)

        # Stockout reduction
        stockout_reduction = metadata.get('stockout_reduction', 0.05)

        # Overstock reduction
        overstock_reduction = metadata.get('overstock_reduction', 0.08)

        # Customer satisfaction impact
        satisfaction_impact = metadata.get('customer_satisfaction_improvement', 0.02)

        # Composite business value
        return inventory_impact + stockout_reduction + overstock_reduction + satisfaction_impact

class KPIManager:
    """Manages KPI calculations and tracking"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().dict()
        self.kpis: Dict[str, BaseKPI] = {}
        self.results_history: List[Dict[str, KPIResult]] = []
        self.max_history = 1000

        self._initialize_kpis()

    def _initialize_kpis(self):
        """Initialize standard KPIs"""
        self.kpis = {
            'wmape': WMAPEKpi(),
            'mape': MAPEKPI(),
            'directional_accuracy': DirectionalAccuracyKPI(),
            'forecast_bias': ForecastBiasKPI(),
            'prediction_coverage': CoverageKPI(),
            'data_quality': DataQualityKPI(),
            'model_performance': ModelPerformanceKPI(),
            'business_value': BusinessValueKPI()
        }

    def add_kpi(self, kpi: BaseKPI):
        """Add custom KPI"""
        self.kpis[kpi.name.lower()] = kpi

    def calculate_all_kpis(self, actual: np.ndarray, predicted: np.ndarray,
                          metadata: Optional[Dict[str, Any]] = None,
                          segment: Optional[str] = None) -> Dict[str, KPIResult]:
        """Calculate all registered KPIs"""
        results = {}

        for kpi_name, kpi in self.kpis.items():
            try:
                result = kpi.calculate(actual, predicted, metadata, segment)
                results[kpi_name] = result

                # Publish KPI result event
                event_publisher.publish_event('kpi_calculated', {
                    'kpi_name': kpi_name,
                    'value': result.value,
                    'status': result.status.value,
                    'segment': segment
                })

            except Exception as e:
                logger.error(f"Failed to calculate KPI {kpi_name}: {e}")

        # Store in history
        self.results_history.append(results)
        if len(self.results_history) > self.max_history:
            self.results_history.pop(0)

        return results

    def calculate_segmented_kpis(self, data_segments: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, KPIResult]]:
        """Calculate KPIs for different data segments"""
        segmented_results = {}

        for segment_name, (actual, predicted) in data_segments.items():
            segment_metadata = metadata.copy() if metadata else {}
            segment_metadata['segment'] = segment_name

            segmented_results[segment_name] = self.calculate_all_kpis(
                actual, predicted, segment_metadata, segment_name
            )

        return segmented_results

    def get_kpi_summary(self) -> Dict[str, Any]:
        """Get comprehensive KPI summary"""
        if not self.results_history:
            return {'status': 'No KPI results available'}

        latest_results = self.results_history[-1]

        # Status distribution
        status_counts = {}
        for status in KPIStatus:
            status_counts[status.value] = sum(
                1 for result in latest_results.values()
                if result.status == status
            )

        # Category breakdown
        category_scores = {}
        for category in KPICategory:
            category_results = [
                result for result in latest_results.values()
                if result.category == category
            ]
            if category_results:
                avg_score = np.mean([r.value for r in category_results if not np.isnan(r.value)])
                category_scores[category.value] = avg_score

        # Overall health score (0-100)
        excellent_count = status_counts.get('excellent', 0)
        good_count = status_counts.get('good', 0)
        total_kpis = len(latest_results)

        health_score = ((excellent_count * 100 + good_count * 75) / total_kpis) if total_kpis > 0 else 0

        # Critical issues
        critical_kpis = [
            {'name': result.name, 'value': result.value}
            for result in latest_results.values()
            if result.status == KPIStatus.CRITICAL
        ]

        return {
            'overall_health_score': health_score,
            'total_kpis': total_kpis,
            'status_distribution': status_counts,
            'category_scores': category_scores,
            'critical_issues': critical_kpis,
            'last_calculation': max(r.timestamp for r in latest_results.values()).isoformat(),
            'trending_kpis': self._calculate_trending_metrics()
        }

    def _calculate_trending_metrics(self) -> Dict[str, str]:
        """Calculate trending indicators for key KPIs"""
        if len(self.results_history) < 2:
            return {}

        current_results = self.results_history[-1]
        previous_results = self.results_history[-2]

        trends = {}

        for kpi_name in ['wmape', 'directional_accuracy', 'business_value']:
            if kpi_name in current_results and kpi_name in previous_results:
                current_val = current_results[kpi_name].value
                previous_val = previous_results[kpi_name].value

                if np.isnan(current_val) or np.isnan(previous_val):
                    trends[kpi_name] = 'stable'
                else:
                    change = current_val - previous_val
                    if abs(change) < 0.001:  # Minimal change threshold
                        trends[kpi_name] = 'stable'
                    elif change > 0:
                        if self.kpis[kpi_name].higher_is_better:
                            trends[kpi_name] = 'improving'
                        else:
                            trends[kpi_name] = 'degrading'
                    else:
                        if self.kpis[kpi_name].higher_is_better:
                            trends[kpi_name] = 'degrading'
                        else:
                            trends[kpi_name] = 'improving'

        return trends

    def generate_kpi_report(self, format: str = 'dict') -> Any:
        """Generate comprehensive KPI report"""
        summary = self.get_kpi_summary()

        if not self.results_history:
            return summary

        latest_results = self.results_history[-1]

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detailed_results': {
                name: result.to_dict() for name, result in latest_results.items()
            },
            'recommendations': self._generate_recommendations(latest_results)
        }

        if format == 'json':
            return json.dumps(report, indent=2)
        elif format == 'dataframe':
            df_data = []
            for name, result in latest_results.items():
                df_data.append({
                    'KPI': name,
                    'Category': result.category.value,
                    'Value': result.value,
                    'Status': result.status.value,
                    'Segment': result.segment
                })
            return pd.DataFrame(df_data)

        return report

    def _generate_recommendations(self, results: Dict[str, KPIResult]) -> List[str]:
        """Generate improvement recommendations based on KPI results"""
        recommendations = []

        # Check WMAPE
        if 'wmape' in results:
            wmape_result = results['wmape']
            if wmape_result.status == KPIStatus.CRITICAL:
                recommendations.append("CRITICAL: WMAPE is very high. Consider feature engineering or model ensemble.")
            elif wmape_result.status == KPIStatus.WARNING:
                recommendations.append("WARNING: WMAPE exceeds target. Review hyperparameters and validation strategy.")

        # Check directional accuracy
        if 'directional_accuracy' in results:
            dir_result = results['directional_accuracy']
            if dir_result.status == KPIStatus.WARNING:
                recommendations.append("Directional accuracy is low. Consider trend-focused features or models.")

        # Check data quality
        if 'data_quality' in results:
            quality_result = results['data_quality']
            if quality_result.status != KPIStatus.EXCELLENT:
                recommendations.append("Data quality issues detected. Implement data cleaning and validation.")

        # Check model performance
        if 'model_performance' in results:
            perf_result = results['model_performance']
            if perf_result.status == KPIStatus.WARNING:
                recommendations.append("Model performance is suboptimal. Consider model optimization or scaling.")

        # Check business value
        if 'business_value' in results:
            bv_result = results['business_value']
            if bv_result.status == KPIStatus.WARNING:
                recommendations.append("Business value is below target. Review model impact on business KPIs.")

        return recommendations

# Global KPI manager instance
kpi_manager = KPIManager()

def calculate_kpis(actual: np.ndarray, predicted: np.ndarray,
                  metadata: Optional[Dict[str, Any]] = None,
                  segment: Optional[str] = None) -> Dict[str, KPIResult]:
    """Calculate all KPIs (convenience function)"""
    return kpi_manager.calculate_all_kpis(actual, predicted, metadata, segment)

def get_kpi_summary() -> Dict[str, Any]:
    """Get current KPI summary"""
    return kpi_manager.get_kpi_summary()

def generate_kpi_report() -> Dict[str, Any]:
    """Generate comprehensive KPI report"""
    return kpi_manager.generate_kpi_report()


if __name__ == "__main__":
    # Demo KPI system
    print("📊 KPI System Demo")
    print("=" * 50)

    # Generate sample data
    np.random.seed(42)
    actual = np.random.exponential(100, 1000) + 50
    predicted = actual + np.random.normal(0, 10, 1000)

    # Add some systematic bias and errors
    predicted = predicted * 1.05  # 5% overforecast bias
    predicted[100:150] = predicted[100:150] * 1.5  # Some outlier predictions

    # Sample metadata
    metadata = {
        'missing_rate': 0.02,
        'outlier_rate': 0.01,
        'duplicate_rate': 0.001,
        'training_time_minutes': 45,
        'inference_time_ms': 50,
        'memory_usage_gb': 6,
        'lower_bounds': predicted - 15,
        'upper_bounds': predicted + 15,
        'inventory_optimization': 0.12,
        'stockout_reduction': 0.08,
        'overstock_reduction': 0.06,
        'customer_satisfaction_improvement': 0.03
    }

    print("\n🧮 Calculating KPIs...")
    results = kpi_manager.calculate_all_kpis(actual, predicted, metadata)

    # Display results
    print("\n📈 KPI Results:")
    for kpi_name, result in results.items():
        status_icons = {
            KPIStatus.EXCELLENT: "🟢",
            KPIStatus.GOOD: "🟡",
            KPIStatus.WARNING: "🟠",
            KPIStatus.CRITICAL: "🔴"
        }
        icon = status_icons.get(result.status, "⚪")
        print(f"  {icon} {result.name}: {result.value:.3f} ({result.status.value})")

    # Generate summary
    print("\n📊 KPI Summary:")
    summary = kpi_manager.get_kpi_summary()
    print(f"  Overall Health Score: {summary['overall_health_score']:.1f}/100")
    print(f"  Total KPIs: {summary['total_kpis']}")

    # Status distribution
    print("  Status Distribution:")
    for status, count in summary['status_distribution'].items():
        print(f"    {status}: {count}")

    # Category scores
    print("  Category Scores:")
    for category, score in summary['category_scores'].items():
        print(f"    {category}: {score:.3f}")

    # Recommendations
    report = kpi_manager.generate_kpi_report()
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

    print("\n✅ KPI system demo completed!")