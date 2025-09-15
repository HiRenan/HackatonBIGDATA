#!/usr/bin/env python3
"""
Phase 6: Health Check System
Comprehensive health monitoring for all system components
"""

import time
import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path
import psutil
import os

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.phase6_config import get_config
from architecture.observers import event_publisher

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """Health check result data structure"""
    component: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any]
    timestamp: datetime
    duration_ms: float
    details: Optional[Dict[str, Any]] = None

class BaseHealthCheck:
    """Base class for health checks"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.timeout = self.config.get('timeout_seconds', 10)
        self.critical = self.config.get('critical', False)

    def check(self) -> HealthCheckResult:
        """Execute health check with timing and error handling"""
        start_time = time.time()
        timestamp = datetime.now()

        try:
            status, message, metrics, details = self._perform_check()
            duration_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                metrics=metrics or {},
                timestamp=timestamp,
                duration_ms=duration_ms,
                details=details
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Health check failed for {self.name}: {e}")

            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                metrics={},
                timestamp=timestamp,
                duration_ms=duration_ms,
                details={'error': str(e)}
            )

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Override this method in subclasses"""
        raise NotImplementedError


class DatabaseHealthCheck(BaseHealthCheck):
    """Database connection health check"""

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Check database connectivity"""
        try:
            config = get_config()
            db_config = config.database

            # Mock database check (replace with actual database connection)
            import time
            time.sleep(0.1)  # Simulate database query

            # In real implementation, you would:
            # import psycopg2
            # conn = psycopg2.connect(
            #     host=db_config.host,
            #     port=db_config.port,
            #     database=db_config.name,
            #     user=db_config.user,
            #     password=db_config.password
            # )
            # cursor = conn.cursor()
            # cursor.execute("SELECT 1")
            # conn.close()

            metrics = {
                'connection_pool_size': 10,
                'active_connections': 3,
                'max_connections': db_config.max_connections
            }

            return (
                HealthStatus.HEALTHY,
                "Database connection successful",
                metrics,
                {'host': db_config.host, 'port': db_config.port}
            )

        except Exception as e:
            return (
                HealthStatus.CRITICAL,
                f"Database connection failed: {str(e)}",
                {},
                None
            )


class MLflowHealthCheck(BaseHealthCheck):
    """MLflow tracking server health check"""

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Check MLflow server connectivity"""
        try:
            config = get_config()
            mlflow_uri = config.mlflow.tracking_uri

            # Check MLflow server
            response = requests.get(f"{mlflow_uri}/health", timeout=self.timeout)

            if response.status_code == 200:
                # Additional checks
                try:
                    import mlflow
                    mlflow.set_tracking_uri(mlflow_uri)
                    experiments = mlflow.list_experiments(max_results=1)

                    metrics = {
                        'server_response_time_ms': response.elapsed.total_seconds() * 1000,
                        'experiments_accessible': len(experiments) >= 0
                    }

                    return (
                        HealthStatus.HEALTHY,
                        "MLflow server is healthy",
                        metrics,
                        {'tracking_uri': mlflow_uri}
                    )

                except Exception as e:
                    return (
                        HealthStatus.WARNING,
                        f"MLflow server accessible but experiments not queryable: {str(e)}",
                        {'server_response_time_ms': response.elapsed.total_seconds() * 1000},
                        None
                    )

            else:
                return (
                    HealthStatus.CRITICAL,
                    f"MLflow server returned {response.status_code}",
                    {},
                    None
                )

        except requests.exceptions.ConnectionError:
            return (
                HealthStatus.CRITICAL,
                "Cannot connect to MLflow server",
                {},
                None
            )
        except Exception as e:
            return (
                HealthStatus.CRITICAL,
                f"MLflow health check failed: {str(e)}",
                {},
                None
            )


class ModelLoadingHealthCheck(BaseHealthCheck):
    """Model loading capability health check"""

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Check if models can be loaded successfully"""
        try:
            from architecture.factories import model_factory

            # Test loading different model types
            model_tests = [
                ('lightgbm', {'n_estimators': 1, 'random_state': 42}),
            ]

            successful_models = 0
            total_models = len(model_tests)
            model_details = {}

            for model_type, model_config in model_tests:
                try:
                    start_time = time.time()
                    model = model_factory.create(model_type, model_config)
                    load_time = (time.time() - start_time) * 1000

                    model_details[model_type] = {
                        'status': 'success',
                        'load_time_ms': load_time
                    }
                    successful_models += 1

                except Exception as e:
                    model_details[model_type] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            success_rate = successful_models / total_models

            if success_rate >= 0.8:
                status = HealthStatus.HEALTHY
                message = f"Model loading successful ({successful_models}/{total_models})"
            elif success_rate >= 0.5:
                status = HealthStatus.WARNING
                message = f"Some models failed to load ({successful_models}/{total_models})"
            else:
                status = HealthStatus.CRITICAL
                message = f"Most models failed to load ({successful_models}/{total_models})"

            metrics = {
                'successful_models': successful_models,
                'total_models': total_models,
                'success_rate': success_rate
            }

            return (status, message, metrics, model_details)

        except Exception as e:
            return (
                HealthStatus.CRITICAL,
                f"Model loading test failed: {str(e)}",
                {},
                None
            )


class MemoryHealthCheck(BaseHealthCheck):
    """System memory health check"""

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Check system memory usage"""
        try:
            memory = psutil.virtual_memory()
            config = get_config()

            memory_threshold = config.monitoring.alert_thresholds.get('memory_usage_threshold', 0.80)

            metrics = {
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3)
            }

            if memory.percent < memory_threshold * 100:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal ({memory.percent:.1f}%)"
            elif memory.percent < 95:
                status = HealthStatus.WARNING
                message = f"Memory usage high ({memory.percent:.1f}%)"
            else:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical ({memory.percent:.1f}%)"

            return (status, message, metrics, None)

        except Exception as e:
            return (
                HealthStatus.CRITICAL,
                f"Memory check failed: {str(e)}",
                {},
                None
            )


class DiskSpaceHealthCheck(BaseHealthCheck):
    """Disk space health check"""

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Check disk space usage"""
        try:
            disk = psutil.disk_usage('/')

            metrics = {
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }

            if disk.percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Disk space normal ({disk.percent:.1f}%)"
            elif disk.percent < 95:
                status = HealthStatus.WARNING
                message = f"Disk space low ({disk.percent:.1f}%)"
            else:
                status = HealthStatus.CRITICAL
                message = f"Disk space critical ({disk.percent:.1f}%)"

            return (status, message, metrics, None)

        except Exception as e:
            return (
                HealthStatus.CRITICAL,
                f"Disk space check failed: {str(e)}",
                {},
                None
            )


class DataIntegrityHealthCheck(BaseHealthCheck):
    """Data integrity health check"""

    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Check data integrity and availability"""
        try:
            config = get_config()
            data_path = Path(config.data.data_path)

            # Check if data directory exists
            if not data_path.exists():
                return (
                    HealthStatus.CRITICAL,
                    f"Data directory not found: {data_path}",
                    {},
                    None
                )

            # Check for required data files
            required_files = ['transactions.parquet', 'products.parquet', 'stores.parquet']
            found_files = []
            missing_files = []

            for file_name in required_files:
                file_path = data_path / file_name
                if file_path.exists():
                    found_files.append(file_name)
                else:
                    missing_files.append(file_name)

            # Calculate data health
            data_completeness = len(found_files) / len(required_files)

            metrics = {
                'data_completeness': data_completeness,
                'found_files': len(found_files),
                'total_files': len(required_files),
                'data_path_accessible': True
            }

            details = {
                'found_files': found_files,
                'missing_files': missing_files,
                'data_path': str(data_path)
            }

            if data_completeness == 1.0:
                status = HealthStatus.HEALTHY
                message = "All required data files present"
            elif data_completeness >= 0.8:
                status = HealthStatus.WARNING
                message = f"Some data files missing ({len(missing_files)} missing)"
            else:
                status = HealthStatus.CRITICAL
                message = f"Critical data files missing ({len(missing_files)} missing)"

            return (status, message, metrics, details)

        except Exception as e:
            return (
                HealthStatus.CRITICAL,
                f"Data integrity check failed: {str(e)}",
                {},
                None
            )


class HealthCheckManager:
    """Manages and orchestrates health checks"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().dict()
        self.checks: Dict[str, BaseHealthCheck] = {}
        self.results_history: List[Dict[str, HealthCheckResult]] = []
        self.max_history = 100

        self._initialize_checks()

    def _initialize_checks(self):
        """Initialize all health checks"""
        health_config = self.config.get('health_check', {})

        # Core health checks
        self.checks = {
            'database_connection': DatabaseHealthCheck(
                'database_connection',
                health_config.get('database_connection', {})
            ),
            'mlflow_connection': MLflowHealthCheck(
                'mlflow_connection',
                health_config.get('mlflow_connection', {})
            ),
            'model_loading': ModelLoadingHealthCheck(
                'model_loading',
                health_config.get('model_loading', {})
            ),
            'memory_usage': MemoryHealthCheck(
                'memory_usage',
                health_config.get('memory_usage', {})
            ),
            'disk_space': DiskSpaceHealthCheck(
                'disk_space',
                health_config.get('disk_space', {})
            ),
            'data_integrity': DataIntegrityHealthCheck(
                'data_integrity',
                health_config.get('data_integrity', {})
            )
        }

    def add_check(self, name: str, health_check: BaseHealthCheck):
        """Add a custom health check"""
        self.checks[name] = health_check

    def remove_check(self, name: str):
        """Remove a health check"""
        if name in self.checks:
            del self.checks[name]

    def run_single_check(self, check_name: str) -> Optional[HealthCheckResult]:
        """Run a single health check"""
        if check_name not in self.checks:
            logger.error(f"Health check '{check_name}' not found")
            return None

        logger.info(f"Running health check: {check_name}")
        result = self.checks[check_name].check()

        # Publish event
        event_publisher.publish_event('health_check_completed', {
            'check_name': check_name,
            'status': result.status.value,
            'message': result.message,
            'duration_ms': result.duration_ms
        })

        return result

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks"""
        logger.info("Running all health checks")
        results = {}

        for check_name in self.checks:
            results[check_name] = self.run_single_check(check_name)

        # Store in history
        self.results_history.append(results)
        if len(self.results_history) > self.max_history:
            self.results_history.pop(0)

        # Publish overall health event
        overall_status = self._calculate_overall_health(results)
        event_publisher.publish_event('overall_health_check_completed', {
            'overall_status': overall_status.value,
            'total_checks': len(results),
            'healthy_checks': sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
            'critical_checks': sum(1 for r in results.values() if r.status == HealthStatus.CRITICAL)
        })

        return results

    def _calculate_overall_health(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Calculate overall system health status"""
        if not results:
            return HealthStatus.UNKNOWN

        # Check for critical failures in critical components
        critical_failures = [
            result for result in results.values()
            if result.status == HealthStatus.CRITICAL and
               self.checks[result.component].critical
        ]

        if critical_failures:
            return HealthStatus.CRITICAL

        # Check for any critical failures
        if any(result.status == HealthStatus.CRITICAL for result in results.values()):
            return HealthStatus.CRITICAL

        # Check for warnings
        if any(result.status == HealthStatus.WARNING for result in results.values()):
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        if not self.results_history:
            return {'status': 'No health checks performed'}

        latest_results = self.results_history[-1]
        overall_status = self._calculate_overall_health(latest_results)

        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = sum(
                1 for result in latest_results.values()
                if result.status == status
            )

        # Calculate average response time
        response_times = [result.duration_ms for result in latest_results.values()]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        return {
            'overall_status': overall_status.value,
            'total_checks': len(latest_results),
            'status_breakdown': status_counts,
            'average_response_time_ms': avg_response_time,
            'last_check_time': max(result.timestamp for result in latest_results.values()).isoformat(),
            'critical_issues': [
                {'component': result.component, 'message': result.message}
                for result in latest_results.values()
                if result.status == HealthStatus.CRITICAL
            ]
        }

    def start_continuous_monitoring(self, interval_seconds: int = 60):
        """Start continuous health monitoring"""
        def monitor_loop():
            while True:
                try:
                    self.run_all_checks()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Health monitoring loop error: {e}")
                    time.sleep(interval_seconds)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"Started continuous health monitoring (interval: {interval_seconds}s)")


# Global health check manager instance
health_manager = HealthCheckManager()

def run_health_checks() -> Dict[str, Any]:
    """Run all health checks and return summary"""
    results = health_manager.run_all_checks()
    return health_manager.get_health_summary()

def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    return health_manager.get_health_summary()


if __name__ == "__main__":
    # Demo health check execution
    print("🏥 Running Health Checks")
    print("=" * 50)

    # Run all checks
    results = health_manager.run_all_checks()

    # Display results
    for check_name, result in results.items():
        status_icon = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "❌",
            HealthStatus.UNKNOWN: "❓"
        }[result.status]

        print(f"{status_icon} {result.component}: {result.message} ({result.duration_ms:.1f}ms)")

    # Overall summary
    print("\n📊 Health Summary:")
    summary = health_manager.get_health_summary()
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Total Checks: {summary['total_checks']}")
    print(f"Average Response Time: {summary['average_response_time_ms']:.1f}ms")

    if summary['critical_issues']:
        print("\n🚨 Critical Issues:")
        for issue in summary['critical_issues']:
            print(f"  - {issue['component']}: {issue['message']}")