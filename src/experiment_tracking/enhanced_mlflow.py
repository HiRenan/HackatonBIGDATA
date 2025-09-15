#!/usr/bin/env python3
"""
Phase 6: Enhanced MLflow Integration
Advanced MLflow tracking with auto-logging and comprehensive monitoring
"""

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.prophet
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
import logging
import json
import pickle
import warnings
from datetime import datetime
import time
import psutil
from functools import wraps
from contextlib import contextmanager
import tempfile

logger = logging.getLogger(__name__)

class EnhancedMLflowTracker:
    """Enhanced MLflow tracker with auto-logging and advanced features"""

    def __init__(self,
                 tracking_uri: Optional[str] = None,
                 experiment_name: str = "hackathon_forecast_2025",
                 auto_log: bool = True):
        """
        Initialize enhanced MLflow tracker

        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Experiment name
            auto_log: Enable auto-logging for supported libraries
        """
        self.tracking_uri = tracking_uri or "http://localhost:5000"
        self.experiment_name = experiment_name
        self.auto_log = auto_log

        # Set up MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

        # Create or get experiment
        self.experiment = self._get_or_create_experiment()

        # Enable auto-logging
        if self.auto_log:
            self._enable_auto_logging()

        logger.info(f"MLflow tracker initialized - Experiment: {experiment_name}")

    def _get_or_create_experiment(self):
        """Get or create MLflow experiment"""
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(
                    self.experiment_name,
                    tags={
                        "project": "hackathon_forecast_2025",
                        "created_at": datetime.now().isoformat(),
                        "phase": "6"
                    }
                )
                experiment = self.client.get_experiment(experiment_id)
            return experiment
        except Exception as e:
            logger.error(f"Failed to create/get experiment: {str(e)}")
            raise

    def _enable_auto_logging(self):
        """Enable auto-logging for supported libraries"""
        try:
            # Enable auto-logging for various libraries
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                disable=False,
                exclusive=False,
                disable_for_unsupported_versions=False,
                silent=True
            )

            mlflow.lightgbm.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                disable=False,
                exclusive=False,
                disable_for_unsupported_versions=False,
                silent=True
            )

            logger.info("Auto-logging enabled for scikit-learn and LightGBM")

        except Exception as e:
            logger.warning(f"Failed to enable auto-logging: {str(e)}")

    @contextmanager
    def start_run(self,
                  run_name: Optional[str] = None,
                  nested: bool = False,
                  tags: Optional[Dict[str, str]] = None):
        """
        Context manager for MLflow runs with enhanced tracking

        Args:
            run_name: Run name
            nested: Whether this is a nested run
            tags: Additional tags

        Example:
            with tracker.start_run("model_training") as run:
                # Training code here
                pass
        """
        # Prepare tags
        run_tags = {
            "phase": "6",
            "started_at": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "hostname": os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        }
        if tags:
            run_tags.update(tags)

        # System information
        system_info = self._get_system_info()
        run_tags.update({f"system_{k}": str(v) for k, v in system_info.items()})

        try:
            with mlflow.start_run(run_name=run_name, nested=nested, tags=run_tags) as run:
                # Log system information
                mlflow.log_params(system_info)

                # Start performance monitoring
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024

                yield run

                # Log run duration and memory usage
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                mlflow.log_metrics({
                    "run_duration_seconds": end_time - start_time,
                    "memory_usage_mb": end_memory,
                    "memory_delta_mb": end_memory - start_memory
                })

                # Mark as successful
                self.client.set_tag(run.info.run_id, "status", "success")

        except Exception as e:
            # Mark as failed and log error
            if mlflow.active_run():
                self.client.set_tag(mlflow.active_run().info.run_id, "status", "failed")
                self.client.set_tag(mlflow.active_run().info.run_id, "error", str(e))
                mlflow.log_param("error_type", type(e).__name__)
                mlflow.log_param("error_message", str(e))

            logger.error(f"MLflow run failed: {str(e)}")
            raise

    def log_dataset_info(self, dataset: pd.DataFrame, dataset_name: str):
        """Log comprehensive dataset information"""
        try:
            info = {
                f"{dataset_name}_shape_rows": dataset.shape[0],
                f"{dataset_name}_shape_cols": dataset.shape[1],
                f"{dataset_name}_memory_mb": dataset.memory_usage(deep=True).sum() / 1024 / 1024,
                f"{dataset_name}_null_percentage": (dataset.isnull().sum().sum() / dataset.size) * 100
            }

            mlflow.log_params(info)

            # Log column information
            column_info = {
                "columns": list(dataset.columns),
                "dtypes": dataset.dtypes.astype(str).to_dict(),
                "null_counts": dataset.isnull().sum().to_dict()
            }

            # Save as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(column_info, f, indent=2, default=str)
                f.flush()
                mlflow.log_artifact(f.name, f"{dataset_name}_info.json")

            logger.info(f"Logged dataset info for {dataset_name}")

        except Exception as e:
            logger.warning(f"Failed to log dataset info: {str(e)}")

    def log_feature_importance(self, model, feature_names: List[str], top_n: int = 20):
        """Log feature importance with visualization"""
        try:
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                logger.warning("Model doesn't have feature importance")
                return

            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)

            # Log as table
            mlflow.log_table(importance_df, "feature_importance.json")

            # Log top features as params
            top_features = importance_df.head(10)
            for i, (_, row) in enumerate(top_features.iterrows()):
                mlflow.log_param(f"top_feature_{i+1}", row['feature'])
                mlflow.log_metric(f"top_feature_{i+1}_importance", row['importance'])

            logger.info(f"Logged feature importance for top {top_n} features")

        except Exception as e:
            logger.warning(f"Failed to log feature importance: {str(e)}")

    def log_cross_validation_results(self, cv_scores: Dict[str, List[float]]):
        """Log cross-validation results with statistics"""
        try:
            for metric, scores in cv_scores.items():
                scores_array = np.array(scores)

                # Log statistics
                mlflow.log_metrics({
                    f"cv_{metric}_mean": scores_array.mean(),
                    f"cv_{metric}_std": scores_array.std(),
                    f"cv_{metric}_min": scores_array.min(),
                    f"cv_{metric}_max": scores_array.max(),
                    f"cv_{metric}_median": np.median(scores_array)
                })

                # Log individual fold scores
                for fold, score in enumerate(scores):
                    mlflow.log_metric(f"cv_{metric}_fold_{fold}", score)

            logger.info("Logged cross-validation results")

        except Exception as e:
            logger.warning(f"Failed to log CV results: {str(e)}")

    def log_model_with_metadata(self,
                               model,
                               model_name: str,
                               model_type: str,
                               metrics: Dict[str, float],
                               hyperparameters: Dict[str, Any],
                               feature_names: Optional[List[str]] = None):
        """Log model with comprehensive metadata"""
        try:
            # Log hyperparameters
            mlflow.log_params(hyperparameters)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model based on type
            if model_type.lower() == 'lightgbm':
                mlflow.lightgbm.log_model(model, model_name)
            elif model_type.lower() in ['sklearn', 'scikit-learn']:
                mlflow.sklearn.log_model(model, model_name)
            else:
                # Use pickle for other models
                mlflow.log_artifact(self._pickle_model(model), f"{model_name}.pkl")

            # Log feature importance if available
            if feature_names:
                self.log_feature_importance(model, feature_names)

            # Log model metadata
            metadata = {
                "model_type": model_type,
                "model_class": model.__class__.__name__,
                "logged_at": datetime.now().isoformat(),
                "n_features": len(feature_names) if feature_names else "unknown"
            }

            for key, value in metadata.items():
                mlflow.log_param(f"model_{key}", value)

            logger.info(f"Logged {model_type} model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")

    def log_prediction_results(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              dataset_name: str = "test"):
        """Log prediction results with comprehensive evaluation"""
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            # Calculate WMAPE (Weighted Mean Absolute Percentage Error)
            wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

            metrics = {
                f"{dataset_name}_mae": mae,
                f"{dataset_name}_rmse": rmse,
                f"{dataset_name}_r2": r2,
                f"{dataset_name}_wmape": wmape
            }

            mlflow.log_metrics(metrics)

            # Log prediction statistics
            pred_stats = {
                f"{dataset_name}_pred_mean": np.mean(y_pred),
                f"{dataset_name}_pred_std": np.std(y_pred),
                f"{dataset_name}_pred_min": np.min(y_pred),
                f"{dataset_name}_pred_max": np.max(y_pred),
                f"{dataset_name}_actual_mean": np.mean(y_true),
                f"{dataset_name}_actual_std": np.std(y_true)
            }

            mlflow.log_metrics(pred_stats)

            # Save predictions as artifact
            predictions_df = pd.DataFrame({
                'actual': y_true,
                'predicted': y_pred,
                'residual': y_true - y_pred,
                'abs_residual': np.abs(y_true - y_pred)
            })

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                predictions_df.to_csv(f.name, index=False)
                mlflow.log_artifact(f.name, f"{dataset_name}_predictions.csv")

            logger.info(f"Logged prediction results for {dataset_name}")

        except Exception as e:
            logger.warning(f"Failed to log prediction results: {str(e)}")

    def log_data_quality_report(self, data_quality_report: Dict[str, Any]):
        """Log data quality assessment results"""
        try:
            # Log quality metrics
            quality_metrics = {
                "data_quality_score": data_quality_report.get('overall_score', 0),
                "missing_data_percentage": data_quality_report.get('missing_percentage', 0),
                "duplicate_percentage": data_quality_report.get('duplicate_percentage', 0),
                "outlier_percentage": data_quality_report.get('outlier_percentage', 0)
            }

            mlflow.log_metrics(quality_metrics)

            # Save full report as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data_quality_report, f, indent=2, default=str)
                f.flush()
                mlflow.log_artifact(f.name, "data_quality_report.json")

            logger.info("Logged data quality report")

        except Exception as e:
            logger.warning(f"Failed to log data quality report: {str(e)}")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for tracking"""
        import sys
        import os

        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "hostname": os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        }

    def _pickle_model(self, model) -> str:
        """Pickle model and return temp file path"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(model, f)
            return f.name

    def get_best_run(self, metric: str, ascending: bool = True) -> Optional[mlflow.entities.Run]:
        """Get the best run based on a metric"""
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
            )

            if runs:
                return runs[0]
            return None

        except Exception as e:
            logger.error(f"Failed to get best run: {str(e)}")
            return None

    def get_run_comparison(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs"""
        try:
            runs_data = []

            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time
                }

                # Add metrics
                run_data.update(run.data.metrics)

                # Add key parameters
                run_data.update(run.data.params)

                runs_data.append(run_data)

            return pd.DataFrame(runs_data)

        except Exception as e:
            logger.error(f"Failed to compare runs: {str(e)}")
            return pd.DataFrame()

def auto_log_experiment(experiment_name: str,
                       run_name: Optional[str] = None,
                       tags: Optional[Dict[str, str]] = None):
    """Decorator for automatic experiment logging"""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = EnhancedMLflowTracker(experiment_name=experiment_name)

            with tracker.start_run(run_name=run_name, tags=tags):
                # Log function arguments
                if args:
                    mlflow.log_param("args_count", len(args))
                if kwargs:
                    safe_kwargs = {k: str(v) for k, v in kwargs.items()
                                 if isinstance(v, (str, int, float, bool))}
                    mlflow.log_params(safe_kwargs)

                # Execute function
                result = func(*args, **kwargs)

                # Log result type
                mlflow.log_param("result_type", type(result).__name__)

                return result

        return wrapper
    return decorator

if __name__ == "__main__":
    # Demo usage
    print("🔬 Enhanced MLflow Integration Demo")
    print("=" * 50)

    # Initialize tracker
    tracker = EnhancedMLflowTracker()
    print("✅ MLflow tracker initialized")

    # Demo run
    with tracker.start_run("demo_run", tags={"demo": "true"}):
        # Log some demo data
        mlflow.log_param("demo_param", "demo_value")
        mlflow.log_metric("demo_metric", 0.95)

        print("✅ Demo run completed")

    print("\n🔬 Enhanced MLflow integration ready!")
    print("Features: Auto-logging, system info, feature importance, CV results, and more.")