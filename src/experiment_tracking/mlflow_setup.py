#!/usr/bin/env python3
"""
MLflow Setup and Configuration for Hackathon Forecast 2025
Optimized experiment tracking for forecasting competition
"""

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime

class HackathonMLflowTracker:
    """Specialized MLflow tracker for hackathon forecasting experiments"""
    
    def __init__(self, experiment_name: str = "hackathon_forecast_2025"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self.setup_experiment()
        
    def setup_experiment(self):
        """Initialize MLflow experiment with proper configuration"""
        
        # Set tracking URI to local directory
        mlflow_dir = Path("mlruns").absolute()
        mlflow_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                tags={
                    "project": "hackathon_forecast_2025",
                    "competition": "big_data_forecast",
                    "metric": "WMAPE",
                    "domain": "retail_forecasting"
                }
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(self.experiment_name)
        self.experiment_id = experiment_id
        
        print(f"MLflow experiment '{self.experiment_name}' initialized")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        
    def start_run(self, run_name: str, 
                  model_type: str,
                  phase: str = "development",
                  tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new MLflow run with comprehensive tagging"""
        
        # Default tags
        default_tags = {
            "model_type": model_type,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "hackathon_phase": "development"
        }
        
        if tags:
            default_tags.update(tags)
            
        # Start run
        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        
        # Log environment info
        mlflow.log_params({
            "python_version": os.sys.version.split()[0],
            "experiment_name": self.experiment_name,
            "run_started": datetime.now().isoformat()
        })
        
        return run.info.run_id
    
    def log_data_info(self, data_info: Dict[str, Any]):
        """Log dataset characteristics and preprocessing steps"""
        
        mlflow.log_params({
            f"data_{key}": str(value)[:250]  # MLflow param limit
            for key, value in data_info.items()
            if not isinstance(value, (dict, list))
        })
        
        # Log detailed data info as artifact
        data_info_path = "temp_data_info.json"
        with open(data_info_path, 'w') as f:
            json.dump(data_info, f, indent=2, default=str)
        mlflow.log_artifact(data_info_path, "data_info")
        os.remove(data_info_path)
    
    def log_feature_engineering(self, feature_config: Dict[str, Any]):
        """Log feature engineering configuration and results"""
        
        # Log key feature params
        if "n_features" in feature_config:
            mlflow.log_param("n_features", feature_config["n_features"])
        if "feature_types" in feature_config:
            mlflow.log_param("feature_types", str(feature_config["feature_types"])[:250])
            
        # Log full config as artifact
        feature_config_path = "temp_feature_config.json"
        with open(feature_config_path, 'w') as f:
            json.dump(feature_config, f, indent=2, default=str)
        mlflow.log_artifact(feature_config_path, "features")
        os.remove(feature_config_path)
    
    def log_model_config(self, model_config: Dict[str, Any]):
        """Log model hyperparameters and configuration"""
        
        # Log hyperparameters
        for key, value in model_config.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(f"model_{key}", value)
    
    def log_training_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics (WMAPE, MAPE, etc.)"""
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value, step=step)
    
    def log_validation_results(self, cv_results: Dict[str, Any]):
        """Log cross-validation results"""
        
        # Log CV summary metrics
        for metric_name, scores in cv_results.items():
            if isinstance(scores, (list, np.ndarray)):
                mlflow.log_metric(f"cv_{metric_name}_mean", np.mean(scores))
                mlflow.log_metric(f"cv_{metric_name}_std", np.std(scores))
                mlflow.log_metric(f"cv_{metric_name}_min", np.min(scores))
                mlflow.log_metric(f"cv_{metric_name}_max", np.max(scores))
        
        # Log detailed CV results as artifact
        cv_results_path = "temp_cv_results.json"
        with open(cv_results_path, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        mlflow.log_artifact(cv_results_path, "validation")
        os.remove(cv_results_path)
    
    def log_model(self, model, model_type: str, signature=None):
        """Log trained model with appropriate MLflow logger"""
        
        if model_type.lower() == "lightgbm":
            mlflow.lightgbm.log_model(model, "model", signature=signature)
        elif model_type.lower() in ["prophet", "facebook_prophet"]:
            # Prophet models need custom serialization
            mlflow.sklearn.log_model(model, "model", signature=signature)
        elif model_type.lower() in ["pytorch", "torch", "lstm"]:
            mlflow.pytorch.log_model(model, "model", signature=signature)
        else:
            # Default to sklearn format
            mlflow.sklearn.log_model(model, "model", signature=signature)
    
    def log_predictions(self, predictions: pd.DataFrame, 
                       prediction_type: str = "validation"):
        """Log prediction results"""
        
        # Save predictions as CSV
        pred_filename = f"{prediction_type}_predictions.csv"
        predictions.to_csv(pred_filename, index=False)
        mlflow.log_artifact(pred_filename, "predictions")
        os.remove(pred_filename)
        
        # Log prediction statistics
        if 'prediction' in predictions.columns:
            pred_stats = {
                'prediction_mean': predictions['prediction'].mean(),
                'prediction_std': predictions['prediction'].std(),
                'prediction_min': predictions['prediction'].min(),
                'prediction_max': predictions['prediction'].max(),
                'n_predictions': len(predictions)
            }
            
            for stat_name, value in pred_stats.items():
                mlflow.log_metric(f"{prediction_type}_{stat_name}", value)
    
    def log_submission(self, submission_df: pd.DataFrame, 
                      submission_name: str,
                      final_score: Optional[float] = None):
        """Log competition submission"""
        
        # Save submission
        submission_path = f"submission_{submission_name}.csv"
        submission_df.to_csv(submission_path, index=False)
        mlflow.log_artifact(submission_path, "submissions")
        
        # Log submission metadata
        mlflow.log_params({
            "submission_name": submission_name,
            "submission_rows": len(submission_df),
            "submission_timestamp": datetime.now().isoformat()
        })
        
        if final_score is not None:
            mlflow.log_metric("final_submission_score", final_score)
        
        # Keep local copy
        submissions_dir = Path("submissions")
        submissions_dir.mkdir(exist_ok=True)
        submission_df.to_csv(submissions_dir / submission_path, index=False)
        
        os.remove(submission_path)
        print(f"Submission '{submission_name}' logged and saved")
    
    def compare_experiments(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiment runs"""
        
        runs_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            
            run_data = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'unnamed'),
                'model_type': run.data.tags.get('model_type', 'unknown'),
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time
            }
            
            # Add metrics
            for metric_name, metric_value in run.data.metrics.items():
                run_data[f"metric_{metric_name}"] = metric_value
            
            # Add key parameters
            for param_name, param_value in run.data.params.items():
                if param_name in ['model_type', 'n_features', 'data_shape']:
                    run_data[f"param_{param_name}"] = param_value
            
            runs_data.append(run_data)
        
        return pd.DataFrame(runs_data)
    
    def get_best_runs(self, metric_name: str = "cv_wmape_mean", 
                     top_k: int = 5,
                     ascending: bool = True) -> pd.DataFrame:
        """Get best performing runs based on specified metric"""
        
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        return runs.head(top_k)
    
    def log_ensemble_info(self, ensemble_config: Dict[str, Any]):
        """Log ensemble model configuration"""
        
        mlflow.log_params({
            "ensemble_type": ensemble_config.get("type", "unknown"),
            "n_models": ensemble_config.get("n_models", 0),
            "ensemble_method": ensemble_config.get("method", "unknown")
        })
        
        # Log full ensemble config
        config_path = "temp_ensemble_config.json"
        with open(config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2, default=str)
        mlflow.log_artifact(config_path, "ensemble")
        os.remove(config_path)
    
    def end_run(self, status: str = "FINISHED"):
        """End current MLflow run"""
        mlflow.end_run(status=status)


def setup_experiment_tracking():
    """Initialize experiment tracking for the hackathon"""
    
    print("Setting up MLflow experiment tracking...")
    
    tracker = HackathonMLflowTracker()
    
    # Create sample run to test setup
    with mlflow.start_run(run_name="setup_test") as run:
        mlflow.log_params({
            "test": "setup",
            "status": "initialized"
        })
        mlflow.log_metric("setup_score", 1.0)
        
        print(f"Test run created: {run.info.run_id}")
    
    print("✓ MLflow experiment tracking setup complete!")
    print(f"  - Experiment: {tracker.experiment_name}")
    print(f"  - Tracking URI: {mlflow.get_tracking_uri()}")
    print("  - Use: mlflow ui (to view experiments)")
    
    return tracker


if __name__ == "__main__":
    # Test the setup
    setup_experiment_tracking()