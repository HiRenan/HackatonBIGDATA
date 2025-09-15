#!/usr/bin/env python3
"""
Phase 6: Production-Ready Configuration Management
Environment-specific configurations with validation and secret management
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pydantic import BaseModel, Field, validator
from enum import Enum

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = Field(..., description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    ssl_mode: str = Field(default="require", description="SSL mode")
    connection_timeout: int = Field(default=30, ge=1, description="Connection timeout in seconds")
    max_connections: int = Field(default=20, ge=1, description="Maximum connections")

    @validator('host')
    def validate_host(cls, v):
        if not v or v.isspace():
            raise ValueError('Host cannot be empty')
        return v

class MLflowConfig(BaseModel):
    """MLflow configuration"""
    tracking_uri: str = Field(..., description="MLflow tracking URI")
    experiment_name: str = Field(..., description="Default experiment name")
    artifact_location: Optional[str] = Field(None, description="Artifact storage location")
    registry_uri: Optional[str] = Field(None, description="Model registry URI")
    enable_system_metrics: bool = Field(default=True, description="Enable system metrics logging")
    auto_log_models: bool = Field(default=True, description="Auto-log models")

class ModelConfig(BaseModel):
    """Model configuration"""
    default_model_type: str = Field(default="lightgbm", description="Default model type")
    ensemble_enabled: bool = Field(default=True, description="Enable ensemble models")
    auto_hyperparameter_tuning: bool = Field(default=True, description="Enable auto HP tuning")
    max_training_time: int = Field(default=3600, ge=60, description="Max training time in seconds")
    cross_validation_folds: int = Field(default=5, ge=2, description="CV folds")
    early_stopping_rounds: int = Field(default=50, ge=1, description="Early stopping rounds")

    # Model-specific configurations
    lightgbm_config: Dict[str, Any] = Field(default_factory=lambda: {
        "n_estimators": 1000,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "random_state": 42
    })

    prophet_config: Dict[str, Any] = Field(default_factory=lambda: {
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "interval_width": 0.95
    })

class DataConfig(BaseModel):
    """Data processing configuration"""
    data_path: str = Field(..., description="Path to data directory")
    batch_size: int = Field(default=10000, ge=1, description="Processing batch size")
    chunk_size: int = Field(default=100000, ge=1000, description="Data chunk size")
    max_memory_usage_gb: float = Field(default=8.0, gt=0, description="Max memory usage in GB")
    cache_enabled: bool = Field(default=True, description="Enable data caching")
    validation_split: float = Field(default=0.2, gt=0, lt=1, description="Validation split ratio")

    # Data quality thresholds
    max_missing_ratio: float = Field(default=0.5, ge=0, le=1, description="Max missing data ratio")
    min_data_points: int = Field(default=100, ge=1, description="Minimum data points required")

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    enable_file_logging: bool = Field(default=True, description="Enable file logging")
    log_file_path: str = Field(default="logs/application.log", description="Log file path")
    max_file_size_mb: int = Field(default=100, ge=1, description="Max log file size in MB")
    backup_count: int = Field(default=5, ge=1, description="Number of backup log files")
    enable_structured_logging: bool = Field(default=True, description="Enable JSON logging")

class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration"""
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    metrics_collection_interval: int = Field(default=60, ge=10, description="Metrics collection interval in seconds")
    alert_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "wmape_threshold": 0.20,
        "memory_usage_threshold": 0.80,
        "error_rate_threshold": 0.05,
        "response_time_threshold": 30.0
    })
    enable_email_alerts: bool = Field(default=False, description="Enable email alerts")
    slack_webhook_url: Optional[str] = Field(None, description="Slack webhook URL for alerts")

class SecurityConfig(BaseModel):
    """Security configuration"""
    enable_api_key_auth: bool = Field(default=True, description="Enable API key authentication")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(default=60, ge=1, description="Rate limit per minute")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    allowed_origins: List[str] = Field(default=["http://localhost:3000"], description="Allowed CORS origins")

class Phase6Config(BaseModel):
    """Complete Phase 6 configuration"""
    environment: Environment = Field(..., description="Current environment")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    version: str = Field(default="1.0.0", description="Application version")

    # Component configurations
    database: DatabaseConfig
    mlflow: MLflowConfig
    models: ModelConfig
    data: DataConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig
    security: SecurityConfig

    # Feature flags
    feature_flags: Dict[str, bool] = Field(default_factory=lambda: {
        "enable_phase5_models": True,
        "enable_bayesian_calibration": True,
        "enable_business_rules": True,
        "enable_real_time_inference": False,
        "enable_model_explanability": True,
        "enable_drift_detection": True
    })

    class Config:
        use_enum_values = True
        validate_assignment = True

class ConfigManager:
    """Configuration manager with environment-specific loading"""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "environments"
        self.config_dir.mkdir(exist_ok=True)

        self._config: Optional[Phase6Config] = None
        self._environment: Optional[Environment] = None

    def load_config(self, environment: Optional[str] = None) -> Phase6Config:
        """Load configuration for specified environment"""
        env = environment or os.getenv("FORECAST_ENV", "development")

        try:
            self._environment = Environment(env)
        except ValueError:
            logger.warning(f"Unknown environment '{env}', defaulting to development")
            self._environment = Environment.DEVELOPMENT

        # Load base configuration
        base_config = self._load_base_config()

        # Load environment-specific configuration
        env_config = self._load_environment_config(self._environment)

        # Merge configurations
        merged_config = self._merge_configs(base_config, env_config)

        # Apply environment variable overrides
        final_config = self._apply_env_overrides(merged_config)

        # Validate configuration
        self._config = Phase6Config(**final_config)

        logger.info(f"Configuration loaded for environment: {self._environment.value}")
        return self._config

    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        base_config_path = self.config_dir / "base.yaml"

        if not base_config_path.exists():
            logger.info("Base configuration not found, creating default")
            self._create_default_configs()

        return self._load_yaml_file(base_config_path)

    def _load_environment_config(self, environment: Environment) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        env_config_path = self.config_dir / f"{environment.value}.yaml"

        if env_config_path.exists():
            return self._load_yaml_file(env_config_path)
        else:
            logger.warning(f"Environment config not found: {env_config_path}")
            return {}

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded config from: {file_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return {}

    def _merge_configs(self, base_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base and environment configurations"""
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(base_config, env_config)

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Database overrides
        if "database" in config:
            config["database"]["host"] = os.getenv("DB_HOST", config["database"]["host"])
            config["database"]["port"] = int(os.getenv("DB_PORT", config["database"]["port"]))
            config["database"]["name"] = os.getenv("DB_NAME", config["database"]["name"])
            config["database"]["user"] = os.getenv("DB_USER", config["database"]["user"])
            config["database"]["password"] = os.getenv("DB_PASSWORD", config["database"]["password"])

        # MLflow overrides
        if "mlflow" in config:
            config["mlflow"]["tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
            config["mlflow"]["experiment_name"] = os.getenv("MLFLOW_EXPERIMENT_NAME", config["mlflow"]["experiment_name"])

        # Security overrides
        if "security" in config:
            api_key = os.getenv("API_KEY")
            if api_key:
                config["security"]["api_key"] = api_key

        return config

    def _create_default_configs(self) -> None:
        """Create default configuration files"""
        # Base configuration
        base_config = {
            "environment": "development",
            "debug_mode": True,
            "version": "1.0.0",
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "hackathon_forecast",
                "user": "forecast_user",
                "password": "forecast_password",
                "ssl_mode": "prefer"
            },
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "hackathon_forecast_2025",
                "enable_system_metrics": True
            },
            "models": {
                "default_model_type": "lightgbm",
                "ensemble_enabled": True,
                "auto_hyperparameter_tuning": True
            },
            "data": {
                "data_path": "data/raw",
                "batch_size": 10000,
                "chunk_size": 100000,
                "max_memory_usage_gb": 8.0
            },
            "logging": {
                "level": "INFO",
                "enable_file_logging": True,
                "log_file_path": "logs/application.log"
            },
            "monitoring": {
                "enable_monitoring": True,
                "metrics_collection_interval": 60
            },
            "security": {
                "enable_api_key_auth": False,
                "enable_rate_limiting": False,
                "enable_cors": True
            }
        }

        # Development environment
        dev_config = {
            "debug_mode": True,
            "database": {
                "host": "localhost",
                "ssl_mode": "disable"
            },
            "logging": {
                "level": "DEBUG"
            },
            "security": {
                "enable_api_key_auth": False
            }
        }

        # Production environment
        prod_config = {
            "debug_mode": False,
            "database": {
                "ssl_mode": "require",
                "connection_timeout": 10,
                "max_connections": 50
            },
            "logging": {
                "level": "INFO",
                "enable_structured_logging": True
            },
            "monitoring": {
                "enable_monitoring": True,
                "enable_email_alerts": True
            },
            "security": {
                "enable_api_key_auth": True,
                "enable_rate_limiting": True,
                "rate_limit_per_minute": 100
            }
        }

        # Save configurations
        self._save_yaml_file(self.config_dir / "base.yaml", base_config)
        self._save_yaml_file(self.config_dir / "development.yaml", dev_config)
        self._save_yaml_file(self.config_dir / "production.yaml", prod_config)

        logger.info("Default configuration files created")

    def _save_yaml_file(self, file_path: Path, config: Dict[str, Any]) -> None:
        """Save configuration to YAML file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.debug(f"Saved config to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {file_path}: {e}")

    def get_config(self) -> Optional[Phase6Config]:
        """Get current configuration"""
        return self._config

    def get_environment(self) -> Optional[Environment]:
        """Get current environment"""
        return self._environment

    def reload_config(self) -> Phase6Config:
        """Reload configuration"""
        return self.load_config(self._environment.value if self._environment else None)

    def validate_config(self, config_dict: Dict[str, Any]) -> bool:
        """Validate configuration dictionary"""
        try:
            Phase6Config(**config_dict)
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

# Global configuration manager instance
config_manager = ConfigManager()

def get_config(environment: Optional[str] = None) -> Phase6Config:
    """Get configuration for environment"""
    return config_manager.load_config(environment)

def get_current_config() -> Optional[Phase6Config]:
    """Get current loaded configuration"""
    return config_manager.get_config()

def reload_config() -> Phase6Config:
    """Reload current configuration"""
    return config_manager.reload_config()

if __name__ == "__main__":
    # Demo usage
    print("⚙️ Phase 6 Configuration Management Demo")
    print("=" * 50)

    # Load configuration
    print("\n🔧 Loading configuration...")
    config = get_config("development")

    print(f"Environment: {config.environment}")
    print(f"Debug mode: {config.debug_mode}")
    print(f"Database host: {config.database.host}")
    print(f"MLflow URI: {config.mlflow.tracking_uri}")
    print(f"Default model: {config.models.default_model_type}")

    # Show feature flags
    print(f"\n🚀 Feature flags:")
    for flag, enabled in config.feature_flags.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {flag}")

    # Test environment switching
    print(f"\n🔄 Testing environment switching...")
    prod_config = get_config("production")
    print(f"Production debug mode: {prod_config.debug_mode}")
    print(f"Production API auth: {prod_config.security.enable_api_key_auth}")

    print("\n✅ Configuration demo completed!")