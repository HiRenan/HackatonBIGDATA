#!/usr/bin/env python3
"""
Phase 6: Configuration Management Utilities
Advanced configuration handling with validation and environment support
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "forecast_db"
    username: str = "forecast_user"
    password: str = "forecast_password"

@dataclass
class MLflowConfig:
    """MLflow configuration"""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "hackathon_forecast_2025"
    artifact_root: str = "./mlruns"

@dataclass
class DataConfig:
    """Data processing configuration"""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    chunk_size: int = 100000
    max_memory_gb: float = 8.0
    enable_sampling: bool = True
    sample_size: Optional[int] = None

@dataclass
class ModelConfig:
    """Model training configuration"""
    models_path: str = "models/trained"
    random_state: int = 42
    cross_validation_folds: int = 5
    enable_hyperparameter_tuning: bool = True
    early_stopping_rounds: int = 100

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_monitoring: bool = True
    health_check_interval: int = 30
    metrics_collection_interval: int = 60
    alert_thresholds: Dict[str, float] = None

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'wmape_threshold': 0.20,
                'memory_threshold': 0.80,
                'error_rate_threshold': 0.05
            }

class ConfigManager:
    """Advanced configuration manager"""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("src/config")
        self.environments_dir = self.config_dir / "environments"
        self._config_cache = {}
        self._watchers = []

    def load_config(self,
                   environment: str = "development",
                   config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration for specified environment

        Args:
            environment: Environment name (development, testing, production)
            config_file: Specific config file path

        Returns:
            Merged configuration dictionary
        """
        logger.info(f"Loading configuration for environment: {environment}")

        # Use cache if available
        cache_key = f"{environment}_{config_file}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        try:
            if config_file:
                # Load specific config file
                config = self._load_yaml_file(Path(config_file))
            else:
                # Load base + environment config
                config = self._load_environment_config(environment)

            # Apply environment variable overrides
            config = self._apply_env_overrides(config)

            # Validate configuration
            self._validate_config(config)

            # Cache the config
            self._config_cache[cache_key] = config

            logger.info("Configuration loaded successfully")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def _load_environment_config(self, environment: str) -> Dict[str, Any]:
        """Load base config + environment-specific overrides"""

        # Start with base configuration
        base_file = self.environments_dir / "base.yaml"
        if not base_file.exists():
            logger.warning(f"Base config file not found: {base_file}")
            base_config = {}
        else:
            base_config = self._load_yaml_file(base_file)

        # Load environment-specific config
        env_file = self.environments_dir / f"{environment}.yaml"
        if not env_file.exists():
            logger.warning(f"Environment config file not found: {env_file}")
            env_config = {}
        else:
            env_config = self._load_yaml_file(env_file)

        # Merge configurations (environment overrides base)
        merged_config = self._deep_merge(base_config, env_config)

        # Add metadata
        merged_config['_metadata'] = {
            'environment': environment,
            'loaded_at': datetime.now().isoformat(),
            'config_files': [str(base_file), str(env_file)]
        }

        return merged_config

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                return content or {}
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {file_path}: {str(e)}")
            raise

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Define environment variable mappings
        env_mappings = {
            # Database
            'DATABASE_HOST': 'database.host',
            'DATABASE_PORT': 'database.port',
            'DATABASE_NAME': 'database.database',
            'DATABASE_USER': 'database.username',
            'DATABASE_PASSWORD': 'database.password',

            # MLflow
            'MLFLOW_TRACKING_URI': 'mlflow.tracking_uri',
            'MLFLOW_EXPERIMENT_NAME': 'mlflow.experiment_name',

            # Data
            'DATA_PATH': 'data.raw_data_path',
            'MAX_MEMORY_GB': 'data.max_memory_gb',
            'SAMPLE_SIZE': 'data.sample_size',

            # Monitoring
            'ENABLE_MONITORING': 'monitoring.enable_monitoring',
            'WMAPE_THRESHOLD': 'monitoring.alert_thresholds.wmape_threshold',
        }

        config_copy = config.copy()

        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]

                # Convert value to appropriate type
                if config_path.endswith(('_port', '_size', '_interval', '_folds')):
                    value = int(value)
                elif config_path.endswith(('_gb', '_threshold', '_rate')):
                    value = float(value)
                elif config_path.endswith(('_enable', '_monitoring')):
                    value = value.lower() in ('true', '1', 'yes', 'on')

                # Set nested value
                self._set_nested_value(config_copy, config_path, value)
                logger.debug(f"Applied environment override: {env_var} = {value}")

        return config_copy

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration"""
        required_sections = ['database', 'mlflow', 'data', 'model']

        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing configuration section: {section}")

        # Validate specific values
        if 'data' in config:
            if 'max_memory_gb' in config['data'] and config['data']['max_memory_gb'] < 1:
                raise ValueError("max_memory_gb must be at least 1")

        logger.debug("Configuration validation completed")

    def get_typed_config(self, environment: str = "development") -> Dict[str, Any]:
        """Get configuration with typed dataclass objects"""
        raw_config = self.load_config(environment)

        # Convert to typed configs
        typed_config = {}

        if 'database' in raw_config:
            typed_config['database'] = DatabaseConfig(**raw_config['database'])

        if 'mlflow' in raw_config:
            typed_config['mlflow'] = MLflowConfig(**raw_config['mlflow'])

        if 'data' in raw_config:
            typed_config['data'] = DataConfig(**raw_config['data'])

        if 'model' in raw_config:
            typed_config['model'] = ModelConfig(**raw_config['model'])

        if 'monitoring' in raw_config:
            typed_config['monitoring'] = MonitoringConfig(**raw_config['monitoring'])

        # Keep other sections as-is
        for key, value in raw_config.items():
            if key not in typed_config and not key.startswith('_'):
                typed_config[key] = value

        return typed_config

    def save_config(self, config: Dict[str, Any], file_path: Union[str, Path]):
        """Save configuration to file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass objects to dictionaries
        serializable_config = self._make_serializable(config)

        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_config, f, indent=2, default=str)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(serializable_config, f, default_flow_style=False)

            logger.info(f"Configuration saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to serializable format"""
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    def reload_config(self, environment: str = "development"):
        """Reload configuration (clear cache)"""
        self._config_cache.clear()
        logger.info(f"Configuration cache cleared for {environment}")

def get_config_manager() -> ConfigManager:
    """Get default configuration manager"""
    return ConfigManager()

def load_environment_config(environment: str = "development") -> Dict[str, Any]:
    """Convenience function to load configuration"""
    manager = get_config_manager()
    return manager.load_config(environment)

def get_database_config(environment: str = "development") -> DatabaseConfig:
    """Get database configuration"""
    manager = get_config_manager()
    config = manager.get_typed_config(environment)
    return config.get('database', DatabaseConfig())

def get_mlflow_config(environment: str = "development") -> MLflowConfig:
    """Get MLflow configuration"""
    manager = get_config_manager()
    config = manager.get_typed_config(environment)
    return config.get('mlflow', MLflowConfig())

def get_data_config(environment: str = "development") -> DataConfig:
    """Get data processing configuration"""
    manager = get_config_manager()
    config = manager.get_typed_config(environment)
    return config.get('data', DataConfig())

def setup_environment_from_config(config: Dict[str, Any]):
    """Setup environment variables from configuration"""
    if 'mlflow' in config:
        os.environ['MLFLOW_TRACKING_URI'] = config['mlflow'].get('tracking_uri', '')
        os.environ['MLFLOW_EXPERIMENT_NAME'] = config['mlflow'].get('experiment_name', '')

    if 'database' in config:
        os.environ['DB_HOST'] = config['database'].get('host', '')
        os.environ['DB_PORT'] = str(config['database'].get('port', ''))

    logger.info("Environment variables set from configuration")

if __name__ == "__main__":
    # Demo usage
    print("⚙️ Configuration Manager Demo")
    print("=" * 50)

    # Create config manager
    manager = get_config_manager()

    try:
        # Load development config
        config = manager.load_config("development")
        print("✅ Loaded development configuration")

        # Get typed configs
        typed_config = manager.get_typed_config("development")
        print("✅ Created typed configuration objects")

        # Display some config values
        if 'database' in typed_config:
            db_config = typed_config['database']
            print(f"📊 Database: {db_config.host}:{db_config.port}")

        if 'data' in typed_config:
            data_config = typed_config['data']
            print(f"💾 Max Memory: {data_config.max_memory_gb}GB")

        print("\n⚙️ Configuration management ready!")

    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        print("Make sure config files exist in src/config/environments/")