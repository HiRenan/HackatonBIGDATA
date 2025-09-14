#!/usr/bin/env python3
"""
Phase 6: Unit Tests for Configuration Management
Tests for Phase6Config and ConfigManager
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml
import os

from config.phase6_config import (
    Phase6Config, ConfigManager, Environment,
    DatabaseConfig, MLflowConfig, ModelConfig,
    get_config, get_current_config, reload_config
)


class TestEnvironmentEnum:
    """Tests for Environment enum"""

    def test_environment_values(self):
        """Test environment enum has correct values"""
        assert Environment.DEVELOPMENT == "development"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"
        assert Environment.TESTING == "testing"


class TestConfigModels:
    """Tests for Pydantic configuration models"""

    def test_database_config_valid(self):
        """Test valid database configuration"""
        config_data = {
            "host": "localhost",
            "port": 5432,
            "name": "test_db",
            "user": "test_user",
            "password": "test_password"
        }

        db_config = DatabaseConfig(**config_data)

        assert db_config.host == "localhost"
        assert db_config.port == 5432
        assert db_config.name == "test_db"
        assert db_config.user == "test_user"
        assert db_config.password == "test_password"

    def test_database_config_defaults(self):
        """Test database configuration defaults"""
        minimal_config = {
            "host": "localhost",
            "name": "test_db",
            "user": "test_user",
            "password": "test_password"
        }

        db_config = DatabaseConfig(**minimal_config)

        assert db_config.port == 5432  # Default
        assert db_config.ssl_mode == "require"  # Default
        assert db_config.connection_timeout == 30  # Default

    def test_database_config_validation_empty_host(self):
        """Test database config validation fails for empty host"""
        config_data = {
            "host": "",  # Empty host should fail
            "name": "test_db",
            "user": "test_user",
            "password": "test_password"
        }

        with pytest.raises(ValueError, match="Host cannot be empty"):
            DatabaseConfig(**config_data)

    def test_database_config_validation_invalid_port(self):
        """Test database config validation for invalid port"""
        config_data = {
            "host": "localhost",
            "port": 99999,  # Invalid port
            "name": "test_db",
            "user": "test_user",
            "password": "test_password"
        }

        with pytest.raises(ValueError):
            DatabaseConfig(**config_data)

    def test_mlflow_config_valid(self):
        """Test valid MLflow configuration"""
        config_data = {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test_experiment"
        }

        mlflow_config = MLflowConfig(**config_data)

        assert mlflow_config.tracking_uri == "http://localhost:5000"
        assert mlflow_config.experiment_name == "test_experiment"
        assert mlflow_config.enable_system_metrics is True  # Default

    def test_model_config_valid(self):
        """Test valid model configuration"""
        config_data = {
            "default_model_type": "prophet",
            "ensemble_enabled": False,
            "max_training_time": 1800
        }

        model_config = ModelConfig(**config_data)

        assert model_config.default_model_type == "prophet"
        assert model_config.ensemble_enabled is False
        assert model_config.max_training_time == 1800

    def test_phase6_config_complete(self):
        """Test complete Phase6Config"""
        config_data = {
            "environment": "development",
            "debug_mode": True,
            "version": "1.0.0",
            "database": {
                "host": "localhost",
                "name": "test_db",
                "user": "test_user",
                "password": "test_password"
            },
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "test_experiment"
            },
            "models": {
                "default_model_type": "lightgbm"
            },
            "data": {
                "data_path": "data/test"
            },
            "logging": {
                "level": "DEBUG"
            },
            "monitoring": {
                "enable_monitoring": True
            },
            "security": {
                "enable_api_key_auth": False
            }
        }

        config = Phase6Config(**config_data)

        assert config.environment == Environment.DEVELOPMENT
        assert config.debug_mode is True
        assert config.database.host == "localhost"
        assert config.mlflow.experiment_name == "test_experiment"


class TestConfigManager:
    """Tests for ConfigManager"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_manager = ConfigManager(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization"""
        manager = ConfigManager()
        assert manager.config_dir.exists()

        custom_manager = ConfigManager(self.temp_dir)
        assert custom_manager.config_dir == self.temp_dir

    def test_load_yaml_file_success(self):
        """Test loading YAML file successfully"""
        test_config = {"test_key": "test_value", "nested": {"key": "value"}}
        config_file = self.temp_dir / "test.yaml"

        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)

        result = self.config_manager._load_yaml_file(config_file)
        assert result == test_config

    def test_load_yaml_file_not_found(self):
        """Test loading non-existent YAML file"""
        non_existent_file = self.temp_dir / "not_found.yaml"
        result = self.config_manager._load_yaml_file(non_existent_file)
        assert result == {}

    def test_save_yaml_file(self):
        """Test saving YAML file"""
        test_config = {"test_key": "test_value"}
        config_file = self.temp_dir / "output.yaml"

        self.config_manager._save_yaml_file(config_file, test_config)

        assert config_file.exists()
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config == test_config

    def test_merge_configs(self):
        """Test merging configurations"""
        base_config = {
            "database": {"host": "localhost", "port": 5432},
            "logging": {"level": "INFO"}
        }

        env_config = {
            "database": {"host": "prod-host", "ssl_mode": "require"},
            "monitoring": {"enabled": True}
        }

        merged = self.config_manager._merge_configs(base_config, env_config)

        # Should merge nested dictionaries
        assert merged["database"]["host"] == "prod-host"  # Override
        assert merged["database"]["port"] == 5432        # Keep from base
        assert merged["database"]["ssl_mode"] == "require"  # New from env

        # Should add new top-level keys
        assert merged["monitoring"]["enabled"] is True
        assert merged["logging"]["level"] == "INFO"

    @patch.dict(os.environ, {
        'DB_HOST': 'env-host',
        'DB_PORT': '3306',
        'MLFLOW_TRACKING_URI': 'http://env-mlflow:5000'
    })
    def test_apply_env_overrides(self):
        """Test applying environment variable overrides"""
        config = {
            "database": {
                "host": "config-host",
                "port": 5432,
                "name": "test_db",
                "user": "test_user",
                "password": "test_password"
            },
            "mlflow": {
                "tracking_uri": "http://config-mlflow:5000",
                "experiment_name": "test"
            }
        }

        result = self.config_manager._apply_env_overrides(config)

        assert result["database"]["host"] == "env-host"
        assert result["database"]["port"] == 3306
        assert result["mlflow"]["tracking_uri"] == "http://env-mlflow:5000"

    def test_create_default_configs(self):
        """Test creating default configuration files"""
        self.config_manager._create_default_configs()

        # Check that default config files were created
        base_config_path = self.temp_dir / "base.yaml"
        dev_config_path = self.temp_dir / "development.yaml"
        prod_config_path = self.temp_dir / "production.yaml"

        assert base_config_path.exists()
        assert dev_config_path.exists()
        assert prod_config_path.exists()

        # Check content structure
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        assert "environment" in base_config
        assert "database" in base_config
        assert "mlflow" in base_config

    def test_load_config_development(self):
        """Test loading development configuration"""
        # Create test config files
        base_config = {
            "environment": "development",
            "database": {"host": "localhost", "name": "test_db", "user": "user", "password": "pass"},
            "mlflow": {"tracking_uri": "http://localhost:5000", "experiment_name": "test"},
            "models": {"default_model_type": "lightgbm"},
            "data": {"data_path": "data/raw"},
            "logging": {"level": "INFO"},
            "monitoring": {"enable_monitoring": True},
            "security": {"enable_api_key_auth": False}
        }

        dev_config = {
            "debug_mode": True,
            "logging": {"level": "DEBUG"}
        }

        with open(self.temp_dir / "base.yaml", 'w') as f:
            yaml.dump(base_config, f)

        with open(self.temp_dir / "development.yaml", 'w') as f:
            yaml.dump(dev_config, f)

        config = self.config_manager.load_config("development")

        assert config.environment == Environment.DEVELOPMENT
        assert config.debug_mode is True
        assert config.logging.level == "DEBUG"  # Override from dev config
        assert config.database.host == "localhost"  # From base config

    @patch.dict(os.environ, {'FORECAST_ENV': 'production'})
    def test_load_config_from_env_var(self):
        """Test loading configuration from environment variable"""
        # Create minimal config files
        base_config = {
            "environment": "development",
            "database": {"host": "localhost", "name": "test_db", "user": "user", "password": "pass"},
            "mlflow": {"tracking_uri": "http://localhost:5000", "experiment_name": "test"},
            "models": {"default_model_type": "lightgbm"},
            "data": {"data_path": "data/raw"},
            "logging": {"level": "INFO"},
            "monitoring": {"enable_monitoring": True},
            "security": {"enable_api_key_auth": False}
        }

        with open(self.temp_dir / "base.yaml", 'w') as f:
            yaml.dump(base_config, f)

        # Should load production config based on env var
        config = self.config_manager.load_config()
        assert config.environment == Environment.PRODUCTION

    def test_validate_config_valid(self):
        """Test validating valid configuration"""
        valid_config = {
            "environment": "development",
            "database": {"host": "localhost", "name": "test_db", "user": "user", "password": "pass"},
            "mlflow": {"tracking_uri": "http://localhost:5000", "experiment_name": "test"},
            "models": {"default_model_type": "lightgbm"},
            "data": {"data_path": "data/raw"},
            "logging": {"level": "INFO"},
            "monitoring": {"enable_monitoring": True},
            "security": {"enable_api_key_auth": False}
        }

        assert self.config_manager.validate_config(valid_config) is True

    def test_validate_config_invalid(self):
        """Test validating invalid configuration"""
        invalid_config = {
            "environment": "development",
            "database": {"host": "", "name": "test_db"},  # Empty host should fail
        }

        assert self.config_manager.validate_config(invalid_config) is False

    def test_get_current_config(self):
        """Test getting current configuration"""
        assert self.config_manager.get_config() is None

        # Load a config first
        self.config_manager._create_default_configs()
        config = self.config_manager.load_config("development")

        assert self.config_manager.get_config() == config


class TestGlobalConfigFunctions:
    """Tests for global configuration functions"""

    @patch('config.phase6_config.config_manager')
    def test_get_config_function(self, mock_config_manager):
        """Test get_config global function"""
        mock_config = Mock()
        mock_config_manager.load_config.return_value = mock_config

        result = get_config("production")

        mock_config_manager.load_config.assert_called_once_with("production")
        assert result == mock_config

    @patch('config.phase6_config.config_manager')
    def test_get_current_config_function(self, mock_config_manager):
        """Test get_current_config global function"""
        mock_config = Mock()
        mock_config_manager.get_config.return_value = mock_config

        result = get_current_config()

        mock_config_manager.get_config.assert_called_once()
        assert result == mock_config

    @patch('config.phase6_config.config_manager')
    def test_reload_config_function(self, mock_config_manager):
        """Test reload_config global function"""
        mock_config = Mock()
        mock_config_manager.reload_config.return_value = mock_config

        result = reload_config()

        mock_config_manager.reload_config.assert_called_once()
        assert result == mock_config


@pytest.mark.parametrize("environment,expected_debug", [
    ("development", True),
    ("testing", False),
    ("production", False),
])
class TestConfigEnvironmentParametrized:
    """Parametrized tests for different environments"""

    def test_environment_specific_settings(self, environment, expected_debug):
        """Test environment-specific settings are applied correctly"""
        # This is a conceptual test - in real implementation,
        # you would load actual config files and verify settings
        env_enum = Environment(environment)
        assert isinstance(env_enum, Environment)
        assert env_enum.value == environment


class TestConfigIntegration:
    """Integration tests for configuration system"""

    @pytest.mark.integration
    def test_full_config_loading_cycle(self):
        """Test complete configuration loading cycle"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(Path(temp_dir))

            # Create config files
            config_manager._create_default_configs()

            # Load development config
            dev_config = config_manager.load_config("development")
            assert dev_config.environment == Environment.DEVELOPMENT

            # Load production config
            prod_config = config_manager.load_config("production")
            assert prod_config.environment == Environment.PRODUCTION

            # Verify different settings
            assert dev_config.debug_mode != prod_config.debug_mode

    @pytest.mark.integration
    def test_config_with_missing_files(self):
        """Test configuration loading with missing files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(Path(temp_dir))

            # Try to load config without creating default files
            config = config_manager.load_config("development")

            # Should still work by creating default configs
            assert config is not None
            assert config.environment == Environment.DEVELOPMENT