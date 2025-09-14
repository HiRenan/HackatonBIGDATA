#!/usr/bin/env python3
"""
Phase 6: Advanced Logging Utilities
Structured logging with performance monitoring and MLflow integration
"""

import logging
import logging.config
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import traceback
from functools import wraps
import psutil

class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)

class PerformanceFormatter(logging.Formatter):
    """Enhanced formatter with performance metrics"""

    def format(self, record):
        # Get memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Enhanced format with metrics
        formatted_time = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

        log_line = (
            f"[{formatted_time}] "
            f"[{record.levelname:8s}] "
            f"[{record.name}] "
            f"[{memory_mb:.0f}MB] "
            f"{record.getMessage()}"
        )

        # Add location info for debug level
        if record.levelno == logging.DEBUG:
            log_line += f" ({record.filename}:{record.lineno})"

        # Add exception info if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line

def setup_logger(name: str,
                level: Union[str, int] = "INFO",
                log_file: Optional[str] = None,
                structured: bool = False,
                performance: bool = True) -> logging.Logger:
    """
    Setup a logger with enhanced formatting

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        structured: Use JSON structured logging
        performance: Include performance metrics

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Clear existing handlers
    logger.handlers.clear()

    # Set level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)

    # Choose formatter
    if structured:
        formatter = StructuredFormatter()
    elif performance:
        formatter = PerformanceFormatter()
    else:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)8s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger

class LogContext:
    """Context manager for structured logging"""

    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info("Operation started", extra={'extra_fields': self.context})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        context = {**self.context, 'duration_seconds': round(duration, 3)}

        if exc_type:
            context['error_type'] = exc_type.__name__
            context['error_message'] = str(exc_val)
            self.logger.error("Operation failed", extra={'extra_fields': context})
        else:
            self.logger.info("Operation completed", extra={'extra_fields': context})

def performance_monitor(logger: Optional[logging.Logger] = None):
    """Decorator to monitor function performance"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)

            # Start monitoring
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            try:
                result = func(*args, **kwargs)

                # Success metrics
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                metrics = {
                    'function': func.__name__,
                    'duration_seconds': round(end_time - start_time, 3),
                    'memory_mb': round(end_memory, 1),
                    'memory_delta_mb': round(end_memory - start_memory, 1),
                    'status': 'success'
                }

                func_logger.info(
                    f"Function {func.__name__} completed in {metrics['duration_seconds']}s",
                    extra={'extra_fields': metrics}
                )

                return result

            except Exception as e:
                # Error metrics
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                metrics = {
                    'function': func.__name__,
                    'duration_seconds': round(end_time - start_time, 3),
                    'memory_mb': round(end_memory, 1),
                    'memory_delta_mb': round(end_memory - start_memory, 1),
                    'status': 'error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }

                func_logger.error(
                    f"Function {func.__name__} failed after {metrics['duration_seconds']}s",
                    extra={'extra_fields': metrics}
                )

                raise

        return wrapper
    return decorator

class MLflowLoggingHandler(logging.Handler):
    """Custom logging handler that sends logs to MLflow"""

    def __init__(self, run_id: Optional[str] = None):
        super().__init__()
        self.run_id = run_id
        self.logs = []

    def emit(self, record):
        """Emit a log record to MLflow"""
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'message': self.format(record)
            }
            self.logs.append(log_entry)

            # Send to MLflow in batches
            if len(self.logs) >= 10:
                self.flush_to_mlflow()

        except Exception:
            self.handleError(record)

    def flush_to_mlflow(self):
        """Flush accumulated logs to MLflow"""
        if not self.logs:
            return

        try:
            import mlflow
            if mlflow.active_run():
                log_text = "\n".join([
                    f"[{log['timestamp']}] [{log['level']}] {log['message']}"
                    for log in self.logs
                ])
                mlflow.log_text(log_text, "logs/batch_logs.txt")
                self.logs.clear()
        except ImportError:
            pass  # MLflow not available
        except Exception as e:
            print(f"Failed to send logs to MLflow: {e}")

class DataQualityLogger:
    """Specialized logger for data quality monitoring"""

    def __init__(self, name: str = "data_quality"):
        self.logger = setup_logger(name, structured=True)

    def log_data_shape(self, data_name: str, shape: tuple, operation: str = "loaded"):
        """Log data shape information"""
        self.logger.info(
            f"Data {operation}: {data_name}",
            extra={'extra_fields': {
                'data_name': data_name,
                'shape': shape,
                'rows': shape[0] if shape else 0,
                'columns': shape[1] if len(shape) > 1 else 0,
                'operation': operation
            }}
        )

    def log_missing_values(self, data_name: str, missing_stats: Dict[str, Any]):
        """Log missing value statistics"""
        self.logger.warning(
            f"Missing values detected in {data_name}",
            extra={'extra_fields': {
                'data_name': data_name,
                'missing_stats': missing_stats
            }}
        )

    def log_outliers(self, data_name: str, column: str, outlier_count: int, total_count: int):
        """Log outlier detection results"""
        percentage = (outlier_count / total_count * 100) if total_count > 0 else 0

        self.logger.warning(
            f"Outliers detected in {data_name}.{column}",
            extra={'extra_fields': {
                'data_name': data_name,
                'column': column,
                'outlier_count': outlier_count,
                'total_count': total_count,
                'outlier_percentage': round(percentage, 2)
            }}
        )

    def log_validation_result(self, validation_name: str, passed: bool, details: Dict[str, Any]):
        """Log validation results"""
        level = logging.INFO if passed else logging.ERROR
        self.logger.log(
            level,
            f"Validation {validation_name}: {'PASSED' if passed else 'FAILED'}",
            extra={'extra_fields': {
                'validation_name': validation_name,
                'passed': passed,
                'details': details
            }}
        )

def configure_project_logging(config: Dict[str, Any]):
    """Configure logging for the entire project"""

    # Create logs directory
    log_dir = Path(config.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)

    # Main application log
    app_logger = setup_logger(
        name='hackathon_forecast',
        level=config.get('level', 'INFO'),
        log_file=log_dir / 'application.log',
        performance=True
    )

    # Data processing log
    data_logger = setup_logger(
        name='data_processing',
        level=config.get('level', 'INFO'),
        log_file=log_dir / 'data.log',
        structured=True
    )

    # Model training log
    model_logger = setup_logger(
        name='model_training',
        level=config.get('level', 'INFO'),
        log_file=log_dir / 'models.log',
        performance=True
    )

    # Error log
    error_logger = setup_logger(
        name='errors',
        level='ERROR',
        log_file=log_dir / 'errors.log',
        structured=True
    )

    app_logger.info("Project logging configured successfully")

    return {
        'app': app_logger,
        'data': data_logger,
        'model': model_logger,
        'error': error_logger
    }

# Pre-configured loggers
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a pre-configured logger"""
    return setup_logger(name, level=level, performance=True)

def get_data_logger(name: str = "data") -> DataQualityLogger:
    """Get a data quality logger"""
    return DataQualityLogger(name)

if __name__ == "__main__":
    # Demo usage
    print("📝 Advanced Logging Demo")
    print("=" * 50)

    # Setup demo logger
    logger = setup_logger("demo", level="DEBUG", performance=True)

    # Basic logging
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.debug("This is a debug message")

    # Performance monitoring demo
    @performance_monitor(logger)
    def slow_function():
        time.sleep(1)
        return "completed"

    result = slow_function()
    print(f"Function result: {result}")

    # Context logging demo
    with LogContext(logger, operation="data_processing", dataset="transactions"):
        logger.info("Processing data...")
        time.sleep(0.5)

    # Data quality logger demo
    data_logger = get_data_logger()
    data_logger.log_data_shape("transactions", (100000, 10), "loaded")
    data_logger.log_outliers("transactions", "sales", 150, 100000)

    print("\n✅ Logging system ready!")
    print("Advanced logging with performance monitoring configured.")