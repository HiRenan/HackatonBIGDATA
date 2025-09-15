#!/usr/bin/env python3
"""
Phase 6: Data Loading Pipeline Script
Memory-efficient data loading with validation and optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import sys
from typing import Dict, Any, Optional, Tuple
import psutil
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.loaders import DataLoaderFactory, load_competition_data
from src.data.validators import create_transaction_validator
from src.config.phase6_config import get_config
from src.utils.logging import setup_logger

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percentage': memory.percent
    }

def load_data_pipeline(data_path: str,
                      config: Optional[Dict[str, Any]] = None,
                      sample_size: Optional[int] = None,
                      validate: bool = True,
                      save_processed: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete data loading pipeline with validation

    Args:
        data_path: Path to data directory
        config: Configuration dictionary
        sample_size: Sample size for development
        validate: Whether to run validation
        save_processed: Whether to save processed data

    Returns:
        Tuple of (transactions, products, stores) DataFrames
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()

    logger.info("🚀 Starting data loading pipeline")
    logger.info(f"Data path: {data_path}")

    # Initial memory check
    memory_before = get_memory_usage()
    logger.info(f"Memory before loading: {memory_before['used_gb']:.1f}GB ({memory_before['percentage']:.1f}%)")

    try:
        # Load data using optimized loaders
        logger.info("📁 Loading competition data...")

        # Use factory pattern for consistent loading
        config = config or {}

        if sample_size:
            logger.info(f"🎯 Using sample size: {sample_size:,}")
            config['sample_size'] = sample_size

        transactions, products, stores = load_competition_data(data_path, config)

        # Memory check after loading
        memory_after = get_memory_usage()
        logger.info(f"Memory after loading: {memory_after['used_gb']:.1f}GB ({memory_after['percentage']:.1f}%)")
        memory_used = memory_after['used_gb'] - memory_before['used_gb']
        logger.info(f"Memory used for data: {memory_used:.1f}GB")

        # Log data shapes
        logger.info("📊 Data shapes:")
        logger.info(f"  Transactions: {transactions.shape}")
        logger.info(f"  Products: {products.shape}")
        logger.info(f"  Stores: {stores.shape}")

        # Data validation
        if validate:
            logger.info("✅ Running data validation...")

            # Validate transactions (most critical)
            validator = create_transaction_validator()
            validation_report = validator.validate(transactions)

            logger.info(f"Validation results:")
            logger.info(f"  Success rate: {validation_report.success_rate:.1%}")
            logger.info(f"  Passed: {validation_report.passed}")
            logger.info(f"  Failed: {validation_report.failed}")
            logger.info(f"  Warnings: {validation_report.warnings}")

            # Log critical errors
            if validation_report.failed > 0:
                logger.error("❌ Critical validation errors found!")
                summary = validation_report.get_summary()
                for error in summary['errors']:
                    logger.error(f"  - {error}")

                if config.get('strict_validation', False):
                    raise ValueError("Data validation failed in strict mode")

            # Log warnings
            if validation_report.warnings > 0:
                logger.warning("⚠️ Data quality warnings:")
                summary = validation_report.get_summary()
                for warning in summary['warnings_list']:
                    logger.warning(f"  - {warning}")

        # Save processed data if requested
        if save_processed:
            processed_path = Path(data_path).parent / "processed"
            processed_path.mkdir(exist_ok=True)

            logger.info(f"💾 Saving processed data to {processed_path}")

            # Save with optimized formats
            transactions.to_parquet(processed_path / "transactions_processed.parquet",
                                   compression='snappy', index=False)
            products.to_parquet(processed_path / "products_processed.parquet",
                               compression='snappy', index=False)
            stores.to_parquet(processed_path / "stores_processed.parquet",
                             compression='snappy', index=False)

            logger.info("✅ Processed data saved successfully")

        # Final memory and timing
        end_time = time.time()
        duration = end_time - start_time

        memory_final = get_memory_usage()
        logger.info(f"Final memory usage: {memory_final['used_gb']:.1f}GB ({memory_final['percentage']:.1f}%)")
        logger.info(f"⏱️ Pipeline completed in {duration:.1f} seconds")

        # Data summary
        total_records = len(transactions) + len(products) + len(stores)
        logger.info(f"🎯 Pipeline Summary:")
        logger.info(f"  Total records loaded: {total_records:,}")
        logger.info(f"  Memory efficiency: {total_records / (memory_used * 1024 * 1024):.0f} records/MB")
        logger.info(f"  Processing speed: {total_records / duration:.0f} records/second")

        return transactions, products, stores

    except Exception as e:
        logger.error(f"❌ Data loading pipeline failed: {str(e)}")

        # Memory cleanup on error
        try:
            del locals()['transactions'], locals()['products'], locals()['stores']
        except:
            pass

        raise

def main():
    """Main entry point for data loading script"""
    parser = argparse.ArgumentParser(description="Load competition data with optimizations")

    parser.add_argument("--data-path", "-d", type=str, required=True,
                       help="Path to raw data directory")
    parser.add_argument("--config", "-c", type=str,
                       help="Path to config file")
    parser.add_argument("--sample-size", "-s", type=int,
                       help="Sample size for development (e.g., 100000)")
    parser.add_argument("--no-validation", action="store_true",
                       help="Skip data validation")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save processed data")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--memory-limit", type=float, default=8.0,
                       help="Memory limit in GB")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger("data_loading", level=args.log_level)

    # Load configuration
    if args.config:
        config = get_config(args.config)
    else:
        config = get_config("development")  # Default to development config

    # Override config with CLI arguments
    if args.sample_size:
        config['data_loading']['sample_size'] = args.sample_size

    config['data_loading']['max_memory_usage_gb'] = args.memory_limit
    config['data_loading']['enable_validation'] = not args.no_validation

    try:
        # Check data path exists
        data_path = Path(args.data_path)
        if not data_path.exists():
            logger.error(f"Data path does not exist: {data_path}")
            return 1

        # Check available memory
        memory_info = get_memory_usage()
        if memory_info['available_gb'] < args.memory_limit:
            logger.warning(f"Available memory ({memory_info['available_gb']:.1f}GB) < limit ({args.memory_limit}GB)")

        # Run data loading pipeline
        logger.info(f"🚀 Starting data loading with config: {config['environment']}")

        transactions, products, stores = load_data_pipeline(
            data_path=str(data_path),
            config=config.get('data_loading', {}),
            sample_size=args.sample_size,
            validate=not args.no_validation,
            save_processed=not args.no_save
        )

        logger.info("✅ Data loading completed successfully!")

        # Print summary
        print("\n" + "="*60)
        print("📊 DATA LOADING SUMMARY")
        print("="*60)
        print(f"Transactions: {transactions.shape[0]:,} records")
        print(f"Products:     {products.shape[0]:,} records")
        print(f"Stores:       {stores.shape[0]:,} records")
        print(f"Memory usage: {get_memory_usage()['used_gb']:.1f}GB")
        print("="*60)

        return 0

    except KeyboardInterrupt:
        logger.info("❌ Data loading interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Data loading failed: {str(e)}")
        logger.debug("Full error traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())