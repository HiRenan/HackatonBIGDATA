#!/usr/bin/env python3
"""
Phase 7: Submission Validation Script
Comprehensive validation tool for competition submissions
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime
import json
from typing import Optional, Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.metrics import wmape, retail_forecast_evaluation
from src.utils.logging import setup_logger
from src.utils.config import get_config_manager

def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging for script"""
    return setup_logger("validate_submission", level=log_level, performance=True)

def load_submission_file(file_path: str, hackathon_format: bool = False) -> pd.DataFrame:
    """Load and validate submission file format"""
    logger = logging.getLogger("validate_submission")

    try:
        # Try different formats
        if file_path.endswith('.csv'):
            if hackathon_format:
                # Try semicolon separator for hackathon format
                try:
                    submission = pd.read_csv(file_path, sep=';', encoding='utf-8')
                except:
                    submission = pd.read_csv(file_path)
            else:
                submission = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            submission = pd.read_parquet(file_path)
        else:
            # Try CSV first
            try:
                if hackathon_format:
                    submission = pd.read_csv(file_path, sep=';', encoding='utf-8')
                else:
                    submission = pd.read_csv(file_path)
            except:
                submission = pd.read_parquet(file_path)

        logger.info(f"Loaded submission file: {submission.shape[0]:,} rows, {submission.shape[1]} columns")
        return submission

    except Exception as e:
        logger.error(f"Failed to load submission file: {str(e)}")
        raise

def validate_submission_format(submission: pd.DataFrame,
                              required_columns: List[str] = None,
                              hackathon_format: bool = False) -> Dict[str, Any]:
    """Validate submission format requirements"""
    logger = logging.getLogger("validate_submission")

    if required_columns is None:
        if hackathon_format:
            required_columns = ["semana", "pdv", "produto", "quantidade"]
        else:
            required_columns = ["store_id", "product_id", "date", "prediction"]

    validation_result = {
        'format_valid': True,
        'issues': [],
        'warnings': [],
        'stats': {}
    }

    logger.info("[CHECK] Validating submission format")

    # Check required columns
    missing_columns = [col for col in required_columns if col not in submission.columns]
    if missing_columns:
        validation_result['format_valid'] = False
        validation_result['issues'].append(f"Missing required columns: {missing_columns}")

    # Check for extra columns
    extra_columns = [col for col in submission.columns if col not in required_columns]
    if extra_columns:
        validation_result['warnings'].append(f"Extra columns found: {extra_columns}")

    # Check data types
    if hackathon_format:
        # Hackathon specific validations
        for col in required_columns:
            if col in submission.columns:
                if not pd.api.types.is_integer_dtype(submission[col]):
                    validation_result['format_valid'] = False
                    validation_result['issues'].append(f"Column {col} should be integer type")

        # Check value ranges for hackathon format
        if 'semana' in submission.columns:
            invalid_weeks = ~submission['semana'].between(1, 5)
            if invalid_weeks.any():
                validation_result['format_valid'] = False
                validation_result['issues'].append(f"Invalid week values: must be 1-5, found {submission.loc[invalid_weeks, 'semana'].unique()}")

        if 'quantidade' in submission.columns:
            negative_quantities = submission['quantidade'] < 0
            if negative_quantities.any():
                validation_result['format_valid'] = False
                validation_result['issues'].append(f"Negative quantities found: {negative_quantities.sum():,} rows")

    else:
        # Standard format validations
        if 'prediction' in submission.columns:
            if not pd.api.types.is_numeric_dtype(submission['prediction']):
                validation_result['format_valid'] = False
                validation_result['issues'].append("Prediction column is not numeric")

    # Check for missing values
    missing_values = submission.isnull().sum()
    if missing_values.any():
        validation_result['format_valid'] = False
        for col, count in missing_values[missing_values > 0].items():
            validation_result['issues'].append(f"Missing values in {col}: {count:,}")

    # Check for duplicates
    if len(required_columns) >= 3:
        key_columns = required_columns[:-1]  # All except prediction
        duplicates = submission.duplicated(subset=key_columns).sum()
        if duplicates > 0:
            validation_result['format_valid'] = False
            validation_result['issues'].append(f"Duplicate rows found: {duplicates:,}")

    # Basic statistics
    target_col = 'quantidade' if hackathon_format else 'prediction'
    if target_col in submission.columns:
        pred_stats = submission[target_col].describe()
        validation_result['stats'][target_col] = {
            'count': int(pred_stats['count']),
            'mean': float(pred_stats['mean']),
            'std': float(pred_stats['std']),
            'min': float(pred_stats['min']),
            'max': float(pred_stats['max']),
            'negative_count': int((submission[target_col] < 0).sum()),
            'zero_count': int((submission[target_col] == 0).sum())
        }

        # Check for negative predictions
        negative_count = (submission[target_col] < 0).sum()
        if negative_count > 0:
            validation_result['warnings'].append(f"Negative {target_col} found: {negative_count:,}")

    # Add hackathon specific statistics
    if hackathon_format:
        if 'semana' in submission.columns:
            validation_result['stats']['semana'] = {
                'min': int(submission['semana'].min()),
                'max': int(submission['semana'].max()),
                'unique_count': int(submission['semana'].nunique())
            }

        if 'pdv' in submission.columns:
            validation_result['stats']['pdv'] = {
                'unique_count': int(submission['pdv'].nunique()),
                'min': int(submission['pdv'].min()),
                'max': int(submission['pdv'].max())
            }

        if 'produto' in submission.columns:
            validation_result['stats']['produto'] = {
                'unique_count': int(submission['produto'].nunique()),
                'min': int(submission['produto'].min()),
                'max': int(submission['produto'].max())
            }

        # Check for extreme values (only for non-hackathon format)
        if 'quantidade' in submission.columns:
            target_col = 'quantidade'
        else:
            target_col = 'prediction'

        if target_col in submission.columns:
            q99 = submission[target_col].quantile(0.99)
            q01 = submission[target_col].quantile(0.01)
            extreme_high = (submission[target_col] > q99 * 10).sum()
            extreme_low = (submission[target_col] < q01 / 10).sum()

            if extreme_high > 0:
                validation_result['warnings'].append(f"Extremely high {target_col} (>10x Q99): {extreme_high:,}")
            if extreme_low > 0:
                validation_result['warnings'].append(f"Extremely low {target_col} (<Q01/10): {extreme_low:,}")

    validation_result['stats']['total_rows'] = len(submission)
    validation_result['stats']['total_columns'] = len(submission.columns)

    return validation_result

def validate_business_rules(submission: pd.DataFrame) -> Dict[str, Any]:
    """Validate business logic rules"""
    logger = logging.getLogger("validate_submission")

    validation_result = {
        'business_valid': True,
        'issues': [],
        'warnings': [],
        'stats': {}
    }

    logger.info("[BUSINESS] Validating business rules")

    if 'prediction' not in submission.columns:
        validation_result['business_valid'] = False
        validation_result['issues'].append("No prediction column found for business validation")
        return validation_result

    # Rule 1: No negative sales
    negative_count = (submission['prediction'] < 0).sum()
    if negative_count > 0:
        validation_result['business_valid'] = False
        validation_result['issues'].append(f"Negative predictions not allowed: {negative_count:,} found")

    # Rule 2: Reasonable growth rates
    if 'store_id' in submission.columns and 'product_id' in submission.columns and 'date' in submission.columns:
        try:
            submission['date'] = pd.to_datetime(submission['date'])
            submission_sorted = submission.sort_values(['store_id', 'product_id', 'date'])

            # Calculate period-over-period growth
            submission_sorted['prev_prediction'] = submission_sorted.groupby(['store_id', 'product_id'])['prediction'].shift(1)
            submission_sorted['growth_rate'] = (submission_sorted['prediction'] / submission_sorted['prev_prediction'] - 1)

            extreme_growth = (submission_sorted['growth_rate'].abs() > 10).sum()  # 1000% growth
            if extreme_growth > 0:
                validation_result['warnings'].append(f"Extreme growth rates (>1000%): {extreme_growth:,}")

            validation_result['stats']['max_growth_rate'] = float(submission_sorted['growth_rate'].max())
            validation_result['stats']['min_growth_rate'] = float(submission_sorted['growth_rate'].min())

        except Exception as e:
            validation_result['warnings'].append(f"Could not validate growth rates: {str(e)}")

    # Rule 3: Seasonality patterns (basic check)
    if 'date' in submission.columns:
        try:
            submission['date'] = pd.to_datetime(submission['date'])
            submission['weekday'] = submission['date'].dt.dayofweek
            submission['month'] = submission['date'].dt.month

            # Check for realistic weekly patterns
            weekly_avg = submission.groupby('weekday')['prediction'].mean()
            weekly_cv = weekly_avg.std() / weekly_avg.mean()

            if weekly_cv > 2.0:  # Very high coefficient of variation
                validation_result['warnings'].append(f"Unusual weekly seasonality pattern (CV: {weekly_cv:.2f})")

            validation_result['stats']['weekly_cv'] = float(weekly_cv)

        except Exception as e:
            validation_result['warnings'].append(f"Could not validate seasonality: {str(e)}")

    return validation_result

def calculate_submission_metrics(submission: pd.DataFrame,
                               actual_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Calculate metrics for submission"""
    logger = logging.getLogger("validate_submission")

    metrics_result = {
        'metrics_available': actual_data is not None,
        'descriptive_stats': {},
        'performance_metrics': {}
    }

    logger.info("[METRICS] Calculating submission metrics")

    # Descriptive statistics
    if 'prediction' in submission.columns:
        pred_stats = submission['prediction'].describe()
        metrics_result['descriptive_stats'] = {
            'count': int(pred_stats['count']),
            'mean': float(pred_stats['mean']),
            'median': float(submission['prediction'].median()),
            'std': float(pred_stats['std']),
            'min': float(pred_stats['min']),
            'max': float(pred_stats['max']),
            'q25': float(pred_stats['25%']),
            'q75': float(pred_stats['75%']),
            'skewness': float(submission['prediction'].skew()),
            'kurtosis': float(submission['prediction'].kurtosis())
        }

    # Performance metrics (if actual data available)
    if actual_data is not None and 'prediction' in submission.columns:
        try:
            # Merge with actual data
            merge_cols = [col for col in ['store_id', 'product_id', 'date'] if col in submission.columns and col in actual_data.columns]
            if merge_cols:
                merged = submission.merge(actual_data, on=merge_cols, how='inner')

                if 'actual' in merged.columns or 'total_sales' in merged.columns:
                    actual_col = 'actual' if 'actual' in merged.columns else 'total_sales'

                    # Calculate WMAPE
                    wmape_score = wmape(merged[actual_col], merged['prediction'])

                    # Calculate other metrics
                    mae = np.mean(np.abs(merged[actual_col] - merged['prediction']))
                    mape = np.mean(np.abs((merged[actual_col] - merged['prediction']) / merged[actual_col])) * 100
                    rmse = np.sqrt(np.mean((merged[actual_col] - merged['prediction']) ** 2))

                    metrics_result['performance_metrics'] = {
                        'wmape': float(wmape_score),
                        'mae': float(mae),
                        'mape': float(mape),
                        'rmse': float(rmse),
                        'correlation': float(np.corrcoef(merged[actual_col], merged['prediction'])[0, 1]),
                        'coverage': len(merged) / len(submission)
                    }

                    logger.info(f"WMAPE: {wmape_score:.2f}%")

        except Exception as e:
            logger.warning(f"Could not calculate performance metrics: {str(e)}")

    return metrics_result

def generate_validation_report(submission_path: str,
                             format_validation: Dict[str, Any],
                             business_validation: Dict[str, Any],
                             metrics: Dict[str, Any]) -> str:
    """Generate comprehensive validation report"""

    report_lines = [
        "[REPORT] SUBMISSION VALIDATION REPORT",
        "=" * 60,
        "",
        f"File: {submission_path}",
        f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "[FORMAT] FORMAT VALIDATION:",
        f"  Status: {'[PASS]' if format_validation['format_valid'] else '[FAIL]'}",
        f"  Total Rows: {format_validation['stats'].get('total_rows', 'N/A'):,}",
        f"  Total Columns: {format_validation['stats'].get('total_columns', 'N/A')}",
    ]

    if format_validation['issues']:
        report_lines.append("  [ISSUES] Issues:")
        for issue in format_validation['issues']:
            report_lines.append(f"    - {issue}")

    if format_validation['warnings']:
        report_lines.append("  [WARNINGS] Warnings:")
        for warning in format_validation['warnings']:
            report_lines.append(f"    - {warning}")

    report_lines.extend([
        "",
        "[BUSINESS] BUSINESS RULES VALIDATION:",
        f"  Status: {'[PASS]' if business_validation['business_valid'] else '[FAIL]'}",
    ])

    if business_validation['issues']:
        report_lines.append("  [ISSUES] Issues:")
        for issue in business_validation['issues']:
            report_lines.append(f"    - {issue}")

    if business_validation['warnings']:
        report_lines.append("  [WARNINGS] Warnings:")
        for warning in business_validation['warnings']:
            report_lines.append(f"    - {warning}")

    # Prediction/Quantidade statistics
    target_col = 'quantidade' if 'quantidade' in format_validation['stats'] else 'prediction'
    if target_col in format_validation['stats']:
        pred_stats = format_validation['stats'][target_col]
        report_lines.extend([
            "",
            f"[STATS] {target_col.upper()} STATISTICS:",
            f"  Count: {pred_stats['count']:,}",
            f"  Mean: {pred_stats['mean']:.2f}",
            f"  Std: {pred_stats['std']:.2f}",
            f"  Min: {pred_stats['min']:.2f}",
            f"  Max: {pred_stats['max']:.2f}",
            f"  Negative Count: {pred_stats['negative_count']:,}",
            f"  Zero Count: {pred_stats['zero_count']:,}",
        ])

    # Performance metrics
    if metrics['metrics_available'] and metrics['performance_metrics']:
        perf = metrics['performance_metrics']
        report_lines.extend([
            "",
            "[PERFORMANCE] PERFORMANCE METRICS:",
            f"  WMAPE: {perf['wmape']:.2f}%",
            f"  MAE: {perf['mae']:.2f}",
            f"  MAPE: {perf['mape']:.2f}%",
            f"  RMSE: {perf['rmse']:.2f}",
            f"  Correlation: {perf['correlation']:.3f}",
            f"  Coverage: {perf['coverage']:.1%}",
        ])

    # Overall assessment
    overall_status = format_validation['format_valid'] and business_validation['business_valid']

    report_lines.extend([
        "",
        "[OVERALL] OVERALL ASSESSMENT:",
        f"  Status: {'[READY] READY FOR SUBMISSION' if overall_status else '[FIXES] REQUIRES FIXES'}",
    ])

    if not overall_status:
        report_lines.append("  Action Required: Fix all issues before submitting")
    elif format_validation['warnings'] or business_validation['warnings']:
        report_lines.append("  Recommendation: Review warnings before submitting")
    else:
        report_lines.append("  Result: Submission appears ready")

    report_lines.extend([
        "",
        "=" * 60
    ])

    return "\n".join(report_lines)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validate competition submission file")

    # Required arguments
    parser.add_argument("submission_file", type=str,
                       help="Path to submission file (CSV or Parquet)")

    # Optional arguments
    parser.add_argument("--actual-data", "-a", type=str,
                       help="Path to actual data for performance calculation")
    parser.add_argument("--output-report", "-o", type=str,
                       help="Output file for validation report")
    parser.add_argument("--format", "-f", type=str, choices=["text", "json", "both"], default="text",
                       help="Output format for report")

    # Configuration
    parser.add_argument("--config", "-c", type=str,
                       help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    # Validation options
    parser.add_argument("--skip-business-rules", action="store_true",
                       help="Skip business rules validation")
    parser.add_argument("--required-columns", type=str, nargs="+",
                       default=["store_id", "product_id", "date", "prediction"],
                       help="Required columns in submission")
    parser.add_argument("--hackathon-format", action="store_true",
                       help="Validate as hackathon format (semana;pdv;produto;quantidade)")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("[START] Starting submission validation")
    logger.info(f"Submission file: {args.submission_file}")

    try:
        # Load submission file
        submission = load_submission_file(args.submission_file, args.hackathon_format)

        # Load actual data if provided
        actual_data = None
        if args.actual_data:
            logger.info(f"Loading actual data from {args.actual_data}")
            actual_data = pd.read_csv(args.actual_data) if args.actual_data.endswith('.csv') else pd.read_parquet(args.actual_data)

        # Override required columns for hackathon format
        if args.hackathon_format:
            args.required_columns = ["semana", "pdv", "produto", "quantidade"]
            logger.info("[HACKATHON] Using hackathon format validation")

        # Perform validations
        format_validation = validate_submission_format(submission, args.required_columns, args.hackathon_format)

        business_validation = {'business_valid': True, 'issues': [], 'warnings': [], 'stats': {}}
        if not args.skip_business_rules and not args.hackathon_format:
            business_validation = validate_business_rules(submission)
        elif args.hackathon_format:
            logger.info("[HACKATHON] Skipping business rules validation for hackathon format")

        metrics = calculate_submission_metrics(submission, actual_data)

        # Generate report
        text_report = generate_validation_report(
            args.submission_file, format_validation, business_validation, metrics
        )

        # Output results
        if args.format in ["text", "both"]:
            if args.output_report:
                output_file = args.output_report if args.output_report.endswith('.txt') else f"{args.output_report}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text_report)
                logger.info(f"Text report saved to {output_file}")

            # Print to console
            print(text_report)

        if args.format in ["json", "both"]:
            json_result = {
                'submission_file': args.submission_file,
                'validation_timestamp': datetime.now().isoformat(),
                'format_validation': format_validation,
                'business_validation': business_validation,
                'metrics': metrics
            }

            if args.output_report:
                json_file = args.output_report if args.output_report.endswith('.json') else f"{args.output_report}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, indent=2, default=str)
                logger.info(f"JSON report saved to {json_file}")

        # Exit code based on validation result
        overall_status = format_validation['format_valid'] and business_validation['business_valid']

        if overall_status:
            logger.info("[OK] Submission validation completed - READY FOR SUBMISSION")
            return 0
        else:
            logger.error("[FAIL] Submission validation failed - REQUIRES FIXES")
            return 1

    except KeyboardInterrupt:
        logger.info("[INTERRUPT] Validation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"[ERROR] Validation failed with error: {str(e)}")
        logger.debug("Full error traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())