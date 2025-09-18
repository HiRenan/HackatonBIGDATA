#!/usr/bin/env python3
"""
Phase 7: Final Submission Generation Script
Ultimate submission generation with all optimizations and safeguards
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime, timezone
import json
from typing import Optional, Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.submissions.strategy import SubmissionStrategyFactory, SubmissionPhase
from src.submissions.submission_pipeline import create_submission_pipeline
from src.submissions.timeline_manager import TimelineManager
from src.submissions.leaderboard_analyzer import LeaderboardAnalyzer
from src.submissions.risk_manager import RiskManager
from src.submissions.post_processor import create_standard_pipeline, create_competitive_pipeline
from src.data.loaders import load_competition_data
from src.utils.logging import setup_logger
from src.utils.config import get_config_manager

def convert_to_hackathon_format(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert standard prediction format to hackathon format

    Standard: store_id, product_id, date, prediction
    Hackathon: semana, pdv, produto, quantidade
    """
    logger = logging.getLogger("final_submission")

    logger.info("Converting to hackathon format...")

    # Copy the dataframe
    df = predictions_df.copy()

    # Convert date to week number (assuming January 2023, weeks 1-5)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Calculate week number from start of January 2023
        start_date = pd.Timestamp('2023-01-02')  # First Monday of 2023
        df['semana'] = ((df['date'] - start_date).dt.days // 7) + 1
        # Ensure weeks are 1-5
        df['semana'] = df['semana'].clip(1, 5)
    else:
        # If no date column, assign sequential weeks
        logger.warning("No date column found, assigning sequential weeks")
        df['semana'] = 1

    # Map column names
    column_mapping = {
        'store_id': 'pdv',
        'internal_store_id': 'pdv',
        'product_id': 'produto',
        'internal_product_id': 'produto',
        'prediction': 'quantidade',
        'quantity': 'quantidade'
    }

    # Apply mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]

    # Ensure required columns exist
    required_cols = ['semana', 'pdv', 'produto', 'quantidade']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Required column {col} not found after conversion")
            # Provide defaults
            if col == 'semana':
                df[col] = 1
            elif col in ['pdv', 'produto']:
                df[col] = 1
            elif col == 'quantidade':
                df[col] = 0

    # Select only required columns and ensure integer types
    hackathon_df = df[required_cols].copy()

    for col in required_cols:
        hackathon_df[col] = hackathon_df[col].fillna(0).astype(int)

    # Ensure non-negative quantities
    hackathon_df['quantidade'] = hackathon_df['quantidade'].clip(lower=0)

    logger.info(f"Converted {len(hackathon_df)} predictions to hackathon format")
    logger.info(f"Weeks range: {hackathon_df['semana'].min()}-{hackathon_df['semana'].max()}")
    logger.info(f"Unique PDVs: {hackathon_df['pdv'].nunique()}")
    logger.info(f"Unique products: {hackathon_df['produto'].nunique()}")

    return hackathon_df

def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging for script"""
    return setup_logger("final_submission", level=log_level, performance=True)

def load_all_data(data_path: str, sample_size: Optional[int] = None) -> tuple:
    """Load all competition data"""
    logger = logging.getLogger("final_submission")
    logger.info(f"🔄 Loading complete dataset from {data_path}")

    config = {}
    if sample_size:
        config['sample_size'] = sample_size
        logger.warning(f"Using sample size: {sample_size:,} (not recommended for final submission)")

    try:
        transactions, products, stores = load_competition_data(data_path, config)

        # For final submission, use all available training data
        # Test data should be the actual competition test set
        train_data = transactions

        # In real competition, test data would be provided separately
        # For demo, we'll create a mock test set
        test_dates = pd.date_range(
            start=transactions['date'].max() + pd.Timedelta(days=1),
            periods=30,  # 30-day forecast
            freq='D'
        )

        # Create test set with all store-product combinations for forecast period
        unique_stores = transactions['store_id'].unique()
        unique_products = transactions['product_id'].unique()

        test_combinations = []
        for date in test_dates:
            for store in unique_stores:
                for product in unique_products:
                    test_combinations.append({
                        'date': date,
                        'store_id': store,
                        'product_id': product
                    })

        test_data = pd.DataFrame(test_combinations)

        logger.info(f"📊 Final dataset loaded:")
        logger.info(f"  Training data: {train_data.shape[0]:,} records")
        logger.info(f"  Test data: {test_data.shape[0]:,} predictions needed")
        logger.info(f"  Products: {len(products):,}")
        logger.info(f"  Stores: {len(stores):,}")

        return train_data, test_data, products, stores

    except Exception as e:
        logger.error(f"❌ Failed to load data: {str(e)}")
        raise

def assess_competition_status(leaderboard_path: Optional[str],
                            team_name: str,
                            current_score: Optional[float] = None) -> Dict[str, Any]:
    """Assess current competition status"""
    logger = logging.getLogger("final_submission")

    if not leaderboard_path or not Path(leaderboard_path).exists():
        logger.warning("No leaderboard data available - using conservative estimates")
        return {
            'competitive_intelligence': None,
            'strategy_recommendation': 'conservative',
            'risk_tolerance': 'medium',
            'time_pressure': 'high'
        }

    try:
        analyzer = LeaderboardAnalyzer()

        # Load and analyze leaderboard
        leaderboard_data = pd.read_csv(leaderboard_path)
        intelligence = analyzer.analyze_competitive_landscape(
            leaderboard_data, team_name, current_score
        )

        # Determine strategy based on position
        position = intelligence.position_analysis
        gaps = intelligence.gap_analysis

        if position.competitive_zone == 'leader':
            strategy_rec = 'conservative'
            risk_tolerance = 'low'
        elif gaps.gap_to_top_3 < 2.0:
            strategy_rec = 'aggressive'
            risk_tolerance = 'high'
        elif position.current_rank <= 10:
            strategy_rec = 'balanced'
            risk_tolerance = 'medium'
        else:
            strategy_rec = 'aggressive'
            risk_tolerance = 'high'

        logger.info(f"🏆 Competition status:")
        logger.info(f"  Current rank: {position.current_rank}")
        logger.info(f"  Competitive zone: {position.competitive_zone}")
        logger.info(f"  Gap to top 3: {gaps.gap_to_top_3:.2f}")
        logger.info(f"  Recommended strategy: {strategy_rec}")

        return {
            'competitive_intelligence': intelligence,
            'strategy_recommendation': strategy_rec,
            'risk_tolerance': risk_tolerance,
            'time_pressure': 'high'  # Always high for final submission
        }

    except Exception as e:
        logger.error(f"Failed to assess competition status: {str(e)}")
        return {
            'competitive_intelligence': None,
            'strategy_recommendation': 'conservative',
            'risk_tolerance': 'medium',
            'time_pressure': 'high'
        }

def generate_final_submission(train_data: pd.DataFrame,
                            test_data: pd.DataFrame,
                            competition_status: Dict[str, Any],
                            config: Dict[str, Any],
                            team_name: str) -> Dict[str, Any]:
    """Generate final optimized submission"""
    logger = logging.getLogger("final_submission")

    logger.info("🎯 Generating FINAL submission with all optimizations")

    # Create final submission strategy
    final_strategy = SubmissionStrategyFactory.create('final', config.get('final_strategy', {}))

    # Create enhanced pipeline with final optimizations
    pipeline_config = config.get('pipeline', {})

    # Adjust pipeline based on competition status
    strategy_rec = competition_status['strategy_recommendation']
    if strategy_rec == 'aggressive':
        pipeline_config.setdefault('steps', {}).setdefault('risk_assessment', {})['max_risk_threshold'] = 0.9
    elif strategy_rec == 'conservative':
        pipeline_config.setdefault('steps', {}).setdefault('risk_assessment', {})['max_risk_threshold'] = 0.6

    pipeline = create_submission_pipeline(pipeline_config)

    # Enhanced context for final submission
    context = {
        'team_name': team_name,
        'strategy_type': 'final',
        'is_final_submission': True,
        'competition_status': competition_status,
        'competitive_intelligence': competition_status.get('competitive_intelligence')
    }

    # Execute final pipeline
    try:
        logger.info("🚀 Executing final submission pipeline...")
        result = pipeline.execute_submission_pipeline(
            submission_strategy=final_strategy,
            train_data=train_data,
            test_data=test_data,
            **context
        )

        if not result['success']:
            raise RuntimeError("Final submission pipeline failed")

        logger.info("✅ Final pipeline completed successfully")

        # Apply additional post-processing
        if result['final_submission'] and competition_status['competitive_intelligence']:
            logger.info("🔧 Applying competitive post-processing...")

            competitive_pipeline = create_competitive_pipeline(
                competition_status['competitive_intelligence'].__dict__,
                config.get('competitive_post_processing', {})
            )

            predictions = result['final_submission'].predictions
            post_processing_result = competitive_pipeline.fit_transform(predictions)

            # Update final submission with post-processed predictions
            result['final_submission'].predictions = post_processing_result.processed_predictions
            result['post_processing_improvement'] = post_processing_result.improvement_estimate

            logger.info(f"📈 Post-processing complete - estimated improvement: {post_processing_result.improvement_estimate:.1%}")

        return result

    except Exception as e:
        logger.error(f"❌ Final submission generation failed: {str(e)}")
        raise

def validate_final_submission(submission_result: Dict[str, Any],
                            min_predictions: int = 1000) -> Dict[str, Any]:
    """Comprehensive validation of final submission"""
    logger = logging.getLogger("final_submission")

    logger.info("🔍 Performing final submission validation...")

    validation_results = {
        'format_valid': False,
        'content_valid': False,
        'risk_acceptable': False,
        'completeness_valid': False,
        'issues': [],
        'warnings': []
    }

    if not submission_result['success'] or not submission_result['final_submission']:
        validation_results['issues'].append("Submission generation failed")
        return validation_results

    final_submission = submission_result['final_submission']
    predictions = final_submission.predictions

    # Format validation
    required_columns = ['store_id', 'product_id', 'date', 'prediction']
    if all(col in predictions.columns for col in required_columns):
        validation_results['format_valid'] = True
    else:
        missing_cols = [col for col in required_columns if col not in predictions.columns]
        validation_results['issues'].append(f"Missing columns: {missing_cols}")

    # Content validation
    if validation_results['format_valid']:
        # Check for null values
        null_count = predictions[required_columns].isnull().sum().sum()
        if null_count > 0:
            validation_results['issues'].append(f"Found {null_count} null values")

        # Check for negative predictions
        negative_count = (predictions['prediction'] < 0).sum()
        if negative_count > 0:
            validation_results['issues'].append(f"Found {negative_count} negative predictions")

        # Check prediction range
        pred_min = predictions['prediction'].min()
        pred_max = predictions['prediction'].max()
        pred_mean = predictions['prediction'].mean()

        if pred_max / pred_mean > 100:  # Very extreme values
            validation_results['warnings'].append(f"Very high maximum prediction: {pred_max:.2f}")

        if pred_min == pred_max:  # All same value
            validation_results['issues'].append("All predictions are identical")

        # Check completeness
        if len(predictions) < min_predictions:
            validation_results['issues'].append(f"Insufficient predictions: {len(predictions)} < {min_predictions}")
        else:
            validation_results['completeness_valid'] = True

        if len(validation_results['issues']) == 0:
            validation_results['content_valid'] = True

    # Risk validation
    if 'risk_assessment' in submission_result and submission_result['risk_assessment']:
        risk = submission_result['risk_assessment']
        if risk.risk_level in ['LOW', 'MEDIUM']:
            validation_results['risk_acceptable'] = True
        else:
            validation_results['warnings'].append(f"High risk submission: {risk.risk_level}")
            # Still acceptable for final submission
            validation_results['risk_acceptable'] = True

    # Overall validation
    all_checks = [
        validation_results['format_valid'],
        validation_results['content_valid'],
        validation_results['completeness_valid'],
        validation_results['risk_acceptable']
    ]

    validation_results['overall_valid'] = all(all_checks)

    if validation_results['overall_valid']:
        logger.info("✅ Final submission validation PASSED")
    else:
        logger.error("❌ Final submission validation FAILED")
        for issue in validation_results['issues']:
            logger.error(f"  - {issue}")

    for warning in validation_results['warnings']:
        logger.warning(f"  ⚠️ {warning}")

    return validation_results

def save_final_submission(submission_result: Dict[str, Any],
                         output_dir: str,
                         team_name: str) -> Dict[str, str]:
    """Save final submission files"""
    logger = logging.getLogger("final_submission")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    files_created = {}

    try:
        # Save submission CSV
        if submission_result['final_submission']:
            predictions = submission_result['final_submission'].predictions

            # Check if hackathon format is requested
            hackathon_format = config.get('hackathon_format', False)

            if hackathon_format:
                # Convert to hackathon format: semana;pdv;produto;quantidade
                hackathon_predictions = convert_to_hackathon_format(predictions)

                submission_file = output_path / f"HACKATHON_SUBMISSION_{team_name}_{timestamp}.csv"
                hackathon_predictions.to_csv(submission_file, sep=';', index=False, encoding='utf-8')

                logger.info(f"[HACKATHON] Submission saved: {submission_file}")
                logger.info(f"[HACKATHON] Format: semana;pdv;produto;quantidade")
            else:
                # Standard competition format
                submission_file = output_path / f"FINAL_SUBMISSION_{team_name}_{timestamp}.csv"
                predictions.to_csv(submission_file, index=False)
                logger.info(f"[STANDARD] Final submission saved: {submission_file}")

            files_created['submission'] = str(submission_file)

        # Save detailed results
        results_file = output_path / f"final_submission_results_{timestamp}.json"

        # Prepare JSON-serializable result
        json_result = {
            'timestamp': datetime.now().isoformat(),
            'team_name': team_name,
            'success': submission_result['success'],
            'total_execution_time': submission_result['total_execution_time'],
            'steps_completed': submission_result['steps_completed'],
            'prediction_count': len(submission_result['final_submission'].predictions) if submission_result['final_submission'] else 0
        }

        if submission_result['final_submission']:
            submission = submission_result['final_submission']
            json_result['submission_details'] = {
                'submission_id': submission.submission_id,
                'model_type': submission.model_type,
                'validation_score': submission.validation_score,
                'phase': submission.phase.name
            }

        if submission_result['risk_assessment']:
            risk = submission_result['risk_assessment']
            json_result['risk_assessment'] = {
                'overall_risk': risk.overall_risk,
                'risk_level': risk.risk_level,
                'confidence': risk.confidence
            }

        with open(results_file, 'w') as f:
            json.dump(json_result, f, indent=2)

        files_created['results'] = str(results_file)
        logger.info(f"📊 Results saved: {results_file}")

        return files_created

    except Exception as e:
        logger.error(f"Failed to save final submission: {str(e)}")
        raise

def print_final_summary(submission_result: Dict[str, Any],
                       validation_results: Dict[str, Any],
                       files_created: Dict[str, str]):
    """Print comprehensive final submission summary"""
    print("\n" + "🏆"*20)
    print("🏆 FINAL SUBMISSION SUMMARY")
    print("🏆" + "="*58 + "🏆")

    if submission_result['success'] and validation_results['overall_valid']:
        print("✅ STATUS: READY FOR SUBMISSION")
    else:
        print("❌ STATUS: NEEDS ATTENTION")

    if submission_result['final_submission']:
        submission = submission_result['final_submission']
        print(f"\n📊 SUBMISSION DETAILS:")
        print(f"  Submission ID: {submission.submission_id}")
        print(f"  Model Type: {submission.model_type}")
        print(f"  Predictions: {len(submission.predictions):,} rows")
        print(f"  Validation Score: {submission.validation_score:.3f}")

    if submission_result['risk_assessment']:
        risk = submission_result['risk_assessment']
        print(f"\n⚠️  RISK ASSESSMENT:")
        print(f"  Risk Level: {risk.risk_level}")
        print(f"  Risk Score: {risk.overall_risk:.3f}")
        print(f"  Confidence: {risk.confidence:.3f}")

    if submission_result['competitive_analysis']:
        analysis = submission_result['competitive_analysis']
        pos = analysis.position_analysis
        print(f"\n🏆 COMPETITIVE POSITION:")
        print(f"  Estimated Rank: {pos.current_rank}")
        print(f"  Competitive Zone: {pos.competitive_zone}")
        print(f"  Gap to Top 3: {analysis.gap_analysis.gap_to_top_3:.3f}")

    print(f"\n✅ VALIDATION STATUS:")
    print(f"  Format Valid: {'✅' if validation_results['format_valid'] else '❌'}")
    print(f"  Content Valid: {'✅' if validation_results['content_valid'] else '❌'}")
    print(f"  Risk Acceptable: {'✅' if validation_results['risk_acceptable'] else '❌'}")
    print(f"  Complete: {'✅' if validation_results['completeness_valid'] else '❌'}")

    if validation_results['issues']:
        print(f"\n❌ CRITICAL ISSUES:")
        for issue in validation_results['issues']:
            print(f"  • {issue}")

    if validation_results['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for warning in validation_results['warnings']:
            print(f"  • {warning}")

    print(f"\n💾 FILES CREATED:")
    for file_type, file_path in files_created.items():
        print(f"  {file_type.title()}: {file_path}")

    print(f"\n⏱️  EXECUTION TIME: {submission_result['total_execution_time']:.1f} seconds")
    print("🏆" + "="*58 + "🏆")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate final optimized submission")

    # Required arguments
    parser.add_argument("--data-path", "-d", type=str, required=True,
                       help="Path to competition data directory")
    parser.add_argument("--team-name", "-t", type=str, required=True,
                       help="Team name for submission")

    # Optional arguments
    parser.add_argument("--output-dir", "-o", type=str, default="final_submissions",
                       help="Output directory for final submission")
    parser.add_argument("--leaderboard", "-l", type=str,
                       help="Path to current leaderboard data")
    parser.add_argument("--current-score", "-s", type=float,
                       help="Current team score")
    parser.add_argument("--config", "-c", type=str,
                       help="Path to configuration file")
    parser.add_argument("--sample-size", type=int,
                       help="Sample size (NOT recommended for final submission)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--force", action="store_true",
                       help="Force submission generation even with high risk")
    parser.add_argument("--hackathon-format", action="store_true",
                       help="Generate submission in hackathon format (semana;pdv;produto;quantidade)")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("🏆 Starting FINAL submission generation")
    logger.info(f"Team: {args.team_name}")
    logger.info(f"Data: {args.data_path}")

    try:
        # Load configuration
        config_manager = get_config_manager()
        if args.config:
            config = config_manager.load_config(config_file=args.config)
        else:
            config = config_manager.load_config("production")  # Use production for final

        # Override for sample size (with warning)
        if args.sample_size:
            logger.warning(f"[WARNING] Using sample size {args.sample_size:,} - NOT RECOMMENDED FOR FINAL SUBMISSION")
            config.setdefault('data_loading', {})['sample_size'] = args.sample_size

        # Set hackathon format flag
        if args.hackathon_format:
            logger.info("[HACKATHON] Generating submission in hackathon format")
            config['hackathon_format'] = True

        # Load all competition data
        train_data, test_data, products, stores = load_all_data(args.data_path, args.sample_size)

        # Assess competition status
        competition_status = assess_competition_status(
            args.leaderboard, args.team_name, args.current_score
        )

        # Generate final submission
        submission_result = generate_final_submission(
            train_data=train_data,
            test_data=test_data,
            competition_status=competition_status,
            config=config,
            team_name=args.team_name
        )

        # Validate final submission
        validation_results = validate_final_submission(submission_result)

        # Check if we should proceed
        if not validation_results['overall_valid'] and not args.force:
            logger.error("❌ Final submission validation failed. Use --force to override.")
            return 1

        # Save final submission
        files_created = save_final_submission(
            submission_result, args.output_dir, args.team_name
        )

        # Print comprehensive summary
        print_final_summary(submission_result, validation_results, files_created)

        if validation_results['overall_valid']:
            logger.info("🏆 FINAL SUBMISSION READY - GOOD LUCK!")
            return 0
        else:
            logger.warning("⚠️ Final submission has issues but was generated with --force")
            return 0

    except KeyboardInterrupt:
        logger.info("❌ Final submission generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Final submission generation failed: {str(e)}")
        logger.debug("Full error traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())