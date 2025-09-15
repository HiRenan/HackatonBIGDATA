#!/usr/bin/env python3
"""
Phase 7: Strategic Model Submission Script
Command-line interface for executing strategic submissions with full pipeline
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime
import json
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.submissions.strategy import SubmissionStrategyFactory, SubmissionPhase
from src.submissions.submission_pipeline import create_submission_pipeline
from src.submissions.timeline_manager import TimelineManager
from src.submissions.leaderboard_analyzer import LeaderboardAnalyzer
from src.data.loaders import load_competition_data
from src.utils.logging import setup_logger
from src.utils.config import get_config_manager

def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging for script"""
    return setup_logger("submit_model", level=log_level, performance=True)

def load_data(data_path: str, sample_size: Optional[int] = None) -> tuple:
    """Load competition data"""
    logger = logging.getLogger("submit_model")
    logger.info(f"Loading data from {data_path}")

    config = {}
    if sample_size:
        config['sample_size'] = sample_size

    try:
        transactions, products, stores = load_competition_data(data_path, config)

        # Split into train/test (mock split for demo)
        # In real scenario, test data would be separate
        split_date = transactions['date'].quantile(0.8)
        train_data = transactions[transactions['date'] <= split_date]
        test_data = transactions[transactions['date'] > split_date].drop('total_sales', axis=1)

        logger.info(f"Data loaded - Train: {train_data.shape}, Test: {test_data.shape}")
        return train_data, test_data, products, stores

    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def execute_submission(strategy_type: str,
                      train_data: pd.DataFrame,
                      test_data: pd.DataFrame,
                      config: Dict[str, Any],
                      leaderboard_data: Optional[str] = None,
                      team_name: str = "our_team") -> Dict[str, Any]:
    """Execute strategic submission"""
    logger = logging.getLogger("submit_model")

    logger.info(f"🎯 Executing {strategy_type} submission strategy")

    # Create submission strategy
    strategy = SubmissionStrategyFactory.create(strategy_type, config.get('strategy', {}))

    # Create submission pipeline
    pipeline = create_submission_pipeline(config.get('pipeline', {}))

    # Prepare context
    context = {
        'team_name': team_name,
        'strategy_type': strategy_type
    }

    # Add leaderboard data if available
    if leaderboard_data:
        try:
            if Path(leaderboard_data).exists():
                lb_data = pd.read_csv(leaderboard_data)
                context['leaderboard_data'] = lb_data.to_dict('records')
                logger.info(f"Loaded leaderboard data with {len(lb_data)} entries")
        except Exception as e:
            logger.warning(f"Failed to load leaderboard data: {str(e)}")

    # Execute pipeline
    try:
        result = pipeline.execute_submission_pipeline(
            submission_strategy=strategy,
            train_data=train_data,
            test_data=test_data,
            **context
        )

        logger.info(f"✅ Submission pipeline completed successfully")
        logger.info(f"   Steps completed: {result['steps_completed']}")
        logger.info(f"   Total execution time: {result['total_execution_time']:.1f}s")

        return result

    except Exception as e:
        logger.error(f"❌ Submission pipeline failed: {str(e)}")
        raise

def print_submission_summary(result: Dict[str, Any]):
    """Print submission summary"""
    print("\n" + "="*60)
    print("🎯 SUBMISSION SUMMARY")
    print("="*60)

    if result['success']:
        print("✅ Status: SUCCESS")

        if result['final_submission']:
            submission = result['final_submission']
            print(f"📊 Submission ID: {submission.submission_id}")
            print(f"🏷️  Model Type: {submission.model_type}")
            print(f"📈 Validation Score: {submission.validation_score:.2f}")
            print(f"📝 Predictions: {len(submission.predictions):,} rows")

        if result['risk_assessment']:
            risk = result['risk_assessment']
            print(f"⚠️  Risk Level: {risk.risk_level}")
            print(f"🎲 Risk Score: {risk.overall_risk:.2f}")
            print(f"🔍 Confidence: {risk.confidence:.2f}")

        if result['competitive_analysis']:
            analysis = result['competitive_analysis']
            pos = analysis.position_analysis
            print(f"🏆 Estimated Rank: {pos.current_rank}")
            print(f"📊 Current Score: {pos.current_score:.2f}")
            print(f"🎯 Gap to Top: {analysis.gap_analysis.gap_to_top_3:.2f}")

    else:
        print("❌ Status: FAILED")
        print(f"🚫 Failed Steps: {result['steps_failed']}")

    print(f"⏱️  Total Time: {result['total_execution_time']:.1f}s")
    print("="*60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Execute strategic model submission")

    # Required arguments
    parser.add_argument("strategy", choices=['baseline', 'single_model', 'ensemble', 'optimized_ensemble', 'final'],
                       help="Submission strategy to execute")
    parser.add_argument("--data-path", "-d", type=str, required=True,
                       help="Path to competition data directory")

    # Optional arguments
    parser.add_argument("--config", "-c", type=str,
                       help="Path to configuration file")
    parser.add_argument("--leaderboard", "-l", type=str,
                       help="Path to leaderboard CSV file")
    parser.add_argument("--team-name", "-t", type=str, default="our_team",
                       help="Team name for competitive analysis")
    parser.add_argument("--sample-size", "-s", type=int,
                       help="Sample size for development/testing")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--output-dir", "-o", type=str, default="submissions",
                       help="Output directory for submission files")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run pipeline without saving submission")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("🚀 Starting strategic model submission")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Data path: {args.data_path}")

    try:
        # Load configuration
        config_manager = get_config_manager()
        if args.config:
            config = config_manager.load_config(config_file=args.config)
        else:
            config = config_manager.load_config("development")

        # Override config with CLI arguments
        if args.sample_size:
            config.setdefault('data_loading', {})['sample_size'] = args.sample_size

        if args.dry_run:
            config.setdefault('pipeline', {}).setdefault('steps', {}).setdefault('submission_execution', {})['enabled'] = False

        logger.info(f"Configuration loaded: {config.get('environment', 'unknown')}")

        # Load data
        train_data, test_data, products, stores = load_data(
            args.data_path, args.sample_size
        )

        # Execute submission
        result = execute_submission(
            strategy_type=args.strategy,
            train_data=train_data,
            test_data=test_data,
            config=config,
            leaderboard_data=args.leaderboard,
            team_name=args.team_name
        )

        # Print results
        print_submission_summary(result)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        result_file = output_dir / f"submission_result_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Prepare JSON-serializable result
        json_result = {
            'strategy': args.strategy,
            'timestamp': datetime.now().isoformat(),
            'success': result['success'],
            'total_execution_time': result['total_execution_time'],
            'steps_completed': result['steps_completed'],
            'steps_failed': result['steps_failed']
        }

        if result['final_submission']:
            submission = result['final_submission']
            json_result['submission'] = {
                'submission_id': submission.submission_id,
                'model_type': submission.model_type,
                'validation_score': submission.validation_score,
                'prediction_count': len(submission.predictions)
            }

        if result['risk_assessment']:
            risk = result['risk_assessment']
            json_result['risk_assessment'] = {
                'overall_risk': risk.overall_risk,
                'risk_level': risk.risk_level,
                'confidence': risk.confidence,
                'recommendations_count': len(risk.recommendations)
            }

        with open(result_file, 'w') as f:
            json.dump(json_result, f, indent=2)

        logger.info(f"Results saved to {result_file}")

        if result['success']:
            logger.info("✅ Submission completed successfully!")
            return 0
        else:
            logger.error("❌ Submission failed!")
            return 1

    except KeyboardInterrupt:
        logger.info("❌ Submission interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Submission failed with error: {str(e)}")
        logger.debug("Full error traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())