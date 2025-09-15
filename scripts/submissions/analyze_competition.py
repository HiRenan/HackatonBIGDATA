#!/usr/bin/env python3
"""
Phase 7: Competitive Analysis Script
Advanced competitive intelligence and leaderboard analysis tool
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime
import json
from typing import Optional, Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.submissions.leaderboard_analyzer import LeaderboardAnalyzer, analyze_competition
from src.submissions.timeline_manager import TimelineManager, create_competition_timeline
from src.utils.logging import setup_logger
from src.utils.config import get_config_manager

def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging for script"""
    return setup_logger("analyze_competition", level=log_level, performance=True)

def load_leaderboard_data(file_path: str) -> pd.DataFrame:
    """Load leaderboard data from file"""
    logger = logging.getLogger("analyze_competition")

    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        else:
            # Try to determine format
            try:
                data = pd.read_csv(file_path)
            except:
                data = pd.read_json(file_path)

        logger.info(f"Loaded leaderboard data: {len(data)} entries")

        # Validate required columns
        required_columns = ['rank', 'team_name', 'score']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            # Try to infer columns
            if 'team' in data.columns and 'team_name' not in data.columns:
                data['team_name'] = data['team']
            if 'position' in data.columns and 'rank' not in data.columns:
                data['rank'] = data['position']

        return data

    except Exception as e:
        logger.error(f"Failed to load leaderboard data: {str(e)}")
        raise

def create_mock_leaderboard(num_teams: int = 50, our_rank: int = 20, our_score: float = 18.5) -> pd.DataFrame:
    """Create mock leaderboard for testing"""
    logger = logging.getLogger("analyze_competition")
    logger.info(f"Creating mock leaderboard with {num_teams} teams")

    # Generate realistic WMAPE scores
    np.random.seed(42)

    scores = []
    # Top performers (good scores)
    scores.extend(np.random.uniform(8.0, 12.0, 5))
    # Good performers
    scores.extend(np.random.uniform(12.0, 16.0, 15))
    # Average performers
    scores.extend(np.random.uniform(16.0, 22.0, 20))
    # Below average
    scores.extend(np.random.uniform(22.0, 30.0, 10))

    # Ensure we have enough scores
    while len(scores) < num_teams:
        scores.append(np.random.uniform(15.0, 25.0))

    scores = sorted(scores)[:num_teams]

    # Insert our team at specified rank
    if our_rank <= num_teams:
        scores[our_rank - 1] = our_score

    leaderboard = pd.DataFrame({
        'rank': range(1, num_teams + 1),
        'team_name': [f'Team_{i}' if i != our_rank else 'our_team' for i in range(1, num_teams + 1)],
        'score': scores,
        'submissions': np.random.randint(1, 6, num_teams)
    })

    return leaderboard

def analyze_competitive_landscape(leaderboard_data: pd.DataFrame,
                                team_name: str,
                                current_score: Optional[float] = None) -> Dict[str, Any]:
    """Perform comprehensive competitive analysis"""
    logger = logging.getLogger("analyze_competition")

    logger.info(f"🏆 Analyzing competitive landscape for {team_name}")

    # Perform analysis
    intelligence = analyze_competition(leaderboard_data, team_name, current_score)

    # Extract key insights
    position = intelligence.position_analysis
    gaps = intelligence.gap_analysis
    trends = intelligence.trend_analysis

    analysis_result = {
        'timestamp': datetime.now().isoformat(),
        'team_name': team_name,
        'current_position': {
            'rank': position.current_rank,
            'score': position.current_score,
            'percentile': position.percentile,
            'competitive_zone': position.competitive_zone,
            'total_teams': position.total_teams
        },
        'gaps': {
            'gap_to_top': position.gap_to_top,
            'gap_to_next': position.gap_to_next,
            'gap_to_top_3': gaps.gap_to_top_3,
            'gap_to_top_10': gaps.gap_to_top_10,
            'points_to_advance': position.points_to_advance
        },
        'strategic_analysis': {
            'recommended_target': gaps.recommended_target,
            'achievability_score': gaps.achievability_score,
            'improvement_needed': gaps.improvement_needed,
            'strategic_recommendations': intelligence.strategic_recommendations,
            'risk_assessment': intelligence.risk_assessment
        },
        'competitive_trends': trends
    }

    logger.info(f"✅ Analysis complete - Rank {position.current_rank} of {position.total_teams}")

    return analysis_result

def generate_strategic_report(analysis: Dict[str, Any]) -> str:
    """Generate human-readable strategic report"""
    position = analysis['current_position']
    gaps = analysis['gaps']
    strategy = analysis['strategic_analysis']

    report_lines = [
        "🏆 COMPETITIVE ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Team: {analysis['team_name']}",
        f"Analysis Date: {analysis['timestamp'][:19]}",
        "",
        "📊 CURRENT POSITION:",
        f"  Rank: {position['rank']} of {position['total_teams']} ({position['percentile']:.1f}th percentile)",
        f"  Score: {position['score']:.2f}",
        f"  Competitive Zone: {position['competitive_zone'].title()}",
        "",
        "🎯 COMPETITIVE GAPS:",
        f"  Gap to Leader: {gaps['gap_to_top']:.2f} points",
        f"  Gap to Next Position: {gaps['gap_to_next']:.2f} points",
        f"  Gap to Top 3: {gaps['gap_to_top_3']:.2f} points",
        f"  Gap to Top 10: {gaps['gap_to_top_10']:.2f} points",
        "",
        "📈 STRATEGIC OUTLOOK:",
        f"  Recommended Target Rank: {strategy['recommended_target']}",
        f"  Achievability Score: {strategy['achievability_score']:.2f}",
        f"  Points Needed to Advance: {gaps['points_to_advance']:.2f}",
        "",
        "🎲 IMPROVEMENT TARGETS:",
    ]

    for target, improvement in strategy['improvement_needed'].items():
        report_lines.append(f"  {target.replace('_', ' ').title()}: {improvement:.2f} points")

    report_lines.extend([
        "",
        "💡 STRATEGIC RECOMMENDATIONS:",
    ])

    for i, rec in enumerate(strategy['strategic_recommendations'][:8], 1):
        report_lines.append(f"  {i}. {rec}")

    report_lines.extend([
        "",
        "⚠️ RISK FACTORS:",
    ])

    for risk_type, risk_value in strategy['risk_assessment'].items():
        report_lines.append(f"  {risk_type.replace('_', ' ').title()}: {risk_value:.2f}")

    report_lines.extend([
        "",
        "📊 COMPETITIVE INSIGHTS:",
        f"  Score Distribution Mean: {analysis['competitive_trends']['score_distribution']['mean']:.2f}",
        f"  Score Distribution Std: {analysis['competitive_trends']['score_distribution']['std']:.2f}",
        f"  Top 10 Score Range: {analysis['competitive_trends']['score_distribution']['top_10_range']:.2f}",
        f"  Competitive Intensity: {analysis['competitive_trends']['competitive_intensity']:.2f}",
        "",
        "=" * 60
    ])

    return "\n".join(report_lines)

def analyze_timeline_implications(analysis: Dict[str, Any],
                                competition_days_remaining: int) -> List[str]:
    """Analyze timeline implications"""
    position = analysis['current_position']
    gaps = analysis['gaps']
    strategy = analysis['strategic_analysis']

    implications = []

    # Time pressure analysis
    if competition_days_remaining <= 3:
        if position['rank'] > 10:
            implications.append("⚠️ CRITICAL: Limited time for major improvements - focus on execution")
        implications.append("🔥 URGENT: Final optimization window - implement highest-impact changes only")

    elif competition_days_remaining <= 7:
        if gaps['gap_to_top_3'] < 2.0:
            implications.append("🎯 OPPORTUNITY: Close to top 3 - aggressive optimization recommended")
        implications.append("⏰ MEDIUM TIME PRESSURE: Focus on proven strategies")

    else:
        if strategy['achievability_score'] > 0.6:
            implications.append("📈 GOOD TIMING: Sufficient time for comprehensive improvements")
        implications.append("🚀 FULL OPTIMIZATION: Time available for experimental approaches")

    # Position-specific implications
    if position['competitive_zone'] == 'leader':
        implications.append("👑 LEADER STRATEGY: Conservative approach to maintain position")
    elif position['competitive_zone'] == 'contender':
        implications.append("🥈 CONTENDER STRATEGY: Balanced risk-taking for advancement")
    else:
        implications.append("🚀 CATCH-UP STRATEGY: Aggressive improvements needed")

    return implications

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Analyze competitive landscape and generate strategic insights")

    # Data input arguments
    parser.add_argument("--leaderboard", "-l", type=str,
                       help="Path to leaderboard data file (CSV or JSON)")
    parser.add_argument("--team-name", "-t", type=str, default="our_team",
                       help="Our team name in the leaderboard")
    parser.add_argument("--current-score", "-s", type=float,
                       help="Our current score (if not in leaderboard)")

    # Mock data arguments
    parser.add_argument("--mock-teams", type=int, default=50,
                       help="Number of teams in mock leaderboard")
    parser.add_argument("--mock-rank", type=int, default=20,
                       help="Our rank in mock leaderboard")
    parser.add_argument("--mock-score", type=float, default=18.5,
                       help="Our score in mock leaderboard")

    # Analysis arguments
    parser.add_argument("--competition-days-remaining", "-d", type=int, default=7,
                       help="Days remaining in competition")
    parser.add_argument("--output-dir", "-o", type=str, default="analysis_reports",
                       help="Output directory for analysis reports")

    # Configuration arguments
    parser.add_argument("--config", "-c", type=str,
                       help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    # Output format
    parser.add_argument("--format", "-f", type=str, choices=["text", "json", "both"], default="both",
                       help="Output format for analysis report")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("🏆 Starting competitive analysis")

    try:
        # Load or create leaderboard data
        if args.leaderboard:
            leaderboard_data = load_leaderboard_data(args.leaderboard)
        else:
            logger.info("No leaderboard file provided, creating mock data")
            leaderboard_data = create_mock_leaderboard(
                num_teams=args.mock_teams,
                our_rank=args.mock_rank,
                our_score=args.mock_score
            )

        # Perform analysis
        analysis = analyze_competitive_landscape(
            leaderboard_data=leaderboard_data,
            team_name=args.team_name,
            current_score=args.current_score
        )

        # Timeline implications
        timeline_implications = analyze_timeline_implications(
            analysis, args.competition_days_remaining
        )
        analysis['timeline_implications'] = timeline_implications

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Generate and save reports
        if args.format in ["text", "both"]:
            # Text report
            text_report = generate_strategic_report(analysis)

            # Add timeline implications
            text_report += "\n\n⏰ TIMELINE IMPLICATIONS:\n"
            for imp in timeline_implications:
                text_report += f"  • {imp}\n"

            text_file = output_dir / f"competitive_analysis_{timestamp}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_report)

            logger.info(f"Text report saved to {text_file}")

            # Print summary to console
            print("\n" + text_report)

        if args.format in ["json", "both"]:
            # JSON report
            json_file = output_dir / f"competitive_analysis_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, default=str)

            logger.info(f"JSON report saved to {json_file}")

        # Print key insights
        position = analysis['current_position']
        gaps = analysis['gaps']

        print(f"\n🎯 KEY INSIGHTS:")
        print(f"  Current Rank: {position['rank']} of {position['total_teams']}")
        print(f"  Competitive Zone: {position['competitive_zone']}")
        print(f"  Gap to Top 3: {gaps['gap_to_top_3']:.2f} points")
        print(f"  Recommended Target: Rank {analysis['strategic_analysis']['recommended_target']}")

        print(f"\n⏰ TIMELINE ASSESSMENT ({args.competition_days_remaining} days remaining):")
        for imp in timeline_implications[:3]:  # Show top 3 implications
            print(f"  • {imp}")

        logger.info("✅ Competitive analysis completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("❌ Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Analysis failed with error: {str(e)}")
        logger.debug("Full error traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())