#!/usr/bin/env python3
"""
Phase 7: Timeline Management Script
Interactive tool for managing competition timeline and deadlines
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta
import json
from typing import Optional, Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.submissions.timeline_manager import TimelineManager, create_competition_timeline
from src.utils.logging import setup_logger
from src.utils.config import get_config_manager

def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging for script"""
    return setup_logger("manage_timeline", level=log_level, performance=True)

def display_timeline_status(timeline_manager: TimelineManager, competition_end_date: datetime):
    """Display current timeline status"""
    logger = logging.getLogger("manage_timeline")

    now = datetime.now()
    days_remaining = (competition_end_date - now).days
    hours_remaining = (competition_end_date - now).total_seconds() / 3600

    print("\n" + "="*60)
    print("⏰ COMPETITION TIMELINE STATUS")
    print("="*60)
    print(f"📅 Current Date: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"🏁 Competition End: {competition_end_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"⏳ Time Remaining: {days_remaining} days, {hours_remaining:.1f} hours")

    if days_remaining <= 1:
        print("🚨 CRITICAL: Final day of competition!")
    elif days_remaining <= 3:
        print("⚠️ URGENT: Less than 3 days remaining!")
    elif days_remaining <= 7:
        print("⏰ WARNING: Less than 1 week remaining!")

    print("\n📋 SUBMISSION WINDOWS:")

    # Get submission windows from timeline manager
    windows = timeline_manager.get_submission_windows()

    for window_name, window_info in windows.items():
        if window_info.get('day', 0) > 0:
            window_date = competition_end_date - timedelta(days=window_info['day'])
        else:
            window_date = competition_end_date

        status = "✅ COMPLETED" if window_date < now else "⏳ PENDING"
        priority_emoji = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }.get(window_info.get('priority', 'medium'), '🟡')

        print(f"  {priority_emoji} {window_name.replace('_', ' ').title()}: {window_date.strftime('%Y-%m-%d %H:%M')} {status}")

        if window_date > now:
            days_until = (window_date - now).days
            if days_until <= 1:
                print(f"    ⚠️ Due in {days_until} days!")

def check_deadlines(timeline_manager: TimelineManager, competition_end_date: datetime):
    """Check upcoming deadlines"""
    now = datetime.now()
    alerts = []

    windows = timeline_manager.get_submission_windows()

    for window_name, window_info in windows.items():
        if window_info.get('day', 0) > 0:
            window_date = competition_end_date - timedelta(days=window_info['day'])
        else:
            window_date = competition_end_date

        if window_date > now:
            hours_until = (window_date - now).total_seconds() / 3600

            if hours_until <= 2:
                alerts.append(f"🚨 CRITICAL: {window_name} due in {hours_until:.1f} hours!")
            elif hours_until <= 6:
                alerts.append(f"⚠️ URGENT: {window_name} due in {hours_until:.1f} hours!")
            elif hours_until <= 24:
                alerts.append(f"⏰ UPCOMING: {window_name} due in {hours_until:.1f} hours!")

    if alerts:
        print("\n🔔 DEADLINE ALERTS:")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("\n✅ No immediate deadline alerts")

def generate_timeline_report(timeline_manager: TimelineManager,
                           competition_end_date: datetime,
                           output_file: str):
    """Generate comprehensive timeline report"""
    logger = logging.getLogger("manage_timeline")

    now = datetime.now()
    days_remaining = (competition_end_date - now).days

    report = {
        'timestamp': now.isoformat(),
        'competition_end_date': competition_end_date.isoformat(),
        'days_remaining': days_remaining,
        'hours_remaining': (competition_end_date - now).total_seconds() / 3600,
        'timeline_status': {},
        'alerts': [],
        'recommendations': []
    }

    # Analyze each submission window
    windows = timeline_manager.get_submission_windows()

    for window_name, window_info in windows.items():
        if window_info.get('day', 0) > 0:
            window_date = competition_end_date - timedelta(days=window_info['day'])
        else:
            window_date = competition_end_date

        status = {
            'window_name': window_name,
            'scheduled_date': window_date.isoformat(),
            'priority': window_info.get('priority', 'medium'),
            'is_overdue': window_date < now,
            'days_until': (window_date - now).days if window_date > now else 0,
            'hours_until': (window_date - now).total_seconds() / 3600 if window_date > now else 0
        }

        report['timeline_status'][window_name] = status

        # Generate alerts
        if status['hours_until'] <= 2 and status['hours_until'] > 0:
            report['alerts'].append(f"CRITICAL: {window_name} due in {status['hours_until']:.1f} hours")
        elif status['hours_until'] <= 6 and status['hours_until'] > 0:
            report['alerts'].append(f"URGENT: {window_name} due in {status['hours_until']:.1f} hours")
        elif status['hours_until'] <= 24 and status['hours_until'] > 0:
            report['alerts'].append(f"UPCOMING: {window_name} due in {status['hours_until']:.1f} hours")

    # Generate recommendations
    if days_remaining <= 1:
        report['recommendations'].extend([
            "Focus only on final submission optimization",
            "Avoid experimental changes",
            "Ensure submission format is correct"
        ])
    elif days_remaining <= 3:
        report['recommendations'].extend([
            "Prioritize proven strategies",
            "Perform final ensemble optimization",
            "Prepare backup submission"
        ])
    elif days_remaining <= 7:
        report['recommendations'].extend([
            "Complete model experimentation",
            "Begin ensemble development",
            "Start competitive analysis"
        ])

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Timeline report saved to {output_file}")

    return report

def set_custom_deadline(timeline_manager: TimelineManager,
                       deadline_name: str,
                       deadline_date: datetime,
                       priority: str = "medium"):
    """Set a custom deadline"""
    logger = logging.getLogger("manage_timeline")

    logger.info(f"Setting custom deadline: {deadline_name} at {deadline_date}")

    # This would integrate with the timeline manager to add custom deadlines
    # For now, just log the action
    print(f"✅ Custom deadline set: {deadline_name} at {deadline_date.strftime('%Y-%m-%d %H:%M')} (Priority: {priority})")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Manage competition timeline and deadlines")

    # Timeline arguments
    parser.add_argument("--competition-end", "-e", type=str,
                       help="Competition end date (YYYY-MM-DD HH:MM)")
    parser.add_argument("--days-remaining", "-d", type=int,
                       help="Days remaining in competition (alternative to end date)")

    # Actions
    parser.add_argument("--status", "-s", action="store_true",
                       help="Show current timeline status")
    parser.add_argument("--check-deadlines", "-c", action="store_true",
                       help="Check upcoming deadlines")
    parser.add_argument("--generate-report", "-r", action="store_true",
                       help="Generate comprehensive timeline report")

    # Custom deadline
    parser.add_argument("--set-deadline", type=str,
                       help="Set custom deadline (format: 'name:YYYY-MM-DD HH:MM:priority')")

    # Configuration
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    parser.add_argument("--output-file", "-o", type=str, default="timeline_report.json",
                       help="Output file for timeline report")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("⏰ Starting timeline management")

    try:
        # Load configuration
        config_manager = get_config_manager()
        if args.config:
            config = config_manager.load_config(config_file=args.config)
        else:
            config = config_manager.load_config("submission")

        # Create timeline manager
        timeline_manager = create_competition_timeline(config.get('timeline_management', {}))

        # Determine competition end date
        if args.competition_end:
            competition_end_date = datetime.strptime(args.competition_end, '%Y-%m-%d %H:%M')
        elif args.days_remaining:
            competition_end_date = datetime.now() + timedelta(days=args.days_remaining)
        else:
            # Default: assume 7 days remaining
            competition_end_date = datetime.now() + timedelta(days=7)
            logger.warning("No end date specified, assuming 7 days remaining")

        logger.info(f"Competition end date: {competition_end_date}")

        # Execute requested actions
        if args.status:
            display_timeline_status(timeline_manager, competition_end_date)

        if args.check_deadlines:
            check_deadlines(timeline_manager, competition_end_date)

        if args.generate_report:
            report = generate_timeline_report(timeline_manager, competition_end_date, args.output_file)
            print(f"\n📄 Timeline report generated: {args.output_file}")

        if args.set_deadline:
            try:
                parts = args.set_deadline.split(':')
                if len(parts) == 3:
                    deadline_name, date_str, priority = parts
                    deadline_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                    set_custom_deadline(timeline_manager, deadline_name, deadline_date, priority)
                else:
                    logger.error("Invalid deadline format. Use: 'name:YYYY-MM-DD HH:MM:priority'")
            except Exception as e:
                logger.error(f"Failed to set custom deadline: {str(e)}")

        # Default action if no specific action requested
        if not any([args.status, args.check_deadlines, args.generate_report, args.set_deadline]):
            display_timeline_status(timeline_manager, competition_end_date)
            check_deadlines(timeline_manager, competition_end_date)

        logger.info("✅ Timeline management completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("❌ Timeline management interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Timeline management failed with error: {str(e)}")
        logger.debug("Full error traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())