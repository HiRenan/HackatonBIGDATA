#!/usr/bin/env python3
"""
Phase 7: Timeline Management System
Advanced timeline management with deadline tracking and strategic scheduling
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sys
from enum import Enum
import json
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.submissions.strategy import SubmissionPhase, SubmissionPlan
from src.utils.logging import get_logger

logger = get_logger(__name__)

class TimelineStatus(Enum):
    """Timeline status enumeration"""
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"

class PriorityLevel(Enum):
    """Priority level enumeration"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SubmissionWindow:
    """Submission window definition"""
    phase: SubmissionPhase
    start_time: datetime
    end_time: datetime
    optimal_time: Optional[datetime] = None
    buffer_hours: int = 2
    priority: PriorityLevel = PriorityLevel.MEDIUM
    dependencies: List[SubmissionPhase] = field(default_factory=list)
    status: TimelineStatus = TimelineStatus.SCHEDULED

    @property
    def duration(self) -> timedelta:
        """Get window duration"""
        return self.end_time - self.start_time

    @property
    def time_remaining(self) -> timedelta:
        """Get time remaining until end"""
        now = datetime.now(timezone.utc)
        if self.end_time.tzinfo is None:
            self.end_time = self.end_time.replace(tzinfo=timezone.utc)
        return self.end_time - now

    @property
    def is_active(self) -> bool:
        """Check if window is currently active"""
        now = datetime.now(timezone.utc)
        if self.start_time.tzinfo is None:
            start_time = self.start_time.replace(tzinfo=timezone.utc)
        else:
            start_time = self.start_time
        if self.end_time.tzinfo is None:
            end_time = self.end_time.replace(tzinfo=timezone.utc)
        else:
            end_time = self.end_time
        return start_time <= now <= end_time

    @property
    def is_overdue(self) -> bool:
        """Check if window is overdue"""
        now = datetime.now(timezone.utc)
        if self.end_time.tzinfo is None:
            end_time = self.end_time.replace(tzinfo=timezone.utc)
        else:
            end_time = self.end_time
        return now > end_time

@dataclass
class SubmissionSchedule:
    """Complete submission schedule"""
    competition_name: str
    start_date: datetime
    end_date: datetime
    windows: List[SubmissionWindow] = field(default_factory=list)
    timezone: str = "UTC"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_window(self, window: SubmissionWindow):
        """Add submission window to schedule"""
        self.windows.append(window)
        self.windows.sort(key=lambda w: w.start_time)

    def get_active_windows(self) -> List[SubmissionWindow]:
        """Get currently active windows"""
        return [w for w in self.windows if w.is_active]

    def get_upcoming_windows(self, hours_ahead: int = 24) -> List[SubmissionWindow]:
        """Get upcoming windows within specified hours"""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        upcoming = []
        for window in self.windows:
            if window.start_time.tzinfo is None:
                start_time = window.start_time.replace(tzinfo=timezone.utc)
            else:
                start_time = window.start_time

            if now <= start_time <= cutoff:
                upcoming.append(window)

        return sorted(upcoming, key=lambda w: w.start_time)

    def get_overdue_windows(self) -> List[SubmissionWindow]:
        """Get overdue windows"""
        return [w for w in self.windows if w.is_overdue and w.status != TimelineStatus.COMPLETED]

@dataclass
class DeadlineAlert:
    """Deadline alert definition"""
    window: SubmissionWindow
    alert_type: str  # "upcoming", "urgent", "overdue"
    message: str
    severity: PriorityLevel
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class DeadlineTracker:
    """Deadline tracking and alerting system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alert_thresholds = {
            'upcoming_hours': self.config.get('upcoming_alert_hours', 24),
            'urgent_hours': self.config.get('urgent_alert_hours', 6),
            'critical_hours': self.config.get('critical_alert_hours', 2)
        }
        self.alerts_sent = set()

    def check_deadlines(self, schedule: SubmissionSchedule) -> List[DeadlineAlert]:
        """Check for deadline alerts"""
        alerts = []
        now = datetime.now(timezone.utc)

        for window in schedule.windows:
            if window.status == TimelineStatus.COMPLETED:
                continue

            # Calculate time until deadline
            if window.end_time.tzinfo is None:
                end_time = window.end_time.replace(tzinfo=timezone.utc)
            else:
                end_time = window.end_time

            time_until_deadline = (end_time - now).total_seconds() / 3600  # hours

            alert_key = f"{window.phase.name}_{end_time.isoformat()}"

            # Check for overdue
            if time_until_deadline < 0 and alert_key not in self.alerts_sent:
                alerts.append(DeadlineAlert(
                    window=window,
                    alert_type="overdue",
                    message=f"OVERDUE: {window.phase.name} deadline passed {abs(time_until_deadline):.1f} hours ago",
                    severity=PriorityLevel.CRITICAL
                ))
                self.alerts_sent.add(alert_key)

            # Check for critical (within critical hours)
            elif 0 <= time_until_deadline <= self.alert_thresholds['critical_hours'] and alert_key not in self.alerts_sent:
                alerts.append(DeadlineAlert(
                    window=window,
                    alert_type="critical",
                    message=f"CRITICAL: {window.phase.name} deadline in {time_until_deadline:.1f} hours",
                    severity=PriorityLevel.CRITICAL
                ))
                self.alerts_sent.add(alert_key)

            # Check for urgent (within urgent hours)
            elif 0 <= time_until_deadline <= self.alert_thresholds['urgent_hours'] and alert_key not in self.alerts_sent:
                alerts.append(DeadlineAlert(
                    window=window,
                    alert_type="urgent",
                    message=f"URGENT: {window.phase.name} deadline in {time_until_deadline:.1f} hours",
                    severity=PriorityLevel.HIGH
                ))
                self.alerts_sent.add(alert_key)

            # Check for upcoming (within upcoming hours)
            elif 0 <= time_until_deadline <= self.alert_thresholds['upcoming_hours'] and alert_key not in self.alerts_sent:
                alerts.append(DeadlineAlert(
                    window=window,
                    alert_type="upcoming",
                    message=f"Upcoming: {window.phase.name} deadline in {time_until_deadline:.1f} hours",
                    severity=PriorityLevel.MEDIUM
                ))
                self.alerts_sent.add(alert_key)

        return alerts

    def get_next_deadline(self, schedule: SubmissionSchedule) -> Optional[SubmissionWindow]:
        """Get next upcoming deadline"""
        now = datetime.now(timezone.utc)

        upcoming_windows = []
        for window in schedule.windows:
            if window.status == TimelineStatus.COMPLETED:
                continue

            if window.end_time.tzinfo is None:
                end_time = window.end_time.replace(tzinfo=timezone.utc)
            else:
                end_time = window.end_time

            if end_time > now:
                upcoming_windows.append((window, end_time))

        if not upcoming_windows:
            return None

        # Sort by deadline and return the earliest
        upcoming_windows.sort(key=lambda x: x[1])
        return upcoming_windows[0][0]

class TimelineManager:
    """Advanced timeline management system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.schedule = None
        self.deadline_tracker = DeadlineTracker(config.get('deadline_tracker', {}))

        # Default timeline configuration
        self.default_phase_durations = {
            SubmissionPhase.BASELINE: timedelta(hours=6),
            SubmissionPhase.SINGLE_MODEL: timedelta(hours=12),
            SubmissionPhase.INITIAL_ENSEMBLE: timedelta(hours=24),
            SubmissionPhase.OPTIMIZED_ENSEMBLE: timedelta(hours=36),
            SubmissionPhase.FINAL_SUBMISSION: timedelta(hours=48)
        }

    def create_competition_schedule(self,
                                  competition_name: str,
                                  competition_start: datetime,
                                  competition_end: datetime,
                                  custom_schedule: Optional[Dict[SubmissionPhase, Dict[str, Any]]] = None) -> SubmissionSchedule:
        """Create complete competition schedule"""

        logger.info(f"Creating competition schedule: {competition_name}")

        schedule = SubmissionSchedule(
            competition_name=competition_name,
            start_date=competition_start,
            end_date=competition_end
        )

        # Default 5-phase timeline based on specifications
        default_timeline = {
            SubmissionPhase.BASELINE: {
                'day': 3,
                'optimal_hour': 18,  # 6 PM
                'duration_hours': 6,
                'priority': PriorityLevel.LOW
            },
            SubmissionPhase.SINGLE_MODEL: {
                'day': 7,
                'optimal_hour': 20,  # 8 PM
                'duration_hours': 12,
                'priority': PriorityLevel.MEDIUM
            },
            SubmissionPhase.INITIAL_ENSEMBLE: {
                'day': 10,
                'optimal_hour': 22,  # 10 PM
                'duration_hours': 24,
                'priority': PriorityLevel.MEDIUM,
                'dependencies': [SubmissionPhase.SINGLE_MODEL]
            },
            SubmissionPhase.OPTIMIZED_ENSEMBLE: {
                'day': 13,
                'optimal_hour': 20,  # 8 PM
                'duration_hours': 36,
                'priority': PriorityLevel.HIGH,
                'dependencies': [SubmissionPhase.INITIAL_ENSEMBLE]
            },
            SubmissionPhase.FINAL_SUBMISSION: {
                'day': -1,  # Last day
                'optimal_hour': 18,  # 6 PM (with buffer before deadline)
                'duration_hours': 48,
                'priority': PriorityLevel.CRITICAL,
                'dependencies': [SubmissionPhase.OPTIMIZED_ENSEMBLE]
            }
        }

        # Merge with custom schedule if provided
        timeline = default_timeline
        if custom_schedule:
            for phase, config in custom_schedule.items():
                if phase in timeline:
                    timeline[phase].update(config)
                else:
                    timeline[phase] = config

        # Create submission windows
        for phase, config in timeline.items():
            window = self._create_submission_window(
                phase=phase,
                competition_start=competition_start,
                competition_end=competition_end,
                config=config
            )
            schedule.add_window(window)

        self.schedule = schedule
        logger.info(f"Created schedule with {len(schedule.windows)} submission windows")

        return schedule

    def _create_submission_window(self,
                                phase: SubmissionPhase,
                                competition_start: datetime,
                                competition_end: datetime,
                                config: Dict[str, Any]) -> SubmissionWindow:
        """Create a single submission window"""

        day = config.get('day', 1)
        optimal_hour = config.get('optimal_hour', 12)
        duration_hours = config.get('duration_hours', 12)
        buffer_hours = config.get('buffer_hours', 2)
        priority = config.get('priority', PriorityLevel.MEDIUM)
        dependencies = config.get('dependencies', [])

        # Calculate dates
        if day == -1:  # Last day
            optimal_time = competition_end - timedelta(hours=24-optimal_hour)
        else:
            optimal_time = competition_start + timedelta(days=day-1, hours=optimal_hour)

        # Create window around optimal time
        start_time = optimal_time - timedelta(hours=duration_hours//2)
        end_time = optimal_time + timedelta(hours=duration_hours//2)

        # Ensure window doesn't exceed competition bounds
        start_time = max(start_time, competition_start)
        end_time = min(end_time, competition_end)

        return SubmissionWindow(
            phase=phase,
            start_time=start_time,
            end_time=end_time,
            optimal_time=optimal_time,
            buffer_hours=buffer_hours,
            priority=priority,
            dependencies=dependencies
        )

    def get_current_recommendations(self) -> Dict[str, Any]:
        """Get current timeline recommendations"""
        if not self.schedule:
            return {'error': 'No schedule created'}

        now = datetime.now(timezone.utc)
        recommendations = {
            'current_time': now.isoformat(),
            'active_windows': [],
            'upcoming_windows': [],
            'next_deadline': None,
            'recommended_actions': [],
            'alerts': []
        }

        # Get active windows
        active_windows = self.schedule.get_active_windows()
        recommendations['active_windows'] = [
            {
                'phase': window.phase.name,
                'priority': window.priority.name,
                'time_remaining': window.time_remaining.total_seconds() / 3600,  # hours
                'optimal_time_passed': (now - window.optimal_time).total_seconds() / 3600 if window.optimal_time else None
            }
            for window in active_windows
        ]

        # Get upcoming windows
        upcoming_windows = self.schedule.get_upcoming_windows(hours_ahead=48)
        recommendations['upcoming_windows'] = [
            {
                'phase': window.phase.name,
                'start_time': window.start_time.isoformat(),
                'optimal_time': window.optimal_time.isoformat() if window.optimal_time else None,
                'priority': window.priority.name,
                'hours_until_start': (window.start_time.replace(tzinfo=timezone.utc) - now).total_seconds() / 3600
            }
            for window in upcoming_windows
        ]

        # Get next deadline
        next_deadline = self.deadline_tracker.get_next_deadline(self.schedule)
        if next_deadline:
            recommendations['next_deadline'] = {
                'phase': next_deadline.phase.name,
                'deadline': next_deadline.end_time.isoformat(),
                'hours_remaining': next_deadline.time_remaining.total_seconds() / 3600,
                'priority': next_deadline.priority.name
            }

        # Check for alerts
        alerts = self.deadline_tracker.check_deadlines(self.schedule)
        recommendations['alerts'] = [
            {
                'type': alert.alert_type,
                'message': alert.message,
                'severity': alert.severity.name,
                'phase': alert.window.phase.name
            }
            for alert in alerts
        ]

        # Generate recommended actions
        recommendations['recommended_actions'] = self._generate_recommendations(
            active_windows, upcoming_windows, alerts
        )

        return recommendations

    def _generate_recommendations(self,
                                active_windows: List[SubmissionWindow],
                                upcoming_windows: List[SubmissionWindow],
                                alerts: List[DeadlineAlert]) -> List[str]:
        """Generate timeline-based recommendations"""
        recommendations = []

        # High priority alerts
        critical_alerts = [a for a in alerts if a.severity == PriorityLevel.CRITICAL]
        if critical_alerts:
            recommendations.append("CRITICAL: Immediate action required on overdue/critical deadlines")

        # Active window recommendations
        for window in active_windows:
            if window.priority == PriorityLevel.CRITICAL:
                recommendations.append(f"URGENT: Complete {window.phase.name} submission immediately")
            elif window.time_remaining.total_seconds() < 3600:  # Less than 1 hour
                recommendations.append(f"WARNING: {window.phase.name} window closing soon")

        # Upcoming window recommendations
        for window in upcoming_windows:
            hours_until = (window.start_time.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).total_seconds() / 3600
            if hours_until <= 6:
                recommendations.append(f"PREPARE: {window.phase.name} window opens in {hours_until:.1f} hours")

        # General recommendations
        if not active_windows and not upcoming_windows:
            recommendations.append("All submission windows completed or not yet active")
        elif len(active_windows) > 1:
            recommendations.append("Multiple submission windows active - prioritize by importance")

        return recommendations

    def update_window_status(self, phase: SubmissionPhase, status: TimelineStatus):
        """Update submission window status"""
        if not self.schedule:
            logger.warning("No schedule available to update")
            return

        for window in self.schedule.windows:
            if window.phase == phase:
                window.status = status
                logger.info(f"Updated {phase.name} status to {status.name}")
                break

    def export_schedule(self, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export schedule to JSON format"""
        if not self.schedule:
            return {}

        schedule_data = {
            'competition_name': self.schedule.competition_name,
            'start_date': self.schedule.start_date.isoformat(),
            'end_date': self.schedule.end_date.isoformat(),
            'timezone': self.schedule.timezone,
            'created_at': self.schedule.created_at.isoformat(),
            'windows': []
        }

        for window in self.schedule.windows:
            window_data = {
                'phase': window.phase.name,
                'start_time': window.start_time.isoformat(),
                'end_time': window.end_time.isoformat(),
                'optimal_time': window.optimal_time.isoformat() if window.optimal_time else None,
                'buffer_hours': window.buffer_hours,
                'priority': window.priority.name,
                'dependencies': [dep.name for dep in window.dependencies],
                'status': window.status.name
            }
            schedule_data['windows'].append(window_data)

        if file_path:
            with open(file_path, 'w') as f:
                json.dump(schedule_data, f, indent=2)
            logger.info(f"Schedule exported to {file_path}")

        return schedule_data

    def get_schedule_summary(self) -> str:
        """Get human-readable schedule summary"""
        if not self.schedule:
            return "No schedule created"

        summary_lines = [
            f"Competition: {self.schedule.competition_name}",
            f"Duration: {self.schedule.start_date.strftime('%Y-%m-%d')} to {self.schedule.end_date.strftime('%Y-%m-%d')}",
            f"Windows: {len(self.schedule.windows)}",
            ""
        ]

        summary_lines.append("Submission Timeline:")
        for window in self.schedule.windows:
            status_icon = {
                TimelineStatus.SCHEDULED: "📅",
                TimelineStatus.ACTIVE: "🔴",
                TimelineStatus.COMPLETED: "✅",
                TimelineStatus.OVERDUE: "⚠️",
                TimelineStatus.CANCELLED: "❌"
            }.get(window.status, "❓")

            summary_lines.append(
                f"  {status_icon} {window.phase.name}: "
                f"{window.start_time.strftime('%m/%d %H:%M')} - {window.end_time.strftime('%m/%d %H:%M')} "
                f"(Priority: {window.priority.name})"
            )

        return "\n".join(summary_lines)

def create_competition_timeline(competition_name: str,
                              start_date: datetime,
                              end_date: datetime,
                              config: Optional[Dict[str, Any]] = None) -> TimelineManager:
    """Create configured timeline manager"""
    manager = TimelineManager(config)
    manager.create_competition_schedule(competition_name, start_date, end_date)
    return manager

if __name__ == "__main__":
    # Demo usage
    print("⏰ Timeline Management System Demo")
    print("=" * 50)

    # Create timeline manager
    manager = TimelineManager()

    # Create competition schedule (15-day competition)
    start_date = datetime.now(timezone.utc)
    end_date = start_date + timedelta(days=15)

    schedule = manager.create_competition_schedule(
        "Hackathon Forecast 2025",
        start_date,
        end_date
    )

    print(f"✅ Created schedule with {len(schedule.windows)} submission windows")

    # Get current recommendations
    recommendations = manager.get_current_recommendations()
    print(f"\nCurrent recommendations:")
    for rec in recommendations['recommended_actions']:
        print(f"  • {rec}")

    # Show schedule summary
    print(f"\n{manager.get_schedule_summary()}")

    print("\n⏰ Timeline management system ready!")
    print("Ready to manage strategic submission timing and deadlines.")