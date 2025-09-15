#!/usr/bin/env python3
"""
Phase 6: Alerting System
Enterprise-grade alerting with multiple channels (email, Slack, webhooks)
"""

import smtplib
import requests
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import threading
import time
from queue import Queue
import uuid

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.phase6_config import get_config
from architecture.observers import event_publisher, Observer

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    component: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledgment_time: Optional[datetime] = None
    resolution_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'metadata': self.metadata,
            'acknowledgment_time': self.acknowledgment_time.isoformat() if self.acknowledgment_time else None,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None
        }

class AlertChannel:
    """Base class for alert notification channels"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)

    def send_alert(self, alert: Alert) -> bool:
        """Send alert notification. Override in subclasses."""
        raise NotImplementedError

    def test_connection(self) -> bool:
        """Test channel connectivity. Override in subclasses."""
        return True

class EmailAlertChannel(AlertChannel):
    """Email alert notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("email", config)
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_address = config.get('from_address', 'alerts@company.com')
        self.recipients = config.get('recipients', [])
        self.use_tls = config.get('use_tls', True)

    def send_alert(self, alert: Alert) -> bool:
        """Send email alert"""
        if not self.enabled or not self.recipients:
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Email body
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'html'))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()

                if self.username and self.password:
                    server.login(self.username, self.password)

                server.send_message(msg)

            logger.info(f"Email alert sent: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert {alert.alert_id}: {e}")
            return False

    def _format_email_body(self, alert: Alert) -> str:
        """Format email body with HTML"""
        severity_colors = {
            AlertSeverity.LOW: '#28a745',
            AlertSeverity.MEDIUM: '#ffc107',
            AlertSeverity.HIGH: '#fd7e14',
            AlertSeverity.CRITICAL: '#dc3545'
        }

        color = severity_colors.get(alert.severity, '#6c757d')

        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px;">
                    <h2>🚨 {alert.title}</h2>
                    <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                    <p><strong>Component:</strong> {alert.component}</p>
                    <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </div>

                <div style="padding: 20px; background-color: #f8f9fa; margin-top: 10px; border-radius: 5px;">
                    <h3>Alert Details</h3>
                    <p>{alert.message}</p>

                    {self._format_metadata(alert.metadata) if alert.metadata else ''}
                </div>

                <div style="padding: 10px; text-align: center; color: #6c757d; font-size: 12px;">
                    <p>Hackathon Forecast 2025 - Monitoring System</p>
                    <p>Alert ID: {alert.alert_id}</p>
                </div>
            </div>
        </body>
        </html>
        """

    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for email display"""
        if not metadata:
            return ""

        html = "<h4>Additional Information</h4><ul>"
        for key, value in metadata.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        return html

    def test_connection(self) -> bool:
        """Test email server connection"""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()

                if self.username and self.password:
                    server.login(self.username, self.password)

            return True

        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False

class SlackAlertChannel(AlertChannel):
    """Slack alert notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("slack", config)
        self.webhook_url = config.get('webhook_url', '')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'Forecast Bot')

    def send_alert(self, alert: Alert) -> bool:
        """Send Slack alert"""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            # Format Slack message
            payload = self._format_slack_payload(alert)

            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert.alert_id}")
                return True
            else:
                logger.error(f"Slack alert failed with status {response.status_code}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert {alert.alert_id}: {e}")
            return False

    def _format_slack_payload(self, alert: Alert) -> Dict[str, Any]:
        """Format Slack message payload"""
        severity_colors = {
            AlertSeverity.LOW: '#28a745',
            AlertSeverity.MEDIUM: '#ffc107',
            AlertSeverity.HIGH: '#fd7e14',
            AlertSeverity.CRITICAL: '#dc3545'
        }

        severity_emojis = {
            AlertSeverity.LOW: ':information_source:',
            AlertSeverity.MEDIUM: ':warning:',
            AlertSeverity.HIGH: ':exclamation:',
            AlertSeverity.CRITICAL: ':rotating_light:'
        }

        color = severity_colors.get(alert.severity, '#6c757d')
        emoji = severity_emojis.get(alert.severity, ':question:')

        # Build metadata fields
        fields = [
            {
                "title": "Component",
                "value": alert.component,
                "short": True
            },
            {
                "title": "Severity",
                "value": alert.severity.value.upper(),
                "short": True
            },
            {
                "title": "Time",
                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                "short": False
            }
        ]

        # Add metadata fields
        for key, value in alert.metadata.items():
            fields.append({
                "title": key.replace('_', ' ').title(),
                "value": str(value),
                "short": True
            })

        return {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": ":chart_with_upwards_trend:",
            "attachments": [{
                "color": color,
                "title": f"{emoji} {alert.title}",
                "text": alert.message,
                "fields": fields,
                "footer": "Hackathon Forecast 2025",
                "footer_icon": "https://cdn-icons-png.flaticon.com/512/2103/2103658.png",
                "ts": int(alert.timestamp.timestamp())
            }]
        }

    def test_connection(self) -> bool:
        """Test Slack webhook connection"""
        if not self.webhook_url:
            return False

        try:
            test_payload = {
                "channel": self.channel,
                "username": self.username,
                "text": "🧪 Test connection from Hackathon Forecast 2025"
            }

            response = requests.post(
                self.webhook_url,
                json=test_payload,
                timeout=10
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Slack connection test failed: {e}")
            return False

class WebhookAlertChannel(AlertChannel):
    """Generic webhook alert notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("webhook", config)
        self.webhook_url = config.get('webhook_url', '')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 10)

    def send_alert(self, alert: Alert) -> bool:
        """Send webhook alert"""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now().isoformat(),
                "source": "hackathon_forecast_2025"
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )

            if 200 <= response.status_code < 300:
                logger.info(f"Webhook alert sent: {alert.alert_id}")
                return True
            else:
                logger.error(f"Webhook alert failed with status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to send webhook alert {alert.alert_id}: {e}")
            return False

    def test_connection(self) -> bool:
        """Test webhook connection"""
        if not self.webhook_url:
            return False

        try:
            test_payload = {
                "test": True,
                "message": "Test connection from Hackathon Forecast 2025",
                "timestamp": datetime.now().isoformat()
            }

            response = requests.post(
                self.webhook_url,
                json=test_payload,
                headers=self.headers,
                timeout=self.timeout
            )

            return 200 <= response.status_code < 300

        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False

class AlertManager:
    """Central alert management system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().dict()
        self.channels: Dict[str, AlertChannel] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_queue: Queue = Queue()
        self.max_history = 1000

        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.rate_limit_window = 300  # 5 minutes

        self._initialize_channels()
        self._start_alert_processor()

    def _initialize_channels(self):
        """Initialize alert channels from configuration"""
        alerting_config = self.config.get('alerting', {})

        # Email channel
        email_config = alerting_config.get('alert_channels', {}).get('email', {})
        if email_config.get('enabled', False):
            self.channels['email'] = EmailAlertChannel(email_config)

        # Slack channel
        slack_config = alerting_config.get('alert_channels', {}).get('slack', {})
        if slack_config.get('enabled', False):
            self.channels['slack'] = SlackAlertChannel(slack_config)

        # Webhook channel
        webhook_config = alerting_config.get('alert_channels', {}).get('webhook', {})
        if webhook_config.get('enabled', False):
            self.channels['webhook'] = WebhookAlertChannel(webhook_config)

    def _start_alert_processor(self):
        """Start background alert processing thread"""
        def process_alerts():
            while True:
                try:
                    alert = self.alert_queue.get(timeout=1)
                    self._process_alert(alert)
                    self.alert_queue.task_done()
                except:
                    pass  # Timeout or other error, continue

        processor_thread = threading.Thread(target=process_alerts, daemon=True)
        processor_thread.start()

    def create_alert(self, title: str, message: str, severity: AlertSeverity,
                    component: str, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert"""
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            title=title,
            message=message,
            severity=severity,
            component=component,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        # Check rate limiting
        if self._is_rate_limited(alert):
            logger.info(f"Alert rate limited: {alert.alert_id}")
            return alert

        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert

        # Queue for processing
        self.alert_queue.put(alert)

        # Publish event
        event_publisher.publish_event('alert_created', {
            'alert_id': alert.alert_id,
            'title': alert.title,
            'severity': alert.severity.value,
            'component': alert.component
        })

        logger.info(f"Alert created: {alert.alert_id} - {alert.title}")
        return alert

    def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert should be rate limited"""
        # Create rate limit key based on component and severity
        rate_key = f"{alert.component}_{alert.severity.value}"

        current_time = datetime.now()

        if rate_key not in self.rate_limits:
            self.rate_limits[rate_key] = {
                'count': 1,
                'window_start': current_time
            }
            return False

        rate_info = self.rate_limits[rate_key]

        # Check if we're in a new window
        if (current_time - rate_info['window_start']).total_seconds() > self.rate_limit_window:
            rate_info['count'] = 1
            rate_info['window_start'] = current_time
            return False

        # Increment count
        rate_info['count'] += 1

        # Check limits (configurable by severity)
        limits = {
            AlertSeverity.CRITICAL: 5,  # Max 5 critical alerts per 5min
            AlertSeverity.HIGH: 10,
            AlertSeverity.MEDIUM: 20,
            AlertSeverity.LOW: 50
        }

        return rate_info['count'] > limits.get(alert.severity, 10)

    def _process_alert(self, alert: Alert):
        """Process alert by sending to all enabled channels"""
        success_count = 0
        total_channels = len(self.channels)

        for channel_name, channel in self.channels.items():
            try:
                if channel.send_alert(alert):
                    success_count += 1
                else:
                    logger.warning(f"Failed to send alert {alert.alert_id} via {channel_name}")
            except Exception as e:
                logger.error(f"Error sending alert {alert.alert_id} via {channel_name}: {e}")

        # Log processing results
        if success_count == total_channels:
            logger.info(f"Alert {alert.alert_id} sent successfully to all channels")
        elif success_count > 0:
            logger.warning(f"Alert {alert.alert_id} sent to {success_count}/{total_channels} channels")
        else:
            logger.error(f"Alert {alert.alert_id} failed to send to any channels")

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an active alert"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledgment_time = datetime.now()

        # Publish event
        event_publisher.publish_event('alert_acknowledged', {
            'alert_id': alert_id,
            'user': user,
            'acknowledgment_time': alert.acknowledgment_time.isoformat()
        })

        logger.info(f"Alert acknowledged: {alert_id} by {user}")
        return True

    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an active alert"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolution_time = datetime.now()

        # Remove from active alerts
        del self.active_alerts[alert_id]

        # Publish event
        event_publisher.publish_event('alert_resolved', {
            'alert_id': alert_id,
            'user': user,
            'resolution_time': alert.resolution_time.isoformat()
        })

        logger.info(f"Alert resolved: {alert_id} by {user}")
        return True

    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get list of active alerts"""
        alerts = list(self.active_alerts.values())

        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)

        # Recent alerts
        recent_24h = [alert for alert in self.alert_history if alert.timestamp >= last_24h]
        recent_7d = [alert for alert in self.alert_history if alert.timestamp >= last_7d]

        # Severity breakdown
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                alert for alert in self.active_alerts.values()
                if alert.severity == severity
            ])

        # Component breakdown
        component_counts = {}
        for alert in self.active_alerts.values():
            component_counts[alert.component] = component_counts.get(alert.component, 0) + 1

        return {
            'active_alerts': len(self.active_alerts),
            'alerts_last_24h': len(recent_24h),
            'alerts_last_7d': len(recent_7d),
            'total_alerts_history': len(self.alert_history),
            'severity_breakdown': severity_counts,
            'component_breakdown': component_counts,
            'channels_configured': len(self.channels),
            'channels_enabled': len([ch for ch in self.channels.values() if ch.enabled])
        }

    def test_all_channels(self) -> Dict[str, bool]:
        """Test all configured alert channels"""
        results = {}

        for channel_name, channel in self.channels.items():
            try:
                results[channel_name] = channel.test_connection()
            except Exception as e:
                logger.error(f"Channel test failed for {channel_name}: {e}")
                results[channel_name] = False

        return results

class MetricsAlertObserver(Observer):
    """Observer that creates alerts based on metrics thresholds"""

    def __init__(self, alert_manager: AlertManager, config: Optional[Dict[str, Any]] = None):
        self.alert_manager = alert_manager
        self.config = config or get_config().dict()
        self.thresholds = self.config.get('monitoring', {}).get('alert_thresholds', {})

    def update(self, event):
        """Handle metric events and create alerts if thresholds are exceeded"""
        if event.event_type == 'validation_result':
            self._check_wmape_threshold(event)
        elif event.event_type == 'training_end':
            self._check_training_metrics(event)
        elif event.event_type == 'health_check_completed':
            self._check_health_status(event)

    def _check_wmape_threshold(self, event):
        """Check if WMAPE exceeds threshold"""
        score = event.data.get('score')
        threshold = self.thresholds.get('wmape_threshold', 0.20)

        if score and score > threshold:
            self.alert_manager.create_alert(
                title="WMAPE Threshold Exceeded",
                message=f"Model WMAPE ({score:.3f}) exceeds threshold ({threshold:.3f})",
                severity=AlertSeverity.HIGH,
                component="model_validation",
                metadata={
                    'wmape_score': score,
                    'threshold': threshold,
                    'metric': event.data.get('metric', 'wmape'),
                    'fold': event.data.get('fold', 'unknown')
                }
            )

    def _check_training_metrics(self, event):
        """Check training completion metrics"""
        final_metrics = event.data.get('final_metrics', {})

        # Check for training failure indicators
        if 'error' in final_metrics:
            self.alert_manager.create_alert(
                title="Model Training Failed",
                message=f"Training completed with errors: {final_metrics.get('error')}",
                severity=AlertSeverity.CRITICAL,
                component="model_training",
                metadata=final_metrics
            )

    def _check_health_status(self, event):
        """Check health status and create alerts for critical issues"""
        status = event.data.get('status')
        check_name = event.data.get('check_name')

        if status == 'critical':
            self.alert_manager.create_alert(
                title=f"Health Check Failed: {check_name}",
                message=event.data.get('message', 'Health check failed'),
                severity=AlertSeverity.CRITICAL,
                component="health_check",
                metadata={
                    'check_name': check_name,
                    'duration_ms': event.data.get('duration_ms')
                }
            )

    def get_observer_id(self) -> str:
        return "metrics_alert_observer"

# Global alert manager instance
alert_manager = AlertManager()

# Global metrics alert observer
metrics_observer = MetricsAlertObserver(alert_manager)
event_publisher.attach(metrics_observer)

def create_alert(title: str, message: str, severity: str, component: str,
                metadata: Optional[Dict[str, Any]] = None) -> Alert:
    """Create a new alert (convenience function)"""
    severity_enum = AlertSeverity(severity.lower())
    return alert_manager.create_alert(title, message, severity_enum, component, metadata)

def get_alert_statistics() -> Dict[str, Any]:
    """Get current alert statistics"""
    return alert_manager.get_alert_statistics()


if __name__ == "__main__":
    # Demo alert system
    print("🚨 Alert System Demo")
    print("=" * 50)

    # Test channel connections
    print("\n🔧 Testing Alert Channels:")
    test_results = alert_manager.test_all_channels()
    for channel, result in test_results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {channel}")

    # Create sample alerts
    print("\n📢 Creating Sample Alerts:")

    # Low severity alert
    alert1 = alert_manager.create_alert(
        "Model Performance Warning",
        "WMAPE slightly elevated but within acceptable range",
        AlertSeverity.LOW,
        "model_validation",
        {"wmape": 0.18, "threshold": 0.20}
    )
    print(f"  📘 Low: {alert1.alert_id}")

    # High severity alert
    alert2 = alert_manager.create_alert(
        "High Memory Usage Detected",
        "System memory usage is at 89% capacity",
        AlertSeverity.HIGH,
        "system_monitoring",
        {"memory_percent": 89, "threshold": 85}
    )
    print(f"  🔶 High: {alert2.alert_id}")

    # Critical alert
    alert3 = alert_manager.create_alert(
        "Database Connection Lost",
        "Unable to connect to primary database server",
        AlertSeverity.CRITICAL,
        "database",
        {"connection_attempts": 5, "last_error": "Connection timeout"}
    )
    print(f"  🔴 Critical: {alert3.alert_id}")

    # Wait for processing
    time.sleep(2)

    # Show statistics
    print("\n📊 Alert Statistics:")
    stats = alert_manager.get_alert_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Acknowledge and resolve alerts
    print("\n✅ Managing Alerts:")
    alert_manager.acknowledge_alert(alert2.alert_id, "admin")
    print(f"  Acknowledged: {alert2.alert_id}")

    alert_manager.resolve_alert(alert1.alert_id, "system")
    print(f"  Resolved: {alert1.alert_id}")

    # Final statistics
    print("\n📊 Final Statistics:")
    final_stats = alert_manager.get_alert_statistics()
    print(f"  Active Alerts: {final_stats['active_alerts']}")
    print(f"  Total Processed: {final_stats['total_alerts_history']}")