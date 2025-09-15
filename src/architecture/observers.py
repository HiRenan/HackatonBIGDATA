#!/usr/bin/env python3
"""
Phase 6: Observer Pattern Implementation
Event-driven monitoring and notification system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional
import time
import threading
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

class Event:
    """Event object containing event data"""

    def __init__(self, event_type: str, data: Dict[str, Any], timestamp: Optional[datetime] = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        return f"Event({self.event_type}) at {self.timestamp}"

class Observer(ABC):
    """Abstract observer interface"""

    @abstractmethod
    def update(self, event: Event) -> None:
        """Handle event notification"""
        pass

    @abstractmethod
    def get_observer_id(self) -> str:
        """Get unique observer identifier"""
        pass

class Subject(ABC):
    """Abstract subject that observers can watch"""

    def __init__(self):
        self._observers: List[Observer] = []
        self._lock = threading.Lock()

    def attach(self, observer: Observer) -> None:
        """Attach an observer"""
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)
                logger.info(f"Attached observer: {observer.get_observer_id()}")

    def detach(self, observer: Observer) -> None:
        """Detach an observer"""
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)
                logger.info(f"Detached observer: {observer.get_observer_id()}")

    def notify(self, event: Event) -> None:
        """Notify all observers of an event"""
        with self._lock:
            observers_copy = self._observers.copy()

        for observer in observers_copy:
            try:
                observer.update(event)
            except Exception as e:
                logger.error(f"Observer {observer.get_observer_id()} failed to handle event: {e}")

    def get_observer_count(self) -> int:
        """Get number of attached observers"""
        return len(self._observers)

class TrainingObserver(Observer):
    """Observer for monitoring model training progress"""

    def __init__(self, observer_id: str, log_interval: int = 100):
        self.observer_id = observer_id
        self.log_interval = log_interval
        self.training_history: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.current_epoch = 0
        self.best_score = float('inf')

    def update(self, event: Event) -> None:
        """Handle training events"""
        if event.event_type == 'training_start':
            self._handle_training_start(event)
        elif event.event_type == 'epoch_end':
            self._handle_epoch_end(event)
        elif event.event_type == 'training_end':
            self._handle_training_end(event)
        elif event.event_type == 'validation_score':
            self._handle_validation_score(event)

    def _handle_training_start(self, event: Event) -> None:
        """Handle training start event"""
        self.start_time = event.timestamp
        self.current_epoch = 0
        self.best_score = float('inf')
        self.training_history = []

        logger.info(f"🚀 Training started - Model: {event.data.get('model_name', 'Unknown')}")

    def _handle_epoch_end(self, event: Event) -> None:
        """Handle epoch end event"""
        self.current_epoch = event.data.get('epoch', self.current_epoch + 1)
        metrics = event.data.get('metrics', {})

        # Log progress
        if self.current_epoch % self.log_interval == 0:
            logger.info(f"Epoch {self.current_epoch}: {metrics}")

        # Store history
        history_entry = {
            'epoch': self.current_epoch,
            'timestamp': event.timestamp.isoformat(),
            'metrics': metrics
        }
        self.training_history.append(history_entry)

    def _handle_training_end(self, event: Event) -> None:
        """Handle training end event"""
        if self.start_time:
            duration = event.timestamp - self.start_time
            logger.info(f"✅ Training completed in {duration}")

        final_metrics = event.data.get('final_metrics', {})
        logger.info(f"Final metrics: {final_metrics}")

    def _handle_validation_score(self, event: Event) -> None:
        """Handle validation score event"""
        score = event.data.get('score', float('inf'))
        metric_name = event.data.get('metric', 'loss')

        if score < self.best_score:
            self.best_score = score
            logger.info(f"🎯 New best {metric_name}: {score:.4f}")

    def get_observer_id(self) -> str:
        """Get observer ID"""
        return self.observer_id

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.training_history:
            return {'status': 'No training data'}

        return {
            'total_epochs': self.current_epoch,
            'best_score': self.best_score,
            'training_duration': str(datetime.now() - self.start_time) if self.start_time else None,
            'final_metrics': self.training_history[-1]['metrics'] if self.training_history else {}
        }

class ValidationObserver(Observer):
    """Observer for monitoring validation performance"""

    def __init__(self, observer_id: str, alert_threshold: float = 0.05):
        self.observer_id = observer_id
        self.alert_threshold = alert_threshold  # Threshold for performance degradation alerts
        self.validation_history: List[Dict[str, Any]] = []
        self.baseline_score: Optional[float] = None

    def update(self, event: Event) -> None:
        """Handle validation events"""
        if event.event_type == 'validation_result':
            self._handle_validation_result(event)
        elif event.event_type == 'baseline_set':
            self._handle_baseline_set(event)
        elif event.event_type == 'validation_start':
            self._handle_validation_start(event)

    def _handle_validation_result(self, event: Event) -> None:
        """Handle validation result event"""
        score = event.data.get('score')
        metric = event.data.get('metric', 'wmape')
        fold = event.data.get('fold', 0)

        # Store result
        result_entry = {
            'timestamp': event.timestamp.isoformat(),
            'score': score,
            'metric': metric,
            'fold': fold,
            'data': event.data
        }
        self.validation_history.append(result_entry)

        logger.info(f"📊 Validation {metric} (fold {fold}): {score:.4f}")

        # Check for performance degradation
        if self.baseline_score is not None and score is not None:
            degradation = (score - self.baseline_score) / self.baseline_score
            if degradation > self.alert_threshold:
                self._trigger_degradation_alert(score, degradation, metric)

    def _handle_baseline_set(self, event: Event) -> None:
        """Handle baseline setting event"""
        self.baseline_score = event.data.get('baseline_score')
        logger.info(f"🎯 Baseline score set: {self.baseline_score:.4f}")

    def _handle_validation_start(self, event: Event) -> None:
        """Handle validation start event"""
        strategy = event.data.get('strategy', 'Unknown')
        n_folds = event.data.get('n_folds', 1)
        logger.info(f"🔍 Starting validation: {strategy} ({n_folds} folds)")

    def _trigger_degradation_alert(self, current_score: float, degradation: float, metric: str) -> None:
        """Trigger performance degradation alert"""
        logger.warning(
            f"⚠️ Performance degradation detected! "
            f"{metric}: {current_score:.4f} (baseline: {self.baseline_score:.4f}, "
            f"degradation: {degradation:.2%})"
        )

    def get_observer_id(self) -> str:
        """Get observer ID"""
        return self.observer_id

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        if not self.validation_history:
            return {'status': 'No validation data'}

        scores = [entry['score'] for entry in self.validation_history if entry['score'] is not None]

        return {
            'total_validations': len(self.validation_history),
            'mean_score': sum(scores) / len(scores) if scores else None,
            'best_score': min(scores) if scores else None,
            'worst_score': max(scores) if scores else None,
            'baseline_score': self.baseline_score,
            'recent_validations': self.validation_history[-5:]  # Last 5 validations
        }

class BusinessRulesObserver(Observer):
    """Observer for monitoring business rule violations"""

    def __init__(self, observer_id: str):
        self.observer_id = observer_id
        self.violations: List[Dict[str, Any]] = []
        self.rule_stats: Dict[str, Dict[str, int]] = {}

    def update(self, event: Event) -> None:
        """Handle business rule events"""
        if event.event_type == 'rule_violation':
            self._handle_rule_violation(event)
        elif event.event_type == 'rule_applied':
            self._handle_rule_applied(event)
        elif event.event_type == 'constraint_check':
            self._handle_constraint_check(event)

    def _handle_rule_violation(self, event: Event) -> None:
        """Handle rule violation event"""
        rule_name = event.data.get('rule_name')
        severity = event.data.get('severity', 'medium')
        description = event.data.get('description')

        # Store violation
        violation = {
            'timestamp': event.timestamp.isoformat(),
            'rule_name': rule_name,
            'severity': severity,
            'description': description,
            'data': event.data
        }
        self.violations.append(violation)

        # Update stats
        if rule_name not in self.rule_stats:
            self.rule_stats[rule_name] = {'violations': 0, 'applications': 0}
        self.rule_stats[rule_name]['violations'] += 1

        # Log based on severity
        if severity == 'high':
            logger.error(f"🚨 HIGH severity rule violation: {rule_name} - {description}")
        elif severity == 'medium':
            logger.warning(f"⚠️ MEDIUM severity rule violation: {rule_name} - {description}")
        else:
            logger.info(f"ℹ️ LOW severity rule violation: {rule_name} - {description}")

    def _handle_rule_applied(self, event: Event) -> None:
        """Handle rule application event"""
        rule_name = event.data.get('rule_name')
        adjustments_made = event.data.get('adjustments_made', 0)

        # Update stats
        if rule_name not in self.rule_stats:
            self.rule_stats[rule_name] = {'violations': 0, 'applications': 0}
        self.rule_stats[rule_name]['applications'] += 1

        logger.info(f"✅ Rule applied: {rule_name} ({adjustments_made} adjustments)")

    def _handle_constraint_check(self, event: Event) -> None:
        """Handle constraint check event"""
        constraint_type = event.data.get('constraint_type')
        passed = event.data.get('passed', False)
        details = event.data.get('details', {})

        if not passed:
            logger.warning(f"❌ Constraint check failed: {constraint_type} - {details}")
        else:
            logger.debug(f"✅ Constraint check passed: {constraint_type}")

    def get_observer_id(self) -> str:
        """Get observer ID"""
        return self.observer_id

    def get_violations_summary(self) -> Dict[str, Any]:
        """Get violations summary"""
        if not self.violations:
            return {'status': 'No violations recorded'}

        # Count by severity
        severity_counts = {}
        for violation in self.violations:
            severity = violation['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Recent violations
        recent_violations = sorted(self.violations, key=lambda x: x['timestamp'])[-10:]

        return {
            'total_violations': len(self.violations),
            'severity_breakdown': severity_counts,
            'rule_stats': self.rule_stats,
            'recent_violations': recent_violations
        }

class EventPublisher(Subject):
    """Central event publisher for the forecasting system"""

    def __init__(self):
        super().__init__()
        self.event_history: List[Event] = []
        self.max_history_size = 1000

    def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event to all observers"""
        event = Event(event_type, data)

        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)

        # Notify observers
        self.notify(event)

    def get_recent_events(self, n: int = 10) -> List[Event]:
        """Get recent events"""
        return self.event_history[-n:]

    def get_events_by_type(self, event_type: str, n: int = 10) -> List[Event]:
        """Get recent events of specific type"""
        filtered_events = [e for e in self.event_history if e.event_type == event_type]
        return filtered_events[-n:]

# Global event publisher instance
event_publisher = EventPublisher()

# Convenience functions for publishing common events
def publish_training_start(model_name: str, **kwargs) -> None:
    """Publish training start event"""
    data = {'model_name': model_name, **kwargs}
    event_publisher.publish_event('training_start', data)

def publish_epoch_end(epoch: int, metrics: Dict[str, float], **kwargs) -> None:
    """Publish epoch end event"""
    data = {'epoch': epoch, 'metrics': metrics, **kwargs}
    event_publisher.publish_event('epoch_end', data)

def publish_validation_result(score: float, metric: str = 'wmape', **kwargs) -> None:
    """Publish validation result event"""
    data = {'score': score, 'metric': metric, **kwargs}
    event_publisher.publish_event('validation_result', data)

def publish_rule_violation(rule_name: str, severity: str = 'medium', description: str = '', **kwargs) -> None:
    """Publish rule violation event"""
    data = {'rule_name': rule_name, 'severity': severity, 'description': description, **kwargs}
    event_publisher.publish_event('rule_violation', data)

if __name__ == "__main__":
    # Demo usage
    print("👀 Phase 6 Observer Pattern Demo")
    print("=" * 50)

    # Create observers
    training_observer = TrainingObserver("training_monitor")
    validation_observer = ValidationObserver("validation_monitor")
    business_observer = BusinessRulesObserver("business_monitor")

    # Attach observers
    event_publisher.attach(training_observer)
    event_publisher.attach(validation_observer)
    event_publisher.attach(business_observer)

    print(f"Attached {event_publisher.get_observer_count()} observers")

    # Simulate events
    print("\n📊 Simulating events:")

    # Training events
    publish_training_start("LightGBM", n_estimators=1000)

    for epoch in range(1, 6):
        metrics = {'loss': 1.0 / epoch, 'accuracy': epoch * 0.1}
        publish_epoch_end(epoch, metrics)

    # Validation events
    publish_validation_result(0.15, 'wmape', fold=1)
    publish_validation_result(0.12, 'wmape', fold=2)

    # Business rule events
    publish_rule_violation('minimum_order_quantity', 'high', 'Order quantity below minimum threshold')
    publish_rule_violation('non_negativity', 'medium', 'Negative prediction detected')

    # Get summaries
    print("\n📋 Observer Summaries:")
    print("Training Observer:", training_observer.get_training_summary())
    print("Validation Observer:", validation_observer.get_validation_summary())
    print("Business Observer:", business_observer.get_violations_summary())

    print(f"\nTotal events in history: {len(event_publisher.event_history)}")
    print("Recent events:")
    for event in event_publisher.get_recent_events(5):
        print(f"  - {event}")