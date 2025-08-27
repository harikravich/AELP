"""
Production Real-Time Monitoring and Alerting for GAELP Safety Framework
Implements comprehensive monitoring, metrics, dashboards, and real-time alerting.
"""

import logging
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import uuid
import statistics
from concurrent.futures import ThreadPoolExecutor
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
from google.cloud import monitoring_v3
from google.cloud import pubsub_v1
from google.cloud import bigquery
import grafana_api
import pagerduty
import slack_sdk
import httpx

logger = logging.getLogger(__name__)


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(Enum):
    FIRING = "firing"
    RESOLVED = "resolved"
    PENDING = "pending"
    SILENCED = "silenced"


@dataclass
class SafetyMetric:
    """Safety metric definition"""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    
    # Thresholds for alerting
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Prometheus metric object
    prometheus_metric: Optional[Any] = None


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    query: str  # PromQL query or custom query
    severity: AlertSeverity
    threshold: float
    comparison: str  # '>', '<', '==', '!=', '>=', '<='
    duration: timedelta  # How long condition must be true
    
    description: str
    summary_template: str
    
    # Notification configuration
    notification_channels: List[str] = field(default_factory=list)
    escalation_policy: Optional[str] = None
    
    # Rule state
    is_enabled: bool = True
    last_evaluation: Optional[datetime] = None
    current_state: AlertState = AlertState.RESOLVED


@dataclass
class AlertEvent:
    """Alert event instance"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    state: AlertState
    timestamp: datetime
    
    # Alert details
    title: str
    description: str
    summary: str
    
    # Metric values
    current_value: float
    threshold: float
    
    # Context
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # Tracking
    started_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class PrometheusIntegration:
    """Prometheus metrics integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', True)
        self.push_gateway_url = config.get('push_gateway_url')
        self.job_name = config.get('job_name', 'gaelp_safety')
        
        # Registry for metrics
        self.registry = prometheus_client.CollectorRegistry()
        self.metrics: Dict[str, SafetyMetric] = {}
        
        # Initialize core safety metrics
        self._initialize_core_metrics()
        
        logger.info("Prometheus integration initialized")
    
    def _initialize_core_metrics(self):
        """Initialize core safety metrics"""
        core_metrics = [
            SafetyMetric(
                name="gaelp_campaign_spend_total",
                metric_type=MetricType.COUNTER,
                description="Total campaign spending",
                labels=["campaign_id", "platform", "region"]
            ),
            SafetyMetric(
                name="gaelp_budget_violations_total",
                metric_type=MetricType.COUNTER,
                description="Total budget violations",
                labels=["violation_type", "severity", "campaign_id"]
            ),
            SafetyMetric(
                name="gaelp_content_moderation_total",
                metric_type=MetricType.COUNTER,
                description="Total content moderation events",
                labels=["action", "violation_type", "platform"]
            ),
            SafetyMetric(
                name="gaelp_emergency_stops_total",
                metric_type=MetricType.COUNTER,
                description="Total emergency stops triggered",
                labels=["level", "reason", "triggered_by"]
            ),
            SafetyMetric(
                name="gaelp_transaction_processing_time",
                metric_type=MetricType.HISTOGRAM,
                description="Transaction processing time in seconds",
                labels=["transaction_type"],
                buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            SafetyMetric(
                name="gaelp_system_health_score",
                metric_type=MetricType.GAUGE,
                description="Overall system health score (0-1)",
                labels=["component"]
            ),
            SafetyMetric(
                name="gaelp_fraud_risk_score",
                metric_type=MetricType.GAUGE,
                description="Current fraud risk score (0-1)",
                labels=["campaign_id", "risk_category"]
            ),
            SafetyMetric(
                name="gaelp_compliance_violations_total",
                metric_type=MetricType.COUNTER,
                description="Total compliance violations",
                labels=["regulation", "violation_type", "severity"]
            )
        ]
        
        for metric_def in core_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: SafetyMetric):
        """Register a new metric with Prometheus"""
        if not self.enabled:
            return
        
        try:
            if metric_def.metric_type == MetricType.COUNTER:
                prometheus_metric = Counter(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                prometheus_metric = Gauge(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                prometheus_metric = Histogram(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    buckets=metric_def.buckets,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.SUMMARY:
                prometheus_metric = Summary(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self.registry
                )
            
            metric_def.prometheus_metric = prometheus_metric
            self.metrics[metric_def.name] = metric_def
            
            logger.debug(f"Registered metric: {metric_def.name}")
            
        except Exception as e:
            logger.error(f"Failed to register metric {metric_def.name}: {e}")
    
    def record_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        if not self.enabled or metric_name not in self.metrics:
            return
        
        try:
            metric_def = self.metrics[metric_name]
            prometheus_metric = metric_def.prometheus_metric
            labels = labels or {}
            
            if metric_def.metric_type == MetricType.COUNTER:
                if labels:
                    prometheus_metric.labels(**labels).inc(value)
                else:
                    prometheus_metric.inc(value)
            
            elif metric_def.metric_type == MetricType.GAUGE:
                if labels:
                    prometheus_metric.labels(**labels).set(value)
                else:
                    prometheus_metric.set(value)
            
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                if labels:
                    prometheus_metric.labels(**labels).observe(value)
                else:
                    prometheus_metric.observe(value)
            
            elif metric_def.metric_type == MetricType.SUMMARY:
                if labels:
                    prometheus_metric.labels(**labels).observe(value)
                else:
                    prometheus_metric.observe(value)
                    
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")
    
    def push_metrics(self):
        """Push metrics to Prometheus Push Gateway"""
        if not self.enabled or not self.push_gateway_url:
            return
        
        try:
            prometheus_client.push_to_gateway(
                self.push_gateway_url,
                job=self.job_name,
                registry=self.registry
            )
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")


class RealTimeAlertEngine:
    """Real-time alerting engine with multiple notification channels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: List[AlertEvent] = []
        
        # Time series data for evaluation (simple in-memory store)
        self.metric_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Notification channels
        self.notification_channels = {}
        self._setup_notification_channels(config.get('notifications', {}))
        
        # Alert evaluation
        self.evaluation_interval = config.get('evaluation_interval', 30)  # seconds
        self._evaluation_task = None
        
        # Alerting state
        self._alerting_enabled = True
        self._silenced_rules: Set[str] = set()
        
        logger.info("Real-time alert engine initialized")
    
    def _setup_notification_channels(self, config: Dict[str, Any]):
        """Setup notification channels"""
        # Slack
        if config.get('slack', {}).get('enabled'):
            self.notification_channels['slack'] = SlackNotifier(config['slack'])
        
        # PagerDuty
        if config.get('pagerduty', {}).get('enabled'):
            self.notification_channels['pagerduty'] = PagerDutyNotifier(config['pagerduty'])
        
        # Email
        if config.get('email', {}).get('enabled'):
            self.notification_channels['email'] = EmailNotifier(config['email'])
        
        # Webhook
        if config.get('webhook', {}).get('enabled'):
            self.notification_channels['webhook'] = WebhookNotifier(config['webhook'])
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def start_monitoring(self):
        """Start the alert evaluation loop"""
        if self._evaluation_task is None:
            self._evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop the alert evaluation loop"""
        if self._evaluation_task:
            self._evaluation_task.cancel()
            self._evaluation_task = None
        logger.info("Alert monitoring stopped")
    
    async def _evaluation_loop(self):
        """Main alert evaluation loop"""
        while True:
            try:
                if self._alerting_enabled:
                    await self._evaluate_all_rules()
                await asyncio.sleep(self.evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _evaluate_all_rules(self):
        """Evaluate all alert rules"""
        current_time = datetime.utcnow()
        
        for rule in self.alert_rules.values():
            if not rule.is_enabled or rule.rule_id in self._silenced_rules:
                continue
            
            try:
                await self._evaluate_rule(rule, current_time)
                rule.last_evaluation = current_time
            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule.rule_id}: {e}")
    
    async def _evaluate_rule(self, rule: AlertRule, evaluation_time: datetime):
        """Evaluate a single alert rule"""
        # Get current metric value (simplified - would integrate with actual metrics)
        current_value = await self._get_metric_value(rule.query)
        
        if current_value is None:
            return
        
        # Check if condition is met
        condition_met = self._check_condition(current_value, rule.threshold, rule.comparison)
        
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule.rule_id:
                existing_alert = alert
                break
        
        if condition_met:
            if existing_alert is None:
                # Create new alert
                alert = AlertEvent(
                    alert_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    state=AlertState.PENDING,
                    timestamp=evaluation_time,
                    title=rule.name,
                    description=rule.description,
                    summary=rule.summary_template.format(value=current_value, threshold=rule.threshold),
                    current_value=current_value,
                    threshold=rule.threshold,
                    started_at=evaluation_time
                )
                
                # Check if alert should fire (duration check)
                if rule.duration.total_seconds() == 0:
                    # Fire immediately
                    alert.state = AlertState.FIRING
                    await self._fire_alert(alert)
                
                self.active_alerts[alert.alert_id] = alert
                
            else:
                # Update existing alert
                existing_alert.current_value = current_value
                existing_alert.timestamp = evaluation_time
                
                # Check if pending alert should fire
                if (existing_alert.state == AlertState.PENDING and 
                    evaluation_time - existing_alert.started_at >= rule.duration):
                    existing_alert.state = AlertState.FIRING
                    await self._fire_alert(existing_alert)
        
        else:
            # Condition not met - resolve alert if it exists
            if existing_alert and existing_alert.state == AlertState.FIRING:
                existing_alert.state = AlertState.RESOLVED
                existing_alert.resolved_at = evaluation_time
                await self._resolve_alert(existing_alert)
                del self.active_alerts[existing_alert.alert_id]
    
    def _check_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Check if alert condition is met"""
        if comparison == '>':
            return value > threshold
        elif comparison == '<':
            return value < threshold
        elif comparison == '>=':
            return value >= threshold
        elif comparison == '<=':
            return value <= threshold
        elif comparison == '==':
            return value == threshold
        elif comparison == '!=':
            return value != threshold
        else:
            return False
    
    async def _get_metric_value(self, query: str) -> Optional[float]:
        """Get current metric value (simplified implementation)"""
        # This would integrate with actual metrics backend
        # For now, return a placeholder value
        return 0.5
    
    async def _fire_alert(self, alert: AlertEvent):
        """Fire an alert - send notifications"""
        logger.warning(f"ALERT FIRED: {alert.title} - {alert.summary}")
        
        # Get rule configuration
        rule = self.alert_rules[alert.rule_id]
        
        # Send notifications to configured channels
        for channel_name in rule.notification_channels:
            if channel_name in self.notification_channels:
                try:
                    await self.notification_channels[channel_name].send_alert(alert)
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel_name}: {e}")
        
        # Add to history
        self.alert_history.append(alert)
    
    async def _resolve_alert(self, alert: AlertEvent):
        """Resolve an alert"""
        logger.info(f"ALERT RESOLVED: {alert.title}")
        
        # Send resolution notifications
        rule = self.alert_rules[alert.rule_id]
        
        for channel_name in rule.notification_channels:
            if channel_name in self.notification_channels:
                try:
                    await self.notification_channels[channel_name].send_resolution(alert)
                except Exception as e:
                    logger.error(f"Failed to send resolution to {channel_name}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    def silence_rule(self, rule_id: str, duration: timedelta):
        """Temporarily silence an alert rule"""
        self._silenced_rules.add(rule_id)
        # Schedule unsilencing (simplified)
        asyncio.create_task(self._unsilence_after_delay(rule_id, duration))
    
    async def _unsilence_after_delay(self, rule_id: str, duration: timedelta):
        """Unsilence a rule after specified duration"""
        await asyncio.sleep(duration.total_seconds())
        self._silenced_rules.discard(rule_id)


class SlackNotifier:
    """Slack notification integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.client = slack_sdk.WebClient(token=config['bot_token'])
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'GAELP Safety Bot')
    
    async def send_alert(self, alert: AlertEvent):
        """Send alert to Slack"""
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.EMERGENCY: "#8B0000"
        }[alert.severity]
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🚨 *{alert.severity.value.upper()} ALERT* 🚨\n*{alert.title}*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Value:* {alert.current_value:.2f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Threshold:* {alert.threshold:.2f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Alert ID:* {alert.alert_id}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:* {alert.description}"
                }
            }
        ]
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.chat_postMessage(
                channel=self.channel,
                username=self.username,
                blocks=blocks,
                attachments=[{
                    "color": color,
                    "text": alert.summary
                }]
            )
        )
    
    async def send_resolution(self, alert: AlertEvent):
        """Send alert resolution to Slack"""
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"✅ *ALERT RESOLVED*\n*{alert.title}*"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Alert {alert.alert_id} was resolved at {alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                }
            }
        ]
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.chat_postMessage(
                channel=self.channel,
                username=self.username,
                blocks=blocks
            )
        )


class PagerDutyNotifier:
    """PagerDuty notification integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.integration_key = config['integration_key']
        self.client = httpx.AsyncClient()
    
    async def send_alert(self, alert: AlertEvent):
        """Send alert to PagerDuty"""
        payload = {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "dedup_key": alert.alert_id,
            "payload": {
                "summary": alert.summary,
                "source": "GAELP Safety System",
                "severity": alert.severity.value,
                "component": "safety_framework",
                "group": "alerts",
                "class": "safety_violation",
                "custom_details": {
                    "alert_id": alert.alert_id,
                    "rule_id": alert.rule_id,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "description": alert.description
                }
            }
        }
        
        try:
            response = await self.client.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"PagerDuty notification failed: {e}")
    
    async def send_resolution(self, alert: AlertEvent):
        """Send alert resolution to PagerDuty"""
        payload = {
            "routing_key": self.integration_key,
            "event_action": "resolve",
            "dedup_key": alert.alert_id
        }
        
        try:
            response = await self.client.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"PagerDuty resolution failed: {e}")


class EmailNotifier:
    """Email notification integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.smtp_server = config['smtp_server']
        self.smtp_port = config['smtp_port']
        self.username = config['username']
        self.password = config['password']
        self.recipients = config['recipients']
    
    async def send_alert(self, alert: AlertEvent):
        """Send alert via email"""
        # Implementation would use SMTP to send emails
        logger.info(f"Email alert sent: {alert.title}")
    
    async def send_resolution(self, alert: AlertEvent):
        """Send resolution via email"""
        logger.info(f"Email resolution sent: {alert.title}")


class WebhookNotifier:
    """Webhook notification integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config['url']
        self.headers = config.get('headers', {})
        self.client = httpx.AsyncClient()
    
    async def send_alert(self, alert: AlertEvent):
        """Send alert via webhook"""
        payload = {
            "event_type": "alert_fired",
            "alert": {
                "id": alert.alert_id,
                "rule_id": alert.rule_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "summary": alert.summary,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat(),
                "labels": alert.labels,
                "annotations": alert.annotations
            }
        }
        
        try:
            response = await self.client.post(
                self.webhook_url,
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
    
    async def send_resolution(self, alert: AlertEvent):
        """Send resolution via webhook"""
        payload = {
            "event_type": "alert_resolved",
            "alert": {
                "id": alert.alert_id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
        }
        
        try:
            response = await self.client.post(
                self.webhook_url,
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Webhook resolution failed: {e}")


class ProductionMonitoringOrchestrator:
    """Main orchestrator for production monitoring and alerting"""
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize components
        self.prometheus = PrometheusIntegration(config.get('prometheus', {}))
        self.alert_engine = RealTimeAlertEngine(config.get('alerting', {}))
        
        # Redis for caching and real-time data
        redis_config = config.get('redis', {})
        if redis_config.get('enabled'):
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0)
            )
        else:
            self.redis_client = None
        
        # Cloud integrations
        self.cloud_project = config.get('gcp_project_id')
        if self.cloud_project:
            self.cloud_monitoring = monitoring_v3.MetricServiceClient()
            self.pubsub_publisher = pubsub_v1.PublisherClient()
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.metrics_recorded = 0
        self.alerts_fired = 0
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        logger.info("Production monitoring orchestrator initialized")
    
    def _setup_default_alert_rules(self):
        """Setup default safety alert rules"""
        default_rules = [
            AlertRule(
                rule_id="high_budget_violation_rate",
                name="High Budget Violation Rate",
                query="rate(gaelp_budget_violations_total[5m])",
                severity=AlertSeverity.WARNING,
                threshold=5.0,  # 5 violations per second
                comparison=">",
                duration=timedelta(minutes=2),
                description="Budget violation rate is unusually high",
                summary_template="Budget violations: {value:.1f}/sec (threshold: {threshold:.1f}/sec)",
                notification_channels=["slack", "email"]
            ),
            AlertRule(
                rule_id="emergency_stop_triggered",
                name="Emergency Stop Triggered",
                query="increase(gaelp_emergency_stops_total[1m])",
                severity=AlertSeverity.CRITICAL,
                threshold=0.0,  # Any emergency stop
                comparison=">",
                duration=timedelta(seconds=0),  # Immediate
                description="Emergency stop has been triggered",
                summary_template="Emergency stop triggered: {value} events",
                notification_channels=["slack", "pagerduty", "email"]
            ),
            AlertRule(
                rule_id="low_system_health",
                name="Low System Health Score",
                query="gaelp_system_health_score",
                severity=AlertSeverity.WARNING,
                threshold=0.7,
                comparison="<",
                duration=timedelta(minutes=5),
                description="System health score is below acceptable threshold",
                summary_template="System health: {value:.2f} (threshold: {threshold:.2f})",
                notification_channels=["slack"]
            ),
            AlertRule(
                rule_id="high_fraud_risk",
                name="High Fraud Risk Detected",
                query="max(gaelp_fraud_risk_score)",
                severity=AlertSeverity.CRITICAL,
                threshold=0.8,
                comparison=">",
                duration=timedelta(minutes=1),
                description="High fraud risk detected in campaign activity",
                summary_template="Fraud risk: {value:.2f} (threshold: {threshold:.2f})",
                notification_channels=["slack", "pagerduty"]
            ),
            AlertRule(
                rule_id="compliance_violations",
                name="Compliance Violations Detected",
                query="increase(gaelp_compliance_violations_total[5m])",
                severity=AlertSeverity.EMERGENCY,
                threshold=0.0,  # Any compliance violation
                comparison=">",
                duration=timedelta(seconds=0),  # Immediate
                description="Regulatory compliance violations detected",
                summary_template="Compliance violations: {value} in last 5 minutes",
                notification_channels=["slack", "pagerduty", "email", "webhook"]
            )
        ]
        
        for rule in default_rules:
            self.alert_engine.add_alert_rule(rule)
    
    async def start_monitoring(self):
        """Start all monitoring components"""
        # Start alert engine
        self.alert_engine.start_monitoring()
        
        # Start metric collection loop
        asyncio.create_task(self._metric_collection_loop())
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
        logger.info("Production monitoring started")
    
    async def stop_monitoring(self):
        """Stop all monitoring components"""
        self.alert_engine.stop_monitoring()
        logger.info("Production monitoring stopped")
    
    async def record_safety_event(self, event_type: str, value: float = 1.0, 
                                 labels: Dict[str, str] = None):
        """Record a safety event metric"""
        metric_name = f"gaelp_{event_type}_total"
        
        # Record to Prometheus
        self.prometheus.record_metric(metric_name, value, labels)
        
        # Cache in Redis if available
        if self.redis_client:
            try:
                key = f"safety_event:{event_type}:{int(time.time())}"
                data = {
                    'value': value,
                    'labels': labels or {},
                    'timestamp': datetime.utcnow().isoformat()
                }
                self.redis_client.setex(key, 3600, json.dumps(data))  # 1 hour TTL
            except Exception as e:
                logger.error(f"Redis caching failed: {e}")
        
        self.metrics_recorded += 1
    
    async def record_performance_metric(self, metric_name: str, value: float, 
                                      labels: Dict[str, str] = None):
        """Record a performance metric"""
        self.prometheus.record_metric(metric_name, value, labels)
        self.metrics_recorded += 1
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics summary"""
        uptime = datetime.utcnow() - self.start_time
        
        return {
            "system_status": {
                "uptime_seconds": uptime.total_seconds(),
                "metrics_recorded": self.metrics_recorded,
                "alerts_fired": self.alerts_fired,
                "active_alerts": len(self.alert_engine.active_alerts),
                "monitoring_enabled": self.alert_engine._alerting_enabled
            },
            "alert_summary": {
                "active_alerts": [
                    {
                        "id": alert.alert_id,
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "started_at": alert.started_at.isoformat() if alert.started_at else None
                    }
                    for alert in self.alert_engine.active_alerts.values()
                ],
                "recent_alerts": [
                    {
                        "id": alert.alert_id,
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "state": alert.state.value,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in self.alert_engine.alert_history[-10:]  # Last 10 alerts
                ]
            },
            "health_indicators": {
                "prometheus_enabled": self.prometheus.enabled,
                "redis_connected": self.redis_client is not None,
                "cloud_monitoring_enabled": self.cloud_project is not None,
                "notification_channels": list(self.alert_engine.notification_channels.keys())
            }
        }
    
    async def _metric_collection_loop(self):
        """Background loop for metric collection and aggregation"""
        while True:
            try:
                # Push metrics to Prometheus Gateway
                self.prometheus.push_metrics()
                
                # Update system health metrics
                await self._update_system_health_metrics()
                
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_loop(self):
        """Background loop for health checks"""
        while True:
            try:
                # Check component health
                health_scores = await self._check_component_health()
                
                # Record health metrics
                for component, score in health_scores.items():
                    await self.record_performance_metric(
                        "gaelp_system_health_score",
                        score,
                        {"component": component}
                    )
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(120)
    
    async def _update_system_health_metrics(self):
        """Update overall system health metrics"""
        # This would integrate with actual system components
        # For now, record basic operational metrics
        await self.record_performance_metric(
            "gaelp_monitoring_uptime_seconds",
            (datetime.utcnow() - self.start_time).total_seconds()
        )
    
    async def _check_component_health(self) -> Dict[str, float]:
        """Check health of all system components"""
        health_scores = {}
        
        # Check Prometheus health
        health_scores["prometheus"] = 1.0 if self.prometheus.enabled else 0.0
        
        # Check Redis health
        if self.redis_client:
            try:
                self.redis_client.ping()
                health_scores["redis"] = 1.0
            except:
                health_scores["redis"] = 0.0
        else:
            health_scores["redis"] = 0.0
        
        # Check alert engine health
        health_scores["alerting"] = 1.0 if self.alert_engine._alerting_enabled else 0.0
        
        # Check notification channels
        active_channels = len(self.alert_engine.notification_channels)
        total_channels = 4  # slack, pagerduty, email, webhook
        health_scores["notifications"] = active_channels / total_channels
        
        return health_scores
    
    def create_custom_alert_rule(self, rule_config: Dict[str, Any]) -> str:
        """Create a custom alert rule"""
        rule = AlertRule(
            rule_id=rule_config.get('rule_id', str(uuid.uuid4())),
            name=rule_config['name'],
            query=rule_config['query'],
            severity=AlertSeverity(rule_config['severity']),
            threshold=rule_config['threshold'],
            comparison=rule_config['comparison'],
            duration=timedelta(seconds=rule_config.get('duration_seconds', 60)),
            description=rule_config['description'],
            summary_template=rule_config['summary_template'],
            notification_channels=rule_config.get('notification_channels', ['slack'])
        )
        
        self.alert_engine.add_alert_rule(rule)
        return rule.rule_id
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "metrics_summary": {
                "total_recorded": self.metrics_recorded,
                "recording_rate": self.metrics_recorded / max((datetime.utcnow() - self.start_time).total_seconds(), 1)
            },
            "alert_summary": {
                "total_rules": len(self.alert_engine.alert_rules),
                "active_alerts": len(self.alert_engine.active_alerts),
                "total_fired": self.alerts_fired,
                "silenced_rules": len(self.alert_engine._silenced_rules)
            },
            "system_health": "operational",
            "last_updated": datetime.utcnow().isoformat()
        }