"""
Safety Orchestrator for GAELP Ad Campaign Safety
Central coordination of all safety mechanisms and comprehensive monitoring.
"""

import logging
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import uuid

from .budget_controls import BudgetController, BudgetMonitor, BudgetViolation
from .content_safety import ContentSafetyOrchestrator, ContentItem, ContentViolation
from .performance_safety import PerformanceSafetyOrchestrator, PerformanceDataPoint
from .operational_safety import OperationalSafetyOrchestrator, EmergencyLevel
from .data_safety import DataSafetyOrchestrator
from .agent_behavior_safety import AgentBehaviorSafetyOrchestrator, AgentAction

logger = logging.getLogger(__name__)


class SafetyEventType(Enum):
    BUDGET_VIOLATION = "budget_violation"
    CONTENT_VIOLATION = "content_violation"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    BEHAVIOR_VIOLATION = "behavior_violation"
    DATA_PRIVACY_ISSUE = "data_privacy_issue"
    EMERGENCY_STOP = "emergency_stop"
    SYSTEM_ERROR = "system_error"


class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyEvent:
    """Unified safety event across all safety modules"""
    event_id: str
    event_type: SafetyEventType
    safety_level: SafetyLevel
    timestamp: datetime
    source_module: str
    campaign_id: Optional[str]
    agent_id: Optional[str]
    description: str
    details: Dict[str, Any]
    actions_taken: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class SafetyConfiguration:
    """Configuration for safety systems"""
    enable_budget_controls: bool = True
    enable_content_safety: bool = True
    enable_performance_safety: bool = True
    enable_operational_safety: bool = True
    enable_data_safety: bool = True
    enable_behavior_safety: bool = True
    
    # Alert thresholds
    max_daily_budget: float = 10000.0
    content_violation_threshold: int = 3
    performance_anomaly_threshold: float = 0.8
    behavior_violation_threshold: int = 5
    
    # Integration settings
    alert_webhooks: List[str] = field(default_factory=list)
    human_review_required: bool = True
    auto_pause_on_critical: bool = True
    emergency_contacts: List[str] = field(default_factory=list)


class SafetyDashboard:
    """Real-time safety monitoring dashboard"""
    
    def __init__(self):
        self.metrics = {
            'total_campaigns_monitored': 0,
            'total_safety_events': 0,
            'active_violations': 0,
            'emergency_stops': 0,
            'system_health_score': 1.0,
            'last_updated': datetime.utcnow()
        }
        self.alerts = []
        self.system_status = {}
    
    def update_metrics(self, safety_events: List[SafetyEvent], 
                      system_status: Dict[str, Any]):
        """Update dashboard metrics"""
        try:
            now = datetime.utcnow()
            recent_events = [e for e in safety_events 
                           if e.timestamp > now - timedelta(hours=24)]
            
            self.metrics.update({
                'total_safety_events': len(safety_events),
                'recent_events_24h': len(recent_events),
                'active_violations': len([e for e in safety_events if not e.resolved]),
                'critical_events': len([e for e in recent_events 
                                      if e.safety_level == SafetyLevel.CRITICAL]),
                'emergency_stops': len([e for e in recent_events 
                                      if e.event_type == SafetyEventType.EMERGENCY_STOP]),
                'last_updated': now
            })
            
            # Calculate system health score
            self.metrics['system_health_score'] = self._calculate_health_score(
                recent_events, system_status
            )
            
            self.system_status = system_status
            
        except Exception as e:
            logger.error(f"Dashboard metrics update failed: {e}")
    
    def _calculate_health_score(self, recent_events: List[SafetyEvent], 
                              system_status: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-1)"""
        try:
            base_score = 1.0
            
            # Deduct for recent critical events
            critical_events = len([e for e in recent_events 
                                 if e.safety_level == SafetyLevel.CRITICAL])
            base_score -= critical_events * 0.1
            
            # Deduct for emergency stops
            emergency_events = len([e for e in recent_events 
                                  if e.event_type == SafetyEventType.EMERGENCY_STOP])
            base_score -= emergency_events * 0.2
            
            # Factor in system component health
            component_scores = []
            for component, status in system_status.items():
                if isinstance(status, dict) and 'health' in status:
                    component_scores.append(status['health'])
                elif status == 'operational':
                    component_scores.append(1.0)
                else:
                    component_scores.append(0.5)
            
            if component_scores:
                avg_component_health = sum(component_scores) / len(component_scores)
                base_score = (base_score + avg_component_health) / 2
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return 0.5
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for display"""
        return {
            'metrics': self.metrics,
            'system_status': self.system_status,
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'health_indicators': {
                'overall_health': self.metrics['system_health_score'],
                'components_operational': sum(
                    1 for status in self.system_status.values()
                    if status.get('health', 0) > 0.8
                ),
                'total_components': len(self.system_status)
            }
        }


class ComprehensiveSafetyOrchestrator:
    """
    Main safety orchestrator that coordinates all safety mechanisms
    and provides unified safety monitoring and intervention.
    """
    
    def __init__(self, config: SafetyConfiguration = None):
        self.config = config or SafetyConfiguration()
        
        # Initialize safety modules
        self.budget_controller = BudgetController(
            alert_callback=self._handle_budget_alert
        ) if self.config.enable_budget_controls else None
        
        self.content_safety = ContentSafetyOrchestrator() if self.config.enable_content_safety else None
        
        self.performance_safety = PerformanceSafetyOrchestrator(
            alert_callback=self._handle_performance_alert
        ) if self.config.enable_performance_safety else None
        
        self.operational_safety = OperationalSafetyOrchestrator() if self.config.enable_operational_safety else None
        
        self.data_safety = DataSafetyOrchestrator() if self.config.enable_data_safety else None
        
        self.behavior_safety = AgentBehaviorSafetyOrchestrator() if self.config.enable_behavior_safety else None
        
        # Safety event tracking
        self.safety_events: List[SafetyEvent] = []
        self.active_campaigns: Dict[str, Dict[str, Any]] = {}
        self.dashboard = SafetyDashboard()
        
        # Integration callbacks
        self.alert_callbacks: List[Callable] = []
        self.human_review_callbacks: List[Callable] = []
        
        # Start monitoring
        self._monitoring_task = None
        self._monitoring_active = False
    
    async def start_monitoring(self):
        """Start comprehensive safety monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start individual module monitoring
        if self.budget_controller:
            budget_monitor = BudgetMonitor(self.budget_controller)
            await budget_monitor.start_monitoring()
        
        logger.info("Comprehensive safety monitoring started")
    
    async def stop_monitoring(self):
        """Stop safety monitoring"""
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Safety monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                await self._update_safety_status()
                await self._process_safety_events()
                await self._update_dashboard()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _update_safety_status(self):
        """Update overall safety status"""
        try:
            system_status = {}
            
            # Check each safety module
            if self.budget_controller:
                system_status['budget_controls'] = {
                    'active_campaigns': len(self.budget_controller.campaigns),
                    'violations': len(self.budget_controller.violations),
                    'health': 1.0 if len(self.budget_controller.violations) == 0 else 0.7
                }
            
            if self.content_safety:
                stats = self.content_safety.get_moderation_stats()
                system_status['content_safety'] = {
                    'approval_rate': stats.get('approval_rate', 1.0),
                    'total_processed': stats.get('total_processed', 0),
                    'health': stats.get('approval_rate', 1.0)
                }
            
            if self.performance_safety:
                dashboard = self.performance_safety.get_safety_dashboard()
                anomaly_rate = (dashboard['performance_monitoring']['critical_anomalies'] / 
                              max(dashboard['performance_monitoring']['total_anomalies'], 1))
                system_status['performance_safety'] = {
                    'anomaly_rate': anomaly_rate,
                    'health': max(0.0, 1.0 - anomaly_rate)
                }
            
            if self.operational_safety:
                op_dashboard = self.operational_safety.get_safety_dashboard()
                system_status['operational_safety'] = {
                    'emergency_stops': op_dashboard['emergency_stops']['active'],
                    'health': 1.0 if op_dashboard['emergency_stops']['active'] == 0 else 0.3
                }
            
            if self.data_safety:
                privacy_dashboard = self.data_safety.get_privacy_dashboard()
                system_status['data_safety'] = {
                    'compliance_ready': privacy_dashboard['compliance_status']['gdpr_ready'],
                    'health': 1.0 if privacy_dashboard['compliance_status']['gdpr_ready'] else 0.5
                }
            
            if self.behavior_safety:
                behavior_dashboard = self.behavior_safety.get_safety_dashboard()
                system_status['behavior_safety'] = {
                    'active_interventions': behavior_dashboard['interventions']['active_interventions'],
                    'health': max(0.0, 1.0 - behavior_dashboard['interventions']['active_interventions'] * 0.1)
                }
            
            # Update dashboard
            self.dashboard.update_metrics(self.safety_events, system_status)
            
        except Exception as e:
            logger.error(f"Safety status update failed: {e}")
    
    async def _process_safety_events(self):
        """Process unresolved safety events"""
        try:
            unresolved_events = [e for e in self.safety_events if not e.resolved]
            
            for event in unresolved_events:
                # Check if event should auto-resolve
                if await self._should_auto_resolve_event(event):
                    event.resolved = True
                    event.resolution_time = datetime.utcnow()
                    logger.info(f"Safety event auto-resolved: {event.event_id}")
                
                # Check if escalation is needed
                elif await self._should_escalate_event(event):
                    await self._escalate_safety_event(event)
            
        except Exception as e:
            logger.error(f"Safety event processing failed: {e}")
    
    async def _should_auto_resolve_event(self, event: SafetyEvent) -> bool:
        """Check if a safety event can be auto-resolved"""
        try:
            # Events older than 24 hours with low severity can auto-resolve
            if (event.safety_level == SafetyLevel.WARNING and 
                event.timestamp < datetime.utcnow() - timedelta(hours=24)):
                return True
            
            # Budget violations that are no longer active
            if (event.event_type == SafetyEventType.BUDGET_VIOLATION and
                event.campaign_id):
                # Check if campaign is still violating budget
                # This would integrate with actual campaign status
                return False  # Conservative approach
            
            return False
            
        except Exception as e:
            logger.error(f"Auto-resolve check failed: {e}")
            return False
    
    async def _should_escalate_event(self, event: SafetyEvent) -> bool:
        """Check if a safety event needs escalation"""
        try:
            # Critical events older than 1 hour need escalation
            if (event.safety_level == SafetyLevel.CRITICAL and 
                event.timestamp < datetime.utcnow() - timedelta(hours=1)):
                return True
            
            # Multiple related events
            related_events = [
                e for e in self.safety_events
                if (e.campaign_id == event.campaign_id and 
                    e.event_type == event.event_type and
                    e.timestamp > datetime.utcnow() - timedelta(hours=1))
            ]
            
            if len(related_events) > 3:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Escalation check failed: {e}")
            return False
    
    async def _escalate_safety_event(self, event: SafetyEvent):
        """Escalate a safety event for human review"""
        try:
            logger.critical(f"ESCALATING SAFETY EVENT: {event.event_id}")
            
            # Trigger human review callbacks
            for callback in self.human_review_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Human review callback failed: {e}")
            
            # Add escalation to event actions
            event.actions_taken.append("escalated_to_human_review")
            
        except Exception as e:
            logger.error(f"Safety event escalation failed: {e}")
    
    async def _update_dashboard(self):
        """Update safety dashboard"""
        try:
            # This would update real-time dashboard displays
            dashboard_data = self.dashboard.get_dashboard_data()
            
            # Log critical metrics
            health_score = dashboard_data['metrics']['system_health_score']
            if health_score < 0.7:
                logger.warning(f"System health score low: {health_score:.2f}")
            
        except Exception as e:
            logger.error(f"Dashboard update failed: {e}")
    
    # Core safety validation methods
    
    async def validate_campaign_creation(self, campaign_config: Dict[str, Any], 
                                       creator_id: str) -> Tuple[bool, List[str]]:
        """Comprehensive validation for new campaign creation"""
        validation_results = []
        all_passed = True
        
        try:
            # Budget validation
            if self.budget_controller:
                budget = campaign_config.get('budget', 0)
                if budget > self.config.max_daily_budget:
                    validation_results.append(f"Budget {budget} exceeds maximum {self.config.max_daily_budget}")
                    all_passed = False
            
            # Content safety validation
            if self.content_safety:
                for content_key in ['title', 'description', 'ad_copy']:
                    if content_key in campaign_config:
                        content = ContentItem(
                            content_id=f"campaign_{content_key}",
                            content_type=ContentType.TEXT,
                            content=campaign_config[content_key],
                            campaign_id=campaign_config.get('id', 'new')
                        )
                        
                        is_approved, violations = await self.content_safety.moderate_content(
                            content, platform=campaign_config.get('platform')
                        )
                        
                        if not is_approved:
                            validation_results.extend([v.description for v in violations])
                            all_passed = False
            
            # Data safety validation
            if self.data_safety:
                targeting_data = campaign_config.get('targeting', {})
                safety_result = await self.data_safety.process_data_safely(
                    targeting_data, 'campaign_targeting', creator_id
                )
                
                if not safety_result['safe_to_process']:
                    validation_results.extend(safety_result['warnings'])
                    all_passed = False
            
            # Agent behavior validation
            if self.behavior_safety:
                action = AgentAction(
                    action_id=str(uuid.uuid4()),
                    agent_id=creator_id,
                    action_type=ActionType.CREATE_CAMPAIGN,
                    parameters=campaign_config,
                    timestamp=datetime.utcnow()
                )
                
                action_result = await self.behavior_safety.validate_and_execute_action(action)
                
                if not action_result['action_allowed']:
                    validation_results.extend(action_result['warnings'])
                    all_passed = False
            
            # Record validation event
            await self._record_safety_event(
                SafetyEventType.SYSTEM_ERROR if not all_passed else None,
                SafetyLevel.WARNING if not all_passed else SafetyLevel.SAFE,
                "campaign_validation",
                campaign_config.get('id'),
                creator_id,
                f"Campaign validation: {'PASSED' if all_passed else 'FAILED'}",
                {'validation_results': validation_results}
            )
            
            return all_passed, validation_results
            
        except Exception as e:
            logger.error(f"Campaign validation failed: {e}")
            return False, [f"Validation system error: {str(e)}"]
    
    async def monitor_campaign_activity(self, campaign_id: str, activity_data: Dict[str, Any]):
        """Monitor ongoing campaign activity for safety issues"""
        try:
            # Performance monitoring
            if self.performance_safety and 'performance_metrics' in activity_data:
                for metric_name, value in activity_data['performance_metrics'].items():
                    if hasattr(PerformanceMetric, metric_name.upper()):
                        metric = getattr(PerformanceMetric, metric_name.upper())
                        data_point = PerformanceDataPoint(
                            campaign_id=campaign_id,
                            metric=metric,
                            value=float(value),
                            timestamp=datetime.utcnow()
                        )
                        
                        result = await self.performance_safety.process_performance_data(data_point)
                        
                        if result.get('anomaly_detected'):
                            await self._record_safety_event(
                                SafetyEventType.PERFORMANCE_ANOMALY,
                                SafetyLevel.WARNING,
                                "performance_safety",
                                campaign_id,
                                None,
                                f"Performance anomaly detected: {result['anomaly']['description']}",
                                result
                            )
            
            # Budget monitoring
            if self.budget_controller and 'spend_data' in activity_data:
                # This would integrate with actual spend tracking
                pass
            
        except Exception as e:
            logger.error(f"Campaign activity monitoring failed: {e}")
    
    async def emergency_stop_system(self, reason: str, initiated_by: str, 
                                  level: EmergencyLevel = EmergencyLevel.HIGH) -> str:
        """Trigger emergency stop across all safety systems"""
        try:
            stop_id = str(uuid.uuid4())
            
            # Stop all campaigns via operational safety
            if self.operational_safety:
                emergency_id = await self.operational_safety.emergency_manager.trigger_emergency_stop(
                    level, reason, initiated_by
                )
            
            # Stop budget spending
            if self.budget_controller:
                stopped_count = await self.budget_controller.emergency_stop_all(reason)
                logger.critical(f"Emergency stop: {stopped_count} campaigns stopped")
            
            # Record emergency event
            await self._record_safety_event(
                SafetyEventType.EMERGENCY_STOP,
                SafetyLevel.EMERGENCY,
                "safety_orchestrator",
                None,
                initiated_by,
                f"Emergency stop triggered: {reason}",
                {
                    'stop_id': stop_id,
                    'level': level.value,
                    'reason': reason,
                    'initiated_by': initiated_by
                }
            )
            
            # Send critical alerts
            for callback in self.alert_callbacks:
                try:
                    await callback({
                        'type': 'emergency_stop',
                        'stop_id': stop_id,
                        'level': level.value,
                        'reason': reason,
                        'timestamp': datetime.utcnow()
                    })
                except Exception as e:
                    logger.error(f"Emergency alert callback failed: {e}")
            
            return stop_id
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return ""
    
    async def _record_safety_event(self, event_type: SafetyEventType, 
                                  safety_level: SafetyLevel, 
                                  source_module: str,
                                  campaign_id: str = None, 
                                  agent_id: str = None,
                                  description: str = "", 
                                  details: Dict[str, Any] = None):
        """Record a safety event"""
        try:
            if event_type is None:  # Skip recording for successful validations
                return
            
            event = SafetyEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                safety_level=safety_level,
                timestamp=datetime.utcnow(),
                source_module=source_module,
                campaign_id=campaign_id,
                agent_id=agent_id,
                description=description,
                details=details or {}
            )
            
            self.safety_events.append(event)
            
            # Trigger alerts for critical events
            if safety_level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
                for callback in self.alert_callbacks:
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to record safety event: {e}")
    
    # Callback management
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for safety alerts"""
        self.alert_callbacks.append(callback)
    
    def register_human_review_callback(self, callback: Callable):
        """Register callback for human review requests"""
        self.human_review_callbacks.append(callback)
    
    async def _handle_budget_alert(self, violation: BudgetViolation):
        """Handle budget violation alerts"""
        await self._record_safety_event(
            SafetyEventType.BUDGET_VIOLATION,
            SafetyLevel.CRITICAL if violation.violation_type.value in ['campaign_limit_exceeded'] else SafetyLevel.WARNING,
            "budget_controls",
            violation.campaign_id,
            None,
            violation.description,
            {
                'violation_type': violation.violation_type.value,
                'current_spend': float(violation.current_spend),
                'limit_exceeded': float(violation.limit_exceeded),
                'actions_taken': violation.actions_taken
            }
        )
    
    async def _handle_performance_alert(self, alert_data: Dict[str, Any]):
        """Handle performance anomaly alerts"""
        await self._record_safety_event(
            SafetyEventType.PERFORMANCE_ANOMALY,
            SafetyLevel.WARNING,
            "performance_safety",
            alert_data.get('campaign_id'),
            None,
            f"Performance anomaly: {alert_data.get('description', 'Unknown')}",
            alert_data
        )
    
    # Public API methods
    
    def get_comprehensive_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status across all systems"""
        try:
            dashboard_data = self.dashboard.get_dashboard_data()
            
            return {
                'overall_status': {
                    'health_score': dashboard_data['metrics']['system_health_score'],
                    'safety_level': self._determine_overall_safety_level(),
                    'monitoring_active': self._monitoring_active,
                    'last_updated': datetime.utcnow()
                },
                'module_status': dashboard_data['system_status'],
                'recent_events': [
                    {
                        'event_id': e.event_id,
                        'type': e.event_type.value,
                        'level': e.safety_level.value,
                        'timestamp': e.timestamp,
                        'description': e.description,
                        'resolved': e.resolved
                    }
                    for e in self.safety_events[-10:]  # Last 10 events
                ],
                'system_metrics': dashboard_data['metrics'],
                'alerts': dashboard_data['recent_alerts']
            }
        except Exception as e:
            logger.error(f"Failed to get safety status: {e}")
            return {'error': str(e)}
    
    def _determine_overall_safety_level(self) -> str:
        """Determine overall system safety level"""
        try:
            health_score = self.dashboard.metrics['system_health_score']
            active_violations = self.dashboard.metrics['active_violations']
            emergency_stops = self.dashboard.metrics.get('emergency_stops', 0)
            
            if emergency_stops > 0:
                return "EMERGENCY"
            elif health_score < 0.5 or active_violations > 10:
                return "CRITICAL"
            elif health_score < 0.8 or active_violations > 3:
                return "WARNING"
            else:
                return "SAFE"
                
        except Exception as e:
            logger.error(f"Safety level determination failed: {e}")
            return "UNKNOWN"
    
    def get_campaign_safety_report(self, campaign_id: str) -> Dict[str, Any]:
        """Get detailed safety report for a specific campaign"""
        try:
            campaign_events = [e for e in self.safety_events if e.campaign_id == campaign_id]
            
            return {
                'campaign_id': campaign_id,
                'safety_events': len(campaign_events),
                'unresolved_events': len([e for e in campaign_events if not e.resolved]),
                'safety_violations': [
                    {
                        'type': e.event_type.value,
                        'level': e.safety_level.value,
                        'timestamp': e.timestamp,
                        'description': e.description,
                        'resolved': e.resolved
                    }
                    for e in campaign_events
                ],
                'budget_status': self.budget_controller.get_spend_summary(campaign_id) if self.budget_controller else {},
                'performance_health': 'safe',  # Would integrate with performance safety
                'compliance_status': 'compliant',  # Would integrate with data safety
                'last_assessment': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Failed to generate campaign safety report: {e}")
            return {'error': str(e)}


# Integration helpers for external systems

async def create_safety_middleware(orchestrator: ComprehensiveSafetyOrchestrator):
    """Create middleware for integration with other GAELP components"""
    
    async def campaign_validation_middleware(campaign_data: Dict[str, Any], 
                                           creator_id: str) -> Tuple[bool, List[str]]:
        """Middleware for campaign validation"""
        return await orchestrator.validate_campaign_creation(campaign_data, creator_id)
    
    async def activity_monitoring_middleware(campaign_id: str, 
                                           activity_data: Dict[str, Any]):
        """Middleware for campaign activity monitoring"""
        await orchestrator.monitor_campaign_activity(campaign_id, activity_data)
    
    async def emergency_stop_middleware(reason: str, initiated_by: str) -> str:
        """Middleware for emergency stops"""
        return await orchestrator.emergency_stop_system(reason, initiated_by)
    
    return {
        'validate_campaign': campaign_validation_middleware,
        'monitor_activity': activity_monitoring_middleware,
        'emergency_stop': emergency_stop_middleware
    }