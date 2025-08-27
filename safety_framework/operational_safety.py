"""
Operational Safety Module for GAELP Ad Campaign Safety
Implements sandbox environments, deployment controls, and emergency procedures.
"""

import logging
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from pathlib import Path
import uuid
import shutil
import tempfile

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    SANDBOX = "sandbox"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStage(Enum):
    SIMULATION = "simulation"
    SMALL_BUDGET = "small_budget"
    MEDIUM_BUDGET = "medium_budget"
    FULL_DEPLOYMENT = "full_deployment"


class EmergencyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditEventType(Enum):
    CAMPAIGN_CREATED = "campaign_created"
    CAMPAIGN_MODIFIED = "campaign_modified"
    CAMPAIGN_PAUSED = "campaign_paused"
    CAMPAIGN_STOPPED = "campaign_stopped"
    BUDGET_EXCEEDED = "budget_exceeded"
    CONTENT_VIOLATION = "content_violation"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    EMERGENCY_ACTION = "emergency_action"
    HUMAN_INTERVENTION = "human_intervention"
    SYSTEM_ERROR = "system_error"


@dataclass
class SandboxConfig:
    """Configuration for sandbox environment"""
    max_budget: float = 10.0  # Maximum budget allowed in sandbox
    max_duration: timedelta = timedelta(hours=2)  # Maximum runtime
    allowed_platforms: Set[str] = field(default_factory=lambda: {"simulation"})
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        'cpu_limit': '100m',
        'memory_limit': '256Mi',
        'storage_limit': '1Gi'
    })
    network_isolation: bool = True
    monitoring_enabled: bool = True


@dataclass
class DeploymentConfig:
    """Configuration for graduated deployment"""
    simulation_budget: float = 0.0
    small_budget_limit: float = 50.0
    medium_budget_limit: float = 500.0
    stage_duration: timedelta = timedelta(days=1)
    success_threshold: float = 0.8  # Success rate to advance to next stage
    rollback_threshold: float = 0.3  # Performance threshold for rollback


@dataclass
class AuditEvent:
    """Audit log event"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    campaign_id: Optional[str]
    details: Dict[str, Any]
    severity: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class EmergencyStop:
    """Emergency stop event"""
    stop_id: str
    level: EmergencyLevel
    reason: str
    timestamp: datetime
    initiated_by: str
    affected_campaigns: List[str]
    actions_taken: List[str]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class SandboxManager:
    """Manages sandbox environments for safe testing"""
    
    def __init__(self, base_path: str = "/tmp/gaelp_sandbox"):
        self.base_path = Path(base_path)
        self.active_sandboxes: Dict[str, Dict[str, Any]] = {}
        self.sandbox_configs: Dict[str, SandboxConfig] = {}
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def create_sandbox(self, sandbox_id: str, config: SandboxConfig = None) -> bool:
        """Create a new sandbox environment"""
        try:
            if sandbox_id in self.active_sandboxes:
                logger.error(f"Sandbox {sandbox_id} already exists")
                return False
            
            if config is None:
                config = SandboxConfig()
            
            # Create sandbox directory
            sandbox_path = self.base_path / sandbox_id
            sandbox_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize sandbox state
            sandbox_state = {
                'id': sandbox_id,
                'path': str(sandbox_path),
                'config': config,
                'created_at': datetime.utcnow(),
                'status': 'active',
                'campaigns': {},
                'resource_usage': {
                    'cpu': 0,
                    'memory': 0,
                    'storage': 0,
                    'network': 0
                },
                'audit_log': []
            }
            
            self.active_sandboxes[sandbox_id] = sandbox_state
            self.sandbox_configs[sandbox_id] = config
            
            # Set up monitoring if enabled
            if config.monitoring_enabled:
                await self._setup_sandbox_monitoring(sandbox_id)
            
            logger.info(f"Sandbox {sandbox_id} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create sandbox {sandbox_id}: {e}")
            return False
    
    async def deploy_to_sandbox(self, sandbox_id: str, campaign_config: Dict[str, Any]) -> bool:
        """Deploy a campaign to sandbox for testing"""
        try:
            if sandbox_id not in self.active_sandboxes:
                logger.error(f"Sandbox {sandbox_id} not found")
                return False
            
            sandbox = self.active_sandboxes[sandbox_id]
            config = self.sandbox_configs[sandbox_id]
            
            # Validate campaign against sandbox limits
            campaign_budget = campaign_config.get('budget', 0)
            if campaign_budget > config.max_budget:
                logger.error(f"Campaign budget {campaign_budget} exceeds sandbox limit {config.max_budget}")
                return False
            
            # Check platform restrictions
            campaign_platform = campaign_config.get('platform', 'unknown')
            if campaign_platform not in config.allowed_platforms:
                logger.error(f"Platform {campaign_platform} not allowed in sandbox")
                return False
            
            # Deploy campaign in sandbox
            campaign_id = campaign_config.get('id', str(uuid.uuid4()))
            sandbox['campaigns'][campaign_id] = {
                'config': campaign_config,
                'deployed_at': datetime.utcnow(),
                'status': 'running',
                'metrics': {}
            }
            
            # Log deployment
            await self._log_sandbox_event(sandbox_id, 'campaign_deployed', {
                'campaign_id': campaign_id,
                'config': campaign_config
            })
            
            logger.info(f"Campaign {campaign_id} deployed to sandbox {sandbox_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy to sandbox {sandbox_id}: {e}")
            return False
    
    async def validate_sandbox_performance(self, sandbox_id: str) -> Dict[str, Any]:
        """Validate performance of campaigns in sandbox"""
        try:
            if sandbox_id not in self.active_sandboxes:
                return {'valid': False, 'error': 'Sandbox not found'}
            
            sandbox = self.active_sandboxes[sandbox_id]
            validation_result = {
                'valid': True,
                'campaigns': {},
                'overall_health': 'good',
                'issues': [],
                'recommendations': []
            }
            
            # Check each campaign
            for campaign_id, campaign_data in sandbox['campaigns'].items():
                campaign_validation = await self._validate_campaign_in_sandbox(
                    sandbox_id, campaign_id, campaign_data
                )
                validation_result['campaigns'][campaign_id] = campaign_validation
                
                if not campaign_validation['valid']:
                    validation_result['issues'].extend(campaign_validation.get('issues', []))
            
            # Determine overall health
            failed_campaigns = sum(1 for c in validation_result['campaigns'].values() if not c['valid'])
            total_campaigns = len(validation_result['campaigns'])
            
            if failed_campaigns == 0:
                validation_result['overall_health'] = 'good'
            elif failed_campaigns / max(total_campaigns, 1) < 0.3:
                validation_result['overall_health'] = 'warning'
            else:
                validation_result['overall_health'] = 'critical'
                validation_result['valid'] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate sandbox {sandbox_id}: {e}")
            return {'valid': False, 'error': str(e)}
    
    async def _validate_campaign_in_sandbox(self, sandbox_id: str, campaign_id: str, 
                                          campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a specific campaign in sandbox"""
        validation = {
            'valid': True,
            'issues': [],
            'performance': {},
            'resource_usage': {}
        }
        
        try:
            # Check if campaign is still within time limits
            config = self.sandbox_configs[sandbox_id]
            deployed_at = campaign_data['deployed_at']
            runtime = datetime.utcnow() - deployed_at
            
            if runtime > config.max_duration:
                validation['issues'].append(f"Campaign exceeded maximum duration {config.max_duration}")
                validation['valid'] = False
            
            # Check resource usage (simulated)
            resource_usage = campaign_data.get('resource_usage', {})
            for resource, limit in config.resource_limits.items():
                current_usage = resource_usage.get(resource, 0)
                # Simple validation - would be more sophisticated in real implementation
                if current_usage > 100:  # Arbitrary threshold
                    validation['issues'].append(f"High {resource} usage: {current_usage}")
            
            return validation
            
        except Exception as e:
            logger.error(f"Campaign validation failed: {e}")
            validation['valid'] = False
            validation['issues'].append(f"Validation error: {str(e)}")
            return validation
    
    async def cleanup_sandbox(self, sandbox_id: str) -> bool:
        """Clean up and remove sandbox environment"""
        try:
            if sandbox_id not in self.active_sandboxes:
                logger.warning(f"Sandbox {sandbox_id} not found for cleanup")
                return True
            
            sandbox = self.active_sandboxes[sandbox_id]
            
            # Stop all campaigns
            for campaign_id in sandbox['campaigns']:
                await self._stop_sandbox_campaign(sandbox_id, campaign_id)
            
            # Clean up files
            sandbox_path = Path(sandbox['path'])
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path)
            
            # Remove from active sandboxes
            del self.active_sandboxes[sandbox_id]
            del self.sandbox_configs[sandbox_id]
            
            logger.info(f"Sandbox {sandbox_id} cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup sandbox {sandbox_id}: {e}")
            return False
    
    async def _setup_sandbox_monitoring(self, sandbox_id: str):
        """Set up monitoring for sandbox environment"""
        # This would integrate with actual monitoring systems
        logger.info(f"Monitoring enabled for sandbox {sandbox_id}")
    
    async def _log_sandbox_event(self, sandbox_id: str, event_type: str, details: Dict[str, Any]):
        """Log an event in the sandbox"""
        event = {
            'timestamp': datetime.utcnow(),
            'event_type': event_type,
            'details': details
        }
        
        if sandbox_id in self.active_sandboxes:
            self.active_sandboxes[sandbox_id]['audit_log'].append(event)
    
    async def _stop_sandbox_campaign(self, sandbox_id: str, campaign_id: str):
        """Stop a campaign in sandbox"""
        if sandbox_id in self.active_sandboxes:
            campaigns = self.active_sandboxes[sandbox_id]['campaigns']
            if campaign_id in campaigns:
                campaigns[campaign_id]['status'] = 'stopped'
                await self._log_sandbox_event(sandbox_id, 'campaign_stopped', {'campaign_id': campaign_id})


class GraduatedDeployment:
    """Manages graduated deployment from simulation to production"""
    
    def __init__(self, config: DeploymentConfig = None):
        self.config = config or DeploymentConfig()
        self.deployments: Dict[str, Dict[str, Any]] = {}
        self.stage_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def start_deployment(self, campaign_id: str, campaign_config: Dict[str, Any]) -> bool:
        """Start graduated deployment for a campaign"""
        try:
            if campaign_id in self.deployments:
                logger.error(f"Deployment already in progress for campaign {campaign_id}")
                return False
            
            deployment = {
                'campaign_id': campaign_id,
                'campaign_config': campaign_config,
                'current_stage': DeploymentStage.SIMULATION,
                'started_at': datetime.utcnow(),
                'stage_started_at': datetime.utcnow(),
                'performance_metrics': {},
                'stage_results': {},
                'status': 'active'
            }
            
            self.deployments[campaign_id] = deployment
            self.stage_history[campaign_id] = []
            
            # Start with simulation
            await self._deploy_to_stage(campaign_id, DeploymentStage.SIMULATION)
            
            logger.info(f"Graduated deployment started for campaign {campaign_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start deployment for {campaign_id}: {e}")
            return False
    
    async def _deploy_to_stage(self, campaign_id: str, stage: DeploymentStage) -> bool:
        """Deploy campaign to a specific stage"""
        try:
            deployment = self.deployments[campaign_id]
            
            # Update deployment state
            deployment['current_stage'] = stage
            deployment['stage_started_at'] = datetime.utcnow()
            
            # Configure stage-specific parameters
            stage_config = await self._get_stage_config(stage, deployment['campaign_config'])
            
            # Record stage deployment
            stage_record = {
                'stage': stage,
                'started_at': datetime.utcnow(),
                'config': stage_config,
                'status': 'active'
            }
            self.stage_history[campaign_id].append(stage_record)
            
            logger.info(f"Campaign {campaign_id} deployed to stage {stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy {campaign_id} to stage {stage}: {e}")
            return False
    
    async def _get_stage_config(self, stage: DeploymentStage, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get stage-specific configuration"""
        stage_config = base_config.copy()
        
        if stage == DeploymentStage.SIMULATION:
            stage_config['budget'] = self.config.simulation_budget
            stage_config['platform'] = 'simulation'
            stage_config['real_money'] = False
        
        elif stage == DeploymentStage.SMALL_BUDGET:
            stage_config['budget'] = min(stage_config.get('budget', 0), self.config.small_budget_limit)
            stage_config['real_money'] = True
        
        elif stage == DeploymentStage.MEDIUM_BUDGET:
            stage_config['budget'] = min(stage_config.get('budget', 0), self.config.medium_budget_limit)
            stage_config['real_money'] = True
        
        elif stage == DeploymentStage.FULL_DEPLOYMENT:
            # Use original budget
            stage_config['real_money'] = True
        
        return stage_config
    
    async def evaluate_stage_progress(self, campaign_id: str, performance_data: Dict[str, Any]) -> bool:
        """Evaluate if campaign can progress to next stage"""
        try:
            if campaign_id not in self.deployments:
                logger.error(f"No deployment found for campaign {campaign_id}")
                return False
            
            deployment = self.deployments[campaign_id]
            current_stage = deployment['current_stage']
            
            # Update performance metrics
            deployment['performance_metrics'].update(performance_data)
            
            # Check stage duration
            stage_duration = datetime.utcnow() - deployment['stage_started_at']
            if stage_duration < self.config.stage_duration:
                return False  # Not enough time in current stage
            
            # Evaluate performance
            success_rate = performance_data.get('success_rate', 0)
            
            # Check for rollback condition
            if success_rate < self.config.rollback_threshold:
                await self._rollback_deployment(campaign_id, "Performance below rollback threshold")
                return False
            
            # Check for advancement
            if success_rate >= self.config.success_threshold:
                return await self._advance_to_next_stage(campaign_id)
            
            # Continue in current stage
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate stage progress for {campaign_id}: {e}")
            return False
    
    async def _advance_to_next_stage(self, campaign_id: str) -> bool:
        """Advance campaign to next deployment stage"""
        try:
            deployment = self.deployments[campaign_id]
            current_stage = deployment['current_stage']
            
            # Determine next stage
            stage_progression = {
                DeploymentStage.SIMULATION: DeploymentStage.SMALL_BUDGET,
                DeploymentStage.SMALL_BUDGET: DeploymentStage.MEDIUM_BUDGET,
                DeploymentStage.MEDIUM_BUDGET: DeploymentStage.FULL_DEPLOYMENT,
                DeploymentStage.FULL_DEPLOYMENT: None  # Already at final stage
            }
            
            next_stage = stage_progression.get(current_stage)
            
            if next_stage is None:
                # Campaign is fully deployed
                deployment['status'] = 'completed'
                logger.info(f"Campaign {campaign_id} graduated deployment completed")
                return True
            
            # Mark current stage as completed
            if self.stage_history[campaign_id]:
                self.stage_history[campaign_id][-1]['status'] = 'completed'
                self.stage_history[campaign_id][-1]['completed_at'] = datetime.utcnow()
            
            # Deploy to next stage
            await self._deploy_to_stage(campaign_id, next_stage)
            
            logger.info(f"Campaign {campaign_id} advanced from {current_stage.value} to {next_stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to advance {campaign_id} to next stage: {e}")
            return False
    
    async def _rollback_deployment(self, campaign_id: str, reason: str) -> bool:
        """Rollback deployment due to poor performance"""
        try:
            deployment = self.deployments[campaign_id]
            deployment['status'] = 'rolled_back'
            deployment['rollback_reason'] = reason
            deployment['rolled_back_at'] = datetime.utcnow()
            
            # Mark current stage as failed
            if self.stage_history[campaign_id]:
                self.stage_history[campaign_id][-1]['status'] = 'failed'
                self.stage_history[campaign_id][-1]['failed_at'] = datetime.utcnow()
                self.stage_history[campaign_id][-1]['failure_reason'] = reason
            
            logger.warning(f"Campaign {campaign_id} deployment rolled back: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback deployment for {campaign_id}: {e}")
            return False
    
    def get_deployment_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get current deployment status"""
        return self.deployments.get(campaign_id)


class AuditLogger:
    """Comprehensive audit logging for safety events"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file or "/tmp/gaelp_audit.log"
        self.audit_events: List[AuditEvent] = []
        self.event_handlers: Dict[AuditEventType, List[Callable]] = {}
    
    async def log_event(self, event_type: AuditEventType, details: Dict[str, Any],
                       user_id: str = None, campaign_id: str = None, 
                       severity: str = "info") -> str:
        """Log an audit event"""
        try:
            event_id = str(uuid.uuid4())
            
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                campaign_id=campaign_id,
                details=details,
                severity=severity
            )
            
            self.audit_events.append(event)
            
            # Write to log file
            await self._write_to_log_file(event)
            
            # Trigger event handlers
            await self._trigger_event_handlers(event)
            
            logger.info(f"Audit event logged: {event_type.value} [{event_id}]")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return ""
    
    async def _write_to_log_file(self, event: AuditEvent):
        """Write event to audit log file"""
        try:
            log_entry = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'user_id': event.user_id,
                'campaign_id': event.campaign_id,
                'details': event.details,
                'severity': event.severity
            }
            
            # Append to log file (in real implementation, use proper log rotation)
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write to audit log file: {e}")
    
    async def _trigger_event_handlers(self, event: AuditEvent):
        """Trigger registered event handlers"""
        try:
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed for {event.event_type}: {e}")
        except Exception as e:
            logger.error(f"Failed to trigger event handlers: {e}")
    
    def register_event_handler(self, event_type: AuditEventType, handler: Callable):
        """Register an event handler for specific event types"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_events(self, event_type: AuditEventType = None, 
                  campaign_id: str = None, 
                  since: datetime = None) -> List[AuditEvent]:
        """Get audit events with optional filtering"""
        events = self.audit_events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if campaign_id:
            events = [e for e in events if e.campaign_id == campaign_id]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for the last N hours"""
        since = datetime.utcnow() - timedelta(hours=hours)
        recent_events = self.get_events(since=since)
        
        summary = {
            'total_events': len(recent_events),
            'by_type': {},
            'by_severity': {},
            'critical_events': [],
            'active_campaigns': set()
        }
        
        for event in recent_events:
            # Count by type
            event_type = event.event_type.value
            summary['by_type'][event_type] = summary['by_type'].get(event_type, 0) + 1
            
            # Count by severity
            summary['by_severity'][event.severity] = summary['by_severity'].get(event.severity, 0) + 1
            
            # Track critical events
            if event.severity == 'critical':
                summary['critical_events'].append(event.event_id)
            
            # Track active campaigns
            if event.campaign_id:
                summary['active_campaigns'].add(event.campaign_id)
        
        summary['active_campaigns'] = list(summary['active_campaigns'])
        return summary


class EmergencyStopManager:
    """Manages emergency stop procedures"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.active_stops: Dict[str, EmergencyStop] = {}
        self.stop_procedures: Dict[EmergencyLevel, List[Callable]] = {}
    
    async def trigger_emergency_stop(self, level: EmergencyLevel, reason: str,
                                   initiated_by: str, affected_campaigns: List[str] = None) -> str:
        """Trigger emergency stop procedure"""
        try:
            stop_id = str(uuid.uuid4())
            
            emergency_stop = EmergencyStop(
                stop_id=stop_id,
                level=level,
                reason=reason,
                timestamp=datetime.utcnow(),
                initiated_by=initiated_by,
                affected_campaigns=affected_campaigns or [],
                actions_taken=[]
            )
            
            self.active_stops[stop_id] = emergency_stop
            
            # Execute stop procedures
            actions = await self._execute_stop_procedures(emergency_stop)
            emergency_stop.actions_taken = actions
            
            # Log emergency event
            await self.audit_logger.log_event(
                AuditEventType.EMERGENCY_ACTION,
                {
                    'stop_id': stop_id,
                    'level': level.value,
                    'reason': reason,
                    'affected_campaigns': affected_campaigns,
                    'actions_taken': actions
                },
                user_id=initiated_by,
                severity='critical'
            )
            
            logger.critical(f"Emergency stop triggered: {level.value} - {reason} [{stop_id}]")
            return stop_id
            
        except Exception as e:
            logger.error(f"Failed to trigger emergency stop: {e}")
            return ""
    
    async def _execute_stop_procedures(self, emergency_stop: EmergencyStop) -> List[str]:
        """Execute emergency stop procedures based on level"""
        actions_taken = []
        
        try:
            level = emergency_stop.level
            
            if level == EmergencyLevel.LOW:
                actions_taken.extend([
                    "alert_sent_to_operators",
                    "monitoring_increased"
                ])
            
            elif level == EmergencyLevel.MEDIUM:
                actions_taken.extend([
                    "campaigns_paused",
                    "human_review_triggered",
                    "stakeholders_notified"
                ])
                # Pause affected campaigns
                for campaign_id in emergency_stop.affected_campaigns:
                    await self._pause_campaign(campaign_id)
            
            elif level == EmergencyLevel.HIGH:
                actions_taken.extend([
                    "all_campaigns_paused",
                    "budget_limits_reduced",
                    "executive_team_notified",
                    "incident_response_activated"
                ])
                # Pause all campaigns
                await self._pause_all_campaigns()
            
            elif level == EmergencyLevel.CRITICAL:
                actions_taken.extend([
                    "complete_system_shutdown",
                    "all_spending_stopped",
                    "crisis_team_activated",
                    "external_stakeholders_notified"
                ])
                # Complete shutdown
                await self._complete_system_shutdown()
            
            # Execute custom procedures if registered
            procedures = self.stop_procedures.get(level, [])
            for procedure in procedures:
                try:
                    result = await procedure(emergency_stop)
                    if result:
                        actions_taken.append(result)
                except Exception as e:
                    logger.error(f"Emergency procedure failed: {e}")
                    actions_taken.append(f"procedure_failed: {str(e)}")
            
            return actions_taken
            
        except Exception as e:
            logger.error(f"Failed to execute stop procedures: {e}")
            return ["execution_failed"]
    
    async def _pause_campaign(self, campaign_id: str):
        """Pause a specific campaign"""
        # This would integrate with the campaign management system
        logger.critical(f"EMERGENCY: Pausing campaign {campaign_id}")
    
    async def _pause_all_campaigns(self):
        """Pause all active campaigns"""
        # This would integrate with the campaign management system
        logger.critical("EMERGENCY: Pausing all campaigns")
    
    async def _complete_system_shutdown(self):
        """Complete system shutdown"""
        # This would trigger complete platform shutdown
        logger.critical("EMERGENCY: Complete system shutdown initiated")
    
    async def resolve_emergency_stop(self, stop_id: str, resolved_by: str) -> bool:
        """Resolve an emergency stop"""
        try:
            if stop_id not in self.active_stops:
                logger.error(f"Emergency stop {stop_id} not found")
                return False
            
            emergency_stop = self.active_stops[stop_id]
            emergency_stop.resolved = True
            emergency_stop.resolution_time = datetime.utcnow()
            
            # Log resolution
            await self.audit_logger.log_event(
                AuditEventType.EMERGENCY_ACTION,
                {
                    'stop_id': stop_id,
                    'action': 'resolved',
                    'resolved_by': resolved_by,
                    'resolution_time': emergency_stop.resolution_time.isoformat()
                },
                user_id=resolved_by,
                severity='info'
            )
            
            logger.info(f"Emergency stop {stop_id} resolved by {resolved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve emergency stop {stop_id}: {e}")
            return False
    
    def register_stop_procedure(self, level: EmergencyLevel, procedure: Callable):
        """Register custom emergency stop procedure"""
        if level not in self.stop_procedures:
            self.stop_procedures[level] = []
        self.stop_procedures[level].append(procedure)
    
    def get_active_stops(self) -> List[EmergencyStop]:
        """Get all active emergency stops"""
        return [stop for stop in self.active_stops.values() if not stop.resolved]
    
    def get_stop_history(self, hours: int = 24) -> List[EmergencyStop]:
        """Get emergency stop history"""
        since = datetime.utcnow() - timedelta(hours=hours)
        return [
            stop for stop in self.active_stops.values()
            if stop.timestamp >= since
        ]


class OperationalSafetyOrchestrator:
    """Main orchestrator for operational safety"""
    
    def __init__(self):
        self.sandbox_manager = SandboxManager()
        self.graduated_deployment = GraduatedDeployment()
        self.audit_logger = AuditLogger()
        self.emergency_manager = EmergencyStopManager(self.audit_logger)
        
        # Register default emergency procedures
        self._register_default_procedures()
    
    def _register_default_procedures(self):
        """Register default emergency procedures"""
        
        async def high_level_procedure(emergency_stop: EmergencyStop) -> str:
            """Default high-level emergency procedure"""
            # Could trigger external alerts, API calls, etc.
            return "default_high_level_procedure_executed"
        
        async def critical_level_procedure(emergency_stop: EmergencyStop) -> str:
            """Default critical-level emergency procedure"""
            # Could trigger external escalations, regulatory notifications, etc.
            return "default_critical_level_procedure_executed"
        
        self.emergency_manager.register_stop_procedure(EmergencyLevel.HIGH, high_level_procedure)
        self.emergency_manager.register_stop_procedure(EmergencyLevel.CRITICAL, critical_level_procedure)
    
    async def create_test_environment(self, test_id: str, campaign_config: Dict[str, Any]) -> bool:
        """Create a complete test environment for campaign"""
        try:
            # Create sandbox
            sandbox_config = SandboxConfig(
                max_budget=campaign_config.get('test_budget', 10.0),
                max_duration=timedelta(hours=campaign_config.get('test_duration_hours', 2))
            )
            
            sandbox_created = await self.sandbox_manager.create_sandbox(test_id, sandbox_config)
            if not sandbox_created:
                return False
            
            # Deploy to sandbox
            deployment_success = await self.sandbox_manager.deploy_to_sandbox(test_id, campaign_config)
            if not deployment_success:
                await self.sandbox_manager.cleanup_sandbox(test_id)
                return False
            
            # Log test environment creation
            await self.audit_logger.log_event(
                AuditEventType.CAMPAIGN_CREATED,
                {
                    'test_id': test_id,
                    'environment': 'sandbox',
                    'config': campaign_config
                },
                campaign_id=test_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create test environment {test_id}: {e}")
            return False
    
    async def promote_to_production(self, campaign_id: str, campaign_config: Dict[str, Any]) -> bool:
        """Promote campaign from testing to production using graduated deployment"""
        try:
            # Start graduated deployment
            deployment_started = await self.graduated_deployment.start_deployment(campaign_id, campaign_config)
            if not deployment_started:
                return False
            
            # Log promotion
            await self.audit_logger.log_event(
                AuditEventType.CAMPAIGN_CREATED,
                {
                    'campaign_id': campaign_id,
                    'environment': 'production',
                    'deployment_type': 'graduated',
                    'config': campaign_config
                },
                campaign_id=campaign_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote campaign {campaign_id} to production: {e}")
            return False
    
    def get_safety_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive operational safety dashboard"""
        try:
            return {
                'sandbox_environments': {
                    'active': len(self.sandbox_manager.active_sandboxes),
                    'total_campaigns': sum(
                        len(sandbox['campaigns']) 
                        for sandbox in self.sandbox_manager.active_sandboxes.values()
                    )
                },
                'graduated_deployments': {
                    'active': len([d for d in self.graduated_deployment.deployments.values() 
                                 if d['status'] == 'active']),
                    'completed': len([d for d in self.graduated_deployment.deployments.values() 
                                    if d['status'] == 'completed']),
                    'rolled_back': len([d for d in self.graduated_deployment.deployments.values() 
                                      if d['status'] == 'rolled_back'])
                },
                'audit_summary': self.audit_logger.get_audit_summary(),
                'emergency_stops': {
                    'active': len(self.emergency_manager.get_active_stops()),
                    'recent': len(self.emergency_manager.get_stop_history())
                },
                'system_health': {
                    'operational': True,
                    'last_updated': datetime.utcnow(),
                    'components_status': {
                        'sandbox_manager': 'operational',
                        'graduated_deployment': 'operational',
                        'audit_logger': 'operational',
                        'emergency_manager': 'operational'
                    }
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate safety dashboard: {e}")
            return {'error': str(e)}