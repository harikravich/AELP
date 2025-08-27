"""
Production Integration Layer for GAELP Safety Framework
Orchestrates all production safety systems with real financial controls and regulatory compliance.
"""

import logging
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import uuid
import os
from contextlib import asynccontextmanager

# Import production safety modules
from .production_budget_controls import (
    RealMoneyBudgetController, ProductionBudgetLimits, PaymentMethod, 
    RealSpendRecord, EmergencyStopEvent as BudgetEmergencyStop
)
from .production_content_safety import (
    ProductionContentSafetyOrchestrator, ProductionContentItem, ModerationResult
)
from .emergency_controls import (
    EmergencyControlSystem, EmergencyLevel, ComplianceRegion, EmergencyContact
)
from .production_monitoring import (
    ProductionMonitoringOrchestrator, AlertSeverity, AlertRule
)

logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    INITIALIZING = "initializing"
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


class SafetyValidationResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"
    CONDITIONAL_APPROVAL = "conditional_approval"


@dataclass
class ProductionSafetyConfig:
    """Production safety configuration"""
    # Environment
    environment: str = "production"  # production, staging, development
    
    # Financial controls
    stripe_api_key: str = ""
    max_daily_global_spend: float = 1000000.0  # $1M daily limit
    fraud_detection_threshold: float = 0.7
    
    # Content safety
    openai_api_key: str = ""
    perspective_api_key: str = ""
    content_moderation_strict: bool = True
    
    # Emergency controls
    emergency_contacts: List[Dict[str, Any]] = field(default_factory=list)
    regulatory_notifications_enabled: bool = True
    
    # Monitoring
    prometheus_enabled: bool = True
    slack_bot_token: str = ""
    pagerduty_integration_key: str = ""
    
    # Cloud integrations
    gcp_project_id: str = ""
    bigquery_dataset: str = "safety_audit"
    
    # Compliance
    gdpr_enabled: bool = True
    ccpa_enabled: bool = True
    coppa_enabled: bool = True
    
    # Feature flags
    enable_real_money_controls: bool = True
    enable_ai_content_moderation: bool = True
    enable_emergency_stops: bool = True
    enable_real_time_monitoring: bool = True


@dataclass
class SafetyValidationRequest:
    """Request for safety validation"""
    request_id: str
    request_type: str  # 'campaign_creation', 'content_approval', 'spend_authorization'
    
    # Campaign data
    campaign_data: Optional[Dict[str, Any]] = None
    content_data: Optional[Dict[str, Any]] = None
    financial_data: Optional[Dict[str, Any]] = None
    
    # Context
    user_id: str = ""
    organization_id: str = ""
    target_regions: List[ComplianceRegion] = field(default_factory=list)
    target_platforms: List[str] = field(default_factory=list)
    
    # Compliance requirements
    gdpr_applicable: bool = False
    ccpa_applicable: bool = False
    coppa_applicable: bool = False
    
    # Metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SafetyValidationResponse:
    """Response from safety validation"""
    request_id: str
    result: SafetyValidationResult
    
    # Validation details
    budget_approved: bool = False
    content_approved: bool = False
    compliance_approved: bool = False
    
    # Issues found
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Required actions
    required_actions: List[str] = field(default_factory=list)
    human_review_required: bool = False
    
    # Financial controls
    approved_spend_limit: Optional[float] = None
    payment_method_restrictions: List[str] = field(default_factory=list)
    
    # Next steps
    can_proceed: bool = False
    retry_after: Optional[datetime] = None
    escalation_required: bool = False
    
    # Audit trail
    validation_time: float = 0.0
    validated_by: str = "system"
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ProductionSafetyOrchestrator:
    """
    Main orchestrator for production safety systems.
    Coordinates budget controls, content safety, emergency systems, and monitoring.
    """
    
    def __init__(self, config: ProductionSafetyConfig):
        self.config = config
        self.status = SystemStatus.INITIALIZING
        
        # Initialize core safety systems
        self.budget_controller = None
        self.content_safety = None
        self.emergency_controls = None
        self.monitoring = None
        
        # System state
        self._emergency_active = False
        self._master_kill_switch = False
        self._initialization_complete = False
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.validations_processed = 0
        self.emergencies_triggered = 0
        
        logger.info("Production safety orchestrator created")
    
    async def initialize(self) -> bool:
        """Initialize all production safety systems"""
        try:
            self.status = SystemStatus.INITIALIZING
            logger.info("Initializing production safety systems...")
            
            # Validate configuration
            config_valid = await self._validate_configuration()
            if not config_valid:
                logger.error("Configuration validation failed")
                return False
            
            # Initialize budget controls
            if self.config.enable_real_money_controls:
                await self._initialize_budget_controls()
            
            # Initialize content safety
            if self.config.enable_ai_content_moderation:
                await self._initialize_content_safety()
            
            # Initialize emergency controls
            if self.config.enable_emergency_stops:
                await self._initialize_emergency_controls()
            
            # Initialize monitoring
            if self.config.enable_real_time_monitoring:
                await self._initialize_monitoring()
            
            # Start monitoring loops
            await self._start_monitoring_loops()
            
            # Final health check
            health_check = await self._perform_health_check()
            if not health_check['healthy']:
                logger.error(f"Health check failed: {health_check['issues']}")
                self.status = SystemStatus.DEGRADED
            else:
                self.status = SystemStatus.OPERATIONAL
                self._initialization_complete = True
                logger.info("Production safety systems initialized successfully")
            
            return self._initialization_complete
            
        except Exception as e:
            logger.error(f"Safety system initialization failed: {e}")
            self.status = SystemStatus.SHUTDOWN
            return False
    
    async def _validate_configuration(self) -> bool:
        """Validate production configuration"""
        issues = []
        
        # Check required API keys
        if self.config.enable_real_money_controls and not self.config.stripe_api_key:
            issues.append("Stripe API key required for real money controls")
        
        if self.config.enable_ai_content_moderation and not self.config.openai_api_key:
            issues.append("OpenAI API key required for content moderation")
        
        # Check Cloud project
        if not self.config.gcp_project_id:
            issues.append("GCP project ID required for production deployment")
        
        # Check emergency contacts
        if self.config.enable_emergency_stops and not self.config.emergency_contacts:
            issues.append("Emergency contacts required for emergency stop system")
        
        # Check monitoring configuration
        if self.config.enable_real_time_monitoring:
            if not self.config.slack_bot_token and not self.config.pagerduty_integration_key:
                issues.append("At least one notification channel required for monitoring")
        
        if issues:
            logger.error(f"Configuration validation failed: {issues}")
            return False
        
        return True
    
    async def _initialize_budget_controls(self):
        """Initialize real money budget controls"""
        budget_config = {
            'stripe_api_key': self.config.stripe_api_key,
            'gcp_project_id': self.config.gcp_project_id,
            'fraud_threshold': self.config.fraud_detection_threshold
        }
        
        self.budget_controller = RealMoneyBudgetController(budget_config)
        logger.info("Real money budget controls initialized")
    
    async def _initialize_content_safety(self):
        """Initialize AI-powered content safety"""
        content_config = {
            'openai_api_key': self.config.openai_api_key,
            'perspective_api_key': self.config.perspective_api_key,
            'gcp_project_id': self.config.gcp_project_id,
            'strict_mode': self.config.content_moderation_strict,
            'gdpr_mode': self.config.gdpr_enabled,
            'ccpa_mode': self.config.ccpa_enabled
        }
        
        self.content_safety = ProductionContentSafetyOrchestrator(content_config)
        logger.info("AI content safety initialized")
    
    async def _initialize_emergency_controls(self):
        """Initialize emergency control systems"""
        emergency_config = {
            'emergency_contacts': self.config.emergency_contacts,
            'notifications': {
                'slack': {
                    'enabled': bool(self.config.slack_bot_token),
                    'bot_token': self.config.slack_bot_token
                },
                'pagerduty': {
                    'enabled': bool(self.config.pagerduty_integration_key),
                    'integration_key': self.config.pagerduty_integration_key
                },
                'email': {'enabled': True}  # Basic email always enabled
            },
            'compliance': {
                'enabled_regulations': [
                    reg for reg, enabled in [
                        ('GDPR', self.config.gdpr_enabled),
                        ('CCPA', self.config.ccpa_enabled),
                        ('COPPA', self.config.coppa_enabled)
                    ] if enabled
                ]
            },
            'gcp_project_id': self.config.gcp_project_id
        }
        
        self.emergency_controls = EmergencyControlSystem(emergency_config)
        logger.info("Emergency control systems initialized")
    
    async def _initialize_monitoring(self):
        """Initialize production monitoring"""
        monitoring_config = {
            'prometheus': {'enabled': self.config.prometheus_enabled},
            'alerting': {
                'notifications': {
                    'slack': {
                        'enabled': bool(self.config.slack_bot_token),
                        'bot_token': self.config.slack_bot_token,
                        'channel': '#safety-alerts'
                    },
                    'pagerduty': {
                        'enabled': bool(self.config.pagerduty_integration_key),
                        'integration_key': self.config.pagerduty_integration_key
                    }
                }
            },
            'gcp_project_id': self.config.gcp_project_id
        }
        
        self.monitoring = ProductionMonitoringOrchestrator(monitoring_config)
        await self.monitoring.start_monitoring()
        logger.info("Production monitoring initialized")
    
    async def _start_monitoring_loops(self):
        """Start background monitoring loops"""
        # Start system health monitoring
        asyncio.create_task(self._system_health_loop())
        
        # Start performance monitoring
        asyncio.create_task(self._performance_monitoring_loop())
        
        # Start compliance monitoring
        asyncio.create_task(self._compliance_monitoring_loop())
    
    async def validate_campaign_creation(self, request: SafetyValidationRequest) -> SafetyValidationResponse:
        """Comprehensive validation for campaign creation"""
        start_time = datetime.utcnow()
        response = SafetyValidationResponse(
            request_id=request.request_id,
            result=SafetyValidationResult.REJECTED  # Default to rejected
        )
        
        try:
            # Check system status
            if not self._can_process_request():
                response.warnings.append("System in emergency mode - request denied")
                response.escalation_required = True
                return response
            
            validation_tasks = []
            
            # Budget validation
            if request.financial_data and self.budget_controller:
                validation_tasks.append(self._validate_budget(request, response))
            
            # Content validation
            if request.content_data and self.content_safety:
                validation_tasks.append(self._validate_content(request, response))
            
            # Compliance validation
            validation_tasks.append(self._validate_compliance(request, response))
            
            # Execute all validations in parallel
            await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Determine overall result
            response.result = self._determine_validation_result(response)
            response.can_proceed = (response.result in [
                SafetyValidationResult.APPROVED,
                SafetyValidationResult.CONDITIONAL_APPROVAL
            ])
            
            # Record metrics
            await self._record_validation_metrics(request, response)
            
            self.validations_processed += 1
            response.validation_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Campaign validation completed: {response.result.value} in {response.validation_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Campaign validation failed: {e}")
            response.violations.append({
                'type': 'system_error',
                'severity': 'critical',
                'description': f"Validation system error: {str(e)}"
            })
            response.escalation_required = True
            return response
    
    async def _validate_budget(self, request: SafetyValidationRequest, 
                             response: SafetyValidationResponse):
        """Validate budget and financial requirements"""
        try:
            financial_data = request.financial_data
            
            # Create budget limits
            limits = ProductionBudgetLimits(
                daily_limit=financial_data.get('daily_limit', 1000.0),
                weekly_limit=financial_data.get('weekly_limit', 5000.0),
                monthly_limit=financial_data.get('monthly_limit', 20000.0),
                campaign_limit=financial_data.get('total_limit', 50000.0),
                roi_threshold=financial_data.get('min_roi', 0.1),
                cost_per_acquisition_limit=financial_data.get('max_cpa', 100.0)
            )
            
            # Validate payment method
            payment_method_id = financial_data.get('payment_method_id')
            if not payment_method_id:
                response.violations.append({
                    'type': 'missing_payment_method',
                    'severity': 'critical',
                    'description': 'Valid payment method required'
                })
                return
            
            # Check global spending limits
            if limits.daily_limit > self.config.max_daily_global_spend:
                response.violations.append({
                    'type': 'exceeds_global_limit',
                    'severity': 'high',
                    'description': f'Daily limit exceeds global maximum of ${self.config.max_daily_global_spend:,.2f}'
                })
                return
            
            # Validate with budget controller
            compliance_data = {
                'gdpr_applicable': request.gdpr_applicable,
                'ccpa_applicable': request.ccpa_applicable
            }
            
            budget_valid = await self.budget_controller.register_campaign(
                request.campaign_data.get('id', str(uuid.uuid4())),
                limits,
                payment_method_id,
                compliance_data
            )
            
            if budget_valid:
                response.budget_approved = True
                response.approved_spend_limit = float(limits.daily_limit)
            else:
                response.violations.append({
                    'type': 'budget_registration_failed',
                    'severity': 'critical',
                    'description': 'Failed to register campaign budget'
                })
                
        except Exception as e:
            logger.error(f"Budget validation failed: {e}")
            response.violations.append({
                'type': 'budget_validation_error',
                'severity': 'critical',
                'description': f'Budget validation error: {str(e)}'
            })
    
    async def _validate_content(self, request: SafetyValidationRequest, 
                              response: SafetyValidationResponse):
        """Validate content safety"""
        try:
            content_data = request.content_data
            
            # Create content items for validation
            content_items = []
            
            for content_type, content in content_data.items():
                if content_type in ['title', 'description', 'ad_copy']:
                    content_item = ProductionContentItem(
                        content_id=f"{request.request_id}_{content_type}",
                        content_type=self._map_content_type(content_type),
                        content=content,
                        campaign_id=request.campaign_data.get('id', ''),
                        target_audience=request.campaign_data.get('target_audience'),
                        geographic_targets=[region.value for region in request.target_regions],
                        platform_targets=request.target_platforms,
                        gdpr_applicable=request.gdpr_applicable,
                        ccpa_applicable=request.ccpa_applicable,
                        coppa_applicable=request.coppa_applicable,
                        submitted_by=request.user_id,
                        ip_address=request.ip_address,
                        user_agent=request.user_agent
                    )
                    content_items.append(content_item)
            
            # Moderate all content items
            all_approved = True
            content_violations = []
            
            for content_item in content_items:
                moderation_result = await self.content_safety.moderate_content(
                    content_item, request.target_platforms
                )
                
                if moderation_result.action.value in ['reject', 'block_permanently']:
                    all_approved = False
                
                # Add violations to response
                for violation in moderation_result.violations:
                    content_violations.append({
                        'type': 'content_violation',
                        'severity': violation.severity.value,
                        'description': violation.description,
                        'content_type': content_item.content_type.value,
                        'violation_category': violation.violation_type.value,
                        'confidence': violation.confidence
                    })
                
                # Check if human review is required
                if moderation_result.requires_human_review:
                    response.human_review_required = True
            
            response.content_approved = all_approved
            response.violations.extend(content_violations)
            
            if not all_approved:
                response.required_actions.append("Modify content to comply with platform policies")
                
        except Exception as e:
            logger.error(f"Content validation failed: {e}")
            response.violations.append({
                'type': 'content_validation_error',
                'severity': 'critical',
                'description': f'Content validation error: {str(e)}'
            })
    
    async def _validate_compliance(self, request: SafetyValidationRequest, 
                                 response: SafetyValidationResponse):
        """Validate regulatory compliance"""
        try:
            compliance_issues = []
            
            # GDPR compliance check
            if request.gdpr_applicable and self.config.gdpr_enabled:
                gdpr_issues = await self._check_gdpr_compliance(request)
                compliance_issues.extend(gdpr_issues)
            
            # CCPA compliance check
            if request.ccpa_applicable and self.config.ccpa_enabled:
                ccpa_issues = await self._check_ccpa_compliance(request)
                compliance_issues.extend(ccpa_issues)
            
            # COPPA compliance check
            if request.coppa_applicable and self.config.coppa_enabled:
                coppa_issues = await self._check_coppa_compliance(request)
                compliance_issues.extend(coppa_issues)
            
            # Age restrictions
            target_age = request.campaign_data.get('min_age', 18)
            if target_age < 13:
                compliance_issues.append({
                    'type': 'age_restriction_violation',
                    'severity': 'critical',
                    'description': 'Targeting users under 13 requires special compliance measures'
                })
            
            response.compliance_approved = len(compliance_issues) == 0
            response.violations.extend(compliance_issues)
            
            if compliance_issues:
                response.required_actions.append("Address compliance violations before proceeding")
                
        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            response.violations.append({
                'type': 'compliance_validation_error',
                'severity': 'critical',
                'description': f'Compliance validation error: {str(e)}'
            })
    
    async def trigger_emergency_stop(self, level: EmergencyLevel, reason: str, 
                                   triggered_by: str, context: Dict[str, Any] = None) -> str:
        """Trigger comprehensive emergency stop"""
        if not self.emergency_controls:
            logger.error("Emergency controls not initialized")
            return ""
        
        try:
            # Set system emergency state
            self._emergency_active = True
            if level in [EmergencyLevel.CRITICAL, EmergencyLevel.CATASTROPHIC]:
                self._master_kill_switch = True
                self.status = SystemStatus.EMERGENCY
            
            # Get affected campaigns and regions from context
            affected_campaigns = context.get('affected_campaigns', []) if context else []
            affected_regions = context.get('affected_regions', []) if context else []
            compliance_data = context.get('compliance_data', {}) if context else {}
            
            # Trigger emergency stop
            stop_id = await self.emergency_controls.trigger_emergency_stop(
                level=level,
                reason=reason,
                description=f"Emergency stop triggered: {reason}",
                triggered_by=triggered_by,
                affected_campaigns=affected_campaigns,
                affected_regions=affected_regions,
                compliance_data=compliance_data
            )
            
            # Stop financial operations if needed
            if level in [EmergencyLevel.HIGH, EmergencyLevel.CRITICAL, EmergencyLevel.CATASTROPHIC]:
                if self.budget_controller:
                    await self.budget_controller.emergency_stop_all_campaigns(
                        reason, triggered_by, level.value
                    )
            
            # Record emergency metrics
            if self.monitoring:
                await self.monitoring.record_safety_event(
                    'emergency_stop',
                    1.0,
                    {'level': level.value, 'reason': reason}
                )
            
            self.emergencies_triggered += 1
            
            logger.critical(f"EMERGENCY STOP ACTIVATED: Level {level.value} - {reason} [ID: {stop_id}]")
            return stop_id
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            # Fallback: activate maximum protection
            self._master_kill_switch = True
            self.status = SystemStatus.EMERGENCY
            return "emergency_stop_failed"
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = datetime.utcnow() - self.start_time
        
        status_data = {
            "system": {
                "status": self.status.value,
                "uptime_seconds": uptime.total_seconds(),
                "initialization_complete": self._initialization_complete,
                "emergency_active": self._emergency_active,
                "master_kill_switch": self._master_kill_switch
            },
            "components": {
                "budget_controls": {
                    "enabled": self.budget_controller is not None,
                    "operational": self.budget_controller is not None and not self._emergency_active
                },
                "content_safety": {
                    "enabled": self.content_safety is not None,
                    "operational": self.content_safety is not None
                },
                "emergency_controls": {
                    "enabled": self.emergency_controls is not None,
                    "operational": self.emergency_controls is not None
                },
                "monitoring": {
                    "enabled": self.monitoring is not None,
                    "operational": self.monitoring is not None
                }
            },
            "metrics": {
                "validations_processed": self.validations_processed,
                "emergencies_triggered": self.emergencies_triggered,
                "processing_rate": self.validations_processed / max(uptime.total_seconds(), 1)
            },
            "health_indicators": await self._get_health_indicators(),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return status_data
    
    # Helper methods
    
    def _can_process_request(self) -> bool:
        """Check if system can process requests"""
        return (
            self._initialization_complete and
            not self._master_kill_switch and
            self.status in [SystemStatus.OPERATIONAL, SystemStatus.DEGRADED]
        )
    
    def _determine_validation_result(self, response: SafetyValidationResponse) -> SafetyValidationResult:
        """Determine overall validation result"""
        # Check for critical violations
        critical_violations = [v for v in response.violations if v.get('severity') == 'critical']
        if critical_violations:
            return SafetyValidationResult.REJECTED
        
        # Check if human review is required
        if response.human_review_required:
            return SafetyValidationResult.REQUIRES_REVIEW
        
        # Check if all major components approved
        if response.budget_approved and response.content_approved and response.compliance_approved:
            return SafetyValidationResult.APPROVED
        
        # Check for conditional approval
        if (response.budget_approved and response.compliance_approved and 
            len([v for v in response.violations if v.get('severity') in ['high', 'critical']]) == 0):
            return SafetyValidationResult.CONDITIONAL_APPROVAL
        
        return SafetyValidationResult.REJECTED
    
    def _map_content_type(self, content_type_str: str):
        """Map string content type to enum"""
        from .production_content_safety import ContentType
        mapping = {
            'title': ContentType.TEXT,
            'description': ContentType.TEXT,
            'ad_copy': ContentType.TEXT,
            'image': ContentType.IMAGE,
            'video': ContentType.VIDEO,
            'url': ContentType.URL
        }
        return mapping.get(content_type_str, ContentType.TEXT)
    
    # Background monitoring loops
    
    async def _system_health_loop(self):
        """Monitor overall system health"""
        while True:
            try:
                health_check = await self._perform_health_check()
                
                if not health_check['healthy'] and self.status == SystemStatus.OPERATIONAL:
                    self.status = SystemStatus.DEGRADED
                    logger.warning(f"System degraded: {health_check['issues']}")
                elif health_check['healthy'] and self.status == SystemStatus.DEGRADED:
                    self.status = SystemStatus.OPERATIONAL
                    logger.info("System health restored")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring_loop(self):
        """Monitor system performance"""
        while True:
            try:
                if self.monitoring:
                    # Record system metrics
                    await self.monitoring.record_performance_metric(
                        "gaelp_validations_processed_total",
                        self.validations_processed
                    )
                    
                    await self.monitoring.record_performance_metric(
                        "gaelp_system_uptime_seconds",
                        (datetime.utcnow() - self.start_time).total_seconds()
                    )
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _compliance_monitoring_loop(self):
        """Monitor compliance status"""
        while True:
            try:
                # This would check for ongoing compliance issues
                # and trigger alerts if violations are detected
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Compliance monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        issues = []
        healthy = True
        
        # Check component health
        if self.budget_controller and self._emergency_active:
            issues.append("Budget controller in emergency mode")
            healthy = False
        
        if self.content_safety is None and self.config.enable_ai_content_moderation:
            issues.append("Content safety not initialized")
            healthy = False
        
        if self.emergency_controls is None and self.config.enable_emergency_stops:
            issues.append("Emergency controls not initialized")
            healthy = False
        
        if self.monitoring is None and self.config.enable_real_time_monitoring:
            issues.append("Monitoring not initialized")
            healthy = False
        
        return {
            'healthy': healthy,
            'issues': issues,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _get_health_indicators(self) -> Dict[str, Any]:
        """Get current health indicators"""
        indicators = {}
        
        if self.monitoring:
            monitoring_data = await self.monitoring.get_real_time_metrics()
            indicators['monitoring'] = monitoring_data.get('system_status', {})
        
        if self.budget_controller:
            # Would get budget controller health metrics
            indicators['budget_controls'] = {'operational': not self._emergency_active}
        
        if self.content_safety:
            # Would get content safety health metrics
            indicators['content_safety'] = {'operational': True}
        
        if self.emergency_controls:
            # Would get emergency controls health metrics
            indicators['emergency_controls'] = {'operational': True}
        
        return indicators
    
    async def _record_validation_metrics(self, request: SafetyValidationRequest, 
                                       response: SafetyValidationResponse):
        """Record validation metrics"""
        if self.monitoring:
            await self.monitoring.record_safety_event(
                'validation_completed',
                1.0,
                {
                    'result': response.result.value,
                    'request_type': request.request_type,
                    'human_review_required': str(response.human_review_required)
                }
            )
    
    # Compliance check methods
    
    async def _check_gdpr_compliance(self, request: SafetyValidationRequest) -> List[Dict[str, Any]]:
        """Check GDPR compliance"""
        issues = []
        
        # Check for explicit consent
        if not request.campaign_data.get('gdpr_consent_obtained'):
            issues.append({
                'type': 'gdpr_consent_missing',
                'severity': 'critical',
                'description': 'GDPR consent not obtained for EU data subjects'
            })
        
        # Check data processing lawful basis
        if not request.campaign_data.get('lawful_basis'):
            issues.append({
                'type': 'gdpr_lawful_basis_missing',
                'severity': 'high',
                'description': 'GDPR lawful basis for processing not specified'
            })
        
        return issues
    
    async def _check_ccpa_compliance(self, request: SafetyValidationRequest) -> List[Dict[str, Any]]:
        """Check CCPA compliance"""
        issues = []
        
        # Check for California residents
        if 'CA' in request.target_regions or 'US' in request.target_regions:
            if not request.campaign_data.get('ccpa_opt_out_provided'):
                issues.append({
                    'type': 'ccpa_opt_out_missing',
                    'severity': 'high',
                    'description': 'CCPA opt-out mechanism not provided for California residents'
                })
        
        return issues
    
    async def _check_coppa_compliance(self, request: SafetyValidationRequest) -> List[Dict[str, Any]]:
        """Check COPPA compliance"""
        issues = []
        
        # Check age targeting
        min_age = request.campaign_data.get('min_age', 18)
        if min_age < 13:
            if not request.campaign_data.get('parental_consent_obtained'):
                issues.append({
                    'type': 'coppa_parental_consent_missing',
                    'severity': 'critical',
                    'description': 'COPPA parental consent required for children under 13'
                })
        
        return issues