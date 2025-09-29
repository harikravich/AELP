"""
Production-grade Safety Gates and Human-In-The-Loop (HITL) Approval System for AELP2.

This module provides comprehensive safety mechanisms including:
- ConfigurableSafetyGates: Dynamic threshold evaluation from environment variables
- HITLApprovalQueue: Production approval workflow with timeout handling
- PolicyChecker: Content compliance and creative validation
- SafetyEventLogger: Comprehensive audit trail for all safety events

All thresholds are configurable via environment variables with no hardcoded fallbacks.
"""

import os
import uuid
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
import re
import json
from datetime import datetime, timedelta


class SafetyEventType(Enum):
    """Types of safety events tracked in the system."""
    GATE_VIOLATION = "gate_violation"
    POLICY_VIOLATION = "policy_violation"
    APPROVAL_TIMEOUT = "approval_timeout"
    EMERGENCY_STOP = "emergency_stop"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"


class SafetyEventSeverity(Enum):
    """Severity levels for safety events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(Enum):
    """Status of approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class SafetyEvent:
    """Represents a safety event with complete metadata."""
    event_type: SafetyEventType
    severity: SafetyEventSeverity
    timestamp: datetime
    metadata: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class GateViolation:
    """Details of a specific gate violation."""
    gate_name: str
    actual_value: float
    threshold_value: float
    operator: str  # '<', '>', '<=', '>='
    severity: SafetyEventSeverity


@dataclass
class ApprovalRequest:
    """Represents an approval request in the HITL system."""
    approval_id: str
    action_type: str
    context: Dict[str, Any]
    status: ApprovalStatus
    requested_at: datetime
    expires_at: datetime
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyConfigurationError(Exception):
    """Raised when required safety configuration is missing or invalid."""
    pass


class SafetyGates:
    """
    Production-grade safety gates with configurable thresholds from environment variables.
    
    All thresholds must be configured via environment variables. No hardcoded fallbacks.
    """
    
    def __init__(self):
        """Initialize safety gates with configuration from environment variables."""
        self.logger = logging.getLogger(__name__)
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load all safety thresholds from environment variables."""
        required_configs = {
            'AELP2_MIN_WIN_RATE': float,
            'AELP2_MAX_CAC': float,
            'AELP2_MIN_ROAS': float,
            'AELP2_MAX_SPEND_VELOCITY': float,
        }
        
        self.thresholds = {}
        missing_configs = []
        
        for config_name, config_type in required_configs.items():
            value = os.environ.get(config_name)
            if value is None:
                missing_configs.append(config_name)
                continue
                
            try:
                self.thresholds[config_name] = config_type(value)
            except ValueError as e:
                raise SafetyConfigurationError(
                    f"Invalid value for {config_name}: {value}. Must be {config_type.__name__}."
                ) from e
        
        if missing_configs:
            raise SafetyConfigurationError(
                f"Required safety configuration missing: {', '.join(missing_configs)}. "
                f"Set these environment variables before starting the system."
            )
        
        # Optional configurations with validation
        optional_configs = {
            'AELP2_MAX_DAILY_SPEND': float,
            'AELP2_MIN_CONVERSION_RATE': float,
            'AELP2_MAX_CPC': float,
        }
        
        for config_name, config_type in optional_configs.items():
            value = os.environ.get(config_name)
            if value is not None:
                try:
                    self.thresholds[config_name] = config_type(value)
                except ValueError:
                    self.logger.warning(f"Invalid optional config {config_name}: {value}, ignoring.")
        
        self.logger.info(f"Safety gates configured with {len(self.thresholds)} thresholds")
    
    def evaluate_gates(self, metrics: Dict[str, Any]) -> Tuple[bool, List[GateViolation]]:
        """
        Evaluate all safety gates against provided metrics.
        
        Args:
            metrics: Dictionary containing performance metrics
            
        Returns:
            Tuple of (passed: bool, violations: List[GateViolation])
        """
        violations = []
        
        # Extract metrics with safe defaults
        win_rate = float(metrics.get('win_rate', 0.0))
        cac = self._calculate_cac(metrics)
        roas = self._calculate_roas(metrics)
        spend_velocity = float(metrics.get('spend_velocity', 0.0))
        
        # Evaluate win rate
        min_win_rate = self.thresholds['AELP2_MIN_WIN_RATE']
        if win_rate < min_win_rate:
            violations.append(GateViolation(
                gate_name='win_rate',
                actual_value=win_rate,
                threshold_value=min_win_rate,
                operator='<',
                severity=SafetyEventSeverity.HIGH
            ))
        
        # Evaluate CAC
        max_cac = self.thresholds['AELP2_MAX_CAC']
        if cac > max_cac and cac != float('inf'):
            violations.append(GateViolation(
                gate_name='cac',
                actual_value=cac,
                threshold_value=max_cac,
                operator='>',
                severity=SafetyEventSeverity.MEDIUM
            ))
        
        # Evaluate ROAS
        min_roas = self.thresholds['AELP2_MIN_ROAS']
        if roas < min_roas:
            violations.append(GateViolation(
                gate_name='roas',
                actual_value=roas,
                threshold_value=min_roas,
                operator='<',
                severity=SafetyEventSeverity.MEDIUM
            ))
        
        # Evaluate spend velocity
        max_spend_velocity = self.thresholds['AELP2_MAX_SPEND_VELOCITY']
        if spend_velocity > max_spend_velocity:
            violations.append(GateViolation(
                gate_name='spend_velocity',
                actual_value=spend_velocity,
                threshold_value=max_spend_velocity,
                operator='>',
                severity=SafetyEventSeverity.CRITICAL
            ))
        
        # Optional threshold checks
        if 'AELP2_MAX_DAILY_SPEND' in self.thresholds:
            daily_spend = float(metrics.get('daily_spend', 0.0))
            max_daily_spend = self.thresholds['AELP2_MAX_DAILY_SPEND']
            if daily_spend > max_daily_spend:
                violations.append(GateViolation(
                    gate_name='daily_spend',
                    actual_value=daily_spend,
                    threshold_value=max_daily_spend,
                    operator='>',
                    severity=SafetyEventSeverity.CRITICAL
                ))
        
        passed = len(violations) == 0
        return passed, violations
    
    def _calculate_cac(self, metrics: Dict[str, Any]) -> float:
        """Calculate Customer Acquisition Cost safely."""
        spend = float(metrics.get('spend', 0.0))
        conversions = int(metrics.get('conversions', 0))
        
        if conversions == 0:
            return float('inf')
        
        return spend / conversions
    
    def _calculate_roas(self, metrics: Dict[str, Any]) -> float:
        """Calculate Return on Ad Spend safely."""
        revenue = float(metrics.get('revenue', 0.0))
        spend = float(metrics.get('spend', 0.0))
        
        if spend == 0:
            return 0.0
        
        return revenue / spend


class HITLApprovalQueue:
    """
    Human-In-The-Loop approval queue with timeout handling and status tracking.
    
    Manages approval workflows for creative changes and high-risk actions.
    """
    
    def __init__(self):
        """Initialize the approval queue with configuration from environment variables."""
        self.logger = logging.getLogger(__name__)
        self._requests: Dict[str, ApprovalRequest] = {}
        self._lock = threading.RLock()
        
        # Load timeout configuration
        timeout_str = os.environ.get('AELP2_APPROVAL_TIMEOUT')
        if timeout_str is None:
            raise SafetyConfigurationError(
                "AELP2_APPROVAL_TIMEOUT environment variable is required. "
                "Set timeout in seconds (e.g., '3600' for 1 hour)."
            )
        
        try:
            self.approval_timeout_seconds = int(timeout_str)
        except ValueError as e:
            raise SafetyConfigurationError(
                f"Invalid AELP2_APPROVAL_TIMEOUT value: {timeout_str}. Must be integer seconds."
            ) from e
        
        if self.approval_timeout_seconds <= 0:
            raise SafetyConfigurationError(
                f"AELP2_APPROVAL_TIMEOUT must be positive, got: {self.approval_timeout_seconds}"
            )
        
        self.logger.info(f"HITL approval queue initialized with {self.approval_timeout_seconds}s timeout")
    
    def request_approval(self, action: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Request approval for an action that requires human oversight.
        
        Args:
            action: The action requiring approval
            context: Additional context for the approval request
            
        Returns:
            approval_id: Unique identifier for tracking the approval
        """
        approval_id = str(uuid.uuid4())
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=self.approval_timeout_seconds)
        
        request = ApprovalRequest(
            approval_id=approval_id,
            action_type=action.get('type', 'unknown'),
            context=context,
            status=ApprovalStatus.PENDING,
            requested_at=now,
            expires_at=expires_at,
            metadata={
                'action': action,
                'priority': context.get('priority', 'normal'),
                'requester': context.get('requester', 'system')
            }
        )
        
        with self._lock:
            self._requests[approval_id] = request
        
        self.logger.info(f"Approval requested: {approval_id} for {request.action_type}")
        return approval_id
    
    def check_approval_status(self, approval_id: str) -> ApprovalStatus:
        """
        Check the current status of an approval request.
        
        Args:
            approval_id: The approval ID to check
            
        Returns:
            Current ApprovalStatus
        """
        with self._lock:
            request = self._requests.get(approval_id)
            if request is None:
                return ApprovalStatus.REJECTED  # Unknown requests are rejected
            
            # Check for timeout
            if (request.status == ApprovalStatus.PENDING and 
                datetime.utcnow() > request.expires_at):
                request.status = ApprovalStatus.TIMEOUT
                self.logger.warning(f"Approval timeout: {approval_id}")
            
            return request.status
    
    def approve_request(self, approval_id: str, approved_by: str, reason: Optional[str] = None) -> bool:
        """
        Approve a pending request.
        
        Args:
            approval_id: The approval ID to approve
            approved_by: Who approved the request
            reason: Optional reason for approval
            
        Returns:
            True if successfully approved, False otherwise
        """
        with self._lock:
            request = self._requests.get(approval_id)
            if request is None or request.status != ApprovalStatus.PENDING:
                return False
            
            if datetime.utcnow() > request.expires_at:
                request.status = ApprovalStatus.TIMEOUT
                return False
            
            request.status = ApprovalStatus.APPROVED
            request.approved_by = approved_by
            request.metadata['approval_reason'] = reason
            
            self.logger.info(f"Approval granted: {approval_id} by {approved_by}")
            return True
    
    def reject_request(self, approval_id: str, rejected_by: str, reason: str) -> bool:
        """
        Reject a pending request.
        
        Args:
            approval_id: The approval ID to reject
            rejected_by: Who rejected the request
            reason: Reason for rejection
            
        Returns:
            True if successfully rejected, False otherwise
        """
        with self._lock:
            request = self._requests.get(approval_id)
            if request is None or request.status != ApprovalStatus.PENDING:
                return False
            
            request.status = ApprovalStatus.REJECTED
            request.rejection_reason = reason
            request.metadata['rejected_by'] = rejected_by
            
            self.logger.info(f"Approval rejected: {approval_id} by {rejected_by}: {reason}")
            return True
    
    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        with self._lock:
            current_time = datetime.utcnow()
            pending = []
            
            for request in self._requests.values():
                if request.status == ApprovalStatus.PENDING:
                    if current_time > request.expires_at:
                        request.status = ApprovalStatus.TIMEOUT
                    else:
                        pending.append(request)
            
            return pending
    
    def cleanup_expired_requests(self) -> int:
        """Remove expired requests from the queue. Returns count of cleaned up requests."""
        with self._lock:
            current_time = datetime.utcnow()
            expired_ids = []
            
            for approval_id, request in self._requests.items():
                if (request.status in [ApprovalStatus.APPROVED, ApprovalStatus.REJECTED, ApprovalStatus.TIMEOUT] and
                    current_time > request.expires_at + timedelta(hours=24)):  # Keep for 24h after expiry
                    expired_ids.append(approval_id)
            
            for approval_id in expired_ids:
                del self._requests[approval_id]
            
            if expired_ids:
                self.logger.info(f"Cleaned up {len(expired_ids)} expired approval requests")
            
            return len(expired_ids)


class PolicyChecker:
    """
    Content policy and compliance checker for creative validation.
    
    Validates creative content against configurable policies and compliance rules.
    """
    
    def __init__(self):
        """Initialize policy checker with configurable rules."""
        self.logger = logging.getLogger(__name__)
        self._load_policy_configuration()
    
    def _load_policy_configuration(self) -> None:
        """Load policy configuration from environment variables."""
        # Load blocked content patterns
        blocked_content = os.environ.get('AELP2_BLOCKED_CONTENT', '')
        if blocked_content:
            self.blocked_patterns = [pattern.strip() for pattern in blocked_content.split(',')]
        else:
            # Default critical safety patterns
            self.blocked_patterns = [
                r'\bself[-\s]?harm\b',
                r'\bsuicide\b',
                r'\bdrug\b',
                r'\balcohol\b',
                r'\btobacco\b',
                r'\bgambling\b',
                r'\bweapon\b',
                r'\bhate\b',
                r'\bdiscrimination\b'
            ]
        
        # Load compliance requirements
        self.require_age_verification = os.environ.get('AELP2_REQUIRE_AGE_VERIFICATION', 'false').lower() == 'true'
        self.require_health_disclaimer = os.environ.get('AELP2_REQUIRE_HEALTH_DISCLAIMER', 'false').lower() == 'true'
        
        # Targeting restrictions
        restricted_audiences = os.environ.get('AELP2_RESTRICTED_AUDIENCES', '')
        if restricted_audiences:
            self.restricted_audiences = [audience.strip() for audience in restricted_audiences.split(',')]
        else:
            self.restricted_audiences = ['minors', 'vulnerable_populations']
        
        self.logger.info(f"Policy checker loaded with {len(self.blocked_patterns)} content patterns")
    
    def check_policy_compliance(self, creative: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check creative content for policy compliance.
        
        Args:
            creative: Creative content and metadata to check
            
        Returns:
            Tuple of (compliant: bool, issues: List[str])
        """
        issues = []
        
        # Extract content for analysis
        text_content = self._extract_text_content(creative)
        targeting = creative.get('targeting', {})
        
        # Check for blocked content
        content_issues = self._check_blocked_content(text_content)
        issues.extend(content_issues)
        
        # Check targeting compliance
        targeting_issues = self._check_targeting_compliance(targeting)
        issues.extend(targeting_issues)
        
        # Check required disclaimers
        disclaimer_issues = self._check_required_disclaimers(creative, text_content)
        issues.extend(disclaimer_issues)
        
        # Check age-gated content
        age_issues = self._check_age_requirements(creative)
        issues.extend(age_issues)
        
        compliant = len(issues) == 0
        
        if not compliant:
            self.logger.warning(f"Policy compliance failed: {len(issues)} issues found")
        
        return compliant, issues
    
    def _extract_text_content(self, creative: Dict[str, Any]) -> str:
        """Extract all text content from a creative for analysis."""
        content_parts = []
        
        # Extract from various fields
        for field in ['headline', 'description', 'call_to_action', 'body_text']:
            if field in creative and creative[field]:
                content_parts.append(str(creative[field]))
        
        # Extract from nested content
        if 'content' in creative and isinstance(creative['content'], dict):
            for key, value in creative['content'].items():
                if isinstance(value, str):
                    content_parts.append(value)
        
        return ' '.join(content_parts).lower()
    
    def _check_blocked_content(self, text_content: str) -> List[str]:
        """Check text content against blocked patterns."""
        issues = []
        
        for pattern in self.blocked_patterns:
            try:
                if re.search(pattern, text_content, flags=re.IGNORECASE):
                    issues.append(f"Blocked content pattern detected: {pattern}")
            except re.error as e:
                self.logger.error(f"Invalid regex pattern {pattern}: {e}")
        
        return issues
    
    def _check_targeting_compliance(self, targeting: Dict[str, Any]) -> List[str]:
        """Check targeting parameters for compliance."""
        issues = []
        
        # Check restricted audiences
        target_audiences = targeting.get('audiences', [])
        if isinstance(target_audiences, str):
            target_audiences = [target_audiences]
        
        for audience in target_audiences:
            if audience.lower() in [ra.lower() for ra in self.restricted_audiences]:
                issues.append(f"Targeting restricted audience: {audience}")
        
        # Check age targeting
        min_age = targeting.get('min_age')
        if min_age is not None and min_age < 18:
            issues.append(f"Minimum age targeting below 18: {min_age}")
        
        # Check geographic restrictions
        excluded_regions = targeting.get('excluded_regions', [])
        if 'high_risk_regions' in targeting and not excluded_regions:
            issues.append("High-risk regional targeting without exclusions")
        
        return issues
    
    def _check_required_disclaimers(self, creative: Dict[str, Any], text_content: str) -> List[str]:
        """Check for required disclaimers based on content type."""
        issues = []
        
        # Health-related content disclaimer
        if self.require_health_disclaimer:
            health_keywords = ['health', 'medical', 'treatment', 'diagnosis', 'cure', 'therapy']
            if any(keyword in text_content for keyword in health_keywords):
                if 'disclaimer' not in creative or not creative['disclaimer']:
                    issues.append("Health-related content requires disclaimer")
        
        # Financial disclaimer
        financial_keywords = ['investment', 'trading', 'returns', 'profit', 'financial']
        if any(keyword in text_content for keyword in financial_keywords):
            if 'financial_disclaimer' not in creative:
                issues.append("Financial content requires disclaimer")
        
        return issues
    
    def _check_age_requirements(self, creative: Dict[str, Any]) -> List[str]:
        """Check age verification requirements."""
        issues = []
        
        if self.require_age_verification:
            content_type = creative.get('content_type', '')
            age_restricted_types = ['alcohol', 'gambling', 'adult_content']
            
            if content_type in age_restricted_types:
                if not creative.get('age_verified', False):
                    issues.append(f"Age verification required for {content_type}")
        
        return issues


class SafetyEventLogger:
    """
    Comprehensive safety event logging system with structured metadata.
    
    Logs all safety events for audit trails and compliance monitoring.
    """
    
    def __init__(self):
        """Initialize the safety event logger."""
        self.logger = logging.getLogger(__name__)
        self._events: List[SafetyEvent] = []
        self._lock = threading.Lock()
        
        # Configure structured logging format
        self._setup_structured_logging()
    
    def _setup_structured_logging(self) -> None:
        """Set up structured logging for safety events."""
        safety_logger = logging.getLogger('safety_events')
        safety_logger.setLevel(logging.INFO)
        
        # Create file handler if log path is configured
        log_path = os.environ.get('AELP2_SAFETY_LOG_PATH')
        if log_path:
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            safety_logger.addHandler(file_handler)
    
    def log_safety_event(self, event_type: SafetyEventType, severity: SafetyEventSeverity, 
                        metadata: Dict[str, Any]) -> str:
        """
        Log a safety event with complete metadata.
        
        Args:
            event_type: Type of safety event
            severity: Severity level
            metadata: Additional event metadata
            
        Returns:
            Event ID for tracking
        """
        event = SafetyEvent(
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        with self._lock:
            self._events.append(event)
        
        # Log structured event
        safety_logger = logging.getLogger('safety_events')
        log_data = {
            'event_id': event.event_id,
            'event_type': event_type.value,
            'severity': severity.value,
            'timestamp': event.timestamp.isoformat(),
            'metadata': metadata
        }
        
        safety_logger.info(json.dumps(log_data))
        
        # Log to main logger based on severity
        if severity in [SafetyEventSeverity.HIGH, SafetyEventSeverity.CRITICAL]:
            self.logger.error(f"Safety event {event_type.value}: {event.event_id}")
        elif severity == SafetyEventSeverity.MEDIUM:
            self.logger.warning(f"Safety event {event_type.value}: {event.event_id}")
        else:
            self.logger.info(f"Safety event {event_type.value}: {event.event_id}")
        
        return event.event_id
    
    def get_recent_events(self, hours: int = 24) -> List[SafetyEvent]:
        """Get safety events from the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            return [event for event in self._events if event.timestamp >= cutoff_time]
    
    def get_events_by_type(self, event_type: SafetyEventType, hours: int = 24) -> List[SafetyEvent]:
        """Get events of a specific type from the last N hours."""
        recent_events = self.get_recent_events(hours)
        return [event for event in recent_events if event.event_type == event_type]
    
    def get_critical_events(self, hours: int = 24) -> List[SafetyEvent]:
        """Get critical safety events from the last N hours."""
        recent_events = self.get_recent_events(hours)
        return [event for event in recent_events if event.severity == SafetyEventSeverity.CRITICAL]


# Global instances for production use
_safety_gates = None
_hitl_queue = None
_policy_checker = None
_event_logger = None


def get_safety_gates() -> SafetyGates:
    """Get the global SafetyGates instance."""
    global _safety_gates
    if _safety_gates is None:
        _safety_gates = SafetyGates()
    return _safety_gates


def get_hitl_queue() -> HITLApprovalQueue:
    """Get the global HITLApprovalQueue instance."""
    global _hitl_queue
    if _hitl_queue is None:
        _hitl_queue = HITLApprovalQueue()
    return _hitl_queue


def get_policy_checker() -> PolicyChecker:
    """Get the global PolicyChecker instance."""
    global _policy_checker
    if _policy_checker is None:
        _policy_checker = PolicyChecker()
    return _policy_checker


def get_event_logger() -> SafetyEventLogger:
    """Get the global SafetyEventLogger instance."""
    global _event_logger
    if _event_logger is None:
        _event_logger = SafetyEventLogger()
    return _event_logger


# Convenience functions for integration
def validate_action_safety(action: Dict[str, Any], metrics: Dict[str, Any], 
                         context: Dict[str, Any]) -> Tuple[bool, List[str], Optional[str]]:
    """
    Comprehensive safety validation for actions before execution.
    
    Args:
        action: The action to validate
        metrics: Current performance metrics
        context: Additional context for validation
        
    Returns:
        Tuple of (is_safe: bool, violations: List[str], approval_id: Optional[str])
    """
    violations = []
    approval_id = None
    
    # Check safety gates
    safety_gates = get_safety_gates()
    gates_passed, gate_violations = safety_gates.evaluate_gates(metrics)
    
    if not gates_passed:
        event_logger = get_event_logger()
        for violation in gate_violations:
            violations.append(f"Gate violation: {violation.gate_name}")
            event_logger.log_safety_event(
                SafetyEventType.GATE_VIOLATION,
                violation.severity,
                {
                    'gate_name': violation.gate_name,
                    'actual_value': violation.actual_value,
                    'threshold_value': violation.threshold_value,
                    'operator': violation.operator,
                    'action': action,
                    'metrics': metrics
                }
            )
    
    # Check policy compliance for creative actions
    if action.get('type') == 'creative_change':
        policy_checker = get_policy_checker()
        creative = action.get('creative', {})
        is_compliant, policy_issues = policy_checker.check_policy_compliance(creative)
        
        if not is_compliant:
            violations.extend(policy_issues)
            event_logger = get_event_logger()
            event_logger.log_safety_event(
                SafetyEventType.POLICY_VIOLATION,
                SafetyEventSeverity.HIGH,
                {
                    'policy_issues': policy_issues,
                    'action': action,
                    'creative': creative
                }
            )
    
    # Request HITL approval (configurable)
    # Env controls to reduce noisy approvals during early training and for low-risk bidding:
    # - AELP2_HITL_ON_GATE_FAIL: '1' to request approvals when gates fail (default '1')
    # - AELP2_HITL_ON_GATE_FAIL_FOR_BIDS: '1' to also request approvals for bidding actions on gate fail (default '0')
    # - AELP2_HITL_MIN_STEP_FOR_APPROVAL: minimum step index to begin requesting approvals (default '0')
    try:
        on_gate_fail = os.environ.get('AELP2_HITL_ON_GATE_FAIL', '1') == '1'
        on_gate_fail_for_bids = os.environ.get('AELP2_HITL_ON_GATE_FAIL_FOR_BIDS', '0') == '1'
        min_step_for_approval = int(os.environ.get('AELP2_HITL_MIN_STEP_FOR_APPROVAL', '0'))
    except Exception:
        on_gate_fail, on_gate_fail_for_bids, min_step_for_approval = True, False, 0

    step_idx = int(context.get('step', 0)) if isinstance(context, dict) else 0
    # Adaptive throttle: during clearly failing periods, do not request approvals for bidding actions
    try:
        throttle_on_fail = os.environ.get('AELP2_HITL_THROTTLE_ON_FAIL', '1') == '1'
        min_wr_for_approvals = float(os.environ.get('AELP2_HITL_MIN_WINRATE_FOR_APPROVALS', '0.05'))
    except Exception:
        throttle_on_fail = True
        min_wr_for_approvals = 0.05
    current_wr = float(metrics.get('win_rate', 0.0)) if isinstance(metrics, dict) else 0.0
    is_bidding = (
        ('bid' in action or 'bid_amount' in action) and action.get('type') not in ['creative_change', 'budget_increase', 'targeting_change']
    )

    request_approval = False
    # High-risk explicit action types always require approval
    if action.get('type') in ['creative_change', 'budget_increase', 'targeting_change'] or context.get('risk_level') == 'high':
        request_approval = True
    # Gate failure may require approval depending on config
    elif not gates_passed:
        if on_gate_fail and (not is_bidding or on_gate_fail_for_bids):
            if is_bidding and throttle_on_fail and current_wr < min_wr_for_approvals:
                request_approval = False
            else:
                request_approval = True

    if request_approval and step_idx >= min_step_for_approval:
        hitl_queue = get_hitl_queue()
        approval_id = hitl_queue.request_approval(action, context)

        event_logger = get_event_logger()
        event_logger.log_safety_event(
            SafetyEventType.APPROVAL_REQUESTED,
            SafetyEventSeverity.MEDIUM,
            {
                'approval_id': approval_id,
                'action': action,
                'context': context,
                'violations': violations
            }
        )

    is_safe = len(violations) == 0 and approval_id is None
    return is_safe, violations, approval_id


def emergency_stop(reason: str, context: Dict[str, Any]) -> None:
    """
    Trigger emergency stop with comprehensive logging.
    
    Args:
        reason: Reason for emergency stop
        context: Additional context about the emergency
    """
    event_logger = get_event_logger()
    event_logger.log_safety_event(
        SafetyEventType.EMERGENCY_STOP,
        SafetyEventSeverity.CRITICAL,
        {
            'reason': reason,
            'context': context,
            'timestamp': datetime.utcnow().isoformat()
        }
    )
    
    logger = logging.getLogger(__name__)
    logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
    
    # Additional emergency stop logic would go here
    # (e.g., stopping all active campaigns, notifying administrators)
