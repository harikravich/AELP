"""
Safety Monitoring and Compliance

Ensures safety constraints are enforced throughout training,
especially during real-world testing phases.
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum

import numpy as np

from .phases import TrainingPhase
from .episode_manager import EpisodeResult


class SafetyViolationType(Enum):
    """Types of safety violations"""
    BUDGET_EXCEEDED = "budget_exceeded"
    CONTENT_INAPPROPRIATE = "content_inappropriate"
    TARGETING_INAPPROPRIATE = "targeting_inappropriate"
    BID_TOO_HIGH = "bid_too_high"
    BRAND_SAFETY_LOW = "brand_safety_low"
    POLICY_VIOLATION = "policy_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class SafetyViolation:
    """Records a safety violation incident"""
    violation_type: SafetyViolationType
    severity: str  # "low", "medium", "high", "critical"
    episode_id: str
    phase: TrainingPhase
    timestamp: datetime
    details: Dict[str, Any]
    action_taken: str
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class SafetyConstraints:
    """Safety constraints for each phase"""
    max_daily_budget: float
    max_episode_budget: float
    min_content_safety_score: float
    min_brand_safety_score: float
    max_bid_amount: float
    allowed_audience_categories: Set[str]
    forbidden_keywords: Set[str]
    require_human_approval: bool
    max_violations_per_day: int
    anomaly_detection_threshold: float


@dataclass
class SafetyMetrics:
    """Safety-related metrics tracking"""
    total_violations: int = 0
    violations_by_type: Dict[SafetyViolationType, int] = field(default_factory=dict)
    violations_by_severity: Dict[str, int] = field(default_factory=dict)
    daily_budget_spent: float = 0.0
    total_budget_spent: float = 0.0
    content_safety_scores: List[float] = field(default_factory=list)
    brand_safety_scores: List[float] = field(default_factory=list)
    last_violation_time: Optional[datetime] = None
    consecutive_safe_episodes: int = 0


class SafetyMonitor:
    """
    Monitors and enforces safety constraints during training,
    with special focus on budget controls and content safety.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Safety tracking
        self.violations: List[SafetyViolation] = []
        self.metrics = SafetyMetrics()
        
        # Phase-specific constraints
        self.safety_constraints = self._initialize_safety_constraints()
        
        # Monitoring state
        self.last_daily_reset = datetime.now().date()
        self.monitoring_enabled = True
        
        # Alert callbacks
        self.alert_callbacks: List[callable] = []
        
        self.logger.info("Safety Monitor initialized")
    
    def _initialize_safety_constraints(self) -> Dict[TrainingPhase, SafetyConstraints]:
        """Initialize safety constraints for each phase"""
        
        return {
            TrainingPhase.SIMULATION: SafetyConstraints(
                max_daily_budget=float('inf'),  # No real budget limits in simulation
                max_episode_budget=1000.0,  # Simulated budget
                min_content_safety_score=0.8,
                min_brand_safety_score=0.7,
                max_bid_amount=100.0,
                allowed_audience_categories=set(),  # All allowed in simulation
                forbidden_keywords={"violence", "hate", "adult", "illegal"},
                require_human_approval=False,
                max_violations_per_day=20,  # More lenient in simulation
                anomaly_detection_threshold=3.0
            ),
            
            TrainingPhase.HISTORICAL_VALIDATION: SafetyConstraints(
                max_daily_budget=float('inf'),  # No real budget
                max_episode_budget=500.0,
                min_content_safety_score=0.9,
                min_brand_safety_score=0.8,
                max_bid_amount=50.0,
                allowed_audience_categories={"general", "business", "education"},
                forbidden_keywords={"violence", "hate", "adult", "illegal", "gambling"},
                require_human_approval=False,
                max_violations_per_day=10,
                anomaly_detection_threshold=2.5
            ),
            
            TrainingPhase.REAL_TESTING: SafetyConstraints(
                max_daily_budget=self.config.budget.real_testing_daily_limit,
                max_episode_budget=10.0,  # Small per-episode limit
                min_content_safety_score=0.95,
                min_brand_safety_score=0.9,
                max_bid_amount=5.0,
                allowed_audience_categories={"general", "business"},
                forbidden_keywords={"violence", "hate", "adult", "illegal", "gambling", "politics"},
                require_human_approval=True,
                max_violations_per_day=2,  # Very strict
                anomaly_detection_threshold=2.0
            ),
            
            TrainingPhase.SCALED_DEPLOYMENT: SafetyConstraints(
                max_daily_budget=self.config.budget.scaled_deployment_daily_limit,
                max_episode_budget=100.0,
                min_content_safety_score=0.9,
                min_brand_safety_score=0.85,
                max_bid_amount=20.0,
                allowed_audience_categories={"general", "business", "education", "technology"},
                forbidden_keywords={"violence", "hate", "adult", "illegal"},
                require_human_approval=False,  # Automated with monitoring
                max_violations_per_day=5,
                anomaly_detection_threshold=2.5
            )
        }
    
    async def check_safety_constraints(self, 
                                     episode_result: EpisodeResult,
                                     phase: TrainingPhase) -> bool:
        """
        Check if episode result violates safety constraints
        
        Args:
            episode_result: Results from the episode
            phase: Current training phase
            
        Returns:
            bool: True if safe, False if violations detected
        """
        
        if not self.monitoring_enabled:
            return True
        
        constraints = self.safety_constraints[phase]
        violations = []
        
        # Check daily budget reset
        await self._check_daily_reset()
        
        # Budget constraints
        budget_violations = self._check_budget_constraints(
            episode_result, constraints, phase
        )
        violations.extend(budget_violations)
        
        # Content safety
        content_violations = self._check_content_safety(
            episode_result, constraints, phase
        )
        violations.extend(content_violations)
        
        # Brand safety
        brand_violations = self._check_brand_safety(
            episode_result, constraints, phase
        )
        violations.extend(brand_violations)
        
        # Targeting appropriateness
        targeting_violations = self._check_targeting_safety(
            episode_result, constraints, phase
        )
        violations.extend(targeting_violations)
        
        # Bid amount checks
        bid_violations = self._check_bid_safety(
            episode_result, constraints, phase
        )
        violations.extend(bid_violations)
        
        # Anomaly detection
        anomaly_violations = await self._check_anomalies(
            episode_result, constraints, phase
        )
        violations.extend(anomaly_violations)
        
        # Process violations
        if violations:
            await self._process_violations(violations, episode_result, phase)
            return False
        else:
            self.metrics.consecutive_safe_episodes += 1
            return True
    
    async def check_real_campaign_safety(self, episode_result: EpisodeResult) -> bool:
        """
        Enhanced safety checks for real campaigns with stricter enforcement
        
        Args:
            episode_result: Results from real campaign episode
            
        Returns:
            bool: True if safe for real deployment
        """
        
        # Standard safety checks
        basic_safety = await self.check_safety_constraints(
            episode_result, TrainingPhase.REAL_TESTING
        )
        
        if not basic_safety:
            return False
        
        # Additional real campaign checks
        constraints = self.safety_constraints[TrainingPhase.REAL_TESTING]
        
        # Check for human approval requirement
        if constraints.require_human_approval:
            approval_status = episode_result.info.get("human_approval", False)
            if not approval_status:
                await self._create_violation(
                    SafetyViolationType.POLICY_VIOLATION,
                    "critical",
                    episode_result.episode_id,
                    TrainingPhase.REAL_TESTING,
                    {"reason": "Human approval required but not obtained"},
                    "Episode blocked"
                )
                return False
        
        # Real-time performance monitoring
        roi = episode_result.metrics.roi
        if roi < -0.5:  # More than 50% loss
            await self._create_violation(
                SafetyViolationType.PERFORMANCE_DEGRADATION,
                "high",
                episode_result.episode_id,
                TrainingPhase.REAL_TESTING,
                {"roi": roi, "threshold": -0.5},
                "Performance monitoring triggered"
            )
            return False
        
        # Check daily violation limit
        today_violations = len([
            v for v in self.violations
            if v.timestamp.date() == datetime.now().date()
        ])
        
        if today_violations >= constraints.max_violations_per_day:
            self.logger.error(f"Daily violation limit reached: {today_violations}")
            return False
        
        return True
    
    def _check_budget_constraints(self, 
                                 episode_result: EpisodeResult,
                                 constraints: SafetyConstraints,
                                 phase: TrainingPhase) -> List[SafetyViolation]:
        """Check budget-related safety constraints"""
        
        violations = []
        budget_spent = episode_result.metrics.budget_spent
        
        # Episode budget check
        if budget_spent > constraints.max_episode_budget:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.BUDGET_EXCEEDED,
                severity="high" if budget_spent > constraints.max_episode_budget * 2 else "medium",
                episode_id=episode_result.episode_id,
                phase=phase,
                timestamp=datetime.now(),
                details={
                    "budget_spent": budget_spent,
                    "limit": constraints.max_episode_budget,
                    "overage": budget_spent - constraints.max_episode_budget
                },
                action_taken="Episode budget exceeded"
            ))
        
        # Daily budget check
        self.metrics.daily_budget_spent += budget_spent
        self.metrics.total_budget_spent += budget_spent
        
        if self.metrics.daily_budget_spent > constraints.max_daily_budget:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.BUDGET_EXCEEDED,
                severity="critical",
                episode_id=episode_result.episode_id,
                phase=phase,
                timestamp=datetime.now(),
                details={
                    "daily_spent": self.metrics.daily_budget_spent,
                    "daily_limit": constraints.max_daily_budget,
                    "overage": self.metrics.daily_budget_spent - constraints.max_daily_budget
                },
                action_taken="Daily budget limit exceeded"
            ))
        
        return violations
    
    def _check_content_safety(self, 
                             episode_result: EpisodeResult,
                             constraints: SafetyConstraints,
                             phase: TrainingPhase) -> List[SafetyViolation]:
        """Check content safety constraints"""
        
        violations = []
        content_score = episode_result.metrics.content_quality_score
        
        # Content safety score check
        if content_score < constraints.min_content_safety_score:
            severity = "critical" if content_score < 0.5 else "high"
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.CONTENT_INAPPROPRIATE,
                severity=severity,
                episode_id=episode_result.episode_id,
                phase=phase,
                timestamp=datetime.now(),
                details={
                    "content_score": content_score,
                    "threshold": constraints.min_content_safety_score,
                    "content_issues": episode_result.info.get("content_flags", [])
                },
                action_taken="Content flagged for review"
            ))
        
        # Forbidden keywords check
        content_text = episode_result.info.get("ad_content", "")
        found_keywords = [
            keyword for keyword in constraints.forbidden_keywords
            if keyword.lower() in content_text.lower()
        ]
        
        if found_keywords:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.CONTENT_INAPPROPRIATE,
                severity="high",
                episode_id=episode_result.episode_id,
                phase=phase,
                timestamp=datetime.now(),
                details={
                    "forbidden_keywords": found_keywords,
                    "content_excerpt": content_text[:100]
                },
                action_taken="Forbidden keywords detected"
            ))
        
        # Track content safety scores
        self.metrics.content_safety_scores.append(content_score)
        
        return violations
    
    def _check_brand_safety(self, 
                           episode_result: EpisodeResult,
                           constraints: SafetyConstraints,
                           phase: TrainingPhase) -> List[SafetyViolation]:
        """Check brand safety constraints"""
        
        violations = []
        brand_score = episode_result.metrics.brand_safety_score
        
        if brand_score < constraints.min_brand_safety_score:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.BRAND_SAFETY_LOW,
                severity="medium" if brand_score > 0.5 else "high",
                episode_id=episode_result.episode_id,
                phase=phase,
                timestamp=datetime.now(),
                details={
                    "brand_score": brand_score,
                    "threshold": constraints.min_brand_safety_score,
                    "brand_concerns": episode_result.info.get("brand_flags", [])
                },
                action_taken="Brand safety review required"
            ))
        
        # Track brand safety scores
        self.metrics.brand_safety_scores.append(brand_score)
        
        return violations
    
    def _check_targeting_safety(self, 
                               episode_result: EpisodeResult,
                               constraints: SafetyConstraints,
                               phase: TrainingPhase) -> List[SafetyViolation]:
        """Check targeting appropriateness"""
        
        violations = []
        targeting_info = episode_result.info.get("targeting", {})
        
        # Check allowed audience categories
        if constraints.allowed_audience_categories:
            target_categories = set(targeting_info.get("categories", []))
            forbidden_categories = target_categories - constraints.allowed_audience_categories
            
            if forbidden_categories:
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.TARGETING_INAPPROPRIATE,
                    severity="medium",
                    episode_id=episode_result.episode_id,
                    phase=phase,
                    timestamp=datetime.now(),
                    details={
                        "forbidden_categories": list(forbidden_categories),
                        "allowed_categories": list(constraints.allowed_audience_categories)
                    },
                    action_taken="Inappropriate targeting detected"
                ))
        
        # Check for sensitive targeting (age, location restrictions)
        if targeting_info.get("min_age", 18) < 18:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.TARGETING_INAPPROPRIATE,
                severity="high",
                episode_id=episode_result.episode_id,
                phase=phase,
                timestamp=datetime.now(),
                details={"reason": "Targeting minors not allowed"},
                action_taken="Underage targeting blocked"
            ))
        
        return violations
    
    def _check_bid_safety(self, 
                         episode_result: EpisodeResult,
                         constraints: SafetyConstraints,
                         phase: TrainingPhase) -> List[SafetyViolation]:
        """Check bid amount safety"""
        
        violations = []
        bid_amount = episode_result.info.get("bid_amount", 0.0)
        
        if bid_amount > constraints.max_bid_amount:
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.BID_TOO_HIGH,
                severity="medium" if bid_amount < constraints.max_bid_amount * 2 else "high",
                episode_id=episode_result.episode_id,
                phase=phase,
                timestamp=datetime.now(),
                details={
                    "bid_amount": bid_amount,
                    "max_allowed": constraints.max_bid_amount,
                    "overage": bid_amount - constraints.max_bid_amount
                },
                action_taken="Bid amount capped"
            ))
        
        return violations
    
    async def _check_anomalies(self, 
                              episode_result: EpisodeResult,
                              constraints: SafetyConstraints,
                              phase: TrainingPhase) -> List[SafetyViolation]:
        """Check for anomalous behavior patterns"""
        
        violations = []
        
        # Performance anomaly detection
        if len(self.metrics.content_safety_scores) >= 10:
            recent_scores = self.metrics.content_safety_scores[-10:]
            current_score = episode_result.metrics.content_quality_score
            
            mean_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)
            
            if std_score > 0:
                z_score = abs(current_score - mean_score) / std_score
                if z_score > constraints.anomaly_detection_threshold:
                    violations.append(SafetyViolation(
                        violation_type=SafetyViolationType.ANOMALOUS_BEHAVIOR,
                        severity="medium",
                        episode_id=episode_result.episode_id,
                        phase=phase,
                        timestamp=datetime.now(),
                        details={
                            "z_score": z_score,
                            "threshold": constraints.anomaly_detection_threshold,
                            "metric": "content_safety_score",
                            "current_value": current_score,
                            "historical_mean": mean_score
                        },
                        action_taken="Anomaly detected in performance"
                    ))
        
        return violations
    
    async def _process_violations(self, 
                                 violations: List[SafetyViolation],
                                 episode_result: EpisodeResult,
                                 phase: TrainingPhase):
        """Process and record safety violations"""
        
        for violation in violations:
            # Record violation
            self.violations.append(violation)
            self.metrics.total_violations += 1
            self.metrics.last_violation_time = violation.timestamp
            self.metrics.consecutive_safe_episodes = 0
            
            # Update violation counts
            if violation.violation_type not in self.metrics.violations_by_type:
                self.metrics.violations_by_type[violation.violation_type] = 0
            self.metrics.violations_by_type[violation.violation_type] += 1
            
            if violation.severity not in self.metrics.violations_by_severity:
                self.metrics.violations_by_severity[violation.severity] = 0
            self.metrics.violations_by_severity[violation.severity] += 1
            
            # Log violation
            self.logger.warning(
                f"Safety violation {violation.violation_type.value} "
                f"(severity: {violation.severity}) in episode {violation.episode_id}: "
                f"{violation.action_taken}"
            )
            
            # Trigger alerts for critical violations
            if violation.severity == "critical":
                await self._trigger_critical_alert(violation)
    
    async def _create_violation(self, 
                               violation_type: SafetyViolationType,
                               severity: str,
                               episode_id: str,
                               phase: TrainingPhase,
                               details: Dict[str, Any],
                               action_taken: str):
        """Create and process a new violation"""
        
        violation = SafetyViolation(
            violation_type=violation_type,
            severity=severity,
            episode_id=episode_id,
            phase=phase,
            timestamp=datetime.now(),
            details=details,
            action_taken=action_taken
        )
        
        await self._process_violations([violation], None, phase)
    
    async def _trigger_critical_alert(self, violation: SafetyViolation):
        """Trigger alerts for critical safety violations"""
        
        alert_message = {
            "type": "critical_safety_violation",
            "violation": {
                "type": violation.violation_type.value,
                "severity": violation.severity,
                "episode_id": violation.episode_id,
                "phase": violation.phase.value,
                "timestamp": violation.timestamp.isoformat(),
                "details": violation.details
            },
            "action_required": True
        }
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_message)
                else:
                    callback(alert_message)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        self.logger.critical(f"CRITICAL SAFETY VIOLATION: {violation.violation_type.value}")
    
    async def _check_daily_reset(self):
        """Check if daily metrics need to be reset"""
        
        current_date = datetime.now().date()
        if current_date > self.last_daily_reset:
            self.metrics.daily_budget_spent = 0.0
            self.last_daily_reset = current_date
            self.logger.info("Daily safety metrics reset")
    
    def register_alert_callback(self, callback: callable):
        """Register a callback for safety alerts"""
        self.alert_callbacks.append(callback)
    
    def get_safety_summary(self) -> Dict[str, Any]:
        """Get comprehensive safety summary"""
        
        return {
            "total_violations": self.metrics.total_violations,
            "violations_by_type": {
                vtype.value: count 
                for vtype, count in self.metrics.violations_by_type.items()
            },
            "violations_by_severity": self.metrics.violations_by_severity.copy(),
            "daily_budget_spent": self.metrics.daily_budget_spent,
            "total_budget_spent": self.metrics.total_budget_spent,
            "consecutive_safe_episodes": self.metrics.consecutive_safe_episodes,
            "last_violation": self.metrics.last_violation_time.isoformat() if self.metrics.last_violation_time else None,
            "average_content_safety": np.mean(self.metrics.content_safety_scores) if self.metrics.content_safety_scores else 0.0,
            "average_brand_safety": np.mean(self.metrics.brand_safety_scores) if self.metrics.brand_safety_scores else 0.0,
            "monitoring_enabled": self.monitoring_enabled
        }
    
    def get_recent_violations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent safety violations"""
        
        recent_violations = sorted(
            self.violations, 
            key=lambda v: v.timestamp, 
            reverse=True
        )[:limit]
        
        return [
            {
                "type": v.violation_type.value,
                "severity": v.severity,
                "episode_id": v.episode_id,
                "phase": v.phase.value,
                "timestamp": v.timestamp.isoformat(),
                "details": v.details,
                "action_taken": v.action_taken,
                "resolved": v.resolved
            }
            for v in recent_violations
        ]
    
    def enable_monitoring(self):
        """Enable safety monitoring"""
        self.monitoring_enabled = True
        self.logger.info("Safety monitoring enabled")
    
    def disable_monitoring(self):
        """Disable safety monitoring (use with caution)"""
        self.monitoring_enabled = False
        self.logger.warning("Safety monitoring DISABLED")
    
    def resolve_violation(self, violation_id: int, resolution_notes: str = ""):
        """Mark a violation as resolved"""
        
        if 0 <= violation_id < len(self.violations):
            violation = self.violations[violation_id]
            violation.resolved = True
            violation.resolution_time = datetime.now()
            
            self.logger.info(
                f"Violation {violation_id} resolved: {violation.violation_type.value}. "
                f"Notes: {resolution_notes}"
            )
    
    def update_safety_constraints(self, 
                                 phase: TrainingPhase, 
                                 constraint_updates: Dict[str, Any]):
        """Update safety constraints for a specific phase"""
        
        constraints = self.safety_constraints[phase]
        
        for key, value in constraint_updates.items():
            if hasattr(constraints, key):
                setattr(constraints, key, value)
                self.logger.info(f"Updated {key} for {phase.value}: {value}")
            else:
                self.logger.warning(f"Unknown constraint {key} for {phase.value}")
    
    def get_budget_status(self) -> Dict[str, float]:
        """Get current budget status"""
        
        return {
            "daily_spent": self.metrics.daily_budget_spent,
            "total_spent": self.metrics.total_budget_spent,
            "daily_limit": self.safety_constraints[TrainingPhase.REAL_TESTING].max_daily_budget,
            "daily_remaining": max(0.0, 
                self.safety_constraints[TrainingPhase.REAL_TESTING].max_daily_budget - 
                self.metrics.daily_budget_spent
            )
        }