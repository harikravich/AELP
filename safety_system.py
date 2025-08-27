"""
GAELP Safety System - Comprehensive Bid Management Safety Framework

This module implements critical safety mechanisms to prevent catastrophic losses
from bugs, bad models, or anomalous behavior in automated bidding systems.

Core Safety Principles:
- Defense in depth with multiple safety layers
- Fail-safe mechanisms that stop operations when unsafe
- Real-time monitoring and anomaly detection
- Hard limits that cannot be overridden programmatically
- Comprehensive logging for audit trails
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import statistics
import json


class SafetyLevel(Enum):
    """Safety alert levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyViolationType(Enum):
    """Types of safety violations"""
    BID_CAP_EXCEEDED = "bid_cap_exceeded"
    BUDGET_CIRCUIT_BREAKER = "budget_circuit_breaker"
    ANOMALY_DETECTED = "anomaly_detected"
    ROI_THRESHOLD_VIOLATED = "roi_threshold_violated"
    COMPETITIVE_SPEND_EXCEEDED = "competitive_spend_exceeded"
    BLACKLISTED_QUERY = "blacklisted_query"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class SafetyConfig:
    """Safety system configuration"""
    # Absolute maximum bid cap - CANNOT be exceeded under any circumstances
    max_bid_absolute: float = 10.0
    
    # Budget circuit breaker thresholds
    daily_loss_threshold: float = 100.0  # Stop if daily loss exceeds this
    hourly_loss_threshold: float = 25.0  # Stop if hourly loss exceeds this
    consecutive_loss_limit: int = 5      # Stop after N consecutive losing bids
    
    # ROI thresholds
    minimum_roi_threshold: float = 0.1   # 10% minimum ROI
    roi_lookback_hours: int = 24         # Hours to look back for ROI calculation
    
    # Competitive spend limits
    max_competitor_spend_ratio: float = 1.5  # Don't spend more than 1.5x competitor
    
    # Anomaly detection parameters
    anomaly_z_score_threshold: float = 3.0   # Standard deviations for anomaly detection
    anomaly_lookback_hours: int = 168        # 7 days for baseline calculation
    
    # Emergency shutdown triggers
    emergency_loss_threshold: float = 500.0  # Emergency stop threshold
    emergency_bid_spike_multiplier: float = 10.0  # Emergency stop if bid spikes 10x
    
    # System limits
    max_concurrent_campaigns: int = 100
    max_daily_budget_total: float = 1000.0


@dataclass
class BidRecord:
    """Record of a bid attempt"""
    timestamp: datetime
    query: str
    bid_amount: float
    campaign_id: str
    predicted_roi: float
    actual_cost: Optional[float] = None
    actual_revenue: Optional[float] = None
    won: bool = False
    safety_flags: List[str] = None
    
    def __post_init__(self):
        if self.safety_flags is None:
            self.safety_flags = []


@dataclass
class SafetyViolation:
    """Record of a safety violation"""
    timestamp: datetime
    violation_type: SafetyViolationType
    severity: SafetyLevel
    details: Dict[str, Any]
    campaign_id: Optional[str] = None
    query: Optional[str] = None
    bid_amount: Optional[float] = None
    action_taken: str = ""


class SafetySystem:
    """
    Comprehensive safety system for automated bidding
    
    Implements multiple layers of safety checks including:
    - Hard bid caps
    - Budget circuit breakers
    - Anomaly detection
    - ROI monitoring
    - Competitive spend analysis
    - Query blacklisting
    - Emergency shutdown mechanisms
    """
    
    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        self.is_shutdown = False
        self.shutdown_reason = None
        self.bid_history: List[BidRecord] = []
        self.violations: List[SafetyViolation] = []
        self.blacklisted_queries: set = set()
        self.campaign_budgets: Dict[str, float] = {}
        self.campaign_spend: Dict[str, float] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize baseline metrics
        self._baseline_metrics = {}
        self._last_baseline_update = datetime.now()
        
        self.logger.info("Safety system initialized with configuration: %s", self.config)
    
    def check_bid_safety(self, query: str, bid_amount: float, campaign_id: str, 
                        predicted_roi: float = 0.0) -> Tuple[bool, List[str]]:
        """
        Comprehensive bid safety check
        
        Args:
            query: Search query
            bid_amount: Proposed bid amount
            campaign_id: Campaign identifier
            predicted_roi: Model's predicted ROI
        
        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        if self.is_shutdown:
            return False, [f"System is in emergency shutdown: {self.shutdown_reason}"]
        
        violations = []
        
        # 1. Absolute bid cap check (CANNOT be overridden)
        if bid_amount > self.config.max_bid_absolute:
            violation = self._create_violation(
                SafetyViolationType.BID_CAP_EXCEEDED,
                SafetyLevel.CRITICAL,
                {
                    "bid_amount": bid_amount,
                    "max_allowed": self.config.max_bid_absolute,
                    "query": query,
                    "campaign_id": campaign_id
                },
                campaign_id=campaign_id,
                query=query,
                bid_amount=bid_amount
            )
            self.violations.append(violation)
            violations.append(f"Bid ${bid_amount:.2f} exceeds absolute maximum ${self.config.max_bid_absolute:.2f}")
            self.logger.critical("BID CAP VIOLATION: %s", violation.details)
        
        # 2. Blacklisted query check
        if query.lower().strip() in self.blacklisted_queries:
            violation = self._create_violation(
                SafetyViolationType.BLACKLISTED_QUERY,
                SafetyLevel.HIGH,
                {"query": query, "campaign_id": campaign_id},
                campaign_id=campaign_id,
                query=query
            )
            self.violations.append(violation)
            violations.append(f"Query '{query}' is blacklisted")
        
        # 3. Budget circuit breaker check
        budget_violations = self._check_budget_circuit_breaker(campaign_id, bid_amount)
        violations.extend(budget_violations)
        
        # 4. ROI threshold check
        if predicted_roi < self.config.minimum_roi_threshold:
            violation = self._create_violation(
                SafetyViolationType.ROI_THRESHOLD_VIOLATED,
                SafetyLevel.MEDIUM,
                {
                    "predicted_roi": predicted_roi,
                    "minimum_threshold": self.config.minimum_roi_threshold,
                    "campaign_id": campaign_id
                },
                campaign_id=campaign_id
            )
            self.violations.append(violation)
            violations.append(f"Predicted ROI {predicted_roi:.2%} below minimum {self.config.minimum_roi_threshold:.2%}")
        
        # 5. Anomaly detection
        anomaly_violations = self.detect_anomaly(query, bid_amount, campaign_id)
        violations.extend(anomaly_violations)
        
        # 6. Emergency checks
        if bid_amount > self.config.emergency_loss_threshold:
            self.emergency_stop(f"Bid amount ${bid_amount:.2f} exceeds emergency threshold")
            return False, ["Emergency shutdown triggered"]
        
        is_safe = len(violations) == 0
        return is_safe, violations
    
    def validate_bid(self, bid_amount: float, context: Dict[str, Any] = None) -> float:
        """
        Validate and potentially adjust a bid amount for safety.
        This is a compatibility wrapper for check_bid_safety.
        
        Args:
            bid_amount: Proposed bid amount
            context: Optional context dictionary with budget_remaining, etc.
            
        Returns:
            Safe bid amount (may be adjusted down or 0 if unsafe)
        """
        # Extract context parameters
        if context is None:
            context = {}
        
        query = context.get('query', '')
        campaign_id = context.get('campaign_id', 'default')
        predicted_roi = context.get('predicted_roi', 0.0)
        
        # Check bid safety
        is_safe, violations = self.check_bid_safety(
            query=query,
            bid_amount=bid_amount,
            campaign_id=campaign_id,
            predicted_roi=predicted_roi
        )
        
        if is_safe:
            return bid_amount
        
        # If not safe, try to find a safe bid amount
        # Check if it's just too high
        if bid_amount > self.config.max_bid_absolute:
            return min(bid_amount, self.config.max_bid_absolute * 0.9)  # Return 90% of max
        
        # If there are other violations, return 0 (don't bid)
        return 0.0
    
    def detect_anomaly(self, query: str, bid_amount: float, campaign_id: str) -> List[str]:
        """
        Detect anomalous bidding patterns using statistical analysis
        
        Args:
            query: Search query
            bid_amount: Proposed bid amount
            campaign_id: Campaign identifier
        
        Returns:
            List of anomaly descriptions
        """
        anomalies = []
        
        try:
            # Get recent bid history for baseline
            recent_bids = self._get_recent_bids(hours=self.config.anomaly_lookback_hours)
            
            if len(recent_bids) < 10:  # Need minimum data for statistical analysis
                return anomalies
            
            # 1. Bid amount anomaly detection
            recent_bid_amounts = [bid.bid_amount for bid in recent_bids]
            mean_bid = statistics.mean(recent_bid_amounts)
            
            if len(recent_bid_amounts) > 1:
                stdev_bid = statistics.stdev(recent_bid_amounts)
                if stdev_bid > 0:
                    z_score = abs(bid_amount - mean_bid) / stdev_bid
                    if z_score > self.config.anomaly_z_score_threshold:
                        violation = self._create_violation(
                            SafetyViolationType.ANOMALY_DETECTED,
                            SafetyLevel.HIGH,
                            {
                                "type": "bid_amount_anomaly",
                                "bid_amount": bid_amount,
                                "mean_baseline": mean_bid,
                                "z_score": z_score,
                                "threshold": self.config.anomaly_z_score_threshold
                            },
                            campaign_id=campaign_id
                        )
                        self.violations.append(violation)
                        anomalies.append(f"Bid amount anomaly detected (z-score: {z_score:.2f})")
            
            # 2. Bid spike detection
            if recent_bids:
                recent_max = max(bid.bid_amount for bid in recent_bids[-10:])  # Last 10 bids
                if bid_amount > recent_max * self.config.emergency_bid_spike_multiplier:
                    self.emergency_stop(f"Bid spike detected: ${bid_amount:.2f} vs recent max ${recent_max:.2f}")
                    anomalies.append("Emergency shutdown: Extreme bid spike detected")
            
            # 3. Campaign-specific anomalies
            campaign_bids = [bid for bid in recent_bids if bid.campaign_id == campaign_id]
            if len(campaign_bids) >= 5:
                campaign_amounts = [bid.bid_amount for bid in campaign_bids]
                campaign_mean = statistics.mean(campaign_amounts)
                
                if bid_amount > campaign_mean * 5:  # 5x campaign average
                    anomalies.append(f"Bid 5x higher than campaign average (${campaign_mean:.2f})")
            
        except Exception as e:
            self.logger.error("Error in anomaly detection: %s", e)
            # Fail safe: flag as anomaly if we can't analyze
            anomalies.append("Anomaly detection system error - flagging for manual review")
        
        return anomalies
    
    def emergency_stop(self, reason: str):
        """
        Trigger emergency shutdown of the entire system
        
        Args:
            reason: Reason for emergency shutdown
        """
        self.is_shutdown = True
        self.shutdown_reason = reason
        
        violation = self._create_violation(
            SafetyViolationType.EMERGENCY_SHUTDOWN,
            SafetyLevel.CRITICAL,
            {"reason": reason, "timestamp": datetime.now().isoformat()},
            action_taken="System shutdown"
        )
        self.violations.append(violation)
        
        self.logger.critical("EMERGENCY SHUTDOWN TRIGGERED: %s", reason)
        
        # Send emergency notifications (in real system, this would integrate with alerting)
        self._send_emergency_alert(reason)
    
    def validate_campaign(self, campaign_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate campaign configuration for safety compliance
        
        Args:
            campaign_config: Campaign configuration dictionary
        
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        try:
            campaign_id = campaign_config.get('campaign_id', 'unknown')
            
            # 1. Budget validation
            daily_budget = campaign_config.get('daily_budget', 0)
            if daily_budget > self.config.max_daily_budget_total:
                violations.append(f"Daily budget ${daily_budget:.2f} exceeds maximum ${self.config.max_daily_budget_total:.2f}")
            
            # 2. Bid strategy validation
            max_bid = campaign_config.get('max_bid', 0)
            if max_bid > self.config.max_bid_absolute:
                violations.append(f"Max bid ${max_bid:.2f} exceeds absolute maximum ${self.config.max_bid_absolute:.2f}")
            
            # 3. Target validation
            target_queries = campaign_config.get('target_queries', [])
            blacklisted_in_targets = [q for q in target_queries if q.lower().strip() in self.blacklisted_queries]
            if blacklisted_in_targets:
                violations.append(f"Campaign targets blacklisted queries: {blacklisted_in_targets}")
            
            # 4. ROI target validation
            target_roi = campaign_config.get('target_roi', 0)
            if target_roi < self.config.minimum_roi_threshold:
                violations.append(f"Target ROI {target_roi:.2%} below minimum threshold {self.config.minimum_roi_threshold:.2%}")
            
            # 5. Campaign limit validation
            active_campaigns = len(set(bid.campaign_id for bid in self.bid_history if 
                                     bid.timestamp > datetime.now() - timedelta(days=1)))
            if active_campaigns >= self.config.max_concurrent_campaigns:
                violations.append(f"Maximum concurrent campaigns ({self.config.max_concurrent_campaigns}) reached")
            
            if violations:
                violation = self._create_violation(
                    SafetyViolationType.VALIDATION_FAILED,
                    SafetyLevel.HIGH,
                    {
                        "campaign_id": campaign_id,
                        "violations": violations,
                        "config": campaign_config
                    },
                    campaign_id=campaign_id
                )
                self.violations.append(violation)
        
        except Exception as e:
            self.logger.error("Error validating campaign: %s", e)
            violations.append(f"Campaign validation error: {e}")
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def record_bid_outcome(self, query: str, bid_amount: float, campaign_id: str,
                          won: bool, actual_cost: float = None, actual_revenue: float = None):
        """
        Record the outcome of a bid for tracking and analysis
        
        Args:
            query: Search query
            bid_amount: Bid amount
            campaign_id: Campaign identifier
            won: Whether the bid was won
            actual_cost: Actual cost incurred
            actual_revenue: Actual revenue generated
        """
        bid_record = BidRecord(
            timestamp=datetime.now(),
            query=query,
            bid_amount=bid_amount,
            campaign_id=campaign_id,
            predicted_roi=0.0,  # Would be filled from the safety check
            actual_cost=actual_cost,
            actual_revenue=actual_revenue,
            won=won
        )
        
        self.bid_history.append(bid_record)
        
        # Update campaign spend tracking
        if actual_cost:
            if campaign_id not in self.campaign_spend:
                self.campaign_spend[campaign_id] = 0.0
            self.campaign_spend[campaign_id] += actual_cost
        
        # Check for circuit breaker conditions after each bid
        self._check_post_bid_circuit_breakers(bid_record)
        
        # Trim old history to prevent memory issues
        if len(self.bid_history) > 10000:
            self.bid_history = self.bid_history[-5000:]  # Keep last 5000 records
    
    def add_to_blacklist(self, query: str, reason: str = ""):
        """
        Add a query to the blacklist
        
        Args:
            query: Query to blacklist
            reason: Reason for blacklisting
        """
        normalized_query = query.lower().strip()
        self.blacklisted_queries.add(normalized_query)
        self.logger.warning("Added query to blacklist: '%s' (Reason: %s)", query, reason)
    
    def remove_from_blacklist(self, query: str):
        """
        Remove a query from the blacklist
        
        Args:
            query: Query to remove from blacklist
        """
        normalized_query = query.lower().strip()
        self.blacklisted_queries.discard(normalized_query)
        self.logger.info("Removed query from blacklist: '%s'", query)
    
    def get_safety_status(self) -> Dict[str, Any]:
        """
        Get current safety system status and metrics
        
        Returns:
            Dictionary containing safety status information
        """
        recent_violations = [v for v in self.violations 
                           if v.timestamp > datetime.now() - timedelta(hours=24)]
        
        recent_bids = self._get_recent_bids(hours=24)
        total_spend_24h = sum(bid.actual_cost or 0 for bid in recent_bids)
        total_revenue_24h = sum(bid.actual_revenue or 0 for bid in recent_bids)
        
        status = {
            "is_operational": not self.is_shutdown,
            "shutdown_reason": self.shutdown_reason,
            "total_violations_24h": len(recent_violations),
            "critical_violations_24h": len([v for v in recent_violations 
                                          if v.severity == SafetyLevel.CRITICAL]),
            "total_spend_24h": total_spend_24h,
            "total_revenue_24h": total_revenue_24h,
            "roi_24h": (total_revenue_24h / total_spend_24h - 1) if total_spend_24h > 0 else 0,
            "blacklisted_queries_count": len(self.blacklisted_queries),
            "active_campaigns": len(set(bid.campaign_id for bid in recent_bids)),
            "config": self.config.__dict__,
            "last_updated": datetime.now().isoformat()
        }
        
        return status
    
    def reset_emergency_shutdown(self, authorization_code: str = None):
        """
        Reset emergency shutdown (requires authorization in production)
        
        Args:
            authorization_code: Authorization code for resetting shutdown
        """
        # In production, this would require proper authorization
        if authorization_code != "SAFETY_OVERRIDE_2024":
            self.logger.warning("Attempted shutdown reset without proper authorization")
            return False
        
        self.is_shutdown = False
        self.shutdown_reason = None
        self.logger.info("Emergency shutdown reset by authorized user")
        return True
    
    def export_safety_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Export comprehensive safety report
        
        Args:
            hours: Number of hours to include in report
        
        Returns:
            Dictionary containing safety report data
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_violations = [v for v in self.violations if v.timestamp > cutoff_time]
        recent_bids = [b for b in self.bid_history if b.timestamp > cutoff_time]
        
        report = {
            "report_period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "system_status": self.get_safety_status(),
            "violations": [
                {
                    "timestamp": v.timestamp.isoformat(),
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "details": v.details,
                    "campaign_id": v.campaign_id,
                    "query": v.query,
                    "action_taken": v.action_taken
                }
                for v in recent_violations
            ],
            "bid_summary": {
                "total_bids": len(recent_bids),
                "total_spend": sum(bid.actual_cost or 0 for bid in recent_bids),
                "total_revenue": sum(bid.actual_revenue or 0 for bid in recent_bids),
                "win_rate": sum(1 for bid in recent_bids if bid.won) / len(recent_bids) if recent_bids else 0,
                "average_bid": sum(bid.bid_amount for bid in recent_bids) / len(recent_bids) if recent_bids else 0
            },
            "safety_metrics": {
                "violation_rate": len(recent_violations) / len(recent_bids) if recent_bids else 0,
                "critical_violations": len([v for v in recent_violations 
                                          if v.severity == SafetyLevel.CRITICAL]),
                "emergency_shutdowns": len([v for v in recent_violations 
                                          if v.violation_type == SafetyViolationType.EMERGENCY_SHUTDOWN])
            }
        }
        
        return report
    
    # Private helper methods
    
    def _create_violation(self, violation_type: SafetyViolationType, severity: SafetyLevel,
                         details: Dict[str, Any], campaign_id: str = None, query: str = None,
                         bid_amount: float = None, action_taken: str = "") -> SafetyViolation:
        """Create a safety violation record"""
        return SafetyViolation(
            timestamp=datetime.now(),
            violation_type=violation_type,
            severity=severity,
            details=details,
            campaign_id=campaign_id,
            query=query,
            bid_amount=bid_amount,
            action_taken=action_taken
        )
    
    def _get_recent_bids(self, hours: int) -> List[BidRecord]:
        """Get bid records from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [bid for bid in self.bid_history if bid.timestamp > cutoff_time]
    
    def _check_budget_circuit_breaker(self, campaign_id: str, bid_amount: float) -> List[str]:
        """Check budget circuit breaker conditions"""
        violations = []
        
        # Daily loss check
        daily_bids = self._get_recent_bids(hours=24)
        campaign_daily_bids = [bid for bid in daily_bids if bid.campaign_id == campaign_id]
        
        daily_spend = sum(bid.actual_cost or bid.bid_amount for bid in campaign_daily_bids)
        daily_revenue = sum(bid.actual_revenue or 0 for bid in campaign_daily_bids)
        daily_loss = daily_spend - daily_revenue
        
        if daily_loss > self.config.daily_loss_threshold:
            violation = self._create_violation(
                SafetyViolationType.BUDGET_CIRCUIT_BREAKER,
                SafetyLevel.HIGH,
                {
                    "type": "daily_loss_threshold",
                    "daily_loss": daily_loss,
                    "threshold": self.config.daily_loss_threshold,
                    "campaign_id": campaign_id
                },
                campaign_id=campaign_id
            )
            self.violations.append(violation)
            violations.append(f"Daily loss ${daily_loss:.2f} exceeds threshold ${self.config.daily_loss_threshold:.2f}")
        
        # Hourly loss check
        hourly_bids = self._get_recent_bids(hours=1)
        campaign_hourly_bids = [bid for bid in hourly_bids if bid.campaign_id == campaign_id]
        
        hourly_spend = sum(bid.actual_cost or bid.bid_amount for bid in campaign_hourly_bids)
        hourly_revenue = sum(bid.actual_revenue or 0 for bid in campaign_hourly_bids)
        hourly_loss = hourly_spend - hourly_revenue
        
        if hourly_loss > self.config.hourly_loss_threshold:
            violations.append(f"Hourly loss ${hourly_loss:.2f} exceeds threshold ${self.config.hourly_loss_threshold:.2f}")
        
        # Consecutive loss check
        recent_campaign_bids = [bid for bid in self.bid_history[-20:] if bid.campaign_id == campaign_id]
        consecutive_losses = 0
        for bid in reversed(recent_campaign_bids):
            if bid.actual_cost and bid.actual_revenue:
                if bid.actual_cost > bid.actual_revenue:
                    consecutive_losses += 1
                else:
                    break
            elif bid.actual_cost and bid.actual_cost > 0:  # Assume loss if no revenue data
                consecutive_losses += 1
            else:
                break
        
        if consecutive_losses >= self.config.consecutive_loss_limit:
            violations.append(f"Consecutive losses ({consecutive_losses}) exceeds limit ({self.config.consecutive_loss_limit})")
        
        return violations
    
    def _check_post_bid_circuit_breakers(self, bid_record: BidRecord):
        """Check circuit breaker conditions after a bid is recorded"""
        if not bid_record.actual_cost or not bid_record.actual_revenue:
            return  # Can't check without cost/revenue data
        
        loss = bid_record.actual_cost - bid_record.actual_revenue
        
        # Check if this single bid exceeds emergency threshold
        if loss > self.config.emergency_loss_threshold:
            self.emergency_stop(f"Single bid loss ${loss:.2f} exceeds emergency threshold")
    
    def _send_emergency_alert(self, reason: str):
        """Send emergency alert (placeholder for real alerting system)"""
        # In production, this would integrate with PagerDuty, Slack, email, etc.
        alert_message = f"EMERGENCY: GAELP Safety System Shutdown - {reason}"
        self.logger.critical("EMERGENCY ALERT: %s", alert_message)
        # Additional alerting logic would go here


# Example usage and testing
if __name__ == "__main__":
    # Initialize safety system
    config = SafetyConfig(
        max_bid_absolute=10.0,
        daily_loss_threshold=100.0,
        minimum_roi_threshold=0.15  # 15% minimum ROI
    )
    
    safety = SafetySystem(config)
    
    # Add some blacklisted queries
    safety.add_to_blacklist("illegal content", "Prohibited content")
    safety.add_to_blacklist("spam query", "Low quality")
    
    # Test bid safety check
    is_safe, violations = safety.check_bid_safety(
        query="test query",
        bid_amount=5.0,
        campaign_id="campaign_001",
        predicted_roi=0.20
    )
    
    print(f"Bid safety check: {'SAFE' if is_safe else 'UNSAFE'}")
    if violations:
        print("Violations:", violations)
    
    # Test campaign validation
    campaign_config = {
        "campaign_id": "campaign_001",
        "daily_budget": 50.0,
        "max_bid": 8.0,
        "target_queries": ["valid query", "another query"],
        "target_roi": 0.20
    }
    
    is_valid, campaign_violations = safety.validate_campaign(campaign_config)
    print(f"Campaign validation: {'VALID' if is_valid else 'INVALID'}")
    if campaign_violations:
        print("Campaign violations:", campaign_violations)
    
    # Test emergency conditions
    print("\nTesting emergency conditions...")
    
    # Test bid cap violation
    is_safe, violations = safety.check_bid_safety("test", 15.0, "campaign_001", 0.5)
    print(f"Over-limit bid: {'SAFE' if is_safe else 'UNSAFE'} - {violations}")
    
    # Test blacklisted query
    is_safe, violations = safety.check_bid_safety("illegal content", 5.0, "campaign_001", 0.3)
    print(f"Blacklisted query: {'SAFE' if is_safe else 'UNSAFE'} - {violations}")
    
    # Get safety status
    status = safety.get_safety_status()
    print(f"\nSafety status: {'OPERATIONAL' if status['is_operational'] else 'SHUTDOWN'}")
    print(f"Total violations (24h): {status['total_violations_24h']}")
    
    # Export safety report
    report = safety.export_safety_report(hours=1)
    print(f"\nSafety report generated with {len(report['violations'])} violations")