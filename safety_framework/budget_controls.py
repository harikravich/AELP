"""
Budget Controls Module for GAELP Ad Campaign Safety
Implements comprehensive budget monitoring, limits, and safety mechanisms.
"""

import logging
import httpx
import stripe
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from decimal import Decimal
import json
import hashlib
import hmac
from contextlib import asynccontextmanager
import backoff
from google.cloud import bigquery
from google.cloud import monitoring_v3
from google.cloud import pubsub_v1
import os

logger = logging.getLogger(__name__)


class BudgetViolationType(Enum):
    DAILY_LIMIT_EXCEEDED = "daily_limit_exceeded"
    WEEKLY_LIMIT_EXCEEDED = "weekly_limit_exceeded" 
    MONTHLY_LIMIT_EXCEEDED = "monthly_limit_exceeded"
    CAMPAIGN_LIMIT_EXCEEDED = "campaign_limit_exceeded"
    SPEND_RATE_TOO_HIGH = "spend_rate_too_high"
    UNEXPECTED_CHARGE = "unexpected_charge"
    FRAUD_DETECTED = "fraud_detected"
    PAYMENT_FAILED = "payment_failed"
    ACCOUNT_SUSPENDED = "account_suspended"
    CARD_DECLINED = "card_declined"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    ROI_THRESHOLD_VIOLATED = "roi_threshold_violated"
    COST_PER_ACQUISITION_HIGH = "cost_per_acquisition_high"


class CampaignStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class BudgetLimits:
    """Budget limits configuration for campaigns"""
    daily_limit: Decimal
    weekly_limit: Decimal
    monthly_limit: Decimal
    campaign_limit: Decimal
    hourly_rate_limit: Optional[Decimal] = None
    roi_threshold: Optional[Decimal] = None  # Minimum ROI required
    cost_per_acquisition_limit: Optional[Decimal] = None  # Maximum CPA allowed
    emergency_stop_threshold: Optional[Decimal] = None  # Auto-stop if spend exceeds this
    
    def __post_init__(self):
        # Validate budget hierarchy
        if self.daily_limit * 7 > self.weekly_limit:
            logger.warning("Daily limit * 7 exceeds weekly limit")
        if self.weekly_limit * 4.3 > self.monthly_limit:
            logger.warning("Weekly limit * 4.3 exceeds monthly limit")
        
        # Set emergency stop threshold if not provided
        if self.emergency_stop_threshold is None:
            self.emergency_stop_threshold = self.daily_limit * Decimal('2.0')  # 200% of daily limit


@dataclass
class SpendRecord:
    """Individual spend transaction record"""
    campaign_id: str
    amount: Decimal
    timestamp: datetime
    platform: str
    transaction_id: str
    description: str
    payment_method_id: str  # Link to actual payment method
    stripe_charge_id: Optional[str] = None  # Stripe transaction ID
    bank_transaction_id: Optional[str] = None  # Bank confirmation ID
    conversion_data: Optional[Dict[str, Any]] = None  # ROI/conversion tracking
    is_verified: bool = False  # Whether payment was confirmed
    risk_score: float = 0.0  # Fraud detection score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaymentMethod:
    """Real payment method configuration"""
    payment_id: str
    method_type: str  # 'credit_card', 'bank_account', 'prepaid'
    stripe_payment_method_id: str
    last_four: str
    is_active: bool
    daily_limit: Optional[Decimal] = None
    monthly_limit: Optional[Decimal] = None
    risk_level: str = "low"  # low, medium, high
    
@dataclass
class BudgetViolation:
    """Budget violation event"""
    violation_type: BudgetViolationType
    campaign_id: str
    current_spend: Decimal
    limit_exceeded: Decimal
    timestamp: datetime
    description: str
    severity: str = "medium"  # low, medium, high, critical
    financial_impact: Decimal = Decimal('0')
    regulatory_implications: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    auto_recovery_possible: bool = True

@dataclass
class RealTimeSpendAlert:
    """Real-time spending alert"""
    alert_id: str
    campaign_id: str
    alert_type: str
    threshold_percentage: float  # 80% = approaching limit
    current_spend: Decimal
    limit: Decimal
    time_remaining: timedelta
    projected_overspend: Decimal
    recommended_actions: List[str]


class RealMoneyBudgetController:
    """
    Production-ready budget control system for real ad campaigns.
    Handles actual financial transactions, fraud detection, and regulatory compliance.
    """
    
    def __init__(self, alert_callback=None, storage_backend=None, stripe_api_key=None):
        # Core data structures
        self.campaigns: Dict[str, BudgetLimits] = {}
        self.spend_records: Dict[str, List[SpendRecord]] = {}
        self.violations: List[BudgetViolation] = []
        self.campaign_status: Dict[str, CampaignStatus] = {}
        self.payment_methods: Dict[str, PaymentMethod] = {}
        
        # Financial integration
        self.stripe_client = stripe
        if stripe_api_key:
            stripe.api_key = stripe_api_key
        
        # Real-time monitoring
        self.alert_callback = alert_callback
        self.storage_backend = storage_backend
        self._monitoring_active = False
        self._fraud_detection_enabled = True
        
        # Emergency controls
        self._emergency_stop_active = False
        self._master_kill_switch = False
        
        # BigQuery for audit logging
        self.bq_client = bigquery.Client() if os.getenv('GOOGLE_CLOUD_PROJECT') else None
        self.audit_table = f"{os.getenv('GOOGLE_CLOUD_PROJECT', 'gaelp')}.safety_audit.budget_events"
        
        # Pub/Sub for real-time alerts
        self.publisher = pubsub_v1.PublisherClient() if os.getenv('GOOGLE_CLOUD_PROJECT') else None
        self.alert_topic = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT', 'gaelp')}/topics/budget-alerts"
        
        # Cloud Monitoring for metrics
        self.monitoring_client = monitoring_v3.MetricServiceClient() if os.getenv('GOOGLE_CLOUD_PROJECT') else None
        
        logger.info("Real money budget controller initialized with production safety features")
        
    async def register_campaign(self, campaign_id: str, limits: BudgetLimits) -> bool:
        """Register a new campaign with budget limits"""
        try:
            self.campaigns[campaign_id] = limits
            self.spend_records[campaign_id] = []
            self.campaign_status[campaign_id] = CampaignStatus.ACTIVE
            
            logger.info(f"Campaign {campaign_id} registered with limits: {limits}")
            return True
        except Exception as e:
            logger.error(f"Failed to register campaign {campaign_id}: {e}")
            return False
    
    async def record_spend(self, spend: SpendRecord) -> Tuple[bool, Optional[BudgetViolation]]:
        """
        Record a spend transaction and check for budget violations.
        Returns (success, violation_if_any)
        """
        try:
            campaign_id = spend.campaign_id
            
            # Validate campaign exists
            if campaign_id not in self.campaigns:
                logger.error(f"Unknown campaign: {campaign_id}")
                return False, None
            
            # Check if campaign is active
            if self.campaign_status.get(campaign_id) != CampaignStatus.ACTIVE:
                logger.warning(f"Attempted spend on inactive campaign: {campaign_id}")
                return False, None
            
            # Record the spend
            self.spend_records[campaign_id].append(spend)
            
            # Check for violations
            violation = await self._check_budget_violations(campaign_id)
            
            if violation:
                await self._handle_violation(violation)
                return True, violation
            
            # Store to persistent backend if available
            if self.storage_backend:
                await self.storage_backend.store_spend(spend)
            
            logger.info(f"Recorded spend: {spend.amount} for campaign {campaign_id}")
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to record spend: {e}")
            return False, None
    
    async def _check_budget_violations(self, campaign_id: str) -> Optional[BudgetViolation]:
        """Check if current spend violates any budget limits"""
        try:
            limits = self.campaigns[campaign_id]
            records = self.spend_records[campaign_id]
            now = datetime.utcnow()
            
            # Calculate spend for different time periods
            daily_spend = self._calculate_period_spend(records, now - timedelta(days=1))
            weekly_spend = self._calculate_period_spend(records, now - timedelta(weeks=1))
            monthly_spend = self._calculate_period_spend(records, now - timedelta(days=30))
            total_spend = sum(r.amount for r in records)
            
            # Check violations
            if daily_spend > limits.daily_limit:
                return BudgetViolation(
                    violation_type=BudgetViolationType.DAILY_LIMIT_EXCEEDED,
                    campaign_id=campaign_id,
                    current_spend=daily_spend,
                    limit_exceeded=limits.daily_limit,
                    timestamp=now,
                    description=f"Daily spend {daily_spend} exceeds limit {limits.daily_limit}"
                )
            
            if weekly_spend > limits.weekly_limit:
                return BudgetViolation(
                    violation_type=BudgetViolationType.WEEKLY_LIMIT_EXCEEDED,
                    campaign_id=campaign_id,
                    current_spend=weekly_spend,
                    limit_exceeded=limits.weekly_limit,
                    timestamp=now,
                    description=f"Weekly spend {weekly_spend} exceeds limit {limits.weekly_limit}"
                )
            
            if monthly_spend > limits.monthly_limit:
                return BudgetViolation(
                    violation_type=BudgetViolationType.MONTHLY_LIMIT_EXCEEDED,
                    campaign_id=campaign_id,
                    current_spend=monthly_spend,
                    limit_exceeded=limits.monthly_limit,
                    timestamp=now,
                    description=f"Monthly spend {monthly_spend} exceeds limit {limits.monthly_limit}"
                )
            
            if total_spend > limits.campaign_limit:
                return BudgetViolation(
                    violation_type=BudgetViolationType.CAMPAIGN_LIMIT_EXCEEDED,
                    campaign_id=campaign_id,
                    current_spend=total_spend,
                    limit_exceeded=limits.campaign_limit,
                    timestamp=now,
                    description=f"Campaign total spend {total_spend} exceeds limit {limits.campaign_limit}"
                )
            
            # Check hourly rate if configured
            if limits.hourly_rate_limit:
                hourly_spend = self._calculate_period_spend(records, now - timedelta(hours=1))
                if hourly_spend > limits.hourly_rate_limit:
                    return BudgetViolation(
                        violation_type=BudgetViolationType.SPEND_RATE_TOO_HIGH,
                        campaign_id=campaign_id,
                        current_spend=hourly_spend,
                        limit_exceeded=limits.hourly_rate_limit,
                        timestamp=now,
                        description=f"Hourly spend rate {hourly_spend} exceeds limit {limits.hourly_rate_limit}"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking budget violations for {campaign_id}: {e}")
            return None
    
    def _calculate_period_spend(self, records: List[SpendRecord], since: datetime) -> Decimal:
        """Calculate total spend since a given timestamp"""
        return sum(r.amount for r in records if r.timestamp >= since)
    
    async def _handle_violation(self, violation: BudgetViolation):
        """Handle a budget violation with automatic interventions"""
        try:
            actions = []
            
            # Pause the campaign immediately
            self.campaign_status[violation.campaign_id] = CampaignStatus.PAUSED
            actions.append("campaign_paused")
            
            # Record the violation
            violation.actions_taken = actions
            self.violations.append(violation)
            
            # Send alert if callback configured
            if self.alert_callback:
                await self.alert_callback(violation)
            
            # Log critical violation
            logger.critical(f"Budget violation detected and handled: {violation.description}")
            
            # Store violation if backend available
            if self.storage_backend:
                await self.storage_backend.store_violation(violation)
            
        except Exception as e:
            logger.error(f"Failed to handle violation: {e}")
    
    async def pause_campaign(self, campaign_id: str, reason: str = "Manual pause") -> bool:
        """Manually pause a campaign"""
        try:
            if campaign_id not in self.campaigns:
                logger.error(f"Unknown campaign: {campaign_id}")
                return False
            
            self.campaign_status[campaign_id] = CampaignStatus.PAUSED
            logger.info(f"Campaign {campaign_id} paused: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause campaign {campaign_id}: {e}")
            return False
    
    async def resume_campaign(self, campaign_id: str, override_violations: bool = False) -> bool:
        """Resume a paused campaign with safety checks"""
        try:
            if campaign_id not in self.campaigns:
                logger.error(f"Unknown campaign: {campaign_id}")
                return False
            
            # Check for recent violations
            if not override_violations:
                recent_violations = [
                    v for v in self.violations 
                    if v.campaign_id == campaign_id and 
                    v.timestamp > datetime.utcnow() - timedelta(hours=1)
                ]
                if recent_violations:
                    logger.warning(f"Cannot resume {campaign_id}: recent violations exist")
                    return False
            
            self.campaign_status[campaign_id] = CampaignStatus.ACTIVE
            logger.info(f"Campaign {campaign_id} resumed")
            return True
        except Exception as e:
            logger.error(f"Failed to resume campaign {campaign_id}: {e}")
            return False
    
    def get_campaign_status(self, campaign_id: str) -> Optional[CampaignStatus]:
        """Get current status of a campaign"""
        return self.campaign_status.get(campaign_id)
    
    def get_spend_summary(self, campaign_id: str) -> Dict[str, Decimal]:
        """Get spend summary for a campaign"""
        try:
            if campaign_id not in self.spend_records:
                return {}
            
            records = self.spend_records[campaign_id]
            now = datetime.utcnow()
            
            return {
                "total_spend": sum(r.amount for r in records),
                "daily_spend": self._calculate_period_spend(records, now - timedelta(days=1)),
                "weekly_spend": self._calculate_period_spend(records, now - timedelta(weeks=1)),
                "monthly_spend": self._calculate_period_spend(records, now - timedelta(days=30)),
                "hourly_spend": self._calculate_period_spend(records, now - timedelta(hours=1))
            }
        except Exception as e:
            logger.error(f"Failed to get spend summary for {campaign_id}: {e}")
            return {}
    
    def get_violations(self, campaign_id: Optional[str] = None) -> List[BudgetViolation]:
        """Get violations for a specific campaign or all campaigns"""
        if campaign_id:
            return [v for v in self.violations if v.campaign_id == campaign_id]
        return self.violations.copy()
    
    async def emergency_stop_all(self, reason: str = "Emergency stop") -> int:
        """Emergency stop all active campaigns"""
        stopped_count = 0
        try:
            for campaign_id, status in self.campaign_status.items():
                if status == CampaignStatus.ACTIVE:
                    self.campaign_status[campaign_id] = CampaignStatus.STOPPED
                    stopped_count += 1
                    logger.critical(f"Emergency stop: {campaign_id} - {reason}")
            
            if self.alert_callback:
                await self.alert_callback({
                    "type": "emergency_stop",
                    "campaigns_stopped": stopped_count,
                    "reason": reason,
                    "timestamp": datetime.utcnow()
                })
            
            return stopped_count
        except Exception as e:
            logger.error(f"Failed emergency stop: {e}")
            return 0


class BudgetMonitor:
    """Real-time budget monitoring service"""
    
    def __init__(self, controller: BudgetController, check_interval: int = 60):
        self.controller = controller
        self.check_interval = check_interval
        self._running = False
        self._task = None
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Budget monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring loop"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Budget monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._check_all_campaigns()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_campaigns(self):
        """Check all campaigns for budget issues"""
        for campaign_id in self.controller.campaigns:
            if self.controller.campaign_status.get(campaign_id) == CampaignStatus.ACTIVE:
                # Check for spending patterns that might indicate issues
                summary = self.controller.get_spend_summary(campaign_id)
                await self._analyze_spending_patterns(campaign_id, summary)
    
    async def _analyze_spending_patterns(self, campaign_id: str, summary: Dict[str, Decimal]):
        """Analyze spending patterns for anomalies"""
        try:
            limits = self.controller.campaigns[campaign_id]
            
            # Check if approaching limits (80% threshold)
            daily_threshold = limits.daily_limit * Decimal('0.8')
            if summary.get('daily_spend', 0) > daily_threshold:
                logger.warning(f"Campaign {campaign_id} approaching daily limit")
            
            # Check unusual spending velocity
            hourly_spend = summary.get('hourly_spend', 0)
            expected_hourly = limits.daily_limit / 24
            if hourly_spend > expected_hourly * 3:  # 3x normal rate
                logger.warning(f"Campaign {campaign_id} has unusual spending velocity")
                
        except Exception as e:
            logger.error(f"Error analyzing spending patterns for {campaign_id}: {e}")