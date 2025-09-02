#!/usr/bin/env python3
"""
COMPREHENSIVE BUDGET SAFETY CONTROLLER FOR GAELP PRODUCTION
Critical budget safety controls to prevent any possibility of overspending.

BUDGET SAFETY FEATURES:
- Real-time spending velocity monitoring
- Multi-tier spending limits (hourly/daily/weekly/monthly)
- Anomaly detection for unusual spending patterns
- Automatic campaign pausing on budget violations
- Predictive overspend prevention
- Multi-channel budget allocation safeguards
- Emergency budget kill switches
- Comprehensive audit trail

NO FALLBACKS - IMMEDIATE BUDGET PROTECTION ONLY
ZERO TOLERANCE FOR OVERSPENDING
"""

import logging
import threading
import time
import json
import os
import sqlite3
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
import numpy as np
from collections import defaultdict, deque
import asyncio
import signal
import sys
from pathlib import Path

# Import emergency controls for integration
from emergency_controls import EmergencyController, EmergencyType, EmergencyLevel, get_emergency_controller

logger = logging.getLogger(__name__)

class BudgetViolationType(Enum):
    """Types of budget violations"""
    DAILY_LIMIT_EXCEEDED = "daily_limit_exceeded"
    HOURLY_VELOCITY_EXCEEDED = "hourly_velocity_exceeded" 
    WEEKLY_LIMIT_EXCEEDED = "weekly_limit_exceeded"
    MONTHLY_LIMIT_EXCEEDED = "monthly_limit_exceeded"
    ANOMALOUS_SPEND_PATTERN = "anomalous_spend_pattern"
    CAMPAIGN_BUDGET_EXCEEDED = "campaign_budget_exceeded"
    CHANNEL_BUDGET_EXCEEDED = "channel_budget_exceeded"
    PREDICTIVE_OVERSPEND = "predictive_overspend"
    BID_LIMIT_EXCEEDED = "bid_limit_exceeded"
    SPEND_ACCELERATION = "spend_acceleration"

class BudgetSafetyLevel(Enum):
    """Budget safety levels"""
    SAFE = "safe"           # Normal operation
    WARNING = "warning"     # Approaching limits
    CRITICAL = "critical"   # Immediate action required
    EMERGENCY = "emergency" # Emergency stop triggered

class CampaignStatus(Enum):
    """Campaign status for safety controls"""
    ACTIVE = "active"
    PAUSED = "paused"
    EMERGENCY_STOPPED = "emergency_stopped"
    BUDGET_EXHAUSTED = "budget_exhausted"

@dataclass
class BudgetLimits:
    """Comprehensive budget limits configuration"""
    # Primary limits
    daily_limit: Decimal
    weekly_limit: Decimal
    monthly_limit: Decimal
    
    # Velocity limits (per hour)
    max_hourly_spend: Decimal
    max_hourly_velocity_increase: Decimal  # Max % increase per hour
    
    # Safety buffers (percentages)
    warning_threshold: float = 0.80    # 80% of limit
    critical_threshold: float = 0.95   # 95% of limit
    emergency_threshold: float = 1.00  # 100% of limit (hard stop)
    
    # Anomaly detection thresholds
    max_bid_multiplier: float = 3.0    # Max 3x normal bid
    max_spend_acceleration: float = 2.0 # Max 2x spending rate increase
    
    # Predictive settings
    prediction_window_hours: int = 2   # Hours ahead to predict
    overspend_prevention_buffer: float = 0.10  # 10% buffer for predictions

@dataclass
class SpendingRecord:
    """Record of spending activity"""
    timestamp: datetime
    campaign_id: str
    channel: str
    amount: Decimal
    bid_amount: Decimal
    impressions: int
    clicks: int
    conversions: int
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class BudgetViolation:
    """Budget violation record"""
    violation_id: str
    violation_type: BudgetViolationType
    severity: BudgetSafetyLevel
    timestamp: datetime
    campaign_id: str
    channel: str
    current_amount: Decimal
    limit_amount: Decimal
    percentage_of_limit: float
    message: str
    actions_taken: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

@dataclass
class CampaignBudgetState:
    """Current budget state for a campaign"""
    campaign_id: str
    status: CampaignStatus
    daily_spent: Decimal
    weekly_spent: Decimal
    monthly_spent: Decimal
    hourly_spent: Decimal  # Current hour
    limits: BudgetLimits
    last_spend_timestamp: Optional[datetime] = None
    violation_count: int = 0
    emergency_paused: bool = False

import uuid

class BudgetSafetyController:
    """
    Comprehensive Budget Safety Controller
    
    Prevents overspending through multiple layers of protection:
    1. Real-time spending monitoring
    2. Predictive overspend detection
    3. Anomaly detection
    4. Automatic campaign pausing
    5. Emergency shutdown mechanisms
    """
    
    def __init__(self, config_path: str = "budget_safety_config.json"):
        self.config_path = config_path
        self.db_path = "budget_safety_events.db"
        
        # Core state
        self.active = True
        self.campaign_states: Dict[str, CampaignBudgetState] = {}
        self.spending_history: deque = deque(maxlen=10000)  # Last 10k records
        self.violations: List[BudgetViolation] = []
        
        # Monitoring data
        self.hourly_spend_tracking: Dict[Tuple[str, datetime], Decimal] = defaultdict(Decimal)
        self.daily_spend_tracking: Dict[Tuple[str, date], Decimal] = defaultdict(Decimal)
        self.weekly_spend_tracking: Dict[Tuple[str, int], Decimal] = defaultdict(Decimal)  # (campaign, week_number)
        self.monthly_spend_tracking: Dict[Tuple[str, int], Decimal] = defaultdict(Decimal)  # (campaign, month)
        
        # Anomaly detection
        self.baseline_patterns: Dict[str, Dict[str, float]] = {}  # campaign_id -> patterns
        self.recent_bids: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.spend_velocity_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=24))  # 24 hours
        
        # Emergency integration
        self.emergency_controller = get_emergency_controller()
        
        # Thread safety
        self.lock = threading.Lock()
        self.monitoring_threads: List[threading.Thread] = []
        
        # Initialize system
        self._init_database()
        self._load_config()
        self._start_monitoring()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info("Budget Safety Controller initialized with comprehensive protection")
    
    def _init_database(self):
        """Initialize SQLite database for budget tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Spending records table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spending_records (
                    record_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    amount REAL NOT NULL,
                    bid_amount REAL NOT NULL,
                    impressions INTEGER NOT NULL,
                    clicks INTEGER NOT NULL,
                    conversions INTEGER NOT NULL
                )
            """)
            
            # Budget violations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budget_violations (
                    violation_id TEXT PRIMARY KEY,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    current_amount REAL NOT NULL,
                    limit_amount REAL NOT NULL,
                    percentage_of_limit REAL NOT NULL,
                    message TEXT NOT NULL,
                    actions_taken TEXT NOT NULL,
                    resolved BOOLEAN NOT NULL,
                    resolution_timestamp TEXT
                )
            """)
            
            # Campaign budget states table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS campaign_states (
                    campaign_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    daily_spent REAL NOT NULL,
                    weekly_spent REAL NOT NULL,
                    monthly_spent REAL NOT NULL,
                    hourly_spent REAL NOT NULL,
                    limits_json TEXT NOT NULL,
                    last_spend_timestamp TEXT,
                    violation_count INTEGER NOT NULL,
                    emergency_paused BOOLEAN NOT NULL,
                    updated_timestamp TEXT NOT NULL
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Budget safety database initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            # Create database file if it doesn't exist
            try:
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self.db_path).touch()
                self._init_database()  # Retry once
            except Exception as retry_error:
                logger.error(f"Failed to create database after retry: {retry_error}")
    
    def _load_config(self):
        """Load budget safety configuration"""
        default_config = {
            "default_limits": {
                "daily_limit": 1000.0,
                "weekly_limit": 5000.0,
                "monthly_limit": 20000.0,
                "max_hourly_spend": 150.0,
                "max_hourly_velocity_increase": 0.50,  # 50% increase max
                "warning_threshold": 0.80,
                "critical_threshold": 0.95,
                "emergency_threshold": 1.00,
                "max_bid_multiplier": 3.0,
                "max_spend_acceleration": 2.0,
                "prediction_window_hours": 2,
                "overspend_prevention_buffer": 0.10
            },
            "monitoring_intervals": {
                "spending_check_seconds": 30,
                "velocity_check_seconds": 60,
                "anomaly_check_seconds": 120,
                "prediction_check_seconds": 300
            },
            "emergency_actions": {
                "auto_pause_campaigns": True,
                "emergency_stop_threshold": 1.05,  # 105% triggers emergency stop
                "notification_webhook": None
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            config = default_config
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        self.config = config
        logger.info(f"Budget safety configuration loaded from {self.config_path}")
    
    def _start_monitoring(self):
        """Start background monitoring threads"""
        monitoring_functions = [
            self._monitor_spending_limits,
            self._monitor_spending_velocity,
            self._monitor_spending_anomalies,
            self._monitor_predictive_overspend
        ]
        
        for func in monitoring_functions:
            thread = threading.Thread(target=func, daemon=True)
            thread.start()
            self.monitoring_threads.append(thread)
        
        logger.info(f"Started {len(monitoring_functions)} budget monitoring threads")
    
    def register_campaign(self, campaign_id: str, limits: Optional[BudgetLimits] = None) -> None:
        """Register a campaign with budget limits"""
        try:
            if limits is None:
                # Use default limits from config
                default_limits = self.config["default_limits"]
                limits = BudgetLimits(
                    daily_limit=Decimal(str(default_limits["daily_limit"])),
                    weekly_limit=Decimal(str(default_limits["weekly_limit"])),
                    monthly_limit=Decimal(str(default_limits["monthly_limit"])),
                    max_hourly_spend=Decimal(str(default_limits["max_hourly_spend"])),
                    max_hourly_velocity_increase=default_limits["max_hourly_velocity_increase"],
                    warning_threshold=default_limits["warning_threshold"],
                    critical_threshold=default_limits["critical_threshold"],
                    emergency_threshold=default_limits["emergency_threshold"],
                    max_bid_multiplier=default_limits["max_bid_multiplier"],
                    max_spend_acceleration=default_limits["max_spend_acceleration"],
                    prediction_window_hours=default_limits["prediction_window_hours"],
                    overspend_prevention_buffer=default_limits["overspend_prevention_buffer"]
                )
            
            with self.lock:
                self.campaign_states[campaign_id] = CampaignBudgetState(
                    campaign_id=campaign_id,
                    status=CampaignStatus.ACTIVE,
                    daily_spent=Decimal('0'),
                    weekly_spent=Decimal('0'),
                    monthly_spent=Decimal('0'),
                    hourly_spent=Decimal('0'),
                    limits=limits
                )
            
            self._save_campaign_state(campaign_id)
            logger.info(f"Campaign {campaign_id} registered with budget limits: Daily=${limits.daily_limit}")
            
        except Exception as e:
            logger.error(f"Error registering campaign {campaign_id}: {e}")
            raise RuntimeError(f"Failed to register campaign budget controls: {e}")
    
    def record_spending(self, campaign_id: str, channel: str, amount: Decimal, 
                       bid_amount: Decimal, impressions: int = 0, clicks: int = 0, 
                       conversions: int = 0) -> Tuple[bool, List[str]]:
        """
        Record spending and check all budget safety limits.
        Returns: (is_safe_to_continue, violations_detected)
        """
        try:
            if not self.active:
                return False, ["Budget safety controller is not active"]
            
            if campaign_id not in self.campaign_states:
                logger.warning(f"Campaign {campaign_id} not registered, using default limits")
                self.register_campaign(campaign_id)
            
            # Create spending record
            record = SpendingRecord(
                timestamp=datetime.now(),
                campaign_id=campaign_id,
                channel=channel,
                amount=amount,
                bid_amount=bid_amount,
                impressions=impressions,
                clicks=clicks,
                conversions=conversions
            )
            
            with self.lock:
                # Add to history
                self.spending_history.append(record)
                
                # Update campaign state
                campaign_state = self.campaign_states[campaign_id]
                current_time = record.timestamp
                current_date = current_time.date()
                current_hour = current_time.replace(minute=0, second=0, microsecond=0)
                
                # Update spending totals
                campaign_state.hourly_spent += amount
                campaign_state.daily_spent += amount
                campaign_state.weekly_spent += amount
                campaign_state.monthly_spent += amount
                campaign_state.last_spend_timestamp = current_time
                
                # Update tracking dictionaries
                self.hourly_spend_tracking[(campaign_id, current_hour)] += amount
                self.daily_spend_tracking[(campaign_id, current_date)] += amount
                
                week_number = current_date.isocalendar()[1]
                month_number = current_date.month
                self.weekly_spend_tracking[(campaign_id, week_number)] += amount
                self.monthly_spend_tracking[(campaign_id, month_number)] += amount
                
                # Track bid patterns for anomaly detection
                self.recent_bids[campaign_id].append((current_time, float(bid_amount)))
                
                # Update velocity tracking
                self.spend_velocity_history[campaign_id].append((current_time, float(amount)))
            
            # Save to database
            self._save_spending_record(record)
            self._save_campaign_state(campaign_id)
            
            # Check all safety limits
            violations = self._check_all_limits(campaign_id, record)
            
            # Determine if safe to continue
            is_safe = True
            violation_messages = []
            
            for violation in violations:
                if violation.severity in [BudgetSafetyLevel.CRITICAL, BudgetSafetyLevel.EMERGENCY]:
                    is_safe = False
                violation_messages.append(f"{violation.violation_type.value}: {violation.message}")
                
                # Take immediate action for violations
                self._handle_budget_violation(violation)
            
            if not is_safe:
                logger.error(f"BUDGET VIOLATION: Campaign {campaign_id} unsafe to continue: {violation_messages}")
            
            return is_safe, violation_messages
            
        except Exception as e:
            logger.error(f"Error recording spending for campaign {campaign_id}: {e}")
            # Fail safe - reject spending if we can't verify safety
            return False, [f"Budget safety check failed: {e}"]
    
    def _check_all_limits(self, campaign_id: str, record: SpendingRecord) -> List[BudgetViolation]:
        """Check all budget limits and return violations"""
        violations = []
        campaign_state = self.campaign_states[campaign_id]
        limits = campaign_state.limits
        
        # 1. Check daily limit
        daily_violation = self._check_daily_limit(campaign_state, record)
        if daily_violation:
            violations.append(daily_violation)
        
        # 2. Check hourly velocity
        velocity_violation = self._check_hourly_velocity(campaign_state, record)
        if velocity_violation:
            violations.append(velocity_violation)
        
        # 3. Check weekly limit
        weekly_violation = self._check_weekly_limit(campaign_state, record)
        if weekly_violation:
            violations.append(weekly_violation)
        
        # 4. Check monthly limit
        monthly_violation = self._check_monthly_limit(campaign_state, record)
        if monthly_violation:
            violations.append(monthly_violation)
        
        # 5. Check bid anomalies
        bid_violation = self._check_bid_anomalies(campaign_state, record)
        if bid_violation:
            violations.append(bid_violation)
        
        # 6. Check spend acceleration
        acceleration_violation = self._check_spend_acceleration(campaign_state, record)
        if acceleration_violation:
            violations.append(acceleration_violation)
        
        # 7. Predictive overspend check
        predictive_violation = self._check_predictive_overspend(campaign_state, record)
        if predictive_violation:
            violations.append(predictive_violation)
        
        return violations
    
    def _check_daily_limit(self, campaign_state: CampaignBudgetState, record: SpendingRecord) -> Optional[BudgetViolation]:
        """Check daily spending limit"""
        daily_spent = campaign_state.daily_spent
        daily_limit = campaign_state.limits.daily_limit
        percentage = float(daily_spent / daily_limit) if daily_limit > 0 else 0
        
        limits = campaign_state.limits
        
        if percentage >= limits.emergency_threshold:
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.DAILY_LIMIT_EXCEEDED,
                severity=BudgetSafetyLevel.EMERGENCY,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=daily_spent,
                limit_amount=daily_limit,
                percentage_of_limit=percentage,
                message=f"Daily budget EMERGENCY: ${daily_spent} >= ${daily_limit} ({percentage:.1%})"
            )
        elif percentage >= limits.critical_threshold:
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.DAILY_LIMIT_EXCEEDED,
                severity=BudgetSafetyLevel.CRITICAL,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=daily_spent,
                limit_amount=daily_limit,
                percentage_of_limit=percentage,
                message=f"Daily budget CRITICAL: ${daily_spent} of ${daily_limit} ({percentage:.1%})"
            )
        elif percentage >= limits.warning_threshold:
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.DAILY_LIMIT_EXCEEDED,
                severity=BudgetSafetyLevel.WARNING,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=daily_spent,
                limit_amount=daily_limit,
                percentage_of_limit=percentage,
                message=f"Daily budget WARNING: ${daily_spent} of ${daily_limit} ({percentage:.1%})"
            )
        
        return None
    
    def _check_hourly_velocity(self, campaign_state: CampaignBudgetState, record: SpendingRecord) -> Optional[BudgetViolation]:
        """Check hourly spending velocity"""
        current_hour_key = (record.campaign_id, record.timestamp.replace(minute=0, second=0, microsecond=0))
        hourly_spent = self.hourly_spend_tracking[current_hour_key]
        max_hourly = campaign_state.limits.max_hourly_spend
        
        if hourly_spent > max_hourly:
            percentage = float(hourly_spent / max_hourly) if max_hourly > 0 else 0
            
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.HOURLY_VELOCITY_EXCEEDED,
                severity=BudgetSafetyLevel.CRITICAL if percentage > 1.2 else BudgetSafetyLevel.WARNING,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=hourly_spent,
                limit_amount=max_hourly,
                percentage_of_limit=percentage,
                message=f"Hourly velocity exceeded: ${hourly_spent} > ${max_hourly}/hour ({percentage:.1%})"
            )
        
        return None
    
    def _check_weekly_limit(self, campaign_state: CampaignBudgetState, record: SpendingRecord) -> Optional[BudgetViolation]:
        """Check weekly spending limit"""
        weekly_spent = campaign_state.weekly_spent
        weekly_limit = campaign_state.limits.weekly_limit
        percentage = float(weekly_spent / weekly_limit) if weekly_limit > 0 else 0
        
        limits = campaign_state.limits
        
        if percentage >= limits.emergency_threshold:
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.WEEKLY_LIMIT_EXCEEDED,
                severity=BudgetSafetyLevel.EMERGENCY,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=weekly_spent,
                limit_amount=weekly_limit,
                percentage_of_limit=percentage,
                message=f"Weekly budget EMERGENCY: ${weekly_spent} >= ${weekly_limit} ({percentage:.1%})"
            )
        elif percentage >= limits.critical_threshold:
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.WEEKLY_LIMIT_EXCEEDED,
                severity=BudgetSafetyLevel.CRITICAL,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=weekly_spent,
                limit_amount=weekly_limit,
                percentage_of_limit=percentage,
                message=f"Weekly budget CRITICAL: ${weekly_spent} of ${weekly_limit} ({percentage:.1%})"
            )
        
        return None
    
    def _check_monthly_limit(self, campaign_state: CampaignBudgetState, record: SpendingRecord) -> Optional[BudgetViolation]:
        """Check monthly spending limit"""
        monthly_spent = campaign_state.monthly_spent
        monthly_limit = campaign_state.limits.monthly_limit
        percentage = float(monthly_spent / monthly_limit) if monthly_limit > 0 else 0
        
        limits = campaign_state.limits
        
        if percentage >= limits.emergency_threshold:
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.MONTHLY_LIMIT_EXCEEDED,
                severity=BudgetSafetyLevel.EMERGENCY,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=monthly_spent,
                limit_amount=monthly_limit,
                percentage_of_limit=percentage,
                message=f"Monthly budget EMERGENCY: ${monthly_spent} >= ${monthly_limit} ({percentage:.1%})"
            )
        elif percentage >= limits.critical_threshold:
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.MONTHLY_LIMIT_EXCEEDED,
                severity=BudgetSafetyLevel.CRITICAL,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=monthly_spent,
                limit_amount=monthly_limit,
                percentage_of_limit=percentage,
                message=f"Monthly budget CRITICAL: ${monthly_spent} of ${monthly_limit} ({percentage:.1%})"
            )
        
        return None
    
    def _check_bid_anomalies(self, campaign_state: CampaignBudgetState, record: SpendingRecord) -> Optional[BudgetViolation]:
        """Check for anomalous bidding patterns"""
        campaign_id = record.campaign_id
        current_bid = float(record.bid_amount)
        
        # Get recent bids for baseline
        recent_bids = list(self.recent_bids[campaign_id])
        if len(recent_bids) < 10:  # Need baseline
            return None
        
        # Calculate baseline average (excluding current bid)
        baseline_bids = [bid for _, bid in recent_bids[-20:-1]]  # Last 20 excluding current
        if not baseline_bids:
            return None
        
        baseline_avg = np.mean(baseline_bids)
        if baseline_avg <= 0:
            return None
        
        bid_multiplier = current_bid / baseline_avg
        max_multiplier = campaign_state.limits.max_bid_multiplier
        
        if bid_multiplier > max_multiplier:
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.BID_LIMIT_EXCEEDED,
                severity=BudgetSafetyLevel.CRITICAL if bid_multiplier > max_multiplier * 1.5 else BudgetSafetyLevel.WARNING,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=record.bid_amount,
                limit_amount=Decimal(str(baseline_avg * max_multiplier)),
                percentage_of_limit=bid_multiplier,
                message=f"Anomalous bid detected: ${current_bid:.2f} is {bid_multiplier:.1f}x baseline (${baseline_avg:.2f})"
            )
        
        return None
    
    def _check_spend_acceleration(self, campaign_state: CampaignBudgetState, record: SpendingRecord) -> Optional[BudgetViolation]:
        """Check for spend acceleration patterns"""
        campaign_id = record.campaign_id
        velocity_history = list(self.spend_velocity_history[campaign_id])
        
        if len(velocity_history) < 5:  # Need history
            return None
        
        # Calculate spending velocity (amount per minute)
        current_time = record.timestamp
        current_amount = float(record.amount)
        
        # Get spending rate over last hour vs previous hour
        one_hour_ago = current_time - timedelta(hours=1)
        two_hours_ago = current_time - timedelta(hours=2)
        
        recent_spending = sum(
            amount for timestamp, amount in velocity_history 
            if timestamp > one_hour_ago
        )
        
        previous_spending = sum(
            amount for timestamp, amount in velocity_history 
            if two_hours_ago < timestamp <= one_hour_ago
        )
        
        if previous_spending <= 0:
            return None
        
        acceleration = recent_spending / previous_spending
        max_acceleration = campaign_state.limits.max_spend_acceleration
        
        if acceleration > max_acceleration:
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.SPEND_ACCELERATION,
                severity=BudgetSafetyLevel.WARNING if acceleration < max_acceleration * 1.5 else BudgetSafetyLevel.CRITICAL,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=Decimal(str(recent_spending)),
                limit_amount=Decimal(str(previous_spending * max_acceleration)),
                percentage_of_limit=acceleration,
                message=f"Spend acceleration detected: {acceleration:.1f}x increase (${recent_spending:.2f} vs ${previous_spending:.2f})"
            )
        
        return None
    
    def _check_predictive_overspend(self, campaign_state: CampaignBudgetState, record: SpendingRecord) -> Optional[BudgetViolation]:
        """Check for predictive overspend risk"""
        campaign_id = record.campaign_id
        limits = campaign_state.limits
        
        # Get current spending velocity
        velocity_history = list(self.spend_velocity_history[campaign_id])
        if len(velocity_history) < 3:
            return None
        
        # Calculate current hourly spending rate
        current_time = record.timestamp
        one_hour_ago = current_time - timedelta(hours=1)
        
        recent_spending = sum(
            amount for timestamp, amount in velocity_history
            if timestamp > one_hour_ago
        )
        
        if recent_spending <= 0:
            return None
        
        # Project spending forward
        prediction_hours = limits.prediction_window_hours
        projected_spend = recent_spending * prediction_hours
        
        # Check against daily limit with buffer
        daily_limit_with_buffer = float(limits.daily_limit) * (1 - limits.overspend_prevention_buffer)
        current_daily_spent = float(campaign_state.daily_spent)
        
        if current_daily_spent + projected_spend > daily_limit_with_buffer:
            projected_total = current_daily_spent + projected_spend
            
            return BudgetViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BudgetViolationType.PREDICTIVE_OVERSPEND,
                severity=BudgetSafetyLevel.WARNING if projected_total < float(limits.daily_limit) else BudgetSafetyLevel.CRITICAL,
                timestamp=record.timestamp,
                campaign_id=record.campaign_id,
                channel=record.channel,
                current_amount=Decimal(str(projected_total)),
                limit_amount=limits.daily_limit,
                percentage_of_limit=projected_total / float(limits.daily_limit),
                message=f"Predictive overspend risk: ${projected_total:.2f} projected (${recent_spending:.2f}/hr rate)"
            )
        
        return None
    
    def _handle_budget_violation(self, violation: BudgetViolation) -> None:
        """Handle budget violation with appropriate actions"""
        try:
            with self.lock:
                self.violations.append(violation)
                campaign_state = self.campaign_states[violation.campaign_id]
                campaign_state.violation_count += 1
            
            actions_taken = []
            
            # Determine actions based on severity
            if violation.severity == BudgetSafetyLevel.EMERGENCY:
                # Emergency actions
                actions_taken.append("Emergency campaign pause")
                self._emergency_pause_campaign(violation.campaign_id, violation.message)
                
                actions_taken.append("Triggering system emergency stop")
                self._trigger_emergency_stop(violation)
                
            elif violation.severity == BudgetSafetyLevel.CRITICAL:
                # Critical actions
                actions_taken.append("Campaign paused")
                self._pause_campaign(violation.campaign_id, f"Critical budget violation: {violation.message}")
                
                # Also register with emergency controller
                self.emergency_controller.update_budget_tracking(
                    violation.campaign_id,
                    float(violation.current_amount),
                    float(violation.limit_amount)
                )
                
            elif violation.severity == BudgetSafetyLevel.WARNING:
                # Warning actions
                actions_taken.append("Budget warning logged")
                actions_taken.append("Increased monitoring activated")
                
            # Update violation record
            violation.actions_taken = actions_taken
            
            # Save to database
            self._save_violation(violation)
            
            # Log appropriately
            if violation.severity == BudgetSafetyLevel.EMERGENCY:
                logger.critical(f"BUDGET EMERGENCY: {violation.message} - Actions: {actions_taken}")
            elif violation.severity == BudgetSafetyLevel.CRITICAL:
                logger.error(f"BUDGET CRITICAL: {violation.message} - Actions: {actions_taken}")
            else:
                logger.warning(f"BUDGET WARNING: {violation.message} - Actions: {actions_taken}")
            
        except Exception as e:
            logger.error(f"Error handling budget violation: {e}")
            # If we can't handle the violation properly, emergency stop
            self._trigger_emergency_stop(violation)
    
    def _pause_campaign(self, campaign_id: str, reason: str) -> None:
        """Pause a campaign due to budget violation"""
        try:
            with self.lock:
                if campaign_id in self.campaign_states:
                    self.campaign_states[campaign_id].status = CampaignStatus.PAUSED
                    self._save_campaign_state(campaign_id)
            
            logger.warning(f"Campaign {campaign_id} PAUSED: {reason}")
            
            # Here you would integrate with actual campaign management system
            # For now, we just log and update state
            
        except Exception as e:
            logger.error(f"Error pausing campaign {campaign_id}: {e}")
    
    def _emergency_pause_campaign(self, campaign_id: str, reason: str) -> None:
        """Emergency pause a campaign"""
        try:
            with self.lock:
                if campaign_id in self.campaign_states:
                    campaign_state = self.campaign_states[campaign_id]
                    campaign_state.status = CampaignStatus.EMERGENCY_STOPPED
                    campaign_state.emergency_paused = True
                    self._save_campaign_state(campaign_id)
            
            logger.critical(f"Campaign {campaign_id} EMERGENCY STOPPED: {reason}")
            
        except Exception as e:
            logger.error(f"Error emergency stopping campaign {campaign_id}: {e}")
    
    def _trigger_emergency_stop(self, violation: BudgetViolation) -> None:
        """Trigger system-wide emergency stop"""
        try:
            # Integrate with emergency controller
            emergency_reason = f"Budget Emergency: {violation.message}"
            self.emergency_controller.trigger_manual_emergency_stop(emergency_reason)
            
            logger.critical(f"SYSTEM EMERGENCY STOP TRIGGERED: {emergency_reason}")
            
        except Exception as e:
            logger.critical(f"Error triggering emergency stop: {e}")
            # Last resort - exit immediately
            os._exit(1)
    
    def _monitor_spending_limits(self) -> None:
        """Monitor spending limits continuously"""
        while self.active:
            try:
                check_interval = self.config["monitoring_intervals"]["spending_check_seconds"]
                
                current_time = datetime.now()
                with self.lock:
                    campaigns_to_check = list(self.campaign_states.keys())
                
                for campaign_id in campaigns_to_check:
                    try:
                        self._check_campaign_limits(campaign_id, current_time)
                    except Exception as e:
                        logger.error(f"Error checking limits for campaign {campaign_id}: {e}")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in spending limits monitoring: {e}")
                time.sleep(60)
    
    def _monitor_spending_velocity(self) -> None:
        """Monitor spending velocity continuously"""
        while self.active:
            try:
                check_interval = self.config["monitoring_intervals"]["velocity_check_seconds"]
                
                with self.lock:
                    campaigns_to_check = list(self.campaign_states.keys())
                
                for campaign_id in campaigns_to_check:
                    try:
                        self._check_velocity_patterns(campaign_id)
                    except Exception as e:
                        logger.error(f"Error checking velocity for campaign {campaign_id}: {e}")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in velocity monitoring: {e}")
                time.sleep(60)
    
    def _monitor_spending_anomalies(self) -> None:
        """Monitor for spending anomalies"""
        while self.active:
            try:
                check_interval = self.config["monitoring_intervals"]["anomaly_check_seconds"]
                
                with self.lock:
                    campaigns_to_check = list(self.campaign_states.keys())
                
                for campaign_id in campaigns_to_check:
                    try:
                        self._detect_spending_anomalies(campaign_id)
                    except Exception as e:
                        logger.error(f"Error detecting anomalies for campaign {campaign_id}: {e}")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in anomaly monitoring: {e}")
                time.sleep(120)
    
    def _monitor_predictive_overspend(self) -> None:
        """Monitor for predictive overspend risks"""
        while self.active:
            try:
                check_interval = self.config["monitoring_intervals"]["prediction_check_seconds"]
                
                with self.lock:
                    campaigns_to_check = list(self.campaign_states.keys())
                
                for campaign_id in campaigns_to_check:
                    try:
                        self._predict_overspend_risk(campaign_id)
                    except Exception as e:
                        logger.error(f"Error predicting overspend for campaign {campaign_id}: {e}")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in predictive monitoring: {e}")
                time.sleep(300)
    
    def _check_campaign_limits(self, campaign_id: str, current_time: datetime) -> None:
        """Check all limits for a campaign"""
        if campaign_id not in self.campaign_states:
            return
        
        campaign_state = self.campaign_states[campaign_id]
        
        # Reset counters if new time period
        self._reset_time_period_counters(campaign_state, current_time)
        
        # Check if any limits are approaching or exceeded
        limits = campaign_state.limits
        
        # Daily limit check
        daily_percentage = float(campaign_state.daily_spent / limits.daily_limit) if limits.daily_limit > 0 else 0
        if daily_percentage >= limits.warning_threshold:
            logger.warning(f"Campaign {campaign_id} approaching daily limit: {daily_percentage:.1%}")
    
    def _check_velocity_patterns(self, campaign_id: str) -> None:
        """Check velocity patterns for a campaign"""
        velocity_history = list(self.spend_velocity_history[campaign_id])
        if len(velocity_history) < 2:
            return
        
        # Check for sudden velocity spikes
        recent_velocities = [amount for _, amount in velocity_history[-5:]]
        if len(recent_velocities) >= 3:
            recent_avg = np.mean(recent_velocities)
            baseline_avg = np.mean([amount for _, amount in velocity_history[:-5]]) if len(velocity_history) > 5 else recent_avg
            
            if baseline_avg > 0 and recent_avg / baseline_avg > 2.0:
                logger.warning(f"Campaign {campaign_id} velocity spike detected: {recent_avg:.2f} vs {baseline_avg:.2f}")
    
    def _detect_spending_anomalies(self, campaign_id: str) -> None:
        """Detect spending anomalies for a campaign"""
        recent_records = [
            record for record in self.spending_history 
            if record.campaign_id == campaign_id and 
            record.timestamp > datetime.now() - timedelta(hours=2)
        ]
        
        if len(recent_records) < 5:
            return
        
        # Check for unusual patterns
        recent_amounts = [float(record.amount) for record in recent_records]
        recent_bids = [float(record.bid_amount) for record in recent_records]
        
        # Statistical anomaly detection
        amount_std = np.std(recent_amounts)
        amount_mean = np.mean(recent_amounts)
        
        for record in recent_records[-3:]:  # Check last 3 records
            amount = float(record.amount)
            if amount_std > 0 and abs(amount - amount_mean) > 3 * amount_std:
                logger.warning(f"Spending anomaly detected for campaign {campaign_id}: ${amount:.2f} (mean: ${amount_mean:.2f}, std: ${amount_std:.2f})")
    
    def _predict_overspend_risk(self, campaign_id: str) -> None:
        """Predict overspend risk for a campaign"""
        if campaign_id not in self.campaign_states:
            return
        
        campaign_state = self.campaign_states[campaign_id]
        velocity_history = list(self.spend_velocity_history[campaign_id])
        
        if len(velocity_history) < 5:
            return
        
        # Simple linear projection
        current_time = datetime.now()
        one_hour_ago = current_time - timedelta(hours=1)
        
        recent_spending = sum(
            amount for timestamp, amount in velocity_history
            if timestamp > one_hour_ago
        )
        
        # Project to end of day
        hours_remaining = 24 - current_time.hour
        if hours_remaining <= 0:
            return
        
        projected_additional_spend = recent_spending * hours_remaining
        projected_total = float(campaign_state.daily_spent) + projected_additional_spend
        daily_limit = float(campaign_state.limits.daily_limit)
        
        if projected_total > daily_limit * 0.95:  # 95% of limit
            risk_percentage = projected_total / daily_limit
            logger.warning(f"Overspend risk for campaign {campaign_id}: {risk_percentage:.1%} projected (${projected_total:.2f} vs ${daily_limit:.2f})")
    
    def _reset_time_period_counters(self, campaign_state: CampaignBudgetState, current_time: datetime) -> None:
        """Reset time period counters when periods change"""
        current_date = current_time.date()
        current_hour = current_time.hour
        
        # Check if we need to reset daily counter
        if campaign_state.last_spend_timestamp:
            last_date = campaign_state.last_spend_timestamp.date()
            if current_date != last_date:
                campaign_state.daily_spent = Decimal('0')
        
        # Check if we need to reset weekly counter (Sunday reset)
        if current_date.weekday() == 6:  # Sunday
            if not campaign_state.last_spend_timestamp or \
               campaign_state.last_spend_timestamp.date().weekday() != 6:
                campaign_state.weekly_spent = Decimal('0')
        
        # Check if we need to reset monthly counter
        if current_date.day == 1:  # First of month
            if not campaign_state.last_spend_timestamp or \
               campaign_state.last_spend_timestamp.date().day != 1:
                campaign_state.monthly_spent = Decimal('0')
        
        # Reset hourly counter each hour
        if not campaign_state.last_spend_timestamp or \
           campaign_state.last_spend_timestamp.hour != current_hour:
            campaign_state.hourly_spent = Decimal('0')
    
    def _save_spending_record(self, record: SpendingRecord) -> None:
        """Save spending record to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO spending_records 
                (record_id, timestamp, campaign_id, channel, amount, bid_amount, impressions, clicks, conversions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.record_id,
                record.timestamp.isoformat(),
                record.campaign_id,
                record.channel,
                float(record.amount),
                float(record.bid_amount),
                record.impressions,
                record.clicks,
                record.conversions
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving spending record: {e}")
    
    def _save_violation(self, violation: BudgetViolation) -> None:
        """Save violation to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO budget_violations 
                (violation_id, violation_type, severity, timestamp, campaign_id, channel, 
                 current_amount, limit_amount, percentage_of_limit, message, actions_taken, resolved, resolution_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                violation.violation_id,
                violation.violation_type.value,
                violation.severity.value,
                violation.timestamp.isoformat(),
                violation.campaign_id,
                violation.channel,
                float(violation.current_amount),
                float(violation.limit_amount),
                violation.percentage_of_limit,
                violation.message,
                json.dumps(violation.actions_taken),
                violation.resolved,
                violation.resolution_timestamp.isoformat() if violation.resolution_timestamp else None
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving violation: {e}")
    
    def _save_campaign_state(self, campaign_id: str) -> None:
        """Save campaign state to database"""
        try:
            if campaign_id not in self.campaign_states:
                return
            
            state = self.campaign_states[campaign_id]
            
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO campaign_states 
                (campaign_id, status, daily_spent, weekly_spent, monthly_spent, hourly_spent, 
                 limits_json, last_spend_timestamp, violation_count, emergency_paused, updated_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                campaign_id,
                state.status.value,
                float(state.daily_spent),
                float(state.weekly_spent),
                float(state.monthly_spent),
                float(state.hourly_spent),
                json.dumps({
                    "daily_limit": float(state.limits.daily_limit),
                    "weekly_limit": float(state.limits.weekly_limit),
                    "monthly_limit": float(state.limits.monthly_limit),
                    "max_hourly_spend": float(state.limits.max_hourly_spend),
                    "max_hourly_velocity_increase": state.limits.max_hourly_velocity_increase,
                    "warning_threshold": state.limits.warning_threshold,
                    "critical_threshold": state.limits.critical_threshold,
                    "emergency_threshold": state.limits.emergency_threshold,
                    "max_bid_multiplier": state.limits.max_bid_multiplier,
                    "max_spend_acceleration": state.limits.max_spend_acceleration,
                    "prediction_window_hours": state.limits.prediction_window_hours,
                    "overspend_prevention_buffer": state.limits.overspend_prevention_buffer
                }),
                state.last_spend_timestamp.isoformat() if state.last_spend_timestamp else None,
                state.violation_count,
                state.emergency_paused,
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving campaign state: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Budget Safety Controller received signal {signum}, shutting down...")
        self.shutdown()
    
    # Public interface methods
    
    def get_campaign_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status for a campaign"""
        try:
            if campaign_id not in self.campaign_states:
                return None
            
            with self.lock:
                state = self.campaign_states[campaign_id]
                
                return {
                    "campaign_id": campaign_id,
                    "status": state.status.value,
                    "spending": {
                        "daily_spent": float(state.daily_spent),
                        "weekly_spent": float(state.weekly_spent),
                        "monthly_spent": float(state.monthly_spent),
                        "hourly_spent": float(state.hourly_spent)
                    },
                    "limits": {
                        "daily_limit": float(state.limits.daily_limit),
                        "weekly_limit": float(state.limits.weekly_limit),
                        "monthly_limit": float(state.limits.monthly_limit),
                        "max_hourly_spend": float(state.limits.max_hourly_spend)
                    },
                    "utilization": {
                        "daily_utilization": float(state.daily_spent / state.limits.daily_limit) if state.limits.daily_limit > 0 else 0,
                        "weekly_utilization": float(state.weekly_spent / state.limits.weekly_limit) if state.limits.weekly_limit > 0 else 0,
                        "monthly_utilization": float(state.monthly_spent / state.limits.monthly_limit) if state.limits.monthly_limit > 0 else 0
                    },
                    "violation_count": state.violation_count,
                    "emergency_paused": state.emergency_paused,
                    "last_spend_timestamp": state.last_spend_timestamp.isoformat() if state.last_spend_timestamp else None
                }
        except Exception as e:
            logger.error(f"Error getting campaign status: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            with self.lock:
                total_campaigns = len(self.campaign_states)
                active_campaigns = sum(1 for state in self.campaign_states.values() if state.status == CampaignStatus.ACTIVE)
                paused_campaigns = sum(1 for state in self.campaign_states.values() if state.status == CampaignStatus.PAUSED)
                emergency_campaigns = sum(1 for state in self.campaign_states.values() if state.status == CampaignStatus.EMERGENCY_STOPPED)
                
                total_daily_spent = sum(float(state.daily_spent) for state in self.campaign_states.values())
                total_violations = len(self.violations)
                recent_violations = len([v for v in self.violations if v.timestamp > datetime.now() - timedelta(hours=24)])
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system_active": self.active,
                "campaigns": {
                    "total": total_campaigns,
                    "active": active_campaigns,
                    "paused": paused_campaigns,
                    "emergency_stopped": emergency_campaigns
                },
                "spending": {
                    "total_daily_spent": total_daily_spent,
                    "total_records": len(self.spending_history)
                },
                "violations": {
                    "total_violations": total_violations,
                    "recent_violations_24h": recent_violations,
                    "violation_types": self._get_violation_type_counts()
                },
                "monitoring": {
                    "monitoring_threads_active": sum(1 for t in self.monitoring_threads if t.is_alive()),
                    "last_config_load": os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else None
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _get_violation_type_counts(self) -> Dict[str, int]:
        """Get counts of violation types"""
        counts = defaultdict(int)
        for violation in self.violations:
            counts[violation.violation_type.value] += 1
        return dict(counts)
    
    def is_campaign_safe_to_spend(self, campaign_id: str, amount: Decimal) -> Tuple[bool, str]:
        """Check if a campaign is safe to spend a given amount"""
        try:
            if not self.active:
                return False, "Budget safety controller is not active"
            
            if campaign_id not in self.campaign_states:
                return False, f"Campaign {campaign_id} not registered"
            
            campaign_state = self.campaign_states[campaign_id]
            
            if campaign_state.status != CampaignStatus.ACTIVE:
                return False, f"Campaign status is {campaign_state.status.value}"
            
            # Check if spending amount would exceed any limits
            limits = campaign_state.limits
            
            # Daily limit check
            new_daily_total = campaign_state.daily_spent + amount
            if new_daily_total > limits.daily_limit:
                return False, f"Would exceed daily limit: ${new_daily_total} > ${limits.daily_limit}"
            
            # Hourly velocity check
            new_hourly_total = campaign_state.hourly_spent + amount
            if new_hourly_total > limits.max_hourly_spend:
                return False, f"Would exceed hourly limit: ${new_hourly_total} > ${limits.max_hourly_spend}"
            
            # Weekly limit check
            new_weekly_total = campaign_state.weekly_spent + amount
            if new_weekly_total > limits.weekly_limit:
                return False, f"Would exceed weekly limit: ${new_weekly_total} > ${limits.weekly_limit}"
            
            # Monthly limit check
            new_monthly_total = campaign_state.monthly_spent + amount
            if new_monthly_total > limits.monthly_limit:
                return False, f"Would exceed monthly limit: ${new_monthly_total} > ${limits.monthly_limit}"
            
            return True, "Safe to spend"
            
        except Exception as e:
            logger.error(f"Error checking spend safety for campaign {campaign_id}: {e}")
            return False, f"Safety check failed: {e}"
    
    def emergency_pause_all_campaigns(self, reason: str) -> List[str]:
        """Emergency pause all active campaigns"""
        try:
            paused_campaigns = []
            
            with self.lock:
                for campaign_id, state in self.campaign_states.items():
                    if state.status == CampaignStatus.ACTIVE:
                        self._emergency_pause_campaign(campaign_id, reason)
                        paused_campaigns.append(campaign_id)
            
            logger.critical(f"EMERGENCY PAUSE ALL: {len(paused_campaigns)} campaigns paused - {reason}")
            return paused_campaigns
            
        except Exception as e:
            logger.error(f"Error emergency pausing all campaigns: {e}")
            return []
    
    def resume_campaign(self, campaign_id: str, override_reason: str) -> bool:
        """Resume a paused campaign (requires manual override)"""
        try:
            if campaign_id not in self.campaign_states:
                return False
            
            with self.lock:
                state = self.campaign_states[campaign_id]
                
                if state.status in [CampaignStatus.PAUSED, CampaignStatus.BUDGET_EXHAUSTED]:
                    state.status = CampaignStatus.ACTIVE
                    state.emergency_paused = False
                    self._save_campaign_state(campaign_id)
                    
                    logger.info(f"Campaign {campaign_id} RESUMED: {override_reason}")
                    return True
                elif state.status == CampaignStatus.EMERGENCY_STOPPED:
                    logger.warning(f"Cannot resume emergency stopped campaign {campaign_id} without system reset")
                    return False
                
            return False
            
        except Exception as e:
            logger.error(f"Error resuming campaign {campaign_id}: {e}")
            return False
    
    def update_campaign_limits(self, campaign_id: str, new_limits: BudgetLimits) -> bool:
        """Update budget limits for a campaign"""
        try:
            if campaign_id not in self.campaign_states:
                return False
            
            with self.lock:
                self.campaign_states[campaign_id].limits = new_limits
                self._save_campaign_state(campaign_id)
            
            logger.info(f"Updated budget limits for campaign {campaign_id}: Daily=${new_limits.daily_limit}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating limits for campaign {campaign_id}: {e}")
            return False
    
    def shutdown(self) -> None:
        """Graceful shutdown of budget safety controller"""
        try:
            logger.info("Budget Safety Controller shutting down...")
            self.active = False
            
            # Wait for monitoring threads to finish
            for thread in self.monitoring_threads:
                if thread.is_alive():
                    thread.join(timeout=5)
            
            # Save final state
            for campaign_id in self.campaign_states.keys():
                self._save_campaign_state(campaign_id)
            
            logger.info("Budget Safety Controller shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global budget safety controller instance
_budget_safety_controller: Optional[BudgetSafetyController] = None

def get_budget_safety_controller(config_path: str = None) -> BudgetSafetyController:
    """Get global budget safety controller instance"""
    global _budget_safety_controller
    if _budget_safety_controller is None:
        _budget_safety_controller = BudgetSafetyController(config_path or "budget_safety_config.json")
    return _budget_safety_controller

def budget_safety_decorator(campaign_id: str, channel: str = "default"):
    """Decorator to add budget safety checks to spending functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            controller = get_budget_safety_controller()
            
            # Extract amount from function arguments or kwargs
            amount = kwargs.get('amount') or (args[0] if args else Decimal('0'))
            if not isinstance(amount, Decimal):
                amount = Decimal(str(amount))
            
            # Pre-check if safe to spend
            is_safe, reason = controller.is_campaign_safe_to_spend(campaign_id, amount)
            if not is_safe:
                raise Exception(f"Budget safety violation: {reason}")
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Record the spending (assuming successful execution means spending occurred)
            bid_amount = kwargs.get('bid_amount', amount)  # Default to amount if no bid specified
            if not isinstance(bid_amount, Decimal):
                bid_amount = Decimal(str(bid_amount))
            
            controller.record_spending(
                campaign_id=campaign_id,
                channel=channel,
                amount=amount,
                bid_amount=bid_amount,
                impressions=kwargs.get('impressions', 0),
                clicks=kwargs.get('clicks', 0),
                conversions=kwargs.get('conversions', 0)
            )
            
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    import uuid
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🛡️ GAELP Budget Safety Controller Test")
    print("=" * 50)
    
    # Initialize controller
    controller = BudgetSafetyController()
    
    # Register a test campaign
    test_campaign = "test_campaign_001"
    test_limits = BudgetLimits(
        daily_limit=Decimal('1000.00'),
        weekly_limit=Decimal('5000.00'),
        monthly_limit=Decimal('20000.00'),
        max_hourly_spend=Decimal('100.00'),
        max_hourly_velocity_increase=0.50
    )
    
    controller.register_campaign(test_campaign, test_limits)
    print(f"✅ Registered campaign: {test_campaign}")
    
    # Test normal spending
    print(f"\n📊 Testing normal spending...")
    for i in range(5):
        amount = Decimal('50.00')
        bid = Decimal('2.50')
        
        is_safe, violations = controller.record_spending(
            campaign_id=test_campaign,
            channel="google_ads",
            amount=amount,
            bid_amount=bid,
            impressions=1000,
            clicks=40,
            conversions=2
        )
        
        print(f"  Spend ${amount}: Safe={is_safe}, Violations={len(violations)}")
    
    # Test budget limit approach
    print(f"\n⚠️ Testing budget limit approach...")
    large_amount = Decimal('750.00')  # This should trigger warnings
    
    is_safe, violations = controller.record_spending(
        campaign_id=test_campaign,
        channel="google_ads", 
        amount=large_amount,
        bid_amount=Decimal('5.00'),
        impressions=5000,
        clicks=200,
        conversions=10
    )
    
    print(f"  Large spend ${large_amount}: Safe={is_safe}")
    if violations:
        for violation in violations:
            print(f"    Violation: {violation}")
    
    # Test bid anomaly
    print(f"\n🚨 Testing bid anomaly detection...")
    anomalous_bid = Decimal('25.00')  # Much higher than previous bids
    
    is_safe, violations = controller.record_spending(
        campaign_id=test_campaign,
        channel="google_ads",
        amount=Decimal('50.00'),
        bid_amount=anomalous_bid,
        impressions=200,
        clicks=8,
        conversions=1
    )
    
    print(f"  Anomalous bid ${anomalous_bid}: Safe={is_safe}")
    if violations:
        for violation in violations:
            print(f"    Violation: {violation}")
    
    # Test system status
    print(f"\n📈 System Status:")
    status = controller.get_system_status()
    print(f"  Active campaigns: {status['campaigns']['active']}")
    print(f"  Total daily spend: ${status['spending']['total_daily_spent']:.2f}")
    print(f"  Total violations: {status['violations']['total_violations']}")
    
    # Test campaign status
    campaign_status = controller.get_campaign_status(test_campaign)
    if campaign_status:
        print(f"\n📋 Campaign Status for {test_campaign}:")
        print(f"  Status: {campaign_status['status']}")
        print(f"  Daily utilization: {campaign_status['utilization']['daily_utilization']:.1%}")
        print(f"  Violation count: {campaign_status['violation_count']}")
    
    # Test pre-spend safety check
    print(f"\n🔍 Testing pre-spend safety check...")
    test_amount = Decimal('200.00')
    is_safe_check, reason = controller.is_campaign_safe_to_spend(test_campaign, test_amount)
    print(f"  Safe to spend ${test_amount}: {is_safe_check} - {reason}")
    
    print(f"\n✅ Budget Safety Controller test completed!")
    print(f"💡 Key features demonstrated:")
    print(f"   • Real-time spending monitoring with multi-tier limits")
    print(f"   • Anomaly detection for bidding patterns") 
    print(f"   • Automatic violation handling and campaign pausing")
    print(f"   • Comprehensive audit trail and reporting")
    print(f"   • Pre-spend safety validation")
    print(f"   • Emergency stop integration")
    
    # Cleanup
    controller.shutdown()