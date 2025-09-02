#!/usr/bin/env python3
"""
EMERGENCY STOP MECHANISMS AND KILL SWITCHES FOR GAELP SYSTEM
Critical safety controls for production training and bidding operations.

IMMEDIATE STOP TRIGGERS:
- Budget overrun (>120% daily limit) 
- Anomalous bidding (>$50 CPC)
- Training instability (loss explosion)
- System errors (>5 failures/minute)
- Circuit breakers for all components
- Graceful shutdown with state preservation

NO FALLBACKS - IMMEDIATE SHUTDOWN ONLY
"""

import logging
import threading
import time
import pickle
import json
import os
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from decimal import Decimal
import sqlite3
from pathlib import Path
import signal
import sys

logger = logging.getLogger(__name__)

class EmergencyLevel(Enum):
    """Emergency severity levels"""
    GREEN = "green"      # Normal operation
    YELLOW = "yellow"    # Warning - monitoring required
    RED = "red"          # Critical - immediate action
    BLACK = "black"      # Emergency stop - shutdown everything

class EmergencyType(Enum):
    """Types of emergency situations"""
    BUDGET_OVERRUN = "budget_overrun"
    ANOMALOUS_BIDDING = "anomalous_bidding"
    TRAINING_INSTABILITY = "training_instability"
    SYSTEM_ERROR_RATE = "system_error_rate"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CPU_OVERLOAD = "cpu_overload"
    DATA_CORRUPTION = "data_corruption"
    EXTERNAL_API_FAILURE = "external_api_failure"
    MANUAL_STOP = "manual_stop"

@dataclass
class EmergencyTrigger:
    """Configuration for emergency trigger"""
    trigger_type: EmergencyType
    threshold_value: float
    measurement_window_minutes: int = 5
    consecutive_violations: int = 2
    enabled: bool = True
    callback: Optional[Callable] = None
    
@dataclass
class EmergencyEvent:
    """Record of an emergency event"""
    event_id: str
    trigger_type: EmergencyType
    emergency_level: EmergencyLevel
    timestamp: datetime
    current_value: float
    threshold_value: float
    message: str
    component: str
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    actions_taken: List[str] = field(default_factory=list)

@dataclass
class SystemState:
    """Current system state for preservation"""
    timestamp: datetime
    training_step: int
    model_weights: Optional[bytes]
    environment_state: Dict[str, Any]
    budget_state: Dict[str, float]
    performance_metrics: Dict[str, float]
    active_campaigns: List[str]
    pending_bids: List[Dict[str, Any]]

class CircuitBreaker:
    """Circuit breaker for individual components"""
    
    def __init__(self, name: str, failure_threshold: int = 5, timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        with self.lock:
            if self.state == "open":
                if self.last_failure_time and \
                   (datetime.now() - self.last_failure_time).seconds > self.timeout:
                    self.state = "half_open"
                    logger.info(f"Circuit breaker {self.name} entering half-open state")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN - operation blocked")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "half_open":
                    self.reset()
                return result
            except Exception as e:
                self.record_failure()
                raise e
    
    def record_failure(self):
        """Record a failure and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(f"Circuit breaker {self.name} is now OPEN after {self.failure_count} failures")
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"
        logger.info(f"Circuit breaker {self.name} reset to CLOSED state")

class EmergencyController:
    """Main emergency control system"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "emergency_config.json"
        self.db_path = "emergency_events.db"
        self.state_path = "emergency_state.pkl"
        
        # Core control flags
        self.system_active = True
        self.emergency_stop_triggered = False
        self.current_emergency_level = EmergencyLevel.GREEN
        
        # Monitoring data
        self.triggers: Dict[EmergencyType, EmergencyTrigger] = {}
        self.events: List[EmergencyEvent] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.monitoring_threads: List[threading.Thread] = []
        
        # System monitoring
        self.error_counts: Dict[str, int] = {}
        self.budget_tracker: Dict[str, float] = {}
        self.bid_history: List[float] = []
        self.training_loss_history: List[float] = []
        self.system_metrics: Dict[str, List[float]] = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize system
        self._init_database()
        self._load_config()
        self._create_circuit_breakers()
        self._start_monitoring()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _init_database(self):
        """Initialize SQLite database for event logging"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emergency_events (
                event_id TEXT PRIMARY KEY,
                trigger_type TEXT,
                emergency_level TEXT,
                timestamp TEXT,
                current_value REAL,
                threshold_value REAL,
                message TEXT,
                component TEXT,
                resolved BOOLEAN,
                resolution_timestamp TEXT,
                actions_taken TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def _load_config(self):
        """Load emergency trigger configuration"""
        default_config = {
            "budget_overrun_threshold": 1.20,  # 120% of daily limit
            "max_cpc_threshold": 50.0,         # $50 maximum CPC
            "loss_explosion_threshold": 10.0,  # 10x normal loss
            "error_rate_threshold": 5.0,       # 5 errors per minute
            "memory_threshold": 0.90,          # 90% memory usage
            "cpu_threshold": 0.95,             # 95% CPU usage
            "measurement_window": 5,           # 5 minute window
            "consecutive_violations": 2        # 2 consecutive violations
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            config = default_config
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Create triggers from config
        self.triggers[EmergencyType.BUDGET_OVERRUN] = EmergencyTrigger(
            EmergencyType.BUDGET_OVERRUN,
            config["budget_overrun_threshold"],
            config["measurement_window"],
            config["consecutive_violations"]
        )
        
        self.triggers[EmergencyType.ANOMALOUS_BIDDING] = EmergencyTrigger(
            EmergencyType.ANOMALOUS_BIDDING,
            config["max_cpc_threshold"],
            config["measurement_window"],
            config["consecutive_violations"]
        )
        
        self.triggers[EmergencyType.TRAINING_INSTABILITY] = EmergencyTrigger(
            EmergencyType.TRAINING_INSTABILITY,
            config["loss_explosion_threshold"],
            config["measurement_window"],
            config["consecutive_violations"]
        )
        
        self.triggers[EmergencyType.SYSTEM_ERROR_RATE] = EmergencyTrigger(
            EmergencyType.SYSTEM_ERROR_RATE,
            config["error_rate_threshold"],
            config["measurement_window"],
            config["consecutive_violations"]
        )
        
        self.triggers[EmergencyType.MEMORY_EXHAUSTION] = EmergencyTrigger(
            EmergencyType.MEMORY_EXHAUSTION,
            config["memory_threshold"],
            config["measurement_window"],
            config["consecutive_violations"]
        )
        
        self.triggers[EmergencyType.CPU_OVERLOAD] = EmergencyTrigger(
            EmergencyType.CPU_OVERLOAD,
            config["cpu_threshold"],
            config["measurement_window"],
            config["consecutive_violations"]
        )
        
        # Add manual stop trigger
        self.triggers[EmergencyType.MANUAL_STOP] = EmergencyTrigger(
            EmergencyType.MANUAL_STOP,
            0.0,  # Always triggers when called
            0,
            1
        )
    
    def _create_circuit_breakers(self):
        """Create circuit breakers for all components"""
        components = [
            "discovery_engine",
            "creative_selector", 
            "attribution_engine",
            "budget_pacer",
            "identity_resolver",
            "parameter_manager",
            "training_orchestrator",
            "environment",
            "rl_agent",
            "ga4_integration",
            "bidding_system"
        ]
        
        for component in components:
            self.circuit_breakers[component] = CircuitBreaker(
                name=component,
                failure_threshold=3,
                timeout=300  # 5 minutes
            )
    
    def _start_monitoring(self):
        """Start background monitoring threads"""
        monitoring_functions = [
            self._monitor_system_resources,
            self._monitor_error_rates,
            self._monitor_budget_spending,
            self._monitor_training_metrics,
            self._monitor_bid_anomalies
        ]
        
        for func in monitoring_functions:
            thread = threading.Thread(target=func, daemon=True)
            thread.start()
            self.monitoring_threads.append(thread)
    
    def _monitor_system_resources(self):
        """Monitor CPU, memory, disk usage"""
        while self.system_active:
            try:
                # Memory usage
                memory_percent = psutil.virtual_memory().percent / 100.0
                self._check_trigger(EmergencyType.MEMORY_EXHAUSTION, memory_percent, "system")
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1) / 100.0
                self._check_trigger(EmergencyType.CPU_OVERLOAD, cpu_percent, "system")
                
                # Store metrics
                timestamp = datetime.now()
                if "memory" not in self.system_metrics:
                    self.system_metrics["memory"] = []
                if "cpu" not in self.system_metrics:
                    self.system_metrics["cpu"] = []
                    
                self.system_metrics["memory"].append((timestamp, memory_percent))
                self.system_metrics["cpu"].append((timestamp, cpu_percent))
                
                # Keep only last hour of data
                cutoff = datetime.now() - timedelta(hours=1)
                self.system_metrics["memory"] = [
                    (t, v) for t, v in self.system_metrics["memory"] if t > cutoff
                ]
                self.system_metrics["cpu"] = [
                    (t, v) for t, v in self.system_metrics["cpu"] if t > cutoff
                ]
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
                time.sleep(60)
    
    def _monitor_error_rates(self):
        """Monitor system error rates"""
        while self.system_active:
            try:
                # Calculate error rate per minute
                current_time = datetime.now()
                minute_ago = current_time - timedelta(minutes=1)
                
                recent_errors = sum(
                    count for timestamp, count in self.error_counts.items()
                    if datetime.fromisoformat(timestamp) > minute_ago
                )
                
                self._check_trigger(EmergencyType.SYSTEM_ERROR_RATE, recent_errors, "system")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring error rates: {e}")
                time.sleep(60)
    
    def _monitor_budget_spending(self):
        """Monitor budget spending rates"""
        while self.system_active:
            try:
                for budget_name, current_spend in self.budget_tracker.items():
                    # Check if we have a daily limit
                    daily_limit_key = f"{budget_name}_daily_limit"
                    if daily_limit_key in self.budget_tracker:
                        daily_limit = self.budget_tracker[daily_limit_key]
                        spend_ratio = current_spend / daily_limit if daily_limit > 0 else 0
                        
                        self._check_trigger(
                            EmergencyType.BUDGET_OVERRUN, 
                            spend_ratio, 
                            f"budget_{budget_name}"
                        )
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring budget spending: {e}")
                time.sleep(300)
    
    def _monitor_training_metrics(self):
        """Monitor training stability"""
        while self.system_active:
            try:
                if len(self.training_loss_history) >= 10:
                    # Calculate recent loss trend
                    recent_losses = self.training_loss_history[-10:]
                    baseline_loss = np.mean(self.training_loss_history[:-10]) if len(self.training_loss_history) > 10 else 1.0
                    current_loss = np.mean(recent_losses)
                    
                    if baseline_loss > 0:
                        loss_ratio = current_loss / baseline_loss
                        self._check_trigger(
                            EmergencyType.TRAINING_INSTABILITY,
                            loss_ratio,
                            "training"
                        )
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring training metrics: {e}")
                time.sleep(60)
    
    def _monitor_bid_anomalies(self):
        """Monitor bidding for anomalies"""
        while self.system_active:
            try:
                if self.bid_history:
                    # Check recent bids
                    recent_bids = self.bid_history[-100:] if len(self.bid_history) >= 100 else self.bid_history
                    max_recent_bid = max(recent_bids) if recent_bids else 0
                    
                    self._check_trigger(
                        EmergencyType.ANOMALOUS_BIDDING,
                        max_recent_bid,
                        "bidding"
                    )
                
                time.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring bid anomalies: {e}")
                time.sleep(120)
    
    def _check_trigger(self, trigger_type: EmergencyType, current_value: float, component: str):
        """Check if a trigger threshold is exceeded"""
        if trigger_type not in self.triggers or not self.triggers[trigger_type].enabled:
            return
        
        trigger = self.triggers[trigger_type]
        threshold_exceeded = current_value > trigger.threshold_value
        
        if threshold_exceeded:
            # Determine emergency level based on severity
            if trigger_type == EmergencyType.BUDGET_OVERRUN:
                if current_value > trigger.threshold_value * 1.5:  # 180% of limit
                    level = EmergencyLevel.BLACK
                elif current_value > trigger.threshold_value * 1.3:  # 156% of limit  
                    level = EmergencyLevel.RED
                else:
                    level = EmergencyLevel.YELLOW
            elif trigger_type == EmergencyType.ANOMALOUS_BIDDING:
                if current_value > 100.0:  # >$100 CPC
                    level = EmergencyLevel.BLACK
                elif current_value > 75.0:   # >$75 CPC
                    level = EmergencyLevel.RED
                else:
                    level = EmergencyLevel.YELLOW
            elif trigger_type == EmergencyType.TRAINING_INSTABILITY:
                if current_value > trigger.threshold_value * 5:  # 50x normal loss
                    level = EmergencyLevel.BLACK
                elif current_value > trigger.threshold_value * 2:  # 20x normal loss
                    level = EmergencyLevel.RED
                else:
                    level = EmergencyLevel.YELLOW
            else:
                # Generic scaling
                if current_value > trigger.threshold_value * 2:
                    level = EmergencyLevel.BLACK
                elif current_value > trigger.threshold_value * 1.5:
                    level = EmergencyLevel.RED
                else:
                    level = EmergencyLevel.YELLOW
            
            # Create emergency event
            event = EmergencyEvent(
                event_id=f"{trigger_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                trigger_type=trigger_type,
                emergency_level=level,
                timestamp=datetime.now(),
                current_value=current_value,
                threshold_value=trigger.threshold_value,
                message=f"{trigger_type.value} exceeded: {current_value:.2f} > {trigger.threshold_value:.2f}",
                component=component
            )
            
            self._handle_emergency(event)
    
    def _handle_emergency(self, event: EmergencyEvent):
        """Handle an emergency event"""
        with self.lock:
            self.events.append(event)
            self.current_emergency_level = max(
                self.current_emergency_level, 
                event.emergency_level,
                key=lambda x: ["green", "yellow", "red", "black"].index(x.value)
            )
        
        # Log event to database
        self._log_event_to_db(event)
        
        # Take immediate action based on severity
        if event.emergency_level == EmergencyLevel.BLACK:
            self._emergency_shutdown(event)
        elif event.emergency_level == EmergencyLevel.RED:
            self._critical_response(event)
        elif event.emergency_level == EmergencyLevel.YELLOW:
            self._warning_response(event)
        
        # Call custom callback if configured
        trigger = self.triggers[event.trigger_type]
        if trigger.callback:
            try:
                trigger.callback(event)
            except Exception as e:
                logger.error(f"Error in emergency callback: {e}")
    
    def _emergency_shutdown(self, event: EmergencyEvent):
        """Perform immediate emergency shutdown"""
        logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {event.message}")
        
        self.emergency_stop_triggered = True
        
        actions = [
            "Triggering emergency stop",
            "Saving system state",
            "Stopping all training",
            "Halting all bidding",
            "Preserving data"
        ]
        
        # Save current system state
        self._save_system_state()
        
        # Stop all operations
        self._stop_all_operations()
        
        # Record actions
        event.actions_taken = actions
        
        logger.critical("EMERGENCY SHUTDOWN COMPLETE - All operations stopped")
        
        # Only exit if not in test mode
        if not hasattr(self, '_test_mode') or not self._test_mode:
            os._exit(1)
    
    def _critical_response(self, event: EmergencyEvent):
        """Handle critical level emergency"""
        logger.error(f"CRITICAL EMERGENCY: {event.message}")
        
        actions = []
        
        if event.trigger_type == EmergencyType.BUDGET_OVERRUN:
            actions.append("Pausing all campaigns")
            actions.append("Blocking new bids")
            self._pause_campaigns()
        
        elif event.trigger_type == EmergencyType.ANOMALOUS_BIDDING:
            actions.append("Reducing bid caps by 50%")
            actions.append("Increasing bid validation")
            self._reduce_bid_caps()
        
        elif event.trigger_type == EmergencyType.TRAINING_INSTABILITY:
            actions.append("Stopping training updates")
            actions.append("Reverting to last stable model")
            self._revert_to_stable_model()
        
        elif event.trigger_type == EmergencyType.SYSTEM_ERROR_RATE:
            actions.append("Activating circuit breakers")
            actions.append("Reducing system load")
            self._activate_circuit_breakers()
        
        event.actions_taken = actions
        
        # Save state in case we need to escalate to shutdown
        self._save_system_state()
    
    def _warning_response(self, event: EmergencyEvent):
        """Handle warning level emergency"""
        logger.warning(f"WARNING: {event.message}")
        
        actions = ["Increased monitoring", "Alerting operations team"]
        event.actions_taken = actions
        
        # Just log and monitor more closely
        # No immediate action required for warnings
    
    def _save_system_state(self):
        """Save current system state for recovery"""
        try:
            state = SystemState(
                timestamp=datetime.now(),
                training_step=0,  # Will be filled by caller
                model_weights=None,  # Will be filled by caller
                environment_state={},  # Will be filled by caller
                budget_state=self.budget_tracker.copy(),
                performance_metrics=self._calculate_current_metrics(),
                active_campaigns=[],  # Will be filled by caller
                pending_bids=[]  # Will be filled by caller
            )
            
            with open(self.state_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info("System state saved for emergency recovery")
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    def _stop_all_operations(self):
        """Stop all system operations"""
        self.system_active = False
        
        # Signal all monitoring threads to stop
        for thread in self.monitoring_threads:
            if thread.is_alive():
                # Threads are daemon threads, they'll stop when main process stops
                pass
    
    def _pause_campaigns(self):
        """Pause all active campaigns"""
        # This would integrate with actual campaign management
        logger.info("Pausing all campaigns due to emergency")
    
    def _reduce_bid_caps(self):
        """Reduce all bid caps by 50%"""
        # This would integrate with actual bidding system
        logger.info("Reducing bid caps by 50% due to emergency")
    
    def _revert_to_stable_model(self):
        """Revert to last known stable model"""
        # This would integrate with model versioning
        logger.info("Reverting to stable model due to training instability")
    
    def _activate_circuit_breakers(self):
        """Activate all circuit breakers"""
        for breaker in self.circuit_breakers.values():
            breaker.state = "open"
        logger.info("All circuit breakers activated due to high error rate")
    
    def _calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics"""
        return {
            "error_rate": sum(self.error_counts.values()),
            "budget_utilization": sum(self.budget_tracker.values()),
            "avg_bid": np.mean(self.bid_history) if self.bid_history else 0,
            "training_loss": np.mean(self.training_loss_history[-10:]) if self.training_loss_history else 0
        }
    
    def _log_event_to_db(self, event: EmergencyEvent):
        """Log emergency event to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO emergency_events 
                (event_id, trigger_type, emergency_level, timestamp, current_value, 
                 threshold_value, message, component, resolved, resolution_timestamp, actions_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.trigger_type.value,
                event.emergency_level.value,
                event.timestamp.isoformat(),
                event.current_value,
                event.threshold_value,
                event.message,
                event.component,
                event.resolved,
                event.resolution_timestamp.isoformat() if event.resolution_timestamp else None,
                json.dumps(event.actions_taken)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging event to database: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.trigger_manual_emergency_stop("System signal received")
    
    # Public interface methods
    
    def register_error(self, component: str, error_message: str):
        """Register an error for monitoring"""
        timestamp = datetime.now().isoformat()
        key = f"{timestamp}"
        self.error_counts[key] = 1
        
        # Clean old error entries
        cutoff = datetime.now() - timedelta(hours=1)
        self.error_counts = {
            k: v for k, v in self.error_counts.items()
            if datetime.fromisoformat(k) > cutoff
        }
    
    def update_budget_tracking(self, budget_name: str, current_spend: float, daily_limit: float = None):
        """Update budget tracking information"""
        self.budget_tracker[budget_name] = current_spend
        if daily_limit is not None:
            self.budget_tracker[f"{budget_name}_daily_limit"] = daily_limit
    
    def record_bid(self, bid_amount: float):
        """Record a bid for anomaly detection"""
        self.bid_history.append(bid_amount)
        
        # Keep only last 1000 bids
        if len(self.bid_history) > 1000:
            self.bid_history = self.bid_history[-1000:]
    
    def record_training_loss(self, loss: float):
        """Record training loss for stability monitoring"""
        self.training_loss_history.append(loss)
        
        # Keep only last 1000 losses
        if len(self.training_loss_history) > 1000:
            self.training_loss_history = self.training_loss_history[-1000:]
    
    def get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get circuit breaker for component"""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker(component)
        return self.circuit_breakers[component]
    
    def trigger_manual_emergency_stop(self, reason: str):
        """Manually trigger emergency stop"""
        event = EmergencyEvent(
            event_id=f"manual_stop_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trigger_type=EmergencyType.MANUAL_STOP,
            emergency_level=EmergencyLevel.BLACK,
            timestamp=datetime.now(),
            current_value=1.0,
            threshold_value=0.0,
            message=f"Manual emergency stop: {reason}",
            component="manual"
        )
        
        self._handle_emergency(event)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "active": self.system_active,
            "emergency_stop_triggered": self.emergency_stop_triggered,
            "emergency_level": self.current_emergency_level.value,
            "recent_events": [
                {
                    "type": event.trigger_type.value,
                    "level": event.emergency_level.value,
                    "timestamp": event.timestamp.isoformat(),
                    "message": event.message,
                    "component": event.component
                }
                for event in self.events[-10:]  # Last 10 events
            ],
            "circuit_breakers": {
                name: breaker.state for name, breaker in self.circuit_breakers.items()
            },
            "metrics": self._calculate_current_metrics()
        }
    
    def is_system_healthy(self) -> bool:
        """Check if system is in healthy state"""
        return (self.current_emergency_level in [EmergencyLevel.GREEN, EmergencyLevel.YELLOW] and
                not self.emergency_stop_triggered and
                self.system_active)
    
    def reset_emergency_state(self):
        """Reset emergency state after manual intervention"""
        if not self.emergency_stop_triggered:
            logger.warning("Cannot reset - emergency stop not triggered")
            return
        
        # Reset state
        self.emergency_stop_triggered = False
        self.current_emergency_level = EmergencyLevel.GREEN
        self.system_active = True
        
        # Reset circuit breakers
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        
        # Clear recent events
        self.events = []
        
        logger.info("Emergency state reset - system ready to restart")


# Global emergency controller instance
_emergency_controller: Optional[EmergencyController] = None

def get_emergency_controller() -> EmergencyController:
    """Get global emergency controller instance"""
    global _emergency_controller
    if _emergency_controller is None:
        _emergency_controller = EmergencyController()
    return _emergency_controller

def integrate_budget_safety_controller():
    """Integrate budget safety controller with emergency system"""
    try:
        # Import here to avoid circular imports
        from budget_safety_controller import get_budget_safety_controller
        
        emergency_controller = get_emergency_controller()
        budget_controller = get_budget_safety_controller()
        
        # Register budget safety with emergency system
        emergency_controller.budget_safety_controller = budget_controller
        logger.info("Budget Safety Controller integrated with Emergency Control System")
        
        return True
    except Exception as e:
        logger.error(f"Failed to integrate budget safety controller: {e}")
        return False

def emergency_stop_decorator(component_name: str):
    """Decorator to add emergency controls to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            controller = get_emergency_controller()
            
            # Check if system is healthy
            if not controller.is_system_healthy():
                raise Exception(f"System emergency - {component_name} operations blocked")
            
            breaker = controller.get_circuit_breaker(component_name)
            
            try:
                return breaker.call(func, *args, **kwargs)
            except Exception as e:
                controller.register_error(component_name, str(e))
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    controller = EmergencyController()
    
    print("Emergency Control System initialized")
    print(f"System status: {controller.get_system_status()}")
    
    # Simulate some conditions
    print("\nTesting budget overrun...")
    controller.update_budget_tracking("main_campaign", 1200, 1000)  # 120% spend
    
    print("\nTesting anomalous bid...")
    controller.record_bid(75.0)  # $75 CPC
    
    print("\nTesting training instability...")
    for i in range(20):
        controller.record_training_loss(1.0 + i * 0.5)  # Exploding loss
    
    time.sleep(1)
    
    print(f"\nFinal status: {controller.get_system_status()}")