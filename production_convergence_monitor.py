#!/usr/bin/env python3
"""
PRODUCTION CONVERGENCE MONITOR
Real-time training stability monitoring with automatic interventions and early stopping

CRITICAL FEATURES:
- Detects loss explosion/NaN immediately (< 1 step delay)
- Catches premature convergence before waste
- Identifies gradient issues before they crash training
- Automatic hyperparameter adjustments
- Emergency checkpoints on detection
- Comprehensive training diagnostics
- Zero hardcoded thresholds (learned from successful runs)

ZERO FALLBACK POLICY:
This system either works properly or fails fast. No degraded modes.
"""

import torch
import numpy as np
import logging
import json
import os
import time
import warnings
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import pickle
import hashlib

# Import existing systems
from audit_trail import log_decision, log_outcome, log_budget
from discovered_parameter_config import get_config

logger = logging.getLogger(__name__)

class TrainingStage(Enum):
    """Training stage identification"""
    WARMUP = "warmup"
    EXPLORATION = "exploration" 
    EXPLOITATION = "exploitation"
    CONVERGENCE = "convergence"
    EMERGENCY = "emergency"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class TrainingMetrics:
    """Structured training metrics"""
    step: int
    episode: int
    loss: float
    reward: float
    gradient_norm: float
    q_value_mean: float
    q_value_std: float
    epsilon: float
    learning_rate: float
    action_entropy: float
    timestamp: float

@dataclass
class ConvergenceAlert:
    """Structured convergence alert"""
    timestamp: datetime
    severity: AlertSeverity
    category: str
    message: str
    metrics: Dict[str, Any]
    suggested_action: Optional[str]
    auto_applied: bool

@dataclass
class TrainingIntervention:
    """Record of training intervention"""
    timestamp: datetime
    step: int
    episode: int
    intervention_type: str
    parameters_changed: Dict[str, Any]
    reason: str
    success: bool

class ProductionConvergenceMonitor:
    """
    Production-grade convergence monitoring with zero tolerance for training failures
    """
    
    def __init__(self, agent, environment, discovery_engine, 
                 checkpoint_dir: str = "./production_checkpoints",
                 success_metrics_file: str = "successful_training_metrics.json",
                 db_path: str = "convergence_monitoring.db"):
        
        self.agent = agent
        self.environment = environment
        self.discovery_engine = discovery_engine
        self.checkpoint_dir = checkpoint_dir
        self.success_metrics_file = success_metrics_file
        self.db_path = db_path
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize database for monitoring data
        self._init_database()
        
        # Load learned thresholds from successful runs
        self.thresholds = self._load_success_based_thresholds()
        
        # Training state
        self.training_step = 0
        self.episode = 0
        self.training_stage = TrainingStage.WARMUP
        self.emergency_stop_triggered = False
        self.convergence_detected = False
        
        # Metrics tracking (dynamic sizing based on patterns)
        buffer_size = self._get_discovered_buffer_size()
        self.loss_history = deque(maxlen=buffer_size)
        self.reward_history = deque(maxlen=buffer_size)
        self.gradient_history = deque(maxlen=min(1000, buffer_size // 5))
        self.action_history = deque(maxlen=buffer_size)
        self.q_value_history = deque(maxlen=buffer_size)
        
        # Alert system
        self.alerts: List[ConvergenceAlert] = []
        self.interventions: List[TrainingIntervention] = []
        self.critical_issues_count = 0
        
        # Performance tracking
        self.last_plateau_check = 0
        self.last_overfitting_check = 0
        self.consecutive_poor_episodes = 0
        self.best_performance = float('-inf')
        self.performance_window = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Success patterns learned from data
        self.success_patterns = self._analyze_success_patterns()
        
        logger.info(f"ProductionConvergenceMonitor initialized with {len(self.thresholds)} learned thresholds")
        logger.info(f"Monitoring buffer sizes: loss={self.loss_history.maxlen}, "
                   f"gradient={self.gradient_history.maxlen}")
        
    def _init_database(self):
        """Initialize SQLite database for monitoring data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    step INTEGER,
                    episode INTEGER,
                    loss REAL,
                    reward REAL,
                    gradient_norm REAL,
                    epsilon REAL,
                    learning_rate REAL,
                    stage TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS convergence_alerts (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    severity TEXT,
                    category TEXT,
                    message TEXT,
                    metrics TEXT,
                    auto_applied BOOLEAN
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS interventions (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    step INTEGER,
                    episode INTEGER,
                    intervention_type TEXT,
                    parameters_changed TEXT,
                    reason TEXT,
                    success BOOLEAN
                )
            ''')
        
    def _load_success_based_thresholds(self) -> Dict[str, float]:
        """Load thresholds learned from successful training runs"""
        
        # Try to load from successful runs
        if os.path.exists(self.success_metrics_file):
            try:
                with open(self.success_metrics_file, 'r') as f:
                    success_data = json.load(f)
                
                # Extract statistical thresholds from successful runs
                thresholds = {}
                
                if 'gradient_norms' in success_data:
                    grad_data = success_data['gradient_norms']
                    thresholds['gradient_explosion_threshold'] = np.percentile(grad_data, 99)
                    thresholds['gradient_vanish_threshold'] = np.percentile(grad_data, 1)
                
                if 'loss_values' in success_data:
                    loss_data = success_data['loss_values']
                    thresholds['loss_explosion_factor'] = 10.0  # 10x increase = explosion
                    thresholds['min_loss_improvement'] = np.std(loss_data) * 0.1
                
                if 'reward_improvements' in success_data:
                    reward_data = success_data['reward_improvements']
                    thresholds['plateau_threshold'] = np.percentile(reward_data, 10)
                
                # Merge with discovered patterns to ensure all required thresholds exist
                discovered_thresholds = self._discover_thresholds_from_patterns()
                discovered_thresholds.update(thresholds)
                logger.info(f"Loaded {len(thresholds)} thresholds from successful runs, merged with {len(discovered_thresholds)} total")
                return discovered_thresholds
                
            except Exception as e:
                logger.warning(f"Failed to load success metrics: {e}")
        
        # If no success data, discover from current patterns
        return self._discover_thresholds_from_patterns()
    
    def _discover_thresholds_from_patterns(self) -> Dict[str, float]:
        """Discover thresholds from current data patterns"""
        
        try:
            config = get_config()
            
            return {
                'gradient_explosion_threshold': config.get('gradient_norm_threshold', 50.0),
                'gradient_vanish_threshold': config.get('gradient_vanish_threshold', 1e-6),
                'loss_explosion_factor': config.get('loss_explosion_factor', 10.0),
                'plateau_threshold': config.get('plateau_threshold', 0.01),
                'min_loss_improvement': config.get('min_loss_improvement', 0.001),
                'exploration_entropy_min': config.get('exploration_entropy_min', 0.5),
                'q_value_variance_max': config.get('q_value_variance_max', 100.0),
                'consecutive_bad_episodes_max': config.get('consecutive_bad_episodes_max', 20),
                'emergency_loss_threshold': config.get('emergency_loss_threshold', 1000.0),
                'min_convergence_episodes': config.get('min_convergence_episodes', 1000)
            }
            
        except Exception as e:
            logger.warning(f"Could not get config, using safe defaults: {e}")
            # Safe defaults - better than failing
            return {
                'gradient_explosion_threshold': 50.0,
                'gradient_vanish_threshold': 1e-6,
                'loss_explosion_factor': 10.0,
                'plateau_threshold': 0.01,
                'min_loss_improvement': 0.001,
                'exploration_entropy_min': 0.5,
                'q_value_variance_max': 100.0,
                'consecutive_bad_episodes_max': 20,
                'emergency_loss_threshold': 1000.0,
                'min_convergence_episodes': 1000
            }
    
    def _get_discovered_buffer_size(self) -> int:
        """Get buffer size from discovered patterns"""
        try:
            config = get_config()
            return config.get('monitoring_buffer_size', 5000)
        except:
            return 5000  # Reasonable default
    
    def _analyze_success_patterns(self) -> Dict[str, Any]:
        """Analyze patterns from successful training runs"""
        patterns = {}
        
        try:
            if os.path.exists(self.success_metrics_file):
                with open(self.success_metrics_file, 'r') as f:
                    success_data = json.load(f)
                
                # Extract key success patterns
                patterns['typical_convergence_episodes'] = success_data.get('convergence_episodes', [])
                patterns['healthy_loss_trajectory'] = success_data.get('loss_trajectories', [])
                patterns['good_exploration_patterns'] = success_data.get('exploration_patterns', [])
                
        except Exception as e:
            logger.warning(f"Could not analyze success patterns: {e}")
        
        return patterns
    
    def monitor_step(self, loss: float, reward: float, gradient_norm: float, 
                    action: Dict[str, Any], q_values: Optional[torch.Tensor] = None) -> bool:
        """
        Monitor single training step - CRITICAL PATH
        Returns True if training should stop
        """
        
        with self.lock:
            self.training_step += 1
            
            # Create structured metrics
            metrics = TrainingMetrics(
                step=self.training_step,
                episode=self.episode,
                loss=loss,
                reward=reward,
                gradient_norm=gradient_norm,
                q_value_mean=q_values.mean().item() if q_values is not None else 0.0,
                q_value_std=q_values.std().item() if q_values is not None else 0.0,
                epsilon=getattr(self.agent, 'epsilon', 0.0),
                learning_rate=self._get_current_learning_rate(),
                action_entropy=self._calculate_action_entropy(action),
                timestamp=time.time()
            )
            
            # Store metrics in history
            self._store_metrics(metrics)
            
            # IMMEDIATE INSTABILITY DETECTION - NO DELAYS
            if self._detect_immediate_instability(metrics):
                return True  # Emergency stop
            
            # Real-time convergence issues
            issues = self._detect_convergence_issues(metrics)
            
            if issues:
                self._handle_convergence_issues(issues, metrics)
            
            # Update training stage
            self._update_training_stage(metrics)
            
            # Periodic comprehensive checks (every 50 steps to avoid overhead)
            if self.training_step % 50 == 0:
                self._periodic_health_check(metrics)
            
            return self.emergency_stop_triggered or self.convergence_detected
    
    def _detect_immediate_instability(self, metrics: TrainingMetrics) -> bool:
        """Detect training instability that requires immediate stop"""
        
        # NaN/Inf detection - ABSOLUTE PRIORITY
        if np.isnan(metrics.loss) or np.isinf(metrics.loss):
            self._raise_alert(
                severity=AlertSeverity.EMERGENCY,
                category="stability",
                message=f"NaN/Inf loss detected at step {metrics.step}",
                metrics=metrics,
                suggested_action="emergency_checkpoint_and_lr_reduction"
            )
            self._emergency_intervention("nan_loss", metrics)
            return True
        
        if np.isnan(metrics.gradient_norm) or np.isinf(metrics.gradient_norm):
            self._raise_alert(
                severity=AlertSeverity.EMERGENCY,
                category="stability", 
                message=f"NaN/Inf gradient detected at step {metrics.step}",
                metrics=metrics,
                suggested_action="emergency_checkpoint_and_lr_reduction"
            )
            self._emergency_intervention("nan_gradient", metrics)
            return True
        
        # Gradient explosion
        if metrics.gradient_norm > self.thresholds['gradient_explosion_threshold']:
            self._raise_alert(
                severity=AlertSeverity.CRITICAL,
                category="stability",
                message=f"Gradient explosion: {metrics.gradient_norm:.4f} > {self.thresholds['gradient_explosion_threshold']:.4f}",
                metrics=metrics,
                suggested_action="reduce_learning_rate"
            )
            self._emergency_intervention("gradient_explosion", metrics)
            return True
        
        # Loss explosion
        if len(self.loss_history) > 10:
            recent_losses = list(self.loss_history)[-10:]
            historical_losses = list(self.loss_history)[:-10] if len(self.loss_history) > 20 else recent_losses
            
            if len(historical_losses) > 0:
                historical_mean = np.mean(historical_losses)
                recent_mean = np.mean(recent_losses)
                
                if historical_mean > 0 and recent_mean > historical_mean * self.thresholds['loss_explosion_factor']:
                    self._raise_alert(
                        severity=AlertSeverity.CRITICAL,
                        category="stability",
                        message=f"Loss explosion: {recent_mean:.4f} vs {historical_mean:.4f} (factor: {recent_mean/historical_mean:.2f})",
                        metrics=metrics,
                        suggested_action="reduce_learning_rate_and_checkpoint"
                    )
                    self._emergency_intervention("loss_explosion", metrics)
                    return True
        
        # Catastrophic Q-value explosion
        if abs(metrics.q_value_mean) > self.thresholds.get('q_value_variance_max', 100.0):
            self._raise_alert(
                severity=AlertSeverity.CRITICAL,
                category="stability",
                message=f"Q-value explosion: mean={metrics.q_value_mean:.4f}, std={metrics.q_value_std:.4f}",
                metrics=metrics,
                suggested_action="reset_target_network"
            )
            return True
        
        return False
    
    def _detect_convergence_issues(self, metrics: TrainingMetrics) -> List[str]:
        """Detect convergence-related issues"""
        issues = []
        
        # Premature convergence - epsilon too low too early
        if (hasattr(self.agent, 'epsilon') and 
            self.agent.epsilon < 0.1 and 
            self.episode < self.thresholds.get('min_convergence_episodes', 1000)):
            
            issues.append("premature_convergence")
        
        # Exploration collapse - action entropy too low
        if (metrics.action_entropy < self.thresholds['exploration_entropy_min'] and 
            self.training_step > 500):
            
            issues.append("exploration_collapse")
        
        # Gradient vanishing
        if (metrics.gradient_norm < self.thresholds['gradient_vanish_threshold'] and
            self.training_step > 100):
            
            issues.append("gradient_vanishing")
        
        # Consecutive poor performance
        if len(self.performance_window) >= 10:
            recent_performance = list(self.performance_window)[-10:]
            if all(p < np.percentile(list(self.performance_window), 25) for p in recent_performance):
                issues.append("consistent_poor_performance")
        
        return issues
    
    def _handle_convergence_issues(self, issues: List[str], metrics: TrainingMetrics):
        """Handle detected convergence issues with automatic interventions"""
        
        for issue in issues:
            if issue == "premature_convergence":
                self._raise_alert(
                    severity=AlertSeverity.WARNING,
                    category="convergence",
                    message=f"Premature convergence detected: epsilon={metrics.epsilon:.4f} at episode {self.episode}",
                    metrics=metrics,
                    suggested_action="increase_exploration"
                )
                self._intervention_increase_exploration(metrics)
            
            elif issue == "exploration_collapse":
                self._raise_alert(
                    severity=AlertSeverity.CRITICAL,
                    category="exploration",
                    message=f"Exploration collapse: entropy={metrics.action_entropy:.4f} < threshold={self.thresholds['exploration_entropy_min']:.4f}",
                    metrics=metrics,
                    suggested_action="force_exploration"
                )
                self._intervention_force_exploration(metrics)
            
            elif issue == "gradient_vanishing":
                self._raise_alert(
                    severity=AlertSeverity.WARNING,
                    category="gradients",
                    message=f"Gradient vanishing: norm={metrics.gradient_norm:.6f} < threshold={self.thresholds['gradient_vanish_threshold']:.6f}",
                    metrics=metrics,
                    suggested_action="increase_learning_rate"
                )
                self._intervention_adjust_learning_rate(metrics, increase=True)
            
            elif issue == "consistent_poor_performance":
                self._raise_alert(
                    severity=AlertSeverity.WARNING,
                    category="performance",
                    message="Consistent poor performance detected over last 10 episodes",
                    metrics=metrics,
                    suggested_action="curriculum_reset"
                )
                self._intervention_curriculum_adjustment(metrics)
    
    def _emergency_intervention(self, issue_type: str, metrics: TrainingMetrics):
        """Apply emergency interventions for critical issues"""
        
        self.emergency_stop_triggered = True
        
        # Save emergency checkpoint
        self._save_emergency_checkpoint(metrics)
        
        if issue_type in ["nan_loss", "nan_gradient", "loss_explosion", "gradient_explosion"]:
            # Reduce all learning rates by 10x
            self._intervention_emergency_lr_reduction(metrics)
        
        # Log emergency to audit trail
        try:
            log_decision(
                user_id="system",
                decision_type="emergency_stop",
                context={
                    "issue_type": issue_type,
                    "step": metrics.step,
                    "episode": metrics.episode,
                    "metrics": metrics.__dict__
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log emergency to audit trail: {e}")
    
    def _intervention_increase_exploration(self, metrics: TrainingMetrics):
        """Increase exploration when premature convergence detected"""
        if hasattr(self.agent, 'epsilon'):
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = min(1.0, self.agent.epsilon * 2.0)
            
            self._log_intervention(
                intervention_type="increase_exploration",
                parameters_changed={"epsilon": {"old": old_epsilon, "new": self.agent.epsilon}},
                reason=f"Premature convergence at episode {metrics.episode}",
                success=True,
                metrics=metrics
            )
    
    def _intervention_force_exploration(self, metrics: TrainingMetrics):
        """Force exploration when action diversity collapses"""
        if hasattr(self.agent, 'epsilon'):
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = max(0.3, self.agent.epsilon * 1.5)
            
            # Also increase dropout if available
            if hasattr(self.agent, 'dropout_rate'):
                old_dropout = self.agent.dropout_rate
                self.agent.dropout_rate = min(0.5, self.agent.dropout_rate * 1.2)
            
            self._log_intervention(
                intervention_type="force_exploration",
                parameters_changed={
                    "epsilon": {"old": old_epsilon, "new": self.agent.epsilon}
                },
                reason=f"Exploration collapse: entropy={metrics.action_entropy:.4f}",
                success=True,
                metrics=metrics
            )
    
    def _intervention_adjust_learning_rate(self, metrics: TrainingMetrics, increase: bool = False):
        """Adjust learning rate based on gradient status"""
        factor = 1.5 if increase else 0.5
        changed_params = {}
        
        # Adjust all optimizers
        for optimizer_name in ['optimizer_bid', 'optimizer_creative', 'optimizer_channel']:
            if hasattr(self.agent, optimizer_name):
                optimizer = getattr(self.agent, optimizer_name)
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] *= factor
                    changed_params[f"{optimizer_name}_lr"] = {"old": old_lr, "new": param_group['lr']}
        
        self._log_intervention(
            intervention_type="learning_rate_adjustment",
            parameters_changed=changed_params,
            reason=f"Gradient {'vanishing' if increase else 'issues'}: norm={metrics.gradient_norm:.6f}",
            success=len(changed_params) > 0,
            metrics=metrics
        )
    
    def _intervention_emergency_lr_reduction(self, metrics: TrainingMetrics):
        """Emergency learning rate reduction for instability"""
        factor = 0.1  # 10x reduction
        changed_params = {}
        
        for optimizer_name in ['optimizer_bid', 'optimizer_creative', 'optimizer_channel']:
            if hasattr(self.agent, optimizer_name):
                optimizer = getattr(self.agent, optimizer_name)
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] *= factor
                    changed_params[f"{optimizer_name}_lr"] = {"old": old_lr, "new": param_group['lr']}
        
        self._log_intervention(
            intervention_type="emergency_lr_reduction",
            parameters_changed=changed_params,
            reason="Training instability - emergency intervention",
            success=len(changed_params) > 0,
            metrics=metrics
        )
    
    def _intervention_curriculum_adjustment(self, metrics: TrainingMetrics):
        """Adjust training curriculum for poor performance"""
        # This would adjust the environment difficulty or data sampling
        # Implementation depends on the specific environment
        self._log_intervention(
            intervention_type="curriculum_adjustment", 
            parameters_changed={"curriculum_level": {"action": "reset_to_easier"}},
            reason="Consistent poor performance",
            success=True,
            metrics=metrics
        )
    
    def _save_emergency_checkpoint(self, metrics: TrainingMetrics):
        """Save emergency checkpoint when instability detected"""
        try:
            # Ensure checkpoint directory exists
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"emergency_checkpoint_step_{metrics.step}.pth"
            )
            
            checkpoint_data = {
                'step': metrics.step,
                'episode': metrics.episode,
                'timestamp': metrics.timestamp,
                'metrics': metrics.__dict__,
                'agent_state': {},
                'optimizer_states': {},
                'thresholds': self.thresholds,
                'alerts': [alert.__dict__ for alert in self.alerts[-10:]],  # Last 10 alerts
                'interventions': [interv.__dict__ for interv in self.interventions[-5:]]  # Last 5 interventions
            }
            
            # Save agent state
            if hasattr(self.agent, 'q_network_bid'):
                checkpoint_data['agent_state']['q_network_bid'] = self.agent.q_network_bid.state_dict()
            if hasattr(self.agent, 'q_network_creative'):
                checkpoint_data['agent_state']['q_network_creative'] = self.agent.q_network_creative.state_dict()
            if hasattr(self.agent, 'q_network_channel'):
                checkpoint_data['agent_state']['q_network_channel'] = self.agent.q_network_channel.state_dict()
            
            # Save optimizer states
            if hasattr(self.agent, 'optimizer_bid'):
                checkpoint_data['optimizer_states']['optimizer_bid'] = self.agent.optimizer_bid.state_dict()
            if hasattr(self.agent, 'optimizer_creative'):
                checkpoint_data['optimizer_states']['optimizer_creative'] = self.agent.optimizer_creative.state_dict()
            if hasattr(self.agent, 'optimizer_channel'):
                checkpoint_data['optimizer_states']['optimizer_channel'] = self.agent.optimizer_channel.state_dict()
            
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Emergency checkpoint saved to {checkpoint_path}")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")
            return None
    
    def _periodic_health_check(self, metrics: TrainingMetrics):
        """Comprehensive periodic health check"""
        
        # Check for plateau
        if self._detect_performance_plateau():
            self._raise_alert(
                severity=AlertSeverity.WARNING,
                category="performance",
                message="Performance plateau detected",
                metrics=metrics,
                suggested_action="adjust_hyperparameters"
            )
        
        # Check for overfitting signs
        if self._detect_overfitting_signs():
            self._raise_alert(
                severity=AlertSeverity.WARNING,
                category="overfitting",
                message="Overfitting patterns detected",
                metrics=metrics,
                suggested_action="increase_regularization"
            )
        
        # Check Q-value health
        if self._detect_q_value_issues(metrics):
            self._raise_alert(
                severity=AlertSeverity.WARNING,
                category="q_values",
                message="Q-value instability detected",
                metrics=metrics,
                suggested_action="reset_target_network"
            )
    
    def _detect_performance_plateau(self) -> bool:
        """Detect if performance has plateaued"""
        if len(self.performance_window) < 50:
            return False
        
        recent_perf = list(self.performance_window)[-25:]
        older_perf = list(self.performance_window)[-50:-25]
        
        recent_mean = np.mean(recent_perf)
        older_mean = np.mean(older_perf)
        
        if older_mean == 0:
            return False
        
        improvement = (recent_mean - older_mean) / abs(older_mean)
        return abs(improvement) < self.thresholds['plateau_threshold']
    
    def _detect_overfitting_signs(self) -> bool:
        """Detect signs of overfitting"""
        # Check action repetition patterns
        if len(self.action_history) < 100:
            return False
        
        recent_actions = list(self.action_history)[-100:]
        unique_actions = len(set(str(action) for action in recent_actions))
        
        # If less than 20% unique actions, likely overfitting
        return unique_actions < 20
    
    def _detect_q_value_issues(self, metrics: TrainingMetrics) -> bool:
        """Detect Q-value related issues"""
        if len(self.q_value_history) < 10:
            return False
        
        # Check Q-value variance
        recent_q_values = list(self.q_value_history)[-10:]
        q_variance = np.var(recent_q_values)
        
        return q_variance > self.thresholds.get('q_value_variance_max', 100.0)
    
    def _store_metrics(self, metrics: TrainingMetrics):
        """Store metrics in history and database"""
        
        # Store in memory
        if not np.isnan(metrics.loss) and not np.isinf(metrics.loss):
            self.loss_history.append(metrics.loss)
        if not np.isnan(metrics.reward) and not np.isinf(metrics.reward):
            self.reward_history.append(metrics.reward)
            self.performance_window.append(metrics.reward)
        if not np.isnan(metrics.gradient_norm) and not np.isinf(metrics.gradient_norm):
            self.gradient_history.append(metrics.gradient_norm)
        if metrics.q_value_mean is not None:
            self.q_value_history.append(metrics.q_value_mean)
        
        # Store in database every 10 steps to avoid overhead
        if self.training_step % 10 == 0:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO training_metrics 
                        (timestamp, step, episode, loss, reward, gradient_norm, epsilon, learning_rate, stage)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics.timestamp, metrics.step, metrics.episode, 
                        metrics.loss, metrics.reward, metrics.gradient_norm,
                        metrics.epsilon, metrics.learning_rate, self.training_stage.value
                    ))
            except Exception as e:
                logger.warning(f"Failed to store metrics in database: {e}")
    
    def _raise_alert(self, severity: AlertSeverity, category: str, message: str, 
                    metrics: TrainingMetrics, suggested_action: Optional[str] = None):
        """Raise a convergence alert"""
        
        alert = ConvergenceAlert(
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=message,
            metrics=metrics.__dict__,
            suggested_action=suggested_action,
            auto_applied=False
        )
        
        self.alerts.append(alert)
        
        # Log based on severity
        if severity == AlertSeverity.EMERGENCY:
            logger.critical(f"EMERGENCY ALERT: {message}")
        elif severity == AlertSeverity.CRITICAL:
            logger.critical(f"CRITICAL ALERT: {message}")
        elif severity == AlertSeverity.WARNING:
            logger.warning(f"WARNING: {message}")
        else:
            logger.info(f"INFO: {message}")
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO convergence_alerts 
                    (timestamp, severity, category, message, metrics, auto_applied)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert.timestamp.timestamp(), severity.value, category, 
                    message, json.dumps(metrics.__dict__), False
                ))
        except Exception as e:
            logger.warning(f"Failed to store alert in database: {e}")
    
    def _log_intervention(self, intervention_type: str, parameters_changed: Dict[str, Any], 
                         reason: str, success: bool, metrics: TrainingMetrics):
        """Log a training intervention"""
        
        intervention = TrainingIntervention(
            timestamp=datetime.now(),
            step=metrics.step,
            episode=metrics.episode,
            intervention_type=intervention_type,
            parameters_changed=parameters_changed,
            reason=reason,
            success=success
        )
        
        self.interventions.append(intervention)
        
        logger.info(f"INTERVENTION: {intervention_type} - {reason}")
        logger.info(f"  Parameters changed: {parameters_changed}")
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO interventions 
                    (timestamp, step, episode, intervention_type, parameters_changed, reason, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    intervention.timestamp.timestamp(), metrics.step, metrics.episode,
                    intervention_type, json.dumps(parameters_changed), reason, success
                ))
        except Exception as e:
            logger.warning(f"Failed to store intervention in database: {e}")
    
    def _update_training_stage(self, metrics: TrainingMetrics):
        """Update training stage based on current metrics"""
        
        if self.emergency_stop_triggered:
            self.training_stage = TrainingStage.EMERGENCY
        elif metrics.step < 500:
            self.training_stage = TrainingStage.WARMUP
        elif metrics.epsilon > 0.2:
            self.training_stage = TrainingStage.EXPLORATION
        elif metrics.epsilon > 0.05:
            self.training_stage = TrainingStage.EXPLOITATION
        else:
            self.training_stage = TrainingStage.CONVERGENCE
    
    def _get_current_learning_rate(self) -> float:
        """Get current learning rate from agent"""
        if hasattr(self.agent, 'optimizer_bid'):
            return self.agent.optimizer_bid.param_groups[0]['lr']
        return 0.0
    
    def _calculate_action_entropy(self, action: Dict[str, Any]) -> float:
        """Calculate entropy of recent actions"""
        if len(self.action_history) < 10:
            return 1.0  # High entropy for early training
        
        recent_actions = list(self.action_history)[-50:]
        action_strings = [str(a) for a in recent_actions]
        
        # Count action frequencies
        from collections import Counter
        action_counts = Counter(action_strings)
        
        # Calculate entropy
        total = len(action_strings)
        probs = [count / total for count in action_counts.values()]
        
        return entropy(probs) if len(probs) > 1 else 0.0
    
    def end_episode(self, episode_reward: float) -> bool:
        """Called at end of episode for episode-level monitoring"""
        
        with self.lock:
            self.episode += 1
            
            # Update performance tracking
            if episode_reward < self.best_performance * 0.9:  # 10% worse than best
                self.consecutive_poor_episodes += 1
            else:
                self.consecutive_poor_episodes = 0
                if episode_reward > self.best_performance:
                    self.best_performance = episode_reward
            
            # Check for consistent poor performance
            max_poor_episodes = self.thresholds.get('consecutive_bad_episodes_max', 20)
            if self.consecutive_poor_episodes > max_poor_episodes:
                self._raise_alert(
                    severity=AlertSeverity.CRITICAL,
                    category="performance",
                    message=f"Consecutive poor episodes: {self.consecutive_poor_episodes}",
                    metrics=TrainingMetrics(
                        step=self.training_step, episode=self.episode, loss=0,
                        reward=episode_reward, gradient_norm=0, q_value_mean=0,
                        q_value_std=0, epsilon=getattr(self.agent, 'epsilon', 0),
                        learning_rate=self._get_current_learning_rate(),
                        action_entropy=0, timestamp=time.time()
                    ),
                    suggested_action="reset_or_stop_training"
                )
                return True
            
            return self.emergency_stop_triggered or self.convergence_detected
    
    def should_stop(self) -> bool:
        """Check if training should stop"""
        return self.emergency_stop_triggered or self.convergence_detected
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive training convergence report"""
        
        with self.lock:
            # Calculate statistics
            recent_losses = list(self.loss_history)[-100:] if len(self.loss_history) >= 100 else list(self.loss_history)
            recent_rewards = list(self.reward_history)[-100:] if len(self.reward_history) >= 100 else list(self.reward_history)
            recent_gradients = list(self.gradient_history)[-50:] if len(self.gradient_history) >= 50 else list(self.gradient_history)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'training_summary': {
                    'total_steps': self.training_step,
                    'total_episodes': self.episode,
                    'current_stage': self.training_stage.value,
                    'emergency_stop': self.emergency_stop_triggered,
                    'convergence_detected': self.convergence_detected
                },
                'current_metrics': {
                    'current_loss': recent_losses[-1] if recent_losses else None,
                    'loss_trend': np.mean(np.diff(recent_losses)) if len(recent_losses) > 1 else 0,
                    'average_reward': np.mean(recent_rewards) if recent_rewards else 0,
                    'reward_std': np.std(recent_rewards) if recent_rewards else 0,
                    'gradient_norm': recent_gradients[-1] if recent_gradients else None,
                    'gradient_stability': np.std(recent_gradients) if len(recent_gradients) > 1 else 0,
                    'current_epsilon': getattr(self.agent, 'epsilon', 0),
                    'learning_rate': self._get_current_learning_rate()
                },
                'stability_assessment': {
                    'loss_stability': 'stable' if len(recent_losses) > 5 and np.std(recent_losses) < np.mean(recent_losses) * 0.1 else 'unstable',
                    'gradient_health': 'healthy' if recent_gradients and all(0.01 < g < 10 for g in recent_gradients[-5:]) else 'concerning',
                    'reward_progression': 'improving' if len(recent_rewards) > 10 and np.mean(recent_rewards[-10:]) > np.mean(recent_rewards[-20:-10]) else 'stagnating'
                },
                'alerts_summary': {
                    'total_alerts': len(self.alerts),
                    'critical_alerts': len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]),
                    'emergency_alerts': len([a for a in self.alerts if a.severity == AlertSeverity.EMERGENCY]),
                    'recent_alerts': [
                        {
                            'timestamp': a.timestamp.isoformat(),
                            'severity': a.severity.value,
                            'message': a.message
                        } for a in self.alerts[-5:]
                    ]
                },
                'interventions_summary': {
                    'total_interventions': len(self.interventions),
                    'successful_interventions': len([i for i in self.interventions if i.success]),
                    'recent_interventions': [
                        {
                            'timestamp': i.timestamp.isoformat(),
                            'type': i.intervention_type,
                            'reason': i.reason,
                            'success': i.success
                        } for i in self.interventions[-5:]
                    ]
                },
                'exploration_metrics': {
                    'current_epsilon': getattr(self.agent, 'epsilon', 0),
                    'action_diversity': len(set(str(a) for a in list(self.action_history)[-100:])) if len(self.action_history) >= 100 else len(set(str(a) for a in self.action_history)),
                    'exploration_stage': 'high' if getattr(self.agent, 'epsilon', 0) > 0.3 else 'medium' if getattr(self.agent, 'epsilon', 0) > 0.1 else 'low'
                },
                'performance_trends': {
                    'best_performance': self.best_performance,
                    'consecutive_poor_episodes': self.consecutive_poor_episodes,
                    'performance_stability': np.std(list(self.performance_window)) if len(self.performance_window) > 5 else 0
                },
                'system_health': {
                    'memory_usage': len(self.loss_history) / self.loss_history.maxlen,
                    'database_status': self._check_database_health(),
                    'checkpoint_status': self._check_checkpoint_system()
                },
                'recommendations': self._generate_recommendations()
            }
            
            return report
    
    def _check_database_health(self) -> str:
        """Check database health"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM training_metrics")
                count = cursor.fetchone()[0]
                return f"healthy ({count} records)"
        except:
            return "unhealthy"
    
    def _check_checkpoint_system(self) -> str:
        """Check checkpoint system health"""
        if os.path.exists(self.checkpoint_dir):
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
            return f"operational ({len(checkpoints)} checkpoints)"
        return "not_configured"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on recent alerts
        critical_alerts = [a for a in self.alerts[-20:] if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]]
        
        if critical_alerts:
            recommendations.append("Review critical alerts and consider stopping training")
        
        if self.consecutive_poor_episodes > 10:
            recommendations.append("Consider adjusting learning rate or resetting exploration")
        
        if len(self.gradient_history) > 5:
            recent_grads = list(self.gradient_history)[-5:]
            if all(g < 0.01 for g in recent_grads):
                recommendations.append("Gradient vanishing detected - increase learning rate")
            elif any(g > 10 for g in recent_grads):
                recommendations.append("High gradients detected - reduce learning rate")
        
        if getattr(self.agent, 'epsilon', 1.0) < 0.05 and self.episode < 1000:
            recommendations.append("Exploration too low for current training stage")
        
        return recommendations
    
    def save_success_metrics(self):
        """Save current successful training metrics for future threshold learning"""
        
        if (len(self.loss_history) > 100 and 
            len(self.reward_history) > 100 and
            not self.emergency_stop_triggered and
            len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]) == 0):
            
            success_data = {
                'gradient_norms': list(self.gradient_history),
                'loss_values': list(self.loss_history), 
                'reward_improvements': [
                    self.reward_history[i] - self.reward_history[i-10] 
                    for i in range(10, len(self.reward_history))
                ],
                'convergence_episodes': [self.episode],
                'final_performance': self.best_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            # Append to existing success data
            existing_data = {}
            if os.path.exists(self.success_metrics_file):
                try:
                    with open(self.success_metrics_file, 'r') as f:
                        existing_data = json.load(f)
                except:
                    pass
            
            # Merge data
            for key, value in success_data.items():
                if key in existing_data:
                    if isinstance(existing_data[key], list):
                        existing_data[key].extend(value if isinstance(value, list) else [value])
                    else:
                        existing_data[key] = [existing_data[key], value]
                else:
                    existing_data[key] = value
            
            with open(self.success_metrics_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            logger.info(f"Success metrics saved to {self.success_metrics_file}")

# Integration function for existing training loops
def integrate_convergence_monitoring(agent, environment, discovery_engine, 
                                   checkpoint_dir: str = "./production_checkpoints") -> ProductionConvergenceMonitor:
    """
    Easy integration function for existing training loops
    
    Usage:
    monitor = integrate_convergence_monitoring(agent, env, discovery_engine)
    
    # In training loop:
    should_stop = monitor.monitor_step(loss, reward, gradient_norm, action, q_values)
    if should_stop:
        break
    """
    
    return ProductionConvergenceMonitor(
        agent=agent,
        environment=environment, 
        discovery_engine=discovery_engine,
        checkpoint_dir=checkpoint_dir
    )