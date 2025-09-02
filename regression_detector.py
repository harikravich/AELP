#!/usr/bin/env python3
"""
PERFORMANCE REGRESSION DETECTION AND AUTOMATIC ROLLBACK SYSTEM
Real-time monitoring, degradation detection, and model rollback mechanisms.

CRITICAL FEATURES:
- Real-time performance monitoring across all key metrics
- Statistical change detection using control limits and Z-scores  
- Automatic baseline establishment from historical performance
- Multi-level degradation alerts (yellow/red/black)
- Automatic model rollback to last good checkpoint
- Comprehensive regression testing framework
- Integration with emergency control system

NO SIMPLIFIED DETECTION - Full statistical analysis only
"""

import logging
import threading
import time
import json
import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import deque, defaultdict
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import shutil
import subprocess
import hashlib
import warnings

# Import emergency controls for integration
from emergency_controls import EmergencyController, EmergencyLevel, EmergencyType

logger = logging.getLogger(__name__)

class RegressionSeverity(IntEnum):
    """Regression severity levels with ordering support"""
    NONE = 0           # No regression detected
    MINOR = 1          # Small degradation, monitor only
    MODERATE = 2       # Significant degradation, alert
    SEVERE = 3         # Major degradation, consider rollback
    CRITICAL = 4       # Critical degradation, immediate rollback
    
    @property
    def value_str(self):
        """String representation for compatibility"""
        return {
            0: "none",
            1: "minor", 
            2: "moderate",
            3: "severe",
            4: "critical"
        }[self.value]

class MetricType(Enum):
    """Types of metrics to monitor"""
    ROAS = "roas"                   # Return on Ad Spend
    CONVERSION_RATE = "cvr"         # Conversion Rate
    CPC = "cpc"                     # Cost Per Click
    CTR = "ctr"                     # Click Through Rate
    TRAINING_LOSS = "training_loss" # Training Loss
    REWARD = "reward"               # Episode Reward
    BID_ACCURACY = "bid_accuracy"   # Bid Prediction Accuracy
    SPEND_EFFICIENCY = "spend_eff"  # Spend Efficiency
    USER_SATISFACTION = "user_sat"  # User Satisfaction Score
    LATENCY = "latency"             # Response Latency

@dataclass
class MetricSnapshot:
    """Single metric measurement"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    episode: Optional[int] = None
    user_id: Optional[str] = None
    campaign_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegressionAlert:
    """Regression detection alert"""
    metric_type: MetricType
    severity: RegressionSeverity
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    p_value: float
    detection_time: datetime
    confidence: float
    recommended_action: str
    historical_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelCheckpoint:
    """Model checkpoint metadata"""
    checkpoint_id: str
    model_path: str
    performance_metrics: Dict[str, float]
    creation_time: datetime
    episodes_trained: int
    validation_scores: Dict[str, float]
    is_baseline: bool = False
    is_rollback_candidate: bool = True

class StatisticalDetector:
    """Advanced statistical methods for regression detection"""
    
    def __init__(self, window_size: int = 100, alpha: float = 0.01):
        self.window_size = window_size
        self.alpha = alpha  # Significance level
        self.baseline_windows = defaultdict(deque)
        self.current_windows = defaultdict(deque)
        self.control_limits = {}
        self.baseline_stats = {}
    
    def update_baseline(self, metric_type: MetricType, values: List[float]):
        """Update baseline statistics from historical data"""
        if len(values) < 10:
            logger.warning(f"Insufficient baseline data for {metric_type.value}: {len(values)} samples")
            return
        
        values_array = np.array(values)
        
        # Remove outliers using IQR method
        q1, q3 = np.percentile(values_array, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        clean_values = values_array[(values_array >= lower_bound) & (values_array <= upper_bound)]
        
        if len(clean_values) < len(values) * 0.7:  # Too many outliers
            logger.warning(f"Too many outliers in baseline for {metric_type.value}")
            clean_values = values_array
        
        # Calculate robust statistics
        mean_val = np.mean(clean_values)
        std_val = np.std(clean_values, ddof=1)
        median_val = np.median(clean_values)
        
        # Establish control limits (3-sigma for critical metrics)
        if metric_type in [MetricType.ROAS, MetricType.CONVERSION_RATE, MetricType.REWARD]:
            sigma_multiplier = 2.5  # Tighter control for key business metrics
        else:
            sigma_multiplier = 3.0
        
        upper_limit = mean_val + sigma_multiplier * std_val
        lower_limit = mean_val - sigma_multiplier * std_val
        
        self.baseline_stats[metric_type] = {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'count': len(clean_values),
            'min': np.min(clean_values),
            'max': np.max(clean_values)
        }
        
        self.control_limits[metric_type] = {
            'upper': upper_limit,
            'lower': lower_limit,
            'sigma_multiplier': sigma_multiplier
        }
        
        logger.info(f"Baseline updated for {metric_type.value}: "
                   f"mean={mean_val:.4f}, std={std_val:.4f}, "
                   f"limits=[{lower_limit:.4f}, {upper_limit:.4f}]")
    
    def detect_regression(self, metric_type: MetricType, current_values: List[float]) -> Optional[RegressionAlert]:
        """Detect regression using multiple statistical tests"""
        if metric_type not in self.baseline_stats:
            logger.warning(f"No baseline available for {metric_type.value}")
            return None
        
        if len(current_values) < 10:
            logger.debug(f"Insufficient current data for {metric_type.value}: {len(current_values)} samples")
            return None
        
        baseline_stats = self.baseline_stats[metric_type]
        current_mean = np.mean(current_values)
        current_std = np.std(current_values, ddof=1)
        
        # Test 1: Z-score test for mean shift
        z_score = (current_mean - baseline_stats['mean']) / (baseline_stats['std'] / np.sqrt(len(current_values)))
        
        # Test 2: Welch's t-test for mean difference
        baseline_values = np.random.normal(baseline_stats['mean'], baseline_stats['std'], baseline_stats['count'])
        t_stat, p_value = stats.ttest_ind(current_values, baseline_values, equal_var=False)
        
        # Test 3: Control limits check
        control_limits = self.control_limits[metric_type]
        out_of_control = (current_mean < control_limits['lower'] or 
                         current_mean > control_limits['upper'])
        
        # Test 4: Variance ratio test (F-test)
        f_ratio = current_std**2 / baseline_stats['std']**2
        f_critical = stats.f.ppf(1 - self.alpha/2, len(current_values)-1, baseline_stats['count']-1)
        variance_changed = f_ratio > f_critical or f_ratio < 1/f_critical
        
        # Determine regression severity
        severity = RegressionSeverity.NONE
        confidence = 0.0
        
        if p_value < self.alpha:  # Statistically significant change
            abs_z = abs(z_score)
            if abs_z > 4.0 or out_of_control:
                severity = RegressionSeverity.CRITICAL
                confidence = 0.99
            elif abs_z > 3.0:
                severity = RegressionSeverity.SEVERE
                confidence = 0.95
            elif abs_z > 2.0:
                severity = RegressionSeverity.MODERATE
                confidence = 0.90
            elif abs_z > 1.5:
                severity = RegressionSeverity.MINOR
                confidence = 0.80
        
        # Additional checks for critical metrics
        if metric_type in [MetricType.ROAS, MetricType.CONVERSION_RATE]:
            # For business metrics, any significant decrease is concerning
            relative_change = (current_mean - baseline_stats['mean']) / baseline_stats['mean']
            if relative_change < -0.10 and severity == RegressionSeverity.NONE:
                severity = RegressionSeverity.MODERATE
                confidence = 0.85
            elif relative_change < -0.20:
                # IntEnum supports direct comparison
                if RegressionSeverity.SEVERE > severity:
                    severity = RegressionSeverity.SEVERE
                confidence = max(confidence, 0.95)
        
        if severity == RegressionSeverity.NONE:
            return None
        
        # Determine recommended action
        if severity == RegressionSeverity.CRITICAL:
            action = "IMMEDIATE_ROLLBACK"
        elif severity == RegressionSeverity.SEVERE:
            action = "CONSIDER_ROLLBACK"
        elif severity == RegressionSeverity.MODERATE:
            action = "ALERT_AND_MONITOR"
        else:
            action = "MONITOR_CLOSELY"
        
        return RegressionAlert(
            metric_type=metric_type,
            severity=severity,
            current_value=current_mean,
            baseline_mean=baseline_stats['mean'],
            baseline_std=baseline_stats['std'],
            z_score=z_score,
            p_value=p_value,
            detection_time=datetime.now(),
            confidence=confidence,
            recommended_action=action,
            historical_context={
                'baseline_count': baseline_stats['count'],
                'current_count': len(current_values),
                'relative_change': (current_mean - baseline_stats['mean']) / baseline_stats['mean'],
                'variance_ratio': f_ratio,
                'out_of_control': out_of_control,
                'variance_changed': variance_changed
            }
        )

class ModelManager:
    """Manages model checkpoints and rollbacks"""
    
    def __init__(self, checkpoint_dir: str = "/home/hariravichandran/AELP/model_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.checkpoints = {}
        self.current_model_id = None
        self.baseline_model_id = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Load checkpoint metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for checkpoint_id, metadata in data.get('checkpoints', {}).items():
                        checkpoint = ModelCheckpoint(
                            checkpoint_id=checkpoint_id,
                            model_path=metadata['model_path'],
                            performance_metrics=metadata['performance_metrics'],
                            creation_time=datetime.fromisoformat(metadata['creation_time']),
                            episodes_trained=metadata['episodes_trained'],
                            validation_scores=metadata['validation_scores'],
                            is_baseline=metadata.get('is_baseline', False),
                            is_rollback_candidate=metadata.get('is_rollback_candidate', True)
                        )
                        self.checkpoints[checkpoint_id] = checkpoint
                    
                    self.current_model_id = data.get('current_model_id')
                    self.baseline_model_id = data.get('baseline_model_id')
                    
                logger.info(f"Loaded {len(self.checkpoints)} checkpoints from metadata")
            except Exception as e:
                logger.error(f"Failed to load checkpoint metadata: {e}")
                self.checkpoints = {}
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk"""
        try:
            data = {
                'checkpoints': {},
                'current_model_id': self.current_model_id,
                'baseline_model_id': self.baseline_model_id
            }
            
            for checkpoint_id, checkpoint in self.checkpoints.items():
                data['checkpoints'][checkpoint_id] = {
                    'model_path': checkpoint.model_path,
                    'performance_metrics': checkpoint.performance_metrics,
                    'creation_time': checkpoint.creation_time.isoformat(),
                    'episodes_trained': checkpoint.episodes_trained,
                    'validation_scores': checkpoint.validation_scores,
                    'is_baseline': checkpoint.is_baseline,
                    'is_rollback_candidate': checkpoint.is_rollback_candidate
                }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")
    
    def create_checkpoint(self, model, performance_metrics: Dict[str, float], 
                         episodes_trained: int, validation_scores: Dict[str, float]) -> str:
        """Create a new model checkpoint"""
        checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{episodes_trained}"
        
        try:
            # Determine file extension and save method based on model type
            if hasattr(model, 'state_dict'):
                model_path = str(self.checkpoint_dir / f"{checkpoint_id}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'performance_metrics': performance_metrics,
                    'episodes_trained': episodes_trained,
                    'validation_scores': validation_scores,
                    'creation_time': datetime.now().isoformat()
                }, model_path)
            else:
                # Handle other model types with pickle
                model_path = str(self.checkpoint_dir / f"{checkpoint_id}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'performance_metrics': performance_metrics,
                        'episodes_trained': episodes_trained,
                        'validation_scores': validation_scores,
                        'creation_time': datetime.now().isoformat()
                    }, f)
            
            # Create checkpoint metadata
            checkpoint = ModelCheckpoint(
                checkpoint_id=checkpoint_id,
                model_path=model_path,
                performance_metrics=performance_metrics,
                creation_time=datetime.now(),
                episodes_trained=episodes_trained,
                validation_scores=validation_scores,
                is_baseline=len(self.checkpoints) == 0,  # First checkpoint is baseline
                is_rollback_candidate=True
            )
            
            self.checkpoints[checkpoint_id] = checkpoint
            self.current_model_id = checkpoint_id
            
            if checkpoint.is_baseline:
                self.baseline_model_id = checkpoint_id
            
            self._save_metadata()
            logger.info(f"Created checkpoint {checkpoint_id} with ROAS={performance_metrics.get('roas', 0):.4f}")
            
            # Clean up old checkpoints (keep last 20)
            self._cleanup_old_checkpoints()
            
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def _cleanup_old_checkpoints(self, keep_count: int = 20):
        """Remove old checkpoints to save disk space"""
        if len(self.checkpoints) <= keep_count:
            return
        
        # Sort by creation time, keep most recent and baseline
        sorted_checkpoints = sorted(
            [(cid, cp) for cid, cp in self.checkpoints.items()],
            key=lambda x: x[1].creation_time,
            reverse=True
        )
        
        to_remove = []
        kept_count = 0
        
        for checkpoint_id, checkpoint in sorted_checkpoints:
            if checkpoint.is_baseline or checkpoint_id == self.current_model_id:
                continue  # Never remove baseline or current model
            
            if kept_count < keep_count:
                kept_count += 1
            else:
                to_remove.append(checkpoint_id)
        
        for checkpoint_id in to_remove:
            try:
                checkpoint = self.checkpoints[checkpoint_id]
                if os.path.exists(checkpoint.model_path):
                    os.remove(checkpoint.model_path)
                del self.checkpoints[checkpoint_id]
                logger.info(f"Cleaned up old checkpoint {checkpoint_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup checkpoint {checkpoint_id}: {e}")
        
        if to_remove:
            self._save_metadata()
    
    def find_rollback_candidate(self, min_performance: Dict[str, float]) -> Optional[str]:
        """Find the best checkpoint for rollback based on performance criteria"""
        candidates = []
        
        for checkpoint_id, checkpoint in self.checkpoints.items():
            if not checkpoint.is_rollback_candidate:
                continue
            
            # Check if checkpoint meets minimum performance requirements
            meets_requirements = True
            score = 0.0
            
            for metric, min_value in min_performance.items():
                checkpoint_value = checkpoint.performance_metrics.get(metric, 0)
                if checkpoint_value < min_value:
                    meets_requirements = False
                    break
                score += checkpoint_value
            
            if meets_requirements:
                candidates.append((checkpoint_id, score, checkpoint.creation_time))
        
        if not candidates:
            logger.warning("No rollback candidates meet performance requirements")
            return None
        
        # Select candidate with best performance, preferring more recent if tied
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_candidate = candidates[0][0]
        
        logger.info(f"Selected rollback candidate: {best_candidate}")
        return best_candidate
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to a specific checkpoint"""
        if checkpoint_id not in self.checkpoints:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        if not os.path.exists(checkpoint.model_path):
            logger.error(f"Checkpoint file not found: {checkpoint.model_path}")
            return False
        
        try:
            # Create backup of current model if it exists
            if self.current_model_id and self.current_model_id != checkpoint_id:
                current_checkpoint = self.checkpoints[self.current_model_id]
                backup_path = current_checkpoint.model_path + ".backup"
                if os.path.exists(current_checkpoint.model_path):
                    shutil.copy2(current_checkpoint.model_path, backup_path)
            
            # Update current model reference
            old_model_id = self.current_model_id
            self.current_model_id = checkpoint_id
            self._save_metadata()
            
            logger.info(f"Successfully rolled back from {old_model_id} to {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback to checkpoint {checkpoint_id}: {e}")
            return False
    
    def get_current_checkpoint(self) -> Optional[ModelCheckpoint]:
        """Get current active checkpoint"""
        if self.current_model_id:
            return self.checkpoints.get(self.current_model_id)
        return None
    
    def load_checkpoint(self, checkpoint_id: str):
        """Load a checkpoint's model state"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        if checkpoint.model_path.endswith('.pt'):
            return torch.load(checkpoint.model_path, weights_only=False)
        elif checkpoint.model_path.endswith('.pkl'):
            with open(checkpoint.model_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Legacy support - try both methods
            try:
                return torch.load(checkpoint.model_path, weights_only=False)
            except:
                with open(checkpoint.model_path, 'rb') as f:
                    return pickle.load(f)

class RegressionDetector:
    """Main regression detection and rollback system"""
    
    def __init__(self, db_path: str = "/home/hariravichandran/AELP/regression_monitoring.db",
                 checkpoint_dir: str = "/home/hariravichandran/AELP/model_checkpoints",
                 emergency_controller: Optional[EmergencyController] = None):
        
        self.db_path = db_path
        self.statistical_detector = StatisticalDetector()
        self.model_manager = ModelManager(checkpoint_dir)
        self.emergency_controller = emergency_controller
        
        # Metric storage
        self.metric_history = defaultdict(deque)
        self.recent_metrics = defaultdict(lambda: deque(maxlen=100))
        self.alerts = deque(maxlen=1000)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
        # Configuration
        self.baseline_window_size = 500  # Episodes for baseline
        self.detection_window_size = 50   # Episodes for current assessment
        self.checkpoint_frequency = 100   # Episodes between checkpoints
        self.auto_rollback_enabled = True
        self.rollback_thresholds = {
            MetricType.ROAS: 0.80,      # 80% of baseline ROAS
            MetricType.CONVERSION_RATE: 0.85,  # 85% of baseline CVR
            MetricType.REWARD: 0.75      # 75% of baseline reward
        }
        
        self._init_database()
        self._load_historical_data()
        
    def _init_database(self):
        """Initialize SQLite database for metric storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        episode INTEGER,
                        user_id TEXT,
                        campaign_id TEXT,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        baseline_mean REAL NOT NULL,
                        z_score REAL NOT NULL,
                        p_value REAL NOT NULL,
                        confidence REAL NOT NULL,
                        detection_time TEXT NOT NULL,
                        recommended_action TEXT NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolution_time TEXT,
                        resolution_action TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS rollbacks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trigger_alert_id INTEGER,
                        from_checkpoint_id TEXT NOT NULL,
                        to_checkpoint_id TEXT NOT NULL,
                        rollback_time TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        performance_before TEXT,
                        performance_after TEXT,
                        success BOOLEAN NOT NULL,
                        FOREIGN KEY (trigger_alert_id) REFERENCES alerts (id)
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_type_time ON metrics(metric_type, timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(detection_time)')
                
                logger.info("Regression monitoring database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _load_historical_data(self):
        """Load historical metrics to establish baselines"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for metric_type in MetricType:
                    cursor.execute('''
                        SELECT value FROM metrics 
                        WHERE metric_type = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (metric_type.value, self.baseline_window_size))
                    
                    values = [row[0] for row in cursor.fetchall()]
                    if len(values) >= 50:  # Minimum for meaningful baseline
                        self.statistical_detector.update_baseline(metric_type, values)
                        logger.info(f"Loaded {len(values)} historical values for {metric_type.value}")
                
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
    
    def record_metric(self, snapshot: MetricSnapshot):
        """Record a new metric measurement"""
        with self.lock:
            # Store in memory
            self.metric_history[snapshot.metric_type].append(snapshot)
            self.recent_metrics[snapshot.metric_type].append(snapshot.value)
            
            # Store in database
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO metrics (metric_type, value, timestamp, episode, user_id, campaign_id, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        snapshot.metric_type.value,
                        snapshot.value,
                        snapshot.timestamp.isoformat(),
                        snapshot.episode,
                        snapshot.user_id,
                        snapshot.campaign_id,
                        json.dumps(snapshot.metadata)
                    ))
            except Exception as e:
                logger.error(f"Failed to record metric: {e}")
    
    def check_for_regressions(self) -> List[RegressionAlert]:
        """Check all metrics for regressions"""
        alerts = []
        
        with self.lock:
            for metric_type in MetricType:
                if len(self.recent_metrics[metric_type]) < 10:
                    continue
                
                recent_values = list(self.recent_metrics[metric_type])
                alert = self.statistical_detector.detect_regression(metric_type, recent_values)
                
                if alert:
                    alerts.append(alert)
                    self.alerts.append(alert)
                    self._store_alert(alert)
                    
                    logger.warning(f"Regression detected: {alert.metric_type.value} "
                                 f"severity={alert.severity.value_str} "
                                 f"z_score={alert.z_score:.2f} "
                                 f"confidence={alert.confidence:.2f}")
                    
                    # Trigger emergency controls if integrated
                    if self.emergency_controller:
                        if alert.severity == RegressionSeverity.CRITICAL:
                            self.emergency_controller.trigger_emergency(
                                EmergencyType.TRAINING_INSTABILITY,
                                f"Critical regression in {alert.metric_type.value}"
                            )
                        elif alert.severity == RegressionSeverity.SEVERE:
                            self.emergency_controller.set_emergency_level(EmergencyLevel.RED)
        
        return alerts
    
    def _store_alert(self, alert: RegressionAlert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO alerts (metric_type, severity, current_value, baseline_mean, 
                                      z_score, p_value, confidence, detection_time, recommended_action)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.metric_type.value,
                    alert.severity.value_str,
                    alert.current_value,
                    alert.baseline_mean,
                    alert.z_score,
                    alert.p_value,
                    alert.confidence,
                    alert.detection_time.isoformat(),
                    alert.recommended_action
                ))
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def evaluate_rollback_need(self, alerts: List[RegressionAlert]) -> bool:
        """Evaluate if automatic rollback is needed"""
        if not self.auto_rollback_enabled:
            return False
        
        critical_alerts = [a for a in alerts if a.severity == RegressionSeverity.CRITICAL]
        severe_alerts = [a for a in alerts if a.severity == RegressionSeverity.SEVERE]
        
        # Immediate rollback conditions
        if len(critical_alerts) >= 1:
            logger.error(f"Critical regression detected in {critical_alerts[0].metric_type.value}")
            return True
        
        if len(severe_alerts) >= 2:
            logger.error(f"Multiple severe regressions detected: {[a.metric_type.value for a in severe_alerts]}")
            return True
        
        # Business metric degradation
        for alert in severe_alerts:
            if alert.metric_type in [MetricType.ROAS, MetricType.CONVERSION_RATE]:
                relative_change = abs(alert.current_value - alert.baseline_mean) / alert.baseline_mean
                if relative_change > 0.25:  # 25% degradation in key metrics
                    logger.error(f"Severe business metric degradation: {alert.metric_type.value}")
                    return True
        
        return False
    
    def perform_automatic_rollback(self, trigger_alerts: List[RegressionAlert]) -> bool:
        """Perform automatic rollback to best checkpoint"""
        logger.info("Initiating automatic model rollback due to performance regression")
        
        # Find appropriate rollback candidate
        min_performance = {}
        for alert in trigger_alerts:
            if alert.metric_type in self.rollback_thresholds:
                min_performance[alert.metric_type.value] = (
                    alert.baseline_mean * self.rollback_thresholds[alert.metric_type]
                )
        
        candidate_id = self.model_manager.find_rollback_candidate(min_performance)
        if not candidate_id:
            logger.error("No suitable rollback candidate found")
            return False
        
        current_checkpoint = self.model_manager.get_current_checkpoint()
        current_id = current_checkpoint.checkpoint_id if current_checkpoint else "unknown"
        
        success = self.model_manager.rollback_to_checkpoint(candidate_id)
        
        # Record rollback
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO rollbacks (from_checkpoint_id, to_checkpoint_id, rollback_time, 
                                         reason, performance_before, performance_after, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    current_id,
                    candidate_id,
                    datetime.now().isoformat(),
                    f"Automatic rollback due to regressions: {[a.metric_type.value for a in trigger_alerts]}",
                    json.dumps({a.metric_type.value: a.current_value for a in trigger_alerts}),
                    json.dumps(self.model_manager.checkpoints[candidate_id].performance_metrics),
                    success
                ))
        except Exception as e:
            logger.error(f"Failed to record rollback: {e}")
        
        if success:
            logger.info(f"Successfully rolled back from {current_id} to {candidate_id}")
            
            # Update baselines to exclude problematic period
            self._reset_baselines_after_rollback()
            
        return success
    
    def _reset_baselines_after_rollback(self):
        """Reset baselines after rollback to avoid contamination"""
        logger.info("Resetting baselines after rollback")
        
        # Clear recent metrics that may be contaminated
        for metric_type in MetricType:
            self.recent_metrics[metric_type].clear()
        
        # Reload baselines from pre-regression data
        self._load_historical_data()
    
    def create_model_checkpoint(self, model, performance_metrics: Dict[str, float], 
                               episodes_trained: int) -> str:
        """Create a new model checkpoint"""
        # Calculate validation scores
        validation_scores = self._calculate_validation_scores(performance_metrics)
        
        checkpoint_id = self.model_manager.create_checkpoint(
            model, performance_metrics, episodes_trained, validation_scores
        )
        
        return checkpoint_id
    
    def _calculate_validation_scores(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate validation scores from performance metrics"""
        validation_scores = {}
        
        # Composite score based on multiple metrics
        roas = performance_metrics.get('roas', 0)
        cvr = performance_metrics.get('conversion_rate', 0)
        reward = performance_metrics.get('reward', 0)
        
        # Weighted composite score
        composite_score = (0.4 * roas + 0.3 * cvr * 100 + 0.3 * reward)
        validation_scores['composite_score'] = composite_score
        
        # Individual metric scores
        for metric, value in performance_metrics.items():
            validation_scores[f'validation_{metric}'] = value
        
        return validation_scores
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Regression monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Regression monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check for regressions
                alerts = self.check_for_regressions()
                
                if alerts:
                    # Evaluate need for rollback
                    if self.evaluate_rollback_need(alerts):
                        self.perform_automatic_rollback(alerts)
                
                # Update baselines periodically
                if len(self.metric_history[MetricType.ROAS]) > self.baseline_window_size:
                    self._update_baselines_from_recent_data()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Longer delay on error
    
    def _update_baselines_from_recent_data(self):
        """Update baselines from recent stable data"""
        for metric_type in MetricType:
            if len(self.metric_history[metric_type]) >= self.baseline_window_size:
                # Get recent stable data (excluding very recent data that might be problematic)
                recent_data = list(self.metric_history[metric_type])[-self.baseline_window_size:-self.detection_window_size]
                values = [snapshot.value for snapshot in recent_data]
                
                if len(values) >= 100:  # Minimum for meaningful update
                    self.statistical_detector.update_baseline(metric_type, values)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'total_alerts': len(self.alerts),
            'recent_alerts': len([a for a in self.alerts if a.detection_time > datetime.now() - timedelta(hours=24)]),
            'current_checkpoint': None,
            'checkpoint_count': len(self.model_manager.checkpoints),
            'metric_summaries': {},
            'system_health': 'healthy'
        }
        
        # Current checkpoint info
        current_checkpoint = self.model_manager.get_current_checkpoint()
        if current_checkpoint:
            summary['current_checkpoint'] = {
                'checkpoint_id': current_checkpoint.checkpoint_id,
                'creation_time': current_checkpoint.creation_time.isoformat(),
                'episodes_trained': current_checkpoint.episodes_trained,
                'performance_metrics': current_checkpoint.performance_metrics
            }
        
        # Metric summaries
        for metric_type in MetricType:
            if metric_type in self.recent_metrics:
                recent_values = list(self.recent_metrics[metric_type])
                if recent_values:
                    summary['metric_summaries'][metric_type.value] = {
                        'current_mean': np.mean(recent_values),
                        'current_std': np.std(recent_values),
                        'sample_count': len(recent_values),
                        'baseline_available': metric_type in self.statistical_detector.baseline_stats
                    }
                    
                    if metric_type in self.statistical_detector.baseline_stats:
                        baseline = self.statistical_detector.baseline_stats[metric_type]
                        summary['metric_summaries'][metric_type.value].update({
                            'baseline_mean': baseline['mean'],
                            'baseline_std': baseline['std'],
                            'relative_change': (np.mean(recent_values) - baseline['mean']) / baseline['mean']
                        })
        
        # Recent critical alerts affect system health
        critical_alerts = [a for a in self.alerts if a.severity == RegressionSeverity.CRITICAL 
                          and a.detection_time > datetime.now() - timedelta(hours=1)]
        if critical_alerts:
            summary['system_health'] = 'critical'
        elif len([a for a in self.alerts if a.severity == RegressionSeverity.SEVERE 
                 and a.detection_time > datetime.now() - timedelta(hours=4)]) > 0:
            summary['system_health'] = 'degraded'
        
        return summary

class RegressionTestSuite:
    """Comprehensive regression testing framework"""
    
    def __init__(self, regression_detector: RegressionDetector):
        self.detector = regression_detector
        self.test_results = []
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all regression tests"""
        logger.info("Starting comprehensive regression testing")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': [],
            'overall_status': 'PASS'
        }
        
        # Test 1: Statistical detection accuracy
        result1 = self._test_statistical_detection()
        test_results['test_details'].append(result1)
        
        # Test 2: Baseline establishment
        result2 = self._test_baseline_establishment()
        test_results['test_details'].append(result2)
        
        # Test 3: Rollback mechanism
        result3 = self._test_rollback_mechanism()
        test_results['test_details'].append(result3)
        
        # Test 4: Alert generation and storage
        result4 = self._test_alert_system()
        test_results['test_details'].append(result4)
        
        # Test 5: Performance under load
        result5 = self._test_performance_under_load()
        test_results['test_details'].append(result5)
        
        # Summarize results
        for result in test_results['test_details']:
            test_results['tests_run'] += 1
            if result['status'] == 'PASS':
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
        
        if test_results['tests_failed'] > 0:
            test_results['overall_status'] = 'FAIL'
        
        logger.info(f"Regression testing complete: {test_results['tests_passed']}/{test_results['tests_run']} passed")
        
        return test_results
    
    def _test_statistical_detection(self) -> Dict[str, Any]:
        """Test statistical regression detection with known patterns"""
        test_name = "Statistical Detection Accuracy"
        
        try:
            # Create baseline data
            baseline_roas = np.random.normal(2.5, 0.5, 100)
            self.detector.statistical_detector.update_baseline(MetricType.ROAS, baseline_roas.tolist())
            
            # Test 1: No regression (should not alert)
            normal_data = np.random.normal(2.5, 0.5, 50)
            alert = self.detector.statistical_detector.detect_regression(MetricType.ROAS, normal_data.tolist())
            
            if alert is not None:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': 'False positive on normal data'}
            
            # Test 2: Clear regression (should alert)
            regression_data = np.random.normal(1.8, 0.3, 50)  # Significant drop
            alert = self.detector.statistical_detector.detect_regression(MetricType.ROAS, regression_data.tolist())
            
            if alert is None or alert.severity == RegressionSeverity.NONE:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': 'Failed to detect clear regression'}
            
            # Test 3: Gradual degradation
            gradual_data = np.random.normal(2.2, 0.4, 50)  # Moderate drop
            alert = self.detector.statistical_detector.detect_regression(MetricType.ROAS, gradual_data.tolist())
            
            if alert is None:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': 'Failed to detect gradual regression'}
            
            return {'test_name': test_name, 'status': 'PASS', 
                   'details': 'All detection patterns working correctly'}
            
        except Exception as e:
            return {'test_name': test_name, 'status': 'FAIL', 
                   'reason': f'Exception: {str(e)}'}
    
    def _test_baseline_establishment(self) -> Dict[str, Any]:
        """Test baseline establishment from historical data"""
        test_name = "Baseline Establishment"
        
        try:
            # Generate test data with known properties
            test_data = np.random.normal(1000, 200, 500)
            
            # Update baseline
            self.detector.statistical_detector.update_baseline(MetricType.REWARD, test_data.tolist())
            
            # Check if baseline was established correctly
            if MetricType.REWARD not in self.detector.statistical_detector.baseline_stats:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': 'Baseline not established'}
            
            baseline = self.detector.statistical_detector.baseline_stats[MetricType.REWARD]
            
            # Verify statistics are reasonable
            if abs(baseline['mean'] - 1000) > 50:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': f'Baseline mean incorrect: {baseline["mean"]}'}
            
            if abs(baseline['std'] - 200) > 50:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': f'Baseline std incorrect: {baseline["std"]}'}
            
            return {'test_name': test_name, 'status': 'PASS', 
                   'details': f'Baseline established: mean={baseline["mean"]:.2f}, std={baseline["std"]:.2f}'}
            
        except Exception as e:
            return {'test_name': test_name, 'status': 'FAIL', 
                   'reason': f'Exception: {str(e)}'}
    
    def _test_rollback_mechanism(self) -> Dict[str, Any]:
        """Test model rollback functionality"""
        test_name = "Rollback Mechanism"
        
        try:
            # Create test checkpoints
            test_model_1 = {'weights': np.random.randn(100)}
            metrics_1 = {'roas': 2.5, 'conversion_rate': 0.05}
            
            test_model_2 = {'weights': np.random.randn(100)}
            metrics_2 = {'roas': 1.8, 'conversion_rate': 0.03}  # Degraded performance
            
            # Create checkpoints
            checkpoint_1 = self.detector.create_model_checkpoint(test_model_1, metrics_1, 100)
            checkpoint_2 = self.detector.create_model_checkpoint(test_model_2, metrics_2, 200)
            
            # Find rollback candidate
            min_performance = {'roas': 2.0, 'conversion_rate': 0.04}
            candidate = self.detector.model_manager.find_rollback_candidate(min_performance)
            
            if candidate != checkpoint_1:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': f'Wrong rollback candidate selected: {candidate}'}
            
            # Test rollback
            success = self.detector.model_manager.rollback_to_checkpoint(checkpoint_1)
            
            if not success:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': 'Rollback operation failed'}
            
            # Verify current model is correct
            current = self.detector.model_manager.get_current_checkpoint()
            if current.checkpoint_id != checkpoint_1:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': 'Current model not updated after rollback'}
            
            return {'test_name': test_name, 'status': 'PASS', 
                   'details': f'Successfully rolled back to {checkpoint_1}'}
            
        except Exception as e:
            return {'test_name': test_name, 'status': 'FAIL', 
                   'reason': f'Exception: {str(e)}'}
    
    def _test_alert_system(self) -> Dict[str, Any]:
        """Test alert generation and storage"""
        test_name = "Alert System"
        
        try:
            # Create test metric snapshots
            timestamp = datetime.now()
            
            # Good metrics
            for i in range(20):
                snapshot = MetricSnapshot(
                    metric_type=MetricType.CPC,
                    value=1.5 + np.random.normal(0, 0.2),
                    timestamp=timestamp - timedelta(minutes=i),
                    episode=100 + i
                )
                self.detector.record_metric(snapshot)
            
            # Establish baseline
            baseline_values = [1.5 + np.random.normal(0, 0.2) for _ in range(100)]
            self.detector.statistical_detector.update_baseline(MetricType.CPC, baseline_values)
            
            # Add problematic metrics
            for i in range(10):
                snapshot = MetricSnapshot(
                    metric_type=MetricType.CPC,
                    value=3.0 + np.random.normal(0, 0.3),  # Much higher CPC
                    timestamp=timestamp + timedelta(minutes=i),
                    episode=120 + i
                )
                self.detector.record_metric(snapshot)
            
            # Check for regressions
            alerts = self.detector.check_for_regressions()
            
            if not alerts:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': 'No alerts generated for clear regression'}
            
            cpc_alert = next((a for a in alerts if a.metric_type == MetricType.CPC), None)
            if not cpc_alert:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': 'CPC regression not detected'}
            
            if cpc_alert.severity == RegressionSeverity.NONE:
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': 'Regression severity not set correctly'}
            
            return {'test_name': test_name, 'status': 'PASS', 
                   'details': f'Alert generated with severity {cpc_alert.severity.value}'}
            
        except Exception as e:
            return {'test_name': test_name, 'status': 'FAIL', 
                   'reason': f'Exception: {str(e)}'}
    
    def _test_performance_under_load(self) -> Dict[str, Any]:
        """Test system performance under high load"""
        test_name = "Performance Under Load"
        
        try:
            start_time = time.time()
            
            # Generate high volume of metrics
            timestamp = datetime.now()
            
            for i in range(1000):  # High volume test
                for metric_type in [MetricType.ROAS, MetricType.CPC, MetricType.CTR]:
                    snapshot = MetricSnapshot(
                        metric_type=metric_type,
                        value=np.random.normal(2.0, 0.5),
                        timestamp=timestamp + timedelta(seconds=i),
                        episode=i
                    )
                    self.detector.record_metric(snapshot)
            
            # Run regression checks
            alerts = self.detector.check_for_regressions()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if processing_time > 30:  # Should complete within 30 seconds
                return {'test_name': test_name, 'status': 'FAIL', 
                       'reason': f'Processing too slow: {processing_time:.2f}s'}
            
            return {'test_name': test_name, 'status': 'PASS', 
                   'details': f'Processed 3000 metrics in {processing_time:.2f}s'}
            
        except Exception as e:
            return {'test_name': test_name, 'status': 'FAIL', 
                   'reason': f'Exception: {str(e)}'}

# Integration function for GAELP system
def integrate_with_gaelp_training(agent, environment, regression_detector: RegressionDetector):
    """Integration wrapper for GAELP training with regression detection"""
    
    def training_step_with_monitoring(episode: int, total_reward: float, 
                                    metrics: Dict[str, float], model_state):
        """Wrapper for training steps with regression monitoring"""
        
        # Record key metrics
        timestamp = datetime.now()
        
        # Record ROAS if available
        if 'roas' in metrics:
            snapshot = MetricSnapshot(
                metric_type=MetricType.ROAS,
                value=metrics['roas'],
                timestamp=timestamp,
                episode=episode
            )
            regression_detector.record_metric(snapshot)
        
        # Record conversion rate
        if 'conversion_rate' in metrics:
            snapshot = MetricSnapshot(
                metric_type=MetricType.CONVERSION_RATE,
                value=metrics['conversion_rate'],
                timestamp=timestamp,
                episode=episode
            )
            regression_detector.record_metric(snapshot)
        
        # Record episode reward
        snapshot = MetricSnapshot(
            metric_type=MetricType.REWARD,
            value=total_reward,
            timestamp=timestamp,
            episode=episode
        )
        regression_detector.record_metric(snapshot)
        
        # Record training loss if available
        if hasattr(agent, 'last_loss') and agent.last_loss is not None:
            snapshot = MetricSnapshot(
                metric_type=MetricType.TRAINING_LOSS,
                value=float(agent.last_loss),
                timestamp=timestamp,
                episode=episode
            )
            regression_detector.record_metric(snapshot)
        
        # Check for regressions every 25 episodes
        if episode % 25 == 0:
            alerts = regression_detector.check_for_regressions()
            
            if alerts:
                logger.warning(f"Episode {episode}: {len(alerts)} regression alerts")
                
                # Log alert details
                for alert in alerts:
                    logger.warning(f"  {alert.metric_type.value}: {alert.severity.value} "
                                 f"(current={alert.current_value:.4f}, "
                                 f"baseline={alert.baseline_mean:.4f})")
                
                # Check if rollback is needed
                if regression_detector.evaluate_rollback_need(alerts):
                    logger.error(f"Episode {episode}: Initiating automatic rollback")
                    success = regression_detector.perform_automatic_rollback(alerts)
                    
                    if success:
                        logger.info("Automatic rollback completed successfully")
                        # Load the rolled back model state
                        current_checkpoint = regression_detector.model_manager.get_current_checkpoint()
                        if current_checkpoint:
                            checkpoint_data = regression_detector.model_manager.load_checkpoint(
                                current_checkpoint.checkpoint_id
                            )
                            # Agent should reload its state from checkpoint
                            if hasattr(agent, 'load_state_dict') and 'model_state_dict' in checkpoint_data:
                                agent.load_state_dict(checkpoint_data['model_state_dict'])
                    else:
                        logger.error("Automatic rollback failed")
        
        # Create checkpoint every 100 episodes
        if episode % 100 == 0 and episode > 0:
            checkpoint_metrics = {
                'roas': metrics.get('roas', 0),
                'conversion_rate': metrics.get('conversion_rate', 0),
                'reward': total_reward,
                'episode': episode
            }
            
            checkpoint_id = regression_detector.create_model_checkpoint(
                model_state, checkpoint_metrics, episode
            )
            logger.info(f"Episode {episode}: Created checkpoint {checkpoint_id}")
    
    return training_step_with_monitoring

# Main function for standalone usage
def main():
    """Main function for testing regression detection system"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize regression detector
    detector = RegressionDetector()
    
    # Start monitoring
    detector.start_monitoring()
    
    try:
        # Run comprehensive tests
        test_suite = RegressionTestSuite(detector)
        results = test_suite.run_comprehensive_tests()
        
        print("\n" + "="*70)
        print("REGRESSION DETECTION SYSTEM TEST RESULTS")
        print("="*70)
        print(f"Overall Status: {results['overall_status']}")
        print(f"Tests Passed: {results['tests_passed']}/{results['tests_run']}")
        
        for test_detail in results['test_details']:
            status_symbol = "✅" if test_detail['status'] == 'PASS' else "❌"
            print(f"{status_symbol} {test_detail['test_name']}: {test_detail['status']}")
            if 'reason' in test_detail:
                print(f"   Reason: {test_detail['reason']}")
            if 'details' in test_detail:
                print(f"   Details: {test_detail['details']}")
        
        # Generate performance summary
        summary = detector.get_performance_summary()
        print(f"\nSystem Health: {summary['system_health']}")
        print(f"Active Monitoring: {summary['monitoring_active']}")
        print(f"Total Checkpoints: {summary['checkpoint_count']}")
        
        print("\n" + "="*70)
        
    finally:
        detector.stop_monitoring()

if __name__ == "__main__":
    main()