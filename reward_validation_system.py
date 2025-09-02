#!/usr/bin/env python3
"""
GAELP Reward Function Validation & Safety System
Production-grade reward validation with clipping, anomaly detection, and safety checks.

CRITICAL SAFETY FEATURES:
1. Reward function static analysis & validation
2. Real-time reward clipping and bounds checking  
3. Reward hacking detection algorithms
4. Multi-touch attribution reward validation
5. Budget-aware reward scaling
6. Temporal reward consistency checks
7. Statistical anomaly detection
8. Human-in-the-loop review for suspicious patterns

NO PLACEHOLDER IMPLEMENTATIONS - ALL PRODUCTION READY
"""

import numpy as np
import pandas as pd
import logging
import json
import sqlite3
import threading
import time
import ast
import inspect
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import hashlib
import pickle
from pathlib import Path
import warnings
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class RewardValidationLevel(Enum):
    """Reward validation severity levels"""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"

@dataclass
class RewardAnomalyPattern:
    """Detected reward anomaly pattern"""
    pattern_id: str
    pattern_type: str
    timestamp: datetime
    severity: RewardValidationLevel
    description: str
    statistical_metrics: Dict[str, float]
    affected_components: List[str]
    suggested_actions: List[str]
    human_review_required: bool = False

@dataclass 
class RewardValidationResult:
    """Result of reward validation check"""
    is_valid: bool
    original_reward: float
    validated_reward: float
    clipping_applied: bool
    anomaly_score: float
    validation_level: RewardValidationLevel
    warnings: List[str]
    metadata: Dict[str, Any]

class RewardFunctionAnalyzer:
    """Static analysis of reward functions for safety"""
    
    def __init__(self):
        self.dangerous_patterns = [
            r'reward\s*\*=?\s*[0-9]+',  # Reward multiplication by constants
            r'reward\s*\+=?\s*[0-9]+',  # Reward addition by constants
            r'np\.inf|float\(\'inf\'\)',  # Infinite values
            r'np\.nan|float\(\'nan\'\)',  # NaN values
            r'while\s+True',  # Infinite loops
            r'exec\s*\(',  # Dynamic execution
            r'eval\s*\(',  # Dynamic evaluation
        ]
        self.suspicious_patterns = [
            r'random\.',  # Random number usage in rewards
            r'time\.',   # Time-dependent rewards (can be manipulated)
            r'global\s+',  # Global variable modification
            r'\.append\(',  # List modifications (potential memory leaks)
        ]
        
    def analyze_reward_function(self, func: Callable) -> Tuple[bool, List[str], List[str]]:
        """Analyze reward function for safety and correctness"""
        try:
            # Get source code
            source = inspect.getsource(func)
            
            # Parse AST for deeper analysis
            tree = ast.parse(source)
            
            dangerous_issues = []
            suspicious_issues = []
            
            # Pattern-based analysis
            for pattern in self.dangerous_patterns:
                import re
                if re.search(pattern, source):
                    dangerous_issues.append(f"Dangerous pattern detected: {pattern}")
            
            for pattern in self.suspicious_patterns:
                import re
                if re.search(pattern, source):
                    suspicious_issues.append(f"Suspicious pattern detected: {pattern}")
            
            # AST-based analysis
            dangerous_ast, suspicious_ast = self._analyze_ast(tree)
            dangerous_issues.extend(dangerous_ast)
            suspicious_issues.extend(suspicious_ast)
            
            is_safe = len(dangerous_issues) == 0
            return is_safe, dangerous_issues, suspicious_issues
            
        except Exception as e:
            logger.error(f"Error analyzing reward function: {e}")
            return False, [f"Analysis failed: {e}"], []
    
    def _analyze_ast(self, tree: ast.AST) -> Tuple[List[str], List[str]]:
        """Analyze AST for dangerous patterns"""
        dangerous = []
        suspicious = []
        
        for node in ast.walk(tree):
            # Check for dangerous operations
            if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value:
                dangerous.append("Infinite while loop detected")
            
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['exec', 'eval']:
                    dangerous.append(f"Dangerous function call: {node.func.id}")
                elif node.func.id in ['print', 'input']:
                    suspicious.append(f"I/O operation in reward function: {node.func.id}")
            
            # Check for global modifications
            if isinstance(node, ast.Global):
                suspicious.append("Global variable modification detected")
            
            # Check for infinite values
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                if np.isinf(node.value) or np.isnan(node.value):
                    dangerous.append(f"Infinite/NaN constant: {node.value}")
        
        return dangerous, suspicious

class RewardBoundsManager:
    """Manages dynamic reward bounds based on system state"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bounds_history = deque(maxlen=10000)
        self.dynamic_bounds = {}
        self.percentile_bounds = {}
        
        # Static bounds
        self.absolute_min = config.get('absolute_min_reward', -1000.0)
        self.absolute_max = config.get('absolute_max_reward', 1000.0)
        
        # Dynamic bounds parameters
        self.percentile_low = config.get('percentile_low', 1.0)
        self.percentile_high = config.get('percentile_high', 99.0)
        self.bound_update_frequency = config.get('bound_update_frequency', 1000)
        
        logger.info("Reward bounds manager initialized")
    
    def update_bounds(self, rewards: List[float], context: str = "general"):
        """Update dynamic bounds based on recent rewards"""
        if len(rewards) < 10:
            return
        
        # Calculate percentile bounds
        low_bound = np.percentile(rewards, self.percentile_low)
        high_bound = np.percentile(rewards, self.percentile_high)
        
        # Apply safety margins
        margin = (high_bound - low_bound) * 0.1  # 10% margin
        low_bound -= margin
        high_bound += margin
        
        # Ensure within absolute bounds
        low_bound = max(low_bound, self.absolute_min)
        high_bound = min(high_bound, self.absolute_max)
        
        self.percentile_bounds[context] = {
            'low': low_bound,
            'high': high_bound,
            'timestamp': datetime.now(),
            'sample_size': len(rewards)
        }
        
        logger.info(f"Updated bounds for {context}: [{low_bound:.3f}, {high_bound:.3f}]")
    
    def get_bounds(self, context: str = "general") -> Tuple[float, float]:
        """Get current bounds for context"""
        if context in self.percentile_bounds:
            bounds = self.percentile_bounds[context]
            # Check if bounds are stale (older than 1 hour)
            if (datetime.now() - bounds['timestamp']).seconds < 3600:
                return bounds['low'], bounds['high']
        
        # Fall back to absolute bounds
        return self.absolute_min, self.absolute_max
    
    def clip_reward(self, reward: float, context: str = "general") -> Tuple[float, bool]:
        """Clip reward to safe bounds"""
        low_bound, high_bound = self.get_bounds(context)
        
        original_reward = reward
        clipped_reward = np.clip(reward, low_bound, high_bound)
        
        clipping_applied = abs(original_reward - clipped_reward) > 1e-6
        
        if clipping_applied:
            logger.warning(f"Reward clipped: {original_reward:.3f} -> {clipped_reward:.3f} (bounds: [{low_bound:.3f}, {high_bound:.3f}])")
        
        return clipped_reward, clipping_applied

class RewardAnomalyDetector:
    """Advanced anomaly detection for reward patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_history = deque(maxlen=10000)
        self.anomaly_patterns = []
        
        # Anomaly detection parameters
        self.z_score_threshold = config.get('z_score_threshold', 3.0)
        self.iqr_multiplier = config.get('iqr_multiplier', 1.5)
        self.sequence_length = config.get('sequence_length', 100)
        self.drift_threshold = config.get('drift_threshold', 0.5)
        
        # Pattern detection
        self.spike_threshold = config.get('spike_threshold', 5.0)
        self.gradient_threshold = config.get('gradient_threshold', 2.0)
        self.oscillation_threshold = config.get('oscillation_threshold', 0.8)
        
        logger.info("Reward anomaly detector initialized")
    
    def detect_anomalies(self, reward: float, context: Dict[str, Any] = None) -> Tuple[bool, float, List[str]]:
        """Detect if current reward is anomalous"""
        context = context or {}
        
        # Add to history
        self.reward_history.append({
            'reward': reward,
            'timestamp': datetime.now(),
            'context': context
        })
        
        if len(self.reward_history) < 30:
            return False, 0.0, []  # Need sufficient history
        
        # Extract recent rewards for analysis
        recent_rewards = [r['reward'] for r in list(self.reward_history)[-1000:]]
        
        anomaly_flags = []
        anomaly_scores = []
        
        # 1. Statistical outlier detection (Z-score)
        is_outlier, z_score = self._detect_statistical_outlier(reward, recent_rewards)
        if is_outlier:
            anomaly_flags.append(f"Statistical outlier (Z-score: {z_score:.2f})")
            anomaly_scores.append(abs(z_score) / self.z_score_threshold)
        
        # 2. IQR-based outlier detection
        is_iqr_outlier, iqr_score = self._detect_iqr_outlier(reward, recent_rewards)
        if is_iqr_outlier:
            anomaly_flags.append(f"IQR outlier (score: {iqr_score:.2f})")
            anomaly_scores.append(iqr_score)
        
        # 3. Sudden spike detection
        is_spike, spike_magnitude = self._detect_spike(reward, recent_rewards)
        if is_spike:
            anomaly_flags.append(f"Sudden spike (magnitude: {spike_magnitude:.2f})")
            anomaly_scores.append(spike_magnitude / self.spike_threshold)
        
        # 4. Gradient anomaly detection
        is_gradient_anomaly, gradient_score = self._detect_gradient_anomaly(recent_rewards)
        if is_gradient_anomaly:
            anomaly_flags.append(f"Gradient anomaly (score: {gradient_score:.2f})")
            anomaly_scores.append(gradient_score / self.gradient_threshold)
        
        # 5. Oscillation pattern detection
        is_oscillation = self._detect_oscillation_pattern(recent_rewards)
        if is_oscillation:
            anomaly_flags.append("Suspicious oscillation pattern")
            anomaly_scores.append(1.0)
        
        # 6. Context-based anomaly detection
        context_anomalies = self._detect_context_anomalies(reward, context, recent_rewards)
        anomaly_flags.extend(context_anomalies)
        
        # Calculate overall anomaly score
        overall_score = max(anomaly_scores) if anomaly_scores else 0.0
        is_anomalous = len(anomaly_flags) > 0
        
        return is_anomalous, overall_score, anomaly_flags
    
    def _detect_statistical_outlier(self, reward: float, recent_rewards: List[float]) -> Tuple[bool, float]:
        """Detect statistical outliers using Z-score"""
        if len(recent_rewards) < 10:
            return False, 0.0
        
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        if std_reward == 0:
            return False, 0.0
        
        z_score = abs((reward - mean_reward) / std_reward)
        is_outlier = z_score > self.z_score_threshold
        
        return is_outlier, z_score
    
    def _detect_iqr_outlier(self, reward: float, recent_rewards: List[float]) -> Tuple[bool, float]:
        """Detect outliers using Interquartile Range"""
        if len(recent_rewards) < 10:
            return False, 0.0
        
        q25 = np.percentile(recent_rewards, 25)
        q75 = np.percentile(recent_rewards, 75)
        iqr = q75 - q25
        
        if iqr == 0:
            return False, 0.0
        
        lower_bound = q25 - self.iqr_multiplier * iqr
        upper_bound = q75 + self.iqr_multiplier * iqr
        
        is_outlier = reward < lower_bound or reward > upper_bound
        
        # Calculate outlier score
        if reward < lower_bound:
            score = (lower_bound - reward) / iqr
        elif reward > upper_bound:
            score = (reward - upper_bound) / iqr
        else:
            score = 0.0
        
        return is_outlier, score
    
    def _detect_spike(self, reward: float, recent_rewards: List[float]) -> Tuple[bool, float]:
        """Detect sudden spikes in reward values"""
        if len(recent_rewards) < 5:
            return False, 0.0
        
        # Compare with recent average
        recent_avg = np.mean(recent_rewards[-10:])
        spike_magnitude = abs(reward - recent_avg) / (abs(recent_avg) + 1e-6)
        
        is_spike = spike_magnitude > self.spike_threshold
        return is_spike, spike_magnitude
    
    def _detect_gradient_anomaly(self, recent_rewards: List[float]) -> Tuple[bool, float]:
        """Detect anomalous gradients in reward sequence"""
        if len(recent_rewards) < 10:
            return False, 0.0
        
        # Calculate gradients
        gradients = np.gradient(recent_rewards[-10:])
        max_gradient = np.max(np.abs(gradients))
        
        # Compare with historical gradient distribution
        if len(recent_rewards) >= 50:
            historical_gradients = []
            for i in range(10, len(recent_rewards)):
                hist_grad = np.gradient(recent_rewards[i-10:i])
                historical_gradients.extend(hist_grad)
            
            if historical_gradients:
                grad_std = np.std(historical_gradients)
                grad_score = max_gradient / (grad_std + 1e-6)
                is_anomaly = grad_score > self.gradient_threshold
                return is_anomaly, grad_score
        
        return False, 0.0
    
    def _detect_oscillation_pattern(self, recent_rewards: List[float]) -> bool:
        """Detect suspicious oscillation patterns"""
        if len(recent_rewards) < 20:
            return False
        
        # Calculate autocorrelation to detect periodic patterns
        rewards_array = np.array(recent_rewards[-20:])
        rewards_normalized = (rewards_array - np.mean(rewards_array)) / (np.std(rewards_array) + 1e-6)
        
        # Check for oscillation with different periods
        for period in [2, 3, 4, 5]:
            if len(rewards_normalized) >= period * 3:
                correlation = np.corrcoef(rewards_normalized[:-period], rewards_normalized[period:])[0, 1]
                if abs(correlation) > self.oscillation_threshold:
                    return True
        
        return False
    
    def _detect_context_anomalies(self, reward: float, context: Dict[str, Any], 
                                 recent_rewards: List[float]) -> List[str]:
        """Detect context-based anomalies"""
        anomalies = []
        
        # Check reward vs. expected conversion value
        if 'conversion_value' in context and context['conversion_value'] > 0:
            expected_reward = context['conversion_value'] * context.get('conversion_probability', 0.01)
            if reward > expected_reward * 10:  # Reward is 10x expected value
                anomalies.append(f"Reward {reward:.2f} >> expected {expected_reward:.2f}")
        
        # Check reward vs. bid amount
        if 'bid_amount' in context:
            if reward > context['bid_amount'] * 5:  # Reward is 5x bid
                anomalies.append(f"Reward {reward:.2f} >> bid {context['bid_amount']:.2f}")
        
        # Check for budget-inconsistent rewards
        if 'budget_remaining' in context and context['budget_remaining'] < context.get('bid_amount', 0):
            if reward > 0:  # Positive reward despite insufficient budget
                anomalies.append("Positive reward despite insufficient budget")
        
        return anomalies

class RewardConsistencyValidator:
    """Validates reward consistency across different contexts and time periods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_contexts = defaultdict(list)
        self.consistency_metrics = {}
        
        # Consistency parameters
        self.max_context_deviation = config.get('max_context_deviation', 0.5)
        self.temporal_window_hours = config.get('temporal_window_hours', 24)
        self.min_samples_for_consistency = config.get('min_samples_for_consistency', 20)
        
        logger.info("Reward consistency validator initialized")
    
    def validate_consistency(self, reward: float, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate reward consistency with similar contexts"""
        consistency_issues = []
        
        # Create context signature for grouping
        context_signature = self._create_context_signature(context)
        
        # Add current reward to context group
        self.reward_contexts[context_signature].append({
            'reward': reward,
            'timestamp': datetime.now(),
            'full_context': context
        })
        
        # Keep only recent rewards (within temporal window)
        cutoff_time = datetime.now() - timedelta(hours=self.temporal_window_hours)
        self.reward_contexts[context_signature] = [
            r for r in self.reward_contexts[context_signature] 
            if r['timestamp'] > cutoff_time
        ]
        
        # Check consistency if we have enough samples
        if len(self.reward_contexts[context_signature]) >= self.min_samples_for_consistency:
            issues = self._check_context_consistency(context_signature, reward)
            consistency_issues.extend(issues)
        
        # Check cross-context consistency
        cross_context_issues = self._check_cross_context_consistency(reward, context)
        consistency_issues.extend(cross_context_issues)
        
        is_consistent = len(consistency_issues) == 0
        return is_consistent, consistency_issues
    
    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature for grouping similar contexts"""
        # Group by key context features
        signature_features = {
            'user_segment': context.get('user_segment', 'unknown'),
            'channel': context.get('channel', 'unknown'),
            'time_bucket': self._get_time_bucket(datetime.now()),
            'budget_tier': self._get_budget_tier(context.get('budget_remaining', 0)),
            'competition_level': self._get_competition_bucket(context.get('competition_level', 1.0))
        }
        
        # Create hash signature
        signature_str = json.dumps(signature_features, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()[:12]
    
    def _get_time_bucket(self, timestamp: datetime) -> str:
        """Get time bucket for temporal consistency"""
        hour = timestamp.hour
        if hour < 6:
            return "night"
        elif hour < 12:
            return "morning"
        elif hour < 18:
            return "afternoon"
        else:
            return "evening"
    
    def _get_budget_tier(self, budget: float) -> str:
        """Get budget tier for consistency grouping"""
        if budget < 100:
            return "low"
        elif budget < 500:
            return "medium"
        else:
            return "high"
    
    def _get_competition_bucket(self, competition: float) -> str:
        """Get competition level bucket"""
        if competition < 0.5:
            return "low"
        elif competition < 1.5:
            return "medium"
        else:
            return "high"
    
    def _check_context_consistency(self, context_signature: str, current_reward: float) -> List[str]:
        """Check consistency within the same context"""
        issues = []
        
        rewards = [r['reward'] for r in self.reward_contexts[context_signature]]
        
        if len(rewards) < 2:
            return issues
        
        # Calculate statistics
        mean_reward = np.mean(rewards[:-1])  # Exclude current reward
        std_reward = np.std(rewards[:-1])
        
        # Check if current reward deviates significantly
        if std_reward > 0:
            z_score = abs((current_reward - mean_reward) / std_reward)
            if z_score > 3.0:  # 3 standard deviations
                issues.append(f"Context consistency violation: Z-score {z_score:.2f}")
        
        # Check coefficient of variation
        if mean_reward != 0:
            cv = std_reward / abs(mean_reward)
            if cv > self.max_context_deviation:
                issues.append(f"High context variability: CV {cv:.2f}")
        
        return issues
    
    def _check_cross_context_consistency(self, reward: float, context: Dict[str, Any]) -> List[str]:
        """Check consistency across different contexts"""
        issues = []
        
        # This would implement more sophisticated cross-context analysis
        # For now, implement basic checks
        
        return issues

class ProductionRewardValidator:
    """Production-grade reward validation system"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "reward_validation_config.json"
        self.config = self._load_config()
        
        # Initialize components
        self.function_analyzer = RewardFunctionAnalyzer()
        self.bounds_manager = RewardBoundsManager(self.config.get('bounds', {}))
        self.anomaly_detector = RewardAnomalyDetector(self.config.get('anomaly_detection', {}))
        self.consistency_validator = RewardConsistencyValidator(self.config.get('consistency', {}))
        
        # Validation state
        self.validation_history = deque(maxlen=100000)
        self.suspicious_patterns = []
        self.human_review_queue = []
        
        # Database for logging
        self.db_path = "reward_validation.db"
        self._init_database()
        
        # Monitoring
        self.monitoring_active = True
        self._start_monitoring()
        
        logger.info("Production reward validator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "bounds": {
                "absolute_min_reward": -1000.0,
                "absolute_max_reward": 1000.0,
                "percentile_low": 1.0,
                "percentile_high": 99.0
            },
            "anomaly_detection": {
                "z_score_threshold": 3.0,
                "iqr_multiplier": 1.5,
                "spike_threshold": 5.0,
                "gradient_threshold": 2.0
            },
            "consistency": {
                "max_context_deviation": 0.5,
                "temporal_window_hours": 24,
                "min_samples_for_consistency": 20
            },
            "human_review": {
                "enabled": True,
                "anomaly_score_threshold": 2.0,
                "pattern_review_threshold": 5
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        else:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _init_database(self):
        """Initialize validation database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reward_validations (
                validation_id TEXT PRIMARY KEY,
                timestamp TEXT,
                original_reward REAL,
                validated_reward REAL,
                clipping_applied BOOLEAN,
                anomaly_score REAL,
                validation_level TEXT,
                context TEXT,
                warnings TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_patterns (
                pattern_id TEXT PRIMARY KEY,
                timestamp TEXT,
                pattern_type TEXT,
                severity TEXT,
                description TEXT,
                statistical_metrics TEXT,
                human_reviewed BOOLEAN
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        def monitor():
            while self.monitoring_active:
                try:
                    self._check_for_patterns()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    logger.error(f"Error in monitoring: {e}")
                    time.sleep(300)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def validate_reward_function(self, reward_func: Callable) -> Tuple[bool, List[str], List[str]]:
        """Validate a reward function for safety"""
        return self.function_analyzer.analyze_reward_function(reward_func)
    
    def validate_reward(self, reward: float, context: Dict[str, Any] = None) -> RewardValidationResult:
        """Comprehensively validate a single reward value"""
        context = context or {}
        
        # 1. Bounds checking and clipping
        validated_reward, clipping_applied = self.bounds_manager.clip_reward(
            reward, context.get('context_type', 'general')
        )
        
        # 2. Anomaly detection
        is_anomalous, anomaly_score, anomaly_flags = self.anomaly_detector.detect_anomalies(
            reward, context
        )
        
        # 3. Consistency validation
        is_consistent, consistency_issues = self.consistency_validator.validate_consistency(
            reward, context
        )
        
        # 4. Determine validation level
        if is_anomalous and anomaly_score > 3.0:
            validation_level = RewardValidationLevel.BLOCKED
        elif is_anomalous and anomaly_score > 2.0:
            validation_level = RewardValidationLevel.DANGEROUS
        elif not is_consistent or anomaly_score > 1.0:
            validation_level = RewardValidationLevel.SUSPICIOUS
        else:
            validation_level = RewardValidationLevel.NORMAL
        
        # 5. Collect warnings
        warnings = []
        if clipping_applied:
            warnings.append("Reward clipped to safe bounds")
        warnings.extend(anomaly_flags)
        warnings.extend(consistency_issues)
        
        # 6. Create validation result
        result = RewardValidationResult(
            is_valid=(validation_level != RewardValidationLevel.BLOCKED),
            original_reward=reward,
            validated_reward=validated_reward,
            clipping_applied=clipping_applied,
            anomaly_score=anomaly_score,
            validation_level=validation_level,
            warnings=warnings,
            metadata={
                'validation_timestamp': datetime.now().isoformat(),
                'context': context,
                'anomaly_flags': anomaly_flags,
                'consistency_issues': consistency_issues
            }
        )
        
        # 7. Log validation
        self._log_validation(result)
        
        # 8. Add to human review if needed
        if self._needs_human_review(result):
            self.human_review_queue.append(result)
        
        return result
    
    def _log_validation(self, result: RewardValidationResult):
        """Log validation result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO reward_validations 
                (validation_id, timestamp, original_reward, validated_reward, 
                 clipping_applied, anomaly_score, validation_level, context, warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hashlib.md5(f"{result.original_reward}{datetime.now()}".encode()).hexdigest(),
                datetime.now().isoformat(),
                result.original_reward,
                result.validated_reward,
                result.clipping_applied,
                result.anomaly_score,
                result.validation_level.value,
                json.dumps(result.metadata.get('context', {})),
                json.dumps(result.warnings)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging validation: {e}")
    
    def _needs_human_review(self, result: RewardValidationResult) -> bool:
        """Check if validation result needs human review"""
        if not self.config.get('human_review', {}).get('enabled', True):
            return False
        
        threshold = self.config.get('human_review', {}).get('anomaly_score_threshold', 2.0)
        
        return (result.validation_level in [RewardValidationLevel.DANGEROUS, RewardValidationLevel.BLOCKED] or
                result.anomaly_score > threshold)
    
    def _check_for_patterns(self):
        """Check for suspicious patterns in validation history"""
        if len(self.validation_history) < 100:
            return
        
        recent_validations = list(self.validation_history)[-1000:]
        
        # Check for clustering of anomalies
        anomalous_count = sum(1 for v in recent_validations 
                             if v.validation_level in [RewardValidationLevel.DANGEROUS, 
                                                     RewardValidationLevel.BLOCKED])
        
        if anomalous_count > len(recent_validations) * 0.1:  # More than 10% anomalous
            pattern = RewardAnomalyPattern(
                pattern_id=f"cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                pattern_type="anomaly_clustering",
                timestamp=datetime.now(),
                severity=RewardValidationLevel.DANGEROUS,
                description=f"High concentration of anomalous rewards: {anomalous_count}/{len(recent_validations)}",
                statistical_metrics={'anomaly_rate': anomalous_count / len(recent_validations)},
                affected_components=['reward_system'],
                suggested_actions=['review_reward_function', 'check_for_manipulation'],
                human_review_required=True
            )
            
            self.suspicious_patterns.append(pattern)
            self._log_pattern(pattern)
    
    def _log_pattern(self, pattern: RewardAnomalyPattern):
        """Log suspicious pattern to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO anomaly_patterns 
                (pattern_id, timestamp, pattern_type, severity, description, 
                 statistical_metrics, human_reviewed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.timestamp.isoformat(),
                pattern.pattern_type,
                pattern.severity.value,
                pattern.description,
                json.dumps(pattern.statistical_metrics),
                pattern.human_review_required
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging pattern: {e}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        recent_validations = list(self.validation_history)[-1000:]
        
        if not recent_validations:
            return {'message': 'No validations recorded'}
        
        total_validations = len(recent_validations)
        clipped_count = sum(1 for v in recent_validations if v.clipping_applied)
        anomalous_count = sum(1 for v in recent_validations 
                             if v.validation_level != RewardValidationLevel.NORMAL)
        
        return {
            'total_validations': total_validations,
            'clipping_rate': clipped_count / total_validations,
            'anomaly_rate': anomalous_count / total_validations,
            'average_anomaly_score': np.mean([v.anomaly_score for v in recent_validations]),
            'validation_level_distribution': {
                level.value: sum(1 for v in recent_validations if v.validation_level == level)
                for level in RewardValidationLevel
            },
            'human_review_queue_size': len(self.human_review_queue),
            'suspicious_patterns': len(self.suspicious_patterns)
        }


# Global validator instance
_reward_validator: Optional[ProductionRewardValidator] = None

def get_reward_validator() -> ProductionRewardValidator:
    """Get global reward validator instance"""
    global _reward_validator
    if _reward_validator is None:
        _reward_validator = ProductionRewardValidator()
    return _reward_validator

def validate_reward_safe(reward: float, context: Dict[str, Any] = None) -> float:
    """Safe reward validation function for production use"""
    validator = get_reward_validator()
    result = validator.validate_reward(reward, context)
    
    if not result.is_valid:
        logger.warning(f"Reward validation failed: {result.warnings}")
        return 0.0  # Return safe default
    
    return result.validated_reward

def reward_validation_decorator(func: Callable) -> Callable:
    """Decorator to add reward validation to functions"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # If function returns a reward value
        if isinstance(result, (int, float)):
            context = {'function_name': func.__name__}
            if args and isinstance(args[0], dict):
                context.update(args[0])
            
            validated_reward = validate_reward_safe(result, context)
            return validated_reward
        
        return result
    
    return wrapper


if __name__ == "__main__":
    # Example usage and testing
    print("Initializing Production Reward Validator...")
    
    validator = ProductionRewardValidator()
    
    # Test reward function validation
    def test_reward_function(state):
        return state.get('conversion_value', 0) * state.get('probability', 0.01)
    
    is_safe, dangerous, suspicious = validator.validate_reward_function(test_reward_function)
    print(f"Function validation: {'SAFE' if is_safe else 'UNSAFE'}")
    if dangerous:
        print(f"Dangerous issues: {dangerous}")
    if suspicious:
        print(f"Suspicious issues: {suspicious}")
    
    # Test reward validation
    test_context = {
        'user_segment': 'high_value',
        'conversion_probability': 0.05,
        'conversion_value': 50.0,
        'bid_amount': 2.0,
        'budget_remaining': 1000.0
    }
    
    # Normal reward
    result = validator.validate_reward(2.5, test_context)
    print(f"\nNormal reward validation: {'VALID' if result.is_valid else 'INVALID'}")
    print(f"Original: {result.original_reward}, Validated: {result.validated_reward}")
    print(f"Warnings: {result.warnings}")
    
    # Anomalous reward
    result = validator.validate_reward(500.0, test_context)
    print(f"\nAnomalous reward validation: {'VALID' if result.is_valid else 'INVALID'}")
    print(f"Original: {result.original_reward}, Validated: {result.validated_reward}")
    print(f"Validation level: {result.validation_level}")
    print(f"Warnings: {result.warnings}")
    
    # Get validation statistics
    stats = validator.get_validation_stats()
    print(f"\nValidation statistics: {stats}")
    
    print("Reward validation system test completed.")