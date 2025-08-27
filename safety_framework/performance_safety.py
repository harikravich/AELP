"""
Performance Safety Module for GAELP Ad Campaign Safety
Implements reward clipping, performance monitoring, and anomaly detection.
"""

import logging
import asyncio
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import json
import numpy as np
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    CLICK_THROUGH_RATE = "ctr"
    CONVERSION_RATE = "conversion_rate"
    COST_PER_CLICK = "cpc"
    COST_PER_ACQUISITION = "cpa"
    RETURN_ON_AD_SPEND = "roas"
    IMPRESSIONS = "impressions"
    CLICKS = "clicks"
    CONVERSIONS = "conversions"
    SPEND = "spend"


class AnomalyType(Enum):
    SUDDEN_SPIKE = "sudden_spike"
    SUDDEN_DROP = "sudden_drop"
    SUSTAINED_ANOMALY = "sustained_anomaly"
    STATISTICAL_OUTLIER = "statistical_outlier"
    REWARD_HACKING = "reward_hacking"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class SafetyAction(Enum):
    CLIP_REWARD = "clip_reward"
    PAUSE_CAMPAIGN = "pause_campaign"
    REDUCE_BUDGET = "reduce_budget"
    HUMAN_REVIEW = "human_review"
    ROLLBACK = "rollback"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class PerformanceDataPoint:
    """Single performance measurement"""
    campaign_id: str
    metric: PerformanceMetric
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAnomaly:
    """Detected performance anomaly"""
    campaign_id: str
    anomaly_type: AnomalyType
    metric: PerformanceMetric
    current_value: float
    expected_value: Optional[float]
    confidence: float
    severity: str  # low, medium, high, critical
    description: str
    timestamp: datetime
    actions_taken: List[SafetyAction] = field(default_factory=list)


@dataclass
class RewardClipping:
    """Reward clipping configuration"""
    min_reward: float
    max_reward: float
    clip_percentile: float = 95.0  # Clip values above this percentile
    adaptive: bool = True  # Adapt clipping bounds based on data


class PerformanceMonitor:
    """Monitors campaign performance for anomalies and safety issues"""
    
    def __init__(self, history_window: int = 1000):
        self.history_window = history_window
        self.performance_data: Dict[str, Dict[PerformanceMetric, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=history_window))
        )
        self.anomalies: List[PerformanceAnomaly] = []
        self.baseline_stats: Dict[str, Dict[PerformanceMetric, Dict]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.campaign_timeouts: Dict[str, datetime] = {}
        self.max_campaign_runtime = timedelta(days=30)  # Default max runtime
    
    async def record_performance(self, data_point: PerformanceDataPoint) -> Optional[PerformanceAnomaly]:
        """Record a performance data point and check for anomalies"""
        try:
            campaign_id = data_point.campaign_id
            metric = data_point.metric
            
            # Store the data point
            self.performance_data[campaign_id][metric].append(data_point)
            
            # Update baseline statistics
            await self._update_baseline_stats(campaign_id, metric)
            
            # Check for anomalies
            anomaly = await self._detect_anomaly(data_point)
            
            if anomaly:
                self.anomalies.append(anomaly)
                logger.warning(f"Performance anomaly detected: {anomaly.description}")
            
            return anomaly
            
        except Exception as e:
            logger.error(f"Failed to record performance data: {e}")
            return None
    
    async def _update_baseline_stats(self, campaign_id: str, metric: PerformanceMetric):
        """Update baseline statistics for anomaly detection"""
        try:
            data_points = self.performance_data[campaign_id][metric]
            if len(data_points) < 10:  # Need minimum data for statistics
                return
            
            values = [dp.value for dp in data_points]
            
            self.baseline_stats[campaign_id][metric] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'q95': np.percentile(values, 95),
                'last_updated': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Failed to update baseline stats: {e}")
    
    async def _detect_anomaly(self, data_point: PerformanceDataPoint) -> Optional[PerformanceAnomaly]:
        """Detect if the data point represents an anomaly"""
        try:
            campaign_id = data_point.campaign_id
            metric = data_point.metric
            value = data_point.value
            
            # Get baseline statistics
            stats = self.baseline_stats[campaign_id].get(metric, {})
            if not stats:
                return None  # Not enough data yet
            
            mean = stats['mean']
            stdev = stats['stdev']
            
            # Statistical outlier detection (3-sigma rule)
            if stdev > 0:
                z_score = abs(value - mean) / stdev
                if z_score > 3:
                    return PerformanceAnomaly(
                        campaign_id=campaign_id,
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        metric=metric,
                        current_value=value,
                        expected_value=mean,
                        confidence=min(z_score / 3, 1.0),
                        severity=self._determine_severity(z_score),
                        description=f"{metric.value} value {value} is {z_score:.2f} standard deviations from mean {mean:.4f}",
                        timestamp=data_point.timestamp
                    )
            
            # Sudden spike detection (value > 95th percentile)
            q95 = stats.get('q95', float('inf'))
            if value > q95 * 1.5:  # 50% above 95th percentile
                return PerformanceAnomaly(
                    campaign_id=campaign_id,
                    anomaly_type=AnomalyType.SUDDEN_SPIKE,
                    metric=metric,
                    current_value=value,
                    expected_value=q95,
                    confidence=0.8,
                    severity="high",
                    description=f"{metric.value} sudden spike: {value} vs expected ~{q95:.4f}",
                    timestamp=data_point.timestamp
                )
            
            # Sudden drop detection (value < 5th percentile)
            q5 = np.percentile([dp.value for dp in self.performance_data[campaign_id][metric]], 5)
            if value < q5 * 0.5:  # 50% below 5th percentile
                return PerformanceAnomaly(
                    campaign_id=campaign_id,
                    anomaly_type=AnomalyType.SUDDEN_DROP,
                    metric=metric,
                    current_value=value,
                    expected_value=q5,
                    confidence=0.8,
                    severity="medium",
                    description=f"{metric.value} sudden drop: {value} vs expected ~{q5:.4f}",
                    timestamp=data_point.timestamp
                )
            
            # Check for reward hacking patterns
            reward_hack_anomaly = await self._detect_reward_hacking(data_point)
            if reward_hack_anomaly:
                return reward_hack_anomaly
            
            return None
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return None
    
    async def _detect_reward_hacking(self, data_point: PerformanceDataPoint) -> Optional[PerformanceAnomaly]:
        """Detect potential reward hacking patterns"""
        try:
            campaign_id = data_point.campaign_id
            metric = data_point.metric
            
            # Get recent data points for this metric
            recent_data = list(self.performance_data[campaign_id][metric])[-10:]
            if len(recent_data) < 5:
                return None
            
            recent_values = [dp.value for dp in recent_data]
            
            # Check for unrealistic performance improvements
            if metric in [PerformanceMetric.CLICK_THROUGH_RATE, PerformanceMetric.CONVERSION_RATE]:
                # CTR or conversion rate shouldn't exceed realistic bounds
                if data_point.value > 0.5:  # 50% CTR is unrealistic
                    return PerformanceAnomaly(
                        campaign_id=campaign_id,
                        anomaly_type=AnomalyType.REWARD_HACKING,
                        metric=metric,
                        current_value=data_point.value,
                        expected_value=None,
                        confidence=0.9,
                        severity="critical",
                        description=f"Unrealistic {metric.value}: {data_point.value:.4f} (>50%)",
                        timestamp=data_point.timestamp
                    )
            
            # Check for sudden, sustained improvements (possible manipulation)
            if len(recent_values) >= 5:
                baseline_avg = statistics.mean(recent_values[:-3])
                recent_avg = statistics.mean(recent_values[-3:])
                
                if recent_avg > baseline_avg * 5:  # 5x improvement is suspicious
                    return PerformanceAnomaly(
                        campaign_id=campaign_id,
                        anomaly_type=AnomalyType.REWARD_HACKING,
                        metric=metric,
                        current_value=data_point.value,
                        expected_value=baseline_avg,
                        confidence=0.7,
                        severity="high",
                        description=f"Suspicious performance improvement: {recent_avg:.4f} vs baseline {baseline_avg:.4f}",
                        timestamp=data_point.timestamp
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Reward hacking detection failed: {e}")
            return None
    
    def _determine_severity(self, z_score: float) -> str:
        """Determine anomaly severity based on z-score"""
        if z_score > 5:
            return "critical"
        elif z_score > 4:
            return "high"
        elif z_score > 3:
            return "medium"
        else:
            return "low"
    
    def get_campaign_performance_summary(self, campaign_id: str) -> Dict[str, Any]:
        """Get performance summary for a campaign"""
        try:
            summary = {}
            
            for metric, data_points in self.performance_data[campaign_id].items():
                if not data_points:
                    continue
                
                values = [dp.value for dp in data_points]
                summary[metric.value] = {
                    'current': values[-1] if values else None,
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'trend': self._calculate_trend(values),
                    'data_points': len(values)
                }
            
            # Add anomaly information
            campaign_anomalies = [a for a in self.anomalies if a.campaign_id == campaign_id]
            summary['anomalies'] = {
                'total': len(campaign_anomalies),
                'critical': len([a for a in campaign_anomalies if a.severity == "critical"]),
                'recent': len([a for a in campaign_anomalies 
                             if a.timestamp > datetime.utcnow() - timedelta(hours=24)])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple trend calculation using linear regression slope
        x = list(range(len(values)))
        n = len(values)
        
        slope = (n * sum(i * values[i] for i in range(n)) - sum(x) * sum(values)) / \
                (n * sum(i * i for i in range(n)) - sum(x) ** 2)
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    async def check_campaign_timeout(self, campaign_id: str) -> bool:
        """Check if campaign has exceeded maximum runtime"""
        if campaign_id not in self.campaign_timeouts:
            self.campaign_timeouts[campaign_id] = datetime.utcnow()
            return False
        
        runtime = datetime.utcnow() - self.campaign_timeouts[campaign_id]
        return runtime > self.max_campaign_runtime


class RewardClipper:
    """Implements reward clipping to prevent exploitation"""
    
    def __init__(self, initial_config: RewardClipping):
        self.config = initial_config
        self.reward_history: deque = deque(maxlen=10000)
        self.clipping_stats = {
            'total_rewards': 0,
            'clipped_rewards': 0,
            'clipped_high': 0,
            'clipped_low': 0
        }
    
    def clip_reward(self, reward: float, campaign_id: str = None) -> Tuple[float, bool]:
        """
        Clip reward value to safe bounds.
        Returns (clipped_reward, was_clipped)
        """
        try:
            original_reward = reward
            was_clipped = False
            
            # Update adaptive bounds if enabled
            if self.config.adaptive and len(self.reward_history) > 100:
                self._update_adaptive_bounds()
            
            # Apply clipping
            if reward > self.config.max_reward:
                reward = self.config.max_reward
                was_clipped = True
                self.clipping_stats['clipped_high'] += 1
                logger.debug(f"Reward clipped high: {original_reward} -> {reward}")
            
            elif reward < self.config.min_reward:
                reward = self.config.min_reward
                was_clipped = True
                self.clipping_stats['clipped_low'] += 1
                logger.debug(f"Reward clipped low: {original_reward} -> {reward}")
            
            # Update statistics
            self.reward_history.append(original_reward)
            self.clipping_stats['total_rewards'] += 1
            if was_clipped:
                self.clipping_stats['clipped_rewards'] += 1
            
            return reward, was_clipped
            
        except Exception as e:
            logger.error(f"Reward clipping failed: {e}")
            return reward, False
    
    def _update_adaptive_bounds(self):
        """Update clipping bounds based on reward distribution"""
        try:
            if not self.reward_history:
                return
            
            rewards = list(self.reward_history)
            
            # Calculate percentile-based bounds
            lower_bound = np.percentile(rewards, 5)  # 5th percentile
            upper_bound = np.percentile(rewards, self.config.clip_percentile)
            
            # Add some margin to prevent over-clipping
            margin = (upper_bound - lower_bound) * 0.1
            
            self.config.min_reward = max(self.config.min_reward, lower_bound - margin)
            self.config.max_reward = min(self.config.max_reward, upper_bound + margin)
            
            logger.debug(f"Updated adaptive bounds: [{self.config.min_reward:.4f}, {self.config.max_reward:.4f}]")
            
        except Exception as e:
            logger.error(f"Failed to update adaptive bounds: {e}")
    
    def get_clipping_stats(self) -> Dict[str, Any]:
        """Get reward clipping statistics"""
        total = max(self.clipping_stats['total_rewards'], 1)
        
        return {
            **self.clipping_stats,
            'clipping_rate': self.clipping_stats['clipped_rewards'] / total,
            'high_clipping_rate': self.clipping_stats['clipped_high'] / total,
            'low_clipping_rate': self.clipping_stats['clipped_low'] / total,
            'current_bounds': {
                'min': self.config.min_reward,
                'max': self.config.max_reward
            },
            'reward_distribution': {
                'mean': statistics.mean(self.reward_history) if self.reward_history else 0,
                'median': statistics.median(self.reward_history) if self.reward_history else 0,
                'std': statistics.stdev(self.reward_history) if len(self.reward_history) > 1 else 0
            }
        }


class ABTestValidator:
    """Validates A/B test statistical significance and safety"""
    
    def __init__(self, min_sample_size: int = 1000, significance_level: float = 0.05):
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.active_tests: Dict[str, Dict] = {}
    
    async def validate_ab_test(self, test_id: str, control_data: List[float], 
                             treatment_data: List[float]) -> Dict[str, Any]:
        """Validate A/B test for statistical significance and safety"""
        try:
            validation_result = {
                'test_id': test_id,
                'is_valid': False,
                'is_significant': False,
                'is_safe': True,
                'sample_sizes': {
                    'control': len(control_data),
                    'treatment': len(treatment_data)
                },
                'statistics': {},
                'warnings': [],
                'recommendations': []
            }
            
            # Check minimum sample size
            if len(control_data) < self.min_sample_size or len(treatment_data) < self.min_sample_size:
                validation_result['warnings'].append(
                    f"Insufficient sample size. Need at least {self.min_sample_size} samples per group."
                )
                validation_result['recommendations'].append("Continue collecting data before drawing conclusions")
                return validation_result
            
            # Calculate basic statistics
            control_mean = statistics.mean(control_data)
            treatment_mean = statistics.mean(treatment_data)
            control_std = statistics.stdev(control_data) if len(control_data) > 1 else 0
            treatment_std = statistics.stdev(treatment_data) if len(treatment_data) > 1 else 0
            
            validation_result['statistics'] = {
                'control_mean': control_mean,
                'treatment_mean': treatment_mean,
                'control_std': control_std,
                'treatment_std': treatment_std,
                'effect_size': (treatment_mean - control_mean) / max(control_std, 0.001),
                'relative_improvement': (treatment_mean - control_mean) / max(control_mean, 0.001)
            }
            
            # Perform t-test (simplified)
            pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
            t_stat = (treatment_mean - control_mean) / (pooled_std * np.sqrt(2/len(control_data)))
            
            # Simplified p-value calculation (normally would use scipy.stats)
            # For demonstration purposes, using a basic threshold
            p_value = 2 * (1 - min(abs(t_stat) / 2, 0.99))  # Simplified approximation
            
            validation_result['statistics']['t_statistic'] = t_stat
            validation_result['statistics']['p_value'] = p_value
            validation_result['is_significant'] = p_value < self.significance_level
            
            # Safety checks
            if validation_result['statistics']['relative_improvement'] > 5.0:  # 500% improvement
                validation_result['is_safe'] = False
                validation_result['warnings'].append("Unrealistic improvement detected - possible data quality issue")
            
            if validation_result['statistics']['relative_improvement'] < -0.5:  # 50% degradation
                validation_result['is_safe'] = False
                validation_result['warnings'].append("Significant performance degradation detected")
            
            # Overall validation
            validation_result['is_valid'] = (
                validation_result['is_safe'] and
                len(control_data) >= self.min_sample_size and
                len(treatment_data) >= self.min_sample_size
            )
            
            # Generate recommendations
            if validation_result['is_significant'] and validation_result['is_safe']:
                validation_result['recommendations'].append("Test shows significant results - consider implementation")
            elif not validation_result['is_significant']:
                validation_result['recommendations'].append("No significant difference detected - consider longer test duration")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"A/B test validation failed: {e}")
            return {
                'test_id': test_id,
                'is_valid': False,
                'error': str(e)
            }


class PerformanceSafetyOrchestrator:
    """Main orchestrator for performance safety"""
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        self.monitor = PerformanceMonitor()
        self.reward_clipper = RewardClipper(RewardClipping(
            min_reward=-10.0,
            max_reward=10.0,
            clip_percentile=95.0,
            adaptive=True
        ))
        self.ab_validator = ABTestValidator()
        self.alert_callback = alert_callback
        
        self.safety_actions_taken: List[Tuple[str, SafetyAction, datetime]] = []
    
    async def process_performance_data(self, data_point: PerformanceDataPoint) -> Dict[str, Any]:
        """Process performance data with safety checks"""
        result = {
            'data_recorded': False,
            'anomaly_detected': False,
            'actions_taken': [],
            'reward_clipped': False,
            'campaign_status': 'active'
        }
        
        try:
            # Record performance data and check for anomalies
            anomaly = await self.monitor.record_performance(data_point)
            result['data_recorded'] = True
            
            if anomaly:
                result['anomaly_detected'] = True
                result['anomaly'] = {
                    'type': anomaly.anomaly_type.value,
                    'metric': anomaly.metric.value,
                    'severity': anomaly.severity,
                    'description': anomaly.description
                }
                
                # Take appropriate safety actions
                actions = await self._handle_anomaly(anomaly)
                result['actions_taken'] = [action.value for action in actions]
                anomaly.actions_taken = actions
            
            # Check for campaign timeout
            timeout_exceeded = await self.monitor.check_campaign_timeout(data_point.campaign_id)
            if timeout_exceeded:
                result['actions_taken'].append('campaign_timeout')
                result['campaign_status'] = 'timed_out'
                await self._execute_safety_action(data_point.campaign_id, SafetyAction.PAUSE_CAMPAIGN)
            
            return result
            
        except Exception as e:
            logger.error(f"Performance data processing failed: {e}")
            result['error'] = str(e)
            return result
    
    async def clip_reward(self, reward: float, campaign_id: str) -> Tuple[float, Dict[str, Any]]:
        """Clip reward with safety information"""
        clipped_reward, was_clipped = self.reward_clipper.clip_reward(reward, campaign_id)
        
        return clipped_reward, {
            'original_reward': reward,
            'clipped_reward': clipped_reward,
            'was_clipped': was_clipped,
            'clipping_stats': self.reward_clipper.get_clipping_stats()
        }
    
    async def _handle_anomaly(self, anomaly: PerformanceAnomaly) -> List[SafetyAction]:
        """Handle detected anomalies with appropriate safety actions"""
        actions = []
        
        try:
            if anomaly.severity == "critical":
                if anomaly.anomaly_type == AnomalyType.REWARD_HACKING:
                    actions.extend([SafetyAction.PAUSE_CAMPAIGN, SafetyAction.HUMAN_REVIEW])
                else:
                    actions.extend([SafetyAction.EMERGENCY_STOP, SafetyAction.HUMAN_REVIEW])
            
            elif anomaly.severity == "high":
                if anomaly.anomaly_type == AnomalyType.SUDDEN_SPIKE:
                    actions.extend([SafetyAction.CLIP_REWARD, SafetyAction.REDUCE_BUDGET])
                elif anomaly.anomaly_type == AnomalyType.SUDDEN_DROP:
                    actions.extend([SafetyAction.PAUSE_CAMPAIGN, SafetyAction.HUMAN_REVIEW])
                else:
                    actions.append(SafetyAction.HUMAN_REVIEW)
            
            elif anomaly.severity == "medium":
                actions.append(SafetyAction.CLIP_REWARD)
            
            # Execute the actions
            for action in actions:
                await self._execute_safety_action(anomaly.campaign_id, action)
            
            # Send alert if callback configured
            if self.alert_callback:
                await self.alert_callback({
                    'type': 'performance_anomaly',
                    'anomaly': anomaly,
                    'actions_taken': actions
                })
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to handle anomaly: {e}")
            return []
    
    async def _execute_safety_action(self, campaign_id: str, action: SafetyAction):
        """Execute a specific safety action"""
        try:
            self.safety_actions_taken.append((campaign_id, action, datetime.utcnow()))
            
            if action == SafetyAction.PAUSE_CAMPAIGN:
                # This would integrate with the actual campaign management system
                logger.critical(f"SAFETY ACTION: Pausing campaign {campaign_id}")
            
            elif action == SafetyAction.EMERGENCY_STOP:
                logger.critical(f"SAFETY ACTION: Emergency stop for campaign {campaign_id}")
            
            elif action == SafetyAction.REDUCE_BUDGET:
                logger.warning(f"SAFETY ACTION: Reducing budget for campaign {campaign_id}")
            
            elif action == SafetyAction.HUMAN_REVIEW:
                logger.warning(f"SAFETY ACTION: Flagging campaign {campaign_id} for human review")
            
            elif action == SafetyAction.ROLLBACK:
                logger.warning(f"SAFETY ACTION: Rolling back campaign {campaign_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute safety action {action}: {e}")
    
    def get_safety_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive safety dashboard data"""
        try:
            return {
                'performance_monitoring': {
                    'active_campaigns': len(self.monitor.performance_data),
                    'total_anomalies': len(self.monitor.anomalies),
                    'critical_anomalies': len([a for a in self.monitor.anomalies if a.severity == "critical"]),
                    'recent_anomalies': len([
                        a for a in self.monitor.anomalies 
                        if a.timestamp > datetime.utcnow() - timedelta(hours=24)
                    ])
                },
                'reward_clipping': self.reward_clipper.get_clipping_stats(),
                'safety_actions': {
                    'total_actions': len(self.safety_actions_taken),
                    'recent_actions': len([
                        action for _, _, timestamp in self.safety_actions_taken
                        if timestamp > datetime.utcnow() - timedelta(hours=24)
                    ]),
                    'action_breakdown': self._get_action_breakdown()
                },
                'system_health': {
                    'monitoring_active': True,
                    'last_updated': datetime.utcnow(),
                    'alerts_sent': 0  # Would track actual alerts
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate safety dashboard: {e}")
            return {}
    
    def _get_action_breakdown(self) -> Dict[str, int]:
        """Get breakdown of safety actions taken"""
        breakdown = {}
        for _, action, _ in self.safety_actions_taken:
            action_name = action.value
            breakdown[action_name] = breakdown.get(action_name, 0) + 1
        return breakdown