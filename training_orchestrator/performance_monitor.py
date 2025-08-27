"""
Performance Monitoring and Analysis

Tracks agent performance across training phases, analyzes trends,
and determines graduation criteria for phase transitions.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np
from scipy import stats

from .phases import TrainingPhase


class PerformanceMetric(Enum):
    """Core performance metrics for monitoring"""
    REWARD = "reward"
    SUCCESS_RATE = "success_rate"
    ROI = "roi"
    BUDGET_EFFICIENCY = "budget_efficiency"
    CLICK_THROUGH_RATE = "click_through_rate"
    CONVERSION_RATE = "conversion_rate"
    COST_PER_ACQUISITION = "cost_per_acquisition"
    BRAND_SAFETY_SCORE = "brand_safety_score"
    AUDIENCE_ENGAGEMENT = "audience_engagement"


@dataclass
class PerformanceWindow:
    """Performance data for a specific time window"""
    start_time: datetime
    end_time: datetime
    episode_count: int
    metrics: Dict[PerformanceMetric, List[float]] = field(default_factory=dict)
    statistics: Dict[PerformanceMetric, Dict[str, float]] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Analysis of performance trends"""
    metric: PerformanceMetric
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float  # 0.0 to 1.0
    slope: float
    r_squared: float
    confidence: float
    projection_10_episodes: float
    projection_50_episodes: float


@dataclass
class GraduationAssessment:
    """Assessment of graduation readiness"""
    phase: TrainingPhase
    ready_to_graduate: bool
    confidence: float
    criteria_met: Dict[str, bool]
    criteria_scores: Dict[str, float]
    missing_requirements: List[str]
    recommendation: str


class PerformanceMonitor:
    """
    Monitors agent performance across training phases and provides
    analysis for curriculum progression and graduation decisions.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_history: Dict[TrainingPhase, List[Dict[str, float]]] = {
            phase: [] for phase in TrainingPhase
        }
        
        # Window-based analysis
        self.performance_windows: Dict[TrainingPhase, List[PerformanceWindow]] = {
            phase: [] for phase in TrainingPhase
        }
        
        # Trend analysis
        self.trend_analyses: Dict[TrainingPhase, Dict[PerformanceMetric, TrendAnalysis]] = {
            phase: {} for phase in TrainingPhase
        }
        
        # Graduation thresholds
        self.graduation_thresholds = self._initialize_graduation_thresholds()
        
        self.logger.info("Performance Monitor initialized")
    
    def _initialize_graduation_thresholds(self) -> Dict[TrainingPhase, Dict[str, float]]:
        """Initialize graduation thresholds for each phase"""
        
        return {
            TrainingPhase.SIMULATION: {
                "min_average_reward": 0.7,
                "min_success_rate": 0.8,
                "performance_improvement_rate": 0.05,
                "consistency_window": 50,
                "min_episodes": 100,
                "trend_stability_episodes": 30
            },
            
            TrainingPhase.HISTORICAL_VALIDATION: {
                "historical_performance_match": 0.95,  # 95% of known good campaigns
                "prediction_accuracy": 0.85,
                "correlation_with_actual": 0.8,
                "min_validation_episodes": 50,
                "confidence_threshold": 0.9
            },
            
            TrainingPhase.REAL_TESTING: {
                "consecutive_positive_roi": 5,
                "min_roi_threshold": 0.05,  # 5% ROI minimum
                "budget_efficiency": 0.8,
                "safety_compliance": 1.0,  # Perfect safety compliance
                "min_real_episodes": 20
            },
            
            TrainingPhase.SCALED_DEPLOYMENT: {
                "sustained_performance": 0.15,  # 15% improvement threshold
                "multi_campaign_success": 0.9,
                "transfer_learning_effectiveness": 0.8,
                "operational_stability": 0.95,
                "min_deployment_episodes": 100
            }
        }
    
    def record_episode_performance(self, 
                                  phase: TrainingPhase,
                                  episode_metrics: Dict[str, float]):
        """Record performance metrics for an episode"""
        
        # Add timestamp
        episode_data = episode_metrics.copy()
        episode_data["timestamp"] = datetime.now().timestamp()
        episode_data["episode_id"] = len(self.performance_history[phase])
        
        # Store in history
        self.performance_history[phase].append(episode_data)
        
        # Update performance windows
        self._update_performance_windows(phase, episode_data)
        
        # Update trend analysis
        self._update_trend_analysis(phase)
        
        self.logger.debug(f"Recorded performance for {phase.value}: {episode_metrics}")
    
    def _update_performance_windows(self, 
                                   phase: TrainingPhase, 
                                   episode_data: Dict[str, float]):
        """Update sliding window performance analysis"""
        
        window_sizes = [10, 25, 50, 100]  # Different window sizes for analysis
        
        for window_size in window_sizes:
            recent_episodes = self.performance_history[phase][-window_size:]
            
            if len(recent_episodes) >= min(window_size, 10):  # Minimum 10 episodes
                window = PerformanceWindow(
                    start_time=datetime.fromtimestamp(recent_episodes[0]["timestamp"]),
                    end_time=datetime.fromtimestamp(recent_episodes[-1]["timestamp"]),
                    episode_count=len(recent_episodes)
                )
                
                # Calculate metrics for this window
                for metric in PerformanceMetric:
                    metric_values = [
                        ep.get(metric.value, 0.0) for ep in recent_episodes
                        if metric.value in ep
                    ]
                    
                    if metric_values:
                        window.metrics[metric] = metric_values
                        window.statistics[metric] = {
                            "mean": np.mean(metric_values),
                            "std": np.std(metric_values),
                            "min": np.min(metric_values),
                            "max": np.max(metric_values),
                            "median": np.median(metric_values),
                            "q25": np.percentile(metric_values, 25),
                            "q75": np.percentile(metric_values, 75)
                        }
                
                # Store window (keep only recent windows)
                if phase not in self.performance_windows:
                    self.performance_windows[phase] = []
                
                self.performance_windows[phase].append(window)
                
                # Keep only last 10 windows per size
                if len(self.performance_windows[phase]) > 10:
                    self.performance_windows[phase] = self.performance_windows[phase][-10:]
    
    def _update_trend_analysis(self, phase: TrainingPhase):
        """Update trend analysis for performance metrics"""
        
        if len(self.performance_history[phase]) < 10:
            return  # Need minimum data for trend analysis
        
        recent_episodes = self.performance_history[phase][-50:]  # Last 50 episodes
        episode_indices = list(range(len(recent_episodes)))
        
        for metric in PerformanceMetric:
            metric_values = [
                ep.get(metric.value, 0.0) for ep in recent_episodes
                if metric.value in ep
            ]
            
            if len(metric_values) >= 10:
                trend_analysis = self._calculate_trend(episode_indices, metric_values, metric)
                self.trend_analyses[phase][metric] = trend_analysis
    
    def _calculate_trend(self, 
                        x_values: List[int], 
                        y_values: List[float], 
                        metric: PerformanceMetric) -> TrendAnalysis:
        """Calculate trend analysis for a metric"""
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        r_squared = r_value ** 2
        
        # Determine trend direction and strength
        if abs(slope) < 0.001:  # Very small slope
            trend_direction = "stable"
            trend_strength = 1.0 - abs(slope) * 100  # Closer to 0 slope = more stable
        elif slope > 0:
            trend_direction = "improving"
            trend_strength = min(1.0, abs(slope) * 10)  # Scale appropriately
        else:
            trend_direction = "declining"
            trend_strength = min(1.0, abs(slope) * 10)
        
        # Calculate confidence based on R-squared and p-value
        confidence = r_squared * (1.0 - p_value)
        
        # Project future performance
        last_x = x_values[-1]
        projection_10 = slope * (last_x + 10) + intercept
        projection_50 = slope * (last_x + 50) + intercept
        
        return TrendAnalysis(
            metric=metric,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            r_squared=r_squared,
            confidence=confidence,
            projection_10_episodes=projection_10,
            projection_50_episodes=projection_50
        )
    
    def check_phase_progression(self, 
                               phase: TrainingPhase,
                               performance_history: List[float]) -> bool:
        """
        Check if agent should continue in current phase or if ready to progress
        
        Args:
            phase: Current training phase
            performance_history: Recent performance scores
            
        Returns:
            bool: True if should continue, False if should stop/reassess
        """
        
        if len(performance_history) < 10:
            return True  # Continue until sufficient data
        
        # Check for concerning trends
        recent_performance = performance_history[-10:]
        trend_slope = self._calculate_simple_trend(recent_performance)
        
        # If performance is declining significantly, may need intervention
        if trend_slope < -0.1:  # Declining by more than 10% per episode
            self.logger.warning(f"Declining performance detected in {phase.value}")
            return False
        
        # Check for stagnation
        if len(performance_history) >= 100:
            recent_50 = performance_history[-50:]
            previous_50 = performance_history[-100:-50]
            
            recent_avg = np.mean(recent_50)
            previous_avg = np.mean(previous_50)
            
            improvement = (recent_avg - previous_avg) / previous_avg
            if improvement < 0.01:  # Less than 1% improvement
                self.logger.warning(f"Performance stagnation detected in {phase.value}")
                return False
        
        return True
    
    def check_graduation_criteria(self, 
                                 phase: TrainingPhase,
                                 performance_history: List[float]) -> bool:
        """
        Check if agent meets graduation criteria for current phase
        
        Args:
            phase: Current training phase
            performance_history: Performance history for the phase
            
        Returns:
            bool: True if graduation criteria are met
        """
        
        thresholds = self.graduation_thresholds[phase]
        
        if phase == TrainingPhase.SIMULATION:
            return self._check_simulation_graduation(performance_history, thresholds)
        elif phase == TrainingPhase.HISTORICAL_VALIDATION:
            return self._check_historical_graduation(thresholds)
        elif phase == TrainingPhase.REAL_TESTING:
            return self._check_real_testing_graduation(thresholds)
        elif phase == TrainingPhase.SCALED_DEPLOYMENT:
            return self._check_scaled_deployment_graduation(thresholds)
        
        return False
    
    def _check_simulation_graduation(self, 
                                   performance_history: List[float],
                                   thresholds: Dict[str, float]) -> bool:
        """Check simulation phase graduation criteria"""
        
        if len(performance_history) < thresholds["min_episodes"]:
            return False
        
        # Check average reward over recent window
        window_size = min(thresholds["consistency_window"], len(performance_history))
        recent_performance = performance_history[-window_size:]
        avg_reward = np.mean(recent_performance)
        
        if avg_reward < thresholds["min_average_reward"]:
            return False
        
        # Check success rate (assuming binary success/failure)
        success_count = sum(1 for p in recent_performance if p > 0.5)
        success_rate = success_count / len(recent_performance)
        
        if success_rate < thresholds["min_success_rate"]:
            return False
        
        # Check performance improvement trend
        if len(performance_history) >= thresholds["trend_stability_episodes"] * 2:
            early_window = performance_history[-thresholds["trend_stability_episodes"]*2:-thresholds["trend_stability_episodes"]]
            late_window = performance_history[-thresholds["trend_stability_episodes"]:]
            
            early_avg = np.mean(early_window)
            late_avg = np.mean(late_window)
            
            improvement_rate = (late_avg - early_avg) / early_avg if early_avg > 0 else 0
            
            if improvement_rate < thresholds["performance_improvement_rate"]:
                return False
        
        return True
    
    def _check_historical_graduation(self, thresholds: Dict[str, float]) -> bool:
        """Check historical validation graduation criteria"""
        
        # This would check against historical benchmark data
        # For now, return a simplified check
        phase_history = self.performance_history[TrainingPhase.HISTORICAL_VALIDATION]
        
        if len(phase_history) < thresholds["min_validation_episodes"]:
            return False
        
        # Check validation metrics from recent episodes
        recent_episodes = phase_history[-10:]
        
        # Extract historical performance metrics
        historical_matches = [ep.get("historical_performance_match", 0.0) for ep in recent_episodes]
        prediction_accuracies = [ep.get("prediction_accuracy", 0.0) for ep in recent_episodes]
        correlations = [ep.get("correlation_with_actual", 0.0) for ep in recent_episodes]
        
        if historical_matches and np.mean(historical_matches) < thresholds["historical_performance_match"]:
            return False
        
        if prediction_accuracies and np.mean(prediction_accuracies) < thresholds["prediction_accuracy"]:
            return False
        
        if correlations and np.mean(correlations) < thresholds["correlation_with_actual"]:
            return False
        
        return True
    
    def _check_real_testing_graduation(self, thresholds: Dict[str, float]) -> bool:
        """Check real testing graduation criteria"""
        
        phase_history = self.performance_history[TrainingPhase.REAL_TESTING]
        
        if len(phase_history) < thresholds["min_real_episodes"]:
            return False
        
        # Check consecutive positive ROI
        recent_rois = [ep.get("roi", 0.0) for ep in phase_history[-10:]]
        consecutive_positive = 0
        max_consecutive = 0
        
        for roi in reversed(recent_rois):
            if roi > thresholds["min_roi_threshold"]:
                consecutive_positive += 1
                max_consecutive = max(max_consecutive, consecutive_positive)
            else:
                consecutive_positive = 0
        
        if max_consecutive < thresholds["consecutive_positive_roi"]:
            return False
        
        # Check budget efficiency
        budget_efficiencies = [ep.get("budget_efficiency", 0.0) for ep in phase_history[-10:]]
        if budget_efficiencies and np.mean(budget_efficiencies) < thresholds["budget_efficiency"]:
            return False
        
        # Check safety compliance
        safety_scores = [ep.get("safety_compliance", 0.0) for ep in phase_history[-10:]]
        if safety_scores and np.mean(safety_scores) < thresholds["safety_compliance"]:
            return False
        
        return True
    
    def _check_scaled_deployment_graduation(self, thresholds: Dict[str, float]) -> bool:
        """Check scaled deployment graduation criteria"""
        
        phase_history = self.performance_history[TrainingPhase.SCALED_DEPLOYMENT]
        
        if len(phase_history) < thresholds["min_deployment_episodes"]:
            return False
        
        # Check sustained performance improvement
        sustained_performances = [ep.get("sustained_performance", 0.0) for ep in phase_history[-20:]]
        if sustained_performances and np.mean(sustained_performances) < thresholds["sustained_performance"]:
            return False
        
        # Check multi-campaign success
        multi_campaign_successes = [ep.get("multi_campaign_success", 0.0) for ep in phase_history[-20:]]
        if multi_campaign_successes and np.mean(multi_campaign_successes) < thresholds["multi_campaign_success"]:
            return False
        
        # Check transfer learning effectiveness
        transfer_learning_scores = [ep.get("transfer_learning_effectiveness", 0.0) for ep in phase_history[-20:]]
        if transfer_learning_scores and np.mean(transfer_learning_scores) < thresholds["transfer_learning_effectiveness"]:
            return False
        
        return True
    
    def validate_historical_performance(self, 
                                       validation_results: List[Dict[str, float]]) -> bool:
        """
        Validate performance against historical benchmarks
        
        Args:
            validation_results: List of validation metrics from episodes
            
        Returns:
            bool: True if validation passes
        """
        
        if not validation_results:
            return False
        
        # Extract key validation metrics
        historical_matches = [result.get("historical_performance_match", 0.0) for result in validation_results]
        prediction_accuracies = [result.get("prediction_accuracy", 0.0) for result in validation_results]
        correlations = [result.get("correlation_with_actual", 0.0) for result in validation_results]
        
        thresholds = self.graduation_thresholds[TrainingPhase.HISTORICAL_VALIDATION]
        
        # Check each metric
        if np.mean(historical_matches) < thresholds["historical_performance_match"]:
            self.logger.warning(f"Historical performance match insufficient: {np.mean(historical_matches):.3f}")
            return False
        
        if np.mean(prediction_accuracies) < thresholds["prediction_accuracy"]:
            self.logger.warning(f"Prediction accuracy insufficient: {np.mean(prediction_accuracies):.3f}")
            return False
        
        if np.mean(correlations) < thresholds["correlation_with_actual"]:
            self.logger.warning(f"Correlation with actual insufficient: {np.mean(correlations):.3f}")
            return False
        
        self.logger.info("Historical validation passed successfully")
        return True
    
    def get_graduation_assessment(self, phase: TrainingPhase) -> GraduationAssessment:
        """Get detailed graduation assessment for a phase"""
        
        thresholds = self.graduation_thresholds[phase]
        phase_history = self.performance_history[phase]
        
        # Initialize assessment
        assessment = GraduationAssessment(
            phase=phase,
            ready_to_graduate=False,
            confidence=0.0,
            criteria_met={},
            criteria_scores={},
            missing_requirements=[],
            recommendation=""
        )
        
        if not phase_history:
            assessment.recommendation = "No performance data available"
            return assessment
        
        # Phase-specific assessment
        if phase == TrainingPhase.SIMULATION:
            self._assess_simulation_graduation(assessment, phase_history, thresholds)
        elif phase == TrainingPhase.HISTORICAL_VALIDATION:
            self._assess_historical_graduation(assessment, phase_history, thresholds)
        elif phase == TrainingPhase.REAL_TESTING:
            self._assess_real_testing_graduation(assessment, phase_history, thresholds)
        elif phase == TrainingPhase.SCALED_DEPLOYMENT:
            self._assess_scaled_deployment_graduation(assessment, phase_history, thresholds)
        
        # Calculate overall confidence
        met_criteria = sum(assessment.criteria_met.values())
        total_criteria = len(assessment.criteria_met)
        assessment.confidence = met_criteria / total_criteria if total_criteria > 0 else 0.0
        
        # Determine if ready to graduate
        assessment.ready_to_graduate = all(assessment.criteria_met.values())
        
        # Generate recommendation
        if assessment.ready_to_graduate:
            assessment.recommendation = f"Ready to graduate from {phase.value}"
        else:
            missing_count = len(assessment.missing_requirements)
            assessment.recommendation = f"Not ready to graduate. {missing_count} criteria not met."
        
        return assessment
    
    def _assess_simulation_graduation(self, 
                                     assessment: GraduationAssessment,
                                     phase_history: List[Dict[str, float]],
                                     thresholds: Dict[str, float]):
        """Assess simulation phase graduation"""
        
        # Episode count check
        episode_count = len(phase_history)
        assessment.criteria_met["min_episodes"] = episode_count >= thresholds["min_episodes"]
        assessment.criteria_scores["episode_count"] = episode_count
        
        if episode_count < thresholds["min_episodes"]:
            assessment.missing_requirements.append(f"Need {thresholds['min_episodes'] - episode_count} more episodes")
        
        if episode_count >= 10:  # Need minimum data for other checks
            recent_rewards = [ep.get("reward", 0.0) for ep in phase_history[-50:]]
            
            # Average reward check
            avg_reward = np.mean(recent_rewards)
            assessment.criteria_met["average_reward"] = avg_reward >= thresholds["min_average_reward"]
            assessment.criteria_scores["average_reward"] = avg_reward
            
            if avg_reward < thresholds["min_average_reward"]:
                assessment.missing_requirements.append(f"Average reward {avg_reward:.3f} below {thresholds['min_average_reward']}")
            
            # Success rate check
            success_count = sum(1 for r in recent_rewards if r > 0.5)
            success_rate = success_count / len(recent_rewards)
            assessment.criteria_met["success_rate"] = success_rate >= thresholds["min_success_rate"]
            assessment.criteria_scores["success_rate"] = success_rate
            
            if success_rate < thresholds["min_success_rate"]:
                assessment.missing_requirements.append(f"Success rate {success_rate:.3f} below {thresholds['min_success_rate']}")
    
    def _assess_historical_graduation(self, 
                                     assessment: GraduationAssessment,
                                     phase_history: List[Dict[str, float]],
                                     thresholds: Dict[str, float]):
        """Assess historical validation graduation"""
        
        episode_count = len(phase_history)
        assessment.criteria_met["min_episodes"] = episode_count >= thresholds["min_validation_episodes"]
        assessment.criteria_scores["episode_count"] = episode_count
        
        if episode_count >= 10:
            recent_episodes = phase_history[-10:]
            
            # Historical performance match
            matches = [ep.get("historical_performance_match", 0.0) for ep in recent_episodes]
            avg_match = np.mean(matches) if matches else 0.0
            assessment.criteria_met["historical_match"] = avg_match >= thresholds["historical_performance_match"]
            assessment.criteria_scores["historical_match"] = avg_match
            
            # Prediction accuracy
            accuracies = [ep.get("prediction_accuracy", 0.0) for ep in recent_episodes]
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            assessment.criteria_met["prediction_accuracy"] = avg_accuracy >= thresholds["prediction_accuracy"]
            assessment.criteria_scores["prediction_accuracy"] = avg_accuracy
    
    def _assess_real_testing_graduation(self, 
                                       assessment: GraduationAssessment,
                                       phase_history: List[Dict[str, float]],
                                       thresholds: Dict[str, float]):
        """Assess real testing graduation"""
        
        episode_count = len(phase_history)
        assessment.criteria_met["min_episodes"] = episode_count >= thresholds["min_real_episodes"]
        assessment.criteria_scores["episode_count"] = episode_count
        
        if episode_count >= 5:
            recent_episodes = phase_history[-10:]
            
            # ROI checks
            rois = [ep.get("roi", 0.0) for ep in recent_episodes]
            avg_roi = np.mean(rois) if rois else 0.0
            assessment.criteria_met["roi_threshold"] = avg_roi >= thresholds["min_roi_threshold"]
            assessment.criteria_scores["average_roi"] = avg_roi
            
            # Consecutive positive ROI
            consecutive_positive = self._count_consecutive_positive_roi(rois, thresholds["min_roi_threshold"])
            assessment.criteria_met["consecutive_positive"] = consecutive_positive >= thresholds["consecutive_positive_roi"]
            assessment.criteria_scores["consecutive_positive_roi"] = consecutive_positive
    
    def _assess_scaled_deployment_graduation(self, 
                                           assessment: GraduationAssessment,
                                           phase_history: List[Dict[str, float]],
                                           thresholds: Dict[str, float]):
        """Assess scaled deployment graduation"""
        
        episode_count = len(phase_history)
        assessment.criteria_met["min_episodes"] = episode_count >= thresholds["min_deployment_episodes"]
        assessment.criteria_scores["episode_count"] = episode_count
        
        if episode_count >= 20:
            recent_episodes = phase_history[-20:]
            
            # Sustained performance
            sustained_perfs = [ep.get("sustained_performance", 0.0) for ep in recent_episodes]
            avg_sustained = np.mean(sustained_perfs) if sustained_perfs else 0.0
            assessment.criteria_met["sustained_performance"] = avg_sustained >= thresholds["sustained_performance"]
            assessment.criteria_scores["sustained_performance"] = avg_sustained
    
    def _count_consecutive_positive_roi(self, rois: List[float], threshold: float) -> int:
        """Count consecutive positive ROI episodes"""
        consecutive = 0
        max_consecutive = 0
        
        for roi in reversed(rois):
            if roi >= threshold:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _calculate_simple_trend(self, values: List[float]) -> float:
        """Calculate simple trend slope for a list of values"""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def get_performance_summary(self, phase: TrainingPhase) -> Dict[str, Any]:
        """Get comprehensive performance summary for a phase"""
        
        phase_history = self.performance_history[phase]
        
        if not phase_history:
            return {"phase": phase.value, "no_data": True}
        
        # Basic statistics
        rewards = [ep.get("reward", 0.0) for ep in phase_history]
        
        summary = {
            "phase": phase.value,
            "total_episodes": len(phase_history),
            "average_reward": np.mean(rewards) if rewards else 0.0,
            "reward_std": np.std(rewards) if rewards else 0.0,
            "min_reward": np.min(rewards) if rewards else 0.0,
            "max_reward": np.max(rewards) if rewards else 0.0,
            "latest_episodes": phase_history[-5:] if len(phase_history) >= 5 else phase_history
        }
        
        # Add trend analysis
        if phase in self.trend_analyses:
            summary["trends"] = {
                metric.value: {
                    "direction": analysis.trend_direction,
                    "strength": analysis.trend_strength,
                    "confidence": analysis.confidence
                }
                for metric, analysis in self.trend_analyses[phase].items()
            }
        
        # Add graduation assessment
        assessment = self.get_graduation_assessment(phase)
        summary["graduation"] = {
            "ready": assessment.ready_to_graduate,
            "confidence": assessment.confidence,
            "recommendation": assessment.recommendation
        }
        
        return summary