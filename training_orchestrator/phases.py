"""
Training Phase Management

Defines the four phases of the simulation-to-real-world learning progression
and manages transitions between phases.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class TrainingPhase(Enum):
    """Training phases for ad campaign agent learning progression"""
    SIMULATION = "simulation"
    HISTORICAL_VALIDATION = "historical_validation"
    REAL_TESTING = "real_testing"
    SCALED_DEPLOYMENT = "scaled_deployment"


@dataclass
class PhaseConfiguration:
    """Configuration for a specific training phase"""
    name: str
    max_episodes: int
    graduation_criteria: Dict[str, float]
    safety_constraints: Dict[str, Any]
    environment_config: Dict[str, Any]
    budget_limits: Optional[Dict[str, float]] = None
    time_limits: Optional[timedelta] = None


@dataclass
class PhaseTransition:
    """Records a phase transition event"""
    from_phase: TrainingPhase
    to_phase: TrainingPhase
    timestamp: datetime
    graduation_metrics: Dict[str, float]
    reason: str


class PhaseManager:
    """
    Manages training phases and transitions between them based on
    graduation criteria and performance thresholds.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_phase = TrainingPhase.SIMULATION
        self.phase_history: List[PhaseTransition] = []
        
        # Define phase configurations
        self.phase_configs = self._setup_phase_configurations()
        
        self.logger.info("Phase Manager initialized")
    
    def _setup_phase_configurations(self) -> Dict[TrainingPhase, PhaseConfiguration]:
        """Setup configurations for each training phase"""
        
        return {
            TrainingPhase.SIMULATION: PhaseConfiguration(
                name="Simulation Training",
                max_episodes=self.config.phases.simulation_episodes,
                graduation_criteria={
                    "min_average_reward": 0.7,
                    "min_success_rate": 0.8,
                    "performance_improvement_rate": 0.05,
                    "consistency_window": 50
                },
                safety_constraints={
                    "max_safety_violations": 5,
                    "min_content_safety_score": 0.9
                },
                environment_config={
                    "persona_diversity": "high",
                    "scenario_complexity": "progressive",
                    "feedback_frequency": "episode"
                }
            ),
            
            TrainingPhase.HISTORICAL_VALIDATION: PhaseConfiguration(
                name="Historical Data Validation", 
                max_episodes=self.config.phases.historical_validation_episodes,
                graduation_criteria={
                    "historical_performance_match": 0.95,  # 95% of known good campaigns
                    "prediction_accuracy": 0.85,
                    "correlation_with_actual": 0.8
                },
                safety_constraints={
                    "max_safety_violations": 2,
                    "content_policy_compliance": 1.0
                },
                environment_config={
                    "historical_data_coverage": "comprehensive",
                    "validation_method": "cross_validation",
                    "benchmark_campaigns": "top_performers"
                }
            ),
            
            TrainingPhase.REAL_TESTING: PhaseConfiguration(
                name="Small Budget Real Testing",
                max_episodes=100,  # Conservative for real money
                graduation_criteria={
                    "consecutive_positive_roi": 5,
                    "min_roi_threshold": 0.05,  # 5% ROI minimum
                    "budget_efficiency": 0.8
                },
                safety_constraints={
                    "max_daily_budget": self.config.budget.real_testing_daily_limit,
                    "max_safety_violations": 0,  # Zero tolerance in real testing
                    "content_approval_required": True,
                    "human_oversight_required": True
                },
                environment_config={
                    "platform_integration": "sandbox_first",
                    "budget_controls": "strict",
                    "monitoring_frequency": "real_time"
                },
                budget_limits={
                    "daily_limit": self.config.budget.real_testing_daily_limit,
                    "episode_limit": 10.0,
                    "total_phase_limit": 500.0
                },
                time_limits=timedelta(days=30)  # Max 30 days for real testing
            ),
            
            TrainingPhase.SCALED_DEPLOYMENT: PhaseConfiguration(
                name="Scaled Deployment",
                max_episodes=1000,  # Ongoing operation
                graduation_criteria={
                    "sustained_performance": 0.15,  # 15% improvement threshold
                    "multi_campaign_success": 0.9,
                    "transfer_learning_effectiveness": 0.8
                },
                safety_constraints={
                    "max_daily_budget": self.config.budget.scaled_deployment_daily_limit,
                    "performance_monitoring": "continuous",
                    "anomaly_detection": "enabled"
                },
                environment_config={
                    "multi_platform_support": True,
                    "advanced_optimization": True,
                    "transfer_learning": True
                },
                budget_limits={
                    "daily_limit": self.config.budget.scaled_deployment_daily_limit,
                    "scaling_factor": 2.0  # Can scale up based on performance
                }
            )
        }
    
    def get_current_phase_config(self) -> PhaseConfiguration:
        """Get configuration for current phase"""
        return self.phase_configs[self.current_phase]
    
    def can_graduate_phase(self, 
                          phase: TrainingPhase, 
                          performance_metrics: Dict[str, float],
                          episode_count: int) -> tuple[bool, str]:
        """
        Check if agent meets graduation criteria for current phase
        
        Args:
            phase: The phase to check graduation for
            performance_metrics: Current performance metrics
            episode_count: Number of episodes completed in phase
            
        Returns:
            tuple: (can_graduate, reason)
        """
        phase_config = self.phase_configs[phase]
        criteria = phase_config.graduation_criteria
        
        # Check minimum episode requirement
        min_episodes = max(50, phase_config.max_episodes * 0.1)  # At least 10% of max episodes
        if episode_count < min_episodes:
            return False, f"Insufficient episodes: {episode_count} < {min_episodes}"
        
        # Check specific graduation criteria for each phase
        if phase == TrainingPhase.SIMULATION:
            return self._check_simulation_graduation(performance_metrics, criteria)
        
        elif phase == TrainingPhase.HISTORICAL_VALIDATION:
            return self._check_historical_graduation(performance_metrics, criteria)
        
        elif phase == TrainingPhase.REAL_TESTING:
            return self._check_real_testing_graduation(performance_metrics, criteria)
        
        elif phase == TrainingPhase.SCALED_DEPLOYMENT:
            return self._check_scaled_deployment_graduation(performance_metrics, criteria)
        
        return False, "Unknown phase"
    
    def _check_simulation_graduation(self, 
                                   metrics: Dict[str, float], 
                                   criteria: Dict[str, float]) -> tuple[bool, str]:
        """Check graduation criteria for simulation phase"""
        
        # Check average reward threshold
        if metrics.get("average_reward", 0) < criteria["min_average_reward"]:
            return False, f"Average reward {metrics.get('average_reward', 0):.3f} below threshold {criteria['min_average_reward']}"
        
        # Check success rate
        if metrics.get("success_rate", 0) < criteria["min_success_rate"]:
            return False, f"Success rate {metrics.get('success_rate', 0):.3f} below threshold {criteria['min_success_rate']}"
        
        # Check performance improvement trend
        improvement_rate = metrics.get("performance_improvement_rate", 0)
        if improvement_rate < criteria["performance_improvement_rate"]:
            return False, f"Performance improvement rate {improvement_rate:.3f} below threshold {criteria['performance_improvement_rate']}"
        
        # Check consistency over recent episodes
        consistency = metrics.get("performance_consistency", 0)
        if consistency < 0.8:  # 80% consistency threshold
            return False, f"Performance consistency {consistency:.3f} below 0.8"
        
        return True, "All simulation graduation criteria met"
    
    def _check_historical_graduation(self, 
                                   metrics: Dict[str, float], 
                                   criteria: Dict[str, float]) -> tuple[bool, str]:
        """Check graduation criteria for historical validation phase"""
        
        # Check performance match with historical campaigns
        historical_match = metrics.get("historical_performance_match", 0)
        if historical_match < criteria["historical_performance_match"]:
            return False, f"Historical performance match {historical_match:.3f} below threshold {criteria['historical_performance_match']}"
        
        # Check prediction accuracy
        prediction_accuracy = metrics.get("prediction_accuracy", 0)
        if prediction_accuracy < criteria["prediction_accuracy"]:
            return False, f"Prediction accuracy {prediction_accuracy:.3f} below threshold {criteria['prediction_accuracy']}"
        
        # Check correlation with actual results
        correlation = metrics.get("correlation_with_actual", 0)
        if correlation < criteria["correlation_with_actual"]:
            return False, f"Correlation with actual {correlation:.3f} below threshold {criteria['correlation_with_actual']}"
        
        return True, "All historical validation criteria met"
    
    def _check_real_testing_graduation(self, 
                                     metrics: Dict[str, float], 
                                     criteria: Dict[str, float]) -> tuple[bool, str]:
        """Check graduation criteria for real testing phase"""
        
        # Check consecutive positive ROI campaigns
        consecutive_positive = metrics.get("consecutive_positive_roi", 0)
        required_consecutive = criteria["consecutive_positive_roi"]
        if consecutive_positive < required_consecutive:
            return False, f"Consecutive positive ROI {consecutive_positive} below required {required_consecutive}"
        
        # Check minimum ROI threshold
        avg_roi = metrics.get("average_roi", 0)
        if avg_roi < criteria["min_roi_threshold"]:
            return False, f"Average ROI {avg_roi:.3f} below threshold {criteria['min_roi_threshold']}"
        
        # Check budget efficiency
        budget_efficiency = metrics.get("budget_efficiency", 0)
        if budget_efficiency < criteria["budget_efficiency"]:
            return False, f"Budget efficiency {budget_efficiency:.3f} below threshold {criteria['budget_efficiency']}"
        
        return True, "All real testing criteria met"
    
    def _check_scaled_deployment_graduation(self, 
                                          metrics: Dict[str, float], 
                                          criteria: Dict[str, float]) -> tuple[bool, str]:
        """Check graduation criteria for scaled deployment phase"""
        
        # Check sustained performance improvement
        sustained_performance = metrics.get("sustained_performance", 0)
        if sustained_performance < criteria["sustained_performance"]:
            return False, f"Sustained performance {sustained_performance:.3f} below threshold {criteria['sustained_performance']}"
        
        # Check multi-campaign success rate
        multi_campaign_success = metrics.get("multi_campaign_success", 0)
        if multi_campaign_success < criteria["multi_campaign_success"]:
            return False, f"Multi-campaign success {multi_campaign_success:.3f} below threshold {criteria['multi_campaign_success']}"
        
        # Check transfer learning effectiveness
        transfer_learning = metrics.get("transfer_learning_effectiveness", 0)
        if transfer_learning < criteria["transfer_learning_effectiveness"]:
            return False, f"Transfer learning effectiveness {transfer_learning:.3f} below threshold {criteria['transfer_learning_effectiveness']}"
        
        return True, "All scaled deployment criteria met"
    
    def transition_to_next_phase(self, 
                               current_metrics: Dict[str, float], 
                               reason: str) -> bool:
        """
        Transition to the next phase if graduation criteria are met
        
        Args:
            current_metrics: Performance metrics that justify transition
            reason: Reason for the transition
            
        Returns:
            bool: True if transition was successful
        """
        
        # Define phase progression order
        phase_order = [
            TrainingPhase.SIMULATION,
            TrainingPhase.HISTORICAL_VALIDATION,
            TrainingPhase.REAL_TESTING,
            TrainingPhase.SCALED_DEPLOYMENT
        ]
        
        current_index = phase_order.index(self.current_phase)
        
        # Check if there's a next phase
        if current_index >= len(phase_order) - 1:
            self.logger.info("Already in final phase")
            return False
        
        next_phase = phase_order[current_index + 1]
        
        # Record the transition
        transition = PhaseTransition(
            from_phase=self.current_phase,
            to_phase=next_phase,
            timestamp=datetime.now(),
            graduation_metrics=current_metrics.copy(),
            reason=reason
        )
        
        self.phase_history.append(transition)
        
        # Update current phase
        previous_phase = self.current_phase
        self.current_phase = next_phase
        
        self.logger.info(
            f"Phase transition: {previous_phase.value} -> {next_phase.value}. "
            f"Reason: {reason}"
        )
        
        return True
    
    def get_phase_history(self) -> List[PhaseTransition]:
        """Get history of phase transitions"""
        return self.phase_history.copy()
    
    def get_current_phase(self) -> TrainingPhase:
        """Get current training phase"""
        return self.current_phase
    
    def get_phase_progress(self, episode_count: int) -> Dict[str, Any]:
        """
        Get progress information for current phase
        
        Args:
            episode_count: Number of episodes completed in current phase
            
        Returns:
            Dict with progress information
        """
        current_config = self.get_current_phase_config()
        
        progress = {
            "current_phase": self.current_phase.value,
            "phase_name": current_config.name,
            "episodes_completed": episode_count,
            "max_episodes": current_config.max_episodes,
            "progress_percentage": min(100.0, (episode_count / current_config.max_episodes) * 100),
            "graduation_criteria": current_config.graduation_criteria,
            "safety_constraints": current_config.safety_constraints
        }
        
        # Add budget information if applicable
        if current_config.budget_limits:
            progress["budget_limits"] = current_config.budget_limits
        
        # Add time limits if applicable
        if current_config.time_limits:
            progress["time_limits"] = current_config.time_limits
        
        return progress
    
    def reset_to_phase(self, phase: TrainingPhase, reason: str = "Manual reset"):
        """
        Reset to a specific phase (for debugging or retraining)
        
        Args:
            phase: Phase to reset to
            reason: Reason for the reset
        """
        if phase != self.current_phase:
            # Record the reset as a transition
            transition = PhaseTransition(
                from_phase=self.current_phase,
                to_phase=phase,
                timestamp=datetime.now(),
                graduation_metrics={},
                reason=f"RESET: {reason}"
            )
            
            self.phase_history.append(transition)
            self.current_phase = phase
            
            self.logger.warning(
                f"Phase reset to {phase.value}. Reason: {reason}"
            )