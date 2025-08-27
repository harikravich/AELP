"""
Curriculum Learning Scheduler

Manages progressive difficulty scaling and task scheduling for 
simulation-to-real-world learning progression.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from .phases import TrainingPhase


class DifficultyMetric(Enum):
    """Metrics used to assess task difficulty"""
    AUDIENCE_COMPLEXITY = "audience_complexity"
    BUDGET_CONSTRAINTS = "budget_constraints"
    COMPETITION_LEVEL = "competition_level"
    CREATIVE_REQUIREMENTS = "creative_requirements"
    TARGETING_PRECISION = "targeting_precision"
    MARKET_VOLATILITY = "market_volatility"


@dataclass
class CurriculumTask:
    """Represents a single curriculum task with difficulty parameters"""
    task_id: str
    task_name: str
    difficulty_level: float  # 0.0 to 1.0
    difficulty_metrics: Dict[DifficultyMetric, float]
    prerequisites: List[str] = field(default_factory=list)
    success_threshold: float = 0.8
    min_episodes: int = 10
    max_episodes: int = 100
    environment_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumProgress:
    """Tracks progress through curriculum tasks"""
    task_id: str
    episodes_completed: int = 0
    success_count: int = 0
    total_reward: float = 0.0
    average_performance: float = 0.0
    completed: bool = False
    completion_time: Optional[datetime] = None
    performance_history: List[float] = field(default_factory=list)


class CurriculumScheduler:
    """
    Manages curriculum learning progression with adaptive difficulty scaling
    based on agent performance and phase requirements.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Curriculum state
        self.current_tasks: Dict[str, CurriculumProgress] = {}
        self.completed_tasks: Dict[str, CurriculumProgress] = {}
        self.available_tasks: Dict[str, CurriculumTask] = {}
        
        # Performance tracking
        self.performance_window = config.curriculum.performance_window_size
        self.difficulty_progression_rate = config.curriculum.difficulty_progression_rate
        
        # Initialize curriculum for each phase
        self._initialize_curricula()
        
        self.logger.info("Curriculum Scheduler initialized")
    
    def _initialize_curricula(self):
        """Initialize curriculum tasks for each training phase"""
        
        # Phase 1: Simulation Training Curriculum
        self._initialize_simulation_curriculum()
        
        # Phase 2: Historical Validation Curriculum  
        self._initialize_historical_curriculum()
        
        # Phase 3: Real Testing Curriculum
        self._initialize_real_testing_curriculum()
        
        # Phase 4: Scaled Deployment Curriculum
        self._initialize_scaled_deployment_curriculum()
    
    def _initialize_simulation_curriculum(self):
        """Initialize simulation phase curriculum"""
        
        simulation_tasks = [
            # Beginner tasks
            CurriculumTask(
                task_id="sim_basic_targeting",
                task_name="Basic Audience Targeting",
                difficulty_level=0.2,
                difficulty_metrics={
                    DifficultyMetric.AUDIENCE_COMPLEXITY: 0.1,
                    DifficultyMetric.BUDGET_CONSTRAINTS: 0.2,
                    DifficultyMetric.COMPETITION_LEVEL: 0.1,
                    DifficultyMetric.CREATIVE_REQUIREMENTS: 0.2,
                    DifficultyMetric.TARGETING_PRECISION: 0.3
                },
                environment_config={
                    "audience_size": "large",
                    "competition": "low",
                    "budget_flexibility": "high",
                    "success_metrics": ["reach", "impressions"]
                }
            ),
            
            CurriculumTask(
                task_id="sim_budget_optimization",
                task_name="Budget Allocation Optimization",
                difficulty_level=0.4,
                difficulty_metrics={
                    DifficultyMetric.BUDGET_CONSTRAINTS: 0.7,
                    DifficultyMetric.AUDIENCE_COMPLEXITY: 0.3,
                    DifficultyMetric.COMPETITION_LEVEL: 0.2
                },
                prerequisites=["sim_basic_targeting"],
                environment_config={
                    "budget_constraints": "medium",
                    "cost_sensitivity": "high",
                    "success_metrics": ["cost_per_click", "roi"]
                }
            ),
            
            # Intermediate tasks
            CurriculumTask(
                task_id="sim_competitive_markets",
                task_name="Competitive Market Navigation",
                difficulty_level=0.6,
                difficulty_metrics={
                    DifficultyMetric.COMPETITION_LEVEL: 0.8,
                    DifficultyMetric.BUDGET_CONSTRAINTS: 0.5,
                    DifficultyMetric.TARGETING_PRECISION: 0.6
                },
                prerequisites=["sim_budget_optimization"],
                environment_config={
                    "competition": "high",
                    "market_saturation": "medium",
                    "bid_competition": "aggressive"
                }
            ),
            
            # Advanced tasks
            CurriculumTask(
                task_id="sim_complex_creative",
                task_name="Complex Creative Optimization",
                difficulty_level=0.8,
                difficulty_metrics={
                    DifficultyMetric.CREATIVE_REQUIREMENTS: 0.9,
                    DifficultyMetric.AUDIENCE_COMPLEXITY: 0.7,
                    DifficultyMetric.TARGETING_PRECISION: 0.8
                },
                prerequisites=["sim_competitive_markets"],
                environment_config={
                    "creative_formats": "multiple",
                    "audience_segments": "complex",
                    "personalization_required": True
                }
            ),
            
            CurriculumTask(
                task_id="sim_volatile_markets",
                task_name="Market Volatility Adaptation",
                difficulty_level=1.0,
                difficulty_metrics={
                    DifficultyMetric.MARKET_VOLATILITY: 1.0,
                    DifficultyMetric.COMPETITION_LEVEL: 0.8,
                    DifficultyMetric.BUDGET_CONSTRAINTS: 0.7
                },
                prerequisites=["sim_complex_creative"],
                environment_config={
                    "market_volatility": "high",
                    "demand_fluctuation": "extreme",
                    "real_time_adaptation": True
                }
            )
        ]
        
        for task in simulation_tasks:
            self.available_tasks[task.task_id] = task
    
    def _initialize_historical_curriculum(self):
        """Initialize historical validation curriculum"""
        
        historical_tasks = [
            CurriculumTask(
                task_id="hist_benchmark_matching",
                task_name="Historical Benchmark Matching",
                difficulty_level=0.5,
                difficulty_metrics={
                    DifficultyMetric.TARGETING_PRECISION: 0.8,
                    DifficultyMetric.BUDGET_CONSTRAINTS: 0.4
                },
                environment_config={
                    "validation_method": "cross_validation",
                    "benchmark_campaigns": "successful_only",
                    "time_period": "last_year"
                }
            ),
            
            CurriculumTask(
                task_id="hist_prediction_accuracy",
                task_name="Performance Prediction Accuracy",
                difficulty_level=0.7,
                difficulty_metrics={
                    DifficultyMetric.MARKET_VOLATILITY: 0.6,
                    DifficultyMetric.AUDIENCE_COMPLEXITY: 0.8
                },
                prerequisites=["hist_benchmark_matching"],
                environment_config={
                    "prediction_horizon": "7_days",
                    "accuracy_threshold": 0.85,
                    "confidence_intervals": True
                }
            )
        ]
        
        for task in historical_tasks:
            self.available_tasks[task.task_id] = task
    
    def _initialize_real_testing_curriculum(self):
        """Initialize real testing curriculum"""
        
        real_testing_tasks = [
            CurriculumTask(
                task_id="real_micro_budget",
                task_name="Micro Budget Campaigns",
                difficulty_level=0.3,
                difficulty_metrics={
                    DifficultyMetric.BUDGET_CONSTRAINTS: 0.9,
                    DifficultyMetric.TARGETING_PRECISION: 0.5
                },
                environment_config={
                    "daily_budget": 10.0,
                    "campaign_duration": "3_days",
                    "safety_monitoring": "continuous"
                }
            ),
            
            CurriculumTask(
                task_id="real_small_budget",
                task_name="Small Budget Optimization",
                difficulty_level=0.5,
                difficulty_metrics={
                    DifficultyMetric.BUDGET_CONSTRAINTS: 0.7,
                    DifficultyMetric.COMPETITION_LEVEL: 0.6
                },
                prerequisites=["real_micro_budget"],
                environment_config={
                    "daily_budget": 50.0,
                    "campaign_duration": "7_days",
                    "performance_tracking": "real_time"
                }
            )
        ]
        
        for task in real_testing_tasks:
            self.available_tasks[task.task_id] = task
    
    def _initialize_scaled_deployment_curriculum(self):
        """Initialize scaled deployment curriculum"""
        
        scaled_tasks = [
            CurriculumTask(
                task_id="scaled_multi_campaign",
                task_name="Multi-Campaign Management",
                difficulty_level=0.6,
                difficulty_metrics={
                    DifficultyMetric.AUDIENCE_COMPLEXITY: 0.8,
                    DifficultyMetric.BUDGET_CONSTRAINTS: 0.6,
                    DifficultyMetric.TARGETING_PRECISION: 0.7
                },
                environment_config={
                    "concurrent_campaigns": 5,
                    "budget_allocation": "dynamic",
                    "cross_campaign_optimization": True
                }
            ),
            
            CurriculumTask(
                task_id="scaled_transfer_learning",
                task_name="Transfer Learning Across Products",
                difficulty_level=0.8,
                difficulty_metrics={
                    DifficultyMetric.AUDIENCE_COMPLEXITY: 0.9,
                    DifficultyMetric.CREATIVE_REQUIREMENTS: 0.8,
                    DifficultyMetric.MARKET_VOLATILITY: 0.7
                },
                prerequisites=["scaled_multi_campaign"],
                environment_config={
                    "product_categories": "multiple",
                    "audience_overlap": "minimal",
                    "knowledge_transfer": True
                }
            )
        ]
        
        for task in scaled_tasks:
            self.available_tasks[task.task_id] = task
    
    def get_current_task(self, phase: TrainingPhase) -> Optional[CurriculumTask]:
        """Get the current task for the given phase"""
        
        # Get available tasks for this phase
        phase_tasks = self._get_phase_tasks(phase)
        
        # Find the next available task that meets prerequisites
        for task in phase_tasks:
            if task.task_id in self.completed_tasks:
                continue
                
            if self._check_prerequisites(task):
                return task
        
        return None
    
    def _get_phase_tasks(self, phase: TrainingPhase) -> List[CurriculumTask]:
        """Get tasks appropriate for the given phase"""
        
        phase_prefixes = {
            TrainingPhase.SIMULATION: "sim_",
            TrainingPhase.HISTORICAL_VALIDATION: "hist_", 
            TrainingPhase.REAL_TESTING: "real_",
            TrainingPhase.SCALED_DEPLOYMENT: "scaled_"
        }
        
        prefix = phase_prefixes.get(phase, "")
        return [
            task for task in self.available_tasks.values()
            if task.task_id.startswith(prefix)
        ]
    
    def _check_prerequisites(self, task: CurriculumTask) -> bool:
        """Check if task prerequisites are met"""
        
        for prereq_id in task.prerequisites:
            if prereq_id not in self.completed_tasks:
                return False
            
            # Check if prerequisite was completed successfully
            prereq_progress = self.completed_tasks[prereq_id]
            if not prereq_progress.completed:
                return False
        
        return True
    
    def update_curriculum(self, 
                         phase: TrainingPhase,
                         performance_history: List[float],
                         episode_info: Optional[Dict[str, Any]] = None):
        """
        Update curriculum based on recent performance
        
        Args:
            phase: Current training phase
            performance_history: Recent performance scores
            episode_info: Additional episode information
        """
        
        current_task = self.get_current_task(phase)
        if not current_task:
            return
        
        # Update task progress
        task_progress = self.current_tasks.get(
            current_task.task_id,
            CurriculumProgress(task_id=current_task.task_id)
        )
        
        if performance_history:
            latest_performance = performance_history[-1]
            task_progress.performance_history.append(latest_performance)
            task_progress.episodes_completed += 1
            task_progress.total_reward += latest_performance
            
            # Update average performance
            task_progress.average_performance = (
                task_progress.total_reward / task_progress.episodes_completed
            )
            
            # Check for success
            if latest_performance > current_task.success_threshold:
                task_progress.success_count += 1
        
        # Update current tasks tracking
        self.current_tasks[current_task.task_id] = task_progress
        
        # Check for task completion
        self._check_task_completion(current_task, task_progress)
        
        # Adaptive difficulty adjustment
        self._adjust_difficulty(current_task, performance_history)
    
    def _check_task_completion(self, 
                              task: CurriculumTask, 
                              progress: CurriculumProgress):
        """Check if a task should be marked as completed"""
        
        # Minimum episode requirement
        if progress.episodes_completed < task.min_episodes:
            return
        
        # Success rate requirement
        success_rate = progress.success_count / progress.episodes_completed
        if success_rate < task.success_threshold:
            # Check if we've exceeded max episodes
            if progress.episodes_completed >= task.max_episodes:
                self.logger.warning(
                    f"Task {task.task_id} failed to meet success threshold "
                    f"after {progress.episodes_completed} episodes"
                )
                # Mark as completed but unsuccessful
                progress.completed = True
                progress.completion_time = datetime.now()
            return
        
        # Check performance consistency
        recent_performance = progress.performance_history[-self.performance_window:]
        if len(recent_performance) >= self.performance_window:
            consistency = self._calculate_performance_consistency(recent_performance)
            if consistency < 0.7:  # 70% consistency threshold
                return
        
        # Task completed successfully
        progress.completed = True
        progress.completion_time = datetime.now()
        
        # Move to completed tasks
        self.completed_tasks[task.task_id] = progress
        self.current_tasks.pop(task.task_id, None)
        
        self.logger.info(
            f"Task {task.task_id} completed successfully. "
            f"Episodes: {progress.episodes_completed}, "
            f"Success rate: {success_rate:.3f}, "
            f"Average performance: {progress.average_performance:.3f}"
        )
    
    def _adjust_difficulty(self, 
                          task: CurriculumTask, 
                          performance_history: List[float]):
        """Adaptively adjust task difficulty based on performance"""
        
        if len(performance_history) < 10:  # Need sufficient data
            return
        
        recent_performance = performance_history[-10:]
        avg_performance = np.mean(recent_performance)
        
        # Calculate performance trend
        if len(recent_performance) >= 5:
            early_avg = np.mean(recent_performance[:5])
            late_avg = np.mean(recent_performance[-5:])
            trend = late_avg - early_avg
        else:
            trend = 0.0
        
        # Adjust difficulty based on performance and trend
        if avg_performance > 0.9 and trend > 0.05:
            # Performing very well and improving - increase difficulty
            self._increase_task_difficulty(task, 0.1)
        elif avg_performance < 0.6 and trend < -0.05:
            # Performing poorly and declining - decrease difficulty
            self._decrease_task_difficulty(task, 0.1)
    
    def _increase_task_difficulty(self, task: CurriculumTask, factor: float):
        """Increase task difficulty"""
        
        # Increase relevant difficulty metrics
        for metric, value in task.difficulty_metrics.items():
            new_value = min(1.0, value + factor)
            task.difficulty_metrics[metric] = new_value
        
        # Update overall difficulty level
        task.difficulty_level = min(1.0, task.difficulty_level + factor)
        
        self.logger.info(f"Increased difficulty for task {task.task_id} to {task.difficulty_level:.2f}")
    
    def _decrease_task_difficulty(self, task: CurriculumTask, factor: float):
        """Decrease task difficulty"""
        
        # Decrease relevant difficulty metrics
        for metric, value in task.difficulty_metrics.items():
            new_value = max(0.0, value - factor)
            task.difficulty_metrics[metric] = new_value
        
        # Update overall difficulty level
        task.difficulty_level = max(0.0, task.difficulty_level - factor)
        
        self.logger.info(f"Decreased difficulty for task {task.task_id} to {task.difficulty_level:.2f}")
    
    def _calculate_performance_consistency(self, performance_history: List[float]) -> float:
        """Calculate performance consistency score"""
        
        if len(performance_history) < 2:
            return 1.0
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_performance = np.mean(performance_history)
        std_performance = np.std(performance_history)
        
        if mean_performance == 0:
            return 0.0
        
        cv = std_performance / mean_performance
        
        # Convert to consistency score (higher is more consistent)
        consistency = max(0.0, 1.0 - cv)
        
        return consistency
    
    def get_curriculum_progress(self, phase: TrainingPhase) -> Dict[str, Any]:
        """Get comprehensive curriculum progress for a phase"""
        
        phase_tasks = self._get_phase_tasks(phase)
        
        progress_info = {
            "phase": phase.value,
            "total_tasks": len(phase_tasks),
            "completed_tasks": 0,
            "current_task": None,
            "overall_progress": 0.0,
            "task_details": []
        }
        
        for task in phase_tasks:
            task_info = {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "difficulty_level": task.difficulty_level,
                "status": "not_started"
            }
            
            if task.task_id in self.completed_tasks:
                progress = self.completed_tasks[task.task_id]
                task_info.update({
                    "status": "completed",
                    "episodes_completed": progress.episodes_completed,
                    "success_rate": progress.success_count / max(1, progress.episodes_completed),
                    "average_performance": progress.average_performance,
                    "completion_time": progress.completion_time
                })
                progress_info["completed_tasks"] += 1
                
            elif task.task_id in self.current_tasks:
                progress = self.current_tasks[task.task_id]
                task_info.update({
                    "status": "in_progress",
                    "episodes_completed": progress.episodes_completed,
                    "success_rate": progress.success_count / max(1, progress.episodes_completed),
                    "average_performance": progress.average_performance
                })
                progress_info["current_task"] = task.task_id
            
            progress_info["task_details"].append(task_info)
        
        # Calculate overall progress
        if progress_info["total_tasks"] > 0:
            progress_info["overall_progress"] = progress_info["completed_tasks"] / progress_info["total_tasks"]
        
        return progress_info
    
    def reset_curriculum(self, phase: Optional[TrainingPhase] = None):
        """Reset curriculum progress for a specific phase or all phases"""
        
        if phase is None:
            # Reset all curriculum progress
            self.current_tasks.clear()
            self.completed_tasks.clear()
            self.logger.info("All curriculum progress reset")
        else:
            # Reset progress for specific phase
            phase_tasks = self._get_phase_tasks(phase)
            for task in phase_tasks:
                self.current_tasks.pop(task.task_id, None)
                self.completed_tasks.pop(task.task_id, None)
            self.logger.info(f"Curriculum progress reset for phase {phase.value}")
    
    def get_recommended_environment_config(self, phase: TrainingPhase) -> Dict[str, Any]:
        """Get recommended environment configuration for current task"""
        
        current_task = self.get_current_task(phase)
        if current_task:
            return current_task.environment_config.copy()
        
        # Return default configuration if no current task
        return {
            "difficulty_level": 0.5,
            "adaptive_difficulty": True,
            "performance_feedback": True
        }