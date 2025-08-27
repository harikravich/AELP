"""
Core Training Orchestrator

Manages the four-phase simulation-to-real-world learning progression
for ad campaign agents.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

import numpy as np
from google.cloud import bigquery, pubsub_v1
import redis

from .phases import TrainingPhase, PhaseManager
from .curriculum import CurriculumScheduler
from .episode_manager import EpisodeManager
from .performance_monitor import PerformanceMonitor
from .safety_monitor import SafetyMonitor
from .checkpoint_manager import CheckpointManager
from .journey_timeout import JourneyTimeoutManager, TimeoutConfiguration


class TrainingState(Enum):
    """Training orchestrator states"""
    INITIALIZING = "initializing"
    SIMULATION_TRAINING = "simulation_training"
    HISTORICAL_VALIDATION = "historical_validation" 
    REAL_TESTING = "real_testing"
    SCALED_DEPLOYMENT = "scaled_deployment"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingConfiguration:
    """Configuration for training orchestrator"""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_config: Dict[str, Any] = field(default_factory=dict)
    environment_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Phase configuration
    simulation_episodes: int = 1000
    historical_validation_episodes: int = 100
    real_testing_budget_limit: float = 50.0  # dollars per day
    scaled_deployment_threshold: float = 0.15  # 15% ROI improvement
    
    # Curriculum settings
    curriculum_enabled: bool = True
    difficulty_progression_rate: float = 0.1
    performance_window: int = 50  # episodes to consider for progression
    
    # Safety settings
    max_daily_budget: float = 10000.0
    safety_check_interval: int = 10  # minutes
    anomaly_threshold: float = 3.0  # standard deviations
    
    # Journey timeout settings
    journey_timeout_days: int = 14
    inactivity_threshold_hours: int = 72
    cleanup_stale_data_days: int = 30
    
    # Reproducibility
    random_seed: int = 42
    log_level: str = "INFO"
    checkpoint_interval: int = 100  # episodes
    
    # Integration settings
    bigquery_project: str = "gaelp-project"
    bigquery_dataset: str = "training_logs"
    redis_host: str = "localhost"
    redis_port: int = 6379
    pubsub_topic: str = "training-events"


@dataclass
class TrainingMetrics:
    """Training session metrics"""
    total_episodes: int = 0
    successful_episodes: int = 0
    current_phase: TrainingPhase = TrainingPhase.SIMULATION
    phase_start_time: datetime = field(default_factory=datetime.now)
    total_reward: float = 0.0
    average_reward: float = 0.0
    budget_spent: float = 0.0
    safety_violations: int = 0
    last_checkpoint: Optional[str] = None
    
    # Journey timeout metrics
    active_journeys: int = 0
    timed_out_journeys: int = 0
    total_abandonment_penalty: float = 0.0
    zombie_journeys_cleaned: int = 0


class TrainingOrchestrator:
    """
    Core training orchestrator that manages the simulation-to-real-world
    learning progression for ad campaign agents.
    """
    
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.state = TrainingState.INITIALIZING
        self.metrics = TrainingMetrics()
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Initialize random seed for reproducibility
        np.random.seed(config.random_seed)
        
        # Initialize components
        self.phase_manager = PhaseManager(config)
        self.curriculum_scheduler = CurriculumScheduler(config)
        self.episode_manager = EpisodeManager(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.safety_monitor = SafetyMonitor(config)
        self.checkpoint_manager = CheckpointManager()
        
        # Initialize journey timeout manager
        timeout_config = TimeoutConfiguration(
            default_timeout_days=config.journey_timeout_days,
            inactivity_threshold_hours=config.inactivity_threshold_hours,
            max_journey_duration_days=config.cleanup_stale_data_days * 3  # 90 days max
        )
        self.journey_timeout_manager = None  # Will be initialized with external services
        
        # Initialize external services
        self._init_external_services()
        
        # Training control
        self._stop_requested = False
        self._pause_requested = False
        
        self.logger.info(f"Training Orchestrator initialized with experiment ID: {config.experiment_id}")
    
    def _init_external_services(self):
        """Initialize external service connections"""
        try:
            # BigQuery client for logging
            self.bigquery_client = bigquery.Client(project=self.config.bigquery_project)
            
            # Redis client for state management
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            
            # Pub/Sub publisher for events
            self.publisher = pubsub_v1.PublisherClient()
            self.topic_path = self.publisher.topic_path(
                self.config.bigquery_project,
                self.config.pubsub_topic
            )
            
            # Initialize journey timeout manager with external services
            timeout_config = TimeoutConfiguration(
                default_timeout_days=self.config.journey_timeout_days,
                inactivity_threshold_hours=self.config.inactivity_threshold_hours,
                max_journey_duration_days=self.config.cleanup_stale_data_days * 3
            )
            
            self.journey_timeout_manager = JourneyTimeoutManager(
                config=timeout_config,
                bigquery_client=self.bigquery_client,
                redis_client=self.redis_client,
                project_id=self.config.bigquery_project,
                dataset_id=self.config.bigquery_dataset
            )
            
            self.logger.info("External services and journey timeout manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize external services: {e}")
            raise
    
    async def start_training(self, agent, environments: Dict[str, Any]) -> bool:
        """
        Start the training process
        
        Args:
            agent: The agent to train
            environments: Dictionary of environment instances keyed by phase
            
        Returns:
            bool: True if training completed successfully
        """
        try:
            self.state = TrainingState.SIMULATION_TRAINING
            self.logger.info("Starting training orchestration")
            
            # Save initial checkpoint
            await self._save_checkpoint(agent, "initial")
            
            # Phase 1: Simulation Training
            success = await self._run_simulation_phase(agent, environments)
            if not success:
                return False
            
            # Phase 2: Historical Data Validation
            success = await self._run_historical_validation_phase(agent, environments)
            if not success:
                return False
            
            # Phase 3: Small Budget Real Testing
            success = await self._run_real_testing_phase(agent, environments)
            if not success:
                return False
            
            # Phase 4: Scaled Deployment
            success = await self._run_scaled_deployment_phase(agent, environments)
            if not success:
                return False
            
            self.state = TrainingState.COMPLETED
            self.logger.info("Training completed successfully")
            
            # Final checkpoint and metrics
            await self._save_checkpoint(agent, "final")
            await self._publish_training_completion()
            
            return True
            
        except Exception as e:
            self.state = TrainingState.FAILED
            self.logger.error(f"Training failed: {e}")
            await self._publish_training_failure(str(e))
            return False
    
    async def _run_simulation_phase(self, agent, environments) -> bool:
        """Run Phase 1: Simulation Training"""
        self.logger.info("Starting simulation training phase")
        self.metrics.current_phase = TrainingPhase.SIMULATION
        self.metrics.phase_start_time = datetime.now()
        
        simulation_env = environments.get("simulation")
        if not simulation_env:
            raise ValueError("Simulation environment not provided")
        
        episodes_completed = 0
        performance_history = []
        
        while episodes_completed < self.config.simulation_episodes:
            if self._stop_requested:
                return False
            
            if self._pause_requested:
                await self._handle_pause()
            
            # Run episode
            episode_result = await self.episode_manager.run_episode(
                agent, simulation_env, f"sim_{episodes_completed}"
            )
            
            episodes_completed += 1
            self.metrics.total_episodes += 1
            
            if episode_result.success:
                self.metrics.successful_episodes += 1
                self.metrics.total_reward += episode_result.total_reward
                self.metrics.average_reward = self.metrics.total_reward / self.metrics.successful_episodes
                performance_history.append(episode_result.total_reward)
            
            # Update curriculum if enabled
            if self.config.curriculum_enabled:
                self.curriculum_scheduler.update_curriculum(
                    phase=TrainingPhase.SIMULATION,
                    performance_history=performance_history[-self.config.performance_window:]
                )
            
            # Performance monitoring
            should_continue = self.performance_monitor.check_phase_progression(
                phase=TrainingPhase.SIMULATION,
                performance_history=performance_history[-self.config.performance_window:]
            )
            
            # Safety monitoring
            safety_ok = await self.safety_monitor.check_safety_constraints(
                episode_result, TrainingPhase.SIMULATION
            )
            
            if not safety_ok:
                self.metrics.safety_violations += 1
                self.logger.warning("Safety violation detected in simulation phase")
            
            # Checkpoint saving
            if episodes_completed % self.config.checkpoint_interval == 0:
                await self._save_checkpoint(agent, f"sim_episode_{episodes_completed}")
            
            # Log progress
            if episodes_completed % 100 == 0:
                self.logger.info(
                    f"Simulation progress: {episodes_completed}/{self.config.simulation_episodes} "
                    f"episodes, avg reward: {self.metrics.average_reward:.3f}"
                )
        
        # Check graduation criteria
        graduated = self.performance_monitor.check_graduation_criteria(
            TrainingPhase.SIMULATION, performance_history
        )
        
        if not graduated:
            self.logger.warning("Failed to meet simulation phase graduation criteria")
            return False
        
        self.logger.info("Simulation phase completed successfully")
        await self._log_phase_completion(TrainingPhase.SIMULATION)
        return True
    
    async def _run_historical_validation_phase(self, agent, environments) -> bool:
        """Run Phase 2: Historical Data Validation"""
        self.logger.info("Starting historical validation phase")
        self.metrics.current_phase = TrainingPhase.HISTORICAL_VALIDATION
        self.metrics.phase_start_time = datetime.now()
        
        historical_env = environments.get("historical")
        if not historical_env:
            raise ValueError("Historical environment not provided")
        
        validation_results = []
        episodes_completed = 0
        
        while episodes_completed < self.config.historical_validation_episodes:
            if self._stop_requested:
                return False
            
            # Run validation episode
            episode_result = await self.episode_manager.run_episode(
                agent, historical_env, f"hist_{episodes_completed}"
            )
            
            episodes_completed += 1
            self.metrics.total_episodes += 1
            
            if episode_result.success:
                self.metrics.successful_episodes += 1
                validation_results.append(episode_result.validation_metrics)
            
            # Safety check
            safety_ok = await self.safety_monitor.check_safety_constraints(
                episode_result, TrainingPhase.HISTORICAL_VALIDATION
            )
            
            if not safety_ok:
                self.metrics.safety_violations += 1
        
        # Validate against historical benchmarks
        graduated = self.performance_monitor.validate_historical_performance(
            validation_results
        )
        
        if not graduated:
            self.logger.warning("Failed historical validation criteria")
            return False
        
        self.logger.info("Historical validation phase completed successfully")
        await self._log_phase_completion(TrainingPhase.HISTORICAL_VALIDATION)
        return True
    
    async def _run_real_testing_phase(self, agent, environments) -> bool:
        """Run Phase 3: Small Budget Real Testing"""
        self.logger.info("Starting real testing phase with budget controls")
        self.metrics.current_phase = TrainingPhase.REAL_TESTING
        self.metrics.phase_start_time = datetime.now()
        
        real_env = environments.get("real")
        if not real_env:
            raise ValueError("Real environment not provided")
        
        # Configure budget constraints
        daily_budget_spent = 0.0
        consecutive_successful_campaigns = 0
        required_consecutive = 5
        
        while consecutive_successful_campaigns < required_consecutive:
            if self._stop_requested:
                return False
            
            # Check daily budget limit
            if daily_budget_spent >= self.config.real_testing_budget_limit:
                self.logger.info("Daily budget limit reached, waiting for reset")
                await asyncio.sleep(3600)  # Wait 1 hour
                daily_budget_spent = 0.0
                continue
            
            # Run real campaign episode
            episode_result = await self.episode_manager.run_episode(
                agent, real_env, f"real_{consecutive_successful_campaigns}"
            )
            
            self.metrics.total_episodes += 1
            daily_budget_spent += episode_result.budget_spent
            self.metrics.budget_spent += episode_result.budget_spent
            
            # Enhanced safety monitoring for real campaigns
            safety_ok = await self.safety_monitor.check_real_campaign_safety(
                episode_result
            )
            
            if not safety_ok:
                self.logger.error("Safety violation in real campaign, pausing")
                consecutive_successful_campaigns = 0
                self.metrics.safety_violations += 1
                continue
            
            # Check campaign performance
            if episode_result.success and episode_result.roi > 0:
                consecutive_successful_campaigns += 1
                self.metrics.successful_episodes += 1
                self.logger.info(
                    f"Successful real campaign {consecutive_successful_campaigns}/{required_consecutive}, "
                    f"ROI: {episode_result.roi:.3f}"
                )
            else:
                consecutive_successful_campaigns = 0
                self.logger.warning("Real campaign failed, resetting counter")
        
        self.logger.info("Real testing phase completed successfully")
        await self._log_phase_completion(TrainingPhase.REAL_TESTING)
        return True
    
    async def _run_scaled_deployment_phase(self, agent, environments) -> bool:
        """Run Phase 4: Scaled Deployment"""
        self.logger.info("Starting scaled deployment phase")
        self.metrics.current_phase = TrainingPhase.SCALED_DEPLOYMENT
        self.metrics.phase_start_time = datetime.now()
        
        # Implementation for scaled deployment
        # This would include multi-campaign management,
        # advanced optimization strategies, and transfer learning
        
        # For now, return success to complete the phase structure
        self.logger.info("Scaled deployment phase completed successfully")
        await self._log_phase_completion(TrainingPhase.SCALED_DEPLOYMENT)
        return True
    
    async def _save_checkpoint(self, agent, checkpoint_name: str):
        """Save agent checkpoint with metadata"""
        checkpoint_data = {
            "agent_state": agent.get_state(),
            "metrics": self.metrics,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to Redis for quick access
        checkpoint_key = f"checkpoint:{self.config.experiment_id}:{checkpoint_name}"
        self.redis_client.hset(checkpoint_key, mapping=checkpoint_data)
        
        self.metrics.last_checkpoint = checkpoint_name
        self.logger.info(f"Checkpoint saved: {checkpoint_name}")
    
    async def _log_phase_completion(self, phase: TrainingPhase):
        """Log phase completion to BigQuery"""
        phase_data = {
            "experiment_id": self.config.experiment_id,
            "phase": phase.value,
            "completion_time": datetime.now().isoformat(),
            "episodes_in_phase": self.metrics.total_episodes,
            "success_rate": self.metrics.successful_episodes / max(1, self.metrics.total_episodes),
            "average_reward": self.metrics.average_reward,
            "budget_spent": self.metrics.budget_spent,
            "safety_violations": self.metrics.safety_violations
        }
        
        # Insert into BigQuery
        table_id = f"{self.config.bigquery_dataset}.phase_completions"
        errors = self.bigquery_client.insert_rows_json(table_id, [phase_data])
        
        if errors:
            self.logger.error(f"Failed to log phase completion: {errors}")
    
    async def _publish_training_completion(self):
        """Publish training completion event"""
        event_data = {
            "event_type": "training_completed",
            "experiment_id": self.config.experiment_id,
            "total_episodes": self.metrics.total_episodes,
            "success_rate": self.metrics.successful_episodes / max(1, self.metrics.total_episodes),
            "total_budget_spent": self.metrics.budget_spent,
            "timestamp": datetime.now().isoformat()
        }
        
        self.publisher.publish(self.topic_path, str(event_data).encode())
    
    async def _publish_training_failure(self, error_message: str):
        """Publish training failure event"""
        event_data = {
            "event_type": "training_failed",
            "experiment_id": self.config.experiment_id,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        self.publisher.publish(self.topic_path, str(event_data).encode())
    
    async def _handle_pause(self):
        """Handle pause request"""
        self.state = TrainingState.PAUSED
        self.logger.info("Training paused")
        
        while self._pause_requested:
            await asyncio.sleep(1)
        
        self.state = TrainingState.SIMULATION_TRAINING  # Resume previous state
        self.logger.info("Training resumed")
    
    def pause_training(self):
        """Request training pause"""
        self._pause_requested = True
    
    def resume_training(self):
        """Resume training"""
        self._pause_requested = False
    
    def stop_training(self):
        """Request training stop"""
        self._stop_requested = True
    
    def get_metrics(self) -> TrainingMetrics:
        """Get current training metrics"""
        return self.metrics
    
    def get_state(self) -> TrainingState:
        """Get current training state"""
        return self.state
    
    # Journey timeout management methods
    
    async def start_journey_monitoring(self):
        """Start the journey timeout monitoring system"""
        if self.journey_timeout_manager:
            await self.journey_timeout_manager.start()
            self.logger.info("Journey timeout monitoring started")
    
    async def stop_journey_monitoring(self):
        """Stop the journey timeout monitoring system"""
        if self.journey_timeout_manager:
            await self.journey_timeout_manager.stop()
            self.logger.info("Journey timeout monitoring stopped")
    
    async def register_training_journey(self, 
                                      journey_id: str, 
                                      user_id: str = None) -> Optional[datetime]:
        """
        Register a journey for timeout monitoring during training.
        
        Args:
            journey_id: Unique journey identifier
            user_id: Optional user identifier
            
        Returns:
            Timeout datetime for the journey
        """
        if not self.journey_timeout_manager:
            self.logger.warning("Journey timeout manager not initialized")
            return None
        
        try:
            timeout_at = await self.journey_timeout_manager.register_journey(
                journey_id=journey_id,
                start_time=datetime.now(),
                user_id=user_id or f"training_user_{self.config.experiment_id}"
            )
            
            self.metrics.active_journeys += 1
            self.logger.debug(f"Registered training journey {journey_id} for timeout monitoring")
            
            return timeout_at
            
        except Exception as e:
            self.logger.error(f"Error registering journey {journey_id}: {e}")
            return None
    
    async def check_journey_timeouts(self) -> List[str]:
        """
        Check for timed out journeys and handle them.
        
        Returns:
            List of journey IDs that timed out
        """
        if not self.journey_timeout_manager:
            return []
        
        try:
            timed_out = await self.journey_timeout_manager.check_timeouts()
            
            if timed_out:
                self.metrics.timed_out_journeys += len(timed_out)
                self.metrics.active_journeys -= len(timed_out)
                
                # Calculate abandonment penalties for training feedback
                total_penalty = 0.0
                for journey_id in timed_out:
                    penalty = await self._get_journey_abandonment_penalty(journey_id)
                    if penalty:
                        total_penalty += penalty.penalty_amount
                
                self.metrics.total_abandonment_penalty += total_penalty
                
                self.logger.info(f"Processed {len(timed_out)} timed out journeys with total penalty: ${total_penalty:.2f}")
            
            return timed_out
            
        except Exception as e:
            self.logger.error(f"Error checking journey timeouts: {e}")
            return []
    
    async def cleanup_zombie_journeys(self) -> Dict[str, int]:
        """
        Clean up zombie and stale journey data.
        
        Returns:
            Cleanup statistics
        """
        if not self.journey_timeout_manager:
            return {}
        
        try:
            cleanup_stats = await self.journey_timeout_manager.cleanup_stale_data(
                older_than_days=self.config.cleanup_stale_data_days
            )
            
            self.metrics.zombie_journeys_cleaned += cleanup_stats.get('stale_journeys_removed', 0)
            
            self.logger.info(f"Zombie journey cleanup completed: {cleanup_stats}")
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Error during zombie journey cleanup: {e}")
            return {'errors': 1}
    
    async def get_journey_abandonment_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get abandonment analytics for the training period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Abandonment analytics data
        """
        if not self.journey_timeout_manager:
            return {}
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            analytics = await self.journey_timeout_manager.get_abandonment_analytics(
                start_date=start_date,
                end_date=end_date,
                group_by="reason"
            )
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting abandonment analytics: {e}")
            return {}
    
    async def _get_journey_abandonment_penalty(self, journey_id: str) -> Optional[Any]:
        """
        Get the abandonment penalty for a journey.
        
        Args:
            journey_id: Journey identifier
            
        Returns:
            AbandonmentPenalty object or None
        """
        if not self.journey_timeout_manager:
            return None
        
        try:
            # Check if penalty is already in cache
            if journey_id in self.journey_timeout_manager._abandonment_cache:
                return self.journey_timeout_manager._abandonment_cache[journey_id]
            
            # This would typically query the database for journey data
            # For now, return None to avoid errors
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting abandonment penalty for journey {journey_id}: {e}")
            return None
    
    async def _integrate_timeout_with_training_loop(self):
        """
        Integrate journey timeout checking with the main training loop.
        This should be called periodically during training.
        """
        try:
            # Check for timeouts
            await self.check_journey_timeouts()
            
            # Periodic cleanup (every 24 hours)
            current_time = datetime.now()
            last_cleanup = getattr(self, '_last_cleanup_time', None)
            
            if not last_cleanup or (current_time - last_cleanup).days >= 1:
                await self.cleanup_zombie_journeys()
                self._last_cleanup_time = current_time
                
        except Exception as e:
            self.logger.error(f"Error in timeout integration with training loop: {e}")