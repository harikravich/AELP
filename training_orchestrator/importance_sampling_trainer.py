"""
Enhanced Training Orchestrator with Importance Sampling for Crisis Parent Weighting

This module extends the core training orchestrator to integrate importance sampling
for proper crisis parent weighting (10% population, 50% value) with 5x weight.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

from .core import TrainingOrchestrator, TrainingConfiguration, TrainingMetrics
from .rl_agents.importance_sampling_integration import (
    ImportanceSamplingReplayBuffer,
    ExperienceAggregator, 
    integrate_importance_sampling_with_training
)
from ..importance_sampler import ImportanceSampler


logger = logging.getLogger(__name__)


class ImportanceSamplingTrainingConfiguration(TrainingConfiguration):
    """Extended configuration with importance sampling parameters"""
    
    # Importance sampling parameters
    enable_importance_sampling: bool = True
    crisis_parent_weight: float = 5.0  # 5x weight for crisis parents
    importance_sampling_alpha: float = 0.6
    importance_sampling_beta: float = 0.4
    replay_buffer_capacity: int = 50000
    
    # Crisis parent identification
    crisis_indicators: List[str] = None
    value_threshold_percentile: float = 90.0  # Top 10% by value
    
    # Sampling strategy
    use_importance_sampling: bool = True
    temperature: float = 1.0
    min_crisis_parent_ratio: float = 0.05  # Minimum 5% crisis parents in batch


class ImportanceSamplingTrainingMetrics(TrainingMetrics):
    """Extended metrics with importance sampling statistics"""
    
    # Importance sampling metrics
    crisis_parent_experiences: int = 0
    regular_parent_experiences: int = 0
    crisis_parent_ratio_in_batches: float = 0.0
    average_importance_weight: float = 1.0
    
    # Crisis parent performance
    crisis_parent_success_rate: float = 0.0
    crisis_parent_average_value: float = 0.0
    regular_parent_average_value: float = 0.0


class ImportanceSamplingTrainingOrchestrator(TrainingOrchestrator):
    """
    Enhanced training orchestrator with importance sampling for crisis parent weighting.
    
    Ensures that crisis parents (rare but valuable) receive proper attention during
    training through weighted sampling and bias correction.
    """
    
    def __init__(self, config: ImportanceSamplingTrainingConfiguration):
        super().__init__(config)
        
        self.is_config = config
        self.is_metrics = ImportanceSamplingTrainingMetrics()
        
        # Initialize importance sampling components
        self.importance_sampler = ImportanceSampler(
            population_ratios={"crisis_parent": 0.1, "regular_parent": 0.9},
            conversion_ratios={"crisis_parent": 0.5, "regular_parent": 0.5},
            max_weight=config.crisis_parent_weight,
            alpha=config.importance_sampling_alpha,
            beta_start=config.importance_sampling_beta
        )
        
        self.importance_buffer = None
        self.experience_aggregator = None
        
        self.logger.info(f"Importance sampling training orchestrator initialized with {config.crisis_parent_weight}x crisis parent weighting")
    
    async def start_training(self, agent, environments: Dict[str, Any]) -> bool:
        """
        Start training with importance sampling integration
        """
        try:
            # Initialize importance sampling components
            self.importance_buffer, self.experience_aggregator = integrate_importance_sampling_with_training(
                agent=agent,
                replay_buffer_capacity=self.is_config.replay_buffer_capacity,
                importance_sampler=self.importance_sampler
            )
            
            # Set up crisis indicators if provided
            if self.is_config.crisis_indicators:
                self.experience_aggregator.crisis_indicators = self.is_config.crisis_indicators
            
            # Run the standard training process with importance sampling enhancements
            return await super().start_training(agent, environments)
            
        except Exception as e:
            self.logger.error(f"Importance sampling training failed: {e}")
            return False
    
    async def _run_simulation_phase(self, agent, environments) -> bool:
        """Enhanced simulation phase with importance sampling"""
        self.logger.info("Starting simulation training phase with importance sampling")
        self.is_metrics.current_phase = self.config.TrainingPhase.SIMULATION if hasattr(self.config, 'TrainingPhase') else "SIMULATION"
        self.is_metrics.phase_start_time = datetime.now()
        
        simulation_env = environments.get("simulation")
        if not simulation_env:
            raise ValueError("Simulation environment not provided")
        
        episodes_completed = 0
        performance_history = []
        batch_crisis_ratios = []
        
        while episodes_completed < self.config.simulation_episodes:
            if self._stop_requested:
                return False
            
            if self._pause_requested:
                await self._handle_pause()
            
            # Run episode with experience collection
            episode_result = await self.episode_manager.run_episode(
                agent, simulation_env, f"sim_{episodes_completed}"
            )
            
            # Process experiences through importance sampling pipeline
            if hasattr(episode_result, 'experiences') and episode_result.experiences:
                aggregation_stats = self.experience_aggregator.aggregate_experiences(
                    episode_result.experiences, 
                    self.importance_buffer
                )
                
                # Update crisis parent statistics
                self.is_metrics.crisis_parent_experiences += aggregation_stats.get('crisis_parent', 0)
                self.is_metrics.regular_parent_experiences += aggregation_stats.get('regular_parent', 0)
            
            episodes_completed += 1
            self.is_metrics.total_episodes += 1
            
            if episode_result.success:
                self.is_metrics.successful_episodes += 1
                self.is_metrics.total_reward += episode_result.total_reward
                self.is_metrics.average_reward = self.is_metrics.total_reward / self.is_metrics.successful_episodes
                performance_history.append(episode_result.total_reward)
            
            # Update agent with importance-weighted experiences
            if len(self.importance_buffer) >= 64:  # Minimum batch size
                await self._update_agent_with_importance_sampling(agent)
            
            # Update curriculum if enabled
            if self.config.curriculum_enabled:
                self.curriculum_scheduler.update_curriculum(
                    phase="SIMULATION",  # Use string instead of enum
                    performance_history=performance_history[-self.config.performance_window:]
                )
            
            # Performance monitoring with importance sampling metrics
            should_continue = self.performance_monitor.check_phase_progression(
                phase="SIMULATION",
                performance_history=performance_history[-self.config.performance_window:]
            )
            
            # Safety monitoring
            safety_ok = await self.safety_monitor.check_safety_constraints(
                episode_result, "SIMULATION"
            )
            
            if not safety_ok:
                self.is_metrics.safety_violations += 1
                self.logger.warning("Safety violation detected in simulation phase")
            
            # Checkpoint saving
            if episodes_completed % self.config.checkpoint_interval == 0:
                await self._save_checkpoint_with_importance_sampling(agent, f"sim_episode_{episodes_completed}")
            
            # Log progress with importance sampling statistics
            if episodes_completed % 100 == 0:
                sampling_stats = self.importance_buffer.get_sampling_statistics()
                crisis_ratio = self.is_metrics.crisis_parent_experiences / max(1, 
                    self.is_metrics.crisis_parent_experiences + self.is_metrics.regular_parent_experiences)
                
                self.logger.info(
                    f"Simulation progress: {episodes_completed}/{self.config.simulation_episodes} episodes, "
                    f"avg reward: {self.is_metrics.average_reward:.3f}, "
                    f"crisis parent ratio: {crisis_ratio:.2%}, "
                    f"crisis weight: {sampling_stats.get('crisis_parent_weight', 1.0):.1f}x"
                )
        
        # Check graduation criteria
        graduated = self.performance_monitor.check_graduation_criteria(
            "SIMULATION", performance_history
        )
        
        if not graduated:
            self.logger.warning("Failed to meet simulation phase graduation criteria")
            return False
        
        # Log importance sampling effectiveness
        await self._log_importance_sampling_effectiveness()
        
        self.logger.info("Simulation phase with importance sampling completed successfully")
        await self._log_phase_completion("SIMULATION")
        return True
    
    async def _update_agent_with_importance_sampling(self, agent):
        """Update agent using importance-weighted experience sampling"""
        
        try:
            # Sample experiences with importance weighting
            batch_dict, importance_weights, sampled_indices = self.importance_buffer.sample_importance_weighted(
                batch_size=64
            )
            
            # Calculate crisis parent ratio in batch
            crisis_count = sum(1 for event_type in batch_dict['event_types'] if event_type == 'crisis_parent')
            crisis_ratio = crisis_count / len(batch_dict['event_types'])
            
            # Update metrics
            self.is_metrics.crisis_parent_ratio_in_batches = (
                self.is_metrics.crisis_parent_ratio_in_batches * 0.9 + crisis_ratio * 0.1
            )
            self.is_metrics.average_importance_weight = np.mean(importance_weights)
            
            # Convert batch to experiences format for agent
            experiences = []
            for i in range(len(batch_dict['states'])):
                exp = {
                    'state': batch_dict['states'][i],
                    'action': batch_dict['actions'][i],
                    'reward': batch_dict['rewards'][i],
                    'next_state': batch_dict['next_states'][i],
                    'done': batch_dict['dones'][i],
                    'event_type': batch_dict['event_types'][i],
                    'value': batch_dict['values'][i],
                    'importance_weight': importance_weights[i]
                }
                experiences.append(exp)
            
            # Update agent policy with importance-weighted experiences
            if hasattr(agent, 'update_policy'):
                training_metrics = agent.update_policy(experiences)
                
                # Log training metrics with importance sampling info
                if training_metrics:
                    self.logger.debug(
                        f"Agent update: policy_loss={training_metrics.get('policy_loss', 0):.4f}, "
                        f"crisis_ratio={crisis_ratio:.2%}, "
                        f"avg_importance_weight={self.is_metrics.average_importance_weight:.3f}"
                    )
            
        except Exception as e:
            self.logger.error(f"Error updating agent with importance sampling: {e}")
    
    async def _save_checkpoint_with_importance_sampling(self, agent, checkpoint_name: str):
        """Save checkpoint including importance sampling state"""
        
        checkpoint_data = {
            "agent_state": agent.get_state(),
            "metrics": self.is_metrics,
            "config": self.is_config,
            "importance_sampler_state": self.importance_sampler.get_sampling_statistics(),
            "importance_buffer_state": self.importance_buffer.get_state(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to Redis for quick access
        checkpoint_key = f"checkpoint:{self.config.experiment_id}:{checkpoint_name}"
        self.redis_client.hset(checkpoint_key, mapping={
            k: str(v) for k, v in checkpoint_data.items()
        })
        
        self.is_metrics.last_checkpoint = checkpoint_name
        self.logger.info(f"Checkpoint with importance sampling saved: {checkpoint_name}")
    
    async def _log_importance_sampling_effectiveness(self):
        """Log statistics about importance sampling effectiveness"""
        
        sampling_stats = self.importance_buffer.get_sampling_statistics()
        buffer_stats = sampling_stats.get('buffer_stats', {})
        importance_stats = sampling_stats.get('importance_stats', {})
        
        effectiveness_data = {
            "experiment_id": self.config.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "crisis_parent_experiences": self.is_metrics.crisis_parent_experiences,
            "regular_parent_experiences": self.is_metrics.regular_parent_experiences,
            "crisis_parent_ratio_buffer": buffer_stats.get('event_ratios', {}).get('crisis_parent', 0),
            "crisis_parent_ratio_batches": self.is_metrics.crisis_parent_ratio_in_batches,
            "crisis_parent_weight": sampling_stats.get('crisis_parent_weight', 1.0),
            "average_importance_weight": self.is_metrics.average_importance_weight,
            "buffer_size": len(self.importance_buffer),
            "importance_sampling_frame_count": importance_stats.get('frame_count', 0)
        }
        
        # Log to BigQuery
        table_id = f"{self.config.bigquery_dataset}.importance_sampling_effectiveness"
        try:
            errors = self.bigquery_client.insert_rows_json(table_id, [effectiveness_data])
            if errors:
                self.logger.error(f"Failed to log importance sampling effectiveness: {errors}")
            else:
                self.logger.info("Importance sampling effectiveness logged successfully")
        except Exception as e:
            self.logger.error(f"Error logging importance sampling effectiveness: {e}")
    
    def get_importance_sampling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive importance sampling metrics"""
        
        sampling_stats = self.importance_buffer.get_sampling_statistics() if self.importance_buffer else {}
        
        return {
            "crisis_parent_experiences": self.is_metrics.crisis_parent_experiences,
            "regular_parent_experiences": self.is_metrics.regular_parent_experiences,
            "crisis_parent_ratio_in_batches": self.is_metrics.crisis_parent_ratio_in_batches,
            "average_importance_weight": self.is_metrics.average_importance_weight,
            "crisis_parent_weight": sampling_stats.get('crisis_parent_weight', 1.0),
            "regular_parent_weight": sampling_stats.get('regular_parent_weight', 1.0),
            "buffer_stats": sampling_stats.get('buffer_stats', {}),
            "importance_stats": sampling_stats.get('importance_stats', {})
        }
    
    def update_crisis_parent_ratios(
        self, 
        population_ratio: float = None,
        conversion_ratio: float = None
    ):
        """Update crisis parent ratios based on observed data"""
        
        if population_ratio is not None:
            self.importance_sampler.population_ratios['crisis_parent'] = population_ratio
            self.importance_sampler.population_ratios['regular_parent'] = 1.0 - population_ratio
        
        if conversion_ratio is not None:
            self.importance_sampler.conversion_ratios['crisis_parent'] = conversion_ratio
            self.importance_sampler.conversion_ratios['regular_parent'] = 1.0 - conversion_ratio
        
        # Update importance weights
        self.importance_sampler._update_importance_weights()
        
        # Update buffer if available
        if self.importance_buffer:
            self.importance_buffer.update_importance_weights(
                population_ratios=self.importance_sampler.population_ratios,
                conversion_ratios=self.importance_sampler.conversion_ratios
            )
        
        self.logger.info(f"Updated crisis parent ratios - population: {population_ratio}, conversion: {conversion_ratio}")


# Factory function to create importance sampling training orchestrator
def create_importance_sampling_trainer(
    experiment_id: str = None,
    crisis_parent_weight: float = 5.0,
    importance_sampling_alpha: float = 0.6,
    importance_sampling_beta: float = 0.4,
    **kwargs
) -> ImportanceSamplingTrainingOrchestrator:
    """
    Create an importance sampling training orchestrator with default configuration.
    
    Args:
        experiment_id: Unique experiment identifier
        crisis_parent_weight: Weight multiplier for crisis parents (default 5x)
        importance_sampling_alpha: Prioritization exponent
        importance_sampling_beta: Bias correction parameter
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ImportanceSamplingTrainingOrchestrator
    """
    
    config = ImportanceSamplingTrainingConfiguration(
        experiment_id=experiment_id or f"crisis_parent_training_{int(time.time())}",
        crisis_parent_weight=crisis_parent_weight,
        importance_sampling_alpha=importance_sampling_alpha,
        importance_sampling_beta=importance_sampling_beta,
        **kwargs
    )
    
    return ImportanceSamplingTrainingOrchestrator(config)


# Example usage
if __name__ == "__main__":
    # Create trainer with crisis parent weighting
    trainer = create_importance_sampling_trainer(
        experiment_id="crisis_parent_test",
        crisis_parent_weight=5.0,
        simulation_episodes=1000
    )
    
    print(f"Created importance sampling trainer: {trainer.config.experiment_id}")
    print(f"Crisis parent weight: {trainer.is_config.crisis_parent_weight}x")
    print(f"Importance sampling enabled: {trainer.is_config.enable_importance_sampling}")