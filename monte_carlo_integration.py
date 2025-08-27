#!/usr/bin/env python3
"""
Monte Carlo Integration with GAELP Training Orchestrator

This module shows how to integrate the Monte Carlo parallel simulation framework
with the existing GAELP training orchestrator for enhanced learning from
diverse scenarios.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from monte_carlo_simulator import MonteCarloSimulator, WorldType, EpisodeExperience

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloTrainingConfig:
    """Configuration for Monte Carlo enhanced training"""
    # Monte Carlo settings
    n_worlds: int = 100
    max_concurrent_worlds: int = 20
    crisis_parent_frequency: float = 0.10  # 10% frequency
    crisis_parent_value_multiplier: float = 5.0  # 50% of value
    
    # Training settings
    batch_size: int = 200
    importance_sampling_ratio: float = 0.5  # 50% importance sampled
    experience_buffer_size: int = 1000000
    
    # World distribution
    world_distribution: Dict[WorldType, float] = None
    
    def __post_init__(self):
        if self.world_distribution is None:
            self.world_distribution = {
                WorldType.NORMAL_MARKET: 0.25,
                WorldType.HIGH_COMPETITION: 0.20,
                WorldType.LOW_COMPETITION: 0.15,
                WorldType.SEASONAL_PEAK: 0.12,
                WorldType.ECONOMIC_DOWNTURN: 0.08,
                WorldType.CRISIS_PARENT: 0.10,
                WorldType.TECH_SAVVY: 0.04,
                WorldType.BUDGET_CONSCIOUS: 0.03,
                WorldType.IMPULSE_BUYER: 0.02,
                WorldType.LUXURY_SEEKER: 0.01
            }


class MonteCarloTrainingOrchestrator:
    """
    Enhanced training orchestrator that uses Monte Carlo parallel simulation
    for more diverse and efficient training data generation.
    """
    
    def __init__(self, config: MonteCarloTrainingConfig):
        self.config = config
        
        # Initialize Monte Carlo simulator
        self.monte_carlo_simulator = MonteCarloSimulator(
            n_worlds=config.n_worlds,
            world_types_distribution=config.world_distribution,
            max_concurrent_worlds=config.max_concurrent_worlds,
            experience_buffer_size=config.experience_buffer_size
        )
        
        # Training statistics
        self.total_training_steps = 0
        self.total_episodes_generated = 0
        self.crisis_parent_episodes = 0
        
        logger.info(f"Monte Carlo Training Orchestrator initialized with {config.n_worlds} parallel worlds")
    
    async def generate_training_batch(self, agent, target_batch_size: int = None) -> Dict[str, Any]:
        """
        Generate a diverse training batch using Monte Carlo simulation.
        
        Args:
            agent: The RL agent to generate experiences for
            target_batch_size: Size of training batch to generate
            
        Returns:
            Dict containing training batch and metadata
        """
        if target_batch_size is None:
            target_batch_size = self.config.batch_size
        
        # Generate episodes across parallel worlds
        episodes_to_generate = max(target_batch_size // 10, 50)  # Generate more episodes than needed
        
        logger.info(f"Generating {episodes_to_generate} episodes across {self.config.n_worlds} worlds")
        
        # Run episode batch
        experiences = await self.monte_carlo_simulator.run_episode_batch(
            agent, 
            batch_size=episodes_to_generate
        )
        
        # Update statistics
        self.total_episodes_generated += len(experiences)
        self.crisis_parent_episodes += sum(exp.crisis_parent_interactions for exp in experiences)
        
        # Create training batch with importance sampling
        training_batch = self._create_training_batch(experiences, target_batch_size)
        
        return training_batch
    
    def _create_training_batch(self, experiences: List[EpisodeExperience], target_size: int) -> Dict[str, Any]:
        """Create training batch from experiences with importance sampling"""
        
        # Separate regular and importance sampled experiences
        importance_samples_count = int(target_size * self.config.importance_sampling_ratio)
        regular_samples_count = target_size - importance_samples_count
        
        # Get importance sampled experiences (focus on crisis parents)
        importance_samples = self.monte_carlo_simulator.importance_sampling(
            target_samples=importance_samples_count,
            focus_rare_events=True
        )
        
        # Get regular samples
        regular_samples = self.monte_carlo_simulator.experience_buffer.sample_batch(
            regular_samples_count,
            importance_sampling=False
        )
        
        # Combine samples
        all_samples = importance_samples + regular_samples
        np.random.shuffle(all_samples)  # Shuffle to mix importance and regular samples
        
        # Convert to training format
        training_data = self._prepare_training_data(all_samples[:target_size])
        
        # Add metadata
        training_batch = {
            'training_data': training_data,
            'batch_size': len(training_data['states']),
            'importance_sampled_ratio': len(importance_samples) / len(all_samples[:target_size]),
            'crisis_parent_experiences': sum(1 for exp in all_samples[:target_size] if exp.crisis_parent_interactions > 0),
            'world_type_distribution': self._get_world_type_distribution(all_samples[:target_size]),
            'average_importance_weight': np.mean([exp.importance_weight for exp in all_samples[:target_size]]),
            'generation_metadata': {
                'total_episodes_generated': len(experiences),
                'total_worlds_used': len(set(exp.world_id for exp in experiences)),
                'crisis_interactions_generated': sum(exp.crisis_parent_interactions for exp in experiences)
            }
        }
        
        return training_batch
    
    def _prepare_training_data(self, experiences: List[EpisodeExperience]) -> Dict[str, List]:
        """Convert experiences to training data format"""
        states, actions, rewards, next_states, dones, importance_weights = [], [], [], [], [], []
        
        for exp in experiences:
            for i in range(len(exp.states)):
                states.append(exp.states[i])
                actions.append(exp.actions[i])
                rewards.append(exp.rewards[i])
                dones.append(exp.dones[i])
                importance_weights.append(exp.importance_weight)
                
                # Next state
                if i < len(exp.states) - 1:
                    next_states.append(exp.states[i + 1])
                else:
                    next_states.append(exp.states[i])  # Terminal state
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'importance_weights': importance_weights
        }
    
    def _get_world_type_distribution(self, experiences: List[EpisodeExperience]) -> Dict[str, int]:
        """Get distribution of world types in the batch"""
        world_type_counts = {}
        for exp in experiences:
            world_type = exp.world_type.value
            world_type_counts[world_type] = world_type_counts.get(world_type, 0) + 1
        return world_type_counts
    
    async def train_agent_with_monte_carlo(self, agent, n_training_steps: int = 1000) -> Dict[str, Any]:
        """
        Train agent using Monte Carlo generated experiences.
        
        Args:
            agent: RL agent to train
            n_training_steps: Number of training steps to perform
            
        Returns:
            Training results and statistics
        """
        logger.info(f"Starting Monte Carlo enhanced training for {n_training_steps} steps")
        
        training_results = {
            'training_losses': [],
            'batch_statistics': [],
            'crisis_parent_utilization': [],
            'world_performance': {}
        }
        
        for step in range(n_training_steps):
            # Generate diverse training batch
            training_batch = await self.generate_training_batch(agent)
            
            # Train agent on batch
            if hasattr(agent, 'update_policy'):
                loss_metrics = agent.update_policy(training_batch['training_data'])
                training_results['training_losses'].append(loss_metrics)
            
            # Record batch statistics
            batch_stats = {
                'step': step,
                'batch_size': training_batch['batch_size'],
                'crisis_parent_experiences': training_batch['crisis_parent_experiences'],
                'importance_sampled_ratio': training_batch['importance_sampled_ratio'],
                'world_diversity': len(training_batch['world_type_distribution']),
                'average_importance_weight': training_batch['average_importance_weight']
            }
            training_results['batch_statistics'].append(batch_stats)
            
            # Calculate crisis parent utilization
            crisis_utilization = training_batch['crisis_parent_experiences'] / training_batch['batch_size']
            training_results['crisis_parent_utilization'].append(crisis_utilization)
            
            self.total_training_steps += 1
            
            # Log progress
            if step % 100 == 0:
                logger.info(f"Training step {step}/{n_training_steps}")
                logger.info(f"  Crisis parent utilization: {crisis_utilization:.1%}")
                logger.info(f"  World diversity: {len(training_batch['world_type_distribution'])} types")
                logger.info(f"  Avg importance weight: {training_batch['average_importance_weight']:.2f}")
        
        # Final statistics
        final_stats = self.get_training_statistics()
        training_results.update(final_stats)
        
        logger.info(f"Monte Carlo training completed!")
        logger.info(f"  Total episodes generated: {self.total_episodes_generated}")
        logger.info(f"  Crisis parent episodes: {self.crisis_parent_episodes}")
        logger.info(f"  Crisis parent rate: {self.crisis_parent_episodes / max(1, self.total_episodes_generated):.1%}")
        
        return training_results
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        sim_stats = self.monte_carlo_simulator.get_simulation_stats()
        buffer_stats = self.monte_carlo_simulator.experience_buffer.get_buffer_stats()
        
        return {
            'training_overview': {
                'total_training_steps': self.total_training_steps,
                'total_episodes_generated': self.total_episodes_generated,
                'crisis_parent_episodes': self.crisis_parent_episodes,
                'crisis_parent_rate': self.crisis_parent_episodes / max(1, self.total_episodes_generated),
                'episodes_per_second': sim_stats['simulation_overview']['episodes_per_second']
            },
            'world_statistics': sim_stats['world_statistics'],
            'buffer_statistics': buffer_stats,
            'simulation_performance': sim_stats['simulation_overview']
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.monte_carlo_simulator.cleanup()
        logger.info("Monte Carlo Training Orchestrator cleanup completed")


# Integration example with existing GAELP components
async def integration_example():
    """Example showing integration with GAELP training orchestrator"""
    
    print("🔗 MONTE CARLO INTEGRATION EXAMPLE")
    print("=" * 50)
    
    # Create Monte Carlo training orchestrator
    config = MonteCarloTrainingConfig(
        n_worlds=50,  # Smaller for example
        max_concurrent_worlds=10,
        batch_size=100,
        importance_sampling_ratio=0.4  # 40% importance sampled
    )
    
    orchestrator = MonteCarloTrainingOrchestrator(config)
    
    # Mock agent
    class MockAgent:
        def __init__(self):
            self.training_step = 0
        
        def select_action(self, state, deterministic=False):
            return {
                'bid': np.random.uniform(1.0, 10.0),
                'budget': np.random.uniform(100.0, 1000.0),
                'creative': {'quality_score': np.random.uniform(0.5, 1.0)},
                'quality_score': np.random.uniform(0.6, 1.0)
            }
        
        def update_policy(self, training_data):
            self.training_step += 1
            return {
                'policy_loss': np.random.uniform(0.1, 0.5),
                'value_loss': np.random.uniform(0.05, 0.3),
                'entropy_loss': np.random.uniform(0.01, 0.1)
            }
    
    agent = MockAgent()
    
    print(f"✅ Created orchestrator with {config.n_worlds} worlds")
    print(f"📊 Batch size: {config.batch_size}")
    print(f"🎯 Importance sampling: {config.importance_sampling_ratio:.0%}")
    
    try:
        # Generate a training batch
        print("\n🔄 Generating training batch...")
        training_batch = await orchestrator.generate_training_batch(agent)
        
        print(f"✅ Generated batch with {training_batch['batch_size']} experiences")
        print(f"🚨 Crisis parent experiences: {training_batch['crisis_parent_experiences']}")
        print(f"📈 Importance sampled: {training_batch['importance_sampled_ratio']:.1%}")
        print(f"🌍 World types represented: {len(training_batch['world_type_distribution'])}")
        
        # Show world distribution
        print("\nWorld type distribution in batch:")
        for world_type, count in training_batch['world_type_distribution'].items():
            percentage = count / training_batch['batch_size'] * 100
            print(f"  {world_type}: {count} ({percentage:.1f}%)")
        
        # Run short training session
        print(f"\n🎓 Running training session...")
        results = await orchestrator.train_agent_with_monte_carlo(agent, n_training_steps=50)
        
        print(f"✅ Training completed!")
        print(f"📊 Final statistics:")
        overview = results['training_overview']
        print(f"  Episodes generated: {overview['total_episodes_generated']}")
        print(f"  Crisis parent rate: {overview['crisis_parent_rate']:.1%}")
        print(f"  Performance: {overview['episodes_per_second']:.1f} episodes/sec")
        
        # Average crisis parent utilization
        avg_crisis_util = np.mean(results['crisis_parent_utilization'])
        print(f"  Avg crisis utilization: {avg_crisis_util:.1%}")
        
        print(f"\n🎯 INTEGRATION SUCCESS!")
        print("Key Benefits:")
        print("✅ Parallel world simulation for diverse training data")
        print("✅ Automatic importance sampling for rare events")
        print("✅ Crisis parent events properly weighted")
        print("✅ Seamless integration with existing agents")
        print("✅ Real-time performance monitoring")
        
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(integration_example())