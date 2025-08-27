#!/usr/bin/env python3
"""
Simple Monte Carlo Simulation Demo

Shows the key features with a smaller scale for quick testing.
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime

from monte_carlo_simulator import MonteCarloSimulator, WorldType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleAgent:
    def __init__(self):
        self.agent_id = "simple_agent"
    
    def select_action(self, state, deterministic=False):
        return {
            'bid': np.random.uniform(1.0, 5.0),
            'budget': np.random.uniform(200.0, 800.0),
            'creative': {'quality_score': np.random.uniform(0.5, 0.9)},
            'quality_score': np.random.uniform(0.6, 0.9)
        }


async def simple_demo():
    """Run a simple Monte Carlo demo"""
    
    logger.info("Starting Simple Monte Carlo Demo")
    logger.info("=" * 50)
    
    # Create simulator with fewer worlds for quick demo
    simulator = MonteCarloSimulator(
        n_worlds=10,  # Small number for quick demo
        max_concurrent_worlds=5
    )
    
    agent = SimpleAgent()
    
    logger.info(f"Created simulator with {simulator.n_worlds} worlds")
    
    try:
        # Run a single batch
        logger.info("Running episode batch...")
        experiences = await simulator.run_episode_batch(agent, batch_size=15)
        
        logger.info(f"Completed {len(experiences)} episodes")
        
        # Aggregate experiences
        aggregated = simulator.aggregate_experiences(experiences)
        
        logger.info("Results:")
        logger.info(f"  Total Episodes: {aggregated.get('total_experiences', 0)}")
        logger.info(f"  Total Reward: {aggregated.get('total_reward', 0):.2f}")
        logger.info(f"  Average Reward: {aggregated.get('average_reward', 0):.3f}")
        logger.info(f"  Success Rate: {aggregated.get('success_rate', 0):.2%}")
        logger.info(f"  Crisis Interactions: {aggregated.get('total_crisis_interactions', 0)}")
        
        # Test importance sampling
        samples = simulator.importance_sampling(target_samples=10)
        crisis_samples = sum(1 for exp in samples if exp.crisis_parent_interactions > 0)
        
        logger.info(f"Importance sampling: {crisis_samples}/10 crisis parent samples")
        
        # Get buffer stats
        buffer_stats = simulator.experience_buffer.get_buffer_stats()
        logger.info(f"Buffer: {buffer_stats['total_experiences']} experiences, "
                   f"{buffer_stats['crisis_parent_experiences']} crisis parent")
        
        # Get simulation stats
        sim_stats = simulator.get_simulation_stats()
        overview = sim_stats['simulation_overview']
        
        logger.info(f"Performance: {overview['episodes_per_second']:.1f} episodes/second")
        
        # Show world breakdown
        logger.info("\nWorld Type Performance:")
        world_breakdown = aggregated.get('world_type_breakdown', {})
        for world_type, stats in world_breakdown.items():
            logger.info(f"  {world_type}: {stats.get('success_rate', 0):.2%} success, "
                       f"{stats.get('count', 0)} episodes")
        
        logger.info("\nDemo completed successfully!")
        
        return {
            'total_episodes': len(experiences),
            'average_reward': aggregated.get('average_reward', 0),
            'success_rate': aggregated.get('success_rate', 0),
            'crisis_interactions': aggregated.get('total_crisis_interactions', 0),
            'episodes_per_second': overview['episodes_per_second']
        }
        
    finally:
        simulator.cleanup()


if __name__ == "__main__":
    result = asyncio.run(simple_demo())
    print(f"\nFinal Result: {result}")