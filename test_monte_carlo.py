#!/usr/bin/env python3
"""
Test script for Monte Carlo Simulator
"""

import asyncio
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_functionality():
    """Test basic Monte Carlo simulator functionality"""
    try:
        logger.info("Starting Monte Carlo Simulator test")
        
        # Import the simulator
        from monte_carlo_simulator import MonteCarloSimulator, WorldType
        
        logger.info("Successfully imported MonteCarloSimulator")
        
        # Create a small simulator for testing
        simulator = MonteCarloSimulator(
            n_worlds=5,  # Very small for testing
            max_concurrent_worlds=2
        )
        
        logger.info("Successfully created simulator instance")
        
        # Mock agent
        class TestAgent:
            def __init__(self):
                self.agent_id = "test_agent"
            
            def select_action(self, state, deterministic=False):
                return {
                    'bid': 5.0,
                    'budget': 500.0,
                    'creative': {'quality_score': 0.7},
                    'quality_score': 0.8
                }
        
        agent = TestAgent()
        logger.info("Created test agent")
        
        # Test single batch
        logger.info("Running episode batch...")
        experiences = await simulator.run_episode_batch(agent, batch_size=3)
        
        logger.info(f"Completed batch with {len(experiences)} experiences")
        
        # Test experience aggregation
        if experiences:
            aggregated = simulator.aggregate_experiences(experiences)
            logger.info(f"Aggregated results - Total reward: {aggregated.get('total_reward', 0):.2f}")
        
        # Test importance sampling
        if len(experiences) > 0:
            samples = simulator.importance_sampling(target_samples=2)
            logger.info(f"Importance sampling returned {len(samples)} samples")
        
        # Get stats
        stats = simulator.get_simulation_stats()
        logger.info(f"Simulation stats: {stats['simulation_overview']['total_episodes_run']} total episodes")
        
        # Cleanup
        simulator.cleanup()
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main test function"""
    logger.info("Starting Monte Carlo Simulator tests...")
    
    success = await test_basic_functionality()
    
    if success:
        logger.info("All tests passed!")
        sys.exit(0)
    else:
        logger.error("Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())