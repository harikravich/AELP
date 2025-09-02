#!/usr/bin/env python3
"""Start production training directly - NO HARDCODING"""
import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

import logging
import time
from datetime import datetime

# Import production components
from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent, DynamicEnrichedState
from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
from creative_selector import CreativeSelector
from attribution_models import AttributionEngine
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from gaelp_parameter_manager import ParameterManager

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fortified_training_output.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("STARTING PRODUCTION TRAINING - NO HARDCODING")
    logger.info("="*70)
    
    # Initialize components
    discovery = DiscoveryEngine(write_enabled=True, cache_only=False)
    creative_selector = CreativeSelector()
    attribution = AttributionEngine()
    budget_pacer = BudgetPacer()
    identity_resolver = IdentityResolver()
    pm = ParameterManager()
    
    # Create production environment
    logger.info("Creating production environment...")
    env = ProductionFortifiedEnvironment(
        parameter_manager=pm,
        use_real_ga4_data=False,
        is_parallel=False
    )
    
    # Create production agent
    logger.info("Creating production agent with discovered parameters...")
    agent = ProductionFortifiedRLAgent(
        discovery_engine=discovery,
        creative_selector=creative_selector,
        attribution_engine=attribution,
        budget_pacer=budget_pacer,
        identity_resolver=identity_resolver,
        parameter_manager=pm
    )
    
    logger.info(f"Agent initialized with:")
    logger.info(f"  - {len(agent.discovered_channels)} discovered channels: {agent.discovered_channels}")
    logger.info(f"  - {len(agent.discovered_segments)} discovered segments: {agent.discovered_segments}")
    logger.info(f"  - {len(agent.discovered_creatives)} discovered creatives")
    logger.info(f"  - Warm start enabled: {len(agent.replay_buffer)} samples")
    logger.info(f"  - Epsilon: {agent.epsilon:.3f}")
    
    # Training loop
    num_episodes = 100  # Start with fewer episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Episode metrics
        episode_conversions = 0
        episode_revenue = 0
        episode_spend = 0
        
        while not done and step < 1000:  # Max 1000 steps per episode
            # Get current state
            state = env.current_user_state
            
            # Select action
            action = agent.select_action(state, explore=True)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Get next state
            next_state = env.current_user_state
            
            # Train agent
            agent.train(state, action, reward, next_state, done)
            
            episode_reward += reward
            step += 1
            
            # Track metrics
            metrics = info.get('metrics', {})
            if 'total_conversions' in metrics:
                episode_conversions = metrics['total_conversions']
            if 'total_revenue' in metrics:
                episode_revenue = metrics['total_revenue']
            if 'budget_spent' in metrics:
                episode_spend = metrics['budget_spent']
            
            if step % 100 == 0:
                logger.info(f"Episode {episode}, Step {step}: Reward={reward:.2f}, "
                          f"Conversions={episode_conversions}, "
                          f"ROAS={episode_revenue/max(1, episode_spend):.2f}x, "
                          f"Epsilon={agent.epsilon:.3f}")
        
        # Log episode results
        logger.info(f"Episode {episode} complete:")
        logger.info(f"  Total Reward: {episode_reward:.2f}")
        logger.info(f"  Steps: {step}")
        logger.info(f"  Conversions: {episode_conversions}")
        logger.info(f"  Revenue: ${episode_revenue:.2f}")
        logger.info(f"  Spend: ${episode_spend:.2f}")
        if episode_spend > 0:
            logger.info(f"  ROAS: {episode_revenue/episode_spend:.2f}x")
        logger.info(f"  Epsilon: {agent.epsilon:.3f}")
        
        # Show channel performance
        if 'channel_performance' in metrics:
            logger.info("  Channel Performance:")
            for channel, perf in list(metrics['channel_performance'].items())[:3]:
                if perf.get('impressions', 0) > 0:
                    logger.info(f"    {channel}: {perf['impressions']} impr, "
                              f"{perf.get('conversions', 0)} conv")
        
        # Save model periodically
        if episode % 20 == 0 and episode > 0:
            logger.info(f"Checkpoint at episode {episode}")
    
    logger.info("="*70)
    logger.info("PRODUCTION TRAINING COMPLETE")
    logger.info(f"Final epsilon: {agent.epsilon:.3f}")
    logger.info(f"Total episodes: {num_episodes}")
    logger.info("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Shutting down...")