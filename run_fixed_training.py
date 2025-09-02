#!/usr/bin/env python3
"""FIXED PRODUCTION TRAINING - Optimized for learning conversions"""
import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

import logging
from datetime import datetime

# Import production components
from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent
from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
from creative_selector import CreativeSelector
from attribution_models import AttributionEngine
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from gaelp_parameter_manager import ParameterManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixed_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("FIXED TRAINING - OPTIMIZED FOR CONVERSIONS")
    logger.info("="*70)
    
    # Initialize components
    discovery = DiscoveryEngine(write_enabled=False, cache_only=True)
    creative_selector = CreativeSelector()
    attribution = AttributionEngine()
    budget_pacer = BudgetPacer()
    identity_resolver = IdentityResolver()
    pm = ParameterManager()
    
    # Create environment
    env = ProductionFortifiedEnvironment(
        parameter_manager=pm,
        use_real_ga4_data=False,
        is_parallel=False
    )
    
    # Create agent with FIXED epsilon settings
    agent = ProductionFortifiedRLAgent(
        discovery_engine=discovery,
        creative_selector=creative_selector,
        attribution_engine=attribution,
        budget_pacer=budget_pacer,
        identity_resolver=identity_resolver,
        parameter_manager=pm,
        epsilon=0.3,  # Start with more exploration
        learning_rate=5e-4  # Faster learning
    )
    
    logger.info(f"Starting with epsilon: {agent.epsilon:.3f}")
    logger.info(f"Channels: {agent.discovered_channels}")
    logger.info(f"Segments: {agent.discovered_segments}")
    
    # Training metrics
    total_conversions = 0
    total_revenue = 0
    total_spend = 0
    
    # Run MORE EPISODES with SHORTER length
    num_episodes = 50
    max_steps = 500  # Shorter episodes
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_conversions = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            state = env.current_user_state
            action = agent.select_action(state, explore=True)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_state = env.current_user_state
            agent.train(state, action, reward, next_state, done)
            
            episode_reward += reward
            step += 1
            
            # Check for conversions
            metrics = info.get('metrics', {})
            new_conversions = metrics.get('total_conversions', 0) - total_conversions
            if new_conversions > 0:
                logger.info(f"🎯 CONVERSION! Episode {episode}, Step {step}")
                episode_conversions += new_conversions
                total_conversions += new_conversions
                total_revenue = metrics.get('total_revenue', 0)
            
            total_spend = metrics.get('budget_spent', 0)
            
            if step % 100 == 0 and step > 0:
                roas = total_revenue / max(1, total_spend)
                logger.info(f"Ep {episode}, Step {step}: "
                          f"Conv={total_conversions}, "
                          f"ROAS={roas:.2f}, "
                          f"ε={agent.epsilon:.3f}")
        
        # Episode summary
        roas = total_revenue / max(1, total_spend)
        logger.info(f"Episode {episode}: "
                   f"Reward={episode_reward:.1f}, "
                   f"Conv={episode_conversions}, "
                   f"Total Conv={total_conversions}, "
                   f"ROAS={roas:.2f}x, "
                   f"ε={agent.epsilon:.3f}")
        
        # Success check
        if total_conversions > 10:
            logger.info(f"✅ SUCCESS! Got {total_conversions} conversions!")
            if roas > 1.0:
                logger.info(f"✅ PROFITABLE! ROAS = {roas:.2f}x")
    
    # Final summary
    logger.info("="*70)
    logger.info(f"TRAINING COMPLETE:")
    logger.info(f"  Total Conversions: {total_conversions}")
    logger.info(f"  Total Revenue: ${total_revenue:.2f}")
    logger.info(f"  Total Spend: ${total_spend:.2f}")
    if total_spend > 0:
        logger.info(f"  Final ROAS: {total_revenue/total_spend:.2f}x")
    logger.info(f"  Final Epsilon: {agent.epsilon:.3f}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
