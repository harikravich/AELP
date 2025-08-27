#!/usr/bin/env python3
"""Minimal test of fixed simulator to verify conversions are happening"""

import logging
import numpy as np
from enhanced_simulator_fixed import FixedGAELPEnvironment

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_minimal():
    """Run minimal test to verify simulator is working"""
    
    logger.info("Creating environment...")
    env = FixedGAELPEnvironment(max_budget=500.0, max_steps=50)
    
    logger.info("Running 2 episodes with 20 steps each...")
    
    for episode in range(2):
        logger.info(f"\n=== Episode {episode + 1} ===")
        obs = env.reset(f"test_ep_{episode}")
        
        for step in range(20):
            # Vary bids to test auction mechanics
            bid = np.random.uniform(0.5, 3.0)
            
            action = {
                'bid': bid,
                'quality_score': 0.8,
                'creative': {
                    'id': f'creative_{step % 3}',
                    'quality_score': 0.75
                }
            }
            
            obs, reward, done, info = env.step(action)
            
            if step % 5 == 0:
                logger.info(f"Step {step}: Win rate={info['win_rate']:.1%}, Budget left=${obs['budget_remaining']:.2f}")
            
            if done:
                logger.info("Episode ended early (budget exhausted)")
                break
        
        # Report episode results
        logger.info(f"\nEpisode {episode + 1} Results:")
        logger.info(f"  Impressions: {env.metrics['total_impressions']}")
        logger.info(f"  Clicks: {env.metrics['total_clicks']}")
        logger.info(f"  Conversions: {env.metrics['total_conversions']}")
        logger.info(f"  Delayed conversions scheduled: {env.metrics['delayed_conversions_scheduled']}")
        logger.info(f"  Delayed conversions executed: {env.metrics['delayed_conversions_executed']}")
        logger.info(f"  Win rate: {info['win_rate']:.1%}")
        logger.info(f"  Unique users: {len(env.metrics['unique_users'])}")
        logger.info(f"  Persistent users: {len(env.metrics['persistent_users'])}")
    
    # Final check
    logger.info("\n=== FINAL VERIFICATION ===")
    
    success = True
    
    # Check conversions
    if env.metrics['total_conversions'] == 0 and env.metrics['delayed_conversions_scheduled'] == 0:
        logger.error("❌ FAIL: No conversions happening!")
        success = False
    else:
        logger.info(f"✅ Conversions working: {env.metrics['total_conversions']} executed, {env.metrics['delayed_conversions_scheduled']} scheduled")
    
    # Check win rate
    win_rate = env.metrics['auction_wins'] / max(1, env.metrics['auction_wins'] + env.metrics['auction_losses'])
    if win_rate > 0.8:
        logger.error(f"❌ FAIL: Win rate too high: {win_rate:.1%}")
        success = False
    else:
        logger.info(f"✅ Auction mechanics fixed: {win_rate:.1%} win rate")
    
    # Check user persistence
    if len(env.metrics['persistent_users']) < 2:
        logger.error("❌ FAIL: Users not persisting")
        success = False
    else:
        logger.info(f"✅ Users persisting: {len(env.metrics['persistent_users'])} active users")
    
    return success

if __name__ == "__main__":
    success = test_minimal()
    
    if success:
        print("\n🎉 SUCCESS: Fixed simulator working properly!")
    else:
        print("\n❌ FAILURE: Simulator still has issues")
        exit(1)