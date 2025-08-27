#!/usr/bin/env python3
"""
Test script to debug GAELP components and ensure they work correctly
"""

import numpy as np
from gaelp_gym_env import GAELPGymEnv
from enhanced_simulator import EnhancedGAELPEnvironment

def test_enhanced_simulator():
    """Test the enhanced simulator directly"""
    print("Testing Enhanced Simulator")
    print("=" * 40)
    
    env = EnhancedGAELPEnvironment(max_budget=1000, max_steps=50)
    obs = env.reset()
    
    for step in range(10):
        action = {
            'bid': 3.0,  # Higher bid to increase win chance
            'budget': 1000,
            'quality_score': 0.8,
            'creative': {
                'quality_score': 0.7,
                'price_shown': 50
            }
        }
        
        obs, reward, done, info = env.step(action)
        print(f"Step {step}: Won: {info.get('won', False)}, Cost: ${info.get('cost', 0):.2f}, "
              f"Revenue: ${info.get('revenue', 0):.2f}, Reward: {reward:.3f}")
        
        if done:
            break
    
    print(f"Final ROAS: {obs['roas']:.2f}x")
    print()

def test_gym_env():
    """Test the Gymnasium environment"""
    print("Testing Gymnasium Environment")
    print("=" * 40)
    
    env = GAELPGymEnv(max_budget=1000.0, max_steps=50)
    observation, info = env.reset(seed=42)
    
    print(f"Initial observation: {observation}")
    
    for step in range(10):
        # Use strategic action
        action = np.array([3.0, 0.8, 0.7, 50.0, 0.6])  # [bid, quality_score, creative_quality, price_shown, targeting]
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        if step % 2 == 0:
            print(f"Step {step}: Reward: {reward:.3f}, Total spent: ${info['total_spent']:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    final_metrics = env.get_episode_metrics()
    print(f"Final metrics: {final_metrics}")
    env.close()

def test_with_basic_rl():
    """Test with a simple random agent that learns"""
    print("Testing with Basic RL Agent")
    print("=" * 40)
    
    env = GAELPGymEnv(max_budget=1000.0, max_steps=30)
    
    # Simple strategy: start conservative, increase bids if performance is good
    base_bid = 1.0
    
    for episode in range(5):
        observation, info = env.reset()
        total_reward = 0
        episode_metrics = []
        
        for step in range(20):
            # Adaptive strategy
            if len(episode_metrics) > 5:
                recent_roas = np.mean([m['roas'] for m in episode_metrics[-3:]])
                if recent_roas > 2.0:
                    bid_multiplier = 1.2  # Increase bids if doing well
                elif recent_roas < 1.0:
                    bid_multiplier = 0.8  # Decrease bids if losing money
                else:
                    bid_multiplier = 1.0
            else:
                bid_multiplier = 1.0
            
            action = np.array([
                base_bid * bid_multiplier,
                np.random.uniform(0.6, 0.9),  # quality_score
                np.random.uniform(0.5, 0.8),  # creative_quality  
                np.random.uniform(30, 80),    # price_shown
                np.random.uniform(0.4, 0.8)   # targeting_precision
            ])
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            episode_metrics.append({
                'roas': info['raw_metrics']['roas'],
                'cost': info['total_spent'],
                'reward': reward
            })
            
            if terminated or truncated:
                break
        
        final_metrics = env.get_episode_metrics()
        print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}, "
              f"ROAS: {final_metrics['final_roas']:.2f}x, "
              f"Budget Used: {final_metrics['budget_utilization']*100:.1f}%")

if __name__ == "__main__":
    test_enhanced_simulator()
    test_gym_env()  
    test_with_basic_rl()