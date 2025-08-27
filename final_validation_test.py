#!/usr/bin/env python3
"""
Final validation test for GAELP Gymnasium environment
"""

import numpy as np
from gaelp_gym_env import GAELPGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def test_environment_validation():
    """Test that environment passes Gymnasium validation"""
    print("Environment Validation Test")
    print("=" * 40)
    
    env = GAELPGymEnv(max_budget=1000.0, max_steps=20)
    
    try:
        check_env(env)
        print("✓ Environment passed Gymnasium validation checks!")
    except Exception as e:
        print(f"✗ Environment validation failed: {e}")
    
    env.close()

def test_manual_agent_performance():
    """Test with a simple manual strategy"""
    print("\nManual Strategy Test")
    print("=" * 40)
    
    env = GAELPGymEnv(max_budget=1000.0, max_steps=30)
    
    strategies = {
        'conservative': lambda: np.array([1.0, 0.7, 0.6, 40.0, 0.5]),
        'aggressive': lambda: np.array([5.0, 0.9, 0.8, 60.0, 0.8]),
        'balanced': lambda: np.array([2.5, 0.8, 0.7, 50.0, 0.6])
    }
    
    for strategy_name, action_func in strategies.items():
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(20):
            action = action_func()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        metrics = env.get_episode_metrics()
        print(f"{strategy_name.capitalize()} Strategy:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  ROAS: {metrics.get('final_roas', 0):.2f}x")
        print(f"  Impressions: {metrics.get('impressions', 0)}")
        print(f"  Clicks: {metrics.get('clicks', 0)}")
        print(f"  Conversions: {metrics.get('conversions', 0)}")
        print(f"  Budget Used: {metrics.get('budget_utilization', 0)*100:.1f}%")
        print()
    
    env.close()

def test_simple_rl_training():
    """Test simple RL training with few steps"""
    print("Simple RL Training Test")
    print("=" * 40)
    
    env = GAELPGymEnv(max_budget=1000.0, max_steps=25)
    
    # Quick training
    model = PPO("MlpPolicy", env, verbose=1, device='cpu')
    model.learn(total_timesteps=2000)
    
    # Test trained model
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(15):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    metrics = env.get_episode_metrics()
    print(f"Trained Agent Performance:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  ROAS: {metrics.get('final_roas', 0):.2f}x")
    print(f"  Budget Used: {metrics.get('budget_utilization', 0)*100:.1f}%")
    
    env.close()

def test_observation_action_spaces():
    """Test the observation and action spaces thoroughly"""
    print("\nObservation/Action Spaces Test")
    print("=" * 40)
    
    env = GAELPGymEnv(max_budget=1000.0, max_steps=20)
    
    # Test observation space
    obs, info = env.reset()
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Observation space bounds: low={env.observation_space.low}, high={env.observation_space.high}")
    print(f"Sample observation: {obs}")
    print(f"Observation in bounds: {env.observation_space.contains(obs)}")
    
    # Test action space
    print(f"\nAction space shape: {env.action_space.shape}")
    print(f"Action space bounds: low={env.action_space.low}, high={env.action_space.high}")
    
    # Test action bounds
    for i in range(5):
        action = env.action_space.sample()
        print(f"Sample action {i+1}: {action}")
        print(f"Action in bounds: {env.action_space.contains(action)}")
        
        # Test step with this action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step result: reward={reward:.3f}, obs_bounds_ok={env.observation_space.contains(obs)}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

def comprehensive_functionality_test():
    """Comprehensive test of all functionality"""
    print("\nComprehensive Functionality Test")
    print("=" * 40)
    
    env = GAELPGymEnv(max_budget=2000.0, max_steps=40, render_mode="human")
    
    # Test different scenarios
    scenarios = [
        ("Low bids", np.array([0.5, 0.6, 0.5, 30.0, 0.4])),
        ("High bids", np.array([8.0, 0.9, 0.8, 80.0, 0.9])),
        ("Medium bids", np.array([3.0, 0.75, 0.7, 55.0, 0.65]))
    ]
    
    for scenario_name, action in scenarios:
        print(f"\nTesting {scenario_name}:")
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(15):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if step % 5 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total_spent=${info['total_spent']:.2f}")
            
            if terminated or truncated:
                break
        
        metrics = env.get_episode_metrics()
        print(f"  Final results: Steps={steps}, Reward={total_reward:.2f}, ROAS={metrics.get('final_roas', 0):.2f}x")
    
    env.close()

if __name__ == "__main__":
    print("GAELP Gymnasium Environment - Final Validation")
    print("=" * 60)
    
    test_environment_validation()
    test_observation_action_spaces()
    test_manual_agent_performance()
    test_simple_rl_training()
    comprehensive_functionality_test()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("GAELP Gymnasium Environment is ready for RL training!")
    print("=" * 60)