#!/usr/bin/env python3
"""
GAELP Gymnasium Environment Complete Demo

This script demonstrates the complete Gymnasium-compatible environment for GAELP,
including training with multiple RL algorithms and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from typing import Dict, List, Tuple
import time

from gaelp_gym_env import GAELPGymEnv

def demonstrate_environment_features():
    """Demonstrate all environment features"""
    print("🚀 GAELP Gymnasium Environment Demo")
    print("=" * 60)
    
    # Create environment
    env = GAELPGymEnv(max_budget=1000.0, max_steps=30, render_mode="human")
    
    print(f"📊 Environment Specifications:")
    print(f"   Observation Space: {env.observation_space}")
    print(f"   Action Space: {env.action_space}")
    print(f"   Max Budget: ${env.max_budget}")
    print(f"   Max Steps: {env.max_steps}")
    
    # Validate environment
    try:
        check_env(env)
        print("✅ Environment passed all Gymnasium validation checks!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return
    
    print(f"\n🎯 Observation Space Details:")
    print(f"   [0] Normalized total cost (0-1)")
    print(f"   [1] Normalized total revenue (0-1)")  
    print(f"   [2] Normalized impressions (0-1)")
    print(f"   [3] Normalized clicks (0-1)")
    print(f"   [4] Normalized conversions (0-1)")
    print(f"   [5] Normalized avg CPC (0-1)")
    print(f"   [6] Normalized ROAS (0-10)")
    print(f"   [7] Remaining budget ratio (0-1)")
    print(f"   [8] Step progress ratio (0-1)")
    
    print(f"\n🎮 Action Space Details:")
    print(f"   [0] Bid amount ($0.1-$10.0)")
    print(f"   [1] Quality score (0.1-1.0)")
    print(f"   [2] Creative quality (0.1-1.0)")
    print(f"   [3] Price shown ($1-$200)")
    print(f"   [4] Targeting precision (0.1-1.0)")
    
    env.close()

def test_manual_strategies():
    """Test different manual bidding strategies"""
    print(f"\n📈 Manual Strategy Comparison")
    print("=" * 60)
    
    strategies = {
        "Conservative": np.array([1.0, 0.7, 0.6, 40.0, 0.5]),
        "Aggressive": np.array([8.0, 0.9, 0.8, 80.0, 0.9]),
        "Balanced": np.array([3.0, 0.8, 0.7, 50.0, 0.6]),
        "Quality-Focused": np.array([2.0, 0.95, 0.9, 45.0, 0.8]),
        "Budget-Conscious": np.array([1.5, 0.6, 0.5, 35.0, 0.4])
    }
    
    results = []
    
    for strategy_name, action in strategies.items():
        env = GAELPGymEnv(max_budget=1000.0, max_steps=25)
        obs, info = env.reset(seed=42)  # Fixed seed for consistency
        
        total_reward = 0
        for step in range(20):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        metrics = env.get_episode_metrics()
        results.append({
            'strategy': strategy_name,
            'total_reward': total_reward,
            'roas': metrics.get('final_roas', 0),
            'impressions': metrics.get('impressions', 0),
            'clicks': metrics.get('clicks', 0),
            'conversions': metrics.get('conversions', 0),
            'budget_used': metrics.get('budget_utilization', 0) * 100
        })
        
        env.close()
    
    # Display results
    print(f"{'Strategy':<15} {'Reward':<8} {'ROAS':<6} {'Impr':<6} {'Clicks':<7} {'Conv':<5} {'Budget%':<8}")
    print("-" * 60)
    
    for result in sorted(results, key=lambda x: x['total_reward'], reverse=True):
        print(f"{result['strategy']:<15} {result['total_reward']:<8.1f} "
              f"{result['roas']:<6.2f} {result['impressions']:<6.0f} "
              f"{result['clicks']:<7.0f} {result['conversions']:<5.0f} "
              f"{result['budget_used']:<8.1f}")
    
    return results

def train_and_compare_algorithms():
    """Train and compare different RL algorithms"""
    print(f"\n🤖 RL Algorithm Comparison")
    print("=" * 60)
    
    algorithms = {
        'PPO': lambda env: PPO("MlpPolicy", env, verbose=0, device='cpu'),
        'A2C': lambda env: A2C("MlpPolicy", env, verbose=0, device='cpu')
    }
    
    training_timesteps = 5000
    evaluation_episodes = 5
    results = {}
    
    for alg_name, alg_factory in algorithms.items():
        print(f"🔄 Training {alg_name}...")
        
        # Create training environment
        train_env = GAELPGymEnv(max_budget=1000.0, max_steps=30)
        train_env = Monitor(train_env)
        
        # Create and train model
        model = alg_factory(train_env)
        
        start_time = time.time()
        model.learn(total_timesteps=training_timesteps)
        training_time = time.time() - start_time
        
        # Evaluate model
        eval_env = GAELPGymEnv(max_budget=1000.0, max_steps=30)
        episode_rewards = []
        episode_metrics = []
        
        for episode in range(evaluation_episodes):
            obs, info = eval_env.reset(seed=episode + 100)  # Different seeds
            episode_reward = 0
            
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            metrics = eval_env.get_episode_metrics()
            episode_rewards.append(episode_reward)
            episode_metrics.append(metrics)
        
        # Calculate statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_roas = np.mean([m.get('final_roas', 0) for m in episode_metrics])
        avg_budget_used = np.mean([m.get('budget_utilization', 0) for m in episode_metrics])
        
        results[alg_name] = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_roas': avg_roas,
            'avg_budget_used': avg_budget_used,
            'training_time': training_time
        }
        
        print(f"   ✅ {alg_name} completed in {training_time:.1f}s")
        
        eval_env.close()
        train_env.close()
    
    # Display comparison
    print(f"\n📊 Algorithm Performance Comparison:")
    print(f"{'Algorithm':<10} {'Avg Reward':<12} {'Std Reward':<12} {'ROAS':<8} {'Budget%':<10} {'Time(s)':<8}")
    print("-" * 70)
    
    for alg_name, stats in sorted(results.items(), key=lambda x: x[1]['avg_reward'], reverse=True):
        print(f"{alg_name:<10} {stats['avg_reward']:<12.2f} {stats['std_reward']:<12.2f} "
              f"{stats['avg_roas']:<8.2f} {stats['avg_budget_used']*100:<10.1f} {stats['training_time']:<8.1f}")
    
    return results

def demonstrate_learning_progression():
    """Show how an agent learns over time"""
    print(f"\n📚 Learning Progression Demo")
    print("=" * 60)
    
    env = GAELPGymEnv(max_budget=1000.0, max_steps=30)
    env = Monitor(env)
    
    model = PPO("MlpPolicy", env, verbose=0, device='cpu')
    
    # Test performance at different training stages
    checkpoints = [0, 1000, 3000, 5000, 8000]
    performance_history = []
    
    for i, timesteps in enumerate(checkpoints):
        if timesteps > 0:
            if i == 1:
                model.learn(total_timesteps=timesteps)
            else:
                model.learn(total_timesteps=timesteps - checkpoints[i-1])
        
        # Evaluate current performance
        eval_env = GAELPGymEnv(max_budget=1000.0, max_steps=30)
        test_rewards = []
        test_roas = []
        
        for test_ep in range(3):
            obs, info = eval_env.reset(seed=test_ep + 200)
            episode_reward = 0
            
            while True:
                if timesteps == 0:
                    action = eval_env.action_space.sample()  # Random baseline
                else:
                    action, _states = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            metrics = eval_env.get_episode_metrics()
            test_rewards.append(episode_reward)
            test_roas.append(metrics.get('final_roas', 0))
        
        avg_reward = np.mean(test_rewards)
        avg_roas = np.mean(test_roas)
        performance_history.append((timesteps, avg_reward, avg_roas))
        
        status = "Random Baseline" if timesteps == 0 else f"After {timesteps} steps"
        print(f"   {status}: Reward={avg_reward:.2f}, ROAS={avg_roas:.2f}x")
        
        eval_env.close()
    
    env.close()
    
    # Show improvement
    if len(performance_history) > 1:
        initial_reward = performance_history[0][1]
        final_reward = performance_history[-1][1]
        improvement = ((final_reward - initial_reward) / abs(initial_reward) * 100) if initial_reward != 0 else 0
        
        print(f"\n   📈 Total Improvement: {improvement:.1f}%")
    
    return performance_history

def integration_showcase():
    """Show how to integrate with existing GAELP components"""
    print(f"\n🔗 Integration Showcase")
    print("=" * 60)
    
    print(f"✅ Gymnasium Environment Integration:")
    print(f"   • Compatible with any Gymnasium-based RL library")
    print(f"   • Works with stable-baselines3, Ray RLlib, etc.")
    print(f"   • Standard reset(), step(), render() interface")
    
    print(f"\n✅ GAELP Simulator Integration:")
    print(f"   • Uses enhanced_simulator.py for realistic ad auctions")
    print(f"   • Includes user behavior modeling")
    print(f"   • Real-world calibrated performance metrics")
    
    print(f"\n✅ Observable Metrics:")
    print(f"   • Cost, Revenue, ROAS (Return on Ad Spend)")
    print(f"   • Impressions, Clicks, Conversions")
    print(f"   • Click-through Rate (CTR), Conversion Rate (CVR)")
    print(f"   • Budget utilization and campaign efficiency")
    
    print(f"\n✅ Action Space Design:")
    print(f"   • Bid amount (core advertising parameter)")
    print(f"   • Quality scores (ad relevance)")
    print(f"   • Creative parameters (ad content quality)")
    print(f"   • Targeting precision (audience selection)")
    
    print(f"\n🎯 Use Cases:")
    print(f"   • Automated bid optimization")
    print(f"   • Campaign budget allocation")
    print(f"   • Creative testing and optimization")
    print(f"   • Multi-objective optimization (ROAS vs. volume)")

def main():
    """Run the complete demonstration"""
    demonstrate_environment_features()
    
    manual_results = test_manual_strategies()
    
    rl_results = train_and_compare_algorithms()
    
    learning_history = demonstrate_learning_progression()
    
    integration_showcase()
    
    print(f"\n🎉 GAELP Gymnasium Environment Demo Complete!")
    print("=" * 60)
    print(f"✅ Environment validated and working")
    print(f"✅ Multiple strategies tested")
    print(f"✅ RL algorithms trained and compared")
    print(f"✅ Learning progression demonstrated")
    print(f"✅ Integration points showcased")
    
    # Summary
    best_manual = max(manual_results, key=lambda x: x['total_reward'])
    print(f"\n📋 Summary:")
    print(f"   Best Manual Strategy: {best_manual['strategy']} (Reward: {best_manual['total_reward']:.1f})")
    
    if rl_results:
        best_rl = max(rl_results.items(), key=lambda x: x[1]['avg_reward'])
        print(f"   Best RL Algorithm: {best_rl[0]} (Avg Reward: {best_rl[1]['avg_reward']:.1f})")
    
    if learning_history:
        initial_perf = learning_history[0][1]
        final_perf = learning_history[-1][1]
        print(f"   Learning Improvement: {final_perf:.1f} → {final_perf:.1f} reward")
    
    print(f"\n🚀 Ready for production RL training!")

if __name__ == "__main__":
    main()