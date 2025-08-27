#!/usr/bin/env python3
"""
RL Training Demo with GAELP Gymnasium Environment

This demonstrates training RL agents on the GAELP advertising environment
using stable-baselines3 algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from typing import Dict, Any
import os

from gaelp_gym_env import GAELPGymEnv


class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress
    """
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.training_rewards = []
        self.episode_metrics = []
        
    def _on_step(self) -> bool:
        # Log every N steps
        if self.n_calls % self.check_freq == 0:
            # Get training environment
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
                if hasattr(env, 'get_episode_metrics'):
                    metrics = env.get_episode_metrics()
                    if metrics:
                        self.episode_metrics.append(metrics)
                        
                        if self.verbose > 0:
                            print(f"Step {self.n_calls}: ROAS: {metrics.get('final_roas', 0):.2f}x, "
                                  f"Budget Used: {metrics.get('budget_utilization', 0)*100:.1f}%")
        
        return True


def test_environment_compatibility():
    """Test that our environment works with standard RL libraries"""
    print("Testing Environment Compatibility")
    print("=" * 50)
    
    # Test basic functionality
    env = GAELPGymEnv(max_budget=1000.0, max_steps=30)
    
    # Check spaces
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset and step
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Test random actions
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i == 0:
            print(f"First step - Reward: {reward:.3f}, Info keys: {list(info.keys())}")
        
        if terminated or truncated:
            break
    
    final_metrics = env.get_episode_metrics()
    print(f"Episode completed - Total reward: {total_reward:.2f}")
    print(f"Final ROAS: {final_metrics.get('final_roas', 0):.2f}x")
    env.close()
    print("Environment compatibility test passed!\n")


def train_ppo_agent(total_timesteps: int = 10000):
    """Train a PPO agent on GAELP environment"""
    print("Training PPO Agent")
    print("=" * 50)
    
    # Create environment
    env = GAELPGymEnv(max_budget=1000.0, max_steps=50)
    env = Monitor(env)  # Wrap with Monitor for logging
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.01,
        device='cpu'
    )
    
    # Create callback
    callback = TrainingCallback(check_freq=1000, verbose=1)
    
    # Train the model
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save model
    model.save("gaelp_ppo_agent")
    print("Model saved as 'gaelp_ppo_agent'")
    
    return model, callback


def evaluate_trained_agent(model, n_episodes: int = 5):
    """Evaluate the trained agent"""
    print("Evaluating Trained Agent")
    print("=" * 50)
    
    env = GAELPGymEnv(max_budget=1000.0, max_steps=50)
    
    episode_rewards = []
    episode_metrics = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        metrics = env.get_episode_metrics()
        episode_rewards.append(episode_reward)
        episode_metrics.append(metrics)
        
        print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}, "
              f"ROAS: {metrics.get('final_roas', 0):.2f}x, "
              f"Budget Used: {metrics.get('budget_utilization', 0)*100:.1f}%")
    
    env.close()
    
    # Summary statistics
    avg_reward = np.mean(episode_rewards)
    avg_roas = np.mean([m.get('final_roas', 0) for m in episode_metrics])
    avg_budget_used = np.mean([m.get('budget_utilization', 0) for m in episode_metrics])
    
    print(f"\nEvaluation Summary:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average ROAS: {avg_roas:.2f}x")
    print(f"  Average Budget Used: {avg_budget_used*100:.1f}%")
    
    return episode_rewards, episode_metrics


def compare_algorithms(timesteps: int = 5000):
    """Compare different RL algorithms on GAELP environment"""
    print("Comparing RL Algorithms")
    print("=" * 50)
    
    # Test different algorithms
    algorithms = {
        'PPO': lambda env: PPO("MlpPolicy", env, verbose=0, device='cpu'),
        'A2C': lambda env: A2C("MlpPolicy", env, verbose=0, device='cpu'),
    }
    
    results = {}
    
    for alg_name, alg_factory in algorithms.items():
        print(f"Training {alg_name}...")
        
        env = GAELPGymEnv(max_budget=1000.0, max_steps=30)
        env = Monitor(env)
        
        model = alg_factory(env)
        model.learn(total_timesteps=timesteps)
        
        # Evaluate
        eval_env = GAELPGymEnv(max_budget=1000.0, max_steps=30)
        episode_rewards = []
        
        for _ in range(3):  # 3 evaluation episodes
            obs, info = eval_env.reset()
            episode_reward = 0
            
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
        
        avg_reward = np.mean(episode_rewards)
        results[alg_name] = avg_reward
        print(f"{alg_name} average reward: {avg_reward:.2f}")
        
        eval_env.close()
        env.close()
    
    # Print comparison
    print(f"\nAlgorithm Comparison:")
    for alg, reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {alg}: {reward:.2f}")
    
    return results


def demonstrate_learning_curve():
    """Demonstrate learning progression over time"""
    print("Demonstrating Learning Curve")
    print("=" * 50)
    
    env = GAELPGymEnv(max_budget=1000.0, max_steps=50)
    env = Monitor(env)
    
    model = PPO("MlpPolicy", env, verbose=0, device='cpu')
    
    # Track performance at different training stages
    training_stages = [1000, 3000, 5000, 8000, 10000]
    performance_over_time = []
    
    for i, timesteps in enumerate(training_stages):
        if i == 0:
            model.learn(total_timesteps=timesteps)
        else:
            model.learn(total_timesteps=timesteps - training_stages[i-1])
        
        # Evaluate current performance
        eval_env = GAELPGymEnv(max_budget=1000.0, max_steps=50)
        rewards = []
        roas_values = []
        
        for _ in range(3):
            obs, info = eval_env.reset()
            episode_reward = 0
            
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            metrics = eval_env.get_episode_metrics()
            rewards.append(episode_reward)
            roas_values.append(metrics.get('final_roas', 0))
        
        avg_reward = np.mean(rewards)
        avg_roas = np.mean(roas_values)
        performance_over_time.append((timesteps, avg_reward, avg_roas))
        
        print(f"After {timesteps} steps: Reward={avg_reward:.2f}, ROAS={avg_roas:.2f}x")
        eval_env.close()
    
    env.close()
    return performance_over_time


if __name__ == "__main__":
    print("GAELP RL Training Demo")
    print("=" * 60)
    
    # Test 1: Environment compatibility
    test_environment_compatibility()
    
    # Test 2: Train PPO agent
    model, callback = train_ppo_agent(total_timesteps=8000)
    
    # Test 3: Evaluate trained agent
    rewards, metrics = evaluate_trained_agent(model, n_episodes=5)
    
    # Test 4: Compare algorithms
    algorithm_results = compare_algorithms(timesteps=5000)
    
    # Test 5: Learning curve
    learning_progress = demonstrate_learning_curve()
    
    print("\n" + "=" * 60)
    print("GAELP RL Training Demo Completed Successfully!")
    print("=" * 60)
    
    # Summary
    final_avg_reward = np.mean(rewards)
    final_avg_roas = np.mean([m.get('final_roas', 0) for m in metrics])
    
    print(f"Final PPO Performance:")
    print(f"  Average Reward: {final_avg_reward:.2f}")
    print(f"  Average ROAS: {final_avg_roas:.2f}x")
    print(f"  Best Algorithm: {max(algorithm_results.items(), key=lambda x: x[1])[0]}")
    
    if learning_progress:
        initial_perf = learning_progress[0][1]
        final_perf = learning_progress[-1][1]
        improvement = ((final_perf - initial_perf) / abs(initial_perf) * 100) if initial_perf != 0 else 0
        print(f"  Learning Improvement: {improvement:.1f}%")