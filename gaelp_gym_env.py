#!/usr/bin/env python3
"""
GAELP Gymnasium Environment

A Gymnasium-compatible environment for training RL agents on advertising campaigns.
This environment follows the standard Gym interface with proper observation and action spaces.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

from enhanced_simulator import EnhancedGAELPEnvironment

logger = logging.getLogger(__name__)


class GAELPGymEnv(gym.Env):
    """
    Gymnasium-compatible environment for GAELP advertising optimization.
    
    Observation Space:
    - Box space with normalized advertising metrics
    
    Action Space:
    - Box space for continuous bid, budget, and targeting parameters
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self, 
        max_budget: float = 10000.0,
        max_steps: int = 100,
        render_mode: Optional[str] = None
    ):
        """
        Initialize GAELP Gym Environment
        
        Args:
            max_budget: Maximum campaign budget
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        self.max_budget = max_budget
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Initialize the enhanced simulator
        self.simulator = EnhancedGAELPEnvironment()
        
        # Define observation space (normalized metrics)
        # [total_cost, total_revenue, impressions, clicks, conversions, avg_cpc, roas, remaining_budget, step_ratio]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0]),
            dtype=np.float32,
            shape=(9,)
        )
        
        # Define action space
        # [bid_amount, quality_score, creative_quality, price_shown, targeting_precision]
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.1, 0.1, 1.0, 0.1]),
            high=np.array([10.0, 1.0, 1.0, 200.0, 1.0]),
            dtype=np.float32,
            shape=(5,)
        )
        
        # Episode tracking
        self.current_step = 0
        self.total_spent = 0.0
        self.episode_history = []
        
        # Normalization parameters
        self.max_impressions = 100  # For normalization
        self.max_clicks = 20
        self.max_conversions = 5
        self.max_cpc = 20.0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to initial state
        
        Returns:
            observation: Initial observation
            info: Additional info dict
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset environment state
        self.current_step = 0
        self.total_spent = 0.0
        self.episode_history = []
        
        # Reset simulator
        self.simulator.reset()
        
        # Get initial observation
        observation = self._get_normalized_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Action array [bid_amount, quality_score, creative_quality, price_shown, targeting_precision]
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional info dict
        """
        
        # Validate action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Convert action to simulator format
        sim_action = {
            'bid': float(action[0]),
            'budget': self.max_budget,
            'quality_score': float(action[1]),
            'creative': {
                'quality_score': float(action[2]),
                'price_shown': float(action[3])
            },
            'targeting_precision': float(action[4])
        }
        
        # Execute step in simulator
        sim_obs, sim_reward, sim_done, sim_info = self.simulator.step(sim_action)
        
        # Update episode tracking
        self.current_step += 1
        self.total_spent += sim_info.get('cost', 0.0)
        self.episode_history.append({
            'action': sim_action,
            'reward': sim_reward,
            'metrics': sim_info
        })
        
        # Calculate Gymnasium-style reward
        reward = self._calculate_reward(sim_reward, sim_info)
        
        # Check termination conditions
        terminated = bool(sim_done or self.total_spent >= self.max_budget)
        truncated = bool(self.current_step >= self.max_steps)
        
        # Get normalized observation
        observation = self._get_normalized_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment
        """
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        
    def close(self):
        """
        Clean up resources
        """
        pass
    
    def _get_normalized_observation(self) -> np.ndarray:
        """
        Get normalized observation vector
        """
        # Get raw observation from simulator
        raw_obs = self.simulator._get_observation()
        
        # Normalize values
        normalized_obs = np.array([
            raw_obs['total_cost'] / self.max_budget,  # Normalized cost
            raw_obs['total_revenue'] / (self.max_budget * 3),  # Normalized revenue (assuming 3x max ROAS)
            raw_obs['impressions'] / self.max_impressions,  # Normalized impressions
            raw_obs['clicks'] / self.max_clicks,  # Normalized clicks
            raw_obs['conversions'] / self.max_conversions,  # Normalized conversions
            min(raw_obs['avg_cpc'] / self.max_cpc, 1.0),  # Normalized avg CPC
            min(raw_obs['roas'] / 10.0, 1.0),  # Normalized ROAS (capped at 10x)
            (self.max_budget - self.total_spent) / self.max_budget,  # Remaining budget ratio
            self.current_step / self.max_steps  # Step progress ratio
        ], dtype=np.float32)
        
        return normalized_obs
    
    def _calculate_reward(self, sim_reward: float, sim_info: Dict[str, Any]) -> float:
        """
        Calculate Gymnasium-style reward
        """
        # Base reward is the simulator's ROAS-based reward
        base_reward = sim_reward
        
        # Add efficiency bonuses
        cost = sim_info.get('cost', 0.0)
        revenue = sim_info.get('revenue', 0.0)
        conversions = sim_info.get('conversions', 0)
        
        # Reward efficient spending
        if cost > 0:
            efficiency_bonus = min(revenue / cost, 5.0) - 1.0  # Bonus for ROAS > 1
        else:
            efficiency_bonus = 0.0
        
        # Reward conversions
        conversion_bonus = conversions * 0.1
        
        # Penalty for overspending
        overspend_penalty = 0.0
        if self.total_spent > self.max_budget * 0.9:  # Warning zone
            overspend_penalty = -0.1
        
        total_reward = base_reward + efficiency_bonus + conversion_bonus + overspend_penalty
        
        return float(total_reward)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional info for debugging/monitoring
        """
        raw_obs = self.simulator._get_observation()
        
        return {
            'step': self.current_step,
            'total_spent': self.total_spent,
            'remaining_budget': self.max_budget - self.total_spent,
            'raw_metrics': raw_obs,
            'episode_length': len(self.episode_history),
            'avg_reward': np.mean([h['reward'] for h in self.episode_history]) if self.episode_history else 0.0
        }
    
    def _render_human(self):
        """
        Render in human-readable format
        """
        if not self.episode_history:
            print("Episode not started yet")
            return
            
        raw_obs = self.simulator._get_observation()
        
        print(f"\n{'='*50}")
        print(f"GAELP Campaign Status - Step {self.current_step}")
        print(f"{'='*50}")
        print(f"Budget: ${self.total_spent:.2f} / ${self.max_budget:.2f} ({(self.total_spent/self.max_budget*100):.1f}%)")
        print(f"Impressions: {raw_obs['impressions']}")
        print(f"Clicks: {raw_obs['clicks']} (CTR: {raw_obs['clicks']/max(raw_obs['impressions'], 1)*100:.2f}%)")
        print(f"Conversions: {raw_obs['conversions']} (CVR: {raw_obs['conversions']/max(raw_obs['clicks'], 1)*100:.2f}%)")
        print(f"Revenue: ${raw_obs['total_revenue']:.2f}")
        print(f"ROAS: {raw_obs['roas']:.2f}x")
        print(f"Avg CPC: ${raw_obs['avg_cpc']:.2f}")
        
        if len(self.episode_history) >= 5:
            recent_rewards = [h['reward'] for h in self.episode_history[-5:]]
            print(f"Recent Avg Reward: {np.mean(recent_rewards):.3f}")
    
    def _render_rgb_array(self) -> np.ndarray:
        """
        Render as RGB array for video recording
        """
        # Simple visualization - could be enhanced with matplotlib
        # For now, return a placeholder array
        return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive episode metrics for analysis
        """
        if not self.episode_history:
            return {}
        
        raw_obs = self.simulator._get_observation()
        
        rewards = [h['reward'] for h in self.episode_history]
        costs = [h['metrics'].get('cost', 0) for h in self.episode_history]
        revenues = [h['metrics'].get('revenue', 0) for h in self.episode_history]
        
        return {
            'episode_length': len(self.episode_history),
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards),
            'total_cost': sum(costs),
            'total_revenue': sum(revenues),
            'final_roas': raw_obs['roas'],
            'impressions': raw_obs['impressions'],
            'clicks': raw_obs['clicks'],
            'conversions': raw_obs['conversions'],
            'ctr': raw_obs['clicks'] / max(raw_obs['impressions'], 1),
            'cvr': raw_obs['conversions'] / max(raw_obs['clicks'], 1),
            'budget_utilization': self.total_spent / self.max_budget
        }


def test_gaelp_gym_env():
    """
    Test the GAELP Gymnasium environment
    """
    print("Testing GAELP Gymnasium Environment")
    print("=" * 50)
    
    # Create environment
    env = GAELPGymEnv(max_budget=1000.0, max_steps=50, render_mode="human")
    
    # Test reset
    observation, info = env.reset(seed=42)
    print(f"Initial observation shape: {observation.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    total_reward = 0
    
    for step in range(20):
        # Sample random action
        action = env.action_space.sample()
        
        # Execute step
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            env.render()
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    # Get final metrics
    final_metrics = env.get_episode_metrics()
    print(f"\nFinal Episode Metrics:")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_gaelp_gym_env()