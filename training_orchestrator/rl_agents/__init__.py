"""
Real Reinforcement Learning Agents for GAELP

This module implements production-ready RL algorithms for ad campaign optimization:
- PPO (Proximal Policy Optimization) for stable policy learning
- SAC (Soft Actor-Critic) for continuous action spaces  
- DQN (Deep Q-Network) for discrete campaign choices
- Multi-algorithm comparison and ensemble methods
"""

from .base_agent import BaseRLAgent, AgentConfig
from .ppo_agent import PPOAgent, PPOConfig
from .sac_agent import SACAgent, SACConfig
from .dqn_agent import DQNAgent, DQNConfig
from .networks import PolicyNetwork, ValueNetwork, QNetwork
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .environment_wrappers import AdCampaignEnvWrapper
from .state_processor import StateProcessor
from .action_space import ActionSpaceManager
from .reward_engineering import RewardEngineer

__version__ = "1.0.0"
__all__ = [
    "BaseRLAgent",
    "AgentConfig", 
    "PPOAgent",
    "PPOConfig",
    "SACAgent", 
    "SACConfig",
    "DQNAgent",
    "DQNConfig",
    "PolicyNetwork",
    "ValueNetwork", 
    "QNetwork",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "AdCampaignEnvWrapper",
    "StateProcessor",
    "ActionSpaceManager",
    "RewardEngineer"
]