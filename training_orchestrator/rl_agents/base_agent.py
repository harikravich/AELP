"""
Base RL Agent Interface

Defines the common interface that all RL agents must implement for GAELP integration.
"""

import abc
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Base configuration for RL agents"""
    
    # Model architecture
    state_dim: int = 128
    action_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # soft update rate
    
    # Experience replay
    buffer_size: int = 1000000
    prioritized_replay: bool = False
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta: float = 0.4
    
    # Training schedule
    train_every: int = 4
    target_update_interval: int = 10000
    gradient_steps: int = 1
    
    # Exploration
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    
    # Device and optimization
    device: str = "auto"  # "auto", "cpu", "cuda"
    optimizer: str = "adam"
    grad_clip_norm: Optional[float] = 10.0
    
    # Logging and checkpointing
    log_interval: int = 1000
    checkpoint_interval: int = 10000
    eval_frequency: int = 5000
    
    # Safety and constraints
    max_action_value: float = 1.0
    min_action_value: float = -1.0
    action_noise_std: float = 0.1
    
    def __post_init__(self):
        """Validate and setup configuration"""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate dimensions
        assert self.state_dim > 0, "State dimension must be positive"
        assert self.action_dim > 0, "Action dimension must be positive"
        assert len(self.hidden_dims) > 0, "Must specify at least one hidden layer"
        
        # Validate hyperparameters
        assert 0 < self.learning_rate < 1, "Learning rate must be in (0, 1)"
        assert 0 < self.gamma <= 1, "Gamma must be in (0, 1]"
        assert 0 <= self.tau <= 1, "Tau must be in [0, 1]"


class BaseRLAgent(abc.ABC):
    """
    Abstract base class for all RL agents in GAELP.
    
    Provides common functionality and defines the interface that training
    orchestrator expects from RL agents.
    """
    
    def __init__(self, config: AgentConfig, agent_id: str):
        self.config = config
        self.agent_id = agent_id
        self.device = torch.device(config.device)
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.total_timesteps = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = {}
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Setup networks and optimizers (implemented by subclasses)
        self._setup_networks()
        self._setup_optimizers()
        
        self.logger.info(f"Initialized {self.__class__.__name__} agent {agent_id}")
    
    @abc.abstractmethod
    def _setup_networks(self):
        """Setup neural networks (policy, value, etc.)"""
        pass
    
    @abc.abstractmethod
    def _setup_optimizers(self):
        """Setup optimizers for neural networks"""
        pass
    
    @abc.abstractmethod
    async def select_action(self, state: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """
        Select action given current state.
        
        Args:
            state: Current environment state
            deterministic: If True, select best action without exploration
            
        Returns:
            Dict containing action components for ad campaign
        """
        pass
    
    @abc.abstractmethod
    def update_policy(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Update agent policy based on experience batch.
        
        Args:
            experiences: List of experience tuples (state, action, reward, next_state, done)
            
        Returns:
            Dict of training metrics (loss values, etc.)
        """
        pass
    
    @abc.abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing"""
        pass
    
    @abc.abstractmethod
    def load_state(self, state: Dict[str, Any]):
        """Load agent state from checkpoint"""
        pass
    
    def preprocess_state(self, raw_state: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess raw environment state into tensor format.
        
        Args:
            raw_state: Raw state dict from environment
            
        Returns:
            Preprocessed state tensor
        """
        # Extract and normalize state features
        features = []
        
        # Market context features
        market_context = raw_state.get("market_context", {})
        features.extend([
            market_context.get("competition_level", 0.5),
            market_context.get("seasonality_factor", 1.0),
            market_context.get("trend_momentum", 0.0),
            market_context.get("market_volatility", 0.1)
        ])
        
        # Historical performance features  
        performance_history = raw_state.get("performance_history", {})
        features.extend([
            performance_history.get("avg_roas", 1.0),
            performance_history.get("avg_ctr", 0.02),
            performance_history.get("avg_conversion_rate", 0.05),
            performance_history.get("total_spend", 0.0) / 10000.0,  # normalize
            performance_history.get("total_revenue", 0.0) / 50000.0  # normalize
        ])
        
        # Budget constraints
        budget_info = raw_state.get("budget_constraints", {})
        features.extend([
            budget_info.get("daily_budget", 100.0) / 1000.0,  # normalize
            budget_info.get("remaining_budget", 100.0) / 1000.0,  # normalize
            budget_info.get("budget_utilization", 0.0)
        ])
        
        # Persona/audience features
        persona_data = raw_state.get("persona", {})
        demographics = persona_data.get("demographics", {})
        interests = persona_data.get("interests", [])
        
        # Age group encoding (one-hot)
        age_groups = ["18-25", "25-35", "35-45", "45-55", "55-65", "65+"]
        age_encoding = [1.0 if demographics.get("age_group") == age else 0.0 for age in age_groups]
        features.extend(age_encoding)
        
        # Income level encoding
        income_levels = ["low", "medium", "high"] 
        income_encoding = [1.0 if demographics.get("income") == inc else 0.0 for inc in income_levels]
        features.extend(income_encoding)
        
        # Interest categories encoding
        interest_categories = ["technology", "entertainment", "health", "finance", "travel", 
                             "food", "sports", "fashion", "education", "home"]
        interest_encoding = [1.0 if cat in interests else 0.0 for cat in interest_categories]
        features.extend(interest_encoding)
        
        # Time-based features
        time_info = raw_state.get("time_context", {})
        features.extend([
            time_info.get("hour_of_day", 12) / 24.0,  # normalize to [0,1]
            time_info.get("day_of_week", 3) / 7.0,    # normalize to [0,1]
            time_info.get("day_of_month", 15) / 31.0,  # normalize to [0,1]
            time_info.get("month", 6) / 12.0           # normalize to [0,1]
        ])
        
        # Previous action features (for recurrent behavior)
        prev_action = raw_state.get("previous_action", {})
        features.extend([
            float(prev_action.get("creative_type", "image") == "video"),
            float(prev_action.get("creative_type", "image") == "carousel"), 
            prev_action.get("budget", 50.0) / 100.0,  # normalize
            float(prev_action.get("bid_strategy", "cpc") == "cpm"),
            float(prev_action.get("bid_strategy", "cpc") == "cpa")
        ])
        
        # Pad or truncate to exact state dimension
        while len(features) < self.config.state_dim:
            features.append(0.0)
        features = features[:self.config.state_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def postprocess_action(self, action_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Convert model output tensor to structured action dict.
        
        Args:
            action_tensor: Raw action tensor from policy network
            
        Returns:
            Structured action dict for ad campaign
        """
        action_values = action_tensor.cpu().numpy()
        
        # Map continuous outputs to discrete choices and continuous parameters
        creative_type_logits = action_values[:3]  # image, video, carousel
        creative_type = ["image", "video", "carousel"][np.argmax(creative_type_logits)]
        
        audience_logits = action_values[3:6]  # young_adults, professionals, families  
        target_audience = ["young_adults", "professionals", "families"][np.argmax(audience_logits)]
        
        bid_strategy_logits = action_values[6:9]  # cpc, cpm, cpa
        bid_strategy = ["cpc", "cpm", "cpa"][np.argmax(bid_strategy_logits)]
        
        # Continuous parameters (normalized outputs)
        budget = np.clip(action_values[9] * 100.0, 10.0, 200.0)  # $10-200 daily budget
        bid_amount = np.clip(action_values[10] * 10.0, 0.5, 20.0)  # $0.5-20 bid
        audience_size = np.clip(action_values[11], 0.1, 1.0)  # 10%-100% of target audience
        
        # A/B test configuration
        ab_test_enabled = action_values[12] > 0.0
        ab_test_split = np.clip(action_values[13], 0.1, 0.9) if ab_test_enabled else 0.5
        
        return {
            "creative_type": creative_type,
            "target_audience": target_audience,
            "bid_strategy": bid_strategy,
            "budget": float(budget),
            "bid_amount": float(bid_amount),
            "audience_size": float(audience_size),
            "ab_test_enabled": bool(ab_test_enabled),
            "ab_test_split": float(ab_test_split),
            "action_metadata": {
                "agent_id": self.agent_id,
                "training_step": self.training_step,
                "exploration_rate": self.get_exploration_rate(),
                "confidence": float(np.max(self._softmax(creative_type_logits)))
            }
        }
    
    def _softmax(self, x):
        """Compute softmax values for array x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def get_exploration_rate(self) -> float:
        """Get current exploration rate (epsilon for epsilon-greedy)"""
        progress = min(1.0, self.training_step / (self.config.exploration_fraction * 1000000))
        return self.config.exploration_final_eps + (
            self.config.exploration_initial_eps - self.config.exploration_final_eps
        ) * (1 - progress)
    
    def record_episode(self, total_reward: float, episode_length: int):
        """Record episode statistics"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1
        
        # Keep only recent episodes for memory efficiency
        max_episodes = 10000
        if len(self.episode_rewards) > max_episodes:
            self.episode_rewards = self.episode_rewards[-max_episodes:]
            self.episode_lengths = self.episode_lengths[-max_episodes:]
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get current training metrics"""
        metrics = {
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "total_timesteps": self.total_timesteps,
            "exploration_rate": self.get_exploration_rate()
        }
        
        if self.episode_rewards:
            metrics.update({
                "mean_episode_reward": np.mean(self.episode_rewards[-100:]),
                "std_episode_reward": np.std(self.episode_rewards[-100:]),
                "max_episode_reward": np.max(self.episode_rewards),
                "min_episode_reward": np.min(self.episode_rewards),
                "mean_episode_length": np.mean(self.episode_lengths[-100:])
            })
        
        # Add algorithm-specific metrics
        metrics.update(self.training_metrics)
        
        return metrics
    
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint to file"""
        checkpoint = {
            "agent_class": self.__class__.__name__,
            "agent_id": self.agent_id,
            "config": self.config.__dict__,
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "total_timesteps": self.total_timesteps,
            "episode_rewards": self.episode_rewards[-1000:],  # last 1000 episodes
            "episode_lengths": self.episode_lengths[-1000:],
            "agent_state": self.get_state(),
            "timestamp": datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]
        self.total_timesteps = checkpoint["total_timesteps"]
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.episode_lengths = checkpoint.get("episode_lengths", [])
        
        self.load_state(checkpoint["agent_state"])
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def set_training_mode(self, training: bool = True):
        """Set training mode for all networks"""
        for name, module in self.__dict__.items():
            if isinstance(module, torch.nn.Module):
                module.train(training)
    
    def to_device(self, device: str):
        """Move all networks to specified device"""
        self.device = torch.device(device)
        for name, module in self.__dict__.items():
            if isinstance(module, torch.nn.Module):
                module.to(self.device)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model architecture and parameters"""
        summary = {
            "agent_class": self.__class__.__name__,
            "agent_id": self.agent_id,
            "device": str(self.device),
            "total_parameters": 0,
            "trainable_parameters": 0,
            "networks": {}
        }
        
        for name, module in self.__dict__.items():
            if isinstance(module, torch.nn.Module):
                total_params = sum(p.numel() for p in module.parameters())
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                summary["networks"][name] = {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params
                }
                
                summary["total_parameters"] += total_params
                summary["trainable_parameters"] += trainable_params
        
        return summary