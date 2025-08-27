"""
Agent Factory for GAELP

Creates and configures RL agents based on algorithm type and environment requirements.
Handles agent lifecycle management and provides unified interface for training orchestrator.
"""

import torch
import logging
from typing import Dict, Any, Optional, Union, Type
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import BaseRLAgent, AgentConfig
from .ppo_agent import PPOAgent, PPOConfig
from .sac_agent import SACAgent, SACConfig
from .dqn_agent import DQNAgent, DQNConfig
from .state_processor import StateProcessor, StateProcessorConfig
from .reward_engineering import RewardEngineer, RewardConfig

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Supported RL agent types"""
    PPO = "ppo"
    SAC = "sac"
    DQN = "dqn"
    ENSEMBLE = "ensemble"


@dataclass
class AgentFactoryConfig:
    """Configuration for agent factory"""
    
    # Agent selection
    agent_type: AgentType = AgentType.PPO
    agent_id: str = "gaelp_agent"
    
    # Environment specifications
    state_dim: int = 128
    action_dim: int = 64
    
    # Hardware configuration
    device: str = "auto"  # "auto", "cpu", "cuda"
    mixed_precision: bool = False
    
    # State processing
    enable_state_processing: bool = True
    state_processor_config: StateProcessorConfig = field(default_factory=StateProcessorConfig)
    
    # Reward engineering
    enable_reward_engineering: bool = True
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    
    # Algorithm-specific configurations
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
    sac_config: SACConfig = field(default_factory=SACConfig)
    dqn_config: DQNConfig = field(default_factory=DQNConfig)
    
    # Ensemble configuration (if using ensemble)
    ensemble_agents: list = field(default_factory=lambda: [AgentType.PPO, AgentType.SAC])
    ensemble_weights: list = field(default_factory=lambda: [0.6, 0.4])
    
    # Training configuration
    enable_curriculum_learning: bool = True
    enable_transfer_learning: bool = False
    pretrained_model_path: Optional[str] = None
    
    # Monitoring and checkpointing
    checkpoint_dir: str = "./checkpoints"
    log_training_metrics: bool = True
    wandb_project: Optional[str] = None


class AgentFactory:
    """
    Factory class for creating and managing RL agents in GAELP.
    
    Provides a unified interface for creating different types of RL agents
    with appropriate configurations for ad campaign optimization.
    """
    
    def __init__(self, config: AgentFactoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        
        # Initialize state processor
        if config.enable_state_processing:
            self.state_processor = StateProcessor(config.state_processor_config)
        else:
            self.state_processor = None
        
        # Initialize reward engineer
        if config.enable_reward_engineering:
            self.reward_engineer = RewardEngineer(config.reward_config)
        else:
            self.reward_engineer = None
        
        self.logger.info(f"Agent factory initialized for {config.agent_type.value} on {self.device}")
    
    def create_agent(self) -> Union[BaseRLAgent, 'EnsembleAgent']:
        """Create RL agent based on configuration"""
        
        if self.config.agent_type == AgentType.ENSEMBLE:
            return self._create_ensemble_agent()
        else:
            return self._create_single_agent(self.config.agent_type)
    
    def _create_single_agent(self, agent_type: AgentType) -> BaseRLAgent:
        """Create single RL agent"""
        
        # Update agent configurations with factory settings
        if agent_type == AgentType.PPO:
            config = self.config.ppo_config
            config.state_dim = self.config.state_dim
            config.action_dim = self.config.action_dim
            config.device = self.device
            agent = PPOAgent(config, self.config.agent_id)
            
        elif agent_type == AgentType.SAC:
            config = self.config.sac_config
            config.state_dim = self.config.state_dim
            config.action_dim = self.config.action_dim
            config.device = self.device
            agent = SACAgent(config, self.config.agent_id)
            
        elif agent_type == AgentType.DQN:
            config = self.config.dqn_config
            config.state_dim = self.config.state_dim
            config.action_dim = self.config.action_dim
            config.device = self.device
            agent = DQNAgent(config, self.config.agent_id)
            
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        
        # Load pretrained model if specified
        if self.config.pretrained_model_path:
            try:
                agent.load_checkpoint(self.config.pretrained_model_path)
                self.logger.info(f"Loaded pretrained model from {self.config.pretrained_model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load pretrained model: {e}")
        
        return agent
    
    def _create_ensemble_agent(self) -> 'EnsembleAgent':
        """Create ensemble of multiple agents"""
        
        agents = []
        for agent_type in self.config.ensemble_agents:
            agent = self._create_single_agent(agent_type)
            agents.append(agent)
        
        ensemble = EnsembleAgent(
            agents=agents,
            weights=self.config.ensemble_weights,
            agent_id=self.config.agent_id
        )
        
        return ensemble
    
    def get_state_processor(self) -> Optional[StateProcessor]:
        """Get state processor instance"""
        return self.state_processor
    
    def get_reward_engineer(self) -> Optional[RewardEngineer]:
        """Get reward engineer instance"""
        return self.reward_engineer
    
    @staticmethod
    def get_recommended_config(environment_type: str) -> AgentFactoryConfig:
        """Get recommended configuration for environment type"""
        
        base_config = AgentFactoryConfig()
        
        if environment_type == "simulation":
            # Fast training for simulation
            base_config.agent_type = AgentType.PPO
            base_config.ppo_config.batch_size = 128
            base_config.ppo_config.rollout_length = 1024
            base_config.ppo_config.ppo_epochs = 3
            
        elif environment_type == "historical":
            # Sample efficiency for limited historical data
            base_config.agent_type = AgentType.SAC
            base_config.sac_config.batch_size = 64
            base_config.sac_config.warm_up_steps = 1000
            
        elif environment_type == "real":
            # Conservative and safe for real campaigns
            base_config.agent_type = AgentType.ENSEMBLE
            base_config.ensemble_agents = [AgentType.PPO, AgentType.SAC]
            base_config.ensemble_weights = [0.7, 0.3]  # More weight on stable PPO
            
            # Enhanced safety for real campaigns
            base_config.reward_config.brand_safety_weight = 1.5
            base_config.reward_config.safety_violation_penalty = 10.0
            
        elif environment_type == "scaled":
            # High performance for scaled deployment
            base_config.agent_type = AgentType.SAC
            base_config.sac_config.batch_size = 256
            base_config.sac_config.gradient_steps_per_update = 2
            base_config.enable_state_processing = True
            
        return base_config


class EnsembleAgent:
    """
    Ensemble agent that combines multiple RL algorithms.
    
    Uses weighted voting for action selection and distributes
    training across multiple agents.
    """
    
    def __init__(self, agents: list, weights: list, agent_id: str):
        self.agents = agents
        self.weights = weights
        self.agent_id = agent_id
        
        assert len(agents) == len(weights), "Number of agents must match number of weights"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
        
        self.training_step = 0
        self.episode_count = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Ensemble agent created with {len(agents)} agents")
    
    async def select_action(self, state: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """Select action using weighted ensemble voting"""
        
        # Get actions from all agents
        agent_actions = []
        for agent in self.agents:
            action = await agent.select_action(state, deterministic)
            agent_actions.append(action)
        
        # Combine actions using weighted voting
        combined_action = self._combine_actions(agent_actions)
        
        # Add ensemble metadata
        combined_action["action_metadata"]["ensemble_info"] = {
            "num_agents": len(self.agents),
            "weights": self.weights,
            "agent_types": [type(agent).__name__ for agent in self.agents]
        }
        
        return combined_action
    
    def _combine_actions(self, actions: list) -> Dict[str, Any]:
        """Combine actions from multiple agents using weighted voting"""
        
        # For discrete choices, use weighted voting
        creative_votes = {}
        audience_votes = {}
        bid_strategy_votes = {}
        
        for action, weight in zip(actions, self.weights):
            # Creative type voting
            creative = action["creative_type"]
            creative_votes[creative] = creative_votes.get(creative, 0) + weight
            
            # Audience voting
            audience = action["target_audience"]
            audience_votes[audience] = audience_votes.get(audience, 0) + weight
            
            # Bid strategy voting
            bid_strategy = action["bid_strategy"]
            bid_strategy_votes[bid_strategy] = bid_strategy_votes.get(bid_strategy, 0) + weight
        
        # Select winning choices
        creative_type = max(creative_votes, key=creative_votes.get)
        target_audience = max(audience_votes, key=audience_votes.get)
        bid_strategy = max(bid_strategy_votes, key=bid_strategy_votes.get)
        
        # For continuous values, use weighted average
        budget = sum(action["budget"] * weight for action, weight in zip(actions, self.weights))
        bid_amount = sum(action["bid_amount"] * weight for action, weight in zip(actions, self.weights))
        audience_size = sum(action["audience_size"] * weight for action, weight in zip(actions, self.weights))
        
        # A/B test decisions
        ab_test_enabled = sum(float(action["ab_test_enabled"]) * weight 
                             for action, weight in zip(actions, self.weights)) > 0.5
        ab_test_split = sum(action["ab_test_split"] * weight for action, weight in zip(actions, self.weights))
        
        return {
            "creative_type": creative_type,
            "target_audience": target_audience,
            "bid_strategy": bid_strategy,
            "budget": budget,
            "bid_amount": bid_amount,
            "audience_size": audience_size,
            "ab_test_enabled": ab_test_enabled,
            "ab_test_split": ab_test_split,
            "action_metadata": {
                "agent_id": self.agent_id,
                "training_step": self.training_step,
                "ensemble_method": "weighted_voting"
            }
        }
    
    def update_policy(self, experiences: list) -> Dict[str, float]:
        """Update all agents in ensemble"""
        
        combined_metrics = {}
        
        for i, agent in enumerate(self.agents):
            agent_metrics = agent.update_policy(experiences)
            
            # Add agent-specific prefix to metrics
            agent_name = type(agent).__name__
            for key, value in agent_metrics.items():
                combined_metrics[f"{agent_name}_{key}"] = value
        
        self.training_step += 1
        
        return combined_metrics
    
    def get_state(self) -> Dict[str, Any]:
        """Get ensemble state for checkpointing"""
        return {
            "agent_states": [agent.get_state() for agent in self.agents],
            "weights": self.weights,
            "agent_id": self.agent_id,
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "agent_types": [type(agent).__name__ for agent in self.agents]
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load ensemble state from checkpoint"""
        agent_states = state["agent_states"]
        
        for agent, agent_state in zip(self.agents, agent_states):
            agent.load_state(agent_state)
        
        self.weights = state["weights"]
        self.training_step = state.get("training_step", 0)
        self.episode_count = state.get("episode_count", 0)
    
    def save_checkpoint(self, filepath: str):
        """Save ensemble checkpoint"""
        checkpoint = {
            "ensemble_state": self.get_state(),
            "timestamp": torch.datetime.now().isoformat()
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load ensemble checkpoint"""
        checkpoint = torch.load(filepath)
        self.load_state(checkpoint["ensemble_state"])
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get combined training metrics from all agents"""
        combined_metrics = {
            "ensemble_training_step": self.training_step,
            "ensemble_episode_count": self.episode_count
        }
        
        for i, agent in enumerate(self.agents):
            agent_metrics = agent.get_training_metrics()
            agent_name = type(agent).__name__
            
            for key, value in agent_metrics.items():
                combined_metrics[f"{agent_name}_{key}"] = value
        
        return combined_metrics