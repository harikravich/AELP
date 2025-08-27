"""
DQN (Deep Q-Network) Agent

Implements DQN algorithm optimized for discrete ad campaign choices with
experience replay, target networks, and double DQN for stable learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import logging

from .base_agent import BaseRLAgent, AgentConfig
from .networks import QNetwork, DoubleQNetwork
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class DQNConfig(AgentConfig):
    """DQN-specific configuration"""
    
    # DQN variants
    double_dqn: bool = True
    dueling_dqn: bool = True
    noisy_networks: bool = False
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000
    
    # Training parameters
    target_update_frequency: int = 10000
    train_frequency: int = 4
    gradient_steps: int = 1
    
    # Memory
    buffer_size: int = 1000000
    prioritized_replay: bool = True
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta: float = 0.4
    prioritized_replay_beta_end: float = 1.0
    prioritized_replay_beta_steps: int = 100000
    
    # Multi-step learning
    n_step: int = 1
    
    # Categorical DQN (distributional RL)
    distributional: bool = False
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0


class DQNAgent(BaseRLAgent):
    """
    DQN agent for discrete ad campaign decisions.
    
    Implements various DQN improvements including double DQN, dueling architecture,
    prioritized experience replay, and optional distributional learning.
    """
    
    def __init__(self, config: DQNConfig, agent_id: str):
        self.dqn_config = config
        super().__init__(config, agent_id)
        
        # Initialize replay buffer
        if config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                config.buffer_size,
                alpha=config.prioritized_replay_alpha,
                beta=config.prioritized_replay_beta
            )
        else:
            self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Exploration schedule
        self.epsilon_schedule = self._create_epsilon_schedule()
        
        # For multi-step learning
        if config.n_step > 1:
            self.n_step_buffer = []
        
        # Beta schedule for prioritized replay
        if config.prioritized_replay:
            self.beta_schedule = self._create_beta_schedule()
        
        self.logger.info(f"DQN agent initialized with double_dqn={config.double_dqn}, dueling={config.dueling_dqn}")
    
    def _setup_networks(self):
        """Setup DQN networks"""
        
        # Compute discrete action space size
        self.discrete_action_space = self._compute_discrete_action_space()
        
        if self.dqn_config.double_dqn:
            # Double DQN with separate Q-networks
            self.q_network = QNetwork(
                state_dim=self.config.state_dim,
                action_dim=self.discrete_action_space,
                hidden_dims=self.config.hidden_dims,
                activation=self.config.activation,
                dueling=self.dqn_config.dueling_dqn
            ).to(self.device)
            
            self.target_q_network = QNetwork(
                state_dim=self.config.state_dim,
                action_dim=self.discrete_action_space,
                hidden_dims=self.config.hidden_dims,
                activation=self.config.activation,
                dueling=self.dqn_config.dueling_dqn
            ).to(self.device)
        else:
            # Standard DQN
            self.q_network = QNetwork(
                state_dim=self.config.state_dim,
                action_dim=self.discrete_action_space,
                hidden_dims=self.config.hidden_dims,
                activation=self.config.activation,
                dueling=self.dqn_config.dueling_dqn
            ).to(self.device)
            
            self.target_q_network = QNetwork(
                state_dim=self.config.state_dim,
                action_dim=self.discrete_action_space,
                hidden_dims=self.config.hidden_dims,
                activation=self.config.activation,
                dueling=self.dqn_config.dueling_dqn
            ).to(self.device)
        
        # Initialize target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Freeze target network
        for param in self.target_q_network.parameters():
            param.requires_grad = False
    
    def _setup_optimizers(self):
        """Setup DQN optimizer"""
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-4
        )
    
    def _compute_discrete_action_space(self) -> int:
        """Compute total discrete action space size"""
        # For ad campaigns, we'll use a combination of discrete choices
        creative_types = 3  # image, video, carousel
        audiences = 3       # young_adults, professionals, families  
        bid_strategies = 3  # cpc, cpm, cpa
        budget_levels = 5   # discretized budget levels
        
        return creative_types * audiences * bid_strategies * budget_levels
    
    def _create_epsilon_schedule(self):
        """Create epsilon decay schedule for exploration"""
        def epsilon_schedule(step):
            if step < self.dqn_config.epsilon_decay_steps:
                progress = step / self.dqn_config.epsilon_decay_steps
                return self.dqn_config.epsilon_start + progress * (
                    self.dqn_config.epsilon_end - self.dqn_config.epsilon_start
                )
            else:
                return self.dqn_config.epsilon_end
        
        return epsilon_schedule
    
    def _create_beta_schedule(self):
        """Create beta schedule for prioritized replay"""
        def beta_schedule(step):
            if step < self.dqn_config.prioritized_replay_beta_steps:
                progress = step / self.dqn_config.prioritized_replay_beta_steps
                return self.dqn_config.prioritized_replay_beta + progress * (
                    self.dqn_config.prioritized_replay_beta_end - self.dqn_config.prioritized_replay_beta
                )
            else:
                return self.dqn_config.prioritized_replay_beta_end
        
        return beta_schedule
    
    async def select_action(self, state: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """Select action using epsilon-greedy policy"""
        
        self.q_network.eval()
        
        with torch.no_grad():
            state_tensor = self.preprocess_state(state).unsqueeze(0)
            
            # Get Q-values for all actions
            if hasattr(self.q_network, 'q1_forward'):
                q_values = self.q_network.q1_forward(state_tensor, None)  # For compatibility
            else:
                # For standard Q-network, we need to evaluate all possible actions
                q_values = self._evaluate_all_actions(state_tensor)
            
            # Epsilon-greedy action selection
            epsilon = 0.0 if deterministic else self.epsilon_schedule(self.training_step)
            
            if np.random.random() < epsilon:
                # Random action
                action_idx = np.random.randint(0, self.discrete_action_space)
            else:
                # Greedy action
                action_idx = torch.argmax(q_values, dim=-1).item()
        
        self.q_network.train()
        
        # Convert action index to structured action
        structured_action = self._action_index_to_dict(action_idx)
        
        return structured_action
    
    def _evaluate_all_actions(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Evaluate Q-values for all possible discrete actions"""
        batch_size = state_tensor.shape[0]
        q_values = torch.zeros(batch_size, self.discrete_action_space, device=self.device)
        
        # This is inefficient but works for demonstration
        # In practice, you'd want a more efficient network architecture
        for action_idx in range(self.discrete_action_space):
            action_tensor = self._action_index_to_tensor(action_idx).unsqueeze(0).repeat(batch_size, 1)
            q_val = self.q_network(state_tensor, action_tensor)
            q_values[:, action_idx] = q_val.squeeze()
        
        return q_values
    
    def _action_index_to_dict(self, action_idx: int) -> Dict[str, Any]:
        """Convert action index to structured action dictionary"""
        
        # Decompose action index into components
        creative_types = ["image", "video", "carousel"]
        audiences = ["young_adults", "professionals", "families"]
        bid_strategies = ["cpc", "cpm", "cpa"]
        budget_levels = [20.0, 40.0, 60.0, 80.0, 100.0]
        
        # Decompose index
        budget_idx = action_idx % len(budget_levels)
        action_idx //= len(budget_levels)
        
        bid_idx = action_idx % len(bid_strategies)
        action_idx //= len(bid_strategies)
        
        audience_idx = action_idx % len(audiences)
        action_idx //= len(audiences)
        
        creative_idx = action_idx % len(creative_types)
        
        return {
            "creative_type": creative_types[creative_idx],
            "target_audience": audiences[audience_idx],
            "bid_strategy": bid_strategies[bid_idx],
            "budget": budget_levels[budget_idx],
            "bid_amount": 5.0,  # Fixed for DQN
            "audience_size": 0.8,  # Fixed for DQN
            "ab_test_enabled": False,  # Simplified for DQN
            "ab_test_split": 0.5,
            "action_metadata": {
                "agent_id": self.agent_id,
                "training_step": self.training_step,
                "epsilon": self.epsilon_schedule(self.training_step),
                "action_index": action_idx
            }
        }
    
    def _action_index_to_tensor(self, action_idx: int) -> torch.Tensor:
        """Convert action index to tensor format for Q-network input"""
        action_dict = self._action_index_to_dict(action_idx)
        
        # Convert to tensor format (matching SAC/PPO format)
        action_values = []
        
        # Creative type (one-hot)
        creative_types = ["image", "video", "carousel"]
        creative_one_hot = [1.0 if action_dict["creative_type"] == ct else 0.0 for ct in creative_types]
        action_values.extend(creative_one_hot)
        
        # Target audience (one-hot)
        audiences = ["young_adults", "professionals", "families"]
        audience_one_hot = [1.0 if action_dict["target_audience"] == aud else 0.0 for aud in audiences]
        action_values.extend(audience_one_hot)
        
        # Bid strategy (one-hot)
        bid_strategies = ["cpc", "cpm", "cpa"]
        bid_one_hot = [1.0 if action_dict["bid_strategy"] == bs else 0.0 for bs in bid_strategies]
        action_values.extend(bid_one_hot)
        
        # Continuous values (normalized)
        action_values.extend([
            action_dict["budget"] / 100.0,
            action_dict["bid_amount"] / 10.0,
            action_dict["audience_size"],
            float(action_dict["ab_test_enabled"]),
            action_dict["ab_test_split"]
        ])
        
        # Pad to action dimension if needed
        while len(action_values) < self.config.action_dim:
            action_values.append(0.0)
        
        return torch.tensor(action_values[:self.config.action_dim], dtype=torch.float32, device=self.device)
    
    def update_policy(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update DQN policy using experience replay"""
        
        # Add experiences to replay buffer
        for exp in experiences:
            if self.dqn_config.n_step > 1:
                self._add_to_n_step_buffer(exp)
            else:
                self._add_experience(exp)
        
        # Update only if we have enough samples
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # Train every N steps
        if self.training_step % self.dqn_config.train_frequency != 0:
            self.training_step += 1
            return {}
        
        metrics = {}
        
        # Perform gradient steps
        for _ in range(self.dqn_config.gradient_steps):
            
            # Sample batch from replay buffer
            if self.dqn_config.prioritized_replay:
                beta = self.beta_schedule(self.training_step)
                self.replay_buffer.beta = beta
                batch, indices, weights = self.replay_buffer.sample(self.config.batch_size)
                weights = torch.tensor(weights, device=self.device)
            else:
                batch = self.replay_buffer.sample(self.config.batch_size)
                weights = None
                indices = None
            
            # Update Q-network
            update_metrics = self._update_q_network(batch, weights, indices)
            metrics.update(update_metrics)
        
        # Update target network
        if self.training_step % self.dqn_config.target_update_frequency == 0:
            self._update_target_network()
        
        self.training_step += 1
        self.training_metrics.update(metrics)
        
        return metrics
    
    def _add_experience(self, exp: Dict[str, Any]):
        """Add single experience to replay buffer"""
        state = self.preprocess_state(exp['state'])
        action_idx = self._action_dict_to_index(exp['action'])
        reward = float(exp['reward'])
        next_state = self.preprocess_state(exp['next_state'])
        done = bool(exp['done'])
        
        self.replay_buffer.add(state, action_idx, reward, next_state, done)
    
    def _add_to_n_step_buffer(self, exp: Dict[str, Any]):
        """Add experience to n-step buffer"""
        self.n_step_buffer.append(exp)
        
        if len(self.n_step_buffer) >= self.dqn_config.n_step:
            # Compute n-step return
            n_step_exp = self._compute_n_step_return(self.n_step_buffer)
            self._add_experience(n_step_exp)
            
            # Remove oldest experience
            self.n_step_buffer.pop(0)
    
    def _compute_n_step_return(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute n-step return from experience sequence"""
        n_step_return = 0.0
        gamma_power = 1.0
        
        for i, exp in enumerate(experiences):
            n_step_return += gamma_power * exp['reward']
            gamma_power *= self.config.gamma
            
            if exp['done']:
                break
        
        # Use first state and action, last next_state, n-step return
        return {
            'state': experiences[0]['state'],
            'action': experiences[0]['action'],
            'reward': n_step_return,
            'next_state': experiences[-1]['next_state'],
            'done': experiences[-1]['done']
        }
    
    def _action_dict_to_index(self, action_dict: Dict[str, Any]) -> int:
        """Convert action dictionary to action index"""
        creative_types = ["image", "video", "carousel"]
        audiences = ["young_adults", "professionals", "families"]
        bid_strategies = ["cpc", "cpm", "cpa"]
        budget_levels = [20.0, 40.0, 60.0, 80.0, 100.0]
        
        creative_idx = creative_types.index(action_dict["creative_type"])
        audience_idx = audiences.index(action_dict["target_audience"])
        bid_idx = bid_strategies.index(action_dict["bid_strategy"])
        
        # Find closest budget level
        budget = action_dict["budget"]
        budget_idx = min(range(len(budget_levels)), 
                        key=lambda i: abs(budget_levels[i] - budget))
        
        # Compose index
        action_idx = (creative_idx * len(audiences) * len(bid_strategies) * len(budget_levels) +
                     audience_idx * len(bid_strategies) * len(budget_levels) +
                     bid_idx * len(budget_levels) +
                     budget_idx)
        
        return action_idx
    
    def _update_q_network(self, batch: Dict[str, torch.Tensor], 
                         weights: Optional[torch.Tensor] = None,
                         indices: Optional[List[int]] = None) -> Dict[str, float]:
        """Update Q-network using DQN loss"""
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device).long()
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Current Q-values
        current_q_values = self._evaluate_all_actions(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            if self.dqn_config.double_dqn:
                # Double DQN: use online network for action selection
                next_q_values_online = self._evaluate_all_actions(next_states)
                next_actions = torch.argmax(next_q_values_online, dim=1)
                
                # Use target network for Q-value estimation
                next_q_values_target = self._evaluate_all_actions_target(next_states)
                next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1))
            else:
                # Standard DQN
                next_q_values_target = self._evaluate_all_actions_target(next_states)
                next_q_values = torch.max(next_q_values_target, dim=1)[0].unsqueeze(1)
            
            # Compute target
            target_q_values = rewards.unsqueeze(1) + self.config.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        
        if weights is not None:
            # Weighted loss for prioritized replay
            loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.grad_clip_norm
        )
        
        self.optimizer.step()
        
        # Update priorities for prioritized replay
        if self.dqn_config.prioritized_replay and indices is not None:
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, priorities)
        
        return {
            'q_loss': loss.item(),
            'q_values_mean': current_q_values.mean().item(),
            'target_q_mean': target_q_values.mean().item(),
            'td_error_mean': td_errors.abs().mean().item(),
            'grad_norm': grad_norm.item(),
            'epsilon': self.epsilon_schedule(self.training_step)
        }
    
    def _evaluate_all_actions_target(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Evaluate Q-values using target network"""
        batch_size = state_tensor.shape[0]
        q_values = torch.zeros(batch_size, self.discrete_action_space, device=self.device)
        
        for action_idx in range(self.discrete_action_space):
            action_tensor = self._action_index_to_tensor(action_idx).unsqueeze(0).repeat(batch_size, 1)
            q_val = self.target_q_network(state_tensor, action_tensor)
            q_values[:, action_idx] = q_val.squeeze()
        
        return q_values
    
    def _update_target_network(self):
        """Hard update of target network"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing"""
        state = {
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'replay_buffer': self.replay_buffer.get_state()
        }
        
        if self.dqn_config.n_step > 1:
            state['n_step_buffer'] = self.n_step_buffer
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load agent state from checkpoint"""
        self.q_network.load_state_dict(state['q_network'])
        self.target_q_network.load_state_dict(state['target_q_network'])
        self.optimizer.load_state_dict(state['optimizer'])
        
        self.training_step = state.get('training_step', 0)
        self.episode_count = state.get('episode_count', 0)
        
        if 'replay_buffer' in state:
            self.replay_buffer.load_state(state['replay_buffer'])
        
        if 'n_step_buffer' in state:
            self.n_step_buffer = state['n_step_buffer']