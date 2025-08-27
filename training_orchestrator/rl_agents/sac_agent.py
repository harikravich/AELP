"""
SAC (Soft Actor-Critic) Agent

Implements SAC algorithm optimized for continuous action spaces in ad campaign optimization.
Features entropy regularization and automatic temperature tuning for robust exploration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import logging

from .base_agent import BaseRLAgent, AgentConfig
from .networks import PolicyNetwork, DoubleQNetwork
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class SACConfig(AgentConfig):
    """SAC-specific configuration"""
    
    # SAC hyperparameters
    target_entropy: Optional[float] = None  # Auto-computed if None
    initial_temperature: float = 1.0
    automatic_entropy_tuning: bool = True
    
    # Q-network parameters
    double_q: bool = True
    dueling_q: bool = False
    
    # Soft update parameters
    tau: float = 0.005
    target_update_interval: int = 1
    
    # Training parameters
    warm_up_steps: int = 10000
    gradient_steps_per_update: int = 1
    
    # Policy network parameters
    log_std_min: float = -20
    log_std_max: float = 2
    
    # Replay buffer
    buffer_size: int = 1000000
    prioritized_replay: bool = True


class SACAgent(BaseRLAgent):
    """
    SAC agent for ad campaign optimization.
    
    Uses soft Q-learning with entropy regularization for sample-efficient
    learning in continuous action spaces.
    """
    
    def __init__(self, config: SACConfig, agent_id: str):
        self.sac_config = config
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
        
        # Auto-compute target entropy if not specified
        if config.target_entropy is None:
            # Target entropy = -|A| (heuristic for continuous actions)
            self.target_entropy = -config.action_dim
        else:
            self.target_entropy = config.target_entropy
        
        # Temperature parameter (log scale for numerical stability)
        if config.automatic_entropy_tuning:
            self.log_temperature = torch.tensor(
                np.log(config.initial_temperature), 
                device=self.device, 
                requires_grad=True
            )
            self.temperature_optimizer = torch.optim.Adam(
                [self.log_temperature], lr=config.learning_rate
            )
        else:
            self.temperature = config.initial_temperature
        
        self.logger.info(f"SAC agent initialized with target_entropy={self.target_entropy}")
    
    def _setup_networks(self):
        """Setup SAC networks"""
        # Policy network
        self.policy_network = PolicyNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            log_std_min=self.sac_config.log_std_min,
            log_std_max=self.sac_config.log_std_max
        ).to(self.device)
        
        # Q-networks (double Q-learning)
        self.q_networks = DoubleQNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            dueling=self.sac_config.dueling_q
        ).to(self.device)
        
        # Target Q-networks
        self.target_q_networks = DoubleQNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            dueling=self.sac_config.dueling_q
        ).to(self.device)
        
        # Initialize target networks
        self.target_q_networks.load_state_dict(self.q_networks.state_dict())
        
        # Freeze target networks
        for param in self.target_q_networks.parameters():
            param.requires_grad = False
    
    def _setup_optimizers(self):
        """Setup SAC optimizers"""
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.config.learning_rate
        )
        
        self.q_optimizer = torch.optim.Adam(
            self.q_networks.parameters(),
            lr=self.config.learning_rate
        )
    
    async def select_action(self, state: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """Select action using SAC policy"""
        self.policy_network.eval()
        
        with torch.no_grad():
            state_tensor = self.preprocess_state(state).unsqueeze(0)
            action_tensor, _ = self.policy_network.sample_action(
                state_tensor, deterministic=deterministic
            )
        
        self.policy_network.train()
        
        # Convert to structured action
        structured_action = self.postprocess_action(action_tensor.squeeze(0))
        
        return structured_action
    
    def update_policy(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update SAC policy using replay buffer"""
        
        # Add experiences to replay buffer
        for exp in experiences:
            state = self.preprocess_state(exp['state'])
            action = self._action_dict_to_tensor(exp['action'])
            reward = float(exp['reward'])
            next_state = self.preprocess_state(exp['next_state'])
            done = bool(exp['done'])
            
            self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Update only if we have enough samples and past warm-up
        if (len(self.replay_buffer) < self.config.batch_size or 
            self.training_step < self.sac_config.warm_up_steps):
            return {}
        
        metrics = {}
        
        # Perform gradient steps
        for _ in range(self.sac_config.gradient_steps_per_update):
            
            # Sample batch from replay buffer
            if self.sac_config.prioritized_replay:
                batch, indices, weights = self.replay_buffer.sample(self.config.batch_size)
                weights = torch.tensor(weights, device=self.device)
            else:
                batch = self.replay_buffer.sample(self.config.batch_size)
                weights = None
            
            # Update Q-networks
            q_metrics = self._update_q_networks(batch, weights)
            metrics.update(q_metrics)
            
            # Update policy network
            policy_metrics = self._update_policy_network(batch)
            metrics.update(policy_metrics)
            
            # Update temperature (if automatic tuning)
            if self.sac_config.automatic_entropy_tuning:
                temp_metrics = self._update_temperature(batch)
                metrics.update(temp_metrics)
            
            # Update target networks
            if self.training_step % self.sac_config.target_update_interval == 0:
                self._soft_update_targets()
            
            # Update priorities (if using prioritized replay)
            if self.sac_config.prioritized_replay and indices is not None:
                td_errors = q_metrics.get('td_error', torch.zeros(len(indices)))
                self.replay_buffer.update_priorities(indices, td_errors.abs().cpu().numpy())
        
        self.training_step += 1
        self.training_metrics.update(metrics)
        
        return metrics
    
    def _update_q_networks(self, batch: Dict[str, torch.Tensor], 
                          weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Update Q-networks using Bellman equation"""
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Compute current Q-values
        q1_current, q2_current = self.q_networks(states, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions, next_log_probs = self.policy_network.sample_action(next_states)
            
            # Compute target Q-values using target networks
            q1_target, q2_target = self.target_q_networks(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            
            # Apply entropy regularization
            temperature = self._get_temperature()
            q_target = q_target - temperature * next_log_probs.unsqueeze(1)
            
            # Compute Bellman target
            target_values = rewards.unsqueeze(1) + self.config.gamma * (1 - dones.unsqueeze(1)) * q_target
        
        # Compute Q-losses
        q1_loss = F.mse_loss(q1_current, target_values, reduction='none')
        q2_loss = F.mse_loss(q2_current, target_values, reduction='none')
        
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            q1_loss = (q1_loss.squeeze() * weights).mean()
            q2_loss = (q2_loss.squeeze() * weights).mean()
        else:
            q1_loss = q1_loss.mean()
            q2_loss = q2_loss.mean()
        
        q_loss = q1_loss + q2_loss
        
        # Backward pass
        self.q_optimizer.zero_grad()
        q_loss.backward()
        
        # Gradient clipping
        q_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_networks.parameters(), self.config.grad_clip_norm
        )
        
        self.q_optimizer.step()
        
        # Compute TD error for prioritized replay
        with torch.no_grad():
            td_error = torch.abs(q1_current - target_values).squeeze()
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1_mean': q1_current.mean().item(),
            'q2_mean': q2_current.mean().item(),
            'target_q_mean': target_values.mean().item(),
            'q_grad_norm': q_grad_norm.item(),
            'td_error': td_error
        }
    
    def _update_policy_network(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy network using SAC policy gradient"""
        
        states = batch['states'].to(self.device)
        
        # Sample actions from current policy
        actions, log_probs = self.policy_network.sample_action(states)
        
        # Compute Q-values for sampled actions
        q1_values, q2_values = self.q_networks(states, actions)
        q_values = torch.min(q1_values, q2_values)
        
        # Policy loss with entropy regularization
        temperature = self._get_temperature()
        policy_loss = (temperature * log_probs.unsqueeze(1) - q_values).mean()
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.config.grad_clip_norm
        )
        
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'policy_entropy': -log_probs.mean().item(),
            'policy_grad_norm': policy_grad_norm.item(),
            'q_values_mean': q_values.mean().item()
        }
    
    def _update_temperature(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update temperature parameter for entropy regularization"""
        
        states = batch['states'].to(self.device)
        
        with torch.no_grad():
            _, log_probs = self.policy_network.sample_action(states)
        
        # Temperature loss
        temperature_loss = -(
            self.log_temperature * (log_probs + self.target_entropy).detach()
        ).mean()
        
        # Backward pass
        self.temperature_optimizer.zero_grad()
        temperature_loss.backward()
        self.temperature_optimizer.step()
        
        temperature = torch.exp(self.log_temperature).item()
        
        return {
            'temperature_loss': temperature_loss.item(),
            'temperature': temperature,
            'target_entropy': self.target_entropy
        }
    
    def _get_temperature(self) -> torch.Tensor:
        """Get current temperature value"""
        if self.sac_config.automatic_entropy_tuning:
            return torch.exp(self.log_temperature)
        else:
            return torch.tensor(self.temperature, device=self.device)
    
    def _soft_update_targets(self):
        """Soft update of target networks"""
        with torch.no_grad():
            for target_param, param in zip(
                self.target_q_networks.parameters(), 
                self.q_networks.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                )
    
    def _action_dict_to_tensor(self, action_dict: Dict[str, Any]) -> torch.Tensor:
        """Convert action dictionary to tensor format"""
        # This should match the postprocess_action output format
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
            action_dict["budget"] / 100.0,  # normalize
            action_dict["bid_amount"] / 10.0,  # normalize
            action_dict["audience_size"],
            float(action_dict["ab_test_enabled"]),
            action_dict["ab_test_split"]
        ])
        
        # Pad to action dimension if needed
        while len(action_values) < self.config.action_dim:
            action_values.append(0.0)
        
        return torch.tensor(action_values[:self.config.action_dim], dtype=torch.float32)
    
    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing"""
        state = {
            'policy_network': self.policy_network.state_dict(),
            'q_networks': self.q_networks.state_dict(),
            'target_q_networks': self.target_q_networks.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'replay_buffer': self.replay_buffer.get_state()
        }
        
        if self.sac_config.automatic_entropy_tuning:
            state.update({
                'log_temperature': self.log_temperature,
                'temperature_optimizer': self.temperature_optimizer.state_dict()
            })
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load agent state from checkpoint"""
        self.policy_network.load_state_dict(state['policy_network'])
        self.q_networks.load_state_dict(state['q_networks'])
        self.target_q_networks.load_state_dict(state['target_q_networks'])
        self.policy_optimizer.load_state_dict(state['policy_optimizer'])
        self.q_optimizer.load_state_dict(state['q_optimizer'])
        
        self.training_step = state.get('training_step', 0)
        self.episode_count = state.get('episode_count', 0)
        
        if 'replay_buffer' in state:
            self.replay_buffer.load_state(state['replay_buffer'])
        
        if (self.sac_config.automatic_entropy_tuning and 
            'log_temperature' in state):
            self.log_temperature = state['log_temperature']
            self.temperature_optimizer.load_state_dict(state['temperature_optimizer'])