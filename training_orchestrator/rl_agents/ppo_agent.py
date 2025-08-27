"""
PPO (Proximal Policy Optimization) Agent

Implements PPO algorithm optimized for ad campaign optimization with stable
policy updates and effective exploration-exploitation balance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import logging

from .base_agent import BaseRLAgent, AgentConfig
from .networks import PolicyNetwork, ValueNetwork
from .replay_buffer import ReplayBuffer
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from importance_sampler import ImportanceSampler, Experience

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig(AgentConfig):
    """PPO-specific configuration"""
    
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    
    # Training parameters
    ppo_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    
    # Value function
    value_clipping: bool = True
    value_clip_epsilon: float = 0.2
    
    # Adaptive parameters
    adaptive_kl_target: float = 0.01
    adaptive_kl_factor: float = 1.5
    adaptive_clip: bool = False
    
    # Rollout collection
    rollout_length: int = 2048
    normalize_advantages: bool = True
    normalize_values: bool = False


class PPOAgent(BaseRLAgent):
    """
    PPO agent for ad campaign optimization.
    
    Uses clipped policy gradients and generalized advantage estimation
    for stable and sample-efficient learning.
    """
    
    def __init__(self, config: PPOConfig, agent_id: str):
        self.ppo_config = config
        super().__init__(config, agent_id)
        
        # PPO-specific tracking
        self.rollout_buffer = []
        self.current_rollout = []
        
        # Initialize importance sampler for crisis parent weighting
        self.importance_sampler = ImportanceSampler(
            population_ratios={"crisis_parent": 0.1, "regular_parent": 0.9},
            conversion_ratios={"crisis_parent": 0.5, "regular_parent": 0.5},
            max_weight=5.0,  # 5x weight for crisis parents
            alpha=0.6,
            beta_start=0.4
        )
        
        # Adaptive hyperparameters
        self.current_clip_epsilon = config.clip_epsilon
        self.current_lr = config.learning_rate
        
        self.logger.info(f"PPO agent initialized with clip_epsilon={config.clip_epsilon} and importance sampling")
    
    def _setup_networks(self):
        """Setup PPO networks"""
        self.policy_network = PolicyNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation
        ).to(self.device)
        
        self.value_network = ValueNetwork(
            state_dim=self.config.state_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation
        ).to(self.device)
        
        # Target networks for value clipping (optional)
        if self.ppo_config.value_clipping:
            self.old_value_network = ValueNetwork(
                state_dim=self.config.state_dim,
                hidden_dims=self.config.hidden_dims,
                activation=self.config.activation
            ).to(self.device)
            self.old_value_network.load_state_dict(self.value_network.state_dict())
    
    def _setup_optimizers(self):
        """Setup PPO optimizers"""
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )
        
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )
    
    async def select_action(self, state: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """Select action using PPO policy"""
        self.policy_network.eval()
        
        with torch.no_grad():
            state_tensor = self.preprocess_state(state).unsqueeze(0)
            action_tensor, log_prob = self.policy_network.sample_action(
                state_tensor, deterministic=deterministic
            )
            
            # Get value estimate for this state
            value = self.value_network(state_tensor)
            
            # Store for rollout collection
            if not deterministic:  # Only store during training
                self.current_rollout.append({
                    'state': state_tensor.cpu(),
                    'action': action_tensor.cpu(),
                    'log_prob': log_prob.cpu(),
                    'value': value.cpu(),
                    'raw_state': state
                })
        
        self.policy_network.train()
        
        # Convert to structured action
        structured_action = self.postprocess_action(action_tensor.squeeze(0))
        
        return structured_action
    
    def update_policy(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update PPO policy using collected rollout data with importance sampling"""
        
        # Convert experiences to importance sampler format and add to buffer
        for i, exp in enumerate(experiences):
            if i < len(self.current_rollout):
                rollout_item = self.current_rollout[i]
                
                # Determine event type based on experience metadata
                event_type = exp.get('event_type', 'regular_parent')
                if 'crisis' in exp.get('metadata', {}).get('user_profile', '').lower():
                    event_type = 'crisis_parent'
                
                # Create importance sampler experience
                importance_exp = Experience(
                    state=rollout_item['state'].numpy(),
                    action=rollout_item['action'].numpy(),
                    reward=exp.get('reward', 0.0),
                    next_state=exp.get('next_state', rollout_item['state'].numpy()),
                    done=exp.get('done', False),
                    value=exp.get('value', rollout_item['value'].item()),
                    event_type=event_type,
                    timestamp=exp.get('timestamp', self.training_step),
                    metadata=exp.get('metadata', {})
                )
                
                self.importance_sampler.add_experience(importance_exp)
        
        # If we have enough experience for an update
        if len(self.current_rollout) >= self.ppo_config.rollout_length:
            
            # Use importance sampling for experience selection
            sampled_experiences, importance_weights, sampled_indices = self.importance_sampler.weighted_sampling(
                batch_size=min(len(self.current_rollout), self.ppo_config.rollout_length),
                temperature=1.0
            )
            
            # Process rollout with importance-weighted experiences
            rollout_data = self._process_rollout_with_importance_sampling(
                experiences, sampled_experiences, importance_weights, sampled_indices
            )
            
            # Perform PPO update with bias correction
            metrics = self._ppo_update_with_importance_sampling(rollout_data, importance_weights)
            
            # Clear rollout buffer
            self.current_rollout = []
            
            self.training_step += 1
            
            # Update learning rate and clip epsilon if adaptive
            if self.ppo_config.adaptive_clip:
                self._update_adaptive_parameters(metrics)
            
            # Add importance sampling statistics to metrics
            sampling_stats = self.importance_sampler.get_sampling_statistics()
            metrics.update({
                'importance_sampling_stats': sampling_stats,
                'crisis_parent_weight': sampling_stats.get('importance_weights', {}).get('crisis_parent', 1.0),
                'regular_parent_weight': sampling_stats.get('importance_weights', {}).get('regular_parent', 1.0)
            })
            
            return metrics
        
        return {}
    
    def _process_rollout(self, experiences: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process rollout data and compute advantages"""
        
        # Match experiences with rollout data
        rewards = []
        dones = []
        
        for i, exp in enumerate(experiences[-len(self.current_rollout):]):
            rewards.append(exp.get('reward', 0.0))
            dones.append(exp.get('done', False))
        
        # Pad if necessary
        while len(rewards) < len(self.current_rollout):
            rewards.append(0.0)
            dones.append(True)
        
        # Convert rollout to tensors
        states = torch.cat([item['state'] for item in self.current_rollout])
        actions = torch.cat([item['action'] for item in self.current_rollout])
        old_log_probs = torch.cat([item['log_prob'] for item in self.current_rollout])
        values = torch.cat([item['value'] for item in self.current_rollout])
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        # Normalize advantages
        if self.ppo_config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': states.to(self.device),
            'actions': actions.to(self.device),
            'old_log_probs': old_log_probs.to(self.device),
            'advantages': advantages.to(self.device),
            'returns': returns.to(self.device),
            'old_values': values.to(self.device)
        }
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                     dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Bootstrap value for last state (assume 0 if episode ended)
        next_values = torch.cat([values[1:], torch.zeros(1)])
        next_values[dones] = 0  # Zero value for terminal states
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * next_values[t] - values[t]
            advantages[t] = delta + self.config.gamma * self.ppo_config.gae_lambda * last_advantage
            
            if dones[t]:
                last_advantage = 0
            else:
                last_advantage = advantages[t]
        
        returns = advantages + values
        
        return advantages, returns
    
    def _process_rollout_with_importance_sampling(
        self,
        experiences: List[Dict[str, Any]],
        sampled_experiences: List[Experience],
        importance_weights: List[float],
        sampled_indices: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Process rollout data with importance-weighted experiences"""
        
        # Use sampled indices to select corresponding rollout items and experiences
        selected_rollout_items = [self.current_rollout[i % len(self.current_rollout)] for i in sampled_indices]
        selected_experiences = experiences[:len(selected_rollout_items)]
        
        # Match experiences with rollout data
        rewards = []
        dones = []
        
        for i, (exp, sampled_exp) in enumerate(zip(selected_experiences, sampled_experiences)):
            rewards.append(exp.get('reward', sampled_exp.reward))
            dones.append(exp.get('done', sampled_exp.done))
        
        # Convert rollout to tensors using selected items
        states = torch.cat([item['state'] for item in selected_rollout_items])
        actions = torch.cat([item['action'] for item in selected_rollout_items])
        old_log_probs = torch.cat([item['log_prob'] for item in selected_rollout_items])
        values = torch.cat([item['value'] for item in selected_rollout_items])
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        importance_weights_tensor = torch.tensor(importance_weights, dtype=torch.float32)
        
        # Compute advantages using GAE (same as before)
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        # Apply importance weights to advantages
        weighted_advantages = advantages * importance_weights_tensor
        
        # Normalize weighted advantages
        if self.ppo_config.normalize_advantages:
            weighted_advantages = (weighted_advantages - weighted_advantages.mean()) / (weighted_advantages.std() + 1e-8)
        
        return {
            'states': states.to(self.device),
            'actions': actions.to(self.device),
            'old_log_probs': old_log_probs.to(self.device),
            'advantages': weighted_advantages.to(self.device),
            'returns': returns.to(self.device),
            'old_values': values.to(self.device),
            'importance_weights': importance_weights_tensor.to(self.device)
        }
    
    def _ppo_update_with_importance_sampling(
        self,
        rollout_data: Dict[str, torch.Tensor],
        importance_weights: List[float]
    ) -> Dict[str, float]:
        """Perform PPO policy update with importance sampling bias correction"""
        
        metrics = {}
        
        # Get data
        states = rollout_data['states']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['old_log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        old_values = rollout_data['old_values']
        is_weights = rollout_data['importance_weights']
        
        batch_size = len(states)
        
        # Update target value network if using value clipping
        if self.ppo_config.value_clipping:
            self.old_value_network.load_state_dict(self.value_network.state_dict())
        
        # Multiple epochs of updates
        for epoch in range(self.ppo_config.ppo_epochs):
            
            # Create minibatches
            indices = torch.randperm(batch_size)
            
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_entropy_losses = []
            epoch_kl_divs = []
            
            for start in range(0, batch_size, self.ppo_config.minibatch_size):
                end = min(start + self.ppo_config.minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                # Get minibatch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_values[mb_indices]
                mb_is_weights = is_weights[mb_indices]
                
                # Forward pass through current policy
                new_log_probs = self.policy_network.log_prob(mb_states, mb_actions)
                new_values = self.value_network(mb_states).squeeze()
                
                # Policy loss (clipped) with importance sampling correction
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 
                    1 - self.current_clip_epsilon, 
                    1 + self.current_clip_epsilon
                )
                
                policy_loss1 = ratio * mb_advantages
                policy_loss2 = clipped_ratio * mb_advantages
                policy_loss = -torch.min(policy_loss1, policy_loss2)
                
                # Apply importance sampling weights to policy loss
                weighted_policy_loss = (policy_loss * mb_is_weights).mean()
                
                # Value loss (optionally clipped)
                if self.ppo_config.value_clipping:
                    old_values_pred = self.old_value_network(mb_states).squeeze()
                    value_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values,
                        -self.ppo_config.value_clip_epsilon,
                        self.ppo_config.value_clip_epsilon
                    )
                    value_loss1 = (new_values - mb_returns) ** 2
                    value_loss2 = (value_clipped - mb_returns) ** 2
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = (new_values - mb_returns) ** 2
                
                # Apply importance sampling weights to value loss
                weighted_value_loss = (value_loss * mb_is_weights).mean()
                
                # Entropy loss (for exploration) - no IS weighting needed
                entropy_loss = -self._compute_entropy(mb_states).mean()
                
                # Total loss
                total_loss = (
                    weighted_policy_loss + 
                    self.ppo_config.value_loss_coef * weighted_value_loss +
                    self.ppo_config.entropy_coef * entropy_loss
                )
                
                # Compute gradients
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Apply bias correction to gradients using importance sampler
                policy_grads = []
                for param in self.policy_network.parameters():
                    if param.grad is not None:
                        policy_grads.append(param.grad.view(-1))
                
                if policy_grads:
                    policy_grad_vector = torch.cat(policy_grads)
                    corrected_policy_grads = self.importance_sampler.bias_correction(
                        policy_grad_vector.cpu().numpy(),
                        importance_weights,
                        len(mb_indices)
                    )
                    
                    # Redistribute corrected gradients back to parameters
                    start_idx = 0
                    for param in self.policy_network.parameters():
                        if param.grad is not None:
                            param_size = param.grad.numel()
                            param.grad = torch.tensor(
                                corrected_policy_grads[start_idx:start_idx + param_size]
                            ).view(param.grad.shape).to(param.device)
                            start_idx += param_size
                
                # Gradient clipping
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(), self.ppo_config.max_grad_norm
                )
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_network.parameters(), self.ppo_config.max_grad_norm
                )
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Track metrics
                epoch_policy_losses.append(weighted_policy_loss.item())
                epoch_value_losses.append(weighted_value_loss.item())
                epoch_entropy_losses.append(entropy_loss.item())
                
                # KL divergence for adaptive clipping
                with torch.no_grad():
                    kl_div = (mb_old_log_probs - new_log_probs).mean()
                    epoch_kl_divs.append(kl_div.item())
        
        # Aggregate metrics
        metrics.update({
            'policy_loss': np.mean(epoch_policy_losses),
            'value_loss': np.mean(epoch_value_losses), 
            'entropy_loss': np.mean(epoch_entropy_losses),
            'kl_divergence': np.mean(epoch_kl_divs),
            'clip_epsilon': self.current_clip_epsilon,
            'learning_rate': self.current_lr,
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'value_mean': old_values.mean().item(),
            'importance_weights_mean': is_weights.mean().item(),
            'importance_weights_std': is_weights.std().item()
        })
        
        self.training_metrics.update(metrics)
        
        return metrics
    
    def _ppo_update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform PPO policy update"""
        
        metrics = {}
        
        # Get data
        states = rollout_data['states']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['old_log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        old_values = rollout_data['old_values']
        
        batch_size = len(states)
        
        # Update target value network if using value clipping
        if self.ppo_config.value_clipping:
            self.old_value_network.load_state_dict(self.value_network.state_dict())
        
        # Multiple epochs of updates
        for epoch in range(self.ppo_config.ppo_epochs):
            
            # Create minibatches
            indices = torch.randperm(batch_size)
            
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_entropy_losses = []
            epoch_kl_divs = []
            
            for start in range(0, batch_size, self.ppo_config.minibatch_size):
                end = min(start + self.ppo_config.minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                # Get minibatch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_values[mb_indices]
                
                # Forward pass through current policy
                new_log_probs = self.policy_network.log_prob(mb_states, mb_actions)
                new_values = self.value_network(mb_states).squeeze()
                
                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 
                    1 - self.current_clip_epsilon, 
                    1 + self.current_clip_epsilon
                )
                
                policy_loss1 = ratio * mb_advantages
                policy_loss2 = clipped_ratio * mb_advantages
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                
                # Value loss (optionally clipped)
                if self.ppo_config.value_clipping:
                    old_values_pred = self.old_value_network(mb_states).squeeze()
                    value_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values,
                        -self.ppo_config.value_clip_epsilon,
                        self.ppo_config.value_clip_epsilon
                    )
                    value_loss1 = (new_values - mb_returns) ** 2
                    value_loss2 = (value_clipped - mb_returns) ** 2
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = F.mse_loss(new_values, mb_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -self._compute_entropy(mb_states).mean()
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.ppo_config.value_loss_coef * value_loss +
                    self.ppo_config.entropy_coef * entropy_loss
                )
                
                # Backward pass
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(), self.ppo_config.max_grad_norm
                )
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_network.parameters(), self.ppo_config.max_grad_norm
                )
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Track metrics
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropy_losses.append(entropy_loss.item())
                
                # KL divergence for adaptive clipping
                with torch.no_grad():
                    kl_div = (mb_old_log_probs - new_log_probs).mean()
                    epoch_kl_divs.append(kl_div.item())
        
        # Aggregate metrics
        metrics.update({
            'policy_loss': np.mean(epoch_policy_losses),
            'value_loss': np.mean(epoch_value_losses), 
            'entropy_loss': np.mean(epoch_entropy_losses),
            'kl_divergence': np.mean(epoch_kl_divs),
            'clip_epsilon': self.current_clip_epsilon,
            'learning_rate': self.current_lr,
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'value_mean': old_values.mean().item()
        })
        
        self.training_metrics.update(metrics)
        
        return metrics
    
    def _compute_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """Compute policy entropy for given states"""
        continuous_mean, continuous_log_std, discrete_logits = self.policy_network(states)
        
        # Continuous entropy
        continuous_entropy = 0.5 * (continuous_log_std + np.log(2 * np.pi * np.e)).sum(dim=-1)
        
        # Discrete entropy  
        discrete_entropy = 0
        for logits in discrete_logits:
            probs = F.softmax(logits, dim=-1)
            discrete_entropy += -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        return continuous_entropy + discrete_entropy
    
    def _update_adaptive_parameters(self, metrics: Dict[str, float]):
        """Update adaptive hyperparameters based on training metrics"""
        
        kl_div = metrics.get('kl_divergence', 0)
        
        # Adaptive clip epsilon based on KL divergence
        if kl_div > self.ppo_config.adaptive_kl_target * self.ppo_config.adaptive_kl_factor:
            self.current_clip_epsilon = max(0.01, self.current_clip_epsilon * 0.9)
        elif kl_div < self.ppo_config.adaptive_kl_target / self.ppo_config.adaptive_kl_factor:
            self.current_clip_epsilon = min(0.5, self.current_clip_epsilon * 1.1)
        
        self.logger.debug(f"Adaptive clip epsilon: {self.current_clip_epsilon:.4f}, KL: {kl_div:.6f}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing"""
        return {
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'current_clip_epsilon': self.current_clip_epsilon,
            'current_lr': self.current_lr,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'importance_sampler_stats': self.importance_sampler.get_sampling_statistics()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load agent state from checkpoint"""
        self.policy_network.load_state_dict(state['policy_network'])
        self.value_network.load_state_dict(state['value_network'])
        self.policy_optimizer.load_state_dict(state['policy_optimizer'])
        self.value_optimizer.load_state_dict(state['value_optimizer'])
        
        self.current_clip_epsilon = state.get('current_clip_epsilon', self.ppo_config.clip_epsilon)
        self.current_lr = state.get('current_lr', self.config.learning_rate)
        self.training_step = state.get('training_step', 0)
        self.episode_count = state.get('episode_count', 0)
        
        # Restore importance sampler statistics if available
        if 'importance_sampler_stats' in state:
            stats = state['importance_sampler_stats']
            if 'population_ratios' in stats:
                self.importance_sampler.population_ratios = stats['population_ratios']
            if 'conversion_ratios' in stats:
                self.importance_sampler.conversion_ratios = stats['conversion_ratios']
            if 'frame_count' in stats:
                self.importance_sampler.frame_count = stats['frame_count']
            self.importance_sampler._update_importance_weights()
    
    def set_learning_rate(self, lr: float):
        """Update learning rate for both optimizers"""
        self.current_lr = lr
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = lr