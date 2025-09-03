"""
Advanced RL Agent with State-of-the-Art Features
Implements all sophisticated RL techniques for production-grade performance marketing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from collections import deque
import random
import copy
import json
import os
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass 
class AdvancedConfig:
    """Configuration for advanced RL features."""
    
    # Core parameters
    learning_rate: float = 0.0001
    gamma: float = 0.99
    tau: float = 0.001  # Soft update parameter
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    ucb_c: float = 2.0  # UCB exploration constant
    thompson_prior_alpha: float = 1.0
    thompson_prior_beta: float = 1.0
    
    # Advanced DQN features
    double_dqn: bool = True
    dueling_dqn: bool = True
    noisy_nets: bool = True
    categorical_dqn: bool = False  # C51
    n_atoms: int = 51  # For distributional RL
    v_min: float = -10.0
    v_max: float = 10.0
    
    # Prioritized Experience Replay
    per_alpha: float = 0.6  # Prioritization exponent
    per_beta_start: float = 0.4  # Importance sampling exponent
    per_beta_end: float = 1.0
    per_beta_steps: int = 100000
    per_epsilon: float = 1e-6  # Small constant for stability
    
    # Multi-objective optimization
    n_objectives: int = 4  # ROI, CTR, Budget, Safety
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    pareto_archive_size: int = 100
    
    # Curiosity and intrinsic motivation
    curiosity_weight: float = 0.1
    curiosity_type: str = "rnd"  # "rnd", "icm", "ngu"
    
    # Action masking
    use_action_masking: bool = True
    invalid_action_penalty: float = -10.0
    
    # Reward shaping
    use_reward_shaping: bool = True
    potential_discount: float = 0.99
    
    # Safety
    safe_exploration: bool = True
    constraint_threshold: float = 0.1
    
    # Memory
    buffer_size: int = 100000
    batch_size: int = 64
    min_buffer_size: int = 1000
    
    # Training
    update_frequency: int = 4
    target_update_frequency: int = 1000
    gradient_clip: float = 10.0
    
    # Checkpointing
    checkpoint_frequency: int = 5000
    checkpoint_dir: str = "checkpoints/advanced"


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration via parameter noise."""
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Factorized noise
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingNetwork(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 noisy: bool = False, n_atoms: int = 1):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        
        Linear = NoisyLinear if noisy else nn.Linear
        
        # Shared layers
        self.shared = nn.Sequential(
            Linear(state_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, n_atoms)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, action_dim * n_atoms)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.shared(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        if self.n_atoms > 1:
            # Distributional RL (C51)
            value = value.view(-1, 1, self.n_atoms)
            advantage = advantage.view(-1, self.action_dim, self.n_atoms)
        else:
            value = value.view(-1, 1)
            advantage = advantage.view(-1, self.action_dim)
        
        # Combine streams (dueling architecture)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def reset_noise(self):
        """Reset noise for noisy networks."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with importance sampling."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4,
                 beta_steps: int = 100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.frame = 0
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done, info=None):
        """Add experience with maximum priority."""
        experience = (state, action, reward, next_state, done, info)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with importance sampling weights."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        
        # Calculate sampling probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.frame / self.beta_steps)
        self.frame += 1
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = list(zip(*samples))
        
        # Handle states that might be stored as different types
        states_list = batch[0]
        if len(states_list) > 0 and isinstance(states_list[0], np.ndarray):
            # States are already arrays, stack them
            states = np.stack(states_list)
        else:
            # Convert to array safely
            states = np.array(states_list, dtype=np.float32)
        
        actions = np.array(batch[1], dtype=np.int64)
        rewards = np.array(batch[2], dtype=np.float32)
        
        # Handle next_states similarly
        next_states_list = batch[3]
        if len(next_states_list) > 0 and isinstance(next_states_list[0], np.ndarray):
            next_states = np.stack(next_states_list)
        else:
            next_states = np.array(next_states_list, dtype=np.float32)
        
        dones = np.array(batch[4], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class CuriosityModule(nn.Module):
    """Random Network Distillation (RND) for curiosity-driven exploration."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Fixed random target network
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Predictor network (trainable)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_features = self.target(state)
        predicted_features = self.predictor(state)
        return target_features, predicted_features
    
    def compute_intrinsic_reward(self, state: torch.Tensor) -> float:
        """Compute curiosity bonus based on prediction error."""
        with torch.no_grad():
            target_features, predicted_features = self.forward(state)
            intrinsic_reward = F.mse_loss(predicted_features, target_features).item()
        return intrinsic_reward


class RewardShaper:
    """Potential-based reward shaping for faster learning."""
    
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.potential_cache = {}
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Handle dataclasses and other objects with __dict__
            return self._make_serializable(obj.__dict__)
        elif hasattr(obj, 'value'):
            # Handle Enums
            return obj.value
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert to string as fallback
            return str(obj)
    
    def compute_potential(self, state: Dict[str, Any]) -> float:
        """Compute potential function for state."""
        # Handle both dict and dataclass states
        if hasattr(state, '__dict__'):
            # Convert dataclass to dict
            state_dict = state.__dict__
        else:
            state_dict = state
            
        # Convert state to serializable format
        serializable_state = self._make_serializable(state_dict)
        
        # Hash state for caching
        state_hash = hashlib.md5(json.dumps(serializable_state, sort_keys=True).encode()).hexdigest()
        
        if state_hash in self.potential_cache:
            return self.potential_cache[state_hash]
        
        # Compute potential based on progress metrics
        potential = 0.0
        
        # Progress towards conversion
        if isinstance(state_dict, dict) and 'conversion_probability' in state_dict:
            potential += state_dict['conversion_probability'] * 10.0
        
        # Budget efficiency
        if isinstance(state_dict, dict) and 'budget_spent' in state_dict and 'budget_total' in state_dict:
            efficiency = 1.0 - (state_dict['budget_spent'] / max(state_dict['budget_total'], 1))
            potential += efficiency * 5.0
        
        # CTR/CVR improvements
        if isinstance(state_dict, dict) and 'ctr' in state_dict:
            potential += state_dict['ctr'] * 20.0
        
        self.potential_cache[state_hash] = potential
        return potential
    
    def shape_reward(self, reward: float, state: Dict, next_state: Dict, done: bool) -> float:
        """Apply potential-based reward shaping."""
        current_potential = self.compute_potential(state)
        
        if done:
            next_potential = 0
        else:
            next_potential = self.compute_potential(next_state)
        
        shaped_reward = reward + self.gamma * next_potential - current_potential
        return shaped_reward


class ActionMasker:
    """Handles invalid action masking for constrained action spaces."""
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    
    def get_action_mask(self, state: Dict[str, Any]) -> np.ndarray:
        """Generate action mask based on current constraints."""
        mask = np.ones(self.action_dim, dtype=np.float32)
        
        # Budget constraints
        if state.get('budget_remaining', float('inf')) <= 0:
            # Mask high-bid actions
            mask[7:10] = 0  # Assuming high bid indices
        
        # Platform availability
        if not state.get('google_enabled', True):
            mask[0] = 0  # Google platform action
        if not state.get('facebook_enabled', True):
            mask[1] = 0  # Facebook platform action
        
        # Time-based constraints
        hour = state.get('hour', 12)
        if 0 <= hour < 6:  # Late night
            mask[5:7] = 0  # Mask aggressive bidding
        
        # Safety constraints
        if state.get('safety_violations', 0) > 3:
            mask[8:10] = 0  # Mask risky actions
        
        return mask
    
    def apply_action_mask(self, q_values: torch.Tensor, mask: np.ndarray,
                         invalid_penalty: float = -1e9) -> torch.Tensor:
        """Apply mask to Q-values to prevent invalid actions."""
        mask_tensor = torch.FloatTensor(mask).to(q_values.device)
        masked_q_values = q_values.clone()
        masked_q_values[mask_tensor == 0] = invalid_penalty
        return masked_q_values


class MultiObjectiveOptimizer:
    """Handles multi-objective optimization with Pareto frontiers."""
    
    def __init__(self, n_objectives: int, weights: Optional[List[float]] = None):
        self.n_objectives = n_objectives
        self.weights = weights or [1.0 / n_objectives] * n_objectives
        self.pareto_archive = []
    
    def scalarize_rewards(self, rewards: List[float]) -> float:
        """Weighted sum scalarization of multiple objectives."""
        return sum(r * w for r, w in zip(rewards, self.weights))
    
    def is_dominated(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 is dominated by obj2."""
        return all(o2 >= o1 for o1, o2 in zip(obj1, obj2)) and \
               any(o2 > o1 for o1, o2 in zip(obj1, obj2))
    
    def update_pareto_archive(self, objectives: List[float], state_action: Tuple):
        """Update Pareto archive with non-dominated solutions."""
        # Remove dominated solutions
        self.pareto_archive = [
            (obj, sa) for obj, sa in self.pareto_archive
            if not self.is_dominated(obj, objectives)
        ]
        
        # Add if not dominated
        if not any(self.is_dominated(objectives, obj) for obj, _ in self.pareto_archive):
            self.pareto_archive.append((objectives, state_action))
        
        # Limit archive size
        if len(self.pareto_archive) > 100:
            self.pareto_archive = self.pareto_archive[-100:]
    
    def get_pareto_optimal_action(self, state: Any, q_values: Dict[str, torch.Tensor]) -> int:
        """Select action from Pareto frontier."""
        if not self.pareto_archive:
            # Use weighted sum if needed
            combined_q = sum(q * w for q, w in zip(q_values.values(), self.weights))
            return combined_q.argmax().item()
        
        # Find closest Pareto solution
        # (Simplified - in practice would use more sophisticated selection)
        return random.choice([sa[1] for _, sa in self.pareto_archive])[1]


class AdvancedRLAgent:
    """
    State-of-the-art RL agent with all advanced features for production use.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[AdvancedConfig] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or AdvancedConfig()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DuelingNetwork(
            state_dim, action_dim,
            noisy=self.config.noisy_nets,
            n_atoms=self.config.n_atoms if self.config.categorical_dqn else 1
        ).to(self.device)
        
        self.target_network = DuelingNetwork(
            state_dim, action_dim,
            noisy=self.config.noisy_nets,
            n_atoms=self.config.n_atoms if self.config.categorical_dqn else 1
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        
        # Memory
        self.memory = PrioritizedReplayBuffer(
            self.config.buffer_size,
            alpha=self.config.per_alpha,
            beta_start=self.config.per_beta_start,
            beta_steps=self.config.per_beta_steps
        )
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        self.exploration_steps = 0
        
        # Advanced modules
        self.curiosity = CuriosityModule(state_dim).to(self.device)
        self.curiosity_optimizer = optim.Adam(self.curiosity.predictor.parameters(), lr=0.0001)
        
        self.reward_shaper = RewardShaper(self.config.gamma)
        self.action_masker = ActionMasker(action_dim)
        self.multi_objective = MultiObjectiveOptimizer(
            self.config.n_objectives,
            self.config.objective_weights
        )
        
        # Thompson Sampling arms for exploration
        self.thompson_arms = {}
        for i in range(action_dim):
            self.thompson_arms[i] = {
                'alpha': self.config.thompson_prior_alpha,
                'beta': self.config.thompson_prior_beta
            }
        
        # Metrics
        self.steps = 0
        self.episodes = 0
        self.losses = deque(maxlen=100)
        
        # Checkpointing
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        logger.info("Advanced RL Agent initialized with all features")
    
    def select_action(self, state: Dict[str, Any], explore: bool = True) -> int:
        """
        Select action with advanced exploration strategies.
        """
        state_array = self._dict_to_array(state)
        
        # Get action mask
        action_mask = self.action_masker.get_action_mask(state) if self.config.use_action_masking else None
        
        # Exploration decision
        if explore and random.random() < self.epsilon:
            # Advanced exploration
            if random.random() < 0.5:
                # Thompson Sampling
                action = self._thompson_sampling_action()
            else:
                # UCB exploration
                action = self._ucb_action(state_array)
        else:
            # Exploitation with action masking
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze(0)
                
                if action_mask is not None:
                    q_values = self.action_masker.apply_action_mask(q_values, action_mask)
                
                if self.config.noisy_nets:
                    self.q_network.reset_noise()
                
                action = q_values.argmax().item()
        
        # Validate action
        if action_mask is not None and action_mask[action] == 0:
            # Use valid action if needed
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
        
        self.exploration_steps += 1
        return action
    
    def _thompson_sampling_action(self) -> int:
        """Thompson Sampling for exploration."""
        samples = {}
        for action, params in self.thompson_arms.items():
            samples[action] = np.random.beta(params['alpha'], params['beta'])
        return max(samples, key=samples.get)
    
    def _ucb_action(self, state: np.ndarray) -> int:
        """Upper Confidence Bound exploration."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
        
        ucb_values = q_values + self.config.ucb_c * np.sqrt(np.log(self.steps + 1) / (self.exploration_steps + 1))
        return ucb_values.argmax()
    
    def train(self) -> Dict[str, float]:
        """
        Train with all advanced features.
        """
        if len(self.memory) < self.config.min_buffer_size:
            return {'loss': 0.0, 'td_error': 0.0, 'epsilon': self.epsilon, 'q_value': 0.0}
        
        # Sample from prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values with Double DQN
        with torch.no_grad():
            if self.config.double_dqn:
                # Use online network for action selection
                next_actions = self.q_network(next_states).argmax(1)
                # Use target network for value estimation
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(1)[0]
            
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # Compute loss with importance sampling weights
        td_errors = target_q - current_q
        loss = (weights * td_errors.pow(2)).mean()
        
        # Add curiosity loss if enabled
        if self.config.curiosity_weight > 0:
            target_features, predicted_features = self.curiosity(states)
            curiosity_loss = F.mse_loss(predicted_features, target_features.detach())
            loss += self.config.curiosity_weight * curiosity_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        priorities = td_errors.abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities)
        
        # Update curiosity module
        if self.config.curiosity_weight > 0:
            self.curiosity_optimizer.zero_grad()
            curiosity_loss.backward()
            self.curiosity_optimizer.step()
        
        # Soft update target network
        if self.steps % self.config.target_update_frequency == 0:
            self._soft_update_target_network()
        
        # Update exploration
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        # Update Thompson Sampling based on rewards
        for i, (action, reward) in enumerate(zip(actions.cpu().numpy(), rewards.cpu().numpy())):
            if reward > 0:
                self.thompson_arms[action]['alpha'] += 1
            else:
                self.thompson_arms[action]['beta'] += 1
        
        self.steps += 1
        self.losses.append(loss.item())
        
        # Checkpoint
        if self.steps % self.config.checkpoint_frequency == 0:
            self.save_checkpoint()
        
        return {
            'loss': loss.item(),
            'td_error': td_errors.mean().item(),
            'epsilon': self.epsilon,
            'q_value': current_q.mean().item()
        }
    
    def store_experience(self, state: Dict, action: int, reward: float,
                        next_state: Dict, done: bool, info: Dict = None):
        """Store experience with reward shaping and intrinsic motivation."""
        
        # Apply reward shaping
        if self.config.use_reward_shaping:
            reward = self.reward_shaper.shape_reward(reward, state, next_state, done)
        
        # Add intrinsic reward
        if self.config.curiosity_weight > 0:
            state_array = self._dict_to_array(state)
            intrinsic_reward = self.curiosity.compute_intrinsic_reward(
                torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
            )
            reward += self.config.curiosity_weight * intrinsic_reward
        
        # Handle multi-objective rewards
        if info and 'multi_objectives' in info:
            objectives = info['multi_objectives']
            reward = self.multi_objective.scalarize_rewards(objectives)
            self.multi_objective.update_pareto_archive(objectives, (state, action))
        
        # Store in prioritized buffer
        state_array = self._dict_to_array(state)
        next_state_array = self._dict_to_array(next_state)
        
        self.memory.add(state_array, action, reward, next_state_array, done, info)
    
    def _soft_update_target_network(self):
        """Soft update of target network parameters."""
        for target_param, param in zip(self.target_network.parameters(), 
                                      self.q_network.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def _dict_to_array(self, state: Dict) -> np.ndarray:
        """Convert state dict to numpy array."""
        # Handle both dict and dataclass states
        if hasattr(state, '__dict__'):
            # Convert dataclass to dict
            state_dict = state.__dict__
        else:
            state_dict = state
            
        # Extract relevant features
        features = []
        
        # Numeric features
        for key in ['hour', 'day_of_week', 'budget_remaining', 'ctr', 'cvr',
                   'competition_level', 'channel_performance']:
            if isinstance(state_dict, dict) and key in state_dict:
                features.append(float(state_dict[key]))
            else:
                features.append(0.0)
        
        # Categorical features (one-hot encode)
        if isinstance(state_dict, dict) and 'channel' in state_dict:
            channels = ['google', 'facebook', 'instagram', 'tiktok']
            channel_vec = [1.0 if state_dict['channel'] == c else 0.0 for c in channels]
            features.extend(channel_vec)
        
        # Pad to expected dimension
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features[:self.state_dim], dtype=np.float32)
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'curiosity': self.curiosity.state_dict(),
            'curiosity_optimizer': self.curiosity_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'thompson_arms': self.thompson_arms,
            'config': self.config
        }
        
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_{self.steps}.pt')
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        # PyTorch 2.6+ requires weights_only=False for custom classes
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.curiosity.load_state_dict(checkpoint['curiosity'])
        self.curiosity_optimizer.load_state_dict(checkpoint['curiosity_optimizer'])
        self.epsilon = checkpoint['epsilon']
    
    def get_bid_action(self, state, explore: bool = True):
        """Get bid action - wrapper for compatibility with master integration."""
        # Convert JourneyState to dict if needed
        if hasattr(state, 'to_dict'):
            state_dict = state.to_dict()
        else:
            state_dict = state if isinstance(state, dict) else {'budget_remaining': 1000}
        
        # Get action from main method
        action_idx = self.select_action(state_dict, explore=explore)
        
        # Map action index to bid value (simple linear mapping)
        min_bid = 0.5
        max_bid = 10.0
        bid_value = min_bid + (action_idx / self.action_dim) * (max_bid - min_bid)
        
        return action_idx, bid_value
    
    def get_creative_action(self, state, explore: bool = True):
        """Get creative action - wrapper for compatibility."""
        # For now, return a random creative ID
        creative_idx = np.random.randint(0, 10)
        return creative_idx, f"creative_{creative_idx}"
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.thompson_arms = checkpoint['thompson_arms']
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'thompson_arms': self.thompson_arms,
            'pareto_archive_size': len(self.multi_objective.pareto_archive),
            'device': str(self.device)
        }


def create_advanced_agent(state_dim: int = 20, action_dim: int = 10,
                         config_dict: Optional[Dict] = None) -> AdvancedRLAgent:
    """Factory function to create advanced RL agent."""
    
    config = AdvancedConfig()
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return AdvancedRLAgent(state_dim, action_dim, config)


if __name__ == "__main__":
    # Test advanced agent
    agent = create_advanced_agent()
    
    print("Advanced RL Agent Features:")
    print("✅ Double DQN")
    print("✅ Dueling Architecture") 
    print("✅ Noisy Networks")
    print("✅ Prioritized Experience Replay")
    print("✅ Action Masking")
    print("✅ Multi-Objective Optimization")
    print("✅ Curiosity-Driven Exploration (RND)")
    print("✅ Thompson Sampling")
    print("✅ UCB Exploration")
    print("✅ Reward Shaping")
    print("✅ Soft Target Updates")
    print("✅ Gradient Clipping")
    print("✅ Automatic Checkpointing")
    
    # Test action selection
    test_state = {
        'hour': 14,
        'day_of_week': 3,
        'budget_remaining': 50.0,
        'ctr': 0.02,
        'cvr': 0.01,
        'channel': 'google',
        'competition_level': 0.7
    }
    
    action = agent.select_action(test_state)
    print(f"\nSelected action: {action}")
    
    print("\nDiagnostics:", agent.get_diagnostics())