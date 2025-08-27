"""
Neural Network Architectures for RL Agents

Implements policy networks, value networks, and Q-networks optimized 
for ad campaign optimization tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import math


class MLPBlock(nn.Module):
    """Multi-layer perceptron block with customizable activation and normalization"""
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int],
        activation: str = "relu",
        use_layer_norm: bool = True,
        use_dropout: bool = False,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Build layer sequence
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add normalization (except for output layer)
            if i < len(dims) - 2 and use_layer_norm:
                self.layers.append(nn.LayerNorm(dims[i + 1]))
            
            # Add activation (except for output layer)
            if i < len(dims) - 2:
                if activation == "relu":
                    self.layers.append(nn.ReLU())
                elif activation == "gelu":
                    self.layers.append(nn.GELU())
                elif activation == "tanh":
                    self.layers.append(nn.Tanh())
                elif activation == "swish":
                    self.layers.append(nn.SiLU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                
                # Add dropout
                if use_dropout:
                    self.layers.append(nn.Dropout(dropout_rate))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PolicyNetwork(nn.Module):
    """
    Policy network for continuous and discrete action spaces.
    
    Outputs both continuous actions (budget, bid amounts) and discrete action logits
    (creative type, audience, bid strategy) for ad campaign optimization.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        use_layer_norm: bool = True,
        continuous_actions: int = 5,  # budget, bid_amount, audience_size, ab_test_split
        discrete_action_dims: List[int] = [3, 3, 3, 2],  # creative, audience, bid_strategy, ab_test
        log_std_init: float = -0.5,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_actions = continuous_actions
        self.discrete_action_dims = discrete_action_dims
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared feature extractor
        self.feature_extractor = MLPBlock(
            state_dim, hidden_dims[-1], hidden_dims[:-1], 
            activation, use_layer_norm
        )
        
        # Continuous action heads (mean and log_std)
        self.continuous_mean = nn.Linear(hidden_dims[-1], continuous_actions)
        self.continuous_log_std = nn.Parameter(
            torch.ones(continuous_actions) * log_std_init
        )
        
        # Discrete action heads
        self.discrete_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], dim) for dim in discrete_action_dims
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through policy network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            Tuple of (continuous_mean, continuous_log_std, discrete_logits_list)
        """
        features = self.feature_extractor(state)
        
        # Continuous actions
        continuous_mean = torch.tanh(self.continuous_mean(features))
        continuous_log_std = torch.clamp(
            self.continuous_log_std, self.log_std_min, self.log_std_max
        )
        
        # Discrete actions
        discrete_logits = [head(features) for head in self.discrete_heads]
        
        return continuous_mean, continuous_log_std, discrete_logits
    
    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            deterministic: If True, return mean action without noise
            
        Returns:
            Tuple of (action, log_prob)
        """
        continuous_mean, continuous_log_std, discrete_logits = self.forward(state)
        
        if deterministic:
            # Deterministic action
            continuous_action = continuous_mean
            discrete_actions = [torch.argmax(logits, dim=-1) for logits in discrete_logits]
            log_prob = torch.zeros(state.shape[0], device=state.device)
        else:
            # Stochastic action
            continuous_std = torch.exp(continuous_log_std)
            continuous_dist = torch.distributions.Normal(continuous_mean, continuous_std)
            continuous_action = continuous_dist.rsample()
            continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
            
            discrete_dists = [torch.distributions.Categorical(logits=logits) for logits in discrete_logits]
            discrete_actions = [dist.sample() for dist in discrete_dists]
            discrete_log_probs = [dist.log_prob(action) for dist, action in zip(discrete_dists, discrete_actions)]
            
            log_prob = continuous_log_prob + sum(discrete_log_probs)
        
        # Combine actions
        discrete_one_hot = [F.one_hot(action, num_classes=dim).float() 
                           for action, dim in zip(discrete_actions, self.discrete_action_dims)]
        discrete_tensor = torch.cat(discrete_one_hot, dim=-1)
        
        action = torch.cat([continuous_action, discrete_tensor], dim=-1)
        
        return action, log_prob
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given action"""
        continuous_mean, continuous_log_std, discrete_logits = self.forward(state)
        
        # Split action into continuous and discrete parts
        continuous_action = action[:, :self.continuous_actions]
        discrete_start = self.continuous_actions
        
        # Continuous log prob
        continuous_std = torch.exp(continuous_log_std)
        continuous_dist = torch.distributions.Normal(continuous_mean, continuous_std)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
        
        # Discrete log probs
        discrete_log_prob = 0
        for i, (logits, dim) in enumerate(zip(discrete_logits, self.discrete_action_dims)):
            discrete_action_one_hot = action[:, discrete_start:discrete_start + dim]
            discrete_action = torch.argmax(discrete_action_one_hot, dim=-1)
            discrete_dist = torch.distributions.Categorical(logits=logits)
            discrete_log_prob += discrete_dist.log_prob(discrete_action)
            discrete_start += dim
        
        return continuous_log_prob + discrete_log_prob


class ValueNetwork(nn.Module):
    """Value network for state value estimation"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.network = MLPBlock(
            state_dim, 1, hidden_dims, activation, use_layer_norm
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            State values [batch_size, 1]
        """
        return self.network(state)


class QNetwork(nn.Module):
    """Q-network for action-value estimation"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        use_layer_norm: bool = True,
        dueling: bool = False
    ):
        super().__init__()
        
        self.dueling = dueling
        self.action_dim = action_dim
        
        if dueling:
            # Dueling architecture
            self.feature_extractor = MLPBlock(
                state_dim + action_dim, hidden_dims[-1], hidden_dims[:-1],
                activation, use_layer_norm
            )
            
            self.value_head = nn.Linear(hidden_dims[-1], 1)
            self.advantage_head = nn.Linear(hidden_dims[-1], 1)
        else:
            # Standard Q-network
            self.network = MLPBlock(
                state_dim + action_dim, 1, hidden_dims,
                activation, use_layer_norm
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            action: Batch of actions [batch_size, action_dim]
            
        Returns:
            Q-values [batch_size, 1]
        """
        state_action = torch.cat([state, action], dim=-1)
        
        if self.dueling:
            features = self.feature_extractor(state_action)
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_value = self.network(state_action)
        
        return q_value


class DoubleQNetwork(nn.Module):
    """Double Q-network for reduced overestimation bias"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        use_layer_norm: bool = True,
        dueling: bool = False
    ):
        super().__init__()
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims, activation, use_layer_norm, dueling)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims, activation, use_layer_norm, dueling)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both Q-networks"""
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through first Q-network only"""
        return self.q1(state, action)


class AttentionPolicyNetwork(nn.Module):
    """
    Policy network with attention mechanism for processing variable-length
    campaign history and market context.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        attention_heads: int = 8,
        attention_dim: int = 128,
        max_sequence_length: int = 50
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.attention_dim = attention_dim
        self.max_sequence_length = max_sequence_length
        
        # State embedding
        self.state_embedding = nn.Linear(state_dim, attention_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            attention_dim, attention_heads, batch_first=True
        )
        
        # Policy head
        self.policy_network = PolicyNetwork(
            attention_dim, action_dim, hidden_dims
        )
        
        # Positional encoding
        self.register_buffer(
            'positional_encoding',
            self._create_positional_encoding(max_sequence_length, attention_dim)
        )
    
    def _create_positional_encoding(self, max_length: int, d_model: int) -> torch.Tensor:
        """Create positional encoding for sequence input"""
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, state_sequence: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass with attention over state sequence.
        
        Args:
            state_sequence: [batch_size, seq_length, state_dim]
            mask: Optional attention mask [batch_size, seq_length]
        """
        batch_size, seq_length, _ = state_sequence.shape
        
        # Embed states
        embedded_states = self.state_embedding(state_sequence)
        
        # Add positional encoding
        embedded_states += self.positional_encoding[:, :seq_length, :]
        
        # Apply attention
        attended_states, _ = self.attention(
            embedded_states, embedded_states, embedded_states,
            key_padding_mask=mask
        )
        
        # Use last attended state for policy
        final_state = attended_states[:, -1, :]
        
        return self.policy_network(final_state)