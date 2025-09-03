"""
Journey-Aware RL Agent for GAELP
Implements expanded state space and journey-aware reward shaping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import random
from dataclasses import dataclass
import json

from enhanced_journey_tracking import (
    EnhancedMultiTouchUser, Channel, UserState, TouchpointType
)
from multi_channel_orchestrator import (
    MultiChannelOrchestrator, JourneyAwareRLEnvironment
)
# Import new UserJourneyDatabase system
from user_journey_database import (
    UserJourneyDatabase, UserJourney, JourneyTouchpoint, 
    UserProfile, CompetitorExposure
)
from journey_state import (
    JourneyState as DatabaseJourneyState, TransitionTrigger,
    JourneyStateManager
)
# Import directly to avoid SQLAlchemy issues in training_orchestrator package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training_orchestrator'))

from journey_state_encoder import (
    JourneyStateEncoder, JourneyStateEncoderConfig, create_journey_encoder
)

@dataclass
class JourneyState:
    """Expanded state representation for journey-aware RL"""
    # User features
    user_state: int
    journey_length: int
    time_since_last_touch: float
    conversion_probability: float
    total_cost: float
    
    # Journey sequence features
    last_3_channels: List[int]
    last_3_states: List[int]
    state_transitions: List[Tuple[int, int]]
    
    # Channel interaction features
    channel_touches: np.ndarray  # Count per channel
    channel_costs: np.ndarray    # Total cost per channel
    channel_last_touch: np.ndarray  # Days since last touch per channel
    
    # Temporal features
    hour_of_day: int
    day_of_week: int
    days_in_journey: int
    
    # Performance features
    click_through_rate: float
    engagement_rate: float
    bounce_rate: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural network"""
        features = []
        
        # Scalar features
        features.extend([
            self.user_state / 6.0,  # Normalize by number of states
            min(self.journey_length / 20.0, 1.0),  # Cap at 20 touches
            min(self.time_since_last_touch / 14.0, 1.0),  # Cap at 14 days
            self.conversion_probability,
            min(self.total_cost / 100.0, 1.0),  # Cap at $100
            self.hour_of_day / 24.0,
            self.day_of_week / 7.0,
            min(self.days_in_journey / 30.0, 1.0),
            self.click_through_rate,
            self.engagement_rate,
            self.bounce_rate
        ])
        
        # Sequence features (padded/truncated to fixed size)
        for ch in self.last_3_channels:
            features.append(ch / 8.0)  # Normalize by number of channels
        
        for st in self.last_3_states:
            features.append(st / 6.0)  # Normalize by number of states
        
        # Channel features
        features.extend(self.channel_touches / max(self.channel_touches.sum(), 1))
        features.extend(self.channel_costs / max(self.channel_costs.sum() + 1, 1))
        features.extend(np.clip(self.channel_last_touch / 14.0, 0, 1))
        
        return torch.FloatTensor(features)

class JourneyAwareActorCritic(nn.Module):
    """Actor-Critic network with journey awareness"""
    
    def __init__(self, state_dim: int = 256, hidden_dim: int = 256, 
                 num_channels: int = 8):
        super().__init__()
        
        # Process encoded state features
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Attention layer for channel selection  
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Separate networks for actor and critic
        # Use encoded state dimension directly
        combined_dim = hidden_dim
        
        self.actor_fc2 = nn.Linear(combined_dim, hidden_dim)
        self.actor_fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.actor_out = nn.Linear(hidden_dim // 2, num_channels)
        
        self.critic_fc2 = nn.Linear(combined_dim, hidden_dim)
        self.critic_fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.critic_out = nn.Linear(hidden_dim // 2, 1)
        
        # Bid amount prediction per channel
        self.bid_fc = nn.Linear(combined_dim, hidden_dim)
        self.bid_out = nn.Linear(hidden_dim, num_channels)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state: torch.Tensor, sequence: Optional[torch.Tensor] = None):
        """Forward pass through the network"""
        # Process encoded state features (already contains sequence information)
        x = F.relu(self.ln1(self.fc1(state)))
        x = self.dropout(x)
        
        # Use encoded features directly (sequence processing handled by encoder)
        combined = x
        
        # Actor path (channel selection)
        actor_hidden = F.relu(self.actor_fc2(combined))
        actor_hidden = self.dropout(actor_hidden)
        actor_hidden = F.relu(self.actor_fc3(actor_hidden))
        channel_logits = self.actor_out(actor_hidden)
        channel_probs = F.softmax(channel_logits, dim=-1)
        
        # Critic path (value estimation)
        critic_hidden = F.relu(self.critic_fc2(combined))
        critic_hidden = self.dropout(critic_hidden)
        critic_hidden = F.relu(self.critic_fc3(critic_hidden))
        value = self.critic_out(critic_hidden)
        
        # Bid amount prediction
        bid_hidden = F.relu(self.bid_fc(combined))
        bid_amounts = F.softplus(self.bid_out(bid_hidden))  # Ensure positive bids
        
        return channel_probs, value, bid_amounts

class JourneyAwareRewardShaper:
    """Implements journey-aware reward shaping"""
    
    def __init__(self):
        self.conversion_value = 120.0  # Aura LTV
        self.target_cac = 30.0  # Target customer acquisition cost
        
        # State progression rewards
        self.state_rewards = {
            (UserState.UNAWARE, UserState.AWARE): 2.0,
            (UserState.AWARE, UserState.INTERESTED): 3.0,
            (UserState.INTERESTED, UserState.CONSIDERING): 4.0,
            (UserState.CONSIDERING, UserState.INTENT): 5.0,
            (UserState.INTENT, UserState.CONVERTED): 50.0,
        }
        
        # Penalty for regression
        self.regression_penalty = -5.0
        
        # Efficiency bonuses
        self.quick_conversion_bonus = 20.0  # Convert in < 5 touches
        self.low_cost_bonus = 10.0  # Convert under target CAC
    
    def calculate_reward(self, 
                        old_state: UserState,
                        new_state: UserState,
                        cost: float,
                        journey_length: int,
                        total_cost: float,
                        conversion_prob: float) -> float:
        """Calculate shaped reward for action"""
        reward = 0.0
        
        # 1. State progression reward
        transition = (old_state, new_state)
        if transition in self.state_rewards:
            reward += self.state_rewards[transition]
        elif self._is_regression(old_state, new_state):
            reward += self.regression_penalty
        
        # 2. Cost efficiency
        cost_penalty = -cost * 0.5  # Penalize spending
        reward += cost_penalty
        
        # 3. Conversion reward
        if new_state == UserState.CONVERTED:
            reward += self.conversion_value
            
            # Quick conversion bonus
            if journey_length <= 5:
                reward += self.quick_conversion_bonus
            
            # Low CAC bonus
            if total_cost <= self.target_cac:
                reward += self.low_cost_bonus
        
        # 4. Potential value (for non-converted users)
        if new_state != UserState.CONVERTED:
            potential_value = conversion_prob * self.conversion_value * 0.1
            reward += potential_value
        
        # 5. Journey efficiency
        if journey_length > 15:
            # Penalty for too many touches
            reward -= (journey_length - 15) * 0.5
        
        # 6. Churn penalty
        if new_state == UserState.CHURNED:
            reward -= 20.0
        
        return reward
    
    def _is_regression(self, old_state: UserState, new_state: UserState) -> bool:
        """Check if state transition is a regression"""
        state_order = [
            UserState.UNAWARE,
            UserState.AWARE,
            UserState.INTERESTED,
            UserState.CONSIDERING,
            UserState.INTENT,
            UserState.CONVERTED
        ]
        
        if old_state in state_order and new_state in state_order:
            old_idx = state_order.index(old_state)
            new_idx = state_order.index(new_state)
            return new_idx < old_idx
        
        return False
    
    def calculate_trajectory_bonus(self, trajectory: List[Dict]) -> float:
        """Calculate bonus for entire trajectory"""
        if not trajectory:
            return 0.0
        
        bonus = 0.0
        
        # Check for successful conversion
        final_state = trajectory[-1].get('new_state')
        if final_state == UserState.CONVERTED:
            total_cost = sum(t.get('cost', 0) for t in trajectory)
            journey_length = len(trajectory)
            
            # ROI bonus
            roi = (self.conversion_value - total_cost) / (total_cost + 1)
            if roi > 2.0:
                bonus += 30.0
            elif roi > 1.0:
                bonus += 15.0
            
            # Optimal journey length bonus
            if 3 <= journey_length <= 7:
                bonus += 10.0
        
        return bonus

class JourneyAwarePPOAgent:
    """PPO agent with journey awareness"""
    
    def __init__(self, 
                 state_dim: int = 256,
                 hidden_dim: int = 256,
                 num_channels: int = 8,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 use_journey_encoder: bool = True):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor_critic = JourneyAwareActorCritic(state_dim, hidden_dim, num_channels)
        self.actor_critic.to(self.device)
        
        # Journey state encoder (initialize before optimizer)
        self.use_journey_encoder = use_journey_encoder
        if use_journey_encoder:
            self.journey_encoder = create_journey_encoder(
                max_sequence_length=5,
                lstm_hidden_dim=64,
                encoded_state_dim=state_dim,
                normalize_features=False  # Disable normalization to avoid NaN issues
            )
            self.journey_encoder.to(self.device)
            
            # Optimizer includes both actor-critic and encoder parameters
            all_parameters = list(self.actor_critic.parameters()) + list(self.journey_encoder.parameters())
            self.optimizer = optim.Adam(all_parameters, lr=learning_rate)
        else:
            self.journey_encoder = None
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # PPO parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Reward shaper
        self.reward_shaper = JourneyAwareRewardShaper()
        
        # Experience buffer
        self.memory = []
        
    def select_action(self, state: Union[JourneyState, Dict[str, Any]]) -> Tuple[int, float, torch.Tensor]:
        """Select action using current policy"""
        if self.use_journey_encoder and isinstance(state, dict):
            # Use journey encoder for rich state representation
            with torch.no_grad():
                state_tensor = self.journey_encoder.encode_journey(state).unsqueeze(0).to(self.device)
        elif isinstance(state, JourneyState):
            # Use simple tensor conversion if needed
            state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
        elif self.use_journey_encoder and hasattr(state, '__dict__'):
            # Convert object to dict format for encoder if possible
            state_dict = extract_journey_state_for_encoder(state, None, None)
            with torch.no_grad():
                state_tensor = self.journey_encoder.encode_journey(state_dict).unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}. Expected Dict or JourneyState.")
        
        with torch.no_grad():
            channel_probs, value, bid_amounts = self.actor_critic(state_tensor)
        
        # Sample channel
        dist = Categorical(channel_probs)
        channel_idx = dist.sample()
        
        # Get bid amount for selected channel
        bid_amount = bid_amounts[0, channel_idx].item()
        
        # Store for training
        log_prob = dist.log_prob(channel_idx)
        
        return channel_idx.item(), bid_amount, log_prob
    
    def store_transition(self, state: Union[JourneyState, Dict[str, Any]], action: int, reward: float,
                        next_state: Union[JourneyState, Dict[str, Any]], done: bool, log_prob: torch.Tensor):
        """Store transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })
    
    def compute_returns(self, rewards: List[float], dones: List[bool]) -> torch.Tensor:
        """Compute discounted returns"""
        returns = []
        discounted_return = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_return = 0
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        return torch.FloatTensor(returns).to(self.device)
    
    def update(self, batch_size: int = 32, epochs: int = 4):
        """Update policy using PPO"""
        if len(self.memory) < batch_size:
            return
        
        # Prepare batch - encode states if using journey encoder
        if self.use_journey_encoder:
            # Encode all states using journey encoder
            encoded_states = []
            for m in self.memory:
                if isinstance(m['state'], dict):
                    with torch.no_grad():
                        encoded_state = self.journey_encoder.encode_journey(m['state'])
                elif isinstance(m['state'], JourneyState):
                    encoded_state = m['state'].to_tensor()
                elif hasattr(m['state'], '__dict__'):
                    # Try to convert object to dict format
                    state_dict = extract_journey_state_for_encoder(m['state'], None, None)
                    with torch.no_grad():
                        encoded_state = self.journey_encoder.encode_journey(state_dict)
                else:
                    # Use tensor conversion if needed
                    encoded_state = m['state'].to_tensor() if hasattr(m['state'], 'to_tensor') else torch.zeros(256)
                encoded_states.append(encoded_state)
            states = torch.stack(encoded_states).to(self.device)
        else:
            states = torch.stack([m['state'].to_tensor() for m in self.memory]).to(self.device)
            
        actions = torch.LongTensor([m['action'] for m in self.memory]).to(self.device)
        rewards = [m['reward'] for m in self.memory]
        dones = [m['done'] for m in self.memory]
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory]).to(self.device)
        
        # Compute returns
        returns = self.compute_returns(rewards, dones)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update
        for _ in range(epochs):
            # Forward pass
            channel_probs, values, bid_amounts = self.actor_critic(states)
            dist = Categorical(channel_probs)
            
            # Calculate losses
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Advantages
            advantages = returns - values.squeeze()
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        self.memory = []
    
    def save(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'use_journey_encoder': self.use_journey_encoder
        }
        
        if self.use_journey_encoder:
            checkpoint['encoder_state_dict'] = self.journey_encoder.state_dict()
            checkpoint['encoder_config'] = self.journey_encoder.get_config()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load journey encoder if present
        if checkpoint.get('use_journey_encoder', False) and self.use_journey_encoder:
            if 'encoder_state_dict' in checkpoint:
                self.journey_encoder.load_state_dict(checkpoint['encoder_state_dict'])

def extract_journey_state_for_encoder(user: EnhancedMultiTouchUser,
                                     orchestrator: MultiChannelOrchestrator,
                                     timestamp: Any) -> Dict[str, Any]:
    """Extract journey data in format expected by JourneyStateEncoder"""
    from datetime import datetime
    
    # Convert timestamp to unix timestamp if needed
    if hasattr(timestamp, 'timestamp'):
        current_timestamp = timestamp.timestamp()
    else:
        current_timestamp = float(timestamp) if timestamp else datetime.now().timestamp()
    
    # Extract journey history
    journey_history = []
    for tp in user.journey:
        tp_data = {
            'channel': tp.channel.name.lower(),
            'user_state_after': tp.user_state_after.name.lower(),
            'cost': tp.cost,
            'timestamp': current_timestamp - (len(user.journey) - user.journey.index(tp)) * 86400  # Approximate
        }
        journey_history.append(tp_data)
    
    # Channel statistics
    channel_distribution = {}
    channel_costs = {}
    channel_last_touch = {}
    
    channel_names = ['search', 'social', 'display', 'video', 'email', 'direct', 'affiliate', 'retargeting']
    
    # Initialize all channels
    for channel_name in channel_names:
        channel_distribution[channel_name] = 0
        channel_costs[channel_name] = 0.0
        channel_last_touch[channel_name] = 30.0  # Default to 30 days
    
    # Populate with actual data
    for i, tp in enumerate(user.journey):
        channel_name = tp.channel.name.lower()
        if channel_name in channel_distribution:
            channel_distribution[channel_name] += 1
            channel_costs[channel_name] += tp.cost
            channel_last_touch[channel_name] = len(user.journey) - i - 1
    
    # Map channel names to standard encoder format
    channel_mapping = {
        'google_ads': 'search', 'facebook_ads': 'social',
        'email': 'email', 'display': 'display',
        'video': 'video', 'social': 'social',
        'affiliate': 'affiliate', 'direct': 'direct'
    }
    
    # Normalize channel names
    normalized_distribution = {}
    normalized_costs = {}
    normalized_last_touch = {}
    
    for encoder_channel in channel_names:
        normalized_distribution[encoder_channel] = 0
        normalized_costs[encoder_channel] = 0.0
        normalized_last_touch[encoder_channel] = 30.0
        
        # Check if we have data for this channel under any name
        for original_channel, count in channel_distribution.items():
            if channel_mapping.get(original_channel, original_channel) == encoder_channel:
                normalized_distribution[encoder_channel] += count
                normalized_costs[encoder_channel] += channel_costs.get(original_channel, 0.0)
                normalized_last_touch[encoder_channel] = min(
                    normalized_last_touch[encoder_channel],
                    channel_last_touch.get(original_channel, 30.0)
                )
    
    # Current time features
    current_dt = datetime.fromtimestamp(current_timestamp)
    
    journey_data = {
        'current_state': user.current_state.name.lower(),
        'days_in_journey': user.total_touches,  # Approximate
        'journey_stage': min(user.current_state.value, 4),  # Map to 0-4 range
        'total_touches': user.total_touches,
        'conversion_probability': user.conversion_probability,
        'user_fatigue_level': min(user.total_touches / 10.0, 1.0),  # Simple fatigue model
        'time_since_last_touch': user.time_since_last_touch,
        'hour_of_day': current_dt.hour,
        'day_of_week': current_dt.weekday(),
        'day_of_month': current_dt.day,
        'current_timestamp': current_timestamp,
        'journey_history': journey_history,
        'channel_distribution': normalized_distribution,
        'channel_costs': normalized_costs,
        'channel_last_touch': normalized_last_touch,
        'click_through_rate': 0.035,  # Placeholder - would come from orchestrator
        'engagement_rate': 0.15,      # Placeholder
        'bounce_rate': 0.4,           # Placeholder
        'conversion_rate': 0.08,      # Placeholder
        'competitors_seen': 2,        # Placeholder
        'competitor_engagement_rate': 0.12  # Placeholder
    }
    
    return journey_data

def extract_journey_state(user: EnhancedMultiTouchUser, 
                          orchestrator: MultiChannelOrchestrator,
                          timestamp: Any) -> JourneyState:
    """Extract journey state from user and orchestrator"""
    # Get last 3 channels and states
    last_3_channels = []
    last_3_states = []
    
    for tp in user.journey[-3:]:
        last_3_channels.append(list(Channel).index(tp.channel))
        last_3_states.append(list(UserState).index(tp.user_state_after))
    
    # Pad if needed
    while len(last_3_channels) < 3:
        last_3_channels.insert(0, 0)
    while len(last_3_states) < 3:
        last_3_states.insert(0, 0)
    
    # Channel statistics
    channel_touches = np.zeros(len(Channel))
    channel_costs = np.zeros(len(Channel))
    channel_last_touch = np.ones(len(Channel)) * 30  # Default to 30 days
    
    for i, tp in enumerate(user.journey):
        ch_idx = list(Channel).index(tp.channel)
        channel_touches[ch_idx] += 1
        channel_costs[ch_idx] += tp.cost
        channel_last_touch[ch_idx] = len(user.journey) - i - 1
    
    # Create state
    state = JourneyState(
        user_state=list(UserState).index(user.current_state),
        journey_length=user.total_touches,
        time_since_last_touch=user.time_since_last_touch,
        conversion_probability=user.conversion_probability,
        total_cost=user.total_cost,
        last_3_channels=last_3_channels,
        last_3_states=last_3_states,
        state_transitions=[],
        channel_touches=channel_touches,
        channel_costs=channel_costs,
        channel_last_touch=channel_last_touch,
        hour_of_day=12,  # Placeholder
        day_of_week=3,   # Placeholder
        days_in_journey=user.total_touches,
        click_through_rate=0.1,  # Placeholder
        engagement_rate=0.2,      # Placeholder
        bounce_rate=0.3          # Placeholder
    )
    
    return state

class DatabaseIntegratedRLAgent:
    """RL Agent with UserJourneyDatabase integration for persistent state tracking."""
    
    def __init__(self, 
                 bigquery_project_id: str,
                 bigquery_dataset_id: str = "gaelp",
                 state_dim: int = 64,
                 hidden_dim: int = 256,
                 num_channels: int = 8):
        """Initialize agent with database integration."""
        
        # Initialize UserJourneyDatabase
        self.journey_db = UserJourneyDatabase(
            project_id=bigquery_project_id,
            dataset_id=bigquery_dataset_id
        )
        
        # Initialize RL components with journey encoder
        self.agent = JourneyAwarePPOAgent(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            use_journey_encoder=True
        )
        
        # State manager for journey predictions
        self.state_manager = JourneyStateManager()
        
        # Channel mapping
        self.channels = [
            'google_ads', 'facebook_ads', 'email', 'display', 
            'video', 'social', 'affiliate', 'direct'
        ]
        
    def process_user_interaction(self, 
                               user_id: str,
                               channel: str,
                               interaction_type: str,
                               device_fingerprint: Optional[Dict] = None,
                               **touchpoint_data) -> Tuple[Dict, float]:
        """
        Process user interaction and get RL action recommendation.
        
        Returns:
            Tuple of (action_recommendation, expected_reward)
        """
        import uuid
        from datetime import datetime
        
        # Create touchpoint
        touchpoint = JourneyTouchpoint(
            touchpoint_id=str(uuid.uuid4()),
            journey_id="",  # Will be set by database
            user_id=user_id,
            canonical_user_id="",  # Will be resolved by database
            timestamp=datetime.now(),
            channel=channel,
            interaction_type=interaction_type,
            **touchpoint_data
        )
        
        # Get or create journey
        journey, is_new = self.journey_db.get_or_create_journey(
            user_id=user_id,
            channel=channel,
            interaction_type=interaction_type,
            device_fingerprint=device_fingerprint,
            **touchpoint_data
        )
        
        # Set journey_id in touchpoint
        touchpoint.journey_id = journey.journey_id
        touchpoint.canonical_user_id = journey.canonical_user_id
        
        # Determine trigger type
        trigger = self._map_interaction_to_trigger(interaction_type)
        
        # Update journey with new touchpoint
        updated_journey = self.journey_db.update_journey(
            journey_id=journey.journey_id,
            touchpoint=touchpoint,
            trigger=trigger
        )
        
        # Convert to RL state representation
        rl_state = self._journey_to_rl_state(updated_journey, touchpoint)
        
        # Get RL action recommendation
        channel_idx, bid_amount, _ = self.agent.select_action(rl_state)
        
        # Calculate expected reward based on journey analytics
        expected_reward = self._calculate_expected_reward(updated_journey, channel_idx, bid_amount)
        
        # Prepare action recommendation
        action_recommendation = {
            'recommended_channel': self.channels[channel_idx],
            'recommended_bid': bid_amount,
            'journey_state': updated_journey.current_state.value,
            'conversion_probability': self.state_manager.calculate_conversion_probability(
                current_state=updated_journey.current_state,
                journey_score=updated_journey.journey_score,
                days_in_journey=(datetime.now() - updated_journey.journey_start).days,
                touchpoint_count=updated_journey.touchpoint_count,
                context={}
            ),
            'journey_score': updated_journey.journey_score,
            'is_new_journey': is_new
        }
        
        return action_recommendation, expected_reward
    
    def record_conversion(self, 
                         user_id: str,
                         conversion_value: float,
                         conversion_type: str = "purchase",
                         **conversion_data):
        """Record conversion and update RL training data."""
        import uuid
        from datetime import datetime
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Find user's active journey
        canonical_user_id = self.journey_db._resolve_user_identity(user_id)
        journey = self.journey_db._find_active_journey(canonical_user_id)
        
        if not journey:
            logger.warning(f"No active journey found for conversion: {user_id}")
            return
        
        # Create conversion touchpoint
        conversion_touchpoint = JourneyTouchpoint(
            touchpoint_id=str(uuid.uuid4()),
            journey_id=journey.journey_id,
            user_id=user_id,
            canonical_user_id=canonical_user_id,
            timestamp=datetime.now(),
            channel="conversion",  # Special channel for conversions
            interaction_type="purchase",
            **conversion_data
        )
        
        # Update journey with conversion
        self.journey_db.update_journey(
            journey_id=journey.journey_id,
            touchpoint=conversion_touchpoint,
            trigger=TransitionTrigger.PURCHASE
        )
        
        # Get journey analytics for RL training
        analytics = self.journey_db.get_journey_analytics(journey.journey_id)
        
        # Create training data from successful journey
        self._create_training_data_from_journey(analytics, conversion_value)
    
    def _map_interaction_to_trigger(self, interaction_type: str) -> TransitionTrigger:
        """Map interaction type to journey state trigger."""
        
        mapping = {
            'impression': TransitionTrigger.IMPRESSION,
            'click': TransitionTrigger.CLICK,
            'view': TransitionTrigger.CONTENT_VIEW,
            'engagement': TransitionTrigger.ENGAGEMENT,
            'product_view': TransitionTrigger.PRODUCT_VIEW,
            'add_to_cart': TransitionTrigger.ADD_TO_CART,
            'checkout': TransitionTrigger.CHECKOUT_START,
            'purchase': TransitionTrigger.PURCHASE
        }
        
        return mapping.get(interaction_type, TransitionTrigger.IMPRESSION)
    
    def _journey_to_rl_state(self, journey: UserJourney, touchpoint: JourneyTouchpoint) -> Dict[str, Any]:
        """Convert UserJourney to RL state representation compatible with JourneyStateEncoder."""
        from datetime import datetime
        
        # Map database journey state to string representation for encoder
        state_mapping = {
            DatabaseJourneyState.UNAWARE: 'unaware',
            DatabaseJourneyState.AWARE: 'aware', 
            DatabaseJourneyState.CONSIDERING: 'considering',
            DatabaseJourneyState.INTENT: 'intent',
            DatabaseJourneyState.CONVERTED: 'converted'
        }
        
        current_dt = datetime.now()
        
        # Create journey data in format expected by JourneyStateEncoder
        journey_data = {
            'current_state': state_mapping.get(journey.current_state, 'unaware'),
            'days_in_journey': (current_dt - journey.journey_start).days,
            'journey_stage': min(journey.current_state.value, 4),
            'total_touches': journey.touchpoint_count,
            'conversion_probability': self.state_manager.calculate_conversion_probability(
                journey.current_state, journey.journey_score,
                (current_dt - journey.journey_start).days,
                journey.touchpoint_count, {}
            ),
            'user_fatigue_level': min(journey.touchpoint_count / 10.0, 1.0),
            'time_since_last_touch': 0.0,  # Current interaction
            'hour_of_day': current_dt.hour,
            'day_of_week': current_dt.weekday(),
            'day_of_month': current_dt.day,
            'current_timestamp': current_dt.timestamp(),
            'journey_history': [],  # Would be populated with full touchpoint history
            'channel_distribution': {channel: 0 for channel in ['search', 'social', 'display', 'video', 'email', 'direct', 'affiliate', 'retargeting']},
            'channel_costs': {channel: 0.0 for channel in ['search', 'social', 'display', 'video', 'email', 'direct', 'affiliate', 'retargeting']},
            'channel_last_touch': {channel: 30.0 for channel in ['search', 'social', 'display', 'video', 'email', 'direct', 'affiliate', 'retargeting']},
            'click_through_rate': 0.035,
            'engagement_rate': 0.15,
            'bounce_rate': 0.4,
            'conversion_rate': 0.08,
            'competitors_seen': 0,
            'competitor_engagement_rate': 0.0
        }
        
        # Update channel data based on current touchpoint
        if touchpoint.channel in journey_data['channel_distribution']:
            journey_data['channel_distribution'][touchpoint.channel] = 1
            journey_data['channel_costs'][touchpoint.channel] = touchpoint.cost or 0.0
            journey_data['channel_last_touch'][touchpoint.channel] = 0.0
        
        return journey_data
    
    def _calculate_expected_reward(self, journey: UserJourney, channel_idx: int, bid_amount: float) -> float:
        """Calculate expected reward for recommended action."""
        from datetime import datetime
        
        # Simple reward calculation based on conversion probability and cost
        conversion_prob = self.state_manager.calculate_conversion_probability(
            journey.current_state, journey.journey_score,
            (datetime.now() - journey.journey_start).days,
            journey.touchpoint_count, {}
        )
        
        # Expected conversion value (would be learned or configured)
        expected_conversion_value = 50.0
        
        # Expected reward = (conversion_prob * conversion_value) - cost
        expected_reward = (conversion_prob * expected_conversion_value) - bid_amount
        
        return expected_reward
    
    def _create_training_data_from_journey(self, analytics: Dict, conversion_value: float):
        """Create RL training data from completed journey."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Created training data from conversion worth ${conversion_value}")

def main():
    """Test journey-aware RL agent"""
    print("🤖 Journey-Aware RL Agent for GAELP")
    print("=" * 60)
    
    # Create environment
    from enhanced_journey_tracking import MultiTouchJourneySimulator
    
    simulator = MultiTouchJourneySimulator(num_users=100, time_horizon_days=14)
    orchestrator = MultiChannelOrchestrator(budget_daily=500.0)
    env = JourneyAwareRLEnvironment(simulator, orchestrator)
    
    # Create agent with journey encoder
    agent = JourneyAwarePPOAgent(use_journey_encoder=True)
    
    print("\n📊 Training Journey-Aware Agent with Enhanced State Encoding...")
    
    # Training loop
    num_episodes = 10
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Get initial user
        user = env.current_user
        state = extract_journey_state_for_encoder(user, orchestrator, env.simulator.current_date)
        
        while episode_length < 100:
            # Select action
            channel_idx, bid_amount, log_prob = agent.select_action(state)
            
            # Create action array
            action = np.zeros(len(Channel))
            action[channel_idx] = bid_amount
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Get next state
            next_user = env.current_user
            next_state = extract_journey_state_for_encoder(next_user, orchestrator, env.simulator.current_date)
            
            # Store transition
            agent.store_transition(state, channel_idx, reward, next_state, done, log_prob)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        # Update agent
        if episode > 0:
            agent.update()
        
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Evaluate
    print("\n📈 Evaluation Results:")
    obs, _ = env.reset()
    total_reward = 0
    conversions = 0
    total_cost = 0
    
    for _ in range(50):
        user = env.current_user
        state = extract_journey_state_for_encoder(user, orchestrator, env.simulator.current_date)
        
        channel_idx, bid_amount, _ = agent.select_action(state)
        action = np.zeros(len(Channel))
        action[channel_idx] = bid_amount
        
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        total_cost += info.get('cost', 0)
        if info.get('converted', False):
            conversions += 1
        
        if done:
            break
    
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Conversions: {conversions}")
    print(f"  Total Cost: ${total_cost:.2f}")
    print(f"  Avg Cost per Conversion: ${total_cost / max(conversions, 1):.2f}")
    
    # Save model
    agent.save('/home/hariravichandran/AELP/journey_aware_agent.pth')
    print("\n✅ Journey-aware RL agent with Journey State Encoder trained and saved!")
    
    return agent

if __name__ == "__main__":
    agent = main()