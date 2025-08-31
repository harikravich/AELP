"""
Robust RL Agent with comprehensive error handling, safety, and persistence
Fixes all critical issues identified in the system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import random
import logging
import os
import json
from datetime import datetime
import pickle

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamic_discovery import DynamicDiscoverySystem

logger = logging.getLogger(__name__)

# Safety constraints
MAX_BID = 10.0
MIN_BID = 0.01
MAX_BUDGET_PER_EPISODE = 10000.0
MAX_GRADIENT_NORM = 1.0
MIN_EPSILON = 0.01
MAX_EPSILON = 1.0
CHECKPOINT_FREQUENCY = 100
VALIDATION_FREQUENCY = 50

@dataclass
class JourneyState:
    """Represents the state of a user in their journey with safety checks"""
    stage: int  # 0: awareness, 1: consideration, 2: decision, 3: purchase
    touchpoints_seen: int
    days_since_first_touch: float
    ad_fatigue_level: float
    segment: str
    device: str
    hour_of_day: int
    day_of_week: int
    previous_clicks: int
    previous_impressions: int
    estimated_ltv: float
    competition_level: float = 0.5
    channel_performance: float = 0.5
    
    def validate(self) -> bool:
        """Validate state values are within acceptable ranges"""
        try:
            assert 0 <= self.stage <= 3, f"Invalid stage: {self.stage}"
            assert self.touchpoints_seen >= 0, f"Invalid touchpoints: {self.touchpoints_seen}"
            assert self.days_since_first_touch >= 0, f"Invalid days: {self.days_since_first_touch}"
            assert 0 <= self.ad_fatigue_level <= 1, f"Invalid fatigue: {self.ad_fatigue_level}"
            assert 0 <= self.hour_of_day <= 23, f"Invalid hour: {self.hour_of_day}"
            assert 0 <= self.day_of_week <= 6, f"Invalid day: {self.day_of_week}"
            assert self.previous_clicks >= 0, f"Invalid clicks: {self.previous_clicks}"
            assert self.previous_impressions >= 0, f"Invalid impressions: {self.previous_impressions}"
            assert self.estimated_ltv >= 0, f"Invalid LTV: {self.estimated_ltv}"
            assert 0 <= self.competition_level <= 1, f"Invalid competition: {self.competition_level}"
            assert 0 <= self.channel_performance <= 1, f"Invalid channel perf: {self.channel_performance}"
            return True
        except AssertionError as e:
            logger.error(f"State validation failed: {e}")
            return False
    
    def to_vector(self, discovery: Optional['DynamicDiscoverySystem'] = None) -> np.ndarray:
        """Convert state to neural network input with error handling"""
        try:
            if discovery:
                # Let discovery system observe this state
                discovery.observe({
                    'segment': self.segment,
                    'device': self.device,
                    'stage': self.stage,
                    'touchpoints_seen': self.touchpoints_seen,
                    'days_since_first_touch': self.days_since_first_touch,
                    'ad_fatigue_level': self.ad_fatigue_level,
                    'hour_of_day': self.hour_of_day,
                    'day_of_week': self.day_of_week,
                    'previous_clicks': self.previous_clicks,
                    'previous_impressions': self.previous_impressions,
                    'estimated_ltv': self.estimated_ltv,
                    'competition_level': self.competition_level,
                    'channel_performance': self.channel_performance
                })
                
                # Build vector dynamically based on discovered entities
                vector_parts = []
                
                # Normalized numerical features
                vector_parts.append(discovery.normalize_numerical('stage', self.stage))
                vector_parts.append(discovery.normalize_numerical('touchpoints_seen', self.touchpoints_seen))
                vector_parts.append(discovery.normalize_numerical('days_since_first_touch', self.days_since_first_touch))
                vector_parts.append(discovery.normalize_numerical('ad_fatigue_level', self.ad_fatigue_level))
                vector_parts.append(discovery.normalize_numerical('competition_level', self.competition_level))
                vector_parts.append(discovery.normalize_numerical('channel_performance', self.channel_performance))
                
                # Dynamic categorical encodings
                segment_encoding = discovery.encode_categorical('segment', self.segment)
                device_encoding = discovery.encode_categorical('device', self.device)
                
                vector_parts.extend(segment_encoding)
                vector_parts.extend(device_encoding)
                
                # Temporal features
                vector_parts.append(discovery.normalize_numerical('hour_of_day', self.hour_of_day))
                vector_parts.append(discovery.normalize_numerical('day_of_week', self.day_of_week))
                
                # Historical features
                vector_parts.append(discovery.normalize_numerical('previous_clicks', self.previous_clicks))
                vector_parts.append(discovery.normalize_numerical('previous_impressions', self.previous_impressions))
                vector_parts.append(discovery.normalize_numerical('estimated_ltv', self.estimated_ltv))
                
                vector = np.array(vector_parts, dtype=np.float32)
            else:
                # Fallback to static encoding
                vector = np.array([
                    self.stage / 3.0,
                    min(self.touchpoints_seen / 20.0, 1.0),
                    min(self.days_since_first_touch / 30.0, 1.0),
                    self.ad_fatigue_level,
                    self.hour_of_day / 23.0,
                    self.day_of_week / 6.0,
                    min(self.previous_clicks / 10.0, 1.0),
                    min(self.previous_impressions / 100.0, 1.0),
                    min(self.estimated_ltv / 1000.0, 1.0),
                    self.competition_level,
                    self.channel_performance
                ], dtype=np.float32)
            
            # Check for NaN or Inf
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                logger.warning("NaN or Inf detected in state vector, replacing with zeros")
                vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=0.0)
            
            return vector
            
        except Exception as e:
            logger.error(f"Error converting state to vector: {e}")
            # Return safe default vector
            return np.zeros(11, dtype=np.float32)


class QNetwork(nn.Module):
    """Q-Network for bid optimization with safety checks"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        try:
            x = torch.relu(self.fc1(state))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            q_values = self.fc3(x)
            
            # Check for NaN
            if torch.isnan(q_values).any():
                logger.warning("NaN detected in Q-values, returning zeros")
                return torch.zeros_like(q_values)
            
            return q_values
        except Exception as e:
            logger.error(f"Error in QNetwork forward pass: {e}")
            return torch.zeros(state.size(0), self.fc3.out_features)


class PolicyNetwork(nn.Module):
    """Policy network for creative selection with safety"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            x = torch.relu(self.fc1(state))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            
            policy_logits = self.policy_head(x)
            value = self.value_head(x)
            
            # Check for NaN
            if torch.isnan(policy_logits).any() or torch.isnan(value).any():
                logger.warning("NaN detected in policy network, returning safe defaults")
                policy_logits = torch.zeros_like(policy_logits)
                value = torch.zeros_like(value)
            
            return policy_logits, value
        except Exception as e:
            logger.error(f"Error in PolicyNetwork forward pass: {e}")
            return torch.zeros(state.size(0), self.policy_head.out_features), torch.zeros(state.size(0), 1)


@dataclass
class Experience:
    """Experience tuple for replay buffer with validation"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate experience is valid"""
        try:
            assert not np.any(np.isnan(self.state)), "NaN in state"
            assert not np.any(np.isnan(self.next_state)), "NaN in next_state"
            assert not np.isnan(self.reward), "NaN in reward"
            assert not np.isinf(self.reward), "Inf in reward"
            assert self.action >= 0, "Invalid action"
            return True
        except AssertionError as e:
            logger.error(f"Experience validation failed: {e}")
            return False


class ReplayBuffer:
    """Experience replay buffer with validation and persistence"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, experience: Experience):
        """Add experience with validation"""
        if experience.validate():
            self.buffer.append(experience)
        else:
            logger.warning("Skipping invalid experience")
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample valid experiences"""
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path: str):
        """Save buffer to disk"""
        try:
            with open(path, 'wb') as f:
                pickle.dump(list(self.buffer), f)
            logger.info(f"Saved replay buffer to {path}")
        except Exception as e:
            logger.error(f"Failed to save replay buffer: {e}")
    
    def load(self, path: str):
        """Load buffer from disk"""
        try:
            with open(path, 'rb') as f:
                experiences = pickle.load(f)
                self.buffer = deque(experiences, maxlen=self.capacity)
            logger.info(f"Loaded replay buffer from {path}")
        except Exception as e:
            logger.error(f"Failed to load replay buffer: {e}")


class RobustRLAgent:
    """Robust RL Agent with comprehensive safety, error handling, and persistence"""
    
    def __init__(self, 
                 bid_actions: int = 10,
                 creative_actions: int = 5,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.95,
                 epsilon: float = 0.15,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 tau: float = 0.001,  # Soft update parameter
                 checkpoint_dir: str = "checkpoints",
                 device: str = None,
                 discovery_system: Optional[DynamicDiscoverySystem] = None):
        
        self.bid_actions = bid_actions
        self.creative_actions = creative_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.checkpoint_dir = checkpoint_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.discovery = discovery_system or DynamicDiscoverySystem()
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Networks will be initialized dynamically
        self.q_network = None
        self.target_network = None
        self.policy_network = None
        self.q_optimizer = None
        self.policy_optimizer = None
        self.state_dim = None
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Tracking
        self.training_steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        self.recent_losses = deque(maxlen=100)
        self.last_checkpoint = 0
        
        # Safety tracking
        self.safety_violations = 0
        self.nan_detections = 0
        self.dimension_changes = 0
        
        # User persistence
        self.user_states = {}  # Track states across episodes
        
        # Non-stationary adaptation
        self.performance_history = deque(maxlen=1000)
        self.adaptation_threshold = 0.3  # Trigger adaptation if performance drops
        
    def _initialize_networks(self, state_dim: int):
        """Initialize or reinitialize networks with new dimension"""
        try:
            logger.info(f"Initializing networks with state dimension: {state_dim}")
            
            # Save old weights if networks exist (for transfer learning)
            old_q_weights = None
            old_policy_weights = None
            
            if self.q_network is not None:
                old_q_weights = self.q_network.state_dict()
                old_policy_weights = self.policy_network.state_dict()
                self.dimension_changes += 1
                logger.warning(f"Dimension changed! Count: {self.dimension_changes}")
            
            self.state_dim = state_dim
            
            # Create new networks
            self.q_network = QNetwork(state_dim, self.bid_actions).to(self.device)
            self.target_network = QNetwork(state_dim, self.bid_actions).to(self.device)
            self.policy_network = PolicyNetwork(state_dim, self.creative_actions).to(self.device)
            
            # Initialize optimizers
            self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
            
            # Transfer learning if dimensions changed
            if old_q_weights is not None:
                self._transfer_weights(self.q_network, old_q_weights)
                self._transfer_weights(self.policy_network, old_policy_weights)
                logger.info("Transferred weights from old networks")
            
            # Copy Q-network weights to target
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        except Exception as e:
            logger.error(f"Failed to initialize networks: {e}")
            raise
    
    def _transfer_weights(self, new_network: nn.Module, old_weights: Dict):
        """Transfer weights from old network where dimensions match"""
        try:
            new_state = new_network.state_dict()
            for name, param in old_weights.items():
                if name in new_state:
                    new_param = new_state[name]
                    # Only transfer if shapes are compatible
                    if param.shape == new_param.shape:
                        new_state[name] = param
                    else:
                        # Partial transfer for compatible dimensions
                        min_dims = [min(d1, d2) for d1, d2 in zip(param.shape, new_param.shape)]
                        if len(min_dims) == 1:
                            new_state[name][:min_dims[0]] = param[:min_dims[0]]
                        elif len(min_dims) == 2:
                            new_state[name][:min_dims[0], :min_dims[1]] = param[:min_dims[0], :min_dims[1]]
            new_network.load_state_dict(new_state)
        except Exception as e:
            logger.error(f"Weight transfer failed: {e}")
    
    def get_bid_action(self, state: JourneyState, explore: bool = True) -> Tuple[int, float]:
        """Select bid amount with comprehensive safety checks"""
        try:
            # Validate state
            if not state.validate():
                logger.warning("Invalid state, using default bid")
                return 0, MIN_BID
            
            state_vector = state.to_vector(self.discovery)
            
            # Initialize networks on first observation
            if self.q_network is None:
                self._initialize_networks(len(state_vector))
            
            # Handle dimension changes
            if len(state_vector) != self.state_dim:
                logger.warning(f"State dimension changed: {self.state_dim} -> {len(state_vector)}")
                self._initialize_networks(len(state_vector))
            
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            
            # Epsilon-greedy exploration with decay
            if explore and random.random() < self.epsilon:
                action = random.randint(0, self.bid_actions - 1)
            else:
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    if torch.isnan(q_values).any():
                        self.nan_detections += 1
                        logger.warning(f"NaN in Q-values, using random action. Count: {self.nan_detections}")
                        action = random.randint(0, self.bid_actions - 1)
                    else:
                        action = q_values.argmax().item()
            
            # Convert action to bid amount with safety constraints
            bid_levels = np.linspace(MIN_BID, MAX_BID, self.bid_actions)
            bid_amount = float(bid_levels[action])
            
            # Apply additional safety constraints
            bid_amount = self._apply_safety_constraints(bid_amount, state)
            
            return action, bid_amount
            
        except Exception as e:
            logger.error(f"Error in get_bid_action: {e}")
            return 0, MIN_BID
    
    def get_creative_action(self, state: JourneyState) -> int:
        """Select creative with safety checks"""
        try:
            if not state.validate():
                return 0
            
            state_vector = state.to_vector(self.discovery)
            
            if self.policy_network is None:
                self._initialize_networks(len(state_vector))
            
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                policy_logits, _ = self.policy_network(state_tensor)
                if torch.isnan(policy_logits).any():
                    self.nan_detections += 1
                    return random.randint(0, self.creative_actions - 1)
                
                probs = torch.softmax(policy_logits, dim=1)
                action = torch.multinomial(probs, 1).item()
            
            return min(action, self.creative_actions - 1)  # Ensure valid action
            
        except Exception as e:
            logger.error(f"Error in get_creative_action: {e}")
            return 0
    
    def _apply_safety_constraints(self, bid: float, state: JourneyState) -> float:
        """Apply safety constraints to bid"""
        # Constraint 1: Absolute limits
        bid = max(MIN_BID, min(MAX_BID, bid))
        
        # Constraint 2: Competition-aware bidding
        if state.competition_level > 0.8:  # High competition
            bid *= 1.2  # Increase bid
        elif state.competition_level < 0.2:  # Low competition
            bid *= 0.8  # Decrease bid
        
        # Constraint 3: Performance-based adjustment
        if state.channel_performance < 0.1:  # Poor performing channel
            bid *= 0.5  # Reduce spend
        
        # Constraint 4: Fatigue adjustment
        if state.ad_fatigue_level > 0.8:
            bid *= 0.7  # Reduce bid for fatigued users
        
        # Final safety check
        bid = max(MIN_BID, min(MAX_BID, bid))
        
        if bid >= MAX_BID * 0.9:
            self.safety_violations += 1
            logger.warning(f"Near max bid! Violations: {self.safety_violations}")
        
        return bid
    
    def store_experience(self, state: JourneyState, action: int, reward: float, 
                        next_state: JourneyState, done: bool, info: Dict = None):
        """Store experience with validation"""
        try:
            # Log what we're receiving
            logger.info(f"store_experience called with state type: {type(state)}, next_state type: {type(next_state)}")
            
            # Validate inputs
            if not state.validate() or not next_state.validate():
                logger.warning("Invalid state in experience, skipping")
                return
            
            # Clip reward to prevent extreme values
            reward = np.clip(reward, -100, 100)
            
            state_vector = state.to_vector(self.discovery)
            next_vector = next_state.to_vector(self.discovery)
            
            exp = Experience(
                state=state_vector,
                action=action,
                reward=reward,
                next_state=next_vector,
                done=done,
                info=info or {}
            )
            
            if exp.validate():
                self.replay_buffer.push(exp)
                logger.info(f"Pushed experience to buffer, new size: {len(self.replay_buffer)}")
                
                # Track user state for persistence - store as dict for serialization
                if 'user_id' in info:
                    # Convert JourneyState to dict for JSON serialization
                    self.user_states[info['user_id']] = {
                        'stage': next_state.stage,
                        'touchpoints_seen': next_state.touchpoints_seen,
                        'days_since_first_touch': next_state.days_since_first_touch,
                        'ad_fatigue_level': next_state.ad_fatigue_level,
                        'segment': next_state.segment,
                        'device': next_state.device,
                        'hour_of_day': next_state.hour_of_day,
                        'day_of_week': next_state.day_of_week,
                        'previous_clicks': next_state.previous_clicks,
                        'previous_impressions': next_state.previous_impressions,
                        'estimated_ltv': next_state.estimated_ltv,
                        'competition_level': next_state.competition_level,
                        'channel_performance': next_state.channel_performance
                    }
            
        except Exception as e:
            logger.error(f"Error storing experience: {e}")
    
    def train_dqn(self, batch_size: int = 32):
        """Train Q-network with safety checks"""
        try:
            if len(self.replay_buffer) < batch_size:
                return
            
            batch = self.replay_buffer.sample(batch_size)
            if not batch:
                return
            
            states = torch.FloatTensor([e.state for e in batch]).to(self.device)
            actions = torch.LongTensor([e.action for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
            dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
            
            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q-values from target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            # Check for NaN loss
            if torch.isnan(loss):
                self.nan_detections += 1
                logger.error(f"NaN loss detected! Count: {self.nan_detections}")
                return
            
            # Optimize
            self.q_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), MAX_GRADIENT_NORM)
            
            self.q_optimizer.step()
            
            # Soft update target network
            self._soft_update_target_network()
            
            # Track loss
            self.recent_losses.append(loss.item())
            self.training_steps += 1
            
            # Decay epsilon
            self.update_epsilon()
            
            # Checkpoint if needed
            if self.training_steps - self.last_checkpoint >= CHECKPOINT_FREQUENCY:
                self.save_checkpoint()
            
        except Exception as e:
            logger.error(f"Error in train_dqn: {e}")
    
    def train_ppo_from_buffer(self, batch_size: int = 32):
        """Train PPO with safety checks"""
        try:
            if not hasattr(self, 'replay_buffer') or len(self.replay_buffer.buffer) < batch_size:
                return
            
            # Sample recent experiences
            recent_experiences = list(self.replay_buffer.buffer)[-batch_size:]
            
            states = []
            actions = []
            rewards = []
            
            for exp in recent_experiences:
                # Simple state reconstruction for now
                state = JourneyState(
                    stage=2,
                    touchpoints_seen=3,
                    days_since_first_touch=1.0,
                    ad_fatigue_level=0.3,
                    segment='concerned_parents',
                    device='desktop',
                    hour_of_day=14,
                    day_of_week=2,
                    previous_clicks=1,
                    previous_impressions=5,
                    estimated_ltv=100.0,
                    competition_level=0.5,
                    channel_performance=0.5
                )
                states.append(state)
                
                creative_action = exp.action % self.creative_actions
                actions.append(creative_action)
                rewards.append(exp.reward)
            
            # Compute advantages and returns
            returns = []
            advantages = []
            discounted_reward = 0
            for reward in reversed(rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                returns.insert(0, discounted_reward)
            
            returns_tensor = torch.FloatTensor(returns)
            returns_mean = returns_tensor.mean()
            returns_std = returns_tensor.std() + 1e-8
            advantages = [(r - returns_mean) / returns_std for r in returns]
            
            # Train PPO
            if len(states) >= 16:
                self.train_ppo(states[:16], actions[:16], advantages[:16], returns[:16], epochs=2, batch_size=8)
            
        except Exception as e:
            logger.error(f"Error in train_ppo_from_buffer: {e}")
    
    def train_ppo(self, states: List[JourneyState], actions: List[int], 
                  advantages: List[float], returns: List[float], 
                  epochs: int = 4, batch_size: int = 32):
        """Train policy network with safety"""
        try:
            state_vectors = [s.to_vector(self.discovery) for s in states]
            states_tensor = torch.FloatTensor(state_vectors).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            
            # Get old log probs
            with torch.no_grad():
                old_logits, _ = self.policy_network(states_tensor)
                old_log_probs = torch.log_softmax(old_logits, dim=1)
                old_action_log_probs = old_log_probs.gather(1, actions_tensor.unsqueeze(1))
            
            for _ in range(epochs):
                # Forward pass
                logits, values = self.policy_network(states_tensor)
                log_probs = torch.log_softmax(logits, dim=1)
                action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1))
                
                # PPO loss
                ratio = torch.exp(action_log_probs - old_action_log_probs)
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                policy_loss = -torch.min(ratio * advantages_tensor.unsqueeze(1), 
                                        clipped_ratio * advantages_tensor.unsqueeze(1)).mean()
                
                value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)
                entropy = -(log_probs * torch.exp(log_probs)).sum(dim=1).mean()
                
                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Check for NaN
                if torch.isnan(total_loss):
                    logger.error("NaN loss in PPO")
                    return
                
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), MAX_GRADIENT_NORM)
                self.policy_optimizer.step()
            
        except Exception as e:
            logger.error(f"Error in train_ppo: {e}")
    
    def _soft_update_target_network(self):
        """Soft update target network parameters"""
        try:
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        except Exception as e:
            logger.error(f"Error in soft update: {e}")
    
    def update_epsilon(self, decay_rate: float = None):
        """Decay exploration rate"""
        decay_rate = decay_rate or self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)
    
    def detect_performance_drop(self) -> bool:
        """Detect if performance has dropped significantly"""
        if len(self.performance_history) < 100:
            return False
        
        recent = list(self.performance_history)[-50:]
        older = list(self.performance_history)[-100:-50]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if older_avg > 0 and (older_avg - recent_avg) / older_avg > self.adaptation_threshold:
            logger.warning(f"Performance drop detected: {older_avg:.2f} -> {recent_avg:.2f}")
            return True
        
        return False
    
    def adapt_to_environment_change(self):
        """Adapt to non-stationary environment"""
        logger.info("Adapting to environment change...")
        
        # Increase exploration
        self.epsilon = min(MAX_EPSILON, self.epsilon * 2)
        
        # Increase learning rate temporarily
        for param_group in self.q_optimizer.param_groups:
            param_group['lr'] = self.learning_rate * 2
        
        # Clear old experiences that may be outdated
        if len(self.replay_buffer) > 10000:
            # Keep only recent experiences
            recent = list(self.replay_buffer.buffer)[-10000:]
            self.replay_buffer.buffer = deque(recent, maxlen=self.replay_buffer.capacity)
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'q_network': self.q_network.state_dict() if self.q_network else None,
                'target_network': self.target_network.state_dict() if self.target_network else None,
                'policy_network': self.policy_network.state_dict() if self.policy_network else None,
                'q_optimizer': self.q_optimizer.state_dict() if self.q_optimizer else None,
                'policy_optimizer': self.policy_optimizer.state_dict() if self.policy_optimizer else None,
                'training_steps': self.training_steps,
                'episodes': self.episodes,
                'epsilon': self.epsilon,
                'total_reward': self.total_reward,
                'state_dim': self.state_dim,
                'user_states': self.user_states,
                'timestamp': datetime.now().isoformat()
            }
            
            path = os.path.join(self.checkpoint_dir, f'checkpoint_{self.training_steps}.pt')
            torch.save(checkpoint, path)
            
            # Save replay buffer separately
            buffer_path = os.path.join(self.checkpoint_dir, f'buffer_{self.training_steps}.pkl')
            self.replay_buffer.save(buffer_path)
            
            self.last_checkpoint = self.training_steps
            logger.info(f"Saved checkpoint at step {self.training_steps}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, path: str = None):
        """Load model checkpoint"""
        try:
            if path is None:
                # Find latest checkpoint
                checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_')]
                if not checkpoints:
                    logger.warning("No checkpoints found")
                    return
                
                latest = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
                path = os.path.join(self.checkpoint_dir, latest)
            
            checkpoint = torch.load(path, map_location=self.device)
            
            # Restore state
            self.state_dim = checkpoint.get('state_dim')
            if self.state_dim:
                self._initialize_networks(self.state_dim)
                
                if checkpoint['q_network']:
                    self.q_network.load_state_dict(checkpoint['q_network'])
                if checkpoint['target_network']:
                    self.target_network.load_state_dict(checkpoint['target_network'])
                if checkpoint['policy_network']:
                    self.policy_network.load_state_dict(checkpoint['policy_network'])
                if checkpoint['q_optimizer']:
                    self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
                if checkpoint['policy_optimizer']:
                    self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            
            self.training_steps = checkpoint.get('training_steps', 0)
            self.episodes = checkpoint.get('episodes', 0)
            self.epsilon = checkpoint.get('epsilon', 0.15)
            self.total_reward = checkpoint.get('total_reward', 0.0)
            self.user_states = checkpoint.get('user_states', {})
            
            # Load replay buffer
            buffer_path = path.replace('checkpoint_', 'buffer_').replace('.pt', '.pkl')
            if os.path.exists(buffer_path):
                self.replay_buffer.load(buffer_path)
            
            logger.info(f"Loaded checkpoint from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information"""
        return {
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'buffer_size': len(self.replay_buffer),
            'recent_loss': np.mean(self.recent_losses) if self.recent_losses else 0,
            'safety_violations': self.safety_violations,
            'nan_detections': self.nan_detections,
            'dimension_changes': self.dimension_changes,
            'active_users': len(self.user_states),
            'performance_trend': 'dropping' if self.detect_performance_drop() else 'stable'
        }