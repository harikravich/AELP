"""
TransformerWorldModel Integration for GAELP
Adds predictive capability for user behavior and market dynamics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class WorldModelConfig:
    """Configuration for world model."""
    d_model: int = 256  # Smaller for efficiency
    n_heads: int = 4
    n_layers: int = 3
    predict_horizon: int = 30  # Days ahead to predict
    use_diffusion: bool = False  # Simplified version without diffusion
    device: str = "cpu"  # Use CPU by default for compatibility


class SimpleTransformerWorldModel(nn.Module):
    """
    Simplified TransformerWorldModel that predicts:
    - User behavior patterns
    - Market dynamics
    - Conversion probabilities
    - Future trajectories
    
    This is a practical implementation that works with existing system.
    """
    
    def __init__(self, config: WorldModelConfig = None):
        super().__init__()
        self.config = config or WorldModelConfig()
        
        # Embedding dimensions
        self.state_dim = 128
        self.action_dim = 32
        self.d_model = self.config.d_model
        
        # Input projection
        self.state_projection = nn.Linear(self.state_dim, self.d_model)
        self.action_projection = nn.Linear(self.action_dim, self.d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.config.n_layers)
        
        # Prediction heads
        self.user_behavior_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 64)  # User state embedding
        )
        
        self.market_dynamics_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 32)  # Market state
        )
        
        self.conversion_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 2)  # Binary: convert/not convert
        )
        
        # Next state predictor
        self.next_state_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.state_dim)
        )
        
        self.to(self.config.device)
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        max_len = 1000
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                            -(np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def encode_state(self, state: Dict[str, Any]) -> torch.Tensor:
        """Encode environment state to tensor."""
        # Extract key features from state dict
        features = []
        
        # Market features
        features.append(state.get('budget_remaining', 1000) / 1000)
        features.append(state.get('time_remaining', 30) / 30)
        features.append(state.get('current_roi', 0))
        features.append(state.get('auction_position', 5) / 10)
        
        # User journey features
        features.append(state.get('user_engagement', 0))
        features.append(state.get('conversion_probability', 0))
        features.append(state.get('days_in_journey', 0) / 30)
        
        # Competition features
        features.append(state.get('competitor_strength', 0.5))
        features.append(state.get('market_saturation', 0.3))
        
        # Pad to state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.state_dim], dtype=torch.float32)
    
    def encode_action(self, action: Dict[str, Any]) -> torch.Tensor:
        """Encode action to tensor."""
        features = []
        
        # Bid features
        features.append(action.get('bid', 1.0))
        features.append(action.get('bid_modifier', 1.0))
        
        # Creative features
        features.append(float(action.get('creative_id', 0)))
        features.append(float(action.get('use_new_creative', False)))
        
        # Targeting features
        if 'target_segments' in action:
            # One-hot encode segments (simplified)
            segments = action['target_segments']
            features.extend([1.0 if seg in segments else 0.0 
                           for seg in ['high_intent', 'broad', 'narrow']])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Pad to action_dim
        while len(features) < self.action_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.action_dim], dtype=torch.float32)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through world model.
        
        Args:
            states: Batch of state sequences [batch, seq_len, state_dim]
            actions: Batch of action sequences [batch, seq_len, action_dim]
            
        Returns:
            Dictionary of predictions
        """
        batch_size, seq_len = states.shape[:2]
        
        # Project inputs
        state_emb = self.state_projection(states)
        action_emb = self.action_projection(actions)
        
        # Combine state and action
        combined = state_emb + action_emb  # Simple addition for combination
        
        # Add positional encoding
        combined = combined + self.positional_encoding[:, :seq_len, :].to(combined.device)
        
        # Pass through transformer
        hidden = self.transformer(combined)
        
        # Generate predictions from all heads
        predictions = {
            'user_behavior': self.user_behavior_head(hidden),
            'market_state': self.market_dynamics_head(hidden),
            'conversion_logits': self.conversion_head(hidden),
            'next_state': self.next_state_predictor(hidden)
        }
        
        return predictions
    
    def predict_trajectory(self, 
                          initial_state: Dict[str, Any],
                          policy,
                          horizon: int = None) -> List[Dict[str, Any]]:
        """
        Predict future trajectory using the world model.
        
        Args:
            initial_state: Starting state
            policy: Policy to use for action selection
            horizon: How many steps to predict
            
        Returns:
            List of predicted states
        """
        if horizon is None:
            horizon = self.config.predict_horizon
        
        trajectory = []
        state = self.encode_state(initial_state).unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
        
        for t in range(horizon):
            # Get action from policy
            with torch.no_grad():
                if hasattr(policy, 'get_action'):
                    action_dict = policy.get_action(initial_state)
                else:
                    # Simple random policy for testing
                    action_dict = {
                        'bid': random.uniform(0.5, 2.0),
                        'creative_id': random.randint(0, 10)
                    }
                
                action = self.encode_action(action_dict).unsqueeze(0).unsqueeze(0)
                
                # Predict next state
                predictions = self.forward(state, action)
                next_state = predictions['next_state'][:, -1, :]  # Take last timestep
                
                # Decode predictions
                user_behavior = predictions['user_behavior'][:, -1, :].squeeze().numpy()
                market_state = predictions['market_state'][:, -1, :].squeeze().numpy()
                conversion_prob = F.softmax(predictions['conversion_logits'][:, -1, :], dim=-1)
                conversion_prob = conversion_prob[0, 1].item()  # Probability of conversion
                
                # Store in trajectory
                trajectory.append({
                    'timestep': t,
                    'state': next_state.squeeze().numpy(),
                    'action': action_dict,
                    'user_behavior': user_behavior,
                    'market_state': market_state,
                    'conversion_probability': conversion_prob,
                    'predicted_roi': conversion_prob * 119.99 / max(0.01, action_dict['bid'])
                })
                
                # Update state for next iteration
                state = next_state.unsqueeze(0)
        
        return trajectory
    
    def train_on_experience(self, experience_buffer: List[Tuple]) -> float:
        """
        Train world model on collected experience.
        
        Args:
            experience_buffer: List of (state, action, next_state, reward) tuples
            
        Returns:
            Average loss
        """
        if len(experience_buffer) < 10:
            return 0.0
        
        # Sample batch
        batch_size = min(32, len(experience_buffer))
        batch = random.sample(experience_buffer, batch_size)
        
        # Prepare batch tensors
        states = []
        actions = []
        next_states = []
        rewards = []
        
        for state, action, next_state, reward in batch:
            states.append(self.encode_state(state))
            actions.append(self.encode_action(action))
            next_states.append(self.encode_state(next_state))
            rewards.append(reward)
        
        states = torch.stack(states).unsqueeze(1)  # Add seq dimension
        actions = torch.stack(actions).unsqueeze(1)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Forward pass
        predictions = self.forward(states, actions)
        
        # Calculate losses
        state_loss = F.mse_loss(predictions['next_state'].squeeze(1), next_states)
        
        # Simple conversion prediction based on reward
        conversion_targets = (rewards > 0).long()
        conversion_loss = F.cross_entropy(
            predictions['conversion_logits'].squeeze(1),
            conversion_targets
        )
        
        # Total loss
        total_loss = state_loss + 0.5 * conversion_loss
        
        # Backward pass (if training)
        if self.training:
            total_loss.backward()
        
        return total_loss.item()


class WorldModelOrchestrator:
    """
    Orchestrates world model integration with existing GAELP system.
    """
    
    def __init__(self, config: WorldModelConfig = None):
        self.config = config or WorldModelConfig()
        self.world_model = SimpleTransformerWorldModel(self.config)
        self.experience_buffer = []
        self.trajectory_cache = {}
        
        logger.info("✅ TransformerWorldModel initialized")
        logger.info(f"   - Prediction horizon: {self.config.predict_horizon} days")
        logger.info(f"   - Model size: {self.config.d_model}d, {self.config.n_heads}h, {self.config.n_layers}L")
    
    def imagine_campaign(self, 
                        initial_state: Dict[str, Any],
                        strategy: str = "balanced") -> Dict[str, Any]:
        """
        Imagine how a campaign would perform with given strategy.
        
        Args:
            initial_state: Starting conditions
            strategy: Campaign strategy (aggressive, conservative, balanced)
            
        Returns:
            Predicted campaign outcomes
        """
        # Create simple policy based on strategy
        class StrategyPolicy:
            def __init__(self, strategy):
                self.strategy = strategy
            
            def get_action(self, state):
                if self.strategy == "aggressive":
                    return {"bid": 2.0, "creative_id": 1, "target_segments": ["high_intent"]}
                elif self.strategy == "conservative":
                    return {"bid": 0.5, "creative_id": 2, "target_segments": ["broad"]}
                else:  # balanced
                    return {"bid": 1.0, "creative_id": 3, "target_segments": ["narrow"]}
        
        policy = StrategyPolicy(strategy)
        
        # Generate trajectory
        trajectory = self.world_model.predict_trajectory(initial_state, policy)
        
        # Analyze outcomes
        total_spend = sum(t['action']['bid'] for t in trajectory)
        expected_conversions = sum(t['conversion_probability'] for t in trajectory)
        expected_revenue = expected_conversions * 119.99
        predicted_roi = (expected_revenue / max(1, total_spend) - 1) * 100
        
        return {
            'strategy': strategy,
            'trajectory_length': len(trajectory),
            'total_spend': total_spend,
            'expected_conversions': expected_conversions,
            'expected_revenue': expected_revenue,
            'predicted_roi': predicted_roi,
            'trajectory': trajectory[:5]  # First 5 steps for inspection
        }
    
    def compare_strategies(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare different strategies using world model predictions.
        
        Args:
            initial_state: Starting conditions
            
        Returns:
            Comparison of strategy outcomes
        """
        strategies = ["aggressive", "conservative", "balanced"]
        comparisons = {}
        
        for strategy in strategies:
            comparisons[strategy] = self.imagine_campaign(initial_state, strategy)
        
        # Find best strategy
        best_strategy = max(comparisons.keys(), 
                          key=lambda s: comparisons[s]['predicted_roi'])
        
        return {
            'comparisons': comparisons,
            'recommended_strategy': best_strategy,
            'best_predicted_roi': comparisons[best_strategy]['predicted_roi']
        }
    
    def update_with_experience(self, state, action, next_state, reward):
        """
        Update world model with new experience.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            reward: Reward received
        """
        self.experience_buffer.append((state, action, next_state, reward))
        
        # Keep buffer bounded
        if len(self.experience_buffer) > 10000:
            self.experience_buffer.pop(0)
        
        # Periodically train
        if len(self.experience_buffer) % 100 == 0 and len(self.experience_buffer) > 0:
            loss = self.world_model.train_on_experience(self.experience_buffer)
            logger.debug(f"World model training loss: {loss:.4f}")
    
    def get_market_forecast(self, horizon: int = 7) -> Dict[str, Any]:
        """
        Get market forecast for next N days.
        
        Args:
            horizon: Days to forecast
            
        Returns:
            Market predictions
        """
        # Simple forecast based on recent experience
        if len(self.experience_buffer) < 10:
            return {
                'forecast_available': False,
                'reason': 'Insufficient data'
            }
        
        # Use last state as starting point
        last_state = self.experience_buffer[-1][0]
        
        # Generate forecast trajectory
        trajectory = self.world_model.predict_trajectory(
            last_state,
            policy=None,  # Use random policy
            horizon=horizon
        )
        
        # Extract market predictions
        market_states = [t['market_state'] for t in trajectory]
        conversion_probs = [t['conversion_probability'] for t in trajectory]
        
        return {
            'forecast_available': True,
            'horizon_days': horizon,
            'avg_conversion_probability': np.mean(conversion_probs),
            'conversion_trend': 'increasing' if conversion_probs[-1] > conversion_probs[0] else 'decreasing',
            'market_volatility': np.std([m[0] for m in market_states]),  # First dimension as proxy
            'confidence': min(len(self.experience_buffer) / 1000, 1.0)  # Based on data amount
        }


# Integration helper
def create_world_model(config: WorldModelConfig = None) -> WorldModelOrchestrator:
    """
    Create world model orchestrator for integration.
    
    Args:
        config: Optional configuration
        
    Returns:
        WorldModelOrchestrator instance
    """
    return WorldModelOrchestrator(config)