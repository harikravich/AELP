"""
Realistic RL Agent for GAELP
Uses ONLY data available from real ad platforms
No fantasy user tracking or mental states
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)

@dataclass
class RealisticState:
    """State based on ACTUAL observable data from ad platforms"""
    
    # Time context (REAL)
    hour_of_day: int  # 0-23
    day_of_week: int  # 0-6
    
    # Platform context (REAL)
    platform: str  # google, facebook, tiktok
    
    # Campaign performance - YOUR data only (REAL)
    campaign_ctr: float  # Your historical CTR
    campaign_cvr: float  # Your historical CVR
    campaign_cpc: float  # Your average CPC
    
    # Recent performance - last hour (REAL)
    recent_impressions: int
    recent_clicks: int
    recent_spend: float
    recent_conversions: int  # Within attribution window
    
    # Budget state (REAL)
    budget_remaining_pct: float  # % of daily budget left
    hours_remaining: int  # Hours left in day
    pace_vs_target: float  # Are we over/under pacing?
    
    # Market signals - inferred, not direct (REAL)
    avg_position: float  # Google only, 0 for others
    win_rate: float  # % of auctions won
    price_pressure: float  # CPCs vs historical
    
    # NO FANTASY DATA:
    # - No user journey stage (can't know)
    # - No user intent score (can't measure)
    # - No touchpoint history (can't track cross-platform)
    # - No competitor bids (never see these)
    
    def to_vector(self) -> np.ndarray:
        """Convert to neural network input"""
        # One-hot encode platform
        platform_vec = [0, 0, 0]
        if self.platform == 'google':
            platform_vec[0] = 1
        elif self.platform == 'facebook':
            platform_vec[1] = 1
        else:  # tiktok
            platform_vec[2] = 1
        
        # Normalize continuous features
        return np.array([
            # Time features (cyclical encoding)
            np.sin(2 * np.pi * self.hour_of_day / 24),
            np.cos(2 * np.pi * self.hour_of_day / 24),
            np.sin(2 * np.pi * self.day_of_week / 7),
            np.cos(2 * np.pi * self.day_of_week / 7),
            
            # Platform
            *platform_vec,
            
            # Performance (normalized)
            min(1.0, self.campaign_ctr * 20),  # CTR usually 0-5%
            min(1.0, self.campaign_cvr * 50),  # CVR usually 0-2%
            min(1.0, self.campaign_cpc / 10),  # CPC usually $0-10
            
            # Recent performance
            min(1.0, self.recent_impressions / 100),
            min(1.0, self.recent_clicks / 10),
            min(1.0, self.recent_spend / 100),
            min(1.0, self.recent_conversions / 5),
            
            # Budget state
            self.budget_remaining_pct,
            self.hours_remaining / 24,
            np.clip(self.pace_vs_target, -1, 1),
            
            # Market signals
            self.avg_position / 4 if self.avg_position > 0 else 0,
            self.win_rate,
            np.clip(self.price_pressure, 0, 2)
        ], dtype=np.float32)


class RealisticDQN(nn.Module):
    """Deep Q-Network for bid optimization using REAL features only"""
    
    def __init__(self, state_dim: int = 20, action_dim: int = 20):
        super().__init__()
        
        # Smaller network since we have fewer features
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, action_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.output(x)


class RealisticRLAgent:
    """
    RL Agent that learns from REAL ad platform data only
    No fantasy user tracking or competitor visibility
    """
    
    def __init__(self, 
                 bid_range: Tuple[float, float] = (0.5, 10.0),
                 num_bid_actions: int = 20,
                 learning_rate: float = 0.0001):
        
        self.bid_range = bid_range
        self.num_bid_actions = num_bid_actions
        
        # Create bid action space
        self.bid_values = np.linspace(bid_range[0], bid_range[1], num_bid_actions)
        
        # Initialize Q-network
        self.q_network = RealisticDQN(state_dim=20, action_dim=num_bid_actions)
        self.target_network = RealisticDQN(state_dim=20, action_dim=num_bid_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Training tracking
        self.training_step = 0
        self.update_target_every = 100
        
        # Performance tracking (REAL metrics only)
        self.performance_history = {
            'ctr': deque(maxlen=100),
            'cvr': deque(maxlen=100),
            'cpc': deque(maxlen=100),
            'roas': deque(maxlen=100)
        }
        
        logger.info("Initialized REALISTIC RL Agent with observable state space")
    
    def get_action(self, state: RealisticState, explore: bool = True) -> Dict[str, Any]:
        """
        Select action based on REAL observable state
        
        Returns:
            Dictionary with bid and other action parameters
        """
        state_vector = state.to_vector()
        
        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            bid_idx = random.randrange(self.num_bid_actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state_vector).unsqueeze(0))
                bid_idx = q_values.argmax(1).item()
        
        bid_value = self.bid_values[bid_idx]
        
        # Adjust bid based on context (using REAL signals)
        bid_multiplier = 1.0
        
        # Time-based adjustments
        if state.hour_of_day in [22, 23, 0, 1, 2]:
            bid_multiplier *= 1.3  # Late night crisis searches
        elif state.hour_of_day in [3, 4, 5, 6]:
            bid_multiplier *= 0.7  # Early morning low activity
        
        # Platform adjustments
        if state.platform == 'google':
            bid_multiplier *= 1.1  # Higher bids for search intent
        elif state.platform == 'tiktok':
            bid_multiplier *= 0.8  # Lower bids for discovery
        
        # Performance adjustments
        if state.campaign_cvr > 0.02:  # Good conversion rate
            bid_multiplier *= 1.2
        elif state.campaign_cvr < 0.005:  # Poor conversion rate
            bid_multiplier *= 0.8
        
        # Budget pacing adjustments
        if state.pace_vs_target > 0.2:  # Over-pacing
            bid_multiplier *= 0.9
        elif state.pace_vs_target < -0.2:  # Under-pacing
            bid_multiplier *= 1.1
        
        final_bid = bid_value * bid_multiplier
        
        # Select creative and audience based on platform and time
        creative = self._select_creative(state)
        audience = self._select_audience(state)
        
        return {
            'bid': final_bid,
            'bid_idx': bid_idx,
            'creative': creative,
            'audience': audience,
            'platform': state.platform
        }
    
    def _select_creative(self, state: RealisticState) -> str:
        """Select creative based on context (REAL patterns)"""
        
        if state.platform == 'google':
            if state.hour_of_day in [22, 23, 0, 1, 2]:
                return 'crisis_help'  # "Get Help Now"
            elif state.recent_clicks > 5:
                return 'social_proof'  # "50,000 Parents Trust Us"
            else:
                return 'benefit_focused'  # "Monitor Mental Health"
        
        elif state.platform == 'facebook':
            if state.day_of_week in [5, 6]:  # Weekend
                return 'lifestyle'  # Family-focused creative
            else:
                return 'problem_aware'  # "Is Your Teen OK?"
        
        else:  # TikTok
            return 'native_style'  # Looks like user content
    
    def _select_audience(self, state: RealisticState) -> str:
        """Select audience targeting based on performance"""
        
        if state.platform == 'facebook':
            if state.campaign_cvr > 0.01:
                return 'lookalike_1pct'  # Tighter targeting
            else:
                return 'interest_parents'  # Broader
        
        elif state.platform == 'google':
            return 'in_market_parents'  # Google's in-market audiences
        
        else:
            return 'broad_parents_25_45'
    
    def store_experience(self, state: RealisticState, action_idx: int, 
                        reward: float, next_state: RealisticState, done: bool):
        """Store experience for replay learning"""
        
        self.memory.append((
            state.to_vector(),
            action_idx,
            reward,
            next_state.to_vector(),
            done
        ))
    
    def train(self):
        """Train Q-network on batch of experiences"""
        
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def calculate_reward(self, state: RealisticState, action: Dict[str, Any],
                        result: Dict[str, Any]) -> float:
        """
        Calculate reward based on REAL metrics only
        
        This is what actually drives learning!
        """
        reward = 0.0
        
        # Impression reward (small, just for feedback)
        if result.get('won', False):
            reward += 0.01
            
            # Click reward (meaningful signal)
            if result.get('clicked', False):
                ctr_efficiency = 1.0 / max(0.01, result.get('price_paid', 1.0))
                reward += ctr_efficiency * 0.1
            
            # Conversion reward (big but delayed)
            if result.get('converted', False):
                roas = result.get('conversion_value', 0) / max(0.01, result.get('price_paid', 1.0))
                reward += roas * 0.5
            
            # Cost penalty
            reward -= result.get('price_paid', 0) / 100
        
        # Win rate bonus (learning to win auctions)
        if state.win_rate < 0.3 and result.get('won', False):
            reward += 0.05  # Bonus for winning when win rate is low
        
        # Budget pacing bonus
        if abs(state.pace_vs_target) < 0.1:
            reward += 0.02  # Bonus for good pacing
        
        return reward
    
    def get_insights(self) -> Dict[str, Any]:
        """Get learnings and insights (REAL patterns only)"""
        
        insights = {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_steps': self.training_step,
            'performance_trends': {}
        }
        
        # Calculate performance trends
        if self.performance_history['ctr']:
            insights['performance_trends']['ctr_improvement'] = (
                np.mean(list(self.performance_history['ctr'])[-10:]) - 
                np.mean(list(self.performance_history['ctr'])[:10:])
            )
        
        if self.performance_history['roas']:
            insights['performance_trends']['roas_trend'] = (
                np.mean(list(self.performance_history['roas'])[-10:]) /
                max(0.01, np.mean(list(self.performance_history['roas'])[:10:]))
            )
        
        return insights
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        logger.info(f"Model loaded from {path}")


# Example usage showing REAL learning
if __name__ == "__main__":
    agent = RealisticRLAgent()
    
    # Example state from REAL ad platform data
    state = RealisticState(
        hour_of_day=22,  # 10 PM
        day_of_week=2,  # Wednesday
        platform='google',
        campaign_ctr=0.03,  # 3% CTR
        campaign_cvr=0.02,  # 2% CVR
        campaign_cpc=3.50,
        recent_impressions=45,
        recent_clicks=2,
        recent_spend=7.00,
        recent_conversions=0,
        budget_remaining_pct=0.6,
        hours_remaining=2,
        pace_vs_target=-0.1,  # Under-pacing slightly
        avg_position=2.3,
        win_rate=0.35,
        price_pressure=1.1  # Prices 10% higher than normal
    )
    
    # Get action from agent
    action = agent.get_action(state)
    print(f"Action: Bid ${action['bid']:.2f} with {action['creative']} creative")
    
    # This would come from the ad platform
    result = {
        'won': True,
        'clicked': True,
        'price_paid': 3.45,
        'converted': False  # Will be delayed
    }
    
    # Calculate reward
    reward = agent.calculate_reward(state, action, result)
    print(f"Reward: {reward:.4f}")
    
    # Agent learns from REAL patterns, not fantasy!