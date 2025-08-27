#!/usr/bin/env python3
"""
CompetitorAgents System for GAELP - Learning Competitors for Realistic Ad Auction Simulation

This module implements various types of competitor agents that learn and adapt their bidding
strategies based on market dynamics and losses. Each competitor represents a different
real-world ad platform with unique characteristics and pricing models.

Competitors:
- Qustodio: $99/year, aggressive Q-learning agent
- Bark: $144/year, premium policy gradient agent  
- Circle: $129/year, defensive rule-based agent
- Norton: Baseline random agent

Each agent learns from auction losses and adapts strategies to improve performance metrics
like win_rate, avg_position, and spend_efficiency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import random
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of competitor agents"""
    Q_LEARNING = "q_learning"
    POLICY_GRADIENT = "policy_gradient"
    RULE_BASED = "rule_based"
    RANDOM = "random"


class UserValueTier(Enum):
    """User value tiers for strategic bidding"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"


@dataclass
class AuctionContext:
    """Context information for an auction"""
    user_id: str
    user_value_tier: UserValueTier
    timestamp: datetime
    device_type: str
    geo_location: str
    time_of_day: int  # 0-23
    day_of_week: int  # 0-6
    market_competition: float  # 0-1, estimated competition level
    keyword_competition: float  # 0-1, keyword-specific competition
    seasonality_factor: float  # 0-2, seasonal multiplier
    user_engagement_score: float  # 0-1, predicted engagement
    conversion_probability: float  # 0-1, predicted conversion


@dataclass
class AuctionResult:
    """Result of an auction participation"""
    won: bool
    bid_amount: float
    winning_price: float
    position: int  # 1-indexed, 1 being best
    competitor_count: int
    user_value_tier: UserValueTier
    cost_per_click: float
    revenue: float = 0.0  # If conversion occurred
    converted: bool = False


@dataclass
class AgentMetrics:
    """Performance metrics for competitor agents"""
    win_rate: float = 0.0
    avg_position: float = 0.0
    spend_efficiency: float = 0.0  # Revenue / Cost
    total_spend: float = 0.0
    total_revenue: float = 0.0
    total_auctions: int = 0
    conversions: int = 0
    conversion_rate: float = 0.0
    roas: float = 0.0  # Return on Ad Spend
    high_value_wins: int = 0  # Wins on high-value users
    cost_per_acquisition: float = 0.0


class BaseCompetitorAgent(ABC):
    """Base class for all competitor agents"""
    
    def __init__(self, name: str, annual_budget: float, agent_type: AgentType):
        self.name = name
        self.annual_budget = annual_budget
        self.daily_budget = annual_budget / 365
        self.agent_type = agent_type
        
        # Performance tracking
        self.metrics = AgentMetrics()
        self.auction_history: List[AuctionResult] = []
        self.learning_history: List[Dict[str, Any]] = []
        
        # Strategy parameters
        self.base_bid_multiplier = 1.0
        self.aggression_level = 0.5  # 0-1, how aggressive the bidding is
        self.risk_tolerance = 0.5  # 0-1, tolerance for risky bids
        self.learning_rate = 0.01
        
        # Budget management
        self.daily_spend = 0.0
        self.current_date = datetime.now().date()
        
        # Loss tracking for adaptation
        self.recent_losses = deque(maxlen=100)  # Track recent auction losses
        self.high_value_losses = deque(maxlen=50)  # Track losses on high-value users
        
        logger.info(f"Initialized {self.name} agent with ${annual_budget} annual budget")
    
    @abstractmethod
    def calculate_bid(self, context: AuctionContext) -> float:
        """Calculate bid amount for given auction context"""
        pass
    
    @abstractmethod
    def update_strategy(self, result: AuctionResult, context: AuctionContext):
        """Update bidding strategy based on auction result"""
        pass
    
    @abstractmethod
    def learn_from_losses(self):
        """Analyze recent losses and adapt strategy"""
        pass
    
    def should_participate(self, context: AuctionContext) -> bool:
        """Decide whether to participate in auction based on budget and strategy"""
        # Check daily budget
        if self.daily_spend >= self.daily_budget:
            return False
        
        # Reset daily spend if new day
        if context.timestamp.date() > self.current_date:
            self.daily_spend = 0.0
            self.current_date = context.timestamp.date()
        
        # Strategic participation decision
        if context.user_value_tier == UserValueTier.LOW and self.daily_spend > 0.8 * self.daily_budget:
            return False
        
        return True
    
    def record_auction(self, result: AuctionResult, context: AuctionContext):
        """Record auction result and update metrics"""
        self.auction_history.append(result)
        self.daily_spend += result.cost_per_click if result.won else 0
        
        # Update metrics
        self._update_metrics()
        
        # Track losses for learning
        if not result.won:
            self.recent_losses.append((result, context))
            if context.user_value_tier in [UserValueTier.HIGH, UserValueTier.PREMIUM]:
                self.high_value_losses.append((result, context))
        
        # Update strategy based on result
        self.update_strategy(result, context)
        
        # Periodically learn from losses
        if len(self.auction_history) % 10 == 0:
            self.learn_from_losses()
    
    def _update_metrics(self):
        """Update performance metrics based on auction history"""
        if not self.auction_history:
            return
        
        recent_results = self.auction_history[-100:]  # Last 100 auctions
        
        wins = [r for r in recent_results if r.won]
        
        self.metrics.total_auctions = len(recent_results)
        self.metrics.win_rate = len(wins) / len(recent_results) if recent_results else 0
        
        if wins:
            self.metrics.avg_position = np.mean([r.position for r in wins])
            self.metrics.total_spend = sum([r.cost_per_click for r in wins])
            self.metrics.total_revenue = sum([r.revenue for r in wins])
            self.metrics.conversions = sum([1 for r in wins if r.converted])
            self.metrics.conversion_rate = self.metrics.conversions / len(wins)
            
            if self.metrics.total_spend > 0:
                self.metrics.spend_efficiency = self.metrics.total_revenue / self.metrics.total_spend
                self.metrics.roas = self.metrics.total_revenue / self.metrics.total_spend
            
            if self.metrics.conversions > 0:
                self.metrics.cost_per_acquisition = self.metrics.total_spend / self.metrics.conversions
            
            self.metrics.high_value_wins = len([r for r in wins if r.user_value_tier in [UserValueTier.HIGH, UserValueTier.PREMIUM]])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'name': self.name,
            'agent_type': self.agent_type.value,
            'annual_budget': self.annual_budget,
            'metrics': {
                'win_rate': round(self.metrics.win_rate, 3),
                'avg_position': round(self.metrics.avg_position, 2),
                'spend_efficiency': round(self.metrics.spend_efficiency, 3),
                'total_spend': round(self.metrics.total_spend, 2),
                'total_revenue': round(self.metrics.total_revenue, 2),
                'roas': round(self.metrics.roas, 3),
                'conversion_rate': round(self.metrics.conversion_rate, 3),
                'cost_per_acquisition': round(self.metrics.cost_per_acquisition, 2),
                'high_value_wins': self.metrics.high_value_wins,
                'total_auctions': self.metrics.total_auctions
            },
            'strategy_params': {
                'base_bid_multiplier': round(self.base_bid_multiplier, 3),
                'aggression_level': round(self.aggression_level, 3),
                'risk_tolerance': round(self.risk_tolerance, 3)
            }
        }


class QLearningAgent(BaseCompetitorAgent):
    """Q-Learning agent representing Qustodio ($99/year, aggressive)"""
    
    def __init__(self):
        super().__init__("Qustodio", 99.0, AgentType.Q_LEARNING)
        
        # Q-learning parameters
        self.epsilon = 0.3  # Exploration rate
        self.gamma = 0.95   # Discount factor
        self.alpha = 0.1    # Learning rate
        
        # State and action discretization
        self.bid_actions = np.linspace(0.1, 3.0, 20)  # Possible bid multipliers
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # More aggressive by default
        self.aggression_level = 0.8
        self.risk_tolerance = 0.7
        
        logger.info(f"Initialized Q-Learning agent {self.name} with aggressive profile")
    
    def _get_state(self, context: AuctionContext) -> str:
        """Convert auction context to discrete state"""
        # Discretize continuous features
        competition = "low" if context.market_competition < 0.3 else "med" if context.market_competition < 0.7 else "high"
        tier = context.user_value_tier.value
        time_bucket = "morning" if context.time_of_day < 12 else "afternoon" if context.time_of_day < 18 else "evening"
        
        return f"{tier}_{competition}_{time_bucket}"
    
    def calculate_bid(self, context: AuctionContext) -> float:
        """Calculate bid using Q-learning policy with epsilon-greedy exploration"""
        if not self.should_participate(context):
            return 0.0
        
        state = self._get_state(context)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: random action
            bid_multiplier = random.choice(self.bid_actions)
        else:
            # Exploit: best Q-value action
            q_values = [self.q_table[state][str(action)] for action in self.bid_actions]
            best_action_idx = np.argmax(q_values)
            bid_multiplier = self.bid_actions[best_action_idx]
        
        # Base bid calculation
        base_bid = self._calculate_base_bid(context)
        
        # Apply Q-learning multiplier and aggression
        final_bid = base_bid * bid_multiplier * (1.0 + self.aggression_level)
        
        return max(0.1, final_bid)
    
    def _calculate_base_bid(self, context: AuctionContext) -> float:
        """Calculate base bid based on user value and context"""
        # User tier multipliers
        tier_multipliers = {
            UserValueTier.LOW: 0.5,
            UserValueTier.MEDIUM: 1.0,
            UserValueTier.HIGH: 2.0,
            UserValueTier.PREMIUM: 3.5
        }
        
        base_value = tier_multipliers[context.user_value_tier]
        
        # Adjust for context factors
        base_value *= context.conversion_probability
        base_value *= context.seasonality_factor
        base_value *= (1.0 + context.user_engagement_score)
        
        return base_value
    
    def update_strategy(self, result: AuctionResult, context: AuctionContext):
        """Update Q-table based on auction result"""
        state = self._get_state(context)
        action = str(min(self.bid_actions, key=lambda x: abs(x - (result.bid_amount / self._calculate_base_bid(context)))))
        
        # Calculate reward
        if result.won:
            if result.converted:
                reward = result.revenue - result.cost_per_click
            else:
                reward = -result.cost_per_click * 0.5  # Penalty for non-converting wins
        else:
            # Penalty for losing, especially on high-value users
            penalty = 0.1 if result.user_value_tier == UserValueTier.LOW else 0.5
            reward = -penalty
        
        # Update Q-value
        current_q = self.q_table[state][action]
        
        # Simplified next state value (assuming similar future states)
        next_state_value = max([self.q_table[state][str(a)] for a in self.bid_actions]) if self.q_table[state] else 0
        
        new_q = current_q + self.alpha * (reward + self.gamma * next_state_value - current_q)
        self.q_table[state][action] = new_q
        
        # Decay epsilon over time
        self.epsilon = max(0.1, self.epsilon * 0.9999)
    
    def learn_from_losses(self):
        """Analyze losses and increase aggression on high-value users"""
        if len(self.high_value_losses) < 5:
            return
        
        # Analyze recent high-value losses
        recent_high_value_losses = list(self.high_value_losses)[-10:]
        high_value_loss_rate = len(recent_high_value_losses) / min(10, len(self.high_value_losses))
        
        # Increase aggression if losing too many high-value auctions
        if high_value_loss_rate > 0.7:
            self.aggression_level = min(1.0, self.aggression_level * 1.1)
            self.learning_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'increased_aggression',
                'reason': 'high_value_losses',
                'new_aggression': self.aggression_level,
                'loss_rate': high_value_loss_rate
            })
            
            logger.info(f"{self.name}: Increased aggression to {self.aggression_level:.3f} due to high-value losses")


class PolicyGradientAgent(BaseCompetitorAgent):
    """Policy Gradient agent representing Bark ($144/year, premium)"""
    
    def __init__(self):
        super().__init__("Bark", 144.0, AgentType.POLICY_GRADIENT)
        
        # Neural network for policy
        self.input_dim = 10  # Number of context features
        self.hidden_dim = 64
        self.output_dim = 1  # Bid multiplier
        
        self.policy_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.rewards = []
        self.log_probs = []
        
        # Premium positioning - more selective but higher bids
        self.aggression_level = 0.6
        self.risk_tolerance = 0.4
        self.quality_threshold = 0.7  # Only bid on high-quality opportunities
        
        logger.info(f"Initialized Policy Gradient agent {self.name} with premium profile")
    
    def _context_to_features(self, context: AuctionContext) -> torch.Tensor:
        """Convert auction context to feature vector"""
        # User tier encoding
        tier_encoding = {
            UserValueTier.LOW: 0.25,
            UserValueTier.MEDIUM: 0.5,
            UserValueTier.HIGH: 0.75,
            UserValueTier.PREMIUM: 1.0
        }
        
        features = [
            tier_encoding[context.user_value_tier],
            context.market_competition,
            context.keyword_competition,
            context.seasonality_factor,
            context.user_engagement_score,
            context.conversion_probability,
            context.time_of_day / 24.0,  # Normalize
            context.day_of_week / 7.0,   # Normalize
            1.0 if context.device_type == 'mobile' else 0.0,
            min(1.0, self.daily_spend / self.daily_budget)  # Budget utilization
        ]
        
        return torch.FloatTensor(features)
    
    def calculate_bid(self, context: AuctionContext) -> float:
        """Calculate bid using policy gradient network"""
        if not self.should_participate(context):
            return 0.0
        
        # Premium strategy: only bid on high-quality opportunities
        quality_score = (context.user_engagement_score + context.conversion_probability) / 2
        if quality_score < self.quality_threshold:
            return 0.0
        
        # Get features and run through policy network
        features = self._context_to_features(context)
        
        with torch.no_grad():
            bid_multiplier = self.policy_net(features).item()
        
        # Scale multiplier to reasonable range (0.5 to 3.0)
        bid_multiplier = 0.5 + bid_multiplier * 2.5
        
        # Base bid calculation
        base_bid = self._calculate_base_bid(context)
        
        # Apply premium positioning strategy
        premium_multiplier = 1.5 if context.user_value_tier in [UserValueTier.HIGH, UserValueTier.PREMIUM] else 1.0
        
        final_bid = base_bid * bid_multiplier * premium_multiplier
        
        return max(0.1, final_bid)
    
    def _calculate_base_bid(self, context: AuctionContext) -> float:
        """Calculate base bid with premium focus"""
        # Higher base values for premium agent
        tier_multipliers = {
            UserValueTier.LOW: 0.3,  # Lower than other agents
            UserValueTier.MEDIUM: 1.2,
            UserValueTier.HIGH: 2.5,
            UserValueTier.PREMIUM: 4.0
        }
        
        base_value = tier_multipliers[context.user_value_tier]
        
        # Premium adjustments
        base_value *= context.conversion_probability * 1.2
        base_value *= context.seasonality_factor
        base_value *= (1.0 + context.user_engagement_score * 1.5)
        
        return base_value
    
    def update_strategy(self, result: AuctionResult, context: AuctionContext):
        """Update policy network using REINFORCE algorithm"""
        # Calculate reward
        if result.won:
            if result.converted:
                # High reward for conversions
                reward = (result.revenue - result.cost_per_click) * 2.0
            else:
                # Penalty for non-converting wins (premium agent wants quality)
                reward = -result.cost_per_click
        else:
            # Small penalty for losing, larger for high-value users
            if result.user_value_tier in [UserValueTier.HIGH, UserValueTier.PREMIUM]:
                reward = -1.0
            else:
                reward = -0.2
        
        self.rewards.append(reward)
        
        # For policy gradient, we need to store the log probability
        # Simplified version - in practice, we'd store this during forward pass
        features = self._context_to_features(context)
        bid_multiplier = self.policy_net(features)
        
        # Approximate log probability (for demonstration)
        log_prob = -((result.bid_amount / self._calculate_base_bid(context) - bid_multiplier) ** 2)
        self.log_probs.append(log_prob)
        
        # Update policy every 10 steps
        if len(self.rewards) >= 10:
            self._update_policy()
    
    def _update_policy(self):
        """Update policy network using accumulated rewards"""
        if len(self.rewards) == 0:
            return
        
        # Calculate discounted rewards
        discounted_rewards = []
        cumulative = 0
        for reward in reversed(self.rewards):
            cumulative = reward + 0.95 * cumulative
            discounted_rewards.insert(0, cumulative)
        
        # Normalize rewards
        rewards_tensor = torch.FloatTensor(discounted_rewards)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Policy gradient update
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, rewards_tensor):
            policy_loss.append(-log_prob * reward)
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.rewards.clear()
        self.log_probs.clear()
    
    def learn_from_losses(self):
        """Analyze losses and adjust quality threshold"""
        if len(self.recent_losses) < 10:
            return
        
        recent_results = [r for r, c in self.recent_losses[-20:]]
        loss_rate = len(recent_results) / 20
        
        # Adjust quality threshold based on loss rate
        if loss_rate > 0.8:  # Too selective, missing opportunities
            self.quality_threshold = max(0.5, self.quality_threshold * 0.95)
            action = 'decreased_selectivity'
        elif loss_rate < 0.4:  # Maybe too aggressive, increase selectivity
            self.quality_threshold = min(0.9, self.quality_threshold * 1.05)
            action = 'increased_selectivity'
        else:
            return  # No adjustment needed
        
        self.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'reason': 'loss_rate_adjustment',
            'new_quality_threshold': self.quality_threshold,
            'loss_rate': loss_rate
        })
        
        logger.info(f"{self.name}: Adjusted quality threshold to {self.quality_threshold:.3f} (loss rate: {loss_rate:.3f})")


class RuleBasedAgent(BaseCompetitorAgent):
    """Rule-based agent representing Circle ($129/year, defensive)"""
    
    def __init__(self):
        super().__init__("Circle", 129.0, AgentType.RULE_BASED)
        
        # Rule-based parameters
        self.rules = {
            'max_bid_multiplier': 2.0,
            'min_bid_multiplier': 0.8,
            'budget_protection_threshold': 0.8,  # Protect 20% of daily budget
            'peak_hours': [9, 10, 11, 14, 15, 16],  # Peak bidding hours
            'preferred_tiers': [UserValueTier.MEDIUM, UserValueTier.HIGH],
            'competition_threshold': 0.6,  # Avoid highly competitive auctions
            'win_rate_target': 0.35,  # Target win rate
            'position_target': 2.5   # Target average position
        }
        
        # Defensive positioning
        self.aggression_level = 0.3
        self.risk_tolerance = 0.3
        
        # Adaptive rule weights
        self.rule_weights = {
            'budget_conservation': 1.0,
            'time_targeting': 0.8,
            'tier_preference': 1.2,
            'competition_avoidance': 1.0
        }
        
        logger.info(f"Initialized Rule-Based agent {self.name} with defensive profile")
    
    def calculate_bid(self, context: AuctionContext) -> float:
        """Calculate bid using rule-based logic"""
        if not self.should_participate(context):
            return 0.0
        
        # Rule 1: Budget protection
        budget_utilization = self.daily_spend / self.daily_budget
        if budget_utilization > self.rules['budget_protection_threshold']:
            # Only bid on premium users when budget is tight
            if context.user_value_tier != UserValueTier.PREMIUM:
                return 0.0
        
        # Rule 2: Avoid highly competitive auctions
        if context.market_competition > self.rules['competition_threshold']:
            return 0.0
        
        # Rule 3: Time-based bidding
        time_multiplier = 1.2 if context.time_of_day in self.rules['peak_hours'] else 0.8
        
        # Rule 4: User tier preferences
        if context.user_value_tier not in self.rules['preferred_tiers'] and context.user_value_tier != UserValueTier.PREMIUM:
            return 0.0
        
        # Base bid calculation
        base_bid = self._calculate_base_bid(context)
        
        # Apply rule-based multipliers
        bid_multiplier = self.rules['min_bid_multiplier']
        
        # Adjust based on current performance vs targets
        if self.metrics.win_rate < self.rules['win_rate_target']:
            bid_multiplier *= 1.1  # Increase bids if win rate is low
        
        if self.metrics.avg_position > self.rules['position_target']:
            bid_multiplier *= 1.05  # Increase bids if position is poor
        
        # Apply time multiplier
        bid_multiplier *= time_multiplier
        
        # Conservative cap
        bid_multiplier = min(bid_multiplier, self.rules['max_bid_multiplier'])
        
        final_bid = base_bid * bid_multiplier
        
        return max(0.1, final_bid)
    
    def _calculate_base_bid(self, context: AuctionContext) -> float:
        """Calculate base bid with defensive approach"""
        # Conservative tier multipliers
        tier_multipliers = {
            UserValueTier.LOW: 0.4,
            UserValueTier.MEDIUM: 1.0,
            UserValueTier.HIGH: 1.8,
            UserValueTier.PREMIUM: 2.5
        }
        
        base_value = tier_multipliers[context.user_value_tier]
        
        # Conservative adjustments
        base_value *= context.conversion_probability * 0.9  # Slightly pessimistic
        base_value *= min(context.seasonality_factor, 1.5)  # Cap seasonal boost
        base_value *= (1.0 + context.user_engagement_score * 0.8)
        
        return base_value
    
    def update_strategy(self, result: AuctionResult, context: AuctionContext):
        """Update rules based on performance"""
        # Simple rule adaptation based on results
        if result.won and result.converted:
            # Successful conversion - reinforce similar contexts
            if context.time_of_day not in self.rules['peak_hours'] and random.random() < 0.1:
                self.rules['peak_hours'].append(context.time_of_day)
        
        # Adjust competition threshold based on losses
        if not result.won and context.market_competition > 0.5:
            self.rules['competition_threshold'] = max(0.3, self.rules['competition_threshold'] * 0.99)
    
    def learn_from_losses(self):
        """Adapt rules based on loss patterns"""
        if len(self.recent_losses) < 15:
            return
        
        # Analyze loss patterns
        recent_loss_contexts = [context for result, context in self.recent_losses[-15:]]
        
        # Check if losses are concentrated in specific hours
        loss_hours = [c.time_of_day for c in recent_loss_contexts]
        hour_counts = defaultdict(int)
        for hour in loss_hours:
            hour_counts[hour] += 1
        
        # Remove hours with high loss rates from peak hours
        for hour, count in hour_counts.items():
            if count >= 3 and hour in self.rules['peak_hours']:
                self.rules['peak_hours'].remove(hour)
                self.learning_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'removed_peak_hour',
                    'hour': hour,
                    'loss_count': count
                })
        
        # Adjust competition threshold if losing too much
        avg_competition = np.mean([c.market_competition for c in recent_loss_contexts])
        if avg_competition > 0.6:
            self.rules['competition_threshold'] = max(0.2, self.rules['competition_threshold'] * 0.95)
            self.learning_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'lowered_competition_threshold',
                'new_threshold': self.rules['competition_threshold'],
                'avg_loss_competition': avg_competition
            })
        
        logger.info(f"{self.name}: Adapted rules based on recent losses")


class RandomAgent(BaseCompetitorAgent):
    """Random baseline agent representing Norton (baseline)"""
    
    def __init__(self):
        super().__init__("Norton", 100.0, AgentType.RANDOM)  # Assumed budget for baseline
        
        # Random parameters
        self.bid_range = (0.5, 2.5)  # Min and max bid multipliers
        self.participation_rate = 0.7  # 70% participation rate
        
        # Baseline characteristics
        self.aggression_level = 0.5
        self.risk_tolerance = 0.5
        
        logger.info(f"Initialized Random agent {self.name} as baseline")
    
    def calculate_bid(self, context: AuctionContext) -> float:
        """Calculate random bid within range"""
        if not self.should_participate(context):
            return 0.0
        
        # Random participation decision
        if random.random() > self.participation_rate:
            return 0.0
        
        # Random bid multiplier
        bid_multiplier = random.uniform(*self.bid_range)
        
        # Base bid calculation
        base_bid = self._calculate_base_bid(context)
        
        final_bid = base_bid * bid_multiplier
        
        return max(0.1, final_bid)
    
    def _calculate_base_bid(self, context: AuctionContext) -> float:
        """Simple base bid calculation"""
        tier_multipliers = {
            UserValueTier.LOW: 0.5,
            UserValueTier.MEDIUM: 1.0,
            UserValueTier.HIGH: 1.5,
            UserValueTier.PREMIUM: 2.0
        }
        
        return tier_multipliers[context.user_value_tier] * context.conversion_probability
    
    def update_strategy(self, result: AuctionResult, context: AuctionContext):
        """Random agent doesn't learn - maintains baseline behavior"""
        pass
    
    def learn_from_losses(self):
        """Random agent doesn't adapt based on losses"""
        pass


class CompetitorAgentManager:
    """Manages all competitor agents and orchestrates auction simulations"""
    
    def __init__(self):
        self.agents = {
            'qustodio': QLearningAgent(),
            'bark': PolicyGradientAgent(),
            'circle': RuleBasedAgent(),
            'norton': RandomAgent()
        }
        
        self.simulation_history = []
        self.market_conditions = {
            'base_competition': 0.5,
            'seasonal_factors': self._generate_seasonal_factors(),
            'user_distribution': {
                UserValueTier.LOW: 0.4,
                UserValueTier.MEDIUM: 0.35,
                UserValueTier.HIGH: 0.2,
                UserValueTier.PREMIUM: 0.05
            }
        }
        
        logger.info("Initialized CompetitorAgentManager with 4 agents")
    
    def _generate_seasonal_factors(self) -> Dict[int, float]:
        """Generate seasonal factors for different months"""
        # Higher factors during holiday seasons
        factors = {
            1: 0.8,   # January - post-holiday low
            2: 0.9,   # February
            3: 1.1,   # March - spring
            4: 1.2,   # April
            5: 1.0,   # May
            6: 0.9,   # June
            7: 0.8,   # July - summer low
            8: 0.9,   # August
            9: 1.1,   # September - back to school
            10: 1.2,  # October
            11: 1.5,  # November - Black Friday
            12: 1.8   # December - holidays
        }
        return factors
    
    def generate_auction_context(self, timestamp: datetime = None) -> AuctionContext:
        """Generate realistic auction context"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Sample user value tier based on distribution
        tier_values = list(self.market_conditions['user_distribution'].keys())
        tier_weights = list(self.market_conditions['user_distribution'].values())
        user_tier = np.random.choice(tier_values, p=tier_weights)
        
        # Generate context features
        context = AuctionContext(
            user_id=f"user_{random.randint(1000, 9999)}",
            user_value_tier=user_tier,
            timestamp=timestamp,
            device_type=random.choice(['mobile', 'desktop', 'tablet']),
            geo_location=random.choice(['US', 'CA', 'UK', 'AU', 'DE']),
            time_of_day=timestamp.hour,
            day_of_week=timestamp.weekday(),
            market_competition=np.random.beta(2, 2),  # 0-1, tends toward middle
            keyword_competition=np.random.beta(1.5, 2),  # Slightly lower competition
            seasonality_factor=self.market_conditions['seasonal_factors'].get(timestamp.month, 1.0),
            user_engagement_score=np.random.beta(2, 3),  # Slightly lower engagement
            conversion_probability=self._calculate_conversion_probability(user_tier)
        )
        
        return context
    
    def _calculate_conversion_probability(self, user_tier: UserValueTier) -> float:
        """Calculate conversion probability based on user tier"""
        base_rates = {
            UserValueTier.LOW: 0.02,
            UserValueTier.MEDIUM: 0.05,
            UserValueTier.HIGH: 0.12,
            UserValueTier.PREMIUM: 0.25
        }
        
        # Add some randomness
        base_rate = base_rates[user_tier]
        return max(0.001, np.random.normal(base_rate, base_rate * 0.3))
    
    def run_auction(self, context: AuctionContext) -> Dict[str, AuctionResult]:
        """Run a single auction with all participating agents"""
        # Get bids from all agents
        bids = {}
        for name, agent in self.agents.items():
            bid = agent.calculate_bid(context)
            if bid > 0:
                bids[name] = bid
        
        if not bids:
            # No participants
            return {}
        
        # Determine auction outcome
        results = {}
        
        # Sort agents by bid (descending)
        sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
        
        # Simple first-price auction mechanics
        for i, (agent_name, bid) in enumerate(sorted_bids):
            position = i + 1
            won = position == 1  # Winner takes all for simplicity
            
            # Calculate cost per click (second-price logic)
            if won and len(sorted_bids) > 1:
                winning_price = sorted_bids[1][1]  # Second highest bid
            else:
                winning_price = bid * 0.8  # Reserve price
            
            # Simulate conversion
            converted = False
            revenue = 0.0
            if won:
                converted = np.random.random() < context.conversion_probability
                if converted:
                    # Revenue based on user tier and random factors
                    tier_revenues = {
                        UserValueTier.LOW: np.random.normal(10, 3),
                        UserValueTier.MEDIUM: np.random.normal(25, 8),
                        UserValueTier.HIGH: np.random.normal(60, 15),
                        UserValueTier.PREMIUM: np.random.normal(150, 30)
                    }
                    revenue = max(0, tier_revenues[context.user_value_tier])
            
            result = AuctionResult(
                won=won,
                bid_amount=bid,
                winning_price=winning_price if won else 0,
                position=position,
                competitor_count=len(bids),
                user_value_tier=context.user_value_tier,
                cost_per_click=winning_price if won else 0,
                revenue=revenue,
                converted=converted
            )
            
            results[agent_name] = result
            
            # Update agent with result
            self.agents[agent_name].record_auction(result, context)
        
        return results
    
    def run_simulation(self, num_auctions: int = 1000, days: int = 30) -> Dict[str, Any]:
        """Run comprehensive simulation with multiple auctions over time"""
        logger.info(f"Starting simulation: {num_auctions} auctions over {days} days")
        
        start_date = datetime.now()
        auctions_per_day = num_auctions // days
        
        simulation_results = []
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            for auction_idx in range(auctions_per_day):
                # Vary auction times throughout the day
                hour = int(np.random.beta(2, 2) * 24)  # More activity during middle hours
                auction_time = current_date.replace(hour=hour, minute=random.randint(0, 59))
                
                # Generate auction context
                context = self.generate_auction_context(auction_time)
                
                # Run auction
                results = self.run_auction(context)
                
                if results:
                    auction_data = {
                        'timestamp': auction_time.isoformat(),
                        'context': context.__dict__,
                        'results': {name: result.__dict__ for name, result in results.items()}
                    }
                    simulation_results.append(auction_data)
        
        self.simulation_history.extend(simulation_results)
        
        # Generate comprehensive report
        report = self.generate_simulation_report()
        
        logger.info(f"Simulation completed. Processed {len(simulation_results)} auctions")
        
        return report
    
    def generate_simulation_report(self) -> Dict[str, Any]:
        """Generate comprehensive simulation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_auctions': len(self.simulation_history),
            'agents': {},
            'market_analysis': self._analyze_market_dynamics(),
            'competitive_landscape': self._analyze_competition()
        }
        
        # Individual agent performance
        for name, agent in self.agents.items():
            report['agents'][name] = agent.get_performance_summary()
        
        # Add learning insights
        report['learning_insights'] = self._generate_learning_insights()
        
        return report
    
    def _analyze_market_dynamics(self) -> Dict[str, Any]:
        """Analyze overall market dynamics from simulation"""
        if not self.simulation_history:
            return {}
        
        # Extract market metrics
        competitions = []
        winning_prices = []
        positions = []
        
        for auction in self.simulation_history:
            context = auction['context']
            competitions.append(context['market_competition'])
            
            for result in auction['results'].values():
                if result['won']:
                    winning_prices.append(result['winning_price'])
                positions.append(result['position'])
        
        return {
            'avg_market_competition': np.mean(competitions),
            'avg_winning_price': np.mean(winning_prices) if winning_prices else 0,
            'avg_position': np.mean(positions) if positions else 0,
            'price_volatility': np.std(winning_prices) if winning_prices else 0,
            'total_unique_auctions': len(self.simulation_history)
        }
    
    def _analyze_competition(self) -> Dict[str, Any]:
        """Analyze competitive dynamics between agents"""
        head_to_head = defaultdict(lambda: defaultdict(int))
        
        for auction in self.simulation_history:
            results = auction['results']
            participating_agents = list(results.keys())
            
            if len(participating_agents) > 1:
                # Find winner
                winner = None
                for agent_name, result in results.items():
                    if result['won']:
                        winner = agent_name
                        break
                
                if winner:
                    for agent_name in participating_agents:
                        if agent_name != winner:
                            head_to_head[winner][agent_name] += 1
        
        return {
            'head_to_head_wins': dict(head_to_head),
            'most_competitive_matchup': self._find_most_competitive_matchup(),
            'market_share': self._calculate_market_share()
        }
    
    def _find_most_competitive_matchup(self) -> str:
        """Find the most competitive agent matchup"""
        matchup_counts = defaultdict(int)
        
        for auction in self.simulation_history:
            results = auction['results']
            participating_agents = sorted(list(results.keys()))
            
            if len(participating_agents) >= 2:
                for i in range(len(participating_agents)):
                    for j in range(i + 1, len(participating_agents)):
                        matchup = f"{participating_agents[i]}_vs_{participating_agents[j]}"
                        matchup_counts[matchup] += 1
        
        if matchup_counts:
            return max(matchup_counts.items(), key=lambda x: x[1])[0]
        return "none"
    
    def _calculate_market_share(self) -> Dict[str, float]:
        """Calculate market share based on wins"""
        wins = defaultdict(int)
        total_wins = 0
        
        for auction in self.simulation_history:
            for agent_name, result in auction['results'].items():
                if result['won']:
                    wins[agent_name] += 1
                    total_wins += 1
        
        if total_wins == 0:
            return {}
        
        return {agent: count / total_wins for agent, count in wins.items()}
    
    def _generate_learning_insights(self) -> Dict[str, Any]:
        """Generate insights about agent learning and adaptation"""
        insights = {}
        
        for name, agent in self.agents.items():
            agent_insights = {
                'learning_events': len(agent.learning_history),
                'strategy_evolution': self._analyze_strategy_evolution(agent),
                'adaptation_triggers': self._analyze_adaptation_triggers(agent)
            }
            insights[name] = agent_insights
        
        return insights
    
    def _analyze_strategy_evolution(self, agent: BaseCompetitorAgent) -> Dict[str, Any]:
        """Analyze how agent's strategy evolved over time"""
        if not agent.learning_history:
            return {
                'evolution': 'no_changes',
                'total_changes': 0,
                'change_types': {},
                'latest_change': None
            }
        
        changes = defaultdict(int)
        for event in agent.learning_history:
            changes[event.get('action', 'unknown')] += 1
        
        return {
            'total_changes': len(agent.learning_history),
            'change_types': dict(changes),
            'latest_change': agent.learning_history[-1] if agent.learning_history else None
        }
    
    def _analyze_adaptation_triggers(self, agent: BaseCompetitorAgent) -> List[str]:
        """Identify what triggers caused agent adaptations"""
        triggers = []
        for event in agent.learning_history:
            reason = event.get('reason', 'unknown')
            if reason not in triggers:
                triggers.append(reason)
        return triggers
    
    def visualize_performance(self, save_path: str = None) -> None:
        """Create performance visualization charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Competitor Agents Performance Analysis', fontsize=16)
        
        # Prepare data
        agent_names = list(self.agents.keys())
        win_rates = [self.agents[name].metrics.win_rate for name in agent_names]
        spend_efficiency = [self.agents[name].metrics.spend_efficiency for name in agent_names]
        avg_positions = [self.agents[name].metrics.avg_position for name in agent_names]
        roas_values = [self.agents[name].metrics.roas for name in agent_names]
        
        # Win Rate Chart
        axes[0, 0].bar(agent_names, win_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0, 0].set_title('Win Rate by Agent')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Spend Efficiency Chart
        axes[0, 1].bar(agent_names, spend_efficiency, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0, 1].set_title('Spend Efficiency by Agent')
        axes[0, 1].set_ylabel('Revenue/Cost Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average Position Chart (lower is better)
        axes[1, 0].bar(agent_names, avg_positions, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1, 0].set_title('Average Position by Agent')
        axes[1, 0].set_ylabel('Position (lower = better)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # ROAS Chart
        axes[1, 1].bar(agent_names, roas_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1, 1].set_title('Return on Ad Spend (ROAS) by Agent')
        axes[1, 1].set_ylabel('ROAS')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance visualization saved to {save_path}")
        
        plt.show()
    
    def export_simulation_data(self, filepath: str) -> None:
        """Export simulation data to JSON file"""
        export_data = {
            'agents': {name: agent.get_performance_summary() for name, agent in self.agents.items()},
            'simulation_history': self.simulation_history,
            'market_conditions': {
                'base_competition': self.market_conditions['base_competition'],
                'seasonal_factors': self.market_conditions['seasonal_factors'],
                'user_distribution': {tier.value: weight for tier, weight in self.market_conditions['user_distribution'].items()}
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Simulation data exported to {filepath}")


def main():
    """Main function to demonstrate the CompetitorAgents system"""
    print("GAELP CompetitorAgents System Demo")
    print("=" * 50)
    
    # Initialize the manager
    manager = CompetitorAgentManager()
    
    # Run simulation
    print("\n1. Running simulation with 500 auctions over 14 days...")
    simulation_results = manager.run_simulation(num_auctions=500, days=14)
    
    # Display results
    print("\n2. Simulation Results:")
    print("-" * 30)
    
    for agent_name, agent_data in simulation_results['agents'].items():
        metrics = agent_data['metrics']
        print(f"\n{agent_name.upper()} ({agent_data['agent_type']}):")
        print(f"  Annual Budget: ${agent_data['annual_budget']}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Avg Position: {metrics['avg_position']:.1f}")
        print(f"  Spend Efficiency: {metrics['spend_efficiency']:.2f}")
        print(f"  ROAS: {metrics['roas']:.2f}")
        print(f"  Total Auctions: {metrics['total_auctions']}")
        print(f"  High-Value Wins: {metrics['high_value_wins']}")
    
    # Market analysis
    print(f"\n3. Market Analysis:")
    print("-" * 20)
    market = simulation_results['market_analysis']
    print(f"  Average Competition Level: {market['avg_market_competition']:.2f}")
    print(f"  Average Winning Price: ${market['avg_winning_price']:.2f}")
    print(f"  Total Auctions: {market['total_unique_auctions']}")
    
    # Competition analysis
    print(f"\n4. Competitive Landscape:")
    print("-" * 25)
    competition = simulation_results['competitive_landscape']
    print(f"  Most Competitive Matchup: {competition['most_competitive_matchup']}")
    
    if competition['market_share']:
        print("  Market Share:")
        for agent, share in competition['market_share'].items():
            print(f"    {agent}: {share:.1%}")
    
    # Learning insights
    print(f"\n5. Learning Insights:")
    print("-" * 20)
    for agent_name, insights in simulation_results['learning_insights'].items():
        print(f"  {agent_name}:")
        print(f"    Learning Events: {insights['learning_events']}")
        print(f"    Strategy Changes: {insights['strategy_evolution']['total_changes']}")
        if insights['adaptation_triggers']:
            print(f"    Adaptation Triggers: {', '.join(insights['adaptation_triggers'])}")
    
    # Visualization
    print(f"\n6. Generating performance visualization...")
    manager.visualize_performance('/home/hariravichandran/AELP/competitor_performance.png')
    
    # Export data
    print(f"\n7. Exporting simulation data...")
    manager.export_simulation_data('/home/hariravichandran/AELP/competitor_simulation_results.json')
    
    print(f"\nDemo completed! Check the generated files for detailed analysis.")


if __name__ == "__main__":
    main()