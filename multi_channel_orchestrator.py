"""
Multi-Channel Orchestration System for GAELP
Coordinates bidding across multiple channels with journey awareness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from enhanced_journey_tracking import (
    EnhancedMultiTouchUser, Channel, UserState, 
    TouchpointType, MultiTouchJourneySimulator
)

@dataclass
class BidDecision:
    """Represents a bidding decision for a channel"""
    channel: Channel
    bid_amount: float
    expected_value: float
    confidence: float
    reasoning: str

class MultiChannelOrchestrator:
    """Orchestrates bidding across multiple marketing channels"""
    
    def __init__(self, 
                 budget_daily: float = 1000.0,
                 channels: Optional[List[Channel]] = None,
                 learning_rate: float = 0.01):
        self.budget_daily = budget_daily
        self.budget_remaining = budget_daily
        self.channels = channels or list(Channel)
        self.learning_rate = learning_rate
        
        # Channel performance tracking
        self.channel_performance = {
            channel: {
                'impressions': 0,
                'clicks': 0,
                'conversions': 0,
                'spend': 0.0,
                'revenue': 0.0,
                'avg_cpc': 0.0,
                'avg_cpm': 0.0,
                'conversion_rate': 0.0,
                'roas': 0.0
            }
            for channel in self.channels
        }
        
        # Budget allocation strategy
        self.channel_budgets = self._initialize_budgets()
        
        # Journey-aware bid adjustments
        self.state_bid_multipliers = {
            UserState.UNAWARE: 0.6,
            UserState.AWARE: 0.8,
            UserState.INTERESTED: 1.0,
            UserState.CONSIDERING: 1.3,
            UserState.INTENT: 1.8,
            UserState.CONVERTED: 0.0,
            UserState.CHURNED: 0.0
        }
        
        # Channel effectiveness by user state
        self.channel_state_effectiveness = self._initialize_effectiveness()
    
    def _initialize_budgets(self) -> Dict[Channel, float]:
        """Initialize budget allocation across channels"""
        # Start with even distribution
        base_allocation = self.budget_daily / len(self.channels)
        
        # Adjust based on channel characteristics
        allocations = {
            Channel.SEARCH: base_allocation * 1.5,  # Higher for search
            Channel.SOCIAL: base_allocation * 1.2,
            Channel.DISPLAY: base_allocation * 0.8,
            Channel.VIDEO: base_allocation * 1.1,
            Channel.EMAIL: base_allocation * 0.3,  # Low cost channel
            Channel.DIRECT: base_allocation * 0.1,  # Minimal spend
            Channel.AFFILIATE: base_allocation * 0.9,
            Channel.RETARGETING: base_allocation * 1.3
        }
        
        # Normalize to match total budget
        total = sum(allocations.values())
        factor = self.budget_daily / total
        
        return {ch: alloc * factor for ch, alloc in allocations.items()}
    
    def _initialize_effectiveness(self) -> Dict[Tuple[Channel, UserState], float]:
        """Initialize channel effectiveness for different user states"""
        effectiveness = {}
        
        # Define effectiveness matrix
        eff_matrix = {
            (Channel.SEARCH, UserState.UNAWARE): 0.7,
            (Channel.SEARCH, UserState.AWARE): 0.8,
            (Channel.SEARCH, UserState.INTERESTED): 0.9,
            (Channel.SEARCH, UserState.CONSIDERING): 0.85,
            (Channel.SEARCH, UserState.INTENT): 0.95,
            
            (Channel.SOCIAL, UserState.UNAWARE): 0.6,
            (Channel.SOCIAL, UserState.AWARE): 0.7,
            (Channel.SOCIAL, UserState.INTERESTED): 0.65,
            
            (Channel.DISPLAY, UserState.UNAWARE): 0.5,
            (Channel.DISPLAY, UserState.AWARE): 0.4,
            
            (Channel.RETARGETING, UserState.INTERESTED): 0.8,
            (Channel.RETARGETING, UserState.CONSIDERING): 0.85,
            (Channel.RETARGETING, UserState.INTENT): 0.9,
            
            (Channel.EMAIL, UserState.CONSIDERING): 0.75,
            (Channel.EMAIL, UserState.INTENT): 0.8,
        }
        
        # Fill in missing values with defaults
        for channel in self.channels:
            for state in UserState:
                key = (channel, state)
                if key not in eff_matrix:
                    if state in [UserState.CONVERTED, UserState.CHURNED]:
                        effectiveness[key] = 0.0
                    else:
                        effectiveness[key] = 0.3  # Default low effectiveness
                else:
                    effectiveness[key] = eff_matrix[key]
        
        return effectiveness
    
    def decide_bid(self, user: EnhancedMultiTouchUser, 
                   available_channels: List[Channel]) -> List[BidDecision]:
        """Decide bidding strategy for a user across channels"""
        decisions = []
        
        for channel in available_channels:
            # Check budget constraint
            if self.channel_budgets.get(channel, 0) <= 0:
                continue
            
            # Calculate expected value
            expected_value = self._calculate_expected_value(user, channel)
            
            # Determine bid amount
            bid_amount = self._calculate_bid_amount(user, channel, expected_value)
            
            # Calculate confidence
            confidence = self._calculate_confidence(user, channel)
            
            # Create bid decision
            decision = BidDecision(
                channel=channel,
                bid_amount=bid_amount,
                expected_value=expected_value,
                confidence=confidence,
                reasoning=self._generate_reasoning(user, channel)
            )
            
            decisions.append(decision)
        
        # Sort by expected value and return top choices
        decisions.sort(key=lambda x: x.expected_value * x.confidence, reverse=True)
        
        return decisions
    
    def _calculate_expected_value(self, user: EnhancedMultiTouchUser, 
                                  channel: Channel) -> float:
        """Calculate expected value of showing ad to user on channel"""
        # Base LTV estimate
        base_ltv = 120  # $10/month * 12 months for Aura
        
        # Get user's conversion probability
        conv_prob = user.conversion_probability
        
        # Adjust for channel effectiveness
        effectiveness_key = (channel, user.current_state)
        channel_effectiveness = self.channel_state_effectiveness.get(effectiveness_key, 0.3)
        
        # Consider user's channel affinity
        user_affinity = user.channel_affinity.get(channel, 0.5)
        
        # Calculate expected value
        expected_value = base_ltv * conv_prob * channel_effectiveness * user_affinity
        
        # Apply journey length penalty (diminishing returns)
        if user.total_touches > 10:
            expected_value *= 0.8
        elif user.total_touches > 15:
            expected_value *= 0.6
        
        return expected_value
    
    def _calculate_bid_amount(self, user: EnhancedMultiTouchUser, 
                             channel: Channel, expected_value: float) -> float:
        """Calculate optimal bid amount"""
        # Base bid from expected value
        base_bid = expected_value * 0.1  # Bid 10% of expected value
        
        # Apply state multiplier
        state_multiplier = self.state_bid_multipliers.get(user.current_state, 1.0)
        
        # Apply channel-specific adjustments
        channel_costs = {
            Channel.SEARCH: 2.5,
            Channel.SOCIAL: 1.5,
            Channel.DISPLAY: 0.8,
            Channel.VIDEO: 2.0,
            Channel.EMAIL: 0.1,
            Channel.DIRECT: 0.0,
            Channel.AFFILIATE: 1.2,
            Channel.RETARGETING: 1.8
        }
        
        channel_base = channel_costs.get(channel, 1.0)
        
        # Combine factors
        bid_amount = min(base_bid * state_multiplier, channel_base * 1.5)
        
        # Apply budget constraint
        available_budget = self.channel_budgets.get(channel, 0)
        bid_amount = min(bid_amount, available_budget * 0.1)  # Max 10% of channel budget per bid
        
        return max(0.01, bid_amount)  # Minimum bid
    
    def _calculate_confidence(self, user: EnhancedMultiTouchUser, 
                             channel: Channel) -> float:
        """Calculate confidence in bid decision"""
        # Base confidence on data availability
        if user.total_touches == 0:
            base_confidence = 0.3
        elif user.total_touches < 3:
            base_confidence = 0.5
        elif user.total_touches < 7:
            base_confidence = 0.7
        else:
            base_confidence = 0.9
        
        # Adjust for channel-state match
        effectiveness_key = (channel, user.current_state)
        effectiveness = self.channel_state_effectiveness.get(effectiveness_key, 0.3)
        
        confidence = base_confidence * (0.5 + 0.5 * effectiveness)
        
        return min(1.0, confidence)
    
    def _generate_reasoning(self, user: EnhancedMultiTouchUser, 
                           channel: Channel) -> str:
        """Generate reasoning for bid decision"""
        reasons = []
        
        # User state reasoning
        if user.current_state == UserState.INTENT:
            reasons.append("High intent user")
        elif user.current_state == UserState.CONSIDERING:
            reasons.append("User in consideration phase")
        elif user.current_state == UserState.INTERESTED:
            reasons.append("User showing interest")
        
        # Channel effectiveness
        effectiveness_key = (channel, user.current_state)
        effectiveness = self.channel_state_effectiveness.get(effectiveness_key, 0.3)
        if effectiveness > 0.7:
            reasons.append(f"High channel effectiveness ({effectiveness:.2f})")
        
        # Journey length
        if user.total_touches > 5:
            reasons.append(f"Multiple touches ({user.total_touches})")
        
        # User affinity
        affinity = user.channel_affinity.get(channel, 0.5)
        if affinity > 0.7:
            reasons.append(f"High channel affinity ({affinity:.2f})")
        
        return "; ".join(reasons) if reasons else "Standard bid"
    
    def update_performance(self, channel: Channel, touchpoint_type: TouchpointType,
                          cost: float, converted: bool = False, revenue: float = 0):
        """Update channel performance metrics"""
        perf = self.channel_performance[channel]
        
        # Update counts
        if touchpoint_type == TouchpointType.IMPRESSION:
            perf['impressions'] += 1
        elif touchpoint_type == TouchpointType.CLICK:
            perf['clicks'] += 1
        
        if converted:
            perf['conversions'] += 1
            perf['revenue'] += revenue
        
        perf['spend'] += cost
        
        # Update derived metrics
        if perf['clicks'] > 0:
            perf['avg_cpc'] = perf['spend'] / perf['clicks']
        
        if perf['impressions'] > 0:
            perf['avg_cpm'] = (perf['spend'] / perf['impressions']) * 1000
            
        if perf['clicks'] > 0:
            perf['conversion_rate'] = perf['conversions'] / perf['clicks']
        
        if perf['spend'] > 0:
            perf['roas'] = perf['revenue'] / perf['spend']
        
        # Update budget
        self.channel_budgets[channel] -= cost
        self.budget_remaining -= cost
    
    def rebalance_budgets(self):
        """Rebalance budgets based on performance"""
        # Calculate performance scores
        scores = {}
        for channel in self.channels:
            perf = self.channel_performance[channel]
            
            # Score based on ROAS and conversion rate
            if perf['spend'] > 0:
                score = perf['roas'] * 0.7 + perf['conversion_rate'] * 100 * 0.3
            else:
                score = 0.5  # Neutral score for unused channels
            
            scores[channel] = score
        
        # Redistribute remaining budget
        if self.budget_remaining > 0:
            total_score = sum(scores.values())
            if total_score > 0:
                for channel in self.channels:
                    weight = scores[channel] / total_score
                    additional = self.budget_remaining * weight * self.learning_rate
                    self.channel_budgets[channel] += additional
    
    def get_orchestration_state(self) -> Dict[str, Any]:
        """Get current state of orchestration"""
        return {
            'budget_remaining': self.budget_remaining,
            'channel_budgets': {ch.value: budget for ch, budget in self.channel_budgets.items()},
            'channel_performance': {
                ch.value: perf for ch, perf in self.channel_performance.items()
            },
            'total_spend': self.budget_daily - self.budget_remaining,
            'total_conversions': sum(p['conversions'] for p in self.channel_performance.values()),
            'total_revenue': sum(p['revenue'] for p in self.channel_performance.values()),
            'overall_roas': sum(p['revenue'] for p in self.channel_performance.values()) / 
                           (self.budget_daily - self.budget_remaining + 0.01)
        }

class JourneyAwareRLEnvironment(gym.Env):
    """Gymnasium environment for journey-aware RL training"""
    
    def __init__(self, simulator: MultiTouchJourneySimulator, 
                 orchestrator: MultiChannelOrchestrator):
        super().__init__()
        
        self.simulator = simulator
        self.orchestrator = orchestrator
        self.current_step = 0
        self.max_steps = 1000
        
        # Define action space (bid amounts for each channel)
        self.action_space = spaces.Box(
            low=0.0, 
            high=10.0, 
            shape=(len(Channel),), 
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'user_state': spaces.Discrete(len(UserState)),
            'journey_length': spaces.Box(low=0, high=50, shape=(1,), dtype=np.int32),
            'time_since_last': spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32),
            'channel_affinities': spaces.Box(low=0, high=1, shape=(len(Channel),), dtype=np.float32),
            'conversion_probability': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'budget_remaining': spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
            'channel_performance': spaces.Box(low=-np.inf, high=np.inf, 
                                             shape=(len(Channel), 4), dtype=np.float32)
        })
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.orchestrator.budget_remaining = self.orchestrator.budget_daily
        self.orchestrator.channel_budgets = self.orchestrator._initialize_budgets()
        
        # Reset performance metrics
        for channel in self.orchestrator.channels:
            for key in self.orchestrator.channel_performance[channel]:
                if isinstance(self.orchestrator.channel_performance[channel][key], (int, float)):
                    self.orchestrator.channel_performance[channel][key] = 0
        
        # Get random user
        self.current_user = np.random.choice(list(self.simulator.users.values()))
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return results"""
        self.current_step += 1
        
        # Parse action (bid amounts for each channel)
        bid_amounts = action
        
        # Select channel based on bids
        channel_idx = np.argmax(bid_amounts)
        channel = list(Channel)[channel_idx]
        bid_amount = float(bid_amounts[channel_idx])
        
        # Process touchpoint
        touchpoint_type = TouchpointType.CLICK  # Simplified
        timestamp = self.simulator.current_date
        
        old_state = self.current_user.current_state
        new_state, cost = self.current_user.process_touchpoint(
            channel, touchpoint_type, bid_amount, timestamp
        )
        
        # Calculate reward
        reward = self._calculate_reward(old_state, new_state, cost)
        
        # Update orchestrator
        converted = (new_state == UserState.CONVERTED)
        revenue = 120 if converted else 0
        self.orchestrator.update_performance(channel, touchpoint_type, cost, converted, revenue)
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps or 
                self.orchestrator.budget_remaining <= 0 or
                new_state in [UserState.CONVERTED, UserState.CHURNED])
        
        # Get next user if current journey ended
        if new_state in [UserState.CONVERTED, UserState.CHURNED]:
            self.current_user = np.random.choice(list(self.simulator.users.values()))
        
        truncated = False
        info = {
            'user_id': self.current_user.user_id,
            'channel': channel.value,
            'cost': cost,
            'converted': converted,
            'state_transition': f"{old_state.value}->{new_state.value}"
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self):
        """Get current observation"""
        user = self.current_user
        
        # User features
        user_state_idx = list(UserState).index(user.current_state)
        channel_affinities = np.array([user.channel_affinity.get(ch, 0.5) for ch in Channel])
        
        # Channel performance features
        channel_perf = np.zeros((len(Channel), 4))
        for i, channel in enumerate(Channel):
            perf = self.orchestrator.channel_performance[channel]
            channel_perf[i] = [
                perf['conversion_rate'],
                perf['roas'],
                perf['avg_cpc'],
                perf['spend'] / (self.orchestrator.budget_daily + 0.01)
            ]
        
        return {
            'user_state': user_state_idx,
            'journey_length': np.array([user.total_touches], dtype=np.int32),
            'time_since_last': np.array([user.time_since_last_touch], dtype=np.float32),
            'channel_affinities': channel_affinities.astype(np.float32),
            'conversion_probability': np.array([user.conversion_probability], dtype=np.float32),
            'budget_remaining': np.array([self.orchestrator.budget_remaining], dtype=np.float32),
            'channel_performance': channel_perf.astype(np.float32)
        }
    
    def _calculate_reward(self, old_state: UserState, new_state: UserState, 
                         cost: float) -> float:
        """Calculate reward for state transition"""
        # Base reward for conversion
        if new_state == UserState.CONVERTED:
            conversion_reward = 100
        else:
            conversion_reward = 0
        
        # Progress reward
        state_values = {
            UserState.UNAWARE: 0,
            UserState.AWARE: 1,
            UserState.INTERESTED: 2,
            UserState.CONSIDERING: 3,
            UserState.INTENT: 4,
            UserState.CONVERTED: 5,
            UserState.CHURNED: -1
        }
        
        progress = state_values.get(new_state, 0) - state_values.get(old_state, 0)
        progress_reward = progress * 5
        
        # Cost penalty
        cost_penalty = -cost * 0.5
        
        # Efficiency bonus (high conv prob, low cost)
        if self.current_user.conversion_probability > 0.5 and cost < 1.0:
            efficiency_bonus = 10
        else:
            efficiency_bonus = 0
        
        total_reward = conversion_reward + progress_reward + cost_penalty + efficiency_bonus
        
        return total_reward
    
    def render(self):
        """Render environment state"""
        print(f"\nStep {self.current_step}")
        print(f"User: {self.current_user.user_id}")
        print(f"State: {self.current_user.current_state.value}")
        print(f"Journey Length: {self.current_user.total_touches}")
        print(f"Budget Remaining: ${self.orchestrator.budget_remaining:.2f}")
        
        # Show top performing channels
        perfs = [(ch, self.orchestrator.channel_performance[ch]['roas']) 
                 for ch in self.orchestrator.channels]
        perfs.sort(key=lambda x: x[1], reverse=True)
        
        print("Top Channels by ROAS:")
        for ch, roas in perfs[:3]:
            print(f"  {ch.value}: {roas:.2f}x")

def main():
    """Test multi-channel orchestration"""
    print("🎯 Multi-Channel Orchestration System")
    print("=" * 60)
    
    # Create components
    simulator = MultiTouchJourneySimulator(num_users=50, time_horizon_days=7)
    orchestrator = MultiChannelOrchestrator(budget_daily=500.0)
    
    print("\n📊 Testing Orchestration...")
    
    # Test bid decisions for sample users
    sample_users = list(simulator.users.values())[:5]
    
    for user in sample_users:
        print(f"\nUser {user.user_id}:")
        print(f"  State: {user.current_state.value}")
        print(f"  Journey Length: {user.total_touches}")
        
        # Get bid decisions
        decisions = orchestrator.decide_bid(user, list(Channel)[:4])
        
        print("  Bid Decisions:")
        for decision in decisions[:3]:
            print(f"    {decision.channel.value}: ${decision.bid_amount:.2f}")
            print(f"      Expected Value: ${decision.expected_value:.2f}")
            print(f"      Confidence: {decision.confidence:.2f}")
            print(f"      Reasoning: {decision.reasoning}")
    
    # Test RL environment
    print("\n🤖 Testing RL Environment...")
    env = JourneyAwareRLEnvironment(simulator, orchestrator)
    
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs['user_state']}")
    
    # Run a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step reward: {reward:.2f}, State: {info['state_transition']}")
        
        if done:
            break
    
    # Show orchestration state
    state = orchestrator.get_orchestration_state()
    print(f"\n📈 Orchestration Results:")
    print(f"  Total Spend: ${state['total_spend']:.2f}")
    print(f"  Total Conversions: {state['total_conversions']}")
    print(f"  Overall ROAS: {state['overall_roas']:.2f}x")
    
    print("\n✅ Multi-channel orchestration ready!")
    
    return orchestrator, env

if __name__ == "__main__":
    orchestrator, env = main()