"""
Enhanced Multi-Touch Journey Tracking System for GAELP
Implements realistic customer journey simulation with state progression
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
from datetime import datetime, timedelta
import json
import hashlib

class UserState(Enum):
    """Customer journey states"""
    UNAWARE = "unaware"
    AWARE = "aware"
    INTERESTED = "interested"
    CONSIDERING = "considering"
    INTENT = "intent"
    CONVERTED = "converted"
    CHURNED = "churned"

class Channel(Enum):
    """Marketing channels"""
    SEARCH = "search"
    SOCIAL = "social"
    DISPLAY = "display"
    VIDEO = "video"
    EMAIL = "email"
    DIRECT = "direct"
    AFFILIATE = "affiliate"
    RETARGETING = "retargeting"

class TouchpointType(Enum):
    """Types of touchpoints"""
    IMPRESSION = "impression"
    CLICK = "click"
    ENGAGEMENT = "engagement"
    VISIT = "visit"
    SIGNUP = "signup"
    TRIAL = "trial"
    PURCHASE = "purchase"
    CHURN = "churn"

@dataclass
class Touchpoint:
    """Single touchpoint in customer journey"""
    timestamp: datetime
    channel: Channel
    touchpoint_type: TouchpointType
    bid_amount: float
    cost: float
    user_state_before: UserState
    user_state_after: UserState
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'channel': self.channel.value,
            'touchpoint_type': self.touchpoint_type.value,
            'bid_amount': self.bid_amount,
            'cost': self.cost,
            'user_state_before': self.user_state_before.value,
            'user_state_after': self.user_state_after.value,
            'metadata': self.metadata
        }

@dataclass
class EnhancedMultiTouchUser:
    """Enhanced user model with realistic journey progression"""
    user_id: str
    current_state: UserState = UserState.UNAWARE
    journey: List[Touchpoint] = field(default_factory=list)
    channel_affinity: Dict[Channel, float] = field(default_factory=dict)
    state_transition_probs: Dict[Tuple[UserState, Channel], Dict[UserState, float]] = field(default_factory=dict)
    conversion_probability: float = 0.0
    lifetime_value: float = 0.0
    time_since_last_touch: float = 0.0
    total_touches: int = 0
    total_cost: float = 0.0
    
    def __post_init__(self):
        # Initialize channel affinities
        if not self.channel_affinity:
            for channel in Channel:
                self.channel_affinity[channel] = np.random.beta(2, 5)
        
        # Initialize state transition probabilities
        if not self.state_transition_probs:
            self._initialize_transition_probs()
    
    def _initialize_transition_probs(self):
        """Initialize realistic state transition probabilities"""
        # Define base transition probabilities
        transitions = {
            (UserState.UNAWARE, Channel.SEARCH): {
                UserState.AWARE: 0.6, UserState.INTERESTED: 0.3, UserState.UNAWARE: 0.1
            },
            (UserState.UNAWARE, Channel.DISPLAY): {
                UserState.AWARE: 0.4, UserState.UNAWARE: 0.6
            },
            (UserState.AWARE, Channel.SEARCH): {
                UserState.INTERESTED: 0.5, UserState.CONSIDERING: 0.2, UserState.AWARE: 0.3
            },
            (UserState.AWARE, Channel.SOCIAL): {
                UserState.INTERESTED: 0.4, UserState.AWARE: 0.6
            },
            (UserState.INTERESTED, Channel.RETARGETING): {
                UserState.CONSIDERING: 0.6, UserState.INTENT: 0.2, UserState.INTERESTED: 0.2
            },
            (UserState.CONSIDERING, Channel.EMAIL): {
                UserState.INTENT: 0.5, UserState.CONVERTED: 0.2, UserState.CONSIDERING: 0.3
            },
            (UserState.INTENT, Channel.SEARCH): {
                UserState.CONVERTED: 0.7, UserState.INTENT: 0.3
            },
            (UserState.INTENT, Channel.RETARGETING): {
                UserState.CONVERTED: 0.6, UserState.INTENT: 0.4
            }
        }
        
        # Apply user-specific variations
        for key, probs in transitions.items():
            varied_probs = {}
            total = 0
            for state, prob in probs.items():
                varied = max(0.01, prob + np.random.normal(0, 0.05))
                varied_probs[state] = varied
                total += varied
            
            # Normalize
            self.state_transition_probs[key] = {
                state: prob/total for state, prob in varied_probs.items()
            }
    
    def process_touchpoint(self, channel: Channel, touchpoint_type: TouchpointType, 
                          bid_amount: float, timestamp: datetime) -> Tuple[UserState, float]:
        """Process a touchpoint and update user state"""
        cost = self._calculate_cost(channel, bid_amount, touchpoint_type)
        
        # Store previous state
        state_before = self.current_state
        
        # Calculate state transition
        new_state = self._calculate_state_transition(channel)
        
        # Create touchpoint record
        touchpoint = Touchpoint(
            timestamp=timestamp,
            channel=channel,
            touchpoint_type=touchpoint_type,
            bid_amount=bid_amount,
            cost=cost,
            user_state_before=state_before,
            user_state_after=new_state,
            metadata={
                'channel_affinity': self.channel_affinity[channel],
                'journey_length': len(self.journey),
                'time_since_last': self.time_since_last_touch
            }
        )
        
        # Update user state
        self.journey.append(touchpoint)
        self.current_state = new_state
        self.total_touches += 1
        self.total_cost += cost
        self.time_since_last_touch = 0
        
        # Update conversion probability
        self._update_conversion_probability()
        
        return new_state, cost
    
    def _calculate_cost(self, channel: Channel, bid_amount: float, 
                       touchpoint_type: TouchpointType) -> float:
        """Calculate actual cost based on channel and touchpoint type"""
        if touchpoint_type == TouchpointType.IMPRESSION:
            return bid_amount * 0.001  # CPM
        elif touchpoint_type == TouchpointType.CLICK:
            return bid_amount  # CPC
        elif touchpoint_type == TouchpointType.ENGAGEMENT:
            return bid_amount * 0.1
        else:
            return 0
    
    def _calculate_state_transition(self, channel: Channel) -> UserState:
        """Calculate new state based on current state and channel"""
        key = (self.current_state, channel)
        
        # Get transition probabilities
        if key in self.state_transition_probs:
            probs = self.state_transition_probs[key]
        else:
            # Default: stay in current state
            return self.current_state
        
        # Sample new state
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        # Adjust probabilities based on journey factors
        if self.total_touches > 5:
            # Increase conversion probability for users with many touches
            for i, state in enumerate(states):
                if state == UserState.CONVERTED:
                    probabilities[i] *= 1.5
                elif state == UserState.CHURNED:
                    probabilities[i] *= 0.8
        
        # Normalize
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        
        return np.random.choice(states, p=probabilities)
    
    def _update_conversion_probability(self):
        """Update conversion probability based on current state and journey"""
        base_probs = {
            UserState.UNAWARE: 0.01,
            UserState.AWARE: 0.05,
            UserState.INTERESTED: 0.15,
            UserState.CONSIDERING: 0.30,
            UserState.INTENT: 0.60,
            UserState.CONVERTED: 1.0,
            UserState.CHURNED: 0.0
        }
        
        self.conversion_probability = base_probs.get(self.current_state, 0.0)
        
        # Adjust based on journey length
        if self.total_touches > 3:
            self.conversion_probability *= 1.1
        if self.total_touches > 7:
            self.conversion_probability *= 1.2
        
        # Cap at 0.95 for non-converted users
        if self.current_state != UserState.CONVERTED:
            self.conversion_probability = min(0.95, self.conversion_probability)
    
    def calculate_ltv(self) -> float:
        """Calculate lifetime value based on conversion and engagement"""
        if self.current_state == UserState.CONVERTED:
            # Base LTV for Aura Parental Controls subscription
            base_ltv = 120  # $10/month * 12 months average retention
            
            # Adjust based on acquisition cost
            roi_multiplier = max(0.5, min(2.0, base_ltv / (self.total_cost + 1)))
            
            self.lifetime_value = base_ltv * roi_multiplier
        else:
            # Potential value based on conversion probability
            self.lifetime_value = 120 * self.conversion_probability
        
        return self.lifetime_value
    
    def get_journey_features(self) -> Dict[str, Any]:
        """Extract features from journey for RL agent"""
        features = {
            'user_id': self.user_id,
            'current_state': self.current_state.value,
            'total_touches': self.total_touches,
            'total_cost': self.total_cost,
            'conversion_probability': self.conversion_probability,
            'lifetime_value': self.lifetime_value,
            'time_since_last_touch': self.time_since_last_touch,
            'journey_length': len(self.journey),
            'unique_channels': len(set(t.channel for t in self.journey)),
            'channel_distribution': self._get_channel_distribution(),
            'state_progression': self._get_state_progression(),
            'avg_bid_amount': np.mean([t.bid_amount for t in self.journey]) if self.journey else 0,
            'last_channel': self.journey[-1].channel.value if self.journey else None,
            'last_touchpoint_type': self.journey[-1].touchpoint_type.value if self.journey else None
        }
        return features
    
    def _get_channel_distribution(self) -> Dict[str, float]:
        """Get distribution of channels in journey"""
        if not self.journey:
            return {}
        
        channel_counts = {}
        for touchpoint in self.journey:
            channel = touchpoint.channel.value
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        total = sum(channel_counts.values())
        return {ch: count/total for ch, count in channel_counts.items()}
    
    def _get_state_progression(self) -> List[str]:
        """Get sequence of state transitions"""
        if not self.journey:
            return []
        
        progression = []
        for touchpoint in self.journey:
            if touchpoint.user_state_after != touchpoint.user_state_before:
                progression.append(f"{touchpoint.user_state_before.value}->{touchpoint.user_state_after.value}")
        
        return progression

class MultiTouchJourneySimulator:
    """Simulate realistic multi-touch customer journeys"""
    
    def __init__(self, num_users: int = 1000, time_horizon_days: int = 30):
        self.num_users = num_users
        self.time_horizon_days = time_horizon_days
        self.users: Dict[str, EnhancedMultiTouchUser] = {}
        self.start_date = datetime.now()
        self.current_date = self.start_date
        
        # Initialize users
        for i in range(num_users):
            user_id = hashlib.md5(f"user_{i}".encode()).hexdigest()[:12]
            self.users[user_id] = EnhancedMultiTouchUser(user_id=user_id)
    
    def simulate_journeys(self) -> pd.DataFrame:
        """Simulate customer journeys over time horizon"""
        all_touchpoints = []
        
        for day in range(self.time_horizon_days):
            self.current_date = self.start_date + timedelta(days=day)
            
            # Simulate touchpoints for active users
            for user_id, user in self.users.items():
                if user.current_state in [UserState.CONVERTED, UserState.CHURNED]:
                    continue
                
                # Increment time since last touch
                user.time_since_last_touch += 1
                
                # Probability of touchpoint decreases with time
                touch_prob = 0.3 * np.exp(-user.time_since_last_touch / 7)
                
                if np.random.random() < touch_prob:
                    # Select channel based on user state and affinity
                    channel = self._select_channel(user)
                    
                    # Determine touchpoint type
                    touchpoint_type = self._select_touchpoint_type(user, channel)
                    
                    # Calculate bid amount
                    bid_amount = self._calculate_bid_amount(user, channel)
                    
                    # Process touchpoint
                    timestamp = self.current_date + timedelta(
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    )
                    
                    new_state, cost = user.process_touchpoint(
                        channel, touchpoint_type, bid_amount, timestamp
                    )
                    
                    # Record touchpoint
                    touchpoint_data = {
                        'user_id': user_id,
                        'timestamp': timestamp,
                        'day': day,
                        'channel': channel.value,
                        'touchpoint_type': touchpoint_type.value,
                        'bid_amount': bid_amount,
                        'cost': cost,
                        'state_before': user.journey[-1].user_state_before.value,
                        'state_after': new_state.value,
                        'conversion_probability': user.conversion_probability,
                        'ltv': user.calculate_ltv(),
                        'total_cost': user.total_cost,
                        'journey_length': len(user.journey)
                    }
                    all_touchpoints.append(touchpoint_data)
        
        return pd.DataFrame(all_touchpoints)
    
    def _select_channel(self, user: EnhancedMultiTouchUser) -> Channel:
        """Select channel based on user state and preferences"""
        # Channel selection strategy based on user state
        state_channel_prefs = {
            UserState.UNAWARE: [Channel.DISPLAY, Channel.SOCIAL, Channel.SEARCH],
            UserState.AWARE: [Channel.SEARCH, Channel.SOCIAL, Channel.VIDEO],
            UserState.INTERESTED: [Channel.RETARGETING, Channel.EMAIL, Channel.SEARCH],
            UserState.CONSIDERING: [Channel.EMAIL, Channel.RETARGETING, Channel.SEARCH],
            UserState.INTENT: [Channel.SEARCH, Channel.RETARGETING, Channel.DIRECT]
        }
        
        preferred_channels = state_channel_prefs.get(
            user.current_state, 
            list(Channel)
        )
        
        # Weight by user affinity
        weights = [user.channel_affinity.get(ch, 0.5) for ch in preferred_channels]
        total = sum(weights)
        probs = [w/total for w in weights]
        
        return np.random.choice(preferred_channels, p=probs)
    
    def _select_touchpoint_type(self, user: EnhancedMultiTouchUser, 
                                channel: Channel) -> TouchpointType:
        """Select touchpoint type based on channel and user state"""
        if user.current_state == UserState.UNAWARE:
            return TouchpointType.IMPRESSION
        elif user.current_state == UserState.AWARE:
            return np.random.choice([TouchpointType.IMPRESSION, TouchpointType.CLICK], p=[0.7, 0.3])
        elif user.current_state == UserState.INTERESTED:
            return np.random.choice([TouchpointType.CLICK, TouchpointType.ENGAGEMENT], p=[0.6, 0.4])
        elif user.current_state == UserState.CONSIDERING:
            return np.random.choice([TouchpointType.ENGAGEMENT, TouchpointType.VISIT], p=[0.5, 0.5])
        elif user.current_state == UserState.INTENT:
            return np.random.choice([TouchpointType.VISIT, TouchpointType.TRIAL], p=[0.6, 0.4])
        else:
            return TouchpointType.CLICK
    
    def _calculate_bid_amount(self, user: EnhancedMultiTouchUser, 
                             channel: Channel) -> float:
        """Calculate bid amount based on user value and channel"""
        # Base bids by channel
        base_bids = {
            Channel.SEARCH: 2.5,
            Channel.SOCIAL: 1.5,
            Channel.DISPLAY: 0.8,
            Channel.VIDEO: 2.0,
            Channel.EMAIL: 0.1,
            Channel.DIRECT: 0.0,
            Channel.AFFILIATE: 1.2,
            Channel.RETARGETING: 1.8
        }
        
        base_bid = base_bids.get(channel, 1.0)
        
        # Adjust based on user state (bid more for high-intent users)
        state_multipliers = {
            UserState.UNAWARE: 0.5,
            UserState.AWARE: 0.7,
            UserState.INTERESTED: 1.0,
            UserState.CONSIDERING: 1.3,
            UserState.INTENT: 1.8
        }
        
        multiplier = state_multipliers.get(user.current_state, 1.0)
        
        # Add some noise
        noise = np.random.normal(1.0, 0.2)
        
        return max(0.1, base_bid * multiplier * noise)
    
    def get_journey_statistics(self) -> Dict[str, Any]:
        """Get statistics about simulated journeys"""
        converted_users = [u for u in self.users.values() 
                          if u.current_state == UserState.CONVERTED]
        
        stats = {
            'total_users': self.num_users,
            'converted_users': len(converted_users),
            'conversion_rate': len(converted_users) / self.num_users,
            'avg_touches_to_conversion': np.mean([u.total_touches for u in converted_users]) if converted_users else 0,
            'avg_cost_per_conversion': np.mean([u.total_cost for u in converted_users]) if converted_users else 0,
            'avg_ltv': np.mean([u.calculate_ltv() for u in self.users.values()]),
            'total_spend': sum(u.total_cost for u in self.users.values()),
            'roi': sum(u.calculate_ltv() for u in converted_users) / sum(u.total_cost for u in self.users.values()) if self.users else 0
        }
        
        return stats

def main():
    """Test the enhanced journey tracking system"""
    print("🚀 Enhanced Multi-Touch Journey Tracking System")
    print("=" * 60)
    
    # Create simulator
    simulator = MultiTouchJourneySimulator(num_users=100, time_horizon_days=14)
    
    # Run simulation
    print("\n📊 Simulating customer journeys...")
    touchpoints_df = simulator.simulate_journeys()
    
    # Get statistics
    stats = simulator.get_journey_statistics()
    
    print("\n📈 Journey Statistics:")
    print(f"  • Total Users: {stats['total_users']}")
    print(f"  • Converted Users: {stats['converted_users']}")
    print(f"  • Conversion Rate: {stats['conversion_rate']:.2%}")
    print(f"  • Avg Touches to Conversion: {stats['avg_touches_to_conversion']:.1f}")
    print(f"  • Avg Cost per Conversion: ${stats['avg_cost_per_conversion']:.2f}")
    print(f"  • Average LTV: ${stats['avg_ltv']:.2f}")
    print(f"  • Total Spend: ${stats['total_spend']:.2f}")
    print(f"  • ROI: {stats['roi']:.2f}x")
    
    # Show sample journeys
    print("\n🔍 Sample Customer Journeys:")
    sample_users = list(simulator.users.values())[:3]
    
    for user in sample_users:
        print(f"\n  User {user.user_id}:")
        print(f"    • Current State: {user.current_state.value}")
        print(f"    • Total Touches: {user.total_touches}")
        print(f"    • Total Cost: ${user.total_cost:.2f}")
        print(f"    • LTV: ${user.calculate_ltv():.2f}")
        print(f"    • Journey:")
        
        for i, touchpoint in enumerate(user.journey[:5]):  # Show first 5 touches
            print(f"      {i+1}. {touchpoint.channel.value} → "
                  f"{touchpoint.user_state_before.value} to {touchpoint.user_state_after.value} "
                  f"(${touchpoint.cost:.2f})")
        
        if len(user.journey) > 5:
            print(f"      ... and {len(user.journey) - 5} more touches")
    
    # Save results
    print("\n💾 Saving simulation results...")
    touchpoints_df.to_csv('/home/hariravichandran/AELP/journey_simulation_results.csv', index=False)
    
    # Save journey data for RL training
    journey_data = {
        'journeys': [
            {
                'user_id': user.user_id,
                'features': user.get_journey_features(),
                'journey': [t.to_dict() for t in user.journey]
            }
            for user in simulator.users.values()
        ],
        'statistics': stats
    }
    
    with open('/home/hariravichandran/AELP/journey_data.json', 'w') as f:
        json.dump(journey_data, f, indent=2, default=str)
    
    print("✅ Journey tracking system ready for integration!")
    
    return simulator, touchpoints_df

if __name__ == "__main__":
    simulator, df = main()