#!/usr/bin/env python3
"""
RecSim-based User Behavior Model for Ad Campaigns

This module leverages Google's RecSim NG to create sophisticated user models
for different types of ad campaign participants, including impulse buyers,
researchers, loyal customers, and window shoppers.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

# NO FALLBACKS - RecSim MUST work
import sys
sys.path.insert(0, '/home/hariravichandran/AELP')
from NO_FALLBACKS import StrictModeEnforcer

# Apply edward2 compatibility patch BEFORE importing edward2
import edward2_patch

try:
    import edward2 as ed
    import recsim_ng.core.value as value
    import recsim_ng.lib.tensorflow.entity as entity
    import recsim_ng.lib.tensorflow.field_spec as field_spec
    from recsim_ng.lib.tensorflow import runtime
except ImportError as e:
    StrictModeEnforcer.enforce('RECSIM', fallback_attempted=True)
    raise ImportError(f"RecSim NG MUST be installed. NO FALLBACKS! Error: {e}")

logger = logging.getLogger(__name__)


class UserSegment(Enum):
    """Enum for different user segments in ad campaigns"""
    IMPULSE_BUYER = "impulse_buyer"
    RESEARCHER = "researcher"
    LOYAL_CUSTOMER = "loyal_customer"
    WINDOW_SHOPPER = "window_shopper"
    PRICE_CONSCIOUS = "price_conscious"
    BRAND_LOYALIST = "brand_loyalist"


@dataclass
class UserProfile:
    """Represents a user's behavioral profile"""
    segment: UserSegment
    click_propensity: float  # Base tendency to click on ads
    conversion_propensity: float  # Base tendency to convert after clicking
    price_sensitivity: float  # 0-1, higher means more sensitive to price
    brand_affinity: float  # 0-1, loyalty to specific brands
    time_preference: Dict[str, float]  # Preferred times for engagement
    device_preference: Dict[str, float]  # Preferred devices
    attention_span: float  # How long they engage with content
    budget: float  # Disposable income level
    
    # Dynamic state variables
    current_interest: float = 0.5
    fatigue_level: float = 0.0
    recent_purchases: List[str] = field(default_factory=list)


class RecSimUserModel:
    """
    RecSim NG-based user behavior model for ad campaigns.
    Models different user segments with realistic behavior patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.user_profiles = self._create_user_segments()
        self.current_users = {}
        self.interaction_history = []
        
        # Initialize RecSim components if available
        if entity is not None:
            self._init_recsim_components()
    
    def _create_user_segments(self) -> Dict[UserSegment, UserProfile]:
        """Create detailed user segment profiles"""
        
        segments = {
            UserSegment.IMPULSE_BUYER: UserProfile(
                segment=UserSegment.IMPULSE_BUYER,
                click_propensity=0.08,
                conversion_propensity=0.15,
                price_sensitivity=0.3,
                brand_affinity=0.4,
                time_preference={
                    'morning': 0.6, 'afternoon': 0.9, 'evening': 1.2, 'night': 0.8
                },
                device_preference={
                    'mobile': 1.4, 'desktop': 0.8, 'tablet': 1.0
                },
                attention_span=2.5,  # seconds
                budget=150.0
            ),
            
            UserSegment.RESEARCHER: UserProfile(
                segment=UserSegment.RESEARCHER,
                click_propensity=0.12,
                conversion_propensity=0.02,
                price_sensitivity=0.8,
                brand_affinity=0.6,
                time_preference={
                    'morning': 1.0, 'afternoon': 1.3, 'evening': 1.1, 'night': 0.5
                },
                device_preference={
                    'mobile': 0.7, 'desktop': 1.5, 'tablet': 1.1
                },
                attention_span=8.0,
                budget=300.0
            ),
            
            UserSegment.LOYAL_CUSTOMER: UserProfile(
                segment=UserSegment.LOYAL_CUSTOMER,
                click_propensity=0.15,
                conversion_propensity=0.25,
                price_sensitivity=0.2,
                brand_affinity=0.9,
                time_preference={
                    'morning': 1.1, 'afternoon': 1.0, 'evening': 1.2, 'night': 0.9
                },
                device_preference={
                    'mobile': 1.1, 'desktop': 1.2, 'tablet': 0.9
                },
                attention_span=5.0,
                budget=500.0
            ),
            
            UserSegment.WINDOW_SHOPPER: UserProfile(
                segment=UserSegment.WINDOW_SHOPPER,
                click_propensity=0.05,
                conversion_propensity=0.01,
                price_sensitivity=0.9,
                brand_affinity=0.3,
                time_preference={
                    'morning': 0.7, 'afternoon': 0.8, 'evening': 1.4, 'night': 1.0
                },
                device_preference={
                    'mobile': 1.3, 'desktop': 0.6, 'tablet': 1.1
                },
                attention_span=1.5,
                budget=75.0
            ),
            
            UserSegment.PRICE_CONSCIOUS: UserProfile(
                segment=UserSegment.PRICE_CONSCIOUS,
                click_propensity=0.06,
                conversion_propensity=0.08,
                price_sensitivity=0.95,
                brand_affinity=0.2,
                time_preference={
                    'morning': 0.9, 'afternoon': 1.0, 'evening': 1.1, 'night': 0.8
                },
                device_preference={
                    'mobile': 1.2, 'desktop': 1.0, 'tablet': 0.8
                },
                attention_span=4.0,
                budget=100.0
            ),
            
            UserSegment.BRAND_LOYALIST: UserProfile(
                segment=UserSegment.BRAND_LOYALIST,
                click_propensity=0.18,
                conversion_propensity=0.30,
                price_sensitivity=0.1,
                brand_affinity=0.95,
                time_preference={
                    'morning': 1.0, 'afternoon': 1.1, 'evening': 1.0, 'night': 0.9
                },
                device_preference={
                    'mobile': 1.0, 'desktop': 1.3, 'tablet': 1.0
                },
                attention_span=6.0,
                budget=800.0
            )
        }
        
        return segments
    
    def _init_recsim_components(self):
        """Initialize RecSim NG components for probabilistic modeling"""
        if entity is None:
            return
        
        # Define field specifications for user state
        # RecSim NG FieldSpec is just a container - doesn't take dtype/shape args
        # It's used for value checking, not type specification
        self.user_state_spec = {
            'segment_id': field_spec.FieldSpec(),
            'interest_level': field_spec.FieldSpec(),
            'fatigue_level': field_spec.FieldSpec(),
            'time_on_site': field_spec.FieldSpec(),
            'page_views': field_spec.FieldSpec(),
            'ad_exposures': field_spec.FieldSpec()
        }
        
        # Define ad/content specifications
        self.ad_spec = {
            'creative_quality': field_spec.FieldSpec(),
            'price_shown': field_spec.FieldSpec(),
            'brand_match': field_spec.FieldSpec(),
            'relevance_score': field_spec.FieldSpec()
        }
    
    def generate_user(self, user_id: str, segment: Optional[UserSegment] = None) -> UserProfile:
        """Generate a new user with specified or random segment"""
        
        if segment is None:
            # Sample segment based on realistic distribution
            segment_weights = {
                UserSegment.IMPULSE_BUYER: 0.20,
                UserSegment.RESEARCHER: 0.15,
                UserSegment.LOYAL_CUSTOMER: 0.10,
                UserSegment.WINDOW_SHOPPER: 0.25,
                UserSegment.PRICE_CONSCIOUS: 0.20,
                UserSegment.BRAND_LOYALIST: 0.10
            }
            
            segments = list(segment_weights.keys())
            weights = list(segment_weights.values())
            segment = np.random.choice(segments, p=weights)
        
        # Get base profile and add noise
        base_profile = self.user_profiles[segment]
        
        # Add individual variation
        user_profile = UserProfile(
            segment=segment,
            click_propensity=max(0, np.random.normal(
                base_profile.click_propensity, 
                base_profile.click_propensity * 0.3
            )),
            conversion_propensity=max(0, np.random.normal(
                base_profile.conversion_propensity,
                base_profile.conversion_propensity * 0.4
            )),
            price_sensitivity=np.clip(np.random.normal(
                base_profile.price_sensitivity, 0.1
            ), 0, 1),
            brand_affinity=np.clip(np.random.normal(
                base_profile.brand_affinity, 0.15
            ), 0, 1),
            time_preference=base_profile.time_preference.copy(),
            device_preference=base_profile.device_preference.copy(),
            attention_span=max(0.5, np.random.normal(
                base_profile.attention_span, 
                base_profile.attention_span * 0.4
            )),
            budget=max(10, np.random.gamma(2, base_profile.budget / 2))
        )
        
        self.current_users[user_id] = user_profile
        return user_profile
    
    def simulate_ad_response(self, 
                           user_id: str, 
                           ad_content: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate user response to an advertisement
        
        Args:
            user_id: Unique identifier for the user
            ad_content: Dict containing ad properties (creative_quality, price_shown, etc.)
            context: Dict containing contextual information (time, device, etc.)
            
        Returns:
            Dict with response metrics (clicked, converted, time_spent, etc.)
        """
        
        if user_id not in self.current_users:
            self.generate_user(user_id)
        
        user = self.current_users[user_id]
        
        # Calculate contextual modifiers
        time_modifier = self._get_time_modifier(user, context.get('hour', 12))
        device_modifier = self._get_device_modifier(user, context.get('device', 'desktop'))
        
        # Calculate fatigue effect (users get tired of seeing ads)
        fatigue_modifier = max(0.1, 1.0 - user.fatigue_level)
        
        # Brand affinity effect
        brand_modifier = 1.0
        if 'brand_match' in ad_content:
            brand_modifier = 1.0 + (user.brand_affinity * ad_content['brand_match'])
        
        # Calculate click probability
        base_click_prob = user.click_propensity
        creative_quality = ad_content.get('creative_quality', 0.5)
        relevance = ad_content.get('relevance_score', 0.5)
        
        click_prob = (
            base_click_prob * 
            creative_quality * 
            relevance *
            time_modifier * 
            device_modifier * 
            fatigue_modifier *
            brand_modifier
        )
        
        # Apply RecSim probabilistic modeling if available
        if ed is not None:
            click_prob = ed.Beta(
                concentration1=click_prob * 10,
                concentration0=(1 - click_prob) * 10
            ).sample()
            click_prob = float(click_prob.numpy())
        
        clicked = np.random.random() < click_prob
        
        # If clicked, simulate conversion
        converted = False
        time_spent = 0.0
        revenue = 0.0
        
        if clicked:
            # Update user state
            user.fatigue_level = min(1.0, user.fatigue_level + 0.05)
            user.current_interest = min(1.0, user.current_interest + 0.1)
            
            # Time spent on landing page
            time_spent = np.random.exponential(user.attention_span)
            
            # Conversion probability
            price_factor = 1.0
            if 'price_shown' in ad_content:
                price_factor = 1.0 - (user.price_sensitivity * 
                                    min(1.0, ad_content['price_shown'] / user.budget))
            
            conv_prob = (
                user.conversion_propensity * 
                creative_quality *
                price_factor *
                brand_modifier *
                min(1.0, time_spent / user.attention_span)  # Engagement effect
            )
            
            if ed is not None:
                conv_prob = ed.Beta(
                    concentration1=conv_prob * 5,
                    concentration0=(1 - conv_prob) * 5
                ).sample()
                conv_prob = float(conv_prob.numpy())
            
            converted = np.random.random() < conv_prob
            
            if converted:
                # Calculate revenue based on user budget and product appeal
                revenue = np.random.gamma(
                    shape=2, 
                    scale=user.budget * creative_quality * 0.3
                )
                user.recent_purchases.append(ad_content.get('product_id', 'unknown'))
        
        # Store interaction history
        interaction = {
            'user_id': user_id,
            'segment': user.segment.value,
            'ad_content': ad_content,
            'context': context,
            'clicked': clicked,
            'converted': converted,
            'time_spent': time_spent,
            'revenue': revenue,
            'click_prob': click_prob
        }
        
        self.interaction_history.append(interaction)
        
        return {
            'clicked': clicked,
            'converted': converted,
            'time_spent': time_spent,
            'revenue': revenue,
            'click_probability': click_prob,
            'user_segment': user.segment.value,
            'fatigue_level': user.fatigue_level,
            'interest_level': user.current_interest
        }
    
    def _get_time_modifier(self, user: UserProfile, hour: int) -> float:
        """Get time-of-day modifier for user engagement"""
        time_period = self._get_time_period(hour)
        return user.time_preference.get(time_period, 1.0)
    
    def _get_device_modifier(self, user: UserProfile, device: str) -> float:
        """Get device-type modifier for user engagement"""
        return user.device_preference.get(device, 1.0)
    
    def _get_time_period(self, hour: int) -> str:
        """Convert hour to time period"""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'
    
    def get_user_analytics(self) -> Dict[str, Any]:
        """Generate analytics on user behavior patterns"""
        
        if not self.interaction_history:
            return {}
        
        segment_stats = {}
        for segment in UserSegment:
            segment_interactions = [
                i for i in self.interaction_history 
                if i['segment'] == segment.value
            ]
            
            if segment_interactions:
                segment_stats[segment.value] = {
                    'total_interactions': len(segment_interactions),
                    'click_rate': np.mean([i['clicked'] for i in segment_interactions]),
                    'conversion_rate': np.mean([
                        i['converted'] for i in segment_interactions if i['clicked']
                    ]) if any(i['clicked'] for i in segment_interactions) else 0,
                    'avg_revenue': np.mean([i['revenue'] for i in segment_interactions]),
                    'avg_time_spent': np.mean([i['time_spent'] for i in segment_interactions])
                }
        
        return {
            'total_interactions': len(self.interaction_history),
            'overall_ctr': np.mean([i['clicked'] for i in self.interaction_history]),
            'overall_conversion_rate': np.mean([
                i['converted'] for i in self.interaction_history if i['clicked']
            ]) if any(i['clicked'] for i in self.interaction_history) else 0,
            'segment_breakdown': segment_stats
        }
    
    def reset_user_states(self):
        """Reset fatigue and interest levels for all users"""
        for user in self.current_users.values():
            user.fatigue_level = max(0, user.fatigue_level - 0.2)
            user.current_interest = 0.5
            user.recent_purchases = []


def test_recsim_user_model():
    """Test the RecSim user model with various scenarios"""
    
    print("Testing RecSim User Model")
    print("=" * 50)
    
    model = RecSimUserModel()
    
    # Test different user segments
    test_scenarios = [
        {
            'ad_content': {
                'creative_quality': 0.8,
                'price_shown': 50.0,
                'brand_match': 0.9,
                'relevance_score': 0.7,
                'product_id': 'premium_sneakers'
            },
            'context': {
                'hour': 20,  # Evening
                'device': 'mobile'
            }
        },
        {
            'ad_content': {
                'creative_quality': 0.4,
                'price_shown': 200.0,
                'brand_match': 0.3,
                'relevance_score': 0.5,
                'product_id': 'expensive_gadget'
            },
            'context': {
                'hour': 14,  # Afternoon
                'device': 'desktop'
            }
        }
    ]
    
    # Run simulations for each user segment
    for segment in UserSegment:
        print(f"\nTesting {segment.value.upper()}:")
        
        results = []
        for i in range(100):  # 100 simulations per segment
            user_id = f"{segment.value}_{i}"
            model.generate_user(user_id, segment)
            
            # Test both scenarios
            for scenario_idx, scenario in enumerate(test_scenarios):
                result = model.simulate_ad_response(
                    user_id=user_id,
                    ad_content=scenario['ad_content'],
                    context=scenario['context']
                )
                result['scenario'] = scenario_idx
                results.append(result)
        
        # Calculate segment statistics
        scenario_0_results = [r for r in results if r['scenario'] == 0]
        scenario_1_results = [r for r in results if r['scenario'] == 1]
        
        print(f"  Scenario 1 (High Quality, Evening, Mobile):")
        print(f"    Click Rate: {np.mean([r['clicked'] for r in scenario_0_results]):.3f}")
        print(f"    Conversion Rate: {np.mean([r['converted'] for r in scenario_0_results if r['clicked']]):.3f}")
        print(f"    Avg Revenue: ${np.mean([r['revenue'] for r in scenario_0_results]):.2f}")
        
        print(f"  Scenario 2 (Low Quality, Afternoon, Desktop):")
        print(f"    Click Rate: {np.mean([r['clicked'] for r in scenario_1_results]):.3f}")
        print(f"    Conversion Rate: {np.mean([r['converted'] for r in scenario_1_results if r['clicked']]):.3f}")
        print(f"    Avg Revenue: ${np.mean([r['revenue'] for r in scenario_1_results]):.2f}")
    
    # Print overall analytics
    print(f"\nOverall Analytics:")
    analytics = model.get_user_analytics()
    print(f"Total Interactions: {analytics['total_interactions']}")
    print(f"Overall CTR: {analytics['overall_ctr']:.3f}")
    print(f"Overall Conversion Rate: {analytics['overall_conversion_rate']:.3f}")


if __name__ == "__main__":
    test_recsim_user_model()