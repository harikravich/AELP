"""
Environment wrappers for GAELP RL agents.
Provides interfaces between ad campaign environments and RL agents.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import random
from dataclasses import dataclass
import logging

# Import Criteo model for realistic CTR predictions - REQUIRED
from criteo_response_model import CriteoUserResponseModel

logger = logging.getLogger(__name__)


@dataclass
class AdCampaignState:
    """Represents the state of an ad campaign."""
    budget_spent: float
    budget_remaining: float
    impressions: int
    clicks: int
    conversions: int
    revenue: float
    ctr: float
    conversion_rate: float
    roas: float
    hour_of_day: int
    day_of_week: int
    campaign_age_hours: int
    creative_type_encoding: np.ndarray  # one-hot
    audience_segment_encoding: np.ndarray  # one-hot
    platform_encoding: np.ndarray  # one-hot
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for neural network input."""
        return np.concatenate([
            np.array([
                self.budget_spent / 1000.0,  # Normalize
                self.budget_remaining / 1000.0,
                self.impressions / 10000.0,
                self.clicks / 1000.0,
                self.conversions / 100.0,
                self.revenue / 10000.0,
                self.ctr,
                self.conversion_rate,
                self.roas / 10.0,  # Normalize ROAS
                self.hour_of_day / 24.0,
                self.day_of_week / 7.0,
                self.campaign_age_hours / 168.0,  # Week normalization
            ]),
            self.creative_type_encoding,
            self.audience_segment_encoding,
            self.platform_encoding
        ])


class AdCampaignEnvWrapper:
    """
    Wrapper for ad campaign environment compatible with RL agents.
    Simulates the ad campaign dynamics for training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize Criteo response model for realistic CTR predictions
        if CRITEO_MODEL_AVAILABLE:
            try:
                self.criteo_model = CriteoUserResponseModel()
                logger.info("CriteoUserResponseModel initialized in RL environment")
            except Exception as e:
                logger.warning(f"Failed to initialize CriteoUserResponseModel: {e}")
                self.criteo_model = None
        else:
            self.criteo_model = None
        
        # Action space: [bid_adjustment, budget_allocation, creative_index, audience_index]
        self.action_dim = 4
        
        # State dimensions
        self.base_features = 12
        self.creative_types = ['image', 'video', 'carousel', 'text']
        self.audience_segments = ['young_adults', 'professionals', 'families', 'seniors']
        self.platforms = ['google_ads', 'facebook_ads', 'tiktok_ads']
        
        self.state_dim = (
            self.base_features + 
            len(self.creative_types) + 
            len(self.audience_segments) + 
            len(self.platforms)
        )
        
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.budget_total = self.config.get('daily_budget', 100.0)
        self.budget_spent = 0.0
        self.impressions = 0
        self.clicks = 0
        self.conversions = 0
        self.revenue = 0.0
        self.hour = 0
        self.day = 0
        self.campaign_age = 0
        
        # Random initial selections
        self.current_creative = random.choice(self.creative_types)
        self.current_audience = random.choice(self.audience_segments)
        self.current_platform = random.choice(self.platforms)
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        ctr = self.clicks / max(1, self.impressions)
        cvr = self.conversions / max(1, self.clicks)
        roas = self.revenue / max(0.01, self.budget_spent)
        
        # One-hot encodings
        creative_encoding = np.zeros(len(self.creative_types))
        creative_encoding[self.creative_types.index(self.current_creative)] = 1
        
        audience_encoding = np.zeros(len(self.audience_segments))
        audience_encoding[self.audience_segments.index(self.current_audience)] = 1
        
        platform_encoding = np.zeros(len(self.platforms))
        platform_encoding[self.platforms.index(self.current_platform)] = 1
        
        state = AdCampaignState(
            budget_spent=self.budget_spent,
            budget_remaining=self.budget_total - self.budget_spent,
            impressions=self.impressions,
            clicks=self.clicks,
            conversions=self.conversions,
            revenue=self.revenue,
            ctr=ctr,
            conversion_rate=cvr,
            roas=roas,
            hour_of_day=self.hour % 24,
            day_of_week=self.day % 7,
            campaign_age_hours=self.campaign_age,
            creative_type_encoding=creative_encoding,
            audience_segment_encoding=audience_encoding,
            platform_encoding=platform_encoding
        )
        
        return state.to_array()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment.
        
        Action components:
        - action[0]: bid adjustment (-1 to 1)
        - action[1]: budget allocation (0 to 1)
        - action[2]: creative selection (continuous, discretized)
        - action[3]: audience selection (continuous, discretized)
        """
        # Parse actions
        bid_adjustment = np.clip(action[0], -1, 1)
        budget_allocation = np.clip(action[1], 0, 1)
        
        # Discretize creative and audience selections
        creative_idx = int(np.clip(action[2] * len(self.creative_types), 
                                   0, len(self.creative_types) - 1))
        audience_idx = int(np.clip(action[3] * len(self.audience_segments),
                                   0, len(self.audience_segments) - 1))
        
        self.current_creative = self.creative_types[creative_idx]
        self.current_audience = self.audience_segments[audience_idx]
        
        # Simulate campaign performance based on actions
        base_ctr = self._get_base_ctr()
        base_cvr = self._get_base_cvr()
        
        # Apply bid adjustment effects
        ctr_multiplier = 1.0 + (bid_adjustment * 0.3)  # ±30% CTR change
        
        # Allocate budget
        step_budget = min(
            budget_allocation * 10.0,  # Max $10 per step
            self.budget_total - self.budget_spent
        )
        
        # Simulate impressions based on budget
        cost_per_impression = 0.001 * (1.0 + bid_adjustment)  # $0.001 base CPM
        new_impressions = int(step_budget / cost_per_impression)
        
        # Simulate clicks
        expected_ctr = base_ctr * ctr_multiplier
        new_clicks = np.random.binomial(new_impressions, expected_ctr)
        
        # Simulate conversions
        new_conversions = np.random.binomial(new_clicks, base_cvr)
        
        # Calculate revenue (with some randomness)
        revenue_per_conversion = np.random.normal(50.0, 10.0)  # $50 average
        new_revenue = new_conversions * max(0, revenue_per_conversion)
        
        # Update state
        self.budget_spent += step_budget
        self.impressions += new_impressions
        self.clicks += new_clicks
        self.conversions += new_conversions
        self.revenue += new_revenue
        self.campaign_age += 1
        self.hour += 1
        
        # Calculate reward
        step_roas = new_revenue / max(0.01, step_budget)
        reward = self._calculate_reward(step_roas, expected_ctr, new_conversions)
        
        # Check if episode is done
        done = (self.budget_spent >= self.budget_total * 0.95 or 
                self.campaign_age >= 168)  # Week-long campaigns
        
        # Info dict
        info = {
            'step_budget': step_budget,
            'step_impressions': new_impressions,
            'step_clicks': new_clicks,
            'step_conversions': new_conversions,
            'step_revenue': new_revenue,
            'step_roas': step_roas,
            'total_roas': self.revenue / max(0.01, self.budget_spent),
            'creative': self.current_creative,
            'audience': self.current_audience,
            'platform': self.current_platform
        }
        
        return self._get_state(), reward, done, info
    
    def _get_base_ctr(self) -> float:
        """Get base CTR using Criteo model or use static calculations."""
        
        if self.criteo_model:
            try:
                # Prepare ad content for Criteo model
                ad_content = {
                    'category': 'parental_controls',  # GAELP domain
                    'brand': 'gaelp',
                    'price': 99.99,  # Typical GAELP subscription price
                    'creative_quality': 0.8  # High quality creative
                }
                
                # Map creative type to device preference
                device_mapping = {
                    'video': 'mobile',    # Video performs better on mobile
                    'image': 'desktop',   # Images work well on desktop
                    'carousel': 'mobile', # Carousel native to mobile
                    'text': 'desktop'     # Text ads common on search/desktop
                }
                
                # Map audience to user segment
                segment_mapping = {
                    'young_adults': 'tech_enthusiast',
                    'professionals': 'parents', 
                    'families': 'parents',
                    'seniors': 'price_conscious'
                }
                
                # Prepare context for Criteo model
                context = {
                    'device': device_mapping.get(self.current_creative, 'desktop'),
                    'hour': self.hour % 24,
                    'day_of_week': (self.hour // 24) % 7,
                    'session_duration': 120,  # Average session length
                    'page_views': 3,
                    'geo_region': 'US',
                    'user_segment': segment_mapping.get(self.current_audience, 'parents'),
                    'browser': 'chrome',
                    'os': 'windows'
                }
                
                # Get CTR prediction from Criteo model
                user_id = f"rl_user_{hash((self.current_creative, self.current_audience)) % 10000}"
                response = self.criteo_model.simulate_user_response(
                    user_id=user_id,
                    ad_content=ad_content,
                    context=context
                )
                
                predicted_ctr = response.get('predicted_ctr', 0.025)
                logger.debug(f"Criteo model predicted CTR: {predicted_ctr:.4f} for "
                           f"{self.current_creative}/{self.current_audience}")
                
                return float(predicted_ctr)
                
            except Exception as e:
                logger.warning(f"Error using Criteo model for CTR prediction: {e}")
                # Fall through to static matrix
        
        # Use static CTR matrix if needed
        ctr_matrix = {
            ('image', 'young_adults'): 0.025,
            ('image', 'professionals'): 0.032,
            ('image', 'families'): 0.028,
            ('image', 'seniors'): 0.018,
            ('video', 'young_adults'): 0.045,
            ('video', 'professionals'): 0.028,
            ('video', 'families'): 0.035,
            ('video', 'seniors'): 0.015,
            ('carousel', 'young_adults'): 0.038,
            ('carousel', 'professionals'): 0.042,
            ('carousel', 'families'): 0.040,
            ('carousel', 'seniors'): 0.022,
            ('text', 'young_adults'): 0.015,
            ('text', 'professionals'): 0.025,
            ('text', 'families'): 0.020,
            ('text', 'seniors'): 0.028,
        }
        
        base = ctr_matrix.get((self.current_creative, self.current_audience), 0.025)
        
        # Add time-of-day effects
        hour_multiplier = 1.0
        if 6 <= (self.hour % 24) <= 9:  # Morning
            hour_multiplier = 1.2
        elif 18 <= (self.hour % 24) <= 22:  # Evening
            hour_multiplier = 1.15
        
        return base * hour_multiplier
    
    def _get_base_cvr(self) -> float:
        """Get base conversion rate based on audience."""
        cvr_map = {
            'young_adults': 0.012,
            'professionals': 0.025,
            'families': 0.018,
            'seniors': 0.008
        }
        return cvr_map.get(self.current_audience, 0.015)
    
    def _calculate_reward(self, step_roas: float, ctr: float, conversions: int) -> float:
        """
        Calculate reward based on multiple objectives.
        Primary: ROAS
        Secondary: CTR, Conversions
        """
        # ROAS component (primary)
        roas_reward = np.tanh(step_roas / 3.0)  # Normalize around ROAS of 3
        
        # CTR component
        ctr_reward = np.tanh(ctr / 0.03)  # Normalize around 3% CTR
        
        # Conversion component
        conversion_reward = np.tanh(conversions / 5.0)  # Normalize around 5 conversions
        
        # Weighted combination
        reward = (
            1.0 * roas_reward +
            0.3 * ctr_reward +
            0.5 * conversion_reward
        )
        
        # Penalty for spending too fast
        if self.campaign_age < 24 and self.budget_spent > self.budget_total * 0.5:
            reward -= 0.5
        
        return reward
    
    def render(self) -> None:
        """Render current state (text-based)."""
        roas = self.revenue / max(0.01, self.budget_spent)
        ctr = self.clicks / max(1, self.impressions)
        cvr = self.conversions / max(1, self.clicks)
        
        print(f"\n{'='*50}")
        print(f"Campaign Status - Hour {self.campaign_age}")
        print(f"{'='*50}")
        print(f"Budget: ${self.budget_spent:.2f} / ${self.budget_total:.2f}")
        print(f"Impressions: {self.impressions:,}")
        print(f"Clicks: {self.clicks} (CTR: {ctr:.2%})")
        print(f"Conversions: {self.conversions} (CVR: {cvr:.2%})")
        print(f"Revenue: ${self.revenue:.2f}")
        print(f"ROAS: {roas:.2f}x")
        print(f"Creative: {self.current_creative}")
        print(f"Audience: {self.current_audience}")
        print(f"Platform: {self.current_platform}")
        print(f"{'='*50}")


class MultiCampaignEnvWrapper(AdCampaignEnvWrapper):
    """
    Extended environment that manages multiple campaigns simultaneously.
    Useful for meta-learning and portfolio optimization.
    """
    
    def __init__(self, num_campaigns: int = 3, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.num_campaigns = num_campaigns
        self.campaigns = []
        
    def reset(self) -> np.ndarray:
        """Reset all campaigns."""
        self.campaigns = []
        for i in range(self.num_campaigns):
            campaign = AdCampaignEnvWrapper(self.config)
            campaign.reset()
            self.campaigns.append(campaign)
        
        # Return concatenated states
        states = [c._get_state() for c in self.campaigns]
        return np.concatenate(states)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute actions for all campaigns.
        Actions should be shape (num_campaigns, action_dim)
        """
        actions = actions.reshape(self.num_campaigns, self.action_dim)
        
        states = []
        total_reward = 0
        all_done = True
        info = {'campaigns': []}
        
        for i, campaign in enumerate(self.campaigns):
            state, reward, done, campaign_info = campaign.step(actions[i])
            states.append(state)
            total_reward += reward
            all_done = all_done and done
            info['campaigns'].append(campaign_info)
        
        # Portfolio-level metrics
        total_spend = sum(c.budget_spent for c in self.campaigns)
        total_revenue = sum(c.revenue for c in self.campaigns)
        info['portfolio_roas'] = total_revenue / max(0.01, total_spend)
        
        return np.concatenate(states), total_reward, all_done, info