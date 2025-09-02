#!/usr/bin/env python3
"""
SHADOW MODE ENVIRONMENT
Simulates real auction and user behavior without spending real money
"""

import logging
import numpy as np
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from shadow_mode_state import DynamicEnrichedState, create_synthetic_state_for_testing
from discovery_engine import GA4RealTimeDataPipeline as DiscoveryEngine
from gaelp_parameter_manager import ParameterManager

logger = logging.getLogger(__name__)

@dataclass
class AuctionResult:
    """Result of a simulated auction"""
    won: bool
    position: float
    price_paid: float
    competing_bids: List[float]
    quality_score: float
    relevance_score: float
    
@dataclass
class UserInteractionResult:
    """Result of simulated user interaction"""
    clicked: bool
    converted: bool
    revenue: float
    time_on_site: float
    bounce_rate: float
    engagement_score: float

@dataclass
class ShadowEnvironmentMetrics:
    """Metrics tracking for shadow environment"""
    total_auctions: int = 0
    total_wins: int = 0
    total_spend: float = 0.0
    total_revenue: float = 0.0
    total_clicks: int = 0
    total_conversions: int = 0
    
    # Performance by segment
    segment_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Performance by channel
    channel_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Performance by creative
    creative_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Temporal patterns
    hourly_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    def update_auction(self, won: bool, spend: float):
        """Update auction metrics"""
        self.total_auctions += 1
        if won:
            self.total_wins += 1
            self.total_spend += spend
    
    def update_interaction(self, clicked: bool, converted: bool, revenue: float):
        """Update interaction metrics"""
        if clicked:
            self.total_clicks += 1
        if converted:
            self.total_conversions += 1
            self.total_revenue += revenue
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'win_rate': self.total_wins / max(1, self.total_auctions),
            'ctr': self.total_clicks / max(1, self.total_wins),
            'cvr': self.total_conversions / max(1, self.total_clicks),
            'roas': self.total_revenue / max(0.01, self.total_spend),
            'total_auctions': self.total_auctions,
            'total_spend': self.total_spend,
            'total_revenue': self.total_revenue
        }

class ShadowModeEnvironment:
    """
    Shadow mode environment that simulates real auction and user behavior
    without spending real money
    """
    
    def __init__(self,
                 parameter_manager: ParameterManager,
                 discovery_engine: Optional[DiscoveryEngine] = None,
                 realistic_simulation: bool = True):
        
        self.pm = parameter_manager
        self.discovery = discovery_engine or DiscoveryEngine(write_enabled=False)
        self.realistic_simulation = realistic_simulation
        
        # Environment state
        self.current_user_state: Optional[DynamicEnrichedState] = None
        self.current_context: Dict[str, Any] = {}
        self.episode_step = 0
        self.max_steps_per_episode = 100
        
        # Simulation parameters
        self.market_conditions = self._initialize_market_conditions()
        self.competitor_models = self._initialize_competitor_models()
        self.user_behavior_models = self._initialize_user_behavior_models()
        
        # Metrics tracking
        self.metrics = ShadowEnvironmentMetrics()
        self.episode_history = deque(maxlen=1000)
        
        # Discovered patterns for realistic simulation
        try:
            self.patterns = self.discovery.discover_all_patterns()
            logger.info(f"Loaded patterns from discovery engine")
        except Exception as e:
            # Fallback if discovery doesn't work properly
            logger.info(f"Using fallback patterns due to: {e}")
            from collections import namedtuple
            MockPatterns = namedtuple('MockPatterns', ['user_patterns', 'temporal_patterns'])
            self.patterns = MockPatterns(
                user_patterns={
                    'segments': {
                        'researching_parent': {'conversion_rate': 0.025, 'engagement_score': 0.6},
                        'concerned_parent': {'conversion_rate': 0.035, 'engagement_score': 0.7},
                        'crisis_parent': {'conversion_rate': 0.055, 'engagement_score': 0.9},
                        'proactive_parent': {'conversion_rate': 0.015, 'engagement_score': 0.4}
                    }
                },
                temporal_patterns={'peak_hours': [19, 20, 21]}
            )
        
        logger.info("Shadow mode environment initialized")
    
    def _initialize_market_conditions(self) -> Dict[str, Any]:
        """Initialize realistic market conditions"""
        return {
            'base_competition_level': 0.6,
            'competition_variance': 0.2,
            'market_saturation': 0.4,
            'seasonal_factors': {
                'morning': 0.8,
                'afternoon': 1.2,
                'evening': 1.5,
                'night': 0.5
            },
            'day_of_week_factors': {
                0: 1.0,  # Monday
                1: 1.1,  # Tuesday
                2: 1.2,  # Wednesday
                3: 1.1,  # Thursday
                4: 0.9,  # Friday
                5: 0.7,  # Saturday
                6: 0.8   # Sunday
            }
        }
    
    def _initialize_competitor_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize competitor bidding models"""
        return {
            'conservative_competitor': {
                'base_bid_multiplier': 0.8,
                'bid_variance': 0.1,
                'budget_constraint': 500.0,
                'quality_focus': 0.7
            },
            'aggressive_competitor': {
                'base_bid_multiplier': 1.3,
                'bid_variance': 0.3,
                'budget_constraint': 2000.0,
                'quality_focus': 0.4
            },
            'smart_competitor': {
                'base_bid_multiplier': 1.0,
                'bid_variance': 0.15,
                'budget_constraint': 1500.0,
                'quality_focus': 0.9
            }
        }
    
    def _initialize_user_behavior_models(self) -> Dict[str, Any]:
        """Initialize user behavior simulation models"""
        segments = self.patterns.user_patterns.get('segments', {})
        
        behavior_models = {}
        for segment_name, segment_data in segments.items():
            behavior_models[segment_name] = {
                'base_ctr': min(0.1, segment_data.get('engagement_score', 0.5) * 0.08),
                'base_cvr': segment_data.get('conversion_rate', 0.02),
                'price_sensitivity': np.random.beta(2, 3),
                'fatigue_sensitivity': np.random.beta(2, 2),
                'content_preferences': self._generate_content_preferences(segment_name),
                'temporal_preferences': self._generate_temporal_preferences(segment_name)
            }
        
        # Default behavior if no segments discovered
        if not behavior_models:
            behavior_models = {
                'default_user': {
                    'base_ctr': 0.02,
                    'base_cvr': 0.015,
                    'price_sensitivity': 0.5,
                    'fatigue_sensitivity': 0.3,
                    'content_preferences': {},
                    'temporal_preferences': {}
                }
            }
        
        return behavior_models
    
    def _generate_content_preferences(self, segment_name: str) -> Dict[str, float]:
        """Generate content preferences for segment"""
        if 'crisis' in segment_name.lower():
            return {
                'urgency_weight': 0.8,
                'authority_weight': 0.6,
                'social_proof_weight': 0.4,
                'fear_appeal_weight': 0.7
            }
        elif 'researching' in segment_name.lower():
            return {
                'urgency_weight': 0.3,
                'authority_weight': 0.9,
                'social_proof_weight': 0.6,
                'fear_appeal_weight': 0.2
            }
        elif 'concerned' in segment_name.lower():
            return {
                'urgency_weight': 0.6,
                'authority_weight': 0.7,
                'social_proof_weight': 0.8,
                'fear_appeal_weight': 0.5
            }
        else:
            return {
                'urgency_weight': 0.4,
                'authority_weight': 0.5,
                'social_proof_weight': 0.6,
                'fear_appeal_weight': 0.3
            }
    
    def _generate_temporal_preferences(self, segment_name: str) -> Dict[str, float]:
        """Generate temporal preferences for segment"""
        if 'crisis' in segment_name.lower():
            # Crisis users more likely to engage at any time
            return {
                'morning_preference': 0.8,
                'afternoon_preference': 0.9,
                'evening_preference': 1.2,
                'night_preference': 0.6,
                'weekend_preference': 0.8
            }
        elif 'researching' in segment_name.lower():
            # Researchers prefer business hours
            return {
                'morning_preference': 0.7,
                'afternoon_preference': 1.1,
                'evening_preference': 0.9,
                'night_preference': 0.3,
                'weekend_preference': 0.6
            }
        else:
            return {
                'morning_preference': 0.8,
                'afternoon_preference': 1.0,
                'evening_preference': 1.0,
                'night_preference': 0.4,
                'weekend_preference': 0.7
            }
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment for new episode"""
        self.episode_step = 0
        
        # Create new user state
        self.current_user_state = create_synthetic_state_for_testing(
            discovery_patterns=self.patterns.__dict__ if hasattr(self.patterns, '__dict__') else {}
        )
        
        # Create context
        self.current_context = self._generate_context()
        
        # Initial observation
        obs = {
            'state': self.current_user_state,
            'context': self.current_context
        }
        
        info = {
            'episode_start': True,
            'user_id': f"shadow_user_{int(time.time())}_{random.randint(1000, 9999)}"
        }
        
        return obs, info
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute action in shadow environment
        Returns: observation, reward, terminated, truncated, info
        """
        self.episode_step += 1
        
        # Extract action components
        bid_amount = getattr(action, 'bid_amount', 1.0)
        creative_id = getattr(action, 'creative_id', 0)
        channel = getattr(action, 'channel', 'display')
        
        # Simulate auction
        auction_result = self._simulate_auction(bid_amount, creative_id, channel)
        
        # Simulate user interaction if won
        interaction_result = None
        if auction_result.won:
            interaction_result = self._simulate_user_interaction(
                creative_id, channel, auction_result
            )
        
        # Calculate reward
        reward = self._calculate_reward(auction_result, interaction_result)
        
        # Update metrics
        self._update_metrics(auction_result, interaction_result, bid_amount, creative_id, channel)
        
        # Update user state
        self._update_user_state(auction_result, interaction_result, creative_id, channel)
        
        # Check if episode is done
        terminated = self._check_termination_conditions()
        truncated = self.episode_step >= self.max_steps_per_episode
        
        # Create next observation
        next_obs = {
            'state': self.current_user_state,
            'context': self._generate_context()
        }
        
        # Create info
        info = {
            'auction_result': auction_result,
            'interaction_result': interaction_result,
            'spend': auction_result.price_paid if auction_result.won else 0.0,
            'revenue': interaction_result.revenue if interaction_result else 0.0,
            'metrics': self.metrics.get_summary(),
            'step': self.episode_step
        }
        
        return next_obs, reward, terminated, truncated, info
    
    def _simulate_auction(self, bid_amount: float, creative_id: int, channel: str) -> AuctionResult:
        """Simulate realistic auction dynamics"""
        
        # Get market conditions
        competition_factor = self._get_competition_factor()
        
        # Generate competitor bids
        competitor_bids = []
        for competitor_name, competitor_model in self.competitor_models.items():
            base_bid = self._estimate_competitor_base_bid(channel)
            competitor_bid = base_bid * competitor_model['base_bid_multiplier']
            competitor_bid += np.random.normal(0, competitor_bid * competitor_model['bid_variance'])
            competitor_bid = max(0.1, competitor_bid)  # Minimum bid
            competitor_bids.append(competitor_bid)
        
        # Add market noise
        competitor_bids.extend(np.random.lognormal(0.5, 0.5, size=random.randint(2, 8)))
        
        # Sort bids to determine position
        all_bids = [(bid_amount, 'our_bid')] + [(bid, f'competitor_{i}') for i, bid in enumerate(competitor_bids)]
        all_bids.sort(key=lambda x: x[0], reverse=True)
        
        # Find our position
        our_position = next((i + 1 for i, (bid, bidder) in enumerate(all_bids) if bidder == 'our_bid'), len(all_bids))
        
        # Determine if we won (top 10 positions typically win)
        won = our_position <= 10 and bid_amount >= 0.5  # Minimum viable bid
        
        # Calculate price paid (second price auction)
        if won and our_position < len(all_bids):
            price_paid = all_bids[our_position][0] + 0.01  # Second price + increment
        elif won:
            price_paid = bid_amount * 0.8  # If we're the only bidder
        else:
            price_paid = 0.0
        
        # Quality score based on creative and targeting
        quality_score = self._calculate_quality_score(creative_id, channel)
        
        # Relevance score
        relevance_score = self._calculate_relevance_score(creative_id, channel)
        
        # Adjust position based on quality (Google Ads style)
        if won:
            quality_adjustment = (quality_score - 0.5) * 2  # -1 to 1 range
            our_position = max(1.0, our_position + quality_adjustment)
        
        return AuctionResult(
            won=won,
            position=float(our_position),
            price_paid=price_paid,
            competing_bids=competitor_bids,
            quality_score=quality_score,
            relevance_score=relevance_score
        )
    
    def _simulate_user_interaction(self, creative_id: int, channel: str, auction_result: AuctionResult) -> UserInteractionResult:
        """Simulate realistic user interaction"""
        
        # Get user behavior model for current segment
        segment_name = self.current_user_state.segment_name
        behavior_model = self.user_behavior_models.get(segment_name, self.user_behavior_models.get('default_user', {}))
        
        # Base CTR from behavior model
        base_ctr = behavior_model.get('base_ctr', 0.02)
        
        # Adjust CTR based on various factors
        ctr_factors = []
        
        # Position factor (higher positions get more clicks)
        position_factor = max(0.1, 2.0 / auction_result.position)
        ctr_factors.append(position_factor)
        
        # Quality factor
        quality_factor = 0.5 + auction_result.quality_score
        ctr_factors.append(quality_factor)
        
        # Relevance factor
        relevance_factor = 0.5 + auction_result.relevance_score
        ctr_factors.append(relevance_factor)
        
        # Channel factor
        channel_factors = {
            'paid_search': 1.2,
            'display': 0.6,
            'social': 0.8,
            'email': 1.4,
            'organic': 1.0
        }
        channel_factor = channel_factors.get(channel, 0.8)
        ctr_factors.append(channel_factor)
        
        # Temporal factor
        hour = self.current_user_state.hour_of_day
        temporal_factor = self._get_temporal_factor(hour, behavior_model)
        ctr_factors.append(temporal_factor)
        
        # Fatigue factor
        fatigue = self.current_user_state.creative_fatigue
        fatigue_sensitivity = behavior_model.get('fatigue_sensitivity', 0.3)
        fatigue_factor = 1.0 - (fatigue * fatigue_sensitivity)
        ctr_factors.append(max(0.1, fatigue_factor))
        
        # Creative content factor
        content_factor = self._get_creative_content_factor(creative_id, segment_name, behavior_model)
        ctr_factors.append(content_factor)
        
        # Calculate final CTR
        final_ctr = base_ctr * np.prod(ctr_factors)
        final_ctr = min(0.5, max(0.001, final_ctr))  # Reasonable bounds
        
        # Determine if clicked
        clicked = np.random.random() < final_ctr
        
        # If clicked, determine if converted
        converted = False
        revenue = 0.0
        
        if clicked:
            base_cvr = behavior_model.get('base_cvr', 0.015)
            
            # CVR factors
            cvr_factors = []
            
            # Journey stage factor
            stage_multipliers = {0: 0.3, 1: 0.6, 2: 1.0, 3: 1.8, 4: 3.0}
            stage_factor = stage_multipliers.get(self.current_user_state.stage, 1.0)
            cvr_factors.append(stage_factor)
            
            # Channel effectiveness for conversion
            channel_cvr_factors = {
                'paid_search': 1.3,
                'email': 1.2,
                'organic': 1.1,
                'social': 0.9,
                'display': 0.7
            }
            channel_cvr_factor = channel_cvr_factors.get(channel, 1.0)
            cvr_factors.append(channel_cvr_factor)
            
            # Content relevance for conversion
            cvr_factors.append(content_factor)
            
            # Quality and relevance boost conversion
            cvr_factors.append(1.0 + (auction_result.quality_score * 0.5))
            cvr_factors.append(1.0 + (auction_result.relevance_score * 0.3))
            
            # Calculate final CVR
            final_cvr = base_cvr * np.prod(cvr_factors)
            final_cvr = min(0.3, max(0.001, final_cvr))
            
            converted = np.random.random() < final_cvr
            
            # Calculate revenue if converted
            if converted:
                base_revenue = self.current_user_state.segment_avg_ltv
                revenue_variance = np.random.lognormal(0, 0.4)  # Revenue variance
                revenue = base_revenue * revenue_variance
                revenue = max(10.0, min(1000.0, revenue))  # Reasonable bounds
        
        # Additional interaction metrics
        time_on_site = np.random.exponential(30) if clicked else 0.0  # seconds
        bounce_rate = 0.8 if not clicked else (0.3 if converted else 0.6)
        engagement_score = (1.0 - bounce_rate) * (1.0 + np.log1p(time_on_site / 60.0))
        
        return UserInteractionResult(
            clicked=clicked,
            converted=converted,
            revenue=revenue,
            time_on_site=time_on_site,
            bounce_rate=bounce_rate,
            engagement_score=engagement_score
        )
    
    def _get_competition_factor(self) -> float:
        """Get current competition factor based on time and market conditions"""
        hour = self.current_user_state.hour_of_day
        day_of_week = self.current_user_state.day_of_week
        
        base_competition = self.market_conditions['base_competition_level']
        
        # Time-based competition
        time_of_day_factor = self.market_conditions['seasonal_factors'].get(
            self._get_time_period(hour), 1.0
        )
        
        # Day of week factor
        dow_factor = self.market_conditions['day_of_week_factors'].get(day_of_week, 1.0)
        
        # Add random variance
        variance = np.random.normal(0, self.market_conditions['competition_variance'])
        
        return base_competition * time_of_day_factor * dow_factor * (1 + variance)
    
    def _get_time_period(self, hour: int) -> str:
        """Convert hour to time period"""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 23:
            return 'evening'
        else:
            return 'night'
    
    def _estimate_competitor_base_bid(self, channel: str) -> float:
        """Estimate competitor base bid for channel"""
        channel_base_bids = {
            'paid_search': 2.5,
            'display': 1.2,
            'social': 1.8,
            'email': 0.8,
            'organic': 0.0  # No bidding for organic
        }
        
        base_bid = channel_base_bids.get(channel, 1.5)
        
        # Add market condition adjustment
        competition_factor = self._get_competition_factor()
        
        return base_bid * competition_factor
    
    def _calculate_quality_score(self, creative_id: int, channel: str) -> float:
        """Calculate quality score for ad"""
        
        # Base quality
        base_quality = 0.5
        
        # Creative-specific quality (simulated)
        creative_quality = 0.3 + (creative_id % 100) / 200.0  # 0.3 to 0.8
        
        # Channel match quality
        creative_channel_match = np.random.beta(2, 2)
        
        # Landing page quality (simulated)
        landing_page_quality = 0.4 + np.random.beta(3, 2) * 0.4  # 0.4 to 0.8
        
        # Historical performance boost
        if hasattr(self, 'creative_performance') and creative_id in self.creative_performance:
            perf = self.creative_performance[creative_id]
            historical_boost = (perf.get('ctr', 0.02) - 0.02) * 5  # Convert CTR to quality boost
            historical_boost = max(-0.2, min(0.2, historical_boost))
        else:
            historical_boost = 0.0
        
        quality_score = (
            base_quality * 0.2 +
            creative_quality * 0.3 +
            creative_channel_match * 0.2 +
            landing_page_quality * 0.2 +
            historical_boost * 0.1
        )
        
        return max(0.1, min(1.0, quality_score))
    
    def _calculate_relevance_score(self, creative_id: int, channel: str) -> float:
        """Calculate relevance score for targeting"""
        
        # Segment relevance
        segment_name = self.current_user_state.segment_name
        behavior_model = self.user_behavior_models.get(segment_name, {})
        content_prefs = behavior_model.get('content_preferences', {})
        
        # Simulated creative content alignment
        creative_relevance = np.mean(list(content_prefs.values())) if content_prefs else 0.5
        
        # Channel-user match
        device = self.current_user_state.device
        channel_device_match = {
            ('mobile', 'social'): 0.9,
            ('mobile', 'display'): 0.7,
            ('desktop', 'paid_search'): 0.8,
            ('desktop', 'email'): 0.9,
            ('tablet', 'display'): 0.8
        }
        channel_match = channel_device_match.get((device, channel), 0.6)
        
        # Temporal relevance
        hour = self.current_user_state.hour_of_day
        temporal_prefs = behavior_model.get('temporal_preferences', {})
        time_period = self._get_time_period(hour)
        temporal_relevance = temporal_prefs.get(f'{time_period}_preference', 0.5)
        
        relevance_score = (
            creative_relevance * 0.5 +
            channel_match * 0.3 +
            temporal_relevance * 0.2
        )
        
        return max(0.1, min(1.0, relevance_score))
    
    def _get_temporal_factor(self, hour: int, behavior_model: Dict[str, Any]) -> float:
        """Get temporal engagement factor"""
        time_period = self._get_time_period(hour)
        temporal_prefs = behavior_model.get('temporal_preferences', {})
        return temporal_prefs.get(f'{time_period}_preference', 1.0)
    
    def _get_creative_content_factor(self, creative_id: int, segment_name: str, behavior_model: Dict[str, Any]) -> float:
        """Get creative content alignment factor"""
        content_prefs = behavior_model.get('content_preferences', {})
        
        if not content_prefs:
            return 1.0
        
        # Simulate creative content features based on ID
        simulated_features = {
            'urgency_level': (creative_id * 17) % 100 / 100.0,
            'authority_level': (creative_id * 23) % 100 / 100.0,
            'social_proof_level': (creative_id * 31) % 100 / 100.0,
            'fear_appeal_level': (creative_id * 43) % 100 / 100.0
        }
        
        # Calculate alignment
        alignment_score = 0.0
        total_weight = 0.0
        
        for pref_key, pref_weight in content_prefs.items():
            feature_key = pref_key.replace('_weight', '_level')
            if feature_key in simulated_features:
                alignment = simulated_features[feature_key] * pref_weight
                alignment_score += alignment
                total_weight += pref_weight
        
        if total_weight > 0:
            alignment_score /= total_weight
        else:
            alignment_score = 0.5
        
        return 0.5 + alignment_score * 0.5  # Scale to 0.5-1.0
    
    def _calculate_reward(self, auction_result: AuctionResult, interaction_result: Optional[UserInteractionResult]) -> float:
        """Calculate reward for the action"""
        reward = 0.0
        
        # Auction success reward
        if auction_result.won:
            # Position reward (better positions = higher reward)
            position_reward = (11 - min(10, auction_result.position)) / 10.0
            reward += position_reward * 2.0
            
            # Quality reward
            quality_reward = auction_result.quality_score * 1.0
            reward += quality_reward
            
            # Relevance reward
            relevance_reward = auction_result.relevance_score * 1.0
            reward += relevance_reward
            
            # Cost efficiency reward
            if auction_result.price_paid > 0:
                # Reward for paying less than bid
                efficiency = max(0, 1.0 - auction_result.price_paid / max(0.01, auction_result.price_paid + 1.0))
                reward += efficiency * 1.0
        else:
            # Small penalty for losing auction
            reward -= 0.2
        
        # Interaction rewards
        if interaction_result:
            if interaction_result.clicked:
                reward += 3.0  # Click reward
                
                if interaction_result.converted:
                    # Conversion reward based on profit
                    profit = interaction_result.revenue - auction_result.price_paid
                    conversion_reward = profit * 0.1  # Scale profit to reasonable reward
                    reward += conversion_reward
            
            # Engagement reward
            engagement_reward = interaction_result.engagement_score * 0.5
            reward += engagement_reward
        
        # Journey progression reward
        if self.current_user_state.stage < 4:  # Not yet converted
            stage_value = self.current_user_state.stage * 0.5
            reward += stage_value
        
        # Segment-specific rewards
        segment_cvr = self.current_user_state.segment_cvr
        if segment_cvr > 0.03:  # High-value segment
            reward += 1.0
        
        return reward
    
    def _update_metrics(self, auction_result: AuctionResult, interaction_result: Optional[UserInteractionResult],
                       bid_amount: float, creative_id: int, channel: str):
        """Update environment metrics"""
        
        # Update auction metrics
        self.metrics.update_auction(
            won=auction_result.won,
            spend=auction_result.price_paid if auction_result.won else 0.0
        )
        
        # Update interaction metrics
        if interaction_result:
            self.metrics.update_interaction(
                clicked=interaction_result.clicked,
                converted=interaction_result.converted,
                revenue=interaction_result.revenue
            )
        
        # Update segment performance
        segment_name = self.current_user_state.segment_name
        if segment_name not in self.metrics.segment_performance:
            self.metrics.segment_performance[segment_name] = {
                'impressions': 0, 'clicks': 0, 'conversions': 0, 'spend': 0, 'revenue': 0
            }
        
        seg_perf = self.metrics.segment_performance[segment_name]
        if auction_result.won:
            seg_perf['impressions'] += 1
            seg_perf['spend'] += auction_result.price_paid
            
            if interaction_result and interaction_result.clicked:
                seg_perf['clicks'] += 1
                if interaction_result.converted:
                    seg_perf['conversions'] += 1
                    seg_perf['revenue'] += interaction_result.revenue
        
        # Update channel performance
        if channel not in self.metrics.channel_performance:
            self.metrics.channel_performance[channel] = {
                'impressions': 0, 'clicks': 0, 'conversions': 0, 'spend': 0, 'revenue': 0
            }
        
        chan_perf = self.metrics.channel_performance[channel]
        if auction_result.won:
            chan_perf['impressions'] += 1
            chan_perf['spend'] += auction_result.price_paid
            
            if interaction_result and interaction_result.clicked:
                chan_perf['clicks'] += 1
                if interaction_result.converted:
                    chan_perf['conversions'] += 1
                    chan_perf['revenue'] += interaction_result.revenue
        
        # Update creative performance
        if creative_id not in self.metrics.creative_performance:
            self.metrics.creative_performance[creative_id] = {
                'impressions': 0, 'clicks': 0, 'conversions': 0, 'spend': 0, 'revenue': 0
            }
        
        crea_perf = self.metrics.creative_performance[creative_id]
        if auction_result.won:
            crea_perf['impressions'] += 1
            crea_perf['spend'] += auction_result.price_paid
            
            if interaction_result and interaction_result.clicked:
                crea_perf['clicks'] += 1
                if interaction_result.converted:
                    crea_perf['conversions'] += 1
                    crea_perf['revenue'] += interaction_result.revenue
    
    def _update_user_state(self, auction_result: AuctionResult, interaction_result: Optional[UserInteractionResult],
                          creative_id: int, channel: str):
        """Update user state based on interaction"""
        
        # Update touchpoints
        if auction_result.won:
            self.current_user_state.touchpoints_seen += 1
        
        # Update creative fatigue
        if auction_result.won and interaction_result:
            fatigue_increase = 0.1 if interaction_result.clicked else 0.05
            self.current_user_state.creative_fatigue = min(1.0, 
                self.current_user_state.creative_fatigue + fatigue_increase)
        
        # Update journey stage based on interaction
        if interaction_result and interaction_result.clicked:
            if interaction_result.converted:
                self.current_user_state.stage = 4  # Converted
            elif self.current_user_state.stage < 3:
                # Progress through stages
                stage_progression_prob = 0.3 + (interaction_result.engagement_score * 0.2)
                if np.random.random() < stage_progression_prob:
                    self.current_user_state.stage = min(4, self.current_user_state.stage + 1)
        
        # Update conversion probability
        if interaction_result:
            if interaction_result.converted:
                # Reset probability after conversion
                self.current_user_state.conversion_probability = 0.01
            elif interaction_result.clicked:
                # Increase probability after click
                self.current_user_state.conversion_probability = min(0.1, 
                    self.current_user_state.conversion_probability * 1.2)
        
        # Update last creative
        if auction_result.won:
            self.current_user_state.last_creative_id = creative_id
        
        # Update channel attribution
        if auction_result.won:
            self.current_user_state.last_touch_channel = channel
    
    def _generate_context(self) -> Dict[str, Any]:
        """Generate current context"""
        competition_factor = self._get_competition_factor()
        
        return {
            'competition_level': competition_factor,
            'avg_competitor_bid': self._estimate_competitor_base_bid(
                self.current_user_state.channel if self.current_user_state else 'display'
            ),
            'is_peak_hour': self.current_user_state.is_peak_hour if self.current_user_state else False,
            'daily_budget': 1000.0,
            'budget_spent': np.random.uniform(0, 800),
            'time_remaining': np.random.uniform(1, 12),
            'market_saturation': self.market_conditions['market_saturation']
        }
    
    def _check_termination_conditions(self) -> bool:
        """Check if episode should terminate"""
        
        # Terminate if user converted
        if self.current_user_state.stage == 4:
            return True
        
        # Terminate if user is highly fatigued
        if self.current_user_state.creative_fatigue > 0.9:
            return True
        
        # Terminate if too many touchpoints without progression
        if (self.current_user_state.touchpoints_seen > 15 and 
            self.current_user_state.stage < 2):
            return True
        
        return False
    
    def get_environment_metrics(self) -> Dict[str, Any]:
        """Get comprehensive environment metrics"""
        summary = self.metrics.get_summary()
        
        # Add detailed breakdowns
        summary['segment_breakdown'] = {}
        for segment, perf in self.metrics.segment_performance.items():
            if perf['impressions'] > 0:
                summary['segment_breakdown'][segment] = {
                    'ctr': perf['clicks'] / perf['impressions'],
                    'cvr': perf['conversions'] / max(1, perf['clicks']),
                    'roas': perf['revenue'] / max(0.01, perf['spend']),
                    'impressions': perf['impressions']
                }
        
        summary['channel_breakdown'] = {}
        for channel, perf in self.metrics.channel_performance.items():
            if perf['impressions'] > 0:
                summary['channel_breakdown'][channel] = {
                    'ctr': perf['clicks'] / perf['impressions'],
                    'cvr': perf['conversions'] / max(1, perf['clicks']),
                    'roas': perf['revenue'] / max(0.01, perf['spend']),
                    'impressions': perf['impressions']
                }
        
        return summary

if __name__ == "__main__":
    # Demo usage
    print("Shadow Mode Environment Demo")
    print("=" * 40)
    
    # Initialize environment
    pm = ParameterManager()
    env = ShadowModeEnvironment(pm)
    
    # Run a few steps
    obs, info = env.reset()
    print(f"Initial user: {info.get('user_id')}")
    print(f"Initial state: {obs['state'].get_state_summary()}")
    
    # Simulate some actions
    class SimpleAction:
        def __init__(self, bid, creative, channel):
            self.bid_amount = bid
            self.creative_id = creative
            self.channel = channel
    
    for step in range(5):
        action = SimpleAction(
            bid=np.random.uniform(1.0, 5.0),
            creative=np.random.randint(0, 50),
            channel=np.random.choice(['paid_search', 'display', 'social'])
        )
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: ${action.bid_amount:.2f} bid, creative {action.creative_id}, {action.channel}")
        print(f"  Result: Won={info['auction_result'].won}, Reward={reward:.2f}")
        if info.get('interaction_result'):
            ir = info['interaction_result']
            print(f"  Interaction: Clicked={ir.clicked}, Converted={ir.converted}, Revenue=${ir.revenue:.2f}")
        
        if terminated or truncated:
            print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    # Print final metrics
    print(f"\nFinal Metrics:")
    metrics = env.get_environment_metrics()
    print(f"  Win Rate: {metrics['win_rate']:.3f}")
    print(f"  CTR: {metrics['ctr']:.3f}")
    print(f"  CVR: {metrics['cvr']:.3f}")
    print(f"  ROAS: {metrics['roas']:.2f}x")