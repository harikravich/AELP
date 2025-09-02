#!/usr/bin/env python3
"""
SHADOW MODE STATE MANAGEMENT
Dynamic state class optimized for shadow mode testing
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class DynamicEnrichedState:
    """
    Dynamic enriched state for shadow mode testing
    Compatible with both production and shadow models
    """
    
    # Core journey state
    stage: int = 0  # 0=unaware, 1=aware, 2=considering, 3=intent, 4=converted
    touchpoints_seen: int = 0
    days_since_first_touch: float = 0.0
    
    # Segment information (discovered dynamically)
    segment_name: str = "researching_parent"
    segment_cvr: float = 0.02
    segment_engagement: float = 0.5
    segment_index: int = 0
    
    # Device and channel context
    device: str = "mobile"  # mobile, desktop, tablet
    channel: str = "organic"  # organic, paid_search, social, display, email
    device_index: int = 0
    channel_index: int = 0
    channel_performance: float = 0.5
    channel_attribution_credit: float = 0.0
    
    # Creative performance and content
    last_creative_id: int = 0
    creative_ctr: float = 0.02
    creative_cvr: float = 0.01
    creative_fatigue: float = 0.0
    creative_diversity_score: float = 0.5
    
    # Creative content features (from analyzer)
    creative_headline_sentiment: float = 0.0  # -1 to 1
    creative_urgency_score: float = 0.0  # 0 to 1
    creative_cta_strength: float = 0.5  # 0 to 1
    creative_uses_social_proof: float = 0.0  # 0 or 1
    creative_uses_authority: float = 0.0  # 0 or 1
    creative_message_frame_score: float = 0.5
    creative_predicted_ctr: float = 0.02
    creative_fatigue_resistance: float = 0.5
    
    # Temporal patterns
    hour_of_day: int = 12
    day_of_week: int = 0
    is_peak_hour: bool = False
    seasonality_factor: float = 1.0
    
    # Competition and auction context
    competition_level: float = 0.5
    avg_competitor_bid: float = 2.0
    win_rate_last_10: float = 0.5
    avg_position_last_10: float = 5.0
    
    # Budget and pacing
    budget_spent_ratio: float = 0.0
    time_in_day_ratio: float = 0.5
    pacing_factor: float = 1.0
    remaining_budget: float = 1000.0
    
    # Identity resolution
    cross_device_confidence: float = 0.0
    num_devices_seen: int = 1
    is_returning_user: bool = False
    is_logged_in: bool = False
    
    # Attribution signals
    first_touch_channel: str = "organic"
    last_touch_channel: str = "organic"
    touchpoint_credits: List[float] = field(default_factory=list)
    expected_conversion_value: float = 100.0
    
    # A/B test context
    ab_test_variant: int = 0
    variant_performance: float = 0.0
    
    # Delayed conversion signals
    conversion_probability: float = 0.02
    days_to_conversion_estimate: float = 7.0
    has_scheduled_conversion: bool = False
    segment_avg_ltv: float = 100.0
    
    # Competitor exposure
    competitor_impressions_seen: int = 0
    competitor_fatigue_level: float = 0.0
    
    # Shadow mode specific fields
    shadow_mode: bool = True
    model_prediction_confidence: float = 0.5
    state_generation_time: datetime = field(default_factory=datetime.now)
    
    # Dynamic discovered segments mapping
    _discovered_segments: Dict[str, int] = field(default_factory=dict)
    _discovered_channels: Dict[str, int] = field(default_factory=dict)
    _discovered_devices: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize dynamic mappings"""
        if not self._discovered_segments:
            self._discovered_segments = {
                'researching_parent': 0,
                'concerned_parent': 1,
                'crisis_parent': 2,
                'proactive_parent': 3
            }
        
        if not self._discovered_channels:
            self._discovered_channels = {
                'organic': 0,
                'paid_search': 1,
                'social': 2,
                'display': 3,
                'email': 4
            }
        
        if not self._discovered_devices:
            self._discovered_devices = {
                'mobile': 0,
                'desktop': 1,
                'tablet': 2
            }
        
        # Update indices based on current values
        self.segment_index = self._discovered_segments.get(self.segment_name, 0)
        self.channel_index = self._discovered_channels.get(self.channel, 0)
        self.device_index = self._discovered_devices.get(self.device, 0)
    
    def update_discovered_mappings(self, 
                                  segments: Optional[Dict[str, int]] = None,
                                  channels: Optional[Dict[str, int]] = None,
                                  devices: Optional[Dict[str, int]] = None):
        """Update discovered mappings from discovery engine"""
        if segments:
            self._discovered_segments = segments
        if channels:
            self._discovered_channels = channels
        if devices:
            self._discovered_devices = devices
        
        # Update current indices
        self.segment_index = self._discovered_segments.get(self.segment_name, 0)
        self.channel_index = self._discovered_channels.get(self.channel, 0)
        self.device_index = self._discovered_devices.get(self.device, 0)
    
    def to_vector(self) -> np.ndarray:
        """Convert to neural network input vector (compatible with original implementation)"""
        num_segments = max(len(self._discovered_segments), 4)
        num_channels = max(len(self._discovered_channels), 5)
        
        return np.array([
            # Journey features (5)
            self.stage / 4.0,
            min(self.touchpoints_seen / 20.0, 1.0),
            min(self.days_since_first_touch / 14.0, 1.0),
            float(self.is_returning_user),
            self.conversion_probability,
            
            # Segment features (3)
            self.segment_index / float(num_segments),
            self.segment_cvr,
            self.segment_engagement,
            
            # Device/channel features (4)
            self.device_index / 2.0,
            self.channel_index / float(num_channels),
            self.channel_performance,
            self.channel_attribution_credit,
            
            # Creative features (13)
            self.last_creative_id / 50.0,  # Normalize by max creatives
            self.creative_ctr,
            self.creative_cvr,
            self.creative_fatigue,
            self.creative_diversity_score,
            (self.creative_headline_sentiment + 1.0) / 2.0,  # Normalize -1,1 to 0,1
            self.creative_urgency_score,
            self.creative_cta_strength,
            self.creative_uses_social_proof,
            self.creative_uses_authority,
            self.creative_message_frame_score,
            self.creative_predicted_ctr,
            self.creative_fatigue_resistance,
            
            # Temporal features (4)
            self.hour_of_day / 23.0,
            self.day_of_week / 6.0,
            float(self.is_peak_hour),
            self.seasonality_factor,
            
            # Competition features (4)
            self.competition_level,
            min(self.avg_competitor_bid / 10.0, 1.0),  # Normalize by max expected bid
            self.win_rate_last_10,
            self.avg_position_last_10 / 10.0,
            
            # Budget features (4)
            self.budget_spent_ratio,
            self.time_in_day_ratio,
            self.pacing_factor,
            min(self.remaining_budget / 1000.0, 1.0),
            
            # Identity features (3)
            self.cross_device_confidence,
            min(self.num_devices_seen / 5.0, 1.0),
            float(self.is_logged_in),
            
            # Attribution features (4)
            self._discovered_channels.get(self.first_touch_channel, 0) / float(num_channels),
            self._discovered_channels.get(self.last_touch_channel, 0) / float(num_channels),
            min(len(self.touchpoint_credits) / 10.0, 1.0),
            self.expected_conversion_value / 200.0,
            
            # A/B test features (2)
            self.ab_test_variant / 10.0,
            self.variant_performance,
            
            # Delayed conversion features (3)
            self.segment_avg_ltv / 200.0,
            min(self.days_to_conversion_estimate / 14.0, 1.0),
            float(self.has_scheduled_conversion),
            
            # Competitor features (2)
            min(self.competitor_impressions_seen / 10.0, 1.0),
            self.competitor_fatigue_level,
            
            # Shadow mode features (2)
            float(self.shadow_mode),
            self.model_prediction_confidence
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, list):
                result[key] = value
            else:
                result[key] = value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DynamicEnrichedState':
        """Create from dictionary"""
        # Convert datetime string back if present
        if 'state_generation_time' in data and isinstance(data['state_generation_time'], str):
            data['state_generation_time'] = datetime.fromisoformat(data['state_generation_time'])
        
        # Handle touchpoint_credits list
        if 'touchpoint_credits' not in data:
            data['touchpoint_credits'] = []
        
        return cls(**data)
    
    def clone_for_shadow(self, model_id: str = "") -> 'DynamicEnrichedState':
        """Create a clone for shadow testing"""
        clone_data = self.to_dict()
        clone_data['shadow_mode'] = True
        clone_data['state_generation_time'] = datetime.now()
        clone_data['model_prediction_confidence'] = np.random.beta(2, 2)  # Vary confidence
        
        return DynamicEnrichedState.from_dict(clone_data)
    
    def update_from_discovery(self, discovery_patterns: Dict[str, Any]):
        """Update state with latest discovery patterns"""
        # Update segment information
        if 'segments' in discovery_patterns.get('user_patterns', {}):
            segments = discovery_patterns['user_patterns']['segments']
            if self.segment_name in segments:
                segment_data = segments[self.segment_name]
                self.segment_cvr = segment_data.get('conversion_rate', self.segment_cvr)
                self.segment_engagement = segment_data.get('engagement_score', self.segment_engagement)
        
        # Update temporal patterns
        if 'peak_hours' in discovery_patterns.get('temporal_patterns', {}):
            peak_hours = discovery_patterns['temporal_patterns']['peak_hours']
            self.is_peak_hour = self.hour_of_day in peak_hours
        
        # Update seasonality
        if 'seasonality_factor' in discovery_patterns.get('temporal_patterns', {}):
            self.seasonality_factor = discovery_patterns['temporal_patterns']['seasonality_factor']
    
    def get_state_summary(self) -> str:
        """Get human-readable state summary"""
        return f"""
DynamicEnrichedState Summary:
  User Journey: Stage {self.stage}, {self.touchpoints_seen} touchpoints, {self.days_since_first_touch:.1f} days
  Segment: {self.segment_name} (CVR: {self.segment_cvr:.3f})
  Context: {self.device} {self.channel} at {self.hour_of_day}:00
  Creative: ID {self.last_creative_id} (fatigue: {self.creative_fatigue:.3f})
  Budget: {self.budget_spent_ratio:.1%} spent, ${self.remaining_budget:.0f} remaining
  Conversion Prob: {self.conversion_probability:.3f}
  Shadow Mode: {self.shadow_mode}
        """.strip()
    
    @property 
    def state_dim(self) -> int:
        """Dimension of state vector (including shadow mode features)"""
        return 53  # Original 51 + 2 shadow mode features

def create_synthetic_state_for_testing(segment_name: str = None,
                                     channel: str = None,
                                     device: str = None,
                                     stage: int = None,
                                     discovery_patterns: Dict[str, Any] = None) -> DynamicEnrichedState:
    """Create synthetic state for shadow testing"""
    
    # Default patterns if not provided
    if discovery_patterns is None:
        discovery_patterns = {
            'user_patterns': {
                'segments': {
                    'researching_parent': {'conversion_rate': 0.025, 'engagement_score': 0.6},
                    'concerned_parent': {'conversion_rate': 0.035, 'engagement_score': 0.7},
                    'crisis_parent': {'conversion_rate': 0.055, 'engagement_score': 0.9},
                    'proactive_parent': {'conversion_rate': 0.015, 'engagement_score': 0.4}
                }
            },
            'temporal_patterns': {
                'peak_hours': [19, 20, 21],
                'seasonality_factor': 1.1
            }
        }
    
    # Random selections if not specified
    segments = list(discovery_patterns['user_patterns']['segments'].keys())
    channels = ['organic', 'paid_search', 'social', 'display', 'email']
    devices = ['mobile', 'desktop', 'tablet']
    
    selected_segment = segment_name or np.random.choice(segments)
    selected_channel = channel or np.random.choice(channels)
    selected_device = device or np.random.choice(devices)
    selected_stage = stage if stage is not None else np.random.randint(0, 5)
    
    # Get segment data
    segment_data = discovery_patterns['user_patterns']['segments'].get(
        selected_segment, {'conversion_rate': 0.02, 'engagement_score': 0.5}
    )
    
    # Create state
    state = DynamicEnrichedState(
        # Journey
        stage=selected_stage,
        touchpoints_seen=np.random.poisson(3) + 1,
        days_since_first_touch=np.random.exponential(2.0),
        
        # Segment
        segment_name=selected_segment,
        segment_cvr=segment_data['conversion_rate'],
        segment_engagement=segment_data['engagement_score'],
        
        # Context
        device=selected_device,
        channel=selected_channel,
        hour_of_day=np.random.randint(0, 24),
        day_of_week=np.random.randint(0, 7),
        
        # Performance
        creative_fatigue=np.random.beta(2, 5),
        budget_spent_ratio=np.random.beta(2, 3),
        competition_level=np.random.beta(2, 2),
        avg_competitor_bid=np.random.lognormal(0.5, 0.3),
        
        # Identity
        cross_device_confidence=np.random.beta(3, 2),
        is_returning_user=np.random.choice([True, False], p=[0.3, 0.7]),
        
        # Conversion
        conversion_probability=segment_data['conversion_rate'] * (1 + np.random.normal(0, 0.2)),
        segment_avg_ltv=np.random.lognormal(4.6, 0.4),  # ~$100 average
        
        # Shadow mode
        shadow_mode=True,
        model_prediction_confidence=np.random.beta(2, 2)
    )
    
    # Update with discovery patterns
    state.update_from_discovery(discovery_patterns)
    
    return state

def batch_create_synthetic_states(count: int = 100,
                                 discovery_patterns: Dict[str, Any] = None) -> List[DynamicEnrichedState]:
    """Create batch of synthetic states for testing"""
    states = []
    
    for _ in range(count):
        state = create_synthetic_state_for_testing(discovery_patterns=discovery_patterns)
        states.append(state)
    
    return states

if __name__ == "__main__":
    # Demo usage
    print("DynamicEnrichedState for Shadow Mode Testing")
    print("=" * 50)
    
    # Create a test state
    state = create_synthetic_state_for_testing()
    print(state.get_state_summary())
    
    print(f"\nState vector dimension: {state.state_dim}")
    print(f"State vector preview: {state.to_vector()[:10]}...")
    
    # Test cloning for shadow
    shadow_state = state.clone_for_shadow("shadow_model_v1")
    print(f"\nShadow state created at: {shadow_state.state_generation_time}")
    print(f"Shadow mode: {shadow_state.shadow_mode}")
    
    # Test serialization
    state_dict = state.to_dict()
    restored_state = DynamicEnrichedState.from_dict(state_dict)
    print(f"\nSerialization test passed: {restored_state.segment_name == state.segment_name}")
    
    print("\nBatch creation test...")
    batch_states = batch_create_synthetic_states(10)
    segments = [s.segment_name for s in batch_states]
    print(f"Created {len(batch_states)} states with segments: {set(segments)}")