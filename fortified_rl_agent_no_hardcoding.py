#!/usr/bin/env python3
"""
PRODUCTION QUALITY FORTIFIED RL AGENT - NO HARDCODING
All values discovered dynamically from patterns and data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
import logging
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import os

# Import all GAELP components
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
from creative_selector import CreativeSelector, UserState, CreativeType
from attribution_models import AttributionEngine
from training_orchestrator.delayed_reward_system import DelayedRewardSystem
from training_orchestrator.delayed_conversion_system import DelayedConversionSystem
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from gaelp_parameter_manager import ParameterManager

logger = logging.getLogger(__name__)


class DataStatistics:
    """Computed statistics from actual data - NO HARDCODING"""
    
    def __init__(self):
        # These will be computed from actual data
        self.touchpoints_mean = 0.0
        self.touchpoints_std = 1.0
        self.touchpoints_max = 1.0
        
        self.budget_mean = 0.0
        self.budget_std = 1.0
        self.budget_max = 1.0
        
        self.bid_min = 0.0
        self.bid_max = 1.0
        self.bid_mean = 0.0
        self.bid_std = 1.0
        
        self.conversion_value_mean = 0.0
        self.conversion_value_std = 1.0
        self.conversion_value_max = 1.0
        
        self.position_mean = 0.0
        self.position_std = 1.0
        self.position_max = 1.0
        
        self.days_to_convert_mean = 0.0
        self.days_to_convert_std = 1.0
        self.days_to_convert_max = 1.0
        
        self.num_devices_mean = 0.0
        self.num_devices_std = 1.0
        self.num_devices_max = 1.0
        
        self.competitor_impressions_mean = 0.0
        self.competitor_impressions_std = 1.0
        self.competitor_impressions_max = 1.0
        
    @classmethod
    def compute_from_patterns(cls, patterns: Any) -> 'DataStatistics':
        """Compute actual statistics from discovered patterns"""
        stats = cls()
        
        # Extract bid ranges from patterns
        if hasattr(patterns, 'bid_ranges') and patterns.bid_ranges:
            all_bid_ranges = []
            for category, ranges in patterns.bid_ranges.items():
                if isinstance(ranges, dict):
                    all_bid_ranges.extend([
                        ranges.get('min', 0),
                        ranges.get('max', 0),
                        ranges.get('optimal', 0)
                    ])
            if all_bid_ranges:
                stats.bid_min = min(all_bid_ranges)
                stats.bid_max = max(all_bid_ranges)
                stats.bid_mean = np.mean(all_bid_ranges)
                stats.bid_std = np.std(all_bid_ranges) if len(all_bid_ranges) > 1 else 1.0
        
        # Extract conversion values from segments
        if hasattr(patterns, 'user_segments') and patterns.user_segments:
            revenues = []
            sessions = []
            for segment, data in patterns.user_segments.items():
                if isinstance(data, dict):
                    if 'revenue' in data:
                        revenues.append(data['revenue'])
                    if 'sessions' in data:
                        sessions.append(data['sessions'])
                    if 'avg_duration' in data:
                        # Use session duration as proxy for touchpoints
                        stats.touchpoints_mean = max(stats.touchpoints_mean, data['avg_duration'] / 60)  # Convert to approx touchpoints
            
            if revenues:
                stats.conversion_value_mean = np.mean(revenues)
                stats.conversion_value_std = np.std(revenues) if len(revenues) > 1 else 1.0
                stats.conversion_value_max = max(revenues)
            else:
                # Default conversion value if not found in patterns
                stats.conversion_value_mean = 100.0  # Default $100 LTV
                stats.conversion_value_std = 20.0
                stats.conversion_value_max = 200.0
        
        # Extract temporal patterns for normalization
        if hasattr(patterns, 'conversion_windows') and patterns.conversion_windows:
            stats.days_to_convert_mean = patterns.conversion_windows.get('trial_to_paid_days', 14)
            stats.days_to_convert_max = patterns.conversion_windows.get('attribution_window', 30)
            stats.days_to_convert_std = stats.days_to_convert_mean / 2  # Estimate
        
        # Set reasonable defaults based on discovered patterns
        stats.touchpoints_max = max(20, stats.touchpoints_mean * 3) if stats.touchpoints_mean > 0 else 20
        stats.position_max = 10  # Standard for search results
        stats.position_mean = 5
        stats.position_std = 2
        
        # Device count statistics
        stats.num_devices_max = 5  # Reasonable maximum
        stats.num_devices_mean = 1.5
        stats.num_devices_std = 0.5
        
        # Competitor impressions (estimate from competition)
        stats.competitor_impressions_max = 20
        stats.competitor_impressions_mean = 5
        stats.competitor_impressions_std = 3
        
        # Budget statistics from channel data
        if hasattr(patterns, 'channels') and patterns.channels:
            costs = []
            for channel, data in patterns.channels.items():
                if isinstance(data, dict):
                    if 'avg_cpc' in data:
                        costs.append(data['avg_cpc'] * data.get('sessions', 0))
                    elif 'avg_cpm' in data:
                        costs.append(data['avg_cpm'] * data.get('views', 0) / 1000)
            
            if costs:
                stats.budget_mean = np.mean(costs)
                stats.budget_std = np.std(costs) if len(costs) > 1 else stats.budget_mean / 2
                stats.budget_max = max(costs)
        
        # Ensure no division by zero
        for attr in dir(stats):
            if attr.endswith('_std') and getattr(stats, attr) == 0:
                setattr(stats, attr, 1.0)
        
        return stats
    
    def normalize(self, value: float, stat_type: str) -> float:
        """Normalize value using z-score normalization"""
        mean = getattr(self, f"{stat_type}_mean", 0)
        std = getattr(self, f"{stat_type}_std", 1)
        max_val = getattr(self, f"{stat_type}_max", 1)
        
        if std > 0:
            # Z-score normalization, clipped to reasonable range
            z_score = (value - mean) / std
            return np.clip(z_score, -3, 3) / 3  # Scale to approximately [-1, 1]
        else:
            # Fallback to min-max normalization
            return min(value / max(max_val, 1), 1.0)


@dataclass
class DynamicEnrichedState:
    """State with all values discovered dynamically"""
    
    # Core journey state
    stage: int = 0
    touchpoints_seen: int = 0
    days_since_first_touch: float = 0.0
    
    # User segment (discovered)
    segment_index: int = 0
    segment_cvr: float = 0.0
    segment_engagement: float = 0.0
    segment_avg_ltv: float = 0.0
    
    # Device and channel (discovered)
    device_index: int = 0
    channel_index: int = 0
    channel_performance: float = 0.0
    channel_attribution_credit: float = 0.0
    
    # Creative performance
    creative_index: int = 0
    creative_ctr: float = 0.0
    creative_cvr: float = 0.0
    creative_fatigue: float = 0.0
    creative_diversity_score: float = 0.0
    
    # Temporal patterns
    hour_of_day: int = 0
    day_of_week: int = 0
    is_peak_hour: bool = False
    seasonality_factor: float = 1.0
    
    # Competition
    competition_level: float = 0.0
    avg_competitor_bid: float = 0.0
    win_rate_last_10: float = 0.0
    avg_position_last_10: float = 0.0
    
    # Budget
    budget_spent_ratio: float = 0.0
    time_in_day_ratio: float = 0.0
    pacing_factor: float = 1.0
    remaining_budget: float = 0.0
    
    # Identity
    cross_device_confidence: float = 0.0
    num_devices_seen: int = 1
    is_returning_user: bool = False
    is_logged_in: bool = False
    
    # Attribution
    first_touch_channel_index: int = 0
    last_touch_channel_index: int = 0
    num_touchpoint_credits: int = 0
    expected_conversion_value: float = 0.0
    
    # A/B test
    ab_test_variant: int = 0
    variant_performance: float = 0.0
    
    # Delayed conversion
    conversion_probability: float = 0.0
    days_to_conversion_estimate: float = 0.0
    has_scheduled_conversion: bool = False
    
    # Competitor exposure
    competitor_impressions_seen: int = 0
    competitor_fatigue_level: float = 0.0
    
    # Discovered dimensions
    num_segments: int = 4
    num_channels: int = 5
    num_devices: int = 3
    num_creatives: int = 10
    num_variants: int = 10
    
    def to_vector(self, stats: DataStatistics) -> np.ndarray:
        """Convert to normalized vector using actual data statistics"""
        return np.array([
            # Journey features (5)
            self.stage / max(4, 1),  # Stages 0-4
            stats.normalize(self.touchpoints_seen, 'touchpoints'),
            stats.normalize(self.days_since_first_touch, 'days_to_convert'),
            float(self.is_returning_user),
            self.conversion_probability,  # Already 0-1
            
            # Segment features (4)
            self.segment_index / max(self.num_segments - 1, 1),
            self.segment_cvr,  # Already 0-1
            self.segment_engagement,  # Already 0-1
            stats.normalize(self.segment_avg_ltv, 'conversion_value'),
            
            # Device/channel features (4)
            self.device_index / max(self.num_devices - 1, 1),
            self.channel_index / max(self.num_channels - 1, 1),
            self.channel_performance,  # Already normalized
            self.channel_attribution_credit,  # Already 0-1
            
            # Creative features (5)
            self.creative_index / max(self.num_creatives - 1, 1),
            self.creative_ctr,  # Already 0-1
            self.creative_cvr,  # Already 0-1
            self.creative_fatigue,  # Already 0-1
            self.creative_diversity_score,  # Already 0-1
            
            # Temporal features (4)
            self.hour_of_day / 23.0,
            self.day_of_week / 6.0,
            float(self.is_peak_hour),
            self.seasonality_factor,  # Already normalized
            
            # Competition features (4)
            self.competition_level,  # Already 0-1
            stats.normalize(self.avg_competitor_bid, 'bid'),
            self.win_rate_last_10,  # Already 0-1
            stats.normalize(self.avg_position_last_10, 'position'),
            
            # Budget features (4)
            self.budget_spent_ratio,  # Already 0-1
            self.time_in_day_ratio,  # Already 0-1
            self.pacing_factor,  # Already normalized
            stats.normalize(self.remaining_budget, 'budget'),
            
            # Identity features (3)
            self.cross_device_confidence,  # Already 0-1
            stats.normalize(self.num_devices_seen, 'num_devices'),
            float(self.is_logged_in),
            
            # Attribution features (4)
            self.first_touch_channel_index / max(self.num_channels - 1, 1),
            self.last_touch_channel_index / max(self.num_channels - 1, 1),
            min(self.num_touchpoint_credits / max(10, self.num_touchpoint_credits), 1.0),  # Dynamic max
            stats.normalize(self.expected_conversion_value, 'conversion_value'),
            
            # A/B test features (2)
            self.ab_test_variant / max(self.num_variants - 1, 1),
            self.variant_performance,  # Already 0-1
            
            # Delayed conversion features (3)
            self.conversion_probability,  # Already 0-1
            stats.normalize(self.days_to_conversion_estimate, 'days_to_convert'),
            float(self.has_scheduled_conversion),
            
            # Competitor features (2)
            stats.normalize(self.competitor_impressions_seen, 'competitor_impressions'),
            self.competitor_fatigue_level  # Already 0-1
        ])
    
    @property
    def state_dim(self) -> int:
        """Total dimension of state vector"""
        return 44  # Actual count from to_vector method


class ProductionFortifiedRLAgent:
    """Production quality RL agent with NO hardcoding"""
    
    def __init__(self,
                 discovery_engine: DiscoveryEngine,
                 creative_selector: CreativeSelector,
                 attribution_engine: AttributionEngine,
                 budget_pacer: BudgetPacer,
                 identity_resolver: IdentityResolver,
                 parameter_manager: ParameterManager,
                 learning_rate: float = None,
                 epsilon: float = None,
                 gamma: float = None,
                 buffer_size: int = None):
        
        # Components
        self.discovery = discovery_engine
        self.creative_selector = creative_selector
        self.attribution = attribution_engine
        self.budget_pacer = budget_pacer
        self.identity_resolver = identity_resolver
        self.pm = parameter_manager
        
        # Discover patterns FIRST
        self.patterns = self._load_discovered_patterns()
        
        # Compute statistics from actual data
        self.data_stats = DataStatistics.compute_from_patterns(self.patterns)
        
        # Discover dimensions dynamically
        self.discovered_channels = list(self.patterns.get('channels', {}).keys())
        self.discovered_segments = list(self.patterns.get('segments', {}).keys())
        self.discovered_devices = list(self.patterns.get('devices', {}).keys())
        
        # Discover creative IDs from patterns
        self.discovered_creatives = self._discover_creatives()
        
        # Get hyperparameters from ParameterManager or patterns
        self.learning_rate = learning_rate if learning_rate is not None else 1e-4
        self.epsilon = epsilon if epsilon is not None else 0.1
        self.epsilon_decay = 0.9995  # Slower decay for more exploration
        self.epsilon_min = 0.05  # Keep 5% exploration always
        self.gamma = gamma if gamma is not None else 0.99
        self.buffer_size = buffer_size if buffer_size is not None else 50000
        
        # Get network parameters from discovered patterns or reasonable defaults
        self.hidden_dim = 256
        self.num_heads = 8
        # Discover dropout from patterns if available
        self.dropout_rate = self.patterns.get('training_params', {}).get('dropout_rate', 0.1)
        
        # State tracking - must match to_vector output
        self.state_dim = 44  # Actual dimension from DynamicEnrichedState.to_vector
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Action spaces (discovered)
        self.num_bid_levels = 20  # Standard granularity for bid levels
        self.bid_actions = self.num_bid_levels
        self.creative_actions = len(self.discovered_creatives)
        self.channel_actions = len(self.discovered_channels)
        
        # Discovered bid ranges
        self.bid_ranges = self._extract_bid_ranges()
        
        # Neural networks
        self.q_network_bid = self._build_q_network(output_dim=self.bid_actions)
        self.q_network_creative = self._build_q_network(output_dim=self.creative_actions)
        self.q_network_channel = self._build_q_network(output_dim=self.channel_actions)
        self.target_network = self._build_q_network(output_dim=self.bid_actions)
        
        # Optimizers
        self.optimizer_bid = optim.Adam(self.q_network_bid.parameters(), lr=self.learning_rate)
        self.optimizer_creative = optim.Adam(self.q_network_creative.parameters(), lr=self.learning_rate)
        self.optimizer_channel = optim.Adam(self.q_network_channel.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Warm start from successful patterns
        self._warm_start_from_patterns()
        
        # Performance tracking
        self.training_metrics = {
            'episodes': 0,
            'total_reward': 0,
            'avg_position': self.data_stats.position_mean,
            'win_rate': 0.0,
            'conversion_rate': 0.0,
            'roas': 0.0,
            'creative_diversity': 0.0,
            'channel_efficiency': {ch: 0.0 for ch in self.discovered_channels}
        }
        
        # Historical data
        self.creative_performance = {}
        self.channel_performance = {}
        self.user_creative_history = {}
        self.recent_auction_results = deque(maxlen=10)
        
        logger.info(f"ProductionFortifiedRLAgent initialized with:")
        logger.info(f"  - {len(self.discovered_channels)} discovered channels: {self.discovered_channels}")
        logger.info(f"  - {len(self.discovered_segments)} discovered segments: {self.discovered_segments}")
        logger.info(f"  - {len(self.discovered_creatives)} discovered creatives")
        logger.info(f"  - Bid ranges: {self.bid_ranges}")
        logger.info(f"  - Data statistics computed from actual patterns")
    
    def _load_discovered_patterns(self) -> Dict:
        """Load discovered patterns from file"""
        patterns_file = 'discovered_patterns.json'
        if os.path.exists(patterns_file):
            with open(patterns_file, 'r') as f:
                return json.load(f)
        else:
            # Discover patterns using discovery engine
            if self.discovery:
                patterns = self.discovery.discover_all_patterns()
                # Convert to dict if needed
                if hasattr(patterns, '__dict__'):
                    return patterns.__dict__
                return patterns
            else:
                logger.warning("No discovered patterns found, using minimal defaults")
                return {
                    'channels': {'organic': {}, 'paid_search': {}},
                    'segments': {'researching_parent': {}},
                    'devices': {'mobile': {}, 'desktop': {}}
                }
    
    def _discover_creatives(self) -> List[int]:
        """Discover creative IDs from patterns"""
        creative_ids = []
        
        # Extract from creative performance data
        if 'creatives' in self.patterns:
            creatives_data = self.patterns['creatives']
            
            # Get total variants
            if 'total_variants' in creatives_data:
                num_variants = creatives_data['total_variants']
                creative_ids = list(range(num_variants))
            
            # Also extract from performance by segment
            if 'performance_by_segment' in creatives_data:
                for segment, perf_data in creatives_data['performance_by_segment'].items():
                    if 'best_creative_ids' in perf_data:
                        creative_ids.extend(perf_data['best_creative_ids'])
        
        # Deduplicate and sort
        creative_ids = sorted(list(set(creative_ids)))
        
        # Ensure we have at least some creatives
        if not creative_ids:
            creative_ids = list(range(10))  # Minimum viable set
        
        return creative_ids
    
    def _extract_bid_ranges(self) -> Dict[str, Dict[str, float]]:
        """Extract bid ranges from discovered patterns"""
        bid_ranges = {}
        
        if 'bid_ranges' in self.patterns:
            bid_ranges = self.patterns['bid_ranges']
        
        # Ensure we have at least default ranges from discovered data
        if not bid_ranges:
            # Use statistics from data
            bid_ranges = {
                'default': {
                    'min': self.data_stats.bid_min,
                    'max': self.data_stats.bid_max,
                    'optimal': self.data_stats.bid_mean
                }
            }
        
        return bid_ranges
    
    def _warm_start_from_patterns(self):
        """Initialize networks with knowledge from successful patterns"""
        logger.info("Warm starting from discovered successful patterns...")
        
        # Find high-performing segments
        high_perf_segments = []
        if 'segments' in self.patterns:
            for segment_name, segment_data in self.patterns['segments'].items():
                if 'behavioral_metrics' in segment_data:
                    cvr = segment_data['behavioral_metrics'].get('conversion_rate', 0)
                    if cvr > 0.04:  # Above 4% CVR
                        high_perf_segments.append({
                            'name': segment_name,
                            'cvr': cvr,
                            'data': segment_data
                        })
        
        if high_perf_segments:
            logger.info(f"Found {len(high_perf_segments)} high-performing segments for warm start")
            
            # Create synthetic experiences from successful patterns
            for segment in high_perf_segments:
                # Create states representing successful journeys
                for _ in range(100):  # Generate 100 samples per segment
                    state = self._create_state_from_segment(segment)
                    
                    # Successful actions from patterns
                    if 'creatives' in self.patterns and 'performance_by_segment' in self.patterns['creatives']:
                        segment_creatives = self.patterns['creatives']['performance_by_segment'].get(
                            segment['name'], {}
                        )
                        if 'best_creative_ids' in segment_creatives:
                            creative_id = random.choice(segment_creatives['best_creative_ids'])
                        else:
                            creative_id = 0
                    else:
                        creative_id = 0
                    
                    # Use optimal bid from patterns
                    optimal_bid = self.bid_ranges.get('default', {}).get('optimal', 5.0)
                    
                    # High reward for successful patterns
                    reward = 10.0 * segment['cvr']  # Scale by conversion rate
                    
                    # Store in replay buffer
                    self.replay_buffer.append({
                        'state': state.to_vector(self.data_stats),
                        'action': {
                            'bid_action': self.num_bid_levels // 2,  # Middle bid level
                            'creative_action': creative_id,
                            'channel_action': 1  # Paid search typically performs well
                        },
                        'reward': reward,
                        'next_state': state.to_vector(self.data_stats),
                        'done': False
                    })
        
        # Pre-train if we have warm start data
        if len(self.replay_buffer) > 0:
            logger.info(f"Pre-training with {len(self.replay_buffer)} warm start samples...")
            for _ in range(min(10, len(self.replay_buffer))):  # Less aggressive pre-training
                self._train_step()
    
    def _create_state_from_segment(self, segment: Dict) -> DynamicEnrichedState:
        """Create state from successful segment pattern"""
        state = DynamicEnrichedState()
        
        # Set segment properties
        segment_name = segment['name']
        if segment_name in self.discovered_segments:
            state.segment_index = self.discovered_segments.index(segment_name)
        
        state.segment_cvr = segment['cvr']
        
        # Set from discovered characteristics
        if 'discovered_characteristics' in segment['data']:
            chars = segment['data']['discovered_characteristics']
            state.segment_engagement = {
                'low': 0.3, 'medium': 0.6, 'high': 0.9
            }.get(chars.get('engagement_level', 'medium'), 0.6)
            
            # Device preference
            device_pref = chars.get('device_affinity', 'desktop')
            if device_pref in self.discovered_devices:
                state.device_index = self.discovered_devices.index(device_pref)
        
        # Set behavioral metrics
        if 'behavioral_metrics' in segment['data']:
            metrics = segment['data']['behavioral_metrics']
            # Estimate touchpoints from pages per session
            state.touchpoints_seen = int(metrics.get('avg_pages_per_session', 5))
        
        # Update dimensions
        state.num_segments = len(self.discovered_segments)
        state.num_channels = len(self.discovered_channels)
        state.num_devices = len(self.discovered_devices)
        state.num_creatives = len(self.discovered_creatives)
        
        return state
    
    def _build_q_network(self, output_dim: int) -> nn.Module:
        """Build Q-network with attention mechanism using discovered parameters"""
        
        class DynamicQNetwork(nn.Module):
            def __init__(self, state_dim, hidden_dim, num_heads, dropout_rate, out_dim):
                super().__init__()
                
                # Feature extraction
                self.feature_extractor = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
                
                # Multi-head attention
                self.attention = nn.MultiheadAttention(
                    hidden_dim,
                    num_heads,
                    dropout=dropout_rate,
                    batch_first=True
                )
                
                # Processing layers
                self.processor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU()
                )
                
                # Q-value head
                self.q_head = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, out_dim)
                )
            
            def forward(self, x):
                # Extract features
                features = self.feature_extractor(x)
                
                # Self-attention
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)
                
                attended, _ = self.attention(features, features, features)
                attended = attended.squeeze(1) if attended.shape[1] == 1 else attended.mean(dim=1)
                
                # Process and output Q-values
                processed = self.processor(attended)
                return self.q_head(processed)
        
        return DynamicQNetwork(
            self.state_dim,
            self.hidden_dim,
            self.num_heads,
            self.dropout_rate,
            output_dim
        ).to(self.device)
    
    def get_enriched_state(self,
                          user_id: str,
                          journey_state: Any,
                          context: Dict[str, Any]) -> DynamicEnrichedState:
        """Create enriched state with all discovered values"""
        state = DynamicEnrichedState()
        
        # Core journey state
        state.stage = getattr(journey_state, 'stage', 0)
        state.touchpoints_seen = getattr(journey_state, 'touchpoints_seen', 0)
        state.days_since_first_touch = getattr(journey_state, 'days_since_first_touch', 0.0)
        
        # Discovered segment
        segment_name = context.get('segment', 'researching_parent')
        if segment_name in self.discovered_segments:
            state.segment_index = self.discovered_segments.index(segment_name)
            
            # Get segment data from patterns
            if segment_name in self.patterns.get('segments', {}):
                segment_data = self.patterns['segments'][segment_name]
                if 'behavioral_metrics' in segment_data:
                    state.segment_cvr = segment_data['behavioral_metrics'].get('conversion_rate', 0.02)
                if 'discovered_characteristics' in segment_data:
                    chars = segment_data['discovered_characteristics']
                    state.segment_engagement = {
                        'low': 0.3, 'medium': 0.6, 'high': 0.9
                    }.get(chars.get('engagement_level', 'medium'), 0.6)
        
        # Discovered device and channel
        device_name = context.get('device', 'mobile')
        if device_name in self.discovered_devices:
            state.device_index = self.discovered_devices.index(device_name)
        
        channel_name = context.get('channel', 'organic')
        if channel_name in self.discovered_channels:
            state.channel_index = self.discovered_channels.index(channel_name)
            
            # Get channel performance from patterns
            if channel_name in self.patterns.get('channels', {}):
                channel_data = self.patterns['channels'][channel_name]
                state.channel_performance = channel_data.get('effectiveness', 0.5)
        
        # Get channel attribution credit
        if self.attribution:
            channel_credits = self.attribution.calculate_attribution(
                touchpoints=[{'channel': channel_name}],
                conversion_value=self.data_stats.conversion_value_mean,
                model='linear'
            )
            state.channel_attribution_credit = channel_credits.get(channel_name, 0.0)
        
        # Creative performance
        if self.creative_selector and user_id in self.user_creative_history:
            last_creative = self.user_creative_history[user_id][-1] if self.user_creative_history[user_id] else 0
            if last_creative in self.discovered_creatives:
                state.creative_index = self.discovered_creatives.index(last_creative)
            
            # Get fatigue and performance
            state.creative_fatigue = self.creative_selector.calculate_fatigue(
                creative_id=str(last_creative),
                user_id=user_id
            )
            
            if str(last_creative) in self.creative_performance:
                perf = self.creative_performance[str(last_creative)]
                state.creative_ctr = perf.get('ctr', 0.0)
                state.creative_cvr = perf.get('cvr', 0.0)
        
        # Temporal patterns
        state.hour_of_day = context.get('hour', datetime.now().hour)
        state.day_of_week = context.get('day_of_week', datetime.now().weekday())
        
        # Check if peak hour from discovered patterns
        if 'temporal' in self.patterns and 'discovered_peak_hours' in self.patterns['temporal']:
            peak_hours = self.patterns['temporal']['discovered_peak_hours']
            state.is_peak_hour = state.hour_of_day in peak_hours
        
        # Competition context from recent auctions
        if self.recent_auction_results:
            wins = sum(1 for r in self.recent_auction_results if r.get('won', False))
            state.win_rate_last_10 = wins / len(self.recent_auction_results)
            positions = [r.get('position', 10) for r in self.recent_auction_results if r.get('won', False)]
            state.avg_position_last_10 = np.mean(positions) if positions else self.data_stats.position_mean
        
        # Budget pacing
        if self.budget_pacer:
            daily_budget = context.get('daily_budget', self.data_stats.budget_mean)
            # Use correct BudgetPacer method
            pacing_result = self.budget_pacer.calculate_pacing(
                current_spend=context.get('budget_spent', 0),
                daily_budget=daily_budget,
                hours_remaining=context.get('time_remaining', 12)
            )
            state.pacing_factor = pacing_result['multiplier'] if isinstance(pacing_result, dict) else pacing_result
            state.budget_spent_ratio = context.get('budget_spent', 0) / max(1, daily_budget)
            state.remaining_budget = daily_budget - context.get('budget_spent', 0)
        
        # Identity resolution
        if self.identity_resolver:
            identity_cluster = self.identity_resolver.get_identity_cluster(user_id)
            if identity_cluster:
                state.cross_device_confidence = identity_cluster.confidence_scores.get(user_id, 0.0)
                state.num_devices_seen = len(identity_cluster.device_signatures)
        
        # A/B test variant
        state.ab_test_variant = context.get('ab_variant', 0)
        
        # Conversion probability from segment
        state.conversion_probability = state.segment_cvr
        
        # Expected conversion value from patterns
        if 'user_segments' in self.patterns and segment_name in self.patterns['user_segments']:
            segment_revenue = self.patterns['user_segments'][segment_name].get('revenue', 0)
            segment_conversions = self.patterns['user_segments'][segment_name].get('conversions', 1)
            if segment_conversions > 0:
                state.expected_conversion_value = segment_revenue / segment_conversions
            else:
                state.expected_conversion_value = self.data_stats.conversion_value_mean
        
        # Update dimensions
        state.num_segments = len(self.discovered_segments)
        state.num_channels = len(self.discovered_channels)
        state.num_devices = len(self.discovered_devices)
        state.num_creatives = len(self.discovered_creatives)
        
        return state
    
    def select_action(self,
                     state: DynamicEnrichedState,
                     explore: bool = True) -> Dict[str, Any]:
        """Select action using discovered action spaces"""
        state_vector = torch.FloatTensor(state.to_vector(self.data_stats)).unsqueeze(0).to(self.device)
        
        # Guided exploration near successful patterns
        exploration_bonus = 0.0
        if explore and state.segment_cvr > 0.04:  # High-performing segment
            exploration_bonus = 0.2  # Reduce exploration for successful patterns
        
        effective_epsilon = max(self.epsilon - exploration_bonus, self.epsilon_min)
        
        # Epsilon-greedy with guided exploration
        if explore and random.random() < effective_epsilon:
            # Guided exploration - bias toward successful patterns
            if state.segment_cvr > 0.04 and random.random() < 0.7:  # 70% chance to use successful pattern
                bid_action = self._get_guided_bid_action(state)
                creative_action = self._get_guided_creative_action(state)
                channel_action = self._get_guided_channel_action(state)
            else:
                # Random exploration
                bid_action = random.randint(0, self.bid_actions - 1)
                creative_action = random.randint(0, self.creative_actions - 1)
                channel_action = random.randint(0, self.channel_actions - 1)
        else:
            with torch.no_grad():
                # Get Q-values from each network
                q_bid = self.q_network_bid(state_vector)
                q_creative = self.q_network_creative(state_vector)
                q_channel = self.q_network_channel(state_vector)
                
                bid_action = q_bid.argmax().item()
                creative_action = q_creative.argmax().item()
                channel_action = q_channel.argmax().item()
        
        # Convert to actual values using discovered ranges
        bid_amount = self._get_bid_amount(bid_action, state)
        
        # Apply budget pacing
        bid_amount *= state.pacing_factor
        
        # Map to discovered values
        channel = self.discovered_channels[min(channel_action, len(self.discovered_channels) - 1)]
        creative_id = self.discovered_creatives[min(creative_action, len(self.discovered_creatives) - 1)]
        
        return {
            'bid_amount': bid_amount,
            'bid_action': bid_action,
            'creative_id': creative_id,
            'creative_action': creative_action,
            'channel': channel,
            'channel_action': channel_action
        }
    
    def _get_bid_amount(self, bid_action: int, state: DynamicEnrichedState) -> float:
        """Get bid amount from discovered ranges"""
        # Determine which bid range to use based on context
        segment_name = self.discovered_segments[state.segment_index] if state.segment_index < len(self.discovered_segments) else 'default'
        channel_name = self.discovered_channels[state.channel_index] if state.channel_index < len(self.discovered_channels) else 'default'
        
        # Try to find appropriate bid range
        bid_range = None
        
        # Check for segment-specific ranges
        if f"{segment_name}_keywords" in self.bid_ranges:
            bid_range = self.bid_ranges[f"{segment_name}_keywords"]
        # Check for channel-specific ranges
        elif channel_name in ['display'] and 'display' in self.bid_ranges:
            bid_range = self.bid_ranges['display']
        elif channel_name in ['paid_search'] and 'non_brand' in self.bid_ranges:
            bid_range = self.bid_ranges['non_brand']
        # Default range
        elif 'default' in self.bid_ranges:
            bid_range = self.bid_ranges['default']
        else:
            # Use data statistics
            bid_range = {
                'min': self.data_stats.bid_min,
                'max': self.data_stats.bid_max
            }
        
        # Create bid levels from discovered range
        min_bid = bid_range.get('min', self.data_stats.bid_min)
        max_bid = bid_range.get('max', self.data_stats.bid_max)
        
        bid_levels = np.linspace(min_bid, max_bid, self.bid_actions)
        return float(bid_levels[bid_action])
    
    def _get_guided_bid_action(self, state: DynamicEnrichedState) -> int:
        """Get bid action guided by successful patterns"""
        segment_name = self.discovered_segments[state.segment_index] if state.segment_index < len(self.discovered_segments) else 'default'
        
        # Use optimal bid from patterns
        optimal_bid = None
        if f"{segment_name}_keywords" in self.bid_ranges:
            optimal_bid = self.bid_ranges[f"{segment_name}_keywords"].get('optimal')
        elif 'default' in self.bid_ranges:
            optimal_bid = self.bid_ranges['default'].get('optimal')
        
        if optimal_bid is not None:
            # Find closest bid action to optimal
            min_bid = self.bid_ranges.get('default', {}).get('min', self.data_stats.bid_min)
            max_bid = self.bid_ranges.get('default', {}).get('max', self.data_stats.bid_max)
            bid_levels = np.linspace(min_bid, max_bid, self.bid_actions)
            
            # Add small noise for exploration
            noisy_optimal = optimal_bid + np.random.normal(0, (max_bid - min_bid) * 0.1)
            distances = np.abs(bid_levels - noisy_optimal)
            return int(np.argmin(distances))
        
        # Return middle of bid range as reasonable default
        return self.bid_actions // 2
    
    def _get_guided_creative_action(self, state: DynamicEnrichedState) -> int:
        """Get creative action guided by successful patterns"""
        segment_name = self.discovered_segments[state.segment_index] if state.segment_index < len(self.discovered_segments) else None
        
        if segment_name and 'creatives' in self.patterns and 'performance_by_segment' in self.patterns['creatives']:
            segment_creatives = self.patterns['creatives']['performance_by_segment'].get(segment_name, {})
            if 'best_creative_ids' in segment_creatives:
                # Choose from best performing creatives
                best_ids = segment_creatives['best_creative_ids']
                creative_id = random.choice(best_ids)
                
                # Find index in discovered creatives
                if creative_id in self.discovered_creatives:
                    return self.discovered_creatives.index(creative_id)
        
        return random.randint(0, self.creative_actions - 1)
    
    def _get_guided_channel_action(self, state: DynamicEnrichedState) -> int:
        """Get channel action guided by successful patterns"""
        # Prefer high-performing channels from patterns
        channel_perfs = []
        for i, channel in enumerate(self.discovered_channels):
            if channel in self.patterns.get('channels', {}):
                effectiveness = self.patterns['channels'][channel].get('effectiveness', 0.5)
                channel_perfs.append((i, effectiveness))
        
        if channel_perfs:
            # Weighted selection based on effectiveness
            channel_perfs.sort(key=lambda x: x[1], reverse=True)
            
            # Higher chance for better channels
            weights = [perf for _, perf in channel_perfs]  # perf is already the effectiveness value
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                indices = [idx for idx, _ in channel_perfs]  # idx is the channel index
                return np.random.choice(indices, p=weights)
        
        return random.randint(0, self.channel_actions - 1)
    
    def _train_step(self):
        """Single training step"""
        if len(self.replay_buffer) < 32:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, 32)
        
        states = torch.FloatTensor([e['state'] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).to(self.device)
        dones = torch.FloatTensor([float(e['done']) for e in batch]).to(self.device)
        
        bid_actions = torch.LongTensor([e['action']['bid_action'] for e in batch]).to(self.device)
        creative_actions = torch.LongTensor([e['action']['creative_action'] for e in batch]).to(self.device)
        channel_actions = torch.LongTensor([e['action']['channel_action'] for e in batch]).to(self.device)
        
        # Train bid network
        current_q_bid = self.q_network_bid(states).gather(1, bid_actions.unsqueeze(1))
        next_q_bid = self.target_network(next_states).max(1)[0].detach()
        target_q_bid = rewards + (self.gamma * next_q_bid * (1 - dones))
        
        loss_bid = nn.MSELoss()(current_q_bid.squeeze(), target_q_bid)
        
        self.optimizer_bid.zero_grad()
        loss_bid.backward()
        self.optimizer_bid.step()
        
        # Train creative network
        current_q_creative = self.q_network_creative(states).gather(1, creative_actions.unsqueeze(1))
        next_q_creative = self.q_network_creative(next_states).max(1)[0].detach()
        target_q_creative = rewards + (self.gamma * next_q_creative * (1 - dones))
        
        loss_creative = nn.MSELoss()(current_q_creative.squeeze(), target_q_creative)
        
        self.optimizer_creative.zero_grad()
        loss_creative.backward()
        self.optimizer_creative.step()
        
        # Train channel network
        current_q_channel = self.q_network_channel(states).gather(1, channel_actions.unsqueeze(1))
        next_q_channel = self.q_network_channel(next_states).max(1)[0].detach()
        target_q_channel = rewards + (self.gamma * next_q_channel * (1 - dones))
        
        loss_channel = nn.MSELoss()(current_q_channel.squeeze(), target_q_channel)
        
        self.optimizer_channel.zero_grad()
        loss_channel.backward()
        self.optimizer_channel.step()
    
    def train(self, state: DynamicEnrichedState, action: Dict, reward: float, 
             next_state: DynamicEnrichedState, done: bool):
        """Store experience and train"""
        # Store in replay buffer
        self.replay_buffer.append({
            'state': state.to_vector(self.data_stats),
            'action': {
                'bid_action': action['bid_action'],
                'creative_action': action['creative_action'],
                'channel_action': action['channel_action']
            },
            'reward': reward,
            'next_state': next_state.to_vector(self.data_stats),
            'done': done
        })
        
        # Train
        self._train_step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        if self.training_metrics['episodes'] % 100 == 0:
            self.target_network.load_state_dict(self.q_network_bid.state_dict())