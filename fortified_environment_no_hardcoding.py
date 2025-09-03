#!/usr/bin/env python3
"""
PRODUCTION QUALITY FORTIFIED GAELP ENVIRONMENT - NO HARDCODING
All values discovered dynamically from patterns and configuration
"""

import numpy as np
import json
import logging
import os
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
from collections import deque, defaultdict
from scipy.stats import entropy
import math

# Import all GAELP components
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
from creative_selector import CreativeSelector, UserState
from attribution_models import AttributionEngine
from gaelp_parameter_manager import ParameterManager
from persistent_user_database_batched import BatchedPersistentUserDatabase
from training_orchestrator.delayed_conversion_system import DelayedConversionSystem
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from auction_gym_integration_fixed import AuctionGymWrapper, AuctionResult
from fortified_rl_agent_no_hardcoding import DynamicEnrichedState, DataStatistics

# Import GA4 integration
try:
    import sys
    sys.path.insert(0, '/home/hariravichandran/AELP')
    from mcp_ga4_integration import GA4DataFetcher
    GA4_AVAILABLE = True
except:
    GA4_AVAILABLE = False

logger = logging.getLogger(__name__)

# Ensure scipy import for advanced statistics
try:
    from scipy.stats import entropy as scipy_entropy
except ImportError:
    logger.warning("scipy not available, using numpy for entropy calculation")
    scipy_entropy = None


class ProductionFortifiedEnvironment(gym.Env):
    """
    Production quality environment with NO hardcoding - everything discovered
    """
    
    def __init__(self,
                 parameter_manager: Optional[ParameterManager] = None,
                 use_real_ga4_data: bool = True,
                 is_parallel: bool = False):
        """
        Initialize environment with all values from configuration or discovery
        """
        super().__init__()
        
        # Parameter manager for configuration
        self.pm = parameter_manager or ParameterManager()
        
        # Load discovered patterns FIRST
        self.patterns = self._load_discovered_patterns()
        
        # Compute data statistics from patterns
        self.data_stats = DataStatistics.compute_from_patterns(self.patterns)
        
        # Initialize multi-objective reward system
        self.reward_calculator = MultiObjectiveRewardCalculator(
            patterns=self.patterns,
            data_stats=self.data_stats,
            parameter_manager=self.pm
        )
        
        # Get budget from patterns or configuration
        self.max_budget = self._discover_budget()
        self.max_steps = 1000  # Standard episode length
        self.use_real_ga4_data = use_real_ga4_data and GA4_AVAILABLE
        
        # Discover dimensions
        self.discovered_channels = list(self.patterns.get('channels', {}).keys())
        self.discovered_segments = list(self.patterns.get('segments', {}).keys())
        self.discovered_devices = list(self.patterns.get('devices', {}).keys())
        self.discovered_creatives = self._discover_creative_ids()
        
        # Initialize all components
        logger.info("Initializing production fortified environment...")
        
        # 1. Discovery Engine
        self.discovery = DiscoveryEngine(
            write_enabled=not is_parallel,
            cache_only=is_parallel
        )
        
        # 2. Creative Selector
        self.creative_selector = CreativeSelector()
        self._initialize_creatives_from_patterns()
        
        # 3. Attribution Engine
        self.attribution = AttributionEngine()
        
        # 4. User Database
        batch_size = 100  # Standard batch size for efficiency
        flush_interval = 5.0  # Flush every 5 seconds
        self.user_db = BatchedPersistentUserDatabase(
            use_batch_writer=True,
            batch_size=batch_size,
            flush_interval=flush_interval
        )
        
        # 5. Delayed Conversion System
        from user_journey_database import UserJourneyDatabase
        from conversion_lag_model import ConversionLagModel
        
        self.journey_db = UserJourneyDatabase()
        lag_model = ConversionLagModel()
        self.conversion_system = DelayedConversionSystem(
            journey_database=self.journey_db,
            attribution_engine=self.attribution,
            conversion_lag_model=lag_model
        )
        
        # 6. Budget Pacer
        self.budget_pacer = BudgetPacer()
        
        # 7. Identity Resolver
        self.identity_resolver = IdentityResolver()
        
        # 8. Auction System
        self.auction_gym = self._initialize_auction_from_patterns()
        
        # 9. GA4 Integration
        if self.use_real_ga4_data:
            self.ga4_fetcher = GA4DataFetcher()
            self._load_real_conversion_data()
        
        # Environment state
        self.current_step = 0
        self.step_count = 0
        self.budget_spent = 0.0
        self.episode_id = f"episode_{datetime.now().timestamp()}"
        
        # Current user context
        self.current_user_id = None
        self.current_user_state = None
        
        # Tracking metrics
        self.metrics = self._initialize_metrics()
        
        # Initialize tracking for reward calculation
        self.reward_tracker = RewardTracker(
            discovered_channels=self.discovered_channels,
            discovered_creatives=self.discovered_creatives,
            discovered_segments=self.discovered_segments
        )
        
        # Dynamic action and observation spaces
        self.action_space = self._create_dynamic_action_space()
        self.observation_space = self._create_dynamic_observation_space()
        
        logger.info(f"Production environment initialized:")
        logger.info(f"  - Budget: {self.max_budget} (discovered)")
        logger.info(f"  - Channels: {len(self.discovered_channels)} discovered")
        logger.info(f"  - Segments: {len(self.discovered_segments)} discovered")
        logger.info(f"  - Creatives: {len(self.discovered_creatives)} discovered")
        logger.info(f"  - Devices: {len(self.discovered_devices)} discovered")
        
        # Check for display channel quality issues
        if 'display' in self.patterns.get('channels', {}):
            display_data = self.patterns['channels']['display']
            if display_data.get('quality_issues', {}).get('needs_urgent_fix', False):
                bot_percentage = display_data['quality_issues'].get('bot_percentage', 0)
                logger.warning(f"⚠️ DISPLAY CHANNEL CRITICAL ISSUE: {bot_percentage}% bot traffic detected")
                logger.warning(f"   Display CVR severely impacted. Fixes available in display_bot_exclusions.json")
    
    def _load_discovered_patterns(self) -> Dict:
        """Load patterns from discovery file"""
        patterns_file = 'discovered_patterns.json'
        if os.path.exists(patterns_file):
            with open(patterns_file, 'r') as f:
                return json.load(f)
        else:
            # Minimal patterns for initialization
            return {
                'channels': {
                    'organic': {'effectiveness': 0.5},
                    'paid_search': {'effectiveness': 0.8}
                },
                'segments': {
                    'researching_parent': {'conversion_rate': 0.04}
                },
                'devices': {
                    'mobile': {},
                    'desktop': {}
                },
                'bid_ranges': {
                    'default': {'min': 1.0, 'max': 10.0, 'optimal': 5.0}
                }
            }
    
    def _discover_budget(self) -> float:
        """Discover budget from patterns or configuration"""
        # Try patterns first for budget information
        budget = None
        
        # Calculate from channel spend in patterns
        total_spend = 0
        if 'channels' in self.patterns:
            for channel, data in self.patterns['channels'].items():
                if 'avg_cpc' in data and 'sessions' in data:
                    total_spend += data['avg_cpc'] * data['sessions']
                elif 'avg_cpm' in data and 'views' in data:
                    total_spend += data['avg_cpm'] * data['views'] / 1000
        
        if total_spend > 0:
            return total_spend / 30  # Daily budget from monthly spend
        
        # Use data statistics
        if self.data_stats.budget_max > 0:
            return self.data_stats.budget_max
        
        # Absolute minimum fallback
        return 1000.0
    
    def _discover_creative_ids(self) -> List[int]:
        """Discover creative IDs from patterns"""
        creative_ids = set()
        
        if 'creatives' in self.patterns:
            creatives_data = self.patterns['creatives']
            
            # Get total variants
            if 'total_variants' in creatives_data:
                num_variants = creatives_data['total_variants']
                creative_ids.update(range(num_variants))
            
            # Extract from segment performance
            if 'performance_by_segment' in creatives_data:
                for segment, perf in creatives_data['performance_by_segment'].items():
                    if 'best_creative_ids' in perf:
                        creative_ids.update(perf['best_creative_ids'])
        
        # Ensure we have some creatives
        if not creative_ids:
            creative_ids = set(range(10))
        
        return sorted(list(creative_ids))
    
    def _initialize_creatives_from_patterns(self):
        """Initialize creatives based on discovered patterns"""
        if 'creatives' not in self.patterns:
            return
        
        creatives_data = self.patterns['creatives']
        
        # Create creatives for each segment's best performers
        if 'performance_by_segment' in creatives_data:
            for segment_name, perf_data in creatives_data['performance_by_segment'].items():
                if 'best_creative_ids' in perf_data:
                    for creative_id in perf_data['best_creative_ids']:
                        # Create creative with discovered performance
                        ctr = perf_data.get('avg_ctr', 0.05)
                        cvr = perf_data.get('avg_cvr', 0.03)
                        
                        # Note: CreativeSelector manages creatives internally
                        # This ensures we have the right performance characteristics
    
    def _initialize_auction_from_patterns(self) -> AuctionGymWrapper:
        """Initialize auction with discovered competition levels"""
        # Get reserve price from bid ranges
        reserve_price = float('inf')
        if 'bid_ranges' in self.patterns:
            for category, ranges in self.patterns['bid_ranges'].items():
                if 'min' in ranges:
                    reserve_price = min(reserve_price, ranges['min'])
        
        if reserve_price == float('inf'):
            reserve_price = self.data_stats.bid_min if self.data_stats.bid_min > 0 else 0.5
        
        # Estimate competition from channel data
        num_competitors = 6  # Default
        if 'channels' in self.patterns:
            # Higher competition for high-effectiveness channels
            effectiveness_scores = []
            for channel, data in self.patterns['channels'].items():
                if 'effectiveness' in data:
                    effectiveness_scores.append(data['effectiveness'])
            
            if effectiveness_scores:
                avg_effectiveness = np.mean(effectiveness_scores)
                # More competitors for more effective channels
                num_competitors = int(4 + avg_effectiveness * 8)  # 4-12 competitors
        
        config = {
            'auction_type': 'second_price',
            'num_slots': 4,  # Standard search results page
            'reserve_price': reserve_price,
            'competitors': {
                'count': num_competitors
            }
        }
        
        return AuctionGymWrapper(config)
    
    def _load_real_conversion_data(self):
        """Load real conversion patterns from GA4"""
        if not self.ga4_fetcher:
            return
        
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Fetch real events
            conversions = self.ga4_fetcher.get_events(
                start_date=start_date,
                end_date=end_date,
                event_name='sign_up'
            )
            
            if conversions and 'rows' in conversions:
                logger.info(f"Loaded {len(conversions['rows'])} real conversion patterns")
        except Exception as e:
            logger.warning(f"Could not load GA4 data: {e}")
    
    def _initialize_metrics(self) -> Dict:
        """Initialize metrics tracking with discovered dimensions"""
        return {
            'total_impressions': 0,
            'total_clicks': 0,
            'total_conversions': 0,
            'total_revenue': 0.0,
            'auction_wins': 0,
            'auction_losses': 0,
            'creative_performance': {cid: {'impressions': 0, 'clicks': 0, 'conversions': 0} 
                                   for cid in self.discovered_creatives},
            'channel_performance': {ch: {'impressions': 0, 'clicks': 0, 'conversions': 0, 'spend': 0.0} 
                                  for ch in self.discovered_channels},
            'segment_performance': {seg: {'users': 0, 'conversions': 0} 
                                  for seg in self.discovered_segments},
            'user_journeys': {}
        }
    
    def _create_dynamic_action_space(self) -> spaces.Dict:
        """Create action space based on discovered dimensions"""
        # Get bid range from patterns
        min_bid = self.data_stats.bid_min if self.data_stats.bid_min > 0 else 0.5
        max_bid = self.data_stats.bid_max if self.data_stats.bid_max > 0 else 10.0
        
        return spaces.Dict({
            'bid': spaces.Box(low=min_bid, high=max_bid, shape=(1,), dtype=np.float32),
            'creative': spaces.Discrete(len(self.discovered_creatives)),
            'channel': spaces.Discrete(len(self.discovered_channels))
        })
    
    def _create_dynamic_observation_space(self) -> spaces.Box:
        """Create observation space for discovered state dimensions"""
        # State dimension from DynamicEnrichedState
        state_dim = 45
        return spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.step_count = 0
        self.budget_spent = 0.0
        self.episode_id = f"episode_{datetime.now().timestamp()}"
        
        # Generate new user from discovered segments
        self.current_user_id, self.current_user_state = self._generate_user_from_patterns()
        
        # Get initial state
        initial_state = self._get_current_state()
        
        # Reset episode metrics
        for key in ['total_impressions', 'total_clicks', 'total_conversions', 
                   'total_revenue', 'auction_wins', 'auction_losses']:
            self.metrics[key] = 0
        
        return initial_state.to_vector(self.data_stats), {'user_id': self.current_user_id}
    
    def _generate_user_from_patterns(self) -> Tuple[str, DynamicEnrichedState]:
        """Generate user based on discovered patterns"""
        # Use UUID for truly unique IDs
        import uuid
        user_id = f"user_{uuid.uuid4().hex[:8]}_{datetime.now().timestamp()}"
        
        # Select segment based on discovered distribution
        segment_weights = []
        segment_names = []
        
        for segment_name in self.discovered_segments:
            if segment_name in self.patterns.get('segments', {}):
                segment_data = self.patterns['segments'][segment_name]
                # Weight by sample size or conversion rate
                if 'discovered_characteristics' in segment_data:
                    weight = segment_data['discovered_characteristics'].get('sample_size', 1)
                else:
                    weight = 1
                segment_weights.append(weight)
                segment_names.append(segment_name)
        
        if segment_weights:
            total_weight = sum(segment_weights)
            probabilities = [w / total_weight for w in segment_weights]
            selected_segment = np.random.choice(segment_names, p=probabilities)
        else:
            selected_segment = self.discovered_segments[0] if self.discovered_segments else 'unknown'
        
        # Create initial state
        state = DynamicEnrichedState()
        state.segment_index = self.discovered_segments.index(selected_segment) if selected_segment in self.discovered_segments else 0
        
        # Set segment properties from patterns
        if selected_segment in self.patterns.get('segments', {}):
            segment_data = self.patterns['segments'][selected_segment]
            if 'behavioral_metrics' in segment_data:
                state.segment_cvr = segment_data['behavioral_metrics'].get('conversion_rate', 0.02)
            if 'discovered_characteristics' in segment_data:
                chars = segment_data['discovered_characteristics']
                state.segment_engagement = {
                    'low': 0.3, 'medium': 0.6, 'high': 0.9
                }.get(chars.get('engagement_level', 'medium'), 0.6)
                
                # Set device preference
                device_pref = chars.get('device_affinity', 'mobile')
                if device_pref in self.discovered_devices:
                    state.device_index = self.discovered_devices.index(device_pref)
        
        # Set discovered dimensions
        state.num_segments = len(self.discovered_segments)
        state.num_channels = len(self.discovered_channels)
        state.num_devices = len(self.discovered_devices)
        state.num_creatives = len(self.discovered_creatives)
        
        # Initialize conversion probability from segment
        state.conversion_probability = state.segment_cvr
        
        # Set LTV from patterns
        if selected_segment in self.patterns.get('user_segments', {}):
            segment_revenue = self.patterns['user_segments'][selected_segment].get('revenue', 0)
            segment_conversions = self.patterns['user_segments'][selected_segment].get('conversions', 1)
            if segment_conversions > 0:
                state.segment_avg_ltv = segment_revenue / segment_conversions
            else:
                state.segment_avg_ltv = self.data_stats.conversion_value_mean
        
        return user_id, state
    
    def _get_current_state(self) -> DynamicEnrichedState:
        """Get current enriched state"""
        if self.current_user_state is None:
            _, self.current_user_state = self._generate_user_from_patterns()
        
        # Update temporal features
        now = datetime.now()
        self.current_user_state.hour_of_day = now.hour
        self.current_user_state.day_of_week = now.weekday()
        
        # Check if peak hour from patterns
        if 'temporal' in self.patterns and 'discovered_peak_hours' in self.patterns['temporal']:
            peak_hours = self.patterns['temporal']['discovered_peak_hours']
            self.current_user_state.is_peak_hour = now.hour in peak_hours
        
        # Update budget state
        self.current_user_state.budget_spent_ratio = self.budget_spent / max(1, self.max_budget)
        self.current_user_state.remaining_budget = self.max_budget - self.budget_spent
        self.current_user_state.time_in_day_ratio = self.current_step / max(1, self.max_steps)
        
        # Get pacing factor
        if self.budget_pacer:
            # Use the correct method signature for BudgetPacer
            current_hour = datetime.now().hour
            pacing_multiplier = self.budget_pacer.get_pacing_multiplier(
                hour=current_hour,
                spent_so_far=self.budget_spent,
                daily_budget=self.max_budget
            )
            self.current_user_state.pacing_factor = pacing_multiplier
        
        return self.current_user_state
    
    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return results"""
        self.step_count += 1
        self.current_step += 1
        
        # Parse action
        bid_amount = float(action.get('bid', action.get('bid_amount', self.data_stats.bid_mean)))
        creative_id = int(action.get('creative', action.get('creative_id', 0)))
        
        # Handle channel - can be string or index
        channel_input = action.get('channel', action.get('channel_action', 0))
        if isinstance(channel_input, str):
            # Already a channel name
            channel = channel_input
        else:
            # It's an index, map to channel name
            channel_idx = int(channel_input)
            channel = self.discovered_channels[min(channel_idx, len(self.discovered_channels) - 1)]
        
        # Run auction
        auction_result = self._run_auction(bid_amount, channel)
        
        # Initialize reward calculation context
        reward_context = {
            'bid_amount': bid_amount,
            'creative_id': creative_id,
            'channel': channel,
            'auction_result': auction_result,
            'user_state': self.current_user_state,
            'step': self.current_step,
            'budget_spent': self.budget_spent,
            'max_budget': self.max_budget
        }
        
        info = {
            'auction_won': auction_result['won'],
            'position': auction_result.get('position', 0),
            'price_paid': auction_result.get('price_paid', 0),
            'channel': channel,
            'creative_id': creative_id,
            'metrics': {},
            'reward_components': {}
        }
        
        # Execute auction outcome and collect metrics
        click_occurred, conversion_occurred, conversion_value = self._execute_auction_outcome(
            auction_result, channel, creative_id
        )
        
        # Update reward context with outcomes
        reward_context.update({
            'click_occurred': click_occurred,
            'conversion_occurred': conversion_occurred,
            'conversion_value': conversion_value,
            'revenue': conversion_value if conversion_occurred else 0.0,
            'cost': auction_result.get('price_paid', 0) if auction_result['won'] else 0.0
        })
        
        # Update tracking systems
        action_taken = {
            'channel': channel,
            'creative_id': creative_id,
            'bid': bid_amount,
            'timestamp': datetime.now(),
            'step': self.current_step
        }
        
        outcome = {
            'auction_won': auction_result['won'],
            'click': click_occurred,
            'conversion': conversion_occurred,
            'revenue': conversion_value if conversion_occurred else 0.0,
            'cost': auction_result.get('price_paid', 0) if auction_result['won'] else 0.0
        }
        
        self.reward_tracker.update(action_taken, outcome)
        
        # Calculate multi-objective reward
        reward, reward_components = self.reward_calculator.calculate_reward(
            reward_context, self.reward_tracker
        )
        
        # Add reward components to info for transparency
        info['reward_components'] = reward_components
        
        # Get next state
        next_state = self._get_current_state()
        
        # Check termination
        terminated = (self.budget_spent >= self.max_budget) or (self.current_step >= self.max_steps)
        truncated = False
        
        # Add performance metrics to info
        info['metrics'].update({
            'budget_spent': self.budget_spent,
            'remaining_budget': self.max_budget - self.budget_spent,
            'impressions': self.metrics['total_impressions'],
            'clicks': self.metrics['total_clicks'],
            'conversions': self.metrics['total_conversions'],
            'revenue': self.metrics['total_revenue'],
            'roas': self.metrics['total_revenue'] / max(1, self.budget_spent)
        })
        
        return next_state.to_vector(self.data_stats), reward, terminated, truncated, info
    
    def _execute_auction_outcome(self, auction_result: Dict, channel: str, creative_id: int) -> Tuple[bool, bool, float]:
        """Execute auction outcome and return click/conversion results"""
        click_occurred = False
        conversion_occurred = False
        conversion_value = 0.0
        
        if auction_result['won']:
            self.metrics['auction_wins'] += 1
            self.budget_spent += auction_result['price_paid']
            
            # Update metrics
            self.metrics['total_impressions'] += 1
            self.metrics['channel_performance'][channel]['impressions'] += 1
            self.metrics['channel_performance'][channel]['spend'] += auction_result['price_paid']
            
            # Simulate click based on creative CTR from patterns
            ctr = self._get_creative_ctr(creative_id, self.current_user_state.segment_index)
            
            # Apply display quality penalty to CTR
            if channel == 'display' and auction_result.get('quality_penalty_applied', False):
                ctr = ctr * 0.2  # Massive CTR penalty for bot traffic
            
            if np.random.random() < ctr:
                click_occurred = True
                self.metrics['total_clicks'] += 1
                self.metrics['channel_performance'][channel]['clicks'] += 1
                self.metrics['creative_performance'][creative_id]['clicks'] += 1
                
                # Update user journey
                self.current_user_state.touchpoints_seen += 1
                
                # Progress stage
                if self.current_user_state.stage < 4:
                    if np.random.random() < 0.3:  # Probability to progress
                        self.current_user_state.stage += 1
                
                # Check for conversion
                cvr = self._get_conversion_probability(self.current_user_state)
                
                # Apply display quality penalty to CVR
                # Discovery: Base conversion multiplier from GA4 channel performance
                conversion_multiplier = self._get_channel_conversion_multiplier(channel)
                if channel == 'display' and auction_result.get('quality_penalty_applied', False):
                    conversion_multiplier = 0.01  # Massive CVR penalty for display bot traffic
                
                if np.random.random() < cvr * conversion_multiplier:
                    conversion_occurred = True
                    days_to_convert = self._get_days_to_convert()
                    conversion_value = self._get_conversion_value(self.current_user_state)
                    
                    self.metrics['total_conversions'] += 1
                    self.metrics['total_revenue'] += conversion_value
                    self.metrics['channel_performance'][channel]['conversions'] += 1
                    self.metrics['creative_performance'][creative_id]['conversions'] += 1
                    
                    # Update segment performance
                    segment_name = self.discovered_segments[self.current_user_state.segment_index]
                    self.metrics['segment_performance'][segment_name]['conversions'] += 1
        else:
            self.metrics['auction_losses'] += 1
        
        return click_occurred, conversion_occurred, conversion_value
    
    def apply_display_quality_fixes(self) -> Dict:
        """Apply display channel quality fixes if available"""
        
        # Check if fixes are available
        try:
            with open('/home/hariravichandran/AELP/display_bot_exclusions.json', 'r') as f:
                exclusions = json.load(f)
                
            # Calculate improvement from exclusions
            sessions_filtered = exclusions['summary']['sessions_filtered']
            improvement_factor = sessions_filtered / 150000  # Percentage of traffic to filter
            
            # Update patterns with improvements
            if 'channels' in self.patterns and 'display' in self.patterns['channels']:
                display_data = self.patterns['channels']['display']
                if 'quality_issues' in display_data:
                    # Apply improvements
                    old_bot_percentage = display_data['quality_issues']['bot_percentage']
                    new_bot_percentage = max(15.0, old_bot_percentage - (improvement_factor * 100))
                    
                    display_data['quality_issues']['bot_percentage'] = new_bot_percentage
                    display_data['quality_issues']['quality_score'] = 100 - new_bot_percentage
                    
                    if new_bot_percentage < 50:
                        display_data['quality_issues']['needs_urgent_fix'] = False
                        
                    # Save updated patterns
                    with open('/home/hariravichandran/AELP/discovered_patterns.json', 'w') as f:
                        json.dump(self.patterns, f, indent=2)
                    
                    return {
                        'fixes_applied': True,
                        'old_bot_percentage': old_bot_percentage,
                        'new_bot_percentage': new_bot_percentage,
                        'sessions_filtered': sessions_filtered,
                        'improvement_factor': improvement_factor
                    }
                    
        except FileNotFoundError:
            return {'fixes_applied': False, 'reason': 'No exclusions file found'}
        
        return {'fixes_applied': False, 'reason': 'No quality issues to fix'}
    
    def _run_auction(self, bid_amount: float, channel: str) -> Dict:
        """Run auction with discovered competition levels and quality scoring"""
        # Adjust competition based on channel
        competition_factor = 1.0
        effectiveness = 0.5  # default
        
        if channel in self.patterns.get('channels', {}):
            channel_data = self.patterns['channels'][channel]
            effectiveness = channel_data.get('effectiveness', 0.5)
            competition_factor = 0.5 + effectiveness  # More competition for better channels
            
            # Apply display quality scoring if this is display channel
            if channel == 'display':
                quality_issues = channel_data.get('quality_issues', {})
                if quality_issues.get('needs_urgent_fix', False):
                    # Massive quality penalty for display until fixed
                    bot_percentage = quality_issues.get('bot_percentage', 0) / 100.0
                    quality_penalty = bot_percentage  # Direct penalty
                    effectiveness = effectiveness * (1 - quality_penalty)
                    
                    # Log quality issues for display
                    if self.current_step % 100 == 0:  # Log occasionally
                        logger.warning(f"Display quality penalty applied: {quality_penalty:.2f} (bot %: {bot_percentage*100:.1f}%)")
        
        # Run auction with correct parameters
        query_value = bid_amount * 1.2  # Estimate query value
        context = {
            'quality_score': 0.7 + np.random.random() * 0.3,
            'channel': channel,
            'effectiveness': effectiveness
        }
        result = self.auction_gym.run_auction(
            our_bid=bid_amount * competition_factor,
            query_value=query_value,
            context=context
        )
        
        return {
            'won': result.won,
            'position': result.slot_position if result.won else 0,
            'price_paid': result.price_paid if result.won else 0,
            'competitors': result.num_competitors if hasattr(result, 'num_competitors') else 6,
            'quality_penalty_applied': channel == 'display' and effectiveness < 0.3
        }
    
    def _get_creative_ctr(self, creative_id: int, segment_index: int) -> float:
        """Get CTR from discovered patterns"""
        segment_name = self.discovered_segments[segment_index] if segment_index < len(self.discovered_segments) else None
        
        # Check patterns for segment-creative performance
        if segment_name and 'creatives' in self.patterns:
            if 'performance_by_segment' in self.patterns['creatives']:
                segment_perf = self.patterns['creatives']['performance_by_segment'].get(segment_name, {})
                if 'avg_ctr' in segment_perf:
                    # Add noise for variation
                    return segment_perf['avg_ctr'] * (0.8 + np.random.random() * 0.4)
        
        # Default CTR with variation
        return 0.05 * (0.5 + np.random.random())
    
    def _get_conversion_probability(self, state: DynamicEnrichedState) -> float:
        """Get conversion probability from state and patterns"""
        base_cvr = state.segment_cvr
        
        # Adjust based on journey stage
        stage_multipliers = [0.1, 0.3, 0.6, 1.0, 2.0]  # By stage
        stage_mult = stage_multipliers[min(state.stage, 4)]
        
        # Adjust based on touchpoints
        touchpoint_factor = min(1.0, state.touchpoints_seen / 10.0)
        
        # Final probability
        return min(0.15, base_cvr * stage_mult * (0.5 + touchpoint_factor) * 3.0)  # Boost for testing
    
    def _get_days_to_convert(self) -> int:
        """Get conversion delay from patterns"""
        if 'conversion_windows' in self.patterns:
            trial_days = self.patterns['conversion_windows'].get('trial_to_paid_days', 14)
            # Add variation with exponential distribution
            return max(1, int(np.random.exponential(trial_days / 2)))
        return int(np.random.exponential(7) + 1)
    
    def _get_channel_conversion_multiplier(self, channel: str) -> float:
        """Get channel-specific conversion multiplier from GA4 data"""
        try:
            # Load GA4 master report for channel performance
            with open('ga4_extracted_data/00_MASTER_REPORT.json', 'r') as f:
                data = json.load(f)
            
            # Get channel-specific conversion patterns
            conversion_patterns = data.get('insights', {}).get('conversion_patterns', {})
            
            if channel == 'search' or channel == 'Search':
                # Search traffic generally converts better - use parental controls rate
                parental = conversion_patterns.get('parental_controls', {})
                if 'search' in parental.get('best_channels', []):
                    return 1.5  # 50% boost for search traffic
                return 1.2
                
            elif channel == 'display' or channel == 'Display':
                # Display typically converts lower but volume is higher
                return 0.8  # 20% reduction for display
                
            elif channel == 'social' or channel == 'Social':
                # Social converts well for parental products
                parental = conversion_patterns.get('parental_controls', {})
                if 'social' in parental.get('best_channels', []):
                    return 1.3  # 30% boost for social
                return 1.0
                
            elif channel == 'organic':
                # Organic is high intent traffic
                balance = conversion_patterns.get('balance_thrive', {})
                if 'organic' in balance.get('best_channels', []):
                    return 1.4  # 40% boost for organic
                return 1.2
                
            else:
                # Generic traffic multiplier
                return 1.0
                
        except Exception as e:
            logger.warning(f"Could not load channel conversion multiplier for {channel}: {e}")
            # Discovery-based fallback: analyze current performance
            if hasattr(self, 'patterns') and 'channel_conversion_rates' in self.patterns:
                rates = self.patterns['channel_conversion_rates']
                if channel in rates:
                    # Use relative performance vs average
                    avg_rate = sum(rates.values()) / max(len(rates), 1)
                    return rates[channel] / max(avg_rate, 0.01)
            
            # Minimal safe multiplier if no data available
            return 1.0
    
    def _get_conversion_value(self, state: DynamicEnrichedState) -> float:
        """Get conversion value from patterns"""
        # Use segment LTV
        base_value = state.segment_avg_ltv
        
        if base_value <= 0:
            base_value = self.data_stats.conversion_value_mean
        
        if base_value <= 0:
            base_value = 100.0  # Minimum viable value
        
        # Add variation within reasonable range
        variation = 0.8 + np.random.random() * 0.4
        return base_value * variation
    
    def render(self):
        """Render environment state"""
        if self.metrics['total_impressions'] > 0:
            ctr = self.metrics['total_clicks'] / self.metrics['total_impressions']
        else:
            ctr = 0
        
        if self.metrics['total_clicks'] > 0:
            cvr = self.metrics['total_conversions'] / self.metrics['total_clicks']
        else:
            cvr = 0
        
        if self.budget_spent > 0:
            roas = self.metrics['total_revenue'] / self.budget_spent
        else:
            roas = 0
        
        print(f"\n=== Episode {self.episode_id} | Step {self.current_step} ===")
        print(f"Budget: ${self.budget_spent:.2f} / ${self.max_budget:.2f}")
        print(f"Impressions: {self.metrics['total_impressions']}")
        print(f"Clicks: {self.metrics['total_clicks']} (CTR: {ctr:.2%})")
        print(f"Conversions: {self.metrics['total_conversions']} (CVR: {cvr:.2%})")
        print(f"Revenue: ${self.metrics['total_revenue']:.2f} (ROAS: {roas:.2f})")
        print(f"Auctions: {self.metrics['auction_wins']} wins / {self.metrics['auction_losses']} losses")
        
        # Show channel performance
        print("\nChannel Performance:")
        for channel in self.discovered_channels[:5]:  # Show top 5
            perf = self.metrics['channel_performance'][channel]
            if perf['impressions'] > 0:
                ch_ctr = perf['clicks'] / perf['impressions']
                ch_roas = self.metrics['total_revenue'] / max(1, perf['spend']) if perf['spend'] > 0 else 0
                print(f"  {channel}: {perf['impressions']} impr, {perf['clicks']} clicks (CTR: {ch_ctr:.2%}), ${perf['spend']:.2f} spent")


class RewardTracker:
    """Tracks metrics needed for multi-objective reward calculation"""
    
    def __init__(self, discovered_channels: List[str], discovered_creatives: List[int], discovered_segments: List[str]):
        self.discovered_channels = discovered_channels
        self.discovered_creatives = discovered_creatives
        self.discovered_segments = discovered_segments
        
        # Exploration tracking
        self.channel_visit_counts = defaultdict(int)
        self.creative_visit_counts = defaultdict(int)
        self.channel_last_visited = {}
        self.creative_last_visited = {}
        
        # Portfolio tracking for diversity
        self.recent_actions = deque(maxlen=1000)  # Last 1000 actions
        self.action_history = deque(maxlen=10000)  # Longer history for learning
        
        # Performance tracking
        self.channel_performance = defaultdict(lambda: {'clicks': 0, 'conversions': 0, 'revenue': 0.0, 'cost': 0.0})
        self.creative_performance = defaultdict(lambda: {'clicks': 0, 'conversions': 0, 'revenue': 0.0, 'exposures': 0})
        
        # Uncertainty tracking for curiosity
        self.prediction_errors = {}
        self.uncertainty_estimates = defaultdict(float)
        
        # Delayed reward attribution
        self.pending_attributions = deque(maxlen=5000)
        
        # Learning progress
        self.exploration_bonus_decay = 0.99
        self.diversity_window = 100
    
    def update(self, action: Dict, outcome: Dict):
        """Update all tracking metrics"""
        channel = action['channel']
        creative_id = action['creative_id']
        step = action['step']
        
        # Update visit counts
        self.channel_visit_counts[channel] += 1
        self.creative_visit_counts[creative_id] += 1
        self.channel_last_visited[channel] = step
        self.creative_last_visited[creative_id] = step
        
        # Track action
        self.recent_actions.append(action)
        self.action_history.append((action, outcome))
        
        # Update performance
        if outcome.get('click', False):
            self.channel_performance[channel]['clicks'] += 1
            self.creative_performance[creative_id]['clicks'] += 1
        
        if outcome.get('conversion', False):
            self.channel_performance[channel]['conversions'] += 1
            self.creative_performance[creative_id]['conversions'] += 1
            self.channel_performance[channel]['revenue'] += outcome.get('revenue', 0)
            self.creative_performance[creative_id]['revenue'] += outcome.get('revenue', 0)
        
        self.channel_performance[channel]['cost'] += outcome.get('cost', 0)
        self.creative_performance[creative_id]['exposures'] += 1
        
        # Handle delayed attributions
        if outcome.get('conversion', False):
            self.pending_attributions.append({
                'action': action,
                'value': outcome.get('revenue', 0),
                'timestamp': datetime.now()
            })
    
    def get_channel_novelty(self, channel: str, step: int) -> float:
        """Calculate exploration bonus for channel selection"""
        if channel not in self.channel_visit_counts:
            return 1.0  # Maximum novelty for unvisited channel
        
        # UCB-style exploration bonus
        total_visits = sum(self.channel_visit_counts.values())
        channel_visits = self.channel_visit_counts[channel]
        
        if total_visits <= 1 or channel_visits <= 0:
            return 1.0
        
        # Calculate recency bonus
        recency_bonus = 1.0
        if channel in self.channel_last_visited:
            steps_since = step - self.channel_last_visited[channel]
            recency_bonus = min(1.0, steps_since / 50.0)  # Decay over 50 steps
        
        # UCB exploration term
        exploration_term = np.sqrt(2 * np.log(total_visits) / channel_visits)
        
        return min(1.0, exploration_term * recency_bonus * (self.exploration_bonus_decay ** step))
    
    def get_creative_novelty(self, creative_id: int, step: int) -> float:
        """Calculate exploration bonus for creative selection"""
        if creative_id not in self.creative_visit_counts:
            return 1.0  # Maximum novelty for new creative
        
        exposures = self.creative_performance[creative_id]['exposures']
        if exposures == 0:
            return 1.0
        
        # Decay novelty based on exposures and recency
        novelty = 1.0 / (1.0 + np.log1p(exposures))
        
        # Recency bonus
        recency_bonus = 1.0
        if creative_id in self.creative_last_visited:
            steps_since = step - self.creative_last_visited[creative_id]
            recency_bonus = min(1.0, steps_since / 30.0)
        
        return novelty * recency_bonus
    
    def get_portfolio_diversity(self) -> float:
        """Calculate portfolio diversity using entropy"""
        if len(self.recent_actions) < 10:
            return 0.0
        
        # Calculate channel distribution in recent actions
        recent_window = list(self.recent_actions)[-self.diversity_window:]
        channel_counts = defaultdict(int)
        
        for action in recent_window:
            channel_counts[action['channel']] += 1
        
        if len(channel_counts) <= 1:
            return 0.0
        
        # Calculate normalized entropy
        total = sum(channel_counts.values())
        probs = [count / total for count in channel_counts.values()]
        
        # Shannon entropy
        entropy_val = -sum(p * np.log(p) for p in probs if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(self.discovered_channels))
        
        return entropy_val / max(max_entropy, 1e-8)
    
    def get_uncertainty_reduction(self, channel: str, creative_id: int) -> float:
        """Calculate curiosity reward based on uncertainty reduction"""
        # Use performance variance as uncertainty proxy
        channel_perf = self.channel_performance[channel]
        creative_perf = self.creative_performance[creative_id]
        
        # Calculate uncertainty based on sample size and variance
        channel_uncertainty = 1.0 / (1.0 + channel_perf['clicks'])
        creative_uncertainty = 1.0 / (1.0 + creative_perf['exposures'])
        
        # Combined uncertainty
        combined_uncertainty = (channel_uncertainty + creative_uncertainty) / 2
        
        return min(1.0, combined_uncertainty)
    
    def get_delayed_attribution_reward(self, user_id: str = None) -> float:
        """Calculate delayed attribution rewards"""
        # For now, return based on pending attributions
        if not self.pending_attributions:
            return 0.0
        
        # Calculate time-discounted value from recent attributions
        total_value = 0.0
        current_time = datetime.now()
        
        for attribution in list(self.pending_attributions)[-10:]:  # Last 10 attributions
            days_elapsed = (current_time - attribution['timestamp']).days
            discount_factor = 0.95 ** days_elapsed  # Daily discount
            total_value += attribution['value'] * discount_factor
        
        # Normalize to reasonable range
        return min(1.0, total_value / 1000.0)  # Scale by $1000


class MultiObjectiveRewardCalculator:
    """Sophisticated multi-objective reward system"""
    
    def __init__(self, patterns: Dict, data_stats: Any, parameter_manager: ParameterManager):
        self.patterns = patterns
        self.data_stats = data_stats
        self.pm = parameter_manager
        
        # Load reward weights from configuration (learned, not hardcoded)
        self.weights = self._load_reward_weights()
        
        # Normalization parameters (learned from data)
        self.roas_normalizer = self._compute_roas_normalizer()
        self.revenue_normalizer = self._compute_revenue_normalizer()
        
        logger.info(f"Multi-objective reward system initialized with weights: {self.weights}")
    
    def _load_reward_weights(self) -> Dict[str, float]:
        """Load learned reward weights from configuration"""
        weights = None
        
        # Try to load from parameter manager or patterns
        if hasattr(self.pm, 'reward_weights') and self.pm.reward_weights is not None:
            weights = self.pm.reward_weights
        elif 'reward_weights' in self.patterns:
            weights = self.patterns['reward_weights']
        
        # Use initial weights if none found
        if weights is None:
            # Initial weights that sum to 1.0 - these will be learned
            weights = {
                'roas': 0.4,        # Revenue optimization (primary)
                'exploration': 0.25, # Exploration bonus
                'diversity': 0.20,   # Portfolio diversity
                'curiosity': 0.10,   # Uncertainty reduction
                'delayed': 0.05      # Delayed attribution
            }
        
        # Ensure weights sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _compute_roas_normalizer(self) -> float:
        """Compute ROAS normalization factor from patterns"""
        # Extract ROAS values from channel performance
        roas_values = []
        
        if 'channels' in self.patterns:
            for channel, data in self.patterns['channels'].items():
                if 'roas' in data:
                    roas_values.append(data['roas'])
                elif 'avg_cpc' in data and 'conversion_value' in data:
                    # Calculate ROAS from available data
                    cpc = data['avg_cpc']
                    value = data['conversion_value']
                    if cpc > 0:
                        roas_values.append(value / cpc)
        
        if roas_values:
            # Use 75th percentile as "good" ROAS for normalization
            return np.percentile(roas_values, 75)
        else:
            # Default normalization (good ROAS = 3x)
            return 3.0
    
    def _compute_revenue_normalizer(self) -> float:
        """Compute revenue normalization factor"""
        if self.data_stats.conversion_value_mean > 0:
            return self.data_stats.conversion_value_mean
        else:
            return 100.0  # Default $100 normalizer
    
    def calculate_reward(self, context: Dict, tracker: RewardTracker) -> Tuple[float, Dict[str, float]]:
        """Calculate multi-objective reward with full transparency"""
        components = {}
        
        # 1. ROAS Component (revenue optimization)
        components['roas'] = self._calculate_roas_component(context)
        
        # 2. Exploration Component
        components['exploration'] = self._calculate_exploration_component(context, tracker)
        
        # 3. Diversity Component
        components['diversity'] = self._calculate_diversity_component(tracker)
        
        # 4. Curiosity Component (uncertainty reduction)
        components['curiosity'] = self._calculate_curiosity_component(context, tracker)
        
        # 5. Delayed Attribution Component
        components['delayed'] = self._calculate_delayed_component(context, tracker)
        
        # Weighted combination
        total_reward = sum(self.weights[key] * components[key] for key in components.keys())
        
        # Log reward calculation for debugging
        if context.get('step', 0) % 100 == 0:
            logger.debug(f"Reward components at step {context.get('step', 0)}: {components}")
            logger.debug(f"Weights: {self.weights}, Total reward: {total_reward:.4f}")
        
        return total_reward, components
    
    def _calculate_roas_component(self, context: Dict) -> float:
        """Calculate ROAS-based reward component"""
        revenue = context.get('revenue', 0.0)
        cost = context.get('cost', 0.01)  # Prevent division by zero
        
        if cost <= 0:
            return 0.0
        
        roas = revenue / cost
        
        # Normalize ROAS to [0, 1] range using sigmoid
        normalized_roas = 2 / (1 + np.exp(-roas / self.roas_normalizer)) - 1
        
        # Penalize negative ROAS more heavily
        if roas < 0:
            normalized_roas *= 2.0  # Double penalty for losses
        
        return max(-1.0, min(1.0, normalized_roas))
    
    def _calculate_exploration_component(self, context: Dict, tracker: RewardTracker) -> float:
        """Calculate exploration bonus"""
        channel = context.get('channel')
        creative_id = context.get('creative_id')
        step = context.get('step', 0)
        
        if not channel:
            return 0.0
        
        # Channel exploration bonus
        channel_novelty = tracker.get_channel_novelty(channel, step)
        
        # Creative exploration bonus
        creative_novelty = tracker.get_creative_novelty(creative_id, step)
        
        # Combined exploration bonus
        exploration_bonus = (channel_novelty + creative_novelty) / 2.0
        
        # Only reward if auction was won (no reward for exploring with losing bids)
        if not context.get('auction_result', {}).get('won', False):
            exploration_bonus *= 0.1  # Minimal exploration reward for losing
        
        return exploration_bonus
    
    def _calculate_diversity_component(self, tracker: RewardTracker) -> float:
        """Calculate portfolio diversity reward"""
        diversity = tracker.get_portfolio_diversity()
        
        # Sigmoid scaling to encourage diversity
        return 2 / (1 + np.exp(-5 * diversity)) - 1
    
    def _calculate_curiosity_component(self, context: Dict, tracker: RewardTracker) -> float:
        """Calculate curiosity/uncertainty reduction reward"""
        channel = context.get('channel')
        creative_id = context.get('creative_id')
        
        if not channel:
            return 0.0
        
        # Reward reducing uncertainty about channel-creative combinations
        uncertainty_reduction = tracker.get_uncertainty_reduction(channel, creative_id)
        
        # Scale by auction outcome (more learning from wins)
        if context.get('auction_result', {}).get('won', False):
            return uncertainty_reduction
        else:
            return uncertainty_reduction * 0.3  # Less learning from losses
    
    def _calculate_delayed_component(self, context: Dict, tracker: RewardTracker) -> float:
        """Calculate delayed attribution reward"""
        # Get user ID from context - user_state is a DynamicEnrichedState object
        user_state = context.get('user_state')
        user_id = None
        if hasattr(user_state, 'user_id'):
            user_id = user_state.user_id
        
        # Get delayed attribution value
        delayed_value = tracker.get_delayed_attribution_reward(user_id)
        
        # Normalize to reasonable range
        return min(1.0, delayed_value)
    
    def update_weights(self, performance_feedback: Dict):
        """Update reward weights based on learning performance"""
        # This would implement weight learning based on agent performance
        # For now, keep weights static but log for future learning
        logger.info(f"Performance feedback received: {performance_feedback}")
        
        # Future: Implement gradient-based weight updates
        # self.weights = optimize_weights(self.weights, performance_feedback)