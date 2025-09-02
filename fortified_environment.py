#!/usr/bin/env python3
"""
FORTIFIED GAELP ENVIRONMENT
Complete integration of all components for realistic marketing simulation
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces

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
from fortified_rl_agent import EnrichedJourneyState, NUM_CREATIVES, NUM_CHANNELS

# Import GA4 integration
try:
    import sys
    sys.path.insert(0, '/home/hariravichandran/AELP')
    from mcp_ga4_integration import GA4DataFetcher
    GA4_AVAILABLE = True
except:
    GA4_AVAILABLE = False

logger = logging.getLogger(__name__)

class FortifiedGAELPEnvironment(gym.Env):
    """
    Fortified environment with complete component integration
    """
    
    def __init__(self,
                 max_budget: float = 10000.0,
                 max_steps: int = 1000,
                 use_real_ga4_data: bool = True,
                 is_parallel: bool = False):
        """
        Initialize fortified environment with all components
        """
        super().__init__()
        
        self.max_budget = max_budget
        self.max_steps = max_steps
        self.use_real_ga4_data = use_real_ga4_data and GA4_AVAILABLE
        
        # Initialize all components
        logger.info("Initializing fortified GAELP environment...")
        
        # 1. Discovery Engine for pattern discovery
        # Use cache_only mode if running in parallel to prevent file corruption
        import os
        self.discovery = DiscoveryEngine(
            write_enabled=not is_parallel,  # Only main process writes
            cache_only=is_parallel  # Parallel processes use cache only
        )
        patterns = self.discovery.discover_all_patterns()
        if not is_parallel:
            logger.info(f"Discovered {len(patterns.user_patterns.get('segments', {}))} segments")
        
        # 2. Parameter Manager with discovered patterns
        self.pm = ParameterManager()
        
        # 3. Creative Selector with A/B testing
        self.creative_selector = CreativeSelector()
        self._initialize_creatives()
        logger.info(f"Initialized {len(self.creative_selector.creatives)} creatives")
        
        # 4. Attribution Engine
        self.attribution = AttributionEngine()
        
        # 5. Persistent User Database with batching
        self.user_db = BatchedPersistentUserDatabase(
            use_batch_writer=True,
            batch_size=100,
            flush_interval=5.0
        )
        
        # 6. Delayed Conversion System
        from user_journey_database import UserJourneyDatabase
        from conversion_lag_model import ConversionLagModel
        
        self.journey_db = UserJourneyDatabase()
        lag_model = ConversionLagModel()
        self.conversion_system = DelayedConversionSystem(
            journey_database=self.journey_db,
            attribution_engine=self.attribution,
            conversion_lag_model=lag_model
        )
        
        # 7. Budget Pacer
        self.budget_pacer = BudgetPacer()
        
        # 8. Identity Resolver
        self.identity_resolver = IdentityResolver()
        
        # 9. Auction System with competitors
        self.auction_gym = self._initialize_auction_system()
        
        # 10. GA4 Data Integration
        if self.use_real_ga4_data:
            self.ga4_fetcher = GA4DataFetcher()
            self._load_real_conversion_data()
        
        # Environment state
        self.current_step = 0
        self.step_count = 0
        self.budget_spent = 0.0
        self.episode_id = f"episode_{datetime.now().timestamp()}"
        
        # Tracking metrics
        self.metrics = {
            'total_impressions': 0,
            'total_clicks': 0,
            'total_conversions': 0,
            'total_revenue': 0.0,
            'auction_wins': 0,
            'auction_losses': 0,
            'creative_performance': {},
            'channel_performance': {},
            'user_journeys': {}
        }
        
        # Action and observation spaces
        self.action_space = spaces.Dict({
            'bid': spaces.Box(low=0.5, high=10.0, shape=(1,), dtype=np.float32),
            'creative': spaces.Discrete(NUM_CREATIVES),
            'channel': spaces.Discrete(NUM_CHANNELS)
        })
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(45,), dtype=np.float32
        )
        
        logger.info("Fortified environment initialized successfully")
    
    def _initialize_creatives(self):
        """Initialize creative library with discovered segments"""
        patterns = self.discovery.discover_all_patterns()
        segments = patterns.user_patterns.get('segments', {})
        
        # Create creatives for each discovered segment
        for segment_name, segment_data in segments.items():
            # Create multiple creative variants per segment
            for i in range(5):
                creative_id = f"{segment_name}_creative_{i}"
                
                # Use segment characteristics to inform creative
                if segment_data.get('conversion_rate', 0) > 0.04:
                    # High converting segment - urgency messaging
                    headline = f"Limited Time: Solution for {segment_name.replace('_', ' ').title()}"
                    cta = "Get Started Now"
                elif segment_data.get('engagement_score', 0) > 0.7:
                    # High engagement - educational content
                    headline = f"Learn How to Help Your {segment_name.replace('_', ' ').title()}"
                    cta = "Read More"
                else:
                    # General awareness
                    headline = f"Discover Solutions for {segment_name.replace('_', ' ').title()}"
                    cta = "Learn More"
                
                # Note: CreativeSelector handles adding creatives internally
                # This is just to ensure we have segment-specific content
    
    def _initialize_auction_system(self) -> AuctionGymWrapper:
        """Initialize auction with realistic competitors"""
        
        # Use AuctionGymWrapper with proper configuration
        config = {
            'auction_type': 'second_price',
            'num_slots': 4,
            'reserve_price': 0.50,
            'competitors': {
                'count': 6  # Realistic number of competitors
            }
        }
        
        return AuctionGymWrapper(config)
    
    def _load_real_conversion_data(self):
        """Load real conversion patterns from GA4"""
        if not self.ga4_fetcher:
            return
        
        try:
            # Get real conversion data from last 30 days
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Fetch conversion events
            conversions = self.ga4_fetcher.get_events(
                start_date=start_date,
                end_date=end_date,
                event_name='purchase'
            )
            
            # Analyze conversion patterns
            if conversions:
                logger.info(f"Loaded {len(conversions)} real conversion events from GA4")
                
                # Extract patterns for realistic simulation
                self._analyze_conversion_patterns(conversions)
        except Exception as e:
            logger.warning(f"Could not load GA4 data: {e}")
    
    def _analyze_conversion_patterns(self, conversions: List[Dict]):
        """Analyze real conversion patterns for simulation"""
        # Extract:
        # - Time to conversion distribution
        # - Device/channel distribution
        # - Value distribution
        # - User journey patterns
        
        conversion_times = []
        conversion_values = []
        conversion_channels = []
        
        for conv in conversions:
            # Extract relevant metrics
            if 'days_to_convert' in conv:
                conversion_times.append(conv['days_to_convert'])
            if 'value' in conv:
                conversion_values.append(conv['value'])
            if 'channel' in conv:
                conversion_channels.append(conv['channel'])
        
        # Store patterns for use in simulation
        self.real_conversion_patterns = {
            'avg_days_to_convert': np.mean(conversion_times) if conversion_times else 7.0,
            'avg_conversion_value': np.mean(conversion_values) if conversion_values else 100.0,
            'channel_distribution': self._calculate_distribution(conversion_channels)
        }
    
    def _calculate_distribution(self, items: List[str]) -> Dict[str, float]:
        """Calculate probability distribution from list"""
        from collections import Counter
        counts = Counter(items)
        total = sum(counts.values())
        return {k: v/total for k, v in counts.items()} if total > 0 else {}
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        self.current_step = 0
        self.step_count = 0
        self.budget_spent = 0.0
        self.episode_id = f"episode_{datetime.now().timestamp()}"
        
        # Reset metrics
        for key in self.metrics:
            if isinstance(self.metrics[key], dict):
                self.metrics[key].clear()
            elif isinstance(self.metrics[key], (int, float)):
                self.metrics[key] = 0
        
        # Get initial user and state
        self.current_user = self._get_or_create_user()
        self.current_state = self._create_enriched_state()
        
        return self.current_state.to_vector()
    
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action in environment"""
        self.current_step += 1
        self.step_count += 1
        
        # Handle both key formats (from agent and from test)
        bid = action.get('bid', action.get('bid_amount', 0))
        creative = action.get('creative', action.get('creative_id', 0))
        channel = action.get('channel', 'organic')
        
        # 1. Run auction with bid
        auction_result = self._run_auction(bid)
        
        # 2. If won, serve creative and track impression
        if auction_result['won']:
            self._serve_impression(creative, channel)
            
            # 3. Simulate user response
            user_response = self._simulate_user_response(
                creative,
                channel
            )
            
            # 4. Update metrics
            self._update_metrics(auction_result, user_response, 
                                {'bid': bid, 'creative': creative, 'channel': channel})
            
            # 5. Track for attribution
            self._track_touchpoint(channel, auction_result, user_response)
        
        # 6. Process delayed conversions
        self._process_delayed_conversions()
        
        # 7. Update user journey state
        self._update_user_journey(auction_result, action)
        
        # 8. Create next state
        next_state = self._create_enriched_state()
        
        # 9. Calculate sophisticated reward
        reward = self._calculate_reward(
            self.current_state,
            action,
            next_state,
            auction_result
        )
        
        # 10. Check if episode done
        done = (self.current_step >= self.max_steps or 
                self.budget_spent >= self.max_budget)
        
        # 11. Prepare info
        info = {
            'auction_result': auction_result,
            'metrics': self.metrics.copy(),
            'budget_remaining': self.max_budget - self.budget_spent
        }
        
        self.current_state = next_state
        
        return next_state.to_vector(), reward, done, info
    
    def _get_or_create_user(self):
        """Get or create persistent user"""
        # 70% returning users, 30% new
        if np.random.random() < 0.7 and self.metrics.get('user_journeys'):
            # Get existing user
            user_id = np.random.choice(list(self.metrics['user_journeys'].keys()))
        else:
            # Create new user
            user_id = f"user_{datetime.now().timestamp()}_{np.random.randint(10000)}"
        
        user, created = self.user_db.get_or_create_persistent_user(
            user_id=user_id,
            episode_id=self.episode_id,
            device_fingerprint={
                "device_id": f"device_{np.random.randint(10000)}",
                "platform": np.random.choice(['iOS', 'Android']),
                "timezone": "America/New_York"
            }
        )
        
        if user_id not in self.metrics['user_journeys']:
            self.metrics['user_journeys'][user_id] = {
                'touchpoints': [],
                'journey_state': 'unaware',
                'total_spend': 0.0
            }
        
        return user
    
    def _create_enriched_state(self) -> EnrichedJourneyState:
        """Create enriched state with all signals"""
        state = EnrichedJourneyState()
        
        # Get user journey info
        user_journey = self.metrics['user_journeys'].get(
            self.current_user.canonical_user_id, {}
        )
        
        # Map journey state
        journey_state_map = {
            'unaware': 0, 'aware': 1, 'considering': 2,
            'intent': 3, 'converted': 4
        }
        state.stage = journey_state_map.get(
            user_journey.get('journey_state', 'unaware'), 0
        )
        
        state.touchpoints_seen = len(user_journey.get('touchpoints', []))
        
        # Get segment from discovery
        patterns = self.discovery.discover_all_patterns()
        segments = list(patterns.user_patterns.get('segments', {}).keys())
        if segments:
            segment_idx = hash(self.current_user.canonical_user_id) % len(segments)
            segment_name = segments[segment_idx]
            segment_data = patterns.user_patterns['segments'][segment_name]
            
            state.segment = segment_idx
            state.segment_cvr = segment_data.get('conversion_rate', 0.02)
            state.segment_engagement = segment_data.get('engagement_score', 0.5)
        
        # Device and channel context
        state.device = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])  # mobile, desktop, tablet
        
        # Temporal patterns
        state.hour_of_day = datetime.now().hour
        state.day_of_week = datetime.now().weekday()
        peak_hours = patterns.temporal_patterns.get('peak_hours', [19, 20, 21])
        state.is_peak_hour = state.hour_of_day in peak_hours
        
        # Competition level
        state.competition_level = min(1.0, self.auction_gym.num_competitors / 10.0)
        
        # Budget pacing
        state.budget_spent_ratio = self.budget_spent / self.max_budget
        state.time_in_day_ratio = (datetime.now().hour / 24.0)
        state.remaining_budget = self.max_budget - self.budget_spent
        
        if self.budget_pacer:
            state.pacing_factor = self.budget_pacer.get_pacing_multiplier(
                hour=state.hour_of_day,
                spent_so_far=self.budget_spent,
                daily_budget=self.max_budget
            )
        
        # Identity resolution
        state.is_returning_user = not self.current_user.episode_count == 1
        state.num_devices_seen = len(self.current_user.device_ids)
        
        # Creative fatigue
        if self.creative_selector and hasattr(self.current_user, 'user_id'):
            recent_creatives = user_journey.get('recent_creatives', [])
            if recent_creatives:
                last_creative = recent_creatives[-1]
                state.creative_fatigue = self.creative_selector.calculate_fatigue(
                    str(last_creative),
                    self.current_user.user_id
                )
        
        # Channel performance
        if 'channel_performance' in self.metrics:
            for channel, perf in self.metrics['channel_performance'].items():
                if perf.get('spend', 0) > 0:
                    perf['roas'] = perf.get('revenue', 0) / perf['spend']
        
        # Conversion probability from real data or patterns
        if hasattr(self, 'real_conversion_patterns'):
            state.conversion_probability = state.segment_cvr * 1.5  # Boost with real data
            state.days_to_conversion_estimate = self.real_conversion_patterns.get(
                'avg_days_to_convert', 7.0
            )
        
        return state
    
    def _run_auction(self, bid_amount: float) -> Dict[str, Any]:
        """Run auction with competitors"""
        # Prepare context for auction
        context = {
            'quality_score': 7.5,  # Could be dynamic based on creative
            'estimated_ctr': 0.05,
            'user_segment': getattr(self.current_state, 'segment', 0),
            'hour_of_day': getattr(self.current_state, 'hour_of_day', 12),
            'customer_ltv': 199.98,  # Aura customer LTV
            'conversion_rate': 0.02
        }
        
        # Estimate query value based on context
        query_value = bid_amount * 1.2  # Simple heuristic
        
        result = self.auction_gym.run_auction(
            our_bid=bid_amount,
            query_value=query_value,
            context=context
        )
        
        if result.won:
            self.metrics['auction_wins'] += 1
            self.budget_spent += result.price_paid
        else:
            self.metrics['auction_losses'] += 1
        
        # Convert AuctionResult to dict for compatibility
        return {
            'won': result.won,
            'price_paid': result.price_paid,
            'position': result.slot_position,
            'competitors': result.competitors,
            'estimated_ctr': result.estimated_ctr,
            'true_ctr': result.true_ctr,
            'outcome': result.outcome,
            'revenue': result.revenue
        }
    
    def _serve_impression(self, creative_id: int, channel: str):
        """Track impression serving"""
        self.metrics['total_impressions'] += 1
        
        # Track creative performance
        if creative_id not in self.metrics['creative_performance']:
            self.metrics['creative_performance'][creative_id] = {
                'impressions': 0,
                'clicks': 0,
                'conversions': 0
            }
        self.metrics['creative_performance'][creative_id]['impressions'] += 1
        
        # Track channel performance
        if channel not in self.metrics['channel_performance']:
            self.metrics['channel_performance'][channel] = {
                'impressions': 0,
                'clicks': 0,
                'conversions': 0,
                'spend': 0.0,
                'revenue': 0.0
            }
        self.metrics['channel_performance'][channel]['impressions'] += 1
    
    def _simulate_user_response(self, creative_id: int, channel: str) -> Dict[str, Any]:
        """Simulate realistic user response"""
        response = {
            'clicked': False,
            'converted': False,
            'revenue': 0.0
        }
        
        # Base CTR depends on position and creative
        base_ctr = 0.05
        
        # Adjust for creative fatigue
        if hasattr(self.current_state, 'creative_fatigue'):
            ctr_multiplier = 1.0 - (self.current_state.creative_fatigue * 0.5)
            base_ctr *= ctr_multiplier
        
        # Click simulation
        if np.random.random() < base_ctr:
            response['clicked'] = True
            self.metrics['total_clicks'] += 1
            self.metrics['creative_performance'][creative_id]['clicks'] += 1
            self.metrics['channel_performance'][channel]['clicks'] += 1
            
            # Conversion simulation (immediate or delayed)
            if np.random.random() < self.current_state.conversion_probability:
                # Schedule delayed conversion
                days_to_convert = np.random.gamma(2, 3)  # Average 6 days
                
                if days_to_convert < 1:
                    # Immediate conversion
                    response['converted'] = True
                    response['revenue'] = np.random.gamma(2, 50)  # Average $100
                    self.metrics['total_conversions'] += 1
                    self.metrics['total_revenue'] += response['revenue']
                else:
                    # Schedule for later
                    self.conversion_system.schedule_conversion(
                        user_id=self.current_user.canonical_user_id,
                        days_to_convert=days_to_convert,
                        conversion_value=np.random.gamma(2, 50)
                    )
        
        return response
    
    def _track_touchpoint(self, channel: str, auction_result: Dict, user_response: Dict):
        """Track touchpoint for attribution"""
        touchpoint = {
            'timestamp': datetime.now(),
            'channel': channel,
            'cost': auction_result.get('price_paid', 0),
            'clicked': user_response.get('clicked', False),
            'position': auction_result.get('position', 10)
        }
        
        user_journey = self.metrics['user_journeys'][self.current_user.canonical_user_id]
        user_journey['touchpoints'].append(touchpoint)
        user_journey['total_spend'] += touchpoint['cost']
    
    def _process_delayed_conversions(self):
        """Process any delayed conversions that are due"""
        due_conversions = self.conversion_system.get_due_conversions(
            current_time=datetime.now()
        )
        
        for conversion in due_conversions:
            self.metrics['total_conversions'] += 1
            self.metrics['total_revenue'] += conversion['value']
            
            # Attribute credit across touchpoints
            user_id = conversion['user_id']
            if user_id in self.metrics['user_journeys']:
                touchpoints = self.metrics['user_journeys'][user_id]['touchpoints']
                
                # Calculate attribution
                credits = self.attribution.calculate_attribution(
                    touchpoints=touchpoints,
                    conversion_value=conversion['value'],
                    model='time_decay'
                )
                
                # Update channel revenue with attribution
                for channel, credit in credits.items():
                    if channel in self.metrics['channel_performance']:
                        self.metrics['channel_performance'][channel]['revenue'] += credit
    
    def _update_user_journey(self, auction_result: Dict, action: Dict):
        """Update user journey state"""
        user_journey = self.metrics['user_journeys'][self.current_user.canonical_user_id]
        
        # Progress journey stage based on interactions
        current_stage = user_journey['journey_state']
        touchpoint_count = len(user_journey['touchpoints'])
        
        if current_stage == 'unaware' and touchpoint_count >= 1:
            user_journey['journey_state'] = 'aware'
        elif current_stage == 'aware' and touchpoint_count >= 3:
            user_journey['journey_state'] = 'considering'
        elif current_stage == 'considering' and touchpoint_count >= 5:
            user_journey['journey_state'] = 'intent'
        
        # Track recent creatives for fatigue
        if 'recent_creatives' not in user_journey:
            user_journey['recent_creatives'] = []
        
        # Handle both 'creative' and 'creative_id' keys
        creative = action.get('creative', action.get('creative_id', 0))
        user_journey['recent_creatives'].append(creative)
        
        # Keep only last 10 creatives
        user_journey['recent_creatives'] = user_journey['recent_creatives'][-10:]
    
    def _update_metrics(self, auction_result: Dict, user_response: Dict, action: Dict):
        """Update comprehensive metrics"""
        # Channel is already a string in the action dict
        channel = action['channel'] if isinstance(action['channel'], str) else \
                  ['organic', 'paid_search', 'social', 'display', 'email'][action['channel']]
        
        # Update channel spend
        if channel in self.metrics['channel_performance']:
            self.metrics['channel_performance'][channel]['spend'] += auction_result.get('price_paid', 0)
            
            # Update revenue if converted
            if user_response.get('converted'):
                self.metrics['channel_performance'][channel]['revenue'] += user_response.get('revenue', 0)
                self.metrics['creative_performance'][action['creative']]['conversions'] += 1
    
    def _calculate_reward(self,
                         state: EnrichedJourneyState,
                         action: Dict,
                         next_state: EnrichedJourneyState,
                         auction_result: Dict) -> float:
        """Calculate sophisticated multi-component reward"""
        reward = 0.0
        
        # Auction outcome
        if auction_result['won']:
            position = auction_result.get('position', 10)
            price = auction_result.get('price_paid', 0)
            
            # Position value
            position_reward = (11 - position) / 10.0
            
            # Cost efficiency
            bid_amount = action.get('bid', action.get('bid_amount', 0))
            efficiency = (bid_amount - price) / max(0.01, bid_amount)
            
            reward += position_reward * 2.0 + efficiency * 1.0
        else:
            reward -= 0.5
        
        # Journey progression
        if next_state.stage > state.stage:
            reward += (next_state.stage - state.stage) * 5.0
        
        # Creative diversity
        if hasattr(next_state, 'creative_diversity_score'):
            reward += next_state.creative_diversity_score * 2.0
        
        # Channel ROI
        # Handle both string and index formats for channel
        if isinstance(action.get('channel'), str):
            channel = action['channel']
        else:
            channel = ['organic', 'paid_search', 'social', 'display', 'email'][action['channel']]
        if channel in self.metrics['channel_performance']:
            chan_perf = self.metrics['channel_performance'][channel]
            if chan_perf['spend'] > 0:
                roas = chan_perf['revenue'] / chan_perf['spend']
                reward += min(roas, 5.0)  # Cap at 5x ROAS
        
        # Budget pacing
        if state.budget_spent_ratio < state.time_in_day_ratio:
            pacing_penalty = (state.time_in_day_ratio - state.budget_spent_ratio) * 2.0
            reward -= pacing_penalty
        
        # Fatigue penalty
        if next_state.creative_fatigue > 0.7:
            reward -= (next_state.creative_fatigue - 0.7) * 5.0
        
        return reward
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Budget: ${self.budget_spent:.2f} / ${self.max_budget:.2f}")
            print(f"Impressions: {self.metrics['total_impressions']}")
            print(f"Clicks: {self.metrics['total_clicks']} "
                  f"(CTR: {self.metrics['total_clicks']/(self.metrics['total_impressions']+1)*100:.2f}%)")
            print(f"Conversions: {self.metrics['total_conversions']}")
            print(f"Revenue: ${self.metrics['total_revenue']:.2f}")
            print(f"ROAS: {self.metrics['total_revenue']/(self.budget_spent+0.01):.2f}x")
    
    def close(self):
        """Clean up environment"""
        # Flush batch writes
        if hasattr(self.user_db, 'flush_batches'):
            self.user_db.flush_batches()
        
        # Shutdown batch writer
        if hasattr(self.user_db, 'shutdown'):
            self.user_db.shutdown()
        
        logger.info("Environment closed")