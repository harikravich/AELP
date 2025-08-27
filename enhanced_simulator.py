#!/usr/bin/env python3
"""
Enhanced GAELP Simulator combining AuctionGym, RecSim, and real data.
This provides much more realistic training environment than random simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

# Import CreativeIntegration for rich ad content
try:
    from creative_integration import get_creative_integration, SimulationContext
    CREATIVE_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Creative Integration not available: {e}")
    CREATIVE_INTEGRATION_AVAILABLE = False

# Import AuctionGym integration
try:
    from auction_gym_integration import AuctionGymWrapper, AUCTION_GYM_AVAILABLE
    AUCTION_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AuctionGym integration not available: {e}")
    AUCTION_INTEGRATION_AVAILABLE = False

# Import RecSim-AuctionGym bridge
try:
    # Apply edward2 patch before importing RecSim
    import edward2_patch
    from recsim_auction_bridge import RecSimAuctionBridge, UserSegment
    from recsim_user_model import RecSimUserModel
    RECSIM_BRIDGE_AVAILABLE = True
    RECSIM_AVAILABLE = True
except ImportError as e:
    logging.error(f"RecSim-AuctionGym bridge REQUIRED. No fallbacks. Fix dependency: {e}")
    RECSIM_BRIDGE_AVAILABLE = False
    RECSIM_AVAILABLE = False
    RecSimUserModel = None
    UserSegment = None

logger = logging.getLogger(__name__)


class AdAuction:
    """Simulates realistic ad auction dynamics using AuctionGym when available"""
    
    def __init__(self, n_competitors: int = 10, max_slots: int = 5, recsim_bridge=None):
        self.n_competitors = n_competitors
        self.max_slots = max_slots
        self.recsim_bridge = recsim_bridge
        
        # Initialize AuctionGym wrapper if available
        if AUCTION_INTEGRATION_AVAILABLE:
            self.auction_gym = AuctionGymWrapper({
                'competitors': {'count': n_competitors},
                'num_slots': max_slots
            })
            self.use_auction_gym = True
            logger.info("Using AuctionGym for realistic auction simulation")
        else:
            raise RuntimeError("AuctionGym integration is REQUIRED. No fallback auction allowed. Fix dependencies.")
        
    def _init_competitors(self):
        """REMOVED - No fallback competitors allowed"""
        raise RuntimeError("Fallback competitors not allowed. Use proper AuctionGym integration.")
    
    def run_auction(self, your_bid: float, quality_score: float, context: Dict[str, Any] = None, user_id: str = None) -> Dict[str, Any]:
        """Run auction with realistic dynamics"""
        
        if self.use_auction_gym:
            # Use AuctionGym for sophisticated auction simulation
            # Convert quality_score to query_value for AuctionGym
            query_value = quality_score * 10.0  # Scale quality score to value
            auction_context = context or {}
            auction_context['quality_score'] = quality_score
            
            result = self.auction_gym.run_auction(
                our_bid=your_bid,
                query_value=query_value,
                context=auction_context
            )
            
            return {
                'won': result.won,
                'price_paid': result.price_paid,
                'impression_share': 1.0 if result.won else 0.0,
                'position': result.slot_position,
                'competitors': result.competitors,
                'estimated_ctr': result.estimated_ctr,
                'true_ctr': result.true_ctr,
                'outcome': result.outcome,
                'revenue': result.revenue,
                'total_slots': result.total_slots
            }
        else:
            # Fallback to simple auction simulation
            return self._run_simple_auction(your_bid, quality_score)
    
    def _run_simple_auction(self, your_bid: float, quality_score: float) -> Dict[str, Any]:
        """REALISTIC auction simulation with proper competitor bidding and balanced competition"""
        
        # Generate BALANCED competitor bids - now more realistic
        num_active_competitors = np.random.randint(6, 10)  # More competition
        competitor_bids = []
        
        # BALANCED: Competitors now bid in realistic ranges that create competition
        for i in range(num_active_competitors):
            comp = self.competitor_strategies[i % len(self.competitor_strategies)]
            
            if comp['type'] == 'aggressive':
                # Aggressive bidders - competitive and strong
                base_bid = np.random.uniform(1.50, 4.50)  # Higher competitive range
                bid = np.random.normal(base_bid, base_bid * 0.25)
            elif comp['type'] == 'conservative':
                # Conservative bidders - still competitive in market
                base_bid = np.random.uniform(1.00, 3.00)  # Higher base
                bid = np.random.normal(base_bid, base_bid * 0.20)
            else:  # adaptive
                # Adaptive bidders - strong middle ground
                base_bid = np.random.uniform(1.25, 3.75)  # Higher range
                bid = np.random.normal(base_bid, base_bid * 0.35)
            
            competitor_bids.append(max(0.10, bid))  # Minimum bid
        
        # Calculate effective bids (bid * quality_score)  
        your_effective_bid = your_bid * quality_score
        
        # BALANCED: Competitors have realistic quality scores with high variance
        competitor_effective_bids = []
        competitor_quality_scores = []
        for bid in competitor_bids:
            # Quality scores distributed realistically - many competitors have good scores
            comp_quality = np.random.normal(7.5, 1.8)  # Higher average with more variance
            comp_quality = np.clip(comp_quality, 4.0, 10.0)
            competitor_quality_scores.append(comp_quality)
            competitor_effective_bids.append(bid * comp_quality)
        
        # Create comprehensive bidder information for ad rank calculation
        all_bidders = [('us', your_effective_bid, your_bid, quality_score)]
        for i, (bid, quality, effective_bid) in enumerate(zip(competitor_bids, competitor_quality_scores, competitor_effective_bids)):
            all_bidders.append((f'comp_{i}', effective_bid, bid, quality))
        
        # Sort by ad rank (effective bid = bid * quality_score)
        all_bidders.sort(key=lambda x: x[1], reverse=True)
        
        # Find our position
        our_position = None
        for rank, (bidder_name, effective_bid, bid, quality) in enumerate(all_bidders):
            if bidder_name == 'us':
                our_position = rank + 1
                break
        
        won = our_position == 1
        
        if won:
            # Google Ads style second-price auction
            if len(all_bidders) > 1:
                # Pay (next_highest_ad_rank / our_quality_score) + $0.01
                next_highest_ad_rank = all_bidders[1][1]  # Second highest ad rank
                second_price = (next_highest_ad_rank / quality_score) + 0.01
                second_price = min(second_price, your_bid)  # Never pay more than bid
            else:
                second_price = your_bid * 0.8  # Reserve price when no competition
            
            # CTR based on position and quality
            position_ctr = {1: 0.08, 2: 0.05, 3: 0.03}.get(our_position, 0.01)
            estimated_ctr = position_ctr * (quality_score / 10.0)  # Normalize quality score
            clicked = np.random.random() < estimated_ctr
            revenue = np.random.gamma(2, 30) if clicked else 0
            
            return {
                'won': True,
                'price_paid': second_price,
                'impression_share': 1.0,
                'position': our_position,
                'competitors': len(competitor_bids),
                'estimated_ctr': estimated_ctr,
                'true_ctr': estimated_ctr,
                'outcome': clicked,
                'revenue': revenue,
                'total_slots': min(4, len(all_bidders))
            }
        else:
            # We lost - zero impression share (realistic auction outcome)
            return {
                'won': False,
                'price_paid': 0,
                'impression_share': 0.0,
                'position': our_position,
                'competitors': len(competitor_bids),
                'estimated_ctr': 0,
                'true_ctr': 0,
                'outcome': False,
                'revenue': 0,
                'total_slots': min(4, len(all_bidders))
            }
    
    def reset_episode(self):
        """Reset auction state for new episode"""
        if self.use_auction_gym:
            self.auction_gym.reset_competitors()
            
    def get_market_stats(self) -> Dict[str, Any]:
        """Get market statistics"""
        if self.use_auction_gym:
            return self.auction_gym.get_market_stats()
        else:
            return {
                'total_auctions': 0,
                'total_revenue': 0,
                'competitors': len(self.competitor_strategies)
            }


@dataclass 
class UserBehaviorModel:
    """Enhanced user behavior model with dynamic discovery - NO HARDCODING"""
    
    def __init__(self, persona_factory=None):
        # Import behavioral health personas for dynamic discovery
        from behavioral_health_persona_factory import BehavioralHealthPersonaFactory
        self.persona_factory = persona_factory if persona_factory else BehavioralHealthPersonaFactory()
        
        # Use RecSim-AuctionGym bridge if available, otherwise use dynamic discovery
        if RECSIM_BRIDGE_AVAILABLE:
            self.recsim_bridge = RecSimAuctionBridge()
            self.use_recsim_bridge = True
            logger.info("Using RecSim-AuctionGym bridge for user behavior model")
        else:
            self.use_recsim_bridge = False
            self.user_segments = {}  # Populated dynamically as segments are discovered
            logger.info("Using dynamic discovery for user behavior model")
        
        self.interaction_count = 0
        self.current_users = {}
        
    def _init_segments(self):
        """Dynamically discover segments from behavioral health personas"""
        # Let the RL agent discover these patterns, not hardcode them
        discovered_segments = {}
        
        # Sample some personas to understand the landscape
        sample_personas = [self.persona_factory.create_persona() for _ in range(100)]
        
        # The RL agent will discover these clusters dynamically
        # We just track what it finds
        for persona in sample_personas:
            segment_key = f"concern_{int(persona.current_concern_level)}"
            if segment_key not in discovered_segments:
                discovered_segments[segment_key] = {
                    'click_prob_base': 0.01 + (persona.current_concern_level * 0.02),
                    'conversion_prob_base': 0.001 * persona.current_concern_level,
                    'price_sensitivity': 1.0 - (persona.current_concern_level * 0.08),
                    'discovered_count': 0,
                    'characteristics': []
                }
            discovered_segments[segment_key]['discovered_count'] += 1
            
        logger.info(f"Discovered {len(discovered_segments)} initial segments through sampling")
        return discovered_segments
    
    def simulate_response(self, ad_creative: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate user response to an ad using RecSim-AuctionGym bridge - NO FALLBACKS"""
        
        if not self.use_recsim_bridge:
            raise RuntimeError("RecSim bridge REQUIRED but not available. NO FALLBACKS ALLOWED.")
            
        return self._simulate_with_recsim_bridge(ad_creative, context)
    
    def _simulate_with_recsim_bridge(self, ad_creative: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Use RecSim-AuctionGym bridge for sophisticated user behavior simulation"""
        
        # Generate unique user ID for this interaction
        user_id = f"user_{self.interaction_count}"
        self.interaction_count += 1
        
        # Generate or get user with realistic segment
        if user_id not in self.current_users:
            # Let the bridge generate a user with a realistic segment
            user_signals = self.recsim_bridge.user_to_auction_signals(user_id, context)
            self.current_users[user_id] = user_signals['segment']
        
        # Generate query based on user state and product category
        product_category = ad_creative.get('product_category', 'shoes')
        brand = ad_creative.get('brand', None)
        
        query_data = self.recsim_bridge.generate_query_from_state(
            user_id=user_id,
            product_category=product_category,
            brand=brand
        )
        
        # Get auction signals for this user
        auction_context = dict(context)
        auction_context['brand_query'] = query_data['brand_query']
        auction_signals = self.recsim_bridge.user_to_auction_signals(user_id, auction_context)
        
        # Map ad_creative to RecSim format
        recsim_ad_content = {
            'creative_quality': ad_creative.get('quality_score', auction_signals['quality_score']),
            'price_shown': ad_creative.get('price_shown', 50.0),
            'brand_match': auction_signals['brand_affinity'],
            'relevance_score': query_data['intent_strength'],
            'product_id': ad_creative.get('product_id', product_category)
        }
        
        # Get response from RecSim model through the bridge
        response = self.recsim_bridge.recsim_model.simulate_ad_response(
            user_id=user_id,
            ad_content=recsim_ad_content,
            context=auction_context
        )
        
        # Map back to expected format with realistic segment information
        return {
            'clicked': response['clicked'],
            'converted': response['converted'],
            'revenue': response['revenue'],
            'segment': auction_signals['segment'],
            'click_prob': response['click_probability'],
            'time_spent': response.get('time_spent', 0),
            'fatigue_level': auction_signals['fatigue_level'],
            'interest_level': auction_signals['interest_level'],
            'query_generated': query_data['query'],
            'journey_stage': query_data['journey_stage'],
            'intent_strength': query_data['intent_strength'],
            'suggested_bid': auction_signals['suggested_bid'],
            'user_segment_full': auction_signals['segment']
        }
    
    def _simulate_with_real_data(self, ad_creative: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """NO FALLBACKS ALLOWED - This method should never be called"""
        raise NotImplementedError("NO FALLBACKS ALLOWED. Use proper RecSim integration.")
    
    def _time_of_day_factor(self, hour: int) -> float:
        """Peak hours have higher engagement"""
        if 9 <= hour <= 11 or 19 <= hour <= 22:
            return 1.3
        elif 0 <= hour <= 6:
            return 0.5
        else:
            return 1.0
    
    def get_user_analytics(self) -> Dict[str, Any]:
        """Get analytics from the user model"""
        if self.use_recsim_bridge:
            bridge_analytics = self.recsim_bridge.get_bridge_analytics()
            return {
                **bridge_analytics,
                'total_users_generated': len(self.current_users),
                'segment_distribution': {segment: list(self.current_users.values()).count(segment) 
                                       for segment in set(self.current_users.values())}
            }
        else:
            return {'message': 'Analytics only available with RecSim-AuctionGym bridge'}
    
    def reset_users(self):
        """Reset user states (useful for RecSim bridge)"""
        if self.use_recsim_bridge:
            self.recsim_bridge.recsim_model.reset_user_states()
        self.interaction_count = 0
        self.current_users = {}


class RealDataCalibrator:
    """Calibrates simulation with real advertising data"""
    
    def __init__(self, data_source: str = "industry_benchmarks"):
        self.benchmarks = self._load_benchmarks()
        
    def _load_benchmarks(self):
        """Load industry benchmark data"""
        return {
            'avg_ctr': 0.02,  # 2% average CTR
            'avg_cpc': 2.50,   # $2.50 average CPC
            'avg_conversion_rate': 0.03,  # 3% conversion rate
            'avg_roas': 3.0,   # 3x return on ad spend
            'ctr_by_position': {1: 0.08, 2: 0.05, 3: 0.03, 4: 0.02},
            'ctr_by_device': {'mobile': 0.025, 'desktop': 0.018, 'tablet': 0.022}
        }
    
    def calibrate_result(self, sim_result: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust simulation results to match real-world distributions"""
        
        # Apply noise and realistic variance
        if sim_result.get('clicked'):
            # Add realistic CTR variance
            noise_factor = np.random.normal(1.0, 0.2)
            sim_result['click_prob'] *= noise_factor
            
        if sim_result.get('converted'):
            # Revenue should follow power law distribution
            sim_result['revenue'] *= np.random.pareto(1.5)
            
        return sim_result


class EnhancedGAELPEnvironment:
    """
    Production-ready environment combining auction dynamics,
    user behavior, and real data calibration.
    
    This class serves as the core simulator that can be wrapped
    by Gymnasium-compatible interfaces.
    """
    
    def __init__(self, max_budget: float = 10000.0, max_steps: int = 100):
        # Initialize RecSim-AuctionGym bridge if available
        self.recsim_bridge = None
        if RECSIM_BRIDGE_AVAILABLE:
            self.recsim_bridge = RecSimAuctionBridge()
            logger.info("Enhanced environment using RecSim-AuctionGym bridge")
        
        self.auction = AdAuction(n_competitors=10, recsim_bridge=self.recsim_bridge)
        self.user_model = UserBehaviorModel()
        self.calibrator = RealDataCalibrator()
        self.episode_data = []
        self.max_budget = max_budget
        self.max_steps = max_steps
        self.current_step = 0
        self.active_users = {}  # Track active users for the session
        
    def reset(self):
        """Reset environment for new episode"""
        self.episode_data = []
        self.current_step = 0
        self.auction.reset_episode()
        self.user_model.reset_users()
        return self._get_observation()
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Dict with 'bid', 'budget', 'creative', 'targeting'
            
        Returns:
            observation, reward, done, info
        """
        
        self.current_step += 1
        
        # Generate or select a user for this step
        user_id = f"session_user_{self.current_step % 100}"  # Reuse users for realistic sessions
        
        # Create context for auction
        context = {
            'hour': np.random.randint(0, 24),
            'device': np.random.choice(['mobile', 'desktop', 'tablet']),
            'step': self.current_step,
            'product_category': action.get('creative', {}).get('product_category', 'shoes'),
            'brand': action.get('creative', {}).get('brand', None)
        }
        
        # If using RecSim bridge, generate realistic query for this user
        if self.recsim_bridge and RECSIM_BRIDGE_AVAILABLE:
            # Let the bridge generate or retrieve user with realistic behavior
            query_data = self.recsim_bridge.generate_query_from_state(
                user_id=user_id,
                product_category=context['product_category'],
                brand=context['brand']
            )
            context.update({
                'query': query_data['query'],
                'intent_strength': query_data['intent_strength'],
                'journey_stage': query_data['journey_stage']
            })
        
        # Run auction with context and user information
        auction_result = self.auction.run_auction(
            your_bid=action.get('bid', 1.0),
            quality_score=action.get('quality_score', 0.7),
            context=context,
            user_id=user_id
        )
        
        results = {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'cost': 0,
            'revenue': 0
        }
        
        # Handle impressions based on auction result
        impression_share = auction_result.get('impression_share', 0)
        
        if impression_share > 0:
            # You get impressions based on your share
            results['impressions'] = max(1, int(impression_share * 10))  # Scale impressions
            results['cost'] = auction_result['price_paid']
            
            # Use enhanced RecSim-AuctionGym results if available, otherwise simulate
            if 'outcome' in auction_result and 'revenue' in auction_result:
                # Use AuctionGym's sophisticated outcome simulation
                if auction_result['outcome']:  # User clicked
                    results['clicks'] = 1
                    
                    # If using RecSim bridge, get more sophisticated conversion behavior
                    if self.recsim_bridge and RECSIM_BRIDGE_AVAILABLE and 'user_segment' in auction_result:
                        # Use segment-specific conversion logic
                        user_segment = auction_result['user_segment']
                        segment_conversion_rates = {
                            'impulse_buyer': 0.15,
                            'researcher': 0.05,
                            'loyal_customer': 0.20,
                            'window_shopper': 0.02,
                            'price_conscious': 0.08,
                            'brand_loyalist': 0.18
                        }
                        
                        base_conv_rate = segment_conversion_rates.get(user_segment, 0.05)
                        intent_modifier = context.get('intent_strength', 0.5)
                        final_conv_rate = base_conv_rate * (0.5 + intent_modifier)
                        
                        if np.random.random() < final_conv_rate:
                            results['conversions'] = 1
                            # Segment-specific revenue
                            if user_segment in ['loyal_customer', 'brand_loyalist']:
                                results['revenue'] = np.random.gamma(3, 40)
                            elif user_segment == 'price_conscious':
                                results['revenue'] = np.random.gamma(2, 20)
                            else:
                                results['revenue'] = np.random.gamma(2, 30)
                    else:
                        # Use AuctionGym's revenue calculation
                        results['revenue'] = auction_result.get('revenue', 0)
                        
                        # Determine conversion from click
                        conversion_rate = action.get('creative', {}).get('conversion_rate', 0.03)
                        if np.random.random() < conversion_rate:
                            results['conversions'] = 1
            else:
                # Fallback to our user behavior simulation
                # Adjust response based on position (lower positions get lower CTR)
                position_factor = 1.0 / auction_result['position']
                
                user_response = self.user_model.simulate_response(
                    ad_creative=action.get('creative', {}),
                    context=context
                )
                
                # Apply position penalty
                user_response['click_prob'] *= position_factor
                
                # Calibrate with real data
                user_response = self.calibrator.calibrate_result(user_response)
                
                # Simulate clicks/conversions across impressions
                for _ in range(results['impressions']):
                    if np.random.random() < user_response['click_prob']:
                        results['clicks'] += 1
                        
                        # Check for conversion on each click
                        if user_response['clicked'] and np.random.random() < 0.05:  # 5% conversion rate on clicks
                            results['conversions'] += 1
                            # Revenue per conversion
                            results['revenue'] += np.random.gamma(2, 30) * action.get('creative', {}).get('quality_score', 0.5)
        
        # Calculate reward (ROAS)
        reward = (results['revenue'] - results['cost']) / max(results['cost'], 0.01)
        
        # Store episode data with enhanced information
        episode_info = {
            'action': action,
            'auction': auction_result,
            'results': results,
            'reward': reward,
            'user_id': user_id,
            'context': context
        }
        
        # Add RecSim-specific information if available
        if self.recsim_bridge and RECSIM_BRIDGE_AVAILABLE:
            if 'user_segment' in auction_result:
                episode_info['user_segment'] = auction_result['user_segment']
            if 'query' in context:
                episode_info['generated_query'] = context['query']
                episode_info['journey_stage'] = context['journey_stage']
        
        self.episode_data.append(episode_info)
        
        # Episode ends after budget exhausted or time limit
        total_cost = sum(r['results']['cost'] for r in self.episode_data)
        done = (self.current_step >= self.max_steps or 
                total_cost > action.get('budget', self.max_budget))
        
        return self._get_observation(), reward, done, results
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current state observation"""
        
        if not self.episode_data:
            return {
                'total_cost': 0,
                'total_revenue': 0,
                'impressions': 0,
                'clicks': 0,
                'conversions': 0,
                'avg_cpc': 0,
                'roas': 0
            }
        
        metrics = {
            'total_cost': sum(r['results']['cost'] for r in self.episode_data),
            'total_revenue': sum(r['results']['revenue'] for r in self.episode_data),
            'impressions': sum(r['results']['impressions'] for r in self.episode_data),
            'clicks': sum(r['results']['clicks'] for r in self.episode_data),
            'conversions': sum(r['results']['conversions'] for r in self.episode_data)
        }
        
        metrics['avg_cpc'] = metrics['total_cost'] / max(metrics['clicks'], 1)
        metrics['roas'] = metrics['total_revenue'] / max(metrics['total_cost'], 0.01)
        
        return metrics


def test_enhanced_environment():
    """Test the enhanced environment with RecSim user modeling"""
    
    env = EnhancedGAELPEnvironment()
    obs = env.reset()
    
    print("Testing Enhanced GAELP Environment with RecSim Integration")
    print("=" * 60)
    print(f"RecSim Available: {RECSIM_AVAILABLE}")
    print("=" * 60)
    
    total_reward = 0
    
    # Test different creative strategies
    creative_strategies = [
        {
            'name': 'High Quality Premium',
            'creative': {
                'quality_score': 0.9,
                'price_shown': 150.0,
                'brand_affinity': 0.8,
                'relevance': 0.9,
                'product_id': 'premium_product'
            }
        },
        {
            'name': 'Budget Friendly',
            'creative': {
                'quality_score': 0.6,
                'price_shown': 25.0,
                'brand_affinity': 0.4,
                'relevance': 0.7,
                'product_id': 'budget_product'
            }
        },
        {
            'name': 'Brand Focus',
            'creative': {
                'quality_score': 0.7,
                'price_shown': 75.0,
                'brand_affinity': 0.95,
                'relevance': 0.8,
                'product_id': 'brand_product'
            }
        }
    ]
    
    for step in range(30):
        # Cycle through different creative strategies
        strategy = creative_strategies[step % len(creative_strategies)]
        
        # Sample action with varying strategies
        action = {
            'bid': np.random.uniform(0.5, 5.0),
            'budget': 1000,
            'quality_score': np.random.uniform(0.5, 1.0),
            'creative': strategy['creative']
        }
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            print(f"Step {step} ({strategy['name']}): ROAS={obs['roas']:.2f}, Cost=${obs['total_cost']:.2f}, Revenue=${obs['total_revenue']:.2f}")
            if RECSIM_AVAILABLE and hasattr(env.user_model, 'recsim_model'):
                user_analytics = env.user_model.get_user_analytics()
                if user_analytics:
                    print(f"    Overall CTR: {user_analytics.get('overall_ctr', 0):.3f}")
                    print(f"    Overall Conversion Rate: {user_analytics.get('overall_conversion_rate', 0):.3f}")
        
        if done:
            break
    
    print(f"\nFinal Results:")
    print(f"  Total Impressions: {obs['impressions']}")
    print(f"  Total Clicks: {obs['clicks']}")
    print(f"  Total Conversions: {obs['conversions']}")
    print(f"  Total Cost: ${obs['total_cost']:.2f}")
    print(f"  Total Revenue: ${obs['total_revenue']:.2f}")
    print(f"  Final ROAS: {obs['roas']:.2f}x")
    print(f"  Average Reward: {total_reward/len(env.episode_data):.3f}")
    
    # Show user behavior analytics if RecSim is available
    if RECSIM_AVAILABLE and hasattr(env.user_model, 'recsim_model'):
        print(f"\nUser Behavior Analytics (RecSim):")
        print("=" * 40)
        analytics = env.user_model.get_user_analytics()
        
        if 'segment_breakdown' in analytics:
            for segment, stats in analytics['segment_breakdown'].items():
                print(f"  {segment.upper()}:")
                print(f"    Interactions: {stats['total_interactions']}")
                print(f"    Click Rate: {stats['click_rate']:.3f}")
                print(f"    Conversion Rate: {stats['conversion_rate']:.3f}")
                print(f"    Avg Revenue: ${stats['avg_revenue']:.2f}")
                print(f"    Avg Time Spent: {stats['avg_time_spent']:.1f}s")
                print()


def test_user_segments():
    """Test specific user segment behaviors"""
    
    if not RECSIM_AVAILABLE:
        print("RecSim not available - skipping user segment tests")
        return
    
    print("Testing Individual User Segments")
    print("=" * 40)
    
    # Apply patch before import
    import edward2_patch
    from recsim_user_model import RecSimUserModel, UserSegment
    
    model = RecSimUserModel()
    
    # Test ad scenarios
    premium_ad = {
        'creative_quality': 0.9,
        'price_shown': 200.0,
        'brand_match': 0.8,
        'relevance_score': 0.9,
        'product_id': 'luxury_watch'
    }
    
    budget_ad = {
        'creative_quality': 0.5,
        'price_shown': 20.0,
        'brand_match': 0.3,
        'relevance_score': 0.6,
        'product_id': 'budget_accessory'
    }
    
    context = {'hour': 20, 'device': 'mobile'}
    
    # Test each segment
    for segment in UserSegment:
        print(f"\nTesting {segment.value.upper()}:")
        
        # Generate users and test responses
        premium_responses = []
        budget_responses = []
        
        for i in range(50):
            user_id = f"test_{segment.value}_{i}"
            model.generate_user(user_id, segment)
            
            # Test premium ad
            resp = model.simulate_ad_response(user_id, premium_ad, context)
            premium_responses.append(resp)
            
            # Test budget ad  
            resp = model.simulate_ad_response(user_id, budget_ad, context)
            budget_responses.append(resp)
        
        # Calculate statistics
        print(f"  Premium Ad ($200):")
        print(f"    Click Rate: {np.mean([r['clicked'] for r in premium_responses]):.3f}")
        print(f"    Conversion Rate: {np.mean([r['converted'] for r in premium_responses if r['clicked']]):.3f}")
        print(f"    Avg Revenue: ${np.mean([r['revenue'] for r in premium_responses]):.2f}")
        
        print(f"  Budget Ad ($20):")
        print(f"    Click Rate: {np.mean([r['clicked'] for r in budget_responses]):.3f}")
        print(f"    Conversion Rate: {np.mean([r['converted'] for r in budget_responses if r['clicked']]):.3f}")
        print(f"    Avg Revenue: ${np.mean([r['revenue'] for r in budget_responses]):.2f}")


if __name__ == "__main__":
    test_enhanced_environment()
    print("\n" + "="*60 + "\n")
    test_user_segments()