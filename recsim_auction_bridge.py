#!/usr/bin/env python3
"""
RecSim-AuctionGym Bridge for GAELP
Connects RecSim user segments to AuctionGym bidding system to create
realistic user-driven auction participation and query generation.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import json

# Import RecSim components - NO FALLBACKS ALLOWED
from recsim_user_model import RecSimUserModel, UserSegment, UserProfile
from NO_FALLBACKS import StrictModeEnforcer

# Verify RecSim user model is working
if not hasattr(RecSimUserModel, 'simulate_ad_response'):
    StrictModeEnforcer.enforce('RECSIM_USER_MODEL', fallback_attempted=True)
    raise ImportError("RecSim user model MUST be properly implemented. NO FALLBACKS!")

# Import AuctionGym components - NO FALLBACKS ALLOWED
try:
    from auction_gym_integration import AuctionGymWrapper, AuctionResult
except ImportError:
    StrictModeEnforcer.enforce('AUCTION_GYM', fallback_attempted=True)
    raise ImportError("AuctionGym integration MUST be available. NO FALLBACKS!")

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Represents search query intent based on user journey stage"""
    stage: str  # 'awareness', 'consideration', 'purchase', 'loyalty'
    intent_strength: float  # 0-1, how likely to convert
    query_type: str  # 'informational', 'navigational', 'transactional'
    keywords: List[str]
    bid_modifier: float  # Multiplier for base bid amount


class UserJourneyStage(Enum):
    """User journey stages for mapping to query intent"""
    AWARENESS = "awareness"
    CONSIDERATION = "consideration" 
    PURCHASE = "purchase"
    LOYALTY = "loyalty"
    RE_ENGAGEMENT = "re_engagement"


class RecSimAuctionBridge:
    """
    Bridge between RecSim user model and AuctionGym bidding system.
    Maps user segments to auction participation patterns and generates
    realistic search queries based on user state.
    """
    
    def __init__(self, 
                 recsim_model: RecSimUserModel,
                 auction_wrapper: AuctionGymWrapper):
        
        # NO fallbacks - components MUST be provided
        if recsim_model is None:
            StrictModeEnforcer.enforce('RECSIM_MODEL', fallback_attempted=True)
            raise ValueError("RecSim model MUST be provided. NO fallbacks!")
        
        if auction_wrapper is None:
            StrictModeEnforcer.enforce('AUCTION_WRAPPER', fallback_attempted=True)
            raise ValueError("Auction wrapper MUST be provided. NO fallbacks!")
        
        self.recsim_model = recsim_model
        self.auction_wrapper = auction_wrapper
        
        # Initialize segment-to-bidding mappings
        self.segment_bid_profiles = self._init_segment_bid_profiles()
        self.journey_stage_mapping = self._init_journey_stage_mapping()
        self.query_templates = self._init_query_templates()
        
        # Tracking
        self.user_auction_history = {}
        self.query_generation_stats = {}
        
    def _init_segment_bid_profiles(self) -> Dict[UserSegment, Dict[str, Any]]:
        """Initialize bidding profiles for each user segment"""
        
        return {
            UserSegment.IMPULSE_BUYER: {
                'base_bid_range': (0.80, 2.50),
                'bid_volatility': 0.6,  # High volatility in bidding
                'auction_participation_rate': 0.85,  # Very likely to trigger auctions
                'quality_score_range': (0.6, 0.8),
                'preferred_slots': [1, 2, 3],  # Top positions
                'time_sensitivity': 0.9,  # Quick decisions
                'budget_depletion_rate': 0.15  # Burns through budget quickly
            },
            
            UserSegment.RESEARCHER: {
                'base_bid_range': (0.30, 1.20),
                'bid_volatility': 0.2,  # Very consistent bidding
                'auction_participation_rate': 0.95,  # Almost always participates
                'quality_score_range': (0.7, 0.9),
                'preferred_slots': [1, 2, 3, 4, 5],  # Any position acceptable
                'time_sensitivity': 0.3,  # Takes time to decide
                'budget_depletion_rate': 0.05  # Conservative spending
            },
            
            UserSegment.LOYAL_CUSTOMER: {
                'base_bid_range': (1.00, 3.00),
                'bid_volatility': 0.3,  # Moderate volatility
                'auction_participation_rate': 0.70,  # Selective participation
                'quality_score_range': (0.8, 0.95),
                'preferred_slots': [1, 2],  # Premium positions
                'time_sensitivity': 0.6,  # Medium decision speed
                'budget_depletion_rate': 0.08  # Steady spending
            },
            
            UserSegment.WINDOW_SHOPPER: {
                'base_bid_range': (0.10, 0.80),
                'bid_volatility': 0.4,  # Somewhat volatile
                'auction_participation_rate': 0.40,  # Low participation
                'quality_score_range': (0.4, 0.6),
                'preferred_slots': [3, 4, 5],  # Lower positions acceptable
                'time_sensitivity': 0.2,  # Very slow decisions
                'budget_depletion_rate': 0.03  # Very conservative
            },
            
            UserSegment.PRICE_CONSCIOUS: {
                'base_bid_range': (0.05, 0.60),
                'bid_volatility': 0.1,  # Very low volatility
                'auction_participation_rate': 0.30,  # Low participation
                'quality_score_range': (0.5, 0.7),
                'preferred_slots': [4, 5],  # Only low-cost positions
                'time_sensitivity': 0.1,  # Extremely slow decisions
                'budget_depletion_rate': 0.02  # Minimal spending
            },
            
            UserSegment.BRAND_LOYALIST: {
                'base_bid_range': (1.50, 4.00),
                'bid_volatility': 0.2,  # Low volatility
                'auction_participation_rate': 0.60,  # Medium participation
                'quality_score_range': (0.85, 0.98),
                'preferred_slots': [1, 2],  # Premium positions only
                'time_sensitivity': 0.8,  # Quick decisions for preferred brands
                'budget_depletion_rate': 0.12  # Willing to spend for quality
            }
        }
    
    def _init_journey_stage_mapping(self) -> Dict[UserSegment, Dict[UserJourneyStage, float]]:
        """Map user segments to journey stage probabilities"""
        
        return {
            UserSegment.IMPULSE_BUYER: {
                UserJourneyStage.AWARENESS: 0.20,
                UserJourneyStage.CONSIDERATION: 0.15,
                UserJourneyStage.PURCHASE: 0.55,
                UserJourneyStage.LOYALTY: 0.05,
                UserJourneyStage.RE_ENGAGEMENT: 0.05
            },
            
            UserSegment.RESEARCHER: {
                UserJourneyStage.AWARENESS: 0.35,
                UserJourneyStage.CONSIDERATION: 0.45,
                UserJourneyStage.PURCHASE: 0.15,
                UserJourneyStage.LOYALTY: 0.03,
                UserJourneyStage.RE_ENGAGEMENT: 0.02
            },
            
            UserSegment.LOYAL_CUSTOMER: {
                UserJourneyStage.AWARENESS: 0.05,
                UserJourneyStage.CONSIDERATION: 0.10,
                UserJourneyStage.PURCHASE: 0.40,
                UserJourneyStage.LOYALTY: 0.35,
                UserJourneyStage.RE_ENGAGEMENT: 0.10
            },
            
            UserSegment.WINDOW_SHOPPER: {
                UserJourneyStage.AWARENESS: 0.40,
                UserJourneyStage.CONSIDERATION: 0.50,
                UserJourneyStage.PURCHASE: 0.05,
                UserJourneyStage.LOYALTY: 0.02,
                UserJourneyStage.RE_ENGAGEMENT: 0.03
            },
            
            UserSegment.PRICE_CONSCIOUS: {
                UserJourneyStage.AWARENESS: 0.25,
                UserJourneyStage.CONSIDERATION: 0.55,
                UserJourneyStage.PURCHASE: 0.15,
                UserJourneyStage.LOYALTY: 0.03,
                UserJourneyStage.RE_ENGAGEMENT: 0.02
            },
            
            UserSegment.BRAND_LOYALIST: {
                UserJourneyStage.AWARENESS: 0.05,
                UserJourneyStage.CONSIDERATION: 0.15,
                UserJourneyStage.PURCHASE: 0.35,
                UserJourneyStage.LOYALTY: 0.40,
                UserJourneyStage.RE_ENGAGEMENT: 0.05
            }
        }
    
    def _init_query_templates(self) -> Dict[UserJourneyStage, Dict[str, Any]]:
        """Initialize query templates for each journey stage"""
        
        return {
            UserJourneyStage.AWARENESS: {
                'query_type': 'informational',
                'intent_strength': 0.1,
                'bid_modifier': 0.7,
                'templates': [
                    "what is {product_category}",
                    "how to choose {product_category}", 
                    "best {product_category} 2024",
                    "{product_category} guide",
                    "types of {product_category}",
                    "{product_category} reviews"
                ]
            },
            
            UserJourneyStage.CONSIDERATION: {
                'query_type': 'informational',
                'intent_strength': 0.4,
                'bid_modifier': 1.0,
                'templates': [
                    "{product_category} comparison",
                    "best {product_category} for {use_case}",
                    "{product_category} vs {competitor}",
                    "cheap {product_category}",
                    "{product_category} price comparison",
                    "{product_category} features"
                ]
            },
            
            UserJourneyStage.PURCHASE: {
                'query_type': 'transactional', 
                'intent_strength': 0.8,
                'bid_modifier': 1.5,
                'templates': [
                    "buy {product_category}",
                    "{product_category} for sale",
                    "{product_category} discount",
                    "order {product_category}",
                    "{product_category} near me",
                    "{brand} {product_category} buy"
                ]
            },
            
            UserJourneyStage.LOYALTY: {
                'query_type': 'navigational',
                'intent_strength': 0.9,
                'bid_modifier': 1.2,
                'templates': [
                    "{brand} {product_category}",
                    "{brand} official store",
                    "{brand} customer service",
                    "{brand} warranty",
                    "{brand} support",
                    "{brand} new products"
                ]
            },
            
            UserJourneyStage.RE_ENGAGEMENT: {
                'query_type': 'transactional',
                'intent_strength': 0.6,
                'bid_modifier': 1.1,
                'templates': [
                    "{product_category} deals",
                    "{product_category} sale",
                    "discount {product_category}",
                    "{product_category} coupon",
                    "{brand} promo code",
                    "{product_category} clearance"
                ]
            }
        }
    
    def user_to_auction_signals(self, 
                               user_id: str, 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Convert user profile and state to auction bidding signals
        
        Args:
            user_id: User identifier
            context: Additional context (time, device, etc.)
            
        Returns:
            Dict with auction signals for bidding decisions
        """
        
        if context is None:
            context = {}
            
        # Get or create user
        if user_id not in self.recsim_model.current_users:
            user_profile = self.recsim_model.generate_user(user_id)
        else:
            user_profile = self.recsim_model.current_users[user_id]
        
        # Get bidding profile for this segment
        bid_profile = self.segment_bid_profiles[user_profile.segment]
        
        # Calculate base bid based on segment and user state
        base_bid_min, base_bid_max = bid_profile['base_bid_range']
        base_bid = np.random.uniform(base_bid_min, base_bid_max)
        
        # Apply user state modifiers
        interest_modifier = 0.5 + user_profile.current_interest * 0.5
        fatigue_modifier = max(0.3, 1.0 - user_profile.fatigue_level)
        
        # Budget constraint modifier
        budget_modifier = min(1.0, user_profile.budget / 100.0)
        
        # Time and device context modifiers
        time_modifier = self._get_time_modifier(user_profile, context.get('hour', 12))
        device_modifier = self._get_device_modifier(user_profile, context.get('device', 'desktop'))
        
        # Calculate final bid
        suggested_bid = (base_bid * 
                        interest_modifier * 
                        fatigue_modifier * 
                        budget_modifier * 
                        time_modifier * 
                        device_modifier)
        
        # Add volatility based on segment
        volatility = bid_profile['bid_volatility']
        noise = np.random.normal(0, volatility * 0.1)
        suggested_bid = max(0.01, suggested_bid * (1 + noise))
        
        # Quality score estimation
        quality_min, quality_max = bid_profile['quality_score_range']
        quality_score = np.random.uniform(quality_min, quality_max)
        
        # Brand affinity affects quality for brand-related queries
        if context.get('brand_query', False):
            quality_score *= (0.7 + user_profile.brand_affinity * 0.3)
        
        return {
            'suggested_bid': suggested_bid,
            'quality_score': quality_score,
            'participation_probability': bid_profile['auction_participation_rate'],
            'preferred_slots': bid_profile['preferred_slots'],
            'time_sensitivity': bid_profile['time_sensitivity'],
            'segment': user_profile.segment.value,
            'interest_level': user_profile.current_interest,
            'fatigue_level': user_profile.fatigue_level,
            'price_sensitivity': user_profile.price_sensitivity,
            'brand_affinity': user_profile.brand_affinity,
            'budget_remaining': user_profile.budget,
            'attention_span': user_profile.attention_span
        }
    
    def generate_query_from_state(self, 
                                 user_id: str,
                                 product_category: str = "shoes",
                                 brand: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate search query based on user state and journey stage
        
        Args:
            user_id: User identifier
            product_category: Product category for query generation
            brand: Optional brand name for branded queries
            
        Returns:
            Dict with generated query and metadata
        """
        
        # Get user profile
        if user_id not in self.recsim_model.current_users:
            user_profile = self.recsim_model.generate_user(user_id)
        else:
            user_profile = self.recsim_model.current_users[user_id]
        
        # Determine journey stage based on user segment and state
        journey_stage = self._determine_journey_stage(user_profile)
        
        # Get query template for this stage
        stage_config = self.query_templates[journey_stage]
        templates = stage_config['templates']
        
        # Select random template
        template = np.random.choice(templates)
        
        # Fill in template variables
        query = template.format(
            product_category=product_category,
            brand=brand or "nike",  # Default brand
            competitor=np.random.choice(["adidas", "puma", "reebok", "vans"]),
            use_case=np.random.choice(["running", "casual", "work", "sports", "hiking"])
        )
        
        # Generate query intent
        query_intent = QueryIntent(
            stage=journey_stage.value,
            intent_strength=stage_config['intent_strength'],
            query_type=stage_config['query_type'],
            keywords=query.split(),
            bid_modifier=stage_config['bid_modifier']
        )
        
        # Track query generation stats
        segment_name = user_profile.segment.value
        if segment_name not in self.query_generation_stats:
            self.query_generation_stats[segment_name] = {}
        
        stage_name = journey_stage.value
        if stage_name not in self.query_generation_stats[segment_name]:
            self.query_generation_stats[segment_name][stage_name] = 0
        
        self.query_generation_stats[segment_name][stage_name] += 1
        
        return {
            'query': query,
            'intent': query_intent,
            'user_segment': user_profile.segment.value,
            'journey_stage': journey_stage.value,
            'query_type': stage_config['query_type'],
            'intent_strength': stage_config['intent_strength'],
            'brand_query': brand is not None and brand.lower() in query.lower()
        }
    
    def map_segment_to_bid_value(self, 
                                segment: UserSegment,
                                query_intent: QueryIntent,
                                market_context: Dict[str, Any] = None) -> float:
        """
        Map user segment and query intent to appropriate bid value
        
        Args:
            segment: User segment
            query_intent: Generated query intent
            market_context: Current market conditions
            
        Returns:
            Recommended bid value
        """
        
        if market_context is None:
            market_context = {}
        
        # Get base bidding profile
        bid_profile = self.segment_bid_profiles[segment]
        base_bid_min, base_bid_max = bid_profile['base_bid_range']
        
        # Start with segment-appropriate bid range
        base_bid = np.random.uniform(base_bid_min, base_bid_max)
        
        # Apply intent strength modifier
        intent_modifier = 0.5 + (query_intent.intent_strength * 0.8)
        
        # Apply query-specific bid modifier
        query_modifier = query_intent.bid_modifier
        
        # Market competition modifier
        competition_level = market_context.get('competition_level', 0.5)
        competition_modifier = 1.0 + (competition_level * 0.3)
        
        # Time-based modifier
        hour = market_context.get('hour', 12)
        if segment == UserSegment.IMPULSE_BUYER and 19 <= hour <= 22:
            time_modifier = 1.3  # Prime impulse buying time
        elif segment == UserSegment.RESEARCHER and 9 <= hour <= 17:
            time_modifier = 1.2  # Business hours research
        else:
            time_modifier = 1.0
        
        # Calculate final bid
        final_bid = (base_bid * 
                    intent_modifier * 
                    query_modifier * 
                    competition_modifier * 
                    time_modifier)
        
        # Apply segment-specific constraints
        if segment == UserSegment.PRICE_CONSCIOUS:
            final_bid = min(final_bid, 0.75)  # Hard cap for price-conscious users
        elif segment == UserSegment.BRAND_LOYALIST and query_intent.intent_strength > 0.7:
            final_bid *= 1.4  # Premium for high-intent brand searches
        
        return max(0.01, final_bid)
    
    def simulate_user_auction_session(self, 
                                    user_id: str,
                                    num_queries: int = 5,
                                    product_category: str = "shoes") -> Dict[str, Any]:
        """
        Simulate a complete user session with multiple queries and auctions
        
        Args:
            user_id: User identifier
            num_queries: Number of queries to simulate
            product_category: Product category for session
            
        Returns:
            Dict with session results and analytics
        """
        
        session_results = {
            'user_id': user_id,
            'queries': [],
            'auctions': [],
            'total_cost': 0.0,
            'total_revenue': 0.0,
            'clicks': 0,
            'conversions': 0
        }
        
        # Get or create user
        if user_id not in self.recsim_model.current_users:
            user_profile = self.recsim_model.generate_user(user_id)
        else:
            user_profile = self.recsim_model.current_users[user_id]
        
        for query_num in range(num_queries):
            # Generate query for current user state
            query_data = self.generate_query_from_state(
                user_id=user_id,
                product_category=product_category
            )
            
            # Get auction signals
            context = {
                'hour': np.random.randint(0, 24),
                'device': np.random.choice(['mobile', 'desktop', 'tablet']),
                'brand_query': query_data['brand_query']
            }
            
            auction_signals = self.user_to_auction_signals(user_id, context)
            
            # Decide whether to participate in auction
            if np.random.random() < auction_signals['participation_probability']:
                # Calculate bid using query intent
                bid_value = self.map_segment_to_bid_value(
                    segment=user_profile.segment,
                    query_intent=query_data['intent'],
                    market_context=context
                )
                
                # Run auction
                auction_result = self.auction_wrapper.run_auction(
                    your_bid=bid_value,
                    your_quality_score=auction_signals['quality_score'],
                    context=context
                )
                
                # Simulate user response if auction was won
                if auction_result.won:
                    # Use RecSim model to simulate ad response
                    ad_content = {
                        'creative_quality': auction_signals['quality_score'],
                        'price_shown': bid_value * 20,  # Estimated product price
                        'brand_match': user_profile.brand_affinity,
                        'relevance_score': query_data['intent_strength'],
                        'product_id': product_category
                    }
                    
                    user_response = self.recsim_model.simulate_ad_response(
                        user_id=user_id,
                        ad_content=ad_content,
                        context=context
                    )
                    
                    # Update session stats
                    session_results['total_cost'] += auction_result.price_paid
                    if user_response['clicked']:
                        session_results['clicks'] += 1
                        if user_response['converted']:
                            session_results['conversions'] += 1
                            session_results['total_revenue'] += user_response['revenue']
                
                session_results['auctions'].append({
                    'query_num': query_num,
                    'bid_value': bid_value,
                    'auction_result': auction_result,
                    'user_response': user_response if auction_result.won else None
                })
            
            session_results['queries'].append(query_data)
            
            # Update user state (fatigue increases with each query)
            user_profile.fatigue_level = min(1.0, user_profile.fatigue_level + 0.1)
        
        # Calculate session metrics
        if session_results['total_cost'] > 0:
            session_results['roas'] = session_results['total_revenue'] / session_results['total_cost']
            session_results['cost_per_click'] = session_results['total_cost'] / max(session_results['clicks'], 1)
            session_results['cost_per_conversion'] = session_results['total_cost'] / max(session_results['conversions'], 1)
        
        # Store session history
        if user_id not in self.user_auction_history:
            self.user_auction_history[user_id] = []
        self.user_auction_history[user_id].append(session_results)
        
        return session_results
    
    def _determine_journey_stage(self, user_profile: UserProfile) -> UserJourneyStage:
        """Determine journey stage based on user profile and state"""
        
        # Get journey stage probabilities for this segment
        stage_probs = self.journey_stage_mapping[user_profile.segment]
        
        # Adjust probabilities based on user state
        adjusted_probs = {}
        for stage, prob in stage_probs.items():
            # Interest level affects consideration and purchase stages
            if stage in [UserJourneyStage.CONSIDERATION, UserJourneyStage.PURCHASE]:
                prob *= (0.5 + user_profile.current_interest)
            
            # Recent purchases affect loyalty and re-engagement
            if len(user_profile.recent_purchases) > 0:
                if stage == UserJourneyStage.LOYALTY:
                    prob *= 1.5
                elif stage == UserJourneyStage.RE_ENGAGEMENT:
                    prob *= 0.5
            
            adjusted_probs[stage] = prob
        
        # Normalize probabilities
        total_prob = sum(adjusted_probs.values())
        normalized_probs = {k: v/total_prob for k, v in adjusted_probs.items()}
        
        # Sample stage
        stages = list(normalized_probs.keys())
        probs = list(normalized_probs.values())
        
        return np.random.choice(stages, p=probs)
    
    def _get_time_modifier(self, user_profile: UserProfile, hour: int) -> float:
        """Get time-based bid modifier"""
        time_period = self._get_time_period(hour)
        return user_profile.time_preference.get(time_period, 1.0)
    
    def _get_device_modifier(self, user_profile: UserProfile, device: str) -> float:
        """Get device-based bid modifier"""
        return user_profile.device_preference.get(device, 1.0)
    
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
    
    def get_bridge_analytics(self) -> Dict[str, Any]:
        """Get analytics on bridge performance"""
        
        analytics = {
            'query_generation_stats': self.query_generation_stats,
            'total_sessions': len(self.user_auction_history),
            'segment_performance': {}
        }
        
        # Analyze performance by segment
        for user_id, sessions in self.user_auction_history.items():
            if user_id in self.recsim_model.current_users:
                segment = self.recsim_model.current_users[user_id].segment.value
                
                if segment not in analytics['segment_performance']:
                    analytics['segment_performance'][segment] = {
                        'sessions': 0,
                        'total_cost': 0,
                        'total_revenue': 0,
                        'total_clicks': 0,
                        'total_conversions': 0
                    }
                
                for session in sessions:
                    analytics['segment_performance'][segment]['sessions'] += 1
                    analytics['segment_performance'][segment]['total_cost'] += session['total_cost']
                    analytics['segment_performance'][segment]['total_revenue'] += session['total_revenue']
                    analytics['segment_performance'][segment]['total_clicks'] += session['clicks']
                    analytics['segment_performance'][segment]['total_conversions'] += session['conversions']
        
        # Calculate segment-level metrics
        for segment_data in analytics['segment_performance'].values():
            if segment_data['total_cost'] > 0:
                segment_data['roas'] = segment_data['total_revenue'] / segment_data['total_cost']
                segment_data['ctr'] = segment_data['total_clicks'] / segment_data['sessions']
                segment_data['conversion_rate'] = segment_data['total_conversions'] / max(segment_data['total_clicks'], 1)
        
        return analytics


def test_recsim_auction_bridge():
    """Test the RecSim-AuctionGym bridge"""
    print("Testing RecSim-AuctionGym Bridge")
    print("=" * 50)
    
    # Initialize required components - NO fallbacks
    recsim_model = RecSimUserModel()
    auction_wrapper = AuctionGymWrapper()
    
    # Initialize bridge with required components
    bridge = RecSimAuctionBridge(
        recsim_model=recsim_model,
        auction_wrapper=auction_wrapper
    )
    
    # Test user-to-auction signals mapping
    print("\n1. Testing user-to-auction signals mapping:")
    for segment in UserSegment:
        user_id = f"test_{segment.value}"
        signals = bridge.user_to_auction_signals(
            user_id=user_id,
            context={'hour': 20, 'device': 'mobile'}
        )
        
        print(f"  {segment.value}:")
        print(f"    Suggested Bid: ${signals['suggested_bid']:.2f}")
        print(f"    Quality Score: {signals['quality_score']:.2f}")
        print(f"    Participation Prob: {signals['participation_probability']:.2f}")
    
    # Test query generation
    print("\n2. Testing query generation from user state:")
    for i, segment in enumerate(list(UserSegment)[:3]):  # Test first 3 segments
        user_id = f"query_test_{segment.value}"
        
        # Generate 3 queries for this user
        for query_num in range(3):
            query_data = bridge.generate_query_from_state(
                user_id=user_id,
                product_category="sneakers",
                brand="nike" if query_num == 0 else None
            )
            
            if query_num == 0:  # Only print first query per segment
                print(f"  {segment.value}:")
                print(f"    Query: '{query_data['query']}'")
                print(f"    Journey Stage: {query_data['journey_stage']}")
                print(f"    Intent Strength: {query_data['intent_strength']:.2f}")
    
    # Test segment to bid value mapping
    print("\n3. Testing segment-to-bid-value mapping:")
    from dataclasses import dataclass
    
    # Create sample query intent
    @dataclass
    class SampleIntent:
        stage: str = "purchase"
        intent_strength: float = 0.8
        query_type: str = "transactional"
        keywords: List[str] = None
        bid_modifier: float = 1.5
    
    sample_intent = SampleIntent(keywords=["buy", "sneakers"])
    
    for segment in list(UserSegment)[:3]:
        bid_value = bridge.map_segment_to_bid_value(
            segment=segment,
            query_intent=sample_intent,
            market_context={'competition_level': 0.6, 'hour': 20}
        )
        print(f"  {segment.value}: ${bid_value:.2f}")
    
    # Test complete user session
    print("\n4. Testing complete user auction session:")
    session_result = bridge.simulate_user_auction_session(
        user_id="session_test_impulse",
        num_queries=5,
        product_category="sneakers"
    )
    
    print(f"  User: {session_result['user_id']}")
    print(f"  Queries Generated: {len(session_result['queries'])}")
    print(f"  Auctions Participated: {len(session_result['auctions'])}")
    print(f"  Total Cost: ${session_result['total_cost']:.2f}")
    print(f"  Total Revenue: ${session_result['total_revenue']:.2f}")
    print(f"  Clicks: {session_result['clicks']}")
    print(f"  Conversions: {session_result['conversions']}")
    if session_result['total_cost'] > 0:
        print(f"  ROAS: {session_result.get('roas', 0):.2f}x")
    
    # Run multiple sessions for analytics
    print("\n5. Running multiple sessions for analytics...")
    for segment in list(UserSegment)[:3]:
        for i in range(2):  # 2 sessions per segment
            user_id = f"analytics_{segment.value}_{i}"
            bridge.simulate_user_auction_session(
                user_id=user_id,
                num_queries=3,
                product_category="shoes"
            )
    
    # Get analytics
    analytics = bridge.get_bridge_analytics()
    print(f"  Total Sessions Analyzed: {analytics['total_sessions']}")
    print(f"  Query Generation Stats:")
    for segment, stages in analytics['query_generation_stats'].items():
        print(f"    {segment}: {dict(stages)}")
    
    print(f"  Segment Performance:")
    for segment, perf in analytics['segment_performance'].items():
        if perf['sessions'] > 0:
            print(f"    {segment}:")
            print(f"      Sessions: {perf['sessions']}")
            print(f"      Avg ROAS: {perf.get('roas', 0):.2f}x")
            print(f"      CTR: {perf.get('ctr', 0):.3f}")
    
    print("\nBridge testing completed successfully!")


if __name__ == "__main__":
    test_recsim_auction_bridge()