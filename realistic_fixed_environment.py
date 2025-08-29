"""
Realistic GAELP Environment - ONLY REAL DATA
Uses only data actually available from ad platforms and your own tracking
NO FANTASY user tracking or competitor visibility
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import random
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class AdPlatformRequest:
    """What you ACTUALLY know when bidding on ad platforms"""
    # Platform and timing
    platform: str  # google, facebook, tiktok
    timestamp: datetime
    
    # Context you know BEFORE bidding
    keyword: Optional[str] = None  # Only on Google, sometimes hidden
    device_type: str = "mobile"  # mobile, desktop, tablet
    geo_location: str = "US-CA"  # State level
    hour_of_day: int = 12
    day_of_week: int = 1  # Monday
    
    # Your campaign context
    campaign_id: str = ""
    ad_group_id: str = ""
    creative_id: str = ""
    
    # Historical performance (YOUR data only)
    campaign_ctr: float = 0.02  # Your historical CTR
    campaign_cvr: float = 0.01  # Your historical CVR
    campaign_cpc: float = 3.50  # Your average CPC
    
    # Budget state
    daily_budget: float = 1000.0
    budget_spent_today: float = 0.0
    hour_budget_pace: float = 1.0  # 1.0 = on pace

@dataclass
class AdPlatformResponse:
    """What the platform tells you AFTER the auction"""
    # Auction result
    won: bool = False
    position: Optional[int] = None  # 1-4 on Google, None on social
    price_paid: float = 0.0
    
    # Immediate feedback (if won)
    impression_served: bool = False
    clicked: bool = False
    
    # Quality feedback
    quality_score: Optional[float] = None  # Google provides this
    relevance_score: Optional[float] = None  # Facebook provides this
    
    # Market signals (inferred, not direct)
    auction_pressure: str = "medium"  # low/medium/high based on price

@dataclass
class PostClickData:
    """What you track AFTER someone clicks your ad"""
    # Click data (from platform)
    click_id: str  # GCLID, FBCLID, etc.
    landing_page: str
    landing_timestamp: datetime
    
    # On-site behavior (YOUR tracking)
    pages_viewed: List[str] = field(default_factory=list)
    time_on_site: float = 0.0
    scroll_depths: Dict[str, float] = field(default_factory=dict)
    
    # Engagement (YOUR tracking)
    videos_watched: int = 0
    forms_started: int = 0
    buttons_clicked: List[str] = field(default_factory=list)
    
    # Conversion (YOUR tracking)
    converted: bool = False
    conversion_type: Optional[str] = None  # trial, purchase
    conversion_value: float = 0.0
    conversion_timestamp: Optional[datetime] = None

class RealisticFixedEnvironment:
    """
    REALISTIC simulation environment using only real available data
    No fantasy user tracking, no competitor visibility
    """
    
    def __init__(self, max_budget: float = 10000.0, max_steps: int = 1000):
        self.max_budget = max_budget
        self.max_steps = max_steps
        self.current_step = 0
        self.budget_spent = 0.0
        self.episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Realistic metrics (what you ACTUALLY get)
        self.metrics = {
            # Platform metrics (aggregated)
            'impressions': 0,
            'clicks': 0,
            'spend': 0.0,
            'avg_position': 0.0,
            'avg_cpc': 0.0,
            
            # Post-click metrics (YOUR data)
            'landing_page_views': 0,
            'engaged_sessions': 0,  # >30 seconds
            'trial_starts': 0,
            'purchases': 0,
            'revenue': 0.0,
            
            # Attribution metrics (YOUR tracking)
            'last_click_conversions': 0,
            'view_through_conversions': 0,  # If using pixel
            'attributed_revenue': 0.0,
            
            # NO FANTASY METRICS like:
            # - user journey stages
            # - competitor impressions
            # - cross-platform user tracking
        }
        
        # Platform-specific behavior models (based on industry benchmarks)
        self.platform_models = {
            'google': {
                'auction_dynamics': {
                    'avg_competitors': 3.5,
                    'price_volatility': 0.3,
                    'quality_score_impact': 0.4
                },
                'user_behavior': {
                    'base_ctr': {1: 0.08, 2: 0.04, 3: 0.025, 4: 0.015},  # By position
                    'intent_modifiers': {
                        'high': 1.5,  # "buy now", "price"
                        'medium': 1.0,  # "reviews", "compare"
                        'low': 0.5  # "what is", "how to"
                    },
                    'time_modifiers': {
                        'business': 1.0,  # 9am-5pm
                        'evening': 1.2,  # 6pm-10pm
                        'late_night': 1.5,  # 11pm-2am (crisis)
                        'early_morning': 0.7  # 3am-8am
                    }
                },
                'conversion_model': {
                    'base_cvr': 0.025,  # 2.5% baseline
                    'click_to_conversion_window': (0, 7),  # Days
                    'attribution_window': 30
                }
            },
            'facebook': {
                'auction_dynamics': {
                    'avg_competitors': 5.0,
                    'price_volatility': 0.4,
                    'relevance_score_impact': 0.5
                },
                'user_behavior': {
                    'base_ctr': {'feed': 0.009, 'stories': 0.012, 'reels': 0.015},
                    'audience_modifiers': {
                        'broad': 0.6,
                        'interest': 1.0,
                        'lookalike': 1.3,
                        'retargeting': 2.5
                    },
                    'creative_modifiers': {
                        'image': 1.0,
                        'video': 1.4,
                        'carousel': 1.2
                    }
                },
                'conversion_model': {
                    'base_cvr': 0.008,
                    'click_to_conversion_window': (1, 14),
                    'attribution_window': 28
                }
            },
            'tiktok': {
                'auction_dynamics': {
                    'avg_competitors': 4.0,
                    'price_volatility': 0.5,
                    'creative_score_impact': 0.6
                },
                'user_behavior': {
                    'base_ctr': {'for_you': 0.015, 'following': 0.008},
                    'content_modifiers': {
                        'trending': 1.8,
                        'native_style': 1.5,
                        'obvious_ad': 0.4
                    }
                },
                'conversion_model': {
                    'base_cvr': 0.004,
                    'click_to_conversion_window': (3, 21),
                    'attribution_window': 28
                }
            }
        }
        
        # Click tracking (simulates your post-click data)
        self.click_tracking = {}  # click_id -> PostClickData
        
        # Conversion queue (simulates delayed conversions)
        self.pending_conversions = []  # List of (conversion_day, click_id, value)
        
        logger.info(f"Initialized REALISTIC environment: budget=${max_budget}, steps={max_steps}")
    
    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.budget_spent = 0.0
        self.metrics = {k: 0 for k in self.metrics}
        self.click_tracking = {}
        self.pending_conversions = []
        return self._get_observation()
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step with REALISTIC data only
        
        Args:
            action: {
                'platform': 'google',
                'bid': 3.50,
                'keyword': 'parental controls',  # Google only
                'audience': 'parents_25_45',  # Facebook only
                'creative_id': 'creative_001'
            }
        """
        self.current_step += 1
        
        # Create realistic ad request
        request = self._create_ad_request(action)
        
        # Simulate auction (you don't see competitor bids!)
        response = self._simulate_auction(request, action['bid'])
        
        # Track results
        reward = 0.0
        if response.won:
            self.metrics['impressions'] += 1
            self.metrics['spend'] += response.price_paid
            self.budget_spent += response.price_paid
            
            # Update average position (Google only)
            if request.platform == 'google' and response.position:
                self.metrics['avg_position'] = (
                    (self.metrics['avg_position'] * (self.metrics['impressions'] - 1) + 
                     response.position) / self.metrics['impressions']
                )
            
            # Simulate click based on realistic CTR
            if response.clicked:
                self.metrics['clicks'] += 1
                
                # Create post-click tracking
                click_id = f"click_{self.current_step}_{random.randint(1000, 9999)}"
                self._track_click(click_id, request, response)
                
                # Small immediate reward for click
                reward += 0.1
                
                # Check for conversion (might be delayed)
                conversion_data = self._simulate_conversion(click_id, request)
                if conversion_data:
                    if conversion_data['immediate']:
                        # Immediate conversion (rare)
                        self.metrics['purchases'] += 1
                        self.metrics['revenue'] += conversion_data['value']
                        reward += conversion_data['value'] / 100  # Normalized reward
                    else:
                        # Schedule delayed conversion
                        self.pending_conversions.append((
                            self.current_step + conversion_data['delay_steps'],
                            click_id,
                            conversion_data['value']
                        ))
            
            # Negative reward for spend
            reward -= response.price_paid / 100
        
        # Process any delayed conversions that should fire now
        self._process_delayed_conversions()
        
        # Check if episode done
        done = (self.current_step >= self.max_steps or 
                self.budget_spent >= self.max_budget)
        
        # Prepare info dict with REAL data only
        info = {
            'request': request.__dict__,
            'response': response.__dict__,
            'metrics': self.metrics.copy()
        }
        
        return self._get_observation(), reward, done, info
    
    def _create_ad_request(self, action: Dict[str, Any]) -> AdPlatformRequest:
        """Create realistic ad request based on action and context"""
        now = datetime.now()
        
        platform = action.get('platform', 'google')
        
        # Realistic keyword (Google only)
        keyword = None
        if platform == 'google':
            keyword = action.get('keyword', self._sample_keyword())
        
        # Calculate historical performance (YOUR data)
        campaign_ctr = self.metrics['clicks'] / max(1, self.metrics['impressions'])
        campaign_cvr = self.metrics['purchases'] / max(1, self.metrics['clicks'])
        campaign_cpc = self.metrics['spend'] / max(1, self.metrics['clicks'])
        
        return AdPlatformRequest(
            platform=platform,
            timestamp=now,
            keyword=keyword,
            device_type=np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.65, 0.30, 0.05]),
            geo_location=np.random.choice(['US-CA', 'US-TX', 'US-NY', 'US-FL']),
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            campaign_id=action.get('campaign_id', 'campaign_001'),
            creative_id=action.get('creative_id', 'creative_001'),
            campaign_ctr=campaign_ctr or 0.02,
            campaign_cvr=campaign_cvr or 0.01,
            campaign_cpc=campaign_cpc or 3.50,
            daily_budget=self.max_budget,
            budget_spent_today=self.budget_spent
        )
    
    def _simulate_auction(self, request: AdPlatformRequest, bid: float) -> AdPlatformResponse:
        """
        Simulate auction without seeing competitor bids
        This is what ACTUALLY happens - you bid blind!
        """
        platform_model = self.platform_models[request.platform]
        dynamics = platform_model['auction_dynamics']
        
        # Generate hidden competitor landscape (you don't see this!)
        num_competitors = int(np.random.poisson(dynamics['avg_competitors']))
        
        # Simulate market pressure (you only infer this from prices)
        if request.hour_of_day in [9, 10, 14, 15, 20, 21]:  # Peak hours
            competition_multiplier = 1.3
        else:
            competition_multiplier = 1.0
        
        # Hidden competitor bids (YOU NEVER SEE THESE)
        if request.platform == 'google':
            # Google second-price auction
            competitor_bids = np.random.exponential(2.0, num_competitors) * competition_multiplier + 1.0
            
            # Quality score affects ranking
            your_rank_score = bid * (0.7 + 0.3 * random.random())  # Simulated quality
            competitor_rank_scores = competitor_bids * np.random.uniform(0.6, 0.9, num_competitors)
            
            all_scores = np.append(competitor_rank_scores, your_rank_score)
            position = (all_scores > your_rank_score).sum() + 1
            
            won = position <= 4  # Top 4 positions
            
            if won:
                # Second price calculation
                if position < len(all_scores):
                    next_score = sorted(all_scores, reverse=True)[position]
                    price_paid = next_score / (0.7 + 0.3 * random.random()) * 1.01
                    price_paid = min(price_paid, bid)
                else:
                    price_paid = bid * 0.8
            else:
                price_paid = 0.0
                position = None
        else:
            # Facebook/TikTok - CPM model
            avg_cpm = 15.0 if request.platform == 'facebook' else 10.0
            threshold = 1 / (1 + np.exp(-(bid - avg_cpm) / 3))
            won = random.random() < threshold
            price_paid = bid * 0.9 if won else 0.0
            position = None  # No position on social
        
        # Calculate click probability if won
        clicked = False
        if won:
            ctr = self._calculate_realistic_ctr(request, position)
            clicked = random.random() < ctr
        
        # Infer market pressure from price
        if won and bid > 0:
            price_ratio = price_paid / bid
            if price_ratio > 0.8:
                auction_pressure = "high"
            elif price_ratio > 0.5:
                auction_pressure = "medium"
            else:
                auction_pressure = "low"
        else:
            auction_pressure = "unknown"
        
        return AdPlatformResponse(
            won=won,
            position=position,
            price_paid=price_paid,
            impression_served=won,
            clicked=clicked,
            auction_pressure=auction_pressure
        )
    
    def _calculate_realistic_ctr(self, request: AdPlatformRequest, position: Optional[int]) -> float:
        """Calculate CTR based on real industry benchmarks"""
        platform_model = self.platform_models[request.platform]
        behavior = platform_model['user_behavior']
        
        if request.platform == 'google':
            # Position-based CTR
            base_ctr = behavior['base_ctr'].get(position, 0.01)
            
            # Intent modifier based on keyword
            if request.keyword:
                if any(word in request.keyword.lower() for word in ['buy', 'price', 'cost']):
                    intent = 'high'
                elif any(word in request.keyword.lower() for word in ['review', 'compare', 'vs']):
                    intent = 'medium'
                else:
                    intent = 'low'
                base_ctr *= behavior['intent_modifiers'][intent]
            
            # Time modifier
            hour = request.hour_of_day
            if 9 <= hour < 17:
                time_period = 'business'
            elif 17 <= hour < 22:
                time_period = 'evening'
            elif hour >= 22 or hour < 3:
                time_period = 'late_night'
            else:
                time_period = 'early_morning'
            
            base_ctr *= behavior['time_modifiers'][time_period]
            
        elif request.platform == 'facebook':
            base_ctr = behavior['base_ctr']['feed']
            # Simplified - in reality would depend on audience targeting
            base_ctr *= behavior['audience_modifiers']['interest']
            
        else:  # TikTok
            base_ctr = behavior['base_ctr']['for_you']
            base_ctr *= behavior['content_modifiers']['native_style']
        
        # Add noise
        noise = np.random.normal(1.0, 0.15)
        return max(0.0, min(1.0, base_ctr * noise))
    
    def _track_click(self, click_id: str, request: AdPlatformRequest, response: AdPlatformResponse):
        """Track post-click data (what YOU can measure)"""
        # Simulate on-site behavior
        pages = self._simulate_site_behavior(request)
        
        self.click_tracking[click_id] = PostClickData(
            click_id=click_id,
            landing_page="/features/balance",
            landing_timestamp=request.timestamp,
            pages_viewed=pages,
            time_on_site=len(pages) * np.random.uniform(15, 45),
            forms_started=1 if len(pages) > 3 else 0,
            converted=False  # Will be set if conversion happens
        )
        
        self.metrics['landing_page_views'] += 1
        if len(pages) > 2:
            self.metrics['engaged_sessions'] += 1
    
    def _simulate_site_behavior(self, request: AdPlatformRequest) -> List[str]:
        """Simulate realistic on-site behavior"""
        # Different behavior patterns based on source
        if request.platform == 'google':
            # High intent - more pages
            if request.hour_of_day in [22, 23, 0, 1, 2]:
                # Crisis parent - quick to signup
                return random.choice([
                    ['/features/balance', '/pricing', '/signup'],
                    ['/features/balance', '/signup']
                ])
            else:
                # Researcher
                return random.choice([
                    ['/features/balance', '/how-it-works', '/pricing'],
                    ['/features/balance', '/features/alerts', '/blog/teen-mental-health', '/pricing']
                ])
        else:
            # Social traffic - less directed
            return random.choice([
                ['/features/balance'],
                ['/features/balance', '/about'],
                []  # Bounce
            ])
    
    def _simulate_conversion(self, click_id: str, request: AdPlatformRequest) -> Optional[Dict]:
        """Simulate conversion decision with realistic delay"""
        platform_model = self.platform_models[request.platform]
        conversion_model = platform_model['conversion_model']
        
        # Get base CVR
        base_cvr = conversion_model['base_cvr']
        
        # Adjust for context
        if request.platform == 'google' and request.keyword:
            if 'crisis' in request.keyword.lower() or 'emergency' in request.keyword.lower():
                base_cvr *= 2.5  # Crisis converts better
        
        # Time of day adjustment
        if request.hour_of_day in [22, 23, 0, 1, 2]:
            base_cvr *= 1.5  # Late night = higher intent
        
        # Check if converts
        if random.random() < base_cvr:
            # Calculate delay
            min_delay, max_delay = conversion_model['click_to_conversion_window']
            delay_days = np.random.exponential((max_delay - min_delay) / 3) + min_delay
            delay_steps = int(delay_days * 24)  # Convert to hourly steps
            
            return {
                'immediate': delay_steps == 0,
                'delay_steps': delay_steps,
                'value': 199.99  # Aura subscription value
            }
        
        return None
    
    def _process_delayed_conversions(self):
        """Process any conversions that should fire now"""
        conversions_to_process = []
        remaining_conversions = []
        
        for conversion_step, click_id, value in self.pending_conversions:
            if conversion_step <= self.current_step:
                conversions_to_process.append((click_id, value))
            else:
                remaining_conversions.append((conversion_step, click_id, value))
        
        self.pending_conversions = remaining_conversions
        
        for click_id, value in conversions_to_process:
            self.metrics['purchases'] += 1
            self.metrics['revenue'] += value
            self.metrics['last_click_conversions'] += 1
            self.metrics['attributed_revenue'] += value
            
            # Update click tracking
            if click_id in self.click_tracking:
                self.click_tracking[click_id].converted = True
                self.click_tracking[click_id].conversion_value = value
                self.click_tracking[click_id].conversion_timestamp = datetime.now()
            
            logger.info(f"Delayed conversion fired: ${value:.2f}")
    
    def _sample_keyword(self) -> str:
        """Sample realistic search keywords"""
        keywords = [
            "parental control app",
            "monitor teen phone",
            "teen mental health app",
            "track kid location",
            "screen time app",
            "teen depression signs",
            "cyberbullying prevention",
            "aura parental controls",
            "best parental control app 2024",
            "how to monitor teen social media"
        ]
        return random.choice(keywords)
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observable state (ONLY REAL DATA)"""
        return {
            'step': self.current_step,
            'budget_spent': self.budget_spent,
            'budget_remaining': self.max_budget - self.budget_spent,
            'impressions': self.metrics['impressions'],
            'clicks': self.metrics['clicks'],
            'conversions': self.metrics['purchases'],
            'revenue': self.metrics['revenue'],
            'ctr': self.metrics['clicks'] / max(1, self.metrics['impressions']),
            'cvr': self.metrics['purchases'] / max(1, self.metrics['clicks']),
            'roas': self.metrics['revenue'] / max(1, self.budget_spent),
            'avg_cpc': self.metrics['spend'] / max(1, self.metrics['clicks']),
            'pending_conversions': len(self.pending_conversions)
        }