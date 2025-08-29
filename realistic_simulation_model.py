"""
Realistic Simulation Model
Models what data is ACTUALLY available in real advertising platforms
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import random

@dataclass
class RealAdRequest:
    """What you ACTUALLY see when someone triggers your ad"""
    # Platform data
    platform: str  # google, facebook, tiktok
    timestamp: datetime
    
    # What Google Ads Actually Provides
    search_query: Optional[str] = None  # Only on Google, often hidden for privacy
    device_type: str = "mobile"  # mobile, desktop, tablet
    location: str = "California"  # State/region level only
    hour_of_day: int = 12
    
    # Demographics (often unknown)
    age_range: str = "unknown"  # 25-34, 35-44, or "unknown" (50% of time)
    gender: str = "unknown"  # male, female, unknown
    
    # What you bid
    your_bid: float = 0.0
    your_quality_score: float = 0.0
    
    # Auction result (all you learn)
    won_auction: bool = False
    position: Optional[int] = None  # 1-4 for search
    price_paid: float = 0.0
    
    # User action (binary)
    clicked: bool = False
    
    # Conversion (delayed, might never know)
    converted: Optional[bool] = None  # None = don't know yet
    conversion_value: float = 0.0
    conversion_lag_days: Optional[float] = None

class RealisticChannelBehavior:
    """Models how users ACTUALLY behave differently on each platform"""
    
    def __init__(self):
        # Based on real industry data
        self.channel_behaviors = {
            'google': {
                'base_ctr': {
                    'position_1': 0.08,  # Top position
                    'position_2': 0.04,
                    'position_3': 0.025,
                    'position_4': 0.015
                },
                'intent_multipliers': {
                    'crisis': 2.5,  # "emergency teen help" = high CTR
                    'research': 1.0,  # "parental control apps"
                    'comparison': 1.2,  # "bark vs qustodio"
                    'informational': 0.3  # "what is screen time"
                },
                'device_factors': {
                    'mobile': 0.9,  # Slightly lower on mobile
                    'desktop': 1.1,  # Higher on desktop (serious research)
                    'tablet': 1.0
                },
                'time_patterns': {
                    'business_hours': 1.0,  # 9am-5pm
                    'evening': 1.2,  # 6pm-10pm (parents free)
                    'late_night': 1.8,  # 11pm-2am (crisis searches)
                    'early_morning': 0.6  # 5am-8am
                },
                'conversion_window': (1, 7),  # Days
                'conversion_rate': 0.025  # 2.5% baseline
            },
            'facebook': {
                'base_ctr': {
                    'newsfeed': 0.009,  # In feed
                    'stories': 0.012,  # Stories placement
                    'messenger': 0.005,  # Messenger ads
                    'marketplace': 0.003  # Marketplace
                },
                'audience_multipliers': {
                    'lookalike': 1.3,  # Lookalike audiences
                    'interest': 1.0,  # Interest targeting
                    'broad': 0.6,  # Broad targeting
                    'retargeting': 2.5  # Retargeting previous visitors
                },
                'creative_factors': {
                    'video': 1.4,  # Video ads
                    'carousel': 1.2,  # Multiple images
                    'single_image': 1.0,  # Standard
                    'text_only': 0.5  # Just text
                },
                'time_patterns': {
                    'morning_scroll': 1.1,  # 7am-9am
                    'lunch_break': 1.3,  # 12pm-1pm
                    'evening_scroll': 1.5,  # 7pm-10pm
                    'late_night': 1.2  # 10pm-12am
                },
                'conversion_window': (3, 14),  # Longer consideration
                'conversion_rate': 0.008  # Lower than search
            },
            'tiktok': {
                'base_ctr': {
                    'for_you': 0.015,  # Main feed
                    'following': 0.008,  # Following feed
                    'search': 0.006  # Search results
                },
                'content_match': {
                    'trending_format': 1.8,  # Matches current trend
                    'native_feel': 1.5,  # Looks like user content
                    'obvious_ad': 0.4  # Obviously an ad
                },
                'audience_factors': {
                    'gen_z_parent': 1.5,  # Young parents
                    'millennial_parent': 1.0,
                    'gen_x_parent': 0.6  # Older parents
                },
                'time_patterns': {
                    'morning': 0.7,
                    'afternoon': 0.9,
                    'prime_time': 1.4,  # 8pm-11pm
                    'late_night': 1.6  # 11pm-1am
                },
                'conversion_window': (7, 21),  # Very long
                'conversion_rate': 0.004  # Lowest (discovery platform)
            }
        }
    
    def calculate_click_probability(self, request: RealAdRequest) -> float:
        """Calculate REALISTIC click probability based on platform dynamics"""
        
        channel_data = self.channel_behaviors.get(request.platform, self.channel_behaviors['google'])
        
        # Start with base CTR for position/placement
        if request.platform == 'google':
            base_ctr = channel_data['base_ctr'].get(f'position_{request.position}', 0.01)
            
            # Parse intent from search query
            intent = self._parse_search_intent(request.search_query)
            base_ctr *= channel_data['intent_multipliers'].get(intent, 1.0)
            
        elif request.platform == 'facebook':
            base_ctr = channel_data['base_ctr']['newsfeed']  # Assume newsfeed
            # In reality, you'd know if this is retargeting
            if random.random() < 0.1:  # 10% are retargeting
                base_ctr *= channel_data['audience_multipliers']['retargeting']
            
        else:  # TikTok
            base_ctr = channel_data['base_ctr']['for_you']
            # Young parent bonus
            if request.age_range in ['18-24', '25-34']:
                base_ctr *= channel_data['audience_factors']['gen_z_parent']
        
        # Time of day adjustment
        time_period = self._get_time_period(request.hour_of_day)
        time_mult = channel_data['time_patterns'].get(time_period, 1.0)
        
        # Device adjustment
        if request.platform == 'google':
            device_mult = channel_data['device_factors'].get(request.device_type, 1.0)
        else:
            # Mobile-first platforms
            device_mult = 1.2 if request.device_type == 'mobile' else 0.7
        
        final_ctr = base_ctr * time_mult * device_mult
        
        # Add randomness (real user behavior is noisy)
        noise = np.random.normal(1.0, 0.2)  # 20% standard deviation
        final_ctr *= max(0.5, min(1.5, noise))
        
        return min(1.0, max(0.0, final_ctr))
    
    def calculate_conversion_probability(self, request: RealAdRequest, 
                                        clicked: bool) -> Tuple[bool, Optional[float]]:
        """Calculate if and when a conversion happens"""
        
        if not clicked:
            return False, None
        
        channel_data = self.channel_behaviors[request.platform]
        base_cvr = channel_data['conversion_rate']
        
        # Platform-specific conversion factors
        if request.platform == 'google':
            # High intent searches convert better
            if request.search_query and any(word in request.search_query.lower() 
                                           for word in ['buy', 'price', 'cost', 'trial']):
                base_cvr *= 2.0
            # Crisis searches convert quickly
            if request.hour_of_day in [23, 0, 1, 2]:
                base_cvr *= 1.5
                
        elif request.platform == 'facebook':
            # Retargeting converts much better
            if random.random() < 0.1:  # Assume 10% retargeting
                base_cvr *= 3.0
            # Older parents more likely to convert
            if request.age_range in ['35-44', '45-54']:
                base_cvr *= 1.3
                
        else:  # TikTok
            # Very low baseline, but trending content can work
            if random.random() < 0.05:  # 5% hit trending format
                base_cvr *= 3.0
        
        # Roll the dice
        converts = random.random() < base_cvr
        
        if converts:
            # Calculate conversion delay
            min_delay, max_delay = channel_data['conversion_window']
            
            # Most conversions happen early in window
            delay_days = np.random.exponential(scale=(max_delay - min_delay) / 3) + min_delay
            delay_days = min(max_delay, delay_days)
            
            return True, delay_days
        
        return False, None
    
    def _parse_search_intent(self, query: Optional[str]) -> str:
        """Parse search intent from query"""
        if not query:
            return 'research'
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['emergency', 'crisis', 'urgent', 'help now']):
            return 'crisis'
        elif any(word in query_lower for word in ['vs', 'versus', 'compare', 'best']):
            return 'comparison'
        elif any(word in query_lower for word in ['what is', 'how to', 'guide']):
            return 'informational'
        else:
            return 'research'
    
    def _get_time_period(self, hour: int) -> str:
        """Categorize time of day"""
        if 9 <= hour < 17:
            return 'business_hours'
        elif 17 <= hour < 22:
            return 'evening'
        elif hour >= 22 or hour < 3:
            return 'late_night'
        else:
            return 'early_morning'

class RealisticSimulation:
    """Simulates what the ad platform ACTUALLY teaches the agent"""
    
    def __init__(self):
        self.behavior_model = RealisticChannelBehavior()
        self.conversion_tracking = {}  # Track pending conversions
        self.current_day = 0
        
    def simulate_ad_request(self, platform: str, bid: float, 
                           creative_type: str) -> RealAdRequest:
        """Simulate what happens when you participate in an ad auction"""
        
        # Generate realistic request context
        request = RealAdRequest(
            platform=platform,
            timestamp=datetime.now(),
            device_type=np.random.choice(['mobile', 'desktop', 'tablet'], 
                                        p=[0.65, 0.30, 0.05]),
            hour_of_day=self._generate_realistic_hour(platform),
            location=np.random.choice(['California', 'Texas', 'New York', 'Florida']),
        )
        
        # Add platform-specific data
        if platform == 'google':
            request.search_query = self._generate_search_query()
        
        # Simulate auction (you don't see competitor bids!)
        won, position, price = self._simulate_auction(platform, bid)
        request.won_auction = won
        request.position = position
        request.price_paid = price
        request.your_bid = bid
        
        if won:
            # Calculate click probability
            click_prob = self.behavior_model.calculate_click_probability(request)
            request.clicked = random.random() < click_prob
            
            if request.clicked:
                # Check for conversion (might be delayed)
                converts, delay = self.behavior_model.calculate_conversion_probability(
                    request, request.clicked
                )
                
                if converts:
                    # Schedule conversion for future
                    conversion_day = self.current_day + delay
                    if conversion_day not in self.conversion_tracking:
                        self.conversion_tracking[conversion_day] = []
                    self.conversion_tracking[conversion_day].append({
                        'request': request,
                        'value': 199.99  # Aura subscription
                    })
        
        return request
    
    def process_conversions(self) -> List[Dict]:
        """Process any conversions that should fire today"""
        conversions = []
        if self.current_day in self.conversion_tracking:
            conversions = self.conversion_tracking[self.current_day]
            del self.conversion_tracking[self.current_day]
        return conversions
    
    def _generate_realistic_hour(self, platform: str) -> int:
        """Generate realistic hour based on platform usage patterns"""
        if platform == 'google':
            # Business hours + evening peaks
            return np.random.choice(range(24), p=[
                0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 12am-5am
                0.03, 0.04, 0.05, 0.06, 0.06, 0.05,  # 6am-11am
                0.05, 0.05, 0.06, 0.06, 0.05, 0.04,  # 12pm-5pm
                0.04, 0.06, 0.08, 0.08, 0.05, 0.03   # 6pm-11pm
            ])
        elif platform == 'facebook':
            # Morning scroll, lunch, evening peaks
            return np.random.choice(range(24), p=[
                0.02, 0.01, 0.01, 0.01, 0.01, 0.02,  # 12am-5am
                0.03, 0.06, 0.07, 0.04, 0.03, 0.03,  # 6am-11am
                0.06, 0.04, 0.03, 0.03, 0.03, 0.03,  # 12pm-5pm
                0.04, 0.06, 0.08, 0.09, 0.07, 0.04   # 6pm-11pm
            ])
        else:  # TikTok - late evening focused
            return np.random.choice(range(24), p=[
                0.04, 0.03, 0.02, 0.01, 0.01, 0.01,  # 12am-5am
                0.02, 0.03, 0.03, 0.03, 0.03, 0.03,  # 6am-11am
                0.03, 0.03, 0.03, 0.04, 0.04, 0.05,  # 12pm-5pm
                0.06, 0.08, 0.10, 0.12, 0.10, 0.06   # 6pm-11pm
            ])
    
    def _generate_search_query(self) -> str:
        """Generate realistic search query"""
        queries = [
            "parental control app iphone",
            "teen depression warning signs",
            "monitor kids phone without them knowing",
            "best parental control app 2024",
            "teen mental health crisis help",
            "how to know if teen is depressed",
            "bark vs qustodio vs aura",
            "screen time monitoring app",
            "teen social media monitoring",
            "emergency teen mental health"
        ]
        return np.random.choice(queries)
    
    def _simulate_auction(self, platform: str, bid: float) -> Tuple[bool, int, float]:
        """Simulate auction (you don't see competitor bids!)"""
        
        # Generate hidden competitor bids
        if platform == 'google':
            competitor_bids = np.random.exponential(2.5, size=3) + 1.5  # $1.50-$6 range
        else:
            competitor_bids = np.random.exponential(1.5, size=4) + 0.8  # Lower on social
        
        all_bids = list(competitor_bids) + [bid]
        all_bids.sort(reverse=True)
        
        position = all_bids.index(bid) + 1
        won = position <= 4  # Top 4 positions win
        
        if won and position > 1:
            # Second price auction
            price = all_bids[position] * 1.01  # Pay just above next bidder
        elif won:
            price = bid * 0.85  # First position discount
        else:
            price = 0
        
        return won, position if won else None, price

# Example usage showing what agent ACTUALLY learns:
if __name__ == "__main__":
    sim = RealisticSimulation()
    
    # Day 1: Agent tries Google
    request1 = sim.simulate_ad_request('google', bid=3.50, creative_type='crisis')
    print(f"Google: Won={request1.won_auction}, Clicked={request1.clicked}")
    # Agent learns: "Bid $3.50 on Google -> won position 2, got click, no immediate conversion"
    
    # Day 3: Check for conversions
    sim.current_day = 3
    conversions = sim.process_conversions()
    print(f"Conversions on day 3: {len(conversions)}")
    # Agent learns: "That click from 3 days ago converted!"