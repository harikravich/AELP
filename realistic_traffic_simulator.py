#!/usr/bin/env python3
"""
Realistic Traffic Simulator for Behavioral Health Marketing
Models ACTUAL traffic composition when targeting "parents of teens"
Includes mistargeting, bots, non-parents, and wrong-fit parents
Uses Criteo data for baseline CTR calibration
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os

# Import our persona system
from behavioral_health_persona_factory import (
    BehavioralHealthPersonaFactory, 
    ParentPersona,
    TriggerEvent
)
from trigger_event_system import TriggerEventSystem

# Import Criteo data for CTR calibration
from criteo_data_loader import CriteoDataLoader

class VisitorType(Enum):
    """All possible visitor types in our traffic"""
    # Target parents (35%)
    CRISIS_PARENT = "crisis_parent"
    HIGH_CONCERN_PARENT = "high_concern_parent" 
    MODERATE_CONCERN_PARENT = "moderate_concern_parent"
    CURIOUS_PARENT = "curious_parent"
    
    # Wrong-fit parents (25%)
    PARENT_YOUNG_KIDS = "parent_young_kids"  # Kids under 13
    PARENT_ADULT_KIDS = "parent_adult_kids"  # Kids over 18
    LOCATION_ONLY_PARENT = "location_only_parent"  # Just wants GPS
    
    # Non-parents (40%)
    TEACHER = "teacher"
    THERAPIST = "therapist"
    RESEARCHER = "researcher"
    RANDOM_ADULT = "random_adult"
    TEEN_THEMSELVES = "teen_themselves"
    COMPETITOR = "competitor"
    BOT = "bot"

@dataclass
class Visitor:
    """A visitor to our ads/site - may or may not be our target"""
    visitor_id: str
    visitor_type: VisitorType
    timestamp: datetime
    channel: str
    device: str
    location: str
    
    # Behavioral attributes
    attention_span: float  # 0-1, how long they engage
    price_sensitivity: float  # 0-1
    ad_blindness: float  # 0-1, how much they ignore ads
    
    # For actual parents
    parent_persona: Optional[ParentPersona] = None
    
    # Search/browse intent
    search_query: Optional[str] = None
    referrer: Optional[str] = None
    
    # Engagement history
    ads_seen: List[Dict] = field(default_factory=list)
    pages_viewed: List[str] = field(default_factory=list)
    time_on_site: float = 0.0
    
    def get_base_ctr(self, ad_content: Dict[str, Any]) -> float:
        """Get base CTR for this visitor type"""
        
        # Use Criteo baseline of 1.5% and adjust by visitor type
        criteo_baseline = 0.015
        
        if self.visitor_type == VisitorType.BOT:
            return random.uniform(0.001, 0.003)  # Bots have consistent low CTR
        
        if self.visitor_type == VisitorType.TEEN_THEMSELVES:
            # Teens might click out of curiosity
            return random.uniform(0.005, 0.02)
        
        if self.visitor_type in [VisitorType.TEACHER, VisitorType.THERAPIST]:
            # Professionals might be interested
            return random.uniform(0.02, 0.05)
        
        if self.visitor_type == VisitorType.RESEARCHER:
            # Researchers click to study
            return random.uniform(0.03, 0.06)
        
        if self.visitor_type == VisitorType.COMPETITOR:
            # Competitors definitely click
            return random.uniform(0.7, 0.9)
        
        if self.visitor_type == VisitorType.RANDOM_ADULT:
            # Random mistargeting
            return random.uniform(0.001, 0.005)
        
        # Parents with wrong-fit kids
        if self.visitor_type == VisitorType.PARENT_YOUNG_KIDS:
            return random.uniform(0.005, 0.015)  # Might be planning ahead
        
        if self.visitor_type == VisitorType.PARENT_ADULT_KIDS:
            return random.uniform(0.002, 0.008)  # Low relevance
        
        if self.visitor_type == VisitorType.LOCATION_ONLY_PARENT:
            if "location" in str(ad_content).lower() or "gps" in str(ad_content).lower():
                return random.uniform(0.05, 0.10)
            return random.uniform(0.003, 0.01)
        
        # Actual target parents - use persona
        if self.parent_persona:
            return self.parent_persona.should_click_ad(ad_content)
        
        # Default
        return criteo_baseline

@dataclass 
class TrafficSource:
    """Represents a traffic source/channel"""
    channel: str
    targeting_accuracy: float  # How well targeting works (0-1)
    bot_percentage: float  # Percentage of bot traffic
    cost_per_click: float  # Average CPC
    
    # Visitor type distribution for this channel
    visitor_distribution: Dict[VisitorType, float]

class RealisticTrafficSimulator:
    """Simulates realistic traffic with proper composition"""
    
    def __init__(self):
        # Load Criteo data for CTR calibration
        self.criteo_loader = CriteoDataLoader()
        self._load_criteo_stats()
        
        # Define traffic sources
        self.traffic_sources = self._init_traffic_sources()
        
        # Cache for generated visitors
        self.visitor_cache = []
        
        # Statistics tracking
        self.stats = {
            'total_impressions': 0,
            'total_clicks': 0,
            'clicks_by_type': {},
            'cost_by_channel': {}
        }
    
    def _load_criteo_stats(self):
        """Load Criteo statistics for CTR calibration"""
        try:
            with open('data/criteo_statistics.json', 'r') as f:
                self.criteo_stats = json.load(f)
                self.baseline_ctr = self.criteo_stats.get('click_rate', 0.015)
        except:
            self.baseline_ctr = 0.015  # Default 1.5% CTR
    
    def _init_traffic_sources(self) -> Dict[str, TrafficSource]:
        """Initialize different traffic sources with realistic compositions"""
        
        sources = {}
        
        # Google Search - High intent but lots of mistargeting
        sources['google_search'] = TrafficSource(
            channel='google_search',
            targeting_accuracy=0.4,  # Only 40% are actual parents of teens
            bot_percentage=0.05,
            cost_per_click=3.50,
            visitor_distribution={
                VisitorType.CRISIS_PARENT: 0.05,
                VisitorType.HIGH_CONCERN_PARENT: 0.10,
                VisitorType.MODERATE_CONCERN_PARENT: 0.15,
                VisitorType.CURIOUS_PARENT: 0.10,
                VisitorType.PARENT_YOUNG_KIDS: 0.08,
                VisitorType.PARENT_ADULT_KIDS: 0.02,
                VisitorType.LOCATION_ONLY_PARENT: 0.05,
                VisitorType.TEACHER: 0.15,
                VisitorType.THERAPIST: 0.03,
                VisitorType.RESEARCHER: 0.02,
                VisitorType.RANDOM_ADULT: 0.15,
                VisitorType.TEEN_THEMSELVES: 0.05,
                VisitorType.COMPETITOR: 0.01,
                VisitorType.BOT: 0.04
            }
        )
        
        # Facebook - Better parent targeting but lower intent
        sources['facebook'] = TrafficSource(
            channel='facebook',
            targeting_accuracy=0.6,  # Better targeting
            bot_percentage=0.08,
            cost_per_click=2.20,
            visitor_distribution={
                VisitorType.CRISIS_PARENT: 0.02,
                VisitorType.HIGH_CONCERN_PARENT: 0.08,
                VisitorType.MODERATE_CONCERN_PARENT: 0.20,
                VisitorType.CURIOUS_PARENT: 0.25,
                VisitorType.PARENT_YOUNG_KIDS: 0.15,
                VisitorType.PARENT_ADULT_KIDS: 0.05,
                VisitorType.LOCATION_ONLY_PARENT: 0.08,
                VisitorType.TEACHER: 0.03,
                VisitorType.THERAPIST: 0.01,
                VisitorType.RESEARCHER: 0.01,
                VisitorType.RANDOM_ADULT: 0.10,
                VisitorType.TEEN_THEMSELVES: 0.01,
                VisitorType.COMPETITOR: 0.005,
                VisitorType.BOT: 0.005
            }
        )
        
        # TikTok - Younger parents, lots of curiosity clicks
        sources['tiktok'] = TrafficSource(
            channel='tiktok',
            targeting_accuracy=0.3,  # Poor targeting
            bot_percentage=0.15,
            cost_per_click=1.50,
            visitor_distribution={
                VisitorType.CRISIS_PARENT: 0.01,
                VisitorType.HIGH_CONCERN_PARENT: 0.05,
                VisitorType.MODERATE_CONCERN_PARENT: 0.10,
                VisitorType.CURIOUS_PARENT: 0.15,
                VisitorType.PARENT_YOUNG_KIDS: 0.20,
                VisitorType.PARENT_ADULT_KIDS: 0.02,
                VisitorType.LOCATION_ONLY_PARENT: 0.03,
                VisitorType.TEACHER: 0.05,
                VisitorType.THERAPIST: 0.02,
                VisitorType.RESEARCHER: 0.02,
                VisitorType.RANDOM_ADULT: 0.25,
                VisitorType.TEEN_THEMSELVES: 0.08,
                VisitorType.COMPETITOR: 0.005,
                VisitorType.BOT: 0.015
            }
        )
        
        # YouTube - Video viewers, educational content
        sources['youtube'] = TrafficSource(
            channel='youtube',
            targeting_accuracy=0.45,
            bot_percentage=0.10,
            cost_per_click=2.00,
            visitor_distribution={
                VisitorType.CRISIS_PARENT: 0.03,
                VisitorType.HIGH_CONCERN_PARENT: 0.12,
                VisitorType.MODERATE_CONCERN_PARENT: 0.18,
                VisitorType.CURIOUS_PARENT: 0.15,
                VisitorType.PARENT_YOUNG_KIDS: 0.10,
                VisitorType.PARENT_ADULT_KIDS: 0.03,
                VisitorType.LOCATION_ONLY_PARENT: 0.04,
                VisitorType.TEACHER: 0.08,
                VisitorType.THERAPIST: 0.02,
                VisitorType.RESEARCHER: 0.03,
                VisitorType.RANDOM_ADULT: 0.15,
                VisitorType.TEEN_THEMSELVES: 0.06,
                VisitorType.COMPETITOR: 0.005,
                VisitorType.BOT: 0.005
            }
        )
        
        # Display Network - Lowest quality traffic
        sources['display'] = TrafficSource(
            channel='display',
            targeting_accuracy=0.2,
            bot_percentage=0.25,
            cost_per_click=0.80,
            visitor_distribution={
                VisitorType.CRISIS_PARENT: 0.01,
                VisitorType.HIGH_CONCERN_PARENT: 0.03,
                VisitorType.MODERATE_CONCERN_PARENT: 0.08,
                VisitorType.CURIOUS_PARENT: 0.08,
                VisitorType.PARENT_YOUNG_KIDS: 0.05,
                VisitorType.PARENT_ADULT_KIDS: 0.05,
                VisitorType.LOCATION_ONLY_PARENT: 0.05,
                VisitorType.TEACHER: 0.05,
                VisitorType.THERAPIST: 0.01,
                VisitorType.RESEARCHER: 0.01,
                VisitorType.RANDOM_ADULT: 0.40,
                VisitorType.TEEN_THEMSELVES: 0.03,
                VisitorType.COMPETITOR: 0.005,
                VisitorType.BOT: 0.145
            }
        )
        
        return sources
    
    def generate_visitor(self, 
                        channel: str,
                        timestamp: datetime,
                        force_type: Optional[VisitorType] = None) -> Visitor:
        """Generate a realistic visitor based on channel"""
        
        if channel not in self.traffic_sources:
            channel = 'display'  # Default to worst quality
        
        source = self.traffic_sources[channel]
        
        # Determine visitor type
        if force_type:
            visitor_type = force_type
        else:
            visitor_type = np.random.choice(
                list(source.visitor_distribution.keys()),
                p=list(source.visitor_distribution.values())
            )
        
        # Generate visitor attributes
        visitor_id = f"v_{timestamp.timestamp()}_{random.randint(1000, 9999)}"
        
        # Device distribution varies by channel
        if channel == 'tiktok':
            device = random.choices(['mobile', 'tablet'], weights=[0.85, 0.15])[0]
        elif channel == 'facebook':
            device = random.choices(['mobile', 'desktop', 'tablet'], weights=[0.65, 0.30, 0.05])[0]
        elif channel == 'google_search':
            device = random.choices(['mobile', 'desktop', 'tablet'], weights=[0.55, 0.40, 0.05])[0]
        else:
            device = random.choices(['mobile', 'desktop', 'tablet'], weights=[0.50, 0.45, 0.05])[0]
        
        # Location (simplified)
        locations = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        location = random.choice(locations)
        
        # Behavioral attributes
        if visitor_type == VisitorType.BOT:
            attention_span = 0.01
            price_sensitivity = 0.0
            ad_blindness = 0.99
        elif visitor_type == VisitorType.COMPETITOR:
            attention_span = 0.95  # They study everything
            price_sensitivity = 0.0  # Don't care about price
            ad_blindness = 0.0  # Click everything
        elif visitor_type == VisitorType.RANDOM_ADULT:
            attention_span = random.uniform(0.01, 0.1)
            price_sensitivity = random.uniform(0.7, 1.0)
            ad_blindness = random.uniform(0.8, 0.95)
        else:
            attention_span = random.uniform(0.3, 0.8)
            price_sensitivity = random.uniform(0.3, 0.9)
            ad_blindness = random.uniform(0.2, 0.6)
        
        # Create visitor
        visitor = Visitor(
            visitor_id=visitor_id,
            visitor_type=visitor_type,
            timestamp=timestamp,
            channel=channel,
            device=device,
            location=location,
            attention_span=attention_span,
            price_sensitivity=price_sensitivity,
            ad_blindness=ad_blindness
        )
        
        # If it's a parent type, create full persona
        if visitor_type in [VisitorType.CRISIS_PARENT, VisitorType.HIGH_CONCERN_PARENT,
                           VisitorType.MODERATE_CONCERN_PARENT, VisitorType.CURIOUS_PARENT]:
            visitor.parent_persona = self._create_parent_for_type(visitor_type)
            if visitor.parent_persona:
                visitor.search_query = visitor.parent_persona.generate_search_query()
        
        # Generate search query for other types
        elif visitor_type == VisitorType.TEACHER:
            visitor.search_query = random.choice([
                "classroom management apps",
                "student monitoring software",
                "digital citizenship curriculum",
                "cyberbullying prevention schools"
            ])
        elif visitor_type == VisitorType.PARENT_YOUNG_KIDS:
            visitor.search_query = random.choice([
                "parental controls for 10 year old",
                "kids internet safety",
                "youtube kids alternatives",
                "screen time limits elementary"
            ])
        elif visitor_type == VisitorType.TEEN_THEMSELVES:
            visitor.search_query = random.choice([
                "how to bypass parental controls",
                "disable screen time iPhone",
                "get around Aura monitoring",
                "parent tracking apps how to stop"
            ])
        
        return visitor
    
    def _create_parent_for_type(self, visitor_type: VisitorType) -> ParentPersona:
        """Create appropriate parent persona for visitor type"""
        
        if visitor_type == VisitorType.CRISIS_PARENT:
            # High urgency triggers
            trigger = random.choice([
                TriggerEvent.FOUND_SELF_HARM_CONTENT,
                TriggerEvent.CYBERBULLYING_INCIDENT,
                TriggerEvent.SUICIDE_IDEATION_DISCOVERED
            ])
            parent = BehavioralHealthPersonaFactory.create_parent_with_trigger(trigger)
            parent.current_concern_level = random.uniform(8, 10)
            
        elif visitor_type == VisitorType.HIGH_CONCERN_PARENT:
            trigger = random.choice([
                TriggerEvent.GRADES_DROPPING,
                TriggerEvent.SLEEP_DISRUPTION_SEVERE,
                TriggerEvent.SOCIAL_ISOLATION
            ])
            parent = BehavioralHealthPersonaFactory.create_parent_with_trigger(trigger)
            parent.current_concern_level = random.uniform(6, 8)
            
        elif visitor_type == VisitorType.MODERATE_CONCERN_PARENT:
            trigger = random.choice([
                TriggerEvent.TOO_MUCH_SCREEN_TIME,
                TriggerEvent.FRIEND_HAD_INCIDENT,
                TriggerEvent.NEWS_ARTICLE_READ
            ])
            parent = BehavioralHealthPersonaFactory.create_parent_with_trigger(trigger)
            parent.current_concern_level = random.uniform(4, 6)
            
        else:  # CURIOUS_PARENT
            trigger = random.choice([
                TriggerEvent.GENERAL_WORRY,
                TriggerEvent.PREVENTION_MINDED,
                TriggerEvent.SCHOOL_NEWSLETTER
            ])
            parent = BehavioralHealthPersonaFactory.create_parent_with_trigger(trigger)
            parent.current_concern_level = random.uniform(1, 4)
        
        return parent
    
    def simulate_impression(self,
                          visitor: Visitor,
                          ad_content: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate an ad impression and potential click"""
        
        self.stats['total_impressions'] += 1
        
        # Calculate CTR based on visitor and ad match
        base_ctr = visitor.get_base_ctr(ad_content)
        
        # Adjust for ad blindness
        adjusted_ctr = base_ctr * (1 - visitor.ad_blindness)
        
        # Adjust for device (mobile has slightly lower CTR)
        if visitor.device == 'mobile':
            adjusted_ctr *= 0.85
        elif visitor.device == 'tablet':
            adjusted_ctr *= 0.9
        
        # Time of day adjustment (late night and early morning lower)
        hour = visitor.timestamp.hour
        if hour >= 0 and hour < 6:
            adjusted_ctr *= 0.7
        elif hour >= 22:
            adjusted_ctr *= 0.8
        
        # Frequency penalty (seen similar ads)
        similar_ads = sum(1 for ad in visitor.ads_seen 
                         if ad.get('creative_family') == ad_content.get('creative_family'))
        if similar_ads > 0:
            adjusted_ctr *= (0.8 ** similar_ads)
        
        # Determine click
        clicked = random.random() < adjusted_ctr
        
        # Calculate cost
        source = self.traffic_sources[visitor.channel]
        if clicked:
            # Actual CPC varies
            cpc = source.cost_per_click * random.uniform(0.7, 1.3)
            self.stats['total_clicks'] += 1
            
            # Track by type
            vtype = visitor.visitor_type.value
            self.stats['clicks_by_type'][vtype] = self.stats['clicks_by_type'].get(vtype, 0) + 1
            
            # Track cost by channel
            channel = visitor.channel
            self.stats['cost_by_channel'][channel] = self.stats['cost_by_channel'].get(channel, 0) + cpc
        else:
            cpc = 0
        
        # Record ad seen
        visitor.ads_seen.append(ad_content)
        
        return {
            'visitor_id': visitor.visitor_id,
            'visitor_type': visitor.visitor_type.value,
            'timestamp': visitor.timestamp.isoformat(),
            'channel': visitor.channel,
            'device': visitor.device,
            'location': visitor.location,
            'ad_content': ad_content,
            'clicked': clicked,
            'ctr': adjusted_ctr,
            'cpc': cpc,
            'search_query': visitor.search_query,
            'parent_concern': visitor.parent_persona.current_concern_level if visitor.parent_persona else None
        }
    
    def simulate_traffic_hour(self,
                            budget: float,
                            channel_mix: Dict[str, float],
                            timestamp: datetime) -> List[Dict[str, Any]]:
        """Simulate an hour of traffic given budget and channel mix"""
        
        impressions = []
        remaining_budget = budget
        
        for channel, channel_pct in channel_mix.items():
            if channel not in self.traffic_sources:
                continue
            
            channel_budget = budget * channel_pct
            source = self.traffic_sources[channel]
            
            # Estimate impressions based on budget and CPC
            estimated_clicks = channel_budget / source.cost_per_click
            estimated_impressions = estimated_clicks / self.baseline_ctr
            
            # Generate impressions
            for _ in range(int(estimated_impressions)):
                if remaining_budget <= 0:
                    break
                
                # Generate visitor
                visitor = self.generate_visitor(channel, timestamp)
                
                # Create ad content (simplified for now)
                ad_content = self._generate_ad_content(visitor, channel)
                
                # Simulate impression
                result = self.simulate_impression(visitor, ad_content)
                
                if result['clicked']:
                    remaining_budget -= result['cpc']
                
                impressions.append(result)
        
        return impressions
    
    def _generate_ad_content(self, visitor: Visitor, channel: str) -> Dict[str, Any]:
        """Generate appropriate ad content for visitor and channel"""
        
        # Crisis-focused ads for high-concern parents
        if visitor.visitor_type in [VisitorType.CRISIS_PARENT, VisitorType.HIGH_CONCERN_PARENT]:
            return {
                'creative_family': 'crisis',
                'headline': 'Get immediate help for your teen',
                'description': 'Clinician-designed monitoring for mental health',
                'urgency_level': 9,
                'mentions_clinical': True,
                'mentions_crisis': True,
                'price_shown': None  # Don't show price to crisis parents
            }
        
        # Preventative messaging for moderate concern
        elif visitor.visitor_type == VisitorType.MODERATE_CONCERN_PARENT:
            return {
                'creative_family': 'prevention',
                'headline': 'Build healthy screen time habits',
                'description': '73% of parents miss digital warning signs',
                'urgency_level': 5,
                'mentions_clinical': True,
                'mentions_crisis': False,
                'price_shown': 14.99
            }
        
        # General messaging for everyone else
        else:
            return {
                'creative_family': 'general',
                'headline': 'Smart parental controls for modern families',
                'description': 'Monitor and protect your kids online',
                'urgency_level': 3,
                'mentions_clinical': False,
                'mentions_crisis': False,
                'price_shown': 14.99
            }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of simulation"""
        
        overall_ctr = (self.stats['total_clicks'] / self.stats['total_impressions'] 
                      if self.stats['total_impressions'] > 0 else 0)
        
        # CTR by visitor type
        ctr_by_type = {}
        for vtype in VisitorType:
            type_clicks = self.stats['clicks_by_type'].get(vtype.value, 0)
            # Estimate impressions (simplified)
            type_impressions = self.stats['total_impressions'] * 0.1  # Rough estimate
            if type_impressions > 0:
                ctr_by_type[vtype.value] = type_clicks / type_impressions
        
        return {
            'total_impressions': self.stats['total_impressions'],
            'total_clicks': self.stats['total_clicks'],
            'overall_ctr': overall_ctr,
            'clicks_by_type': self.stats['clicks_by_type'],
            'cost_by_channel': self.stats['cost_by_channel'],
            'total_cost': sum(self.stats['cost_by_channel'].values()),
            'ctr_by_type': ctr_by_type
        }


if __name__ == "__main__":
    print("Testing Realistic Traffic Simulator\n")
    
    # Create simulator
    simulator = RealisticTrafficSimulator()
    
    # Test generating different visitor types
    print("Sample Visitors by Channel:\n")
    
    for channel in ['google_search', 'facebook', 'tiktok']:
        print(f"\n{channel.upper()}:")
        for _ in range(5):
            visitor = simulator.generate_visitor(channel, datetime.now())
            print(f"  - {visitor.visitor_type.value}: {visitor.search_query or 'No query'}")
    
    # Simulate an hour of traffic
    print("\n\nSimulating 1 hour of traffic with $100 budget:\n")
    
    channel_mix = {
        'google_search': 0.4,
        'facebook': 0.3,
        'youtube': 0.2,
        'tiktok': 0.1
    }
    
    impressions = simulator.simulate_traffic_hour(
        budget=100,
        channel_mix=channel_mix,
        timestamp=datetime.now()
    )
    
    # Show results
    stats = simulator.get_summary_stats()
    
    print(f"Total Impressions: {stats['total_impressions']}")
    print(f"Total Clicks: {stats['total_clicks']}")
    print(f"Overall CTR: {stats['overall_ctr']:.2%}")
    print(f"Total Cost: ${stats['total_cost']:.2f}")
    
    print("\nClicks by Visitor Type:")
    for vtype, clicks in sorted(stats['clicks_by_type'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {vtype}: {clicks}")
    
    print("\nCost by Channel:")
    for channel, cost in stats['cost_by_channel'].items():
        print(f"  {channel}: ${cost:.2f}")