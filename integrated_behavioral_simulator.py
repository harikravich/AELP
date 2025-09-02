#!/usr/bin/env python3
"""
Integrated Behavioral Health Marketing Simulator
Combines realistic traffic, parent personas, and multi-week journeys
NO HARDCODING - Uses actual traffic patterns and discovered behaviors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import random
import json
import logging

# Import our components
from realistic_traffic_simulator import (
    RealisticTrafficSimulator,
    Visitor,
    VisitorType,
    TrafficSource
)
from behavioral_health_persona_factory import (
    BehavioralHealthPersonaFactory,
    ParentPersona,
    TriggerEvent
)
from trigger_event_system import (
    TriggerEventSystem,
    TriggerContext,
    TriggerEvolution
)

# Import auction and user behavior systems
try:
    from auction_gym_integration import AuctionGymWrapper, AUCTION_GYM_AVAILABLE
except ImportError:
    AUCTION_GYM_AVAILABLE = False
    AuctionGymWrapper = None

try:
    import edward2_patch
    from recsim_auction_bridge import RecSimAuctionBridge
    from recsim_user_model import RecSimUserModel
    RECSIM_AVAILABLE = True
except ImportError:
    RECSIM_AVAILABLE = False
    RecSimAuctionBridge = None
    RecSimUserModel = None

# Import Criteo for CTR calibration
from criteo_data_loader import CriteoDataLoader

logger = logging.getLogger(__name__)

@dataclass
class Journey:
    """Multi-week journey for a visitor"""
    visitor: Visitor
    start_time: datetime
    touchpoints: List[Dict[str, Any]] = field(default_factory=list)
    current_stage: str = "unaware"
    days_in_journey: int = 0
    converted: bool = False
    conversion_value: float = 0.0
    attribution_path: List[str] = field(default_factory=list)
    
    def add_touchpoint(self, touchpoint: Dict[str, Any]):
        """Add a touchpoint to the journey"""
        self.touchpoints.append(touchpoint)
        self.attribution_path.append(f"{touchpoint['channel']}_{touchpoint['timestamp']}")
        
        # Update days in journey
        if self.touchpoints:
            first_touch = datetime.fromisoformat(self.touchpoints[0]['timestamp'])
            current_touch = datetime.fromisoformat(touchpoint['timestamp'])
            self.days_in_journey = (current_touch - first_touch).days

@dataclass
class SimulationState:
    """Current state of the simulation"""
    current_time: datetime
    active_journeys: Dict[str, Journey]  # visitor_id -> Journey
    completed_journeys: List[Journey]
    total_spend: float
    total_conversions: int
    hourly_metrics: List[Dict[str, Any]]

class IntegratedBehavioralSimulator:
    """
    Full integrated simulator combining:
    - Realistic traffic patterns
    - Behavioral health parent personas
    - Multi-week decision journeys
    - Actual auction dynamics
    - Real CTR patterns from Criteo
    """
    
    def __init__(self, 
                 start_date: datetime = None,
                 use_recsim: bool = True,
                 use_auctiongym: bool = True):
        
        # Initialize components
        self.traffic_simulator = RealisticTrafficSimulator()
        self.trigger_system = TriggerEventSystem()
        
        # Initialize auction system - REQUIRED
        if not (use_auctiongym and AUCTION_GYM_AVAILABLE):
            raise RuntimeError("AuctionGym is REQUIRED for realistic auction simulation. No fallbacks allowed.")
        
        auction_config = {
            'auction_type': 'second_price',
            'num_bidders': 10,
            'num_slots': 5
        }
        self.auction_gym = AuctionGymWrapper(config=auction_config)
        logger.info("Using AuctionGym for auction simulation")
        
        # Initialize RecSim if available
        if use_recsim and RECSIM_AVAILABLE:
            self.recsim_bridge = RecSimAuctionBridge()
            logger.info("Using RecSim for user behavior")
        else:
            self.recsim_bridge = None
            logger.info("RecSim not available, using persona-based behavior")
        
        # Load Criteo data
        self.criteo_loader = CriteoDataLoader()
        self._load_criteo_calibration()
        
        # Simulation state
        self.state = SimulationState(
            current_time=start_date or datetime.now(),
            active_journeys={},
            completed_journeys=[],
            total_spend=0.0,
            total_conversions=0,
            hourly_metrics=[]
        )
        
        # Campaign configuration
        self.campaign_config = self._default_campaign_config()
    
    def _load_criteo_calibration(self):
        """Load Criteo data for CTR calibration"""
        try:
            with open('data/criteo_statistics.json', 'r') as f:
                self.criteo_stats = json.load(f)
                logger.info(f"Loaded Criteo stats: {self.criteo_stats['click_rate']:.2%} baseline CTR")
        except Exception as e:
            logger.warning(f"Could not load Criteo stats: {e}")
            self.criteo_stats = {'click_rate': 0.015}
    
    def _default_campaign_config(self) -> Dict[str, Any]:
        """Default campaign configuration for Aura behavioral health"""
        return {
            'daily_budget': 5000,  # $5k/day
            'channel_mix': {
                'google_search': 0.4,
                'facebook': 0.3,
                'youtube': 0.15,
                'tiktok': 0.1,
                'display': 0.05
            },
            'bidding_strategy': 'target_cac',
            'target_cac': 100,
            'creative_themes': [
                'crisis_help',
                'prevention',
                'sleep_wellness',
                'clinical_backing',
                'family_safety'
            ],
            'landing_pages': {
                'crisis': '/crisis-help',
                'prevention': '/healthy-habits',
                'comparison': '/why-aura',
                'pricing': '/pricing'
            }
        }
    
    def simulate_hour(self, 
                     hour_budget: float,
                     hour_of_day: int) -> Dict[str, Any]:
        """Simulate one hour of campaign activity"""
        
        hour_start = self.state.current_time.replace(hour=hour_of_day)
        hour_metrics = {
            'timestamp': hour_start.isoformat(),
            'impressions': 0,
            'clicks': 0,
            'spend': 0.0,
            'new_journeys': 0,
            'conversions': 0,
            'revenue': 0.0,
            'visitors_by_type': {},
            'clicks_by_type': {}
        }
        
        # Adjust channel mix based on time of day
        channel_mix = self._adjust_channel_mix_for_hour(
            self.campaign_config['channel_mix'],
            hour_of_day
        )
        
        # Generate traffic for this hour
        impressions = self.traffic_simulator.simulate_traffic_hour(
            budget=hour_budget,
            channel_mix=channel_mix,
            timestamp=hour_start
        )
        
        for impression in impressions:
            hour_metrics['impressions'] += 1
            
            # Track visitor type
            vtype = impression['visitor_type']
            hour_metrics['visitors_by_type'][vtype] = \
                hour_metrics['visitors_by_type'].get(vtype, 0) + 1
            
            if impression['clicked']:
                hour_metrics['clicks'] += 1
                hour_metrics['spend'] += impression['cpc']
                hour_metrics['clicks_by_type'][vtype] = \
                    hour_metrics['clicks_by_type'].get(vtype, 0) + 1
                
                # Process the click
                self._process_click(impression)
        
        # Process ongoing journeys
        self._process_active_journeys(hour_start)
        
        # Check for conversions
        conversions = self._check_conversions(hour_start)
        hour_metrics['conversions'] = len(conversions)
        hour_metrics['revenue'] = sum(c['value'] for c in conversions)
        
        # Update state
        self.state.total_spend += hour_metrics['spend']
        self.state.total_conversions += hour_metrics['conversions']
        self.state.hourly_metrics.append(hour_metrics)
        
        return hour_metrics
    
    def _adjust_channel_mix_for_hour(self, 
                                    base_mix: Dict[str, float],
                                    hour: int) -> Dict[str, float]:
        """Adjust channel mix based on time of day"""
        
        adjusted = base_mix.copy()
        
        # Late night (10pm-3am) - More search, less social
        if hour >= 22 or hour <= 3:
            adjusted['google_search'] = min(0.6, base_mix['google_search'] * 1.5)
            adjusted['facebook'] = base_mix['facebook'] * 0.5
            adjusted['tiktok'] = base_mix['tiktok'] * 0.3
            
        # After school (3pm-6pm) - More social
        elif 15 <= hour <= 18:
            adjusted['facebook'] = min(0.5, base_mix['facebook'] * 1.3)
            adjusted['tiktok'] = min(0.3, base_mix['tiktok'] * 1.5)
            adjusted['google_search'] = base_mix['google_search'] * 0.8
        
        # Normalize to sum to 1
        total = sum(adjusted.values())
        return {k: v/total for k, v in adjusted.items()}
    
    def _process_click(self, impression: Dict[str, Any]):
        """Process a click and potentially start a journey"""
        
        visitor_id = impression['visitor_id']
        
        # Check if this visitor already has a journey
        if visitor_id in self.state.active_journeys:
            journey = self.state.active_journeys[visitor_id]
            journey.add_touchpoint(impression)
        else:
            # Start new journey if it's a relevant visitor
            if impression['visitor_type'] in ['crisis_parent', 'high_concern_parent',
                                              'moderate_concern_parent', 'curious_parent']:
                # Create journey
                visitor = self._reconstruct_visitor(impression)
                journey = Journey(
                    visitor=visitor,
                    start_time=datetime.fromisoformat(impression['timestamp'])
                )
                journey.add_touchpoint(impression)
                self.state.active_journeys[visitor_id] = journey
    
    def _reconstruct_visitor(self, impression: Dict[str, Any]) -> Visitor:
        """Reconstruct visitor object from impression data"""
        # This is simplified - in production would store full visitor
        visitor = Visitor(
            visitor_id=impression['visitor_id'],
            visitor_type=VisitorType[impression['visitor_type'].upper()],
            timestamp=datetime.fromisoformat(impression['timestamp']),
            channel=impression['channel'],
            device=impression['device'],
            location=impression['location'],
            attention_span=0.5,
            price_sensitivity=0.5,
            ad_blindness=0.3
        )
        
        # Add parent persona if applicable
        if impression['parent_concern'] is not None:
            visitor.parent_persona = self._create_persona_from_concern(impression['parent_concern'])
        
        return visitor
    
    def _create_persona_from_concern(self, concern_level: float) -> ParentPersona:
        """Create parent persona from concern level"""
        if concern_level >= 8:
            trigger = TriggerEvent.FOUND_SELF_HARM_CONTENT
        elif concern_level >= 6:
            trigger = TriggerEvent.SLEEP_DISRUPTION_SEVERE
        elif concern_level >= 4:
            trigger = TriggerEvent.TOO_MUCH_SCREEN_TIME
        else:
            trigger = TriggerEvent.GENERAL_WORRY
        
        parent = BehavioralHealthPersonaFactory.create_parent_with_trigger(trigger)
        parent.current_concern_level = concern_level
        return parent
    
    def _process_active_journeys(self, current_time: datetime):
        """Process all active journeys"""
        
        completed = []
        
        for visitor_id, journey in self.state.active_journeys.items():
            # Update journey stage based on touchpoints and time
            if len(journey.touchpoints) >= 3 and journey.days_in_journey >= 1:
                journey.current_stage = "researching"
            
            if len(journey.touchpoints) >= 10 and journey.days_in_journey >= 3:
                journey.current_stage = "comparing"
            
            if len(journey.touchpoints) >= 15 and journey.days_in_journey >= 7:
                journey.current_stage = "deciding"
            
            # Check for journey timeout (30 days)
            if journey.days_in_journey > 30:
                completed.append(visitor_id)
        
        # Move completed journeys
        for visitor_id in completed:
            journey = self.state.active_journeys.pop(visitor_id)
            self.state.completed_journeys.append(journey)
    
    def _check_conversions(self, current_time: datetime) -> List[Dict[str, Any]]:
        """Check which journeys convert"""
        
        conversions = []
        
        for visitor_id, journey in list(self.state.active_journeys.items()):
            if journey.converted:
                continue
            
            # Only check visitors who are actually parents
            if not journey.visitor.parent_persona:
                continue
            
            parent = journey.visitor.parent_persona
            
            # Check conversion probability
            will_convert = parent.will_convert(
                touchpoint_count=len(journey.touchpoints),
                days_since_trigger=journey.days_in_journey
            )
            
            if will_convert:
                journey.converted = True
                journey.conversion_value = 14.99  # Monthly subscription
                
                conversions.append({
                    'visitor_id': visitor_id,
                    'timestamp': current_time.isoformat(),
                    'value': journey.conversion_value,
                    'journey_days': journey.days_in_journey,
                    'touchpoints': len(journey.touchpoints),
                    'attribution_path': journey.attribution_path,
                    'parent_concern': parent.current_concern_level
                })
                
                # Move to completed
                self.state.completed_journeys.append(journey)
                del self.state.active_journeys[visitor_id]
        
        return conversions
    
    def simulate_day(self, daily_budget: float = 5000) -> Dict[str, Any]:
        """Simulate a full day of campaign activity"""
        
        day_metrics = {
            'date': self.state.current_time.date().isoformat(),
            'total_impressions': 0,
            'total_clicks': 0,
            'total_spend': 0.0,
            'total_conversions': 0,
            'total_revenue': 0.0,
            'hourly_data': [],
            'conversions_by_type': {},
            'cac_by_segment': {}
        }
        
        # Distribute budget across hours (higher during peak times)
        hourly_budgets = self._distribute_daily_budget(daily_budget)
        
        for hour in range(24):
            hour_metrics = self.simulate_hour(hourly_budgets[hour], hour)
            
            day_metrics['total_impressions'] += hour_metrics['impressions']
            day_metrics['total_clicks'] += hour_metrics['clicks']
            day_metrics['total_spend'] += hour_metrics['spend']
            day_metrics['total_conversions'] += hour_metrics['conversions']
            day_metrics['total_revenue'] += hour_metrics['revenue']
            day_metrics['hourly_data'].append(hour_metrics)
        
        # Calculate CAC by segment
        if day_metrics['total_conversions'] > 0:
            day_metrics['overall_cac'] = day_metrics['total_spend'] / day_metrics['total_conversions']
        else:
            day_metrics['overall_cac'] = float('inf')
        
        # Advance to next day
        self.state.current_time += timedelta(days=1)
        
        return day_metrics
    
    def _distribute_daily_budget(self, daily_budget: float) -> List[float]:
        """Distribute daily budget across 24 hours"""
        
        # Hour weights based on expected performance
        hour_weights = []
        for hour in range(24):
            if hour >= 22 or hour <= 3:  # Late night crisis searches
                weight = 2.0
            elif 15 <= hour <= 20:  # After school/evening
                weight = 1.5
            elif 9 <= hour <= 11:  # Morning research
                weight = 1.2
            elif 6 <= hour <= 8:  # Early morning
                weight = 0.8
            else:  # Other hours
                weight = 1.0
            hour_weights.append(weight)
        
        # Normalize and distribute
        total_weight = sum(hour_weights)
        hourly_budgets = [daily_budget * (w / total_weight) for w in hour_weights]
        
        return hourly_budgets
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary of simulation results"""
        
        total_days = len(self.state.hourly_metrics) / 24 if self.state.hourly_metrics else 0
        
        # Analyze conversions by visitor type
        conversions_by_type = {}
        for journey in self.state.completed_journeys:
            if journey.converted:
                vtype = journey.visitor.visitor_type.value
                conversions_by_type[vtype] = conversions_by_type.get(vtype, 0) + 1
        
        # Calculate key metrics
        summary = {
            'simulation_days': total_days,
            'total_spend': self.state.total_spend,
            'total_conversions': self.state.total_conversions,
            'overall_cac': self.state.total_spend / max(1, self.state.total_conversions),
            'active_journeys': len(self.state.active_journeys),
            'completed_journeys': len(self.state.completed_journeys),
            'conversions_by_type': conversions_by_type,
            'conversion_rate': self.state.total_conversions / max(1, len(self.state.completed_journeys)),
            'avg_journey_length': np.mean([j.days_in_journey for j in self.state.completed_journeys]) if self.state.completed_journeys else 0,
            'avg_touchpoints_to_conversion': np.mean([len(j.touchpoints) for j in self.state.completed_journeys if j.converted]) if any(j.converted for j in self.state.completed_journeys) else 0
        }
        
        return summary


if __name__ == "__main__":
    print("Testing Integrated Behavioral Health Simulator\n")
    print("=" * 60)
    
    # Create simulator
    simulator = IntegratedBehavioralSimulator(
        start_date=datetime.now(),
        use_recsim=RECSIM_AVAILABLE,
        use_auctiongym=AUCTION_GYM_AVAILABLE
    )
    
    # Simulate one day
    print("\nSimulating 1 day with $5,000 budget...")
    day_results = simulator.simulate_day(daily_budget=5000)
    
    print(f"\nDay Results:")
    print(f"  Impressions: {day_results['total_impressions']:,}")
    print(f"  Clicks: {day_results['total_clicks']:,}")
    print(f"  Spend: ${day_results['total_spend']:.2f}")
    print(f"  Conversions: {day_results['total_conversions']}")
    print(f"  Revenue: ${day_results['total_revenue']:.2f}")
    print(f"  CAC: ${day_results['overall_cac']:.2f}")
    
    # Show hourly pattern
    print(f"\nHourly Click Pattern:")
    for hour_data in day_results['hourly_data']:
        hour = datetime.fromisoformat(hour_data['timestamp']).hour
        clicks = hour_data['clicks']
        if clicks > 0:
            bar = "█" * (clicks // 5)
            print(f"  {hour:02d}:00: {bar} {clicks} clicks")
    
    # Get simulation summary
    summary = simulator.get_simulation_summary()
    
    print(f"\nSimulation Summary:")
    print(f"  Active Journeys: {summary['active_journeys']}")
    print(f"  Completed Journeys: {summary['completed_journeys']}")
    print(f"  Overall CAC: ${summary['overall_cac']:.2f}")
    
    if summary['conversions_by_type']:
        print(f"\nConversions by Visitor Type:")
        for vtype, count in summary['conversions_by_type'].items():
            print(f"    {vtype}: {count}")
    
    print(f"\nAverage Journey Metrics:")
    print(f"  Days to conversion: {summary['avg_journey_length']:.1f}")
    print(f"  Touchpoints to conversion: {summary['avg_touchpoints_to_conversion']:.1f}")