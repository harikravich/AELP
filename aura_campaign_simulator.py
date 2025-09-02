#!/usr/bin/env python3
"""
Aura Parental Controls Campaign Simulator
Optimizes for CAC (Customer Acquisition Cost) while maximizing volume and AOV
Target: Drive conversions to https://buy.aura.com/parental-controls-app
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
from creative_integration import get_creative_integration, SimulationContext
from dynamic_segment_integration import (
    get_discovered_segments,
    get_segment_conversion_rate,
    get_high_converting_segment,
    get_mobile_segment,
    validate_no_hardcoded_segments
)

@dataclass
class AuraProduct:
    """Aura parental controls product details"""
    name: str = "Aura Parental Controls"
    base_price: float = 14.99  # Monthly subscription
    annual_price: float = 144.00  # Annual subscription (20% discount)
    family_price: float = 19.99  # Family plan
    target_cac: float = 50.00  # Target customer acquisition cost
    ltv_monthly: float = 89.94  # 6-month average lifetime value
    ltv_annual: float = 288.00  # 2-year average lifetime value


class AuraUserSimulator:
    """Simulates user behavior for parental control software buyers"""
    
    def __init__(self):
        self.segments = self._define_parent_segments()
        self.product = AuraProduct()
        
    def _define_parent_segments(self) -> Dict[str, Any]:
        """Define parent segments who buy parental controls"""
        return {
            # Replaced with dynamic segments,
            'tech_savvy_parent': {
                'size': 0.20,  # 20% of audience
                'click_rate': 0.025,  # 2.5% CTR - selective
                'conversion_rate': 0.05,  # 5% conversion
                'price_sensitivity': 0.5,  # Moderate - compares options
                'annual_preference': 0.6,
                'triggers': ['app monitoring', 'location tracking', 'time limits', 'content filtering'],
                'peak_hours': [12, 13, 19, 20],  # Lunch and evening
                'device_preference': 'desktop',
                'urgency': 0.5  # Researches before buying
            },
            'new_parent': {
                'size': 0.15,  # 15% of audience
                'click_rate': 0.06,  # 6% CTR - information seeking
                'conversion_rate': 0.03,  # 3% conversion - just learning
                'price_sensitivity': 0.7,  # High - budget conscious
                'annual_preference': 0.3,  # Prefer monthly trials
                'triggers': ['first phone', 'tween safety', 'parental guidance', 'digital wellness'],
                'peak_hours': [10, 11, 14, 15],  # During baby naps
                'device_preference': 'mobile',
                'urgency': 0.3  # Planning ahead
            },
            # Replaced with dynamic segments,
            'budget_conscious': {
                'size': 0.20,  # 20% of audience
                'click_rate': 0.03,  # 3% CTR
                'conversion_rate': 0.02,  # 2% conversion
                'price_sensitivity': 0.9,  # Very high
                'annual_preference': 0.2,  # Monthly only
                'triggers': ['free trial', 'discount', 'affordable', 'value'],
                'peak_hours': [21, 22, 23],  # Late evening bargain hunting
                'device_preference': 'mobile',
                'urgency': 0.2  # Will wait for deals
            }
        }
    
    def simulate_user_journey(self, ad_content: Dict[str, Any], targeting: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate complete user journey from ad to purchase"""
        
        # Select user segment based on targeting
        segment = self._select_segment(targeting)
        segment_data = self.segments[segment]
        
        # Stage 1: Impression to Click
        click_prob = self._calculate_click_probability(ad_content, segment_data, targeting)
        clicked = np.random.random() < click_prob
        
        if not clicked:
            return {
                'clicked': False,
                'converted': False,
                'revenue': 0,
                'aov': 0,
                'ltv': 0,
                'segment': segment,
                'journey_stage': 'impression'
            }
        
        # Stage 2: Click to Landing Page
        bounce_prob = self._calculate_bounce_probability(ad_content, segment_data)
        if np.random.random() < bounce_prob:
            return {
                'clicked': True,
                'converted': False,
                'revenue': 0,
                'aov': 0,
                'ltv': 0,
                'segment': segment,
                'journey_stage': 'bounced'
            }
        
        # Stage 3: Landing Page to Conversion
        conversion_prob = self._calculate_conversion_probability(ad_content, segment_data, targeting)
        converted = np.random.random() < conversion_prob
        
        if not converted:
            return {
                'clicked': True,
                'converted': False,
                'revenue': 0,
                'aov': 0,
                'ltv': 0,
                'segment': segment,
                'journey_stage': 'abandoned_cart'
            }
        
        # Stage 4: Purchase Decision
        purchase_details = self._calculate_purchase_value(segment_data)
        
        return {
            'clicked': True,
            'converted': True,
            'revenue': purchase_details['revenue'],
            'aov': purchase_details['aov'],
            'ltv': purchase_details['ltv'],
            'segment': segment,
            'journey_stage': 'purchased',
            'plan_type': purchase_details['plan']
        }
    
    def _select_segment(self, targeting: Dict[str, Any]) -> str:
        """Select user segment based on targeting"""
        # Targeting can bias segment selection
        if 'crisis_keywords' in targeting.get('keywords', []):
            if np.random.random() < 0.6:  # 60% chance for crisis parents
                return 'crisis_parent'
        
        if 'concerned_keywords' in targeting.get('keywords', []):
            if np.random.random() < 0.5:  # 50% chance for concerned parents
                return 'concerned_parent'
        
        # Otherwise, select based on natural distribution
        segments = list(self.segments.keys())
        weights = [self.segments[s]['size'] for s in segments]
        return np.random.choice(segments, p=weights)
    
    def _calculate_click_probability(self, ad_content: Dict[str, Any], segment: Dict, targeting: Dict) -> float:
        """Calculate probability of clicking the ad"""
        
        base_ctr = segment['click_rate']
        
        # Ad relevance multiplier
        relevance = 1.0
        ad_text = ad_content.get('headline', '').lower()
        for trigger in segment['triggers']:
            if trigger in ad_text:
                relevance *= 1.3  # 30% boost per matching trigger
        
        # Time of day multiplier
        hour = targeting.get('hour', 12)
        time_multiplier = 1.2 if hour in segment['peak_hours'] else 0.8
        
        # Device match multiplier
        device = targeting.get('device', 'desktop')
        device_multiplier = 1.1 if device == segment['device_preference'] else 0.9
        
        # Urgency boost
        urgency_boost = 1 + (segment['urgency'] * 0.5)
        
        # Creative quality
        creative_quality = ad_content.get('quality_score', 0.7)
        
        final_ctr = base_ctr * relevance * time_multiplier * device_multiplier * urgency_boost * creative_quality
        
        return min(final_ctr, 0.25)  # Cap at 25% CTR
    
    def _calculate_bounce_probability(self, ad_content: Dict, segment: Dict) -> float:
        """Calculate probability of bouncing from landing page"""
        
        # Base bounce rate
        base_bounce = 0.6  # 60% baseline bounce rate
        
        # Message match reduces bounce
        message_match = ad_content.get('landing_page_match', 0.5)
        bounce_reduction = message_match * 0.3
        
        # Urgency reduces bounce
        urgency_reduction = segment['urgency'] * 0.2
        
        return max(0.2, base_bounce - bounce_reduction - urgency_reduction)
    
    def _calculate_conversion_probability(self, ad_content: Dict, segment: Dict, targeting: Dict) -> float:
        """Calculate probability of converting (purchasing)"""
        
        base_conversion = segment['conversion_rate']
        
        # Price shown effect
        price_shown = ad_content.get('price_shown', self.product.base_price)
        if price_shown is None:
            price_shown = self.product.base_price
        price_ratio = price_shown / self.product.base_price
        price_effect = 1 - (segment['price_sensitivity'] * max(0, price_ratio - 1))
        
        # Trust signals
        trust_signals = ad_content.get('trust_signals', 0.5)  # Reviews, badges, etc.
        trust_multiplier = 1 + (trust_signals * 0.5)
        
        # Urgency messaging
        urgency_messaging = ad_content.get('urgency_messaging', 0.3)
        urgency_multiplier = 1 + (urgency_messaging * segment['urgency'] * 0.5)
        
        # Social proof
        social_proof = ad_content.get('social_proof', 0.4)  # "10,000 parents trust Aura"
        social_multiplier = 1 + (social_proof * 0.3)
        
        final_conversion = base_conversion * price_effect * trust_multiplier * urgency_multiplier * social_multiplier
        
        return min(final_conversion, 0.3)  # Cap at 30% conversion
    
    def _calculate_purchase_value(self, segment: Dict) -> Dict[str, Any]:
        """Calculate purchase value and LTV"""
        
        # Determine plan type
        if np.random.random() < segment['annual_preference']:
            plan = 'annual'
            revenue = self.product.annual_price
            ltv = self.product.ltv_annual
        else:
            plan = 'monthly'
            revenue = self.product.base_price
            ltv = self.product.ltv_monthly
        
        # Family plan upsell (20% chance for concerned parents)
        if segment == self.segments.get('concerned_parent') and np.random.random() < 0.2:
            plan = 'family'
            revenue = self.product.family_price * 12  # Annual family
            ltv = revenue * 2  # 2-year retention
        
        return {
            'plan': plan,
            'revenue': revenue,
            'aov': revenue,
            'ltv': ltv
        }


class AuraCampaignEnvironment:
    """Complete campaign environment for Aura parental controls"""
    
    def __init__(self):
        self.user_simulator = AuraUserSimulator()
        self.campaign_budget = 10000  # Daily budget
        self.historical_data = []
        self.creative_integration = get_creative_integration()
        
    def run_campaign(self, strategy: Dict[str, Any], num_impressions: int = 1000) -> Dict[str, Any]:
        """Run a campaign with given strategy"""
        
        results = {
            'impressions': num_impressions,
            'clicks': 0,
            'conversions': 0,
            'revenue': 0,
            'cost': 0,
            'journeys': []
        }
        
        # Calculate cost per impression (CPM)
        cpm = strategy.get('bid', 5.0)  # $5 CPM default
        results['cost'] = (num_impressions / 1000) * cpm
        
        # Simulate each impression
        for _ in range(num_impressions):
            # Create ad content based on strategy
            ad_content = {
                'headline': strategy.get('headline', 'Keep Your Kids Safe Online'),
                'quality_score': strategy.get('creative_quality', 0.7),
                'price_shown': strategy.get('price_display', 14.99),
                'trust_signals': strategy.get('trust_signals', 0.5),
                'urgency_messaging': strategy.get('urgency', 0.3),
                'social_proof': strategy.get('social_proof', 0.4),
                'landing_page_match': strategy.get('lp_match', 0.6)
            }
            
            # Create targeting
            targeting = {
                'keywords': strategy.get('keywords', ['parental controls']),
                'hour': np.random.randint(0, 24),
                'device': np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.3, 0.1])
            }
            
            # Simulate user journey
            journey = self.user_simulator.simulate_user_journey(ad_content, targeting)
            results['journeys'].append(journey)
            
            if journey['clicked']:
                results['clicks'] += 1
                results['cost'] += strategy.get('cpc', 2.0)  # Additional CPC cost
            
            if journey['converted']:
                results['conversions'] += 1
                results['revenue'] += journey['revenue']
        
        # Calculate key metrics
        results['ctr'] = results['clicks'] / results['impressions'] if results['impressions'] > 0 else 0
        results['conversion_rate'] = results['conversions'] / results['clicks'] if results['clicks'] > 0 else 0
        results['cac'] = results['cost'] / results['conversions'] if results['conversions'] > 0 else float('inf')
        results['roas'] = results['revenue'] / results['cost'] if results['cost'] > 0 else 0
        results['aov'] = results['revenue'] / results['conversions'] if results['conversions'] > 0 else 0
        
        # Calculate segment distribution
        segments = [j['segment'] for j in results['journeys']]
        results['segment_distribution'] = {s: segments.count(s)/len(segments) for s in set(segments)}
        
        return results


def test_strategies():
    """Test different campaign strategies for Aura"""
    
    env = AuraCampaignEnvironment()
    
    strategies = [
        {
            'name': 'Safety First',
            'headline': 'Protect Your Kids from Online Predators - Real-Time Alerts',
            'keywords': ['crisis_keywords', 'concerned_keywords'],
            'bid': 8.0,
            'cpc': 3.0,
            'creative_quality': 0.9,
            'trust_signals': 0.8,
            'urgency': 0.7,
            'social_proof': 0.6,
            'price_display': None  # Don't show price
        },
        {
            'name': 'Screen Time Focus',
            'headline': 'Help Your Kids Find Balance Online - Screen Time Controls',
            'keywords': ['screen time', 'digital wellness', 'balance'],
            'bid': 5.0,
            'cpc': 2.0,
            'creative_quality': 0.8,
            'trust_signals': 0.6,
            'urgency': 0.3,
            'social_proof': 0.7,
            'price_display': 14.99
        },
        {
            'name': 'Value Proposition',
            'headline': 'Complete Parental Controls - Only $14.99/month',
            'keywords': ['affordable', 'value', 'discount'],
            'bid': 3.0,
            'cpc': 1.5,
            'creative_quality': 0.7,
            'trust_signals': 0.5,
            'urgency': 0.2,
            'social_proof': 0.5,
            'price_display': 14.99
        },
        {
            'name': 'Crisis Response',
            'headline': 'Caught Something Inappropriate? Block It Now with Aura',
            'keywords': ['crisis_keywords', 'inappropriate content', 'block now'],
            'bid': 10.0,
            'cpc': 4.0,
            'creative_quality': 0.85,
            'trust_signals': 0.7,
            'urgency': 0.9,
            'social_proof': 0.4,
            'price_display': None
        }
    ]
    
    print("🎯 Aura Parental Controls Campaign Simulation")
    print("=" * 80)
    print(f"Target CAC: ${AuraProduct().target_cac:.2f}")
    print(f"Monthly LTV: ${AuraProduct().ltv_monthly:.2f}")
    print(f"Annual LTV: ${AuraProduct().ltv_annual:.2f}")
    print("=" * 80)
    
    best_strategy = None
    best_cac = float('inf')
    
    for strategy in strategies:
        print(f"\n📊 Testing Strategy: {strategy['name']}")
        print("-" * 40)
        
        results = env.run_campaign(strategy, num_impressions=10000)
        
        print(f"Impressions: {results['impressions']:,}")
        print(f"Clicks: {results['clicks']:,} (CTR: {results['ctr']:.2%})")
        print(f"Conversions: {results['conversions']} (CR: {results['conversion_rate']:.2%})")
        print(f"Revenue: ${results['revenue']:.2f}")
        print(f"Cost: ${results['cost']:.2f}")
        print(f"CAC: ${results['cac']:.2f}")
        print(f"ROAS: {results['roas']:.2f}x")
        print(f"AOV: ${results['aov']:.2f}")
        
        if results['cac'] < best_cac and results['conversions'] > 0:
            best_cac = results['cac']
            best_strategy = strategy['name']
        
        # Show segment distribution
        if results['segment_distribution']:
            print("\nAudience Segments Reached:")
            for segment, pct in sorted(results['segment_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  {segment}: {pct:.1%}")
    
    print("\n" + "=" * 80)
    print(f"🏆 BEST STRATEGY: {best_strategy} with CAC of ${best_cac:.2f}")
    
    # Calculate profit margin
    if best_cac < float('inf'):
        margin = ((AuraProduct().ltv_monthly - best_cac) / AuraProduct().ltv_monthly) * 100
        print(f"📈 Profit Margin: {margin:.1f}% (LTV ${AuraProduct().ltv_monthly:.2f} - CAC ${best_cac:.2f})")


if __name__ == "__main__":
    test_strategies()