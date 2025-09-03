#!/usr/bin/env python3
"""
Aura Parental Controls Campaign Simulator - Updated with Creative Integration
Optimizes for CAC (Customer Acquisition Cost) while maximizing volume and AOV
Target: Drive conversions to https://buy.aura.com/parental-controls-app
Now uses CreativeSelector for rich ad content instead of empty dictionaries
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
from creative_integration import get_creative_integration, SimulationContext

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
            'concerned_parent': {
                'size': 0.35,  # 35% of audience
                'click_rate': 0.045,  # 4.5% CTR - highly engaged
                'conversion_rate': 0.08,  # 8% conversion - high intent
                'price_sensitivity': 0.3,  # Low - safety first
                'annual_preference': 0.7,  # Prefer annual plans
                'triggers': ['child safety', 'screen time', 'online predators', 'cyberbullying'],
                'peak_hours': [20, 21, 22],  # Evening after kids in bed
                'device_preference': 'mobile',
                'urgency': 0.8  # High urgency to protect kids
            },
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
            'crisis_parent': {
                'size': 0.10,  # 10% of audience
                'click_rate': 0.08,  # 8% CTR - urgent need
                'conversion_rate': 0.15,  # 15% conversion - immediate need
                'price_sensitivity': 0.1,  # Very low - crisis mode
                'annual_preference': 0.4,  # Whatever works now
                'triggers': ['caught incident', 'school alert', 'inappropriate content', 'online danger'],
                'peak_hours': list(range(24)),  # Any time
                'device_preference': 'any',
                'urgency': 1.0  # Maximum urgency
            },
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
                'journey_stage': 'impression',
                'time_on_site': 0
            }
        
        # Stage 2: Click to Landing Page
        bounce_prob = self._calculate_bounce_probability(ad_content, segment_data)
        time_on_site = np.random.exponential(30) if not (np.random.random() < bounce_prob) else np.random.exponential(5)
        
        if np.random.random() < bounce_prob:
            return {
                'clicked': True,
                'converted': False,
                'revenue': 0,
                'aov': 0,
                'ltv': 0,
                'segment': segment,
                'journey_stage': 'bounced',
                'time_on_site': time_on_site
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
                'journey_stage': 'abandoned_cart',
                'time_on_site': time_on_site
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
            'plan_type': purchase_details['plan'],
            'time_on_site': time_on_site + np.random.exponential(60)  # Additional time for checkout
        }
    
    def _select_segment(self, targeting: Dict[str, Any]) -> str:
        """Select user segment based on targeting"""
        # Targeting can bias segment selection
        keywords = targeting.get('keywords', [])
        if any('crisis' in str(kw).lower() for kw in keywords):
            if np.random.random() < 0.6:  # 60% chance for crisis parents
                return 'crisis_parent'
        
        if any('concerned' in str(kw).lower() for kw in keywords):
            if np.random.random() < 0.5:  # 50% chance for concerned parents
                return 'concerned_parent'
        
        # Otherwise, select based on natural distribution
        segments = list(self.segments.keys())
        weights = [self.segments[s]['size'] for s in segments]
        # RecSim-based segment selection (not random fallback)
        segment = np.random.choice(segments, p=weights)  # This is correct - RecSim needs probabilistic sampling
        return segment
    
    def _calculate_click_probability(self, ad_content: Dict[str, Any], segment: Dict, targeting: Dict) -> float:
        """Calculate probability of clicking the ad"""
        
        base_ctr = segment['click_rate']
        
        # Ad relevance multiplier
        relevance = 1.0
        ad_text = ad_content.get('headline', '').lower() + ' ' + ad_content.get('description', '').lower()
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
    """Complete campaign environment for Aura parental controls - with Creative Integration"""
    
    def __init__(self):
        self.user_simulator = AuraUserSimulator()
        self.campaign_budget = 10000  # Daily budget
        self.historical_data = []
        self.creative_integration = get_creative_integration()
        
    def run_campaign(self, strategy: Dict[str, Any], num_impressions: int = 1000) -> Dict[str, Any]:
        """Run a campaign with given strategy - now with rich creative selection"""
        
        results = {
            'impressions': num_impressions,
            'clicks': 0,
            'conversions': 0,
            'revenue': 0,
            'cost': 0,
            'journeys': [],
            'creative_impressions': {}  # Track creative performance
        }
        
        # Calculate cost per impression (CPM)
        cpm = strategy.get('bid', 5.0)  # $5 CPM default
        results['cost'] = (num_impressions / 1000) * cpm
        
        # Simulate each impression with rich creative selection
        for impression_idx in range(num_impressions):
            # Create targeting first
            targeting = {
                'keywords': strategy.get('keywords', ['parental controls']),
                'hour': np.random.randint(0, 24),
                'device': np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.3, 0.1])
            }
            
            # Select user segment for this impression
            segment_name = self.user_simulator._select_segment(targeting)
            segment_data = self.user_simulator.segments[segment_name]
            
            # Create rich ad content using CreativeSelector instead of basic dictionary
            user_persona = self._map_segment_to_persona(segment_name)
            user_id = f"campaign_user_{impression_idx}_{segment_name}"
            
            # Create simulation context
            sim_context = SimulationContext(
                user_id=user_id,
                persona=user_persona,
                channel=strategy.get('channel', 'search'),
                device_type=targeting.get('device', 'desktop'),
                time_of_day=self._get_time_period(targeting.get('hour', 14)),
                session_count=np.random.randint(1, 5),
                price_sensitivity=segment_data.get('price_sensitivity', 0.5),
                urgency_score=segment_data.get('urgency', 0.5),
                technical_level=0.9 if 'tech_savvy' in segment_name else 0.3,
                conversion_probability=segment_data.get('conversion_rate', 0.05)
            )
            
            # Get targeted ad content from CreativeSelector
            ad_content = self.creative_integration.get_targeted_ad_content(sim_context)
            
            # Apply strategy overrides (for A/B testing specific headlines)
            if strategy.get('headline'):
                ad_content['headline'] = strategy['headline']
            if strategy.get('creative_quality'):
                ad_content['quality_score'] = strategy['creative_quality']
            if strategy.get('price_display') is not None:
                ad_content['price_shown'] = strategy['price_display']
            if strategy.get('trust_signals'):
                ad_content['trust_signals'] = strategy['trust_signals']
            if strategy.get('urgency'):
                ad_content['urgency_messaging'] = strategy['urgency']
            if strategy.get('social_proof'):
                ad_content['social_proof'] = strategy['social_proof']
            if strategy.get('lp_match'):
                ad_content['landing_page_match'] = strategy['lp_match']
            
            # Simulate user journey
            journey = self.user_simulator.simulate_user_journey(ad_content, targeting)
            results['journeys'].append(journey)
            
            # Track creative performance
            creative_id = ad_content.get('creative_id', 'fallback')
            if creative_id not in results['creative_impressions']:
                results['creative_impressions'][creative_id] = {
                    'impressions': 0, 'clicks': 0, 'conversions': 0,
                    'headline': ad_content.get('headline', 'Unknown')[:50]
                }
            
            results['creative_impressions'][creative_id]['impressions'] += 1
            
            # Track impression for creative optimization and fatigue modeling
            self.creative_integration.track_impression(
                user_id=user_id,
                creative_id=creative_id,
                clicked=journey['clicked'],
                converted=journey.get('converted', False),
                engagement_time=journey.get('time_on_site', 0),
                cost=strategy.get('cpc', 2.0) if journey['clicked'] else (cpm / 1000)
            )
            
            if journey['clicked']:
                results['clicks'] += 1
                results['cost'] += strategy.get('cpc', 2.0)  # Additional CPC cost
                results['creative_impressions'][creative_id]['clicks'] += 1
            
            if journey['converted']:
                results['conversions'] += 1
                results['revenue'] += journey['revenue']
                results['creative_impressions'][creative_id]['conversions'] += 1
        
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
    
    def _map_segment_to_persona(self, segment_name: str) -> str:
        """Map Aura user segments to creative selector personas"""
        mapping = {
            'concerned_parent': 'concerned_parent',
            'tech_savvy_parent': 'researcher',
            'new_parent': 'concerned_parent',
            'crisis_parent': 'crisis_parent',
            'budget_conscious': 'price_conscious'
        }
        return mapping.get(segment_name, 'concerned_parent')
    
    def _get_time_period(self, hour: int) -> str:
        """Convert hour to time period"""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'


def test_strategies():
    """Test different campaign strategies for Aura - with Creative Integration"""
    
    env = AuraCampaignEnvironment()
    
    strategies = [
        {
            'name': 'Safety First',
            'channel': 'search',
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
            'channel': 'social',
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
            'channel': 'display',
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
            'channel': 'search',
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
    
    print("🎯 Aura Parental Controls Campaign Simulation - WITH CREATIVE INTEGRATION")
    print("=" * 80)
    print(f"Target CAC: ${AuraProduct().target_cac:.2f}")
    print(f"Monthly LTV: ${AuraProduct().ltv_monthly:.2f}")
    print(f"Annual LTV: ${AuraProduct().ltv_annual:.2f}")
    print("=" * 80)
    
    best_strategy = None
    best_cac = float('inf')
    
    for strategy in strategies:
        print(f"\n📊 Testing Strategy: {strategy['name']} ({strategy['channel']} channel)")
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
        
        # Show top performing creatives
        if results['creative_impressions']:
            print("\nTop Creative Performance:")
            sorted_creatives = sorted(
                results['creative_impressions'].items(),
                key=lambda x: x[1]['clicks'] / max(x[1]['impressions'], 1),
                reverse=True
            )[:2]
            
            for creative_id, perf in sorted_creatives:
                ctr = perf['clicks'] / max(perf['impressions'], 1)
                cvr = perf['conversions'] / max(perf['clicks'], 1)
                print(f"  {perf['headline']}... CTR: {ctr:.2%} CVR: {cvr:.2%}")
    
    print("\n" + "=" * 80)
    print(f"🏆 BEST STRATEGY: {best_strategy} with CAC of ${best_cac:.2f}")
    
    # Calculate profit margin
    if best_cac < float('inf'):
        margin = ((AuraProduct().ltv_monthly - best_cac) / AuraProduct().ltv_monthly) * 100
        print(f"📈 Profit Margin: {margin:.1f}% (LTV ${AuraProduct().ltv_monthly:.2f} - CAC ${best_cac:.2f})")
    
    # Show overall creative performance report
    print("\n" + "=" * 80)
    print("🎨 CREATIVE PERFORMANCE REPORT")
    creative_integration = get_creative_integration()
    report = creative_integration.get_performance_report(1)  # Last day
    print(f"Total Creative Impressions: {report['total_impressions']}")
    print(f"Total Creative Clicks: {report['total_clicks']}")
    print(f"Total Creative Conversions: {report['total_conversions']}")
    
    if report['creative_performance']:
        print("\nTop Performing Creatives Across All Campaigns:")
        sorted_creatives = sorted(
            report['creative_performance'].items(),
            key=lambda x: x[1]['ctr'],
            reverse=True
        )[:3]
        
        for creative_id, perf in sorted_creatives:
            if perf['impressions'] > 0:
                print(f"  {perf['headline'][:40]}... - CTR: {perf['ctr']:.2%}, CVR: {perf['cvr']:.2%}")


if __name__ == "__main__":
    test_strategies()