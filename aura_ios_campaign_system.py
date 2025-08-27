#!/usr/bin/env python3
"""
Aura Balance iOS-Only Campaign System
CRITICAL: Since Aura Balance ONLY works on iPhone, all campaigns target iOS exclusively
Position iOS-only as PREMIUM feature - no Android waste
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from creative_integration import get_creative_integration, SimulationContext
from creative_selector import JourneyStage, CreativeType

# Import iOS targeting components
from ios_targeting_system import (
    iOSTargetingEngine, iOSCampaignConfig, Platform, iOSAudience, 
    iOSDevice, iOSUserProfile, iOSCreativeGenerator
)

# Import Aura campaign components
from aura_campaign_simulator_updated import AuraProduct, AuraCampaignEnvironment


@dataclass
class iOSAuraProduct(AuraProduct):
    """Aura Balance - iOS ONLY product details"""
    name: str = "Aura Balance"
    tagline: str = "Behavioral Health Monitoring for iPhone Families"
    ios_exclusive: bool = True
    app_store_url: str = "https://apps.apple.com/app/aura-balance"
    compatibility: str = "Requires iOS 14.0+ and iPhone"
    
    # Premium positioning due to iOS exclusivity
    premium_multiplier: float = 1.3  # 30% premium for iOS-only features
    
    def get_premium_price(self) -> float:
        """Get premium price for iOS users"""
        return self.base_price * self.premium_multiplier


class iOSAuraUserSimulator:
    """iOS-specific user simulator for Aura Balance"""
    
    def __init__(self):
        self.ios_product = iOSAuraProduct()
        self.ios_segments = self._define_ios_parent_segments()
        
    def _define_ios_parent_segments(self) -> Dict[str, Any]:
        """Define iOS parent segments - premium users only"""
        return {
            'premium_ios_parents': {
                'size': 0.40,  # 40% - largest iOS segment
                'income': '$100k+',
                'devices': ['iPhone 13+', 'iPad Pro', 'Apple Watch'],
                'click_rate': 0.055,  # Higher than Android users
                'conversion_rate': 0.12,  # Premium users convert better
                'price_sensitivity': 0.2,  # Low - they pay for quality
                'annual_preference': 0.8,  # Prefer annual plans
                'apple_ecosystem': True,
                'family_sharing': True,
                'screen_time_active': True,
                'triggers': [
                    'teen behavioral health', 'mental wellness', 'depression detection',
                    'mood tracking', 'social media impact', 'digital wellness'
                ],
                'peak_hours': [19, 20, 21, 22],  # Evening concern time
                'urgency': 0.85  # High concern for teen mental health
            },
            
            'ios_crisis_parents': {
                'size': 0.15,  # 15% - urgent need segment
                'income': '$75k+',
                'devices': ['iPhone 12+', 'iPad'],
                'click_rate': 0.095,  # Very high engagement
                'conversion_rate': 0.20,  # Crisis converts immediately
                'price_sensitivity': 0.1,  # Price irrelevant in crisis
                'annual_preference': 0.5,  # Whatever works now
                'apple_ecosystem': True,
                'family_sharing': True,
                'screen_time_active': True,
                'triggers': [
                    'teen depression', 'suicide ideation', 'self harm',
                    'behavioral crisis', 'mental health emergency', 'intervention needed'
                ],
                'peak_hours': list(range(24)),  # Any time - crisis mode
                'urgency': 1.0  # Maximum urgency
            },
            
            'ios_researchers': {
                'size': 0.20,  # 20% - tech-savvy iOS users
                'income': '$125k+',
                'devices': ['iPhone 14 Pro', 'iPad Pro', 'MacBook', 'Apple Watch'],
                'click_rate': 0.030,  # Selective but engaged
                'conversion_rate': 0.06,  # Research before buying
                'price_sensitivity': 0.4,  # Moderate - want value
                'annual_preference': 0.7,
                'apple_ecosystem': True,
                'family_sharing': True,
                'screen_time_active': True,
                'triggers': [
                    'behavioral analytics', 'AI monitoring', 'data privacy',
                    'algorithm transparency', 'research-backed', 'clinical validation'
                ],
                'peak_hours': [12, 13, 18, 19],  # Lunch and early evening
                'urgency': 0.4  # Methodical approach
            },
            
            'ios_screen_time_upgraders': {
                'size': 0.20,  # 20% - existing Screen Time users
                'income': '$85k+',
                'devices': ['iPhone 12+', 'iPad'],
                'click_rate': 0.070,  # High - they know they need more
                'conversion_rate': 0.15,  # Ready to upgrade
                'price_sensitivity': 0.3,  # Low - paying for better features
                'annual_preference': 0.6,
                'apple_ecosystem': True,
                'family_sharing': True,
                'screen_time_active': True,
                'triggers': [
                    'beyond screen time', 'screen time alternative', 'advanced parental controls',
                    'mood insights', 'behavioral patterns', 'ai-powered monitoring'
                ],
                'peak_hours': [20, 21, 22],  # Evening review of kid's usage
                'urgency': 0.7  # Know current solution isn't enough
            },
            
            'ios_price_conscious': {
                'size': 0.05,  # 5% - smallest segment for iOS
                'income': '$60k+',  # Still higher than general population
                'devices': ['iPhone 11', 'older iPad'],
                'click_rate': 0.040,
                'conversion_rate': 0.04,  # Lowest conversion
                'price_sensitivity': 0.7,  # High for iOS segment
                'annual_preference': 0.3,  # Monthly preferred
                'apple_ecosystem': False,  # Limited Apple services
                'family_sharing': True,
                'screen_time_active': False,
                'triggers': [
                    'affordable ios', 'budget friendly', 'free trial',
                    'parental controls deal', 'ios family plan'
                ],
                'peak_hours': [21, 22, 23],  # Late night deal hunting
                'urgency': 0.3  # Will wait for better price
            }
        }
    
    def simulate_ios_user_journey(self, ad_content: Dict[str, Any], 
                                 targeting: Dict[str, Any],
                                 ios_profile: Optional[iOSUserProfile] = None) -> Dict[str, Any]:
        """Simulate iOS user journey - premium audience only"""
        
        # Ensure we're targeting iOS users only
        if not self._verify_ios_targeting(targeting):
            raise ValueError("NON-iOS TARGETING DETECTED - BLOCKED")
        
        # Select iOS segment
        ios_segment = self._select_ios_segment(targeting, ios_profile)
        segment_data = self.ios_segments[ios_segment]
        
        # Generate iOS user profile if not provided
        if not ios_profile:
            ios_profile = self._generate_ios_profile(ios_segment)
        
        # Stage 1: iOS-specific impression to click
        click_prob = self._calculate_ios_click_probability(
            ad_content, segment_data, targeting, ios_profile
        )
        clicked = np.random.random() < click_prob
        
        if not clicked:
            return {
                'clicked': False,
                'converted': False,
                'revenue': 0,
                'aov': 0,
                'ltv': 0,
                'segment': ios_segment,
                'journey_stage': 'impression',
                'ios_profile': asdict(ios_profile),
                'android_excluded': True
            }
        
        # Stage 2: App Store bounce check
        app_store_bounce_prob = self._calculate_app_store_bounce_probability(
            ad_content, segment_data, ios_profile
        )
        
        if np.random.random() < app_store_bounce_prob:
            return {
                'clicked': True,
                'converted': False,
                'revenue': 0,
                'aov': 0,
                'ltv': 0,
                'segment': ios_segment,
                'journey_stage': 'app_store_bounce',
                'ios_profile': asdict(ios_profile),
                'android_excluded': True
            }
        
        # Stage 3: iOS conversion probability
        conversion_prob = self._calculate_ios_conversion_probability(
            ad_content, segment_data, targeting, ios_profile
        )
        converted = np.random.random() < conversion_prob
        
        if not converted:
            return {
                'clicked': True,
                'converted': False,
                'revenue': 0,
                'aov': 0,
                'ltv': 0,
                'segment': ios_segment,
                'journey_stage': 'ios_abandoned',
                'ios_profile': asdict(ios_profile),
                'android_excluded': True
            }
        
        # Stage 4: iOS premium purchase
        purchase_details = self._calculate_ios_purchase_value(segment_data, ios_profile)
        
        return {
            'clicked': True,
            'converted': True,
            'revenue': purchase_details['revenue'],
            'aov': purchase_details['aov'],
            'ltv': purchase_details['ltv'],
            'segment': ios_segment,
            'journey_stage': 'ios_purchased',
            'plan_type': purchase_details['plan'],
            'ios_profile': asdict(ios_profile),
            'android_excluded': True,
            'apple_pay_used': purchase_details.get('apple_pay', False)
        }
    
    def _verify_ios_targeting(self, targeting: Dict[str, Any]) -> bool:
        """Verify targeting is iOS-only - CRITICAL CHECK"""
        
        # Check device targeting
        device = targeting.get('device', '').lower()
        if 'android' in device or 'samsung' in device:
            return False
        
        # Check keywords for Android terms
        keywords = targeting.get('keywords', [])
        android_terms = ['android', 'samsung', 'google pixel', 'family link']
        
        for keyword in keywords:
            keyword_str = str(keyword).lower()
            if any(term in keyword_str for term in android_terms):
                return False
        
        # Check for iOS-positive signals
        ios_signals = ['ios', 'iphone', 'ipad', 'apple', 'screen time']
        has_ios_signal = any(
            signal in str(targeting).lower() for signal in ios_signals
        )
        
        return has_ios_signal or targeting.get('platform') == 'ios'
    
    def _select_ios_segment(self, targeting: Dict[str, Any], 
                           ios_profile: Optional[iOSUserProfile]) -> str:
        """Select iOS-specific segment"""
        
        if ios_profile and ios_profile.teen_in_household:
            # Crisis keywords with teen = crisis parent
            keywords = [str(k).lower() for k in targeting.get('keywords', [])]
            crisis_terms = ['crisis', 'depression', 'suicide', 'emergency', 'intervention']
            if any(term in ' '.join(keywords) for term in crisis_terms):
                return 'ios_crisis_parents'
        
        # Screen Time upgrade targeting
        if any('screen time' in str(k).lower() for k in targeting.get('keywords', [])):
            if np.random.random() < 0.7:  # 70% chance
                return 'ios_screen_time_upgraders'
        
        # Premium targeting
        if targeting.get('premium', False) or targeting.get('household_income') == '$100k+':
            return 'premium_ios_parents'
        
        # Research-focused
        research_terms = ['research', 'clinical', 'data', 'algorithm', 'ai']
        if any(term in str(targeting).lower() for term in research_terms):
            return 'ios_researchers'
        
        # Natural segment distribution
        segments = list(self.ios_segments.keys())
        weights = [self.ios_segments[s]['size'] for s in segments]
        return np.random.choice(segments, p=weights)
    
    def _generate_ios_profile(self, segment: str) -> iOSUserProfile:
        """Generate realistic iOS user profile"""
        
        segment_data = self.ios_segments[segment]
        
        # Select device based on segment
        device_options = {
            'premium_ios_parents': [iOSDevice.IPHONE_14_PLUS, iOSDevice.IPHONE_13],
            'ios_crisis_parents': [iOSDevice.IPHONE_13, iOSDevice.IPHONE_12],
            'ios_researchers': [iOSDevice.IPHONE_14_PLUS],
            'ios_screen_time_upgraders': [iOSDevice.IPHONE_13, iOSDevice.IPHONE_12],
            'ios_price_conscious': [iOSDevice.IPHONE_12, iOSDevice.IPHONE_11]
        }
        
        device = np.random.choice(device_options.get(segment, [iOSDevice.IPHONE_13]))
        
        # Generate profile
        profile = iOSUserProfile(
            user_id=f"{segment}_{np.random.randint(10000, 99999)}",
            device_model=device,
            ios_version="16.0" if segment in ['premium_ios_parents', 'ios_researchers'] else "15.0",
            household_income=segment_data['income'],
            family_sharing_enabled=segment_data['family_sharing'],
            screen_time_active=segment_data['screen_time_active'],
            apple_services=["iCloud+", "Apple One"] if segment_data['apple_ecosystem'] else [],
            app_store_spending="high" if 'premium' in segment else "medium",
            privacy_focused=True,  # iOS users care about privacy
            teen_in_household=True  # All Aura users have teens
        )
        
        return profile
    
    def _calculate_ios_click_probability(self, ad_content: Dict, segment: Dict,
                                        targeting: Dict, ios_profile: iOSUserProfile) -> float:
        """Calculate iOS-specific click probability - higher than Android"""
        
        base_ctr = segment['click_rate']
        
        # iOS premium user engagement boost
        ios_boost = 1.2 if ios_profile.get_premium_score() > 0.7 else 1.0
        
        # Apple ecosystem integration messaging boost
        apple_messaging_boost = 1.3 if any(
            term in ad_content.get('headline', '').lower() 
            for term in ['apple', 'iphone', 'ios', 'screen time']
        ) else 1.0
        
        # Family Sharing targeting boost
        family_boost = 1.15 if ios_profile.family_sharing_enabled else 1.0
        
        # Device match (always matches for iOS targeting)
        device_match = 1.1
        
        # Time relevance
        hour = targeting.get('hour', 12)
        time_multiplier = 1.2 if hour in segment['peak_hours'] else 0.9
        
        # Urgency multiplier
        urgency_multiplier = 1 + (segment['urgency'] * 0.4)
        
        final_ctr = (base_ctr * ios_boost * apple_messaging_boost * 
                    family_boost * device_match * time_multiplier * urgency_multiplier)
        
        return min(final_ctr, 0.30)  # Cap at 30%
    
    def _calculate_app_store_bounce_probability(self, ad_content: Dict, 
                                               segment: Dict, ios_profile: iOSUserProfile) -> float:
        """Calculate App Store bounce probability - lower for iOS users"""
        
        # Base bounce rate - lower than web because App Store is trusted
        base_bounce = 0.25  # 25% for iOS (vs 60% for web)
        
        # Premium users bounce less
        premium_reduction = 0.1 if ios_profile.get_premium_score() > 0.8 else 0.0
        
        # Urgency reduces bounce significantly
        urgency_reduction = segment['urgency'] * 0.15
        
        # Apple ecosystem users trust App Store more
        ecosystem_reduction = 0.05 if len(ios_profile.apple_services) > 1 else 0.0
        
        final_bounce = max(0.10, base_bounce - premium_reduction - urgency_reduction - ecosystem_reduction)
        
        return final_bounce
    
    def _calculate_ios_conversion_probability(self, ad_content: Dict, segment: Dict,
                                            targeting: Dict, ios_profile: iOSUserProfile) -> float:
        """Calculate iOS conversion probability - premium users convert better"""
        
        base_conversion = segment['conversion_rate']
        
        # Premium users convert better
        premium_multiplier = 1.0 + (ios_profile.get_premium_score() * 0.5)
        
        # Apple Pay reduces friction
        apple_pay_boost = 1.3 if ios_profile.app_store_spending == "high" else 1.1
        
        # Price sensitivity effect (using premium pricing)
        premium_price = self.ios_product.get_premium_price()
        price_ratio = premium_price / self.ios_product.base_price
        price_effect = 1 - (segment['price_sensitivity'] * (price_ratio - 1))
        
        # iOS-specific trust factors
        ios_trust_multiplier = 1.2  # iOS users trust App Store more
        
        # Urgency multiplier
        urgency_multiplier = 1 + (segment['urgency'] * 0.6)
        
        final_conversion = (base_conversion * premium_multiplier * apple_pay_boost * 
                           price_effect * ios_trust_multiplier * urgency_multiplier)
        
        return min(final_conversion, 0.35)  # Cap at 35%
    
    def _calculate_ios_purchase_value(self, segment: Dict, ios_profile: iOSUserProfile) -> Dict[str, Any]:
        """Calculate iOS purchase value - premium pricing"""
        
        # Determine plan preference
        annual_prob = segment['annual_preference'] * (1 + ios_profile.get_premium_score() * 0.2)
        
        if np.random.random() < annual_prob:
            plan = 'annual'
            revenue = self.ios_product.annual_price * self.ios_product.premium_multiplier
            ltv = self.ios_product.ltv_annual * 1.4  # iOS users have higher retention
        else:
            plan = 'monthly'
            revenue = self.ios_product.get_premium_price()
            ltv = self.ios_product.ltv_monthly * 1.3
        
        # Family plan upsell (higher for premium iOS users)
        family_upsell_prob = 0.3 if ios_profile.get_premium_score() > 0.8 else 0.15
        if np.random.random() < family_upsell_prob:
            plan = 'family_annual'
            revenue = self.ios_product.family_price * 12 * self.ios_product.premium_multiplier
            ltv = revenue * 2.5  # Family plans have highest retention
        
        # Apple Pay usage
        apple_pay_used = ios_profile.app_store_spending in ['high', 'medium']
        
        return {
            'plan': plan,
            'revenue': revenue,
            'aov': revenue,
            'ltv': ltv,
            'apple_pay': apple_pay_used,
            'premium_pricing': True
        }


class iOSAuraCampaignEnvironment:
    """Complete iOS-only campaign environment for Aura Balance"""
    
    def __init__(self):
        self.ios_user_simulator = iOSAuraUserSimulator()
        self.ios_targeting_engine = iOSTargetingEngine()
        self.creative_integration = get_creative_integration()
        
        # Track iOS-specific metrics
        self.ios_metrics = {
            'android_requests_blocked': 0,
            'ios_premium_conversions': 0,
            'app_store_installs': 0,
            'apple_pay_transactions': 0,
            'family_sharing_activations': 0,
            'screen_time_integrations': 0
        }
    
    def run_ios_campaign(self, ios_config: iOSCampaignConfig, 
                        num_impressions: int = 1000) -> Dict[str, Any]:
        """Run iOS-only campaign - NO ANDROID TRAFFIC"""
        
        # Verify iOS-only configuration
        if not ios_config.exclude_android:
            raise ValueError("CRITICAL: iOS campaigns MUST exclude Android")
        
        results = {
            'campaign_config': asdict(ios_config),
            'impressions': num_impressions,
            'clicks': 0,
            'conversions': 0,
            'revenue': 0,
            'cost': 0,
            'ios_journeys': [],
            'android_blocked': 0,  # Track blocked Android attempts
            'ios_specific_metrics': {}
        }
        
        # Calculate premium iOS CPM
        base_cpm = 6.0  # Higher for iOS
        premium_multiplier = ios_config.budget_multiplier
        ios_cpm = base_cpm * premium_multiplier
        
        results['cost'] = (num_impressions / 1000) * ios_cpm
        
        # Simulate each impression - iOS ONLY
        for impression_idx in range(num_impressions):
            
            # Create iOS-specific targeting
            ios_targeting = self._create_ios_targeting(ios_config, impression_idx)
            
            # Verify no Android targeting slipped through
            if not self.ios_user_simulator._verify_ios_targeting(ios_targeting):
                self.ios_metrics['android_requests_blocked'] += 1
                results['android_blocked'] += 1
                continue  # Skip this impression - Android blocked
            
            # Generate iOS user profile
            segment = self.ios_user_simulator._select_ios_segment(ios_targeting, None)
            ios_profile = self.ios_user_simulator._generate_ios_profile(segment)
            
            # Get iOS-specific creative
            ios_creative = self._get_ios_creative(ios_config, ios_profile, segment)
            
            # Simulate iOS user journey
            try:
                journey = self.ios_user_simulator.simulate_ios_user_journey(
                    ios_creative, ios_targeting, ios_profile
                )
                
                results['ios_journeys'].append(journey)
                
                # Track results
                if journey['clicked']:
                    results['clicks'] += 1
                    # iOS CPC is higher due to premium audience
                    ios_cpc = 2.5 * ios_config.budget_multiplier
                    results['cost'] += ios_cpc
                
                if journey['converted']:
                    results['conversions'] += 1
                    results['revenue'] += journey['revenue']
                    
                    # Track iOS-specific metrics
                    self.ios_metrics['ios_premium_conversions'] += 1
                    if journey.get('apple_pay_used'):
                        self.ios_metrics['apple_pay_transactions'] += 1
                    
                    if ios_profile.family_sharing_enabled:
                        self.ios_metrics['family_sharing_activations'] += 1
                    
                    if ios_profile.screen_time_active:
                        self.ios_metrics['screen_time_integrations'] += 1
                        
            except ValueError as e:
                # Android targeting detected - block it
                if "NON-iOS TARGETING" in str(e):
                    self.ios_metrics['android_requests_blocked'] += 1
                    results['android_blocked'] += 1
                    continue
                else:
                    raise e
        
        # Calculate iOS-specific metrics
        results['ctr'] = results['clicks'] / results['impressions'] if results['impressions'] > 0 else 0
        results['conversion_rate'] = results['conversions'] / results['clicks'] if results['clicks'] > 0 else 0
        results['cac'] = results['cost'] / results['conversions'] if results['conversions'] > 0 else float('inf')
        results['roas'] = results['revenue'] / results['cost'] if results['cost'] > 0 else 0
        results['aov'] = results['revenue'] / results['conversions'] if results['conversions'] > 0 else 0
        
        # iOS-specific metrics
        results['ios_specific_metrics'] = {
            'premium_conversion_rate': self.ios_metrics['ios_premium_conversions'] / max(results['conversions'], 1),
            'apple_pay_adoption': self.ios_metrics['apple_pay_transactions'] / max(results['conversions'], 1),
            'family_sharing_rate': self.ios_metrics['family_sharing_activations'] / max(results['conversions'], 1),
            'screen_time_integration_rate': self.ios_metrics['screen_time_integrations'] / max(results['conversions'], 1),
            'android_waste_prevented': results['android_blocked'] * (ios_cpm / 1000),
            'ios_purity': 1.0 - (results['android_blocked'] / max(num_impressions, 1))
        }
        
        return results
    
    def _create_ios_targeting(self, ios_config: iOSCampaignConfig, impression_idx: int) -> Dict[str, Any]:
        """Create iOS-specific targeting for each impression"""
        
        return {
            'platform': 'ios',
            'device': np.random.choice(['iPhone 14', 'iPhone 13', 'iPhone 12', 'iPad Pro'], p=[0.3, 0.3, 0.3, 0.1]),
            'ios_version': ios_config.ios_version_min,
            'keywords': ios_config.apple_specific_keywords,
            'excluded_keywords': ios_config.negative_keywords,
            'hour': np.random.randint(0, 24),
            'audience': ios_config.audience.value,
            'premium': True,
            'family_sharing': np.random.choice([True, False], p=[0.7, 0.3]),
            'household_income': np.random.choice(['$75k+', '$100k+', '$150k+'], p=[0.3, 0.4, 0.3])
        }
    
    def _get_ios_creative(self, ios_config: iOSCampaignConfig, 
                         ios_profile: iOSUserProfile, segment: str) -> Dict[str, Any]:
        """Get iOS-specific creative content"""
        
        # Generate iOS creative using targeting engine
        creative = self.ios_targeting_engine.creative_generator.generate_ios_creative(
            audience=ios_config.audience,
            journey_stage=JourneyStage.AWARENESS,
            creative_type=CreativeType.HERO_IMAGE
        )
        
        # Convert to ad content format
        return {
            'creative_id': creative.id,
            'headline': creative.headline,
            'description': creative.description,
            'cta': creative.cta,
            'image_url': creative.image_url,
            'landing_page': creative.landing_page.value,
            'quality_score': 0.9,  # iOS creatives are high quality
            'trust_signals': 0.8,  # High trust on iOS/App Store
            'urgency_messaging': 0.7 if 'crisis' in segment else 0.4,
            'social_proof': 0.6,  # "Join 50,000+ iPhone families"
            'landing_page_match': 0.8,  # iOS landing pages match well
            'ios_branding': True,
            'app_store_focused': True,
            'premium_positioning': True
        }


def run_comprehensive_ios_campaign_test():
    """Run comprehensive iOS campaign test across all audiences"""
    
    print("🍎 AURA BALANCE iOS-ONLY CAMPAIGN SYSTEM")
    print("=" * 80)
    print("Since Aura Balance ONLY works on iPhone - targeting iOS exclusively")  
    print("Positioning iOS-only as PREMIUM feature")
    print("NO ANDROID TRAFFIC - ZERO WASTE")
    print("=" * 80)
    
    # Initialize iOS campaign environment
    ios_env = iOSAuraCampaignEnvironment()
    
    # Test all iOS audience types
    ios_test_configs = [
        {
            'name': 'Premium iPhone Families',
            'config': iOSCampaignConfig(
                campaign_name="premium_iphone_families",
                platform=Platform.GOOGLE_ADS,
                audience=iOSAudience.PREMIUM_IPHONE_FAMILIES,
                budget_multiplier=1.5  # Premium targeting
            ),
            'impressions': 20000
        },
        {
            'name': 'Screen Time Upgraders', 
            'config': iOSCampaignConfig(
                campaign_name="screen_time_upgraders",
                platform=Platform.APPLE_SEARCH_ADS,
                audience=iOSAudience.SCREEN_TIME_UPGRADERS,
                budget_multiplier=1.6  # Highest intent
            ),
            'impressions': 15000
        },
        {
            'name': 'iOS Crisis Parents',
            'config': iOSCampaignConfig(
                campaign_name="ios_crisis_parents",
                platform=Platform.FACEBOOK_ADS,
                audience=iOSAudience.IOS_PARENTS_TEENS,
                budget_multiplier=1.4  # Urgent need
            ),
            'impressions': 10000
        },
        {
            'name': 'Apple Ecosystem Users',
            'config': iOSCampaignConfig(
                campaign_name="apple_ecosystem",
                platform=Platform.TIKTOK_ADS,
                audience=iOSAudience.APPLE_ECOSYSTEM_USERS,
                budget_multiplier=1.3
            ),
            'impressions': 12000
        }
    ]
    
    best_campaign = None
    best_roas = 0
    total_android_blocked = 0
    total_revenue = 0
    total_cost = 0
    
    print("\n🎯 TESTING iOS CAMPAIGNS")
    print("-" * 50)
    
    for test_config in ios_test_configs:
        print(f"\n📱 {test_config['name'].upper()}")
        print(f"Platform: {test_config['config'].platform.value}")
        print(f"Audience: {test_config['config'].audience.value}")
        
        try:
            results = ios_env.run_ios_campaign(
                test_config['config'], 
                test_config['impressions']
            )
            
            # Display results
            print(f"  Impressions: {results['impressions']:,}")
            print(f"  Clicks: {results['clicks']:,} (CTR: {results['ctr']:.2%})")  
            print(f"  Conversions: {results['conversions']} (CVR: {results['conversion_rate']:.2%})")
            print(f"  Revenue: ${results['revenue']:,.2f}")
            print(f"  Cost: ${results['cost']:,.2f}")
            print(f"  CAC: ${results['cac']:.2f}")
            print(f"  ROAS: {results['roas']:.1f}x")
            print(f"  AOV: ${results['aov']:.2f}")
            
            # iOS-specific metrics
            ios_metrics = results['ios_specific_metrics']
            print(f"  🏆 iOS Purity: {ios_metrics['ios_purity']:.1%}")
            print(f"  💳 Apple Pay Rate: {ios_metrics['apple_pay_adoption']:.1%}")
            print(f"  👨‍👩‍👧‍👦 Family Sharing: {ios_metrics['family_sharing_rate']:.1%}")
            print(f"  ⏱️ Screen Time Integration: {ios_metrics['screen_time_integration_rate']:.1%}")
            print(f"  🚫 Android Blocked: {results['android_blocked']} impressions")
            print(f"  💰 Waste Prevented: ${ios_metrics['android_waste_prevented']:.2f}")
            
            # Track best performance
            if results['roas'] > best_roas and results['conversions'] > 0:
                best_roas = results['roas']
                best_campaign = test_config['name']
            
            total_android_blocked += results['android_blocked']
            total_revenue += results['revenue']
            total_cost += results['cost']
            
        except Exception as e:
            print(f"  ❌ Campaign failed: {e}")
    
    print("\n" + "=" * 80)
    print("📊 OVERALL iOS CAMPAIGN RESULTS")
    print(f"🏆 Best Campaign: {best_campaign} (ROAS: {best_roas:.1f}x)")
    print(f"💰 Total Revenue: ${total_revenue:,.2f}")
    print(f"💸 Total Cost: ${total_cost:,.2f}")
    print(f"📈 Overall ROAS: {total_revenue/total_cost:.1f}x")
    print(f"🚫 Total Android Blocked: {total_android_blocked:,} impressions")
    print(f"✅ 100% iOS Traffic - ZERO Android Waste")
    print("=" * 80)
    
    # Run compliance verification
    print("\n🔍 iOS TARGETING COMPLIANCE VERIFICATION")
    print("-" * 50)
    
    sample_config = iOSCampaignConfig(
        campaign_name="compliance_test",
        platform=Platform.GOOGLE_ADS,
        audience=iOSAudience.PREMIUM_IPHONE_FAMILIES
    )
    
    campaign = ios_env.ios_targeting_engine.create_ios_campaign(sample_config)
    compliance = ios_env.ios_targeting_engine.verify_no_android_targeting(campaign)
    
    all_compliant = True
    for check, passed in compliance.items():
        status = "✅" if passed else "❌" 
        print(f"  {status} {check.replace('_', ' ').title()}")
        if not passed:
            all_compliant = False
    
    if all_compliant:
        print("\n✅ FULL iOS TARGETING COMPLIANCE")
        print("🚀 Ready for iOS-only campaign launch!")
    else:
        print("\n❌ COMPLIANCE ISSUES DETECTED")
        print("🚨 Fix required before launch")
    
    return ios_env


if __name__ == "__main__":
    ios_env = run_comprehensive_ios_campaign_test()