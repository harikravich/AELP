#!/usr/bin/env python3
"""
iOS-Specific Targeting System for GAELP
Since Aura Balance ONLY works on iPhone, all campaigns must target iOS exclusively.
Position iOS-only as a premium feature, not a limitation.

CRITICAL: NO ANDROID TARGETING - WASTE OF BUDGET
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib

from creative_selector import (
    CreativeSelector, UserState, UserSegment, JourneyStage,
    Creative, CreativeType, LandingPageType
)


class iOSDevice(Enum):
    """iOS device types for precise targeting"""
    IPHONE_14_PLUS = "iPhone 14+"
    IPHONE_13 = "iPhone 13"
    IPHONE_12 = "iPhone 12"
    IPHONE_11 = "iPhone 11"
    IPAD_PRO = "iPad Pro"
    IPAD_AIR = "iPad Air"
    IPAD_MINI = "iPad Mini"


class iOSAudience(Enum):
    """iOS-specific audience segments"""
    PREMIUM_IPHONE_FAMILIES = "premium_iphone_families"
    IOS_PARENTS_TEENS = "ios_parents_teens"
    APPLE_ECOSYSTEM_USERS = "apple_ecosystem_users"
    SCREEN_TIME_UPGRADERS = "screen_time_upgraders"
    APPLE_PAY_USERS = "apple_pay_users"


class Platform(Enum):
    """Advertising platforms with iOS targeting"""
    GOOGLE_ADS = "google_ads"
    FACEBOOK_ADS = "facebook_ads"
    APPLE_SEARCH_ADS = "apple_search_ads"
    TIKTOK_ADS = "tiktok_ads"  # iOS only


@dataclass
class iOSUserProfile:
    """iOS user profile for premium targeting"""
    user_id: str
    device_model: iOSDevice
    ios_version: str
    household_income: str  # "$75k+", "$100k+", "$150k+"
    family_sharing_enabled: bool
    screen_time_active: bool
    apple_services: List[str]  # ["iCloud+", "Apple One", "Apple Music"]
    app_store_spending: str  # "high", "medium", "low"
    privacy_focused: bool
    teen_in_household: bool
    location: str = "US"
    
    def get_premium_score(self) -> float:
        """Calculate premium user score (0.0-1.0)"""
        score = 0.0
        
        # Device premium score
        device_scores = {
            iOSDevice.IPHONE_14_PLUS: 1.0,
            iOSDevice.IPHONE_13: 0.9,
            iOSDevice.IPHONE_12: 0.8,
            iOSDevice.IPHONE_11: 0.6,
            iOSDevice.IPAD_PRO: 0.9,
            iOSDevice.IPAD_AIR: 0.7,
            iOSDevice.IPAD_MINI: 0.6
        }
        score += device_scores.get(self.device_model, 0.5) * 0.3
        
        # Income score
        income_scores = {
            "$150k+": 1.0,
            "$100k+": 0.8,
            "$75k+": 0.6,
            "$50k+": 0.4
        }
        score += income_scores.get(self.household_income, 0.2) * 0.3
        
        # Apple ecosystem engagement
        if self.family_sharing_enabled:
            score += 0.2
        if len(self.apple_services) >= 2:
            score += 0.1
        if self.app_store_spending == "high":
            score += 0.1
        
        return min(score, 1.0)


@dataclass 
class iOSCampaignConfig:
    """iOS-specific campaign configuration"""
    campaign_name: str
    platform: Platform
    audience: iOSAudience
    ios_version_min: str = "14.0"
    device_types: List[iOSDevice] = None
    budget_multiplier: float = 1.3  # 30% premium for iOS
    exclude_android: bool = True
    apple_specific_keywords: List[str] = None
    negative_keywords: List[str] = None
    
    def __post_init__(self):
        if self.device_types is None:
            self.device_types = [iOSDevice.IPHONE_14_PLUS, iOSDevice.IPHONE_13, iOSDevice.IPHONE_12]
        
        if self.apple_specific_keywords is None:
            self.apple_specific_keywords = [
                "iphone parental controls",
                "ios screen time alternative", 
                "iphone teen monitoring",
                "apple family safety",
                "ios behavioral health app",
                "screen time replacement",
                "iphone family wellness"
            ]
        
        if self.negative_keywords is None:
            self.negative_keywords = [
                "android", "samsung", "google pixel", "kindle",
                "google family link", "family link", "windows phone", "huawei",
                "xiaomi", "oneplus", "lg phone"
            ]


class iOSCreativeGenerator:
    """Generate iOS-specific creatives that celebrate Apple ecosystem"""
    
    def __init__(self):
        self.ios_messaging_templates = self._load_ios_messaging()
        
    def _load_ios_messaging(self) -> Dict[str, Dict[str, List[str]]]:
        """Load iOS-specific messaging templates"""
        return {
            'premium_positioning': {
                'headlines': [
                    "Premium Monitoring for iPhone Families",
                    "Exclusively Designed for iOS",
                    "The Apple of Parental Controls", 
                    "iPhone-First Mental Health Monitoring",
                    "Built for Apple Families Who Care",
                    "Screen Time Pro - For Serious Parents",
                    "Native iOS Performance You Can Trust"
                ],
                'descriptions': [
                    "Seamless Screen Time integration with AI-powered insights",
                    "Native iOS performance with Apple-grade privacy protection", 
                    "Designed for iPhone, not ported from Android",
                    "Works perfectly with Family Sharing and iCloud",
                    "Advanced monitoring that Apple families deserve"
                ],
                'ctas': [
                    "Get on App Store",
                    "Download for iPhone", 
                    "Start iOS Trial",
                    "Experience Apple AI",
                    "Join iPhone Families"
                ]
            },
            'screen_time_upgrade': {
                'headlines': [
                    "Screen Time Shows Time. We Show Why.",
                    "Beyond Screen Time - See the Real Story",
                    "Screen Time 2.0 - With AI Insights",
                    "What Screen Time Can't Tell You",
                    "Upgrade Your Screen Time Experience"
                ],
                'descriptions': [
                    "Get behavioral insights Screen Time can't provide",
                    "See mood patterns, social personas, and wellness trends", 
                    "Advanced AI analysis of your teen's digital behavior",
                    "Screen Time shows hours. We show health."
                ],
                'ctas': [
                    "Upgrade Your Monitoring",
                    "See Beyond Time",
                    "Get AI Insights", 
                    "Try Advanced Features"
                ]
            },
            'apple_ecosystem': {
                'headlines': [
                    "Seamlessly Integrated with Apple",
                    "Built for the Apple Ecosystem",
                    "Your Apple Family, Protected",
                    "iCloud + AI = Perfect Protection"
                ],
                'descriptions': [
                    "Works with Family Sharing, Screen Time, and Find My",
                    "Syncs across iPhone, iPad, and Mac seamlessly",
                    "Apple Pay checkout, Sign in with Apple, iCloud sync"
                ],
                'ctas': [
                    "Connect to Apple",
                    "Sync with iCloud",
                    "Enable Family Features"
                ]
            },
            'crisis_response': {
                'headlines': [
                    "Protect Your iPhone Teen Now",
                    "Immediate iPhone Protection Available",
                    "Crisis? Get iPhone Alerts Instantly",
                    "Real-Time iPhone Safety Monitoring"
                ],
                'descriptions': [
                    "Instant alerts sent directly to your iPhone",
                    "Real-time behavioral monitoring and intervention",
                    "Protect your teen's mental health on iOS"
                ],
                'ctas': [
                    "Protect iPhone Now",
                    "Get Instant Alerts",
                    "Enable Crisis Mode"
                ]
            }
        }
    
    def generate_ios_creative(self, audience: iOSAudience, journey_stage: JourneyStage, 
                             creative_type: CreativeType) -> Creative:
        """Generate iOS-specific creative for audience and stage"""
        
        # Select messaging category based on audience
        if audience == iOSAudience.PREMIUM_IPHONE_FAMILIES:
            category = 'premium_positioning'
        elif audience == iOSAudience.SCREEN_TIME_UPGRADERS:
            category = 'screen_time_upgrade'
        elif audience == iOSAudience.APPLE_ECOSYSTEM_USERS:
            category = 'apple_ecosystem' 
        elif audience == iOSAudience.IOS_PARENTS_TEENS:
            category = 'crisis_response'
        else:
            category = 'premium_positioning'
            
        messages = self.ios_messaging_templates[category]
        
        # Select random message elements
        headline = np.random.choice(messages['headlines'])
        description = np.random.choice(messages['descriptions'])
        cta = np.random.choice(messages['ctas'])
        
        # Map audience to user segment
        segment_mapping = {
            iOSAudience.PREMIUM_IPHONE_FAMILIES: UserSegment.CRISIS_PARENTS,
            iOSAudience.IOS_PARENTS_TEENS: UserSegment.CRISIS_PARENTS,
            iOSAudience.APPLE_ECOSYSTEM_USERS: UserSegment.RESEARCHERS,
            iOSAudience.SCREEN_TIME_UPGRADERS: UserSegment.RETARGETING,
            iOSAudience.APPLE_PAY_USERS: UserSegment.PRICE_CONSCIOUS
        }
        
        segment = segment_mapping.get(audience, UserSegment.CRISIS_PARENTS)
        
        # Determine landing page
        if "trial" in headline.lower() or "free" in headline.lower():
            landing_page = LandingPageType.FREE_TRIAL
        elif "compare" in headline.lower() or "vs" in headline.lower():
            landing_page = LandingPageType.COMPARISON_GUIDE
        elif "crisis" in category or "protect" in headline.lower():
            landing_page = LandingPageType.EMERGENCY_SETUP
        else:
            landing_page = LandingPageType.FEATURE_DEEP_DIVE
            
        # Generate creative ID
        creative_id = f"ios_{audience.value}_{journey_stage.value}_{int(np.random.random() * 1000)}"
        
        return Creative(
            id=creative_id,
            segment=segment,
            journey_stage=journey_stage,
            creative_type=creative_type,
            headline=headline,
            description=description,
            cta=cta,
            image_url=f"/images/ios/{audience.value}_{creative_type.value}.jpg",
            landing_page=landing_page,
            priority=9,  # High priority for iOS-specific
            tags=["ios", "iphone", "premium", audience.value.replace('_', '-')]
        )


class iOSTargetingEngine:
    """Main iOS targeting engine - NO ANDROID ALLOWED"""
    
    def __init__(self):
        self.creative_generator = iOSCreativeGenerator()
        self.creative_selector = CreativeSelector()
        self.ios_audiences = self._initialize_ios_audiences()
        self.platform_configs = self._initialize_platform_configs()
        
        # Initialize iOS-specific creatives
        self._generate_ios_creative_library()
        
        # Performance tracking
        self.ios_metrics = {
            'app_store_impressions': 0,
            'app_store_conversions': 0,
            'ios_ctr': 0.0,
            'ios_cvr': 0.0, 
            'ios_ltv': 0.0,
            'screen_time_comparison_clicks': 0,
            'apple_pay_conversions': 0,
            'android_waste_prevented': 0  # Money saved by not targeting Android
        }
    
    def _initialize_ios_audiences(self) -> Dict[iOSAudience, Dict[str, Any]]:
        """Initialize iOS-specific audience definitions"""
        return {
            iOSAudience.PREMIUM_IPHONE_FAMILIES: {
                'devices': [iOSDevice.IPHONE_14_PLUS, iOSDevice.IPHONE_13, iOSDevice.IPAD_PRO],
                'income_threshold': '$75k+',
                'interests': ['Apple products', 'Premium apps', 'Family sharing', 'Privacy'],
                'behaviors': ['High app store spending', 'Apple Pay user', 'Multiple Apple devices'],
                'messaging_focus': 'Premium experience and family protection',
                'bid_multiplier': 1.5,
                'expected_ltv': 450.0
            },
            
            iOSAudience.IOS_PARENTS_TEENS: {
                'devices': [iOSDevice.IPHONE_13, iOSDevice.IPHONE_12, iOSDevice.IPHONE_11],
                'income_threshold': '$50k+',
                'interests': ['Parenting', 'Teen safety', 'Screen time', 'Digital wellness'],
                'behaviors': ['Family Sharing enabled', 'Screen Time user', 'Parental control searches'],
                'messaging_focus': 'Teen protection and behavioral insights',
                'bid_multiplier': 1.4,
                'expected_ltv': 350.0
            },
            
            iOSAudience.APPLE_ECOSYSTEM_USERS: {
                'devices': [d for d in iOSDevice],  # All devices
                'income_threshold': '$60k+',
                'interests': ['Apple ecosystem', 'iCloud', 'Productivity', 'Privacy'],
                'behaviors': ['Multiple Apple services', 'Mac owner', 'Apple Watch user'],
                'messaging_focus': 'Seamless integration and ecosystem benefits',
                'bid_multiplier': 1.3,
                'expected_ltv': 400.0
            },
            
            iOSAudience.SCREEN_TIME_UPGRADERS: {
                'devices': [iOSDevice.IPHONE_13, iOSDevice.IPHONE_12],
                'income_threshold': '$40k+',
                'interests': ['Screen Time', 'Parental controls', 'Digital wellness'],
                'behaviors': ['Active Screen Time user', 'App limits set', 'Family sharing'],
                'messaging_focus': 'Advanced features beyond Screen Time',
                'bid_multiplier': 1.6,  # Highest - they already use Screen Time
                'expected_ltv': 320.0
            },
            
            iOSAudience.APPLE_PAY_USERS: {
                'devices': [iOSDevice.IPHONE_14_PLUS, iOSDevice.IPHONE_13],
                'income_threshold': '$55k+', 
                'interests': ['Convenience', 'Security', 'Premium services'],
                'behaviors': ['Frequent Apple Pay use', 'In-app purchases', 'Subscription services'],
                'messaging_focus': 'Convenient premium features',
                'bid_multiplier': 1.2,
                'expected_ltv': 280.0
            }
        }
    
    def _initialize_platform_configs(self) -> Dict[Platform, Dict[str, Any]]:
        """Initialize platform-specific iOS targeting configurations"""
        return {
            Platform.GOOGLE_ADS: {
                'device_targeting': {
                    'mobile_os': ['iOS 14.0+'],
                    'exclude_os': ['Android', 'Windows', 'Other'],
                    'device_types': ['iPhone', 'iPad'],
                    'bid_adjustment': '+30%'
                },
                'audience_targeting': {
                    'affinity': ['Apple product enthusiasts', 'Premium mobile users'],
                    'in_market': ['Parental control software', 'Mobile apps'],
                    'custom': ['Visited Apple.com', 'Screen Time users']
                },
                'keyword_strategy': {
                    'ios_specific': [
                        'iphone parental controls',
                        'ios screen time alternative',
                        'iphone family safety',
                        'screen time replacement ios'
                    ],
                    'broad_match_modifier': '+iphone +parental +controls',
                    'negative_keywords': [
                        'android', 'samsung', 'google pixel', 'family link'
                    ]
                }
            },
            
            Platform.FACEBOOK_ADS: {
                'device_targeting': {
                    'platforms': ['iOS'],
                    'min_ios_version': '14.0',
                    'exclude_platforms': ['Android', 'Desktop without mobile']
                },
                'detailed_targeting': {
                    'behaviors': [
                        'iOS device users',
                        'Apple Pay users', 
                        'Premium mobile app users',
                        'Family-focused mobile users'
                    ],
                    'interests': [
                        'iPhone',
                        'Apple Inc.',
                        'iOS apps',
                        'Screen Time (iOS)',
                        'Family Link alternatives'
                    ]
                },
                'exclusions': {
                    'behaviors': ['Android device users', 'Budget smartphone users'],
                    'interests': ['Android apps', 'Google Family Link']
                }
            },
            
            Platform.APPLE_SEARCH_ADS: {
                'targeting': {
                    'app_store_only': True,
                    'device_types': ['iPhone', 'iPad'],
                    'age_range': '25-54',  # Parent demographics
                    'location': ['US', 'CA', 'UK', 'AU']  # Premium markets
                },
                'keyword_strategy': {
                    'exact_match': [
                        'parental controls',
                        'screen time alternative',
                        'teen monitoring',
                        'family safety'
                    ],
                    'broad_match': [
                        'family wellness apps',
                        'digital parenting tools',
                        'behavioral health monitoring'
                    ]
                },
                'bid_strategy': 'aggressive'  # Premium placement
            },
            
            Platform.TIKTOK_ADS: {
                'device_targeting': {
                    'os': ['iOS'],
                    'min_version': '14.0',
                    'exclude_os': ['Android', 'Windows', 'Other']
                },
                'interest_targeting': [
                    'Parenting',
                    'Family',
                    'Teen content',
                    'Digital wellness',
                    'Apple products'
                ],
                'creative_format': ['Video', 'Image'],
                'placement': ['For You feed', 'Video details page'],
                'keyword_strategy': {
                    'ios_specific': [
                        'iphone parental controls',
                        'ios teen monitoring',
                        'iphone family safety'
                    ],
                    'negative_keywords': [
                        'android', 'samsung', 'google pixel', 'family link'
                    ]
                }
            }
        }
    
    def _generate_ios_creative_library(self):
        """Generate comprehensive iOS-specific creative library"""
        creative_count = 0
        
        for audience in iOSAudience:
            for journey_stage in [JourneyStage.AWARENESS, JourneyStage.CONSIDERATION, JourneyStage.DECISION]:
                for creative_type in [CreativeType.HERO_IMAGE, CreativeType.VIDEO, CreativeType.TEXT_AD]:
                    # Generate 2-3 creatives per combination
                    for i in range(3):
                        creative = self.creative_generator.generate_ios_creative(
                            audience, journey_stage, creative_type
                        )
                        self.creative_selector.add_creative(creative)
                        creative_count += 1
        
        print(f"Generated {creative_count} iOS-specific creatives")
    
    def get_ios_user_segment(self, ios_profile: iOSUserProfile) -> UserSegment:
        """Map iOS user profile to appropriate user segment"""
        
        premium_score = ios_profile.get_premium_score()
        
        if ios_profile.teen_in_household and premium_score > 0.7:
            return UserSegment.CRISIS_PARENTS
        elif premium_score > 0.8:
            return UserSegment.RESEARCHERS  
        elif ios_profile.app_store_spending == "low":
            return UserSegment.PRICE_CONSCIOUS
        else:
            return UserSegment.RETARGETING
    
    def create_ios_campaign(self, config: iOSCampaignConfig) -> Dict[str, Any]:
        """Create iOS-specific campaign configuration"""
        
        if not config.exclude_android:
            raise ValueError("ERROR: iOS campaigns MUST exclude Android users")
        
        audience_config = self.ios_audiences[config.audience]
        platform_config = self.platform_configs[config.platform]
        
        campaign = {
            'campaign_id': f"ios_{config.campaign_name}_{config.platform.value}",
            'name': f"iOS - {config.campaign_name}",
            'platform': config.platform.value,
            'audience': config.audience.value,
            
            # iOS-specific targeting
            'device_targeting': {
                'included_devices': [d.value for d in config.device_types],
                'min_ios_version': config.ios_version_min,
                'excluded_os': ['Android', 'Windows', 'Other']
            },
            
            # Budget and bidding
            'budget_multiplier': config.budget_multiplier,
            'bid_strategy': 'target_cpa_premium',
            'target_cpa': audience_config['expected_ltv'] * 0.2,  # 20% of LTV
            
            # Keywords
            'keywords': {
                'positive': config.apple_specific_keywords,
                'negative': config.negative_keywords
            },
            
            # Creative requirements  
            'creative_requirements': {
                'ios_branding': True,
                'apple_design_language': True,
                'app_store_badge': True,
                'no_android_imagery': True
            },
            
            # Landing page
            'landing_page': f"/ios-{config.audience.value}",
            
            # Tracking
            'conversion_tracking': {
                'app_store_installs': True,
                'in_app_events': True,
                'apple_pay_conversions': True
            }
        }
        
        return campaign
    
    def simulate_ios_campaign_performance(self, config: iOSCampaignConfig, 
                                        impressions: int = 10000) -> Dict[str, Any]:
        """Simulate iOS campaign performance (NO Android waste)"""
        
        audience_data = self.ios_audiences[config.audience]
        
        # Base performance metrics for iOS users (premium audience)
        base_ctr = 0.045  # 4.5% CTR (higher than general)
        base_cvr = 0.08   # 8% conversion (premium iOS users)
        
        # Apply audience multipliers
        ctr_multiplier = audience_data['bid_multiplier'] * 0.7
        cvr_multiplier = audience_data['bid_multiplier'] * 0.8
        
        actual_ctr = base_ctr * ctr_multiplier
        actual_cvr = base_cvr * cvr_multiplier
        
        # Calculate results
        clicks = int(impressions * actual_ctr)
        conversions = int(clicks * actual_cvr)
        
        # Cost calculation (premium iOS targeting)
        cpm = 8.0 * config.budget_multiplier  # Premium CPM for iOS
        cpc = cpm / (actual_ctr * 1000) if actual_ctr > 0 else 0
        total_cost = (impressions / 1000) * cpm + (clicks * 1.5)  # CPM + CPC
        
        # Revenue calculation
        avg_ltv = audience_data['expected_ltv']
        revenue = conversions * avg_ltv
        
        # Calculate metrics
        cac = total_cost / conversions if conversions > 0 else float('inf')
        roas = revenue / total_cost if total_cost > 0 else 0
        
        # Estimate Android waste prevented
        android_waste = total_cost * 0.3  # 30% typical waste on incompatible users
        
        results = {
            'campaign_config': asdict(config),
            'audience': config.audience.value,
            'performance': {
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'ctr': actual_ctr,
                'cvr': actual_cvr,
                'cac': cac,
                'roas': roas
            },
            'financial': {
                'total_cost': total_cost,
                'total_revenue': revenue,
                'profit': revenue - total_cost,
                'android_waste_prevented': android_waste
            },
            'ios_specific': {
                'app_store_installs': conversions,
                'premium_score': 0.85,  # iOS users are premium
                'apple_ecosystem_engagement': 0.72,
                'screen_time_integration_rate': 0.65
            }
        }
        
        return results
    
    def verify_no_android_targeting(self, campaign_config: Dict[str, Any]) -> Dict[str, bool]:
        """Verify campaign properly excludes Android (compliance check)"""
        
        checks = {
            'excludes_android': False,
            'ios_only_devices': False,
            'no_android_keywords': False,
            'apple_specific_messaging': False,
            'app_store_focused': False
        }
        
        # Check device exclusions
        excluded_os = campaign_config.get('device_targeting', {}).get('excluded_os', [])
        if 'Android' in excluded_os:
            checks['excludes_android'] = True
            
        # Check device inclusions
        included_devices = campaign_config.get('device_targeting', {}).get('included_devices', [])
        if any('iPhone' in device or 'iPad' in device for device in included_devices):
            checks['ios_only_devices'] = True
            
        # Check negative keywords
        negative_keywords = campaign_config.get('keywords', {}).get('negative', [])
        android_terms = ['android', 'samsung', 'google pixel', 'family link']
        if any(term in negative_keywords for term in android_terms):
            checks['no_android_keywords'] = True
            
        # Check creative requirements
        creative_reqs = campaign_config.get('creative_requirements', {})
        if creative_reqs.get('ios_branding') and creative_reqs.get('app_store_badge'):
            checks['apple_specific_messaging'] = True
            
        # Check landing page
        landing_page = campaign_config.get('landing_page', '')
        if 'ios' in landing_page or 'iphone' in landing_page:
            checks['app_store_focused'] = True
            
        return checks
    
    def get_ios_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate iOS-specific performance report"""
        
        base_report = self.creative_selector.get_performance_report(days)
        
        # Add iOS-specific metrics
        ios_report = {
            'period_days': days,
            'ios_only_traffic': True,
            'android_waste_prevented': self.ios_metrics['android_waste_prevented'],
            
            # Standard metrics
            'impressions': base_report['total_impressions'],
            'clicks': base_report['total_clicks'],
            'conversions': base_report['total_conversions'],
            
            # iOS-specific metrics
            'app_store_impressions': self.ios_metrics['app_store_impressions'],
            'app_store_conversions': self.ios_metrics['app_store_conversions'],
            'screen_time_comparison_clicks': self.ios_metrics['screen_time_comparison_clicks'],
            'apple_pay_conversions': self.ios_metrics['apple_pay_conversions'],
            
            # Creative performance by iOS audience
            'audience_performance': {},
            
            # Device breakdown
            'device_performance': {
                'iPhone 14+': {'impressions': 0, 'conversions': 0, 'cvr': 0},
                'iPhone 13': {'impressions': 0, 'conversions': 0, 'cvr': 0}, 
                'iPhone 12': {'impressions': 0, 'conversions': 0, 'cvr': 0},
                'iPad Pro': {'impressions': 0, 'conversions': 0, 'cvr': 0}
            }
        }
        
        return ios_report


def run_ios_targeting_demo():
    """Demonstrate iOS targeting system"""
    
    print("🍎 iOS Targeting System for GAELP")
    print("=" * 80)
    print("Aura Balance ONLY works on iPhone - targeting iOS exclusively")
    print("Positioning iOS-only as PREMIUM feature, not limitation")
    print("=" * 80)
    
    # Initialize system
    ios_engine = iOSTargetingEngine()
    
    # Test different iOS audiences
    test_audiences = [
        (iOSAudience.PREMIUM_IPHONE_FAMILIES, "Premium iPhone families with teens"),
        (iOSAudience.SCREEN_TIME_UPGRADERS, "Current Screen Time users wanting more"),
        (iOSAudience.APPLE_ECOSYSTEM_USERS, "Heavy Apple ecosystem users"),
        (iOSAudience.IOS_PARENTS_TEENS, "iOS parents with teenagers")
    ]
    
    print("\n🎯 TESTING iOS AUDIENCE CAMPAIGNS")
    print("-" * 50)
    
    best_audience = None
    best_roas = 0
    total_android_waste_prevented = 0
    
    for audience, description in test_audiences:
        print(f"\n📱 {audience.value.upper()} - {description}")
        
        # Create campaign config
        config = iOSCampaignConfig(
            campaign_name=f"Balance_{audience.value}",
            platform=Platform.GOOGLE_ADS,
            audience=audience
        )
        
        # Simulate performance
        results = ios_engine.simulate_ios_campaign_performance(config, impressions=50000)
        
        # Display results
        perf = results['performance']
        financial = results['financial']
        ios_metrics = results['ios_specific']
        
        print(f"  Impressions: {perf['impressions']:,}")
        print(f"  Clicks: {perf['clicks']:,} (CTR: {perf['ctr']:.2%})")
        print(f"  Conversions: {perf['conversions']} (CVR: {perf['cvr']:.2%})")
        print(f"  CAC: ${perf['cac']:.2f}")
        print(f"  ROAS: {perf['roas']:.1f}x")
        print(f"  Revenue: ${financial['total_revenue']:,.2f}")
        print(f"  Profit: ${financial['profit']:,.2f}")
        print(f"  💰 Android Waste Prevented: ${financial['android_waste_prevented']:,.2f}")
        print(f"  🏆 Premium Score: {ios_metrics['premium_score']:.0%}")
        
        # Track best performing
        if perf['roas'] > best_roas and perf['conversions'] > 0:
            best_roas = perf['roas']
            best_audience = audience.value
            
        total_android_waste_prevented += financial['android_waste_prevented']
    
    print("\n" + "=" * 80)
    print(f"🏆 BEST iOS AUDIENCE: {best_audience} (ROAS: {best_roas:.1f}x)")
    print(f"💰 TOTAL ANDROID WASTE PREVENTED: ${total_android_waste_prevented:,.2f}")
    print(f"✅ 100% iOS-only traffic - NO budget wasted on incompatible users")
    
    # Verify compliance
    print(f"\n🔍 COMPLIANCE CHECK")
    print("-" * 30)
    
    sample_config = ios_engine.create_ios_campaign(iOSCampaignConfig(
        campaign_name="Compliance_Test",
        platform=Platform.GOOGLE_ADS,
        audience=iOSAudience.PREMIUM_IPHONE_FAMILIES
    ))
    
    compliance = ios_engine.verify_no_android_targeting(sample_config)
    
    for check, passed in compliance.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    all_passed = all(compliance.values())
    print(f"\n{'✅ FULL COMPLIANCE' if all_passed else '❌ COMPLIANCE FAILED'}")
    
    if not all_passed:
        print("🚨 WARNING: Campaign not properly configured for iOS-only")
        print("Fix required before launching")
    
    return ios_engine


if __name__ == "__main__":
    ios_engine = run_ios_targeting_demo()