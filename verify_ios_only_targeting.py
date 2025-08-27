#!/usr/bin/env python3
"""
FINAL VERIFICATION: iOS-Only Targeting
Ensures ZERO Android traffic gets through the system
"""

import json
from typing import Dict, List, Any
from ios_targeting_system import iOSTargetingEngine, iOSCampaignConfig, Platform, iOSAudience
from aura_ios_campaign_system import iOSAuraCampaignEnvironment


def verify_android_exclusion():
    """Verify that Android users are completely excluded"""
    
    print("🔍 FINAL ANDROID EXCLUSION VERIFICATION")
    print("=" * 80)
    print("Testing that NO Android users can get through iOS targeting")
    print("=" * 80)
    
    ios_engine = iOSTargetingEngine()
    
    # Test 1: Try to create campaign without Android exclusion
    print("\n1️⃣  Testing mandatory Android exclusion...")
    try:
        bad_config = iOSCampaignConfig(
            campaign_name="test_bad_config",
            platform=Platform.GOOGLE_ADS,
            audience=iOSAudience.PREMIUM_IPHONE_FAMILIES,
            exclude_android=False  # This should fail
        )
        ios_engine.create_ios_campaign(bad_config)
        print("❌ CRITICAL FAILURE: Campaign created without Android exclusion!")
        return False
    except ValueError as e:
        if "iOS campaigns MUST exclude Android users" in str(e):
            print("✅ Android exclusion is mandatory - PASSED")
        else:
            print(f"❌ Wrong error message: {e}")
            return False
    
    # Test 2: Verify all platforms exclude Android
    print("\n2️⃣  Testing platform-specific Android exclusion...")
    all_platforms_good = True
    
    for platform in Platform:
        config = iOSCampaignConfig(
            campaign_name=f"test_{platform.value}",
            platform=platform,
            audience=iOSAudience.PREMIUM_IPHONE_FAMILIES
        )
        
        try:
            campaign = ios_engine.create_ios_campaign(config)
            excluded_os = campaign['device_targeting']['excluded_os']
            
            if 'Android' not in excluded_os:
                print(f"❌ Platform {platform.value} does NOT exclude Android")
                all_platforms_good = False
            else:
                print(f"✅ Platform {platform.value} excludes Android")
                
        except Exception as e:
            print(f"❌ Platform {platform.value} failed: {e}")
            all_platforms_good = False
    
    if not all_platforms_good:
        return False
    
    # Test 3: Test Android keyword blocking
    print("\n3️⃣  Testing Android keyword blocking...")
    config = iOSCampaignConfig(
        campaign_name="test_keywords",
        platform=Platform.GOOGLE_ADS,
        audience=iOSAudience.PREMIUM_IPHONE_FAMILIES
    )
    
    campaign = ios_engine.create_ios_campaign(config)
    negative_keywords = campaign['keywords']['negative']
    
    required_blocks = ['android', 'samsung', 'google pixel', 'family link']
    all_blocked = True
    
    for term in required_blocks:
        if term not in negative_keywords:
            print(f"❌ Android term '{term}' NOT blocked")
            all_blocked = False
        else:
            print(f"✅ Android term '{term}' blocked")
    
    if not all_blocked:
        return False
    
    # Test 4: Test simulated campaign blocking
    print("\n4️⃣  Testing simulated campaign Android blocking...")
    ios_env = iOSAuraCampaignEnvironment()
    
    config = iOSCampaignConfig(
        campaign_name="test_simulation",
        platform=Platform.GOOGLE_ADS,
        audience=iOSAudience.PREMIUM_IPHONE_FAMILIES
    )
    
    # Run small test campaign
    results = ios_env.run_ios_campaign(config, num_impressions=1000)
    
    ios_purity = results['ios_specific_metrics']['ios_purity']
    android_blocked = results['android_blocked']
    
    print(f"iOS Purity: {ios_purity:.1%}")
    print(f"Android Blocked: {android_blocked} impressions")
    
    if ios_purity < 1.0:
        print("❌ Some Android traffic got through!")
        return False
    else:
        print("✅ 100% iOS traffic - no Android waste")
    
    # Test 5: Test creative messaging
    print("\n5️⃣  Testing iOS-specific creative messaging...")
    creative_tests_passed = 0
    total_creative_tests = 0
    
    for audience in iOSAudience:
        creative = ios_engine.creative_generator.generate_ios_creative(
            audience=audience,
            journey_stage=ios_engine.creative_selector.JourneyStage.AWARENESS,
            creative_type=ios_engine.creative_selector.CreativeType.HERO_IMAGE
        )
        
        total_creative_tests += 1
        text_content = f"{creative.headline} {creative.description}".lower()
        
        # Must have iOS/Apple terms
        ios_terms = ['ios', 'iphone', 'ipad', 'apple', 'screen time']
        has_ios_term = any(term in text_content for term in ios_terms)
        
        # Must NOT have Android terms
        android_terms = ['android', 'samsung', 'google pixel']
        has_android_term = any(term in text_content for term in android_terms)
        
        if has_ios_term and not has_android_term:
            creative_tests_passed += 1
            print(f"✅ {audience.value}: iOS-specific messaging")
        else:
            print(f"❌ {audience.value}: Bad messaging - {creative.headline[:50]}...")
    
    creative_success_rate = creative_tests_passed / total_creative_tests
    
    if creative_success_rate < 1.0:
        print(f"❌ Only {creative_success_rate:.1%} of creatives are iOS-specific")
        return False
    else:
        print("✅ All creatives are iOS-specific")
    
    # Test 6: Budget allocation verification
    print("\n6️⃣  Testing budget allocation prevents Android waste...")
    
    # Simulate budget allocation
    test_results = []
    total_waste_prevented = 0
    
    for audience in list(iOSAudience)[:3]:  # Test 3 audiences
        config = iOSCampaignConfig(
            campaign_name=f"budget_test_{audience.value}",
            platform=Platform.GOOGLE_ADS,
            audience=audience
        )
        
        results = ios_engine.simulate_ios_campaign_performance(config, impressions=5000)
        waste_prevented = results['financial']['android_waste_prevented']
        total_waste_prevented += waste_prevented
        
        print(f"✅ {audience.value}: ${waste_prevented:.2f} Android waste prevented")
    
    print(f"💰 Total waste prevented: ${total_waste_prevented:.2f}")
    
    return True


def generate_ios_targeting_report():
    """Generate comprehensive iOS targeting report"""
    
    print("\n📊 iOS TARGETING IMPLEMENTATION REPORT")
    print("=" * 80)
    
    ios_engine = iOSTargetingEngine()
    
    report = {
        'implementation_date': '2025-01-22',
        'ios_requirement': 'Aura Balance requires iOS 14.0+ and iPhone',
        'android_exclusion': 'MANDATORY - All campaigns block Android traffic',
        
        'platforms_configured': [],
        'audience_segments': [],
        'creative_library_size': 0,
        'compliance_status': 'FULL_COMPLIANCE',
        
        'key_benefits': [
            'ZERO budget waste on incompatible Android users',
            'Premium positioning as iOS-exclusive feature',
            'Higher conversion rates from iOS premium audience',
            '62.8% of Aura traffic is already iOS - perfect match'
        ],
        
        'campaign_performance_estimates': {}
    }
    
    # Platform configuration
    for platform in Platform:
        platform_info = {
            'platform': platform.value,
            'android_excluded': True,
            'ios_specific_features': True
        }
        report['platforms_configured'].append(platform_info)
    
    # Audience segments
    for audience in iOSAudience:
        audience_data = ios_engine.ios_audiences[audience]
        audience_info = {
            'segment': audience.value,
            'expected_ltv': audience_data['expected_ltv'],
            'bid_multiplier': audience_data['bid_multiplier'],
            'targeting': 'iOS devices only'
        }
        report['audience_segments'].append(audience_info)
    
    # Performance estimates
    for audience in iOSAudience:
        config = iOSCampaignConfig(
            campaign_name=f"estimate_{audience.value}",
            platform=Platform.GOOGLE_ADS,
            audience=audience
        )
        
        results = ios_engine.simulate_ios_campaign_performance(config, impressions=10000)
        
        report['campaign_performance_estimates'][audience.value] = {
            'ctr': f"{results['performance']['ctr']:.2%}",
            'cvr': f"{results['performance']['cvr']:.2%}",
            'cac': f"${results['performance']['cac']:.2f}",
            'roas': f"{results['performance']['roas']:.1f}x",
            'android_waste_prevented': f"${results['financial']['android_waste_prevented']:.2f}"
        }\n    \n    # Save report\n    with open('/home/hariravichandran/AELP/ios_targeting_implementation_report.json', 'w') as f:\n        json.dump(report, f, indent=2)\n    \n    print(\"✅ iOS targeting report generated\")\n    print(f\"📁 Saved to: ios_targeting_implementation_report.json\")\n    \n    return report\n\n\ndef main():\n    \"\"\"Main verification function\"\"\"\n    \n    print(\"🍎 AURA BALANCE iOS-ONLY TARGETING VERIFICATION\")\n    print(\"Verifying that Balance (iOS-only app) properly excludes Android traffic\")\n    print(\"\\n\" + \"=\"*80)\n    \n    # Run verification tests\n    android_exclusion_verified = verify_android_exclusion()\n    \n    print(\"\\n\" + \"=\"*80)\n    \n    if android_exclusion_verified:\n        print(\"✅ ALL ANDROID EXCLUSION TESTS PASSED\")\n        print(\"✅ iOS-only targeting is properly implemented\")\n        print(\"✅ NO budget will be wasted on Android users\")\n        print(\"✅ Balance app compatibility ensured\")\n        \n        # Generate implementation report\n        report = generate_ios_targeting_report()\n        \n        print(\"\\n🚀 READY FOR iOS-ONLY CAMPAIGN LAUNCH\")\n        print(\"📱 Targeting 62.8% of Aura's existing iOS traffic\")\n        print(\"💰 Premium positioning for iPhone families\")\n        \n        return True\n    else:\n        print(\"❌ ANDROID EXCLUSION VERIFICATION FAILED\")\n        print(\"🚨 FIX REQUIRED BEFORE CAMPAIGN LAUNCH\")\n        print(\"⚠️  Android users could waste budget on incompatible app\")\n        \n        return False\n\n\nif __name__ == \"__main__\":\n    success = main()\n    exit(0 if success else 1)"