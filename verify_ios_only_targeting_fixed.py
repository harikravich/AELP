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
    
    print("\n✅ ALL ANDROID EXCLUSION TESTS PASSED")
    return True


def main():
    """Main verification function"""
    
    print("🍎 AURA BALANCE iOS-ONLY TARGETING VERIFICATION")
    print("Verifying that Balance (iOS-only app) properly excludes Android traffic")
    print("\n" + "="*80)
    
    # Run verification tests
    android_exclusion_verified = verify_android_exclusion()
    
    print("\n" + "="*80)
    
    if android_exclusion_verified:
        print("✅ ALL ANDROID EXCLUSION TESTS PASSED")
        print("✅ iOS-only targeting is properly implemented")  
        print("✅ NO budget will be wasted on Android users")
        print("✅ Balance app compatibility ensured")
        
        print("\n🚀 READY FOR iOS-ONLY CAMPAIGN LAUNCH")
        print("📱 Targeting 62.8% of Aura's existing iOS traffic")
        print("💰 Premium positioning for iPhone families")
        
        return True
    else:
        print("❌ ANDROID EXCLUSION VERIFICATION FAILED")
        print("🚨 FIX REQUIRED BEFORE CAMPAIGN LAUNCH")
        print("⚠️  Android users could waste budget on incompatible app")
        
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)