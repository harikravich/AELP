#!/usr/bin/env python3
"""
CRITICAL TEST: Verify iOS-Only Targeting Compliance
Ensures NO ANDROID TRAFFIC gets through - prevents budget waste
"""

import pytest
import json
from typing import Dict, List, Any
from ios_targeting_system import (
    iOSTargetingEngine, iOSCampaignConfig, Platform, iOSAudience, 
    iOSDevice, iOSUserProfile
)


class TestiOSTargetingCompliance:
    """Test iOS targeting compliance - NO ANDROID ALLOWED"""
    
    def setup_method(self):
        """Setup iOS targeting engine"""
        self.ios_engine = iOSTargetingEngine()
    
    def test_android_exclusion_mandatory(self):
        """Test that Android exclusion is mandatory"""
        
        # Try to create campaign without Android exclusion - should fail
        with pytest.raises(ValueError, match="iOS campaigns MUST exclude Android users"):
            config = iOSCampaignConfig(
                campaign_name="test_campaign",
                platform=Platform.GOOGLE_ADS,
                audience=iOSAudience.PREMIUM_IPHONE_FAMILIES,
                exclude_android=False  # This should cause failure
            )
            self.ios_engine.create_ios_campaign(config)
    
    def test_all_platforms_exclude_android(self):
        """Test all platforms properly exclude Android"""
        
        for platform in Platform:
            for audience in iOSAudience:
                config = iOSCampaignConfig(
                    campaign_name=f"test_{platform.value}_{audience.value}",
                    platform=platform,
                    audience=audience
                )
                
                campaign = self.ios_engine.create_ios_campaign(config)
                
                # Verify Android exclusion
                excluded_os = campaign['device_targeting']['excluded_os']
                assert 'Android' in excluded_os, f"Platform {platform.value} must exclude Android"
                
                # Verify iOS-only devices
                included_devices = campaign['device_targeting']['included_devices']
                assert all('iPhone' in device or 'iPad' in device 
                          for device in included_devices), f"Platform {platform.value} must target only iOS devices"
    
    def test_android_keywords_blocked(self):
        """Test that Android keywords are blocked"""
        
        config = iOSCampaignConfig(
            campaign_name="test_keywords",
            platform=Platform.GOOGLE_ADS,
            audience=iOSAudience.PREMIUM_IPHONE_FAMILIES
        )
        
        campaign = self.ios_engine.create_ios_campaign(config)
        
        negative_keywords = campaign['keywords']['negative']
        android_terms = ['android', 'samsung', 'google pixel', 'family link', 'huawei']
        
        for term in android_terms:
            assert term in negative_keywords, f"Must block Android keyword: {term}"
    
    def test_ios_only_messaging(self):
        """Test that messaging is iOS-specific"""
        
        for audience in iOSAudience:
            creative = self.ios_engine.creative_generator.generate_ios_creative(
                audience=audience,
                journey_stage=self.ios_engine.creative_selector.get_journey_stages()[0],
                creative_type=self.ios_engine.creative_selector.get_creative_types()[0]
            )
            
            # Check for iOS/Apple terms in messaging
            text_content = f"{creative.headline} {creative.description} {creative.cta}".lower()
            
            ios_terms = ['ios', 'iphone', 'ipad', 'apple', 'screen time', 'family sharing']
            has_ios_term = any(term in text_content for term in ios_terms)
            assert has_ios_term, f"Creative must include iOS-specific messaging: {creative.headline}"
            
            # Ensure no Android terms
            android_terms = ['android', 'samsung', 'google pixel']
            has_android_term = any(term in text_content for term in android_terms)
            assert not has_android_term, f"Creative must not mention Android: {creative.headline}"
    
    def test_premium_positioning(self):
        """Test that iOS is positioned as premium, not limitation"""
        
        creative = self.ios_engine.creative_generator.generate_ios_creative(
            audience=iOSAudience.PREMIUM_IPHONE_FAMILIES,
            journey_stage=self.ios_engine.creative_selector.get_journey_stages()[0],
            creative_type=self.ios_engine.creative_selector.get_creative_types()[0]
        )
        
        text_content = f"{creative.headline} {creative.description}".lower()
        
        # Should have premium positioning words
        premium_terms = ['premium', 'exclusive', 'designed for', 'built for', 'apple']
        has_premium_term = any(term in text_content for term in premium_terms)
        assert has_premium_term, f"Must position as premium: {creative.headline}"
        
        # Should NOT have apology words
        apology_terms = ['sorry', 'unfortunately', 'limited to', 'only available on']
        has_apology_term = any(term in text_content for term in apology_terms)
        assert not has_apology_term, f"Must not apologize for iOS-only: {creative.headline}"
    
    def test_compliance_verification_strict(self):
        """Test strict compliance verification"""
        
        config = iOSCampaignConfig(
            campaign_name="compliance_test",
            platform=Platform.GOOGLE_ADS,
            audience=iOSAudience.PREMIUM_IPHONE_FAMILIES
        )
        
        campaign = self.ios_engine.create_ios_campaign(config)
        compliance = self.ios_engine.verify_no_android_targeting(campaign)
        
        # ALL compliance checks must pass
        for check, passed in compliance.items():
            assert passed, f"Compliance check failed: {check}"
    
    def test_budget_no_android_waste(self):
        """Test that budget calculations prevent Android waste"""
        
        config = iOSCampaignConfig(
            campaign_name="budget_test",
            platform=Platform.GOOGLE_ADS,
            audience=iOSAudience.PREMIUM_IPHONE_FAMILIES
        )
        
        results = self.ios_engine.simulate_ios_campaign_performance(config, impressions=10000)
        
        # Verify no Android waste tracked
        android_waste = results['financial']['android_waste_prevented']
        assert android_waste > 0, "Should track Android waste prevented"
        
        # Verify premium iOS targeting
        ios_metrics = results['ios_specific']
        assert ios_metrics['premium_score'] > 0.8, "iOS users should be premium"
    
    def test_cross_platform_consistency(self):
        """Test iOS targeting consistency across all platforms"""
        
        audience = iOSAudience.PREMIUM_IPHONE_FAMILIES
        
        platform_campaigns = {}
        for platform in Platform:
            config = iOSCampaignConfig(
                campaign_name=f"consistency_test_{platform.value}",
                platform=platform,
                audience=audience
            )
            
            campaign = self.ios_engine.create_ios_campaign(config)
            platform_campaigns[platform.value] = campaign
        
        # All platforms must exclude Android
        for platform_name, campaign in platform_campaigns.items():
            excluded_os = campaign['device_targeting']['excluded_os']
            assert 'Android' in excluded_os, f"Platform {platform_name} must exclude Android"
            
            # All must target iOS
            included_devices = campaign['device_targeting']['included_devices']
            assert any('iPhone' in device or 'iPad' in device for device in included_devices), \
                f"Platform {platform_name} must include iOS devices"
    
    def test_user_profile_premium_scoring(self):
        """Test iOS user profile premium scoring"""
        
        # High premium user
        premium_user = iOSUserProfile(
            user_id="premium_001",
            device_model=iOSDevice.IPHONE_14_PLUS,
            ios_version="16.0",
            household_income="$150k+",
            family_sharing_enabled=True,
            screen_time_active=True,
            apple_services=["iCloud+", "Apple One", "Apple Music"],
            app_store_spending="high",
            privacy_focused=True,
            teen_in_household=True
        )
        
        premium_score = premium_user.get_premium_score()
        assert premium_score > 0.8, f"Premium user should score high: {premium_score}"
        
        # Lower premium user
        budget_user = iOSUserProfile(
            user_id="budget_001", 
            device_model=iOSDevice.IPHONE_11,
            ios_version="15.0",
            household_income="$50k+",
            family_sharing_enabled=False,
            screen_time_active=False,
            apple_services=[],
            app_store_spending="low",
            privacy_focused=False,
            teen_in_household=True
        )
        
        budget_score = budget_user.get_premium_score()
        assert budget_score < 0.6, f"Budget user should score lower: {budget_score}"
    
    def test_no_fallbacks_android(self):
        """Test that there are NO fallbacks to Android targeting"""
        
        # Search for any fallback code
        import inspect
        import ios_targeting_system
        
        source = inspect.getsource(ios_targeting_system)
        
        # These terms should NOT appear in iOS targeting code
        forbidden_terms = [
            'fallback', 'if android', 'android_targeting', 'backup_android',
            'include_android', 'allow_android', 'android_fallback'
        ]
        
        for term in forbidden_terms:
            assert term.lower() not in source.lower(), f"Forbidden fallback term found: {term}"
    
    def test_performance_ios_only(self):
        """Test performance metrics are iOS-only"""
        
        config = iOSCampaignConfig(
            campaign_name="performance_test",
            platform=Platform.GOOGLE_ADS,
            audience=iOSAudience.PREMIUM_IPHONE_FAMILIES
        )
        
        results = self.ios_engine.simulate_ios_campaign_performance(config, impressions=5000)
        
        # Verify iOS-specific performance tracking
        assert 'ios_specific' in results, "Must track iOS-specific metrics"
        
        ios_specific = results['ios_specific']
        required_ios_metrics = [
            'app_store_installs',
            'premium_score', 
            'apple_ecosystem_engagement',
            'screen_time_integration_rate'
        ]
        
        for metric in required_ios_metrics:
            assert metric in ios_specific, f"Must track iOS metric: {metric}"
            assert ios_specific[metric] > 0, f"iOS metric should be positive: {metric}"


class TestRealWorldiOSScenarios:
    """Test real-world iOS targeting scenarios"""
    
    def setup_method(self):
        self.ios_engine = iOSTargetingEngine()
    
    def test_screen_time_competitor_scenario(self):
        """Test positioning against built-in Screen Time"""
        
        config = iOSCampaignConfig(
            campaign_name="screen_time_competitor",
            platform=Platform.APPLE_SEARCH_ADS,
            audience=iOSAudience.SCREEN_TIME_UPGRADERS
        )
        
        campaign = self.ios_engine.create_ios_campaign(config)
        
        # Should target Screen Time keywords
        positive_keywords = campaign['keywords']['positive']
        screen_time_keywords = [kw for kw in positive_keywords if 'screen time' in kw.lower()]
        assert len(screen_time_keywords) > 0, "Must target Screen Time keywords"
        
        # Should be premium positioned
        assert campaign['bid_strategy'] == 'target_cpa_premium', "Must use premium bidding"
    
    def test_crisis_parent_urgency_scenario(self):
        """Test crisis parent scenario with urgency"""
        
        config = iOSCampaignConfig(
            campaign_name="crisis_parent",
            platform=Platform.GOOGLE_ADS,
            audience=iOSAudience.IOS_PARENTS_TEENS
        )
        
        results = self.ios_engine.simulate_ios_campaign_performance(config, impressions=1000)
        
        # Crisis parents should have higher conversion rates
        cvr = results['performance']['cvr']
        assert cvr > 0.05, f"Crisis parents should convert well: {cvr:.2%}"
        
        # Should be willing to pay premium
        cac = results['performance']['cac']
        assert cac > 0, "Should have reasonable CAC for crisis parents"
    
    def test_apple_ecosystem_integration_scenario(self):
        """Test Apple ecosystem integration messaging"""
        
        config = iOSCampaignConfig(
            campaign_name="apple_ecosystem",
            platform=Platform.FACEBOOK_ADS,
            audience=iOSAudience.APPLE_ECOSYSTEM_USERS
        )
        
        campaign = self.ios_engine.create_ios_campaign(config)
        
        # Should emphasize integration
        creative_reqs = campaign['creative_requirements']
        assert creative_reqs['apple_design_language'], "Must use Apple design language"
        assert creative_reqs['ios_branding'], "Must emphasize iOS branding"


def run_comprehensive_ios_test():
    """Run all iOS targeting compliance tests"""
    
    print("🧪 COMPREHENSIVE iOS TARGETING COMPLIANCE TEST")
    print("=" * 80)
    print("Testing that ALL Android traffic is excluded - NO WASTE")
    print("=" * 80)
    
    # Run pytest with detailed output
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, 
        "-v", "--tb=short", "--no-header"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ ALL iOS TARGETING COMPLIANCE TESTS PASSED")
        print("✅ NO Android traffic will be targeted")
        print("✅ NO budget waste on incompatible users")
        print("✅ iOS positioned as PREMIUM, not limitation")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ iOS TARGETING COMPLIANCE TESTS FAILED")
        print("🚨 FIX REQUIRED BEFORE LAUNCH")
        print("=" * 80)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run individual test methods for demonstration
    test_ios = TestiOSTargetingCompliance()
    test_ios.setup_method()
    
    print("🔍 Quick Compliance Check")
    print("-" * 30)
    
    try:
        test_ios.test_android_exclusion_mandatory()
        print("✅ Android exclusion mandatory")
    except Exception as e:
        print(f"❌ Android exclusion test failed: {e}")
    
    try:
        test_ios.test_all_platforms_exclude_android()
        print("✅ All platforms exclude Android")
    except Exception as e:
        print(f"❌ Platform exclusion test failed: {e}")
    
    try:
        test_ios.test_premium_positioning()
        print("✅ Premium positioning verified")
    except Exception as e:
        print(f"❌ Premium positioning test failed: {e}")
    
    try:
        test_ios.test_no_fallbacks_android()
        print("✅ No Android fallbacks found")
    except Exception as e:
        print(f"❌ Fallback check failed: {e}")
    
    print("\n🚀 Running comprehensive test suite...")
    success = run_comprehensive_ios_test()
    
    if success:
        print("\n🎯 Ready for iOS-only campaign launch!")
    else:
        print("\n⚠️  Fix compliance issues before launch")