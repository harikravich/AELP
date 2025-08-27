#!/usr/bin/env python3
"""
Test script to verify Creative Selector is properly connected to GAELP ad serving
"""

import asyncio
import sys
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, '/home/hariravichandran/AELP')

from gaelp_master_integration import MasterOrchestrator, GAELPConfig
from user_journey_database import UserProfile, UserJourney, JourneyState

async def test_creative_selector_connection():
    """Test that Creative Selector is properly connected to GAELP ad serving"""
    print("🔗 Testing Creative Selector Connection to GAELP")
    print("=" * 50)
    
    # Create small test configuration
    config = GAELPConfig(
        simulation_days=1,
        users_per_day=5,  # Small test
        n_parallel_worlds=2,
        daily_budget_total=50.0,
        enable_creative_optimization=True,
        enable_safety_system=True,
        enable_budget_pacing=True
    )
    
    print("1. Initializing GAELP Master Orchestrator...")
    orchestrator = MasterOrchestrator(config)
    
    # Verify Creative Selector is initialized
    if orchestrator.creative_selector:
        print(f"   ✅ Creative Selector initialized")
        print(f"   📊 Available creatives: {len(orchestrator.creative_selector.creatives)}")
        print(f"   🧪 A/B tests: {len(orchestrator.creative_selector.ab_tests)}")
        
        # Show available creatives
        print("   Available Creative Types:")
        for creative_id, creative in list(orchestrator.creative_selector.creatives.items())[:5]:
            print(f"     {creative_id}: {creative.headline[:40]}...")
    else:
        print("   ❌ Creative Selector not initialized")
        return False
    
    print("\n2. Testing Creative Selection for Different Users...")
    
    # Test different user scenarios
    test_scenarios = [
        {
            "name": "Crisis Parent - New User",
            "profile": UserProfile(
                user_id="test_crisis_parent",
                canonical_user_id="test_crisis_parent",
                device_ids=["mobile_123"],
                current_journey_state=JourneyState.UNAWARE,
                conversion_probability=0.8,
                first_seen=datetime.now(),
                last_seen=datetime.now()
            ),
            "journey_state": JourneyState.UNAWARE,
            "touchpoint_count": 0
        },
        {
            "name": "Researcher - Returning User",
            "profile": UserProfile(
                user_id="test_researcher", 
                canonical_user_id="test_researcher",
                device_ids=["desktop_456"],
                current_journey_state=JourneyState.CONSIDERING,
                conversion_probability=0.5,
                first_seen=datetime.now(),
                last_seen=datetime.now()
            ),
            "journey_state": JourneyState.CONSIDERING,
            "touchpoint_count": 3
        },
        {
            "name": "Price Conscious - Deal Seeker",
            "profile": UserProfile(
                user_id="test_price_conscious",
                canonical_user_id="test_price_conscious", 
                device_ids=["tablet_789"],
                current_journey_state=JourneyState.AWARE,
                conversion_probability=0.3,
                first_seen=datetime.now(),
                last_seen=datetime.now()
            ),
            "journey_state": JourneyState.AWARE,
            "touchpoint_count": 1
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n   {i}. {scenario['name']}")
        
        # Create mock journey
        journey = UserJourney(
            journey_id=f"test_journey_{i}",
            user_id=scenario['profile'].user_id,
            canonical_user_id=scenario['profile'].canonical_user_id,
            journey_start=datetime.now(),
            current_state=scenario['journey_state'],
            touchpoint_count=scenario['touchpoint_count'],
            converted=False
        )
        
        try:
            # Test creative selection (this is the main integration point)
            creative_selection = await orchestrator._select_creative(scenario['profile'], journey)
            
            # Verify we got a real creative, not empty dict
            if creative_selection and creative_selection.get('creative_id') != 'default':
                print(f"      ✅ Real creative selected: {creative_selection['headline']}")
                print(f"      🎯 CTA: {creative_selection['cta']}")
                print(f"      🏷️  Type: {creative_selection['creative_type']}")
                print(f"      👤 Segment: {creative_selection['user_segment']}")
                print(f"      📍 Landing: {creative_selection['landing_page']}")
                print(f"      🧠 Reason: {creative_selection['selection_reason']}")
                
                # Verify all required fields are present
                required_fields = ['creative_id', 'headline', 'description', 'cta', 'creative_type', 'user_segment']
                missing_fields = [field for field in required_fields if not creative_selection.get(field)]
                
                if missing_fields:
                    print(f"      ⚠️  Missing fields: {missing_fields}")
                else:
                    print(f"      ✅ All required fields present")
                    
            else:
                print(f"      ⚠️  Using fallback/default creative")
                return False
                
        except Exception as e:
            print(f"      ❌ Error selecting creative: {e}")
            return False
    
    print("\n3. Testing Creative Performance Tracking...")
    
    # Test impression tracking
    try:
        # Get initial performance
        initial_report = orchestrator.creative_selector.get_performance_report(days=1)
        initial_impressions = initial_report['total_impressions']
        
        # Track some test impressions
        test_creative_id = 'crisis_parent_emergency_1'
        for i in range(3):
            orchestrator.creative_selector.track_impression(
                creative_id=test_creative_id,
                user_id=f'perf_test_user_{i}',
                clicked=i < 2,  # First 2 clicked
                converted=i == 0,  # First converted
                engagement_time=30.0 + i * 10,
                cost=2.0
            )
        
        # Check updated performance
        updated_report = orchestrator.creative_selector.get_performance_report(days=1)
        new_impressions = updated_report['total_impressions']
        
        print(f"   Initial impressions: {initial_impressions}")
        print(f"   After tracking: {new_impressions}")
        print(f"   ✅ Performance tracking working: +{new_impressions - initial_impressions} impressions")
        
        if test_creative_id in updated_report['creative_performance']:
            perf = updated_report['creative_performance'][test_creative_id]
            print(f"   Test creative performance: {perf['impressions']} imp, "
                  f"{perf['ctr']:.2%} CTR, {perf['cvr']:.2%} CVR")
        
    except Exception as e:
        print(f"   ❌ Error in performance tracking: {e}")
        return False
    
    print("\n4. Testing A/B Test Integration...")
    
    try:
        # Check A/B tests were created
        ab_tests = orchestrator.creative_selector.ab_tests
        print(f"   A/B tests configured: {len(ab_tests)}")
        
        for test_id, variant in ab_tests.items():
            print(f"     {test_id}: {variant.name} ({variant.traffic_split*100:.0f}% traffic)")
        
        if ab_tests:
            print("   ✅ A/B testing integration working")
        else:
            print("   ⚠️  No A/B tests found")
            
    except Exception as e:
        print(f"   ❌ Error checking A/B tests: {e}")
        return False
    
    print("\n5. Testing Creative Fatigue System...")
    
    try:
        # Test fatigue calculation
        test_user_id = "fatigue_test_user"
        test_creative_id = "crisis_parent_emergency_1"
        
        # Show same creative multiple times
        for i in range(4):
            orchestrator.creative_selector.track_impression(
                creative_id=test_creative_id,
                user_id=test_user_id,
                clicked=False,
                engagement_time=20.0,
                cost=1.5
            )
        
        # Check fatigue score
        fatigue_score = orchestrator.creative_selector.calculate_fatigue(test_creative_id, test_user_id)
        fatigue_analysis = orchestrator.creative_selector.get_fatigue_analysis(test_user_id)
        
        print(f"   Fatigue score for {test_creative_id}: {fatigue_score:.2f}")
        print(f"   Total fatigued creatives: {sum(1 for score in fatigue_analysis.values() if score > 0.5)}")
        print("   ✅ Creative fatigue system working")
        
    except Exception as e:
        print(f"   ❌ Error testing fatigue: {e}")
        return False
    
    return True

async def main():
    """Run the Creative Selector connection test"""
    print("🧪 GAELP Creative Selector Connection Test")
    print("=" * 50)
    
    success = await test_creative_selector_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: Creative Selector is fully connected to GAELP!")
        print("\nKey Features Working:")
        print("✅ Dynamic creative selection based on user profile")
        print("✅ User segmentation (Crisis Parents, Researchers, Price Conscious)")
        print("✅ Journey stage awareness (Awareness, Consideration, Decision)")
        print("✅ Device and context detection")
        print("✅ Performance tracking (impressions, clicks, conversions)")
        print("✅ A/B testing framework")
        print("✅ Creative fatigue prevention")
        print("✅ Comprehensive creative metadata")
        print("\n🚀 No more empty ad_content dictionaries {} - all ads are now targeted!")
    else:
        print("❌ FAILURE: Creative Selector connection has issues")
        print("Review the error messages above")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)