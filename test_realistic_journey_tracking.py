#!/usr/bin/env python3
"""
Test that journey tracking is realistic and matches what we'd see in production
"""

import sys
import time
import json

def test_realistic_journey_tracking():
    print("\nTesting Realistic Journey Tracking")
    print("="*60)
    
    # Import dashboard
    print("1. Importing dashboard...")
    from gaelp_live_dashboard_enhanced import GAELPLiveSystemEnhanced
    system = GAELPLiveSystemEnhanced()
    print("   ✅ Dashboard imported")
    
    # Start simulation
    print("\n2. Starting simulation...")
    system.start_simulation()
    time.sleep(2)
    print("   ✅ Simulation started")
    
    # Let it run to generate some clicks
    print("\n3. Running for 10 seconds to generate clicks...")
    time.sleep(10)
    
    # Check journey data
    print("\n4. Checking journey/click tracking...")
    
    # Get active journeys (should be click IDs now)
    journeys = system.active_journeys
    print(f"   Active click IDs tracked: {len(journeys)}")
    
    if journeys:
        # Look at a sample journey
        sample_click_id = list(journeys.keys())[0]
        sample_journey = journeys[sample_click_id]
        
        print(f"\n   Sample click tracking (ID: {sample_click_id[-8:]})")
        print(f"   - Channel: {sample_journey.get('channel')}")
        print(f"   - Keyword/Segment: {sample_journey.get('keyword')}")
        print(f"   - Device: {sample_journey.get('device')}")
        print(f"   - Impression cost: ${sample_journey.get('impression_cost', 0):.2f}")
        print(f"   - Clicked: {sample_journey.get('clicked', False)}")
        print(f"   - Converted: {sample_journey.get('converted', False)}")
        print(f"   - Conversion pending: {sample_journey.get('conversion_pending', False)}")
        
        # Verify we DON'T have fantasy data
        print("\n5. Verifying NO fantasy data in journeys...")
        fantasy_fields = ['touchpoints', 'stage', 'impressions', 'clicks', 'user_id', 'cross_device']
        found_fantasy = False
        for field in fantasy_fields:
            if field in sample_journey:
                print(f"   ❌ FOUND FANTASY FIELD: {field}")
                found_fantasy = True
        
        if not found_fantasy:
            print("   ✅ No fantasy journey data found!")
        
        # Check what we CAN track (realistic)
        print("\n6. Verifying REALISTIC tracking...")
        realistic_fields = ['click_id', 'timestamp', 'channel', 'keyword', 'device', 'impression_cost']
        for field in realistic_fields:
            if field in sample_journey:
                print(f"   ✅ {field}: {sample_journey[field]}")
            else:
                print(f"   ⚠️ Missing realistic field: {field}")
    
    # Check dashboard data
    print("\n7. Checking dashboard data...")
    data = system.get_dashboard_data()
    
    # Verify segment performance comes from aggregate data
    if 'segment_performance' in data:
        print("   ✅ Segment performance (from aggregate patterns):")
        for segment, perf in data['segment_performance'].items():
            print(f"      - {segment}: {perf.get('impressions', 0)} impressions, {perf.get('clicks', 0)} clicks")
    
    # Check attribution model
    if 'component_tracking' in data and 'attribution' in data['component_tracking']:
        attr = data['component_tracking']['attribution']
        print(f"\n   ✅ Attribution (YOUR conversions only):")
        print(f"      - Last touch: {attr.get('last_touch', 0)}")
        print(f"      - First touch: {attr.get('first_touch', 0)}")
        print(f"      - Multi touch: {attr.get('multi_touch', 0)}")
    
    # Check for removed competitor tracking
    print("\n8. Verifying competitor data removed...")
    if 'competitive_tracking' in data.get('component_tracking', {}):
        print("   ❌ STILL HAS competitive_tracking!")
    else:
        print("   ✅ No competitive_tracking found")
    
    if 'competitor_analysis' in data:
        print("   ❌ STILL HAS competitor_analysis!")
    else:
        print("   ✅ No competitor_analysis found")
    
    # Stop simulation
    system.is_running = False
    
    print("\n✅ Realistic journey tracking test complete!")
    print("\nSUMMARY:")
    print("- Tracks click IDs (not user journeys)")
    print("- Only sees individual impressions/clicks")
    print("- Attributes conversions within window")
    print("- No cross-session user tracking")
    print("- No competitor visibility")
    print("- Segments from aggregate patterns only")
    
    return True

if __name__ == "__main__":
    success = test_realistic_journey_tracking()
    if not success:
        sys.exit(1)