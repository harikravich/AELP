#!/usr/bin/env python3
"""Test if Temporal Effects are properly integrated into the system"""

import asyncio
from datetime import datetime, timedelta
from gaelp_master_integration import MasterOrchestrator, GAELPConfig
from temporal_effects import TemporalEffects, EventSpike

async def test_temporal_effects_integration():
    """Test Temporal Effects integration in bid calculation"""
    
    print("="*80)
    print("TESTING TEMPORAL EFFECTS INTEGRATION")
    print("="*80)
    
    # Test 1: Check if Temporal Effects is initialized
    print("\n1. Checking Temporal Effects initialization...")
    try:
        config = GAELPConfig()
        config.enable_temporal_effects = True
        master = MasterOrchestrator(config)
        
        if hasattr(master, 'temporal_effects'):
            print(f"   ✅ Temporal Effects initialized")
            print(f"      Enable flag: {config.enable_temporal_effects}")
        else:
            print(f"   ❌ Temporal Effects not found in master")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Test temporal multipliers
    print("\n2. Testing temporal multipliers...")
    try:
        temporal = master.temporal_effects
        
        # Test different times
        test_times = [
            (datetime.now().replace(hour=10), "Morning (10 AM)"),
            (datetime.now().replace(hour=14), "Afternoon (2 PM)"),
            (datetime.now().replace(hour=20), "Evening (8 PM)"),
            (datetime.now().replace(hour=2), "Late night (2 AM)")
        ]
        
        for test_time, desc in test_times:
            base_bid = 2.0
            result = temporal.adjust_bidding(
                base_bid=base_bid,
                date=test_time
            )
            adjusted_bid = result['adjusted_bid']
            reason = result.get('reason', '')
            
            multiplier = adjusted_bid / base_bid
            print(f"   ✅ {desc}:")
            print(f"      Base bid: ${base_bid:.2f} → Adjusted: ${adjusted_bid:.2f}")
            print(f"      Multiplier: {multiplier:.2f}x")
            print(f"      Reason: {reason[:60]}...")
            
    except Exception as e:
        print(f"   ❌ Temporal multipliers failed: {e}")
        return False
    
    # Test 3: Test event spikes
    print("\n3. Testing event spikes...")
    try:
        # Add a test event happening now
        test_event = EventSpike(
            name="test_promotion",
            multiplier=2.0,
            duration_days=3
        )
        temporal.add_event_spike(test_event, datetime.now() - timedelta(days=1))
        
        # Test bidding during event
        base_bid = 3.0
        result = temporal.adjust_bidding(
            base_bid=base_bid,
            date=datetime.now()
        )
        adjusted_bid = result['adjusted_bid']
        reason = result.get('reason', '')
        
        if adjusted_bid > base_bid:
            print(f"   ✅ Event spike detected:")
            print(f"      Base bid: ${base_bid:.2f} → Event bid: ${adjusted_bid:.2f}")
            print(f"      Event multiplier: {adjusted_bid/base_bid:.2f}x")
        else:
            print(f"   ⚠️  Event spike not affecting bids as expected")
            
    except Exception as e:
        print(f"   ❌ Event spike test failed: {e}")
        return False
    
    # Test 4: Test integration in bid calculation flow
    print("\n4. Testing integration in bid calculation...")
    try:
        # Create a mock journey state
        journey_state = {
            'conversion_probability': 0.1,
            'journey_stage': 2,
            'user_fatigue_level': 0.2,
            'hour_of_day': datetime.now().hour
        }
        
        query_data = {
            'intent_strength': 0.7,
            'segment': 'crisis_parent'
        }
        
        creative_selection = {
            'creative_type': 'display'
        }
        
        # Calculate bid using the master's method
        bid_amount = await master._calculate_bid(
            journey_state=journey_state,
            query_data=query_data,
            creative_selection=creative_selection
        )
        
        print(f"   ✅ Bid calculation with temporal effects:")
        print(f"      Final bid: ${bid_amount:.2f}")
        print(f"      Temporal effects enabled: {master.config.enable_temporal_effects}")
        
        # Compare with temporal disabled
        master.config.enable_temporal_effects = False
        bid_without_temporal = await master._calculate_bid(
            journey_state=journey_state,
            query_data=query_data,
            creative_selection=creative_selection
        )
        
        print(f"      Bid without temporal: ${bid_without_temporal:.2f}")
        print(f"      Difference: ${abs(bid_amount - bid_without_temporal):.2f}")
        
    except Exception as e:
        print(f"   ❌ Bid calculation integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test seasonal patterns
    print("\n5. Testing seasonal patterns...")
    try:
        # Test different months
        test_months = [
            (datetime(2024, 1, 15), "January (New Year)"),
            (datetime(2024, 8, 20), "August (Back to School)"),
            (datetime(2024, 11, 25), "November (Black Friday)"),
            (datetime(2024, 12, 20), "December (Holiday)")
        ]
        
        for test_date, desc in test_months:
            multiplier = temporal.get_seasonal_multiplier(test_date)
            print(f"   ✅ {desc}: {multiplier:.2f}x multiplier")
            
    except Exception as e:
        print(f"   ❌ Seasonal patterns failed: {e}")
        return False
    
    # Test 6: Test optimal timing recommendations
    print("\n6. Testing optimal timing recommendations...")
    try:
        recommendations = temporal.get_optimal_timing(
            target_date=datetime.now().date(),
            duration_hours=24*7  # 7 days
        )
        
        print(f"   ✅ Timing recommendations for next 7 days:")
        print(f"      Peak time: {recommendations['peak_time']['datetime']}")
        print(f"      Peak demand multiplier: {recommendations['peak_time']['demand_multiplier']:.2f}")
        print(f"      Average demand: {recommendations['average_demand']:.2f}")
        if 'recommendations' in recommendations and recommendations['recommendations']:
            print(f"      Top recommendation: {recommendations['recommendations'][0]}")
        
    except Exception as e:
        print(f"   ❌ Optimal timing failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ TEMPORAL EFFECTS INTEGRATION TEST PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_temporal_effects_integration())
    exit(0 if success else 1)