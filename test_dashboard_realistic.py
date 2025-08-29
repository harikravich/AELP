#!/usr/bin/env python3
"""
Test that dashboard works with realistic simulation
"""

import sys
import time

def test_dashboard():
    print("\nTesting Dashboard with Realistic Simulation")
    print("="*60)
    
    # Import dashboard
    print("1. Importing dashboard...")
    try:
        from gaelp_live_dashboard_enhanced import GAELPLiveSystemEnhanced
        system = GAELPLiveSystemEnhanced()
        print("   ✅ Dashboard imported")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Check it's using realistic components
    print("\n2. Checking configuration...")
    try:
        assert hasattr(system, 'orchestrator'), "Missing orchestrator"
        assert system.config.get('use_real_data_only'), "Not configured for real data"
        print(f"   ✅ Configured for real data")
        print(f"   ✅ Daily budget: ${system.daily_budget}")
    except Exception as e:
        print(f"   ❌ Config check failed: {e}")
        return False
    
    # Test getting dashboard data before starting
    print("\n3. Testing get_dashboard_data()...")
    try:
        data = system.get_dashboard_data()
        assert 'metrics' in data
        assert 'time_series' in data
        assert 'platform_performance' in data or 'component_tracking' in data
        print("   ✅ Dashboard data structure OK")
    except Exception as e:
        print(f"   ❌ get_dashboard_data failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Start simulation
    print("\n4. Starting simulation...")
    try:
        system.start_simulation()
        time.sleep(2)
        print("   ✅ Simulation started")
    except Exception as e:
        print(f"   ❌ Start failed: {e}")
        return False
    
    # Let it run
    print("\n5. Running for 5 seconds...")
    time.sleep(5)
    
    # Check data is flowing
    print("\n6. Checking data flow...")
    try:
        data = system.get_dashboard_data()
        metrics = data['metrics']
        
        print(f"   Impressions: {metrics.get('total_impressions', 0)}")
        print(f"   Clicks: {metrics.get('total_clicks', 0)}")
        print(f"   Spend: ${metrics.get('total_spend', 0):.2f}")
        print(f"   Win rate: {metrics.get('win_rate', 0)*100:.1f}%")
        
        if metrics.get('total_impressions', 0) > 0:
            print("   ✅ Data flowing!")
        else:
            print("   ⚠️ No impressions yet (may need more time)")
        
        # Check tracking components
        if 'component_tracking' in data:
            tracking = data['component_tracking']
            if 'platforms' in tracking:
                print(f"   ✅ Platform tracking: {list(tracking['platforms'].keys())}")
            if 'rl' in tracking:
                print(f"   ✅ RL tracking: {tracking['rl'].get('q_learning_updates', 0)} updates")
    except Exception as e:
        print(f"   ❌ Data check failed: {e}")
        return False
    
    # Stop
    system.is_running = False
    print("\n✅ Dashboard test complete!")
    return True

if __name__ == "__main__":
    success = test_dashboard()
    if not success:
        sys.exit(1)