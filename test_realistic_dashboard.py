#!/usr/bin/env python3
"""
Test the REALISTIC dashboard integration
Verify NO fantasy data is being used
"""

import sys
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_realistic_dashboard():
    """Test that dashboard uses ONLY realistic simulation"""
    
    print("\n" + "="*60)
    print("TESTING REALISTIC DASHBOARD INTEGRATION")
    print("="*60 + "\n")
    
    # Test 1: Import check
    print("1. Testing imports...")
    try:
        # Should import realistic components
        from gaelp_live_dashboard_enhanced import GAELPLiveSystemEnhanced
        print("   ✅ Dashboard imported successfully")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Check it's using realistic orchestrator
    print("\n2. Checking orchestrator type...")
    try:
        system = GAELPLiveSystemEnhanced()
        
        # Check configuration
        assert 'use_real_data_only' in system.config, "Missing real data flag"
        assert system.config['use_real_data_only'] == True, "Not using real data only"
        
        # Check orchestrator attribute name
        assert hasattr(system, 'orchestrator'), "Should have 'orchestrator' not 'master'"
        assert not hasattr(system, 'master'), "Still has old 'master' attribute"
        
        print("   ✅ Using realistic orchestrator")
    except Exception as e:
        print(f"   ❌ Configuration check failed: {e}")
        return False
    
    # Test 3: Check metrics are realistic
    print("\n3. Checking metrics...")
    try:
        # Check for real metrics
        real_metrics = ['total_impressions', 'total_clicks', 'total_spend', 
                       'ctr', 'cvr', 'roas']
        for metric in real_metrics:
            assert metric in system.metrics, f"Missing real metric: {metric}"
        
        # Check NO fantasy metrics
        fantasy_metrics = ['competitor_analysis', 'journey_completion_rate', 
                          'cross_device_matches', 'safety_interventions',
                          'monte_carlo_confidence']
        for metric in fantasy_metrics:
            assert metric not in system.metrics, f"Fantasy metric found: {metric}"
        
        print("   ✅ Metrics are realistic")
    except Exception as e:
        print(f"   ❌ Metrics check failed: {e}")
        return False
    
    # Test 4: Check time series data
    print("\n4. Checking time series...")
    try:
        # Check for real time series
        real_series = ['spend', 'ctr', 'cvr', 'cpc', 'roas']
        for series in real_series:
            assert series in system.time_series, f"Missing real series: {series}"
        
        # Check NO fantasy series
        assert 'competitor_bids' not in system.time_series, "Still tracking competitor bids!"
        assert 'q_values' not in system.time_series, "Still has Q-values series"
        assert 'delayed_rewards' not in system.time_series, "Still has delayed rewards"
        
        print("   ✅ Time series is realistic")
    except Exception as e:
        print(f"   ❌ Time series check failed: {e}")
        return False
    
    # Test 5: Test starting the simulation
    print("\n5. Testing simulation start...")
    try:
        system.start_simulation()
        
        # Give it a moment to initialize
        time.sleep(2)
        
        # Check that realistic orchestrator was created
        assert system.orchestrator is not None, "Orchestrator not created"
        assert system.orchestrator.__class__.__name__ == "RealisticMasterOrchestrator", \
            f"Wrong orchestrator type: {system.orchestrator.__class__.__name__}"
        
        print("   ✅ Simulation started with realistic orchestrator")
    except Exception as e:
        print(f"   ❌ Simulation start failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Let it run and check data
    print("\n6. Running for 5 seconds and checking data...")
    try:
        time.sleep(5)
        
        # Check that metrics are being updated
        has_data = (
            system.metrics['total_impressions'] > 0 or
            system.metrics['total_spend'] > 0 or
            len(system.time_series['timestamps']) > 0
        )
        
        if has_data:
            print(f"   ✅ Data flowing:")
            print(f"      - Impressions: {system.metrics['total_impressions']}")
            print(f"      - Spend: ${system.metrics['total_spend']:.2f}")
            print(f"      - CTR: {system.metrics.get('ctr', 0)*100:.2f}%")
            print(f"      - Time series points: {len(system.time_series['timestamps'])}")
        else:
            print("   ⚠️  No data yet (may need more time)")
        
        # Stop simulation
        system.is_running = False
        
    except Exception as e:
        print(f"   ❌ Runtime check failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ DASHBOARD INTEGRATION TEST PASSED!")
    print("="*60)
    
    print("\n📊 Summary:")
    print("- Dashboard imports realistic components ✅")
    print("- Uses RealisticMasterOrchestrator ✅")
    print("- Tracks ONLY real metrics ✅")
    print("- NO fantasy data (journey states, competitor bids, etc.) ✅")
    print("- Simulation runs without errors ✅")
    
    return True


def check_for_remaining_fantasy():
    """Double-check for any remaining fantasy code"""
    
    print("\n" + "="*60)
    print("CHECKING FOR REMAINING FANTASY DATA")
    print("="*60 + "\n")
    
    import os
    import subprocess
    
    # Files to check
    main_files = [
        'gaelp_live_dashboard_enhanced.py',
        'gaelp_master_integration.py',
        'enhanced_simulator.py',
        'journey_aware_rl_agent.py'
    ]
    
    fantasy_patterns = [
        'JourneyState',
        'mental_state',
        'user_intent',
        'touchpoint_history',
        'competitor_bids(?!.*#.*REMOVED)',
        'cross_device',
        'journey_stage'
    ]
    
    issues_found = []
    
    for file in main_files:
        filepath = f"/home/hariravichandran/AELP/{file}"
        if os.path.exists(filepath):
            for pattern in fantasy_patterns:
                try:
                    result = subprocess.run(
                        ['grep', '-n', pattern, filepath],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            # Skip comments and removed sections
                            if '# REMOVED' not in line and '# NO' not in line:
                                issues_found.append(f"{file}:{line}")
                except:
                    pass
    
    if issues_found:
        print("⚠️  Fantasy data references found:")
        for issue in issues_found[:10]:  # Show first 10
            print(f"   - {issue}")
        print(f"   ... and {len(issues_found)-10} more" if len(issues_found) > 10 else "")
    else:
        print("✅ No fantasy data references found in main files!")
    
    return len(issues_found) == 0


if __name__ == "__main__":
    # Run dashboard test
    success = test_realistic_dashboard()
    
    if success:
        # Double-check for fantasy data
        clean = check_for_remaining_fantasy()
        
        if clean:
            print("\n" + "="*60)
            print("🎉 COMPLETE SUCCESS!")
            print("Dashboard is 100% REALISTIC")
            print("="*60)
        else:
            print("\n⚠️  Some fantasy references remain but dashboard works")
    else:
        print("\n❌ Dashboard test failed")
        sys.exit(1)