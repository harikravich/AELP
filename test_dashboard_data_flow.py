#!/usr/bin/env python3
"""Test script to verify dashboard data flow is working correctly"""

import sys
import time
from gaelp_master_integration import MasterOrchestrator, GAELPConfig
from gaelp_live_dashboard_enhanced import GAELPLiveSystemEnhanced

def test_dashboard_data_flow():
    """Test that data flows correctly from simulation to dashboard"""
    
    print("🔍 Testing Dashboard Data Flow...")
    print("=" * 50)
    
    # Initialize config and orchestrator
    config = GAELPConfig()
    master = MasterOrchestrator(config)
    
    # Initialize dashboard
    dashboard = GAELPLiveSystemEnhanced()
    dashboard.master = master
    
    print("\n📊 Running 10 simulation steps...")
    
    for step in range(10):
        # Run one step
        result = master.step_fixed_environment()
        
        # Update dashboard
        dashboard.update_from_realistic_step(result)
        
        # Check metrics
        print(f"\nStep {step + 1}:")
        print(f"  Metrics from master:")
        print(f"    - Total spend: {result['metrics'].get('total_spend', 0)}")
        print(f"    - Total auctions: {result['metrics'].get('total_auctions', 0)}")
        print(f"    - Total conversions: {result['metrics'].get('total_conversions', 0)}")
        
        print(f"  Dashboard state:")
        print(f"    - Total spend: ${dashboard.metrics['total_spend']:.2f}")
        print(f"    - Total impressions: {dashboard.metrics['total_impressions']}")
        print(f"    - Win rate: {dashboard.metrics.get('win_rate', 0):.2%}")
        print(f"    - Episode count: {dashboard.episode_count}")
        
        # Check platform tracking
        has_platform_data = False
        for platform, data in dashboard.platform_tracking.items():
            if data['impressions'] > 0 or data['spend'] > 0:
                has_platform_data = True
                print(f"    - {platform}: {data['impressions']} impressions, ${data['spend']:.2f} spend")
        
        if not has_platform_data:
            print("    - ⚠️ No platform data tracked")
        
        # Check step_info structure
        step_info = result.get('step_info', {})
        if step_info:
            auction = step_info.get('auction', {})
            print(f"  Step info structure:")
            print(f"    - Has auction: {'auction' in step_info}")
            print(f"    - Channel: {auction.get('channel', 'NOT FOUND')}")
            print(f"    - Won: {auction.get('won', False)}")
            print(f"    - Price: ${auction.get('price', 0):.2f}")
    
    print("\n" + "=" * 50)
    print("✅ Test Summary:")
    
    # Verify key metrics
    issues = []
    
    if dashboard.metrics['total_spend'] == 0:
        issues.append("❌ Dashboard shows $0 spend")
    else:
        print(f"✅ Dashboard tracking spend: ${dashboard.metrics['total_spend']:.2f}")
    
    if dashboard.metrics.get('win_rate', 0) >= 0.9:
        issues.append("⚠️ Win rate suspiciously high (≥90%)")
    else:
        print(f"✅ Win rate realistic: {dashboard.metrics.get('win_rate', 0):.2%}")
    
    has_channel_data = any(
        data['impressions'] > 0 
        for data in dashboard.platform_tracking.values()
    )
    
    if not has_channel_data:
        issues.append("❌ No channel performance data")
    else:
        print("✅ Channel performance tracking working")
    
    if dashboard.episode_count == 0:
        issues.append("❌ Episode count not updating")
    else:
        print(f"✅ Episode tracking: {dashboard.episode_count} episodes")
    
    # Print issues
    if issues:
        print("\n⚠️ Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n🎉 All systems working correctly!")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = test_dashboard_data_flow()
    sys.exit(0 if success else 1)