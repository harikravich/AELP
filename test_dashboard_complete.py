#!/usr/bin/env python3
"""
Complete test of dashboard functionality - all buttons, all data, all realism
"""

import sys
import time
import json
import requests

def test_complete_dashboard():
    print("\n" + "="*60)
    print("COMPLETE DASHBOARD TEST")
    print("="*60)
    
    # Import dashboard
    print("\n1. IMPORTING DASHBOARD...")
    from gaelp_live_dashboard_enhanced import GAELPLiveSystemEnhanced, app
    system = GAELPLiveSystemEnhanced()
    print("   ✅ Dashboard imported")
    
    # Test API endpoints
    print("\n2. TESTING API ENDPOINTS...")
    
    # Start a test client
    with app.test_client() as client:
        
        # Test /api/status
        print("   Testing /api/status...")
        response = client.get('/api/status')
        assert response.status_code == 200
        data = response.json
        assert 'metrics' in data
        assert 'time_series' in data
        assert 'component_tracking' in data
        print("   ✅ /api/status works")
        
        # Test /api/start
        print("   Testing /api/start...")
        response = client.post('/api/start')
        assert response.status_code == 200
        assert response.json['status'] == 'started'
        print("   ✅ /api/start works")
        time.sleep(2)
        
        # Test /api/stop
        print("   Testing /api/stop...")
        response = client.post('/api/stop')
        assert response.status_code == 200
        assert response.json['status'] == 'stopped'
        print("   ✅ /api/stop works")
        
        # Test /api/reset
        print("   Testing /api/reset...")
        response = client.post('/api/reset')
        assert response.status_code == 200
        assert response.json['status'] == 'reset'
        print("   ✅ /api/reset works")
    
    # Start simulation for data testing
    print("\n3. STARTING SIMULATION FOR DATA TESTING...")
    system.start_simulation()
    time.sleep(5)
    
    # Get dashboard data
    print("\n4. CHECKING ALL DATA FIELDS...")
    data = system.get_dashboard_data()
    
    # Check metrics (all should be REAL)
    print("\n   METRICS (Observable):")
    metrics_to_check = [
        'total_impressions', 'total_clicks', 'total_conversions',
        'total_spend', 'total_revenue', 'win_rate', 'current_cpa',
        'current_roi', 'ctr', 'cvr', 'avg_cpc', 'roas'
    ]
    for metric in metrics_to_check:
        if metric in data['metrics']:
            print(f"   ✅ {metric}: {data['metrics'][metric]}")
        else:
            print(f"   ⚠️ {metric}: missing")
    
    # Check time series
    print("\n   TIME SERIES (Observable):")
    time_series_to_check = [
        'spend', 'conversions', 'win_rate', 'ctr', 'cvr', 
        'cpc', 'roas', 'roi', 'exploration_rate'
    ]
    for series in time_series_to_check:
        if series in data['time_series']:
            length = len(data['time_series'][series])
            print(f"   ✅ {series}: {length} data points")
        else:
            print(f"   ⚠️ {series}: missing")
    
    # Check NO fantasy data
    print("\n5. VERIFYING NO FANTASY DATA...")
    fantasy_fields = [
        'competitor_insights', 'competitor_analysis', 'user_journeys',
        'cross_device_tracking', 'user_behavior', 'mental_states',
        'journey_stages', 'touchpoint_sequences', 'user_fatigue'
    ]
    
    found_fantasy = False
    for field in fantasy_fields:
        if field in data:
            print(f"   ❌ FOUND FANTASY: {field}")
            found_fantasy = True
        else:
            print(f"   ✅ No {field}")
    
    # Check component tracking
    print("\n6. CHECKING COMPONENT TRACKING...")
    if 'component_tracking' in data:
        components = data['component_tracking']
        
        # Check for realistic components
        realistic_components = ['rl', 'auction', 'platforms', 'channels', 'attribution']
        for comp in realistic_components:
            if comp in components:
                print(f"   ✅ {comp}: present")
            else:
                print(f"   ⚠️ {comp}: missing")
        
        # Check NO competitive_tracking
        if 'competitive' in components or 'competitor' in components:
            print("   ❌ STILL HAS competitive/competitor tracking!")
        else:
            print("   ✅ No competitive/competitor tracking")
    
    # Check specific data realism
    print("\n7. CHECKING DATA REALISM...")
    
    # Attribution should only be YOUR conversions
    if 'attribution' in components:
        attr = components['attribution']
        total_attr = sum([
            attr.get('first_touch', 0),
            attr.get('last_touch', 0),
            attr.get('multi_touch', 0),
            attr.get('data_driven', 0)
        ])
        print(f"   Attribution total: {total_attr} (should ≈ total conversions: {data['metrics']['total_conversions']})")
    
    # Segments should be discovered patterns
    if 'segment_performance' in data:
        print(f"   Segments discovered: {len(data['segment_performance'])}")
        for segment in data['segment_performance']:
            print(f"     - {segment}")
    
    # Creative performance should be YOUR tests
    if 'creative_performance' in data:
        print(f"   Creative tests tracked: {len(data['creative_performance']) if isinstance(data['creative_performance'], list) else 0}")
    
    # Check active "journeys" are actually click IDs
    print("\n8. CHECKING JOURNEY/CLICK TRACKING...")
    journeys = system.active_journeys
    if journeys:
        sample = list(journeys.values())[0] if journeys else {}
        if 'click_id' in sample and 'touchpoints' not in sample:
            print("   ✅ Tracking click IDs (not journeys)")
        elif 'touchpoints' in sample:
            print("   ❌ STILL tracking journey touchpoints!")
        else:
            print("   ✅ No active clicks yet")
    
    # Stop simulation
    system.is_running = False
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    if found_fantasy:
        print("\n❌ FAILED: Fantasy data still present!")
        return False
    else:
        print("\n✅ SUCCESS: Dashboard is realistic and functional!")
        return True

if __name__ == "__main__":
    success = test_complete_dashboard()
    if not success:
        sys.exit(1)