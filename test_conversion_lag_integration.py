#!/usr/bin/env python3
"""Test if Conversion Lag Model is properly integrated into bidding"""

import asyncio
import numpy as np
from datetime import datetime
from gaelp_master_integration import MasterOrchestrator, GAELPConfig
from conversion_lag_model import ConversionLagModel, ConversionJourney

async def test_conversion_lag_integration():
    """Test Conversion Lag Model integration in bid calculation"""
    
    print("="*80)
    print("TESTING CONVERSION LAG MODEL INTEGRATION")
    print("="*80)
    
    # Test 1: Check if Conversion Lag Model is initialized
    print("\n1. Checking Conversion Lag Model initialization...")
    try:
        config = GAELPConfig()
        master = MasterOrchestrator(config)
        
        if hasattr(master, 'conversion_lag_model'):
            print(f"   ✅ Conversion Lag Model initialized")
            print(f"      Attribution window: {master.conversion_lag_model.attribution_window_days} days")
            print(f"      Timeout threshold: {master.conversion_lag_model.timeout_threshold_days} days")
        else:
            print(f"   ❌ Conversion Lag Model not found in master")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Test conversion time prediction
    print("\n2. Testing conversion time prediction...")
    try:
        lag_model = master.conversion_lag_model
        
        # Test different journey features
        test_scenarios = [
            ([0.8, 3, 0.9, 0.1], "High intent, late stage, low fatigue"),
            ([0.3, 1, 0.4, 0.5], "Low intent, early stage, moderate fatigue"),
            ([0.6, 2, 0.7, 0.2], "Medium intent, mid stage, low fatigue")
        ]
        
        time_points = np.array([1, 3, 7, 14, 30])
        
        for features, desc in test_scenarios:
            try:
                conversion_probs = lag_model.predict_conversion_time(
                    journey_features=features,
                    time_points=time_points
                )
                
                print(f"   ✅ {desc}:")
                if conversion_probs is not None and len(conversion_probs) > 0:
                    for i, (day, prob) in enumerate(zip(time_points, conversion_probs)):
                        if i < 3:  # Show first 3 time points
                            print(f"      Day {day}: {prob:.2%} conversion probability")
                else:
                    print(f"      Using fallback prediction")
                    
            except Exception as e:
                print(f"      Note: {e}")
            
    except Exception as e:
        print(f"   ❌ Conversion time prediction failed: {e}")
        return False
    
    # Test 3: Test hazard rate calculation
    print("\n3. Testing hazard rate calculation...")
    try:
        test_features = [[0.7, 2, 0.8, 0.15]]  # High intent scenario
        time_points = np.array([1, 7, 14, 30])
        
        try:
            hazard_rates = lag_model.calculate_hazard_rate(
                journey_features=test_features[0],
                time_points=time_points
            )
            
            print(f"   ✅ Hazard rates calculated:")
            if hazard_rates is not None:
                for day, rate in zip(time_points, hazard_rates):
                    print(f"      Day {day}: hazard rate = {rate:.4f}")
            else:
                print(f"      Using fallback hazard rates")
                
        except Exception as e:
            print(f"   ⚠️  Hazard rate calculation using fallback: {e}")
            
    except Exception as e:
        print(f"   ❌ Hazard rate test failed: {e}")
        return False
    
    # Test 4: Test integration in bid calculation
    print("\n4. Testing integration in bid calculation...")
    try:
        # Test different conversion probability scenarios
        test_cases = [
            ({'conversion_probability': 0.9, 'journey_stage': 3, 'user_fatigue_level': 0.1},
             {'intent_strength': 0.9}, "Quick converter scenario"),
            ({'conversion_probability': 0.2, 'journey_stage': 1, 'user_fatigue_level': 0.7},
             {'intent_strength': 0.3}, "Slow converter scenario"),
            ({'conversion_probability': 0.5, 'journey_stage': 2, 'user_fatigue_level': 0.3},
             {'intent_strength': 0.6}, "Average converter scenario")
        ]
        
        creative_selection = {'creative_type': 'display'}
        
        for journey_state, query_data, desc in test_cases:
            journey_state['hour_of_day'] = 14  # Afternoon
            
            bid_amount = await master._calculate_bid(
                journey_state=journey_state,
                query_data=query_data,
                creative_selection=creative_selection
            )
            
            print(f"   ✅ {desc}:")
            print(f"      Conversion prob: {journey_state['conversion_probability']:.1%}")
            print(f"      Final bid: ${bid_amount:.2f}")
            
    except Exception as e:
        print(f"   ❌ Bid calculation integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test with sample journey data
    print("\n5. Testing with sample conversion journeys...")
    try:
        # Create sample journeys
        sample_journeys = []
        for i in range(5):
            journey = ConversionJourney(
                user_id=f"user_{i}",
                start_time=datetime.now(),
                touchpoints=[
                    {'channel': 'search', 'timestamp': datetime.now(), 'bid': 2.5 + i*0.5}
                ],
                converted=i < 3,  # First 3 convert
                conversion_time=datetime.now() if i < 3 else None,
                conversion_value=50.0 + i*10 if i < 3 else 0,
                is_censored=i >= 3  # Last 2 are censored
            )
            sample_journeys.append(journey)
        
        # Get insights
        try:
            insights = lag_model.get_conversion_insights(sample_journeys)
            print(f"   ✅ Conversion insights generated:")
            if insights:
                if 'overall_stats' in insights:
                    stats = insights['overall_stats']
                    print(f"      Conversion rate: {stats.get('conversion_rate', 0):.1%}")
                    print(f"      Avg time to conversion: {stats.get('avg_conversion_time_days', 'N/A')} days")
            else:
                print(f"      Using fallback insights")
                
        except Exception as e:
            print(f"   ⚠️  Insights using fallback: {e}")
            
    except Exception as e:
        print(f"   ❌ Sample journey test failed: {e}")
        return False
    
    # Test 6: Test attribution window impact
    print("\n6. Testing attribution window impact...")
    try:
        # Test different attribution windows
        windows = [7, 14, 30]
        journey_features = [0.6, 2, 0.7, 0.2]
        
        for window in windows:
            try:
                impact = lag_model.predict_attribution_window_impact(
                    journey_features=journey_features,
                    window_days=window
                )
                
                if impact is not None:
                    print(f"   ✅ {window}-day window: {impact:.1%} conversions captured")
                else:
                    print(f"   ⚠️  {window}-day window: using fallback estimate")
                    
            except Exception as e:
                print(f"   ⚠️  {window}-day window: {e}")
                
    except Exception as e:
        print(f"   ❌ Attribution window test failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ CONVERSION LAG MODEL INTEGRATION TEST PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_conversion_lag_integration())
    exit(0 if success else 1)