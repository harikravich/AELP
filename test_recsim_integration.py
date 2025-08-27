#!/usr/bin/env python3
"""
Test script for RecSim integration with GAELP.
Demonstrates the different user segment behaviors and ad response patterns.
"""

import numpy as np
import logging
from typing import Dict, List
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_installation():
    """Test if RecSim is properly installed and integrated"""
    
    print("Testing RecSim Integration")
    print("=" * 50)
    
    try:
        from recsim_user_model import RecSimUserModel, UserSegment
        print("✓ RecSim user model imported successfully")
        
        from enhanced_simulator import EnhancedGAELPEnvironment, RECSIM_AVAILABLE
        print(f"✓ Enhanced simulator imported successfully (RecSim available: {RECSIM_AVAILABLE})")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def compare_user_segments():
    """Compare behavior across different user segments"""
    
    try:
        from recsim_user_model import RecSimUserModel, UserSegment
    except ImportError:
        print("RecSim not available - skipping segment comparison")
        return
    
    print("\nUser Segment Comparison")
    print("=" * 50)
    
    model = RecSimUserModel()
    
    # Define test scenarios
    test_ads = [
        {
            'name': 'Luxury Product',
            'content': {
                'creative_quality': 0.9,
                'price_shown': 500.0,
                'brand_match': 0.9,
                'relevance_score': 0.8,
                'product_id': 'luxury_item'
            }
        },
        {
            'name': 'Budget Product',
            'content': {
                'creative_quality': 0.5,
                'price_shown': 15.0,
                'brand_match': 0.3,
                'relevance_score': 0.6,
                'product_id': 'budget_item'
            }
        },
        {
            'name': 'Mid-Range Brand',
            'content': {
                'creative_quality': 0.7,
                'price_shown': 75.0,
                'brand_match': 0.7,
                'relevance_score': 0.8,
                'product_id': 'mid_range_item'
            }
        }
    ]
    
    contexts = [
        {'name': 'Peak Hours Mobile', 'context': {'hour': 20, 'device': 'mobile'}},
        {'name': 'Work Hours Desktop', 'context': {'hour': 14, 'device': 'desktop'}},
        {'name': 'Late Night Tablet', 'context': {'hour': 23, 'device': 'tablet'}}
    ]
    
    # Test each segment
    results = {}
    
    for segment in UserSegment:
        print(f"\nTesting {segment.value.upper()}:")
        segment_results = {}
        
        for ad_scenario in test_ads:
            ad_results = {}
            
            for context_scenario in contexts:
                # Run multiple simulations
                responses = []
                
                for i in range(100):
                    user_id = f"{segment.value}_{ad_scenario['name']}_{context_scenario['name']}_{i}"
                    model.generate_user(user_id, segment)
                    
                    response = model.simulate_ad_response(
                        user_id=user_id,
                        ad_content=ad_scenario['content'],
                        context=context_scenario['context']
                    )
                    responses.append(response)
                
                # Calculate metrics
                clicks = [r['clicked'] for r in responses]
                conversions = [r['converted'] for r in responses if r['clicked']]
                revenues = [r['revenue'] for r in responses]
                time_spent = [r['time_spent'] for r in responses if r['clicked']]
                
                metrics = {
                    'ctr': np.mean(clicks),
                    'conversion_rate': np.mean(conversions) if conversions else 0,
                    'avg_revenue': np.mean(revenues),
                    'avg_time_spent': np.mean(time_spent) if time_spent else 0
                }
                
                ad_results[context_scenario['name']] = metrics
            
            segment_results[ad_scenario['name']] = ad_results
        
        results[segment.value] = segment_results
        
        # Print summary for this segment
        for ad_name, ad_data in segment_results.items():
            print(f"  {ad_name}:")
            best_context = max(ad_data.keys(), key=lambda k: ad_data[k]['ctr'])
            best_metrics = ad_data[best_context]
            print(f"    Best performing context: {best_context}")
            print(f"    CTR: {best_metrics['ctr']:.3f}, Conv Rate: {best_metrics['conversion_rate']:.3f}")
            print(f"    Avg Revenue: ${best_metrics['avg_revenue']:.2f}")
    
    return results


def test_fatigue_and_interest():
    """Test user fatigue and interest level changes"""
    
    try:
        from recsim_user_model import RecSimUserModel, UserSegment
    except ImportError:
        print("RecSim not available - skipping fatigue test")
        return
    
    print("\nTesting User Fatigue and Interest Dynamics")
    print("=" * 50)
    
    model = RecSimUserModel()
    
    # Generate a user and show them multiple ads
    user_id = "fatigue_test_user"
    model.generate_user(user_id, UserSegment.IMPULSE_BUYER)
    
    ad_content = {
        'creative_quality': 0.7,
        'price_shown': 50.0,
        'brand_match': 0.6,
        'relevance_score': 0.7,
        'product_id': 'test_product'
    }
    
    context = {'hour': 19, 'device': 'mobile'}
    
    print("Showing 10 ads to the same user to observe fatigue effects:")
    
    for i in range(10):
        response = model.simulate_ad_response(user_id, ad_content, context)
        user = model.current_users[user_id]
        
        print(f"  Ad {i+1}: Clicked={response['clicked']}, "
              f"Fatigue={user.fatigue_level:.2f}, "
              f"Interest={user.current_interest:.2f}")


def test_enhanced_environment_integration():
    """Test the full enhanced environment with RecSim"""
    
    print("\nTesting Enhanced Environment Integration")
    print("=" * 50)
    
    from enhanced_simulator import EnhancedGAELPEnvironment, RECSIM_AVAILABLE
    
    env = EnhancedGAELPEnvironment()
    obs = env.reset()
    
    print(f"RecSim Integration Active: {RECSIM_AVAILABLE}")
    
    # Test with different ad strategies
    strategies = [
        {
            'name': 'Premium Strategy',
            'action': {
                'bid': 3.0,
                'budget': 1000,
                'quality_score': 0.9,
                'creative': {
                    'quality_score': 0.9,
                    'price_shown': 200.0,
                    'brand_affinity': 0.8,
                    'relevance': 0.9,
                    'product_id': 'premium_product'
                }
            }
        },
        {
            'name': 'Value Strategy',
            'action': {
                'bid': 1.5,
                'budget': 1000,
                'quality_score': 0.7,
                'creative': {
                    'quality_score': 0.6,
                    'price_shown': 30.0,
                    'brand_affinity': 0.4,
                    'relevance': 0.7,
                    'product_id': 'value_product'
                }
            }
        }
    ]
    
    for i, strategy in enumerate(strategies * 5):  # Test each strategy 5 times
        obs, reward, done, info = env.step(strategy['action'])
        
        if i % 3 == 0:  # Print every 3rd step
            print(f"Step {i+1} ({strategy['name']}): "
                  f"ROAS={obs['roas']:.2f}, "
                  f"Clicks={obs['clicks']}, "
                  f"Conversions={obs['conversions']}")
        
        if done:
            break
    
    print(f"\nFinal Performance:")
    print(f"  Total Cost: ${obs['total_cost']:.2f}")
    print(f"  Total Revenue: ${obs['total_revenue']:.2f}")
    print(f"  Final ROAS: {obs['roas']:.2f}x")
    print(f"  CTR: {obs['clicks'] / max(obs['impressions'], 1):.3f}")
    print(f"  Conversion Rate: {obs['conversions'] / max(obs['clicks'], 1):.3f}")
    
    # Show analytics if available
    if hasattr(env.user_model, 'get_user_analytics'):
        analytics = env.user_model.get_user_analytics()
        if analytics and 'segment_breakdown' in analytics:
            print(f"\nUser Segment Performance:")
            for segment, stats in analytics['segment_breakdown'].items():
                if stats['total_interactions'] > 0:
                    print(f"  {segment}: {stats['total_interactions']} interactions, "
                          f"{stats['click_rate']:.3f} CTR")


def main():
    """Run all tests"""
    
    start_time = time.time()
    
    print("GAELP RecSim Integration Test Suite")
    print("=" * 60)
    
    if not test_installation():
        print("Installation test failed. Please run: python install_recsim.py")
        return
    
    try:
        compare_user_segments()
        test_fatigue_and_interest()
        test_enhanced_environment_integration()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()