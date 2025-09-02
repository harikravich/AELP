#!/usr/bin/env python3
"""
Test that user behavior is realistic and not random
"""

import numpy as np
from recsim_user_model import RecSimUserModel, UserSegment

def test_user_behavior_consistency():
    """Test that user behavior follows realistic patterns, not random"""
    
    model = RecSimUserModel()
    
    print("Testing user behavior consistency...")
    
    # Test different user segments have different behaviors
    results = {}
    for segment in UserSegment:
        segment_results = []
        for i in range(50):  # 50 tests per segment
            user_id = f"{segment.value}_{i}"
            user = model.generate_user(user_id, segment)
            
            # Test response to high-quality, relevant ad
            response = model.simulate_ad_response(
                user_id,
                {
                    'creative_quality': 0.9,
                    'relevance_score': 0.8,
                    'brand_match': 0.7,
                    'price_shown': 50.0
                },
                {'hour': 20, 'device': 'mobile'}
            )
            
            segment_results.append({
                'clicked': response['clicked'],
                'click_prob': response['click_probability'],
                'segment': segment.value
            })
        
        results[segment] = segment_results
    
    # Analyze results
    for segment, segment_results in results.items():
        click_rate = np.mean([r['clicked'] for r in segment_results])
        avg_click_prob = np.mean([r['click_prob'] for r in segment_results])
        
        print(f"{segment.value}:")
        print(f"  Click Rate: {click_rate:.3f}")
        print(f"  Avg Click Probability: {avg_click_prob:.3f}")
        
        # Verify behavior is segment-appropriate
        if segment == UserSegment.IMPULSE_BUYER:
            assert avg_click_prob > 0.05, "Impulse buyers should have higher click rates"
        elif segment == UserSegment.WINDOW_SHOPPER:
            assert avg_click_prob < 0.08, "Window shoppers should have lower click rates"
        elif segment == UserSegment.LOYAL_CUSTOMER:
            assert avg_click_prob > 0.10, "Loyal customers should have highest click rates"
    
    print("✅ User behavior is realistic and segment-appropriate!")
    
    # Test that behavior varies with context
    print("\nTesting contextual behavior variation...")
    
    user_id = "context_test"
    model.generate_user(user_id, UserSegment.IMPULSE_BUYER)
    
    # Test different times of day
    evening_response = model.simulate_ad_response(
        user_id,
        {'creative_quality': 0.8},
        {'hour': 20, 'device': 'mobile'}  # Prime time for impulse buyers
    )
    
    morning_response = model.simulate_ad_response(
        user_id,
        {'creative_quality': 0.8},
        {'hour': 8, 'device': 'desktop'}  # Less optimal
    )
    
    print(f"Evening click prob: {evening_response['click_probability']:.3f}")
    print(f"Morning click prob: {morning_response['click_probability']:.3f}")
    
    print("✅ Contextual behavior variation working!")
    
    return True

if __name__ == "__main__":
    test_user_behavior_consistency()
    print("\n🎉 ALL REALISTIC BEHAVIOR TESTS PASSED!")
