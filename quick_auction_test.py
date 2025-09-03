#!/usr/bin/env python3
"""
Quick test to find the competitive bidding sweet spot
"""

import numpy as np
from auction_gym_integration_fixed import FixedAuctionGymIntegration

def test_bid_range():
    """Test different bid levels to find competitive balance"""
    auction = FixedAuctionGymIntegration()
    
    test_bids = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    print("\nQuick Auction Balance Test:")
    print("="*50)
    
    for bid in test_bids:
        wins = 0
        tests = 10  # Quick test
        
        for i in range(tests):
            query_context = {
                'query_value': bid * 1.5,
                'user_segment': np.random.randint(0, 3),
                'device_type': np.random.randint(0, 3), 
                'channel_index': np.random.randint(0, 3),
                'stage': np.random.randint(0, 5),
                'touchpoints': np.random.randint(1, 8),
                'competition_level': np.random.uniform(0.4, 0.8),
                'hour': np.random.randint(9, 18),  # Business hours
                'cvr': 0.02,
                'ltv': 199.98
            }
            
            result = auction.run_auction(bid, query_context)
            if result['won']:
                wins += 1
        
        win_rate = wins / tests
        print(f"${bid:4.1f}: {win_rate:5.1%} win rate ({wins}/{tests} wins)")
        
        # Good range is 15-35%
        if 0.15 <= win_rate <= 0.35:
            print(f"      ✅ GOOD competitive balance")
        elif win_rate < 0.15:
            print(f"      📉 Too competitive")
        elif win_rate > 0.50:
            print(f"      📈 Not competitive enough")
    
    print("\nLooking for bids with 15-35% win rate for realistic competition")

if __name__ == "__main__":
    test_bid_range()