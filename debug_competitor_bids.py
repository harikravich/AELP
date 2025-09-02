#!/usr/bin/env python3
"""
Debug what competitors are actually bidding
"""

import sys
sys.path.append('/home/hariravichandran/AELP')

from fixed_auction_system import FixedAuctionSystem
import numpy as np

def debug_bids():
    """See what competitors are actually bidding"""
    
    auction = FixedAuctionSystem()
    
    scenarios = [
        {"name": "Crisis at 2am", "hour": 2, "intent": "crisis"},
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Hour: {scenario['hour']}, Intent: {scenario['intent']}")
        print("-" * 60)
        
        # Get competitor bids
        context = {
            'hour': scenario['hour'],
            'query_intent': scenario['intent'],
            'device_type': 'desktop'
        }
        
        # Call the internal method to see bids
        competitor_bids = auction.generate_competitor_bids(context)
        
        # Sort and display
        competitor_bids.sort(reverse=True)
        
        print("Competitor bids:")
        for i, bid in enumerate(competitor_bids, 1):
            print(f"  {i}. ${bid:.2f}")
        
        print(f"\nTo win position 4, we need to bid > ${competitor_bids[3]:.2f}")
        print(f"To win position 1, we need to bid > ${competitor_bids[0]:.2f}")
        
        # Test with our max bid (quality score on 1-10 scale like competitors)
        our_bid = 25.0
        result = auction.run_auction(our_bid, 7.5, context)  # Use same scale as competitors!
        print(f"\nWith our bid of ${our_bid:.2f}:")
        print(f"  Result: {'WON' if result.won else 'LOST'}")
        print(f"  Position: {result.position}")
        if result.won:
            print(f"  Price paid: ${result.price_paid:.2f}")

if __name__ == "__main__":
    np.random.seed(42)
    debug_bids()