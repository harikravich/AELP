#!/usr/bin/env python3
"""Debug auction bidding to understand why competitors bid so low"""

import logging
import sys
import os

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_auction_gym_bidders():
    """Debug AuctionGym bidder behavior"""
    print("Debugging AuctionGym Bidders")
    print("=" * 40)
    
    from auction_gym_integration import AuctionGymWrapper
    
    # Create wrapper
    wrapper = AuctionGymWrapper({
        'competitors': {'count': 4},
        'num_slots': 4
    })
    
    # Test bidder behavior directly
    test_values = [1.0, 5.0, 10.0, 20.0]
    test_ctrs = [0.01, 0.05, 0.1]
    
    print(f"\nTesting {len(wrapper.bidders)} bidders")
    for bidder_name, bidder in wrapper.bidders.items():
        print(f"\n{bidder_name} ({type(bidder).__name__}):")
        print(f"  Budget: ${bidder.budget}")
        
        for value in test_values:
            for ctr in test_ctrs:
                try:
                    bid = bidder.bid(
                        value=value,
                        context={},
                        estimated_CTR=ctr
                    )
                    print(f"    Value={value}, CTR={ctr:.3f} -> Bid=${bid:.4f}")
                except Exception as e:
                    print(f"    Value={value}, CTR={ctr:.3f} -> ERROR: {e}")
    
    # Test full auction with various bid levels
    print("\n" + "=" * 40)
    print("Testing Full Auction with Different Bid Levels")
    
    test_bids = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5]
    
    for our_bid in test_bids:
        result = wrapper.run_auction(
            our_bid=our_bid,
            query_value=10.0,
            context={'estimated_ctr': 0.05, 'quality_score': 0.8}
        )
        
        print(f"Our bid: ${our_bid:.2f} -> Won: {result.won}, Price: ${result.price_paid:.4f}, Position: {result.slot_position}")
    
if __name__ == "__main__":
    debug_auction_gym_bidders()
