#!/usr/bin/env python3
"""
Test AuctionGym with parameter fix
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('Testing AuctionGym with parameter fix...')
from auction_gym_integration import AuctionGymWrapper, AUCTION_GYM_AVAILABLE

# Test that it's available
assert AUCTION_GYM_AVAILABLE == True, 'AuctionGym not available!'
print('✅ AuctionGym available')

# Test initialization
wrapper = AuctionGymWrapper()
print('✅ AuctionGym initialized')

# Test auction run
result = wrapper.run_auction(
    our_bid=2.50,
    query_value=3.00,
    context={'quality_score': 1.0, 'conversion_rate': 0.02, 'customer_ltv': 199.98}
)
print(f'✅ Auction ran successfully')
print(f'   Won: {result.won}, Price: ${result.price_paid:.2f}, Position: {result.slot_position}')