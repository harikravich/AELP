#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/hariravichandran/AELP')
from auction_gym_integration_fixed import AuctionGymWrapper
import numpy as np

print('Testing Balanced AuctionGym Integration')
print('=' * 40)

auction = AuctionGymWrapper({'competitors': {'count': 8}, 'num_slots': 4})

# Test with different bid ranges
test_scenarios = [
    {'bid': 1.0, 'query_value': 15.0, 'name': 'Low bid'},
    {'bid': 2.5, 'query_value': 25.0, 'name': 'Medium bid'},
    {'bid': 4.0, 'query_value': 35.0, 'name': 'High bid'},
]

for scenario in test_scenarios:
    wins = 0
    total_auctions = 200
    for i in range(total_auctions):
        result = auction.run_auction(
            our_bid=scenario['bid'], 
            query_value=scenario['query_value'],
            context={'quality_score': 7.0}
        )
        if result.won:
            wins += 1
    
    win_rate = wins / total_auctions
    print(f'{scenario["name"]} (${scenario["bid"]:.1f}): {win_rate:.1%} win rate')

print('\n✅ Win rates should be between 15-35% for realistic competition')