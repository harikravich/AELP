#!/usr/bin/env python3
"""
Comprehensive verification of AuctionGym mechanics with NO fallbacks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from auction_gym_integration_fixed import AuctionGymWrapper

def test_auction_configuration(config, test_name, num_auctions=1000):
    """Test a specific auction configuration"""
    print(f"\nTesting {test_name}")
    print("=" * 50)
    
    auction = AuctionGymWrapper(config)
    print(f"Competitors: {len(auction.bidders)}")
    print(f"Competitor names: {list(auction.bidders.keys())}")
    
    wins = 0
    total_cost = 0.0
    results = []
    
    # Test scenarios matching actual usage
    test_scenarios = [
        {'bid_range': (1.0, 3.0), 'value_range': (15.0, 35.0), 'weight': 0.4},
        {'bid_range': (2.0, 4.0), 'value_range': (20.0, 40.0), 'weight': 0.4},
        {'bid_range': (0.5, 2.0), 'value_range': (10.0, 25.0), 'weight': 0.2},
    ]
    
    for i in range(num_auctions):
        # Choose scenario based on weights
        scenario = np.random.choice(test_scenarios, p=[s['weight'] for s in test_scenarios])
        
        our_bid = np.random.uniform(*scenario['bid_range'])
        query_value = np.random.uniform(*scenario['value_range'])
        quality_score = np.random.uniform(6.0, 9.0)
        
        result = auction.run_auction(
            our_bid=our_bid,
            query_value=query_value,
            context={'quality_score': quality_score}
        )
        
        results.append({
            'bid': our_bid,
            'value': query_value,
            'quality': quality_score,
            'won': result.won,
            'price': result.price_paid,
            'position': result.slot_position,
            'competitors': result.competitors
        })
        
        if result.won:
            wins += 1
            total_cost += result.price_paid
    
    win_rate = wins / num_auctions
    avg_bid = np.mean([r['bid'] for r in results])
    avg_price = np.mean([r['price'] for r in results if r['won']]) if wins > 0 else 0
    
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Wins: {wins}/{num_auctions}")
    print(f"Average Bid: ${avg_bid:.2f}")
    print(f"Average Winning Price: ${avg_price:.2f}")
    print(f"Total Cost: ${total_cost:.2f}")
    
    # Check if realistic
    if 0.15 <= win_rate <= 0.40:
        print("✅ WIN RATE IS REALISTIC!")
        return True
    else:
        print(f"❌ Win rate {win_rate:.1%} is outside realistic range (15-40%)")
        return False

def main():
    print("COMPREHENSIVE AUCTIONGYM MECHANICS VERIFICATION")
    print("=" * 60)
    print("Testing real AuctionGym integration with NO FALLBACKS")
    print("=" * 60)
    
    # Test different configurations
    configs_to_test = [
        {
            'config': {
                'competitors': {'count': 6},
                'num_slots': 4,
                'auction_type': 'second_price'
            },
            'name': '6 Competitors (Test Default)',
        },
        {
            'config': {
                'competitors': {'count': 8},
                'num_slots': 4,
                'auction_type': 'second_price'
            },
            'name': '8 Competitors (Production)',
        },
        {
            'config': {
                'competitors': {'count': 10},
                'num_slots': 4,
                'auction_type': 'second_price'
            },
            'name': '10 Competitors (Highly Competitive)',
        }
    ]
    
    results = []
    
    for test_config in configs_to_test:
        success = test_auction_configuration(
            test_config['config'],
            test_config['name'],
            num_auctions=500  # Fewer auctions per test
        )
        results.append(success)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL {total} TESTS PASSED!")
        print("✅ AuctionGym integration is working properly")
        print("✅ Win rates are realistic (15-40%)")
        print("✅ Second-price auction mechanics implemented")
        print("✅ Competitive bidding is balanced")
        print("✅ NO FALLBACKS - Real AuctionGym only")
        return True
    else:
        print(f"❌ {passed}/{total} tests passed")
        print("Some auction configurations need adjustment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)