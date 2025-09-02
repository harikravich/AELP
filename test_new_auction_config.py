#!/usr/bin/env python3
"""
Test the updated auction configuration with realistic bids
"""

import sys
sys.path.append('/home/hariravichandran/AELP')

from fixed_auction_system import FixedAuctionSystem
import numpy as np

def test_auction_scenarios():
    """Test various auction scenarios with new config"""
    
    auction = FixedAuctionSystem()
    
    scenarios = [
        {"name": "Normal research", "hour": 10, "intent": "research", "bid": 8.0},
        {"name": "Evening family", "hour": 20, "intent": "purchase", "bid": 12.0},
        {"name": "Crisis midnight", "hour": 23, "intent": "crisis", "bid": 20.0},
        {"name": "Crisis 2am", "hour": 2, "intent": "crisis", "bid": 25.0},
        {"name": "Low competition", "hour": 7, "intent": "awareness", "bid": 5.0},
    ]
    
    print("=" * 80)
    print("TESTING NEW AUCTION CONFIGURATION")
    print("=" * 80)
    
    wins = 0
    total = 0
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Time: {scenario['hour']}:00")
        print(f"  Intent: {scenario['intent']}")
        print(f"  Our bid: ${scenario['bid']:.2f}")
        
        # Run 10 auctions for each scenario
        scenario_wins = 0
        for _ in range(10):
            result = auction.run_auction(
                our_bid=scenario['bid'],
                quality_score=7.5,  # Good quality (on 1-10 scale)
                context={
                    'hour': scenario['hour'],
                    'query_intent': scenario['intent'],
                    'device_type': 'desktop'
                }
            )
            if result.won:
                scenario_wins += 1
                wins += 1
            total += 1
        
        print(f"  Win rate: {scenario_wins}/10 ({scenario_wins*10}%)")
        
        # Show one example result
        result = auction.run_auction(
            our_bid=scenario['bid'],
            quality_score=7.5,
            context={
                'hour': scenario['hour'],
                'query_intent': scenario['intent'],
                'device_type': 'desktop'
            }
        )
        print(f"  Example: {'WON' if result.won else 'LOST'} - Position {result.position}")
        if result.won:
            print(f"  Paid: ${result.price_paid:.2f}")
    
    print("\n" + "=" * 80)
    print(f"OVERALL WIN RATE: {wins}/{total} ({wins*100/total:.1f}%)")
    print("=" * 80)
    
    print("\n🎯 TARGET: 30-50% win rate for healthy training")
    if wins/total < 0.3:
        print("❌ Win rate too low - increase bids or reduce competition")
    elif wins/total > 0.5:
        print("⚠️ Win rate too high - may not be realistic")
    else:
        print("✅ Win rate in healthy range!")

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    test_auction_scenarios()