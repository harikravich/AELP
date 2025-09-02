#!/usr/bin/env python3
"""
Test with realistic bid ranges based on what we'd actually bid
"""

import sys
sys.path.append('/home/hariravichandran/AELP')

from fixed_auction_system import FixedAuctionSystem
import numpy as np

def test_realistic():
    """Test with bid ranges the RL agent would discover"""
    
    auction = FixedAuctionSystem()
    
    # More realistic bid ranges based on CPCs
    scenarios = [
        {"name": "Brand search", "hour": 10, "intent": "brand", "bid": 3.0},
        {"name": "General non-brand", "hour": 14, "intent": "research", "bid": 5.5},
        {"name": "Competitor keywords", "hour": 15, "intent": "purchase", "bid": 8.0},
        {"name": "Crisis evening", "hour": 21, "intent": "crisis", "bid": 12.0},
        {"name": "Crisis late night", "hour": 2, "intent": "crisis", "bid": 15.0},
    ]
    
    print("=" * 80)
    print("REALISTIC BIDDING SCENARIOS")
    print("=" * 80)
    
    total_wins = 0
    total_auctions = 0
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Bid: ${scenario['bid']:.2f}")
        
        wins = 0
        spend = 0
        for _ in range(100):
            # Vary quality score realistically (6-8 range)
            quality = np.random.uniform(6.0, 8.0)
            
            result = auction.run_auction(
                our_bid=scenario['bid'],
                quality_score=quality,
                context={
                    'hour': scenario['hour'],
                    'query_intent': scenario['intent'],
                    'device_type': np.random.choice(['mobile', 'desktop'])
                }
            )
            
            if result.won:
                wins += 1
                spend += result.price_paid
            
            total_wins += result.won
            total_auctions += 1
        
        win_rate = wins / 100
        avg_cpc = spend / wins if wins > 0 else 0
        
        print(f"  Win rate: {win_rate:.1%}")
        if wins > 0:
            print(f"  Avg CPC: ${avg_cpc:.2f}")
    
    overall_win_rate = total_wins / total_auctions
    print("\n" + "=" * 80)
    print(f"OVERALL WIN RATE: {overall_win_rate:.1%}")
    print("=" * 80)
    
    if 0.25 <= overall_win_rate <= 0.50:
        print("✅ Win rate is REALISTIC for training!")
    elif overall_win_rate < 0.25:
        print("❌ Win rate too low - agent won't learn")
    else:
        print("⚠️ Win rate too high - reduce bids or increase competition")

if __name__ == "__main__":
    np.random.seed(42)
    test_realistic()
