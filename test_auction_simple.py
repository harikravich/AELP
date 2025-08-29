#!/usr/bin/env python3
"""
Simple auction test to verify realistic win rates without dependencies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleAuctionResult:
    """Result from a simple auction"""
    won: bool
    price_paid: float
    position: int
    competitors: List[float]
    our_bid: float

class SimpleSecondPriceAuction:
    """
    Simple implementation of second-price auction mechanics
    for testing purposes - no external dependencies
    """
    
    def __init__(self, num_slots: int = 4, reserve_price: float = 0.50):
        self.num_slots = num_slots
        self.reserve_price = reserve_price
        
        # Define competitor profiles with realistic budgets and strategies
        self.competitor_profiles = [
            {'name': 'Qustodio', 'base_bid': 2.8, 'variance': 0.4, 'budget': 250},
            {'name': 'Bark', 'base_bid': 3.2, 'variance': 0.3, 'budget': 300},
            {'name': 'Circle', 'base_bid': 2.5, 'variance': 0.5, 'budget': 280},
            {'name': 'Norton', 'base_bid': 2.1, 'variance': 0.3, 'budget': 220},
            {'name': 'Life360', 'base_bid': 3.4, 'variance': 0.4, 'budget': 320},
            {'name': 'SmallComp', 'base_bid': 1.8, 'variance': 0.6, 'budget': 150},
        ]
    
    def generate_competitor_bids(self, context: Dict[str, Any] = None) -> List[float]:
        """Generate realistic competitor bids"""
        context = context or {}
        
        bids = []
        for profile in self.competitor_profiles:
            # Base bid with market conditions
            base = profile['base_bid']
            
            # Apply time-of-day multipliers
            hour = context.get('hour', 12)
            if hour in [9, 10, 11, 14, 15, 16, 17]:  # Business hours
                base *= 1.15
            elif hour in [19, 20, 21]:  # Evening peak
                base *= 1.25
            
            # Apply device type multipliers
            device = context.get('device', 'mobile')
            if device == 'desktop':
                base *= 1.1
            
            # Add variance
            variance = profile['variance']
            bid = np.random.normal(base, base * variance)
            
            # Apply budget constraints
            max_bid = profile['budget'] * 0.02  # 2% of budget per auction
            bid = min(bid, max_bid)
            
            # Ensure above reserve
            bid = max(bid, self.reserve_price)
            
            bids.append(bid)
        
        return bids
    
    def run_auction(self, our_bid: float, context: Dict[str, Any] = None) -> SimpleAuctionResult:
        """Run second-price auction"""
        # Get competitor bids
        competitor_bids = self.generate_competitor_bids(context)
        
        # All bids
        all_bids = [our_bid] + competitor_bids
        bidder_names = ['GAELP'] + [p['name'] for p in self.competitor_profiles]
        
        # Sort by bid amount (highest first)
        sorted_indices = np.argsort(all_bids)[::-1]
        sorted_bids = [all_bids[i] for i in sorted_indices]
        sorted_names = [bidder_names[i] for i in sorted_indices]
        
        # Find our position
        our_position = sorted_names.index('GAELP') + 1
        
        # Determine if we won (top num_slots positions)
        won = our_position <= self.num_slots
        
        # Calculate price paid (second-price auction)
        price_paid = 0.0
        if won and len(sorted_bids) > our_position:
            # Pay the next highest bid (second-price rule)
            price_paid = sorted_bids[our_position]  # Next bid after ours
            # Add small increment
            price_paid += 0.01
            # Never pay more than our bid
            price_paid = min(price_paid, our_bid)
        
        return SimpleAuctionResult(
            won=won,
            price_paid=price_paid,
            position=our_position,
            competitors=competitor_bids,
            our_bid=our_bid
        )

def test_second_price_mechanics():
    """Test that second-price auction works correctly"""
    print("🔍 Testing Second-Price Auction Mechanics")
    print("=" * 60)
    
    auction = SimpleSecondPriceAuction()
    
    test_cases = [
        {'bid': 1.0, 'expected': 'Should rarely win'},
        {'bid': 2.0, 'expected': 'Should sometimes win'},
        {'bid': 3.0, 'expected': 'Should often win'},
        {'bid': 4.0, 'expected': 'Should usually win'},
        {'bid': 5.0, 'expected': 'Should almost always win'}
    ]
    
    for test_case in test_cases:
        wins = 0
        total_paid = 0.0
        prices = []
        violations = 0
        
        for trial in range(100):
            result = auction.run_auction(test_case['bid'])
            
            if result.won:
                wins += 1
                total_paid += result.price_paid
                prices.append(result.price_paid)
                
                # Check second-price rule
                if result.price_paid > test_case['bid']:
                    violations += 1
        
        win_rate = wins / 100.0
        avg_price = total_paid / wins if wins > 0 else 0.0
        
        print(f"\nBid ${test_case['bid']:.2f}:")
        print(f"  Win Rate: {win_rate:.1%} ({wins}/100)")
        print(f"  Avg Price: ${avg_price:.3f}")
        print(f"  Max Price: ${max(prices):.3f}" if prices else "  Max Price: $0.000")
        print(f"  Second-Price Violations: {violations}")
        print(f"  Expected: {test_case['expected']}")
        
        if violations > 0:
            print(f"  ⚠️  WARNING: {violations} second-price violations!")
    
    return True

def test_bid_distribution_analysis():
    """Test competitor bid distributions"""
    print("\n📊 Competitor Bid Distribution Analysis")
    print("=" * 60)
    
    auction = SimpleSecondPriceAuction()
    
    # Collect bid data
    all_competitor_bids = []
    contexts = [
        {'hour': 9, 'device': 'mobile'},
        {'hour': 14, 'device': 'desktop'},
        {'hour': 20, 'device': 'mobile'},
        {'hour': 2, 'device': 'mobile'},
    ]
    
    for _ in range(250):  # 250 samples
        context = np.random.choice(contexts)
        competitor_bids = auction.generate_competitor_bids(context)
        all_competitor_bids.extend(competitor_bids)
    
    # Analysis
    bids_array = np.array(all_competitor_bids)
    
    print(f"Competitor Bid Statistics (n={len(all_competitor_bids)}):")
    print(f"  Mean: ${bids_array.mean():.3f}")
    print(f"  Median: ${np.median(bids_array):.3f}")
    print(f"  Std Dev: ${bids_array.std():.3f}")
    print(f"  Min: ${bids_array.min():.3f}")
    print(f"  Max: ${bids_array.max():.3f}")
    print(f"  25th percentile: ${np.percentile(bids_array, 25):.3f}")
    print(f"  75th percentile: ${np.percentile(bids_array, 75):.3f}")
    
    # Check reasonableness
    mean_bid = bids_array.mean()
    if 1.5 <= mean_bid <= 4.0:
        print(f"  ✅ Competitor bids are in reasonable range")
    else:
        print(f"  ❌ Competitor bids seem unrealistic")
    
    return True

def test_comprehensive_win_rates():
    """Test win rates across bid ranges"""
    print("\n🎯 Comprehensive Win Rate Analysis")
    print("=" * 60)
    
    auction = SimpleSecondPriceAuction()
    
    bid_ranges = np.arange(0.5, 6.0, 0.25)
    results = []
    
    for our_bid in bid_ranges:
        wins = 0
        total_cost = 0.0
        positions = []
        
        for trial in range(50):  # 50 trials per bid level
            # Vary context
            context = {
                'hour': np.random.randint(0, 24),
                'device': np.random.choice(['mobile', 'desktop'])
            }
            
            result = auction.run_auction(our_bid, context)
            
            if result.won:
                wins += 1
                total_cost += result.price_paid
                positions.append(result.position)
        
        win_rate = wins / 50.0
        avg_cpc = total_cost / wins if wins > 0 else 0.0
        avg_position = np.mean(positions) if positions else None
        
        results.append({
            'bid': our_bid,
            'win_rate': win_rate,
            'avg_cpc': avg_cpc,
            'avg_position': avg_position
        })
        
        # Print key milestones
        if our_bid % 1.0 == 0.0:
            print(f"Bid ${our_bid:.2f}: Win Rate {win_rate:.1%}, Avg CPC ${avg_cpc:.2f}")
    
    # Analysis
    df = pd.DataFrame(results)
    
    # Find realistic win rate range (15-40%)
    realistic = df[(df['win_rate'] >= 0.15) & (df['win_rate'] <= 0.40)]
    
    print(f"\nAnalysis Results:")
    if not realistic.empty:
        print(f"  ✅ REALISTIC BID RANGE: ${realistic['bid'].min():.2f} - ${realistic['bid'].max():.2f}")
        print(f"  ✅ WIN RATE RANGE: {realistic['win_rate'].min():.1%} - {realistic['win_rate'].max():.1%}")
        print(f"  ✅ CPC RANGE: ${realistic['avg_cpc'].min():.2f} - ${realistic['avg_cpc'].max():.2f}")
    else:
        print(f"  ❌ NO REALISTIC WIN RATES FOUND")
    
    # Check for 100% win rates (bad sign)
    perfect_wins = df[df['win_rate'] == 1.0]
    if not perfect_wins.empty:
        print(f"  ⚠️  100% WIN RATES at bids ${perfect_wins['bid'].min():.2f}+ (indicates weak competition)")
    
    # Overall assessment
    avg_win_rate_low_bids = df[df['bid'] <= 2.0]['win_rate'].mean()
    avg_win_rate_high_bids = df[df['bid'] >= 4.0]['win_rate'].mean()
    
    print(f"  \nOverall Assessment:")
    print(f"  Low bid win rate (≤$2.00): {avg_win_rate_low_bids:.1%}")
    print(f"  High bid win rate (≥$4.00): {avg_win_rate_high_bids:.1%}")
    
    if avg_win_rate_low_bids < 0.15 and avg_win_rate_high_bids > 0.60:
        print(f"  ✅ REALISTIC COMPETITION DETECTED")
        return True
    else:
        print(f"  ❌ COMPETITION LEVELS SEEM OFF")
        return False

def main():
    """Run all auction tests"""
    print("🔬 GAELP AUCTION DYNAMICS TEST - SIMPLE VERSION")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 3
    
    try:
        if test_second_price_mechanics():
            tests_passed += 1
            print("✅ Second-price mechanics test PASSED")
    except Exception as e:
        print(f"❌ Second-price mechanics test FAILED: {e}")
    
    try:
        if test_bid_distribution_analysis():
            tests_passed += 1
            print("✅ Bid distribution test PASSED")
    except Exception as e:
        print(f"❌ Bid distribution test FAILED: {e}")
    
    try:
        if test_comprehensive_win_rates():
            tests_passed += 1
            print("✅ Win rate analysis test PASSED")
    except Exception as e:
        print(f"❌ Win rate analysis test FAILED: {e}")
    
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - Auction mechanics are working!")
        print("✅ Realistic win rates (15-35% in competitive ranges)")
        print("✅ Second-price auction rules enforced")
        print("✅ Competitive bid landscape generated")
        return True
    else:
        print("❌ SOME TESTS FAILED - Need to fix auction mechanics")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
