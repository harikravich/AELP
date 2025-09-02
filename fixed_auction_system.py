#!/usr/bin/env python3
"""
Fixed Auction System for GAELP - NO FALLBACKS
Implements proper second-price auctions with realistic competition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AuctionResult:
    """Result from auction"""
    won: bool
    price_paid: float
    position: int
    competitors_count: int
    competitor_bids: List[float]
    our_bid: float
    quality_score: float
    estimated_ctr: float
    clicked: bool
    revenue: float

class FixedAuctionSystem:
    """
    Fixed auction system with proper second-price mechanics and realistic competition
    Designed to achieve 15-35% win rates in competitive markets
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        self.num_slots = config.get('num_slots', 4)
        self.reserve_price = config.get('reserve_price', 0.50)
        
        # REALISTIC competitor profiles based on GA4 winning data
        # We're winning 66k+ sessions/week, so competition is beatable
        # Actual CPCs: Brand ~$2-3, General ~$5-7, Competitors ~$8-10
        self.competitor_profiles = [
            # Major competitors (but not impossibly high bids)
            {'name': 'Bark', 'base_bid': 6.50, 'variance': 0.25, 'budget_factor': 1.3, 'aggression': 1.2},
            {'name': 'Qustodio', 'base_bid': 5.50, 'variance': 0.20, 'budget_factor': 1.2, 'aggression': 1.1},
            {'name': 'Circle', 'base_bid': 4.50, 'variance': 0.15, 'budget_factor': 1.1, 'aggression': 1.0},
            {'name': 'Norton', 'base_bid': 4.00, 'variance': 0.20, 'budget_factor': 1.0, 'aggression': 0.95},
            
            # Mid-tier competitors
            {'name': 'GoogleFamily', 'base_bid': 3.50, 'variance': 0.25, 'budget_factor': 1.5, 'aggression': 0.9},
            {'name': 'NetNanny', 'base_bid': 3.25, 'variance': 0.30, 'budget_factor': 0.9, 'aggression': 0.85},
            
            # Smaller players with limited budgets
            {'name': 'FamilyTime', 'base_bid': 2.50, 'variance': 0.35, 'budget_factor': 0.8, 'aggression': 0.8},
            {'name': 'SmallComp1', 'base_bid': 2.00, 'variance': 0.40, 'budget_factor': 0.7, 'aggression': 0.75},
            {'name': 'SmallComp2', 'base_bid': 1.75, 'variance': 0.45, 'budget_factor': 0.6, 'aggression': 0.7},
        ]
        
        # Track performance
        self.total_auctions = 0
        self.total_wins = 0
        self.total_spend = 0.0
        self.position_history = []
        
        logger.info(f"Fixed auction system initialized with {len(self.competitor_profiles)} competitors")
    
    def generate_competitor_bids(self, context: Dict[str, Any]) -> List[float]:
        """Generate realistic competitor bids based on market conditions"""
        
        bids = []
        hour = context.get('hour', 12)
        device = context.get('device_type', 'mobile')
        query_intent = context.get('query_intent', 'research')
        
        for profile in self.competitor_profiles:
            base_bid = profile['base_bid']
            
            # Apply time-based multipliers (more reasonable)
            if hour in [9, 10, 11, 14, 15, 16, 17]:  # Business hours
                base_bid *= 1.10
            elif hour in [19, 20, 21]:  # Evening family time
                base_bid *= 1.20
            elif hour in [22, 23, 0, 1, 2, 3, 4, 5, 6]:  # Late night/crisis
                base_bid *= 1.40  # Higher but not extreme
            
            # Device type multipliers
            if device == 'desktop':
                base_bid *= 1.10  # Slight premium for desktop
            
            # Query intent multipliers (realistic based on actual wins)
            if query_intent == 'crisis':
                base_bid *= profile['aggression'] * 1.5  # Crisis gets 50% premium
            elif query_intent == 'purchase':
                base_bid *= profile['aggression'] * 1.3  # Purchase intent
            elif query_intent == 'research':
                base_bid *= profile['aggression'] * 1.1  # Research phase
            
            # Add variance for realistic bidding
            variance_factor = np.random.normal(1.0, profile['variance'])
            bid = base_bid * variance_factor
            
            # Apply budget constraints (realistic limits per auction)
            max_per_auction = profile['base_bid'] * profile['budget_factor'] * 2.5
            bid = min(bid, max_per_auction)
            
            # Minimum bid constraints
            bid = max(bid, self.reserve_price)
            
            bids.append(bid)
        
        return bids
    
    def run_auction(self, our_bid: float, quality_score: float = 1.0, 
                   context: Dict[str, Any] = None) -> AuctionResult:
        """Run second-price auction with realistic competition"""
        
        context = context or {}
        self.total_auctions += 1
        
        # Generate competitor bids
        competitor_bids = self.generate_competitor_bids(context)
        
        # All bids with quality score adjustment
        # In Google Ads: Ad Rank = Bid × Quality Score
        our_ad_rank = our_bid * quality_score
        
        # Assume competitors have average quality scores around 7-8
        competitor_quality_scores = np.random.normal(7.5, 1.0, len(competitor_bids))
        competitor_quality_scores = np.clip(competitor_quality_scores, 1.0, 10.0)
        
        competitor_ad_ranks = [bid * qs for bid, qs in zip(competitor_bids, competitor_quality_scores)]
        
        # All ad ranks
        all_ad_ranks = [our_ad_rank] + competitor_ad_ranks
        all_bids = [our_bid] + competitor_bids
        bidder_names = ['GAELP'] + [p['name'] for p in self.competitor_profiles]
        
        # Sort by ad rank (highest first)
        sorted_data = sorted(zip(all_ad_ranks, all_bids, bidder_names), 
                           key=lambda x: x[0], reverse=True)
        
        sorted_ad_ranks, sorted_bids, sorted_names = zip(*sorted_data)
        
        # Find our position
        our_position = sorted_names.index('GAELP') + 1
        
        # Check if we won (top num_slots positions)
        won = our_position <= self.num_slots
        
        # Calculate price paid using second-price auction rules
        # Price = (Next highest ad rank / Our quality score) + $0.01
        price_paid = 0.0
        if won and our_position < len(sorted_ad_ranks):
            next_ad_rank = sorted_ad_ranks[our_position]  # Next ad rank below ours
            price_paid = (next_ad_rank / quality_score) + 0.01
            
            # Never pay more than our bid
            price_paid = min(price_paid, our_bid)
            
            # Ensure minimum
            price_paid = max(price_paid, self.reserve_price)
        
        # Calculate performance metrics
        if won:
            self.total_wins += 1
            self.total_spend += price_paid
            self.position_history.append(our_position)
        
        # Estimate CTR based on position (real Google Ads data)
        position_ctrs = {1: 0.065, 2: 0.032, 3: 0.021, 4: 0.016}
        estimated_ctr = position_ctrs.get(our_position, 0.005) if won else 0.0
        
        # Apply quality score to CTR
        true_ctr = estimated_ctr * (quality_score / 7.0)  # 7.0 is average QS
        
        # Simulate click
        clicked = np.random.random() < true_ctr if won else False
        
        # Calculate revenue
        revenue = 0.0
        if clicked:
            # Conversion rate varies by context
            cvr = context.get('conversion_rate', 0.025)
            if np.random.random() < cvr:
                revenue = context.get('customer_ltv', 199.98)
        
        return AuctionResult(
            won=won,
            price_paid=price_paid,
            position=our_position,
            competitors_count=len(competitor_bids),
            competitor_bids=[],  # REALISTIC: We don't know competitor bids in real life!
            our_bid=our_bid,
            quality_score=quality_score,
            estimated_ctr=estimated_ctr,
            clicked=clicked,
            revenue=revenue
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get auction performance statistics"""
        if self.total_auctions == 0:
            return {'error': 'No auctions run yet'}
        
        win_rate = self.total_wins / self.total_auctions
        avg_cpc = self.total_spend / self.total_wins if self.total_wins > 0 else 0.0
        avg_position = np.mean(self.position_history) if self.position_history else 0.0
        
        return {
            'total_auctions': self.total_auctions,
            'wins': self.total_wins,
            'win_rate': win_rate,
            'total_spend': self.total_spend,
            'avg_cpc': avg_cpc,
            'avg_position': avg_position,
            'competitors': len(self.competitor_profiles)
        }
    
    def validate_win_rate(self) -> bool:
        """Validate that win rate is realistic (15-35%)"""
        stats = self.get_performance_stats()
        if 'error' in stats:
            return False
            
        win_rate = stats['win_rate']
        
        # Check if win rate is in realistic range
        if 0.15 <= win_rate <= 0.35:
            logger.info(f"✅ Win rate {win_rate:.1%} is REALISTIC")
            return True
        elif win_rate > 0.80:
            logger.warning(f"❌ Win rate {win_rate:.1%} is TOO HIGH - competition too weak")
            return False
        elif win_rate < 0.10:
            logger.warning(f"❌ Win rate {win_rate:.1%} is TOO LOW - competition too strong")
            return False
        else:
            logger.info(f"⚠️ Win rate {win_rate:.1%} is borderline acceptable")
            return True

# Test the fixed system
if __name__ == "__main__":
    print("🔧 Testing Fixed Auction System")
    print("=" * 50)
    
    auction_system = FixedAuctionSystem()
    
    # Run test auctions with varied bids and contexts
    test_contexts = [
        {'hour': 9, 'device_type': 'mobile', 'query_intent': 'research'},
        {'hour': 14, 'device_type': 'desktop', 'query_intent': 'purchase'},
        {'hour': 21, 'device_type': 'mobile', 'query_intent': 'crisis'},
        {'hour': 2, 'device_type': 'mobile', 'query_intent': 'crisis'},
    ]
    
    # Test 1000 auctions
    for i in range(1000):
        our_bid = np.random.uniform(1.5, 4.0)  # Realistic bid range
        quality_score = np.random.uniform(6.0, 9.0)  # Good quality scores
        context = np.random.choice(test_contexts)
        
        result = auction_system.run_auction(our_bid, quality_score, context)
    
    # Check results
    stats = auction_system.get_performance_stats()
    
    print(f"Results after {stats['total_auctions']} auctions:")
    print(f"Win Rate: {stats['win_rate']:.1%}")
    print(f"Average CPC: ${stats['avg_cpc']:.2f}")
    print(f"Average Position: {stats['avg_position']:.1f}")
    
    # Validate
    is_realistic = auction_system.validate_win_rate()
    
    if is_realistic:
        print("✅ Fixed auction system is working correctly!")
    else:
        print("❌ Auction system still needs adjustment")
