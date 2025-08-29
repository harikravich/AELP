#!/usr/bin/env python3
"""
Realistic Competitor Intelligence Through Inference
How GAELP learns about competitors WITHOUT cheating
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class AuctionObservation:
    """What we ACTUALLY know from an auction"""
    timestamp: datetime
    our_bid: float
    won: bool
    position: Optional[int]  # Only if we won
    price_paid: Optional[float]  # Only if we won
    hour: int
    day_of_week: int
    query_type: str
    
class RealisticCompetitorIntelligence:
    """
    Learn about competitors through INFERENCE, not cheating
    This is how real advertisers analyze competition
    """
    
    def __init__(self):
        # Store auction history for pattern analysis
        self.auction_history = []
        
        # Inferred competitive landscape
        self.inferred_competition = {
            'min_bid_to_show': {},  # By hour
            'min_bid_for_top': {},  # By hour
            'price_pressure': {},    # By time period
            'competitor_count_estimate': None,
            'aggressive_hours': [],
            'quiet_hours': []
        }
        
        # Win rate tracking by bid level
        self.bid_buckets = defaultdict(lambda: {'wins': 0, 'total': 0})
        
    def observe_auction(self, observation: AuctionObservation):
        """Record what we ACTUALLY know from an auction"""
        self.auction_history.append(observation)
        
        # Update bid bucket stats
        bid_bucket = round(observation.our_bid, 1)  # Round to nearest 10 cents
        self.bid_buckets[bid_bucket]['total'] += 1
        if observation.won:
            self.bid_buckets[bid_bucket]['wins'] += 1
        
        # Learn from patterns
        if len(self.auction_history) >= 100:
            self._infer_competition_patterns()
    
    def _infer_competition_patterns(self):
        """Infer competitive landscape from win/loss patterns"""
        
        # 1. Find minimum bid thresholds by testing different bid levels
        for hour in range(24):
            hour_auctions = [a for a in self.auction_history 
                           if a.hour == hour]
            
            if len(hour_auctions) >= 5:
                # Find lowest winning bid
                winning_bids = [a.our_bid for a in hour_auctions if a.won]
                if winning_bids:
                    self.inferred_competition['min_bid_to_show'][hour] = min(winning_bids)
                
                # Find bid needed for top positions
                top_position_bids = [a.our_bid for a in hour_auctions 
                                    if a.won and a.position <= 2]
                if top_position_bids:
                    self.inferred_competition['min_bid_for_top'][hour] = min(top_position_bids)
        
        # 2. Identify competitive pressure periods
        recent = self.auction_history[-500:]
        hourly_win_rates = defaultdict(lambda: {'wins': 0, 'total': 0})
        
        for auction in recent:
            hourly_win_rates[auction.hour]['total'] += 1
            if auction.won:
                hourly_win_rates[auction.hour]['wins'] += 1
        
        # Find aggressive hours (low win rate despite good bids)
        self.inferred_competition['aggressive_hours'] = []
        self.inferred_competition['quiet_hours'] = []
        
        for hour, stats in hourly_win_rates.items():
            if stats['total'] >= 10:
                win_rate = stats['wins'] / stats['total']
                if win_rate < 0.15:
                    self.inferred_competition['aggressive_hours'].append(hour)
                elif win_rate > 0.50:
                    self.inferred_competition['quiet_hours'].append(hour)
        
        # 3. Estimate number of competitors from position distribution
        positions = [a.position for a in recent if a.won and a.position]
        if positions:
            # If we're getting position 4-5 when winning, probably 8-10 competitors
            avg_win_position = np.mean(positions)
            self.inferred_competition['competitor_count_estimate'] = int(avg_win_position * 2)
    
    def get_bid_recommendation(self, hour: int, target_win_rate: float = 0.3) -> float:
        """Recommend bid based on inferred competition"""
        
        # Start with base bid
        base_bid = 3.0
        
        # Adjust for time of day competition
        if hour in self.inferred_competition['aggressive_hours']:
            base_bid *= 1.5  # Bid more during competitive hours
        elif hour in self.inferred_competition['quiet_hours']:
            base_bid *= 0.8  # Save money during quiet hours
        
        # Adjust based on historical performance
        if hour in self.inferred_competition['min_bid_to_show']:
            min_bid = self.inferred_competition['min_bid_to_show'][hour]
            base_bid = max(base_bid, min_bid * 1.2)  # Bid 20% above minimum
        
        return min(base_bid, 10.0)  # Cap at $10 for safety
    
    def get_competitive_insights(self) -> Dict[str, Any]:
        """Get actionable insights about competition"""
        
        insights = {
            'total_auctions_observed': len(self.auction_history),
            'estimated_competitors': self.inferred_competition['competitor_count_estimate'],
            'competitive_hours': self.inferred_competition['aggressive_hours'],
            'opportunity_hours': self.inferred_competition['quiet_hours'],
            'bid_performance': {},
            'recommendations': []
        }
        
        # Add bid performance by level
        for bid_level, stats in sorted(self.bid_buckets.items()):
            if stats['total'] >= 5:
                win_rate = stats['wins'] / stats['total']
                insights['bid_performance'][f'${bid_level:.1f}'] = {
                    'win_rate': round(win_rate * 100, 1),
                    'auctions': stats['total']
                }
        
        # Generate recommendations
        if insights['competitive_hours']:
            insights['recommendations'].append(
                f"Reduce bids during hours {insights['competitive_hours']} "
                f"when competition is fierce"
            )
        
        if insights['opportunity_hours']:
            insights['recommendations'].append(
                f"Increase bids during hours {insights['opportunity_hours']} "
                f"for easy wins"
            )
        
        # Find sweet spot bid
        best_roi_bid = None
        best_roi = 0
        for bid_level, stats in self.bid_buckets.items():
            if stats['total'] >= 10:
                win_rate = stats['wins'] / stats['total']
                # Simple ROI: wins per dollar spent
                roi = win_rate / bid_level
                if roi > best_roi:
                    best_roi = roi
                    best_roi_bid = bid_level
        
        if best_roi_bid:
            insights['recommendations'].append(
                f"Optimal bid around ${best_roi_bid:.2f} gives best ROI"
            )
        
        return insights

# Example usage
if __name__ == "__main__":
    intel = RealisticCompetitorIntelligence()
    
    # Simulate some auction observations
    import random
    for i in range(200):
        hour = random.randint(0, 23)
        bid = random.uniform(2.0, 5.0)
        
        # Simulate competition based on hour
        if hour in [9, 10, 14, 15, 20, 21]:  # Peak hours
            win_prob = 0.1 if bid < 4.0 else 0.3
        else:
            win_prob = 0.3 if bid < 3.0 else 0.6
        
        won = random.random() < win_prob
        
        obs = AuctionObservation(
            timestamp=datetime.now(),
            our_bid=bid,
            won=won,
            position=random.randint(1, 4) if won else None,
            price_paid=bid * 0.85 if won else None,
            hour=hour,
            day_of_week=datetime.now().weekday(),
            query_type='research'
        )
        
        intel.observe_auction(obs)
    
    # Get insights
    insights = intel.get_competitive_insights()
    print("Competitive Intelligence Report")
    print("=" * 50)
    print(f"Auctions analyzed: {insights['total_auctions_observed']}")
    print(f"Estimated competitors: {insights['estimated_competitors']}")
    print(f"High competition hours: {insights['competitive_hours']}")
    print(f"Low competition hours: {insights['opportunity_hours']}")
    print("\nBid Performance:")
    for bid, perf in insights['bid_performance'].items():
        print(f"  {bid}: {perf['win_rate']}% win rate ({perf['auctions']} auctions)")
    print("\nRecommendations:")
    for rec in insights['recommendations']:
        print(f"  • {rec}")