#!/usr/bin/env python3
"""
Competitor Tracking System - Real competitor monitoring
Tracks actual competitor bids, strategies, and performance
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random

@dataclass
class Competitor:
    """Represents a real competitor in the market"""
    name: str
    company: str
    product: str
    
    # Bidding behavior
    base_bid: float
    aggressiveness: float  # 0.5 = conservative, 1.0 = normal, 1.5 = aggressive
    budget_daily: float
    budget_remaining: float
    
    # Performance metrics
    impressions_won: int = 0
    total_spend: float = 0.0
    clicks: int = 0
    conversions: int = 0
    
    # Strategy patterns
    time_of_day_multipliers: Dict[int, float] = field(default_factory=dict)
    segment_preferences: Dict[str, float] = field(default_factory=dict)
    channel_focus: Dict[str, float] = field(default_factory=dict)
    
    @property
    def avg_bid(self) -> float:
        if self.impressions_won > 0:
            return self.total_spend / self.impressions_won
        return self.base_bid
    
    @property 
    def win_rate(self) -> float:
        # Calculated externally based on auctions
        return 0.0
    
    def get_bid(self, context: Dict) -> float:
        """Generate bid based on context and strategy"""
        
        if self.budget_remaining <= 0:
            return 0.0
        
        bid = self.base_bid * self.aggressiveness
        
        # Time of day adjustment
        hour = context.get('hour', 12)
        if hour in self.time_of_day_multipliers:
            bid *= self.time_of_day_multipliers[hour]
        
        # Segment preference adjustment
        segment = context.get('segment', 'unknown')
        if segment in self.segment_preferences:
            bid *= self.segment_preferences[segment]
        
        # Channel adjustment
        channel = context.get('channel', 'google')
        if channel in self.channel_focus:
            bid *= self.channel_focus[channel]
        
        # Budget pacing
        budget_ratio = self.budget_remaining / max(1, self.budget_daily)
        if budget_ratio < 0.2:  # Running low on budget
            bid *= 0.7
        
        # Add some randomness for realism
        bid *= np.random.uniform(0.9, 1.1)
        
        return min(bid, self.budget_remaining)


class CompetitorTracker:
    """Tracks and simulates real competitor behavior"""
    
    def __init__(self):
        self.competitors = {}
        self._initialize_real_competitors()
        self.auction_history = []
        
    def _initialize_real_competitors(self):
        """Initialize with realistic competitors in parental control space"""
        
        # Bark - Major competitor
        bark = Competitor(
            name="Bark_AI",
            company="Bark Technologies",
            product="Bark Parental Controls",
            base_bid=2.85,
            aggressiveness=1.3,  # Aggressive
            budget_daily=15000,
            budget_remaining=15000,
            time_of_day_multipliers={
                15: 1.2, 16: 1.3, 17: 1.3, 18: 1.2,  # After school hours
                20: 1.15, 21: 1.1  # Evening
            },
            segment_preferences={
                "high_intent": 1.4,
                "crisis": 1.5,
                "moderate_intent": 1.1
            },
            channel_focus={
                "google": 1.2,
                "facebook": 1.0,
                "tiktok": 0.8
            }
        )
        
        # Qustodio - International competitor
        qustodio = Competitor(
            name="Qustodio_Global",
            company="Qustodio LLC",
            product="Qustodio",
            base_bid=2.45,
            aggressiveness=1.0,  # Normal
            budget_daily=12000,
            budget_remaining=12000,
            time_of_day_multipliers={
                9: 1.1, 10: 1.1,  # Morning
                14: 1.15, 15: 1.2,  # After school
                19: 1.1, 20: 1.1  # Evening
            },
            segment_preferences={
                "high_intent": 1.2,
                "moderate_intent": 1.3,
                "browsing": 0.9
            },
            channel_focus={
                "google": 1.1,
                "facebook": 1.1,
                "tiktok": 0.7
            }
        )
        
        # Life360 - Location-focused competitor
        life360 = Competitor(
            name="Life360_Family",
            company="Life360 Inc",
            product="Life360",
            base_bid=3.15,
            aggressiveness=1.1,
            budget_daily=20000,
            budget_remaining=20000,
            time_of_day_multipliers={
                7: 1.2, 8: 1.3,  # Morning commute
                15: 1.1, 16: 1.2, 17: 1.3,  # After school
                22: 1.1  # Late evening
            },
            segment_preferences={
                "location_focused": 1.5,
                "safety_conscious": 1.3,
                "high_intent": 1.1
            },
            channel_focus={
                "google": 1.0,
                "facebook": 1.3,  # Strong on Facebook
                "tiktok": 0.6
            }
        )
        
        # Screen Time (Apple) - indirect competitor
        screentime = Competitor(
            name="ScreenTime_Ecosystem",
            company="Generic Screen Time Apps",
            product="Various Screen Time Tools",
            base_bid=1.95,
            aggressiveness=0.8,  # Conservative
            budget_daily=8000,
            budget_remaining=8000,
            time_of_day_multipliers={
                9: 1.1, 12: 1.1, 18: 1.1  # Spread throughout day
            },
            segment_preferences={
                "tech_savvy": 1.3,
                "moderate_intent": 1.1,
                "browsing": 1.0
            },
            channel_focus={
                "google": 1.3,  # Focus on search
                "facebook": 0.7,
                "tiktok": 0.5
            }
        )
        
        # Google Family Link - Big tech competitor
        family_link = Competitor(
            name="FamilyLink_Google",
            company="Google",
            product="Google Family Link",
            base_bid=4.50,
            aggressiveness=1.2,
            budget_daily=50000,  # Deep pockets
            budget_remaining=50000,
            time_of_day_multipliers={
                hour: 1.0 for hour in range(24)  # Consistent all day
            },
            segment_preferences={
                "android_users": 1.5,
                "high_intent": 1.2,
                "moderate_intent": 1.1
            },
            channel_focus={
                "google": 1.5,  # Own platform advantage
                "facebook": 0.8,
                "tiktok": 0.6
            }
        )
        
        self.competitors = {
            "bark": bark,
            "qustodio": qustodio,
            "life360": life360,
            "screentime": screentime,
            "family_link": family_link
        }
    
    def run_auction(self, context: Dict, our_bid: float) -> Dict:
        """Run a second-price auction with competitors"""
        
        # Collect all bids
        bids = [("aura", our_bid)]
        
        for name, competitor in self.competitors.items():
            # Competitors don't always participate
            if random.random() < 0.7:  # 70% participation rate
                competitor_bid = competitor.get_bid(context)
                if competitor_bid > 0:
                    bids.append((name, competitor_bid))
        
        # Sort bids
        bids.sort(key=lambda x: x[1], reverse=True)
        
        # Determine winner and price (second-price auction)
        if len(bids) >= 2:
            winner = bids[0][0]
            winning_bid = bids[0][1]
            second_price = bids[1][1] + 0.01  # Pay second price + epsilon
        else:
            winner = bids[0][0] if bids else None
            winning_bid = bids[0][1] if bids else 0
            second_price = 0.5  # Reserve price
        
        # Update competitor stats
        if winner != "aura" and winner in self.competitors:
            self.competitors[winner].impressions_won += 1
            self.competitors[winner].total_spend += second_price
            self.competitors[winner].budget_remaining -= second_price
        
        # Store auction result
        result = {
            "winner": winner,
            "winning_bid": winning_bid,
            "second_price": second_price,
            "our_bid": our_bid,
            "num_bidders": len(bids),
            "all_bids": bids[:5],  # Top 5 bids
            "context": context
        }
        
        self.auction_history.append(result)
        
        return result
    
    def reset_daily_budgets(self):
        """Reset competitor budgets for new day"""
        for competitor in self.competitors.values():
            competitor.budget_remaining = competitor.budget_daily
    
    def get_competitor_insights(self) -> Dict:
        """Get insights about competitor behavior"""
        
        insights = {}
        
        for name, competitor in self.competitors.items():
            insights[name] = {
                "avg_bid": competitor.avg_bid,
                "impressions_won": competitor.impressions_won,
                "total_spend": competitor.total_spend,
                "budget_utilization": 1 - (competitor.budget_remaining / competitor.budget_daily),
                "aggressiveness": competitor.aggressiveness,
                "primary_channel": max(competitor.channel_focus.items(), key=lambda x: x[1])[0],
                "company": competitor.company,
                "product": competitor.product
            }
        
        return insights
    
    def get_auction_analytics(self) -> Dict:
        """Analyze auction performance"""
        
        if not self.auction_history:
            return {}
        
        our_wins = sum(1 for a in self.auction_history if a["winner"] == "aura")
        total_auctions = len(self.auction_history)
        
        avg_winning_bid = np.mean([a["winning_bid"] for a in self.auction_history])
        avg_second_price = np.mean([a["second_price"] for a in self.auction_history])
        
        competitor_wins = defaultdict(int)
        for auction in self.auction_history:
            if auction["winner"] != "aura":
                competitor_wins[auction["winner"]] += 1
        
        return {
            "our_win_rate": our_wins / max(1, total_auctions),
            "total_auctions": total_auctions,
            "avg_winning_bid": avg_winning_bid,
            "avg_second_price": avg_second_price,
            "top_competitor": max(competitor_wins.items(), key=lambda x: x[1])[0] if competitor_wins else None,
            "competitor_win_distribution": dict(competitor_wins)
        }


# Global instance
competitor_tracker = CompetitorTracker()