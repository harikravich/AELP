"""
Competitive Intelligence System with Partial Observability

This module implements a competitive intelligence system for ad auctions
where we have limited visibility into competitor behavior. We can only
observe when we lose auctions, not the exact bids or strategies of competitors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class AuctionOutcome:
    """Represents an auction outcome with partial observability."""
    timestamp: datetime
    keyword: str
    our_bid: float
    position: Optional[int]  # None if we didn't win
    cost: Optional[float]   # None if we didn't win
    competitor_count: Optional[int]  # Estimated from available data
    quality_score: float
    daypart: int  # Hour of day (0-23)
    day_of_week: int  # 0=Monday, 6=Sunday
    device_type: str
    location: str


@dataclass
class CompetitorProfile:
    """Profile of a competitor's bidding behavior."""
    competitor_id: str
    bid_distribution: Dict[str, Any] = field(default_factory=dict)
    daypart_patterns: Dict[int, float] = field(default_factory=dict)
    keyword_preferences: Dict[str, float] = field(default_factory=dict)
    aggression_level: float = 0.5
    budget_patterns: Dict[str, float] = field(default_factory=dict)
    last_seen: Optional[datetime] = None
    confidence: float = 0.0


class CompetitiveIntelligence:
    """
    Competitive Intelligence System with Partial Observability
    
    Models competitor behavior from limited auction data where we only
    observe outcomes when we participate, not exact competitor bids.
    """
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.auction_history: List[AuctionOutcome] = []
        self.competitor_profiles: Dict[str, CompetitorProfile] = {}
        self.bid_estimator = None
        self.scaler = StandardScaler()
        
        # Uncertainty parameters
        self.base_uncertainty = 0.2  # 20% base uncertainty in estimates
        self.confidence_decay = 0.95  # Daily confidence decay
        
    def record_auction_outcome(self, outcome: AuctionOutcome) -> None:
        """Record an auction outcome for analysis."""
        self.auction_history.append(outcome)
        
        # Clean old data
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        self.auction_history = [
            o for o in self.auction_history 
            if o.timestamp >= cutoff_date
        ]
        
        logger.info(f"Recorded auction outcome for keyword '{outcome.keyword}' "
                   f"at position {outcome.position}")
    
    def estimate_competitor_bid(
        self, 
        outcome: AuctionOutcome,
        position: int = 1
    ) -> Tuple[float, float]:
        """
        Estimate competitor bid for a specific position.
        
        Since we only see auction outcomes, not actual bids, we use:
        1. Our bid and position to infer minimum competitor bids
        2. Historical patterns to estimate typical bid ranges
        3. Quality score adjustments for effective bids
        
        Returns:
            Tuple of (estimated_bid, confidence_interval)
        """
        if outcome.position is None:
            # We lost - competitor bid was higher than ours
            return self._estimate_winning_bid(outcome)
        
        if outcome.position == 1:
            # We won first position - estimate second place bid
            return self._estimate_second_place_bid(outcome)
        
        # We got a lower position - estimate higher position bids
        return self._estimate_higher_position_bid(outcome, position)
    
    def _estimate_winning_bid(self, outcome: AuctionOutcome) -> Tuple[float, float]:
        """Estimate the winning bid when we lost the auction."""
        # Winner's bid was at least our_bid * quality_adjustment + epsilon
        quality_adjustment = outcome.quality_score / 7.0  # Assume average QS is 7
        
        # Look for similar historical auctions
        similar_outcomes = self._find_similar_auctions(outcome)
        
        if similar_outcomes:
            # Use historical data to estimate bid distribution
            win_ratios = []
            for similar in similar_outcomes:
                if similar.position == 1 and similar.cost:
                    # Reverse-engineer effective bid from cost and quality
                    effective_bid = similar.cost / similar.quality_score * 7.0
                    ratio = effective_bid / outcome.our_bid
                    win_ratios.append(ratio)
            
            if win_ratios:
                mean_ratio = np.mean(win_ratios)
                std_ratio = np.std(win_ratios) if len(win_ratios) > 1 else 0.2
                
                estimated_bid = outcome.our_bid * mean_ratio / quality_adjustment
                confidence_interval = estimated_bid * (std_ratio + self.base_uncertainty)
                
                return estimated_bid, confidence_interval
        
        # Fallback to heuristic estimation
        # Winner likely bid 10-50% more than us, adjusted for quality
        multiplier = np.random.uniform(1.1, 1.5)
        estimated_bid = outcome.our_bid * multiplier / quality_adjustment
        confidence_interval = estimated_bid * 0.3
        
        return estimated_bid, confidence_interval
    
    def _estimate_second_place_bid(self, outcome: AuctionOutcome) -> Tuple[float, float]:
        """Estimate second place bid when we won first position."""
        if not outcome.cost:
            return 0.0, 0.0
        
        # In second-price auctions, we pay second-place bid + $0.01
        # But with quality scores, it's more complex
        quality_adjustment = outcome.quality_score / 7.0
        
        # Our cost approximates second-place effective bid
        second_place_effective = outcome.cost
        second_place_actual = second_place_effective / quality_adjustment
        
        # Add uncertainty based on market volatility
        similar_outcomes = self._find_similar_auctions(outcome, limit=20)
        if len(similar_outcomes) > 5:
            costs = [o.cost for o in similar_outcomes if o.cost]
            cost_std = np.std(costs) if len(costs) > 1 else outcome.cost * 0.2
            confidence_interval = cost_std / quality_adjustment
        else:
            confidence_interval = second_place_actual * 0.25
        
        return second_place_actual, confidence_interval
    
    def _estimate_higher_position_bid(
        self, 
        outcome: AuctionOutcome, 
        target_position: int
    ) -> Tuple[float, float]:
        """Estimate bid for higher position than what we achieved."""
        if not outcome.cost or not outcome.position:
            return 0.0, 0.0
        
        # Estimate position premium
        position_multiplier = self._estimate_position_multiplier(
            outcome.position, target_position, outcome.keyword
        )
        
        quality_adjustment = outcome.quality_score / 7.0
        our_effective_bid = outcome.cost / quality_adjustment
        
        estimated_bid = our_effective_bid * position_multiplier
        confidence_interval = estimated_bid * 0.3  # Higher uncertainty for extrapolation
        
        return estimated_bid, confidence_interval
    
    def track_patterns(self, competitor_id: str = "aggregate") -> CompetitorProfile:
        """
        Track competitor bidding patterns from auction outcomes.
        
        Since we can't directly identify competitors, we analyze
        aggregate market behavior patterns.
        """
        if competitor_id not in self.competitor_profiles:
            self.competitor_profiles[competitor_id] = CompetitorProfile(competitor_id)
        
        profile = self.competitor_profiles[competitor_id]
        
        # Analyze daypart patterns
        self._analyze_daypart_patterns(profile)
        
        # Analyze keyword preferences
        self._analyze_keyword_patterns(profile)
        
        # Estimate aggression level
        self._estimate_aggression_level(profile)
        
        # Analyze budget patterns
        self._analyze_budget_patterns(profile)
        
        # Update confidence and last seen
        profile.last_seen = datetime.now()
        profile.confidence = self._calculate_profile_confidence(profile)
        
        logger.info(f"Updated patterns for competitor {competitor_id}, "
                   f"confidence: {profile.confidence:.2f}")
        
        return profile
    
    def _analyze_daypart_patterns(self, profile: CompetitorProfile) -> None:
        """Analyze time-of-day bidding patterns."""
        hourly_competition = defaultdict(list)
        
        for outcome in self.auction_history:
            # Use loss rate as proxy for competition intensity
            competition_intensity = 0.0 if outcome.position == 1 else 1.0
            hourly_competition[outcome.daypart].append(competition_intensity)
        
        for hour, intensities in hourly_competition.items():
            profile.daypart_patterns[hour] = np.mean(intensities)
    
    def _analyze_keyword_patterns(self, profile: CompetitorProfile) -> None:
        """Analyze keyword-specific competition patterns."""
        keyword_competition = defaultdict(list)
        
        for outcome in self.auction_history:
            # Higher cost per click suggests more competition
            if outcome.cost and outcome.position:
                competition_score = outcome.cost / outcome.our_bid
                keyword_competition[outcome.keyword].append(competition_score)
        
        for keyword, scores in keyword_competition.items():
            profile.keyword_preferences[keyword] = np.mean(scores)
    
    def _estimate_aggression_level(self, profile: CompetitorProfile) -> None:
        """Estimate overall market aggression level."""
        recent_outcomes = [
            o for o in self.auction_history[-100:]  # Last 100 auctions
            if o.timestamp >= datetime.now() - timedelta(days=7)
        ]
        
        if not recent_outcomes:
            return
        
        # Calculate win rate and average position
        wins = sum(1 for o in recent_outcomes if o.position == 1)
        win_rate = wins / len(recent_outcomes)
        
        avg_position = np.mean([
            o.position for o in recent_outcomes if o.position
        ])
        
        # Lower win rate and higher average position = more aggressive competition
        profile.aggression_level = 1.0 - (win_rate * 0.7 + (4 - avg_position) / 4 * 0.3)
        profile.aggression_level = np.clip(profile.aggression_level, 0.0, 1.0)
    
    def _analyze_budget_patterns(self, profile: CompetitorProfile) -> None:
        """Analyze budget-related patterns (time of month, day effects)."""
        # Group by day of month to detect budget cycles
        daily_competition = defaultdict(list)
        
        for outcome in self.auction_history:
            day_of_month = outcome.timestamp.day
            competition_intensity = 0.0 if outcome.position == 1 else 1.0
            daily_competition[day_of_month].append(competition_intensity)
        
        # Calculate competition intensity by day of month
        for day, intensities in daily_competition.items():
            if intensities:
                profile.budget_patterns[f"day_{day}"] = np.mean(intensities)
    
    def predict_response(
        self, 
        our_planned_bid: float,
        keyword: str,
        timestamp: datetime,
        scenario: str = "bid_increase"
    ) -> Dict[str, Any]:
        """
        Predict competitor response to our bidding changes.
        
        Args:
            our_planned_bid: Our intended bid
            keyword: Target keyword
            timestamp: When we plan to bid
            scenario: Type of change ("bid_increase", "new_keyword", "budget_increase")
        """
        prediction = {
            "scenario": scenario,
            "our_planned_bid": our_planned_bid,
            "keyword": keyword,
            "timestamp": timestamp,
            "competitor_responses": {},
            "market_impact": {},
            "confidence": 0.0
        }
        
        # Analyze historical similar scenarios
        similar_outcomes = self._find_similar_auctions(
            AuctionOutcome(
                timestamp=timestamp,
                keyword=keyword,
                our_bid=our_planned_bid,
                position=None,
                cost=None,
                competitor_count=None,
                quality_score=7.0,  # Default
                daypart=timestamp.hour,
                day_of_week=timestamp.weekday(),
                device_type="desktop",
                location="default"
            )
        )
        
        if len(similar_outcomes) < 5:
            prediction["confidence"] = 0.2
            prediction["market_impact"]["warning"] = "Limited historical data"
            return prediction
        
        # Predict based on scenario type
        if scenario == "bid_increase":
            prediction.update(self._predict_bid_increase_response(
                our_planned_bid, keyword, similar_outcomes
            ))
        elif scenario == "new_keyword":
            prediction.update(self._predict_new_keyword_response(
                our_planned_bid, keyword, similar_outcomes
            ))
        elif scenario == "budget_increase":
            prediction.update(self._predict_budget_increase_response(
                our_planned_bid, keyword, similar_outcomes
            ))
        
        # Overall confidence based on data quality
        prediction["confidence"] = min(
            len(similar_outcomes) / 50.0,  # More data = higher confidence
            0.8  # Cap at 80% due to inherent uncertainty
        )
        
        return prediction
    
    def _predict_bid_increase_response(
        self,
        our_planned_bid: float,
        keyword: str,
        similar_outcomes: List[AuctionOutcome]
    ) -> Dict[str, Any]:
        """Predict competitor response to our bid increase."""
        # Analyze historical bid escalations
        bid_escalations = []
        for i in range(1, len(similar_outcomes)):
            prev_outcome = similar_outcomes[i-1]
            curr_outcome = similar_outcomes[i]
            
            if prev_outcome.our_bid < curr_outcome.our_bid:
                # We increased bid - did competition respond?
                if curr_outcome.cost and prev_outcome.cost:
                    escalation_ratio = curr_outcome.cost / prev_outcome.cost
                    bid_escalations.append(escalation_ratio)
        
        if bid_escalations:
            avg_escalation = np.mean(bid_escalations)
            escalation_probability = len([e for e in bid_escalations if e > 1.1]) / len(bid_escalations)
        else:
            avg_escalation = 1.0
            escalation_probability = 0.5
        
        return {
            "competitor_responses": {
                "escalation_probability": escalation_probability,
                "expected_escalation_ratio": avg_escalation,
                "estimated_response_time": "1-3 days"
            },
            "market_impact": {
                "expected_cpc_increase": f"{(avg_escalation - 1) * 100:.1f}%",
                "competition_level": "moderate" if escalation_probability < 0.6 else "high"
            }
        }
    
    def _predict_new_keyword_response(
        self,
        our_planned_bid: float,
        keyword: str,
        similar_outcomes: List[AuctionOutcome]
    ) -> Dict[str, Any]:
        """Predict competitor response to us entering a new keyword."""
        # Check if this is truly a new keyword for us
        keyword_history = [o for o in self.auction_history if o.keyword == keyword]
        
        if keyword_history:
            return self._predict_bid_increase_response(our_planned_bid, keyword, similar_outcomes)
        
        # Analyze market entry patterns from similar keywords
        similar_keywords = [
            o.keyword for o in similar_outcomes
            if self._keyword_similarity(keyword, o.keyword) > 0.7
        ]
        
        return {
            "competitor_responses": {
                "detection_time": "immediate to 1 week",
                "response_probability": 0.7,  # High - new entrants are noticed
                "defensive_bidding": True
            },
            "market_impact": {
                "initial_advantage": "2-7 days",
                "expected_competition_increase": "moderate",
                "market_saturation_risk": len(similar_keywords) > 10
            }
        }
    
    def _predict_budget_increase_response(
        self,
        our_planned_bid: float,
        keyword: str,
        similar_outcomes: List[AuctionOutcome]
    ) -> Dict[str, Any]:
        """Predict competitor response to our budget increase."""
        # Budget increases typically lead to more aggressive bidding
        return {
            "competitor_responses": {
                "escalation_probability": 0.8,  # High - sustained presence triggers response
                "response_timeline": "3-7 days",
                "counter_strategy": "budget_matching"
            },
            "market_impact": {
                "market_share_shift": "gradual",
                "price_inflation": "moderate to high",
                "sustainability": "depends on competitor budgets"
            }
        }
    
    def _find_similar_auctions(
        self, 
        target: AuctionOutcome, 
        limit: int = 50
    ) -> List[AuctionOutcome]:
        """Find similar historical auction outcomes."""
        similarities = []
        
        for outcome in self.auction_history:
            similarity = self._calculate_outcome_similarity(target, outcome)
            similarities.append((similarity, outcome))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [outcome for _, outcome in similarities[:limit] if _ > 0.3]
    
    def _calculate_outcome_similarity(
        self, 
        target: AuctionOutcome, 
        candidate: AuctionOutcome
    ) -> float:
        """Calculate similarity between two auction outcomes."""
        similarity = 0.0
        
        # Keyword similarity (exact match for now, could use semantic similarity)
        if target.keyword == candidate.keyword:
            similarity += 0.4
        
        # Time similarity (daypart and day of week)
        if target.daypart == candidate.daypart:
            similarity += 0.2
        if target.day_of_week == candidate.day_of_week:
            similarity += 0.1
        
        # Bid similarity
        if candidate.our_bid > 0:
            bid_ratio = min(target.our_bid / candidate.our_bid, 
                           candidate.our_bid / target.our_bid)
            similarity += 0.2 * bid_ratio
        
        # Device and location similarity
        if target.device_type == candidate.device_type:
            similarity += 0.05
        if target.location == candidate.location:
            similarity += 0.05
        
        return similarity
    
    def _keyword_similarity(self, keyword1: str, keyword2: str) -> float:
        """Calculate keyword similarity (simple implementation)."""
        # Simple Jaccard similarity on words
        words1 = set(keyword1.lower().split())
        words2 = set(keyword2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _estimate_position_multiplier(
        self,
        current_position: int,
        target_position: int,
        keyword: str
    ) -> float:
        """Estimate bid multiplier needed to move from current to target position."""
        # Heuristic based on typical position premiums
        position_premiums = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4}
        
        current_premium = position_premiums.get(current_position, 0.2)
        target_premium = position_premiums.get(target_position, 0.2)
        
        return target_premium / current_premium if current_premium > 0 else 1.5
    
    def _calculate_profile_confidence(self, profile: CompetitorProfile) -> float:
        """Calculate confidence in competitor profile based on data quality."""
        data_points = len(self.auction_history)
        recency_bonus = 1.0 if profile.last_seen and \
                       (datetime.now() - profile.last_seen).days < 7 else 0.8
        
        base_confidence = min(data_points / 100.0, 0.9)  # More data = higher confidence
        
        return base_confidence * recency_bonus
    
    def get_market_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive market intelligence summary."""
        if not self.auction_history:
            return {"error": "No auction data available"}
        
        recent_data = [
            o for o in self.auction_history
            if o.timestamp >= datetime.now() - timedelta(days=7)
        ]
        
        summary = {
            "market_overview": {
                "total_auctions_analyzed": len(self.auction_history),
                "recent_auctions": len(recent_data),
                "tracking_period_days": self.lookback_days
            },
            "competition_metrics": {},
            "market_trends": {},
            "competitor_profiles": {}
        }
        
        if recent_data:
            # Competition intensity
            win_rate = sum(1 for o in recent_data if o.position == 1) / len(recent_data)
            avg_position = np.mean([o.position for o in recent_data if o.position])
            avg_cpc = np.mean([o.cost for o in recent_data if o.cost])
            
            summary["competition_metrics"] = {
                "win_rate": win_rate,
                "average_position": avg_position,
                "average_cpc": avg_cpc,
                "competition_intensity": 1.0 - win_rate
            }
            
            # Market trends (week-over-week)
            if len(self.auction_history) > len(recent_data):
                prev_week_data = [
                    o for o in self.auction_history
                    if datetime.now() - timedelta(days=14) <= o.timestamp < datetime.now() - timedelta(days=7)
                ]
                
                if prev_week_data:
                    prev_win_rate = sum(1 for o in prev_week_data if o.position == 1) / len(prev_week_data)
                    prev_avg_cpc = np.mean([o.cost for o in prev_week_data if o.cost])
                    
                    summary["market_trends"] = {
                        "win_rate_change": win_rate - prev_win_rate,
                        "cpc_change_pct": (avg_cpc - prev_avg_cpc) / prev_avg_cpc * 100 if prev_avg_cpc > 0 else 0
                    }
        
        # Competitor profiles summary
        for comp_id, profile in self.competitor_profiles.items():
            summary["competitor_profiles"][comp_id] = {
                "aggression_level": profile.aggression_level,
                "confidence": profile.confidence,
                "most_active_hours": sorted(
                    profile.daypart_patterns.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3] if profile.daypart_patterns else []
            }
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    intel_system = CompetitiveIntelligence(lookback_days=30)
    
    # Simulate some auction outcomes
    sample_outcomes = [
        AuctionOutcome(
            timestamp=datetime.now() - timedelta(hours=1),
            keyword="running shoes",
            our_bid=2.50,
            position=2,
            cost=2.10,
            competitor_count=5,
            quality_score=8.0,
            daypart=14,
            day_of_week=1,
            device_type="mobile",
            location="US"
        ),
        AuctionOutcome(
            timestamp=datetime.now() - timedelta(hours=2),
            keyword="running shoes",
            our_bid=2.30,
            position=None,  # Lost auction
            cost=None,
            competitor_count=6,
            quality_score=8.0,
            daypart=13,
            day_of_week=1,
            device_type="mobile",
            location="US"
        )
    ]
    
    # Record outcomes
    for outcome in sample_outcomes:
        intel_system.record_auction_outcome(outcome)
    
    # Estimate competitor bids
    competitor_bid, confidence = intel_system.estimate_competitor_bid(sample_outcomes[1])
    print(f"Estimated competitor bid: ${competitor_bid:.2f} ± ${confidence:.2f}")
    
    # Track patterns
    profile = intel_system.track_patterns("aggregate")
    print(f"Competitor aggression level: {profile.aggression_level:.2f}")
    
    # Predict response
    prediction = intel_system.predict_response(
        our_planned_bid=3.00,
        keyword="running shoes",
        timestamp=datetime.now() + timedelta(hours=1),
        scenario="bid_increase"
    )
    print(f"Response prediction confidence: {prediction['confidence']:.2f}")
    
    # Get market summary
    summary = intel_system.get_market_intelligence_summary()
    print(f"Market overview: {summary['market_overview']}")