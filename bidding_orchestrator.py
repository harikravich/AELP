"""
Bidding Orchestrator with Temporal Effects Integration

Orchestrates bidding decisions by incorporating temporal patterns,
seasonal effects, and event-driven demand spikes to optimize
ad spend based on predicted demand variations.
"""

import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from temporal_effects import TemporalEffects, EventSpike

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BidRequest:
    """Structure for incoming bid requests"""
    campaign_id: str
    base_bid: float
    target_date: datetime.datetime
    keywords: List[str]
    budget_remaining: float
    max_adjustment: float = 2.0


@dataclass
class BidResponse:
    """Structure for bid decisions"""
    campaign_id: str
    original_bid: float
    adjusted_bid: float
    demand_multiplier: float
    adjustment_reason: str
    breakdown: Dict[str, float]
    recommended_action: str


class BiddingOrchestrator:
    """
    Main orchestrator for temporal-aware bidding decisions.
    
    Integrates TemporalEffects to adjust bids based on:
    - Seasonal patterns (back-to-school, holidays)
    - Time-of-day effects (peak hours 8-10 PM)
    - Day-of-week patterns
    - Event-driven spikes (news stories causing 3x surge)
    """
    
    def __init__(self):
        """Initialize the orchestrator with temporal effects"""
        self.temporal_effects = TemporalEffects()
        self.active_campaigns: Dict[str, Dict] = {}
        self.bid_history: List[BidResponse] = []
        
        logger.info("BiddingOrchestrator initialized with TemporalEffects")
        
        # Setup default event spikes for common scenarios
        self._setup_default_events()
    
    def _setup_default_events(self):
        """Setup common event spikes that might occur"""
        # These would typically be added dynamically when events occur
        sample_events = [
            EventSpike("screen_time_study", 3.0, 5, 0.8),  # Major study causes 3x spike
            EventSpike("celebrity_endorsement", 2.5, 3, 0.85),
            EventSpike("competitor_incident", 2.2, 7, 0.9),
            EventSpike("regulatory_news", 2.8, 4, 0.75)
        ]
        
        logger.info(f"Default event templates loaded: {len(sample_events)} types")
    
    def process_bid_request(self, request: BidRequest) -> BidResponse:
        """
        Process a bid request with temporal adjustments.
        
        Args:
            request: BidRequest containing campaign details
            
        Returns:
            BidResponse with temporal adjustments applied
        """
        logger.info(f"Processing bid request for campaign {request.campaign_id}")
        
        # Get seasonal multiplier
        seasonal_multiplier = self.temporal_effects.get_seasonal_multiplier(request.target_date)
        
        # Use predict_demand for comprehensive demand forecasting
        demand_forecast = self.temporal_effects.predict_demand(request.target_date)
        
        # Apply adjust_bidding to get full adjustment details
        bid_adjustment = self.temporal_effects.adjust_bidding(
            base_bid=request.base_bid,
            date=request.target_date,
            max_adjustment=request.max_adjustment
        )
        
        # Handle event spikes if any are active
        event_multiplier = self.temporal_effects.get_event_multiplier(request.target_date)
        if event_multiplier > 1.5:
            logger.warning(f"High event multiplier detected: {event_multiplier:.2f}x")
        
        # Create response
        response = BidResponse(
            campaign_id=request.campaign_id,
            original_bid=bid_adjustment['original_bid'],
            adjusted_bid=bid_adjustment['adjusted_bid'],
            demand_multiplier=bid_adjustment['demand_multiplier'],
            adjustment_reason=bid_adjustment['adjustment_reason'],
            breakdown=bid_adjustment['breakdown'],
            recommended_action=self._get_recommended_action(
                bid_adjustment['demand_multiplier'], 
                request.budget_remaining
            )
        )
        
        # Store in bid history
        self.bid_history.append(response)
        
        # Update campaign tracking
        self._update_campaign_tracking(request, response)
        
        logger.info(
            f"Bid adjusted: ${request.base_bid:.2f} -> ${response.adjusted_bid:.2f} "
            f"({response.demand_multiplier:.2f}x) - {response.adjustment_reason}"
        )
        
        return response
    
    def _get_recommended_action(self, demand_multiplier: float, budget_remaining: float) -> str:
        """
        Generate recommended action based on demand and budget.
        
        Args:
            demand_multiplier: Predicted demand multiplier
            budget_remaining: Remaining budget for campaign
            
        Returns:
            Recommended action string
        """
        if demand_multiplier > 2.0:
            if budget_remaining > 1000:  # High budget threshold
                return "AGGRESSIVE_BID - High demand period, maximize reach"
            else:
                return "CAUTIOUS_BID - High demand but limited budget"
        elif demand_multiplier > 1.5:
            return "INCREASE_BID - Above average demand period"
        elif demand_multiplier > 0.8:
            return "MAINTAIN_BID - Normal demand period"
        else:
            return "REDUCE_BID - Low demand period, conserve budget"
    
    def _update_campaign_tracking(self, request: BidRequest, response: BidResponse):
        """Update campaign tracking information"""
        campaign_id = request.campaign_id
        
        if campaign_id not in self.active_campaigns:
            self.active_campaigns[campaign_id] = {
                'first_bid': request.target_date,
                'total_bids': 0,
                'total_spend_planned': 0.0,
                'avg_multiplier': 0.0
            }
        
        campaign = self.active_campaigns[campaign_id]
        campaign['total_bids'] += 1
        campaign['total_spend_planned'] += response.adjusted_bid
        campaign['avg_multiplier'] = (
            (campaign['avg_multiplier'] * (campaign['total_bids'] - 1) + response.demand_multiplier) 
            / campaign['total_bids']
        )
        campaign['last_bid'] = request.target_date
    
    def add_event_spike(self, event_name: str, multiplier: float, duration_days: int, 
                       decay_factor: float = 0.8, start_date: Optional[datetime.datetime] = None):
        """
        Add an event-driven demand spike (e.g., news story causing 3x surge).
        
        Args:
            event_name: Name/description of the event
            multiplier: Peak demand multiplier (e.g., 3.0 for 3x surge)
            duration_days: How many days the spike lasts
            decay_factor: Daily decay rate (0.8 = 20% decay per day)
            start_date: When spike starts (defaults to now)
        """
        if start_date is None:
            start_date = datetime.datetime.now()
        
        event_spike = EventSpike(event_name, multiplier, duration_days, decay_factor)
        self.temporal_effects.add_event_spike(event_spike, start_date)
        
        logger.warning(
            f"Event spike added: {event_name} - {multiplier:.1f}x multiplier "
            f"for {duration_days} days starting {start_date.strftime('%Y-%m-%d')}"
        )
    
    def get_optimal_timing(self, target_date: datetime.date) -> Dict[str, Any]:
        """
        Get optimal timing recommendations for a target date.
        
        Args:
            target_date: Date to analyze
            
        Returns:
            Timing optimization recommendations
        """
        return self.temporal_effects.get_optimal_timing(target_date)
    
    def predict_demand_range(self, start_date: datetime.datetime, 
                           end_date: datetime.datetime) -> List[Tuple[datetime.datetime, float]]:
        """
        Predict demand over a date range.
        
        Args:
            start_date: Start of prediction range
            end_date: End of prediction range
            
        Returns:
            List of (datetime, demand_multiplier) tuples
        """
        return self.temporal_effects.predict_demand_range(start_date, end_date)
    
    def get_campaign_performance(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance summary for a specific campaign.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            Campaign performance dictionary or None if not found
        """
        if campaign_id not in self.active_campaigns:
            return None
        
        campaign = self.active_campaigns[campaign_id]
        campaign_bids = [bid for bid in self.bid_history if bid.campaign_id == campaign_id]
        
        return {
            **campaign,
            'bid_count': len(campaign_bids),
            'recent_bids': campaign_bids[-5:] if campaign_bids else [],
            'avg_adjusted_bid': sum(bid.adjusted_bid for bid in campaign_bids) / len(campaign_bids) if campaign_bids else 0,
            'max_multiplier_seen': max(bid.demand_multiplier for bid in campaign_bids) if campaign_bids else 0,
            'min_multiplier_seen': min(bid.demand_multiplier for bid in campaign_bids) if campaign_bids else 0
        }
    
    def get_current_conditions(self, target_date: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """
        Get current temporal conditions and recommendations.
        
        Args:
            target_date: Date to analyze (defaults to now)
            
        Returns:
            Current conditions summary
        """
        if target_date is None:
            target_date = datetime.datetime.now()
        
        demand = self.temporal_effects.predict_demand(target_date)
        
        return {
            'current_time': target_date,
            'demand_multiplier': demand,
            'seasonal_multiplier': self.temporal_effects.get_seasonal_multiplier(target_date),
            'hourly_multiplier': self.temporal_effects.get_hourly_multiplier(target_date.hour),
            'daily_multiplier': self.temporal_effects.get_daily_multiplier(target_date.weekday()),
            'event_multiplier': self.temporal_effects.get_event_multiplier(target_date),
            'active_spikes': len(self.temporal_effects.active_spikes),
            'market_condition': self._classify_market_condition(demand),
            'bidding_recommendation': self._get_general_bidding_recommendation(demand)
        }
    
    def _classify_market_condition(self, demand_multiplier: float) -> str:
        """Classify current market condition based on demand"""
        if demand_multiplier > 2.0:
            return "EXTREMELY_HIGH_DEMAND"
        elif demand_multiplier > 1.5:
            return "HIGH_DEMAND"
        elif demand_multiplier > 1.2:
            return "ABOVE_AVERAGE_DEMAND"
        elif demand_multiplier > 0.8:
            return "NORMAL_DEMAND"
        elif demand_multiplier > 0.6:
            return "BELOW_AVERAGE_DEMAND"
        else:
            return "LOW_DEMAND"
    
    def _get_general_bidding_recommendation(self, demand_multiplier: float) -> str:
        """Get general bidding recommendation"""
        if demand_multiplier > 2.0:
            return "Maximize bids for peak performance, monitor budget closely"
        elif demand_multiplier > 1.5:
            return "Increase bids to capture above-average demand"
        elif demand_multiplier > 1.2:
            return "Slightly increase bids for modest gains"
        elif demand_multiplier > 0.8:
            return "Maintain current bidding strategy"
        else:
            return "Reduce bids to preserve budget, focus on high-value keywords"


# Example usage and testing
def demo_orchestrator():
    """Demonstrate orchestrator functionality"""
    orchestrator = BiddingOrchestrator()
    
    # Add a major news event spike (e.g., screen time study)
    orchestrator.add_event_spike(
        event_name="Major Screen Time Study Published",
        multiplier=3.0,
        duration_days=5,
        decay_factor=0.8
    )
    
    # Test different scenarios
    test_scenarios = [
        # Back-to-school season + peak hours
        BidRequest("campaign_001", 10.0, datetime.datetime(2024, 8, 15, 20, 30), 
                  ["parental controls", "screen time"], 5000.0),
        
        # Holiday season + peak hours
        BidRequest("campaign_002", 15.0, datetime.datetime(2024, 12, 20, 21, 0),
                  ["device monitoring", "family safety"], 3000.0),
        
        # Summer concerns + afternoon
        BidRequest("campaign_003", 8.0, datetime.datetime(2024, 6, 15, 14, 0),
                  ["child safety apps"], 2000.0),
        
        # Low demand period - night
        BidRequest("campaign_004", 12.0, datetime.datetime(2024, 3, 10, 3, 0),
                  ["screen time tracker"], 1000.0)
    ]
    
    print("Bidding Orchestrator Demo")
    print("=" * 60)
    
    # Process each bid request
    for request in test_scenarios:
        response = orchestrator.process_bid_request(request)
        
        print(f"\nCampaign: {response.campaign_id}")
        print(f"Date/Time: {request.target_date.strftime('%Y-%m-%d %H:%M')} ({request.target_date.strftime('%A')})")
        print(f"Original Bid: ${response.original_bid:.2f}")
        print(f"Adjusted Bid: ${response.adjusted_bid:.2f}")
        print(f"Demand Multiplier: {response.demand_multiplier:.2f}x")
        print(f"Reason: {response.adjustment_reason}")
        print(f"Action: {response.recommended_action}")
        print(f"Breakdown: S={response.breakdown['seasonal']:.2f}x, "
              f"H={response.breakdown['hourly']:.2f}x, "
              f"D={response.breakdown['daily']:.2f}x, "
              f"E={response.breakdown['events']:.2f}x")
    
    # Show current conditions
    print(f"\n{'Current Market Conditions'}")
    print("=" * 60)
    conditions = orchestrator.get_current_conditions()
    print(f"Market Condition: {conditions['market_condition']}")
    print(f"Overall Demand: {conditions['demand_multiplier']:.2f}x")
    print(f"Active Spikes: {conditions['active_spikes']}")
    print(f"Recommendation: {conditions['bidding_recommendation']}")


if __name__ == "__main__":
    demo_orchestrator()