"""
Temporal Effects Module for Demand Forecasting and Bidding

Models seasonal patterns, time-of-day effects, and event-driven spikes
to predict and adjust for temporal variations in demand.
"""

import datetime
from typing import Dict, Tuple, List, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SeasonalEvent(Enum):
    """Major seasonal events that affect demand"""
    BACK_TO_SCHOOL = "back_to_school"
    HOLIDAY_GIFTS = "holiday_gifts"
    SUMMER_CONCERNS = "summer_concerns"
    NEW_YEAR_RESOLUTIONS = "new_year_resolutions"
    SPRING_AWARENESS = "spring_awareness"


@dataclass
class EventSpike:
    """Configuration for news/event-driven demand spikes"""
    name: str
    multiplier: float
    duration_days: int
    decay_factor: float = 0.8  # Daily decay rate


class TemporalEffects:
    """
    Models temporal patterns in demand including:
    - Seasonal effects (back-to-school, holidays, summer)
    - Time-of-day patterns (parent browsing times)
    - Day-of-week effects
    - Event-driven spikes (news stories)
    """
    
    def __init__(self):
        # Seasonal multipliers by month
        self.seasonal_multipliers = {
            1: 1.2,   # January - New Year resolutions
            2: 1.0,   # February - baseline
            3: 1.1,   # March - spring awareness
            4: 1.0,   # April - baseline
            5: 1.0,   # May - baseline
            6: 1.3,   # June - summer concerns start
            7: 1.3,   # July - summer concerns peak
            8: 2.4,   # August - back-to-school surge
            9: 2.4,   # September - back-to-school continues
            10: 1.2,  # October - post-school adjustment
            11: 1.1,  # November - holiday prep
            12: 1.8   # December - holiday device gifts
        }
        
        # Hour-of-day multipliers (24-hour format)
        # Peak at 8PM-10PM when parents are browsing
        self.hourly_multipliers = {
            0: 0.3, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.3,
            6: 0.5, 7: 0.7, 8: 0.8, 9: 0.9, 10: 1.0, 11: 1.1,
            12: 1.2, 13: 1.1, 14: 1.0, 15: 1.0, 16: 1.1, 17: 1.3,
            18: 1.5, 19: 1.7, 20: 2.0, 21: 2.0, 22: 1.5, 23: 0.8
        }
        
        # Day-of-week multipliers (Monday=0, Sunday=6)
        # Weekends higher as parents have more time to research
        self.daily_multipliers = {
            0: 1.1,   # Monday
            1: 1.0,   # Tuesday
            2: 1.0,   # Wednesday
            3: 1.0,   # Thursday
            4: 1.1,   # Friday
            5: 1.4,   # Saturday - high
            6: 1.3    # Sunday - high
        }
        
        # Active event spikes
        self.active_spikes: List[Tuple[EventSpike, datetime.datetime]] = []
        
        # Base demand level (normalized to 1.0)
        self.base_demand = 1.0
    
    def get_seasonal_multiplier(self, date: datetime.datetime) -> float:
        """
        Get seasonal demand multiplier for a given date
        
        Args:
            date: Target date
            
        Returns:
            Seasonal multiplier (1.0 = baseline)
        """
        return self.seasonal_multipliers.get(date.month, 1.0)
    
    def get_hourly_multiplier(self, hour: int) -> float:
        """
        Get hour-of-day demand multiplier
        
        Args:
            hour: Hour in 24-hour format (0-23)
            
        Returns:
            Hourly multiplier (1.0 = baseline)
        """
        return self.hourly_multipliers.get(hour, 1.0)
    
    def get_daily_multiplier(self, weekday: int) -> float:
        """
        Get day-of-week demand multiplier
        
        Args:
            weekday: Weekday (0=Monday, 6=Sunday)
            
        Returns:
            Daily multiplier (1.0 = baseline)
        """
        return self.daily_multipliers.get(weekday, 1.0)
    
    def add_event_spike(self, event: EventSpike, start_date: datetime.datetime):
        """
        Add an event-driven demand spike
        
        Args:
            event: Event spike configuration
            start_date: When the spike starts
        """
        self.active_spikes.append((event, start_date))
    
    def get_event_multiplier(self, date: datetime.datetime) -> float:
        """
        Calculate combined multiplier from all active event spikes
        
        Args:
            date: Target date
            
        Returns:
            Combined event multiplier
        """
        total_multiplier = 1.0
        
        # Clean up expired spikes and calculate active ones
        active_spikes = []
        for spike, start_date in self.active_spikes:
            days_since_start = (date - start_date).days
            
            if days_since_start >= 0 and days_since_start < spike.duration_days:
                # Calculate decaying multiplier
                decay = spike.decay_factor ** days_since_start
                spike_multiplier = 1.0 + (spike.multiplier - 1.0) * decay
                total_multiplier *= spike_multiplier
                active_spikes.append((spike, start_date))
        
        # Update active spikes list
        self.active_spikes = active_spikes
        
        return total_multiplier
    
    def predict_demand(self, date: datetime.datetime) -> float:
        """
        Predict total demand multiplier for a specific date/time
        
        Args:
            date: Target datetime
            
        Returns:
            Combined demand multiplier
        """
        seasonal = self.get_seasonal_multiplier(date)
        hourly = self.get_hourly_multiplier(date.hour)
        daily = self.get_daily_multiplier(date.weekday())
        events = self.get_event_multiplier(date)
        
        return self.base_demand * seasonal * hourly * daily * events
    
    def predict_demand_range(self, start_date: datetime.datetime, 
                           end_date: datetime.datetime, 
                           interval_hours: int = 1) -> List[Tuple[datetime.datetime, float]]:
        """
        Predict demand over a time range
        
        Args:
            start_date: Start of prediction range
            end_date: End of prediction range
            interval_hours: Hours between predictions
            
        Returns:
            List of (datetime, demand_multiplier) tuples
        """
        predictions = []
        current_date = start_date
        
        while current_date <= end_date:
            demand = self.predict_demand(current_date)
            predictions.append((current_date, demand))
            current_date += datetime.timedelta(hours=interval_hours)
        
        return predictions
    
    def adjust_bidding(self, base_bid: float, date: datetime.datetime, 
                      max_adjustment: float = 2.0) -> Dict[str, float]:
        """
        Adjust bidding based on predicted demand
        
        Args:
            base_bid: Base bid amount
            date: Target datetime
            max_adjustment: Maximum bid adjustment multiplier
            
        Returns:
            Dictionary with bid adjustments and breakdown
        """
        demand_multiplier = self.predict_demand(date)
        
        # Cap the adjustment to prevent excessive bidding
        capped_multiplier = min(demand_multiplier, max_adjustment)
        adjusted_bid = base_bid * capped_multiplier
        
        # Get component breakdown
        seasonal = self.get_seasonal_multiplier(date)
        hourly = self.get_hourly_multiplier(date.hour)
        daily = self.get_daily_multiplier(date.weekday())
        events = self.get_event_multiplier(date)
        
        return {
            'original_bid': base_bid,
            'adjusted_bid': adjusted_bid,
            'demand_multiplier': demand_multiplier,
            'capped_multiplier': capped_multiplier,
            'breakdown': {
                'seasonal': seasonal,
                'hourly': hourly,
                'daily': daily,
                'events': events
            },
            'adjustment_reason': self._get_adjustment_reason(date, demand_multiplier)
        }
    
    def _get_adjustment_reason(self, date: datetime.datetime, multiplier: float) -> str:
        """Generate human-readable explanation for bid adjustment"""
        reasons = []
        
        # Check seasonal effects
        seasonal = self.get_seasonal_multiplier(date)
        if seasonal > 1.5:
            if date.month in [8, 9]:
                reasons.append("Back-to-school surge")
            elif date.month == 12:
                reasons.append("Holiday gift season")
        elif seasonal > 1.2:
            if date.month in [6, 7]:
                reasons.append("Summer screen time concerns")
        
        # Check time effects
        hourly = self.get_hourly_multiplier(date.hour)
        if hourly > 1.5:
            reasons.append("Peak parent browsing hours (8-10PM)")
        
        # Check day effects
        if date.weekday() >= 5:
            reasons.append("Weekend research time")
        
        # Check events
        events = self.get_event_multiplier(date)
        if events > 1.5:
            reasons.append("Active news/event spike")
        
        if not reasons:
            if multiplier < 0.8:
                reasons.append("Low demand period")
            else:
                reasons.append("Normal demand period")
        
        return "; ".join(reasons)
    
    def get_optimal_timing(self, target_date: datetime.date, 
                          duration_hours: int = 24) -> Dict[str, any]:
        """
        Find optimal timing within a target date for campaign launch
        
        Args:
            target_date: Target date for campaign
            duration_hours: Hours to analyze around target date
            
        Returns:
            Dictionary with optimal timing recommendations
        """
        start_datetime = datetime.datetime.combine(target_date, datetime.time(0, 0))
        end_datetime = start_datetime + datetime.timedelta(hours=duration_hours)
        
        predictions = self.predict_demand_range(start_datetime, end_datetime)
        
        # Find peak and valley times
        peak_time = max(predictions, key=lambda x: x[1])
        valley_time = min(predictions, key=lambda x: x[1])
        
        # Calculate average demand
        avg_demand = np.mean([pred[1] for pred in predictions])
        
        return {
            'peak_time': {
                'datetime': peak_time[0],
                'demand_multiplier': peak_time[1],
                'recommended_for': 'Maximum reach campaigns'
            },
            'valley_time': {
                'datetime': valley_time[0],
                'demand_multiplier': valley_time[1],
                'recommended_for': 'Cost-efficient campaigns'
            },
            'average_demand': avg_demand,
            'recommendations': self._generate_timing_recommendations(predictions, avg_demand)
        }
    
    def _generate_timing_recommendations(self, predictions: List[Tuple[datetime.datetime, float]], 
                                       avg_demand: float) -> List[str]:
        """Generate timing recommendations based on demand patterns"""
        recommendations = []
        
        # Find periods above/below average
        high_periods = [p for p in predictions if p[1] > avg_demand * 1.2]
        low_periods = [p for p in predictions if p[1] < avg_demand * 0.8]
        
        if high_periods:
            peak_hours = [p[0].hour for p in high_periods]
            common_peak = max(set(peak_hours), key=peak_hours.count)
            recommendations.append(f"Schedule high-priority campaigns around {common_peak}:00")
        
        if low_periods:
            valley_hours = [p[0].hour for p in low_periods]
            common_valley = max(set(valley_hours), key=valley_hours.count)
            recommendations.append(f"Schedule cost-sensitive campaigns around {common_valley}:00")
        
        # Weekend vs weekday advice
        weekend_avg = np.mean([p[1] for p in predictions if p[0].weekday() >= 5])
        weekday_avg = np.mean([p[1] for p in predictions if p[0].weekday() < 5])
        
        if weekend_avg > weekday_avg * 1.2:
            recommendations.append("Focus weekend campaigns for higher engagement")
        elif weekday_avg > weekend_avg * 1.2:
            recommendations.append("Weekday campaigns more cost-effective")
        
        return recommendations


# Example usage and testing functions
def create_sample_events():
    """Create sample event spikes for testing"""
    return [
        EventSpike("screen_time_news", 3.0, 5, 0.7),  # News story causes 3x spike for 5 days
        EventSpike("celebrity_endorsement", 2.5, 3, 0.8),
        EventSpike("competitor_scandal", 2.0, 7, 0.9)
    ]


def demo_temporal_effects():
    """Demonstrate temporal effects functionality"""
    effects = TemporalEffects()
    
    # Add a sample event spike
    news_spike = EventSpike("major_screen_time_study", 3.0, 5, 0.8)
    effects.add_event_spike(news_spike, datetime.datetime.now())
    
    # Predict demand for different times
    test_dates = [
        datetime.datetime(2024, 8, 15, 20, 30),  # Back-to-school + peak hour
        datetime.datetime(2024, 12, 20, 21, 0),  # Holiday season + peak hour
        datetime.datetime(2024, 6, 15, 14, 0),   # Summer concerns + afternoon
        datetime.datetime(2024, 3, 10, 3, 0),    # Low season + night
    ]
    
    print("Temporal Effects Demo")
    print("=" * 50)
    
    for date in test_dates:
        demand = effects.predict_demand(date)
        bid_info = effects.adjust_bidding(10.0, date)  # $10 base bid
        
        print(f"\nDate: {date.strftime('%Y-%m-%d %H:%M')} ({date.strftime('%A')})")
        print(f"Demand Multiplier: {demand:.2f}x")
        print(f"Recommended Bid: ${bid_info['adjusted_bid']:.2f}")
        print(f"Reason: {bid_info['adjustment_reason']}")
        print(f"Breakdown: Seasonal={bid_info['breakdown']['seasonal']:.2f}x, "
              f"Hourly={bid_info['breakdown']['hourly']:.2f}x, "
              f"Daily={bid_info['breakdown']['daily']:.2f}x, "
              f"Events={bid_info['breakdown']['events']:.2f}x")


if __name__ == "__main__":
    demo_temporal_effects()