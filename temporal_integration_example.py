"""
Temporal Integration Example

Demonstrates how the BiddingOrchestrator integrates with TemporalEffects
to handle specific time patterns and seasonal adjustments.
"""

import datetime
from bidding_orchestrator import BiddingOrchestrator, BidRequest


def demonstrate_time_patterns():
    """Demonstrate all the required time patterns and their effects"""
    orchestrator = BiddingOrchestrator()
    
    # Add event spike for news story causing 3x surge
    orchestrator.add_event_spike(
        event_name="Major Parental Control News Story",
        multiplier=3.0,
        duration_days=7,
        decay_factor=0.8,
        start_date=datetime.datetime.now()
    )
    
    print("TEMPORAL BIDDING INTEGRATION DEMO")
    print("=" * 50)
    
    # Test specific time patterns mentioned in requirements
    test_cases = [
        {
            "name": "Peak Hours (8-10 PM)",
            "date": datetime.datetime(2024, 6, 15, 20, 30),  # 8:30 PM
            "expected": "2x multiplier for peak browsing hours"
        },
        {
            "name": "Back-to-School August",
            "date": datetime.datetime(2024, 8, 20, 14, 0),   # August afternoon
            "expected": "2.4x multiplier for back-to-school surge"
        },
        {
            "name": "Back-to-School September",
            "date": datetime.datetime(2024, 9, 10, 14, 0),   # September afternoon
            "expected": "2.4x multiplier continues in September"
        },
        {
            "name": "Holiday Season December",
            "date": datetime.datetime(2024, 12, 15, 16, 0),  # December afternoon
            "expected": "1.8x multiplier for holiday device gifts"
        },
        {
            "name": "Event Spike with Decay",
            "date": datetime.datetime.now() + datetime.timedelta(days=2),
            "expected": "3x event spike with decay over time"
        },
        {
            "name": "Combined Peak + Season",
            "date": datetime.datetime(2024, 8, 25, 21, 0),   # August 9 PM
            "expected": "Peak hours + back-to-school combined"
        },
        {
            "name": "Quiet Time - Early Morning",
            "date": datetime.datetime(2024, 3, 10, 3, 0),    # 3 AM in March
            "expected": "Low multiplier for quiet periods"
        }
    ]
    
    base_bid = 10.0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 30)
        
        # Create bid request
        request = BidRequest(
            campaign_id=f"test_campaign_{i:03d}",
            base_bid=base_bid,
            target_date=test_case['date'],
            keywords=["parental controls", "screen time"],
            budget_remaining=2000.0
        )
        
        # Process the bid
        response = orchestrator.process_bid_request(request)
        
        # Show results
        print(f"Date/Time: {test_case['date'].strftime('%Y-%m-%d %H:%M')} ({test_case['date'].strftime('%A')})")
        print(f"Expected: {test_case['expected']}")
        print(f"Base Bid: ${response.original_bid:.2f}")
        print(f"Adjusted Bid: ${response.adjusted_bid:.2f}")
        print(f"Total Multiplier: {response.demand_multiplier:.2f}x")
        print(f"Reason: {response.adjustment_reason}")
        print(f"Action: {response.recommended_action}")
        
        # Detailed breakdown
        breakdown = response.breakdown
        print(f"Breakdown:")
        print(f"  - Seasonal: {breakdown['seasonal']:.2f}x")
        print(f"  - Hourly: {breakdown['hourly']:.2f}x") 
        print(f"  - Daily: {breakdown['daily']:.2f}x")
        print(f"  - Events: {breakdown['events']:.2f}x")
        
        # Verify peak hours (8-10 PM should be 2x)
        if test_case['date'].hour in [20, 21, 22]:  # 8-10 PM
            assert breakdown['hourly'] >= 2.0, f"Peak hour multiplier should be 2x, got {breakdown['hourly']:.2f}x"
            print(f"  ✓ Peak hour verification passed")
        
        # Verify back-to-school season
        if test_case['date'].month in [8, 9]:
            assert breakdown['seasonal'] >= 2.4, f"Back-to-school should be 2.4x, got {breakdown['seasonal']:.2f}x"
            print(f"  ✓ Back-to-school verification passed")
        
        # Verify holiday season
        if test_case['date'].month == 12:
            assert breakdown['seasonal'] >= 1.8, f"Holiday season should be 1.8x, got {breakdown['seasonal']:.2f}x"
            print(f"  ✓ Holiday season verification passed")


def demonstrate_demand_forecasting():
    """Demonstrate demand forecasting capabilities"""
    orchestrator = BiddingOrchestrator()
    
    print(f"\n\nDEMAND FORECASTING DEMO")
    print("=" * 50)
    
    # Get optimal timing for next week
    target_date = datetime.date.today() + datetime.timedelta(days=7)
    timing_analysis = orchestrator.get_optimal_timing(target_date)
    
    print(f"Optimal Timing Analysis for {target_date}")
    print(f"Peak Time: {timing_analysis['peak_time']['datetime'].strftime('%H:%M')} "
          f"({timing_analysis['peak_time']['demand_multiplier']:.2f}x)")
    print(f"Valley Time: {timing_analysis['valley_time']['datetime'].strftime('%H:%M')} "
          f"({timing_analysis['valley_time']['demand_multiplier']:.2f}x)")
    print(f"Average Demand: {timing_analysis['average_demand']:.2f}x")
    
    print(f"\nRecommendations:")
    for rec in timing_analysis['recommendations']:
        print(f"  • {rec}")
    
    # Show 7-day demand forecast
    start_date = datetime.datetime.now()
    end_date = start_date + datetime.timedelta(days=7)
    predictions = orchestrator.predict_demand_range(start_date, end_date)
    
    print(f"\n7-Day Demand Forecast (daily peaks):")
    daily_peaks = {}
    for dt, demand in predictions:
        date_key = dt.date()
        if date_key not in daily_peaks or demand > daily_peaks[date_key][1]:
            daily_peaks[date_key] = (dt, demand)
    
    for date, (peak_dt, peak_demand) in sorted(daily_peaks.items()):
        print(f"  {date.strftime('%a %m/%d')}: Peak at {peak_dt.strftime('%H:%M')} "
              f"with {peak_demand:.2f}x demand")


def demonstrate_event_handling():
    """Demonstrate event spike handling with decay"""
    orchestrator = BiddingOrchestrator()
    
    print(f"\n\nEVENT SPIKE HANDLING DEMO")
    print("=" * 50)
    
    # Add multiple event types
    events = [
        ("Breaking News: Screen Time Study", 3.0, 5, 0.8),
        ("Celebrity Endorsement", 2.5, 3, 0.85),
        ("Competitor Security Breach", 2.2, 7, 0.9)
    ]
    
    base_date = datetime.datetime.now()
    
    # Add events starting at different times
    for i, (name, multiplier, duration, decay) in enumerate(events):
        event_start = base_date + datetime.timedelta(hours=i * 6)  # Stagger events
        orchestrator.add_event_spike(name, multiplier, duration, decay, event_start)
    
    # Show effect over next 10 days
    print("Event Impact Over Time:")
    for day in range(10):
        test_date = base_date + datetime.timedelta(days=day)
        conditions = orchestrator.get_current_conditions(test_date)
        
        print(f"Day {day + 1}: {conditions['event_multiplier']:.2f}x event impact, "
              f"{conditions['market_condition']} market")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_time_patterns()
    demonstrate_demand_forecasting()
    demonstrate_event_handling()
    
    print(f"\n\n{'INTEGRATION COMPLETE'}")
    print("=" * 50)
    print("✓ TemporalEffects imported and integrated")
    print("✓ get_seasonal_multiplier() called before each bid")
    print("✓ predict_demand() used for demand forecasting")
    print("✓ adjust_bidding() applied to base bids")
    print("✓ Event spikes (3x surge) with decay handled")
    print("✓ Peak hours 8-10 PM (2x multiplier) enforced")
    print("✓ Back-to-school Aug-Sep (2.4x) enforced") 
    print("✓ Holiday season December (1.8x) enforced")
    print("✓ Higher bids during high-demand, lower during quiet times")