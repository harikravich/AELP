#!/usr/bin/env python3
"""
Demonstration of RecSim-AuctionGym Bridge
Shows how user segments map to auction participation and query generation.
"""

from recsim_auction_bridge import RecSimAuctionBridge, UserSegment
import numpy as np

def main():
    print("🔗 RecSim-AuctionGym Bridge Demo")
    print("=" * 60)
    
    # Initialize the bridge
    bridge = RecSimAuctionBridge()
    
    print("\n📊 User Segment Bidding Profiles")
    print("-" * 40)
    
    # Show how each segment behaves in auctions
    for segment in UserSegment:
        user_id = f"demo_{segment.value}"
        
        # Generate auction signals for this segment
        signals = bridge.user_to_auction_signals(
            user_id=user_id,
            context={'hour': 20, 'device': 'mobile', 'brand_query': False}
        )
        
        print(f"\n🎯 {segment.value.upper().replace('_', ' ')}")
        print(f"   Suggested Bid: ${signals['suggested_bid']:.2f}")
        print(f"   Quality Score: {signals['quality_score']:.2f}")
        print(f"   Participation Rate: {signals['participation_probability']:.0%}")
        print(f"   Price Sensitivity: {signals['price_sensitivity']:.2f}")
        print(f"   Time Sensitivity: {signals['time_sensitivity']:.2f}")
    
    print("\n🔍 Query Generation Examples")
    print("-" * 40)
    
    # Show query generation for different segments
    example_segments = [UserSegment.IMPULSE_BUYER, UserSegment.RESEARCHER, UserSegment.LOYAL_CUSTOMER]
    
    for segment in example_segments:
        print(f"\n🤔 {segment.value.upper().replace('_', ' ')} Queries:")
        
        user_id = f"query_{segment.value}"
        
        for i in range(3):
            query_data = bridge.generate_query_from_state(
                user_id=user_id,
                product_category="running shoes",
                brand="nike" if i == 0 else None
            )
            
            print(f"   • '{query_data['query']}'")
            print(f"     Stage: {query_data['journey_stage']} | Intent: {query_data['intent_strength']:.1f}")
    
    print("\n💰 Complete User Journey Simulation")
    print("-" * 40)
    
    # Simulate complete sessions for different segments
    segments_to_test = [UserSegment.IMPULSE_BUYER, UserSegment.RESEARCHER, UserSegment.LOYAL_CUSTOMER]
    
    for segment in segments_to_test:
        user_id = f"journey_{segment.value}"
        
        # Run a 5-query session
        session = bridge.simulate_user_auction_session(
            user_id=user_id,
            num_queries=5,
            product_category="sneakers"
        )
        
        print(f"\n🛍️  {segment.value.upper().replace('_', ' ')} JOURNEY:")
        print(f"   Queries Generated: {len(session['queries'])}")
        print(f"   Auctions Entered: {len(session['auctions'])}")
        print(f"   Total Spend: ${session['total_cost']:.2f}")
        print(f"   Revenue Generated: ${session['total_revenue']:.2f}")
        print(f"   Click-through Rate: {session['clicks']}/{len(session['queries'])}")
        
        if session['total_cost'] > 0:
            roas = session['total_revenue'] / session['total_cost']
            print(f"   Return on Ad Spend: {roas:.1f}x")
        
        # Show sample queries from this session
        print("   Sample Queries:")
        for i, query in enumerate(session['queries'][:3]):
            print(f"     {i+1}. '{query['query']}' ({query['journey_stage']})")
    
    print("\n📈 Cross-Segment Analytics")
    print("-" * 40)
    
    # Get analytics across all sessions
    analytics = bridge.get_bridge_analytics()
    
    print(f"Total Sessions Analyzed: {analytics['total_sessions']}")
    
    print("\nQuery Generation by Stage:")
    all_stages = {}
    for segment_stats in analytics['query_generation_stats'].values():
        for stage, count in segment_stats.items():
            all_stages[stage] = all_stages.get(stage, 0) + count
    
    for stage, count in sorted(all_stages.items()):
        print(f"   {stage.capitalize()}: {count} queries")
    
    print("\nSegment Performance Summary:")
    for segment, perf in analytics['segment_performance'].items():
        if perf['sessions'] > 0:
            avg_cost = perf['total_cost'] / perf['sessions']
            avg_revenue = perf['total_revenue'] / perf['sessions']
            print(f"   {segment.replace('_', ' ').title()}: "
                  f"${avg_cost:.2f} cost, ${avg_revenue:.2f} revenue per session")
    
    print("\n✅ Bridge successfully connects RecSim users to auction participation!")
    print("   - User segments drive bidding behavior")
    print("   - Journey stages generate appropriate queries")
    print("   - User signals optimize auction performance")
    print("   - Complete simulation tracks user lifecycle")

if __name__ == "__main__":
    main()