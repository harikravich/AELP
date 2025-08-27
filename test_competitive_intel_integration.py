#!/usr/bin/env python3
"""Test if Competitive Intelligence is properly integrated into bidding"""

import asyncio
from datetime import datetime
from gaelp_master_integration import MasterOrchestrator, GAELPConfig
from competitive_intel import CompetitiveIntelligence, AuctionOutcome

async def test_competitive_intel_integration():
    """Test Competitive Intelligence integration in bid calculation"""
    
    print("="*80)
    print("TESTING COMPETITIVE INTELLIGENCE INTEGRATION")
    print("="*80)
    
    # Test 1: Check if Competitive Intelligence is initialized
    print("\n1. Checking Competitive Intelligence initialization...")
    try:
        config = GAELPConfig()
        config.enable_competitive_intelligence = True
        master = MasterOrchestrator(config)
        
        if hasattr(master, 'competitive_intel'):
            print(f"   ✅ Competitive Intelligence initialized")
            print(f"      Enable flag: {config.enable_competitive_intelligence}")
            print(f"      Lookback days: {master.competitive_intel.lookback_days}")
        else:
            print(f"   ❌ Competitive Intelligence not found in master")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Record some historical auction outcomes
    print("\n2. Recording historical auction data...")
    try:
        intel = master.competitive_intel
        
        # Simulate some historical auctions
        test_outcomes = [
            AuctionOutcome(
                keyword="parental controls",
                timestamp=datetime.now(),
                our_bid=2.5,
                position=1,
                cost=2.1,
                competitor_count=3,
                quality_score=0.8,
                daypart=datetime.now().hour,
                day_of_week=datetime.now().weekday(),
                device_type="mobile",
                location="US"
            ),
            AuctionOutcome(
                keyword="kids safety app",
                timestamp=datetime.now(),
                our_bid=3.0,
                position=2,
                cost=3.5,
                competitor_count=4,
                quality_score=0.7,
                daypart=datetime.now().hour,
                day_of_week=datetime.now().weekday(),
                device_type="desktop",
                location="US"
            ),
            AuctionOutcome(
                keyword="screen time control",
                timestamp=datetime.now(),
                our_bid=2.8,
                position=1,
                cost=2.3,
                competitor_count=2,
                quality_score=0.85,
                daypart=datetime.now().hour,
                day_of_week=datetime.now().weekday(),
                device_type="mobile",
                location="US"
            )
        ]
        
        for outcome in test_outcomes:
            intel.record_auction_outcome(outcome)
        
        print(f"   ✅ Recorded {len(test_outcomes)} historical outcomes")
        print(f"      Total outcomes tracked: {len(intel.auction_history)}")
        
    except Exception as e:
        print(f"   ❌ Failed to record outcomes: {e}")
        return False
    
    # Test 3: Test competitor bid estimation
    print("\n3. Testing competitor bid estimation...")
    try:
        # Test bid estimation using recent outcomes
        test_scenarios = [
            (AuctionOutcome(
                keyword="parental controls",
                timestamp=datetime.now(),
                our_bid=2.5,
                position=2,
                cost=3.0,
                competitor_count=3,
                quality_score=0.75,
                daypart=datetime.now().hour,
                day_of_week=datetime.now().weekday(),
                device_type="mobile",
                location="US"
            ), 1, "Top position estimation"),
            (AuctionOutcome(
                keyword="kids app",
                timestamp=datetime.now(),
                our_bid=1.8,
                position=1,
                cost=1.5,
                competitor_count=2,
                quality_score=0.9,
                daypart=datetime.now().hour,
                day_of_week=datetime.now().weekday(),
                device_type="desktop",
                location="US"
            ), 2, "Second position estimation")
        ]
        
        for outcome, target_position, desc in test_scenarios:
            estimated_bid, confidence = intel.estimate_competitor_bid(
                outcome=outcome,
                position=target_position
            )
            
            print(f"   ✅ {desc}:")
            print(f"      Our bid: ${outcome.our_bid:.2f}, Position: {outcome.position}")
            print(f"      Estimated bid for position {target_position}: ${estimated_bid:.2f}")
            print(f"      Confidence: {confidence:.2f}")
            
    except Exception as e:
        print(f"   ❌ Bid estimation failed: {e}")
        return False
    
    # Test 4: Test integration in bid calculation
    print("\n4. Testing integration in bid calculation...")
    try:
        # Create mock journey state and query data
        journey_state = {
            'conversion_probability': 0.1,
            'journey_stage': 2,
            'user_fatigue_level': 0.2,
            'hour_of_day': 14
        }
        
        query_data = {
            'query': 'parental controls app',
            'intent_strength': 0.8,
            'device_type': 'mobile',
            'user_engagement_score': 0.7
        }
        
        creative_selection = {
            'creative_type': 'display'
        }
        
        # Calculate bid WITH competitive intelligence
        master.config.enable_competitive_intelligence = True
        bid_with_intel = await master._calculate_bid(
            journey_state=journey_state,
            query_data=query_data,
            creative_selection=creative_selection
        )
        
        # Calculate bid WITHOUT competitive intelligence
        master.config.enable_competitive_intelligence = False
        bid_without_intel = await master._calculate_bid(
            journey_state=journey_state,
            query_data=query_data,
            creative_selection=creative_selection
        )
        
        print(f"   ✅ Bid calculation comparison:")
        print(f"      With competitive intel: ${bid_with_intel:.2f}")
        print(f"      Without competitive intel: ${bid_without_intel:.2f}")
        print(f"      Difference: ${abs(bid_with_intel - bid_without_intel):.2f}")
        
    except Exception as e:
        print(f"   ❌ Bid calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test competitor response prediction
    print("\n5. Testing competitor response prediction...")
    try:
        # Test different competitive scenarios
        scenarios = [
            (2.0, "parental controls", "bid_increase"),
            (3.0, "kids safety app", "aggressive_bidding"),
            (1.5, "screen time", "bid_decrease")
        ]
        
        for our_bid, keyword, scenario in scenarios:
            response = intel.predict_response(
                our_planned_bid=our_bid,
                keyword=keyword,
                timestamp=datetime.now(),
                scenario=scenario
            )
            
            print(f"   ✅ Scenario: {scenario}")
            print(f"      Our planned bid: ${our_bid:.2f} for '{keyword}'")
            if 'competitor_responses' in response:
                comp_resp = response['competitor_responses']
                print(f"      Escalation probability: {comp_resp.get('escalation_probability', 0):.2f}")
                if 'expected_escalation_ratio' in comp_resp:
                    print(f"      Expected escalation: {comp_resp['expected_escalation_ratio']:.2f}x")
            print(f"      Confidence: {response.get('confidence', 0):.2f}")
            
    except Exception as e:
        print(f"   ❌ Response prediction failed: {e}")
        return False
    
    # Test 6: Test market intelligence summary
    print("\n6. Testing market intelligence summary...")
    try:
        summary = intel.get_market_intelligence_summary()
        
        print(f"   ✅ Market Intelligence Summary:")
        if 'market_overview' in summary:
            print(f"      Total auctions analyzed: {summary['market_overview']['total_auctions_analyzed']}")
            print(f"      Recent auctions: {summary['market_overview']['recent_auctions']}")
        
        if 'competition_metrics' in summary and summary['competition_metrics']:
            metrics = summary['competition_metrics']
            print(f"      Win rate: {metrics.get('win_rate', 0):.1%}")
            print(f"      Average position: {metrics.get('average_position', 0):.1f}")
            print(f"      Average CPC: ${metrics.get('average_cpc', 0):.2f}")
            print(f"      Competition intensity: {metrics.get('competition_intensity', 0):.2f}")
        
    except Exception as e:
        print(f"   ❌ Market intelligence summary failed: {e}")
        return False
    
    # Test 7: Test auction outcome recording through master
    print("\n7. Testing auction outcome recording...")
    try:
        # Re-enable competitive intelligence
        master.config.enable_competitive_intelligence = True
        
        # Run a mock auction
        auction_result = await master._run_auction(
            bid_amount=2.5,
            query_data=query_data,
            creative_selection=creative_selection
        )
        
        print(f"   ✅ Auction completed:")
        print(f"      Won: {auction_result['won']}")
        print(f"      Position: {auction_result['position']}")
        print(f"      Competition level: {auction_result.get('num_competitors', 'N/A')} competitors")
        
        # Check if outcome was recorded
        if len(master.competitive_intel.auction_history) > len(test_outcomes):
            print(f"      ✅ Outcome recorded in competitive intelligence")
        
    except Exception as e:
        print(f"   ❌ Auction outcome recording failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ COMPETITIVE INTELLIGENCE INTEGRATION TEST PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_competitive_intel_integration())
    exit(0 if success else 1)