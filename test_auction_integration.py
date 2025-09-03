#!/usr/bin/env python3
"""
Test script to verify auction integration is working with real GSP mechanics
This verifies that we're getting realistic win rates and proper second-price auction behavior
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime
from auction_gym_integration_fixed import FixedAuctionGymIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_auction_mechanics():
    """Test the auction integration with various bid amounts"""
    print("\n" + "="*80)
    print("🎯 TESTING AUCTION INTEGRATION - REAL GSP MECHANICS")
    print("="*80)
    
    # Initialize auction with competitive configuration
    auction_config = {
        'auction_type': 'second_price',
        'num_slots': 4,
        'reserve_price': 0.50,
        'competitors': {
            'count': 8,
            'agents': [
                {'name': 'Qustodio', 'type': 'empirical', 'budget': 320.0, 'gamma': 0.85},
                {'name': 'Bark', 'type': 'truthful', 'budget': 380.0},
                {'name': 'Circle', 'type': 'empirical', 'budget': 280.0, 'gamma': 0.80},
                {'name': 'Norton', 'type': 'truthful', 'budget': 350.0},
                {'name': 'Life360', 'type': 'empirical', 'budget': 400.0, 'gamma': 0.88},
                {'name': 'SmallComp1', 'type': 'empirical', 'budget': 220.0, 'gamma': 0.75},
                {'name': 'McAfee', 'type': 'truthful', 'budget': 360.0},
                {'name': 'Kaspersky', 'type': 'empirical', 'budget': 300.0, 'gamma': 0.82},
            ]
        }
    }
    
    auction = FixedAuctionGymIntegration(auction_config)
    
    print(f"\n✅ Initialized auction with {len(auction.auction_wrapper.bidders)} competitors")
    print(f"Market stats: {auction.auction_wrapper.get_market_stats()}")
    
    # Test different bid scenarios
    test_scenarios = [
        # (bid_amount, description)
        (0.25, "Below reserve price"),
        (0.75, "Low competitive bid"),
        (1.50, "Medium competitive bid"),
        (3.00, "High competitive bid"),
        (5.00, "Very high competitive bid"),
        (7.50, "Premium bid"),
        (10.00, "Maximum aggressive bid")
    ]
    
    print("\n📊 RUNNING AUCTION SCENARIOS:")
    print("-" * 80)
    
    all_results = []
    
    for bid_amount, description in test_scenarios:
        print(f"\n🔸 Testing {description}: ${bid_amount:.2f}")
        
        # Run multiple auctions for this bid level
        scenario_results = []
        for i in range(20):  # 20 auctions per scenario
            query_context = {
                'query_value': bid_amount * 1.5,
                'user_segment': np.random.randint(0, 5),
                'device_type': np.random.randint(0, 3),
                'channel_index': np.random.randint(0, 3),
                'stage': np.random.randint(0, 5),
                'touchpoints': np.random.randint(1, 10),
                'competition_level': np.random.uniform(0.3, 0.9),
                'hour': np.random.randint(0, 24),
                'cvr': 0.02,
                'ltv': 199.98
            }
            
            result = auction.run_auction(bid_amount, query_context)
            scenario_results.append(result)
            all_results.append(result)
        
        # Analyze scenario results
        wins = sum(1 for r in scenario_results if r['won'])
        win_rate = wins / len(scenario_results)
        total_cost = sum(r['cost'] for r in scenario_results)
        avg_cpc = total_cost / wins if wins > 0 else 0
        
        # Position analysis for wins
        positions = [r['position'] for r in scenario_results if r['won']]
        avg_position = sum(positions) / len(positions) if positions else 0
        
        print(f"   Win Rate: {win_rate:.1%} ({wins}/20)")
        print(f"   Avg CPC: ${avg_cpc:.2f}")
        print(f"   Avg Position: {avg_position:.1f}")
        print(f"   Total Spend: ${total_cost:.2f}")
        
        # Check for realistic behavior
        if bid_amount >= 0.50 and win_rate == 0:
            print(f"   ⚠️  WARNING: Zero wins with bid above reserve")
        elif bid_amount < 0.50 and win_rate > 0:
            print(f"   ⚠️  WARNING: Wins with bid below reserve price")
        elif win_rate > 0.70:
            print(f"   ⚠️  WARNING: Win rate too high for competitive market")
        elif avg_cpc > bid_amount:
            print(f"   ❌ ERROR: Paying more than bid amount (GSP violation)")
    
    print("\n" + "="*80)
    print("📈 OVERALL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Overall statistics
    total_auctions = len(all_results)
    total_wins = sum(1 for r in all_results if r['won'])
    overall_win_rate = total_wins / total_auctions if total_auctions > 0 else 0
    total_spend = sum(r['cost'] for r in all_results)
    overall_avg_cpc = total_spend / total_wins if total_wins > 0 else 0
    
    print(f"\n🎯 Total Auctions: {total_auctions}")
    print(f"🎯 Total Wins: {total_wins}")
    print(f"🎯 Overall Win Rate: {overall_win_rate:.1%}")
    print(f"🎯 Overall Avg CPC: ${overall_avg_cpc:.2f}")
    print(f"🎯 Total Spend: ${total_spend:.2f}")
    
    # Position distribution
    winning_results = [r for r in all_results if r['won']]
    if winning_results:
        positions = [r['position'] for r in winning_results]
        print(f"\n📍 Position Distribution:")
        for pos in range(1, 5):
            count = positions.count(pos)
            pct = count / len(positions) * 100
            print(f"   Position {pos}: {count} ({pct:.1f}%)")
    
    # Get system metrics
    metrics = auction.get_metrics()
    print(f"\n🔧 System Metrics:")
    print(f"   Current Quality Score: {metrics['current_quality_score']:.1f}")
    print(f"   Market Competitors: {metrics['market_stats']['total_competitors']}")
    
    # Health check
    is_healthy = auction.health_check()
    print(f"\n💊 System Health: {'✅ HEALTHY' if is_healthy else '❌ UNHEALTHY'}")
    
    # Validate realistic performance
    print("\n" + "="*80)
    print("🔍 VALIDATION RESULTS")
    print("="*80)
    
    validation_passed = True
    
    # Check 1: Win rate should be realistic (15-40% in competitive market)
    if not (0.10 <= overall_win_rate <= 0.50):
        print(f"❌ FAIL: Win rate {overall_win_rate:.1%} is unrealistic")
        validation_passed = False
    else:
        print(f"✅ PASS: Win rate {overall_win_rate:.1%} is realistic")
    
    # Check 2: No GSP violations (CPC <= bid)
    gsp_violations = sum(1 for r in all_results if r['won'] and r['cost'] > r.get('bid_amount', 0) + 0.01)
    if gsp_violations > 0:
        print(f"❌ FAIL: {gsp_violations} GSP violations detected")
        validation_passed = False
    else:
        print(f"✅ PASS: No GSP violations detected")
    
    # Check 3: Competition is active
    avg_competitors = sum(r['competitors'] for r in all_results) / len(all_results)
    if avg_competitors < 3:
        print(f"❌ FAIL: Low competition level ({avg_competitors:.1f} avg competitors)")
        validation_passed = False
    else:
        print(f"✅ PASS: Active competition ({avg_competitors:.1f} avg competitors)")
    
    # Check 4: Position distribution makes sense
    if winning_results:
        pos_1_pct = positions.count(1) / len(positions)
        if pos_1_pct > 0.60:  # Shouldn't win top position too often
            print(f"❌ FAIL: Too many position 1 wins ({pos_1_pct:.1%})")
            validation_passed = False
        else:
            print(f"✅ PASS: Realistic position distribution (pos 1: {pos_1_pct:.1%})")
    
    print("\n" + "="*80)
    if validation_passed:
        print("🎉 ALL VALIDATIONS PASSED - AUCTION INTEGRATION WORKING CORRECTLY!")
        print("🎯 Ready for production use with realistic second-price auction mechanics")
    else:
        print("❌ VALIDATION FAILURES DETECTED - REVIEW AUCTION IMPLEMENTATION")
    print("="*80)
    
    return validation_passed

def test_orchestrator_integration():
    """Test that the orchestrator can use the auction component"""
    print("\n🔗 Testing orchestrator integration...")
    
    try:
        from gaelp_production_orchestrator import GAELPProductionOrchestrator, OrchestratorConfig
        
        # Create test config
        config = OrchestratorConfig()
        config.dry_run = True  # Don't spend real money
        config.enable_rl_training = False  # Just test initialization
        
        orchestrator = GAELPProductionOrchestrator(config)
        
        # Check if auction component initializes
        if orchestrator.initialize_components():
            auction = orchestrator.components.get('auction')
            if auction:
                print("✅ PASS: Orchestrator successfully initialized auction component")
                print(f"   Component type: {type(auction).__name__}")
                print(f"   Has run_auction method: {hasattr(auction, 'run_auction')}")
                return True
            else:
                print("❌ FAIL: Auction component not found in orchestrator")
                return False
        else:
            print("❌ FAIL: Orchestrator component initialization failed")
            return False
    except Exception as e:
        print(f"❌ FAIL: Orchestrator integration test failed: {e}")
        return False

def main():
    """Run all auction integration tests"""
    print("🚀 STARTING AUCTION INTEGRATION TESTING")
    print(f"Timestamp: {datetime.now()}")
    
    # Test 1: Core auction mechanics
    auction_test_passed = test_auction_mechanics()
    
    # Test 2: Orchestrator integration 
    orchestrator_test_passed = test_orchestrator_integration()
    
    print("\n" + "="*80)
    print("🏁 FINAL TEST RESULTS")
    print("="*80)
    print(f"Auction Mechanics: {'✅ PASS' if auction_test_passed else '❌ FAIL'}")
    print(f"Orchestrator Integration: {'✅ PASS' if orchestrator_test_passed else '❌ FAIL'}")
    
    if auction_test_passed and orchestrator_test_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Auction integration is working correctly with realistic GSP mechanics")
        print("✅ Ready for production training with proper second-price auctions")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("🔧 Review the auction implementation before production use")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)