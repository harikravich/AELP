#!/usr/bin/env python3
"""Test if Competitor Agents enum fix works"""

from datetime import datetime
from competitor_agents import CompetitorAgentManager, AuctionContext, UserValueTier

def test_competitor_agents():
    """Test Competitor Agents with correct enum values"""
    
    print("="*80)
    print("TESTING COMPETITOR AGENTS FIX")
    print("="*80)
    
    # Test 1: Initialize manager
    print("\n1. Initializing CompetitorAgentManager...")
    try:
        manager = CompetitorAgentManager()
        print(f"   ✅ Manager initialized with {len(manager.agents)} agents")
        for agent_id, agent in manager.agents.items():
            print(f"      - {agent_id}: {agent.__class__.__name__}")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Test each UserValueTier enum
    print("\n2. Testing UserValueTier enum values...")
    tiers = [UserValueTier.LOW, UserValueTier.MEDIUM, UserValueTier.HIGH, UserValueTier.PREMIUM]
    
    for tier in tiers:
        try:
            context = AuctionContext(
                user_id=f"test_{tier.value}",
                user_value_tier=tier,
                timestamp=datetime.now(),
                device_type="mobile",
                geo_location="US",
                time_of_day=14,
                day_of_week=2,
                market_competition=0.7,
                keyword_competition=0.5,
                seasonality_factor=1.0,
                user_engagement_score=0.8,
                conversion_probability=0.05
            )
            
            # Run auction with this tier
            results = manager.run_auction(context)
            
            # Find the winner (agent with position 1)
            winner = None
            for agent_name, result in results.items():
                if result.won:
                    winner = agent_name
                    break
            
            print(f"   ✅ {tier.name}: Auction completed, winner: {winner}")
            
        except Exception as e:
            print(f"   ❌ {tier.name} failed: {e}")
            return False
    
    # Test 3: Test auction with different contexts
    print("\n3. Testing various auction scenarios...")
    scenarios = [
        ("Low value user", UserValueTier.LOW, 1.0),
        ("Medium value user", UserValueTier.MEDIUM, 2.0),
        ("High value user", UserValueTier.HIGH, 3.0),
        ("Premium user", UserValueTier.PREMIUM, 4.0)
    ]
    
    for scenario_name, tier, bid in scenarios:
        try:
            context = AuctionContext(
                user_id=f"scenario_{tier.value}",
                user_value_tier=tier,
                timestamp=datetime.now(),
                device_type="desktop",
                geo_location="US",
                time_of_day=10,
                day_of_week=1,
                market_competition=0.5,
                keyword_competition=0.6,
                seasonality_factor=1.2,
                user_engagement_score=0.7,
                conversion_probability=0.08
            )
            
            results = manager.run_auction(context)
            
            # Find the winner
            winner = None
            winning_bid = 0
            clearing_price = 0
            for agent_name, result in results.items():
                if result.won:
                    winner = agent_name
                    winning_bid = result.bid_amount
                    clearing_price = result.winning_price
                    break
            
            print(f"   ✅ {scenario_name}:")
            print(f"      User tier: {tier.value}")
            print(f"      Winner: {winner} at ${winning_bid:.2f}")
            print(f"      Cleared price: ${clearing_price:.2f}")
            
        except Exception as e:
            print(f"   ❌ {scenario_name} failed: {e}")
            return False
    
    print("\n" + "="*80)
    print("✅ COMPETITOR AGENTS TEST PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_competitor_agents()
    exit(0 if success else 1)