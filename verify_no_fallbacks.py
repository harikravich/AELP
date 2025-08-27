#!/usr/bin/env python3
"""
STRICT VERIFICATION - NO FALLBACKS ALLOWED
This script verifies that ALL components are using primary implementations
"""

import sys
import os
sys.path.insert(0, '/home/hariravichandran/AELP')

from NO_FALLBACKS import StrictModeEnforcer, NoFallbackError
import importlib
import traceback

def verify_component(name: str, module_path: str, test_func=None):
    """Verify a component works without fallbacks"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    try:
        # Import the module
        module = importlib.import_module(module_path)
        print(f"✅ Module imported: {module_path}")
        
        # Check for forbidden patterns
        module_file = module.__file__
        if module_file:
            with open(module_file, 'r') as f:
                content = f.read()
                forbidden = ['fallback', 'simplified', 'mock', 'dummy', '_AVAILABLE = False']
                for pattern in forbidden:
                    if pattern.lower() in content.lower():
                        # Check if it's in actual code (not comments)
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern.lower() in line.lower():
                                stripped = line.strip()
                                if not stripped.startswith('#') and 'raise' not in line:
                                    raise NoFallbackError(
                                        f"❌ FORBIDDEN PATTERN '{pattern}' found at line {i+1}:\n{line}"
                                    )
        
        # Run test function if provided
        if test_func:
            test_func(module)
            print(f"✅ Component test passed")
        
        print(f"✅ {name} - NO FALLBACKS DETECTED")
        return True
        
    except Exception as e:
        print(f"❌ {name} FAILED:")
        print(f"   {str(e)}")
        traceback.print_exc()
        return False

def test_rl_agent(module):
    """Test that we're using proper RL, not bandits"""
    from training_orchestrator.rl_agent_proper import ProperRLAgent, JourneyState
    
    # Create a test state
    state = JourneyState(
        stage=1,
        touchpoints_seen=3,
        days_since_first_touch=2.0,
        ad_fatigue_level=0.2,
        segment='crisis_parent',
        device='mobile',
        hour_of_day=14,
        day_of_week=2,
        previous_clicks=1,
        previous_impressions=5,
        estimated_ltv=150.0
    )
    
    # Create agent
    agent = ProperRLAgent()
    
    # Test bid selection (should use Q-learning)
    action, bid = agent.get_bid_action(state)
    assert 0 <= action < 10, "Invalid action"
    assert 0.5 <= bid <= 6.0, f"Invalid bid amount: {bid}"
    
    # Test creative selection (should use PPO)
    creative = agent.get_creative_action(state)
    assert 0 <= creative < 5, "Invalid creative"
    
    print(f"   RL Agent test: Bid=${bid:.2f}, Creative={creative}")

def test_recsim(module):
    """Test RecSim is actually imported"""
    import recsim_ng.core.value as value
    assert value is not None, "RecSim not properly imported"

def test_auctiongym(module):
    """Test AuctionGym is actually imported"""
    from auction_gym_integration import Auction, Agent
    assert Auction is not None, "AuctionGym not properly imported"

def main():
    """Verify all components with NO fallbacks"""
    
    print("\n" + "="*60)
    print("STRICT VERIFICATION - NO FALLBACKS ALLOWED")
    print("="*60)
    
    components = [
        ("RL Agent (NOT Bandits!)", "training_orchestrator.rl_agent_proper", test_rl_agent),
        ("RecSim User Model", "recsim_user_model", test_recsim),
        ("AuctionGym Integration", "auction_gym_integration", test_auctiongym),
        ("Conversion Lag Model", "conversion_lag_model", None),
        ("Attribution Models", "attribution_models", None),
        ("Criteo Response Model", "criteo_response_model", None),
        ("Journey Database", "user_journey_database", None),
        ("Monte Carlo Simulator", "monte_carlo_simulator", None),
        ("Competitive Intelligence", "competitive_intelligence", None),
        ("Safety System", "safety_system", None),
        ("Temporal Effects", "temporal_effects", None),
        ("Budget Pacer", "budget_pacer", None),
        ("Identity Resolver", "identity_resolver", None),
        ("Importance Sampler", "importance_sampler", None),
        ("Creative Selector", "creative_selector", None),
        ("Model Versioning", "model_versioning", None),
        ("Journey Timeout", "training_orchestrator.journey_timeout", None),
        ("Delayed Rewards", "training_orchestrator.delayed_reward_system", None),
        ("Master Integration", "gaelp_master_integration", None)
    ]
    
    results = []
    for name, module, test in components:
        result = verify_component(name, module, test)
        results.append((name, result))
    
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print(f"Total: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n❌ VERIFICATION FAILED - FALLBACKS DETECTED!")
        print("FIX ALL COMPONENTS TO USE PRIMARY IMPLEMENTATIONS")
        sys.exit(1)
    else:
        print("\n✅ ALL COMPONENTS VERIFIED - NO FALLBACKS!")
        sys.exit(0)

if __name__ == "__main__":
    main()