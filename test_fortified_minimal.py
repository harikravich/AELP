#!/usr/bin/env python3
"""
Minimal test of fortified system to verify it works
"""

import sys
import traceback

def test_minimal_flow():
    """Test the absolute minimum to verify system works"""
    print("Testing minimal fortified flow...")
    
    try:
        # Import everything
        from fortified_rl_agent import FortifiedRLAgent, EnrichedJourneyState
        from fortified_environment import FortifiedGAELPEnvironment
        from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
        from creative_selector import CreativeSelector
        from attribution_models import AttributionEngine
        from budget_pacer import BudgetPacer
        from identity_resolver import IdentityResolver
        from gaelp_parameter_manager import ParameterManager
        
        print("✓ All imports successful")
        
        # Create components
        discovery = DiscoveryEngine()
        creative_selector = CreativeSelector()
        attribution = AttributionEngine()
        budget_pacer = BudgetPacer()
        identity_resolver = IdentityResolver()
        parameter_manager = ParameterManager()
        
        print("✓ All components created")
        
        # Create agent
        agent = FortifiedRLAgent(
            discovery_engine=discovery,
            creative_selector=creative_selector,
            attribution_engine=attribution,
            budget_pacer=budget_pacer,
            identity_resolver=identity_resolver,
            parameter_manager=parameter_manager
        )
        print("✓ Agent created")
        
        # Create environment
        env = FortifiedGAELPEnvironment()
        print("✓ Environment created")
        
        # Reset environment
        state = env.reset()
        state_obj = env.current_state
        print(f"✓ Environment reset: state type = {type(state_obj)}")
        
        # Get action from agent
        action = agent.select_action(state_obj, explore=True)
        print(f"✓ Agent action: {action}")
        
        # Verify action has required keys
        required_keys = ['bid_amount', 'creative_id', 'channel']
        for key in required_keys:
            if key in action:
                print(f"  ✓ Has {key}: {action[key]}")
            else:
                print(f"  ✗ Missing {key}")
                return False
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  - Reward: {reward}")
        print(f"  - Done: {done}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_flow()
    if success:
        print("\n✅ MINIMAL TEST PASSED - System is working!")
        print("You can now run: python3 capture_fortified_training.py")
    else:
        print("\n❌ MINIMAL TEST FAILED - Fix errors above")
        sys.exit(1)