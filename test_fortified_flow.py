#!/usr/bin/env python3
"""
Test the complete fortified action flow to ensure all components work together
"""

import sys
import traceback
from datetime import datetime

def test_imports():
    """Test all critical imports"""
    print("=" * 70)
    print("Testing imports...")
    print("=" * 70)
    
    errors = []
    
    try:
        import gymnasium as gym
        print("✓ gymnasium imported successfully")
    except ImportError as e:
        errors.append(f"✗ gymnasium import failed: {e}")
    
    try:
        from fortified_rl_agent import FortifiedRLAgent, EnrichedJourneyState
        print("✓ FortifiedRLAgent imported successfully")
    except ImportError as e:
        errors.append(f"✗ FortifiedRLAgent import failed: {e}")
    
    try:
        from fortified_environment import FortifiedEnvironment
        print("✓ FortifiedEnvironment imported successfully")
    except ImportError as e:
        errors.append(f"✗ FortifiedEnvironment import failed: {e}")
    
    try:
        from training_orchestrator.delayed_conversion_system import DelayedConversionSystem
        print("✓ DelayedConversionSystem imported successfully")
    except ImportError as e:
        errors.append(f"✗ DelayedConversionSystem import failed: {e}")
    
    return errors

def test_environment_creation():
    """Test environment instantiation"""
    print("\n" + "=" * 70)
    print("Testing environment creation...")
    print("=" * 70)
    
    errors = []
    
    try:
        from fortified_environment import FortifiedEnvironment
        env = FortifiedEnvironment()
        print("✓ FortifiedEnvironment created successfully")
        
        # Test reset
        state = env.reset()
        print(f"✓ Environment reset successful, state shape: {state.shape if hasattr(state, 'shape') else 'dict'}")
        
        # Test current_state attribute
        if hasattr(env, 'current_state'):
            print(f"✓ Environment has current_state attribute")
        else:
            errors.append("✗ Environment missing current_state attribute")
            
    except Exception as e:
        errors.append(f"✗ Environment creation failed: {e}")
        traceback.print_exc()
    
    return errors

def test_agent_creation():
    """Test agent instantiation"""
    print("\n" + "=" * 70)
    print("Testing agent creation...")
    print("=" * 70)
    
    errors = []
    
    try:
        from fortified_rl_agent import FortifiedRLAgent
        agent = FortifiedRLAgent()
        print("✓ FortifiedRLAgent created successfully")
        
        # Check agent attributes
        print(f"  - State dimensions: {agent.state_dim}")
        print(f"  - Bid actions: {agent.bid_actions}")
        print(f"  - Creative actions: {agent.creative_actions}")
        print(f"  - Channel actions: {agent.channel_actions}")
        
    except Exception as e:
        errors.append(f"✗ Agent creation failed: {e}")
        traceback.print_exc()
    
    return errors

def test_action_flow():
    """Test the complete action flow from agent to environment"""
    print("\n" + "=" * 70)
    print("Testing complete action flow...")
    print("=" * 70)
    
    errors = []
    
    try:
        from fortified_rl_agent import FortifiedRLAgent, EnrichedJourneyState
        from fortified_environment import FortifiedEnvironment
        
        # Create environment and agent
        env = FortifiedEnvironment()
        agent = FortifiedRLAgent()
        
        # Reset environment
        state = env.reset()
        state_obj = env.current_state
        print("✓ Environment reset successful")
        
        # Get action from agent
        action = agent.select_action(state_obj, explore=True)
        print(f"✓ Agent selected action: {action}")
        
        # Verify action structure
        required_keys = ['bid_amount', 'creative_id', 'channel']
        for key in required_keys:
            if key in action:
                print(f"  ✓ Action has '{key}': {action[key]}")
            else:
                errors.append(f"  ✗ Action missing '{key}'")
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  - Reward: {reward}")
        print(f"  - Done: {done}")
        print(f"  - Info keys: {list(info.keys())}")
        
        # Check if environment has updated current_state
        if hasattr(env, 'current_state'):
            next_state_obj = env.current_state
            print(f"✓ Environment updated current_state")
        else:
            errors.append("✗ Environment missing current_state after step")
        
    except KeyError as e:
        errors.append(f"✗ KeyError in action flow: {e}")
        traceback.print_exc()
    except Exception as e:
        errors.append(f"✗ Action flow failed: {e}")
        traceback.print_exc()
    
    return errors

def test_delayed_conversions():
    """Test delayed conversion system integration"""
    print("\n" + "=" * 70)
    print("Testing delayed conversion system...")
    print("=" * 70)
    
    errors = []
    
    try:
        from training_orchestrator.delayed_conversion_system import DelayedConversionSystem
        from user_journey_database import UserJourneyDatabase
        
        # Create components
        journey_db = UserJourneyDatabase()
        conversion_system = DelayedConversionSystem(journey_db)
        
        print("✓ DelayedConversionSystem created successfully")
        
        # Test get_due_conversions method
        due_conversions = conversion_system.get_due_conversions()
        print(f"✓ get_due_conversions() works: {len(due_conversions)} due conversions")
        
    except AttributeError as e:
        if "get_due_conversions" in str(e):
            errors.append(f"✗ DelayedConversionSystem missing get_due_conversions method: {e}")
        else:
            errors.append(f"✗ AttributeError in conversion system: {e}")
        traceback.print_exc()
    except Exception as e:
        errors.append(f"✗ Delayed conversion system failed: {e}")
        traceback.print_exc()
    
    return errors

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" FORTIFIED SYSTEM FLOW TEST ".center(70))
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(70))
    print("=" * 70)
    
    all_errors = []
    
    # Run tests
    all_errors.extend(test_imports())
    all_errors.extend(test_environment_creation())
    all_errors.extend(test_agent_creation())
    all_errors.extend(test_action_flow())
    all_errors.extend(test_delayed_conversions())
    
    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY ".center(70))
    print("=" * 70)
    
    if all_errors:
        print(f"\n❌ {len(all_errors)} errors found:\n")
        for error in all_errors:
            print(f"  {error}")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! The fortified system flow is working correctly.")
        print("\nYou can now run the fortified training with:")
        print("  python3 run_training.py")
        print("  (Select option 1 for fortified training)")
        sys.exit(0)

if __name__ == "__main__":
    main()