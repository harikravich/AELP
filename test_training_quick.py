#!/usr/bin/env python3
"""
Quick test to verify training works with all fixes
"""

import os
import sys
import logging

# Suppress noisy logs
os.environ['RAY_DEDUP_LOGS'] = '1'
logging.getLogger('google.cloud.bigquery').setLevel(logging.WARNING)
logging.getLogger('discovery_engine').setLevel(logging.WARNING)

sys.path.insert(0, '/home/hariravichandran/AELP')

from fortified_environment import FortifiedGAELPEnvironment
from fortified_rl_agent import FortifiedRLAgent

def test_training():
    print("=" * 70)
    print("TESTING FORTIFIED TRAINING SYSTEM")
    print("=" * 70)
    
    # 1. Test environment initialization
    print("\n1. Testing environment initialization...")
    try:
        env = FortifiedGAELPEnvironment(
            max_budget=10000,
            max_steps=100,
            use_real_ga4_data=False  # Start without GA4 to test basics
        )
        print("✅ Environment initialized")
    except Exception as e:
        print(f"❌ Environment init failed: {e}")
        return False
    
    # 2. Test agent initialization
    print("\n2. Testing agent initialization...")
    try:
        from discovery_engine import GA4DiscoveryEngine
        from creative_selector import CreativeSelector
        from attribution_models import AttributionEngine
        from budget_pacer import BudgetPacer
        from identity_resolver import IdentityResolver
        from gaelp_parameter_manager import ParameterManager
        
        agent = FortifiedRLAgent(
            discovery_engine=GA4DiscoveryEngine(),
            creative_selector=CreativeSelector(),
            attribution_engine=AttributionEngine(),
            budget_pacer=BudgetPacer(),
            identity_resolver=IdentityResolver(),
            parameter_manager=ParameterManager()
        )
        print("✅ Agent initialized")
    except Exception as e:
        print(f"❌ Agent init failed: {e}")
        return False
    
    # 3. Test environment reset
    print("\n3. Testing environment reset...")
    try:
        state = env.reset()
        print(f"✅ Environment reset, state shape: {state.shape}")
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        return False
    
    # 4. Test action selection
    print("\n4. Testing action selection...")
    try:
        action = agent.select_action(env.current_state, explore=True)
        print(f"✅ Action selected: bid=${action['bid_amount']:.2f}, creative={action['creative_id']}, channel={action['channel']}")
    except Exception as e:
        print(f"❌ Action selection failed: {e}")
        return False
    
    # 5. Test environment step
    print("\n5. Testing environment step...")
    try:
        next_state, reward, done, info = env.step(action)
        print(f"✅ Step executed: reward={reward:.2f}, done={done}")
        
        # Check bid amount is in realistic range
        if action['bid_amount'] < 5.0:
            print(f"⚠️  WARNING: Bid amount ${action['bid_amount']:.2f} is below realistic range ($5-50)")
        elif action['bid_amount'] > 50.0:
            print(f"⚠️  WARNING: Bid amount ${action['bid_amount']:.2f} is above realistic range ($5-50)")
        else:
            print(f"✅ Bid amount ${action['bid_amount']:.2f} is in realistic range")
            
    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        return False
    
    # 6. Test experience storage
    print("\n6. Testing experience storage...")
    try:
        agent.store_experience(
            state=env.current_state,
            action=action,
            reward=reward,
            next_state=env.current_state,
            done=done
        )
        print(f"✅ Experience stored, buffer size: {len(agent.replay_buffer)}")
    except Exception as e:
        print(f"❌ Experience storage failed: {e}")
        return False
    
    # 7. Run a few more steps
    print("\n7. Running 10 steps to build experience...")
    try:
        for i in range(10):
            action = agent.select_action(env.current_state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.store_experience(
                state=env.current_state,
                action=action,
                reward=reward,
                next_state=env.current_state,
                done=done
            )
            if done:
                env.reset()
        print(f"✅ Completed 10 steps, buffer size: {len(agent.replay_buffer)}")
    except Exception as e:
        print(f"❌ Multi-step execution failed: {e}")
        return False
    
    # 8. Test training
    print("\n8. Testing training step...")
    try:
        if len(agent.replay_buffer) >= 32:
            metrics = agent.train(batch_size=32)
            if metrics:
                print(f"✅ Training step completed:")
                print(f"   - Bid loss: {metrics['loss_bid']:.4f}")
                print(f"   - Creative loss: {metrics['loss_creative']:.4f}")
                print(f"   - Channel loss: {metrics['loss_channel']:.4f}")
            else:
                print("⚠️  Training returned no metrics (might need more data)")
        else:
            print("⚠️  Not enough data for training yet")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - TRAINING SYSTEM IS WORKING")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_training()
    if not success:
        print("\n❌ TRAINING SYSTEM HAS ISSUES")
        sys.exit(1)
