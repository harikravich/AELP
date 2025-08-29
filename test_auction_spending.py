#!/usr/bin/env python3
"""Direct test of auction spending"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from enhanced_simulator_fixed import FixedGAELPEnvironment

print("Testing FixedEnvironment auction spending...")
print("=" * 50)

env = FixedGAELPEnvironment()
print(f"Initial state:")
print(f"  max_budget: ${env.max_budget}")
print(f"  max_steps: {env.max_steps}")
print(f"  budget_spent: ${env.budget_spent}")

# Run a few steps with explicit bids
for i in range(5):
    action = {
        'channel': 'google',
        'bid': 3.0,  # $3 bid
        'audience_segment': 'concerned_parents'
    }
    
    state, reward, done, info = env.step(action)
    
    print(f"\nStep {i+1}:")
    print(f"  Action bid: ${action['bid']}")
    print(f"  Auction won: {info.get('auction', {}).get('won', False)}")
    print(f"  Auction price: ${info.get('auction', {}).get('price', 0):.2f}")
    print(f"  Budget spent: ${env.budget_spent:.2f}")
    print(f"  Done: {done}")
    
    if done:
        print(f"  Episode ended: budget=${env.budget_spent:.2f}, steps={env.current_step}")
        break

print("\n" + "=" * 50)
print("Summary:")
print(f"  Total budget spent: ${env.budget_spent:.2f}")
print(f"  Total auctions: {env.metrics.get('auction_wins', 0) + env.metrics.get('auction_losses', 0)}")
print(f"  Auctions won: {env.metrics.get('auction_wins', 0)}")
print(f"  Win rate: {env.metrics.get('auction_wins', 0) / max(1, env.metrics.get('auction_wins', 0) + env.metrics.get('auction_losses', 0)):.2%}")