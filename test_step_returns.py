#!/usr/bin/env python3
"""Test what step_fixed_environment actually returns"""

from gaelp_master_integration import MasterOrchestrator, GAELPConfig
import json

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = GAELPConfig()
master = MasterOrchestrator(config)

print("Testing step_fixed_environment returns...")
print("=" * 50)

for i in range(5):
    result = master.step_fixed_environment()
    
    print(f"\nStep {i+1}:")
    print(f"  done: {result.get('done', 'NOT FOUND')}")
    print(f"  reward: {result.get('reward', 'NOT FOUND')}")
    
    metrics = result.get('metrics', {})
    print(f"  metrics.total_spend: {metrics.get('total_spend', 'NOT FOUND')} (type: {type(metrics.get('total_spend')).__name__})")
    print(f"  metrics.total_auctions: {metrics.get('total_auctions', 'NOT FOUND')}")
    
    step_info = result.get('step_info', {})
    auction = step_info.get('auction', {})
    print(f"  auction.channel: {auction.get('channel', 'NOT FOUND')}")
    print(f"  auction.won: {auction.get('won', 'NOT FOUND')}")
    print(f"  auction.price: {auction.get('price', 'NOT FOUND')}")
    
    # Check fixed environment directly
    if hasattr(master, 'fixed_environment'):
        env = master.fixed_environment
        print(f"  env.budget_spent: ${env.budget_spent:.2f}")
        print(f"  env.max_budget: ${env.max_budget:.2f}")
        print(f"  env.current_step: {env.current_step}/{env.max_steps}")