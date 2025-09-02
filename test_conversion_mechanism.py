#!/usr/bin/env python3
"""Test the actual conversion mechanism to verify it's working"""
import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
from gaelp_parameter_manager import ParameterManager
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Initialize environment
pm = ParameterManager()
env = ProductionFortifiedEnvironment(
    parameter_manager=pm,
    use_real_ga4_data=False,
    is_parallel=False
)

print("Testing conversion mechanism...")
print("="*70)

# Reset environment
obs, info = env.reset()
print(f"Initial state: Stage={env.current_user_state.stage}, Segment={env.current_user_state.segment_index}")

# Run 100 steps and count conversions
conversions = 0
for step in range(100):
    # Take random action
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check for conversion
    if 'metrics' in info and info['metrics'].get('total_conversions', 0) > 0:
        conversions += info['metrics']['total_conversions']
        print(f"Step {step}: CONVERSION! Total={conversions}, Reward={reward:.2f}")
    
    if step % 20 == 0:
        cvr = env._get_conversion_probability(env.current_user_state)
        print(f"Step {step}: CVR={cvr:.3f}, Stage={env.current_user_state.stage}, "
              f"Touchpoints={env.current_user_state.touchpoints_seen}, "
              f"Total Conversions={env.metrics['total_conversions']}")

print("="*70)
print(f"Final results after 100 steps:")
print(f"  Total conversions: {env.metrics['total_conversions']}")
print(f"  Total revenue: ${env.metrics['total_revenue']:.2f}")
print(f"  Total spend: ${env.metrics['budget_spent']:.2f}")
print(f"  Auction wins: {env.metrics['auction_wins']}")
print(f"  Auction losses: {env.metrics['auction_losses']}")