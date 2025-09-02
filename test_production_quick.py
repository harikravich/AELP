#!/usr/bin/env python3
"""Quick test of production training"""
import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
from gaelp_parameter_manager import ParameterManager

print("Testing production environment reset...")
pm = ParameterManager()
env = ProductionFortifiedEnvironment(parameter_manager=pm, use_real_ga4_data=False)
obs, info = env.reset()
print(f"✅ Reset successful! Observation shape: {obs.shape}")
print(f"✅ User ID: {info.get('user_id')}")

# Test step
action = {'bid': 5.0, 'creative': 0, 'channel': 0}
obs, reward, terminated, truncated, info = env.step(action)
print(f"✅ Step successful! Reward: {reward:.2f}")
print("✅ All tests passed!")