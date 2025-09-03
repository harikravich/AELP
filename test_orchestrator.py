#!/usr/bin/env python3
"""
Test script for GAELP Production Orchestrator
Tests basic functionality without running full system
"""

import sys
import os

print("=" * 80)
print("GAELP ORCHESTRATOR TEST")
print("=" * 80)
print()

# Test imports
print("Testing imports...")
successful_imports = []
failed_imports = []

# Core components that MUST work
core_imports = [
    ('fortified_rl_agent_no_hardcoding', 'ProductionFortifiedRLAgent'),
    ('fortified_environment_no_hardcoding', 'ProductionFortifiedEnvironment'),
    ('discovery_engine', 'GA4DiscoveryEngine'),
    ('emergency_controls', 'EmergencyController'),
]

for module_name, class_name in core_imports:
    try:
        module = __import__(module_name)
        if hasattr(module, class_name):
            successful_imports.append(f"✅ {module_name}.{class_name}")
        else:
            failed_imports.append(f"❌ {module_name}.{class_name} - class not found")
    except Exception as e:
        failed_imports.append(f"❌ {module_name} - {str(e)[:50]}")

print("\nCore Imports:")
for msg in successful_imports:
    print(f"  {msg}")
for msg in failed_imports:
    print(f"  {msg}")

# Test optional components
print("\nOptional Components:")
optional_components = [
    'attribution_system',
    'segment_discovery',
    'production_checkpoint_manager',
    'shadow_mode_manager',
    'bid_explainability_system',
    'statistical_ab_testing_framework',
    'production_online_learner',
    'google_ads_production_manager',
]

available = []
missing = []

for component in optional_components:
    if os.path.exists(f"{component}.py"):
        available.append(f"✅ {component}.py exists")
    else:
        missing.append(f"⚠️ {component}.py not found")

for msg in available:
    print(f"  {msg}")
for msg in missing:
    print(f"  {msg}")

# Test basic initialization
print("\nTesting Basic Initialization:")
try:
    from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent
    from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
    
    # Try to create environment
    env = ProductionFortifiedEnvironment()
    print("  ✅ Environment created")
    
    # Try to create agent
    agent = ProductionFortifiedRLAgent()
    print("  ✅ Agent created")
    
    # Test basic interaction
    state = env.reset()
    print(f"  ✅ Environment reset - state shape: {state.shape if hasattr(state, 'shape') else len(state)}")
    
    action = agent.act(state)
    print(f"  ✅ Agent action generated: {action}")
    
except Exception as e:
    print(f"  ❌ Initialization failed: {e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if len(failed_imports) == 0 and len(successful_imports) >= 3:
    print("✅ Core system is FUNCTIONAL")
    print("   The orchestrator should be able to run with basic functionality")
else:
    print("❌ Core system has ISSUES")
    print("   Fix the import errors before running orchestrator")

print("\nTo run the orchestrator:")
print("  python3 gaelp_production_orchestrator.py")
print("\nTo monitor:")
print("  python3 gaelp_production_monitor.py --mode terminal")