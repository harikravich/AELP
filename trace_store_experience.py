#!/usr/bin/env python3
"""Trace why store_experience isn't being called"""

from gaelp_master_integration import MasterOrchestrator, GAELPConfig
import logging

# Enable ONLY critical logging to see our custom prints
logging.basicConfig(level=logging.ERROR)

config = GAELPConfig()
orchestrator = MasterOrchestrator(config)

# Patch step_fixed_environment to add detailed tracing
import gaelp_master_integration
original_step = gaelp_master_integration.MasterOrchestrator.step_fixed_environment

def traced_step(self):
    """Traced version that shows why store_experience isn't called"""
    # Initialize journey_state at method level
    journey_state = None
    
    # Check conditions at line 1906
    print(f"Line 1906: hasattr(self, 'rl_agent') = {hasattr(self, 'rl_agent')}")
    print(f"Line 1906: self.rl_agent is not None = {self.rl_agent is not None if hasattr(self, 'rl_agent') else 'N/A'}")
    
    if hasattr(self, 'rl_agent') and self.rl_agent is not None:
        print("✓ Creating journey_state (lines 1907-1973)")
        # Journey state creation happens here
        journey_state = "MOCK_JOURNEY_STATE"  # Simplified for tracing
    
    # Call original step (simplified)
    result = original_step(self)
    
    # Check condition at line 2188
    print(f"\nLine 2188 conditions:")
    print(f"  hasattr(self, 'rl_agent') = {hasattr(self, 'rl_agent')}")
    print(f"  self.rl_agent is not None = {self.rl_agent is not None if hasattr(self, 'rl_agent') else 'N/A'}")
    print(f"  journey_state is not None = {journey_state is not None}")
    print(f"  journey_state value = {journey_state}")
    
    all_conditions = (
        hasattr(self, 'rl_agent') and 
        self.rl_agent is not None and 
        journey_state is not None
    )
    print(f"  ALL conditions met = {all_conditions}")
    
    if not all_conditions:
        print("❌ store_experience NOT called - conditions not met")
    else:
        print("✓ store_experience SHOULD be called")
    
    return result

# Patch the method
gaelp_master_integration.MasterOrchestrator.step_fixed_environment = traced_step

# Run one step
print("=== Running one step ===")
result = orchestrator.step_fixed_environment()

# Check buffer
base_agent = orchestrator.rl_agent.rl_agent
print(f"\n=== Final Check ===")
print(f"Buffer size: {len(base_agent.memory)}")
