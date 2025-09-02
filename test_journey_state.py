#!/usr/bin/env python3
"""Test if journey_state is created and passed to store_experience"""

from gaelp_master_integration import MasterOrchestrator, GAELPConfig
import logging

# Set logging to DEBUG to see everything
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress other loggers
logging.getLogger('discovery_engine').setLevel(logging.WARNING)
logging.getLogger('parameter_manager').setLevel(logging.WARNING)

config = GAELPConfig()
orchestrator = MasterOrchestrator(config)

# Add logging to check journey_state creation
import gaelp_master_integration
original_step = gaelp_master_integration.MasterOrchestrator.step_fixed_environment

def patched_step(self):
    """Patched version that logs journey_state status"""
    result = {}
    
    # Check if journey_state gets created
    if hasattr(self, 'rl_agent') and self.rl_agent is not None:
        print("✓ RL agent exists")
        # The actual creation happens inside step_fixed_environment
    
    # Call original
    result = original_step(self)
    
    # Check if store_experience was called by monitoring buffer size
    if hasattr(self.rl_agent, 'rl_agent'):
        base_agent = self.rl_agent.rl_agent
        if hasattr(base_agent, 'memory'):
            print(f"Buffer size after step: {len(base_agent.memory)}")
    
    return result

# Patch the method
gaelp_master_integration.MasterOrchestrator.step_fixed_environment = patched_step

# Run a few steps
for i in range(3):
    print(f"\n=== Step {i+1} ===")
    result = orchestrator.step_fixed_environment()
    
# Check final buffer size
base_agent = orchestrator.rl_agent.rl_agent
print(f"\nFinal buffer size: {len(base_agent.memory)}")
print(f"Minimum buffer size for training: {base_agent.config.min_buffer_size}")
