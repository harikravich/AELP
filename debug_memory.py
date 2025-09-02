#!/usr/bin/env python3
"""Debug why memory isn't filling"""

from gaelp_master_integration import MasterOrchestrator, GAELPConfig
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

config = GAELPConfig()
orchestrator = MasterOrchestrator(config)

# Get the agents
hybrid_agent = orchestrator.rl_agent
base_agent = hybrid_agent.rl_agent

# Check if store_experience method exists
print(f"HybridAgent has store_experience: {hasattr(hybrid_agent, 'store_experience')}")
print(f"BaseAgent has store_experience: {hasattr(base_agent, 'store_experience')}")
print(f"BaseAgent has memory: {hasattr(base_agent, 'memory')}")

if hasattr(base_agent, 'memory'):
    print(f"Memory type: {type(base_agent.memory)}")
    print(f"Initial memory size: {len(base_agent.memory)}")

# Try to manually add an experience
try:
    state = {'bid': 1.0, 'budget': 100.0, 'hour': 12, 'segment': 0}
    next_state = {'bid': 1.5, 'budget': 99.0, 'hour': 13, 'segment': 0}
    hybrid_agent.store_experience(state, 0, 0.1, next_state, False, {})
    print(f"After manual add: {len(base_agent.memory)}")
except Exception as e:
    print(f"Error adding experience: {e}")
    import traceback
    traceback.print_exc()
