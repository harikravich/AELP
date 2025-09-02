#!/usr/bin/env python3
"""Test if replay buffer is getting experiences"""

from gaelp_master_integration import MasterOrchestrator, GAELPConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = GAELPConfig()
orchestrator = MasterOrchestrator(config)

# Get the agents
hybrid_agent = orchestrator.rl_agent
base_agent = hybrid_agent.rl_agent

# Check initial buffer
if hasattr(base_agent, 'memory'):
    logger.info(f"Initial buffer size: {len(base_agent.memory)}")
elif hasattr(base_agent, 'replay_buffer'):
    logger.info(f"Initial buffer size: {len(base_agent.replay_buffer)}")

# Generate some experiences
for i in range(10):
    result = orchestrator.step_fixed_environment()
    
# Check buffer after steps
if hasattr(base_agent, 'memory'):
    logger.info(f"Buffer after 10 steps: {len(base_agent.memory)}")
    logger.info(f"Min buffer size for training: {base_agent.config.min_buffer_size}")
elif hasattr(base_agent, 'replay_buffer'):
    logger.info(f"Buffer after 10 steps: {len(base_agent.replay_buffer)}")
