#!/usr/bin/env python3
"""Fill the buffer and trigger training"""

from gaelp_master_integration import MasterOrchestrator, GAELPConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = GAELPConfig()
orchestrator = MasterOrchestrator(config)

# Get the agents
hybrid_agent = orchestrator.rl_agent
base_agent = hybrid_agent.rl_agent

logger.info(f"Starting - Buffer size: {len(base_agent.memory)}")
logger.info(f"Min buffer size for training: {base_agent.config.min_buffer_size}")

# Run enough steps to fill the buffer
steps_needed = base_agent.config.min_buffer_size + 100  # A bit extra
logger.info(f"Running {steps_needed} steps to fill buffer...")

for i in range(steps_needed):
    orchestrator.step_fixed_environment()
    
    if i % 100 == 0:
        logger.info(f"Step {i}: Buffer size = {len(base_agent.memory)}")

logger.info(f"Final buffer size: {len(base_agent.memory)}")

# Now try to train
logger.info("Attempting to train...")
metrics = base_agent.train()
logger.info(f"Training metrics: {metrics}")

# Check if weights changed
logger.info("Checking if weights updated...")
# Store initial weight for comparison
if hasattr(base_agent, 'q_network'):
    initial_weight = base_agent.q_network.shared[0].weight_mu.data[0,0].item()
    
    # Train again
    metrics = base_agent.train()
    
    # Check weight after training
    final_weight = base_agent.q_network.shared[0].weight_mu.data[0,0].item()
    
    if initial_weight != final_weight:
        logger.info(f"✅ WEIGHTS UPDATED! Change: {abs(final_weight - initial_weight):.9f}")
    else:
        logger.error("❌ Weights still not updating")
