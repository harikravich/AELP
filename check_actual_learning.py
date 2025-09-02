#!/usr/bin/env python3
"""
Check if the actual RL agent inside the hybrid wrapper is learning
"""

import torch
import numpy as np
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("CHECKING ACTUAL RL AGENT LEARNING")
    logger.info("=" * 80)
    
    # Import and create orchestrator
    from gaelp_master_integration import MasterOrchestrator, GAELPConfig
    
    config = GAELPConfig()
    orchestrator = MasterOrchestrator(config)
    
    # Get the wrapped agent
    hybrid_agent = orchestrator.rl_agent
    logger.info(f"Hybrid agent type: {type(hybrid_agent).__name__}")
    
    # Get the actual RL agent inside
    base_agent = hybrid_agent.rl_agent
    logger.info(f"Base RL agent type: {type(base_agent).__name__}")
    
    # Check the base agent's components
    if hasattr(base_agent, 'q_network'):
        logger.info("✓ Base agent has Q-network")
        
        # Get initial weights
        initial_weights = {}
        for name, param in base_agent.q_network.named_parameters():
            initial_weights[name] = param.data.clone()
            logger.info(f"  {name}: shape {param.shape}")
        
        # Check replay buffer
        if hasattr(base_agent, 'replay_buffer'):
            logger.info(f"✓ Replay buffer exists, size: {len(base_agent.replay_buffer)}")
        else:
            logger.error("✗ No replay buffer found!")
            
    elif hasattr(base_agent, 'policy_net'):
        logger.info("✓ Base agent has policy network")
        initial_weights = {}
        for name, param in base_agent.policy_net.named_parameters():
            initial_weights[name] = param.data.clone()
            logger.info(f"  {name}: shape {param.shape}")
    else:
        logger.error("✗ Base agent has no recognizable networks!")
        return
    
    # Run some steps to generate experiences
    logger.info("\n" + "=" * 40)
    logger.info("GENERATING EXPERIENCES")
    logger.info("=" * 40)
    
    env = orchestrator.fixed_environment
    
    for step in range(50):
        # Run a step
        result = orchestrator.step_fixed_environment()
        
        if step % 10 == 0:
            logger.info(f"Step {step}: Reward={result.get('reward', 0):.4f}")
            
            # Check buffer growth
            if hasattr(base_agent, 'replay_buffer'):
                logger.info(f"  Buffer size: {len(base_agent.replay_buffer)}")
    
    # Now try to train
    logger.info("\n" + "=" * 40)
    logger.info("CHECKING TRAINING")
    logger.info("=" * 40)
    
    # Check if training happens
    if hasattr(base_agent, 'train'):
        logger.info("Calling base_agent.train()...")
        metrics = base_agent.train()
        logger.info(f"Training metrics: {metrics}")
    elif hasattr(base_agent, 'train_dqn'):
        logger.info("Calling base_agent.train_dqn()...")
        base_agent.train_dqn(batch_size=32)
    elif hasattr(base_agent, 'update'):
        logger.info("Calling base_agent.update()...")
        base_agent.update()
    else:
        logger.error("✗ No training method found!")
        return
    
    # Check if weights changed
    logger.info("\n" + "=" * 40)
    logger.info("CHECKING WEIGHT UPDATES")
    logger.info("=" * 40)
    
    weights_changed = False
    max_change = 0
    
    if hasattr(base_agent, 'q_network'):
        for name, param in base_agent.q_network.named_parameters():
            if name in initial_weights:
                change = torch.abs(param.data - initial_weights[name]).max().item()
                max_change = max(max_change, change)
                if change > 1e-6:
                    weights_changed = True
                    logger.info(f"  {name}: changed by {change:.9f}")
    elif hasattr(base_agent, 'policy_net'):
        for name, param in base_agent.policy_net.named_parameters():
            if name in initial_weights:
                change = torch.abs(param.data - initial_weights[name]).max().item()
                max_change = max(max_change, change)
                if change > 1e-6:
                    weights_changed = True
                    logger.info(f"  {name}: changed by {change:.9f}")
    
    logger.info("\n" + "=" * 40)
    logger.info("RESULTS")
    logger.info("=" * 40)
    
    if weights_changed:
        logger.info(f"✅ WEIGHTS ARE UPDATING! Max change: {max_change:.9f}")
    else:
        logger.error(f"❌ WEIGHTS NOT CHANGING! Max change: {max_change:.12f}")
        
        # Debug why
        if hasattr(base_agent, 'replay_buffer'):
            logger.info(f"Debug: Buffer size = {len(base_agent.replay_buffer)}")
            if len(base_agent.replay_buffer) > 0:
                # Check a sample
                sample = base_agent.replay_buffer.buffer[0] if hasattr(base_agent.replay_buffer, 'buffer') else None
                if sample:
                    logger.info(f"Debug: Sample reward = {getattr(sample, 'reward', 'N/A')}")
        
        if hasattr(base_agent, 'optimizer'):
            logger.info(f"Debug: Optimizer LR = {base_agent.optimizer.param_groups[0]['lr']}")
            
        if hasattr(base_agent, 'loss_history'):
            logger.info(f"Debug: Recent losses = {base_agent.loss_history[-5:] if base_agent.loss_history else 'None'}")

if __name__ == "__main__":
    main()