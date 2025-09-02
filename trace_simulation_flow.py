#!/usr/bin/env python3
"""
Trace the complete flow from simulation to agent training to find where learning breaks
"""

import sys
import torch
import numpy as np
import logging
from datetime import datetime
import json

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def trace_complete_flow():
    """Trace through the entire GAELP simulation and training pipeline"""
    
    logger.info("=" * 80)
    logger.info("TRACING COMPLETE GAELP SIMULATION FLOW")
    logger.info("=" * 80)
    
    # Import the master orchestrator
    from gaelp_master_integration import MasterOrchestrator, GAELPConfig
    from training_orchestrator.rl_agent_proper import ProperRLAgent
    
    # Create config
    config = GAELPConfig()
    logger.info(f"Config created with settings: {vars(config)}")
    
    # Create orchestrator
    orchestrator = MasterOrchestrator(config)
    logger.info(f"Orchestrator created with {len(orchestrator._get_component_list())} components")
    
    # Check RL agent
    if hasattr(orchestrator, 'rl_agent'):
        agent = orchestrator.rl_agent
        logger.info(f"RL Agent type: {type(agent).__name__}")
        
        # Check replay buffer
        if hasattr(agent, 'replay_buffer'):
            logger.info(f"Replay buffer size: {len(agent.replay_buffer)}")
        
        # Get initial weights checksum
        initial_weights_sum = 0
        for param in agent.q_network.parameters():
            initial_weights_sum += param.data.sum().item()
        logger.info(f"Initial weights checksum: {initial_weights_sum:.6f}")
        
        # Track gradients
        grad_norms = []
        
        def hook_fn(grad):
            grad_norms.append(grad.norm().item())
            return grad
        
        # Register hooks
        for param in agent.q_network.parameters():
            param.register_hook(hook_fn)
    else:
        logger.error("No RL agent found in orchestrator!")
        return False
    
    # Run simulation steps and trace everything
    logger.info("\n" + "=" * 40)
    logger.info("RUNNING SIMULATION STEPS")
    logger.info("=" * 40)
    
    for step in range(10):
        logger.info(f"\n--- STEP {step + 1} ---")
        
        # Get action from agent
        if hasattr(orchestrator, 'fixed_environment'):
            env = orchestrator.fixed_environment
            
            # Check current state
            logger.info(f"Environment step count: {env.step_count}")
            logger.info(f"Environment metrics: {json.dumps(env.metrics, indent=2)}")
            
            # Get observation
            obs = env._get_observation()
            logger.info(f"Observation shape: {len(obs)}")
            logger.info(f"Observation sample: {obs[:5]}...")
            
            # Get action from agent
            action = agent.get_bid_action(None, explore=True)  # Simplified call
            logger.info(f"Agent action: {action}")
            
            # Step environment
            obs, reward, done, info = env.step({'bid': 2.5, 'quality_score': 7.5})
            logger.info(f"Step result - Reward: {reward:.4f}, Done: {done}")
            logger.info(f"Info: {json.dumps(info, default=str, indent=2)}")
            
            # Check if experience was stored
            if hasattr(agent, 'replay_buffer'):
                buffer_size = len(agent.replay_buffer)
                logger.info(f"Replay buffer size after step: {buffer_size}")
                
                # Try to train if buffer has enough samples
                if buffer_size >= 32:
                    logger.info("Training agent...")
                    
                    # Clear gradient tracking
                    grad_norms.clear()
                    
                    # Train
                    agent.train_dqn(batch_size=32)
                    
                    # Check if gradients flowed
                    if grad_norms:
                        logger.info(f"Gradient norms: {grad_norms[:5]}...")
                        logger.info(f"Max gradient norm: {max(grad_norms):.6f}")
                    else:
                        logger.warning("NO GRADIENTS COMPUTED!")
                    
                    # Check if weights changed
                    new_weights_sum = 0
                    for param in agent.q_network.parameters():
                        new_weights_sum += param.data.sum().item()
                    
                    weight_change = abs(new_weights_sum - initial_weights_sum)
                    logger.info(f"Weight change: {weight_change:.9f}")
                    
                    if weight_change > 1e-6:
                        logger.info("✓ WEIGHTS ARE UPDATING!")
                    else:
                        logger.error("✗ WEIGHTS NOT CHANGING!")
        else:
            # Try step_fixed_environment method
            result = orchestrator.step_fixed_environment()
            logger.info(f"Step result: {json.dumps(result, default=str, indent=2)}")
    
    # Analyze learning
    logger.info("\n" + "=" * 40)
    logger.info("LEARNING ANALYSIS")
    logger.info("=" * 40)
    
    # Check final replay buffer
    if hasattr(agent, 'replay_buffer'):
        logger.info(f"Final replay buffer size: {len(agent.replay_buffer)}")
        
        # Sample and inspect experiences
        if len(agent.replay_buffer) > 0:
            sample = agent.replay_buffer.buffer[0]
            logger.info(f"Sample experience:")
            logger.info(f"  State shape: {len(sample.state) if hasattr(sample, 'state') else 'N/A'}")
            logger.info(f"  Action: {sample.action if hasattr(sample, 'action') else 'N/A'}")
            logger.info(f"  Reward: {sample.reward if hasattr(sample, 'reward') else 'N/A'}")
    
    # Check loss history
    if hasattr(agent, 'loss_history'):
        logger.info(f"Loss history: {agent.loss_history[-10:]}")
    
    return True

def check_reward_generation():
    """Specifically check how rewards are being generated"""
    
    logger.info("\n" + "=" * 80)
    logger.info("CHECKING REWARD GENERATION")
    logger.info("=" * 80)
    
    from enhanced_simulator_fixed import FixedGAELPEnvironment
    
    env = FixedGAELPEnvironment()
    
    # Track rewards over multiple steps
    rewards = []
    revenues = []
    costs = []
    
    for i in range(100):
        action = {
            'bid': np.random.uniform(1.0, 5.0),
            'quality_score': np.random.uniform(5.0, 10.0),
            'channel': 'google',
            'audience_segment': 'concerned_parents'
        }
        
        obs, reward, done, info = env.step(action)
        
        rewards.append(reward)
        revenues.append(info.get('revenue', 0))
        costs.append(info.get('cost', 0))
        
        if i % 20 == 0:
            logger.info(f"Step {i}: Reward={reward:.4f}, Revenue=${info.get('revenue', 0):.2f}, Cost=${info.get('cost', 0):.2f}")
    
    # Analyze reward distribution
    logger.info("\nReward Analysis:")
    logger.info(f"  Mean reward: {np.mean(rewards):.4f}")
    logger.info(f"  Std reward: {np.std(rewards):.4f}")
    logger.info(f"  Min reward: {np.min(rewards):.4f}")
    logger.info(f"  Max reward: {np.max(rewards):.4f}")
    logger.info(f"  Zero rewards: {sum(1 for r in rewards if r == 0)}/{len(rewards)}")
    
    logger.info("\nRevenue Analysis:")
    logger.info(f"  Mean revenue: ${np.mean(revenues):.2f}")
    logger.info(f"  Max revenue: ${np.max(revenues):.2f}")
    logger.info(f"  Zero revenues: {sum(1 for r in revenues if r == 0)}/{len(revenues)}")
    
    # Check if rewards are actually varied (not just random)
    unique_rewards = len(set(rewards))
    logger.info(f"\nUnique reward values: {unique_rewards}/{len(rewards)}")
    
    if unique_rewards < 10:
        logger.error("✗ Rewards seem to be following a fixed pattern!")
    else:
        logger.info("✓ Rewards show variation")
    
    return True

if __name__ == "__main__":
    try:
        trace_complete_flow()
        check_reward_generation()
    except Exception as e:
        logger.error(f"Trace failed: {e}")
        import traceback
        traceback.print_exc()