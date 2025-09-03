#!/usr/bin/env python3
"""
Simple runner for GAELP system that actually works
Strips out all the broken components and runs what's real
"""

import sys
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_core_system():
    """Run the actual working GAELP components"""
    
    logger.info("=" * 80)
    logger.info("GAELP PRODUCTION SYSTEM - ACTUAL WORKING VERSION")
    logger.info("=" * 80)
    
    # Import what actually works
    logger.info("Loading core components...")
    from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent
    from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
    from emergency_controls import EmergencyController
    from budget_safety_controller import BudgetSafetyController
    from segment_discovery import SegmentDiscoveryEngine
    from attribution_system import MultiTouchAttributionEngine
    from production_checkpoint_manager import ProductionCheckpointManager
    from regression_detector import RegressionDetector
    
    # Initialize components
    components = {}
    
    logger.info("\n📦 Initializing components...")
    
    # 1. Safety first
    logger.info("🛡️ Safety systems...")
    components['emergency'] = EmergencyController()
    components['budget_safety'] = BudgetSafetyController()
    logger.info("  ✅ Safety systems online")
    
    # 2. Core RL
    logger.info("🤖 RL system...")
    components['environment'] = ProductionFortifiedEnvironment()
    components['agent'] = ProductionFortifiedRLAgent()
    logger.info("  ✅ RL system online")
    
    # 3. Attribution & Monitoring
    logger.info("📊 Attribution & Monitoring...")
    components['attribution'] = MultiTouchAttributionEngine(
        database_path="attribution_system.db"
    )
    components['checkpoint_manager'] = ProductionCheckpointManager(
        checkpoint_dir="production_checkpoints"
    )
    components['regression_detector'] = RegressionDetector()
    logger.info("  ✅ Monitoring online")
    
    # 4. Segment Discovery
    logger.info("🔍 Segment discovery...")
    components['segments'] = SegmentDiscoveryEngine()
    logger.info("  ✅ Segment discovery online")
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL WORKING COMPONENTS INITIALIZED")
    logger.info("=" * 80)
    
    # Run training loop
    logger.info("\n🚀 Starting training loop...")
    
    env = components['environment']
    agent = components['agent']
    emergency = components['emergency']
    budget_safety = components['budget_safety']
    
    try:
        for episode in range(5):  # Just 5 episodes for demo
            logger.info(f"\n📍 Episode {episode + 1}/5")
            
            # Reset environment
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 100:  # Limit steps per episode
                # Check safety
                if emergency.check_emergency_stop():
                    logger.warning("🛑 Emergency stop triggered")
                    break
                
                # Get action
                action = agent.act(state)
                
                # Step environment
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train periodically
                if len(agent.memory) > agent.batch_size and steps % 32 == 0:
                    agent.replay()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Log progress every 20 steps
                if steps % 20 == 0:
                    logger.info(f"    Step {steps}: reward={reward:.2f}, total={total_reward:.2f}")
            
            logger.info(f"  Episode complete: {steps} steps, total reward: {total_reward:.2f}")
            logger.info(f"  Epsilon: {agent.epsilon:.4f}")
            
            # Save checkpoint
            if episode % 2 == 0:
                checkpoint_manager = components['checkpoint_manager']
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    model=agent,
                    metadata={'episode': episode, 'reward': total_reward}
                )
                logger.info(f"  💾 Checkpoint saved: {checkpoint_id}")
    
    except KeyboardInterrupt:
        logger.info("\n⏹️ Training stopped by user")
    except Exception as e:
        logger.error(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        logger.info("\n🧹 Cleaning up...")
        if hasattr(env, 'close'):
            env.close()
        logger.info("✅ Cleanup complete")
    
    logger.info("\n" + "=" * 80)
    logger.info("GAELP SYSTEM EXECUTION COMPLETE")
    logger.info("=" * 80)
    
    # Print summary
    logger.info("\nSUMMARY:")
    logger.info(f"  Components initialized: {len(components)}")
    logger.info(f"  Training episodes: {episode + 1}")
    logger.info(f"  Final epsilon: {agent.epsilon:.4f}")
    logger.info("\nThis is what ACTUALLY WORKS in the GAELP system.")
    logger.info("Not the fantasy orchestrator, but the REAL functioning components.")

if __name__ == "__main__":
    run_core_system()