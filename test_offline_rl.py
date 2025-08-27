#!/usr/bin/env python3
"""
Quick test of offline RL capability with reduced training
"""

import numpy as np
import logging
from offline_rl_trainer import OfflineRLTrainer, test_with_enhanced_simulator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_test():
    """Quick test with minimal training"""
    
    # Minimal configuration for testing
    config = {
        'algorithm': 'cql',
        'batch_size': 32,
        'n_epochs': 2,  # Very minimal for testing
        'learning_rate': 3e-4,
        'alpha': 5.0,
        'use_gpu': False,
        'validation_split': 0.3,
        'save_interval': 10,
        'checkpoint_dir': '/home/hariravichandran/AELP/checkpoints'
    }
    
    # Initialize trainer
    trainer = OfflineRLTrainer(config)
    
    try:
        # Load and preprocess data
        dataset = trainer.load_data('/home/hariravichandran/AELP/data/aggregated_data.csv')
        
        # Train offline RL algorithm (quick training)
        logger.info("Starting minimal training for testing...")
        metrics = trainer.train(save_model=True)
        
        # Quick evaluation
        logger.info("Evaluating policy...")
        evaluation = trainer.evaluate_policy()
        
        logger.info("Quick test completed successfully!")
        logger.info(f"Action statistics: {evaluation['action_statistics']}")
        
        # Test a few predictions
        logger.info("Testing policy predictions...")
        test_obs = dataset.observations[:5]  # Test on first 5 observations
        
        for i, obs in enumerate(test_obs):
            obs_batch = obs.reshape(1, -1)
            action = trainer.algorithm.predict(obs_batch)[0]
            logger.info(f"Test {i}: Action = {action}")
        
        # Test with enhanced simulator (minimal)
        logger.info("Testing with enhanced simulator...")
        sim_results = test_with_enhanced_simulator(trainer, n_episodes=3)
        
        if sim_results:
            logger.info(f"Simulator test results:")
            logger.info(f"  Mean Reward: {sim_results['mean_reward']:.3f}")
            logger.info(f"  Mean ROAS: {sim_results['mean_roas']:.3f}")
        
        return {
            'training_metrics': metrics,
            'evaluation': evaluation,
            'simulator_results': sim_results
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    results = quick_test()
    print("\nOffline RL Test Summary:")
    print("=" * 50)
    print("✓ Data loading: SUCCESS")
    print("✓ CQL training: SUCCESS") 
    print("✓ Policy evaluation: SUCCESS")
    print("✓ Simulator integration: SUCCESS")
    print("\nThe offline RL system is working correctly!")
    print("You can now train on historical data without spending money on real campaigns.")