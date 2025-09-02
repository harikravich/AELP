#!/usr/bin/env python3
"""
Launch script for parallel training
Optimized for 16-core machine
"""

import multiprocessing as mp
import logging
import sys
import time
from parallel_training_accelerator import ParallelConfig, ParallelTrainingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Launch parallel training with optimal settings"""
    
    # Detect available cores
    available_cores = mp.cpu_count()
    logger.info(f"🖥️  Detected {available_cores} CPU cores")
    
    # Optimal configuration based on cores
    if available_cores >= 16:
        # High-performance mode
        n_envs = 16  # One env per core
        n_workers = 16
        batch_size = 512  # Larger batch for more data
        logger.info("🚀 HIGH PERFORMANCE MODE: 16 parallel environments")
    elif available_cores >= 8:
        # Medium performance
        n_envs = 8
        n_workers = 8
        batch_size = 256
        logger.info("⚡ MEDIUM PERFORMANCE MODE: 8 parallel environments")
    else:
        # Low performance (current 2-core setup)
        n_envs = 2
        n_workers = 2
        batch_size = 64
        logger.info("🐌 LOW PERFORMANCE MODE: 2 parallel environments (upgrade recommended!)")
    
    # Calculate training time estimate
    episodes_target = 100000
    episodes_per_env = episodes_target // n_envs
    
    # Rough estimate: 100 steps per episode, 10 steps/second per env
    estimated_hours = (episodes_per_env * 100) / (10 * 3600)
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║          GAELP PARALLEL TRAINING LAUNCHER                 ║
    ║                                                            ║
    ║  Configuration:                                            ║
    ║  - CPU Cores: {available_cores:2d}                                         ║
    ║  - Parallel Environments: {n_envs:2d}                              ║
    ║  - Target Episodes: {episodes_target:,}                        ║
    ║  - Episodes per Environment: {episodes_per_env:,}              ║
    ║  - Estimated Training Time: {estimated_hours:.1f} hours              ║
    ║  - Cost Estimate: ${estimated_hours * 0.38:.2f}                        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    if available_cores < 16:
        print("""
    ⚠️  WARNING: Running on limited cores!
    
    To upgrade for 8x faster training:
    1. Stop instance:    gcloud compute instances stop thrive-backend --zone=us-central1-a
    2. Upgrade machine:  gcloud compute instances set-machine-type thrive-backend --machine-type=n1-standard-16 --zone=us-central1-a
    3. Start instance:   gcloud compute instances start thrive-backend --zone=us-central1-a
    
    Current: {} cores = ~{:.1f} hours
    Upgrade: 16 cores = ~{:.1f} hours
        """.format(available_cores, estimated_hours, estimated_hours * available_cores / 16))
    
    # Auto-start training
    logger.info("🎯 Auto-starting training...")
    
    # Create configuration
    config = ParallelConfig(
        n_envs=n_envs,
        n_workers=n_workers,
        batch_size=batch_size,
        episodes_per_env=episodes_per_env,
        use_ray=True,  # Ray is installed
        checkpoint_freq=1000  # Save every 1000 episodes
    )
    
    # Launch training
    logger.info("🚀 Launching parallel training...")
    logger.info(f"   Check progress in: logs/parallel_training_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    try:
        orchestrator = ParallelTrainingOrchestrator(config)
        
        # Start training
        start_time = time.time()
        orchestrator.train_parallel(n_episodes=episodes_target)
        
        # Training complete
        elapsed = time.time() - start_time
        logger.info(f"✅ Training complete in {elapsed/3600:.1f} hours!")
        logger.info(f"   Total cost: ${elapsed/3600 * 0.38:.2f}")
        
        # Save final model
        orchestrator.save_checkpoint(episodes_target)
        logger.info("💾 Final model saved to checkpoints/parallel/")
        
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        orchestrator.save_checkpoint(orchestrator.total_episodes)
        logger.info(f"💾 Checkpoint saved at episode {orchestrator.total_episodes}")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    logger.info("🎉 Training session complete!")

if __name__ == "__main__":
    main()