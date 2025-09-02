#!/usr/bin/env python3
"""
Run fortified training with minimal output
"""

import os
import sys
import logging

# Suppress most logging
os.environ['RAY_DEDUP_LOGS'] = '1'  # Enable Ray log deduplication
os.environ['RAY_BACKEND_LOG_LEVEL'] = 'error'  # Only show Ray errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Set all loggers to ERROR level
logging.basicConfig(level=logging.ERROR)
logging.getLogger('ray').setLevel(logging.ERROR)
logging.getLogger('ray.rllib').setLevel(logging.ERROR)
logging.getLogger('ray.tune').setLevel(logging.ERROR)
logging.getLogger('gaelp_parameter_manager').setLevel(logging.ERROR)
logging.getLogger('fortified_environment').setLevel(logging.ERROR)
logging.getLogger('persistent_user_database').setLevel(logging.ERROR)
logging.getLogger('bigquery_batch_writer').setLevel(logging.ERROR)
logging.getLogger('user_journey_database').setLevel(logging.ERROR)
logging.getLogger('training_orchestrator').setLevel(logging.ERROR)
logging.getLogger('auction_gym_integration_fixed').setLevel(logging.ERROR)
logging.getLogger('discovery_engine').setLevel(logging.ERROR)
logging.getLogger('budget_pacer').setLevel(logging.ERROR)

# Import after setting log levels
from fortified_training_loop import main

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FORTIFIED TRAINING (QUIET MODE)")
    print("="*70)
    print("Starting training with minimal output...")
    print("Progress will be shown every 10 episodes")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        sys.exit(1)