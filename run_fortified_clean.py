#!/usr/bin/env python3
"""
Clean fortified training runner with reduced logging noise
"""

import os
import sys
import logging

# Set Ray logging to WARNING level
os.environ['RAY_DEDUP_LOGS'] = '1'
os.environ['RAY_LOG_TO_STDERR'] = '0'

# Suppress BigQuery INFO logs
logging.getLogger('google.cloud.bigquery').setLevel(logging.WARNING)
logging.getLogger('bigquery_batch_writer').setLevel(logging.WARNING)
logging.getLogger('discovery_engine').setLevel(logging.WARNING)
logging.getLogger('gaelp_parameter_manager').setLevel(logging.WARNING)

# Import after setting env vars
import subprocess

def main():
    print("Starting FORTIFIED training with reduced logging...")
    print("=" * 70)
    
    # Run the actual training with cleaner output
    subprocess.run([
        "python3", "fortified_training_loop.py"
    ])

if __name__ == "__main__":
    main()
