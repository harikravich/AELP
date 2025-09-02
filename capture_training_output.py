#!/usr/bin/env python3
"""
Capture training output to a file for monitoring
"""

import subprocess
import sys
import time
from datetime import datetime

def main():
    """Run training and capture all output"""
    
    print(f"Starting training capture at {datetime.now()}")
    
    # Open log file
    log_file = "training_output.log"
    
    with open(log_file, "w", buffering=1) as f:  # Line buffered
        # Write header
        f.write(f"=== GAELP Training Started at {datetime.now()} ===\n")
        f.flush()
        
        # Launch training process
        process = subprocess.Popen(
            ["python3", "launch_parallel_training.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        print(f"Training PID: {process.pid}")
        print(f"Output being written to: {log_file}")
        print("Press Ctrl+C to stop...")
        
        # Stream output to both file and console
        try:
            for line in process.stdout:
                # Write to file
                f.write(line)
                f.flush()
                
                # PRINT ALL TO CONSOLE
                print(line.rstrip())
                    
        except KeyboardInterrupt:
            print("\nStopping training...")
            process.terminate()
            process.wait(timeout=5)
            
        # Wait for process to complete
        return_code = process.wait()
        
        f.write(f"\n=== Training ended with code {return_code} at {datetime.now()} ===\n")
        
    print(f"Training complete. Output saved to {log_file}")
    
if __name__ == "__main__":
    main()