#!/usr/bin/env python3
"""
Run fortified training with filtered output
"""

import subprocess
import sys
import re

def main():
    print("\n" + "="*70)
    print("FORTIFIED TRAINING - FILTERED OUTPUT")
    print("="*70)
    print("Starting training with filtered logs...")
    print("Only showing important progress updates")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Patterns to filter out
    filter_patterns = [
        r'^\(ParallelEnvironment pid=\d+\)\s*$',  # Empty pid lines
        r'^\(pid=\d+\)\s*$',  # More empty pid lines
        r'INFO:.*BigQuery.*established',  # BigQuery connection spam
        r'INFO:.*Dataset.*already exists',  # Dataset spam
        r'INFO:.*Table.*already exists',  # Table spam
        r'INFO:.*initialized with',  # Initialization spam
        r'🔬 DISCOVERY ENGINE',  # Discovery banner
        r'================',  # Banner lines
        r'🎯 Discovering',  # Discovery progress
        r'👥 Discovering',  # More discovery
        r'📊 Discovering',  # More discovery
        r'📈 Processing',  # Processing
        r'⏰ Processing',  # More processing
        r'💾 Saved',  # Save messages
        r'✅ Basic patterns',  # Pattern messages
        r'✅ GA4Discovery',  # GA4 messages
        r'✅ AuctionGym',  # AuctionGym messages
    ]
    
    # Compile patterns for efficiency
    compiled_filters = [re.compile(p) for p in filter_patterns]
    
    # Patterns to highlight (show these)
    important_patterns = [
        r'Episode \d+/\d+',  # Episode progress
        r'Average Reward:',  # Reward info
        r'Total Conversions:',  # Conversion info
        r'ROAS:',  # ROAS metrics
        r'ERROR:',  # Errors
        r'CRITICAL:',  # Critical errors
        r'Training complete',  # Completion
    ]
    compiled_important = [re.compile(p) for p in important_patterns]
    
    try:
        # Run training subprocess
        proc = subprocess.Popen(
            ['python3', 'capture_fortified_training.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Process output line by line
        for line in proc.stdout:
            # Strip trailing whitespace
            line = line.rstrip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip filtered patterns
            skip = False
            for pattern in compiled_filters:
                if pattern.search(line):
                    skip = True
                    break
            
            if skip:
                continue
            
            # Check if it's important
            important = False
            for pattern in compiled_important:
                if pattern.search(line):
                    important = True
                    break
            
            # Show important lines or lines that aren't from Ray workers
            if important or not line.startswith('('):
                print(line)
                sys.stdout.flush()
        
        # Wait for process to complete
        proc.wait()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        if proc:
            proc.terminate()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()