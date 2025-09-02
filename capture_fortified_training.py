#!/usr/bin/env python3
"""
Capture fortified training output to a file for monitoring
Works with the new fortified training system
"""

import subprocess
import sys
import time
from datetime import datetime

def main():
    """Run fortified training and capture all output"""
    
    print(f"Starting FORTIFIED training capture at {datetime.now()}")
    print("="*70)
    
    # Open log file
    log_file = "fortified_training_output.log"
    
    with open(log_file, "w", buffering=1) as f:  # Line buffered
        # Write header
        f.write(f"=== FORTIFIED GAELP Training Started at {datetime.now()} ===\n")
        f.write("=" * 70 + "\n")
        f.write("Training with complete component integration:\n")
        f.write("- 45-dimensional enriched state vector\n")
        f.write("- Multi-dimensional actions (bid + creative + channel)\n")
        f.write("- Sophisticated multi-component rewards\n")
        f.write("- All components integrated (Creative Selector, Attribution, etc.)\n")
        f.write("=" * 70 + "\n\n")
        f.flush()
        
        # Launch fortified training process
        process = subprocess.Popen(
            ["python3", "fortified_training_loop.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        print(f"Fortified Training PID: {process.pid}")
        print(f"Output being written to: {log_file}")
        print("Press Ctrl+C to stop...")
        print("="*70)
        
        # Stream output to both file and console (filtered)
        try:
            import re
            
            # Patterns to filter out Ray noise
            filter_patterns = [
                # Empty pid lines with any whitespace
                re.compile(r'^\(ParallelEnvironment\s+pid=\d+\)\s*$'),  
                re.compile(r'^\(pid=\d+\)\s*$'),
                # Lines that are JUST the pid part with nothing after
                re.compile(r'^\(ParallelEnvironment\s+pid=\d+\)$'),
                # Repeated log lines
                re.compile(r'.*\[repeated \d+x across cluster\]'),
                # Ray messages
                re.compile(r'.*Ray deduplicates logs by default.*'),
                # BigQuery/table spam
                re.compile(r'^INFO:.*BigQuery.*established'),
                re.compile(r'^INFO:.*Dataset.*already exists'),
                re.compile(r'^INFO:.*Table.*already exists'),
                # Discovery engine banners (too many)
                re.compile(r'^={70,}$'),  # Long lines of equals
                re.compile(r'^🔬 DISCOVERY ENGINE'),
                re.compile(r'^🎯 Discovering'),
                re.compile(r'^👥 Discovering'),
                re.compile(r'^📊 Discovering'),
                re.compile(r'^📈 Processing'),
                re.compile(r'^⏰ Processing'),
                re.compile(r'^💾 Saved'),
                re.compile(r'^✅ Basic patterns'),
            ]
            
            for line in process.stdout:
                # Write everything to file
                f.write(line)
                f.flush()
                
                # Strip the line for checking
                line_stripped = line.rstrip()
                
                # Skip completely empty lines
                if not line_stripped:
                    continue
                
                # Skip lines that are ONLY a pid reference (more aggressive)
                if line_stripped.startswith('(ParallelEnvironment pid=') and line_stripped.endswith(')'):
                    # Check if there's any actual content after the pid
                    if line_stripped.count(')') == 1:  # Just the closing paren
                        continue
                
                # Skip other filtered patterns
                skip = False
                for pattern in filter_patterns:
                    if pattern.match(line_stripped):
                        skip = True
                        break
                if skip:
                    continue
                
                # Highlight important lines
                if any(keyword in line_stripped for keyword in [
                    "Episode", "Loss", "ROAS", "Conversions", 
                    "CTR", "CVR", "Channel Performance", "Creative",
                    "Training converged", "Checkpoint saved"
                ]):
                    print(f"► {line_stripped}")
                else:
                    print(line_stripped)
                    
        except KeyboardInterrupt:
            print("\n" + "="*70)
            print("Stopping fortified training...")
            process.terminate()
            process.wait(timeout=5)
            
        # Wait for process to complete
        return_code = process.wait()
        
        f.write(f"\n=== Fortified training ended with code {return_code} at {datetime.now()} ===\n")
        
    print("="*70)
    print(f"Fortified training complete. Output saved to {log_file}")
    print("="*70)
    
if __name__ == "__main__":
    main()