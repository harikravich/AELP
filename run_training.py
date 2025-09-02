#!/usr/bin/env python3
"""
Training launcher - choose between original or fortified training
"""

import sys
import subprocess
from datetime import datetime

def print_banner():
    """Print selection banner"""
    print("\n" + "=" * 70)
    print(" GAELP TRAINING LAUNCHER ".center(70))
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

def main():
    """Main launcher"""
    print_banner()
    
    print("\nSelect training mode:\n")
    print("1. 🚀 FORTIFIED Training (Recommended)")
    print("   - 45-dimensional enriched state vector")
    print("   - Multi-dimensional actions (bid + creative + channel)")
    print("   - All components integrated")
    print("   - Sophisticated rewards")
    print()
    print("2. 📦 Original Training")
    print("   - Basic state vector")
    print("   - Bid-only actions")
    print("   - Simple rewards")
    print()
    print("3. 🧪 Test Fortified System")
    print("   - Run tests to verify all components")
    print()
    print("0. Exit")
    print()
    
    choice = input("Enter choice (1/2/3/0): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 70)
        print("Starting FORTIFIED training...")
        print("=" * 70)
        print("\nThis will:")
        print("- Use all integrated components")
        print("- Learn creative selection, channel optimization, and bidding")
        print("- Apply sophisticated multi-component rewards")
        print("- Run 16 parallel environments")
        print("\nPress Ctrl+C to stop at any time")
        print("=" * 70)
        
        input("\nPress Enter to start fortified training...")
        
        # Run fortified training with output capture
        subprocess.run(["python3", "capture_fortified_training.py"])
        
    elif choice == "2":
        print("\n" + "=" * 70)
        print("Starting ORIGINAL training...")
        print("=" * 70)
        print("\nThis will:")
        print("- Use basic RL agent")
        print("- Learn bidding only")
        print("- Simple reward structure")
        print("\nPress Ctrl+C to stop at any time")
        print("=" * 70)
        
        input("\nPress Enter to start original training...")
        
        # Run original training
        subprocess.run(["python3", "capture_training_output.py"])
        
    elif choice == "3":
        print("\n" + "=" * 70)
        print("Running FORTIFIED SYSTEM TESTS...")
        print("=" * 70)
        
        # Run tests
        subprocess.run(["python3", "test_fortified_system.py"])
        
        print("\n" + "=" * 70)
        print("Tests complete. Check output above.")
        print("=" * 70)
        
    elif choice == "0":
        print("\nExiting...")
        return
    else:
        print("\nInvalid choice. Please try again.")
        return main()
    
    print("\n" + "=" * 70)
    print("Training launcher complete")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nLauncher interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)