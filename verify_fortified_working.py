#!/usr/bin/env python3
"""
Verify that the fortified system is working correctly
"""

import json
import sys
import time
import signal
import subprocess

def check_json_file():
    """Verify JSON file is valid"""
    try:
        with open('discovered_patterns.json', 'r') as f:
            data = json.load(f)
        print("✓ JSON file is valid")
        return True
    except Exception as e:
        print(f"✗ JSON error: {e}")
        return False

def run_minimal_test():
    """Run a minimal test of the system"""
    print("\n" + "="*70)
    print("Running minimal test...")
    print("="*70)
    
    try:
        # Run test with timeout
        result = subprocess.run(
            ['python3', 'test_fortified_minimal.py'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "✅ MINIMAL TEST PASSED" in result.stdout:
            print("✓ Minimal test passed")
            return True
        else:
            print("✗ Minimal test failed")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Test timed out")
        return False
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False

def test_fortified_training():
    """Test fortified training for a few steps"""
    print("\n" + "="*70)
    print("Testing fortified training (5 seconds)...")
    print("="*70)
    
    try:
        # Start training process
        proc = subprocess.Popen(
            ['python3', 'capture_fortified_training.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Let it run for 5 seconds
        time.sleep(5)
        
        # Terminate gracefully
        proc.terminate()
        time.sleep(1)
        
        # Force kill if still running
        if proc.poll() is None:
            proc.kill()
        
        # Check output
        stdout, stderr = proc.communicate(timeout=2)
        
        # Look for successful patterns
        success_indicators = [
            "Episode",
            "INFO:fortified_environment:Fortified environment initialized successfully",
            "AuctionGym initialized",
            "GA4DiscoveryEngine"
        ]
        
        errors = []
        for indicator in success_indicators:
            if indicator not in stdout and indicator not in stderr:
                errors.append(f"Missing: {indicator}")
        
        # Check for critical errors
        critical_errors = [
            "KeyError",
            "AttributeError", 
            "TypeError",
            "CRITICAL:gaelp_parameter_manager:USING EMERGENCY FALLBACK"
        ]
        
        for error in critical_errors:
            if error in stderr:
                errors.append(f"Found error: {error}")
        
        if not errors:
            print("✓ Fortified training is working")
            return True
        else:
            print("✗ Fortified training has issues:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            return False
            
    except Exception as e:
        print(f"✗ Training test error: {e}")
        return False

def main():
    """Run all verification tests"""
    print("\n" + "="*70)
    print("FORTIFIED SYSTEM VERIFICATION")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = []
    
    # Check JSON
    results.append(("JSON file valid", check_json_file()))
    
    # Run minimal test
    results.append(("Minimal test", run_minimal_test()))
    
    # Test training
    results.append(("Training test", test_fortified_training()))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Fortified system is working!")
        print("\nYou can now run full training with:")
        print("  python3 capture_fortified_training.py")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Fix issues above")
        print("\nKey issues to check:")
        print("1. JSON file format (discovered_patterns.json)")
        print("2. Action dictionary keys (bid/bid_amount, creative/creative_id)")
        print("3. Channel format (string vs index)")
        return 1

if __name__ == "__main__":
    sys.exit(main())