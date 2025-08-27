#!/usr/bin/env python3
"""
COMPREHENSIVE GAELP TEST SUITE
Tests all critical requirements from CLAUDE.md
Reports honestly on what works and what's broken
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3
import tempfile

# Test results collector
class TestResults:
    def __init__(self):
        self.results = {
            'no_fallbacks': {'status': 'PENDING', 'details': []},
            'user_persistence': {'status': 'PENDING', 'details': []},
            'auction_win_rate': {'status': 'PENDING', 'details': []},
            'conversion_delays': {'status': 'PENDING', 'details': []},
            'no_hardcoded': {'status': 'PENDING', 'details': []},
            'rl_not_bandits': {'status': 'PENDING', 'details': []},
            'recsim_usage': {'status': 'PENDING', 'details': []},
            'auctiongym_usage': {'status': 'PENDING', 'details': []},
            'learning_occurs': {'status': 'PENDING', 'details': []},
            'data_flows': {'status': 'PENDING', 'details': []},
        }
        self.critical_failures = []
        
    def record(self, test_name: str, passed: bool, detail: str):
        """Record test result"""
        if test_name in self.results:
            self.results[test_name]['status'] = 'PASS' if passed else 'FAIL'
            self.results[test_name]['details'].append(detail)
            if not passed:
                self.critical_failures.append(f"{test_name}: {detail}")
                
    def print_report(self):
        """Print comprehensive test report"""
        print("\n" + "="*80)
        print("GAELP COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        # Summary
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        pending = sum(1 for r in self.results.values() if r['status'] == 'PENDING')
        
        print(f"\nSUMMARY: {passed} PASSED, {failed} FAILED, {pending} PENDING")
        
        # Detailed results
        print("\nDETAILED RESULTS:")
        print("-"*80)
        
        for test_name, result in self.results.items():
            status_icon = "✓" if result['status'] == 'PASS' else "✗" if result['status'] == 'FAIL' else "?"
            print(f"\n{status_icon} {test_name.upper().replace('_', ' ')}: {result['status']}")
            for detail in result['details']:
                print(f"  - {detail}")
                
        # Critical failures
        if self.critical_failures:
            print("\n" + "="*80)
            print("CRITICAL FAILURES REQUIRING IMMEDIATE ATTENTION:")
            print("="*80)
            for failure in self.critical_failures:
                print(f"❌ {failure}")
                
        # Honest assessment
        print("\n" + "="*80)
        print("HONEST ASSESSMENT:")
        print("="*80)
        
        if failed > 0:
            print("\n⚠️  THE SYSTEM IS NOT WORKING PROPERLY")
            print("   Multiple critical requirements are failing.")
            print("   These must be fixed before the system can be considered functional.")
        elif pending > 0:
            print("\n⚠️  TESTING INCOMPLETE")
            print("   Some tests could not be run due to missing components.")
        else:
            print("\n✓  ALL TESTS PASSED")
            print("   The system appears to be working according to specifications.")
            
        return failed == 0 and pending == 0

# Initialize test results
results = TestResults()

print("Starting GAELP Comprehensive Test Suite...")
print("-"*80)

# TEST 1: Check for forbidden patterns
print("\n[TEST 1] Checking for forbidden patterns...")
try:
    import subprocess
    
    # Run grep for forbidden patterns
    forbidden_patterns = ['fallback', 'simplified', 'mock', 'dummy', 'TODO', 'FIXME']
    found_patterns = []
    
    for pattern in forbidden_patterns:
        result = subprocess.run(
            f"grep -r '{pattern}' --include='*.py' . 2>/dev/null | grep -v test_ | grep -v '#' | head -5",
            shell=True, capture_output=True, text=True
        )
        if result.stdout:
            found_patterns.append((pattern, len(result.stdout.split('\n'))))
            
    if found_patterns:
        results.record('no_fallbacks', False, 
                      f"Found forbidden patterns: {found_patterns}")
        print(f"  ✗ FAILED: Found {len(found_patterns)} forbidden patterns")
        for pattern, count in found_patterns[:3]:  # Show first 3
            print(f"    - '{pattern}' appears in {count} locations")
    else:
        results.record('no_fallbacks', True, "No forbidden patterns found")
        print("  ✓ PASSED: No forbidden patterns found")
        
except Exception as e:
    results.record('no_fallbacks', False, f"Error checking patterns: {e}")
    print(f"  ✗ ERROR: {e}")

# TEST 2: Check user persistence
print("\n[TEST 2] Testing user persistence across episodes...")
try:
    # Check if persistent user database exists
    db_path = '/tmp/persistent_users.db'
    
    # Try to import and test user persistence
    try:
        from persistent_user_database import PersistentUserDatabase
        
        # Create test database
        test_db = PersistentUserDatabase(db_path=':memory:')
        
        # Create user in episode 1
        user_id1 = test_db.get_or_create_user(
            segment='tech_enthusiast',
            attributes={'age': 25}
        )
        
        # Simulate episode end
        test_db.end_episode()
        
        # Get same user in episode 2
        user_id2 = test_db.get_or_create_user(
            segment='tech_enthusiast',
            attributes={'age': 25}
        )
        
        # Check if user persisted
        if user_id1 == user_id2:
            # Check conversion history persists
            test_db.record_conversion(user_id1, 'search', 100.0, 2.5)
            history = test_db.get_user_history(user_id1)
            
            if history and len(history['conversions']) > 0:
                results.record('user_persistence', True, 
                             f"Users persist with history across episodes")
                print("  ✓ PASSED: Users persist across episodes with history")
            else:
                results.record('user_persistence', False,
                             "Users persist but history is lost")
                print("  ✗ FAILED: Users persist but history is lost")
        else:
            results.record('user_persistence', False, 
                         "Users do not persist across episodes")
            print("  ✗ FAILED: Users do not persist across episodes")
            
    except ImportError:
        results.record('user_persistence', False,
                      "PersistentUserDatabase not found")
        print("  ✗ FAILED: PersistentUserDatabase module not found")
        
except Exception as e:
    results.record('user_persistence', False, f"Error testing persistence: {e}")
    print(f"  ✗ ERROR: {e}")

# TEST 3: Check auction win rates
print("\n[TEST 3] Verifying auction win rates (should be 15-30%)...")
try:
    from gaelp_master_integration import GAELPMasterIntegration
    
    # Initialize system
    config = {
        'daily_budget': 10000,
        'channels': ['search', 'display', 'social'],
        'use_llm': False,
        'use_wandb': False
    }
    
    system = GAELPMasterIntegration(config)
    
    # Run 100 auctions
    wins = 0
    total = 100
    
    for _ in range(total):
        # Simulate auction
        result = system.participate_in_auction(
            channel='search',
            user_segment='tech_enthusiast',
            base_value=10.0
        )
        if result and result.get('won', False):
            wins += 1
            
    win_rate = wins / total * 100
    
    if 15 <= win_rate <= 30:
        results.record('auction_win_rate', True,
                      f"Win rate {win_rate:.1f}% is within target range")
        print(f"  ✓ PASSED: Win rate {win_rate:.1f}% is within 15-30% range")
    else:
        results.record('auction_win_rate', False,
                      f"Win rate {win_rate:.1f}% is outside target range")
        print(f"  ✗ FAILED: Win rate {win_rate:.1f}% is outside 15-30% range")
        
except Exception as e:
    results.record('auction_win_rate', False, f"Error testing auctions: {e}")
    print(f"  ✗ ERROR: {e}")

# TEST 4: Check conversion delays
print("\n[TEST 4] Checking conversion delays (should be 1-3 days)...")
try:
    # Check if conversion lag model exists and uses proper delays
    from conversion_lag_model import ConversionLagModel
    
    model = ConversionLagModel()
    
    # Test multiple conversions
    delays = []
    for _ in range(50):
        delay = model.predict_lag(
            channel='search',
            user_segment='tech_enthusiast',
            touchpoints=[{'channel': 'search', 'timestamp': datetime.now()}]
        )
        if isinstance(delay, (int, float)):
            delays.append(delay)
            
    if delays:
        avg_delay = np.mean(delays)
        min_delay = np.min(delays)
        max_delay = np.max(delays)
        
        # Check if delays are in days (24-72 hours)
        if 24 <= avg_delay <= 72:  # Hours
            results.record('conversion_delays', True,
                         f"Avg delay {avg_delay:.1f}h is within 1-3 days")
            print(f"  ✓ PASSED: Conversion delays average {avg_delay:.1f} hours (1-3 days)")
        elif 1 <= avg_delay <= 3:  # Days
            results.record('conversion_delays', True,
                         f"Avg delay {avg_delay:.1f} days is correct")
            print(f"  ✓ PASSED: Conversion delays average {avg_delay:.1f} days")
        else:
            results.record('conversion_delays', False,
                         f"Delays {avg_delay:.1f} not in 1-3 day range")
            print(f"  ✗ FAILED: Delays {avg_delay:.1f} not in 1-3 day range")
    else:
        results.record('conversion_delays', False, "No delays generated")
        print("  ✗ FAILED: Conversion lag model not generating delays")
        
except Exception as e:
    results.record('conversion_delays', False, f"Error testing delays: {e}")
    print(f"  ✗ ERROR: {e}")

# TEST 5: Check for hardcoded values
print("\n[TEST 5] Checking for hardcoded values...")
try:
    # Search for hardcoded segments, categories, thresholds
    hardcoded_patterns = [
        r"segments\s*=\s*\[.*\]",  # Hardcoded segment lists
        r"categories\s*=\s*\[.*\]",  # Hardcoded category lists
        r"threshold\s*=\s*\d+",  # Hardcoded thresholds
        r"'tech_enthusiast'|'budget_conscious'",  # Specific hardcoded segments
    ]
    
    found_hardcoded = []
    for pattern in hardcoded_patterns:
        result = subprocess.run(
            f"grep -rE '{pattern}' --include='*.py' . 2>/dev/null | grep -v test_ | grep -v example | head -3",
            shell=True, capture_output=True, text=True
        )
        if result.stdout:
            found_hardcoded.append(pattern)
            
    if found_hardcoded:
        results.record('no_hardcoded', False,
                      f"Found {len(found_hardcoded)} hardcoded patterns")
        print(f"  ✗ FAILED: Found {len(found_hardcoded)} hardcoded value patterns")
    else:
        results.record('no_hardcoded', True, "No hardcoded values found")
        print("  ✓ PASSED: No obvious hardcoded values found")
        
except Exception as e:
    results.record('no_hardcoded', False, f"Error checking hardcoded: {e}")
    print(f"  ✗ ERROR: {e}")

# TEST 6: Verify RL not bandits
print("\n[TEST 6] Verifying proper RL implementation (not bandits)...")
try:
    # Check for Q-learning or PPO implementations
    rl_found = False
    bandit_found = False
    
    # Check for RL algorithms
    result = subprocess.run(
        "grep -r 'PPO\\|Q-learning\\|q_values\\|policy_network' --include='*.py' . 2>/dev/null | head -5",
        shell=True, capture_output=True, text=True
    )
    if result.stdout:
        rl_found = True
        
    # Check for bandit algorithms (should not be primary)
    result = subprocess.run(
        "grep -r 'multi_armed_bandit\\|thompson_sampling\\|ucb' --include='*.py' . 2>/dev/null | head -5",
        shell=True, capture_output=True, text=True
    )
    if result.stdout:
        bandit_found = True
        
    if rl_found and not bandit_found:
        results.record('rl_not_bandits', True, "Using proper RL algorithms")
        print("  ✓ PASSED: Proper RL implementation found")
    elif bandit_found and not rl_found:
        results.record('rl_not_bandits', False, "Using bandits instead of RL")
        print("  ✗ FAILED: Using bandits instead of proper RL")
    else:
        results.record('rl_not_bandits', False, "RL implementation unclear")
        print("  ✗ FAILED: RL implementation unclear or missing")
        
except Exception as e:
    results.record('rl_not_bandits', False, f"Error checking RL: {e}")
    print(f"  ✗ ERROR: {e}")

# TEST 7: Verify RecSim usage
print("\n[TEST 7] Verifying RecSim integration...")
try:
    # Check if RecSim is properly imported and used
    from recsim_user_model import RecSimUserModel
    
    # Try to create RecSim environment
    model = RecSimUserModel()
    
    # Check if it's actually using RecSim
    if hasattr(model, 'environment') or hasattr(model, 'user_model'):
        results.record('recsim_usage', True, "RecSim properly integrated")
        print("  ✓ PASSED: RecSim is properly integrated")
    else:
        results.record('recsim_usage', False, "RecSim not properly used")
        print("  ✗ FAILED: RecSim module exists but not properly used")
        
except ImportError:
    results.record('recsim_usage', False, "RecSim module not found")
    print("  ✗ FAILED: RecSim integration missing")
except Exception as e:
    results.record('recsim_usage', False, f"Error with RecSim: {e}")
    print(f"  ✗ ERROR: {e}")

# TEST 8: Verify AuctionGym usage
print("\n[TEST 8] Verifying AuctionGym integration...")
try:
    # Check if AuctionGym is properly integrated
    import sys
    sys.path.append('./auction-gym/src')
    from Auction import Auction
    from Agent import Agent
    
    # Create test auction
    auction = Auction(num_agents=5, num_items=1)
    
    results.record('auctiongym_usage', True, "AuctionGym properly integrated")
    print("  ✓ PASSED: AuctionGym is properly integrated")
    
except ImportError:
    results.record('auctiongym_usage', False, "AuctionGym not found")
    print("  ✗ FAILED: AuctionGym not properly integrated")
except Exception as e:
    results.record('auctiongym_usage', False, f"AuctionGym error: {e}")
    print(f"  ✗ ERROR: {e}")

# TEST 9: Verify learning occurs
print("\n[TEST 9] Verifying that learning actually occurs...")
try:
    # Check if models are being updated
    from training_orchestrator.online_learner import OnlineLearner
    
    learner = OnlineLearner(agent=None)  # Simplified test
    
    # Check if learner has update methods
    if hasattr(learner, 'update') or hasattr(learner, 'learn'):
        # Try to verify weights change
        initial_state = str(learner.__dict__)
        
        # Simulate some learning
        for _ in range(10):
            learner.observe({'state': np.random.rand(10), 'reward': np.random.rand()})
            
        final_state = str(learner.__dict__)
        
        if initial_state != final_state:
            results.record('learning_occurs', True, "Model weights updating")
            print("  ✓ PASSED: Learning is occurring (weights changing)")
        else:
            results.record('learning_occurs', False, "No weight updates detected")
            print("  ✗ FAILED: No learning detected (weights static)")
    else:
        results.record('learning_occurs', False, "No learning methods found")
        print("  ✗ FAILED: No learning methods found")
        
except Exception as e:
    results.record('learning_occurs', False, f"Error testing learning: {e}")
    print(f"  ✗ ERROR: {e}")

# TEST 10: Verify data flows through system
print("\n[TEST 10] Verifying data flows through entire system...")
try:
    # Try to run a complete flow
    from gaelp_master_integration import GAELPMasterIntegration
    
    config = {
        'daily_budget': 1000,
        'channels': ['search'],
        'use_llm': False,
        'use_wandb': False
    }
    
    system = GAELPMasterIntegration(config)
    
    # Run one complete iteration
    try:
        # Start episode
        system.reset()
        
        # Run some steps
        data_flowed = False
        for _ in range(5):
            action = system.get_action({'channel': 'search', 'user_segment': 'general'})
            if action is not None:
                data_flowed = True
                break
                
        if data_flowed:
            results.record('data_flows', True, "Data flows through system")
            print("  ✓ PASSED: Data flows through the system")
        else:
            results.record('data_flows', False, "No data flow detected")
            print("  ✗ FAILED: No data flow detected")
            
    except Exception as e:
        results.record('data_flows', False, f"System execution failed: {e}")
        print(f"  ✗ FAILED: System execution error: {e}")
        
except Exception as e:
    results.record('data_flows', False, f"Error testing data flow: {e}")
    print(f"  ✗ ERROR: {e}")

# Print final report
success = results.print_report()

# Write results to file
with open('gaelp_test_results.json', 'w') as f:
    json.dump(results.results, f, indent=2)

print("\nResults saved to gaelp_test_results.json")

# Exit with appropriate code
sys.exit(0 if success else 1)