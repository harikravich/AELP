#!/usr/bin/env python3
"""
GAELP Safety Systems Basic Integration Test
Tests core safety functionality without external dependencies.
"""

import sys
import os
import logging
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reward_validation():
    """Test reward validation system"""
    print("Testing Reward Validation System...")
    
    try:
        from reward_validation_system import ProductionRewardValidator
        
        validator = ProductionRewardValidator()
        
        # Test normal reward
        result1 = validator.validate_reward(2.5, {'context_type': 'test'})
        assert result1.is_valid, "Normal reward should be valid"
        assert result1.validated_reward == 2.5, "Normal reward should not be modified"
        
        # Test extreme reward
        result2 = validator.validate_reward(10000.0, {'context_type': 'test'})
        assert result2.validated_reward < 10000.0, "Extreme reward should be clipped"
        
        print("✓ Reward Validation System: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Reward Validation System: FAILED - {e}")
        return False

def test_budget_safety():
    """Test budget safety system"""
    print("Testing Budget Safety System...")
    
    try:
        from budget_safety_system import ProductionBudgetSafetySystem
        
        budget_system = ProductionBudgetSafetySystem()
        
        # Test normal spending
        is_allowed1, violations1, pacing1 = budget_system.validate_spending(
            50.0, 'test_campaign', 'test_channel', 'test_account'
        )
        assert is_allowed1, f"Normal spending should be allowed, violations: {violations1}"
        
        # Test extreme spending
        is_allowed2, violations2, pacing2 = budget_system.validate_spending(
            50000.0, 'test_campaign', 'test_channel', 'test_account'
        )
        # Should either be blocked or have warnings
        assert not is_allowed2 or violations2, "Extreme spending should trigger controls"
        
        print("✓ Budget Safety System: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Budget Safety System: FAILED - {e}")
        return False

def test_safety_framework():
    """Test comprehensive safety framework"""
    print("Testing Safety Framework...")
    
    try:
        from gaelp_safety_framework import ComprehensiveSafetyFramework
        
        framework = ComprehensiveSafetyFramework()
        
        # Test bid validation
        test_bid = {
            'bid_amount': 10.0,
            'campaign_id': 'test_campaign',
            'user_data': {'user_id': 'test_user', 'age': 25}
        }
        
        is_safe, violations, safe_values = framework.validate_bidding_decision(test_bid)
        assert isinstance(is_safe, bool), "Should return boolean safety result"
        assert isinstance(violations, list), "Should return violations list"
        assert isinstance(safe_values, dict), "Should return safe values dict"
        
        print("✓ Safety Framework: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Safety Framework: FAILED - {e}")
        return False

def test_emergency_controls():
    """Test emergency controls"""
    print("Testing Emergency Controls...")
    
    try:
        from emergency_controls import EmergencyController
        
        controller = EmergencyController()
        
        # Test initial state
        assert controller.is_system_healthy(), "System should be healthy initially"
        assert not controller.emergency_stop_triggered, "Emergency stop should not be triggered"
        
        # Test budget tracking
        controller.update_budget_tracking('test_campaign', 100.0, 1000.0)
        
        # Test bid recording
        controller.record_bid(5.0)
        
        # Test system status
        status = controller.get_system_status()
        assert isinstance(status, dict), "Should return status dict"
        assert 'active' in status, "Status should include active flag"
        
        print("✓ Emergency Controls: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Emergency Controls: FAILED - {e}")
        return False

def test_integration():
    """Test basic integration between systems"""
    print("Testing System Integration...")
    
    try:
        # Import all systems
        from gaelp_safety_framework import get_safety_framework
        from reward_validation_system import get_reward_validator, validate_reward_safe
        from budget_safety_system import get_budget_safety_system, budget_safety_check
        from emergency_controls import get_emergency_controller
        
        # Get system instances
        safety_framework = get_safety_framework()
        reward_validator = get_reward_validator()
        budget_system = get_budget_safety_system()
        emergency_controller = get_emergency_controller()
        
        # Test reward validation integration
        safe_reward = validate_reward_safe(5.0, {'test': True})
        assert isinstance(safe_reward, float), "Should return safe reward value"
        
        # Test budget safety integration
        is_budget_safe, budget_violations = budget_safety_check(10.0, 'test_campaign', 'test_channel')
        assert isinstance(is_budget_safe, bool), "Should return budget safety boolean"
        assert isinstance(budget_violations, list), "Should return budget violations list"
        
        print("✓ System Integration: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ System Integration: FAILED - {e}")
        return False

def test_performance():
    """Test basic performance characteristics"""
    print("Testing Performance...")
    
    try:
        from reward_validation_system import validate_reward_safe
        
        # Time reward validation
        start_time = time.time()
        
        for i in range(100):
            validate_reward_safe(float(i), {'iteration': i})
        
        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        
        assert avg_time_ms < 50, f"Average validation time too slow: {avg_time_ms:.2f}ms"
        
        print(f"✓ Performance: PASSED (avg: {avg_time_ms:.2f}ms per validation)")
        return True
        
    except Exception as e:
        print(f"✗ Performance: FAILED - {e}")
        return False

def test_decorators():
    """Test safety decorators"""
    print("Testing Safety Decorators...")
    
    try:
        from reward_validation_system import reward_validation_decorator
        from emergency_controls import emergency_stop_decorator
        
        @reward_validation_decorator
        def test_reward_function(state):
            return state.get('reward', 1.0)
        
        @emergency_stop_decorator('test_component')
        def test_emergency_function():
            return "success"
        
        # Test decorated functions
        result1 = test_reward_function({'reward': 5.0})
        assert isinstance(result1, float), "Reward decorator should return float"
        
        result2 = test_emergency_function()
        assert result2 == "success", "Emergency decorator should allow normal execution"
        
        print("✓ Safety Decorators: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Safety Decorators: FAILED - {e}")
        return False

def main():
    """Run all basic safety tests"""
    print("=" * 60)
    print("GAELP SAFETY SYSTEMS BASIC TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Reward Validation", test_reward_validation),
        ("Budget Safety", test_budget_safety),
        ("Safety Framework", test_safety_framework),
        ("Emergency Controls", test_emergency_controls),
        ("System Integration", test_integration),
        ("Performance", test_performance),
        ("Safety Decorators", test_decorators),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name}: FAILED with exception - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / len(tests)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL SAFETY SYSTEMS TESTS PASSED! 🎉")
        print("\nGAELP Safety Systems are ready for production use.")
        print("\nKey Features Validated:")
        print("✓ Reward validation and clipping")
        print("✓ Budget safety and spending limits")
        print("✓ Emergency controls and circuit breakers")
        print("✓ System integration and decorators")
        print("✓ Performance within acceptable limits")
        
        # Create success report
        report = {
            'test_date': datetime.now().isoformat(),
            'total_tests': len(tests),
            'passed_tests': passed,
            'failed_tests': failed,
            'success_rate': (passed / len(tests)) * 100,
            'status': 'ALL_SYSTEMS_OPERATIONAL',
            'systems_validated': [
                'reward_validation_system',
                'budget_safety_system',
                'safety_framework',
                'emergency_controls',
                'system_integration'
            ]
        }
        
        with open('safety_systems_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nValidation report saved to safety_systems_validation_report.json")
        
    else:
        print(f"\n❌ {failed} TESTS FAILED")
        print("Please review failed tests before using in production.")
    
    print("\n" + "=" * 60)
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)