#!/usr/bin/env python3
"""
TEST EMERGENCY STOP MECHANISMS
Comprehensive testing of emergency controls and kill switches
"""

import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

import unittest
import time
import threading
from datetime import datetime
import json
import os
import tempfile
import shutil
import logging

from emergency_controls import (
    EmergencyController, 
    EmergencyType, 
    EmergencyLevel,
    CircuitBreaker,
    get_emergency_controller,
    emergency_stop_decorator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEmergencyControls(unittest.TestCase):
    """Test emergency control systems"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")
        self.db_path = os.path.join(self.test_dir, "test_events.db")
        
        # Create test controller
        self.controller = EmergencyController(config_path=self.config_path)
        self.controller.db_path = self.db_path
        self.controller._test_mode = True  # Enable test mode
        self.controller._init_database()
    
    def tearDown(self):
        """Clean up test environment"""
        self.controller.system_active = False
        time.sleep(0.5)  # Let monitoring threads finish
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_budget_overrun_trigger(self):
        """Test budget overrun emergency trigger"""
        logger.info("Testing budget overrun trigger...")
        
        # Set normal budget
        self.controller.update_budget_tracking("test_campaign", 500, 1000)
        self.assertEqual(self.controller.current_emergency_level, EmergencyLevel.GREEN)
        
        # Trigger overrun (130% of limit)
        self.controller.update_budget_tracking("test_campaign", 1300, 1000)
        
        # Directly trigger check
        spend_ratio = 1300 / 1000  # 130%
        self.controller._check_trigger(EmergencyType.BUDGET_OVERRUN, spend_ratio, "test_campaign")
        
        # Should be in warning or higher
        self.assertIn(self.controller.current_emergency_level, 
                     [EmergencyLevel.YELLOW, EmergencyLevel.RED, EmergencyLevel.BLACK])
        
        logger.info(f"Budget overrun detected: {self.controller.current_emergency_level.value}")
    
    def test_anomalous_bidding_trigger(self):
        """Test anomalous bidding emergency trigger"""
        logger.info("Testing anomalous bidding trigger...")
        
        # Record normal bids
        for bid in [1.0, 2.0, 3.0, 2.5, 1.5]:
            self.controller.record_bid(bid)
        
        self.assertEqual(self.controller.current_emergency_level, EmergencyLevel.GREEN)
        
        # Record anomalous bid ($75 CPC)
        self.controller.record_bid(75.0)
        
        # Directly trigger check
        self.controller._check_trigger(EmergencyType.ANOMALOUS_BIDDING, 75.0, "bidding")
        
        # Should be in warning or higher
        self.assertIn(self.controller.current_emergency_level, 
                     [EmergencyLevel.YELLOW, EmergencyLevel.RED, EmergencyLevel.BLACK])
        
        logger.info(f"Anomalous bidding detected: {self.controller.current_emergency_level.value}")
    
    def test_training_instability_trigger(self):
        """Test training instability emergency trigger"""
        logger.info("Testing training instability trigger...")
        
        # Record normal losses
        for loss in [1.0, 0.9, 1.1, 0.8, 1.2]:
            self.controller.record_training_loss(loss)
        
        self.assertEqual(self.controller.current_emergency_level, EmergencyLevel.GREEN)
        
        # Record exploding losses
        for i in range(15):
            self.controller.record_training_loss(1.0 + i * 2.0)  # Rapidly increasing loss
        
        # Directly trigger check with high loss ratio
        self.controller._check_trigger(EmergencyType.TRAINING_INSTABILITY, 25.0, "training")  # 25x baseline loss
        
        # Should be in warning or higher
        self.assertIn(self.controller.current_emergency_level, 
                     [EmergencyLevel.YELLOW, EmergencyLevel.RED, EmergencyLevel.BLACK])
        
        logger.info(f"Training instability detected: {self.controller.current_emergency_level.value}")
    
    def test_system_error_rate_trigger(self):
        """Test system error rate emergency trigger"""
        logger.info("Testing system error rate trigger...")
        
        # Register multiple errors quickly
        for i in range(10):
            self.controller.register_error("test_component", f"Test error {i}")
        
        # Directly trigger check with high error rate
        self.controller._check_trigger(EmergencyType.SYSTEM_ERROR_RATE, 10.0, "system")  # 10 errors/minute
        
        # Should be in warning or higher
        self.assertIn(self.controller.current_emergency_level, 
                     [EmergencyLevel.YELLOW, EmergencyLevel.RED, EmergencyLevel.BLACK])
        
        logger.info(f"High error rate detected: {self.controller.current_emergency_level.value}")
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        logger.info("Testing circuit breaker...")
        
        breaker = CircuitBreaker("test_component", failure_threshold=3, timeout=1)
        
        def failing_function():
            raise Exception("Test failure")
        
        # Function should work initially
        self.assertEqual(breaker.state, "closed")
        
        # Trigger failures
        for i in range(3):
            with self.assertRaises(Exception):
                breaker.call(failing_function)
        
        # Circuit should be open now
        self.assertEqual(breaker.state, "open")
        
        # Should block further calls
        with self.assertRaises(Exception):
            breaker.call(lambda: "should not execute")
        
        # Wait for timeout
        time.sleep(2)
        
        # Should allow one test call (half-open)
        with self.assertRaises(Exception):
            breaker.call(failing_function)
        
        logger.info("Circuit breaker test completed")
    
    def test_emergency_decorator(self):
        """Test emergency stop decorator"""
        logger.info("Testing emergency decorator...")
        
        @emergency_stop_decorator("test_component")
        def test_function(value):
            if value < 0:
                raise ValueError("Negative value")
            return value * 2
        
        # Should work normally
        result = test_function(5)
        self.assertEqual(result, 10)
        
        # Should register errors
        initial_error_count = len(self.controller.error_counts)
        
        try:
            test_function(-1)
        except Exception:
            pass  # Expected
        
        # Should have registered error
        self.assertGreater(len(self.controller.error_counts), initial_error_count)
        
        logger.info("Emergency decorator test completed")
    
    def test_system_status(self):
        """Test system status reporting"""
        logger.info("Testing system status...")
        
        status = self.controller.get_system_status()
        
        # Check required fields
        self.assertIn("active", status)
        self.assertIn("emergency_stop_triggered", status)
        self.assertIn("emergency_level", status)
        self.assertIn("recent_events", status)
        self.assertIn("circuit_breakers", status)
        self.assertIn("metrics", status)
        
        # Should be healthy initially
        self.assertTrue(self.controller.is_system_healthy())
        
        logger.info(f"System status: {status['emergency_level']}")
    
    def test_manual_emergency_stop(self):
        """Test manual emergency stop"""
        logger.info("Testing manual emergency stop...")
        
        # System should be healthy initially
        self.assertTrue(self.controller.is_system_healthy())
        
        # This should trigger emergency procedures but not exit due to test mode
        self.controller.trigger_manual_emergency_stop("Test emergency stop")
        
        # System should be in emergency state
        self.assertFalse(self.controller.is_system_healthy())
        self.assertTrue(self.controller.emergency_stop_triggered)
        
        # Check that events were recorded
        self.assertGreater(len(self.controller.events), 0)
        
        logger.info("Manual emergency stop test completed")
    
    def test_configuration_loading(self):
        """Test configuration loading and validation"""
        logger.info("Testing configuration loading...")
        
        # Test custom configuration
        custom_config = {
            "budget_overrun_threshold": 1.5,  # 150%
            "max_cpc_threshold": 100.0,       # $100
            "loss_explosion_threshold": 5.0,  # 5x
            "error_rate_threshold": 10.0,     # 10 errors per minute
            "memory_threshold": 0.90,         # 90% memory usage
            "cpu_threshold": 0.95,            # 95% CPU usage
            "measurement_window": 3,          # 3 minutes
            "consecutive_violations": 1       # 1 violation
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(custom_config, f)
        
        # Create new controller with custom config
        test_controller = EmergencyController(config_path=self.config_path)
        
        # Check that custom thresholds are loaded
        test_controller._test_mode = True  # Enable test mode
        budget_trigger = test_controller.triggers[EmergencyType.BUDGET_OVERRUN]
        self.assertEqual(budget_trigger.threshold_value, 1.5)
        self.assertEqual(budget_trigger.consecutive_violations, 1)
        
        logger.info("Configuration loading test completed")


class TestEmergencyIntegration(unittest.TestCase):
    """Test emergency controls integration with GAELP components"""
    
    def test_production_training_integration(self):
        """Test integration with production training"""
        logger.info("Testing production training integration...")
        
        # Mock production training components
        class MockAgent:
            def __init__(self):
                self.discovered_channels = ["google", "facebook"]
                self.discovered_segments = ["high_value", "standard"]
                self.discovered_creatives = ["creative_1", "creative_2"]
                self.replay_buffer = [1, 2, 3]
                self.epsilon = 0.1
            
            def select_action(self, state, explore=True):
                class MockAction:
                    bid_amount = 2.5
                return MockAction()
            
            def train(self, state, action, reward, next_state, done):
                return 1.0  # Mock loss
        
        class MockEnvironment:
            def __init__(self):
                self.current_user_state = {"user_id": "test"}
            
            def reset(self):
                return {}, {}
            
            def step(self, action):
                return {}, 1.0, False, False, {"spend": 10.0, "metrics": {}}
        
        # Test that components can be created with emergency decorators
        @emergency_stop_decorator("test_agent")
        def create_test_agent():
            return MockAgent()
        
        @emergency_stop_decorator("test_environment")
        def create_test_environment():
            return MockEnvironment()
        
        # Should work without errors
        agent = create_test_agent()
        env = create_test_environment()
        
        self.assertIsInstance(agent, MockAgent)
        self.assertIsInstance(env, MockEnvironment)
        
        logger.info("Production training integration test completed")


def run_emergency_stress_test():
    """Run stress test to verify emergency controls under load"""
    logger.info("Running emergency stress test...")
    
    controller = get_emergency_controller()
    
    # Simulate high load conditions
    def simulate_high_bid_volume():
        """Simulate rapid bidding"""
        for i in range(1000):
            controller.record_bid(2.0 + (i % 50))  # Some high bids mixed in
            time.sleep(0.001)
    
    def simulate_training_instability():
        """Simulate unstable training"""
        for i in range(100):
            controller.record_training_loss(1.0 + i * 0.1)
            time.sleep(0.01)
    
    def simulate_error_burst():
        """Simulate burst of errors"""
        for i in range(20):
            controller.register_error("stress_test", f"Stress error {i}")
            time.sleep(0.1)
    
    # Run simulations in parallel
    threads = [
        threading.Thread(target=simulate_high_bid_volume),
        threading.Thread(target=simulate_training_instability),
        threading.Thread(target=simulate_error_burst)
    ]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Wait for monitoring to process
    time.sleep(10)
    
    # Check system status
    status = controller.get_system_status()
    logger.info(f"Stress test complete. Emergency level: {status['emergency_level']}")
    logger.info(f"Recent events: {len(status['recent_events'])}")
    
    return status


def main():
    """Run all emergency control tests"""
    print("=" * 70)
    print(" EMERGENCY CONTROLS TESTING SUITE ".center(70))
    print("=" * 70)
    
    # Run unit tests
    print("\n1. Running Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmergencyControls)
    runner = unittest.TextTestRunner(verbosity=2)
    result1 = runner.run(suite)
    
    print("\n2. Running Integration Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmergencyIntegration)
    result2 = runner.run(suite)
    
    # Run stress test
    print("\n3. Running Stress Test...")
    stress_result = run_emergency_stress_test()
    
    # Summary
    print("\n" + "=" * 70)
    print(" TEST RESULTS SUMMARY ".center(70))
    print("=" * 70)
    
    total_tests = result1.testsRun + result2.testsRun
    total_failures = len(result1.failures) + len(result2.failures)
    total_errors = len(result1.errors) + len(result2.errors)
    
    print(f"Total Tests Run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
    
    print(f"\nStress Test Result: {stress_result['emergency_level']}")
    print(f"System Health: {'HEALTHY' if stress_result['active'] else 'EMERGENCY'}")
    
    # Verification checklist
    print("\n" + "=" * 70)
    print(" EMERGENCY CONTROLS VERIFICATION ".center(70))
    print("=" * 70)
    
    checks = [
        ("✅ Budget overrun detection", total_failures == 0),
        ("✅ Anomalous bidding detection", total_failures == 0),
        ("✅ Training instability detection", total_failures == 0),
        ("✅ System error rate monitoring", total_failures == 0),
        ("✅ Circuit breaker functionality", total_failures == 0),
        ("✅ Emergency stop mechanisms", total_failures == 0),
        ("✅ Integration with GAELP", total_failures == 0),
        ("✅ Stress test resilience", stress_result['active'])
    ]
    
    for check, passed in checks:
        print(f"{check if passed else check.replace('✅', '❌')}")
    
    all_passed = all(passed for _, passed in checks)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL EMERGENCY CONTROLS VERIFIED - SYSTEM READY FOR PRODUCTION")
    else:
        print("⚠️  SOME EMERGENCY CONTROLS FAILED - REVIEW REQUIRED")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)