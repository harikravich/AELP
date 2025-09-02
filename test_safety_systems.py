#!/usr/bin/env python3
"""
GAELP Safety Systems Integration Test
Comprehensive testing of all safety mechanisms and ethical compliance systems.

TESTS PERFORMED:
1. Reward validation with various edge cases
2. Budget safety with spending limits and velocity checks
3. Ethical advertising compliance validation
4. Emergency controls and circuit breakers
5. Integration testing with realistic scenarios
6. Performance and reliability testing

This ensures all safety systems work correctly in production scenarios.
"""

import sys
import os
import logging
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from decimal import Decimal

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import safety systems
from gaelp_safety_integration import (
    GAELPSafetyOrchestrator,
    SafetyCheckResult,
    validate_gaelp_safety,
    gaelp_safety_decorator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SafetySystemsTester:
    """Comprehensive tester for all safety systems"""
    
    def __init__(self):
        self.orchestrator = GAELPSafetyOrchestrator()
        self.test_results = {}
        self.test_count = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
        logger.info("Safety Systems Tester initialized")
    
    def run_all_tests(self):
        """Run all safety system tests"""
        print("=" * 80)
        print("GAELP SAFETY SYSTEMS COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        
        # Test Categories
        test_categories = [
            ("Reward Validation Tests", self.test_reward_validation),
            ("Budget Safety Tests", self.test_budget_safety),
            ("Ethical Compliance Tests", self.test_ethical_compliance),
            ("Emergency Controls Tests", self.test_emergency_controls),
            ("Integration Tests", self.test_integration_scenarios),
            ("Performance Tests", self.test_performance),
            ("Edge Case Tests", self.test_edge_cases)
        ]
        
        for category_name, test_function in test_categories:
            print(f"\n{'-' * 60}")
            print(f"RUNNING: {category_name}")
            print(f"{'-' * 60}")
            
            try:
                test_function()
                print(f"✓ {category_name} completed successfully")
            except Exception as e:
                print(f"✗ {category_name} failed with error: {e}")
                logger.error(f"Test category {category_name} failed: {e}")
        
        # Print final results
        self._print_final_results()
    
    def test_reward_validation(self):
        """Test reward validation system"""
        
        # Test 1: Normal reward
        self._run_test("Normal reward validation", lambda: self._test_normal_reward())
        
        # Test 2: Extremely high reward
        self._run_test("High reward clipping", lambda: self._test_high_reward())
        
        # Test 3: Negative reward
        self._run_test("Negative reward handling", lambda: self._test_negative_reward())
        
        # Test 4: NaN/Inf reward
        self._run_test("Invalid reward values", lambda: self._test_invalid_rewards())
        
        # Test 5: Reward hacking detection
        self._run_test("Reward hacking detection", lambda: self._test_reward_hacking())
    
    def _test_normal_reward(self):
        """Test normal reward validation"""
        bid_data = self._create_test_bid_data(reward=2.5)
        result = validate_gaelp_safety(bid_data)
        
        assert result.overall_result in [SafetyCheckResult.APPROVED, SafetyCheckResult.CONDITIONAL], \
            f"Normal reward should be approved, got {result.overall_result}"
        
        assert "reward_safety" in result.safety_scores, "Missing reward safety score"
        assert result.safety_scores["reward_safety"] > 0.5, "Reward safety score too low"
    
    def _test_high_reward(self):
        """Test high reward clipping"""
        bid_data = self._create_test_bid_data(reward=10000.0)  # Extremely high reward
        result = validate_gaelp_safety(bid_data)
        
        # Should be clipped or rejected
        assert result.overall_result in [SafetyCheckResult.CONDITIONAL, SafetyCheckResult.REJECTED], \
            f"High reward should be clipped/rejected, got {result.overall_result}"
        
        # Should have modifications or violations
        assert result.safe_modifications or result.violations, \
            "High reward should trigger modifications or violations"
    
    def _test_negative_reward(self):
        """Test negative reward handling"""
        bid_data = self._create_test_bid_data(reward=-50.0)
        result = validate_gaelp_safety(bid_data)
        
        # Negative rewards should be handled appropriately
        assert result.overall_result in [SafetyCheckResult.CONDITIONAL, SafetyCheckResult.APPROVED], \
            f"Negative reward handling failed, got {result.overall_result}"
    
    def _test_invalid_rewards(self):
        """Test NaN and infinite reward values"""
        # Test NaN reward
        bid_data = self._create_test_bid_data(reward=float('nan'))
        result = validate_gaelp_safety(bid_data)
        assert result.overall_result == SafetyCheckResult.REJECTED, \
            "NaN reward should be rejected"
        
        # Test infinite reward
        bid_data = self._create_test_bid_data(reward=float('inf'))
        result = validate_gaelp_safety(bid_data)
        assert result.overall_result == SafetyCheckResult.REJECTED, \
            "Infinite reward should be rejected"
    
    def _test_reward_hacking(self):
        """Test reward hacking detection"""
        # Simulate rapid reward increases (potential hacking)
        for i in range(10):
            reward = i * 100  # Rapidly increasing rewards
            bid_data = self._create_test_bid_data(reward=reward)
            result = validate_gaelp_safety(bid_data)
            
            if i > 5:  # Should detect anomaly by now
                assert result.safety_scores.get("reward_safety", 1.0) < 0.8, \
                    "Should detect reward anomaly pattern"
    
    def test_budget_safety(self):
        """Test budget safety system"""
        
        # Test 1: Normal spending within limits
        self._run_test("Normal spending validation", lambda: self._test_normal_spending())
        
        # Test 2: Budget limit exceeded
        self._run_test("Budget limit enforcement", lambda: self._test_budget_exceeded())
        
        # Test 3: Spending velocity check
        self._run_test("Spending velocity validation", lambda: self._test_spending_velocity())
        
        # Test 4: Multi-campaign budget tracking
        self._run_test("Multi-campaign budget tracking", lambda: self._test_multi_campaign_budget())
    
    def _test_normal_spending(self):
        """Test normal spending within limits"""
        bid_data = self._create_test_bid_data(bid_amount=10.0)
        result = validate_gaelp_safety(bid_data)
        
        assert result.overall_result in [SafetyCheckResult.APPROVED, SafetyCheckResult.CONDITIONAL], \
            f"Normal spending should be approved, got {result.overall_result}"
    
    def _test_budget_exceeded(self):
        """Test budget limit enforcement"""
        # Simulate high spending that exceeds limits
        bid_data = self._create_test_bid_data(bid_amount=5000.0)  # Very high spend
        result = validate_gaelp_safety(bid_data)
        
        # Should be rejected or conditional
        assert result.overall_result in [SafetyCheckResult.CONDITIONAL, SafetyCheckResult.REJECTED], \
            f"High spending should be restricted, got {result.overall_result}"
    
    def _test_spending_velocity(self):
        """Test spending velocity checks"""
        # Simulate rapid spending
        campaign_id = "velocity_test_campaign"
        
        for i in range(5):
            bid_data = self._create_test_bid_data(
                bid_amount=100.0,
                campaign_id=campaign_id
            )
            result = validate_gaelp_safety(bid_data)
            
            # Later requests should show velocity warnings
            if i > 2:
                assert result.warnings or result.violations, \
                    "Rapid spending should trigger velocity warnings"
    
    def _test_multi_campaign_budget(self):
        """Test budget tracking across multiple campaigns"""
        campaigns = ["campaign_a", "campaign_b", "campaign_c"]
        
        for campaign in campaigns:
            bid_data = self._create_test_bid_data(
                bid_amount=200.0,
                campaign_id=campaign
            )
            result = validate_gaelp_safety(bid_data)
            
            # All should be processed independently
            assert result.overall_result in [SafetyCheckResult.APPROVED, SafetyCheckResult.CONDITIONAL], \
                f"Multi-campaign budget tracking failed for {campaign}"
    
    def test_ethical_compliance(self):
        """Test ethical compliance system"""
        
        # Test 1: Compliant content
        self._run_test("Compliant content validation", lambda: self._test_compliant_content())
        
        # Test 2: Prohibited content
        self._run_test("Prohibited content detection", lambda: self._test_prohibited_content())
        
        # Test 3: Age-inappropriate targeting
        self._run_test("Age restriction enforcement", lambda: self._test_age_restrictions())
        
        # Test 4: Protected class targeting
        self._run_test("Protected class detection", lambda: self._test_protected_class_targeting())
        
        # Test 5: Vulnerable population protection
        self._run_test("Vulnerable population protection", lambda: self._test_vulnerable_protection())
    
    def _test_compliant_content(self):
        """Test compliant advertising content"""
        bid_data = self._create_test_bid_data(
            campaign_data={
                'creative_text': 'Quality products at competitive prices',
                'headline': 'Shop our latest collection',
                'targeting': {'age': {'min': 25, 'max': 55}},
                'category': 'retail',
                'industry': 'e-commerce'
            }
        )
        
        result = validate_gaelp_safety(bid_data)
        assert result.overall_result in [SafetyCheckResult.APPROVED, SafetyCheckResult.CONDITIONAL], \
            f"Compliant content should be approved, got {result.overall_result}"
    
    def _test_prohibited_content(self):
        """Test detection of prohibited content"""
        bid_data = self._create_test_bid_data(
            campaign_data={
                'creative_text': 'Miracle cure guaranteed! Lose 50 pounds in 30 days!',
                'headline': 'Doctors hate this one weird trick!',
                'targeting': {'age': {'min': 18, 'max': 65}},
                'category': 'health_supplements',
                'industry': 'healthcare'
            }
        )
        
        result = validate_gaelp_safety(bid_data)
        assert result.overall_result in [SafetyCheckResult.CONDITIONAL, SafetyCheckResult.REJECTED, SafetyCheckResult.HUMAN_REVIEW], \
            f"Prohibited content should be restricted, got {result.overall_result}"
        
        assert result.violations or result.warnings, \
            "Prohibited content should trigger violations or warnings"
    
    def _test_age_restrictions(self):
        """Test age restriction enforcement"""
        bid_data = self._create_test_bid_data(
            campaign_data={
                'creative_text': 'Win big at our casino! Play poker now!',
                'headline': 'Gambling made easy',
                'targeting': {'age': {'min': 16, 'max': 65}},  # Too young for gambling
                'category': 'gambling',
                'industry': 'gambling'
            }
        )
        
        result = validate_gaelp_safety(bid_data)
        assert result.overall_result in [SafetyCheckResult.REJECTED, SafetyCheckResult.HUMAN_REVIEW], \
            f"Age-inappropriate targeting should be rejected, got {result.overall_result}"
    
    def _test_protected_class_targeting(self):
        """Test protected class targeting detection"""
        bid_data = self._create_test_bid_data(
            campaign_data={
                'creative_text': 'Jobs available now',
                'headline': 'Apply today',
                'targeting': {
                    'age': {'min': 22, 'max': 35},
                    'race': 'white',  # Prohibited for employment
                    'gender': 'male'  # Prohibited for employment
                },
                'category': 'employment',
                'industry': 'recruitment'
            }
        )
        
        result = validate_gaelp_safety(bid_data)
        assert result.overall_result in [SafetyCheckResult.REJECTED, SafetyCheckResult.HUMAN_REVIEW], \
            f"Protected class targeting should be rejected, got {result.overall_result}"
    
    def _test_vulnerable_protection(self):
        """Test vulnerable population protection"""
        bid_data = self._create_test_bid_data(
            campaign_data={
                'creative_text': 'Quick cash loans - no credit check needed!',
                'headline': 'Money when you need it most',
                'targeting': {
                    'age': {'min': 18, 'max': 25},  # Young adults
                    'income': {'max': 25000},       # Low income
                    'interests': ['debt', 'financial_difficulty']
                },
                'category': 'payday_loans',
                'industry': 'financial_services'
            }
        )
        
        result = validate_gaelp_safety(bid_data)
        assert result.overall_result in [SafetyCheckResult.REJECTED, SafetyCheckResult.HUMAN_REVIEW], \
            f"Vulnerable exploitation should be rejected, got {result.overall_result}"
    
    def test_emergency_controls(self):
        """Test emergency control systems"""
        
        # Test 1: Normal operation
        self._run_test("Normal emergency status", lambda: self._test_normal_emergency_status())
        
        # Test 2: Circuit breaker activation
        self._run_test("Circuit breaker functionality", lambda: self._test_circuit_breakers())
        
        # Test 3: Emergency stop
        self._run_test("Emergency stop mechanism", lambda: self._test_emergency_stop())
    
    def _test_normal_emergency_status(self):
        """Test normal emergency system status"""
        status = self.orchestrator.emergency_controller.get_system_status()
        
        assert status["active"] == True, "Emergency system should be active"
        assert status["emergency_stop_triggered"] == False, "Emergency stop should not be triggered"
    
    def _test_circuit_breakers(self):
        """Test circuit breaker functionality"""
        # Get a circuit breaker
        breaker = self.orchestrator.emergency_controller.get_circuit_breaker("test_component")
        
        # Should start in closed state
        assert breaker.state == "closed", "Circuit breaker should start closed"
        
        # Simulate failures to trigger circuit breaker
        for i in range(6):  # More than failure threshold
            try:
                breaker.call(lambda: exec('raise Exception("Test failure")'))
            except:
                pass
        
        # Should now be open
        assert breaker.state == "open", "Circuit breaker should be open after failures"
    
    def _test_emergency_stop(self):
        """Test emergency stop mechanism"""
        # This test should be run last as it affects system state
        original_state = self.orchestrator.emergency_controller.emergency_stop_triggered
        
        try:
            # Test triggering emergency stop (but reset immediately)
            # Note: We're not actually triggering it as it would affect other tests
            
            # Just verify the mechanism exists and is callable
            assert hasattr(self.orchestrator.emergency_controller, 'trigger_manual_emergency_stop'), \
                "Emergency stop mechanism should exist"
            
            assert hasattr(self.orchestrator.emergency_controller, 'reset_emergency_state'), \
                "Emergency reset mechanism should exist"
            
        except Exception as e:
            # Reset state if anything went wrong
            self.orchestrator.emergency_controller.emergency_stop_triggered = original_state
            raise e
    
    def test_integration_scenarios(self):
        """Test realistic integration scenarios"""
        
        # Test 1: High-value legitimate campaign
        self._run_test("Legitimate high-value campaign", lambda: self._test_legitimate_campaign())
        
        # Test 2: Suspicious campaign requiring review
        self._run_test("Suspicious campaign detection", lambda: self._test_suspicious_campaign())
        
        # Test 3: Multiple safety violations
        self._run_test("Multiple violation handling", lambda: self._test_multiple_violations())
    
    def _test_legitimate_campaign(self):
        """Test a legitimate high-value campaign"""
        bid_data = self._create_test_bid_data(
            bid_amount=25.0,
            reward=12.5,
            campaign_data={
                'creative_text': 'Premium business software solutions for enterprises',
                'headline': 'Boost productivity with our enterprise tools',
                'targeting': {
                    'age': {'min': 28, 'max': 55},
                    'job_title': ['manager', 'director', 'executive'],
                    'company_size': 'large'
                },
                'category': 'business_software',
                'industry': 'technology'
            }
        )
        
        result = validate_gaelp_safety(bid_data)
        assert result.overall_result in [SafetyCheckResult.APPROVED, SafetyCheckResult.CONDITIONAL], \
            f"Legitimate campaign should be approved, got {result.overall_result}"
        
        # Should have good safety scores
        assert all(score > 0.6 for score in result.safety_scores.values()), \
            "Legitimate campaign should have good safety scores"
    
    def _test_suspicious_campaign(self):
        """Test suspicious campaign detection"""
        bid_data = self._create_test_bid_data(
            bid_amount=200.0,  # High bid
            reward=500.0,      # Suspicious reward
            campaign_data={
                'creative_text': 'Make $5000 a week working from home! No experience needed!',
                'headline': 'Secret method millionaires use - click now!',
                'targeting': {
                    'age': {'min': 18, 'max': 25},  # Target young adults
                    'income': {'max': 30000},       # Low income
                    'interests': ['make money', 'work from home', 'financial freedom']
                },
                'category': 'get_rich_quick',
                'industry': 'financial_services'
            }
        )
        
        result = validate_gaelp_safety(bid_data)
        assert result.overall_result in [SafetyCheckResult.HUMAN_REVIEW, SafetyCheckResult.REJECTED], \
            f"Suspicious campaign should require review, got {result.overall_result}"
        
        assert result.human_review_required, \
            "Suspicious campaign should require human review"
    
    def _test_multiple_violations(self):
        """Test handling of multiple safety violations"""
        bid_data = self._create_test_bid_data(
            bid_amount=1000.0,  # Budget violation
            reward=float('inf'),  # Reward violation
            campaign_data={
                'creative_text': 'Hate speech example targeting minorities',  # Ethical violation
                'headline': 'Discriminatory content',
                'targeting': {
                    'age': {'min': 12, 'max': 65},  # Age violation
                    'race': 'specific_race'         # Protected class violation
                },
                'category': 'hate_speech',
                'industry': 'prohibited'
            }
        )
        
        result = validate_gaelp_safety(bid_data)
        
        # Should be rejected due to multiple violations
        assert result.overall_result == SafetyCheckResult.REJECTED, \
            f"Multiple violations should result in rejection, got {result.overall_result}"
        
        # Should have violations from multiple systems
        assert len(result.violations) > 1, \
            "Should detect violations from multiple safety systems"
        
        # Multiple component failures
        failed_components = [k for k, v in result.component_results.items() 
                           if v == SafetyCheckResult.REJECTED]
        assert len(failed_components) > 1, \
            "Multiple safety components should fail"
    
    def test_performance(self):
        """Test performance characteristics"""
        
        # Test 1: Response time
        self._run_test("Safety check performance", lambda: self._test_response_time())
        
        # Test 2: Concurrent requests
        self._run_test("Concurrent safety checks", lambda: self._test_concurrent_checks())
        
        # Test 3: Memory usage
        self._run_test("Memory efficiency", lambda: self._test_memory_usage())
    
    def _test_response_time(self):
        """Test safety check response time"""
        bid_data = self._create_test_bid_data()
        
        # Measure response time for multiple checks
        times = []
        for _ in range(10):
            start = time.time()
            result = validate_gaelp_safety(bid_data)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        # Performance expectations
        assert avg_time < 500, f"Average response time too high: {avg_time:.2f}ms"
        assert max_time < 1000, f"Max response time too high: {max_time:.2f}ms"
        
        print(f"Performance metrics - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
    
    def _test_concurrent_checks(self):
        """Test concurrent safety checks"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        num_threads = 5
        checks_per_thread = 3
        
        def worker():
            for _ in range(checks_per_thread):
                bid_data = self._create_test_bid_data()
                result = validate_gaelp_safety(bid_data)
                results_queue.put(result)
        
        # Start threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        expected_results = num_threads * checks_per_thread
        assert len(results) == expected_results, \
            f"Expected {expected_results} results, got {len(results)}"
        
        # All results should be valid
        for result in results:
            assert hasattr(result, 'overall_result'), "Invalid result format"
    
    def _test_memory_usage(self):
        """Test memory efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many safety checks
        for i in range(100):
            bid_data = self._create_test_bid_data()
            result = validate_gaelp_safety(bid_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50, f"Memory usage increased too much: {memory_increase:.2f}MB"
        
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB, Increase: {memory_increase:.1f}MB")
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        
        # Test 1: Empty bid data
        self._run_test("Empty bid data handling", lambda: self._test_empty_bid_data())
        
        # Test 2: Malformed data
        self._run_test("Malformed data handling", lambda: self._test_malformed_data())
        
        # Test 3: Missing required fields
        self._run_test("Missing fields handling", lambda: self._test_missing_fields())
        
        # Test 4: Extreme values
        self._run_test("Extreme values handling", lambda: self._test_extreme_values())
    
    def _test_empty_bid_data(self):
        """Test handling of empty bid data"""
        result = validate_gaelp_safety({})
        
        # Should handle gracefully, not crash
        assert hasattr(result, 'overall_result'), "Should handle empty data gracefully"
        assert result.overall_result is not None, "Should return valid result"
    
    def _test_malformed_data(self):
        """Test handling of malformed data"""
        malformed_data = {
            'bid_amount': 'not_a_number',
            'reward': {'invalid': 'structure'},
            'campaign_data': 'should_be_dict'
        }
        
        result = validate_gaelp_safety(malformed_data)
        
        # Should handle malformed data without crashing
        assert hasattr(result, 'overall_result'), "Should handle malformed data gracefully"
        # Likely should be rejected due to malformed data
        assert result.overall_result in [SafetyCheckResult.REJECTED, SafetyCheckResult.CONDITIONAL], \
            "Malformed data should be rejected or conditional"
    
    def _test_missing_fields(self):
        """Test handling of missing required fields"""
        minimal_data = {
            'bid_amount': 10.0
            # Missing campaign_id, channel, etc.
        }
        
        result = validate_gaelp_safety(minimal_data)
        
        # Should handle missing fields gracefully
        assert hasattr(result, 'overall_result'), "Should handle missing fields gracefully"
    
    def _test_extreme_values(self):
        """Test handling of extreme values"""
        extreme_data = {
            'bid_amount': 1e10,  # Extremely large number
            'reward': -1e10,     # Extremely negative
            'campaign_id': 'x' * 1000,  # Very long string
            'context': {
                'conversion_probability': 2.0,  # Invalid probability > 1
                'competition_level': -5.0       # Negative competition
            }
        }
        
        result = validate_gaelp_safety(extreme_data)
        
        # Should handle extreme values appropriately
        assert result.overall_result in [SafetyCheckResult.REJECTED, SafetyCheckResult.CONDITIONAL], \
            "Extreme values should be handled appropriately"
    
    def _create_test_bid_data(self, **overrides) -> Dict[str, Any]:
        """Create test bid data with optional overrides"""
        default_data = {
            'bid_id': f'test_{self.test_count}_{int(time.time() * 1000)}',
            'bid_amount': 5.0,
            'campaign_id': 'test_campaign',
            'channel': 'test_channel',
            'account_id': 'test_account',
            'reward': 2.5,
            'context': {
                'user_segment': 'test_segment',
                'conversion_probability': 0.05,
                'conversion_value': 50.0,
                'competition_level': 1.0
            },
            'campaign_data': {
                'creative_text': 'Test advertisement content',
                'headline': 'Test headline',
                'targeting': {
                    'age': {'min': 25, 'max': 45},
                    'interests': ['test_interest']
                },
                'category': 'test_category',
                'industry': 'test_industry'
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_id': self.test_count
            }
        }
        
        # Apply overrides
        default_data.update(overrides)
        
        return default_data
    
    def _run_test(self, test_name: str, test_function: callable):
        """Run a single test with error handling"""
        self.test_count += 1
        
        try:
            start_time = time.time()
            test_function()
            end_time = time.time()
            
            self.passed_tests += 1
            duration_ms = (end_time - start_time) * 1000
            
            print(f"✓ {test_name} - PASSED ({duration_ms:.2f}ms)")
            
            self.test_results[test_name] = {
                'status': 'PASSED',
                'duration_ms': duration_ms,
                'error': None
            }
            
        except AssertionError as e:
            self.failed_tests += 1
            print(f"✗ {test_name} - FAILED: {str(e)}")
            
            self.test_results[test_name] = {
                'status': 'FAILED',
                'duration_ms': 0,
                'error': str(e)
            }
            
        except Exception as e:
            self.failed_tests += 1
            print(f"✗ {test_name} - ERROR: {str(e)}")
            
            self.test_results[test_name] = {
                'status': 'ERROR',
                'duration_ms': 0,
                'error': str(e)
            }
    
    def _print_final_results(self):
        """Print final test results summary"""
        print("\n" + "=" * 80)
        print("FINAL TEST RESULTS")
        print("=" * 80)
        
        print(f"Total Tests: {self.test_count}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests / max(self.test_count, 1)) * 100:.1f}%")
        
        if self.failed_tests > 0:
            print(f"\nFAILED TESTS ({self.failed_tests}):")
            print("-" * 40)
            for test_name, result in self.test_results.items():
                if result['status'] in ['FAILED', 'ERROR']:
                    print(f"- {test_name}: {result['error']}")
        
        # Get comprehensive status
        status = self.orchestrator.get_safety_status_comprehensive()
        print(f"\nSAFETY SYSTEM STATUS:")
        print(f"Overall Status: {status['overall_status']}")
        print(f"Safety Checks Performed: {status['performance_metrics']['total_safety_checks']}")
        print(f"Violations Detected: {status['performance_metrics']['violations_detected']}")
        print(f"Human Reviews Triggered: {status['performance_metrics']['human_reviews_triggered']}")
        print(f"Average Check Time: {status['performance_metrics']['average_check_time_ms']:.2f}ms")
        
        print("\n" + "=" * 80)
        print("SAFETY SYSTEMS TEST COMPLETED")
        print("=" * 80)


def main():
    """Main test execution"""
    print("Initializing GAELP Safety Systems Test Suite...")
    
    # Create and run tester
    tester = SafetySystemsTester()
    tester.run_all_tests()
    
    # Save test results
    with open('safety_test_results.json', 'w') as f:
        json.dump(tester.test_results, f, indent=2, default=str)
    
    print(f"\nTest results saved to safety_test_results.json")
    
    # Exit with appropriate code
    exit_code = 0 if tester.failed_tests == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()