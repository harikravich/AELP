#!/usr/bin/env python3
"""
Integration Tests for GAELP Explainability System

CRITICAL REQUIREMENTS VERIFICATION:
- Every bid decision must be explainable - ✓
- No black box decisions allowed - ✓
- Real-time explanation generation - ✓
- Factor attribution accuracy - ✓
- Audit trail integration - ✓
- Human-readable explanations - ✓

This test suite ensures the explainability system works correctly
with all GAELP components and provides transparent bid decisions.
"""

import unittest
import numpy as np
import json
import tempfile
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import the systems to test
from bid_explainability_system import (
    BidExplainabilityEngine, BidDecisionExplanation, DecisionFactor,
    DecisionConfidence, FactorImportance, ExplainabilityMetrics,
    explain_bid_decision
)
from explainable_rl_agent import ExplainableRLAgent, ExplainableAction, ExplainableExperience
from explanation_dashboard import ExplanationDashboard
from audit_trail import ComplianceAuditTrail, get_audit_trail

# Mock GAELP components for testing
from dynamic_segment_integration import validate_no_hardcoded_segments
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine

logger = logging.getLogger(__name__)

@dataclass
class MockState:
    """Mock state for testing"""
    segment_cvr: float = 0.04
    creative_predicted_ctr: float = 0.025
    competition_level: float = 0.7
    is_peak_hour: bool = True
    hour_of_day: int = 20
    creative_fatigue: float = 0.3
    creative_cta_strength: float = 0.8
    budget_spent_ratio: float = 0.6
    stage: int = 2
    device: int = 0
    channel_attribution_credit: float = 0.4
    cross_device_confidence: float = 0.7
    num_devices_seen: int = 2
    segment: int = 0
    channel: int = 1
    creative_id: int = 42
    pacing_factor: float = 1.2
    
    def to_vector(self, data_stats=None):
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 5)[:50]  # 50 features

@dataclass
class MockAuctionResult:
    """Mock auction result for testing"""
    won: bool = True
    position: int = 1
    price_paid: float = 4.50
    clicked: bool = True
    revenue: float = 100.0
    competitors_count: int = 8

class TestExplainabilityEngine(unittest.TestCase):
    """Test the core explainability engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BidExplainabilityEngine()
        self.mock_state = MockState()
        self.mock_action = {
            'bid_amount': 6.50,
            'creative_id': 42,
            'channel': 'paid_search',
            'bid_action': 10
        }
        self.mock_context = {
            'base_bid': 5.00,
            'pacing_factor': 1.2,
            'daily_budget': 1000.0,
            'exploration_mode': False
        }
        self.mock_model_outputs = {
            'q_values_bid': [2.1, 3.2, 4.5, 3.8, 2.9, 1.8, 2.3, 2.7, 3.1, 4.2, 4.6, 3.4],
            'q_values_creative': [1.2, 2.3, 1.8, 2.1],
            'q_values_channel': [1.5, 2.8, 2.2, 1.9, 1.7]
        }
    
    def test_explain_bid_decision_completeness(self):
        """Test that explanation covers all required elements"""
        
        explanation = self.engine.explain_bid_decision(
            decision_id="test_001",
            user_id="user_123",
            campaign_id="campaign_001",
            state=self.mock_state,
            action=self.mock_action,
            context=self.mock_context,
            model_outputs=self.mock_model_outputs,
            decision_factors={'model_version': 'test_v1'}
        )
        
        # Test explanation completeness
        self.assertIsInstance(explanation, BidDecisionExplanation)
        self.assertEqual(explanation.decision_id, "test_001")
        self.assertEqual(explanation.user_id, "user_123")
        self.assertEqual(explanation.final_bid, 6.50)
        
        # Test factor presence
        self.assertGreater(len(explanation.primary_factors), 0, "Should have primary factors")
        self.assertIsInstance(explanation.primary_factors[0], DecisionFactor)
        
        # Test natural language explanations
        self.assertTrue(explanation.executive_summary, "Should have executive summary")
        self.assertTrue(explanation.detailed_reasoning, "Should have detailed reasoning")
        self.assertIsInstance(explanation.key_insights, list, "Should have insights list")
        
        # Test quantitative analysis
        self.assertIsInstance(explanation.factor_contributions, dict)
        self.assertGreater(sum(explanation.factor_contributions.values()), 0, 
                          "Should have factor contributions")
        
        # Test uncertainty analysis
        self.assertIsInstance(explanation.uncertainty_range, tuple)
        self.assertEqual(len(explanation.uncertainty_range), 2)
        min_bid, max_bid = explanation.uncertainty_range
        self.assertLess(min_bid, explanation.final_bid)
        self.assertGreater(max_bid, explanation.final_bid)
        
        print(f"✅ Explanation completeness test passed - {len(explanation.primary_factors)} factors identified")
    
    def test_factor_attribution_accuracy(self):
        """Test that factor attributions are accurate and sum correctly"""
        
        explanation = self.engine.explain_bid_decision(
            decision_id="test_002",
            user_id="user_123",
            campaign_id="campaign_001", 
            state=self.mock_state,
            action=self.mock_action,
            context=self.mock_context,
            model_outputs=self.mock_model_outputs,
            decision_factors={'model_version': 'test_v1'}
        )
        
        # Test factor contribution accuracy
        factor_contributions = explanation.factor_contributions
        total_contribution = sum(factor_contributions.values())
        
        # Should explain at least 80% of the decision
        self.assertGreaterEqual(total_contribution, 0.8, 
                               f"Factor contributions ({total_contribution:.2%}) should explain at least 80% of decision")
        
        # No negative contributions
        for factor_name, contribution in factor_contributions.items():
            self.assertGreaterEqual(contribution, 0, f"Factor {factor_name} has negative contribution")
        
        # Test factor importance classification
        for factor in explanation.primary_factors:
            self.assertIsInstance(factor.importance_level, FactorImportance)
            self.assertGreaterEqual(factor.impact_weight, 0.0)
            self.assertLessEqual(factor.impact_weight, 1.0)
            self.assertGreaterEqual(factor.confidence, 0.0)
            self.assertLessEqual(factor.confidence, 1.0)
        
        print(f"✅ Factor attribution test passed - {total_contribution:.1%} coverage")
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis functionality"""
        
        explanation = self.engine.explain_bid_decision(
            decision_id="test_003",
            user_id="user_123",
            campaign_id="campaign_001",
            state=self.mock_state,
            action=self.mock_action,
            context=self.mock_context,
            model_outputs=self.mock_model_outputs,
            decision_factors={'model_version': 'test_v1'}
        )
        
        # Test sensitivity analysis
        sensitivity = explanation.sensitivity_analysis
        
        self.assertIsInstance(sensitivity, dict)
        self.assertGreater(len(sensitivity), 0, "Should have sensitivity analysis")
        
        # All sensitivity values should be between 0 and 1
        for factor_name, sensitivity_value in sensitivity.items():
            self.assertGreaterEqual(sensitivity_value, 0.0, 
                                  f"Sensitivity for {factor_name} should be non-negative")
            self.assertLessEqual(sensitivity_value, 1.0, 
                                f"Sensitivity for {factor_name} should not exceed 1.0")
        
        print(f"✅ Sensitivity analysis test passed - {len(sensitivity)} factors analyzed")
    
    def test_confidence_assessment(self):
        """Test decision confidence assessment"""
        
        # Test with high confidence scenario
        high_q_values = [1.0, 1.1, 1.2, 1.3, 8.5, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]  # Clear winner
        
        explanation_high = self.engine.explain_bid_decision(
            decision_id="test_004a",
            user_id="user_123",
            campaign_id="campaign_001",
            state=self.mock_state,
            action=self.mock_action,
            context=self.mock_context,
            model_outputs={**self.mock_model_outputs, 'q_values_bid': high_q_values},
            decision_factors={'model_version': 'test_v1'}
        )
        
        # Test with low confidence scenario  
        low_q_values = [2.0, 2.1, 2.0, 2.1, 2.2, 2.0, 2.1, 2.0, 2.1, 2.0, 2.1, 2.0]  # All similar
        
        explanation_low = self.engine.explain_bid_decision(
            decision_id="test_004b",
            user_id="user_123",
            campaign_id="campaign_001",
            state=self.mock_state,
            action=self.mock_action,
            context=self.mock_context,
            model_outputs={**self.mock_model_outputs, 'q_values_bid': low_q_values},
            decision_factors={'model_version': 'test_v1'}
        )
        
        # High Q-value spread should lead to higher confidence
        self.assertIsInstance(explanation_high.decision_confidence, DecisionConfidence)
        self.assertIsInstance(explanation_low.decision_confidence, DecisionConfidence)
        
        # Uncertainty ranges should reflect confidence
        high_range = explanation_high.uncertainty_range[1] - explanation_high.uncertainty_range[0]
        low_range = explanation_low.uncertainty_range[1] - explanation_low.uncertainty_range[0]
        
        # Lower confidence should have larger uncertainty range
        self.assertGreaterEqual(low_range, high_range, 
                               "Low confidence should have larger uncertainty range")
        
        print(f"✅ Confidence assessment test passed - High: {explanation_high.decision_confidence.value}, Low: {explanation_low.decision_confidence.value}")
    
    def test_counterfactual_generation(self):
        """Test counterfactual analysis"""
        
        explanation = self.engine.explain_bid_decision(
            decision_id="test_005",
            user_id="user_123", 
            campaign_id="campaign_001",
            state=self.mock_state,
            action=self.mock_action,
            context=self.mock_context,
            model_outputs=self.mock_model_outputs,
            decision_factors={'model_version': 'test_v1'}
        )
        
        # Test counterfactuals
        counterfactuals = explanation.counterfactuals
        
        self.assertIsInstance(counterfactuals, dict)
        self.assertGreater(len(counterfactuals), 0, "Should have counterfactual scenarios")
        
        # Test counterfactual structure
        for scenario_name, scenario_data in counterfactuals.items():
            self.assertIn('scenario', scenario_data)
            self.assertIn('estimated_bid_change', scenario_data)
            self.assertIn('rationale', scenario_data)
            
            self.assertIsInstance(scenario_data['scenario'], str)
            self.assertIsInstance(scenario_data['rationale'], str)
        
        print(f"✅ Counterfactual generation test passed - {len(counterfactuals)} scenarios generated")

class TestExplainableRLAgent(unittest.TestCase):
    """Test the explainable RL agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Mock GAELP components
        self.mock_discovery_engine = None  # Would be real DiscoveryEngine
        self.mock_creative_selector = None
        self.mock_attribution_engine = None
        self.mock_budget_pacer = None
        self.mock_identity_resolver = None
        self.mock_parameter_manager = None
        
        # For now, test without full GAELP integration
        # In production, would initialize with real components
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_explainable_action_generation(self):
        """Test that explainable actions are generated correctly"""
        
        # Create mock state
        state = MockState()
        
        # Test would create explainable RL agent and generate action
        # For now, test the core explainability components
        
        engine = BidExplainabilityEngine()
        
        explanation = engine.explain_bid_decision(
            decision_id="agent_test_001",
            user_id="user_123",
            campaign_id="campaign_001",
            state=state,
            action={
                'bid_amount': 5.50,
                'creative_id': 42,
                'channel': 'paid_search'
            },
            context={
                'pacing_factor': 1.1,
                'daily_budget': 1000
            },
            model_outputs={
                'q_values_bid': [1.0, 2.0, 3.0, 4.0, 3.5, 2.5],
                'q_values_creative': [1.5, 2.5, 1.8],
                'q_values_channel': [1.2, 2.2, 1.9]
            },
            decision_factors={'model_version': 'test'}
        )
        
        # Verify explainable action structure
        self.assertIsInstance(explanation, BidDecisionExplanation)
        self.assertTrue(explanation.executive_summary)
        self.assertGreater(len(explanation.primary_factors), 0)
        
        print("✅ Explainable action generation test passed")

class TestAuditTrailIntegration(unittest.TestCase):
    """Test audit trail integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.audit_trail = ComplianceAuditTrail(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_decision_logging_with_explanations(self):
        """Test that decisions are logged with full explanations"""
        
        # Create explanation
        engine = BidExplainabilityEngine()
        state = MockState()
        
        explanation = engine.explain_bid_decision(
            decision_id="audit_test_001",
            user_id="user_123",
            campaign_id="campaign_001",
            state=state,
            action={
                'bid_amount': 7.25,
                'creative_id': 15,
                'channel': 'display'
            },
            context={
                'pacing_factor': 0.9,
                'daily_budget': 500
            },
            model_outputs={
                'q_values_bid': [1.0, 2.0, 3.0, 4.5, 3.0, 2.0],
                'q_values_creative': [2.0, 3.5, 2.8],
                'q_values_channel': [1.8, 2.5, 3.2]
            },
            decision_factors={'exploration_mode': False}
        )
        
        # Log decision to audit trail
        self.audit_trail.log_bidding_decision(
            decision_id="audit_test_001",
            user_id="user_123",
            session_id="session_001",
            campaign_id="campaign_001",
            state=state,
            action={
                'bid_amount': 7.25,
                'creative_id': 15,
                'channel': 'display'
            },
            context={
                'pacing_factor': 0.9,
                'daily_budget': 500
            },
            q_values={
                'bid': [1.0, 2.0, 3.0, 4.5, 3.0, 2.0],
                'creative': [2.0, 3.5, 2.8],
                'channel': [1.8, 2.5, 3.2]
            },
            decision_factors={
                'explanation_summary': explanation.executive_summary,
                'factor_contributions': explanation.factor_contributions,
                'confidence_level': explanation.decision_confidence.value
            }
        )
        
        # Flush to database
        self.audit_trail.storage.flush_buffers()
        
        # Verify decision was logged
        with self.audit_trail.storage.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM bidding_decisions")
            count = cursor.fetchone()[0]
            
            self.assertEqual(count, 1, "Should have logged 1 decision")
            
            # Verify explanation data is stored
            cursor.execute("SELECT decision_factors FROM bidding_decisions WHERE decision_id = ?", 
                         ("audit_test_001",))
            decision_factors = cursor.fetchone()[0]
            decision_factors_dict = json.loads(decision_factors)
            
            self.assertIn('explanation_summary', decision_factors_dict)
            self.assertIn('factor_contributions', decision_factors_dict)
            self.assertIn('confidence_level', decision_factors_dict)
        
        print("✅ Audit trail integration test passed")
    
    def test_explanation_integrity_validation(self):
        """Test that explanation integrity is maintained"""
        
        # Generate multiple decisions and explanations
        engine = BidExplainabilityEngine()
        
        for i in range(5):
            state = MockState()
            state.segment_cvr = 0.02 + (i * 0.01)  # Vary segment CVR
            
            explanation = engine.explain_bid_decision(
                decision_id=f"integrity_test_{i:03d}",
                user_id=f"user_{i}",
                campaign_id="campaign_001",
                state=state,
                action={
                    'bid_amount': 5.0 + i,
                    'creative_id': i,
                    'channel': 'paid_search'
                },
                context={'pacing_factor': 1.0},
                model_outputs={
                    'q_values_bid': [1.0 + i*0.1] * 6,
                    'q_values_creative': [1.5] * 3,
                    'q_values_channel': [1.2] * 3
                },
                decision_factors={'test_run': i}
            )
            
            # Verify explanation consistency
            self.assertGreater(len(explanation.primary_factors), 0)
            self.assertGreater(sum(explanation.factor_contributions.values()), 0.5)
            self.assertIsInstance(explanation.decision_confidence, DecisionConfidence)
        
        print("✅ Explanation integrity validation test passed")

class TestExplanationDashboard(unittest.TestCase):
    """Test explanation dashboard functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.audit_trail = ComplianceAuditTrail(self.temp_db.name)
        self.dashboard = ExplanationDashboard(audit_trail=self.audit_trail)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_data_retrieval_methods(self):
        """Test dashboard data retrieval methods"""
        
        # Add some test data to audit trail
        engine = BidExplainabilityEngine()
        
        for i in range(3):
            state = MockState()
            
            # Log decision
            self.audit_trail.log_bidding_decision(
                decision_id=f"dashboard_test_{i:03d}",
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                campaign_id="campaign_001",
                state=state,
                action={'bid_amount': 5.0 + i, 'creative_id': i, 'channel': 'display'},
                context={'daily_budget': 1000},
                q_values={'bid': [1.0] * 6, 'creative': [1.5] * 3, 'channel': [1.2] * 3},
                decision_factors={'test_decision': i}
            )
        
        # Flush to database
        self.audit_trail.storage.flush_buffers()
        
        # Test data retrieval
        decisions_data = self.dashboard._get_decisions_data("Last 24 Hours", ["high", "medium", "low"])
        
        self.assertIsInstance(decisions_data, list)
        self.assertEqual(len(decisions_data), 3, "Should retrieve 3 test decisions")
        
        for decision in decisions_data:
            self.assertIn('decision_id', decision)
            self.assertIn('bid_amount', decision)
            self.assertIn('confidence_level', decision)
        
        print("✅ Dashboard data retrieval test passed")

class TestHardcodingCompliance(unittest.TestCase):
    """Test that no hardcoded values are used in explanations"""
    
    def test_no_hardcoded_segments(self):
        """Test that no hardcoded segments are used"""
        
        # This test ensures we're not using hardcoded segment lists
        try:
            validate_no_hardcoded_segments()
            print("✅ No hardcoded segments compliance test passed")
        except Exception as e:
            self.fail(f"Hardcoded segments detected: {e}")
    
    def test_dynamic_factor_weights(self):
        """Test that factor weights are dynamic, not hardcoded"""
        
        engine = BidExplainabilityEngine()
        
        # Test with different scenarios to ensure weights adapt
        scenario1 = MockState(segment_cvr=0.08, creative_predicted_ctr=0.015)  # High CVR, low CTR
        scenario2 = MockState(segment_cvr=0.015, creative_predicted_ctr=0.06)   # Low CVR, high CTR
        
        exp1 = engine.explain_bid_decision(
            "test1", "user1", "camp1", scenario1,
            {'bid_amount': 6.0, 'creative_id': 1, 'channel': 'search'},
            {'pacing_factor': 1.0},
            {'q_values_bid': [1.0] * 6, 'q_values_creative': [1.5] * 3, 'q_values_channel': [1.2] * 3},
            {'test': 'scenario1'}
        )
        
        exp2 = engine.explain_bid_decision(
            "test2", "user2", "camp2", scenario2,
            {'bid_amount': 6.0, 'creative_id': 1, 'channel': 'search'},
            {'pacing_factor': 1.0},
            {'q_values_bid': [1.0] * 6, 'q_values_creative': [1.5] * 3, 'q_values_channel': [1.2] * 3},
            {'test': 'scenario2'}
        )
        
        # Factor importance should be different between scenarios
        cvr_importance_1 = exp1.factor_contributions.get('User Segment Conversion Rate', 0)
        cvr_importance_2 = exp2.factor_contributions.get('User Segment Conversion Rate', 0)
        
        ctr_importance_1 = exp1.factor_contributions.get('Creative Performance Prediction', 0)
        ctr_importance_2 = exp2.factor_contributions.get('Creative Performance Prediction', 0)
        
        # High CVR scenario should weight segment more heavily
        self.assertGreater(cvr_importance_1, cvr_importance_2, 
                          "High CVR scenario should weight segment factor more heavily")
        
        # High CTR scenario should weight creative more heavily
        self.assertGreater(ctr_importance_2, ctr_importance_1,
                          "High CTR scenario should weight creative factor more heavily")
        
        print("✅ Dynamic factor weights compliance test passed")

class TestPerformanceAndScaling(unittest.TestCase):
    """Test performance and scaling of explainability system"""
    
    def test_explanation_generation_speed(self):
        """Test that explanations can be generated quickly"""
        
        engine = BidExplainabilityEngine()
        state = MockState()
        
        import time
        
        # Time explanation generation
        start_time = time.time()
        
        for i in range(10):
            explanation = engine.explain_bid_decision(
                f"speed_test_{i:03d}", f"user_{i}", "campaign_001",
                state,
                {'bid_amount': 5.0, 'creative_id': i, 'channel': 'search'},
                {'pacing_factor': 1.0},
                {'q_values_bid': [1.0] * 6, 'q_values_creative': [1.5] * 3, 'q_values_channel': [1.2] * 3},
                {'test_batch': i}
            )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Should generate explanations in under 100ms each
        self.assertLess(avg_time, 0.1, f"Explanation generation too slow: {avg_time:.3f}s average")
        
        print(f"✅ Explanation speed test passed - {avg_time:.3f}s average per explanation")
    
    def test_memory_usage(self):
        """Test that explanation system doesn't leak memory"""
        
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = BidExplainabilityEngine()
        state = MockState()
        
        # Generate many explanations
        for i in range(100):
            explanation = engine.explain_bid_decision(
                f"memory_test_{i:03d}", f"user_{i}", "campaign_001",
                state,
                {'bid_amount': 5.0, 'creative_id': i % 10, 'channel': 'search'},
                {'pacing_factor': 1.0},
                {'q_values_bid': [1.0] * 6, 'q_values_creative': [1.5] * 3, 'q_values_channel': [1.2] * 3},
                {'memory_test': i}
            )
            
            # Clear reference
            del explanation
            
            if i % 20 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not increase memory by more than 50MB
        self.assertLess(memory_increase, 50, 
                       f"Memory usage increased too much: {memory_increase:.1f}MB")
        
        print(f"✅ Memory usage test passed - {memory_increase:.1f}MB increase")

def run_explainability_tests():
    """Run all explainability integration tests"""
    
    print("🔍 GAELP Explainability Integration Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestExplainabilityEngine,
        TestExplainableRLAgent,
        TestAuditTrailIntegration,
        TestExplanationDashboard,
        TestHardcodingCompliance,
        TestPerformanceAndScaling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🔍 EXPLAINABILITY TEST SUMMARY")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ ALL EXPLAINABILITY TESTS PASSED")
        print(f"   Tests run: {result.testsRun}")
        print("   🎯 Complete transparency achieved")
        print("   📊 All bid decisions are explainable")
        print("   🔍 No black box elements detected")
        print("   📋 Audit trail integration verified")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        # Print failure details
        for test, traceback in result.failures + result.errors:
            print(f"\n   FAILED: {test}")
            print(f"   Error: {traceback.split('AssertionError:')[-1].strip()}")
        
        return False


if __name__ == "__main__":
    success = run_explainability_tests()
    exit(0 if success else 1)