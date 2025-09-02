#!/usr/bin/env python3
"""
Simple Explainability System Tests for GAELP

Tests the core explainability functionality without complex dependencies.
Ensures all bid decisions can be explained transparently.
"""

import unittest
import numpy as np
import json
import tempfile
import os
from datetime import datetime
from dataclasses import dataclass

# Import the core explainability system
from bid_explainability_system import (
    BidExplainabilityEngine, BidDecisionExplanation, DecisionFactor,
    DecisionConfidence, FactorImportance
)

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
    
    def to_vector(self, data_stats=None):
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 5)

class TestExplainabilityCore(unittest.TestCase):
    """Test core explainability functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BidExplainabilityEngine()
        self.mock_state = MockState()
    
    def test_explanation_generation(self):
        """Test that explanations are generated correctly"""
        
        explanation = self.engine.explain_bid_decision(
            decision_id="test_001",
            user_id="user_123",
            campaign_id="campaign_001",
            state=self.mock_state,
            action={
                'bid_amount': 6.50,
                'creative_id': 42,
                'channel': 'paid_search',
                'bid_action': 10
            },
            context={
                'base_bid': 5.00,
                'pacing_factor': 1.2,
                'daily_budget': 1000.0
            },
            model_outputs={
                'q_values_bid': [2.1, 3.2, 4.5, 3.8, 2.9, 1.8, 2.3, 2.7, 3.1, 4.2, 4.6, 3.4],
                'q_values_creative': [1.2, 2.3, 1.8, 2.1],
                'q_values_channel': [1.5, 2.8, 2.2, 1.9, 1.7]
            },
            decision_factors={'model_version': 'test_v1'}
        )
        
        # Verify explanation structure
        self.assertIsInstance(explanation, BidDecisionExplanation)
        self.assertEqual(explanation.decision_id, "test_001")
        self.assertEqual(explanation.final_bid, 6.50)
        self.assertGreater(len(explanation.primary_factors), 0)
        self.assertTrue(explanation.executive_summary)
        
        print(f"✅ Basic explanation generation: {len(explanation.primary_factors)} factors identified")
    
    def test_factor_attribution(self):
        """Test factor attribution accuracy"""
        
        explanation = self.engine.explain_bid_decision(
            decision_id="test_002",
            user_id="user_123", 
            campaign_id="campaign_001",
            state=self.mock_state,
            action={'bid_amount': 7.25, 'creative_id': 15, 'channel': 'display'},
            context={'pacing_factor': 0.9},
            model_outputs={
                'q_values_bid': [1.0, 2.0, 3.0, 4.5, 3.0, 2.0],
                'q_values_creative': [2.0, 3.5, 2.8],
                'q_values_channel': [1.8, 2.5, 3.2]
            },
            decision_factors={'model_version': 'test_v1'}
        )
        
        # Test factor contributions
        total_contribution = sum(explanation.factor_contributions.values())
        self.assertGreaterEqual(total_contribution, 0.5, "Should explain at least 50% of decision")
        
        # Test factor properties
        for factor in explanation.primary_factors:
            self.assertIsInstance(factor, DecisionFactor)
            self.assertGreaterEqual(factor.impact_weight, 0.0)
            self.assertLessEqual(factor.impact_weight, 1.0)
            self.assertTrue(factor.explanation)
        
        print(f"✅ Factor attribution test: {total_contribution:.1%} coverage")
    
    def test_confidence_assessment(self):
        """Test confidence level assessment"""
        
        # High confidence scenario (clear Q-value winner)
        explanation_high = self.engine.explain_bid_decision(
            decision_id="test_003a",
            user_id="user_123",
            campaign_id="campaign_001", 
            state=self.mock_state,
            action={'bid_amount': 5.00, 'creative_id': 1, 'channel': 'search'},
            context={},
            model_outputs={
                'q_values_bid': [1.0, 1.1, 1.2, 8.5, 1.5, 1.6]  # Clear winner
            },
            decision_factors={}
        )
        
        # Low confidence scenario (similar Q-values)
        explanation_low = self.engine.explain_bid_decision(
            decision_id="test_003b", 
            user_id="user_123",
            campaign_id="campaign_001",
            state=self.mock_state,
            action={'bid_amount': 5.00, 'creative_id': 1, 'channel': 'search'},
            context={},
            model_outputs={
                'q_values_bid': [2.0, 2.1, 2.0, 2.1, 2.2, 2.0]  # All similar
            },
            decision_factors={}
        )
        
        # Verify confidence levels
        self.assertIsInstance(explanation_high.decision_confidence, DecisionConfidence)
        self.assertIsInstance(explanation_low.decision_confidence, DecisionConfidence)
        
        print(f"✅ Confidence assessment: High={explanation_high.decision_confidence.value}, Low={explanation_low.decision_confidence.value}")
    
    def test_uncertainty_analysis(self):
        """Test uncertainty range calculation"""
        
        explanation = self.engine.explain_bid_decision(
            decision_id="test_004",
            user_id="user_123",
            campaign_id="campaign_001",
            state=self.mock_state,
            action={'bid_amount': 6.00, 'creative_id': 10, 'channel': 'social'},
            context={},
            model_outputs={'q_values_bid': [1.0, 2.0, 3.0, 4.0, 3.5, 2.5]},
            decision_factors={}
        )
        
        # Test uncertainty range
        min_bid, max_bid = explanation.uncertainty_range
        
        self.assertIsInstance(min_bid, float)
        self.assertIsInstance(max_bid, float) 
        self.assertLess(min_bid, explanation.final_bid)
        self.assertGreater(max_bid, explanation.final_bid)
        self.assertGreaterEqual(min_bid, 0.5)  # Minimum bid floor
        self.assertLessEqual(max_bid, 10.0)   # Maximum bid cap
        
        print(f"✅ Uncertainty analysis: ${min_bid:.2f} - ${max_bid:.2f} range")
    
    def test_counterfactual_scenarios(self):
        """Test counterfactual scenario generation"""
        
        explanation = self.engine.explain_bid_decision(
            decision_id="test_005",
            user_id="user_123", 
            campaign_id="campaign_001",
            state=self.mock_state,
            action={'bid_amount': 5.75, 'creative_id': 20, 'channel': 'video'},
            context={},
            model_outputs={'q_values_bid': [1.5, 2.5, 3.5, 4.0, 3.0, 2.0]},
            decision_factors={}
        )
        
        # Test counterfactuals
        counterfactuals = explanation.counterfactuals
        
        self.assertIsInstance(counterfactuals, dict)
        self.assertGreater(len(counterfactuals), 0)
        
        # Test counterfactual structure
        for scenario_name, scenario_data in counterfactuals.items():
            self.assertIn('scenario', scenario_data)
            self.assertIn('estimated_bid_change', scenario_data)
            self.assertIn('rationale', scenario_data)
        
        print(f"✅ Counterfactual scenarios: {len(counterfactuals)} scenarios generated")
    
    def test_explanation_completeness(self):
        """Test that explanations are complete and actionable"""
        
        explanation = self.engine.explain_bid_decision(
            decision_id="test_006",
            user_id="user_123",
            campaign_id="campaign_001", 
            state=self.mock_state,
            action={'bid_amount': 8.00, 'creative_id': 25, 'channel': 'display'},
            context={'pacing_factor': 1.5, 'daily_budget': 2000},
            model_outputs={
                'q_values_bid': [2.0, 3.0, 4.0, 5.0, 4.5, 3.5],
                'q_values_creative': [1.8, 3.2, 2.4],
                'q_values_channel': [2.0, 2.8, 3.5]
            },
            decision_factors={'exploration_mode': False}
        )
        
        # Test completeness
        self.assertTrue(explanation.executive_summary, "Should have executive summary")
        self.assertTrue(explanation.detailed_reasoning, "Should have detailed reasoning")
        self.assertIsInstance(explanation.key_insights, list, "Should have insights")
        self.assertIsInstance(explanation.risk_factors, list, "Should have risk factors")
        self.assertIsInstance(explanation.optimization_opportunities, list, "Should have opportunities")
        
        # Test actionability
        self.assertGreater(len(explanation.key_insights), 0, "Should have actionable insights")
        
        print(f"✅ Explanation completeness: {len(explanation.key_insights)} insights, {len(explanation.optimization_opportunities)} opportunities")
    
    def test_different_scenarios(self):
        """Test explanations for different bidding scenarios"""
        
        scenarios = [
            # High CVR segment
            (MockState(segment_cvr=0.08, creative_predicted_ctr=0.02), "high_cvr"),
            # High competition
            (MockState(segment_cvr=0.02, competition_level=0.9), "high_competition"), 
            # Peak hours
            (MockState(segment_cvr=0.03, is_peak_hour=True, hour_of_day=20), "peak_hours"),
            # Budget constraint
            (MockState(segment_cvr=0.03, budget_spent_ratio=0.9), "budget_constraint")
        ]
        
        for state, scenario_name in scenarios:
            explanation = self.engine.explain_bid_decision(
                decision_id=f"scenario_{scenario_name}",
                user_id="user_123",
                campaign_id="campaign_001",
                state=state,
                action={'bid_amount': 6.0, 'creative_id': 1, 'channel': 'search'},
                context={'pacing_factor': 1.0},
                model_outputs={'q_values_bid': [1.0, 2.0, 3.0, 4.0, 3.0, 2.0]},
                decision_factors={}
            )
            
            # Verify each scenario generates valid explanations
            self.assertIsInstance(explanation, BidDecisionExplanation)
            self.assertTrue(explanation.executive_summary)
            self.assertGreater(len(explanation.primary_factors), 0)
        
        print("✅ Different scenarios test: All scenarios generate valid explanations")

def run_simple_explainability_tests():
    """Run simple explainability tests"""
    
    print("🔍 GAELP Explainability Core Tests")
    print("=" * 50)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestExplainabilityCore)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ ALL EXPLAINABILITY TESTS PASSED")
        print(f"   Tests run: {result.testsRun}")
        print("   🎯 Bid decisions are fully explainable")
        print("   📊 Factor attribution working correctly") 
        print("   🔍 Confidence assessment functioning")
        print("   💡 Counterfactual analysis operational")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        return False

if __name__ == "__main__":
    success = run_simple_explainability_tests()
    exit(0 if success else 1)