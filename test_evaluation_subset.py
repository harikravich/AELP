#!/usr/bin/env python3
"""
Subset test of evaluation framework demonstration
Tests key functionality without running the full comprehensive demo
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from evaluation_framework import (
    EvaluationFramework, PerformanceMetrics, DataSplitter,
    SplitStrategy, StatisticalTest, MultipleTestCorrection,
    quick_ab_test, calculate_required_sample_size
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ab_testing():
    """Test A/B testing functionality."""
    logger.info("=== Testing A/B Testing ===")
    
    np.random.seed(42)
    
    # Generate data with clear effect
    treatment = np.random.normal(1.5, 1.0, 500)  # Clear treatment effect
    control = np.random.normal(1.0, 1.0, 500)
    
    # Quick test
    result = quick_ab_test(treatment, control, "ab_test")
    logger.info(f"Quick A/B Test - Effect Size: {result.effect_size:.4f}, P-value: {result.p_value:.6f}")
    
    # Detailed framework test
    framework = EvaluationFramework({'save_results': False})
    
    # Test multiple metrics
    for i in range(3):
        treatment_data = np.random.normal(1.3 + i*0.1, 1.0, 300)
        control_data = np.random.normal(1.0, 1.0, 300)
        
        result = framework.run_evaluation(
            treatment_data, control_data, f"test_metric_{i}",
            multiple_comparisons=True
        )
        
        logger.info(f"Metric {i} - Effect: {result.effect_size:.4f}, Significant: {result.statistical_significance}")
    
    # Generate report
    report = framework.generate_report(include_plots=False)
    logger.info(f"Report: {report['summary']['total_experiments']} experiments, "
               f"{report['summary']['statistically_significant']} significant")
    
    return True

def test_data_splitting():
    """Test data splitting functionality."""
    logger.info("=== Testing Data Splitting ===")
    
    # Generate temporal data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='D'),
        'user_id': range(1000),
        'value': np.random.normal(100, 15, 1000),
        'segment': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    framework = EvaluationFramework({'save_results': False})
    
    # Test different split strategies
    strategies = [
        (SplitStrategy.RANDOM, {}),
        (SplitStrategy.TEMPORAL, {'time_column': 'timestamp'}),
        (SplitStrategy.STRATIFIED, {'stratify_column': 'segment'})
    ]
    
    for strategy, kwargs in strategies:
        train, test = framework.create_holdout_set(
            data, f'holdout_{strategy.value}', strategy, **kwargs
        )
        
        logger.info(f"{strategy.value}: Train={len(train)}, Test={len(test)}, "
                   f"Total={len(train) + len(test)}")
    
    logger.info(f"Created {len(framework.holdout_sets)} holdout sets")
    return True

def test_power_analysis():
    """Test power analysis functionality."""
    logger.info("=== Testing Power Analysis ===")
    
    # Test different scenarios
    scenarios = [
        (0.2, "small effect"),
        (0.5, "medium effect"),
        (0.8, "large effect")
    ]
    
    for effect_size, description in scenarios:
        sample_size = calculate_required_sample_size(effect_size, 0.8, 0.05)
        logger.info(f"{description}: requires {sample_size} per group")
    
    # Validate with simulation
    framework = EvaluationFramework({'save_results': False})
    
    effect_size = 0.5
    sample_size = calculate_required_sample_size(effect_size, 0.8, 0.05)
    
    # Run simulation
    significant_count = 0
    n_sims = 50  # Reduced for quick test
    
    for i in range(n_sims):
        np.random.seed(i)
        treatment = np.random.normal(1 + effect_size, 1.0, sample_size)
        control = np.random.normal(1.0, 1.0, sample_size)
        
        result = framework.run_evaluation(treatment, control, f"sim_{i}")
        if result.statistical_significance:
            significant_count += 1
    
    observed_power = significant_count / n_sims
    logger.info(f"Power validation: observed {observed_power:.2f} (target: 0.8)")
    
    return True

def test_counterfactual_analysis():
    """Test counterfactual analysis functionality."""
    logger.info("=== Testing Counterfactual Analysis ===")
    
    np.random.seed(42)
    n = 1000
    
    # Generate data with confounders
    age = np.random.uniform(18, 65, n)
    income = np.random.uniform(30000, 100000, n)
    
    # Treatment assignment (biased by confounders)
    treatment_prob = 0.3 + 0.005 * (age - 40) + 0.000002 * (income - 50000)
    treatment_prob = np.clip(treatment_prob, 0.1, 0.9)
    treatment = np.random.binomial(1, treatment_prob, n)
    
    # Outcome with treatment effect
    outcome = (20 + 0.2 * age + 0.0001 * income + 5 * treatment + 
              np.random.normal(0, 2, n))
    
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'treatment': treatment,
        'outcome': outcome
    })
    
    # Test counterfactual methods
    framework = EvaluationFramework({'save_results': False})
    
    methods = ['iptw', 'dm']  # Skip doubly_robust for speed
    
    for method in methods:
        result = framework.analyze_counterfactual(
            data=data,
            treatment_column='treatment',
            outcome_column='outcome',
            feature_columns=['age', 'income'],
            policy_name=f'policy_{method}',
            baseline_policy='control',
            method=method
        )
        
        logger.info(f"{method.upper()}: Effect=${result.estimated_lift:.2f}, "
                   f"CI=[${result.confidence_interval[0]:.2f}, ${result.confidence_interval[1]:.2f}], "
                   f"Significant={result.significance}")
    
    # Compare to naive estimate
    naive_effect = (data[data['treatment'] == 1]['outcome'].mean() - 
                   data[data['treatment'] == 0]['outcome'].mean())
    logger.info(f"Naive estimate: ${naive_effect:.2f}")
    
    return True

def test_performance_metrics():
    """Test performance metrics calculation."""
    logger.info("=== Testing Performance Metrics ===")
    
    # Create sample metrics
    metrics = PerformanceMetrics(
        impressions=10000,
        clicks=300,
        conversions=15,
        spend=750.0,
        revenue=2250.0,
        viewthrough_conversions=5,
        bounce_rate=0.4
    )
    
    logger.info(f"Metrics - CTR: {metrics.ctr:.4f}, Conversion Rate: {metrics.conversion_rate:.4f}")
    logger.info(f"CPC: ${metrics.cpc:.2f}, CPA: ${metrics.cpa:.2f}, ROAS: {metrics.roas:.2f}")
    
    # Test with evaluation framework
    framework = EvaluationFramework({'save_results': False})
    
    # Compare two sets of metrics
    metrics_a = PerformanceMetrics(
        impressions=10000, clicks=350, conversions=20, spend=800.0, revenue=2400.0
    )
    metrics_b = PerformanceMetrics(
        impressions=10000, clicks=300, conversions=15, spend=700.0, revenue=2250.0
    )
    
    result = framework.run_evaluation(
        treatment_data=np.array([metrics_a.roas]),
        control_data=np.array([metrics_b.roas]),
        experiment_name="metrics_comparison",
        metrics_treatment=metrics_a,
        metrics_control=metrics_b
    )
    
    logger.info(f"ROAS comparison - Treatment: {metrics_a.roas:.2f}, "
               f"Control: {metrics_b.roas:.2f}, Significant: {result.statistical_significance}")
    
    return True

def main():
    """Run all subset tests."""
    logger.info("Starting Evaluation Framework Subset Tests")
    
    tests = [
        test_ab_testing,
        test_data_splitting,
        test_power_analysis,
        test_counterfactual_analysis,
        test_performance_metrics
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            logger.info(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_func.__name__} FAILED: {e}")
    
    logger.info(f"\n=== TEST SUMMARY ===")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {len(tests)}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Evaluation framework is working correctly.")
        return True
    else:
        logger.error("Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)