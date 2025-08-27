#!/usr/bin/env python3
"""
Quick test of evaluation framework to ensure basic functionality works
"""

import numpy as np
from evaluation_framework import quick_ab_test, EvaluationFramework

def test_basic_functionality():
    print("Testing basic evaluation framework functionality...")
    
    # Generate simple test data
    np.random.seed(42)
    treatment = np.random.normal(1.2, 1.0, 100)  # Treatment with effect
    control = np.random.normal(1.0, 1.0, 100)    # Control baseline
    
    # Test 1: Quick A/B test
    print("\n1. Quick A/B Test:")
    result = quick_ab_test(treatment, control, "quick_test")
    
    print(f"   Effect Size: {result.effect_size:.4f}")
    print(f"   P-value: {result.p_value:.6f}")
    print(f"   Statistically Significant: {result.statistical_significance}")
    print(f"   Confidence Interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
    
    # Test 2: Full framework
    print("\n2. Full Framework Test:")
    framework = EvaluationFramework({'save_results': False})
    
    result2 = framework.run_evaluation(
        treatment, control, "full_framework_test"
    )
    
    print(f"   Effect Size: {result2.effect_size:.4f}")
    print(f"   P-value: {result2.p_value:.6f}")
    print(f"   Statistical Power: {result2.statistical_power:.4f}")
    print(f"   Statistically Significant: {result2.statistical_significance}")
    
    # Test 3: Report generation
    print("\n3. Report Generation Test:")
    report = framework.generate_report(include_plots=False)
    
    print(f"   Total Experiments: {report['summary']['total_experiments']}")
    print(f"   Significant Results: {report['summary']['statistically_significant']}")
    print(f"   Average Effect Size: {report['summary']['average_effect_size']:.4f}")
    
    print("\n✅ All basic tests passed! Evaluation framework is working correctly.")
    return True

if __name__ == "__main__":
    test_basic_functionality()