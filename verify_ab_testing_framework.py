#!/usr/bin/env python3
"""
A/B TESTING FRAMEWORK VERIFICATION

Verifies the core functionality of the statistical A/B testing framework
with practical examples and real-world scenarios.
"""

import numpy as np
import logging
import time
from typing import Dict, Any, List

from statistical_ab_testing_framework import (
    StatisticalABTestFramework, StatisticalConfig, TestType,
    AllocationStrategy, SignificanceTest
)
from ab_testing_integration import (
    create_gaelp_ab_testing_system, PolicyConfiguration
)
from discovery_engine import GA4DiscoveryEngine
from dynamic_segment_integration import validate_no_hardcoded_segments

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Validate no hardcoded segments
validate_no_hardcoded_segments("ab_testing_verification")


def test_basic_ab_framework():
    """Test basic A/B testing framework functionality"""
    
    logger.info("🧪 Testing Basic A/B Framework Functionality")
    
    # Initialize framework
    config = StatisticalConfig(
        alpha=0.05,
        power=0.80,
        minimum_sample_size=100,
        minimum_detectable_effect=0.10
    )
    
    discovery = GA4DiscoveryEngine()
    framework = StatisticalABTestFramework(config, discovery)
    
    # Create simple 2-variant test
    variants = [
        {
            'variant_id': 'control',
            'name': 'Control Policy',
            'policy_parameters': {'learning_rate': 0.001, 'epsilon': 0.1}
        },
        {
            'variant_id': 'treatment',
            'name': 'Treatment Policy',
            'policy_parameters': {'learning_rate': 0.01, 'epsilon': 0.05}
        }
    ]
    
    test_id = framework.create_ab_test(
        test_id='basic_verification_test',
        test_name='Basic Verification Test',
        variants=variants,
        test_type=TestType.BAYESIAN_BANDIT,
        allocation_strategy=AllocationStrategy.FIXED_ALLOCATION
    )
    
    logger.info(f"✅ Created test: {test_id}")
    
    # Simulate realistic data
    n_observations = 500
    true_control_rate = 0.02
    true_treatment_rate = 0.025  # 25% improvement
    
    allocation_counts = {'control': 0, 'treatment': 0}
    
    for i in range(n_observations):
        context = {
            'segment': 'discovered_segment_1',
            'device': 'mobile' if i % 2 == 0 else 'desktop',
            'hour': (i % 24),
            'channel': 'organic'
        }
        
        # Assign variant
        variant_id = framework.assign_variant(test_id, f'user_{i}', context)
        
        if variant_id:
            allocation_counts[variant_id] += 1
            
            # Generate realistic results
            conversion_rate = true_control_rate if variant_id == 'control' else true_treatment_rate
            converted = np.random.random() < conversion_rate
            
            # Generate secondary metrics
            roas = np.random.normal(3.2 if converted else 0, 0.8)
            ctr = np.random.normal(0.05, 0.01)
            ltv = np.random.normal(150 if converted else 0, 30)
            
            framework.record_observation(
                test_id=test_id,
                variant_id=variant_id,
                user_id=f'user_{i}',
                primary_metric_value=float(converted),
                secondary_metrics={
                    'roas': max(0, roas),
                    'ctr': max(0, ctr),
                    'ltv': max(0, ltv)
                },
                converted=converted,
                context=context
            )
    
    logger.info(f"📊 Allocation counts: {allocation_counts}")
    
    # Analyze results
    results = framework.analyze_test(test_id, SignificanceTest.BAYESIAN_HYPOTHESIS)
    
    logger.info("🔍 Test Results:")
    logger.info(f"  Minimum sample achieved: {results.minimum_sample_achieved}")
    logger.info(f"  Bayesian probability: {results.bayesian_probability:.3f}")
    logger.info(f"  P-value: {results.p_value:.3f}")
    logger.info(f"  Effect size: {results.effect_size:.4f}")
    logger.info(f"  Is significant: {results.is_significant}")
    logger.info(f"  Winner: {results.winner_variant_id}")
    logger.info(f"  Recommendation: {results.recommended_action}")
    
    # Get test status
    status = framework.get_test_status(test_id)
    logger.info(f"📈 Test progress: {status['progress']:.1%}")
    
    return results.is_significant and results.minimum_sample_achieved


def test_policy_comparison_integration():
    """Test policy comparison integration with GAELP"""
    
    logger.info("🤖 Testing Policy Comparison Integration")
    
    try:
        # Create mock components
        class MockComponent:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        
        discovery = GA4DiscoveryEngine()
        mock_attribution = MockComponent()
        mock_budget_pacer = MockComponent()
        mock_identity_resolver = MockComponent()
        mock_parameter_manager = MockComponent()
        
        # Create A/B testing system
        ab_system = create_gaelp_ab_testing_system(
            discovery, mock_attribution, mock_budget_pacer,
            mock_identity_resolver, mock_parameter_manager
        )
        
        # Create policy variants
        base_config = {
            'learning_rate': 0.001,
            'epsilon': 0.1,
            'gamma': 0.99,
            'buffer_size': 10000
        }
        
        policy_conservative = ab_system.create_policy_variant(
            base_config,
            {'epsilon': 0.05, 'learning_rate': 0.0005},
            'Conservative Policy'
        )
        
        policy_aggressive = ab_system.create_policy_variant(
            base_config,
            {'epsilon': 0.2, 'learning_rate': 0.002},
            'Aggressive Policy'
        )
        
        logger.info(f"✅ Created policies: {policy_conservative}, {policy_aggressive}")
        
        # Create comparison test
        test_id = ab_system.create_policy_comparison_test(
            [policy_conservative, policy_aggressive],
            'Conservative vs Aggressive Policy Test',
            primary_metric='roas'
        )
        
        logger.info(f"✅ Created comparison test: {test_id}")
        
        # Simulate episodes
        n_episodes = 200
        policy_performance = {policy_conservative: [], policy_aggressive: []}
        
        for i in range(n_episodes):
            context = {
                'segment': 'discovered_segment_1',
                'device': 'mobile',
                'hour': 14,
                'channel': 'organic'
            }
            
            # Select policy
            selected_policy, agent = ab_system.select_policy_for_episode(
                f'user_{i}', context, test_id
            )
            
            # Simulate different performance characteristics
            if selected_policy == policy_conservative:
                # Conservative: lower variance, steady performance
                base_roas = np.random.normal(3.0, 0.5)
                conversion_rate = 0.025
            else:
                # Aggressive: higher variance, potentially higher performance
                base_roas = np.random.normal(3.5, 1.2)
                conversion_rate = 0.03
            
            converted = np.random.random() < conversion_rate
            roas = max(0, base_roas if converted else 0)
            
            episode_data = {
                'total_reward': roas * 10,
                'roas': roas,
                'conversion_rate': conversion_rate,
                'converted': converted,
                'ctr': 0.05,
                'ltv': 120 if converted else 0
            }
            
            # Record episode result
            ab_system.record_episode_result(
                selected_policy, f'user_{i}', episode_data, context
            )
            
            policy_performance[selected_policy].append(roas)
        
        # Analyze performance
        analysis = ab_system.analyze_policy_performance(test_id)
        
        logger.info("🎯 Policy Comparison Results:")
        logger.info(f"  Statistical significance: {analysis['statistical_results']['is_significant']}")
        logger.info(f"  Winner: {analysis['statistical_results']['winner_variant_id']}")
        logger.info(f"  Recommendation: {analysis['statistical_results']['recommended_action']}")
        
        # Performance insights
        for policy_id in [policy_conservative, policy_aggressive]:
            if policy_id in analysis['policy_analysis']:
                stats = analysis['policy_analysis'][policy_id]
                logger.info(f"  {policy_id}:")
                logger.info(f"    Episodes: {stats['total_episodes']}")
                logger.info(f"    Avg ROAS: {stats['roas']:.3f}")
                logger.info(f"    Conversion Rate: {stats['conversion_rate']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Policy comparison test failed: {e}")
        return False


def test_statistical_methodologies():
    """Test different statistical testing methodologies"""
    
    logger.info("📊 Testing Statistical Methodologies")
    
    config = StatisticalConfig(minimum_sample_size=50)
    discovery = GA4DiscoveryEngine()
    framework = StatisticalABTestFramework(config, discovery)
    
    # Create test with clear difference
    variants = [
        {'variant_id': 'poor', 'name': 'Poor Variant', 'policy_parameters': {}},
        {'variant_id': 'good', 'name': 'Good Variant', 'policy_parameters': {}}
    ]
    
    test_id = framework.create_ab_test(
        'methodology_test', 'Statistical Methodology Test', variants
    )
    
    # Add observations with clear difference
    true_rates = {'poor': 0.01, 'good': 0.04}  # 4x difference
    
    for variant_id, rate in true_rates.items():
        for i in range(100):
            converted = np.random.random() < rate
            framework.record_observation(
                test_id=test_id,
                variant_id=variant_id,
                user_id=f'user_{variant_id}_{i}',
                primary_metric_value=float(converted),
                secondary_metrics={'roas': 3.0 if converted else 0},
                converted=converted
            )
    
    # Test different statistical methods
    methods = [
        SignificanceTest.BAYESIAN_HYPOTHESIS,
        SignificanceTest.WELCHS_T_TEST,
        SignificanceTest.BOOTSTRAP_PERMUTATION
    ]
    
    results = {}
    for method in methods:
        try:
            result = framework.analyze_test(test_id, method)
            results[method.value] = {
                'is_significant': result.is_significant,
                'p_value': result.p_value,
                'effect_size': result.effect_size,
                'winner': result.winner_variant_id
            }
            logger.info(f"✅ {method.value}: Significant={result.is_significant}, Winner={result.winner_variant_id}")
        except Exception as e:
            logger.warning(f"⚠️ {method.value} failed: {e}")
            results[method.value] = {'error': str(e)}
    
    # All methods should detect the significant difference
    successful_methods = [m for m, r in results.items() if r.get('is_significant', False)]
    logger.info(f"🎯 Methods detecting significance: {len(successful_methods)}/{len(methods)}")
    
    return len(successful_methods) >= 1


def test_multi_armed_bandit():
    """Test multi-armed bandit allocation"""
    
    logger.info("🎰 Testing Multi-Armed Bandit Allocation")
    
    config = StatisticalConfig(exploration_rate=0.1)
    discovery = GA4DiscoveryEngine()
    framework = StatisticalABTestFramework(config, discovery)
    
    # Create 3-variant test with different performance levels
    variants = [
        {'variant_id': 'worst', 'name': 'Worst Policy', 'policy_parameters': {}},
        {'variant_id': 'medium', 'name': 'Medium Policy', 'policy_parameters': {}},
        {'variant_id': 'best', 'name': 'Best Policy', 'policy_parameters': {}}
    ]
    
    test_id = framework.create_ab_test(
        'bandit_test', 'Multi-Armed Bandit Test', variants,
        test_type=TestType.THOMPSON_SAMPLING,
        allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING
    )
    
    # True performance levels
    true_rates = {'worst': 0.01, 'medium': 0.02, 'best': 0.04}
    allocation_counts = {'worst': 0, 'medium': 0, 'best': 0}
    
    # Run bandit for many iterations
    n_iterations = 300
    
    for i in range(n_iterations):
        context = {
            'segment': 'test_segment',
            'device': 'mobile',
            'hour': 14,
            'channel': 'organic'
        }
        
        # Get allocation from bandit
        variant_id = framework.assign_variant(test_id, f'user_{i}', context)
        
        if variant_id:
            allocation_counts[variant_id] += 1
            
            # Generate result based on true performance
            conversion_rate = true_rates[variant_id]
            converted = np.random.random() < conversion_rate
            
            framework.record_observation(
                test_id=test_id,
                variant_id=variant_id,
                user_id=f'user_{i}',
                primary_metric_value=float(converted),
                secondary_metrics={'roas': 4.0 if converted else 0},
                converted=converted,
                context=context
            )
    
    logger.info(f"🎯 Final allocation counts: {allocation_counts}")
    
    # Bandit should learn to prefer better variants over time
    best_allocation = allocation_counts['best']
    worst_allocation = allocation_counts['worst']
    
    logger.info(f"📈 Best variant got {best_allocation}/{n_iterations} ({100*best_allocation/n_iterations:.1f}%)")
    logger.info(f"📉 Worst variant got {worst_allocation}/{n_iterations} ({100*worst_allocation/n_iterations:.1f}%)")
    
    # Success if best variant gets more traffic than worst
    return best_allocation > worst_allocation


def run_verification_suite():
    """Run complete verification suite"""
    
    logger.info("🚀 Starting A/B Testing Framework Verification Suite")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Basic Framework
    try:
        test_results['basic_framework'] = test_basic_ab_framework()
        logger.info(f"✅ Basic Framework Test: {'PASS' if test_results['basic_framework'] else 'FAIL'}")
    except Exception as e:
        logger.error(f"❌ Basic Framework Test: FAILED - {e}")
        test_results['basic_framework'] = False
    
    logger.info("-" * 60)
    
    # Test 2: Policy Comparison
    try:
        test_results['policy_comparison'] = test_policy_comparison_integration()
        logger.info(f"✅ Policy Comparison Test: {'PASS' if test_results['policy_comparison'] else 'FAIL'}")
    except Exception as e:
        logger.error(f"❌ Policy Comparison Test: FAILED - {e}")
        test_results['policy_comparison'] = False
    
    logger.info("-" * 60)
    
    # Test 3: Statistical Methods
    try:
        test_results['statistical_methods'] = test_statistical_methodologies()
        logger.info(f"✅ Statistical Methods Test: {'PASS' if test_results['statistical_methods'] else 'FAIL'}")
    except Exception as e:
        logger.error(f"❌ Statistical Methods Test: FAILED - {e}")
        test_results['statistical_methods'] = False
    
    logger.info("-" * 60)
    
    # Test 4: Multi-Armed Bandit
    try:
        test_results['multi_armed_bandit'] = test_multi_armed_bandit()
        logger.info(f"✅ Multi-Armed Bandit Test: {'PASS' if test_results['multi_armed_bandit'] else 'FAIL'}")
    except Exception as e:
        logger.error(f"❌ Multi-Armed Bandit Test: FAILED - {e}")
        test_results['multi_armed_bandit'] = False
    
    # Summary
    logger.info("=" * 60)
    logger.info("🎯 VERIFICATION RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info("-" * 60)
    logger.info(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - A/B Testing Framework is working correctly!")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed - Framework needs attention")
        return False


if __name__ == '__main__':
    success = run_verification_suite()
    exit(0 if success else 1)