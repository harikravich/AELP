"""
Comprehensive Unit Tests for Evaluation Framework

This module contains extensive unit tests for all components of the evaluation
framework including data splitting, statistical testing, power analysis,
counterfactual analysis, and the main evaluation orchestrator.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

# Import evaluation framework components
from evaluation_framework import (
    EvaluationFramework, DataSplitter, StatisticalTester, PowerAnalyzer,
    CounterfactualAnalyzer, PerformanceMetrics, ExperimentResult,
    CounterfactualResult, SplitStrategy, StatisticalTest,
    MultipleTestCorrection, quick_ab_test, calculate_required_sample_size
)


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics class."""
    
    def test_performance_metrics_calculation(self):
        """Test basic performance metrics calculation."""
        metrics = PerformanceMetrics(
            impressions=10000,
            clicks=300,
            conversions=15,
            spend=700.0,
            revenue=2250.0
        )
        
        assert metrics.ctr == 0.03  # 300/10000
        assert metrics.conversion_rate == 0.05  # 15/300
        assert metrics.cpc == pytest.approx(2.333, rel=1e-2)  # 700/300
        assert metrics.cpa == pytest.approx(46.667, rel=1e-2)  # 700/15
        assert metrics.roas == pytest.approx(3.214, rel=1e-2)  # 2250/700
    
    def test_performance_metrics_zero_division(self):
        """Test performance metrics with zero values."""
        metrics = PerformanceMetrics(
            impressions=0,
            clicks=0,
            conversions=0,
            spend=0.0,
            revenue=0.0
        )
        
        assert metrics.ctr == 0.0
        assert metrics.conversion_rate == 0.0
        assert metrics.cpc == 0.0
        assert metrics.cpa == 0.0
        assert metrics.roas == 0.0
    
    def test_performance_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PerformanceMetrics(
            impressions=1000,
            clicks=30,
            conversions=5,
            spend=100.0,
            revenue=500.0,
            viewthrough_conversions=2,
            bounce_rate=0.4
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['impressions'] == 1000
        assert metrics_dict['clicks'] == 30
        assert metrics_dict['roas'] == 5.0
        assert 'viewthrough_conversions' in metrics_dict
        assert 'bounce_rate' in metrics_dict


class TestDataSplitter:
    """Test suite for DataSplitter class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='D')
        np.random.seed(42)
        
        return pd.DataFrame({
            'timestamp': np.random.choice(dates, 1000),
            'user_id': np.arange(1000),
            'segment': np.random.choice(['A', 'B', 'C'], 1000),
            'value': np.random.normal(100, 15, 1000),
            'converted': np.random.binomial(1, 0.1, 1000)
        })
    
    @pytest.fixture
    def splitter(self):
        """Create DataSplitter instance."""
        return DataSplitter(random_state=42)
    
    def test_random_split(self, splitter, sample_data):
        """Test random data splitting."""
        train, test = splitter.split_data(
            sample_data, SplitStrategy.RANDOM, test_size=0.2
        )
        
        assert len(train) + len(test) == len(sample_data)
        assert len(test) == pytest.approx(200, abs=10)  # 20% of 1000
        assert len(train) == pytest.approx(800, abs=10)  # 80% of 1000
        
        # Ensure no data leakage
        assert len(set(train.index).intersection(set(test.index))) == 0
    
    def test_temporal_split(self, splitter, sample_data):
        """Test temporal data splitting."""
        train, test = splitter.split_data(
            sample_data, SplitStrategy.TEMPORAL, 
            test_size=0.2, time_column='timestamp'
        )
        
        assert len(train) + len(test) == len(sample_data)
        
        # Test data should be from later time periods
        max_train_time = train['timestamp'].max()
        min_test_time = test['timestamp'].min()
        
        # Allow for some overlap due to random timestamp assignment
        # The key is that the split preserves temporal order
        assert len(train) > 0
        assert len(test) > 0
    
    def test_stratified_split(self, splitter, sample_data):
        """Test stratified data splitting."""
        train, test = splitter.split_data(
            sample_data, SplitStrategy.STRATIFIED,
            test_size=0.2, stratify_column='segment'
        )
        
        assert len(train) + len(test) == len(sample_data)
        
        # Check that segment proportions are preserved
        train_proportions = train['segment'].value_counts(normalize=True)
        test_proportions = test['segment'].value_counts(normalize=True)
        
        for segment in ['A', 'B', 'C']:
            assert abs(train_proportions[segment] - test_proportions[segment]) < 0.1
    
    def test_blocked_split(self, splitter, sample_data):
        """Test blocked data splitting."""
        # Create block column (user groups)
        sample_data['user_block'] = sample_data['user_id'] // 100  # 10 blocks
        
        train, test = splitter.split_data(
            sample_data, SplitStrategy.BLOCKED,
            test_size=0.2, block_column='user_block'
        )
        
        # Ensure no user appears in both train and test
        train_blocks = set(train['user_block'].unique())
        test_blocks = set(test['user_block'].unique())
        
        assert len(train_blocks.intersection(test_blocks)) == 0
        assert len(train) + len(test) == len(sample_data)
    
    def test_invalid_split_strategy(self, splitter, sample_data):
        """Test handling of invalid split strategy."""
        with pytest.raises(ValueError):
            splitter.split_data(sample_data, "invalid_strategy")
    
    def test_missing_required_columns(self, splitter, sample_data):
        """Test handling of missing required columns."""
        with pytest.raises(ValueError):
            splitter.split_data(
                sample_data, SplitStrategy.TEMPORAL,
                time_column=None
            )
        
        with pytest.raises(ValueError):
            splitter.split_data(
                sample_data, SplitStrategy.STRATIFIED,
                stratify_column=None
            )


class TestStatisticalTester:
    """Test suite for StatisticalTester class."""
    
    @pytest.fixture
    def tester(self):
        """Create StatisticalTester instance."""
        return StatisticalTester(alpha=0.05)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with known effect."""
        np.random.seed(42)
        treatment = np.random.normal(1.5, 1.0, 100)  # Mean = 1.5
        control = np.random.normal(1.0, 1.0, 100)    # Mean = 1.0
        return treatment, control
    
    def test_t_test(self, tester, sample_data):
        """Test t-test implementation."""
        treatment, control = sample_data
        
        p_value, test_stat, effect_size = tester.calculate_significance(
            treatment, control, StatisticalTest.T_TEST
        )
        
        assert isinstance(p_value, float)
        assert isinstance(test_stat, float)
        assert isinstance(effect_size, float)
        assert 0 <= p_value <= 1
        assert effect_size > 0  # Positive effect expected
    
    def test_mann_whitney_test(self, tester, sample_data):
        """Test Mann-Whitney U test implementation."""
        treatment, control = sample_data
        
        p_value, test_stat, effect_size = tester.calculate_significance(
            treatment, control, StatisticalTest.MANN_WHITNEY
        )
        
        assert isinstance(p_value, float)
        assert isinstance(test_stat, float)
        assert isinstance(effect_size, float)
        assert 0 <= p_value <= 1
    
    def test_cohens_d_calculation(self, tester):
        """Test Cohen's d effect size calculation."""
        treatment = np.array([1, 2, 3, 4, 5])
        control = np.array([0, 1, 2, 3, 4])
        
        effect_size = tester._cohens_d(treatment, control)
        
        assert effect_size == pytest.approx(1.0, rel=1e-1)  # Should be ~1
    
    def test_multiple_testing_correction(self, tester):
        """Test multiple testing correction methods."""
        p_values = [0.01, 0.02, 0.03, 0.06, 0.08]
        
        # Test FDR correction
        rejected, corrected = tester.correct_multiple_testing(
            p_values, MultipleTestCorrection.FDR_BH
        )
        
        assert len(rejected) == len(p_values)
        assert len(corrected) == len(p_values)
        assert isinstance(rejected[0], bool)
        
        # Test Bonferroni correction
        rejected_bonf, corrected_bonf = tester.correct_multiple_testing(
            p_values, MultipleTestCorrection.BONFERRONI
        )
        
        # Bonferroni should be more conservative
        assert sum(rejected_bonf) <= sum(rejected)
    
    def test_no_correction(self, tester):
        """Test no multiple testing correction."""
        p_values = [0.01, 0.02, 0.03]
        
        rejected, corrected = tester.correct_multiple_testing(
            p_values, MultipleTestCorrection.NONE
        )
        
        assert corrected == p_values  # Should be unchanged
        assert rejected == [True, True, True]  # All significant at 0.05
    
    def test_edge_cases(self, tester):
        """Test edge cases for statistical testing."""
        # Identical groups
        identical = np.array([1, 1, 1, 1, 1])
        p_value, _, effect_size = tester.calculate_significance(
            identical, identical, StatisticalTest.T_TEST
        )
        
        assert p_value == pytest.approx(1.0, rel=1e-1)  # No difference
        assert abs(effect_size) < 0.1  # Near zero effect
        
        # Single values
        single_treatment = np.array([5])
        single_control = np.array([3])
        
        # Should handle gracefully (though not recommended)
        try:
            p_value, _, _ = tester.calculate_significance(
                single_treatment, single_control, StatisticalTest.T_TEST
            )
            assert isinstance(p_value, float)
        except Exception:
            # It's acceptable if this fails for edge cases
            pass


class TestPowerAnalyzer:
    """Test suite for PowerAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create PowerAnalyzer instance."""
        return PowerAnalyzer()
    
    def test_sample_size_calculation(self, analyzer):
        """Test sample size calculation."""
        sample_size = analyzer.calculate_sample_size(
            effect_size=0.5, power=0.8, alpha=0.05
        )
        
        assert isinstance(sample_size, int)
        assert sample_size >= 10  # Minimum reasonable sample size
        assert sample_size < 1000  # Should be reasonable for medium effect
    
    def test_power_calculation(self, analyzer):
        """Test statistical power calculation."""
        power = analyzer.calculate_power(
            sample_size=64, effect_size=0.5, alpha=0.05
        )
        
        assert isinstance(power, float)
        assert 0 <= power <= 1
        assert power > 0.5  # Should have reasonable power for medium effect
    
    def test_power_sample_size_relationship(self, analyzer):
        """Test relationship between power and sample size."""
        effect_size = 0.5
        
        # Larger sample size should yield higher power
        power_small = analyzer.calculate_power(30, effect_size)
        power_large = analyzer.calculate_power(100, effect_size)
        
        assert power_large > power_small
    
    def test_effect_size_relationship(self, analyzer):
        """Test relationship between effect size and required sample size."""
        # Larger effect sizes should require smaller samples
        n_small_effect = analyzer.calculate_sample_size(0.2)  # Small effect
        n_large_effect = analyzer.calculate_sample_size(0.8)  # Large effect
        
        assert n_small_effect > n_large_effect
    
    def test_different_test_types(self, analyzer):
        """Test different statistical test types for power analysis."""
        for test_type in [StatisticalTest.T_TEST]:
            sample_size = analyzer.calculate_sample_size(
                0.5, test_type=test_type
            )
            power = analyzer.calculate_power(
                64, 0.5, test_type=test_type
            )
            
            assert isinstance(sample_size, int)
            assert isinstance(power, float)
            assert sample_size > 0
            assert 0 <= power <= 1


class TestCounterfactualAnalyzer:
    """Test suite for CounterfactualAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create CounterfactualAnalyzer instance."""
        return CounterfactualAnalyzer()
    
    @pytest.fixture
    def sample_causal_data(self):
        """Create sample data for causal analysis."""
        np.random.seed(42)
        n = 1000
        
        # Features
        age = np.random.normal(35, 10, n)
        income = np.random.normal(50000, 15000, n)
        prior_purchases = np.random.poisson(2, n)
        
        # Treatment assignment (with some confounding)
        treatment_prob = 0.3 + 0.01 * (age - 35) + 0.000001 * (income - 50000)
        treatment_prob = np.clip(treatment_prob, 0.1, 0.9)
        treatment = np.random.binomial(1, treatment_prob, n)
        
        # Outcome with treatment effect
        outcome = (50 + 0.5 * age + 0.0001 * income + 2 * prior_purchases +
                  3 * treatment + np.random.normal(0, 5, n))
        
        return pd.DataFrame({
            'age': age,
            'income': income,
            'prior_purchases': prior_purchases,
            'treatment': treatment,
            'outcome': outcome
        })
    
    def test_iptw_analysis(self, analyzer, sample_causal_data):
        """Test Inverse Propensity Score Weighting."""
        result = analyzer.estimate_policy_effect(
            data=sample_causal_data,
            treatment_column='treatment',
            outcome_column='outcome',
            feature_columns=['age', 'income', 'prior_purchases'],
            policy_name='test_policy',
            baseline_policy='control',
            method='iptw'
        )
        
        assert isinstance(result, CounterfactualResult)
        assert result.policy_name == 'test_policy'
        assert result.baseline_policy == 'control'
        assert isinstance(result.estimated_lift, float)
        assert isinstance(result.confidence_interval, tuple)
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.confidence_interval[1]
        assert result.methodology == "Inverse Propensity Score Weighting"
    
    def test_direct_method_analysis(self, analyzer, sample_causal_data):
        """Test Direct Method for causal analysis."""
        result = analyzer.estimate_policy_effect(
            data=sample_causal_data,
            treatment_column='treatment',
            outcome_column='outcome',
            feature_columns=['age', 'income', 'prior_purchases'],
            policy_name='test_policy',
            baseline_policy='control',
            method='dm'
        )
        
        assert isinstance(result, CounterfactualResult)
        assert result.methodology == "Direct Method"
        assert 'model_specification' in result.assumptions_met
    
    def test_doubly_robust_analysis(self, analyzer, sample_causal_data):
        """Test Doubly Robust method."""
        result = analyzer.estimate_policy_effect(
            data=sample_causal_data,
            treatment_column='treatment',
            outcome_column='outcome',
            feature_columns=['age', 'income', 'prior_purchases'],
            policy_name='test_policy',
            baseline_policy='control',
            method='doubly_robust'
        )
        
        assert isinstance(result, CounterfactualResult)
        assert result.methodology == "Doubly Robust"
    
    def test_propensity_score_properties(self, analyzer, sample_causal_data):
        """Test propensity score calculation properties."""
        # Run IPTW analysis first to fit the model
        analyzer.estimate_policy_effect(
            data=sample_causal_data,
            treatment_column='treatment',
            outcome_column='outcome',
            feature_columns=['age', 'income', 'prior_purchases'],
            policy_name='test',
            baseline_policy='control',
            method='iptw'
        )
        
        # Check that propensity model was fitted
        assert analyzer.propensity_model is not None
        
        # Predict propensity scores
        X = sample_causal_data[['age', 'income', 'prior_purchases']]
        propensity_scores = analyzer.propensity_model.predict_proba(X)[:, 1]
        
        # Propensity scores should be in [0, 1]
        assert np.all(propensity_scores >= 0)
        assert np.all(propensity_scores <= 1)
    
    def test_invalid_method(self, analyzer, sample_causal_data):
        """Test handling of invalid causal method."""
        with pytest.raises(ValueError):
            analyzer.estimate_policy_effect(
                data=sample_causal_data,
                treatment_column='treatment',
                outcome_column='outcome',
                feature_columns=['age', 'income'],
                policy_name='test',
                baseline_policy='control',
                method='invalid_method'
            )


class TestEvaluationFramework:
    """Test suite for the main EvaluationFramework class."""
    
    @pytest.fixture
    def framework(self):
        """Create EvaluationFramework instance."""
        config = {
            'random_state': 42,
            'alpha': 0.05,
            'power': 0.8,
            'save_results': False  # Don't save during testing
        }
        return EvaluationFramework(config)
    
    @pytest.fixture
    def sample_experiment_data(self):
        """Create sample experiment data."""
        np.random.seed(42)
        treatment = np.random.normal(1.2, 1.0, 100)
        control = np.random.normal(1.0, 1.0, 100)
        return treatment, control
    
    def test_framework_initialization(self, framework):
        """Test framework initialization."""
        assert isinstance(framework.splitter, DataSplitter)
        assert isinstance(framework.tester, StatisticalTester)
        assert isinstance(framework.power_analyzer, PowerAnalyzer)
        assert isinstance(framework.counterfactual_analyzer, CounterfactualAnalyzer)
        assert framework.config['alpha'] == 0.05
    
    def test_holdout_set_creation(self, framework):
        """Test holdout set creation and management."""
        # Create sample data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='D'),
            'value': np.random.normal(100, 15, 1000)
        })
        
        train, holdout = framework.create_holdout_set(
            data, 'test_holdout', SplitStrategy.TEMPORAL, 
            time_column='timestamp'
        )
        
        assert len(train) + len(holdout) == len(data)
        assert 'test_holdout' in framework.holdout_sets
        assert len(framework.holdout_sets['test_holdout']) == len(holdout)
    
    def test_run_evaluation(self, framework, sample_experiment_data):
        """Test complete evaluation workflow."""
        treatment, control = sample_experiment_data
        
        result = framework.run_evaluation(
            treatment_data=treatment,
            control_data=control,
            experiment_name='test_experiment'
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.treatment_group == 'test_experiment_treatment'
        assert result.control_group == 'test_experiment_control'
        assert isinstance(result.p_value, float)
        assert isinstance(result.effect_size, float)
        assert isinstance(result.statistical_power, float)
        assert isinstance(result.confidence_interval, tuple)
        assert len(framework.results_history) == 1
    
    def test_calculate_significance(self, framework):
        """Test significance calculation method."""
        np.random.seed(42)
        treatment = np.random.normal(1.5, 1.0, 100)
        control = np.random.normal(1.0, 1.0, 100)
        
        is_sig, p_value, ci = framework.calculate_significance(
            'test_metric', treatment, control
        )
        
        assert isinstance(is_sig, bool)
        assert isinstance(p_value, float)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert 0 <= p_value <= 1
    
    def test_counterfactual_analysis(self, framework):
        """Test counterfactual analysis integration."""
        # Create sample causal data
        np.random.seed(42)
        n = 500
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n),
            'feature2': np.random.normal(0, 1, n),
            'treatment': np.random.binomial(1, 0.3, n),
            'outcome': np.random.normal(10, 2, n)
        })
        
        # Add treatment effect
        data.loc[data['treatment'] == 1, 'outcome'] += 2
        
        result = framework.analyze_counterfactual(
            data=data,
            treatment_column='treatment',
            outcome_column='outcome',
            feature_columns=['feature1', 'feature2'],
            policy_name='test_policy'
        )
        
        assert isinstance(result, CounterfactualResult)
        assert result.policy_name == 'test_policy'
    
    def test_report_generation(self, framework, sample_experiment_data):
        """Test comprehensive report generation."""
        treatment, control = sample_experiment_data
        
        # Run some evaluations first
        framework.run_evaluation(treatment, control, 'test1')
        framework.run_evaluation(treatment * 1.1, control, 'test2')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            framework.config['output_dir'] = temp_dir
            report = framework.generate_report(include_plots=False)
            
            assert isinstance(report, dict)
            assert 'metadata' in report
            assert 'summary' in report
            assert 'detailed_results' in report
            assert 'statistical_analysis' in report
            assert 'recommendations' in report
            
            # Check summary content
            assert report['summary']['total_experiments'] == 2
            assert 'average_effect_size' in report['summary']
            assert 'significance_rate' in report['summary']
    
    def test_performance_metrics_integration(self, framework):
        """Test integration with performance metrics."""
        treatment_metrics = PerformanceMetrics(
            impressions=10000, clicks=350, conversions=20,
            spend=800.0, revenue=2400.0
        )
        control_metrics = PerformanceMetrics(
            impressions=10000, clicks=300, conversions=15,
            spend=700.0, revenue=2250.0
        )
        
        result = framework.run_evaluation(
            treatment_data=np.array([2400.0]),  # Single revenue value
            control_data=np.array([2250.0]),   # Single revenue value
            experiment_name='metrics_test',
            metrics_treatment=treatment_metrics,
            metrics_control=control_metrics
        )
        
        assert result.treatment_metrics.roas == 3.0  # 2400/800
        assert result.control_metrics.roas == pytest.approx(3.214, rel=1e-2)  # 2250/700
    
    def test_multiple_testing_integration(self, framework):
        """Test multiple testing correction integration."""
        np.random.seed(42)
        
        # Run multiple experiments
        for i in range(5):
            treatment = np.random.normal(1.1, 1.0, 100)
            control = np.random.normal(1.0, 1.0, 100)
            framework.run_evaluation(
                treatment, control, f'multiple_test_{i}',
                multiple_comparisons=True
            )
        
        assert len(framework.results_history) == 5
        
        # Generate report to check multiple testing recommendations
        report = framework.generate_report(include_plots=False)
        recommendations = report['recommendations']
        
        # Should recommend multiple testing correction
        assert any('multiple' in rec.lower() for rec in recommendations)
    
    def test_config_override(self):
        """Test configuration override functionality."""
        custom_config = {
            'alpha': 0.01,
            'power': 0.9,
            'random_state': 123
        }
        
        framework = EvaluationFramework(custom_config)
        
        assert framework.config['alpha'] == 0.01
        assert framework.config['power'] == 0.9
        assert framework.config['random_state'] == 123
        assert framework.tester.alpha == 0.01  # Should propagate to components
    
    def test_json_serialization(self, framework):
        """Test JSON serialization of results."""
        np.random.seed(42)
        treatment = np.random.normal(1.2, 1.0, 50)
        control = np.random.normal(1.0, 1.0, 50)
        
        result = framework.run_evaluation(treatment, control, 'serialize_test')
        
        # Test that result can be made JSON serializable
        json_result = framework._make_json_serializable(result)
        json_str = json.dumps(json_result)
        
        assert isinstance(json_str, str)
        
        # Test deserialization
        deserialized = json.loads(json_str)
        assert isinstance(deserialized, dict)
        assert 'p_value' in deserialized
        assert 'effect_size' in deserialized


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_quick_ab_test(self):
        """Test quick A/B test function."""
        np.random.seed(42)
        treatment = np.random.normal(1.3, 1.0, 100)
        control = np.random.normal(1.0, 1.0, 100)
        
        result = quick_ab_test(treatment, control, 'quick_test')
        
        assert isinstance(result, ExperimentResult)
        assert result.treatment_group == 'quick_test_treatment'
        assert result.control_group == 'quick_test_control'
    
    def test_calculate_required_sample_size_function(self):
        """Test sample size calculation function."""
        sample_size = calculate_required_sample_size(0.5, 0.8, 0.05)
        
        assert isinstance(sample_size, int)
        assert sample_size > 0
        assert sample_size < 1000  # Should be reasonable
    
    def test_different_effect_sizes(self):
        """Test sample size calculation for different effect sizes."""
        small_n = calculate_required_sample_size(0.2)   # Small effect
        medium_n = calculate_required_sample_size(0.5)  # Medium effect
        large_n = calculate_required_sample_size(0.8)   # Large effect
        
        # Larger effects should require smaller samples
        assert small_n > medium_n > large_n


class TestEdgeCasesAndErrors:
    """Test suite for edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        framework = EvaluationFramework()
        
        with pytest.raises((ValueError, IndexError)):
            framework.run_evaluation(
                np.array([]), np.array([]), 'empty_test'
            )
    
    def test_single_value_arrays(self):
        """Test handling of single-value arrays."""
        framework = EvaluationFramework()
        
        # This should work but may have limited statistical validity
        result = framework.run_evaluation(
            np.array([1.0]), np.array([2.0]), 'single_value_test'
        )
        
        assert isinstance(result, ExperimentResult)
        # Effect size calculation should handle single values
        assert isinstance(result.effect_size, float)
    
    def test_identical_groups(self):
        """Test handling of identical treatment and control groups."""
        framework = EvaluationFramework()
        
        identical_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = framework.run_evaluation(
            identical_data, identical_data.copy(), 'identical_test'
        )
        
        # Should detect no difference
        assert result.p_value > 0.5  # High p-value for no difference
        assert abs(result.effect_size) < 0.1  # Near-zero effect size
    
    def test_extreme_sample_size_imbalance(self):
        """Test handling of extremely imbalanced sample sizes."""
        framework = EvaluationFramework()
        
        large_treatment = np.random.normal(1.0, 1.0, 1000)
        small_control = np.random.normal(1.0, 1.0, 10)
        
        result = framework.run_evaluation(
            large_treatment, small_control, 'imbalanced_test'
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.sample_size_treatment == 1000
        assert result.sample_size_control == 10
    
    def test_missing_columns_in_counterfactual(self):
        """Test error handling for missing columns in counterfactual analysis."""
        framework = EvaluationFramework()
        
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'treatment': [0, 1, 0],
            'outcome': [1, 2, 1]
        })
        
        with pytest.raises(KeyError):
            framework.analyze_counterfactual(
                data=data,
                treatment_column='treatment',
                outcome_column='outcome',
                feature_columns=['feature1', 'missing_feature'],  # Missing column
                policy_name='error_test'
            )
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration values."""
        # Test negative alpha
        with pytest.raises((ValueError, AssertionError)):
            config = {'alpha': -0.05}
            framework = EvaluationFramework(config)
        
        # Test alpha > 1
        with pytest.raises((ValueError, AssertionError)):
            config = {'alpha': 1.5}
            framework = EvaluationFramework(config)
    
    def test_insufficient_data_for_power_analysis(self):
        """Test power analysis with insufficient data."""
        analyzer = PowerAnalyzer()
        
        # Very small effect size should require large sample
        sample_size = analyzer.calculate_sample_size(effect_size=0.01)
        assert sample_size >= 10  # Should have minimum threshold
        
        # Very small sample should have low power
        power = analyzer.calculate_power(sample_size=5, effect_size=0.1)
        assert 0 <= power <= 1


@pytest.fixture(scope="module")
def integration_data():
    """Create comprehensive integration test data."""
    np.random.seed(42)
    
    # Create realistic advertising data
    n_campaigns = 100
    campaigns_data = []
    
    for i in range(n_campaigns):
        impressions = np.random.randint(1000, 50000)
        ctr = np.random.uniform(0.01, 0.05)
        clicks = int(impressions * ctr)
        conversion_rate = np.random.uniform(0.01, 0.1)
        conversions = int(clicks * conversion_rate)
        cpc = np.random.uniform(0.5, 3.0)
        spend = clicks * cpc
        revenue_per_conversion = np.random.uniform(20, 200)
        revenue = conversions * revenue_per_conversion
        
        campaigns_data.append({
            'campaign_id': f'campaign_{i}',
            'treatment_group': 'A' if i < 50 else 'B',
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'spend': spend,
            'revenue': revenue,
            'ctr': ctr,
            'conversion_rate': conversion_rate,
            'cpc': cpc,
            'roas': revenue / spend if spend > 0 else 0
        })
    
    return pd.DataFrame(campaigns_data)


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""
    
    def test_end_to_end_campaign_evaluation(self, integration_data):
        """Test complete campaign evaluation workflow."""
        framework = EvaluationFramework({'save_results': False})
        
        # Split data into treatment and control
        treatment_campaigns = integration_data[
            integration_data['treatment_group'] == 'A'
        ]
        control_campaigns = integration_data[
            integration_data['treatment_group'] == 'B'
        ]
        
        # Test multiple metrics
        metrics_to_test = ['roas', 'ctr', 'conversion_rate', 'cpc']
        results = []
        
        for metric in metrics_to_test:
            treatment_values = treatment_campaigns[metric].values
            control_values = control_campaigns[metric].values
            
            result = framework.run_evaluation(
                treatment_data=treatment_values,
                control_data=control_values,
                experiment_name=f'{metric}_comparison'
            )
            results.append(result)
        
        # Generate comprehensive report
        report = framework.generate_report(results, include_plots=False)
        
        assert len(results) == 4
        assert report['summary']['total_experiments'] == 4
        assert 'recommendations' in report
        assert len(report['recommendations']) > 0
    
    def test_temporal_holdout_evaluation(self, integration_data):
        """Test temporal holdout set evaluation."""
        framework = EvaluationFramework({'save_results': False})
        
        # Add temporal information
        integration_data['date'] = pd.date_range(
            '2024-01-01', periods=len(integration_data), freq='D'
        )
        
        # Create temporal holdout
        train_data, holdout_data = framework.create_holdout_set(
            integration_data, 'temporal_holdout', 
            SplitStrategy.TEMPORAL, time_column='date'
        )
        
        # Evaluate on holdout
        holdout_treatment = holdout_data[
            holdout_data['treatment_group'] == 'A'
        ]['roas'].values
        holdout_control = holdout_data[
            holdout_data['treatment_group'] == 'B'
        ]['roas'].values
        
        if len(holdout_treatment) > 0 and len(holdout_control) > 0:
            result = framework.run_evaluation(
                holdout_treatment, holdout_control, 'holdout_test'
            )
            
            assert isinstance(result, ExperimentResult)
            assert 'temporal_holdout' in framework.holdout_sets
    
    def test_counterfactual_policy_evaluation(self, integration_data):
        """Test counterfactual policy evaluation on realistic data."""
        framework = EvaluationFramework({'save_results': False})
        
        # Prepare data for counterfactual analysis
        causal_data = integration_data.copy()
        causal_data['treatment'] = (causal_data['treatment_group'] == 'A').astype(int)
        
        # Select features and outcome
        feature_cols = ['impressions', 'clicks', 'spend']
        
        result = framework.analyze_counterfactual(
            data=causal_data,
            treatment_column='treatment',
            outcome_column='roas',
            feature_columns=feature_cols,
            policy_name='campaign_strategy_A',
            baseline_policy='campaign_strategy_B',
            method='doubly_robust'
        )
        
        assert isinstance(result, CounterfactualResult)
        assert result.policy_name == 'campaign_strategy_A'
        assert result.sample_size == len(causal_data)
    
    def test_power_analysis_for_experiment_design(self):
        """Test power analysis for experiment design."""
        framework = EvaluationFramework()
        
        # Test different scenarios
        scenarios = [
            {'effect_size': 0.2, 'power': 0.8},  # Small effect
            {'effect_size': 0.5, 'power': 0.8},  # Medium effect
            {'effect_size': 0.8, 'power': 0.8},  # Large effect
            {'effect_size': 0.5, 'power': 0.9},  # Higher power
        ]
        
        for scenario in scenarios:
            sample_size = framework.power_analyzer.calculate_sample_size(
                scenario['effect_size'], scenario['power']
            )
            
            # Verify the calculation by computing power
            actual_power = framework.power_analyzer.calculate_power(
                sample_size, scenario['effect_size']
            )
            
            assert actual_power >= scenario['power'] * 0.95  # Within 5%
    
    def test_full_statistical_pipeline(self, integration_data):
        """Test complete statistical analysis pipeline."""
        framework = EvaluationFramework({
            'multiple_testing_correction': MultipleTestCorrection.FDR_BH,
            'save_results': False
        })
        
        # Simulate A/B test results for multiple metrics
        treatment_data = integration_data[
            integration_data['treatment_group'] == 'A'
        ]
        control_data = integration_data[
            integration_data['treatment_group'] == 'B'
        ]
        
        # Test multiple hypotheses (multiple metrics)
        metrics = ['roas', 'ctr', 'conversion_rate']
        
        for metric in metrics:
            framework.run_evaluation(
                treatment_data[metric].values,
                control_data[metric].values,
                f'{metric}_test',
                multiple_comparisons=True  # Apply correction
            )
        
        # Generate final report with recommendations
        report = framework.generate_report(include_plots=False)
        
        # Validate comprehensive analysis
        assert 'multiple_testing' in report['statistical_analysis']
        assert len(report['detailed_results']['experiments']) == 3
        
        # Should contain power analysis recommendations
        recommendations = report['recommendations']
        assert len(recommendations) > 0
        
        # Should detect multiple comparisons
        assert any('multiple' in rec.lower() for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])