"""
Performance and Load Tests for Evaluation Framework

This module contains performance tests, load tests, and benchmarks for the
evaluation framework to ensure it can handle production workloads efficiently.
"""

import pytest
import numpy as np
import pandas as pd
import time
import concurrent.futures
from datetime import datetime, timedelta
import psutil
import gc
from typing import List, Dict, Any
from unittest.mock import Mock
import threading

from evaluation_framework import (
    EvaluationFramework, DataSplitter, StatisticalTester, PowerAnalyzer,
    CounterfactualAnalyzer, PerformanceMetrics, quick_ab_test,
    SplitStrategy, StatisticalTest, MultipleTestCorrection
)


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.results = {}
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    def memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        process = psutil.Process()
        
        # Force garbage collection before measurement
        gc.collect()
        memory_before = process.memory_info().rss
        
        result = func(*args, **kwargs)
        
        gc.collect()
        memory_after = process.memory_info().rss
        memory_used = memory_after - memory_before
        
        return result, memory_used
    
    def profile_function(self, func, *args, **kwargs):
        """Profile both time and memory usage."""
        process = psutil.Process()
        
        gc.collect()
        memory_before = process.memory_info().rss
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        gc.collect()
        memory_after = process.memory_info().rss
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_used': memory_after - memory_before,
            'peak_memory': process.memory_info().rss
        }


@pytest.fixture
def benchmark():
    """Create performance benchmark instance."""
    return PerformanceBenchmark()


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    np.random.seed(42)
    n_samples = 100000
    
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'user_id': np.random.randint(1, 10000, n_samples),
        'campaign_id': np.random.randint(1, 100, n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.uniform(0, 1, n_samples),
        'treatment': np.random.binomial(1, 0.5, n_samples),
        'outcome': np.random.normal(100, 15, n_samples) + 
                  np.random.binomial(1, 0.5, n_samples) * 5  # Treatment effect
    })


class TestDataSplittingPerformance:
    """Performance tests for data splitting operations."""
    
    def test_random_split_performance(self, benchmark, large_dataset):
        """Test performance of random data splitting."""
        splitter = DataSplitter(random_state=42)
        
        profile = benchmark.profile_function(
            splitter.split_data,
            large_dataset, SplitStrategy.RANDOM, test_size=0.2
        )
        
        assert profile['execution_time'] < 2.0  # Should complete within 2 seconds
        assert profile['memory_used'] < 100 * 1024 * 1024  # < 100MB additional memory
        
        train_data, test_data = profile['result']
        assert len(train_data) + len(test_data) == len(large_dataset)
    
    def test_temporal_split_performance(self, benchmark, large_dataset):
        """Test performance of temporal data splitting."""
        splitter = DataSplitter(random_state=42)
        
        profile = benchmark.profile_function(
            splitter.split_data,
            large_dataset, SplitStrategy.TEMPORAL, 
            test_size=0.2, time_column='timestamp'
        )
        
        assert profile['execution_time'] < 3.0  # Sorting takes more time
        assert profile['memory_used'] < 150 * 1024 * 1024  # < 150MB
        
        train_data, test_data = profile['result']
        assert len(train_data) + len(test_data) == len(large_dataset)
    
    def test_stratified_split_performance(self, benchmark, large_dataset):
        """Test performance of stratified data splitting."""
        splitter = DataSplitter(random_state=42)
        
        profile = benchmark.profile_function(
            splitter.split_data,
            large_dataset, SplitStrategy.STRATIFIED,
            test_size=0.2, stratify_column='campaign_id'
        )
        
        assert profile['execution_time'] < 5.0  # Stratification is more complex
        assert profile['memory_used'] < 200 * 1024 * 1024  # < 200MB
        
        train_data, test_data = profile['result']
        assert len(train_data) + len(test_data) == len(large_dataset)
    
    def test_concurrent_splitting(self, large_dataset):
        """Test concurrent data splitting operations."""
        splitter = DataSplitter(random_state=42)
        
        def split_worker(dataset, strategy, worker_id):
            """Worker function for concurrent splitting."""
            try:
                start_time = time.time()
                train, test = splitter.split_data(dataset, strategy, test_size=0.2)
                end_time = time.time()
                return {
                    'worker_id': worker_id,
                    'success': True,
                    'execution_time': end_time - start_time,
                    'train_size': len(train),
                    'test_size': len(test)
                }
            except Exception as e:
                return {
                    'worker_id': worker_id,
                    'success': False,
                    'error': str(e)
                }
        
        # Run concurrent splits
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(8):  # 8 concurrent operations
                future = executor.submit(
                    split_worker, large_dataset, SplitStrategy.RANDOM, i
                )
                futures.append(future)
            
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All operations should succeed
        assert all(result['success'] for result in results)
        
        # Concurrent execution should be faster than sequential
        assert total_time < 8 * 2.0  # Should be faster than 8 * 2 seconds
        
        # Verify results consistency
        for result in results:
            assert result['train_size'] + result['test_size'] == len(large_dataset)


class TestStatisticalTestingPerformance:
    """Performance tests for statistical testing operations."""
    
    def test_large_sample_t_test_performance(self, benchmark):
        """Test t-test performance with large samples."""
        np.random.seed(42)
        
        # Large samples
        treatment = np.random.normal(1.1, 1.0, 50000)
        control = np.random.normal(1.0, 1.0, 50000)
        
        tester = StatisticalTester()
        
        profile = benchmark.profile_function(
            tester.calculate_significance,
            treatment, control, StatisticalTest.T_TEST
        )
        
        assert profile['execution_time'] < 1.0  # Should be fast
        assert profile['memory_used'] < 50 * 1024 * 1024  # < 50MB
        
        p_value, test_stat, effect_size = profile['result']
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
    
    def test_multiple_testing_correction_performance(self, benchmark):
        """Test performance of multiple testing corrections."""
        tester = StatisticalTester()
        
        # Many p-values (simulating many experiments)
        p_values = np.random.uniform(0, 1, 1000).tolist()
        
        # Test different correction methods
        methods = [
            MultipleTestCorrection.BONFERRONI,
            MultipleTestCorrection.HOLM,
            MultipleTestCorrection.FDR_BH,
            MultipleTestCorrection.FDR_BY
        ]
        
        for method in methods:
            profile = benchmark.profile_function(
                tester.correct_multiple_testing,
                p_values, method
            )
            
            assert profile['execution_time'] < 0.5  # Should be fast
            rejected, corrected = profile['result']
            assert len(rejected) == len(p_values)
            assert len(corrected) == len(p_values)
    
    def test_concurrent_statistical_tests(self):
        """Test concurrent statistical testing operations."""
        tester = StatisticalTester()
        
        def test_worker(worker_id, sample_size=1000):
            """Worker function for concurrent testing."""
            np.random.seed(worker_id)  # Different seed for each worker
            
            treatment = np.random.normal(1.1, 1.0, sample_size)
            control = np.random.normal(1.0, 1.0, sample_size)
            
            start_time = time.time()
            p_value, test_stat, effect_size = tester.calculate_significance(
                treatment, control, StatisticalTest.T_TEST
            )
            end_time = time.time()
            
            return {
                'worker_id': worker_id,
                'p_value': p_value,
                'effect_size': effect_size,
                'execution_time': end_time - start_time
            }
        
        # Run concurrent tests
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(test_worker, i) for i in range(20)]
            results = [future.result() for future in futures]
        
        # All tests should complete successfully
        assert len(results) == 20
        assert all(0 <= result['p_value'] <= 1 for result in results)
        
        # Average execution time should be reasonable
        avg_time = np.mean([result['execution_time'] for result in results])
        assert avg_time < 0.1  # Should be fast


class TestCounterfactualAnalysisPerformance:
    """Performance tests for counterfactual analysis."""
    
    def test_iptw_performance_large_dataset(self, benchmark, large_dataset):
        """Test IPTW performance on large dataset."""
        analyzer = CounterfactualAnalyzer()
        
        # Use subset for counterfactual analysis (still large)
        subset_data = large_dataset.sample(n=10000, random_state=42)
        
        profile = benchmark.profile_function(
            analyzer.estimate_policy_effect,
            data=subset_data,
            treatment_column='treatment',
            outcome_column='outcome',
            feature_columns=['feature1', 'feature2', 'feature3'],
            policy_name='test_policy',
            baseline_policy='control',
            method='iptw'
        )
        
        # Should complete within reasonable time
        assert profile['execution_time'] < 30.0  # 30 seconds
        assert profile['memory_used'] < 500 * 1024 * 1024  # < 500MB
        
        result = profile['result']
        assert result.sample_size == 10000
    
    def test_direct_method_performance(self, benchmark, large_dataset):
        """Test direct method performance."""
        analyzer = CounterfactualAnalyzer()
        
        subset_data = large_dataset.sample(n=5000, random_state=42)
        
        profile = benchmark.profile_function(
            analyzer.estimate_policy_effect,
            data=subset_data,
            treatment_column='treatment',
            outcome_column='outcome',
            feature_columns=['feature1', 'feature2', 'feature3'],
            policy_name='test_policy',
            baseline_policy='control',
            method='dm'
        )
        
        # Direct method uses ML models, so may take longer
        assert profile['execution_time'] < 45.0  # 45 seconds
        assert profile['memory_used'] < 1024 * 1024 * 1024  # < 1GB
        
        result = profile['result']
        assert result.methodology == "Direct Method"
    
    def test_bootstrap_confidence_interval_performance(self, benchmark):
        """Test bootstrap confidence interval performance."""
        analyzer = CounterfactualAnalyzer()
        
        # Create smaller dataset for bootstrap testing
        np.random.seed(42)
        n = 1000
        
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n),
            'treatment': np.random.binomial(1, 0.5, n),
            'outcome': np.random.normal(10, 2, n)
        })
        
        # Test bootstrap with reduced iterations for performance
        profile = benchmark.profile_function(
            analyzer._bootstrap_confidence_interval,
            data, 'treatment', 'outcome', ['feature1'], 'iptw', 
            n_bootstrap=100  # Reduced for performance testing
        )
        
        assert profile['execution_time'] < 10.0  # Should be reasonable
        ci_lower, ci_upper = profile['result']
        assert ci_lower < ci_upper


class TestEvaluationFrameworkPerformance:
    """Performance tests for the main evaluation framework."""
    
    def test_framework_initialization_performance(self, benchmark):
        """Test framework initialization performance."""
        profile = benchmark.profile_function(
            EvaluationFramework,
            {'save_results': False, 'random_state': 42}
        )
        
        assert profile['execution_time'] < 1.0  # Quick initialization
        assert profile['memory_used'] < 50 * 1024 * 1024  # < 50MB
        
        framework = profile['result']
        assert isinstance(framework, EvaluationFramework)
    
    def test_large_scale_evaluation_performance(self, benchmark):
        """Test evaluation performance with large datasets."""
        framework = EvaluationFramework({'save_results': False})
        
        # Large treatment and control groups
        np.random.seed(42)
        treatment_data = np.random.normal(1.05, 1.0, 20000)  # Small effect
        control_data = np.random.normal(1.0, 1.0, 20000)
        
        profile = benchmark.profile_function(
            framework.run_evaluation,
            treatment_data, control_data, "large_scale_test"
        )
        
        assert profile['execution_time'] < 5.0  # Should complete quickly
        assert profile['memory_used'] < 100 * 1024 * 1024  # < 100MB
        
        result = profile['result']
        assert result.sample_size_treatment == 20000
        assert result.sample_size_control == 20000
    
    def test_multiple_concurrent_evaluations(self):
        """Test concurrent evaluation performance."""
        framework = EvaluationFramework({'save_results': False})
        
        def evaluation_worker(worker_id, sample_size=5000):
            """Worker function for concurrent evaluations."""
            np.random.seed(worker_id)
            
            # Generate data with slight variation
            effect_size = 0.1 + 0.05 * (worker_id % 5)  # Varying effect sizes
            treatment = np.random.normal(1 + effect_size, 1.0, sample_size)
            control = np.random.normal(1.0, 1.0, sample_size)
            
            start_time = time.time()
            result = framework.run_evaluation(
                treatment, control, f"concurrent_test_{worker_id}"
            )
            end_time = time.time()
            
            return {
                'worker_id': worker_id,
                'result': result,
                'execution_time': end_time - start_time,
                'effect_size': result.effect_size,
                'p_value': result.p_value
            }
        
        # Run multiple concurrent evaluations
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(evaluation_worker, i) for i in range(12)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # All evaluations should succeed
        assert len(results) == 12
        assert all(isinstance(r['result'], type(results[0]['result'])) for r in results)
        
        # Concurrent execution should be efficient
        sequential_estimate = sum(r['execution_time'] for r in results)
        assert total_time < sequential_estimate * 0.8  # At least 20% speedup
        
        # Check that framework tracked all results
        assert len(framework.results_history) == 12
    
    def test_report_generation_performance(self, benchmark):
        """Test performance of report generation."""
        framework = EvaluationFramework({'save_results': False})
        
        # Run multiple evaluations first
        np.random.seed(42)
        for i in range(10):
            treatment = np.random.normal(1.1, 1.0, 1000)
            control = np.random.normal(1.0, 1.0, 1000)
            framework.run_evaluation(treatment, control, f"perf_test_{i}")
        
        # Test report generation performance
        profile = benchmark.profile_function(
            framework.generate_report,
            include_plots=False
        )
        
        assert profile['execution_time'] < 5.0  # Should generate quickly
        assert profile['memory_used'] < 100 * 1024 * 1024  # < 100MB
        
        report = profile['result']
        assert report['summary']['total_experiments'] == 10
    
    def test_holdout_set_management_performance(self, benchmark, large_dataset):
        """Test holdout set management performance."""
        framework = EvaluationFramework({'save_results': False})
        
        profile = benchmark.profile_function(
            framework.create_holdout_set,
            large_dataset, 'perf_holdout', SplitStrategy.TEMPORAL,
            time_column='timestamp'
        )
        
        assert profile['execution_time'] < 5.0  # Should be fast
        assert profile['memory_used'] < 200 * 1024 * 1024  # < 200MB
        
        train_data, holdout_data = profile['result']
        assert 'perf_holdout' in framework.holdout_sets
        assert len(framework.holdout_sets['perf_holdout']) == len(holdout_data)


class TestMemoryUsageAndLeaks:
    """Test memory usage patterns and potential leaks."""
    
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with data size."""
        framework = EvaluationFramework({'save_results': False})
        process = psutil.Process()
        
        memory_usage = []
        sample_sizes = [1000, 5000, 10000, 20000, 50000]
        
        for sample_size in sample_sizes:
            gc.collect()  # Clean up before measurement
            
            memory_before = process.memory_info().rss
            
            # Generate and evaluate data
            np.random.seed(42)
            treatment = np.random.normal(1.1, 1.0, sample_size)
            control = np.random.normal(1.0, 1.0, sample_size)
            
            framework.run_evaluation(treatment, control, f"memory_test_{sample_size}")
            
            memory_after = process.memory_info().rss
            memory_used = memory_after - memory_before
            memory_usage.append(memory_used)
        
        # Memory usage should scale reasonably (roughly linear)
        # Check that memory doesn't explode exponentially
        for i in range(1, len(memory_usage)):
            ratio = memory_usage[i] / memory_usage[i-1]
            size_ratio = sample_sizes[i] / sample_sizes[i-1]
            
            # Memory growth should be roughly proportional to data size
            assert ratio < size_ratio * 2  # Allow 2x overhead
    
    def test_memory_cleanup_after_operations(self):
        """Test that memory is properly cleaned up."""
        framework = EvaluationFramework({'save_results': False})
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss
        
        # Perform many operations
        for i in range(50):
            np.random.seed(i)
            treatment = np.random.normal(1.1, 1.0, 2000)
            control = np.random.normal(1.0, 1.0, 2000)
            framework.run_evaluation(treatment, control, f"cleanup_test_{i}")
        
        # Clear framework history to test cleanup
        framework.results_history.clear()
        framework.holdout_sets.clear()
        
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_concurrent_memory_usage(self):
        """Test memory usage under concurrent load."""
        def memory_intensive_worker(worker_id):
            """Worker that performs memory-intensive operations."""
            framework = EvaluationFramework({'save_results': False})
            
            for i in range(10):
                np.random.seed(worker_id * 100 + i)
                treatment = np.random.normal(1.1, 1.0, 5000)
                control = np.random.normal(1.0, 1.0, 5000)
                framework.run_evaluation(treatment, control, f"worker_{worker_id}_test_{i}")
            
            return len(framework.results_history)
        
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(memory_intensive_worker, i) for i in range(8)]
            results = [future.result() for future in futures]
        
        memory_after = process.memory_info().rss
        memory_used = memory_after - memory_before
        
        # All workers should complete
        assert all(result == 10 for result in results)
        
        # Memory usage should be reasonable for concurrent operations
        assert memory_used < 500 * 1024 * 1024  # < 500MB


class TestBenchmarkingSuite:
    """Comprehensive benchmarking suite for performance validation."""
    
    def test_end_to_end_performance_benchmark(self, benchmark):
        """Complete end-to-end performance benchmark."""
        # Test configuration
        test_config = {
            'sample_sizes': [1000, 5000, 10000],
            'effect_sizes': [0.1, 0.3, 0.5],
            'test_types': [StatisticalTest.T_TEST, StatisticalTest.MANN_WHITNEY]
        }
        
        benchmark_results = []
        
        for sample_size in test_config['sample_sizes']:
            for effect_size in test_config['effect_sizes']:
                for test_type in test_config['test_types']:
                    # Setup
                    framework = EvaluationFramework({'save_results': False})
                    np.random.seed(42)
                    
                    treatment = np.random.normal(1 + effect_size, 1.0, sample_size)
                    control = np.random.normal(1.0, 1.0, sample_size)
                    
                    # Benchmark
                    profile = benchmark.profile_function(
                        framework.run_evaluation,
                        treatment, control, "benchmark_test", test_type=test_type
                    )
                    
                    benchmark_results.append({
                        'sample_size': sample_size,
                        'effect_size': effect_size,
                        'test_type': test_type.value,
                        'execution_time': profile['execution_time'],
                        'memory_used': profile['memory_used'],
                        'detected_effect': profile['result'].statistical_significance
                    })
        
        # Analyze benchmark results
        df_results = pd.DataFrame(benchmark_results)
        
        # Performance should scale reasonably with sample size
        for test_type in test_config['test_types']:
            subset = df_results[df_results['test_type'] == test_type.value]
            
            # Execution time should not explode with sample size
            max_time = subset['execution_time'].max()
            assert max_time < 5.0  # 5 seconds maximum
            
            # Memory usage should be reasonable
            max_memory = subset['memory_used'].max()
            assert max_memory < 200 * 1024 * 1024  # 200MB maximum
        
        # Larger effect sizes should be detected more reliably
        large_effect_detection = df_results[
            df_results['effect_size'] == 0.5
        ]['detected_effect'].mean()
        
        small_effect_detection = df_results[
            df_results['effect_size'] == 0.1
        ]['detected_effect'].mean()
        
        assert large_effect_detection > small_effect_detection
    
    def test_stress_testing(self):
        """Stress test the framework under extreme conditions."""
        framework = EvaluationFramework({'save_results': False})
        
        # Test 1: Very large sample sizes
        large_treatment = np.random.normal(1.01, 1.0, 100000)  # Tiny effect
        large_control = np.random.normal(1.0, 1.0, 100000)
        
        start_time = time.time()
        result = framework.run_evaluation(large_treatment, large_control, "stress_large")
        large_time = time.time() - start_time
        
        assert large_time < 10.0  # Should handle large data
        assert isinstance(result.p_value, float)
        
        # Test 2: Many small evaluations
        start_time = time.time()
        for i in range(100):
            small_treatment = np.random.normal(1.2, 1.0, 100)
            small_control = np.random.normal(1.0, 1.0, 100)
            framework.run_evaluation(small_treatment, small_control, f"stress_small_{i}")
        
        many_small_time = time.time() - start_time
        
        assert many_small_time < 15.0  # Should handle many operations
        assert len(framework.results_history) == 101  # 1 from large + 100 small
        
        # Test 3: Report generation with many results
        start_time = time.time()
        report = framework.generate_report(include_plots=False)
        report_time = time.time() - start_time
        
        assert report_time < 5.0  # Should generate report quickly
        assert report['summary']['total_experiments'] == 101
    
    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        # Baseline performance (simulated historical data)
        baseline_times = {
            'small_evaluation': 0.1,    # 100ms
            'medium_evaluation': 0.5,   # 500ms
            'large_evaluation': 2.0,    # 2s
            'report_generation': 1.0    # 1s
        }
        
        framework = EvaluationFramework({'save_results': False})
        
        # Test current performance
        current_times = {}
        
        # Small evaluation
        np.random.seed(42)
        treatment = np.random.normal(1.1, 1.0, 1000)
        control = np.random.normal(1.0, 1.0, 1000)
        
        start_time = time.time()
        framework.run_evaluation(treatment, control, "regression_small")
        current_times['small_evaluation'] = time.time() - start_time
        
        # Medium evaluation
        treatment = np.random.normal(1.1, 1.0, 10000)
        control = np.random.normal(1.0, 1.0, 10000)
        
        start_time = time.time()
        framework.run_evaluation(treatment, control, "regression_medium")
        current_times['medium_evaluation'] = time.time() - start_time
        
        # Large evaluation
        treatment = np.random.normal(1.1, 1.0, 50000)
        control = np.random.normal(1.0, 1.0, 50000)
        
        start_time = time.time()
        framework.run_evaluation(treatment, control, "regression_large")
        current_times['large_evaluation'] = time.time() - start_time
        
        # Report generation
        start_time = time.time()
        framework.generate_report(include_plots=False)
        current_times['report_generation'] = time.time() - start_time
        
        # Check for performance regression (allow 50% slowdown)
        regression_threshold = 1.5
        
        for operation, baseline_time in baseline_times.items():
            current_time = current_times[operation]
            slowdown_ratio = current_time / baseline_time
            
            print(f"{operation}: {current_time:.3f}s vs baseline {baseline_time:.3f}s "
                  f"(ratio: {slowdown_ratio:.2f})")
            
            # Allow some variance but detect major regressions
            assert slowdown_ratio < regression_threshold, \
                f"Performance regression detected in {operation}: {slowdown_ratio:.2f}x slower"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])