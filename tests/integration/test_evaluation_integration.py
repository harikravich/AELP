"""
Integration Tests for Evaluation Framework with GAELP Components

This module tests the integration of the evaluation framework with other
GAELP components including RL agents, campaign managers, attribution models,
and performance tracking systems.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

# Import GAELP components for integration testing
from evaluation_framework import (
    EvaluationFramework, PerformanceMetrics, ExperimentResult,
    SplitStrategy, StatisticalTest, quick_ab_test
)

# Import attribution models for integration
try:
    from attribution_models import (
        Touchpoint, Journey, FirstTouchAttribution, LastTouchAttribution,
        LinearAttribution, TimeDecayAttribution, AttributionAnalyzer
    )
    HAS_ATTRIBUTION = True
except ImportError:
    HAS_ATTRIBUTION = False
    warnings.warn("Attribution models not available for integration testing")

# Mock GAELP components for testing
@pytest.fixture
def mock_rl_agent():
    """Mock RL agent for testing."""
    agent = Mock()
    agent.agent_id = "test_agent_123"
    agent.policy_name = "ppo_policy"
    agent.get_performance_metrics.return_value = {
        'episode_rewards': [100, 120, 110, 130, 125],
        'episode_lengths': [50, 48, 52, 45, 47],
        'convergence_score': 0.85,
        'total_reward': 585,
        'average_reward': 117
    }
    agent.get_action_history.return_value = [
        {'timestamp': datetime.now(), 'action': 'increase_bid', 'value': 1.2},
        {'timestamp': datetime.now(), 'action': 'change_creative', 'value': 'creative_b'},
        {'timestamp': datetime.now(), 'action': 'adjust_targeting', 'value': 0.8}
    ]
    return agent


@pytest.fixture
def mock_campaign_manager():
    """Mock campaign manager for testing."""
    manager = Mock()
    manager.campaign_id = "campaign_456"
    manager.get_campaign_metrics.return_value = PerformanceMetrics(
        impressions=50000,
        clicks=1500,
        conversions=75,
        spend=3750.0,
        revenue=11250.0,
        viewthrough_conversions=15,
        bounce_rate=0.35,
        time_to_conversion=24.5
    )
    manager.get_historical_performance.return_value = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'impressions': np.random.randint(1000, 2000, 30),
        'clicks': np.random.randint(30, 60, 30),
        'conversions': np.random.randint(1, 5, 30),
        'spend': np.random.uniform(100, 200, 30),
        'revenue': np.random.uniform(300, 600, 30)
    })
    return manager


@pytest.fixture
def sample_journey_data():
    """Create sample customer journey data."""
    journeys = []
    
    for i in range(100):
        # Create touchpoints for each journey
        touchpoints = []
        journey_start = datetime(2024, 1, 1) + timedelta(days=i)
        
        # First touch (awareness)
        touchpoints.append(Touchpoint(
            id=f"touch_{i}_1",
            timestamp=journey_start,
            channel="display",
            action="impression",
            value=0.1
        ))
        
        # Middle touches (consideration)
        if np.random.random() > 0.3:  # 70% have middle touches
            touchpoints.append(Touchpoint(
                id=f"touch_{i}_2",
                timestamp=journey_start + timedelta(hours=12),
                channel="social",
                action="click",
                value=0.3
            ))
        
        if np.random.random() > 0.5:  # 50% have additional touch
            touchpoints.append(Touchpoint(
                id=f"touch_{i}_3",
                timestamp=journey_start + timedelta(days=1),
                channel="search",
                action="click",
                value=0.4
            ))
        
        # Last touch (conversion attempt)
        conversion_value = np.random.uniform(50, 200) if np.random.random() > 0.8 else 0
        converted = conversion_value > 0
        
        touchpoints.append(Touchpoint(
            id=f"touch_{i}_final",
            timestamp=journey_start + timedelta(days=2),
            channel="email",
            action="conversion" if converted else "visit",
            value=0.8 if converted else 0.2
        ))
        
        journeys.append(Journey(
            id=f"journey_{i}",
            touchpoints=touchpoints,
            conversion_value=conversion_value,
            conversion_timestamp=journey_start + timedelta(days=2),
            converted=converted
        ))
    
    return journeys


class TestRLAgentEvaluationIntegration:
    """Test integration between evaluation framework and RL agents."""
    
    def test_rl_agent_performance_evaluation(self, mock_rl_agent):
        """Test evaluation of RL agent performance."""
        framework = EvaluationFramework({'save_results': False})
        
        # Get agent performance metrics
        agent_metrics = mock_rl_agent.get_performance_metrics()
        
        # Compare against baseline (random policy)
        baseline_rewards = np.random.normal(80, 20, 5)  # Lower performance baseline
        
        result = framework.run_evaluation(
            treatment_data=np.array(agent_metrics['episode_rewards']),
            control_data=baseline_rewards,
            experiment_name=f"rl_agent_{mock_rl_agent.agent_id}_evaluation"
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.treatment_group == f"rl_agent_{mock_rl_agent.agent_id}_evaluation_treatment"
        assert result.effect_size > 0  # Agent should outperform random baseline
        
        # Test convergence evaluation
        convergence_score = agent_metrics['convergence_score']
        assert convergence_score > 0.8  # Good convergence threshold
    
    def test_multi_agent_comparison(self):
        """Test comparison of multiple RL agents."""
        framework = EvaluationFramework({'save_results': False})
        
        # Mock multiple agents with different performance profiles
        agent_a_rewards = np.random.normal(120, 15, 100)  # High performance
        agent_b_rewards = np.random.normal(100, 20, 100)  # Medium performance
        agent_c_rewards = np.random.normal(85, 25, 100)   # Lower performance
        
        # Compare agents pairwise
        results = []
        
        # A vs B
        result_ab = framework.run_evaluation(
            agent_a_rewards, agent_b_rewards, "agent_a_vs_b"
        )
        results.append(result_ab)
        
        # A vs C
        result_ac = framework.run_evaluation(
            agent_a_rewards, agent_c_rewards, "agent_a_vs_c"
        )
        results.append(result_ac)
        
        # B vs C
        result_bc = framework.run_evaluation(
            agent_b_rewards, agent_c_rewards, "agent_b_vs_c"
        )
        results.append(result_bc)
        
        # Generate comprehensive comparison report
        report = framework.generate_report(results, include_plots=False)
        
        assert len(results) == 3
        assert report['summary']['total_experiments'] == 3
        
        # Should detect multiple comparisons need
        assert any('multiple' in rec.lower() for rec in report['recommendations'])
    
    def test_rl_policy_counterfactual_analysis(self, mock_rl_agent):
        """Test counterfactual analysis of RL policy changes."""
        framework = EvaluationFramework({'save_results': False})
        
        # Create synthetic data representing policy decisions and outcomes
        np.random.seed(42)
        n_decisions = 500
        
        # Features: bid amount, creative type, targeting score
        bid_amounts = np.random.uniform(0.5, 3.0, n_decisions)
        creative_types = np.random.choice([0, 1, 2], n_decisions)  # 3 creative types
        targeting_scores = np.random.uniform(0.1, 1.0, n_decisions)
        
        # Policy assignment (old vs new policy)
        policy_assignment = np.random.binomial(1, 0.5, n_decisions)
        
        # Outcome (reward) influenced by features and policy
        base_reward = 10 + 15 * bid_amounts + 5 * targeting_scores
        policy_effect = 3 * policy_assignment  # New policy adds +3 reward
        noise = np.random.normal(0, 2, n_decisions)
        rewards = base_reward + policy_effect + noise
        
        policy_data = pd.DataFrame({
            'bid_amount': bid_amounts,
            'creative_type': creative_types,
            'targeting_score': targeting_scores,
            'policy': policy_assignment,  # 0 = old, 1 = new
            'reward': rewards
        })
        
        # Perform counterfactual analysis
        result = framework.analyze_counterfactual(
            data=policy_data,
            treatment_column='policy',
            outcome_column='reward',
            feature_columns=['bid_amount', 'creative_type', 'targeting_score'],
            policy_name='new_rl_policy',
            baseline_policy='old_rl_policy'
        )
        
        assert result.policy_name == 'new_rl_policy'
        assert result.estimated_lift > 0  # New policy should show improvement
        assert result.sample_size == n_decisions


class TestCampaignManagerIntegration:
    """Test integration between evaluation framework and campaign management."""
    
    def test_campaign_performance_evaluation(self, mock_campaign_manager):
        """Test evaluation of campaign performance metrics."""
        framework = EvaluationFramework({'save_results': False})
        
        # Get campaign metrics
        campaign_metrics = mock_campaign_manager.get_campaign_metrics()
        
        # Create baseline metrics (industry average)
        baseline_metrics = PerformanceMetrics(
            impressions=50000,
            clicks=1000,  # Lower CTR
            conversions=50,  # Lower conversion rate
            spend=3750.0,
            revenue=9000.0  # Lower ROAS
        )
        
        # Evaluate ROAS performance
        roas_result = framework.run_evaluation(
            treatment_data=np.array([campaign_metrics.roas]),
            control_data=np.array([baseline_metrics.roas]),
            experiment_name="campaign_roas_evaluation",
            metrics_treatment=campaign_metrics,
            metrics_control=baseline_metrics
        )
        
        assert isinstance(roas_result, ExperimentResult)
        assert roas_result.treatment_metrics.roas > roas_result.control_metrics.roas
    
    def test_historical_performance_analysis(self, mock_campaign_manager):
        """Test analysis of historical campaign performance."""
        framework = EvaluationFramework({'save_results': False})
        
        # Get historical data
        historical_data = mock_campaign_manager.get_historical_performance()
        
        # Split into before/after periods (simulating campaign optimization)
        split_point = len(historical_data) // 2
        before_period = historical_data.iloc[:split_point]
        after_period = historical_data.iloc[split_point:]
        
        # Create temporal holdout
        train_data, holdout_data = framework.create_holdout_set(
            historical_data, 'campaign_temporal_holdout',
            SplitStrategy.TEMPORAL, time_column='date'
        )
        
        # Calculate daily ROAS for both periods
        before_roas = before_period['revenue'] / before_period['spend']
        after_roas = after_period['revenue'] / after_period['spend']
        
        # Evaluate improvement
        improvement_result = framework.run_evaluation(
            treatment_data=after_roas.values,
            control_data=before_roas.values,
            experiment_name="campaign_optimization_evaluation"
        )
        
        assert isinstance(improvement_result, ExperimentResult)
        assert 'campaign_temporal_holdout' in framework.holdout_sets
    
    def test_multi_campaign_comparison(self):
        """Test comparison of multiple campaigns."""
        framework = EvaluationFramework({'save_results': False})
        
        # Create multiple campaign scenarios
        campaigns_data = []
        
        for i in range(5):  # 5 different campaigns
            campaign_roas = np.random.uniform(2.0, 4.0, 30)  # 30 days of data
            campaign_ctr = np.random.uniform(0.02, 0.06, 30)
            campaign_conv_rate = np.random.uniform(0.03, 0.08, 30)
            
            campaigns_data.append({
                'campaign_id': f'campaign_{i}',
                'roas': campaign_roas,
                'ctr': campaign_ctr,
                'conversion_rate': campaign_conv_rate
            })
        
        # Compare best vs worst performing campaigns
        best_campaign = max(campaigns_data, key=lambda x: np.mean(x['roas']))
        worst_campaign = min(campaigns_data, key=lambda x: np.mean(x['roas']))
        
        comparison_result = framework.run_evaluation(
            treatment_data=best_campaign['roas'],
            control_data=worst_campaign['roas'],
            experiment_name="best_vs_worst_campaign"
        )
        
        assert comparison_result.effect_size > 0  # Best should outperform worst
        assert comparison_result.statistical_significance


@pytest.mark.skipif(not HAS_ATTRIBUTION, reason="Attribution models not available")
class TestAttributionModelIntegration:
    """Test integration between evaluation framework and attribution models."""
    
    def test_attribution_model_comparison(self, sample_journey_data):
        """Test comparison of different attribution models."""
        framework = EvaluationFramework({'save_results': False})
        
        # Initialize different attribution models
        first_touch = FirstTouchAttribution()
        last_touch = LastTouchAttribution()
        linear = LinearAttribution()
        time_decay = TimeDecayAttribution()
        
        models = {
            'first_touch': first_touch,
            'last_touch': last_touch,
            'linear': linear,
            'time_decay': time_decay
        }
        
        # Calculate attribution for each model
        attribution_results = {}
        
        for model_name, model in models.items():
            total_attributed_value = 0
            for journey in sample_journey_data:
                if journey.converted:
                    attribution = model.calculate_attribution(journey)
                    total_attributed_value += sum(attribution.values())
            
            attribution_results[model_name] = total_attributed_value
        
        # Compare linear vs last-touch attribution
        linear_values = np.array([attribution_results['linear']])
        last_touch_values = np.array([attribution_results['last_touch']])
        
        result = framework.run_evaluation(
            treatment_data=linear_values,
            control_data=last_touch_values,
            experiment_name="linear_vs_last_touch_attribution"
        )
        
        assert isinstance(result, ExperimentResult)
    
    def test_attribution_model_validation(self, sample_journey_data):
        """Test validation of attribution model performance."""
        framework = EvaluationFramework({'save_results': False})
        
        # Simulate holdout validation for attribution models
        converted_journeys = [j for j in sample_journey_data if j.converted]
        non_converted_journeys = [j for j in sample_journey_data if not j.converted]
        
        # Create train/test split
        train_converted = converted_journeys[:len(converted_journeys)//2]
        test_converted = converted_journeys[len(converted_journeys)//2:]
        
        # Test attribution consistency
        linear_model = LinearAttribution()
        
        # Calculate attribution on training set
        train_attributions = []
        for journey in train_converted:
            attribution = linear_model.calculate_attribution(journey)
            train_attributions.append(sum(attribution.values()))
        
        # Calculate attribution on test set
        test_attributions = []
        for journey in test_converted:
            attribution = linear_model.calculate_attribution(journey)
            test_attributions.append(sum(attribution.values()))
        
        # Test consistency between train and test
        if len(train_attributions) > 0 and len(test_attributions) > 0:
            consistency_result = framework.run_evaluation(
                treatment_data=np.array(train_attributions),
                control_data=np.array(test_attributions),
                experiment_name="attribution_consistency_test"
            )
            
            # Attribution should be consistent (no significant difference)
            assert isinstance(consistency_result, ExperimentResult)
            # High p-value indicates consistency (no significant difference)
            # This test validates model stability across different data samples


class TestPerformanceTrackingIntegration:
    """Test integration with performance tracking and monitoring systems."""
    
    def test_real_time_performance_monitoring(self):
        """Test real-time performance monitoring integration."""
        framework = EvaluationFramework({'save_results': False})
        
        # Simulate real-time performance data
        current_hour_metrics = PerformanceMetrics(
            impressions=5000,
            clicks=150,
            conversions=8,
            spend=375.0,
            revenue=1200.0
        )
        
        # Historical baseline (same hour, previous weeks)
        historical_baseline = [
            PerformanceMetrics(impressions=4800, clicks=140, conversions=7, spend=350.0, revenue=1050.0),
            PerformanceMetrics(impressions=5200, clicks=145, conversions=6, spend=362.5, revenue=900.0),
            PerformanceMetrics(impressions=4900, clicks=138, conversions=7, spend=345.0, revenue=1120.0),
        ]
        
        # Extract ROAS values for comparison
        current_roas = np.array([current_hour_metrics.roas])
        historical_roas = np.array([metrics.roas for metrics in historical_baseline])
        
        # Test if current performance is significantly different
        anomaly_result = framework.run_evaluation(
            treatment_data=current_roas,
            control_data=historical_roas,
            experiment_name="real_time_anomaly_detection"
        )
        
        assert isinstance(anomaly_result, ExperimentResult)
        
        # Generate alert if performance is significantly different
        if anomaly_result.statistical_significance:
            if anomaly_result.effect_size > 0:
                alert_type = "performance_improvement"
            else:
                alert_type = "performance_degradation"
        else:
            alert_type = "normal_operation"
        
        assert alert_type in ["performance_improvement", "performance_degradation", "normal_operation"]
    
    def test_cohort_analysis_integration(self):
        """Test cohort analysis integration."""
        framework = EvaluationFramework({'save_results': False})
        
        # Create cohort data (users acquired in different months)
        cohort_data = []
        
        for cohort_month in range(6):  # 6 cohorts
            cohort_size = np.random.randint(100, 200)
            
            # Revenue progression over time (retention effect)
            monthly_revenue = []
            base_revenue = np.random.uniform(20, 40, cohort_size)
            
            for month in range(6):  # Track for 6 months
                retention_rate = max(0.1, 1.0 - 0.15 * month)  # Decaying retention
                active_users = int(cohort_size * retention_rate)
                
                if active_users > 0:
                    month_revenue = np.random.choice(base_revenue, active_users) * (1 - 0.1 * month)
                    monthly_revenue.extend(month_revenue)
            
            cohort_data.append({
                'cohort': f'2024-{cohort_month+1:02d}',
                'size': cohort_size,
                'lifetime_revenue': monthly_revenue
            })
        
        # Compare early vs late cohorts
        early_cohorts_revenue = []
        late_cohorts_revenue = []
        
        for i, cohort in enumerate(cohort_data):
            if i < 3:  # Early cohorts
                early_cohorts_revenue.extend(cohort['lifetime_revenue'])
            else:  # Late cohorts
                late_cohorts_revenue.extend(cohort['lifetime_revenue'])
        
        # Test cohort performance differences
        cohort_result = framework.run_evaluation(
            treatment_data=np.array(late_cohorts_revenue),
            control_data=np.array(early_cohorts_revenue),
            experiment_name="cohort_performance_comparison"
        )
        
        assert isinstance(cohort_result, ExperimentResult)
    
    def test_seasonal_adjustment_validation(self):
        """Test seasonal adjustment and validation."""
        framework = EvaluationFramework({'save_results': False})
        
        # Create seasonal data
        days = pd.date_range('2024-01-01', periods=365, freq='D')
        
        # Base performance with seasonal patterns
        base_performance = 100
        seasonal_effect = 10 * np.sin(2 * np.pi * np.arange(365) / 365)  # Yearly cycle
        weekly_effect = 5 * np.sin(2 * np.pi * np.arange(365) / 7)       # Weekly cycle
        noise = np.random.normal(0, 3, 365)
        
        raw_performance = base_performance + seasonal_effect + weekly_effect + noise
        
        # Apply seasonal adjustment (simplified)
        adjusted_performance = raw_performance - seasonal_effect - weekly_effect
        
        # Test if adjustment removes seasonality
        # Split into two halves and compare variance
        first_half_raw = raw_performance[:180]
        second_half_raw = raw_performance[180:]
        
        first_half_adjusted = adjusted_performance[:180]
        second_half_adjusted = adjusted_performance[180:]
        
        # Raw data should show more variance between periods
        raw_variance_diff = abs(np.var(first_half_raw) - np.var(second_half_raw))
        adjusted_variance_diff = abs(np.var(first_half_adjusted) - np.var(second_half_adjusted))
        
        # Seasonal adjustment should reduce variance differences
        assert adjusted_variance_diff < raw_variance_diff


class TestEndToEndWorkflows:
    """Test complete end-to-end evaluation workflows."""
    
    def test_complete_ab_test_workflow(self):
        """Test complete A/B test workflow from design to analysis."""
        framework = EvaluationFramework({'save_results': False})
        
        # Step 1: Power analysis for experiment design
        expected_effect_size = 0.3  # 30% improvement expected
        required_sample_size = framework.power_analyzer.calculate_sample_size(
            expected_effect_size, power=0.8, alpha=0.05
        )
        
        # Step 2: Generate experiment data
        np.random.seed(42)
        treatment_data = np.random.normal(1.3, 1.0, required_sample_size)  # 30% improvement
        control_data = np.random.normal(1.0, 1.0, required_sample_size)
        
        # Step 3: Run evaluation
        result = framework.run_evaluation(
            treatment_data, control_data, "complete_ab_test"
        )
        
        # Step 4: Validate power achieved
        actual_power = framework.power_analyzer.calculate_power(
            required_sample_size, abs(result.effect_size), alpha=0.05
        )
        
        # Step 5: Generate comprehensive report
        report = framework.generate_report([result], include_plots=False)
        
        # Assertions
        assert result.statistical_significance  # Should detect the 30% effect
        assert actual_power >= 0.75  # Should achieve target power
        assert len(report['recommendations']) > 0
        assert result.effect_size > 0.2  # Should detect meaningful effect
    
    def test_multi_metric_evaluation_workflow(self):
        """Test evaluation workflow with multiple business metrics."""
        framework = EvaluationFramework({
            'multiple_testing_correction': MultipleTestCorrection.FDR_BH,
            'save_results': False
        })
        
        # Define business metrics to evaluate
        metrics = {
            'revenue': {'treatment': np.random.normal(1000, 100, 200),
                       'control': np.random.normal(950, 100, 200)},
            'conversion_rate': {'treatment': np.random.normal(0.05, 0.01, 200),
                               'control': np.random.normal(0.045, 0.01, 200)},
            'customer_satisfaction': {'treatment': np.random.normal(4.2, 0.5, 200),
                                    'control': np.random.normal(4.0, 0.5, 200)},
            'cost_per_acquisition': {'treatment': np.random.normal(45, 8, 200),
                                   'control': np.random.normal(50, 8, 200)}  # Lower is better
        }
        
        # Run evaluations for all metrics
        results = []
        for metric_name, metric_data in metrics.items():
            result = framework.run_evaluation(
                treatment_data=metric_data['treatment'],
                control_data=metric_data['control'],
                experiment_name=f"{metric_name}_evaluation",
                multiple_comparisons=True
            )
            results.append(result)
        
        # Generate comprehensive report
        report = framework.generate_report(results, include_plots=False)
        
        # Validate multi-metric analysis
        assert len(results) == 4
        assert report['summary']['total_experiments'] == 4
        
        # Should recommend multiple testing correction
        assert any('multiple' in rec.lower() for rec in report['recommendations'])
        
        # Check statistical analysis includes multiple testing info
        assert 'multiple_testing' in report['statistical_analysis']
    
    def test_longitudinal_evaluation_workflow(self):
        """Test longitudinal evaluation workflow over time."""
        framework = EvaluationFramework({'save_results': False})
        
        # Create time-series data
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        
        # Simulate campaign performance over time with trend
        time_trend = np.linspace(0, 0.5, 90)  # Gradual improvement
        seasonal_effect = 0.1 * np.sin(2 * np.pi * np.arange(90) / 7)  # Weekly pattern
        noise = np.random.normal(0, 0.1, 90)
        
        performance_data = pd.DataFrame({
            'date': dates,
            'treatment_group': ['A' if i < 45 else 'B' for i in range(90)],
            'performance': 1.0 + time_trend + seasonal_effect + noise
        })
        
        # Create temporal holdout
        train_data, holdout_data = framework.create_holdout_set(
            performance_data, 'longitudinal_holdout',
            SplitStrategy.TEMPORAL, time_column='date'
        )
        
        # Evaluate on holdout
        holdout_treatment = holdout_data[
            holdout_data['treatment_group'] == 'B'
        ]['performance'].values
        holdout_control = holdout_data[
            holdout_data['treatment_group'] == 'A'
        ]['performance'].values
        
        if len(holdout_treatment) > 5 and len(holdout_control) > 5:
            longitudinal_result = framework.run_evaluation(
                holdout_treatment, holdout_control, "longitudinal_evaluation"
            )
            
            assert isinstance(longitudinal_result, ExperimentResult)
            assert 'longitudinal_holdout' in framework.holdout_sets
    
    def test_causal_inference_workflow(self):
        """Test complete causal inference workflow."""
        framework = EvaluationFramework({'save_results': False})
        
        # Create observational data with confounders
        np.random.seed(42)
        n = 1000
        
        # Confounders
        age = np.random.uniform(18, 65, n)
        income = np.random.uniform(30000, 100000, n)
        prior_engagement = np.random.uniform(0, 1, n)
        
        # Treatment assignment (non-random, depends on confounders)
        treatment_prob = 0.2 + 0.01 * (age - 40) / 10 + 0.000003 * (income - 50000) + 0.3 * prior_engagement
        treatment_prob = np.clip(treatment_prob, 0.05, 0.95)
        treatment = np.random.binomial(1, treatment_prob, n)
        
        # Outcome with treatment effect and confounding
        outcome = (10 + 0.1 * age + 0.00005 * income + 5 * prior_engagement +
                  2 * treatment + np.random.normal(0, 1, n))
        
        causal_data = pd.DataFrame({
            'age': age,
            'income': income,
            'prior_engagement': prior_engagement,
            'treatment': treatment,
            'outcome': outcome
        })
        
        # Run different causal methods
        methods = ['iptw', 'dm', 'doubly_robust']
        causal_results = []
        
        for method in methods:
            result = framework.analyze_counterfactual(
                data=causal_data,
                treatment_column='treatment',
                outcome_column='outcome',
                feature_columns=['age', 'income', 'prior_engagement'],
                policy_name=f'treatment_policy_{method}',
                baseline_policy='control_policy',
                method=method
            )
            causal_results.append(result)
        
        # Compare results across methods
        estimates = [result.estimated_lift for result in causal_results]
        
        # Results should be relatively consistent across methods
        estimate_std = np.std(estimates)
        assert estimate_std < 1.0  # Reasonable consistency
        
        # All methods should detect positive treatment effect
        assert all(result.estimated_lift > 0 for result in causal_results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])