#!/usr/bin/env python3
"""
PRODUCTION A/B TESTING EXAMPLE FOR GAELP

Demonstrates how to use the comprehensive statistical A/B testing framework
in production with real GAELP RL policies. Shows proper setup, monitoring,
and decision-making based on statistical results.

This example shows:
1. Setting up policy variants with different RL configurations
2. Running statistical tests with proper sample sizes
3. Real-time monitoring and adaptive allocation
4. Multi-metric evaluation and segment analysis
5. Decision-making based on statistical significance
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional

# Import the A/B testing framework
from statistical_ab_testing_framework import (
    StatisticalABTestFramework, StatisticalConfig, TestType,
    AllocationStrategy, SignificanceTest
)
from ab_testing_integration import (
    create_gaelp_ab_testing_system, PolicyConfiguration
)

# Import GAELP components
from discovery_engine import GA4DiscoveryEngine
from dynamic_segment_integration import (
    get_discovered_segments, validate_no_hardcoded_segments
)
from attribution_models import AttributionEngine
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from gaelp_parameter_manager import ParameterManager

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Validate no hardcoded segments
validate_no_hardcoded_segments("production_ab_testing")


class ProductionABTestManager:
    """
    Production-ready A/B test manager for GAELP policies
    """
    
    def __init__(self):
        # Initialize GAELP components
        self.discovery = GA4DiscoveryEngine()
        self.attribution = AttributionEngine()
        self.budget_pacer = BudgetPacer()
        self.identity_resolver = IdentityResolver()
        self.parameter_manager = ParameterManager()
        
        # Configure statistical framework
        self.config = StatisticalConfig(
            alpha=0.05,  # 5% significance level
            power=0.80,  # 80% statistical power
            minimum_detectable_effect=0.05,  # 5% minimum detectable effect
            prior_conversion_rate=0.025,  # Expected baseline conversion
            minimum_sample_size=2000,  # Minimum observations per variant
            maximum_sample_size=50000,  # Maximum observations per variant
            confidence_level=0.95,
            
            # Bayesian parameters
            beta_prior_alpha=2.5,  # Prior successes
            beta_prior_beta=97.5,  # Prior failures
            
            # Multi-metric optimization
            primary_metric='roas',
            secondary_metrics=['conversion_rate', 'ctr', 'ltv', 'engagement_score'],
            metric_weights={
                'roas': 0.4,
                'conversion_rate': 0.25,
                'ctr': 0.15,
                'ltv': 0.15,
                'engagement_score': 0.05
            },
            
            # MAB parameters
            exploration_rate=0.10,
            ucb_confidence=2.0,
            thompson_sample_size=50000
        )
        
        # Create A/B testing system
        self.ab_system = create_gaelp_ab_testing_system(
            self.discovery, self.attribution, self.budget_pacer,
            self.identity_resolver, self.parameter_manager, self.config
        )
        
        # Test management
        self.active_tests = {}
        self.test_history = []
        
        logger.info("Production A/B Test Manager initialized")
    
    def create_learning_rate_optimization_test(self) -> str:
        """
        Create test to optimize learning rate parameter
        """
        logger.info("🧠 Creating Learning Rate Optimization Test")
        
        # Base configuration
        base_config = {
            'epsilon': 0.1,
            'gamma': 0.99,
            'buffer_size': 50000,
            'network_architecture': {
                'hidden_layers': [256, 128],
                'dropout': 0.1,
                'activation': 'relu'
            },
            'exploration_strategy': 'epsilon_greedy',
            'reward_weights': {
                'conversion': 1.0,
                'roas': 0.8,
                'ctr': 0.3,
                'ltv': 0.5,
                'engagement': 0.2
            }
        }
        
        # Create learning rate variants
        lr_conservative = self.ab_system.create_policy_variant(
            base_config,
            {'learning_rate': 1e-4},
            'Conservative Learning Rate (1e-4)'
        )
        
        lr_moderate = self.ab_system.create_policy_variant(
            base_config,
            {'learning_rate': 5e-4},
            'Moderate Learning Rate (5e-4)'
        )
        
        lr_aggressive = self.ab_system.create_policy_variant(
            base_config,
            {'learning_rate': 2e-3},
            'Aggressive Learning Rate (2e-3)'
        )
        
        # Create comparison test
        test_id = self.ab_system.create_policy_comparison_test(
            policy_ids=[lr_conservative, lr_moderate, lr_aggressive],
            test_name='Learning Rate Optimization',
            test_type=TestType.BAYESIAN_BANDIT,
            allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING,
            duration_days=21,
            primary_metric='roas'
        )
        
        self.active_tests[test_id] = {
            'name': 'Learning Rate Optimization',
            'type': 'learning_rate',
            'policies': [lr_conservative, lr_moderate, lr_aggressive],
            'start_date': datetime.now(),
            'expected_duration': 21
        }
        
        logger.info(f"✅ Created learning rate test: {test_id}")
        return test_id
    
    def create_exploration_strategy_test(self) -> str:
        """
        Create test to compare exploration strategies
        """
        logger.info("🔍 Creating Exploration Strategy Test")
        
        base_config = {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'buffer_size': 50000,
            'reward_weights': {
                'conversion': 1.0,
                'roas': 0.8,
                'ctr': 0.3,
                'ltv': 0.5
            }
        }
        
        # Epsilon-greedy with different epsilon values
        epsilon_low = self.ab_system.create_policy_variant(
            base_config,
            {'epsilon': 0.05, 'exploration_strategy': 'epsilon_greedy'},
            'Low Exploration (ε=0.05)'
        )
        
        epsilon_medium = self.ab_system.create_policy_variant(
            base_config,
            {'epsilon': 0.15, 'exploration_strategy': 'epsilon_greedy'},
            'Medium Exploration (ε=0.15)'
        )
        
        # UCB exploration
        ucb_policy = self.ab_system.create_policy_variant(
            base_config,
            {
                'exploration_strategy': 'ucb',
                'ucb_confidence': 2.0,
                'epsilon': 0.0  # No epsilon for UCB
            },
            'UCB Exploration'
        )
        
        test_id = self.ab_system.create_policy_comparison_test(
            policy_ids=[epsilon_low, epsilon_medium, ucb_policy],
            test_name='Exploration Strategy Comparison',
            test_type=TestType.BAYESIAN_BANDIT,
            allocation_strategy=AllocationStrategy.ADAPTIVE_ALLOCATION,
            duration_days=28,
            primary_metric='ltv'
        )
        
        self.active_tests[test_id] = {
            'name': 'Exploration Strategy Comparison',
            'type': 'exploration',
            'policies': [epsilon_low, epsilon_medium, ucb_policy],
            'start_date': datetime.now(),
            'expected_duration': 28
        }
        
        logger.info(f"✅ Created exploration strategy test: {test_id}")
        return test_id
    
    def create_reward_weighting_test(self) -> str:
        """
        Create test to optimize reward function weights
        """
        logger.info("🎯 Creating Reward Weighting Test")
        
        base_config = {
            'learning_rate': 1e-3,
            'epsilon': 0.1,
            'gamma': 0.99,
            'buffer_size': 50000
        }
        
        # Conversion-focused
        conversion_focused = self.ab_system.create_policy_variant(
            base_config,
            {
                'reward_weights': {
                    'conversion': 1.0,
                    'roas': 0.3,
                    'ctr': 0.1,
                    'ltv': 0.2
                }
            },
            'Conversion-Focused Rewards'
        )
        
        # ROAS-focused
        roas_focused = self.ab_system.create_policy_variant(
            base_config,
            {
                'reward_weights': {
                    'conversion': 0.5,
                    'roas': 1.0,
                    'ctr': 0.2,
                    'ltv': 0.3
                }
            },
            'ROAS-Focused Rewards'
        )
        
        # Balanced approach
        balanced_rewards = self.ab_system.create_policy_variant(
            base_config,
            {
                'reward_weights': {
                    'conversion': 0.7,
                    'roas': 0.7,
                    'ctr': 0.4,
                    'ltv': 0.6
                }
            },
            'Balanced Rewards'
        )
        
        test_id = self.ab_system.create_policy_comparison_test(
            policy_ids=[conversion_focused, roas_focused, balanced_rewards],
            test_name='Reward Function Optimization',
            test_type=TestType.BAYESIAN_BANDIT,
            allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING,
            duration_days=35,
            primary_metric='conversion_rate'
        )
        
        self.active_tests[test_id] = {
            'name': 'Reward Function Optimization',
            'type': 'reward_weights',
            'policies': [conversion_focused, roas_focused, balanced_rewards],
            'start_date': datetime.now(),
            'expected_duration': 35
        }
        
        logger.info(f"✅ Created reward weighting test: {test_id}")
        return test_id
    
    def simulate_production_traffic(self, test_id: str, n_episodes: int = 5000):
        """
        Simulate realistic production traffic for a test
        """
        logger.info(f"🚀 Simulating {n_episodes} production episodes for test {test_id}")
        
        # Get discovered segments dynamically
        segments = get_discovered_segments()
        if not segments:
            segments = ['discovered_segment_1', 'discovered_segment_2']
        
        # Realistic traffic patterns
        devices = ['mobile', 'desktop', 'tablet']
        device_weights = [0.65, 0.30, 0.05]  # Mobile-heavy traffic
        
        channels = ['organic', 'paid_search', 'social', 'display', 'email']
        channel_weights = [0.35, 0.25, 0.20, 0.15, 0.05]
        
        hours = list(range(24))
        hour_weights = [0.02 if h < 8 or h > 22 else 0.06 if 9 <= h <= 17 else 0.04 for h in hours]
        
        episode_results = []
        
        for i in range(n_episodes):
            # Generate realistic context
            segment = np.random.choice(segments)
            device = np.random.choice(devices, p=device_weights)
            channel = np.random.choice(channels, p=channel_weights)
            hour = np.random.choice(hours, p=hour_weights)
            day_of_week = (i // 100) % 7  # Simulate weekly patterns
            
            context = {
                'segment': segment,
                'device': device,
                'channel': channel,
                'hour': hour,
                'day_of_week': day_of_week,
                'budget_remaining_ratio': max(0.1, 1.0 - (i / n_episodes)),
                'competition_level': np.random.uniform(0.3, 0.8)
            }
            
            # Select policy
            selected_policy, agent = self.ab_system.select_policy_for_episode(
                f'user_{i}', context, test_id
            )
            
            # Simulate realistic performance based on context
            base_conversion_rate = self._get_base_conversion_rate(segment, device, channel)
            base_roas = self._get_base_roas(segment, device, channel)
            
            # Add policy-specific modifiers
            policy_modifier = self._get_policy_performance_modifier(selected_policy, context)
            
            # Generate results
            conversion_rate = base_conversion_rate * policy_modifier['conversion']
            converted = np.random.random() < conversion_rate
            
            if converted:
                roas = max(0, np.random.normal(base_roas * policy_modifier['roas'], base_roas * 0.3))
                ltv = max(0, np.random.normal(150 * policy_modifier['ltv'], 40))
                engagement_score = min(1.0, np.random.normal(0.7 * policy_modifier['engagement'], 0.15))
            else:
                roas = 0
                ltv = 0
                engagement_score = min(1.0, np.random.normal(0.3, 0.1))
            
            ctr = min(1.0, np.random.normal(0.05 * policy_modifier['ctr'], 0.015))
            
            # Record episode result
            episode_data = {
                'total_reward': roas * 10 + (100 if converted else 0),
                'roas': roas,
                'conversion_rate': conversion_rate,
                'converted': converted,
                'ctr': ctr,
                'ltv': ltv,
                'engagement_score': engagement_score,
                'segment': segment
            }
            
            self.ab_system.record_episode_result(
                selected_policy, f'user_{i}', episode_data, context
            )
            
            episode_results.append({
                'episode': i,
                'policy': selected_policy,
                'segment': segment,
                'device': device,
                'channel': channel,
                'converted': converted,
                'roas': roas,
                'ltv': ltv
            })
            
            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i + 1}/{n_episodes} episodes")
        
        logger.info(f"✅ Completed {n_episodes} episodes for test {test_id}")
        return episode_results
    
    def _get_base_conversion_rate(self, segment: str, device: str, channel: str) -> float:
        """Get realistic base conversion rate based on context"""
        base_rate = 0.025
        
        # Segment modifiers (using discovered segments)
        if 'high_intent' in segment.lower() or 'crisis' in segment.lower():
            base_rate *= 1.5
        elif 'research' in segment.lower() or 'exploring' in segment.lower():
            base_rate *= 0.8
        
        # Device modifiers
        if device == 'desktop':
            base_rate *= 1.2
        elif device == 'tablet':
            base_rate *= 0.9
        
        # Channel modifiers
        channel_modifiers = {
            'paid_search': 1.3,
            'email': 1.4,
            'organic': 1.0,
            'social': 0.8,
            'display': 0.7
        }
        base_rate *= channel_modifiers.get(channel, 1.0)
        
        return min(0.15, base_rate)  # Cap at 15%
    
    def _get_base_roas(self, segment: str, device: str, channel: str) -> float:
        """Get realistic base ROAS based on context"""
        base_roas = 3.2
        
        # Segment modifiers
        if 'high_value' in segment.lower():
            base_roas *= 1.3
        elif 'price_sensitive' in segment.lower():
            base_roas *= 0.8
        
        # Device modifiers
        if device == 'desktop':
            base_roas *= 1.1
        
        # Channel modifiers
        channel_modifiers = {
            'email': 1.2,
            'organic': 1.1,
            'paid_search': 1.0,
            'social': 0.9,
            'display': 0.8
        }
        base_roas *= channel_modifiers.get(channel, 1.0)
        
        return base_roas
    
    def _get_policy_performance_modifier(self, policy_id: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Get policy-specific performance modifiers"""
        # This would be based on the actual policy configuration
        # For simulation, we'll use realistic variations
        
        if 'conservative' in policy_id.lower() or 'low' in policy_id.lower():
            return {
                'conversion': 0.95,  # Slightly lower conversion
                'roas': 1.05,       # Slightly better ROAS
                'ctr': 0.98,
                'ltv': 1.02,
                'engagement': 1.0
            }
        elif 'aggressive' in policy_id.lower() or 'high' in policy_id.lower():
            return {
                'conversion': 1.08,  # Higher conversion
                'roas': 0.95,       # Slightly lower ROAS
                'ctr': 1.05,
                'ltv': 0.98,
                'engagement': 1.03
            }
        else:  # Balanced/moderate
            return {
                'conversion': 1.0,
                'roas': 1.0,
                'ctr': 1.0,
                'ltv': 1.0,
                'engagement': 1.0
            }
    
    def monitor_test_progress(self, test_id: str) -> Dict[str, Any]:
        """
        Monitor and analyze test progress
        """
        logger.info(f"📊 Monitoring test progress: {test_id}")
        
        # Get comprehensive analysis
        analysis = self.ab_system.analyze_policy_performance(test_id)
        
        # Get segment-specific recommendations
        segment_recs = self.ab_system.get_segment_specific_recommendations(test_id)
        
        # Get test status
        status = self.ab_system.statistical_framework.get_test_status(test_id)
        
        # Compile monitoring report
        monitoring_report = {
            'test_id': test_id,
            'test_info': self.active_tests.get(test_id, {}),
            'progress': status['progress'],
            'statistical_analysis': analysis['statistical_results'],
            'policy_performance': analysis['policy_analysis'],
            'segment_recommendations': segment_recs,
            'monitoring_timestamp': datetime.now().isoformat()
        }
        
        # Log key metrics
        logger.info(f"  Progress: {status['progress']:.1%}")
        logger.info(f"  Statistical significance: {analysis['statistical_results']['is_significant']}")
        if analysis['statistical_results']['winner_variant_id']:
            logger.info(f"  Current winner: {analysis['statistical_results']['winner_variant_id']}")
            logger.info(f"  Lift: {analysis['statistical_results']['lift_percentage']:.2f}%")
        
        return monitoring_report
    
    def make_deployment_decision(self, test_id: str) -> Dict[str, Any]:
        """
        Make statistically-informed deployment decision
        """
        logger.info(f"🎯 Making deployment decision for test: {test_id}")
        
        analysis = self.ab_system.analyze_policy_performance(test_id)
        statistical_results = analysis['statistical_results']
        
        decision = {
            'test_id': test_id,
            'decision_timestamp': datetime.now().isoformat(),
            'action': 'continue',  # Default action
            'rationale': '',
            'confidence': 'low',
            'recommended_policy': None,
            'rollout_percentage': 0,
            'monitoring_period': 7  # Days
        }
        
        # Decision logic based on statistical results
        if not statistical_results['is_significant']:
            if statistical_results.get('bayesian_probability', 0.5) > 0.90:
                decision.update({
                    'action': 'cautious_rollout',
                    'rationale': 'High Bayesian probability but not statistically significant',
                    'confidence': 'medium',
                    'recommended_policy': statistical_results['winner_variant_id'],
                    'rollout_percentage': 25,
                    'monitoring_period': 14
                })
            else:
                decision.update({
                    'action': 'continue_test',
                    'rationale': 'Insufficient statistical evidence for deployment',
                    'confidence': 'high',
                    'monitoring_period': 7
                })
        
        else:  # Statistically significant
            if statistical_results['lift_percentage'] > 10:
                decision.update({
                    'action': 'full_rollout',
                    'rationale': f'Statistically significant with {statistical_results["lift_percentage"]:.1f}% lift',
                    'confidence': 'high',
                    'recommended_policy': statistical_results['winner_variant_id'],
                    'rollout_percentage': 100,
                    'monitoring_period': 30
                })
            elif statistical_results['lift_percentage'] > 5:
                decision.update({
                    'action': 'gradual_rollout',
                    'rationale': f'Statistically significant with {statistical_results["lift_percentage"]:.1f}% lift',
                    'confidence': 'high',
                    'recommended_policy': statistical_results['winner_variant_id'],
                    'rollout_percentage': 50,
                    'monitoring_period': 21
                })
            else:
                decision.update({
                    'action': 'cautious_rollout',
                    'rationale': 'Statistically significant but small effect size',
                    'confidence': 'medium',
                    'recommended_policy': statistical_results['winner_variant_id'],
                    'rollout_percentage': 30,
                    'monitoring_period': 14
                })
        
        logger.info(f"  Decision: {decision['action']}")
        logger.info(f"  Rationale: {decision['rationale']}")
        if decision['recommended_policy']:
            logger.info(f"  Recommended policy: {decision['recommended_policy']}")
        
        return decision
    
    def export_test_report(self, test_id: str) -> str:
        """
        Export comprehensive test report
        """
        logger.info(f"📄 Exporting comprehensive test report: {test_id}")
        
        # Get all analysis data
        analysis = self.ab_system.analyze_policy_performance(test_id)
        segment_recs = self.ab_system.get_segment_specific_recommendations(test_id)
        decision = self.make_deployment_decision(test_id)
        
        # Create comprehensive report
        report = {
            'test_overview': {
                'test_id': test_id,
                'test_info': self.active_tests.get(test_id, {}),
                'export_timestamp': datetime.now().isoformat()
            },
            'statistical_analysis': analysis['statistical_results'],
            'policy_performance': analysis['policy_analysis'],
            'segment_analysis': segment_recs,
            'deployment_recommendation': decision,
            'test_summary': self.ab_system.export_test_summary(test_id)
        }
        
        # Format as JSON
        report_json = json.dumps(report, indent=2, default=str)
        
        # Save to file
        filename = f"ab_test_report_{test_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            f.write(report_json)
        
        logger.info(f"✅ Test report exported to: {filename}")
        return report_json


async def run_production_example():
    """
    Run complete production example
    """
    logger.info("🚀 Starting Production A/B Testing Example")
    logger.info("=" * 80)
    
    # Initialize manager
    manager = ProductionABTestManager()
    
    # Create different types of tests
    logger.info("📝 Creating Production A/B Tests...")
    
    lr_test_id = manager.create_learning_rate_optimization_test()
    exploration_test_id = manager.create_exploration_strategy_test()
    reward_test_id = manager.create_reward_weighting_test()
    
    # Start monitoring
    manager.ab_system.start_continuous_monitoring()
    
    # Simulate production traffic for one test
    logger.info("🌊 Simulating production traffic...")
    manager.simulate_production_traffic(lr_test_id, n_episodes=3000)
    
    # Monitor progress
    logger.info("📊 Monitoring test progress...")
    monitoring_report = manager.monitor_test_progress(lr_test_id)
    
    # Make deployment decision
    logger.info("🎯 Making deployment decision...")
    decision = manager.make_deployment_decision(lr_test_id)
    
    # Export comprehensive report
    logger.info("📄 Exporting test report...")
    report = manager.export_test_report(lr_test_id)
    
    # Summary
    logger.info("=" * 80)
    logger.info("🎉 Production A/B Testing Example Complete!")
    logger.info(f"  Learning Rate Test: {lr_test_id}")
    logger.info(f"  Exploration Test: {exploration_test_id}")
    logger.info(f"  Reward Weighting Test: {reward_test_id}")
    logger.info(f"  Deployment Decision: {decision['action']}")
    if decision['recommended_policy']:
        logger.info(f"  Winning Policy: {decision['recommended_policy']}")
    logger.info("=" * 80)
    
    return {
        'lr_test_id': lr_test_id,
        'exploration_test_id': exploration_test_id,
        'reward_test_id': reward_test_id,
        'decision': decision,
        'monitoring_report': monitoring_report
    }


if __name__ == '__main__':
    asyncio.run(run_production_example())