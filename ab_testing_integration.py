#!/usr/bin/env python3
"""
A/B TESTING INTEGRATION FOR GAELP

Integrates the statistical A/B testing framework with GAELP's RL training system.
Provides seamless policy comparison, adaptive traffic allocation, and statistical
monitoring of RL policy performance.

Key Features:
- Automatic policy variant creation from RL agent configurations
- Real-time statistical monitoring during training
- Multi-metric evaluation (ROAS, conversion rate, LTV, CTR)
- Contextual bandits for segment-specific policy selection
- Integration with existing GAELP components
"""

import asyncio
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import uuid
from concurrent.futures import ThreadPoolExecutor

# GAELP imports
from statistical_ab_testing_framework import (
    StatisticalABTestFramework, StatisticalConfig, TestType, 
    AllocationStrategy, SignificanceTest, TestResults
)
from fortified_rl_agent import FortifiedRLAgent, EnrichedJourneyState
from discovery_engine import GA4DiscoveryEngine
from dynamic_segment_integration import get_discovered_segments, validate_no_hardcoded_segments
from attribution_models import AttributionEngine
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from gaelp_parameter_manager import ParameterManager

logger = logging.getLogger(__name__)

# Validate no hardcoded segments
validate_no_hardcoded_segments("ab_testing_integration")


@dataclass
class PolicyConfiguration:
    """Configuration for an RL policy variant"""
    policy_id: str
    name: str
    learning_rate: float
    epsilon: float
    gamma: float
    buffer_size: int
    network_architecture: Dict[str, Any]
    exploration_strategy: str
    reward_weights: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'policy_id': self.policy_id,
            'name': self.name,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'buffer_size': self.buffer_size,
            'network_architecture': self.network_architecture,
            'exploration_strategy': self.exploration_strategy,
            'reward_weights': self.reward_weights
        }


@dataclass  
class PolicyPerformanceMetrics:
    """Performance metrics for a policy variant"""
    policy_id: str
    
    # Core metrics
    total_episodes: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0
    
    # Business metrics
    conversion_rate: float = 0.0
    roas: float = 0.0
    ctr: float = 0.0
    ltv: float = 0.0
    
    # RL-specific metrics
    exploration_rate: float = 0.0
    q_value_mean: float = 0.0
    loss_mean: float = 0.0
    
    # Segment-specific performance
    segment_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Temporal performance
    hourly_performance: Dict[int, float] = field(default_factory=dict)
    daily_performance: Dict[str, float] = field(default_factory=dict)
    
    def update_metrics(self, episode_data: Dict[str, Any]):
        """Update metrics with new episode data"""
        self.total_episodes += 1
        episode_reward = episode_data.get('total_reward', 0.0)
        self.total_reward += episode_reward
        self.average_reward = self.total_reward / self.total_episodes
        
        # Update business metrics
        if 'conversion_rate' in episode_data:
            self.conversion_rate = (
                (self.conversion_rate * (self.total_episodes - 1) + episode_data['conversion_rate']) / 
                self.total_episodes
            )
        
        if 'roas' in episode_data:
            self.roas = (
                (self.roas * (self.total_episodes - 1) + episode_data['roas']) / 
                self.total_episodes
            )
        
        if 'ctr' in episode_data:
            self.ctr = (
                (self.ctr * (self.total_episodes - 1) + episode_data['ctr']) / 
                self.total_episodes
            )
        
        if 'ltv' in episode_data:
            self.ltv = (
                (self.ltv * (self.total_episodes - 1) + episode_data['ltv']) / 
                self.total_episodes
            )
        
        # Update segment performance
        segment = episode_data.get('segment', 'unknown')
        if segment not in self.segment_performance:
            self.segment_performance[segment] = {
                'episodes': 0, 'total_reward': 0.0, 'conversion_rate': 0.0
            }
        
        seg_perf = self.segment_performance[segment]
        seg_perf['episodes'] += 1
        seg_perf['total_reward'] += episode_reward
        if 'conversion_rate' in episode_data:
            seg_perf['conversion_rate'] = (
                (seg_perf['conversion_rate'] * (seg_perf['episodes'] - 1) + 
                 episode_data['conversion_rate']) / seg_perf['episodes']
            )
        
        # Update temporal performance
        hour = datetime.now().hour
        if hour not in self.hourly_performance:
            self.hourly_performance[hour] = 0.0
        self.hourly_performance[hour] = (
            (self.hourly_performance[hour] * (self.total_episodes - 1) + episode_reward) / 
            self.total_episodes
        )


class GAELPABTestingIntegration:
    """
    Integration layer between GAELP RL system and statistical A/B testing framework
    """
    
    def __init__(self, 
                 statistical_framework: StatisticalABTestFramework,
                 discovery_engine: GA4DiscoveryEngine,
                 attribution_engine: AttributionEngine,
                 budget_pacer: BudgetPacer,
                 identity_resolver: IdentityResolver,
                 parameter_manager: ParameterManager):
        
        self.statistical_framework = statistical_framework
        self.discovery = discovery_engine
        self.attribution = attribution_engine
        self.budget_pacer = budget_pacer
        self.identity_resolver = identity_resolver
        self.parameter_manager = parameter_manager
        
        # Policy management
        self.policy_variants: Dict[str, PolicyConfiguration] = {}
        self.active_agents: Dict[str, FortifiedRLAgent] = {}
        self.performance_metrics: Dict[str, PolicyPerformanceMetrics] = {}
        
        # Test tracking
        self.active_policy_tests: Dict[str, str] = {}  # policy_id -> test_id
        self.test_to_policies: Dict[str, List[str]] = {}  # test_id -> [policy_ids]
        
        # Traffic allocation history
        self.allocation_history: List[Dict[str, Any]] = []
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_interval = 300  # 5 minutes
        
        logger.info("GAELP A/B Testing Integration initialized")
    
    def create_policy_variant(self,
                            base_policy_config: Dict[str, Any],
                            modifications: Dict[str, Any],
                            variant_name: str) -> str:
        """
        Create a new policy variant by modifying base configuration
        """
        
        # Create unique policy ID
        policy_id = f"policy_{uuid.uuid4().hex[:8]}"
        
        # Merge base config with modifications
        variant_config = base_policy_config.copy()
        variant_config.update(modifications)
        
        # Create policy configuration
        policy_config = PolicyConfiguration(
            policy_id=policy_id,
            name=variant_name,
            learning_rate=variant_config.get('learning_rate', 1e-4),
            epsilon=variant_config.get('epsilon', 0.1),
            gamma=variant_config.get('gamma', 0.99),
            buffer_size=variant_config.get('buffer_size', 50000),
            network_architecture=variant_config.get('network_architecture', {}),
            exploration_strategy=variant_config.get('exploration_strategy', 'epsilon_greedy'),
            reward_weights=variant_config.get('reward_weights', {
                'conversion': 1.0, 'roas': 0.3, 'ctr': 0.1, 'ltv': 0.2
            })
        )
        
        # Initialize RL agent with this configuration
        agent = self._create_rl_agent(policy_config)
        
        # Store configurations
        self.policy_variants[policy_id] = policy_config
        self.active_agents[policy_id] = agent
        self.performance_metrics[policy_id] = PolicyPerformanceMetrics(policy_id)
        
        logger.info(f"Created policy variant {policy_id}: {variant_name}")
        return policy_id
    
    def _create_rl_agent(self, policy_config: PolicyConfiguration) -> FortifiedRLAgent:
        """Create RL agent with specified configuration"""
        
        agent = FortifiedRLAgent(
            discovery_engine=self.discovery,
            creative_selector=None,  # Will be provided during training
            attribution_engine=self.attribution,
            budget_pacer=self.budget_pacer,
            identity_resolver=self.identity_resolver,
            parameter_manager=self.parameter_manager,
            learning_rate=policy_config.learning_rate,
            epsilon=policy_config.epsilon,
            gamma=policy_config.gamma,
            buffer_size=policy_config.buffer_size
        )
        
        return agent
    
    def create_policy_comparison_test(self,
                                    policy_ids: List[str],
                                    test_name: str,
                                    test_type: TestType = TestType.BAYESIAN_BANDIT,
                                    allocation_strategy: AllocationStrategy = AllocationStrategy.ADAPTIVE_ALLOCATION,
                                    duration_days: int = 14,
                                    primary_metric: str = 'roas') -> str:
        """
        Create A/B test to compare multiple policy variants
        """
        
        if len(policy_ids) < 2:
            raise ValueError("At least 2 policies required for comparison")
        
        # Validate all policies exist
        for policy_id in policy_ids:
            if policy_id not in self.policy_variants:
                raise ValueError(f"Policy {policy_id} not found")
        
        # Create test variants
        variants = []
        for policy_id in policy_ids:
            policy_config = self.policy_variants[policy_id]
            variants.append({
                'variant_id': policy_id,
                'name': policy_config.name,
                'policy_parameters': policy_config.to_dict(),
                'allocation_probability': 1.0 / len(policy_ids)
            })
        
        # Create statistical test
        test_id = self.statistical_framework.create_ab_test(
            test_id=f"policy_comparison_{uuid.uuid4().hex[:8]}",
            test_name=test_name,
            variants=variants,
            test_type=test_type,
            allocation_strategy=allocation_strategy,
            duration_days=duration_days
        )
        
        # Track associations
        for policy_id in policy_ids:
            self.active_policy_tests[policy_id] = test_id
        self.test_to_policies[test_id] = policy_ids
        
        # Update statistical config for this test's primary metric
        self.statistical_framework.config.primary_metric = primary_metric
        
        logger.info(f"Created policy comparison test {test_id} for policies: {policy_ids}")
        return test_id
    
    def select_policy_for_episode(self, 
                                user_id: str, 
                                context: Dict[str, Any],
                                test_id: Optional[str] = None) -> Tuple[str, FortifiedRLAgent]:
        """
        Select policy variant for an episode using statistical framework
        """
        
        # If specific test provided
        if test_id and test_id in self.test_to_policies:
            policy_ids = self.test_to_policies[test_id]
        else:
            # Use all active policies in tests
            policy_ids = list(self.active_policy_tests.keys())
        
        if not policy_ids:
            raise ValueError("No active policy tests available")
        
        # Get test ID for allocation
        if test_id:
            allocation_test_id = test_id
        else:
            allocation_test_id = list(self.active_policy_tests.values())[0]
        
        # Enhance context with discovered patterns
        enhanced_context = self._enhance_context(context)
        
        # Assign variant using statistical framework
        selected_policy_id = self.statistical_framework.assign_variant(
            test_id=allocation_test_id,
            user_id=user_id,
            context=enhanced_context
        )
        
        if not selected_policy_id or selected_policy_id not in self.active_agents:
            # Use random selection if needed
            selected_policy_id = np.random.choice(policy_ids)
        
        selected_agent = self.active_agents[selected_policy_id]
        
        # Record allocation
        self.allocation_history.append({
            'user_id': user_id,
            'policy_id': selected_policy_id,
            'test_id': allocation_test_id,
            'timestamp': datetime.now(),
            'context': enhanced_context
        })
        
        return selected_policy_id, selected_agent
    
    def _enhance_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance context with discovered patterns and dynamic information"""
        enhanced = context.copy()
        
        # Add discovered segments
        segments = get_discovered_segments()
        if 'segment' not in enhanced and segments:
            # Default to first discovered segment
            enhanced['segment'] = segments[0]
        
        # Add temporal context
        now = datetime.now()
        enhanced['hour'] = now.hour
        enhanced['day_of_week'] = now.weekday()
        
        # Add discovered patterns
        try:
            patterns = self.discovery.discover_all_patterns()
            enhanced['seasonality_factor'] = patterns.temporal_patterns.get('seasonality_factor', 1.0)
            enhanced['is_peak_hour'] = now.hour in patterns.temporal_patterns.get('peak_hours', [])
        except Exception as e:
            logger.debug(f"Could not get patterns: {e}")
            enhanced['seasonality_factor'] = 1.0
            enhanced['is_peak_hour'] = False
        
        # Add budget context if available
        if self.budget_pacer:
            try:
                budget_info = self.budget_pacer.get_pacing_multiplier(
                    spent=context.get('budget_spent', 0),
                    budget=context.get('daily_budget', 1000),
                    time_remaining=context.get('time_remaining', 12)
                )
                enhanced['budget_pacing_factor'] = budget_info
                enhanced['budget_remaining_ratio'] = 1.0 - context.get('budget_spent', 0) / max(1, context.get('daily_budget', 1000))
            except Exception as e:
                logger.debug(f"Could not get budget info: {e}")
                enhanced['budget_pacing_factor'] = 1.0
                enhanced['budget_remaining_ratio'] = 1.0
        
        return enhanced
    
    def record_episode_result(self,
                            policy_id: str,
                            user_id: str,
                            episode_data: Dict[str, Any],
                            context: Dict[str, Any] = None):
        """
        Record episode results for statistical analysis and performance tracking
        """
        
        # Update policy performance metrics
        if policy_id in self.performance_metrics:
            self.performance_metrics[policy_id].update_metrics(episode_data)
        
        # Extract metrics for statistical framework
        primary_metric_value = episode_data.get('roas', episode_data.get('total_reward', 0.0))
        
        secondary_metrics = {
            'conversion_rate': episode_data.get('conversion_rate', 0.0),
            'ctr': episode_data.get('ctr', 0.0),
            'ltv': episode_data.get('ltv', 0.0),
            'total_reward': episode_data.get('total_reward', 0.0)
        }
        
        converted = episode_data.get('converted', episode_data.get('conversion_rate', 0.0) > 0)
        
        # Record in statistical framework
        if policy_id in self.active_policy_tests:
            test_id = self.active_policy_tests[policy_id]
            
            self.statistical_framework.record_observation(
                test_id=test_id,
                variant_id=policy_id,
                user_id=user_id,
                primary_metric_value=primary_metric_value,
                secondary_metrics=secondary_metrics,
                converted=converted,
                context=context or {}
            )
        
        logger.debug(f"Recorded episode result for policy {policy_id}")
    
    def analyze_policy_performance(self, test_id: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of policy performance in A/B test
        """
        
        if test_id not in self.test_to_policies:
            raise ValueError(f"Test {test_id} not found")
        
        # Get statistical analysis
        statistical_results = self.statistical_framework.analyze_test(test_id)
        
        # Get policy-specific metrics
        policy_ids = self.test_to_policies[test_id]
        policy_analysis = {}
        
        for policy_id in policy_ids:
            if policy_id in self.performance_metrics:
                metrics = self.performance_metrics[policy_id]
                policy_analysis[policy_id] = {
                    'total_episodes': metrics.total_episodes,
                    'average_reward': metrics.average_reward,
                    'conversion_rate': metrics.conversion_rate,
                    'roas': metrics.roas,
                    'ctr': metrics.ctr,
                    'ltv': metrics.ltv,
                    'segment_performance': metrics.segment_performance,
                    'policy_config': self.policy_variants[policy_id].to_dict()
                }
        
        # Combine results
        comprehensive_analysis = {
            'test_id': test_id,
            'statistical_results': {
                'p_value': statistical_results.p_value,
                'is_significant': statistical_results.is_significant,
                'confidence_interval': statistical_results.confidence_interval,
                'effect_size': statistical_results.effect_size,
                'bayesian_probability': statistical_results.bayesian_probability,
                'winner_variant_id': statistical_results.winner_variant_id,
                'recommended_action': statistical_results.recommended_action,
                'lift_percentage': statistical_results.lift_percentage
            },
            'policy_analysis': policy_analysis,
            'test_status': self.statistical_framework.get_test_status(test_id)
        }
        
        return comprehensive_analysis
    
    def get_segment_specific_recommendations(self, test_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get segment-specific policy recommendations
        """
        
        if test_id not in self.test_to_policies:
            return {}
        
        policy_ids = self.test_to_policies[test_id]
        segments = get_discovered_segments()
        
        segment_recommendations = {}
        
        for segment in segments:
            segment_performance = {}
            
            for policy_id in policy_ids:
                if policy_id in self.performance_metrics:
                    metrics = self.performance_metrics[policy_id]
                    if segment in metrics.segment_performance:
                        seg_perf = metrics.segment_performance[segment]
                        segment_performance[policy_id] = seg_perf
            
            if segment_performance:
                # Find best performing policy for this segment
                best_policy = max(segment_performance.keys(), 
                                key=lambda p: segment_performance[p].get('conversion_rate', 0))
                
                segment_recommendations[segment] = {
                    'recommended_policy': best_policy,
                    'policy_performance': segment_performance,
                    'confidence': 'high' if segment_performance[best_policy].get('episodes', 0) > 100 else 'low'
                }
        
        return segment_recommendations
    
    def start_continuous_monitoring(self):
        """Start background monitoring of active tests"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("Started continuous A/B test monitoring")
    
    def stop_continuous_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        logger.info("Stopped continuous A/B test monitoring")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Check all active tests
                for test_id in list(self.test_to_policies.keys()):
                    await self._check_test_status(test_id)
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _check_test_status(self, test_id: str):
        """Check status of a specific test"""
        try:
            # Get current status
            status = self.statistical_framework.get_test_status(test_id)
            
            # Check if we have sufficient data for analysis
            if status.get('progress', 0) > 0.1:  # At least 10% of required data
                results = self.statistical_framework.analyze_test(test_id)
                
                # Log significant findings
                if results.is_significant:
                    logger.info(f"Test {test_id} shows significant results: {results.recommended_action}")
                
                # Check for early stopping conditions
                if results.bayesian_probability > 0.99 or results.bayesian_probability < 0.01:
                    logger.info(f"Test {test_id} eligible for early stopping - very high confidence")
                
        except Exception as e:
            logger.error(f"Error checking test {test_id}: {e}")
    
    def generate_policy_insights(self, policy_id: str) -> Dict[str, Any]:
        """
        Generate insights about a specific policy's performance
        """
        
        if policy_id not in self.performance_metrics:
            return {'error': 'Policy not found'}
        
        metrics = self.performance_metrics[policy_id]
        config = self.policy_variants[policy_id]
        
        insights = {
            'policy_id': policy_id,
            'policy_name': config.name,
            'configuration': config.to_dict(),
            'performance_summary': {
                'total_episodes': metrics.total_episodes,
                'average_reward': metrics.average_reward,
                'conversion_rate': metrics.conversion_rate,
                'roas': metrics.roas,
                'ctr': metrics.ctr,
                'ltv': metrics.ltv
            },
            'segment_analysis': {},
            'temporal_analysis': {
                'hourly_performance': metrics.hourly_performance,
                'daily_performance': metrics.daily_performance
            },
            'recommendations': []
        }
        
        # Segment analysis
        best_segments = []
        worst_segments = []
        
        for segment, perf in metrics.segment_performance.items():
            insights['segment_analysis'][segment] = perf
            
            if perf.get('conversion_rate', 0) > metrics.conversion_rate * 1.2:
                best_segments.append(segment)
            elif perf.get('conversion_rate', 0) < metrics.conversion_rate * 0.8:
                worst_segments.append(segment)
        
        # Generate recommendations
        if best_segments:
            insights['recommendations'].append(f"Policy performs exceptionally well for: {', '.join(best_segments)}")
        
        if worst_segments:
            insights['recommendations'].append(f"Consider optimization for: {', '.join(worst_segments)}")
        
        if metrics.average_reward < 0:
            insights['recommendations'].append("Policy shows negative average reward - consider parameter adjustment")
        
        if config.epsilon > 0.2:
            insights['recommendations'].append("High exploration rate - consider reducing epsilon as policy learns")
        
        return insights
    
    def export_test_summary(self, test_id: str) -> str:
        """Export comprehensive test summary"""
        
        analysis = self.analyze_policy_performance(test_id)
        segment_recs = self.get_segment_specific_recommendations(test_id)
        
        summary = {
            'test_analysis': analysis,
            'segment_recommendations': segment_recs,
            'allocation_history': [
                h for h in self.allocation_history 
                if h.get('test_id') == test_id
            ][-100:],  # Last 100 allocations
            'export_timestamp': datetime.now().isoformat()
        }
        
        return json.dumps(summary, indent=2, default=str)


# Factory function for easy initialization
def create_gaelp_ab_testing_system(discovery_engine: GA4DiscoveryEngine,
                                   attribution_engine: AttributionEngine,
                                   budget_pacer: BudgetPacer,
                                   identity_resolver: IdentityResolver,
                                   parameter_manager: ParameterManager,
                                   config: Optional[StatisticalConfig] = None) -> GAELPABTestingIntegration:
    """
    Factory function to create complete GAELP A/B testing system
    """
    
    if config is None:
        config = StatisticalConfig(
            alpha=0.05,
            power=0.80,
            minimum_detectable_effect=0.05,
            minimum_sample_size=1000,
            primary_metric='roas',
            secondary_metrics=['conversion_rate', 'ctr', 'ltv']
        )
    
    # Create statistical framework
    statistical_framework = StatisticalABTestFramework(config, discovery_engine)
    
    # Create integration system
    integration = GAELPABTestingIntegration(
        statistical_framework=statistical_framework,
        discovery_engine=discovery_engine,
        attribution_engine=attribution_engine,
        budget_pacer=budget_pacer,
        identity_resolver=identity_resolver,
        parameter_manager=parameter_manager
    )
    
    return integration


if __name__ == "__main__":
    # Example usage
    from discovery_engine import GA4DiscoveryEngine
    from attribution_models import AttributionEngine
    from budget_pacer import BudgetPacer
    from identity_resolver import IdentityResolver
    from gaelp_parameter_manager import ParameterManager
    
    # Initialize components (simplified for example)
    discovery = GA4DiscoveryEngine()
    attribution = AttributionEngine()
    budget_pacer = BudgetPacer()
    identity_resolver = IdentityResolver()
    param_manager = ParameterManager()
    
    # Create A/B testing system
    ab_system = create_gaelp_ab_testing_system(
        discovery, attribution, budget_pacer, identity_resolver, param_manager
    )
    
    # Create policy variants
    base_config = {
        'learning_rate': 1e-4,
        'epsilon': 0.1,
        'gamma': 0.99,
        'buffer_size': 50000
    }
    
    policy_a = ab_system.create_policy_variant(
        base_config, 
        {'learning_rate': 1e-3, 'epsilon': 0.05}, 
        'High Learning Rate'
    )
    
    policy_b = ab_system.create_policy_variant(
        base_config,
        {'epsilon': 0.2, 'gamma': 0.95},
        'High Exploration'
    )
    
    # Create comparison test
    test_id = ab_system.create_policy_comparison_test(
        [policy_a, policy_b],
        'Learning Rate vs Exploration Test'
    )
    
    print(f"Created A/B test {test_id} comparing policies {policy_a} and {policy_b}")
    
    # Start monitoring
    ab_system.start_continuous_monitoring()
    
    print("A/B testing system initialized and monitoring started")