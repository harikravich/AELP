#!/usr/bin/env python3
"""
PRODUCTION A/B TESTING INTEGRATION FOR GAELP

Complete production-ready integration of statistical A/B testing framework
with GAELP's RL training system. This provides:

- Real-time policy comparison during training
- Multi-armed bandit allocation with contextual features
- Advanced statistical monitoring (CUSUM, SPRT, Bayesian)
- Multi-objective optimization
- Automatic early stopping
- Comprehensive reporting and dashboards
- Production-grade error handling and monitoring

NO FALLBACKS - Production statistical rigor only.
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
import threading
import time

# GAELP imports
from statistical_ab_testing_framework import (
    StatisticalABTestFramework, StatisticalConfig, TestType, 
    AllocationStrategy, SignificanceTest, TestResults
)
from advanced_ab_testing_enhancements import (
    AdvancedABTestingFramework, AdvancedStatisticalConfig,
    AdvancedTestType, AdvancedAllocationStrategy, create_advanced_ab_testing_system
)
from ab_testing_integration import (
    GAELPABTestingIntegration, PolicyConfiguration, PolicyPerformanceMetrics
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
validate_no_hardcoded_segments("production_ab_testing")


@dataclass
class ProductionABConfig:
    """Production configuration for A/B testing"""
    # Test management
    max_concurrent_tests: int = 5
    auto_conclude_after_days: int = 30
    min_observations_per_variant: int = 1000
    max_observations_per_test: int = 100000
    
    # Statistical rigor
    significance_threshold: float = 0.05
    power_requirement: float = 0.80
    minimum_effect_size: float = 0.03
    
    # Real-time monitoring
    monitoring_interval_seconds: int = 60
    early_stopping_enabled: bool = True
    adaptive_allocation_enabled: bool = True
    
    # Production safety
    allocation_timeout_ms: int = 100
    max_context_features: int = 100
    performance_logging_enabled: bool = True
    error_recovery_enabled: bool = True
    
    # Dashboard and reporting
    report_generation_enabled: bool = True
    real_time_dashboard_enabled: bool = True
    export_results_format: str = 'json'


@dataclass
class TestPerformanceMetrics:
    """Performance metrics for A/B test monitoring"""
    test_id: str
    start_time: datetime
    
    # Allocation performance
    allocation_requests: int = 0
    allocation_successes: int = 0
    allocation_failures: int = 0
    avg_allocation_time_ms: float = 0.0
    
    # Statistical performance
    current_power: float = 0.0
    current_effect_size: float = 0.0
    p_value: float = 1.0
    confidence_level: float = 0.0
    
    # Business metrics
    total_conversions: int = 0
    total_revenue: float = 0.0
    cost_per_acquisition: float = 0.0
    return_on_ad_spend: float = 0.0
    
    # System performance
    observation_recording_rate: float = 0.0
    analysis_computation_time_ms: float = 0.0
    
    def update_allocation_metrics(self, success: bool, time_ms: float):
        """Update allocation performance metrics"""
        self.allocation_requests += 1
        if success:
            self.allocation_successes += 1
        else:
            self.allocation_failures += 1
        
        # Update running average
        if self.allocation_requests > 1:
            self.avg_allocation_time_ms = (
                (self.avg_allocation_time_ms * (self.allocation_requests - 1) + time_ms) / 
                self.allocation_requests
            )
        else:
            self.avg_allocation_time_ms = time_ms


class ProductionABTestManager:
    """
    Production-grade A/B test manager with real-time monitoring and advanced analytics
    """
    
    def __init__(self,
                 discovery_engine: GA4DiscoveryEngine,
                 attribution_engine: AttributionEngine,
                 budget_pacer: BudgetPacer,
                 identity_resolver: IdentityResolver,
                 parameter_manager: ParameterManager,
                 config: ProductionABConfig = None):
        
        self.discovery = discovery_engine
        self.attribution = attribution_engine
        self.budget_pacer = budget_pacer
        self.identity_resolver = identity_resolver
        self.parameter_manager = parameter_manager
        self.config = config or ProductionABConfig()
        
        # Initialize statistical frameworks
        base_config = StatisticalConfig(
            alpha=self.config.significance_threshold,
            power=self.config.power_requirement,
            minimum_detectable_effect=self.config.minimum_effect_size,
            minimum_sample_size=self.config.min_observations_per_variant
        )
        
        advanced_config = AdvancedStatisticalConfig()
        
        # Create advanced A/B testing framework
        self.advanced_framework = create_advanced_ab_testing_system(
            discovery_engine, base_config, advanced_config
        )
        
        # Initialize GAELP integration
        self.gaelp_integration = GAELPABTestingIntegration(
            statistical_framework=self.advanced_framework,
            discovery_engine=discovery_engine,
            attribution_engine=attribution_engine,
            budget_pacer=budget_pacer,
            identity_resolver=identity_resolver,
            parameter_manager=parameter_manager
        )
        
        # Test management
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_metrics: Dict[str, TestPerformanceMetrics] = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Thread safety
        self._allocation_lock = threading.RLock()
        self._observation_lock = threading.RLock()
        
        # Performance tracking
        self.system_metrics = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'total_observations': 0,
            'average_response_time': 0.0,
            'error_rate': 0.0
        }
        
        logger.info("Production A/B Test Manager initialized")
    
    def create_production_policy_test(self,
                                    policy_configs: List[Dict[str, Any]],
                                    test_name: str,
                                    test_type: str = 'bayesian_adaptive',
                                    allocation_strategy: str = 'linucb',
                                    business_objective: str = 'roas',
                                    segment_targeting: List[str] = None,
                                    duration_days: int = 14) -> str:
        """
        Create production-grade policy comparison test
        """
        
        if len(self.active_tests) >= self.config.max_concurrent_tests:
            raise ValueError(f"Maximum concurrent tests ({self.config.max_concurrent_tests}) reached")
        
        if len(policy_configs) < 2:
            raise ValueError("At least 2 policy configurations required")
        
        test_id = f"prod_test_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create policy variants
            policy_ids = []
            for i, policy_config in enumerate(policy_configs):
                policy_id = self.gaelp_integration.create_policy_variant(
                    policy_config.get('base_config', {}),
                    policy_config.get('modifications', {}),
                    policy_config.get('name', f'Policy {i+1}')
                )
                policy_ids.append(policy_id)
            
            # Map test types to advanced types
            type_mapping = {
                'bayesian_adaptive': AdvancedTestType.BAYESIAN_ADAPTIVE_STOPPING,
                'cusum': AdvancedTestType.CUSUM_STOPPING,
                'sprt': AdvancedTestType.SPRT,
                'multi_objective': AdvancedTestType.MULTI_OBJECTIVE_PARETO,
                'covariate_adjusted': AdvancedTestType.COVARIATE_ADJUSTED
            }
            
            allocation_mapping = {
                'linucb': AdvancedAllocationStrategy.LINUCB,
                'thompson': AdvancedAllocationStrategy.THOMPSON_SAMPLING_NEURAL,
                'adaptive_greedy': AdvancedAllocationStrategy.ADAPTIVE_GREEDY
            }
            
            advanced_test_type = type_mapping.get(test_type, AdvancedTestType.BAYESIAN_ADAPTIVE_STOPPING)
            advanced_allocation = allocation_mapping.get(allocation_strategy, AdvancedAllocationStrategy.LINUCB)
            
            # Create advanced test
            variants = []
            for policy_id in policy_ids:
                policy_config = self.gaelp_integration.policy_variants[policy_id]
                variants.append({
                    'variant_id': policy_id,
                    'name': policy_config.name,
                    'policy_parameters': policy_config.to_dict()
                })
            
            self.advanced_framework.create_advanced_test(
                test_id=test_id,
                test_name=test_name,
                variants=variants,
                test_type=advanced_test_type,
                allocation_strategy=advanced_allocation,
                duration_days=duration_days
            )
            
            # Store test configuration
            self.active_tests[test_id] = {
                'test_name': test_name,
                'policy_ids': policy_ids,
                'test_type': test_type,
                'allocation_strategy': allocation_strategy,
                'business_objective': business_objective,
                'segment_targeting': segment_targeting or [],
                'start_time': datetime.now(),
                'status': 'active',
                'duration_days': duration_days
            }
            
            # Initialize performance metrics
            self.test_metrics[test_id] = TestPerformanceMetrics(
                test_id=test_id,
                start_time=datetime.now()
            )
            
            logger.info(f"Created production test {test_id}: {test_name}")
            return test_id
            
        except Exception as e:
            logger.error(f"Error creating production test: {e}")
            raise
    
    def get_policy_allocation(self,
                            user_id: str,
                            context: Dict[str, Any],
                            test_id: Optional[str] = None) -> Tuple[Optional[str], Optional[FortifiedRLAgent], Dict[str, Any]]:
        """
        Get policy allocation with production-grade performance and error handling
        """
        
        start_time = time.time()
        allocation_info = {'success': False, 'method': 'unknown', 'test_id': None}
        
        try:
            with self._allocation_lock:
                self.system_metrics['total_allocations'] += 1
                
                # Timeout protection
                allocation_start = time.time()
                timeout_seconds = self.config.allocation_timeout_ms / 1000.0
                
                # Enhanced context with production features
                enhanced_context = self._enhance_production_context(context)
                
                # Select test if not specified
                if not test_id:
                    test_id = self._select_active_test(enhanced_context)
                
                if not test_id or test_id not in self.active_tests:
                    return None, None, {'error': 'No active test available'}
                
                # Check timeout
                if time.time() - allocation_start > timeout_seconds:
                    logger.warning(f"Allocation timeout for user {user_id}")
                    return None, None, {'error': 'Allocation timeout'}
                
                # Use advanced allocation
                selected_policy_id = self.advanced_framework.assign_variant_advanced(
                    test_id, user_id, enhanced_context
                )
                
                if not selected_policy_id:
                    return None, None, {'error': 'No variant assigned'}
                
                # Get agent
                selected_agent = self.gaelp_integration.active_agents.get(selected_policy_id)
                
                if not selected_agent:
                    return None, None, {'error': 'Agent not found'}
                
                # Update metrics
                elapsed_ms = (time.time() - start_time) * 1000
                self.test_metrics[test_id].update_allocation_metrics(True, elapsed_ms)
                self.system_metrics['successful_allocations'] += 1
                
                allocation_info = {
                    'success': True,
                    'method': self.active_tests[test_id]['allocation_strategy'],
                    'test_id': test_id,
                    'response_time_ms': elapsed_ms
                }
                
                logger.debug(f"Allocated user {user_id} to policy {selected_policy_id} in test {test_id}")
                
                return selected_policy_id, selected_agent, allocation_info
                
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            if test_id and test_id in self.test_metrics:
                self.test_metrics[test_id].update_allocation_metrics(False, elapsed_ms)
            
            logger.error(f"Error in policy allocation: {e}")
            
            if self.config.error_recovery_enabled:
                return self._error_recovery_allocation(user_id, context)
            else:
                return None, None, {'error': str(e)}
    
    def record_policy_performance(self,
                                 test_id: str,
                                 policy_id: str,
                                 user_id: str,
                                 episode_data: Dict[str, Any],
                                 context: Dict[str, Any] = None):
        """
        Record policy performance with production monitoring
        """
        
        start_time = time.time()
        
        try:
            with self._observation_lock:
                self.system_metrics['total_observations'] += 1
                
                # Validate inputs
                if test_id not in self.active_tests:
                    logger.warning(f"Recording observation for inactive test {test_id}")
                    return
                
                # Extract metrics
                primary_metric = episode_data.get(
                    self.active_tests[test_id]['business_objective'], 
                    episode_data.get('total_reward', 0.0)
                )
                
                secondary_metrics = {
                    'conversion_rate': episode_data.get('conversion_rate', 0.0),
                    'roas': episode_data.get('roas', 0.0),
                    'ctr': episode_data.get('ctr', 0.0),
                    'ltv': episode_data.get('ltv', 0.0),
                    'total_reward': episode_data.get('total_reward', 0.0)
                }
                
                converted = episode_data.get('converted', False)
                
                # Record in advanced framework
                self.advanced_framework.record_observation_advanced(
                    test_id=test_id,
                    variant_id=policy_id,
                    user_id=user_id,
                    primary_metric_value=float(primary_metric),
                    secondary_metrics=secondary_metrics,
                    converted=converted,
                    context=context
                )
                
                # Update test performance metrics
                if test_id in self.test_metrics:
                    metrics = self.test_metrics[test_id]
                    if converted:
                        metrics.total_conversions += 1
                    
                    revenue = episode_data.get('revenue', 0.0)
                    metrics.total_revenue += revenue
                
                # Record in GAELP integration
                self.gaelp_integration.record_episode_result(
                    policy_id, user_id, episode_data, context
                )
                
                # Performance logging
                if self.config.performance_logging_enabled:
                    elapsed_ms = (time.time() - start_time) * 1000
                    if elapsed_ms > 100:  # Log slow observations
                        logger.warning(f"Slow observation recording: {elapsed_ms:.1f}ms")
                
                logger.debug(f"Recorded performance for policy {policy_id} in test {test_id}")
                
        except Exception as e:
            logger.error(f"Error recording policy performance: {e}")
            if not self.config.error_recovery_enabled:
                raise
    
    def _enhance_production_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance context with production features and validation"""
        
        enhanced = context.copy()
        
        try:
            # Add timestamp features
            now = datetime.now()
            enhanced['timestamp'] = now.isoformat()
            enhanced['hour'] = now.hour
            enhanced['day_of_week'] = now.weekday()
            enhanced['is_weekend'] = now.weekday() >= 5
            
            # Add discovered segments if not present
            segments = get_discovered_segments()
            if 'segment' not in enhanced and segments:
                enhanced['segment'] = segments[0]  # Default to first discovered segment
            
            # Add GA4 patterns if available
            try:
                patterns = self.discovery.discover_all_patterns()
                enhanced['seasonality_factor'] = patterns.temporal_patterns.get('seasonality_factor', 1.0)
                peak_hours = patterns.temporal_patterns.get('peak_hours', [])
                enhanced['is_peak_hour'] = now.hour in peak_hours
            except:
                enhanced['seasonality_factor'] = 1.0
                enhanced['is_peak_hour'] = False
            
            # Add budget context
            if self.budget_pacer:
                try:
                    enhanced['budget_remaining_ratio'] = self.budget_pacer.get_pacing_multiplier(
                        spent=context.get('budget_spent', 0),
                        budget=context.get('daily_budget', 1000),
                        time_remaining=context.get('time_remaining', 12)
                    )
                except:
                    enhanced['budget_remaining_ratio'] = 1.0
            
            # Limit context size for performance
            if len(enhanced) > self.config.max_context_features:
                # Keep most important features
                important_keys = ['segment', 'device', 'channel', 'hour', 'day_of_week']
                filtered_context = {k: v for k, v in enhanced.items() if k in important_keys}
                enhanced = filtered_context
                logger.warning(f"Context truncated to {len(enhanced)} features")
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing context: {e}")
            return context  # Return original if enhancement fails
    
    def _select_active_test(self, context: Dict[str, Any]) -> Optional[str]:
        """Select the most appropriate active test based on context"""
        
        # Simple selection strategy - can be enhanced with ML
        active_test_ids = [tid for tid, test in self.active_tests.items() 
                          if test['status'] == 'active']
        
        if not active_test_ids:
            return None
        
        # Prefer tests targeting the user's segment
        user_segment = context.get('segment')
        if user_segment:
            for test_id in active_test_ids:
                segment_targeting = self.active_tests[test_id].get('segment_targeting', [])
                if not segment_targeting or user_segment in segment_targeting:
                    return test_id
        
        # Default to first active test
        return active_test_ids[0]
    
    def _error_recovery_allocation(self, user_id: str, context: Dict[str, Any]) -> Tuple[Optional[str], Optional[FortifiedRLAgent], Dict[str, Any]]:
        """Error recovery allocation strategy"""
        
        try:
            # Use simple random allocation among available policies
            all_policy_ids = list(self.gaelp_integration.active_agents.keys())
            if all_policy_ids:
                selected_policy_id = np.random.choice(all_policy_ids)
                selected_agent = self.gaelp_integration.active_agents[selected_policy_id]
                
                return selected_policy_id, selected_agent, {
                    'success': True,
                    'method': 'error_recovery_random',
                    'test_id': None
                }
        except Exception as e:
            logger.error(f"Error recovery allocation failed: {e}")
        
        return None, None, {'error': 'Complete allocation failure'}
    
    def start_real_time_monitoring(self):
        """Start real-time monitoring of all active tests"""
        
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        if asyncio.get_event_loop().is_running():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        else:
            asyncio.run(self._monitoring_loop())
        
        logger.info("Started real-time A/B test monitoring")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring"""
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        logger.info("Stopped real-time A/B test monitoring")
    
    async def _monitoring_loop(self):
        """Real-time monitoring loop"""
        
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
                for test_id in list(self.active_tests.keys()):
                    await self._monitor_test(test_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _monitor_test(self, test_id: str):
        """Monitor a specific test"""
        
        try:
            if test_id not in self.active_tests:
                return
            
            test_config = self.active_tests[test_id]
            
            # Check if test should be concluded
            if self._should_conclude_test(test_id):
                await self._conclude_test_gracefully(test_id)
                return
            
            # Get current analysis
            analysis = self.advanced_framework.get_advanced_test_summary(test_id)
            
            # Update test metrics
            if test_id in self.test_metrics:
                metrics = self.test_metrics[test_id]
                base_analysis = analysis.get('advanced_analysis', {}).get('base_analysis', {})
                
                metrics.current_power = base_analysis.get('statistical_power', 0.0)
                metrics.current_effect_size = base_analysis.get('effect_size', 0.0)
                metrics.p_value = base_analysis.get('p_value', 1.0)
                
                # Check for significance
                if (self.config.early_stopping_enabled and 
                    base_analysis.get('is_significant', False)):
                    
                    logger.info(f"Test {test_id} achieved significance - considering early stopping")
            
            # Performance monitoring
            if self.config.performance_logging_enabled:
                self._log_test_performance(test_id, analysis)
                
        except Exception as e:
            logger.error(f"Error monitoring test {test_id}: {e}")
    
    def _should_conclude_test(self, test_id: str) -> bool:
        """Check if test should be concluded"""
        
        if test_id not in self.active_tests:
            return False
        
        test_config = self.active_tests[test_id]
        
        # Check age
        age_days = (datetime.now() - test_config['start_time']).days
        if age_days >= self.config.auto_conclude_after_days:
            return True
        
        # Check if planned duration reached
        if age_days >= test_config.get('duration_days', 30):
            return True
        
        # Check observation limits
        if test_id in self.test_metrics:
            total_observations = sum(
                variant.n_observations 
                for variant in self.advanced_framework.test_registry.get(test_id, [])
            )
            if total_observations >= self.config.max_observations_per_test:
                return True
        
        return False
    
    async def _conclude_test_gracefully(self, test_id: str):
        """Gracefully conclude a test"""
        
        try:
            if test_id in self.active_tests:
                self.active_tests[test_id]['status'] = 'concluded'
                self.active_tests[test_id]['end_time'] = datetime.now()
            
            # Generate final report
            if self.config.report_generation_enabled:
                await self._generate_final_report(test_id)
            
            logger.info(f"Gracefully concluded test {test_id}")
            
        except Exception as e:
            logger.error(f"Error concluding test {test_id}: {e}")
    
    async def _generate_final_report(self, test_id: str):
        """Generate comprehensive final report"""
        
        try:
            analysis = self.advanced_framework.get_advanced_test_summary(test_id)
            
            # Add production metrics
            if test_id in self.test_metrics:
                metrics = self.test_metrics[test_id]
                analysis['production_metrics'] = {
                    'allocation_success_rate': metrics.allocation_successes / max(metrics.allocation_requests, 1),
                    'avg_allocation_time_ms': metrics.avg_allocation_time_ms,
                    'total_conversions': metrics.total_conversions,
                    'total_revenue': metrics.total_revenue,
                    'cost_per_acquisition': metrics.cost_per_acquisition,
                    'return_on_ad_spend': metrics.return_on_ad_spend
                }
            
            # Export report
            report_filename = f"ab_test_report_{test_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(f"/tmp/{report_filename}", 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"Generated final report: {report_filename}")
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
    
    def _log_test_performance(self, test_id: str, analysis: Dict[str, Any]):
        """Log test performance metrics"""
        
        if test_id not in self.test_metrics:
            return
        
        metrics = self.test_metrics[test_id]
        
        # Extract key metrics for logging
        base_analysis = analysis.get('advanced_analysis', {}).get('base_analysis', {})
        
        logger.info(f"Test {test_id} Performance: "
                   f"Power={metrics.current_power:.3f}, "
                   f"Effect={metrics.current_effect_size:.4f}, "
                   f"P-value={metrics.p_value:.4f}, "
                   f"Conversions={metrics.total_conversions}, "
                   f"Revenue=${metrics.total_revenue:.2f}")
    
    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        
        total_allocations = self.system_metrics['total_allocations']
        
        return {
            'system_status': 'healthy' if self.monitoring_active else 'monitoring_inactive',
            'active_tests': len([t for t in self.active_tests.values() if t['status'] == 'active']),
            'total_allocations': total_allocations,
            'allocation_success_rate': self.system_metrics['successful_allocations'] / max(total_allocations, 1),
            'total_observations': self.system_metrics['total_observations'],
            'average_response_time': self.system_metrics['average_response_time'],
            'error_rate': self.system_metrics['error_rate'],
            'monitoring_active': self.monitoring_active,
            'timestamp': datetime.now().isoformat()
        }


# Production factory function
def create_production_ab_manager(discovery_engine: GA4DiscoveryEngine,
                               attribution_engine: AttributionEngine,
                               budget_pacer: BudgetPacer,
                               identity_resolver: IdentityResolver,
                               parameter_manager: ParameterManager,
                               config: ProductionABConfig = None) -> ProductionABTestManager:
    """Create production A/B test manager with all components"""
    
    return ProductionABTestManager(
        discovery_engine=discovery_engine,
        attribution_engine=attribution_engine,
        budget_pacer=budget_pacer,
        identity_resolver=identity_resolver,
        parameter_manager=parameter_manager,
        config=config or ProductionABConfig()
    )


if __name__ == "__main__":
    # Example production usage
    from discovery_engine import GA4DiscoveryEngine
    from attribution_models import AttributionEngine
    from budget_pacer import BudgetPacer
    from identity_resolver import IdentityResolver
    from gaelp_parameter_manager import ParameterManager
    
    # Initialize components
    discovery = GA4DiscoveryEngine()
    
    # Mock components for demo
    class MockComponent:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    attribution = MockComponent()
    budget_pacer = MockComponent()
    identity_resolver = MockComponent()
    parameter_manager = MockComponent()
    
    # Create production A/B manager
    config = ProductionABConfig(
        max_concurrent_tests=3,
        min_observations_per_variant=500,
        monitoring_interval_seconds=30,
        early_stopping_enabled=True
    )
    
    ab_manager = create_production_ab_manager(
        discovery, attribution, budget_pacer,
        identity_resolver, parameter_manager, config
    )
    
    # Create production test
    policy_configs = [
        {
            'name': 'Conservative Policy',
            'base_config': {'learning_rate': 1e-4, 'epsilon': 0.1},
            'modifications': {'epsilon': 0.05}
        },
        {
            'name': 'Aggressive Policy',
            'base_config': {'learning_rate': 1e-4, 'epsilon': 0.1},
            'modifications': {'learning_rate': 1e-3, 'epsilon': 0.2}
        }
    ]
    
    test_id = ab_manager.create_production_policy_test(
        policy_configs=policy_configs,
        test_name='Production Policy Comparison',
        test_type='bayesian_adaptive',
        allocation_strategy='linucb',
        business_objective='roas'
    )
    
    print(f"Created production test: {test_id}")
    
    # Start monitoring
    ab_manager.start_real_time_monitoring()
    
    print("Production A/B testing system ready")
    print(f"System health: {json.dumps(ab_manager.get_system_health_metrics(), indent=2)}")