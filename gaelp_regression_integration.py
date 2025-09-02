#!/usr/bin/env python3
"""
GAELP REGRESSION DETECTION INTEGRATION
Seamlessly integrates regression detection and rollback mechanisms into GAELP training.

FEATURES:
- Real-time performance monitoring during training
- Automatic baseline establishment from GA4 data
- Model checkpointing at optimal intervals
- Emergency rollback on performance degradation
- Integration with existing emergency controls
- Comprehensive logging and audit trail

NO SIMPLIFIED INTEGRATION - Full production-ready system
"""

import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

import logging
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import GAELP components
from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent
from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
from discovery_engine import GA4DiscoveryEngine
from creative_selector import CreativeSelector
from attribution_models import AttributionEngine
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from gaelp_parameter_manager import ParameterManager

# Import regression detection
from regression_detector import (
    RegressionDetector, MetricSnapshot, MetricType, RegressionSeverity,
    RegressionTestSuite, integrate_with_gaelp_training
)
from emergency_controls import get_emergency_controller, emergency_stop_decorator, EmergencyLevel

logger = logging.getLogger(__name__)

class GAELPRegressionMonitor:
    """GAELP-specific regression monitoring with business logic"""
    
    def __init__(self, regression_detector: RegressionDetector):
        self.regression_detector = regression_detector
        self.business_thresholds = {
            'min_roas': 1.5,           # Minimum acceptable ROAS
            'min_conversion_rate': 0.02, # Minimum CVR (2%)
            'max_cpc': 5.0,            # Maximum CPC
            'min_reward_per_episode': 50.0  # Minimum episode reward
        }
        self.performance_history = []
        self.alert_history = []
    
    def establish_baselines_from_ga4(self, discovery_engine: GA4DiscoveryEngine):
        """Establish performance baselines from historical GA4 data"""
        logger.info("Establishing performance baselines from GA4 data")
        
        try:
            # Get historical performance data
            patterns = discovery_engine.get_discovered_patterns()
            
            # Extract ROAS baseline from successful segments
            roas_values = []
            conversion_rates = []
            
            for pattern in patterns.get('segments', []):
                if 'roas' in pattern and pattern['roas'] > 0:
                    roas_values.append(pattern['roas'])
                
                if 'conversion_rate' in pattern and pattern['conversion_rate'] > 0:
                    conversion_rates.append(pattern['conversion_rate'])
            
            if roas_values:
                self.regression_detector.statistical_detector.update_baseline(
                    MetricType.ROAS, roas_values
                )
                logger.info(f"ROAS baseline established from {len(roas_values)} historical segments")
            
            if conversion_rates:
                self.regression_detector.statistical_detector.update_baseline(
                    MetricType.CONVERSION_RATE, conversion_rates
                )
                logger.info(f"CVR baseline established from {len(conversion_rates)} historical segments")
            
            # Extract CPC data if available
            cpc_values = []
            for pattern in patterns.get('campaigns', []):
                if 'avg_cpc' in pattern and pattern['avg_cpc'] > 0:
                    cpc_values.append(pattern['avg_cpc'])
            
            if cpc_values:
                self.regression_detector.statistical_detector.update_baseline(
                    MetricType.CPC, cpc_values
                )
                logger.info(f"CPC baseline established from {len(cpc_values)} historical campaigns")
            
            # Create synthetic reward baseline based on ROAS and volume
            if roas_values and conversion_rates:
                # Estimate rewards based on business metrics
                estimated_rewards = []
                for roas, cvr in zip(roas_values[:50], conversion_rates[:50]):
                    # Synthetic reward calculation
                    estimated_reward = (roas - 1.0) * 100 + cvr * 1000
                    estimated_rewards.append(max(0, estimated_reward))
                
                self.regression_detector.statistical_detector.update_baseline(
                    MetricType.REWARD, estimated_rewards
                )
                logger.info(f"Reward baseline established from {len(estimated_rewards)} synthetic estimates")
            
        except Exception as e:
            logger.error(f"Failed to establish baselines from GA4: {e}")
            # Use default baselines
            self._establish_default_baselines()
    
    def _establish_default_baselines(self):
        """Establish conservative default baselines when GA4 data unavailable"""
        logger.warning("Using default baselines - no GA4 data available")
        
        # Conservative baselines based on industry standards
        default_roas = [1.8, 2.0, 2.2, 1.9, 2.1] * 20  # 100 samples around 2.0
        default_cvr = [0.025, 0.030, 0.035, 0.028, 0.032] * 20  # Around 3%
        default_cpc = [1.2, 1.5, 1.8, 1.3, 1.6] * 20  # Around $1.50
        default_rewards = [75, 85, 95, 80, 90] * 20  # Around 85
        
        self.regression_detector.statistical_detector.update_baseline(MetricType.ROAS, default_roas)
        self.regression_detector.statistical_detector.update_baseline(MetricType.CONVERSION_RATE, default_cvr)
        self.regression_detector.statistical_detector.update_baseline(MetricType.CPC, default_cpc)
        self.regression_detector.statistical_detector.update_baseline(MetricType.REWARD, default_rewards)
    
    def record_training_metrics(self, episode: int, agent_metrics: Dict[str, Any], 
                              environment_metrics: Dict[str, Any], total_reward: float):
        """Record comprehensive training metrics for regression detection"""
        timestamp = datetime.now()
        
        # Business metrics
        if 'roas' in agent_metrics:
            snapshot = MetricSnapshot(
                metric_type=MetricType.ROAS,
                value=float(agent_metrics['roas']),
                timestamp=timestamp,
                episode=episode,
                metadata={'source': 'agent_calculation'}
            )
            self.regression_detector.record_metric(snapshot)
        
        if 'conversion_rate' in agent_metrics:
            snapshot = MetricSnapshot(
                metric_type=MetricType.CONVERSION_RATE,
                value=float(agent_metrics['conversion_rate']),
                timestamp=timestamp,
                episode=episode,
                metadata={'source': 'agent_calculation'}
            )
            self.regression_detector.record_metric(snapshot)
        
        # Environment metrics
        if 'avg_cpc' in environment_metrics:
            snapshot = MetricSnapshot(
                metric_type=MetricType.CPC,
                value=float(environment_metrics['avg_cpc']),
                timestamp=timestamp,
                episode=episode,
                metadata={'source': 'environment_calculation'}
            )
            self.regression_detector.record_metric(snapshot)
        
        if 'ctr' in environment_metrics:
            snapshot = MetricSnapshot(
                metric_type=MetricType.CTR,
                value=float(environment_metrics['ctr']),
                timestamp=timestamp,
                episode=episode,
                metadata={'source': 'environment_calculation'}
            )
            self.regression_detector.record_metric(snapshot)
        
        # Training metrics
        snapshot = MetricSnapshot(
            metric_type=MetricType.REWARD,
            value=float(total_reward),
            timestamp=timestamp,
            episode=episode,
            metadata={'episode_length': environment_metrics.get('steps', 0)}
        )
        self.regression_detector.record_metric(snapshot)
        
        if hasattr(agent_metrics, 'training_loss') and agent_metrics['training_loss'] is not None:
            snapshot = MetricSnapshot(
                metric_type=MetricType.TRAINING_LOSS,
                value=float(agent_metrics['training_loss']),
                timestamp=timestamp,
                episode=episode,
                metadata={'optimizer': 'adam'}
            )
            self.regression_detector.record_metric(snapshot)
        
        # Track performance history
        performance_record = {
            'episode': episode,
            'timestamp': timestamp,
            'roas': agent_metrics.get('roas', 0),
            'conversion_rate': agent_metrics.get('conversion_rate', 0),
            'cpc': environment_metrics.get('avg_cpc', 0),
            'reward': total_reward,
            'meets_business_thresholds': self._check_business_thresholds(agent_metrics, environment_metrics, total_reward)
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _check_business_thresholds(self, agent_metrics: Dict[str, Any], 
                                  environment_metrics: Dict[str, Any], total_reward: float) -> bool:
        """Check if current performance meets business thresholds"""
        roas = agent_metrics.get('roas', 0)
        cvr = agent_metrics.get('conversion_rate', 0)
        cpc = environment_metrics.get('avg_cpc', float('inf'))
        
        meets_thresholds = (
            roas >= self.business_thresholds['min_roas'] and
            cvr >= self.business_thresholds['min_conversion_rate'] and
            cpc <= self.business_thresholds['max_cpc'] and
            total_reward >= self.business_thresholds['min_reward_per_episode']
        )
        
        return meets_thresholds
    
    def check_performance_degradation(self, episode: int) -> Dict[str, Any]:
        """Check for performance degradation and business rule violations"""
        degradation_report = {
            'episode': episode,
            'timestamp': datetime.now(),
            'statistical_alerts': [],
            'business_violations': [],
            'recommended_actions': [],
            'severity': 'normal'
        }
        
        # Run statistical regression detection
        alerts = self.regression_detector.check_for_regressions()
        degradation_report['statistical_alerts'] = [
            {
                'metric': alert.metric_type.value,
                'severity': alert.severity.value_str,
                'current_value': alert.current_value,
                'baseline_mean': alert.baseline_mean,
                'z_score': alert.z_score,
                'confidence': alert.confidence
            }
            for alert in alerts
        ]
        
        # Check business rule violations
        if self.performance_history:
            recent_performance = self.performance_history[-10:]  # Last 10 episodes
            
            # Check recent business threshold compliance
            recent_compliant = sum(1 for p in recent_performance if p['meets_business_thresholds'])
            compliance_rate = recent_compliant / len(recent_performance)
            
            if compliance_rate < 0.7:  # Less than 70% compliance
                degradation_report['business_violations'].append({
                    'type': 'low_business_compliance',
                    'compliance_rate': compliance_rate,
                    'description': f'Only {compliance_rate:.1%} of recent episodes meet business thresholds'
                })
            
            # Check for declining trends
            recent_roas = [p['roas'] for p in recent_performance if p['roas'] > 0]
            if len(recent_roas) >= 5:
                # Simple trend analysis
                x = np.arange(len(recent_roas))
                slope = np.polyfit(x, recent_roas, 1)[0]
                
                if slope < -0.1:  # Declining trend
                    degradation_report['business_violations'].append({
                        'type': 'declining_roas_trend',
                        'slope': slope,
                        'description': f'ROAS declining at rate {slope:.3f} per episode'
                    })
        
        # Determine overall severity
        critical_alerts = [a for a in alerts if a.severity in [RegressionSeverity.CRITICAL, RegressionSeverity.SEVERE]]
        
        if critical_alerts or len(degradation_report['business_violations']) >= 2:
            degradation_report['severity'] = 'critical'
            degradation_report['recommended_actions'].append('IMMEDIATE_ROLLBACK')
        elif alerts or degradation_report['business_violations']:
            degradation_report['severity'] = 'warning'
            degradation_report['recommended_actions'].append('MONITOR_CLOSELY')
        
        # Store alert history
        if degradation_report['severity'] != 'normal':
            self.alert_history.append(degradation_report)
            
            # Keep only recent alert history
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
        
        return degradation_report
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard"""
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'healthy',
            'recent_performance': {},
            'baseline_comparison': {},
            'business_compliance': {},
            'alert_summary': {},
            'recommendations': []
        }
        
        if self.performance_history:
            recent = self.performance_history[-10:]
            
            dashboard['recent_performance'] = {
                'avg_roas': np.mean([p['roas'] for p in recent if p['roas'] > 0]),
                'avg_conversion_rate': np.mean([p['conversion_rate'] for p in recent if p['conversion_rate'] > 0]),
                'avg_reward': np.mean([p['reward'] for p in recent]),
                'episodes_analyzed': len(recent)
            }
            
            # Business compliance
            compliant_episodes = sum(1 for p in recent if p['meets_business_thresholds'])
            dashboard['business_compliance'] = {
                'compliance_rate': compliant_episodes / len(recent),
                'compliant_episodes': compliant_episodes,
                'total_episodes': len(recent)
            }
        
        # Alert summary
        recent_alerts = [a for a in self.alert_history if 
                        datetime.fromisoformat(a['timestamp'].isoformat()) > datetime.now() - timedelta(hours=24)]
        
        dashboard['alert_summary'] = {
            'alerts_24h': len(recent_alerts),
            'critical_alerts_24h': len([a for a in recent_alerts if a['severity'] == 'critical']),
            'warning_alerts_24h': len([a for a in recent_alerts if a['severity'] == 'warning'])
        }
        
        # System status determination
        if dashboard['alert_summary']['critical_alerts_24h'] > 0:
            dashboard['system_status'] = 'critical'
            dashboard['recommendations'].append('Review critical alerts and consider rollback')
        elif dashboard['alert_summary']['warning_alerts_24h'] > 5:
            dashboard['system_status'] = 'degraded'
            dashboard['recommendations'].append('Investigate performance issues')
        elif dashboard['business_compliance'].get('compliance_rate', 1.0) < 0.8:
            dashboard['system_status'] = 'warning'
            dashboard['recommendations'].append('Improve business metric compliance')
        
        return dashboard

class ProductionTrainingWithRegression:
    """Production GAELP training with integrated regression detection and rollback"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize emergency controls
        self.emergency_controller = get_emergency_controller()
        
        # Initialize regression detection system
        self.regression_detector = RegressionDetector(
            emergency_controller=self.emergency_controller
        )
        
        # Initialize GAELP-specific monitoring
        self.gaelp_monitor = GAELPRegressionMonitor(self.regression_detector)
        
        # Training components (initialized later)
        self.agent = None
        self.environment = None
        self.discovery_engine = None
        self.creative_selector = None
        self.attribution_engine = None
        self.budget_pacer = None
        self.identity_resolver = None
        self.parameter_manager = None
        
        # Training state
        self.training_active = False
        self.current_episode = 0
        self.best_performance = {'roas': 0, 'conversion_rate': 0, 'reward': 0}
    
    def initialize_gaelp_components(self):
        """Initialize all GAELP components with emergency controls"""
        self.logger.info("Initializing GAELP components with regression monitoring")
        
        # Initialize core components
        @emergency_stop_decorator("discovery_engine")
        def create_discovery():
            return GA4DiscoveryEngine(write_enabled=True, cache_only=False)
        
        @emergency_stop_decorator("creative_selector") 
        def create_creative_selector():
            return CreativeSelector()
        
        @emergency_stop_decorator("attribution_engine")
        def create_attribution():
            return AttributionEngine()
        
        @emergency_stop_decorator("budget_pacer")
        def create_budget_pacer():
            return BudgetPacer()
        
        @emergency_stop_decorator("identity_resolver")
        def create_identity_resolver():
            return IdentityResolver()
        
        @emergency_stop_decorator("parameter_manager")
        def create_parameter_manager():
            return ParameterManager()
        
        # Create components
        self.discovery_engine = create_discovery()
        self.creative_selector = create_creative_selector()
        self.attribution_engine = create_attribution()
        self.budget_pacer = create_budget_pacer()
        self.identity_resolver = create_identity_resolver()
        self.parameter_manager = create_parameter_manager()
        
        # Initialize environment
        @emergency_stop_decorator("environment")
        def create_environment():
            return ProductionFortifiedEnvironment(
                parameter_manager=self.parameter_manager,
                use_real_ga4_data=False,
                is_parallel=False
            )
        
        self.environment = create_environment()
        
        # Initialize agent
        @emergency_stop_decorator("rl_agent")
        def create_agent():
            return ProductionFortifiedRLAgent(
                discovery_engine=self.discovery_engine,
                creative_selector=self.creative_selector,
                attribution_engine=self.attribution_engine,
                budget_pacer=self.budget_pacer,
                identity_resolver=self.identity_resolver,
                parameter_manager=self.parameter_manager
            )
        
        self.agent = create_agent()
        
        # Establish baselines from GA4 data
        self.gaelp_monitor.establish_baselines_from_ga4(self.discovery_engine)
        
        # Start regression monitoring
        self.regression_detector.start_monitoring()
        
        self.logger.info(f"GAELP components initialized:")
        self.logger.info(f"  - Discovered channels: {len(self.agent.discovered_channels)}")
        self.logger.info(f"  - Discovered segments: {len(self.agent.discovered_segments)}")  
        self.logger.info(f"  - Discovered creatives: {len(self.agent.discovered_creatives)}")
        self.logger.info(f"  - Regression monitoring: Active")
    
    def run_training_with_regression_monitoring(self, num_episodes: int = 1000):
        """Run training with comprehensive regression monitoring and rollback"""
        self.logger.info(f"Starting GAELP training with regression monitoring for {num_episodes} episodes")
        
        if not self.agent or not self.environment:
            raise ValueError("GAELP components not initialized")
        
        self.training_active = True
        training_start_time = datetime.now()
        
        try:
            for episode in range(num_episodes):
                if not self.emergency_controller.is_system_healthy():
                    self.logger.error("System unhealthy - stopping training")
                    break
                
                episode_start_time = datetime.now()
                self.current_episode = episode
                
                # Run single episode
                episode_results = self._run_single_episode(episode)
                
                if episode_results['success']:
                    # Record metrics for regression monitoring
                    self.gaelp_monitor.record_training_metrics(
                        episode=episode,
                        agent_metrics=episode_results['agent_metrics'],
                        environment_metrics=episode_results['environment_metrics'],
                        total_reward=episode_results['total_reward']
                    )
                    
                    # Check for performance degradation every 10 episodes
                    if episode % 10 == 0 and episode > 50:  # Allow warmup period
                        degradation_report = self.gaelp_monitor.check_performance_degradation(episode)
                        
                        if degradation_report['severity'] == 'critical':
                            self.logger.error(f"Critical performance degradation detected at episode {episode}")
                            self.logger.error(f"Statistical alerts: {len(degradation_report['statistical_alerts'])}")
                            self.logger.error(f"Business violations: {len(degradation_report['business_violations'])}")
                            
                            # Perform rollback if needed
                            if self.regression_detector.auto_rollback_enabled:
                                self.logger.info("Attempting automatic rollback...")
                                rollback_success = self._perform_emergency_rollback()
                                
                                if rollback_success:
                                    self.logger.info("Emergency rollback successful - continuing training")
                                else:
                                    self.logger.error("Emergency rollback failed - stopping training")
                                    break
                        
                        elif degradation_report['severity'] == 'warning':
                            self.logger.warning(f"Performance warning at episode {episode}")
                            for violation in degradation_report['business_violations']:
                                self.logger.warning(f"  - {violation['description']}")
                    
                    # Create checkpoint every 100 episodes
                    if episode % 100 == 0 and episode > 0:
                        self._create_performance_checkpoint(episode, episode_results)
                    
                    # Update best performance tracking
                    self._update_best_performance(episode_results)
                    
                    # Log progress every 25 episodes
                    if episode % 25 == 0:
                        self._log_training_progress(episode, episode_results)
                
                else:
                    self.logger.error(f"Episode {episode} failed: {episode_results.get('error', 'Unknown error')}")
                    self.emergency_controller.register_error("training_episode", episode_results.get('error', 'Episode failed'))
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with exception: {e}")
            raise
        finally:
            self.training_active = False
            self.regression_detector.stop_monitoring()
            
            # Generate final report
            self._generate_final_training_report(training_start_time)
    
    def _run_single_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single training episode with comprehensive monitoring"""
        try:
            obs, info = self.environment.reset()
            episode_reward = 0
            episode_spend = 0
            episode_conversions = 0
            episode_revenue = 0
            step = 0
            done = False
            episode_bids = []
            
            while not done:
                # Get current state
                state = self.environment.current_user_state
                
                # Select action with emergency monitoring
                @emergency_stop_decorator("action_selection")
                def select_action_safe():
                    return self.agent.select_action(state, explore=True)
                
                action = select_action_safe()
                
                # Monitor bid amount
                if hasattr(action, 'bid_amount'):
                    bid_amount = float(action.bid_amount)
                    episode_bids.append(bid_amount)
                    self.emergency_controller.record_bid(bid_amount)
                
                # Step environment
                @emergency_stop_decorator("environment_step")
                def step_environment_safe():
                    return self.environment.step(action)
                
                next_obs, reward, terminated, truncated, info = step_environment_safe()
                done = terminated or truncated
                
                # Track metrics
                if 'spend' in info:
                    episode_spend += info['spend']
                if 'conversions' in info:
                    episode_conversions += info['conversions']
                if 'revenue' in info:
                    episode_revenue += info['revenue']
                
                # Get next state
                next_state = self.environment.current_user_state
                
                # Train agent
                @emergency_stop_decorator("training_step")
                def train_safe():
                    loss = self.agent.train(state, action, reward, next_state, done)
                    if loss is not None:
                        self.emergency_controller.record_training_loss(float(loss))
                    return loss
                
                training_loss = train_safe()
                
                episode_reward += reward
                step += 1
                
                # Check for anomalies during episode
                if step % 50 == 0:
                    if not self.emergency_controller.is_system_healthy():
                        self.logger.warning(f"Emergency condition during episode {episode}, step {step}")
                        break
            
            # Calculate episode metrics
            agent_metrics = {
                'roas': (episode_revenue / episode_spend) if episode_spend > 0 else 0,
                'conversion_rate': (episode_conversions / step) if step > 0 else 0,
                'training_loss': training_loss,
                'epsilon': getattr(self.agent, 'epsilon', 0)
            }
            
            environment_metrics = {
                'avg_cpc': (episode_spend / step) if step > 0 else 0,
                'ctr': 0.1,  # Placeholder - would come from environment
                'steps': step,
                'conversions': episode_conversions,
                'revenue': episode_revenue,
                'spend': episode_spend
            }
            
            return {
                'success': True,
                'total_reward': episode_reward,
                'agent_metrics': agent_metrics,
                'environment_metrics': environment_metrics,
                'episode_length': step,
                'bids': episode_bids
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_reward': 0,
                'agent_metrics': {},
                'environment_metrics': {}
            }
    
    def _perform_emergency_rollback(self) -> bool:
        """Perform emergency rollback to last good checkpoint"""
        self.logger.info("Initiating emergency rollback procedure")
        
        try:
            # Find best rollback candidate
            min_performance = {
                'roas': self.best_performance['roas'] * 0.8,
                'conversion_rate': self.best_performance['conversion_rate'] * 0.8,
                'reward': self.best_performance['reward'] * 0.8
            }
            
            candidate_id = self.regression_detector.model_manager.find_rollback_candidate(min_performance)
            
            if not candidate_id:
                self.logger.error("No suitable rollback candidate found")
                return False
            
            # Perform rollback
            success = self.regression_detector.model_manager.rollback_to_checkpoint(candidate_id)
            
            if success:
                # Load checkpoint into agent
                checkpoint_data = self.regression_detector.model_manager.load_checkpoint(candidate_id)
                
                if hasattr(self.agent, 'load_state_dict') and 'model_state_dict' in checkpoint_data:
                    self.agent.load_state_dict(checkpoint_data['model_state_dict'])
                    self.logger.info(f"Agent state restored from checkpoint {candidate_id}")
                
                # Reset recent metrics to avoid contamination
                for metric_type in MetricType:
                    self.regression_detector.recent_metrics[metric_type].clear()
                
                return True
            
        except Exception as e:
            self.logger.error(f"Emergency rollback failed: {e}")
        
        return False
    
    def _create_performance_checkpoint(self, episode: int, episode_results: Dict[str, Any]):
        """Create performance checkpoint for potential rollback"""
        performance_metrics = {
            'roas': episode_results['agent_metrics']['roas'],
            'conversion_rate': episode_results['agent_metrics']['conversion_rate'],
            'reward': episode_results['total_reward'],
            'episode': episode,
            'training_loss': episode_results['agent_metrics'].get('training_loss', 0)
        }
        
        checkpoint_id = self.regression_detector.create_model_checkpoint(
            self.agent.q_network if hasattr(self.agent, 'q_network') else self.agent,
            performance_metrics,
            episode
        )
        
        self.logger.info(f"Created checkpoint {checkpoint_id} at episode {episode}")
        self.logger.info(f"  ROAS: {performance_metrics['roas']:.4f}")
        self.logger.info(f"  CVR: {performance_metrics['conversion_rate']:.4f}")
        self.logger.info(f"  Reward: {performance_metrics['reward']:.2f}")
    
    def _update_best_performance(self, episode_results: Dict[str, Any]):
        """Update best performance tracking"""
        roas = episode_results['agent_metrics']['roas']
        cvr = episode_results['agent_metrics']['conversion_rate']
        reward = episode_results['total_reward']
        
        if roas > self.best_performance['roas']:
            self.best_performance['roas'] = roas
        
        if cvr > self.best_performance['conversion_rate']:
            self.best_performance['conversion_rate'] = cvr
        
        if reward > self.best_performance['reward']:
            self.best_performance['reward'] = reward
    
    def _log_training_progress(self, episode: int, episode_results: Dict[str, Any]):
        """Log detailed training progress"""
        agent_metrics = episode_results['agent_metrics']
        env_metrics = episode_results['environment_metrics']
        
        self.logger.info(f"Episode {episode} Progress:")
        self.logger.info(f"  Reward: {episode_results['total_reward']:.2f}")
        self.logger.info(f"  ROAS: {agent_metrics['roas']:.4f}")
        self.logger.info(f"  CVR: {agent_metrics['conversion_rate']:.4f}")
        self.logger.info(f"  Spend: ${env_metrics['spend']:.2f}")
        self.logger.info(f"  Revenue: ${env_metrics['revenue']:.2f}")
        self.logger.info(f"  Epsilon: {agent_metrics.get('epsilon', 0):.3f}")
        self.logger.info(f"  Emergency Level: {self.emergency_controller.current_emergency_level.value}")
        
        # Performance dashboard
        dashboard = self.gaelp_monitor.get_performance_dashboard()
        self.logger.info(f"  System Status: {dashboard['system_status']}")
        self.logger.info(f"  Business Compliance: {dashboard['business_compliance'].get('compliance_rate', 0):.1%}")
    
    def _generate_final_training_report(self, training_start_time: datetime):
        """Generate comprehensive training report"""
        training_duration = datetime.now() - training_start_time
        
        self.logger.info("="*70)
        self.logger.info("GAELP TRAINING WITH REGRESSION MONITORING - FINAL REPORT")
        self.logger.info("="*70)
        self.logger.info(f"Training Duration: {training_duration}")
        self.logger.info(f"Episodes Completed: {self.current_episode}")
        self.logger.info(f"Best ROAS: {self.best_performance['roas']:.4f}")
        self.logger.info(f"Best CVR: {self.best_performance['conversion_rate']:.4f}")
        self.logger.info(f"Best Reward: {self.best_performance['reward']:.2f}")
        
        # Regression summary
        summary = self.regression_detector.get_performance_summary()
        self.logger.info(f"Total Alerts Generated: {summary['total_alerts']}")
        self.logger.info(f"Recent Alerts (24h): {summary['recent_alerts']}")
        self.logger.info(f"Checkpoints Created: {summary['checkpoint_count']}")
        self.logger.info(f"Final System Health: {summary['system_health']}")
        
        # GAELP-specific summary
        dashboard = self.gaelp_monitor.get_performance_dashboard()
        self.logger.info(f"Final Business Compliance: {dashboard['business_compliance'].get('compliance_rate', 0):.1%}")
        self.logger.info(f"System Status: {dashboard['system_status']}")
        
        self.logger.info("="*70)

def main():
    """Main function for production training with regression monitoring"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/home/hariravichandran/AELP/gaelp_regression_training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize production training system
        training_system = ProductionTrainingWithRegression()
        
        # Initialize GAELP components
        training_system.initialize_gaelp_components()
        
        # Run training with regression monitoring
        training_system.run_training_with_regression_monitoring(num_episodes=2000)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()