#!/usr/bin/env python3
"""
GAELP Production Orchestrator
Comprehensive system that wires together ALL components from Waves 1-6
Ensures everything is integrated and running properly with monitoring
"""

import sys
import os
import json
import logging
import asyncio
import threading
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CORE IMPORTS ====================
# These are the actual production components we built

# Core RL Components
from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent
from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment

# Data & Discovery
from discovery_engine import GA4RealTimeDataPipeline
try:
    from segment_discovery import SegmentDiscoveryEngine
except ImportError:
    logger.warning("SegmentDiscoveryEngine not available")
    SegmentDiscoveryEngine = None

try:
    from pipeline_integration import GAELPModelUpdater
except ImportError:
    logger.warning("GAELPModelUpdater not available")
    GAELPModelUpdater = None

# Attribution & Budget
try:
    from attribution_system import MultiTouchAttributionEngine
except ImportError:
    logger.warning("MultiTouchAttributionEngine not available")
    MultiTouchAttributionEngine = None

try:
    from budget_optimizer import DynamicBudgetOptimizer
except ImportError:
    logger.warning("DynamicBudgetOptimizer not available")
    DynamicBudgetOptimizer = None

try:
    from budget_safety_controller import BudgetSafetyController
except ImportError:
    logger.warning("BudgetSafetyController not available")
    BudgetSafetyController = None

# Safety & Monitoring  
from emergency_controls import EmergencyController

try:
    from convergence_monitoring_integration_demo import ConvergenceMonitor
except ImportError:
    logger.warning("ConvergenceMonitor not available")
    ConvergenceMonitor = None

try:
    from regression_detector import RegressionDetector
except ImportError:
    logger.warning("RegressionDetector not available")
    RegressionDetector = None

try:
    from production_checkpoint_manager import ProductionCheckpointManager
except ImportError:
    logger.warning("ProductionCheckpointManager not available")
    ProductionCheckpointManager = None

# Production Features
try:
    from production_online_learner import ProductionOnlineLearner
except ImportError:
    logger.warning("ProductionOnlineLearner not available")
    ProductionOnlineLearner = None

try:
    from shadow_mode_manager import ShadowModeManager
except ImportError:
    logger.warning("ShadowModeManager not available")
    ShadowModeManager = None

try:
    from statistical_ab_testing_framework import StatisticalABTestingFramework
except ImportError:
    logger.warning("StatisticalABTestingFramework not available")
    StatisticalABTestingFramework = None

try:
    from bid_explainability_system import BidExplainabilitySystem
except ImportError:
    logger.warning("BidExplainabilitySystem not available")
    BidExplainabilitySystem = None

# Google Ads Integration
try:
    from google_ads_production_manager import GoogleAdsProductionManager
except ImportError:
    logger.warning("GoogleAdsProductionManager not available")
    GoogleAdsProductionManager = None

try:
    from google_ads_gaelp_integration import GoogleAdsGAELPIntegration
except ImportError:
    logger.warning("GoogleAdsGAELPIntegration not available")
    GoogleAdsGAELPIntegration = None

# Success Criteria & Monitoring
try:
    from gaelp_success_criteria_monitor import SuccessCriteriaMonitor
except ImportError:
    logger.warning("SuccessCriteriaMonitor not available")
    SuccessCriteriaMonitor = None

# Creative & Auction
try:
    from creative_content_analyzer import CreativeContentAnalyzer
except ImportError:
    logger.warning("CreativeContentAnalyzer not available")
    CreativeContentAnalyzer = None

try:
    from auction_gym_integration_fixed import FixedAuctionGymIntegration
except ImportError:
    logger.warning("FixedAuctionGymIntegration not available")
    FixedAuctionGymIntegration = None

# ==================== WRAPPER CLASSES ====================

class DiscoveryEngineWrapper:
    """Wrapper to provide DiscoveryEngine interface for GA4RealTimeDataPipeline"""
    
    def __init__(self, pipeline: GA4RealTimeDataPipeline):
        self.pipeline = pipeline
        self.patterns_file = "discovered_patterns.json"
        self._load_patterns()
    
    def _load_patterns(self):
        """Load discovered patterns from file"""
        try:
            with open(self.patterns_file, 'r') as f:
                self.patterns = json.load(f)
        except:
            self.patterns = {}
    
    def get_discovered_patterns(self) -> Dict[str, Any]:
        """Return discovered patterns"""
        return self.patterns
    
    def get_patterns(self) -> Dict[str, Any]:
        """Alias for get_discovered_patterns for compatibility"""
        return self.patterns
    
    def get_conversion_data(self, *args, **kwargs):
        """Delegate to pipeline if available"""
        if hasattr(self.pipeline, 'get_conversion_data'):
            return self.pipeline.get_conversion_data(*args, **kwargs)
        return []

# ==================== ORCHESTRATOR ====================

class ComponentStatus(Enum):
    """Status of each component"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator"""
    # Core settings
    environment: str = "production"  # production, staging, development
    dry_run: bool = False  # Run without real money
    
    # Component toggles (all enabled by default)
    enable_rl_training: bool = True
    enable_online_learning: bool = True
    enable_shadow_mode: bool = True
    enable_ab_testing: bool = True
    enable_google_ads: bool = True
    enable_safety_controls: bool = True
    enable_explainability: bool = True
    
    # Monitoring intervals (seconds)
    health_check_interval: int = 30
    metrics_update_interval: int = 60
    checkpoint_interval: int = 300
    
    # Safety limits
    max_daily_spend: float = 1000.0
    max_bid_amount: float = 50.0
    min_roas_threshold: float = 2.5
    
    # Data settings
    ga4_property_id: str = "308028264"
    google_ads_customer_id: str = ""  # Set from environment
    
    # Model settings
    model_version: str = "gaelp_v1.0"
    enable_double_dqn: bool = True
    enable_prioritized_replay: bool = True
    enable_lstm_sequence: bool = True

class GAELPProductionOrchestrator:
    """Main orchestrator that coordinates all GAELP components"""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.components: Dict[str, Any] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        self.metrics: Dict[str, Any] = {}
        self.running = False
        self.threads: List[threading.Thread] = []
        
        # Initialize locks for thread safety
        self.lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        
        logger.info(f"🚀 GAELP Production Orchestrator initialized in {config.environment} mode")
        
    def initialize_components(self) -> bool:
        """Initialize all components in the correct order"""
        try:
            logger.info("📦 Initializing all components...")
            
            # 1. Safety & Emergency Controls (FIRST - must be ready before anything else)
            if self.config.enable_safety_controls:
                self._init_safety_systems()
            
            # 2. Data Pipeline & Discovery
            self._init_data_pipeline()
            
            # 3. Core RL System
            self._init_rl_system()
            
            # 4. Attribution & Budget
            self._init_attribution_budget()
            
            # 5. Monitoring & Validation
            self._init_monitoring()
            
            # 6. Production Features
            if self.config.enable_online_learning:
                self._init_online_learning()
            
            if self.config.enable_shadow_mode:
                self._init_shadow_mode()
            
            if self.config.enable_ab_testing:
                self._init_ab_testing()
            
            if self.config.enable_explainability:
                self._init_explainability()
            
            # 7. Google Ads Integration (last - needs all other systems)
            if self.config.enable_google_ads and not self.config.dry_run:
                self._init_google_ads()
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    def _init_safety_systems(self):
        """Initialize safety and emergency controls"""
        logger.info("🛡️ Initializing safety systems...")
        
        # Emergency controller
        self.components['emergency_controller'] = EmergencyController()
        self.component_status['emergency_controller'] = ComponentStatus.RUNNING
        
        # Budget safety
        if BudgetSafetyController:
            self.components['budget_safety'] = BudgetSafetyController()
            self.component_status['budget_safety'] = ComponentStatus.RUNNING
        
        # Success criteria monitor
        if SuccessCriteriaMonitor:
            self.components['success_monitor'] = SuccessCriteriaMonitor()
            self.component_status['success_monitor'] = ComponentStatus.RUNNING
        
        logger.info("✅ Safety systems ready")
    
    def _init_data_pipeline(self):
        """Initialize data pipeline and discovery"""
        logger.info("📊 Initializing data pipeline...")
        
        # GA4 real-time pipeline
        self.components['ga4_pipeline'] = GA4RealTimeDataPipeline(
            property_id=self.config.ga4_property_id
        )
        self.component_status['ga4_pipeline'] = ComponentStatus.RUNNING
        
        # Segment discovery
        if SegmentDiscoveryEngine:
            self.components['segment_discovery'] = SegmentDiscoveryEngine()
            self.component_status['segment_discovery'] = ComponentStatus.RUNNING
        
        # Model updater
        if GAELPModelUpdater:
            self.components['model_updater'] = GAELPModelUpdater()
            self.component_status['model_updater'] = ComponentStatus.RUNNING
        
        logger.info("✅ Data pipeline ready")
    
    def _init_rl_system(self):
        """Initialize core RL components"""
        logger.info("🤖 Initializing RL system...")
        
        # Environment
        self.components['environment'] = ProductionFortifiedEnvironment()
        self.component_status['environment'] = ComponentStatus.RUNNING
        
        # RL Agent needs components from environment
        env = self.components['environment']
        
        # Import parameter manager for agent
        from gaelp_parameter_manager import ParameterManager
        
        # Wrap the discovery engine to provide the expected interface
        wrapped_discovery = DiscoveryEngineWrapper(env.discovery)
        
        self.components['rl_agent'] = ProductionFortifiedRLAgent(
            discovery_engine=wrapped_discovery,
            creative_selector=env.creative_selector,
            attribution_engine=env.attribution,
            budget_pacer=env.budget_pacer,
            identity_resolver=env.identity_resolver,
            parameter_manager=ParameterManager()
        )
        self.component_status['rl_agent'] = ComponentStatus.RUNNING
        
        # Auction integration
        if FixedAuctionGymIntegration:
            self.components['auction'] = FixedAuctionGymIntegration()
            self.component_status['auction'] = ComponentStatus.RUNNING
        
        # Creative analyzer
        if CreativeContentAnalyzer:
            self.components['creative_analyzer'] = CreativeContentAnalyzer()
            self.component_status['creative_analyzer'] = ComponentStatus.RUNNING
        
        logger.info("✅ RL system ready")
    
    def _init_attribution_budget(self):
        """Initialize attribution and budget systems"""
        logger.info("💰 Initializing attribution & budget...")
        
        # Multi-touch attribution
        if MultiTouchAttributionEngine:
            self.components['attribution'] = MultiTouchAttributionEngine(
                db_path="attribution_system.db"
            )
            self.component_status['attribution'] = ComponentStatus.RUNNING
        
        # Dynamic budget optimizer
        if DynamicBudgetOptimizer:
            self.components['budget_optimizer'] = DynamicBudgetOptimizer()
            self.component_status['budget_optimizer'] = ComponentStatus.RUNNING
        
        logger.info("✅ Attribution & budget ready")
    
    def _init_monitoring(self):
        """Initialize monitoring and validation"""
        logger.info("📈 Initializing monitoring...")
        
        # Convergence monitor
        if ConvergenceMonitor:
            self.components['convergence_monitor'] = ConvergenceMonitor()
            self.component_status['convergence_monitor'] = ComponentStatus.RUNNING
        
        # Regression detector
        if RegressionDetector:
            self.components['regression_detector'] = RegressionDetector()
            self.component_status['regression_detector'] = ComponentStatus.RUNNING
        
        # Checkpoint manager
        if ProductionCheckpointManager:
            self.components['checkpoint_manager'] = ProductionCheckpointManager(
                checkpoint_dir="production_checkpoints"
            )
            self.component_status['checkpoint_manager'] = ComponentStatus.RUNNING
        
        logger.info("✅ Monitoring ready")
    
    def _init_online_learning(self):
        """Initialize online learning system"""
        logger.info("🔄 Initializing online learning...")
        
        # Create wrapper for discovery engine interface
        discovery_wrapper = DiscoveryEngineWrapper(self.components['ga4_pipeline'])
        
        self.components['online_learner'] = ProductionOnlineLearner(
            agent=self.components['rl_agent'],
            discovery_engine=discovery_wrapper
        )
        self.component_status['online_learner'] = ComponentStatus.RUNNING
        
        logger.info("✅ Online learning ready")
    
    def _init_shadow_mode(self):
        """Initialize shadow mode testing"""
        logger.info("👻 Initializing shadow mode...")
        
        # Create shadow test configuration
        from shadow_mode_manager import ShadowTestConfiguration
        shadow_config = ShadowTestConfiguration(
            test_name="production_shadow_test",
            duration_hours=24.0,
            models={
                "current": {"model_id": "production_v1"},
                "challenger": {"model_id": "production_v2"}
            },
            traffic_percentage=1.0,
            comparison_threshold=0.1,
            statistical_confidence=0.95,
            min_sample_size=100,
            save_all_decisions=True,
            real_time_reporting=True
        )
        
        self.components['shadow_mode'] = ShadowModeManager(shadow_config)
        self.component_status['shadow_mode'] = ComponentStatus.RUNNING
        
        logger.info("✅ Shadow mode ready")
    
    def _init_ab_testing(self):
        """Initialize A/B testing framework"""
        logger.info("🧪 Initializing A/B testing...")
        
        if StatisticalABTestingFramework:
            self.components['ab_testing'] = StatisticalABTestingFramework()
            self.component_status['ab_testing'] = ComponentStatus.RUNNING
            logger.info("✅ A/B testing ready")
        else:
            logger.warning("⚠️ A/B testing not available, skipping")
    
    def _init_explainability(self):
        """Initialize explainability system"""
        logger.info("💡 Initializing explainability...")
        
        if BidExplainabilitySystem:
            self.components['explainability'] = BidExplainabilitySystem()
            self.component_status['explainability'] = ComponentStatus.RUNNING
            logger.info("✅ Explainability ready")
        else:
            logger.warning("⚠️ Explainability not available, skipping")
    
    def _init_google_ads(self):
        """Initialize Google Ads integration"""
        logger.info("📱 Initializing Google Ads...")
        
        if not self.config.google_ads_customer_id:
            logger.warning("⚠️ Google Ads customer ID not set, skipping integration")
            return
        
        if GoogleAdsGAELPIntegration:
            self.components['google_ads'] = GoogleAdsGAELPIntegration(
                customer_id=self.config.google_ads_customer_id,
                rl_agent=self.components['rl_agent']
            )
        else:
            logger.warning("⚠️ GoogleAdsGAELPIntegration not available")
            return
        self.component_status['google_ads'] = ComponentStatus.RUNNING
        
        logger.info("✅ Google Ads ready")
    
    def start(self):
        """Start all components and monitoring threads"""
        logger.info("🚀 Starting GAELP Production Orchestrator...")
        
        if not self.initialize_components():
            logger.error("❌ Failed to initialize components, aborting start")
            return False
        
        self.running = True
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        # Start main training loop if enabled
        if self.config.enable_rl_training:
            self._start_training_loop()
        
        logger.info("✅ GAELP Production Orchestrator started successfully")
        return True
    
    def _start_monitoring_threads(self):
        """Start all monitoring threads"""
        # Health check thread
        health_thread = threading.Thread(
            target=self._health_check_loop,
            name="health_check"
        )
        health_thread.daemon = True
        health_thread.start()
        self.threads.append(health_thread)
        
        # Metrics update thread
        metrics_thread = threading.Thread(
            target=self._metrics_update_loop,
            name="metrics_update"
        )
        metrics_thread.daemon = True
        metrics_thread.start()
        self.threads.append(metrics_thread)
        
        # Checkpoint thread
        checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop,
            name="checkpoint"
        )
        checkpoint_thread.daemon = True
        checkpoint_thread.start()
        self.threads.append(checkpoint_thread)
    
    def _start_training_loop(self):
        """Start the main RL training loop"""
        training_thread = threading.Thread(
            target=self._training_loop,
            name="training"
        )
        training_thread.daemon = True
        training_thread.start()
        self.threads.append(training_thread)
    
    def _health_check_loop(self):
        """Continuous health check of all components"""
        while self.running:
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def _metrics_update_loop(self):
        """Update metrics from all components"""
        while self.running:
            try:
                self._update_metrics()
                time.sleep(self.config.metrics_update_interval)
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
    
    def _checkpoint_loop(self):
        """Save checkpoints periodically"""
        while self.running:
            try:
                time.sleep(self.config.checkpoint_interval)
                self._save_checkpoint()
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")
    
    def _training_loop(self):
        """Main RL training loop"""
        episode = 0
        while self.running:
            try:
                # Run one training episode
                episode_metrics = self._run_training_episode(episode)
                
                # Update metrics
                with self.metrics_lock:
                    self.metrics['last_episode'] = episode
                    self.metrics['episode_metrics'] = episode_metrics
                
                # Check for convergence or issues
                if self._check_training_issues():
                    logger.warning("⚠️ Training issues detected, pausing...")
                    time.sleep(60)
                
                episode += 1
                
            except Exception as e:
                import traceback
                logger.error(f"Training loop error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(30)
    
    def _run_training_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single training episode"""
        env = self.components['environment']
        agent = self.components['rl_agent']
        
        # Reset environment
        state_tuple = env.reset()
        # Handle both tuple and non-tuple returns
        if isinstance(state_tuple, tuple):
            state, _ = state_tuple
        else:
            state = state_tuple
            
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            # Get action from agent - wrap state in proper object
            from fortified_rl_agent_no_hardcoding import DynamicEnrichedState
            
            # Create enriched state from numpy array
            enriched_state = DynamicEnrichedState()
            
            # Parse the numpy state vector and populate fields
            if isinstance(state, np.ndarray) and len(state) >= 10:
                # Map state vector to enriched state fields
                enriched_state.stage = int(state[0]) if len(state) > 0 else 0
                enriched_state.touchpoints_seen = int(state[1]) if len(state) > 1 else 0
                enriched_state.days_since_first_touch = float(state[2]) if len(state) > 2 else 0.0
                enriched_state.segment_index = int(state[3]) if len(state) > 3 else 0
                enriched_state.device_index = int(state[4]) if len(state) > 4 else 0
                enriched_state.channel_index = int(state[5]) if len(state) > 5 else 0
                enriched_state.creative_index = int(state[6]) if len(state) > 6 else 0
                enriched_state.competition_level = float(state[7]) if len(state) > 7 else 0.0
                enriched_state.budget_remaining_pct = float(state[8]) if len(state) > 8 else 1.0
                enriched_state.current_bid = float(state[9]) if len(state) > 9 else 0.0
            
            action = agent.select_action(enriched_state)
            
            # Generate explanation if enabled
            if self.config.enable_explainability and 'explainability' in self.components:
                explanation = self.components['explainability'].explain_bid_decision(
                    state, action
                )
            
            # Step environment
            step_result = env.step(action)
            # Handle both old (4 values) and new (5 values) Gym API
            if len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
                done = done or truncated  # Combine termination conditions
            else:
                next_state, reward, done, info = step_result
            
            # Store experience
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            }
            agent.replay_buffer.add(experience)
            
            # Train if enough experiences
            batch_size = agent._hyperparameters.get('batch_size', 32)
            if len(agent.replay_buffer) > batch_size and steps % 32 == 0:
                # Train the agent with the experience
                enriched_next_state = DynamicEnrichedState()
                if isinstance(next_state, np.ndarray) and len(next_state) >= 10:
                    enriched_next_state.stage = int(next_state[0]) if len(next_state) > 0 else 0
                    enriched_next_state.touchpoints_seen = int(next_state[1]) if len(next_state) > 1 else 0
                    enriched_next_state.days_since_first_touch = float(next_state[2]) if len(next_state) > 2 else 0.0
                    enriched_next_state.segment_index = int(next_state[3]) if len(next_state) > 3 else 0
                    enriched_next_state.device_index = int(next_state[4]) if len(next_state) > 4 else 0
                    enriched_next_state.channel_index = int(next_state[5]) if len(next_state) > 5 else 0
                    enriched_next_state.creative_index = int(next_state[6]) if len(next_state) > 6 else 0
                    enriched_next_state.competition_level = float(next_state[7]) if len(next_state) > 7 else 0.0
                    enriched_next_state.budget_remaining_pct = float(next_state[8]) if len(next_state) > 8 else 1.0
                    enriched_next_state.current_bid = float(next_state[9]) if len(next_state) > 9 else 0.0
                
                # Call the train method
                if hasattr(agent, 'train'):
                    agent.train(enriched_state, action, reward, enriched_next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Check safety violations
            if self._check_safety_violations(action, info):
                logger.warning("🛑 Safety violation detected, ending episode")
                done = True
        
        # Decay epsilon after episode
        if hasattr(agent, 'epsilon_decay') and hasattr(agent, 'epsilon_min'):
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            logger.debug(f"Updated epsilon to {agent.epsilon:.4f}")
        
        return {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': agent.epsilon
        }
    
    def _check_safety_violations(self, action: Any, info: Dict) -> bool:
        """Check for safety violations"""
        if not self.config.enable_safety_controls:
            return False
        
        # Check budget safety
        budget_safety = self.components.get('budget_safety')
        if budget_safety:
            from decimal import Decimal
            is_safe, violations = budget_safety.record_spending(
                campaign_id='training',
                channel=info.get('channel', 'unknown'),
                amount=Decimal(str(info.get('cost', 0))),
                bid_amount=Decimal(str(info.get('bid_amount', 0)))
            )
            if not is_safe:
                logger.warning(f"Budget safety violations: {violations}")
                return True
        
        # Check emergency controls
        emergency = self.components.get('emergency_controller')
        if emergency:
            if emergency.emergency_stop_triggered or not emergency.is_system_healthy():
                logger.warning("Emergency stop triggered or system unhealthy")
                return True
        
        return False
    
    def _check_training_issues(self) -> bool:
        """Check for training convergence issues"""
        # Check convergence monitor
        monitor = self.components.get('convergence_monitor')
        if monitor:
            issues = monitor.check_convergence_issues(self.metrics.get('episode_metrics', {}))
            if issues:
                return True
        
        # Check regression detector
        detector = self.components.get('regression_detector')
        if detector:
            regressions = detector.check_for_regressions()
            if regressions:
                logger.warning(f"Detected {len(regressions)} regressions")
                return True
        
        return False
    
    def _perform_health_check(self):
        """Check health of all components"""
        with self.lock:
            for name, component in self.components.items():
                try:
                    # Check if component has health check method
                    if hasattr(component, 'health_check'):
                        is_healthy = component.health_check()
                        if not is_healthy:
                            self.component_status[name] = ComponentStatus.ERROR
                            logger.warning(f"⚠️ Component {name} is unhealthy")
                    else:
                        # Basic check - component exists
                        self.component_status[name] = ComponentStatus.RUNNING
                except Exception as e:
                    self.component_status[name] = ComponentStatus.ERROR
                    logger.error(f"❌ Health check failed for {name}: {e}")
    
    def _update_metrics(self):
        """Update metrics from all components"""
        with self.metrics_lock:
            # Get metrics from each component
            for name, component in self.components.items():
                try:
                    if hasattr(component, 'get_metrics'):
                        self.metrics[name] = component.get_metrics()
                except Exception as e:
                    logger.error(f"Failed to get metrics from {name}: {e}")
            
            # Add system metrics
            self.metrics['timestamp'] = datetime.now().isoformat()
            self.metrics['component_status'] = {
                name: status.value 
                for name, status in self.component_status.items()
            }
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        try:
            checkpoint_manager = self.components.get('checkpoint_manager')
            if checkpoint_manager and self.components.get('rl_agent'):
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    model=self.components['rl_agent'],
                    model_version=f"episode_{self.metrics.get('last_episode', 0)}",
                    metrics=self.metrics
                )
                logger.info(f"💾 Saved checkpoint: {checkpoint_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all components"""
        return {
            'running': self.running,
            'environment': self.config.environment,
            'components': {
                name: status.value 
                for name, status in self.component_status.items()
            },
            'metrics': self.metrics,
            'config': {
                'dry_run': self.config.dry_run,
                'enable_rl_training': self.config.enable_rl_training,
                'enable_online_learning': self.config.enable_online_learning,
                'enable_shadow_mode': self.config.enable_shadow_mode,
                'enable_ab_testing': self.config.enable_ab_testing,
                'enable_google_ads': self.config.enable_google_ads,
                'enable_safety_controls': self.config.enable_safety_controls,
                'enable_explainability': self.config.enable_explainability
            }
        }
    
    def stop(self):
        """Stop all components gracefully"""
        logger.info("🛑 Stopping GAELP Production Orchestrator...")
        
        self.running = False
        
        # Stop all components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'stop'):
                    component.stop()
                    logger.info(f"Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
        
        logger.info("✅ GAELP Production Orchestrator stopped")

def main():
    """Main entry point"""
    # Load configuration
    config = OrchestratorConfig()
    
    # Override from environment variables
    config.environment = os.getenv('GAELP_ENV', 'production')
    config.dry_run = os.getenv('GAELP_DRY_RUN', 'false').lower() == 'true'
    config.google_ads_customer_id = os.getenv('GOOGLE_ADS_CUSTOMER_ID', '')
    
    # Create orchestrator
    orchestrator = GAELPProductionOrchestrator(config)
    
    # Start orchestrator
    if orchestrator.start():
        logger.info("🎉 GAELP Production Orchestrator is running!")
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(60)
                status = orchestrator.get_status()
                logger.info(f"📊 Status: {json.dumps(status, indent=2)}")
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
            orchestrator.stop()
    else:
        logger.error("Failed to start orchestrator")
        sys.exit(1)

if __name__ == "__main__":
    main()