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
from discovery_engine import RealTimeGA4Pipeline
from segment_discovery import SegmentDiscoveryEngine
from pipeline_integration import GAELPModelUpdater

# Attribution & Budget
from attribution_system import MultiTouchAttributionEngine
from budget_optimizer import DynamicBudgetOptimizer
from budget_safety_controller import BudgetSafetyController

# Safety & Monitoring  
from emergency_controls import EmergencyController
from convergence_monitor import ConvergenceMonitor
from regression_detector import RegressionDetector
from production_checkpoint_manager import ProductionCheckpointManager

# Production Features
from production_online_learner import ProductionOnlineLearner
from shadow_mode_manager import ShadowModeManager
from statistical_ab_testing_framework import StatisticalABTestingFramework
from bid_explainability_system import BidExplainabilitySystem

# Google Ads Integration
from google_ads_production_manager import GoogleAdsProductionManager
from google_ads_gaelp_integration import GoogleAdsGAELPIntegration

# Success Criteria & Monitoring
from gaelp_success_criteria_monitor import SuccessCriteriaMonitor

# Creative & Auction
from creative_content_analyzer import CreativeContentAnalyzer
from auction_gym_integration_fixed import FixedAuctionGymIntegration

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
        budget_config = {
            'daily_limit': self.config.max_daily_spend,
            'max_bid': self.config.max_bid_amount,
            'monitoring_interval': 30
        }
        self.components['budget_safety'] = BudgetSafetyController(budget_config)
        self.component_status['budget_safety'] = ComponentStatus.RUNNING
        
        # Success criteria monitor
        self.components['success_monitor'] = SuccessCriteriaMonitor()
        self.component_status['success_monitor'] = ComponentStatus.RUNNING
        
        logger.info("✅ Safety systems ready")
    
    def _init_data_pipeline(self):
        """Initialize data pipeline and discovery"""
        logger.info("📊 Initializing data pipeline...")
        
        # GA4 real-time pipeline
        self.components['ga4_pipeline'] = RealTimeGA4Pipeline(
            property_id=self.config.ga4_property_id,
            streaming_interval=5
        )
        self.component_status['ga4_pipeline'] = ComponentStatus.RUNNING
        
        # Segment discovery
        self.components['segment_discovery'] = SegmentDiscoveryEngine()
        self.component_status['segment_discovery'] = ComponentStatus.RUNNING
        
        # Model updater
        self.components['model_updater'] = GAELPModelUpdater()
        self.component_status['model_updater'] = ComponentStatus.RUNNING
        
        logger.info("✅ Data pipeline ready")
    
    def _init_rl_system(self):
        """Initialize core RL components"""
        logger.info("🤖 Initializing RL system...")
        
        # Environment
        self.components['environment'] = ProductionFortifiedEnvironment()
        self.component_status['environment'] = ComponentStatus.RUNNING
        
        # RL Agent with all improvements
        agent_config = {
            'enable_double_dqn': self.config.enable_double_dqn,
            'enable_prioritized_replay': self.config.enable_prioritized_replay,
            'enable_lstm': self.config.enable_lstm_sequence,
            'epsilon_decay': 0.99995,  # Fixed from TODO #1
            'training_frequency': 32,   # Fixed from TODO #2
            'target_update_freq': 1000, # Fixed from TODO #13
        }
        self.components['rl_agent'] = ProductionFortifiedRLAgent(**agent_config)
        self.component_status['rl_agent'] = ComponentStatus.RUNNING
        
        # Auction integration
        self.components['auction'] = FixedAuctionGymIntegration()
        self.component_status['auction'] = ComponentStatus.RUNNING
        
        # Creative analyzer
        self.components['creative_analyzer'] = CreativeContentAnalyzer()
        self.component_status['creative_analyzer'] = ComponentStatus.RUNNING
        
        logger.info("✅ RL system ready")
    
    def _init_attribution_budget(self):
        """Initialize attribution and budget systems"""
        logger.info("💰 Initializing attribution & budget...")
        
        # Multi-touch attribution
        self.components['attribution'] = MultiTouchAttributionEngine(
            database_path="attribution_system.db"
        )
        self.component_status['attribution'] = ComponentStatus.RUNNING
        
        # Dynamic budget optimizer
        self.components['budget_optimizer'] = DynamicBudgetOptimizer()
        self.component_status['budget_optimizer'] = ComponentStatus.RUNNING
        
        logger.info("✅ Attribution & budget ready")
    
    def _init_monitoring(self):
        """Initialize monitoring and validation"""
        logger.info("📈 Initializing monitoring...")
        
        # Convergence monitor
        self.components['convergence_monitor'] = ConvergenceMonitor()
        self.component_status['convergence_monitor'] = ComponentStatus.RUNNING
        
        # Regression detector
        self.components['regression_detector'] = RegressionDetector()
        self.component_status['regression_detector'] = ComponentStatus.RUNNING
        
        # Checkpoint manager
        self.components['checkpoint_manager'] = ProductionCheckpointManager(
            checkpoint_dir="production_checkpoints"
        )
        self.component_status['checkpoint_manager'] = ComponentStatus.RUNNING
        
        logger.info("✅ Monitoring ready")
    
    def _init_online_learning(self):
        """Initialize online learning system"""
        logger.info("🔄 Initializing online learning...")
        
        self.components['online_learner'] = ProductionOnlineLearner(
            rl_agent=self.components['rl_agent'],
            environment=self.components['environment']
        )
        self.component_status['online_learner'] = ComponentStatus.RUNNING
        
        logger.info("✅ Online learning ready")
    
    def _init_shadow_mode(self):
        """Initialize shadow mode testing"""
        logger.info("👻 Initializing shadow mode...")
        
        self.components['shadow_mode'] = ShadowModeManager()
        self.component_status['shadow_mode'] = ComponentStatus.RUNNING
        
        logger.info("✅ Shadow mode ready")
    
    def _init_ab_testing(self):
        """Initialize A/B testing framework"""
        logger.info("🧪 Initializing A/B testing...")
        
        self.components['ab_testing'] = StatisticalABTestingFramework()
        self.component_status['ab_testing'] = ComponentStatus.RUNNING
        
        logger.info("✅ A/B testing ready")
    
    def _init_explainability(self):
        """Initialize explainability system"""
        logger.info("💡 Initializing explainability...")
        
        self.components['explainability'] = BidExplainabilitySystem()
        self.component_status['explainability'] = ComponentStatus.RUNNING
        
        logger.info("✅ Explainability ready")
    
    def _init_google_ads(self):
        """Initialize Google Ads integration"""
        logger.info("📱 Initializing Google Ads...")
        
        if not self.config.google_ads_customer_id:
            logger.warning("⚠️ Google Ads customer ID not set, skipping integration")
            return
        
        self.components['google_ads'] = GoogleAdsGAELPIntegration(
            customer_id=self.config.google_ads_customer_id,
            rl_agent=self.components['rl_agent']
        )
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
                logger.error(f"Training loop error: {e}")
                time.sleep(30)
    
    def _run_training_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single training episode"""
        env = self.components['environment']
        agent = self.components['rl_agent']
        
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            # Get action from agent
            action = agent.act(state)
            
            # Generate explanation if enabled
            if self.config.enable_explainability:
                explanation = self.components['explainability'].explain_bid_decision(
                    state, action
                )
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train if enough experiences
            if len(agent.memory) > agent.batch_size and steps % 32 == 0:
                agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Check safety violations
            if self._check_safety_violations(action, info):
                logger.warning("🛑 Safety violation detected, ending episode")
                done = True
        
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
            violation = budget_safety.check_spending_limits(
                campaign_id='training',
                amount=info.get('bid_amount', 0)
            )
            if violation:
                return True
        
        # Check emergency controls
        emergency = self.components.get('emergency_controller')
        if emergency:
            if emergency.check_emergency_stop():
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
            regression = detector.check_for_regression(self.metrics)
            if regression:
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
                    metadata={
                        'metrics': self.metrics,
                        'episode': self.metrics.get('last_episode', 0)
                    }
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