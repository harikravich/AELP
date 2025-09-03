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
    from statistical_ab_testing_framework import StatisticalABTestFramework
except ImportError:
    logger.warning("StatisticalABTestFramework not available")
    StatisticalABTestFramework = None

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
    # Import the GAELP Google Ads agent under the expected name
    from google_ads_gaelp_integration import GAELPGoogleAdsAgent as GoogleAdsGAELPIntegration
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
        
        # Segment discovery tracking
        self.discovered_segments = {}
        self.segments_last_updated = None
        self.segment_update_interval = timedelta(hours=2)  # Update segments every 2 hours
        self.episodes_since_segment_update = 0
        self.segment_update_frequency = 100  # Update every 100 episodes
        
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
            import traceback
            logger.error(f"❌ Component initialization failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
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
        
        # Segment discovery with GA4 integration
        if SegmentDiscoveryEngine:
            # Initialize with GA4 data pipeline for data access
            try:
                # Use the existing GA4 pipeline we already created
                ga4_pipeline = self.components['ga4_pipeline']
                self.components['segment_discovery'] = SegmentDiscoveryEngine(
                    min_cluster_size=20,
                    max_clusters=15,
                    ga4_discovery_engine=ga4_pipeline  # Pass the pipeline
                )
                self.component_status['segment_discovery'] = ComponentStatus.RUNNING
                logger.info("✅ SegmentDiscoveryEngine initialized with GA4 pipeline")
            except Exception as e:
                logger.warning(f"⚠️ Could not pass GA4 pipeline: {e}, using default")
                self.components['segment_discovery'] = SegmentDiscoveryEngine()
                self.component_status['segment_discovery'] = ComponentStatus.RUNNING
        
        # Model updater
        if GAELPModelUpdater:
            self.components['model_updater'] = GAELPModelUpdater()
            self.component_status['model_updater'] = ComponentStatus.RUNNING
        
        logger.info("✅ Data pipeline ready")
    
    def _initial_segment_discovery(self):
        """Discover user segments before training starts"""
        logger.info("🔍 Starting initial segment discovery...")
        
        segment_discovery = self.components.get('segment_discovery')
        if not segment_discovery:
            logger.error("❌ SegmentDiscoveryEngine not available - cannot proceed without dynamic segments")
            raise RuntimeError("Segment discovery is required. No fallbacks allowed.")
        
        try:
            # Force initial discovery with data from GA4 pipeline
            self.discovered_segments = segment_discovery.discover_segments(force_rediscovery=True)
            
            if not self.discovered_segments:
                logger.error("❌ No segments discovered - cannot train without segments")
                raise RuntimeError("Failed to discover segments. Check GA4 data availability.")
            
            self.segments_last_updated = datetime.now()
            
            # Update environment and agent with discovered segments
            self._update_components_with_segments()
            
            logger.info(f"✅ Initial segment discovery complete: {len(self.discovered_segments)} segments found")
            for seg_id, segment in list(self.discovered_segments.items())[:3]:
                logger.info(f"   - {segment.name}: {segment.size} users, CVR: {segment.conversion_rate:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Critical failure in segment discovery: {e}")
            raise RuntimeError(f"Segment discovery failed: {e}. No fallbacks allowed.")
    
    def _should_update_segments(self) -> bool:
        """Check if segments should be updated"""
        if not self.segments_last_updated:
            return True
        
        # Time-based update
        time_based = datetime.now() - self.segments_last_updated > self.segment_update_interval
        
        # Episode-based update
        episode_based = self.episodes_since_segment_update >= self.segment_update_frequency
        
        return time_based or episode_based
    
    def _update_segments_if_needed(self):
        """Update segments if the update interval has passed"""
        if not self._should_update_segments():
            return
        
        logger.info("🔄 Updating discovered segments...")
        
        segment_discovery = self.components.get('segment_discovery')
        if not segment_discovery:
            logger.warning("⚠️ SegmentDiscoveryEngine not available for update")
            return
        
        try:
            # Rediscover segments with fresh data
            new_segments = segment_discovery.discover_segments(force_rediscovery=True)
            
            if new_segments:
                # Compare with existing segments
                old_count = len(self.discovered_segments)
                self.discovered_segments = new_segments
                new_count = len(self.discovered_segments)
                
                # Update components with new segments
                self._update_components_with_segments()
                
                self.segments_last_updated = datetime.now()
                self.episodes_since_segment_update = 0
                
                logger.info(f"✅ Segments updated: {old_count} → {new_count} segments")
            else:
                logger.warning("⚠️ No segments discovered in update - keeping existing segments")
            
        except Exception as e:
            logger.error(f"❌ Failed to update segments: {e}")
    
    def _update_components_with_segments(self):
        """Update environment and agent with discovered segments"""
        if not self.discovered_segments:
            return
        
        # Update environment with segments
        env = self.components.get('environment')
        if env and hasattr(env, 'update_discovered_segments'):
            env.update_discovered_segments(self.discovered_segments)
            logger.debug("✅ Environment updated with discovered segments")
        
        # Update agent with segments
        agent = self.components.get('rl_agent')
        if agent and hasattr(agent, 'update_discovered_segments'):
            agent.update_discovered_segments(self.discovered_segments)
            logger.debug("✅ Agent updated with discovered segments")
        
        # Update online learner with segments
        online_learner = self.components.get('online_learner')
        if online_learner and hasattr(online_learner, 'update_discovered_segments'):
            online_learner.update_discovered_segments(self.discovered_segments)
            logger.debug("✅ Online learner updated with discovered segments")
    
    def _init_rl_system(self):
        """Initialize core RL components"""
        logger.info("🤖 Initializing RL system...")
        
        # Create auction first if available
        auction_orchestrator = None
        if FixedAuctionGymIntegration:
            auction_orchestrator = FixedAuctionGymIntegration()
            self.components['auction'] = auction_orchestrator
            self.component_status['auction'] = ComponentStatus.RUNNING
            logger.info("✅ Auction orchestrator initialized")
        
        # Environment with auction orchestrator
        env = ProductionFortifiedEnvironment()
        # Pass auction orchestrator to environment
        if auction_orchestrator:
            env.auction_orchestrator = auction_orchestrator
        self.components['environment'] = env
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
        
        # Auction integration already created above and passed to environment
        
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
            from decimal import Decimal
            from budget_optimizer import OptimizationObjective
            self.components['budget_optimizer'] = DynamicBudgetOptimizer(
                daily_budget=Decimal(str(self.config.max_daily_spend)),
                optimization_objective=OptimizationObjective.MAXIMIZE_CONVERSIONS
            )
            self.component_status['budget_optimizer'] = ComponentStatus.RUNNING
        
        logger.info("✅ Attribution & budget ready")
    
    def _init_monitoring(self):
        """Initialize monitoring and validation"""
        logger.info("📈 Initializing monitoring...")
        
        # Convergence monitor
        if ConvergenceMonitor:
            try:
                agent = self.components.get('rl_agent')
                discovery = None
                env = self.components.get('environment')
                if env is not None and hasattr(env, 'discovery'):
                    discovery = getattr(env, 'discovery')
                elif 'ga4_pipeline' in self.components:
                    discovery = self.components['ga4_pipeline']

                if agent is not None and discovery is not None:
                    self.components['convergence_monitor'] = ConvergenceMonitor(
                        agent=agent,
                        discovery_engine=discovery,
                        checkpoint_dir="production_checkpoints"
                    )
                    self.component_status['convergence_monitor'] = ComponentStatus.RUNNING
                else:
                    logger.warning("ConvergenceMonitor prerequisites missing (agent or discovery). Skipping initialization.")
            except Exception as e:
                logger.warning(f"ConvergenceMonitor initialization failed: {e}")
        
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
        
        # Start continuous learning cycle in background
        if hasattr(self.components['online_learner'], 'continuous_learning_cycle'):
            learning_thread = threading.Thread(
                target=self._run_continuous_learning,
                name="continuous_learning"
            )
            learning_thread.daemon = True
            learning_thread.start()
            self.threads.append(learning_thread)
            logger.info("🔄 Continuous learning cycle started")
        
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
        
        # Start shadow mode testing - mark as running so it can process decisions
        self.components['shadow_mode'].is_running = True
        
        logger.info("✅ Shadow mode ready")
    
    def _init_ab_testing(self):
        """Initialize A/B testing framework"""
        logger.info("🧪 Initializing A/B testing...")
        
        if StatisticalABTestFramework:
            # Use the GA4 pipeline directly as discovery engine since they have similar interfaces
            ga4_pipeline = self.components['ga4_pipeline']
            
            # Initialize A/B testing with discovery engine and config
            from statistical_ab_testing_framework import StatisticalConfig
            config = StatisticalConfig(
                alpha=0.05,
                power=0.80,
                minimum_detectable_effect=0.05,
                minimum_sample_size=100,  # Lower for faster testing
                confidence_level=0.95,
                exploration_rate=0.1
            )
            
            self.components['ab_testing'] = StatisticalABTestFramework(config, ga4_pipeline)
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
        
        # CRITICAL: Discover user segments before training starts
        self._initial_segment_discovery()
        
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
                
                # CRITICAL: Update model with episode data and newly discovered patterns
                self._update_model_with_episode_data(episode, episode_metrics)
                
                # Update metrics
                with self.metrics_lock:
                    self.metrics['last_episode'] = episode
                    self.metrics['episode_metrics'] = episode_metrics

                # Episode trace (concise)
                try:
                    spend = episode_metrics.get('total_spend', 0.0)
                    revenue = episode_metrics.get('total_revenue', 0.0)
                    conversions = episode_metrics.get('conversions', 0)
                    auction = episode_metrics.get('auction_performance', {}) or {}
                    wins = auction.get('total_wins', 0)
                    total_auctions = auction.get('total_auctions', 0)
                    win_rate = auction.get('win_rate', 0.0)
                    avg_cpc = auction.get('avg_cpc', 0.0)
                    cac = (spend / max(conversions, 1)) if conversions else float('inf')
                    roas = (revenue / max(spend, 1e-6)) if spend > 0 else 0.0
                    logger.info(
                        f"📘 Episode {episode}: steps={episode_metrics.get('steps', 0)}, "
                        f"spend=${spend:.2f}, wins={wins}/{total_auctions} ({win_rate:.1%}), "
                        f"conversions={conversions}, CAC={'∞' if conversions==0 else f'${cac:.2f}'}, ROAS={roas:.2f}x, epsilon={episode_metrics.get('epsilon', 0):.3f}"
                    )
                except Exception as e:
                    logger.debug(f"Episode trace logging failed: {e}")

                # Persist episode metrics to BigQuery if configured
                try:
                    self._write_training_episode_to_bigquery(episode_metrics)
                except Exception as e:
                    logger.debug(f"BQ episode write failed: {e}")
                
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
        """Run a single training episode with A/B testing integration"""
        
        # Update segments periodically during training
        self._update_segments_if_needed()
        self.episodes_since_segment_update += 1
        
        env = self.components['environment']
        agent = self.components['rl_agent']
        shadow_mode = self.components.get('shadow_mode') if self.config.enable_shadow_mode else None
        ab_testing = self.components.get('ab_testing') if self.config.enable_ab_testing else None
        attribution_engine = self.components.get('attribution')
        budget_optimizer = self.components.get('budget_optimizer')
        
        # BUDGET OPTIMIZATION: Get optimal allocation for this episode
        current_hour = datetime.now().hour
        budget_constraint = None
        optimal_allocations = {}
        pacing_multiplier = 1.0
        
        if budget_optimizer:
            try:
                # Get optimal hourly allocation
                from budget_optimizer import PacingStrategy
                optimization_result = budget_optimizer.optimize_hourly_allocation(
                    strategy=PacingStrategy.ADAPTIVE_ML
                )
                optimal_allocations = optimization_result.allocations
                logger.info(f"Budget optimization complete. Confidence: {optimization_result.confidence_score:.2f}")
                
                # Check for early exhaustion risk
                at_risk, risk_reason, recommended_cap = budget_optimizer.prevent_early_exhaustion(current_hour)
                if at_risk:
                    logger.warning(f"🚨 Budget exhaustion risk: {risk_reason}")
                    budget_constraint = recommended_cap
                
                # Get pacing multiplier for current hour
                pacing_multiplier = budget_optimizer.get_pacing_multiplier(current_hour)
                logger.info(f"💰 Budget pacing multiplier for hour {current_hour}: {pacing_multiplier:.2f}")
                
                # Apply performance-based reallocation if needed
                reallocation = budget_optimizer.reallocate_based_on_performance()
                if reallocation:
                    optimal_allocations = reallocation
                    logger.info("📈 Applied performance-based budget reallocation")
                
            except Exception as e:
                logger.error(f"❌ Budget optimization failed: {e}")
                budget_constraint = None
                pacing_multiplier = 1.0
        
        # Initialize episode attribution tracking
        episode_touchpoints = []  # Track all touchpoints in this episode
        episode_user_id = f"training_user_{episode}_{int(datetime.now().timestamp())}"
        attributed_rewards = {}  # Track attributed rewards by touchpoint
        
        # Initialize episode metrics
        total_spend = 0.0  # Track total spend for this episode
        total_revenue = 0.0  # Track total revenue for this episode
        conversions_count = 0
        
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
        episode_experiences = []  # Store experiences for model updating
        shadow_comparisons = []
        auction_metrics = []  # Track auction outcomes for analysis
        
        # A/B testing setup for this episode
        active_test_id = None
        selected_variant = None
        user_id = f"episode_{episode}_user_{np.random.randint(1000000)}"
        
        # Check for active A/B tests
        if ab_testing:
            active_tests = ab_testing.list_active_tests()
            if active_tests:
                # Use the first active test for policy comparison
                test = active_tests[0]
                active_test_id = test['test_id']
                
                # Create context from current state for variant assignment
                context = self._create_ab_context_from_state(state)
                selected_variant = ab_testing.assign_variant(active_test_id, user_id, context)
                logger.debug(f"Episode {episode}: Using A/B test {active_test_id}, variant {selected_variant}")
        
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
                
                # Enrich with discovered segment data
                self._enrich_state_with_segments(enriched_state)
                
                # CRITICAL: Analyze actual creative content, not just IDs
                creative_analyzer = self.components.get('creative_analyzer')
                if creative_analyzer and hasattr(env, 'creative_selector'):
                    creative_content = self._get_creative_content_for_state(env, enriched_state)
                    if creative_content:
                        content_features = creative_analyzer.analyze_creative(creative_content)
                        # Enrich state with actual creative content features
                        enriched_state = self._enrich_state_with_creative_features(enriched_state, content_features)
                        logger.debug(f"Enriched state with creative features: message_frame={content_features.message_frame}, "
                                   f"predicted_ctr={content_features.predicted_ctr:.3f}, urgency={content_features.uses_urgency}")
            
            # SHADOW MODE: Run parallel decisions BEFORE taking the real action
            shadow_decisions = {}
            if shadow_mode and shadow_mode.is_running:
                try:
                    # Generate synthetic user ID for this decision
                    user_id = f"episode_{episode}_step_{steps}"
                    
                    # Create context for shadow decision
                    context = {
                        'competition_level': enriched_state.competition_level,
                        'budget_remaining_pct': enriched_state.budget_remaining_pct,
                        'episode': episode,
                        'step': steps,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Convert enriched_state to shadow mode state
                    from shadow_mode_state import DynamicEnrichedState as ShadowState
                    shadow_user_state = ShadowState()
                    shadow_user_state.stage = enriched_state.stage
                    shadow_user_state.touchpoints_seen = enriched_state.touchpoints_seen
                    shadow_user_state.days_since_first_touch = enriched_state.days_since_first_touch
                    shadow_user_state.segment_index = enriched_state.segment_index
                    shadow_user_state.device_index = enriched_state.device_index
                    shadow_user_state.channel_index = enriched_state.channel_index
                    shadow_user_state.creative_index = enriched_state.creative_index
                    shadow_user_state.competition_level = enriched_state.competition_level
                    shadow_user_state.budget_remaining_pct = enriched_state.budget_remaining_pct
                    shadow_user_state.current_bid = enriched_state.current_bid
                    
                    # Run shadow decisions in parallel
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        shadow_decisions = loop.run_until_complete(
                            shadow_mode._run_parallel_decisions(user_id, shadow_user_state, context)
                        )
                        logger.debug(f"Shadow decisions for step {steps}: {len(shadow_decisions)} models")
                    finally:
                        loop.close()
                        
                except Exception as e:
                    logger.warning(f"Shadow mode failed for step {steps}: {e}")
                    shadow_decisions = {}
            
            # Apply A/B test variant policy parameters if active test
            if selected_variant and active_test_id and ab_testing:
                self._apply_variant_policy_parameters(agent, selected_variant, ab_testing, active_test_id)
            
            # Get PRIMARY action from production agent
            action = agent.select_action(enriched_state)
            
            # BUDGET OPTIMIZATION: Apply intelligent budget constraints to action
            # Handle both dictionary and object actions
            if isinstance(action, dict):
                has_bid_for_budget = 'bid_amount' in action
                original_bid = action.get('bid_amount', 0.0) if has_bid_for_budget else 0.0
            else:
                has_bid_for_budget = hasattr(action, 'bid_amount')
                original_bid = getattr(action, 'bid_amount', 0.0) if has_bid_for_budget else 0.0
            
            if budget_optimizer and has_bid_for_budget:
                
                # Apply pacing multiplier to bid
                adjusted_bid = original_bid * pacing_multiplier
                
                # Apply budget constraint if at risk of exhaustion
                if budget_constraint and adjusted_bid > float(budget_constraint):
                    adjusted_bid = float(budget_constraint)
                    logger.warning(f"💰 Bid constrained due to exhaustion risk: ${original_bid:.2f} → ${adjusted_bid:.2f}")
                
                # Apply hourly allocation constraint if available
                if optimal_allocations and current_hour in optimal_allocations:
                    hourly_budget = float(optimal_allocations[current_hour])
                    # Simple constraint: don't exceed 10% of hourly budget in single bid
                    max_single_bid = hourly_budget * 0.1
                    if adjusted_bid > max_single_bid:
                        adjusted_bid = max_single_bid
                        logger.info(f"💰 Bid constrained to hourly allocation: ${original_bid:.2f} → ${adjusted_bid:.2f}")
                
                # Update action with adjusted bid
                if isinstance(action, dict):
                    action['bid_amount'] = adjusted_bid
                else:
                    action.bid_amount = adjusted_bid
                logger.debug(f"💰 Budget-optimized bid: ${original_bid:.2f} → ${adjusted_bid:.2f} (pacing: {pacing_multiplier:.2f})")
            elif budget_optimizer:
                logger.debug("💰 Budget optimizer active but action has no bid_amount attribute")
            
            # Track this action as a touchpoint if attribution engine is available
            touchpoint_id = None
            if attribution_engine:
                touchpoint_id = self._track_training_touchpoint(
                    attribution_engine, enriched_state, action, episode_user_id, steps
                )
                if touchpoint_id:
                    episode_touchpoints.append({
                        'id': touchpoint_id,
                        'step': steps,
                        'state': enriched_state,
                        'action': action,
                        'timestamp': datetime.now()
                    })
            
            # SHADOW MODE: Record the production decision
            if isinstance(action, dict):
                production_decision = {
                    'model_name': 'production',
                    'user_id': f"episode_{episode}_step_{steps}",
                    'bid_amount': action.get('bid_amount', 0.0),
                    'creative_id': action.get('creative_id', 0),
                    'channel': action.get('channel', 'unknown'),
                    'confidence': action.get('confidence', 0.5),
                    'timestamp': datetime.now()
                }
            else:
                production_decision = {
                    'model_name': 'production',
                    'user_id': f"episode_{episode}_step_{steps}",
                    'bid_amount': getattr(action, 'bid_amount', 0.0),
                    'creative_id': getattr(action, 'creative_id', 0),
                    'channel': getattr(action, 'channel', 'unknown'),
                    'confidence': getattr(action, 'confidence', 0.5),
                    'timestamp': datetime.now()
                }
            
            # Generate explanation if enabled
            if self.config.enable_explainability and 'explainability' in self.components:
                explanation = self.components['explainability'].explain_bid_decision(
                    state, action
                )
                production_decision['explanation'] = explanation
            
            # CRITICAL: Use orchestrator auction component for REAL second-price mechanics
            auction = self.components.get('auction')
            # Handle both dictionary and object actions
            if isinstance(action, dict):
                has_bid = 'bid_amount' in action
                bid_amount = float(action.get('bid_amount', 1.0)) if has_bid else 1.0
            else:
                has_bid = hasattr(action, 'bid_amount')
                bid_amount = float(getattr(action, 'bid_amount', 1.0)) if has_bid else 1.0
            
            if auction and has_bid:
                # Run auction using production auction component with proper GSP mechanics
                
                # Build comprehensive query context from state and action
                query_context = {
                    'query_value': bid_amount * 1.5,  # Estimated query value
                    'user_segment': enriched_state.segment_index,
                    'device_type': enriched_state.device_index, 
                    'channel_index': enriched_state.channel_index,
                    'stage': enriched_state.stage,
                    'touchpoints': enriched_state.touchpoints_seen,
                    'competition_level': enriched_state.competition_level,
                    'hour': current_hour,  # Use actual current hour for time-based competition
                    'cvr': 0.02,  # Base conversion rate
                    'ltv': 199.98,  # Aura customer LTV
                    'budget_remaining_pct': enriched_state.budget_remaining_pct
                }
                
                # Run REAL auction with GSP mechanics and competitive bidding
                auction_outcome = auction.run_auction(bid_amount, query_context)
                
                # Track auction metrics for analysis
                auction_metrics.append({
                    'step': steps,
                    'bid_amount': bid_amount,
                    'won': auction_outcome['won'],
                    'cost': auction_outcome['cost'],
                    'position': auction_outcome['position'],
                    'competitors': auction_outcome['competitors'],
                    'cpc': auction_outcome['cpc'],
                    'source': 'orchestrator_gsp'
                })
                
                # Create enhanced action with auction results for environment
                if isinstance(action, dict):
                    # Preserve original action keys for training
                    enhanced_action = dict(action)  # Copy original action
                    enhanced_action.update({
                        'bid_amount': bid_amount,
                        'creative_id': action.get('creative_id', 0),
                        'channel': action.get('channel', 'search'),
                        'channel_action': enriched_state.channel_index,
                        'auction_override': auction_outcome,
                        'use_real_auction': True  # Signal to environment
                    })
                    action = enhanced_action
                else:
                    # Handle object-based actions
                    enhanced_action = {
                        'bid_amount': bid_amount,
                        'creative_id': getattr(action, 'creative_id', 0),
                        'channel': getattr(action, 'channel', 'search'),
                        'channel_action': enriched_state.channel_index,
                        'auction_override': auction_outcome,
                        'use_real_auction': True  # Signal to environment
                    }
                    for key, value in enhanced_action.items():
                        setattr(action, key, value)
                
                logger.debug(f"Step {steps}: Auction result - Won: {auction_outcome['won']}, "
                           f"Price: ${auction_outcome['cost']:.2f}, Position: {auction_outcome['position']}")
            else:
                # If no auction component or no bid amount, we still need to provide the required fields
                # Since the environment doesn't allow fallbacks, this will trigger an error there
                logger.warning(f"Step {steps}: Auction component not available or no bid amount in action")
                if isinstance(action, dict):
                    action['use_real_auction'] = False
                    action['auction_override'] = None
            
            # Step environment with PRODUCTION action (enhanced with real auction results)
            step_result = env.step(action)
            # Handle both old (4 values) and new (5 values) Gym API
            if len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
                done = done or truncated  # Combine termination conditions
            else:
                next_state, reward, done, info = step_result
            
            # BUDGET OPTIMIZATION: Track performance data for learning
            if budget_optimizer:
                try:
                    # Extract spend and performance data from step result
                    # Count spend only when actual auction cost is reported
                    step_spend = float(info.get('cost', 0.0)) if 'cost' in info else 0.0
                    step_impressions = info.get('impressions', 1) if 'impressions' in info else 1
                    step_clicks = info.get('clicks', 1 if step_spend > 0 else 0) if 'clicks' in info else (1 if step_spend > 0 else 0)
                    step_conversions = 1 if reward > 0 else 0
                    step_revenue = float(info.get('revenue', 0.0)) if 'revenue' in info else max(0.0, float(reward) * 100.0)
                    
                    # Create performance window for this step
                    from budget_optimizer import PerformanceWindow
                    from decimal import Decimal
                    
                    performance_window = PerformanceWindow(
                        start_time=datetime.now() - timedelta(seconds=30),  # Approximate step duration
                        end_time=datetime.now(),
                        spend=Decimal(str(step_spend)),
                        impressions=step_impressions,
                        clicks=step_clicks,
                        conversions=step_conversions,
                        revenue=Decimal(str(step_revenue)),
                        roas=step_revenue / max(0.01, step_spend) if step_spend > 0 else 0,
                        cpa=Decimal(str(step_spend / max(1, step_conversions))) if step_conversions > 0 else Decimal('999'),
                        cvr=step_conversions / max(1, step_clicks) if step_clicks > 0 else 0,
                        cpc=Decimal(str(step_spend / max(1, step_clicks))) if step_clicks > 0 else Decimal('0'),
                        quality_score=info.get('quality_score', 7.0) if 'quality_score' in info else 7.0
                    )
                    
                    # Add to budget optimizer for learning
                    budget_optimizer.add_performance_data(performance_window)
                    
                    # Track total episode spend
                    total_spend += step_spend
                    total_revenue += step_revenue
                    
                    logger.debug(f"💰 Performance tracked: spend=${step_spend:.2f}, reward=${reward:.2f}, "
                               f"conversions={step_conversions}, ROAS={performance_window.roas:.2f}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to track budget performance: {e}")
            
            # Check if this is a conversion (positive reward above threshold)
            is_conversion = self._is_conversion_event(reward, info)
            
            # If conversion detected, track it and attribute rewards
            if is_conversion and attribution_engine and episode_touchpoints:
                conversions_count += 1
                conversion_touchpoint_id = self._track_conversion(
                    attribution_engine, reward, episode_user_id, info
                )
                
                if conversion_touchpoint_id:
                    # Calculate multi-touch attribution for all touchpoints in episode
                    attributed_rewards = self._calculate_episode_attribution(
                        attribution_engine, episode_user_id, episode_touchpoints, reward
                    )
                    
                    # Log attribution results
                    logger.info(f"🎯 Conversion detected (${reward:.2f}) - attributed across {len(attributed_rewards)} touchpoints")
                    for tp_id, attr_reward in attributed_rewards.items():
                        logger.debug(f"   Touchpoint {tp_id}: ${attr_reward:.3f} attributed")
            
            
            # Store experience with attribution information
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'touchpoint_id': touchpoint_id,
                'attributed_reward': attributed_rewards.get(touchpoint_id, reward),
                'is_conversion': is_conversion,
                'episode_touchpoints': len(episode_touchpoints),
                'info': info,
                'step': steps,
                'episode': episode
            }
            agent.replay_buffer.add(experience)
            episode_experiences.append(experience)
            
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
                    
                    # Enrich next state with discovered segment data
                    self._enrich_state_with_segments(enriched_next_state)
                
                # Call the train method with attributed rewards if available
                if hasattr(agent, 'train'):
                    training_reward = attributed_rewards.get(touchpoint_id, reward)
                    agent.train(enriched_state, action, training_reward, enriched_next_state, done)
            
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
        
        # Record A/B test observation if active test
        if selected_variant and active_test_id and ab_testing:
            self._record_ab_test_observation(
                ab_testing, active_test_id, selected_variant, user_id, 
                total_reward, steps, state, info if 'info' in locals() else {}
            )
        
        # Update online learner with episode results
        self._update_online_learner_from_episode({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': agent.epsilon,
            'final_state': state,
            'env_info': info if 'info' in locals() else {}
        })
        
        # Check if we should analyze A/B test results
        if active_test_id and ab_testing and episode % 100 == 0:  # Check every 100 episodes
            self._analyze_ab_test_results(ab_testing, active_test_id)
        
        # Generate shadow mode episode summary
        shadow_summary = {}
        if shadow_mode and shadow_comparisons:
            try:
                # Calculate episode-level shadow metrics
                all_divergences = []
                model_divergences = {}
                
                for comparison in shadow_comparisons:
                    for model_name, divergence in comparison['divergences'].items():
                        if model_name not in model_divergences:
                            model_divergences[model_name] = []
                        model_divergences[model_name].append(divergence)
                        all_divergences.append(divergence)
                
                shadow_summary = {
                    'total_shadow_comparisons': len(shadow_comparisons),
                    'avg_overall_divergence': np.mean(all_divergences) if all_divergences else 0,
                    'max_divergence': np.max(all_divergences) if all_divergences else 0,
                    'high_divergence_count': sum(1 for d in all_divergences if d > 0.2),
                    'model_avg_divergences': {
                        model: np.mean(divs) for model, divs in model_divergences.items()
                    },
                    'shadow_models_active': list(model_divergences.keys())
                }
                
                # Get current shadow mode performance report
                if shadow_mode and hasattr(shadow_mode, 'generate_performance_report'):
                    shadow_performance = shadow_mode.generate_performance_report()
                    shadow_summary['performance_snapshot'] = shadow_performance
                
                logger.info(f"Episode {episode} shadow summary: {shadow_summary['total_shadow_comparisons']} comparisons, "
                          f"avg divergence: {shadow_summary['avg_overall_divergence']:.3f}")
                          
            except Exception as e:
                logger.warning(f"Failed to generate shadow summary for episode {episode}: {e}")
                shadow_summary = {'error': str(e)}

        # Calculate attribution summary for episode
        attribution_summary = {}
        if attributed_rewards:
            attribution_summary = {
                'total_attributed_touchpoints': len(attributed_rewards),
                'total_attributed_value': sum(attributed_rewards.values()),
                'max_attributed_value': max(attributed_rewards.values()) if attributed_rewards else 0,
                'attribution_distribution': list(attributed_rewards.values())
            }

        # BUDGET OPTIMIZATION: Get final budget status
        budget_summary = {}
        if budget_optimizer:
            try:
                budget_status = budget_optimizer.get_optimization_status()
                budget_summary = {
                    'total_spend': total_spend,
                    'pacing_multiplier': pacing_multiplier,
                    'budget_utilization': budget_status['budget_status']['daily_utilization'],
                    'optimal_allocations_count': len(optimal_allocations),
                    'exhaustion_risk': at_risk if 'at_risk' in locals() else False,
                    'pattern_confidence': budget_status['optimization_status']['pattern_confidence'],
                    'learned_patterns': budget_status['optimization_status']['learned_patterns'],
                    'performance_windows': budget_status['risk_assessment']['total_performance_windows']
                }
                logger.info(f"💰 Episode budget summary: spend=${total_spend:.2f}, "
                           f"utilization={budget_summary['budget_utilization']:.1%}")
            except Exception as e:
                logger.error(f"❌ Failed to get budget summary: {e}")
                budget_summary = {'error': str(e), 'total_spend': total_spend}

        # Calculate auction performance summary
        auction_summary = self._calculate_auction_summary(auction_metrics)
        
        return {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'total_spend': total_spend,
            'total_revenue': total_revenue,
            'conversions': conversions_count,
            'epsilon': agent.epsilon,
            'segments_count': len(self.discovered_segments),
            'segments_last_updated': self.segments_last_updated.isoformat() if self.segments_last_updated else None,
            'ab_test_id': active_test_id,
            'ab_variant': selected_variant,
            'shadow_mode': shadow_summary,
            'touchpoints_tracked': len(episode_touchpoints),
            'episode_experiences': episode_experiences,
            'conversions_detected': 1 if attributed_rewards else 0,
            'attribution_summary': attribution_summary,
            'budget_optimization': budget_summary,
            'auction_performance': auction_summary,  # CRITICAL: Real auction performance metrics
            'creative_content_analyzed': 'creative_analyzer' in self.components and self.components['creative_analyzer'] is not None,
            'creative_features_integrated': True  # Now using actual creative content features in state
        }
    
    def _calculate_auction_summary(self, auction_metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive auction performance summary"""
        if not auction_metrics:
            return {
                'total_auctions': 0,
                'win_rate': 0.0,
                'avg_cpc': 0.0,
                'total_spend': 0.0,
                'source': 'no_auctions'
            }
        
        won_auctions = [m for m in auction_metrics if m['won']]
        total_auctions = len(auction_metrics)
        total_wins = len(won_auctions)
        total_spend = sum(m['cost'] for m in auction_metrics)
        
        # Calculate win rate
        win_rate = total_wins / total_auctions if total_auctions > 0 else 0.0
        
        # Calculate average CPC (cost per click)
        avg_cpc = total_spend / total_wins if total_wins > 0 else 0.0
        
        # Position distribution
        position_dist = {}
        positions = [m['position'] for m in won_auctions if m['position'] > 0]
        if positions:
            for pos in range(1, 5):  # Positions 1-4
                position_dist[f'position_{pos}'] = positions.count(pos) / len(positions)
        
        # Competition analysis
        competitor_counts = [m['competitors'] for m in auction_metrics]
        avg_competitors = sum(competitor_counts) / len(competitor_counts) if competitor_counts else 0
        
        # Bid efficiency (win rate vs bid amount)
        avg_bid = sum(m['bid_amount'] for m in auction_metrics) / total_auctions if total_auctions > 0 else 0.0
        
        summary = {
            'total_auctions': total_auctions,
            'total_wins': total_wins,
            'win_rate': win_rate,
            'avg_cpc': avg_cpc,
            'total_spend': total_spend,
            'avg_bid': avg_bid,
            'avg_competitors': avg_competitors,
            'position_distribution': position_dist,
            'source': auction_metrics[0]['source'] if auction_metrics else 'unknown',
            'bid_efficiency': win_rate / max(avg_bid, 0.01) if avg_bid > 0 else 0.0  # wins per dollar bid
        }
        
        # Log auction performance for monitoring
        if total_auctions > 0:
            logger.info(f"🎯 Episode auction summary: {total_wins}/{total_auctions} won "
                       f"({win_rate:.1%}), avg CPC: ${avg_cpc:.2f}, spend: ${total_spend:.2f}")
            
            # CRITICAL: Check if win rate is realistic (should be 15-35% in competitive market)
            if win_rate > 0.50:
                logger.warning(f"⚠️ Win rate suspiciously high: {win_rate:.1%} - check auction competition")
            elif win_rate < 0.05 and total_auctions > 10:
                logger.warning(f"⚠️ Win rate very low: {win_rate:.1%} - bids may be uncompetitive")
        
        return summary
    
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
            try:
                # Prefer explicit API if available
                if hasattr(monitor, 'check_convergence_issues'):
                    issues = monitor.check_convergence_issues(self.metrics.get('episode_metrics', {}))
                    if issues:
                        return True
                # Otherwise, use emergency/stop flags from the production monitor
                elif getattr(monitor, 'emergency_stop_triggered', False):
                    logger.warning("Convergence monitor indicates emergency stop triggered")
                    return True
                elif hasattr(monitor, 'should_stop') and callable(getattr(monitor, 'should_stop')):
                    if monitor.should_stop():
                        logger.warning("Convergence monitor indicates training should stop")
                        return True
            except Exception as e:
                logger.warning(f"Convergence monitor check failed: {e}")
        
        # Check regression detector
        detector = self.components.get('regression_detector')
        if detector:
            regressions = detector.check_for_regressions()
            if regressions:
                logger.warning(f"Detected {len(regressions)} regressions")
                return True
        
        return False

    def _write_training_episode_to_bigquery(self, metrics: Dict[str, Any]):
        """Write episode-level metrics to BigQuery training dataset if configured."""
        import os
        from datetime import datetime
        project = os.getenv('GOOGLE_CLOUD_PROJECT')
        dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
        if not project or not dataset:
            return  # Not configured
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=project)
            table_id = f"{project}.{dataset}.training_episodes"
            # Ensure table exists
            try:
                client.get_table(table_id)
            except Exception:
                schema = [
                    bigquery.SchemaField("timestamp", "TIMESTAMP"),
                    bigquery.SchemaField("episode", "INT64"),
                    bigquery.SchemaField("steps", "INT64"),
                    bigquery.SchemaField("spend", "FLOAT64"),
                    bigquery.SchemaField("revenue", "FLOAT64"),
                    bigquery.SchemaField("conversions", "INT64"),
                    bigquery.SchemaField("win_rate", "FLOAT64"),
                    bigquery.SchemaField("avg_cpc", "FLOAT64"),
                    bigquery.SchemaField("epsilon", "FLOAT64"),
                ]
                table = bigquery.Table(table_id, schema=schema)
                client.create_table(table, exists_ok=True)
            auction = metrics.get('auction_performance', {}) or {}
            row = {
                "timestamp": datetime.now().isoformat(),
                "episode": int(metrics.get('episode', 0)),
                "steps": int(metrics.get('steps', 0)),
                "spend": float(metrics.get('total_spend', 0.0)),
                "revenue": float(metrics.get('total_revenue', 0.0)),
                "conversions": int(metrics.get('conversions', 0)),
                "win_rate": float(auction.get('win_rate', 0.0)),
                "avg_cpc": float(auction.get('avg_cpc', 0.0)),
                "epsilon": float(metrics.get('epsilon', 0.0)),
            }
            client.insert_rows_json(table_id, [row])
        except Exception as e:
            logger.debug(f"Failed to write episode metrics to BigQuery: {e}")
    
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
            
            # Special handling for shadow mode metrics
            shadow_mode = self.components.get('shadow_mode')
            if shadow_mode and self.config.enable_shadow_mode:
                try:
                    shadow_report = shadow_mode.generate_performance_report()
                    self.metrics['shadow_mode'] = {
                        'performance_report': shadow_report,
                        'comparison_count': len(shadow_mode.comparison_history),
                        'models_active': list(shadow_mode.models.keys()),
                        'session_id': shadow_mode.session_id,
                        'database_path': shadow_mode.db_path,
                        'statistical_results': shadow_mode.statistical_results,
                        'is_running': shadow_mode.is_running
                    }
                    logger.debug(f"Shadow mode metrics updated: {self.metrics['shadow_mode']['comparison_count']} comparisons")
                except Exception as e:
                    logger.warning(f"Failed to get shadow mode metrics: {e}")
                    self.metrics['shadow_mode'] = {'error': str(e)}
            
            # Add system metrics
            self.metrics['timestamp'] = datetime.now().isoformat()
            self.metrics['component_status'] = {
                name: status.value 
                for name, status in self.component_status.items()
            }
            
            # Add segment discovery metrics
            self.metrics['segment_discovery'] = self.get_segment_summary()
    
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
    
    def _create_ab_context_from_state(self, state: np.ndarray) -> Dict[str, Any]:
        """Create A/B testing context from environment state"""
        context = {
            'hour': int(datetime.now().hour),
            'day_of_week': int(datetime.now().weekday()),
            'device': 'mobile',  # Default, could be extracted from state
            'channel': 'organic',  # Default, could be extracted from state
            'segment': 'default_segment',  # Could be mapped from segment_index in state
            'seasonality_factor': 1.0,
            'competition_level': 0.5,
            'budget_remaining_ratio': 1.0
        }
        
        # Extract from state if available
        if isinstance(state, np.ndarray) and len(state) >= 10:
            context['competition_level'] = float(state[7]) if len(state) > 7 else 0.5
            context['budget_remaining_ratio'] = float(state[8]) if len(state) > 8 else 1.0
            
            # Map segment index to segment name if possible
            segment_index = int(state[3]) if len(state) > 3 else 0
            try:
                from dynamic_segment_integration import get_discovered_segments
                segments = get_discovered_segments()
                if segments and segment_index < len(segments):
                    context['segment'] = segments[segment_index]
            except ImportError:
                pass  # Use default segment
        
        return context
    
    def _apply_variant_policy_parameters(self, agent, variant_id: str, ab_testing, test_id: str):
        """Apply A/B test variant policy parameters to the agent"""
        try:
            # Get variant parameters
            test_config = ab_testing.active_tests.get(test_id)
            if not test_config:
                return
            
            variant = next((v for v in test_config['variants'] if v.variant_id == variant_id), None)
            if not variant or not variant.policy_parameters:
                return
            
            # Apply parameters to agent
            for param_name, param_value in variant.policy_parameters.items():
                if hasattr(agent, param_name):
                    setattr(agent, param_name, param_value)
                elif hasattr(agent, '_hyperparameters') and param_name in agent._hyperparameters:
                    agent._hyperparameters[param_name] = param_value
                    
            logger.debug(f"Applied variant {variant_id} parameters: {variant.policy_parameters}")
            
        except Exception as e:
            logger.warning(f"Failed to apply variant parameters: {e}")
    
    def _record_ab_test_observation(self, ab_testing, test_id: str, variant_id: str, 
                                    user_id: str, total_reward: float, steps: int, 
                                    final_state: np.ndarray, info: Dict[str, Any]):
        """Record observation for A/B testing analysis"""
        try:
            # Primary metric: total reward (ROAS or conversion)
            primary_metric_value = total_reward
            
            # Secondary metrics
            secondary_metrics = {
                'steps_taken': float(steps),
                'final_bid': float(final_state[9]) if len(final_state) > 9 else 0.0,
                'budget_utilization': 1.0 - (float(final_state[8]) if len(final_state) > 8 else 1.0)
            }
            
            # Add info metrics if available
            if info:
                secondary_metrics.update({
                    'cost': info.get('cost', 0.0),
                    'impressions': info.get('impressions', 0),
                    'clicks': info.get('clicks', 0),
                    'conversions': info.get('conversions', 0)
                })
            
            # Determine if this was a "conversion" (positive reward)
            converted = total_reward > 0
            
            # Create context for this observation
            context = self._create_ab_context_from_state(final_state)
            
            # Record the observation
            ab_testing.record_observation(
                test_id=test_id,
                variant_id=variant_id,
                user_id=user_id,
                primary_metric_value=primary_metric_value,
                secondary_metrics=secondary_metrics,
                converted=converted,
                context=context
            )
            
            logger.debug(f"Recorded A/B observation: {variant_id}, reward={total_reward:.3f}, converted={converted}")
            
        except Exception as e:
            logger.warning(f"Failed to record A/B test observation: {e}")
    
    def _analyze_ab_test_results(self, ab_testing, test_id: str):
        """Analyze A/B test results and log insights"""
        try:
            # Get test status first
            status = ab_testing.get_test_status(test_id)
            if status.get('error'):
                return
            
            # Check if we have enough data for analysis
            min_observations = min(v['n_observations'] for v in status['variants'])
            if min_observations < 50:  # Need minimum samples
                return
            
            # Run statistical analysis
            from statistical_ab_testing_framework import SignificanceTest
            results = ab_testing.analyze_test(test_id, SignificanceTest.BAYESIAN_HYPOTHESIS)
            
            # Log results
            logger.info(f"A/B Test Analysis - {test_id}:")
            logger.info(f"  Statistical Significance: {results.is_significant}")
            logger.info(f"  P-value: {results.p_value:.4f}")
            logger.info(f"  Bayesian Probability: {results.bayesian_probability:.4f}")
            logger.info(f"  Effect Size: {results.effect_size:.4f}")
            logger.info(f"  Winner: {results.winner_variant_id}")
            logger.info(f"  Recommendation: {results.recommended_action}")
            
            # Store results in metrics
            with self.metrics_lock:
                if 'ab_test_results' not in self.metrics:
                    self.metrics['ab_test_results'] = {}
                
                self.metrics['ab_test_results'][test_id] = {
                    'is_significant': results.is_significant,
                    'p_value': results.p_value,
                    'bayesian_probability': results.bayesian_probability,
                    'effect_size': results.effect_size,
                    'winner_variant_id': results.winner_variant_id,
                    'recommendation': results.recommended_action,
                    'last_analysis': datetime.now().isoformat()
                }
            
            # If test is conclusive, consider stopping
            if results.is_significant and results.minimum_sample_achieved:
                logger.info(f"A/B Test {test_id} reached statistical significance - consider concluding")
                
        except Exception as e:
            logger.warning(f"Failed to analyze A/B test results: {e}")
    
    def create_policy_comparison_test(self, policy_a_params: Dict[str, Any], 
                                      policy_b_params: Dict[str, Any], 
                                      test_name: str = "Policy Comparison") -> Optional[str]:
        """Create a new A/B test to compare two policy configurations"""
        ab_testing = self.components.get('ab_testing')
        if not ab_testing:
            logger.warning("A/B testing component not available")
            return None
        
        try:
            from statistical_ab_testing_framework import TestType, AllocationStrategy
            import uuid
            
            variants = [
                {
                    'variant_id': 'policy_a',
                    'name': 'Policy A (Control)',
                    'policy_parameters': policy_a_params,
                    'allocation_probability': 0.5
                },
                {
                    'variant_id': 'policy_b',
                    'name': 'Policy B (Treatment)', 
                    'policy_parameters': policy_b_params,
                    'allocation_probability': 0.5
                }
            ]
            
            test_id = f"policy_test_{uuid.uuid4().hex[:8]}"
            
            created_test_id = ab_testing.create_ab_test(
                test_id=test_id,
                test_name=test_name,
                variants=variants,
                test_type=TestType.BAYESIAN_BANDIT,
                allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING,
                duration_days=7  # One week test
            )
            
            logger.info(f"Created policy comparison A/B test: {created_test_id}")
            return created_test_id
            
        except Exception as e:
            logger.error(f"Failed to create policy comparison test: {e}")
            return None
    
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
        
        # Stop shadow mode first to get final results
        shadow_mode = self.components.get('shadow_mode')
        if shadow_mode:
            try:
                shadow_mode.stop_testing()
                final_results = shadow_mode.get_test_results()
                logger.info(f"Shadow mode final results: {final_results['comparison_count']} comparisons, "
                          f"DB: {final_results['database_path']}")
            except Exception as e:
                logger.error(f"Error stopping shadow mode: {e}")
        
        # Stop all other components
        for name, component in self.components.items():
            if name == 'shadow_mode':  # Already handled above
                continue
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
    
    def _update_model_with_episode_data(self, episode: int, episode_metrics: Dict[str, Any]):
        """Update model with episode data and discovered patterns"""
        try:
            model_updater = self.components.get('model_updater')
            segment_discovery = self.components.get('segment_discovery')
            
            if not model_updater:
                return
            
            # Extract episode experiences from metrics
            episode_experiences = episode_metrics.get('episode_experiences', [])
            total_reward = episode_metrics.get('total_reward', 0)
            
            # Convert episode experiences to GA4-like events format for model updater
            events_data = self._convert_experiences_to_events(episode_experiences, total_reward)
            
            # Update model with episode data
            if events_data:
                # Use asyncio.create_task to run async function in sync context
                import asyncio
                if asyncio.iscoroutinefunction(model_updater.update_gaelp_model):
                    # Run async update in the current thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(model_updater.update_gaelp_model(events_data))
                    finally:
                        loop.close()
                else:
                    # Direct call if not async
                    model_updater.update_gaelp_model(events_data)
                
                logger.info(f"🔄 Updated model with {len(events_data)} events from episode {episode}")
            
            # Update with newly discovered segments if available
            if segment_discovery and episode % 10 == 0:  # Every 10 episodes, check for new segments
                try:
                    new_segments = segment_discovery.discover_segments(force_rediscovery=False)
                    if new_segments:
                        self._integrate_discovered_segments(new_segments)
                        logger.info(f"🔍 Integrated {len(new_segments)} discovered segments into model")
                except Exception as e:
                    logger.error(f"Error updating segments: {e}")
            
            # Model versioning and rollback check
            self._check_model_performance_and_rollback(episode, total_reward)
            
        except Exception as e:
            logger.error(f"Error updating model with episode data: {e}")
    
    def _convert_experiences_to_events(self, experiences: List[Dict], total_reward: float) -> List[Dict[str, Any]]:
        """Convert RL episode experiences to GA4-like events for model updater"""
        events_data = []
        
        for exp in experiences:
            # Extract state information
            state = exp.get('state', [])
            action = exp.get('action', 0)
            reward = exp.get('reward', 0)
            info = exp.get('info', {})
            
            # Create event-like data structure
            event_data = {
                'event_name': 'bid_action' if reward > 0 else 'bid_attempt',
                'timestamp': datetime.now().isoformat(),
                'campaign_name': f"rl_training_campaign_{exp.get('episode', 0)}",
                'revenue': max(0, reward),  # Only positive rewards count as revenue
                'source': 'reinforcement_learning',
                'device_category': self._map_device_index_to_name(int(state[4]) if len(state) > 4 else 0),
                'channel': self._map_channel_index_to_name(int(state[5]) if len(state) > 5 else 0),
                'bid_amount': action if isinstance(action, (int, float)) else 0,
                'conversion_value': reward,
                'stage': int(state[0]) if len(state) > 0 else 0,
                'segment_index': int(state[3]) if len(state) > 3 else 0,
                'page_path': '/rl_training',
                'user_engagement': {
                    'engagement_time_msec': int((exp.get('step', 0) + 1) * 1000)  # Simulate engagement time
                }
            }
            
            # Add conversion event for positive rewards
            if reward > 0:
                conversion_event = {
                    'event_name': 'purchase',
                    'timestamp': datetime.now().isoformat(),
                    'campaign_name': event_data['campaign_name'],
                    'revenue': reward,
                    'source': 'reinforcement_learning',
                    'device_category': event_data['device_category'],
                    'value': reward,
                    'currency': 'USD',
                    'transaction_id': f"rl_txn_{exp.get('episode', 0)}_{exp.get('step', 0)}"
                }
                events_data.append(conversion_event)
            
            events_data.append(event_data)
        
        return events_data
    
    def _map_device_index_to_name(self, device_index: int) -> str:
        """Map device index to device name"""
        device_mapping = {0: 'mobile', 1: 'desktop', 2: 'tablet'}
        return device_mapping.get(device_index, 'mobile')
    
    def _map_channel_index_to_name(self, channel_index: int) -> str:
        """Map channel index to channel name"""
        channel_mapping = {0: 'organic', 1: 'paid_search', 2: 'social', 3: 'email', 4: 'direct'}
        return channel_mapping.get(channel_index, 'organic')
    
    def _integrate_discovered_segments(self, discovered_segments: Dict[str, Any]):
        """Integrate newly discovered segments into the model and agent"""
        try:
            agent = self.components.get('rl_agent')
            if not agent:
                return
            
            # Update agent's segment knowledge
            if hasattr(agent, 'update_segments'):
                rl_compatible_segments = {}
                for seg_id, segment in discovered_segments.items():
                    rl_compatible_segments[seg_id] = {
                        'name': segment.name if hasattr(segment, 'name') else seg_id,
                        'size': segment.size if hasattr(segment, 'size') else 0,
                        'conversion_rate': segment.conversion_rate if hasattr(segment, 'conversion_rate') else 0.0,
                        'characteristics': segment.characteristics if hasattr(segment, 'characteristics') else {},
                        'confidence': segment.confidence_score if hasattr(segment, 'confidence_score') else 0.0
                    }
                
                agent.update_segments(rl_compatible_segments)
                logger.info(f"✅ Agent updated with {len(rl_compatible_segments)} discovered segments")
            
            # Update environment's segment knowledge
            env = self.components.get('environment')
            if env and hasattr(env, 'update_segments'):
                env.update_segments(discovered_segments)
                logger.info(f"✅ Environment updated with discovered segments")
            
            # Store segments for future use
            with self.lock:
                self.metrics['discovered_segments'] = {
                    'count': len(discovered_segments),
                    'timestamp': datetime.now().isoformat(),
                    'segment_names': [seg.name if hasattr(seg, 'name') else seg_id 
                                     for seg_id, seg in discovered_segments.items()]
                }
            
            # Update model_updater with new segment data
            model_updater = self.components.get('model_updater')
            if model_updater and hasattr(model_updater, 'update'):
                model_updater.update(segment_data=discovered_segments)
                logger.info(f"✅ Model updater integrated with {len(discovered_segments)} discovered segments")
            
        except Exception as e:
            logger.error(f"Error integrating discovered segments: {e}")
    
    def _check_model_performance_and_rollback(self, episode: int, reward: float):
        """Check model performance and rollback if needed"""
        try:
            # Store performance metrics
            with self.metrics_lock:
                if 'episode_rewards' not in self.metrics:
                    self.metrics['episode_rewards'] = []
                
                self.metrics['episode_rewards'].append(reward)
                
                # Keep only last 100 episodes for performance tracking
                if len(self.metrics['episode_rewards']) > 100:
                    self.metrics['episode_rewards'] = self.metrics['episode_rewards'][-100:]
                
                # Calculate performance metrics
                recent_rewards = self.metrics['episode_rewards'][-20:] if len(self.metrics['episode_rewards']) >= 20 else self.metrics['episode_rewards']
                avg_recent_reward = np.mean(recent_rewards)
                
                # Check for significant performance degradation
                if len(self.metrics['episode_rewards']) >= 50:
                    older_rewards = self.metrics['episode_rewards'][-50:-20]
                    avg_older_reward = np.mean(older_rewards)
                    
                    performance_degradation = (avg_older_reward - avg_recent_reward) / max(abs(avg_older_reward), 1e-6)
                    
                    # If performance degraded by more than 30%, trigger rollback consideration
                    if performance_degradation > 0.3:
                        logger.warning(f"⚠️ Performance degradation detected: {performance_degradation:.2%}")
                        self._consider_model_rollback(episode, avg_recent_reward, avg_older_reward)
                    
                    # Update performance tracking
                    self.metrics['model_performance'] = {
                        'avg_recent_reward': avg_recent_reward,
                        'avg_older_reward': avg_older_reward if len(older_rewards) > 0 else avg_recent_reward,
                        'performance_degradation': performance_degradation,
                        'episode': episode,
                        'timestamp': datetime.now().isoformat()
                    }
        
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
    
    def _consider_model_rollback(self, episode: int, recent_performance: float, older_performance: float):
        """Consider rolling back to a previous model version"""
        try:
            checkpoint_manager = self.components.get('checkpoint_manager')
            if not checkpoint_manager:
                logger.warning("No checkpoint manager available for rollback")
                return
            
            logger.warning(f"🔄 Considering model rollback at episode {episode}")
            logger.warning(f"Recent performance: {recent_performance:.3f}, Older performance: {older_performance:.3f}")
            
            # Get available checkpoints
            if hasattr(checkpoint_manager, 'list_checkpoints'):
                checkpoints = checkpoint_manager.list_checkpoints()
                
                if checkpoints:
                    # Find a checkpoint from when performance was better
                    best_checkpoint = None
                    for checkpoint in checkpoints[-10:]:
                        checkpoint_metrics = checkpoint.get('metrics', {})
                        checkpoint_performance = checkpoint_metrics.get('model_performance', {}).get('avg_recent_reward', 0)
                        
                        if checkpoint_performance > recent_performance * 1.1:  # 10% better
                            best_checkpoint = checkpoint
                            break
                    
                    if best_checkpoint:
                        logger.info(f"🔄 Rolling back to checkpoint: {best_checkpoint.get('checkpoint_id', 'unknown')}")
                        
                        # Load the checkpoint
                        agent = self.components.get('rl_agent')
                        if agent and hasattr(checkpoint_manager, 'load_checkpoint'):
                            checkpoint_manager.load_checkpoint(best_checkpoint['checkpoint_id'], agent)
                            
                            # Reset metrics to prevent immediate re-rollback
                            with self.metrics_lock:
                                self.metrics['episode_rewards'] = self.metrics['episode_rewards'][-10:]  # Keep only recent few
                            
                            logger.info("✅ Model rollback completed")
                        else:
                            logger.error("Cannot perform rollback - agent or load method not available")
                    else:
                        logger.warning("No suitable checkpoint found for rollback")
                else:
                    logger.warning("No checkpoints available for rollback")
            else:
                logger.warning("Checkpoint manager does not support listing checkpoints")
        
        except Exception as e:
            logger.error(f"Error during model rollback consideration: {e}")
    
    def _enrich_state_with_segments(self, enriched_state):
        """Enrich state with discovered segment information"""
        if not self.discovered_segments:
            return
        
        # Get segment based on segment_index from state
        segment_keys = list(self.discovered_segments.keys())
        if segment_keys and enriched_state.segment_index < len(segment_keys):
            segment_id = segment_keys[enriched_state.segment_index]
            segment = self.discovered_segments[segment_id]
            
            # Populate segment data from discovered segments
            enriched_state.segment_cvr = segment.conversion_rate
            enriched_state.segment_engagement = segment.engagement_metrics.get('high_engagement_rate', 0.0)
            enriched_state.segment_avg_ltv = segment.behavioral_profile.get('avg_session_duration', 0.0) / 100.0  # Normalize
            
            logger.debug(f"State enriched with segment {segment.name}: CVR={segment.conversion_rate:.3f}")
    
    def _get_creative_content_for_state(self, env, enriched_state) -> Optional['CreativeContent']:
        """Get actual creative content for current state, not just ID"""
        try:
            from creative_content_analyzer import CreativeContent
            
            # Get creative selector from environment
            creative_selector = getattr(env, 'creative_selector', None)
            if not creative_selector:
                logger.warning("No creative_selector found in environment")
                return None
            
            # Use creative_index from state to get actual creative content
            creative_index = enriched_state.creative_index
            
            # Try to get creative content based on segment and state
            segment_keys = list(self.discovered_segments.keys()) if self.discovered_segments else ['default']
            segment_name = segment_keys[enriched_state.segment_index] if enriched_state.segment_index < len(segment_keys) else 'default'
            
            # Get creative content from creative selector
            # This should return actual creative content, not hardcoded
            creative_data = None
            
            # Try multiple approaches to get actual creative content
            if hasattr(creative_selector, 'get_creative_content'):
                creative_data = creative_selector.get_creative_content(creative_index)
            elif hasattr(creative_selector, 'select_creative'):
                # Use segment-based selection
                from creative_selector import UserState, CreativeType, JourneyStage, UserSegment
                
                # Discover available JourneyStage values dynamically
                available_stages = [stage for stage in JourneyStage]
                # Map stage index to available stages - distribute evenly
                stage_count = len(available_stages)
                if enriched_state.stage < stage_count:
                    journey_stage = available_stages[enriched_state.stage]
                else:
                    # For stages beyond available, use the last stage (typically retention)
                    journey_stage = available_stages[-1]
                
                # Discover available UserSegment values dynamically
                available_segments = [seg for seg in UserSegment]
                
                # Hash the segment name to deterministically select from available segments
                # This ensures consistent mapping without hardcoding
                import hashlib
                segment_hash = int(hashlib.md5(segment_name.encode()).hexdigest(), 16)
                segment_index = segment_hash % len(available_segments)
                user_segment = available_segments[segment_index]
                
                # Create user state with proper fields
                user_state = UserState(
                    user_id=f"training_user_{segment_name}",
                    segment=user_segment,
                    journey_stage=journey_stage,
                    device_type=f"device_{enriched_state.device_index}",
                    time_of_day='afternoon',
                    previous_interactions=[],
                    conversion_probability=enriched_state.segment_cvr,
                    urgency_score=0.5,
                    price_sensitivity=0.5,
                    technical_level=0.5,
                    session_count=enriched_state.touchpoints_seen,
                    last_seen=time.time()
                )
                creative_result = creative_selector.select_creative(user_state)
                if hasattr(creative_result, 'creative_content'):
                    creative_data = creative_result.creative_content
            
            # If we can't get actual content, discover it from patterns file
            if not creative_data:
                creative_data = self._discover_creative_content_from_patterns(creative_index, segment_name)
            
            # If still no content, log error and return None
            if not creative_data:
                logger.error(f"No creative content available for index {creative_index}")
                return None
            
            # Create CreativeContent object with actual content
            creative_content = CreativeContent(
                creative_id=creative_data.get('creative_id', f'discovered_{creative_index}'),
                headline=creative_data.get('headline', ''),
                description=creative_data.get('description', ''),
                cta=creative_data.get('cta', ''),
                image_url=creative_data.get('image_url', ''),
                impressions=creative_data.get('impressions', 0),
                clicks=creative_data.get('clicks', 0),
                conversions=creative_data.get('conversions', 0)
            )
            
            return creative_content
            
        except Exception as e:
            logger.error(f"Failed to get creative content: {e}")
            return None
    
    def _discover_creative_content_from_patterns(self, creative_index: int, segment_name: str) -> Optional[Dict[str, Any]]:
        """Discover creative content from patterns file or generate dynamically"""
        try:
            patterns_file = "discovered_patterns.json"
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    patterns = json.load(f)
                
                # Look for creative content patterns
                creative_patterns = patterns.get('creative_content', {})
                if creative_patterns:
                    # Find content for this segment or use default
                    segment_content = creative_patterns.get(segment_name) or creative_patterns.get('default')
                    if segment_content and isinstance(segment_content, list):
                        # Use index to select content
                        content_item = segment_content[creative_index % len(segment_content)]
                        if isinstance(content_item, dict):
                            return content_item
                
                # If no specific creative patterns, look for general ad content data
                ad_data = patterns.get('ad_data', [])
                if ad_data and isinstance(ad_data, list):
                    ad_item = ad_data[creative_index % len(ad_data)]
                    if isinstance(ad_item, dict):
                        # Map ad_data fields to creative content format
                        return {
                            'creative_id': ad_item.get('id', f'pattern_{creative_index}'),
                            'headline': ad_item.get('headline', ad_item.get('title', '')),
                            'description': ad_item.get('description', ad_item.get('body', '')),
                            'cta': ad_item.get('cta', ad_item.get('call_to_action', 'Learn More')),
                            'image_url': ad_item.get('image_url', ad_item.get('image', '')),
                            'impressions': ad_item.get('impressions', 0),
                            'clicks': ad_item.get('clicks', 0),
                            'conversions': ad_item.get('conversions', 0)
                        }
                
                # Generate creative content dynamically based on segment data
                return self._generate_dynamic_creative_content(creative_index, segment_name, patterns)
            
            # If no patterns file, still generate content dynamically
            return self._generate_dynamic_creative_content(creative_index, segment_name, {})
            
        except Exception as e:
            logger.error(f"Failed to discover creative content from patterns: {e}")
            return None
    
    def _generate_dynamic_creative_content(self, creative_index: int, segment_name: str, patterns: Dict) -> Dict[str, Any]:
        """Generate creative content dynamically based on discovered segment characteristics"""
        try:
            # Get segment data if available
            segment_data = patterns.get('segments', {}).get(segment_name, {})
            behavioral_metrics = segment_data.get('behavioral_metrics', {})
            characteristics = segment_data.get('discovered_characteristics', {})
            
            # Extract key metrics for creative generation
            cvr = behavioral_metrics.get('conversion_rate', 0.02)
            bounce_rate = behavioral_metrics.get('bounce_rate', 0.4)
            return_rate = behavioral_metrics.get('return_rate', 0.3)
            primary_channel = characteristics.get('primary_channel', 'search')
            device_pref = characteristics.get('device_preference', 'mobile')
            
            # Generate creative variations based on segment characteristics
            # Use creative_index to deterministically generate different variations
            import hashlib
            
            # Create deterministic seed from segment and index
            seed_str = f"{segment_name}_{creative_index}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
            
            # Generate headlines based on segment behavior
            headline_templates = []
            if cvr > 0.04:  # High converting segment
                headline_templates = [
                    "Transform Your Family's Screen Time Today",
                    "Join Thousands Finding Digital Balance",
                    "The #1 Solution Parents Trust"
                ]
            elif return_rate > 0.5:  # High engagement segment
                headline_templates = [
                    "Welcome Back - New Features Inside",
                    "Continue Your Journey to Balance",
                    "Your Progress Matters - Keep Going"
                ]
            elif bounce_rate > 0.5:  # High bounce segment - need attention grabbing
                headline_templates = [
                    "5-Minute Setup, Lifetime Peace of Mind",
                    "Stop Screen Time Battles Forever",
                    "Instant Relief for Digital Overwhelm"
                ]
            else:  # General segment
                headline_templates = [
                    "Discover Healthy Screen Time Balance",
                    "Take Control of Your Digital Life",
                    "Find Your Family's Digital Harmony"
                ]
            
            headline = headline_templates[seed % len(headline_templates)]
            
            # Generate descriptions based on device and channel
            if device_pref == 'mobile' and primary_channel == 'social':
                description = "Quick setup on your phone. Real-time insights. Family-wide protection."
            elif device_pref == 'desktop' and primary_channel == 'search':
                description = "Comprehensive parental controls with detailed analytics and customizable rules for every family member."
            elif primary_channel == 'display':
                description = "See how families like yours found balance. Start your free trial today."
            else:
                description = "Evidence-based tools to help your family thrive in the digital age."
            
            # Generate CTAs based on urgency and segment
            cta_options = []
            if cvr > 0.03:
                cta_options = ["Start Free Trial", "Get Started Now", "Begin Today"]
            elif bounce_rate < 0.3:
                cta_options = ["Learn More", "See How It Works", "Explore Features"]
            else:
                cta_options = ["Try It Free", "Get Instant Access", "Start Now"]
            
            cta = cta_options[(seed // 7) % len(cta_options)]
            
            # Generate other creative properties
            return {
                'creative_id': f'dynamic_{segment_name}_{creative_index}',
                'headline': headline,
                'description': description,
                'cta': cta,
                'image_url': f'/assets/creative_{(seed % 10) + 1}.jpg',
                'segment': segment_name,
                'cvr_estimate': cvr,
                'device_optimized': device_pref,
                'channel_optimized': primary_channel,
                'impressions': 0,
                'clicks': 0,
                'conversions': 0
            }
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic creative content: {e}")
            # Return minimal viable creative content
            return {
                'creative_id': f'basic_{creative_index}',
                'headline': 'Find Digital Balance',
                'description': 'Tools for healthy screen time',
                'cta': 'Learn More',
                'image_url': '/assets/default.jpg',
                'impressions': 0,
                'clicks': 0,
                'conversions': 0
            }
    
    def _enrich_state_with_creative_features(self, enriched_state, content_features) -> 'DynamicEnrichedState':
        """Enrich state with actual creative content features"""
        try:
            from fortified_rl_agent_no_hardcoding import DynamicEnrichedState
            
            # Update existing creative performance metrics with actual content analysis
            enriched_state.creative_ctr = content_features.predicted_ctr
            enriched_state.creative_cvr = content_features.predicted_cvr
            enriched_state.creative_fatigue = 1.0 - content_features.fatigue_resistance
            
            # Calculate diversity score based on content features
            diversity_elements = [
                content_features.headline_emotion != "neutral",
                content_features.uses_numbers,
                content_features.uses_social_proof,
                content_features.uses_authority,
                content_features.visual_style != "unknown",
                content_features.primary_color != "unknown",
                content_features.uses_urgency,
                content_features.message_frame != "benefit"  # Non-default frame
            ]
            enriched_state.creative_diversity_score = sum(diversity_elements) / len(diversity_elements)
            
            # Add new creative content features to state vector
            # These extend the existing state representation
            if not hasattr(enriched_state, 'content_sentiment'):
                enriched_state.content_sentiment = content_features.headline_sentiment
                enriched_state.content_urgency = content_features.headline_urgency
                enriched_state.content_cta_strength = content_features.cta_strength
                enriched_state.content_uses_numbers = float(content_features.uses_numbers)
                enriched_state.content_uses_social_proof = float(content_features.uses_social_proof)
                enriched_state.content_uses_authority = float(content_features.uses_authority)
                enriched_state.content_uses_urgency = float(content_features.uses_urgency)
                
                # Encode message frame as numeric
                frame_encoding = {
                    'benefit': 0.0, 'fear': 0.2, 'urgency': 0.4, 
                    'authority': 0.6, 'social_proof': 0.8
                }
                enriched_state.content_message_frame = frame_encoding.get(content_features.message_frame, 0.0)
                
                # Encode visual style
                style_encoding = {
                    'lifestyle': 0.0, 'clinical': 0.25, 'emotional': 0.5, 
                    'comparison': 0.75, 'unknown': 1.0
                }
                enriched_state.content_visual_style = style_encoding.get(content_features.visual_style, 1.0)
                
                logger.debug(f"Added content features to state: sentiment={content_features.headline_sentiment:.2f}, "
                           f"urgency={content_features.uses_urgency}, frame={content_features.message_frame}")
            
            return enriched_state
            
        except Exception as e:
            logger.error(f"Failed to enrich state with creative features: {e}")
            return enriched_state
    
    def get_discovered_segments(self) -> Dict:
        """Get currently discovered segments"""
        return self.discovered_segments
    
    def get_segment_summary(self) -> Dict[str, Any]:
        """Get summary of discovered segments for monitoring"""
        if not self.discovered_segments:
            return {'total_segments': 0, 'status': 'no_segments'}
        
        total_users = sum(segment.size for segment in self.discovered_segments.values())
        avg_cvr = np.mean([segment.conversion_rate for segment in self.discovered_segments.values()])
        
        segment_names = [segment.name for segment in self.discovered_segments.values()]
        
        return {
            'total_segments': len(self.discovered_segments),
            'total_users': total_users,
            'avg_conversion_rate': avg_cvr,
            'segment_names': segment_names[:5],  # Top 5 for summary
            'last_updated': self.segments_last_updated.isoformat() if self.segments_last_updated else None,
            'episodes_since_update': self.episodes_since_segment_update,
            'status': 'active'
        }
    
    def _run_continuous_learning(self):
        """Run continuous learning cycle in background thread"""
        try:
            import asyncio
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            online_learner = self.components.get('online_learner')
            if online_learner:
                loop.run_until_complete(online_learner.continuous_learning_cycle())
        except Exception as e:
            logger.error(f"Continuous learning thread error: {e}")
        finally:
            if 'loop' in locals():
                loop.close()
    
    def _update_online_learner_from_episode(self, episode_results: Dict[str, Any]):
        """Update online learner with episode outcomes"""
        try:
            online_learner = self.components.get('online_learner')
            if not online_learner:
                return
            
            # Create synthetic production outcome from episode results
            action = {
                'strategy': 'rl_training',
                'episode': episode_results['episode'],
                'epsilon': episode_results.get('epsilon', 0.0),
                'state': episode_results.get('final_state', {})
            }
            
            outcome = {
                'conversion': episode_results['total_reward'] > 0,
                'reward': episode_results['total_reward'],
                'revenue': max(0, episode_results['total_reward'] * 10),  # Estimate revenue
                'spend': episode_results['steps'] * 0.5,  # Estimate spend per step
                'channel': 'rl_training',
                'campaign_id': f"episode_{episode_results['episode']}",
                'done': True,
                'next_state': episode_results.get('final_state', {}),
                'attribution_data': {
                    'episode': episode_results['episode'],
                    'steps': episode_results['steps'],
                    'training_reward': episode_results['total_reward']
                }
            }
            
            # Record the outcome for learning
            online_learner.record_production_outcome(action, outcome)
            
            logger.debug(f"Updated online learner with episode {episode_results['episode']} results")
            
        except Exception as e:
            logger.error(f"Failed to update online learner from episode: {e}")
    
    def record_production_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any], user_id: str = None):
        """Record real production action outcome for learning"""
        try:
            online_learner = self.components.get('online_learner')
            if online_learner:
                online_learner.record_production_outcome(action, outcome, user_id)
                logger.debug(f"Recorded production outcome for action {action.get('id', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to record production outcome: {e}")
    
    async def get_production_action(self, state: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
        """Get action from online learner for production traffic"""
        try:
            online_learner = self.components.get('online_learner')
            if online_learner and hasattr(online_learner, 'select_production_action'):
                return await online_learner.select_production_action(state, user_id)
            else:
                # Use RL agent if needed
                rl_agent = self.components.get('rl_agent')
                if rl_agent:
                    # Convert state to enriched state format
                    from fortified_rl_agent_no_hardcoding import DynamicEnrichedState
                    enriched_state = DynamicEnrichedState()
                    # Populate from state dict
                    for key, value in state.items():
                        if hasattr(enriched_state, key):
                            setattr(enriched_state, key, value)
                    
                    action = rl_agent.select_action(enriched_state)
                    return {
                        'bid_amount': action.get('bid_amount', 1.0),
                        'strategy': 'rl_agent',
                        'state': state,
                        'id': f"action_{int(time.time())}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get production action: {e}")
            
        # Safe default action
        return {
            'bid_amount': 1.0,
            'strategy': 'safe_default',
            'state': state,
            'id': f"safe_action_{int(time.time())}"
        }
    
    def _track_training_touchpoint(self, attribution_engine, enriched_state, action, user_id: str, step: int) -> Optional[str]:
        """Track a training step as a marketing touchpoint for attribution."""
        try:
            # Map RL state to marketing campaign data
            channel_names = ['search', 'social', 'display', 'email', 'direct']
            channel = channel_names[enriched_state.channel_index % len(channel_names)]
            
            # Extract numeric action value for comparison
            action_value = action
            if hasattr(action, 'bid_amount'):
                action_value = action.bid_amount
            elif hasattr(action, '__getitem__'):
                action_value = action.get('bid_amount', action)
            elif not isinstance(action_value, (int, float)):
                action_value = 0.5  # Default middle value
            
            # Determine touchpoint type based on action and state
            if enriched_state.stage == 0:  # First interaction
                touchpoint_type = 'impression'
            elif action_value > 0.7:  # High bid = strong engagement
                touchpoint_type = 'click'
            else:
                touchpoint_type = 'visit'
            
            # Create campaign data from RL state
            campaign_data = {
                'channel': channel,
                'source': f'rl_training_{channel}',
                'medium': 'rl_simulation',
                'campaign': f'gaelp_training_episode_{enriched_state.segment_index}',
                'ad_group': f'segment_{enriched_state.segment_index}',
                'creative_id': f'creative_{enriched_state.creative_index}',
                'keyword': f'auto_keyword_{enriched_state.segment_index}_{enriched_state.device_index}'
            }
            
            # Create user data from RL state
            user_data = {
                'user_id': user_id,
                'device_id': f'training_device_{enriched_state.device_index}',
                'ip_hash': f'training_ip_{user_id[:8]}',
                'platform': 'iOS' if enriched_state.device_index % 2 == 1 else 'Android',
                'is_mobile': enriched_state.device_index % 3 != 0,
                'is_ios': enriched_state.device_index % 2 == 1,
                'timezone': 'America/New_York'
            }
            
            # Track based on type
            if touchpoint_type == 'impression':
                return attribution_engine.track_impression(
                    campaign_data=campaign_data,
                    user_data=user_data,
                    timestamp=datetime.now()
                )
            elif touchpoint_type == 'click':
                click_data = {
                    'click_id': f'rl_click_{step}_{user_id[:8]}',
                    'landing_page': f'/landing/{channel}',
                    'actions_taken': ['view_page'] if action_value > 0.5 else []
                }
                return attribution_engine.track_click(
                    campaign_data=campaign_data,
                    user_data=user_data,
                    click_data=click_data,
                    timestamp=datetime.now()
                )
            else:  # visit
                visit_data = {
                    'page_url': f'/{channel}/landing',
                    'time_on_page': int(action_value * 300),  # Convert action to seconds
                    'pages_viewed': 1 + int(action_value * 3),
                    'actions_taken': ['page_view']
                }
                return attribution_engine.track_site_visit(
                    visit_data=visit_data,
                    user_data=user_data,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Failed to track training touchpoint: {e}")
            return None
    
    def _is_conversion_event(self, reward: float, info: Dict) -> bool:
        """Determine if this step represents a conversion event."""
        # Define conversion thresholds
        CONVERSION_REWARD_THRESHOLD = 1.0  # Minimum reward to be considered conversion
        
        # Check reward threshold
        if reward >= CONVERSION_REWARD_THRESHOLD:
            return True
        
        # Check info signals
        if info.get('conversion', False):
            return True
        
        if info.get('purchase', False):
            return True
        
        if info.get('subscription', False):
            return True
        
        # Check for conversion indicators in action outcomes
        if info.get('action_result') == 'conversion':
            return True
        
        return False
    
    def _track_conversion(self, attribution_engine, reward: float, user_id: str, info: Dict) -> Optional[str]:
        """Track a conversion event in the attribution system."""
        try:
            conversion_data = {
                'value': abs(reward),  # Use absolute reward as conversion value
                'type': info.get('conversion_type', 'rl_training_conversion'),
                'product_category': info.get('product_category', 'behavioral_health_monitoring'),
                'page_url': '/conversion/complete'
            }
            
            user_data = {
                'user_id': user_id,
                'device_id': f'training_device_{user_id[-3:]}',
                'ip_hash': f'training_ip_{user_id[:8]}',
                'platform': 'web',
                'is_mobile': False,
                'is_ios': False
            }
            
            return attribution_engine.track_conversion(
                conversion_data=conversion_data,
                user_data=user_data,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to track conversion: {e}")
            return None
    
    def _calculate_episode_attribution(self, attribution_engine, user_id: str, 
                                     touchpoints: List[Dict], conversion_reward: float) -> Dict[str, float]:
        """Calculate multi-touch attribution for all touchpoints in the episode."""
        try:
            # Get the user journey from attribution engine
            journey_data = attribution_engine.get_user_journey(user_id, days_back=1)
            
            if not journey_data['attribution_results']:
                logger.warning(f"No attribution results found for user {user_id}")
                return {}
            
            # Extract attribution weights by touchpoint
            attributed_rewards = {}
            
            # Use time_decay attribution model results (preferred for RL training)
            time_decay_results = [
                result for result in journey_data['attribution_results']
                if result['result_attribution_model'] == 'time_decay'
            ]
            
            if not time_decay_results:
                # Fall back to any available attribution model
                time_decay_results = journey_data['attribution_results'][:len(touchpoints)]
            
            for result in time_decay_results:
                touchpoint_id = result['touchpoint_id']
                attribution_weight = result['attribution_weight']
                attributed_value = attribution_weight * abs(conversion_reward)
                attributed_rewards[touchpoint_id] = attributed_value
            
            # Ensure all episode touchpoints get some attribution if results are missing
            if len(attributed_rewards) < len(touchpoints):
                # Distribute remaining value equally among unattributed touchpoints
                remaining_touchpoints = [
                    tp for tp in touchpoints 
                    if tp['id'] not in attributed_rewards
                ]
                
                if remaining_touchpoints:
                    remaining_value = abs(conversion_reward) * 0.1  # 10% for unattributed
                    value_per_touchpoint = remaining_value / len(remaining_touchpoints)
                    
                    for tp in remaining_touchpoints:
                        attributed_rewards[tp['id']] = value_per_touchpoint
            
            return attributed_rewards
            
        except Exception as e:
            logger.error(f"Failed to calculate episode attribution: {e}")
            # Fall back to equal distribution if attribution calculation fails
            if touchpoints:
                equal_value = abs(conversion_reward) / len(touchpoints)
                return {tp['id']: equal_value for tp in touchpoints}
            return {}
    
    def create_production_ab_test(self, test_name: str, variants: Dict[str, Dict[str, Any]]) -> str:
        """Create A/B test through online learner"""
        try:
            online_learner = self.components.get('online_learner')
            if online_learner and hasattr(online_learner, 'create_ab_test'):
                experiment_id = online_learner.create_ab_test(test_name, variants)
                logger.info(f"Created A/B test '{test_name}' with ID: {experiment_id}")
                return experiment_id
            else:
                logger.warning("Online learner not available for A/B test creation")
                return ""
        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            return ""
    
    def get_online_learning_status(self) -> Dict[str, Any]:
        """Get detailed status of online learning system"""
        try:
            online_learner = self.components.get('online_learner')
            if online_learner and hasattr(online_learner, 'get_system_status'):
                return online_learner.get_system_status()
            else:
                return {
                    'status': 'not_available',
                    'message': 'Online learner not initialized'
                }
        except Exception as e:
            logger.error(f"Failed to get online learning status: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

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
                # Safely serialize status that may contain numpy types, datetimes, enums, etc.
                def _json_default(o):
                    try:
                        import numpy as _np
                        from decimal import Decimal as _Decimal
                    except Exception:
                        _np = None
                        _Decimal = None
                    # numpy scalars
                    if _np is not None and isinstance(o, (_np.integer, _np.floating)):
                        return o.item()
                    # numpy arrays
                    if _np is not None and isinstance(o, _np.ndarray):
                        return o.tolist()
                    # decimals
                    if _Decimal is not None and isinstance(o, _Decimal):
                        return float(o)
                    # datetimes
                    if isinstance(o, datetime):
                        return o.isoformat()
                    # enums
                    if isinstance(o, Enum):
                        return o.value
                    # sets
                    if isinstance(o, set):
                        return list(o)
                    # objects with to_dict
                    if hasattr(o, 'to_dict') and callable(getattr(o, 'to_dict')):
                        return o.to_dict()
                    # Fallback to string
                    return str(o)
                try:
                    logger.info(f"📊 Status: {json.dumps(status, default=_json_default, indent=2)}")
                except Exception as e:
                    logger.debug(f"Status JSON serialization failed: {e}")
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
            orchestrator.stop()
    else:
        logger.error("Failed to start orchestrator")
        sys.exit(1)

if __name__ == "__main__":
    main()
