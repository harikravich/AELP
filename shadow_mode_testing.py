#!/usr/bin/env python3
"""
PRODUCTION-GRADE SHADOW MODE TESTING SYSTEM
Runs new models alongside production models, compares decisions, tracks performance without spending money
"""

import logging
import asyncio
import json
import time
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import concurrent.futures
from pathlib import Path
import uuid
import copy

# Import GAELP components
try:
    from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent, DynamicEnrichedState
    from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
except ImportError:
    # Fallback for testing
    ProductionFortifiedRLAgent = None
    DynamicEnrichedState = None
    ProductionFortifiedEnvironment = None

from discovery_engine import GA4RealTimeDataPipeline as DiscoveryEngine
try:
    from creative_selector import CreativeSelector
    from attribution_models import AttributionEngine
    from budget_pacer import BudgetPacer
    from identity_resolver import IdentityResolver
except ImportError:
    # Mock for testing
    CreativeSelector = None
    AttributionEngine = None
    BudgetPacer = None
    IdentityResolver = None

from gaelp_parameter_manager import ParameterManager
try:
    from emergency_controls import get_emergency_controller, emergency_stop_decorator
except ImportError:
    # Mock for testing
    def get_emergency_controller():
        class MockController:
            def is_system_healthy(self): return True
        return MockController()
    
    def emergency_stop_decorator(name):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Model types in shadow testing"""
    PRODUCTION = "production"
    SHADOW = "shadow"
    CONTROL = "control"

@dataclass
class ShadowDecision:
    """Individual decision made by a model"""
    model_id: str
    model_type: ModelType
    user_id: str
    timestamp: datetime
    
    # Decision details
    bid_amount: float
    creative_id: int
    channel: str
    confidence_score: float
    
    # Model state
    model_epsilon: float
    exploration_action: bool
    q_values: List[float]
    
    # Context
    user_state: Dict[str, Any]
    environment_context: Dict[str, Any]
    
    # Would-be outcome (not real spending)
    would_win_auction: bool
    would_click: bool
    would_convert: bool
    predicted_position: float
    predicted_cost: float
    predicted_revenue: float

@dataclass
class ShadowComparison:
    """Comparison between model decisions"""
    session_id: str
    timestamp: datetime
    user_id: str
    
    # Decisions from each model
    production_decision: ShadowDecision
    shadow_decision: ShadowDecision
    control_decision: Optional[ShadowDecision] = None
    
    # Divergence metrics
    bid_divergence: float = 0.0
    creative_divergence: bool = False
    channel_divergence: bool = False
    
    # Performance predictions
    production_predicted_value: float = 0.0
    shadow_predicted_value: float = 0.0
    control_predicted_value: float = 0.0
    
    # Flags
    significant_divergence: bool = False
    shadow_outperforms: bool = False

@dataclass
class ShadowMetrics:
    """Aggregated metrics for shadow testing"""
    model_id: str
    model_type: ModelType
    
    # Performance metrics
    total_decisions: int = 0
    avg_bid: float = 0.0
    avg_confidence: float = 0.0
    predicted_win_rate: float = 0.0
    predicted_ctr: float = 0.0
    predicted_cvr: float = 0.0
    predicted_roas: float = 0.0
    
    # Risk metrics
    max_bid_observed: float = 0.0
    bid_volatility: float = 0.0
    creative_diversity: float = 0.0
    channel_distribution: Dict[str, float] = None
    
    # Learning metrics
    epsilon_progression: List[float] = None
    q_value_stability: float = 0.0
    convergence_score: float = 0.0
    
    def __post_init__(self):
        if self.channel_distribution is None:
            self.channel_distribution = {}
        if self.epsilon_progression is None:
            self.epsilon_progression = []

class ShadowModelRunner:
    """Manages execution of a single model in shadow mode"""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 model_type: ModelType,
                 base_components: Dict[str, Any]):
        self.model_id = model_config['model_id']
        self.model_type = model_type
        self.config = model_config
        self.components = base_components
        
        # Initialize model
        self.agent = self._initialize_model()
        self.environment = self._create_shadow_environment()
        
        # Performance tracking
        self.decisions = deque(maxlen=10000)
        self.metrics = ShadowMetrics(
            model_id=self.model_id,
            model_type=model_type
        )
        
        # Risk monitoring
        self.risk_thresholds = {
            'max_bid': model_config.get('max_bid_limit', 20.0),
            'max_daily_spend': model_config.get('max_daily_spend', 5000.0),
            'min_win_rate': model_config.get('min_win_rate', 0.1),
            'max_bid_volatility': model_config.get('max_bid_volatility', 2.0)
        }
        
        logger.info(f"Shadow model {self.model_id} ({model_type.value}) initialized")
    
    def _initialize_model(self) -> ProductionFortifiedRLAgent:
        """Initialize the RL agent for this shadow model"""
        # Create agent with specific configuration
        agent = ProductionFortifiedRLAgent(
            discovery_engine=self.components['discovery'],
            creative_selector=self.components['creative_selector'],
            attribution_engine=self.components['attribution'],
            budget_pacer=self.components['budget_pacer'],
            identity_resolver=self.components['identity_resolver'],
            parameter_manager=self.components['parameter_manager'],
            learning_rate=self.config.get('learning_rate', 1e-4),
            epsilon=self.config.get('epsilon', 0.1)
        )
        
        # Load model weights if specified
        if 'model_path' in self.config and Path(self.config['model_path']).exists():
            agent.load_model(self.config['model_path'])
            logger.info(f"Loaded model weights from {self.config['model_path']}")
        
        return agent
    
    def _create_shadow_environment(self) -> ProductionFortifiedEnvironment:
        """Create environment for shadow testing (no real spending)"""
        return ProductionFortifiedEnvironment(
            parameter_manager=self.components['parameter_manager'],
            use_real_ga4_data=False,  # Use simulated data
            is_parallel=False,
            shadow_mode=True  # Critical: no real money spent
        )
    
    async def make_decision(self, 
                           user_id: str,
                           user_state: DynamicEnrichedState,
                           context: Dict[str, Any]) -> ShadowDecision:
        """Make a decision in shadow mode"""
        start_time = time.time()
        
        try:
            # Get action from agent
            action = self.agent.select_action(user_state, explore=True)
            
            # Calculate confidence based on Q-values
            state_vector = torch.FloatTensor(user_state.to_vector()).unsqueeze(0)
            with torch.no_grad():
                q_values = self.agent.q_network_bid(state_vector).cpu().numpy().flatten()
                confidence = float(np.max(q_values) - np.mean(q_values))
            
            # Predict outcomes without real execution
            would_win, predicted_cost = self._predict_auction_outcome(action, context)
            would_click = self._predict_click(action, user_state, context)
            would_convert = self._predict_conversion(action, user_state, context, would_click)
            predicted_position = self._predict_position(action, context)
            predicted_revenue = self._predict_revenue(would_convert, user_state)
            
            # Create decision record
            decision = ShadowDecision(
                model_id=self.model_id,
                model_type=self.model_type,
                user_id=user_id,
                timestamp=datetime.now(),
                bid_amount=action.bid_amount,
                creative_id=action.creative_id,
                channel=action.channel,
                confidence_score=confidence,
                model_epsilon=self.agent.epsilon,
                exploration_action=(action.bid_amount == 0),  # Simple exploration detection
                q_values=q_values.tolist(),
                user_state=user_state.to_dict(),
                environment_context=context.copy(),
                would_win_auction=would_win,
                would_click=would_click,
                would_convert=would_convert,
                predicted_position=predicted_position,
                predicted_cost=predicted_cost,
                predicted_revenue=predicted_revenue
            )
            
            # Store decision
            self.decisions.append(decision)
            self._update_metrics(decision)
            
            # Check for risk violations
            self._check_risk_thresholds(decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision in shadow model {self.model_id}: {e}")
            raise
    
    def _predict_auction_outcome(self, action: Any, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Predict if bid would win and at what cost"""
        bid = action.bid_amount
        
        # Simulate competition based on context
        competition_level = context.get('competition_level', 0.5)
        avg_competitor_bid = context.get('avg_competitor_bid', 2.0)
        
        # Add realistic variance
        competitor_bid = avg_competitor_bid * (1 + np.random.normal(0, competition_level * 0.3))
        
        would_win = bid > competitor_bid
        predicted_cost = competitor_bid + 0.01 if would_win else 0.0
        
        return would_win, predicted_cost
    
    def _predict_click(self, action: Any, user_state: DynamicEnrichedState, context: Dict[str, Any]) -> bool:
        """Predict if user would click"""
        # Base CTR from creative and context
        base_ctr = 0.02  # Industry average
        
        # Creative quality adjustment
        if hasattr(user_state, 'creative_predicted_ctr'):
            base_ctr = user_state.creative_predicted_ctr
        
        # Context adjustments
        if context.get('is_peak_hour', False):
            base_ctr *= 1.2
        
        # Fatigue adjustment
        if hasattr(user_state, 'creative_fatigue') and user_state.creative_fatigue > 0.5:
            base_ctr *= (1 - user_state.creative_fatigue * 0.5)
        
        return np.random.random() < base_ctr
    
    def _predict_conversion(self, action: Any, user_state: DynamicEnrichedState, 
                          context: Dict[str, Any], clicked: bool) -> bool:
        """Predict if user would convert (only if clicked)"""
        if not clicked:
            return False
        
        # Base CVR from segment
        base_cvr = getattr(user_state, 'segment_cvr', 0.02)
        
        # Journey stage adjustment
        stage_multipliers = {0: 0.5, 1: 0.7, 2: 1.0, 3: 1.5, 4: 2.0}
        stage = getattr(user_state, 'stage', 0)
        base_cvr *= stage_multipliers.get(stage, 1.0)
        
        # Channel effectiveness
        channel_multipliers = {
            'paid_search': 1.2,
            'email': 1.1,
            'display': 0.8,
            'social': 0.9,
            'organic': 1.0
        }
        base_cvr *= channel_multipliers.get(action.channel, 1.0)
        
        return np.random.random() < base_cvr
    
    def _predict_position(self, action: Any, context: Dict[str, Any]) -> float:
        """Predict ad position"""
        bid = action.bid_amount
        avg_competitor_bid = context.get('avg_competitor_bid', 2.0)
        
        if bid <= avg_competitor_bid * 0.5:
            return np.random.uniform(8, 10)
        elif bid <= avg_competitor_bid:
            return np.random.uniform(4, 8)
        elif bid <= avg_competitor_bid * 1.5:
            return np.random.uniform(2, 4)
        else:
            return np.random.uniform(1, 2)
    
    def _predict_revenue(self, converted: bool, user_state: DynamicEnrichedState) -> float:
        """Predict revenue if conversion occurs"""
        if not converted:
            return 0.0
        
        # Base revenue from segment
        base_revenue = getattr(user_state, 'segment_avg_ltv', 100.0)
        
        # Add variance
        return base_revenue * np.random.lognormal(0, 0.3)
    
    def _update_metrics(self, decision: ShadowDecision):
        """Update aggregated metrics"""
        metrics = self.metrics
        
        # Update counters
        metrics.total_decisions += 1
        n = metrics.total_decisions
        
        # Running averages
        metrics.avg_bid = ((n-1) * metrics.avg_bid + decision.bid_amount) / n
        metrics.avg_confidence = ((n-1) * metrics.avg_confidence + decision.confidence_score) / n
        
        # Performance predictions
        metrics.predicted_win_rate = ((n-1) * metrics.predicted_win_rate + float(decision.would_win_auction)) / n
        metrics.predicted_ctr = ((n-1) * metrics.predicted_ctr + float(decision.would_click)) / n
        metrics.predicted_cvr = ((n-1) * metrics.predicted_cvr + float(decision.would_convert)) / n
        
        # ROAS calculation
        if decision.predicted_cost > 0:
            roas = decision.predicted_revenue / decision.predicted_cost
            metrics.predicted_roas = ((n-1) * metrics.predicted_roas + roas) / n
        
        # Risk metrics
        metrics.max_bid_observed = max(metrics.max_bid_observed, decision.bid_amount)
        
        # Channel distribution
        channel = decision.channel
        if channel not in metrics.channel_distribution:
            metrics.channel_distribution[channel] = 0
        metrics.channel_distribution[channel] = (
            (metrics.channel_distribution[channel] * (n-1) + 1) / n
        )
        
        # Learning progression
        metrics.epsilon_progression.append(decision.model_epsilon)
        if len(metrics.epsilon_progression) > 1000:  # Keep last 1000
            metrics.epsilon_progression = metrics.epsilon_progression[-1000:]
    
    def _check_risk_thresholds(self, decision: ShadowDecision):
        """Check if decision violates risk thresholds"""
        if decision.bid_amount > self.risk_thresholds['max_bid']:
            logger.warning(f"Shadow model {self.model_id} bid ${decision.bid_amount:.2f} exceeds threshold ${self.risk_thresholds['max_bid']:.2f}")
        
        if self.metrics.max_bid_observed > self.risk_thresholds['max_bid'] * 1.5:
            logger.error(f"Shadow model {self.model_id} showing dangerous bidding behavior")
    
    def get_current_metrics(self) -> ShadowMetrics:
        """Get current performance metrics"""
        return copy.deepcopy(self.metrics)
    
    def get_recent_decisions(self, count: int = 100) -> List[ShadowDecision]:
        """Get recent decisions"""
        return list(self.decisions)[-count:]

class ShadowTestingEngine:
    """Main engine for shadow mode testing"""
    
    def __init__(self, config_path: str = "shadow_testing_config.json"):
        self.config = self._load_config(config_path)
        self.session_id = str(uuid.uuid4())
        
        # Initialize base components (shared across all models)
        self.base_components = self._initialize_base_components()
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
        # Comparison tracking
        self.comparisons = deque(maxlen=50000)
        self.comparison_metrics = {}
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_comparisons = 0
        
        # Emergency controls
        self.emergency_controller = get_emergency_controller()
        
        logger.info(f"Shadow testing engine initialized with {len(self.models)} models")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load shadow testing configuration"""
        default_config = {
            "models": {
                "production": {
                    "model_id": "production_v1",
                    "model_path": "models/production_model.pt",
                    "learning_rate": 1e-4,
                    "epsilon": 0.05,
                    "max_bid_limit": 15.0
                },
                "shadow": {
                    "model_id": "shadow_v1",
                    "model_path": "models/shadow_model.pt",
                    "learning_rate": 1e-3,
                    "epsilon": 0.15,
                    "max_bid_limit": 20.0
                },
                "control": {
                    "model_id": "control_v1",
                    "learning_rate": 5e-4,
                    "epsilon": 0.1,
                    "max_bid_limit": 10.0
                }
            },
            "comparison_settings": {
                "significant_divergence_threshold": 0.3,
                "performance_comparison_window": 1000,
                "risk_monitoring_enabled": True
            },
            "output_settings": {
                "log_level": "INFO",
                "save_decisions": True,
                "comparison_log_file": "shadow_comparisons.jsonl",
                "metrics_update_frequency": 100
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
        else:
            config = default_config
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created default config at {config_path}")
        
        return config
    
    def _initialize_base_components(self) -> Dict[str, Any]:
        """Initialize shared GAELP components"""
        return {
            'discovery': DiscoveryEngine(write_enabled=False),
            'creative_selector': CreativeSelector() if CreativeSelector else None,
            'attribution': AttributionEngine() if AttributionEngine else None,
            'budget_pacer': BudgetPacer() if BudgetPacer else None,
            'identity_resolver': IdentityResolver() if IdentityResolver else None,
            'parameter_manager': ParameterManager()
        }
    
    def _initialize_models(self):
        """Initialize all shadow testing models"""
        model_configs = self.config.get('models', {})
        
        for model_name, model_config in model_configs.items():
            try:
                model_type = ModelType(model_name.lower())
                model_runner = ShadowModelRunner(
                    model_config=model_config,
                    model_type=model_type,
                    base_components=self.base_components
                )
                self.models[model_name] = model_runner
                logger.info(f"Initialized {model_name} model: {model_config['model_id']}")
            except Exception as e:
                logger.error(f"Failed to initialize {model_name} model: {e}")
    
    async def run_shadow_comparison(self, 
                                   user_id: str,
                                   user_state: DynamicEnrichedState,
                                   context: Dict[str, Any]) -> ShadowComparison:
        """Run parallel decision making across all models"""
        # Make decisions in parallel
        decision_tasks = {}
        for model_name, model_runner in self.models.items():
            decision_tasks[model_name] = asyncio.create_task(
                model_runner.make_decision(user_id, user_state, context)
            )
        
        # Wait for all decisions
        decisions = {}
        for model_name, task in decision_tasks.items():
            try:
                decisions[model_name] = await task
            except Exception as e:
                logger.error(f"Error getting decision from {model_name}: {e}")
                continue
        
        # Create comparison
        comparison = ShadowComparison(
            session_id=self.session_id,
            timestamp=datetime.now(),
            user_id=user_id,
            production_decision=decisions.get('production'),
            shadow_decision=decisions.get('shadow'),
            control_decision=decisions.get('control')
        )
        
        # Calculate divergence metrics
        self._calculate_divergence(comparison)
        
        # Store comparison
        self.comparisons.append(comparison)
        self.total_comparisons += 1
        
        # Log comparison if significant
        if comparison.significant_divergence:
            logger.info(f"Significant divergence detected: User {user_id}")
            logger.info(f"  Production bid: ${comparison.production_decision.bid_amount:.2f}")
            logger.info(f"  Shadow bid: ${comparison.shadow_decision.bid_amount:.2f}")
            logger.info(f"  Bid divergence: {comparison.bid_divergence:.3f}")
        
        # Save to file if configured
        if self.config.get('output_settings', {}).get('save_decisions', False):
            self._save_comparison(comparison)
        
        return comparison
    
    def _calculate_divergence(self, comparison: ShadowComparison):
        """Calculate divergence metrics between decisions"""
        prod = comparison.production_decision
        shadow = comparison.shadow_decision
        
        if not prod or not shadow:
            return
        
        # Bid divergence (relative difference)
        if prod.bid_amount > 0:
            comparison.bid_divergence = abs(shadow.bid_amount - prod.bid_amount) / prod.bid_amount
        else:
            comparison.bid_divergence = abs(shadow.bid_amount - prod.bid_amount)
        
        # Creative and channel divergence
        comparison.creative_divergence = (prod.creative_id != shadow.creative_id)
        comparison.channel_divergence = (prod.channel != shadow.channel)
        
        # Performance predictions
        comparison.production_predicted_value = (
            prod.predicted_revenue - prod.predicted_cost
        )
        comparison.shadow_predicted_value = (
            shadow.predicted_revenue - shadow.predicted_cost
        )
        
        if comparison.control_decision:
            comparison.control_predicted_value = (
                comparison.control_decision.predicted_revenue - 
                comparison.control_decision.predicted_cost
            )
        
        # Determine if divergence is significant
        threshold = self.config.get('comparison_settings', {}).get('significant_divergence_threshold', 0.3)
        comparison.significant_divergence = (
            comparison.bid_divergence > threshold or
            comparison.creative_divergence or
            comparison.channel_divergence
        )
        
        # Check if shadow outperforms
        comparison.shadow_outperforms = (
            comparison.shadow_predicted_value > comparison.production_predicted_value
        )
    
    def _save_comparison(self, comparison: ShadowComparison):
        """Save comparison to log file"""
        log_file = self.config.get('output_settings', {}).get('comparison_log_file', 'shadow_comparisons.jsonl')
        
        try:
            comparison_dict = asdict(comparison)
            # Convert datetime objects to strings
            comparison_dict['timestamp'] = comparison.timestamp.isoformat()
            if comparison_dict['production_decision']:
                comparison_dict['production_decision']['timestamp'] = comparison_dict['production_decision']['timestamp'].isoformat() if 'timestamp' in comparison_dict['production_decision'] else None
            if comparison_dict['shadow_decision']:
                comparison_dict['shadow_decision']['timestamp'] = comparison_dict['shadow_decision']['timestamp'].isoformat() if 'timestamp' in comparison_dict['shadow_decision'] else None
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(comparison_dict) + '\n')
        except Exception as e:
            logger.error(f"Error saving comparison: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.start_time.isoformat(),
                'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
                'total_comparisons': self.total_comparisons
            },
            'model_metrics': {},
            'divergence_analysis': {},
            'performance_comparison': {},
            'risk_analysis': {}
        }
        
        # Individual model metrics
        for model_name, model_runner in self.models.items():
            report['model_metrics'][model_name] = asdict(model_runner.get_current_metrics())
        
        # Divergence analysis
        if self.comparisons:
            comparisons = list(self.comparisons)
            
            report['divergence_analysis'] = {
                'total_significant_divergences': sum(1 for c in comparisons if c.significant_divergence),
                'avg_bid_divergence': np.mean([c.bid_divergence for c in comparisons if c.bid_divergence is not None]),
                'creative_divergence_rate': sum(1 for c in comparisons if c.creative_divergence) / len(comparisons),
                'channel_divergence_rate': sum(1 for c in comparisons if c.channel_divergence) / len(comparisons),
                'shadow_outperform_rate': sum(1 for c in comparisons if c.shadow_outperforms) / len(comparisons)
            }
        
        # Performance comparison
        if len(self.comparisons) > 100:
            recent_comparisons = list(self.comparisons)[-1000:]
            
            prod_values = [c.production_predicted_value for c in recent_comparisons 
                          if c.production_predicted_value is not None]
            shadow_values = [c.shadow_predicted_value for c in recent_comparisons 
                           if c.shadow_predicted_value is not None]
            
            if prod_values and shadow_values:
                report['performance_comparison'] = {
                    'production_avg_value': np.mean(prod_values),
                    'shadow_avg_value': np.mean(shadow_values),
                    'performance_lift': (np.mean(shadow_values) - np.mean(prod_values)) / max(0.001, abs(np.mean(prod_values))),
                    'shadow_win_rate': sum(1 for i in range(min(len(prod_values), len(shadow_values))) 
                                         if shadow_values[i] > prod_values[i]) / min(len(prod_values), len(shadow_values))
                }
        
        # Risk analysis
        risk_flags = []
        for model_name, model_runner in self.models.items():
            metrics = model_runner.get_current_metrics()
            
            if metrics.max_bid_observed > 15.0:
                risk_flags.append(f"{model_name}: High bid observed (${metrics.max_bid_observed:.2f})")
            
            if metrics.predicted_win_rate < 0.1:
                risk_flags.append(f"{model_name}: Low win rate ({metrics.predicted_win_rate:.3f})")
            
            if len(metrics.epsilon_progression) > 10 and metrics.epsilon_progression[-1] > 0.5:
                risk_flags.append(f"{model_name}: High exploration rate ({metrics.epsilon_progression[-1]:.3f})")
        
        report['risk_analysis'] = {
            'risk_flags': risk_flags,
            'total_risk_flags': len(risk_flags)
        }
        
        return report
    
    async def run_continuous_testing(self, duration_minutes: int = 60):
        """Run continuous shadow testing for specified duration"""
        logger.info(f"Starting continuous shadow testing for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Synthetic user stream for testing
        user_counter = 0
        
        while time.time() < end_time:
            try:
                # Generate synthetic user
                user_id = f"shadow_test_user_{user_counter}"
                user_counter += 1
                
                # Create user state (in real system, this comes from production traffic)
                user_state = self._generate_synthetic_user_state()
                context = self._generate_synthetic_context()
                
                # Run comparison
                comparison = await self.run_shadow_comparison(user_id, user_state, context)
                
                # Log progress periodically
                if self.total_comparisons % self.config.get('output_settings', {}).get('metrics_update_frequency', 100) == 0:
                    logger.info(f"Shadow testing progress: {self.total_comparisons} comparisons completed")
                    
                    # Check emergency status
                    if not self.emergency_controller.is_system_healthy():
                        logger.warning("Emergency system indicates unhealthy state - continuing with caution")
                
                # Small delay to simulate realistic traffic
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in continuous testing loop: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Shadow testing completed. Total comparisons: {self.total_comparisons}")
        
        # Generate final report
        final_report = self.get_performance_report()
        
        # Save final report
        report_file = f"shadow_test_report_{self.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Final report saved to {report_file}")
        return final_report
    
    def _generate_synthetic_user_state(self) -> DynamicEnrichedState:
        """Generate synthetic user state for testing"""
        # In production, this would come from real user interactions
        patterns = self.base_components['discovery'].discover_all_patterns()
        segments = list(patterns.user_patterns.get('segments', {}).keys())
        
        if not segments:
            segments = ['researching_parent', 'concerned_parent']
        
        segment_name = np.random.choice(segments)
        segment_data = patterns.user_patterns.get('segments', {}).get(segment_name, {})
        
        return DynamicEnrichedState(
            # Journey state
            stage=np.random.randint(0, 5),
            touchpoints_seen=np.random.randint(1, 10),
            days_since_first_touch=np.random.exponential(3.0),
            
            # Segment info
            segment_name=segment_name,
            segment_cvr=segment_data.get('conversion_rate', 0.02),
            segment_engagement=segment_data.get('engagement_score', 0.5),
            
            # Context
            device=np.random.choice(['mobile', 'desktop', 'tablet']),
            channel=np.random.choice(['organic', 'paid_search', 'social', 'display', 'email']),
            hour_of_day=np.random.randint(0, 24),
            day_of_week=np.random.randint(0, 7),
            
            # Performance indicators
            creative_fatigue=np.random.beta(2, 5),
            budget_spent_ratio=np.random.beta(2, 3),
            
            # Identity
            cross_device_confidence=np.random.beta(3, 2),
            is_returning_user=np.random.choice([True, False])
        )
    
    def _generate_synthetic_context(self) -> Dict[str, Any]:
        """Generate synthetic context for testing"""
        return {
            'competition_level': np.random.beta(2, 2),
            'avg_competitor_bid': np.random.lognormal(1.0, 0.5),
            'is_peak_hour': np.random.choice([True, False], p=[0.3, 0.7]),
            'daily_budget': 1000.0,
            'budget_spent': np.random.uniform(0, 800),
            'time_remaining': np.random.uniform(1, 12)
        }

def create_shadow_testing_config():
    """Create a comprehensive shadow testing configuration"""
    config = {
        "models": {
            "production": {
                "model_id": "production_gaelp_v1",
                "model_path": "models/production_fortified_rl.pt",
                "learning_rate": 1e-4,
                "epsilon": 0.05,  # Low exploration for production
                "max_bid_limit": 15.0,
                "max_daily_spend": 3000.0,
                "description": "Current production model with conservative bidding"
            },
            "shadow": {
                "model_id": "experimental_gaelp_v2",
                "model_path": "models/experimental_model.pt",
                "learning_rate": 2e-4,
                "epsilon": 0.12,  # Higher exploration for testing
                "max_bid_limit": 20.0,
                "max_daily_spend": 4000.0,
                "description": "Experimental model with enhanced creative selection"
            },
            "control": {
                "model_id": "baseline_random",
                "learning_rate": 1e-4,
                "epsilon": 0.3,  # Random baseline
                "max_bid_limit": 10.0,
                "max_daily_spend": 2000.0,
                "description": "Random baseline for comparison"
            }
        },
        "comparison_settings": {
            "significant_divergence_threshold": 0.25,
            "performance_comparison_window": 2000,
            "risk_monitoring_enabled": True,
            "statistical_significance_threshold": 0.05,
            "minimum_sample_size": 100
        },
        "output_settings": {
            "log_level": "INFO",
            "save_decisions": True,
            "comparison_log_file": "shadow_comparisons.jsonl",
            "metrics_update_frequency": 50,
            "generate_reports": True,
            "report_frequency_minutes": 15
        },
        "safety_settings": {
            "max_total_theoretical_spend": 10000.0,
            "emergency_stop_on_anomalies": True,
            "bid_spike_threshold": 5.0,
            "performance_degradation_threshold": 0.5
        }
    }
    
    return config

async def main():
    """Main function for shadow testing demonstration"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('shadow_testing.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("=" * 80)
    logger.info("GAELP SHADOW MODE TESTING - PRODUCTION GRADE")
    logger.info("=" * 80)
    
    # Create configuration
    config = create_shadow_testing_config()
    with open('shadow_testing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize shadow testing engine
    engine = ShadowTestingEngine()
    
    # Run continuous testing
    try:
        final_report = await engine.run_continuous_testing(duration_minutes=30)
        
        logger.info("SHADOW TESTING SUMMARY:")
        logger.info(f"Total comparisons: {final_report['session_info']['total_comparisons']}")
        logger.info(f"Significant divergences: {final_report['divergence_analysis'].get('total_significant_divergences', 'N/A')}")
        logger.info(f"Shadow outperform rate: {final_report['divergence_analysis'].get('shadow_outperform_rate', 'N/A'):.3f}")
        
        if 'performance_comparison' in final_report:
            perf = final_report['performance_comparison']
            logger.info(f"Performance lift: {perf.get('performance_lift', 'N/A'):.3f}")
            logger.info(f"Shadow win rate: {perf.get('shadow_win_rate', 'N/A'):.3f}")
        
        # Risk analysis
        risk_flags = final_report['risk_analysis'].get('risk_flags', [])
        if risk_flags:
            logger.warning("RISK FLAGS DETECTED:")
            for flag in risk_flags:
                logger.warning(f"  - {flag}")
        else:
            logger.info("No risk flags detected")
        
    except KeyboardInterrupt:
        logger.info("Shadow testing interrupted by user")
    except Exception as e:
        logger.error(f"Error in shadow testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())