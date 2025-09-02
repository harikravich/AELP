#!/usr/bin/env python3
"""
SHADOW MODE MANAGER
Orchestrates parallel testing of multiple models without real money spending
"""

import asyncio
import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import sqlite3
import concurrent.futures

from shadow_mode_testing import ShadowTestingEngine, ShadowComparison, ShadowDecision
from shadow_mode_state import DynamicEnrichedState, create_synthetic_state_for_testing
from shadow_mode_environment import ShadowModeEnvironment
try:
    from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent
except ImportError:
    ProductionFortifiedRLAgent = None

from discovery_engine import GA4RealTimeDataPipeline as DiscoveryEngine
from gaelp_parameter_manager import ParameterManager
try:
    from emergency_controls import get_emergency_controller
except ImportError:
    def get_emergency_controller():
        class MockController:
            def is_system_healthy(self): return True
        return MockController()

logger = logging.getLogger(__name__)

@dataclass
class ShadowTestConfiguration:
    """Configuration for shadow testing"""
    test_name: str
    duration_hours: float
    models: Dict[str, Dict[str, Any]]
    traffic_percentage: float = 1.0  # Percentage of traffic to shadow test
    comparison_threshold: float = 0.1  # Threshold for significant differences
    statistical_confidence: float = 0.95
    min_sample_size: int = 100
    save_all_decisions: bool = True
    real_time_reporting: bool = True

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model"""
    model_id: str
    total_decisions: int = 0
    total_impressions: int = 0
    total_clicks: int = 0
    total_conversions: int = 0
    total_spend: float = 0.0
    total_revenue: float = 0.0
    
    avg_bid: float = 0.0
    avg_position: float = 0.0
    win_rate: float = 0.0
    ctr: float = 0.0
    cvr: float = 0.0
    roas: float = 0.0
    
    confidence_score: float = 0.0
    risk_score: float = 0.0
    
    segment_performance: Dict[str, Dict[str, float]] = None
    channel_performance: Dict[str, Dict[str, float]] = None
    temporal_performance: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.segment_performance is None:
            self.segment_performance = {}
        if self.channel_performance is None:
            self.channel_performance = {}
        if self.temporal_performance is None:
            self.temporal_performance = {}

class ShadowModeManager:
    """
    Main manager for shadow mode testing across multiple models
    """
    
    def __init__(self, config: ShadowTestConfiguration):
        self.config = config
        self.start_time = datetime.now()
        self.session_id = f"shadow_{int(time.time())}"
        
        # Initialize components
        self.parameter_manager = ParameterManager()
        self.discovery_engine = DiscoveryEngine(write_enabled=False)
        self.emergency_controller = get_emergency_controller()
        
        # Initialize models
        self.models = {}
        self.environments = {}
        self._initialize_models()
        
        # Data storage
        self.db_path = f"shadow_testing_{self.session_id}.db"
        self._initialize_database()
        
        # Metrics tracking
        self.model_metrics = {}
        self.comparison_history = deque(maxlen=100000)
        self.real_time_stats = defaultdict(lambda: defaultdict(float))
        
        # Control flags
        self.is_running = False
        self.should_stop = False
        
        # Statistical testing
        self.statistical_results = {}
        
        logger.info(f"Shadow Mode Manager initialized for test: {config.test_name}")
    
    def _initialize_models(self):
        """Initialize all models for testing"""
        base_components = {
            'discovery': self.discovery_engine,
            'parameter_manager': self.parameter_manager
        }
        
        for model_name, model_config in self.config.models.items():
            try:
                # Create model
                model = self._create_model(model_config, base_components)
                self.models[model_name] = model
                
                # Create dedicated environment
                env = ShadowModeEnvironment(
                    parameter_manager=self.parameter_manager,
                    discovery_engine=self.discovery_engine
                )
                self.environments[model_name] = env
                
                # Initialize metrics
                self.model_metrics[model_name] = ModelPerformanceMetrics(
                    model_id=model_config.get('model_id', model_name)
                )
                
                logger.info(f"Initialized model {model_name}: {model_config.get('model_id', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {e}")
    
    def _create_model(self, model_config: Dict[str, Any], base_components: Dict[str, Any]):
        """Create a model instance"""
        
        # For now, create a simplified model wrapper
        # In production, this would load actual trained models
        class ShadowModelWrapper:
            def __init__(self, config, components):
                self.config = config
                self.model_id = config.get('model_id', 'unknown')
                self.learning_rate = config.get('learning_rate', 1e-4)
                self.epsilon = config.get('epsilon', 0.1)
                self.components = components
                
                # Simulate model characteristics
                self.bid_bias = config.get('bid_bias', 1.0)  # Tendency to bid higher/lower
                self.exploration_rate = config.get('exploration_rate', self.epsilon)
                self.risk_tolerance = config.get('risk_tolerance', 0.5)
                
                # Performance characteristics
                self.creative_preference = config.get('creative_preference', 'balanced')
                self.channel_preference = config.get('channel_preference', 'balanced')
                
            def make_decision(self, user_state: DynamicEnrichedState, context: Dict[str, Any]) -> Dict[str, Any]:
                """Make a bidding decision"""
                
                # Base bid calculation (simplified)
                base_bid = 2.0 * self.bid_bias
                
                # Adjust for user value
                value_multiplier = 1.0 + (user_state.segment_cvr - 0.02) * 10  # Scale CVR impact
                base_bid *= value_multiplier
                
                # Adjust for competition
                competition_factor = context.get('competition_level', 0.5)
                base_bid *= (1.0 + competition_factor * 0.5)
                
                # Add exploration noise
                if np.random.random() < self.exploration_rate:
                    noise = np.random.normal(0, base_bid * 0.3)
                    base_bid += noise
                
                # Ensure reasonable bounds
                bid_amount = max(0.5, min(15.0, base_bid))
                
                # Creative selection (simplified)
                if self.creative_preference == 'conservative':
                    creative_id = np.random.randint(0, 20)  # Lower creative IDs
                elif self.creative_preference == 'aggressive':
                    creative_id = np.random.randint(30, 50)  # Higher creative IDs
                else:
                    creative_id = np.random.randint(0, 50)
                
                # Channel selection
                channels = ['paid_search', 'display', 'social', 'email']
                if self.channel_preference == 'search_focused':
                    channel = np.random.choice(channels, p=[0.6, 0.2, 0.1, 0.1])
                elif self.channel_preference == 'display_focused':
                    channel = np.random.choice(channels, p=[0.1, 0.6, 0.2, 0.1])
                else:
                    channel = np.random.choice(channels)
                
                # Calculate confidence
                confidence = max(0.1, min(1.0, 1.0 - self.exploration_rate + np.random.normal(0, 0.1)))
                
                return {
                    'bid_amount': bid_amount,
                    'creative_id': creative_id,
                    'channel': channel,
                    'confidence': confidence,
                    'model_epsilon': self.exploration_rate,
                    'exploration_action': np.random.random() < self.exploration_rate
                }
        
        return ShadowModelWrapper(model_config, base_components)
    
    def _initialize_database(self):
        """Initialize SQLite database for storing results"""
        conn = sqlite3.connect(self.db_path)
        
        # Create tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                model_id TEXT,
                user_id TEXT,
                bid_amount REAL,
                creative_id INTEGER,
                channel TEXT,
                confidence_score REAL,
                won_auction BOOLEAN,
                clicked BOOLEAN,
                converted BOOLEAN,
                spend REAL,
                revenue REAL,
                user_state TEXT,
                context TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                user_id TEXT,
                production_bid REAL,
                shadow_bid REAL,
                bid_divergence REAL,
                creative_divergence BOOLEAN,
                channel_divergence BOOLEAN,
                production_value REAL,
                shadow_value REAL,
                significant_divergence BOOLEAN,
                comparison_data TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS metrics_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                model_id TEXT,
                total_decisions INTEGER,
                win_rate REAL,
                ctr REAL,
                cvr REAL,
                roas REAL,
                avg_bid REAL,
                confidence_score REAL,
                risk_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized: {self.db_path}")
    
    async def run_shadow_testing(self):
        """Main loop for shadow testing"""
        logger.info("Starting shadow mode testing...")
        
        self.is_running = True
        start_time = time.time()
        end_time = start_time + (self.config.duration_hours * 3600)
        
        # Metrics reporting task
        reporting_task = asyncio.create_task(self._periodic_reporting())
        
        try:
            user_counter = 0
            
            while time.time() < end_time and not self.should_stop:
                # Check system health
                if not self.emergency_controller.is_system_healthy():
                    logger.warning("Emergency system indicates issues - continuing with caution")
                
                # Generate synthetic user
                user_id = f"shadow_user_{user_counter}"
                user_counter += 1
                
                # Create user state
                user_state = create_synthetic_state_for_testing()
                context = self._generate_context()
                
                # Run parallel decisions across all models
                decisions = await self._run_parallel_decisions(user_id, user_state, context)
                
                # Store decisions and update metrics
                await self._process_decisions(user_id, decisions, user_state, context)
                
                # Compare models
                if len(decisions) >= 2:
                    await self._compare_models(user_id, decisions)
                
                # Small delay to simulate realistic traffic
                await asyncio.sleep(0.05)
                
                # Periodic statistical analysis
                if user_counter % 1000 == 0:
                    await self._run_statistical_analysis()
            
            logger.info("Shadow testing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in shadow testing: {e}")
            raise
        finally:
            self.is_running = False
            reporting_task.cancel()
    
    async def _run_parallel_decisions(self, user_id: str, user_state: DynamicEnrichedState, 
                                    context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run decision making across all models in parallel"""
        
        decisions = {}
        
        # Create tasks for each model
        tasks = {}
        for model_name, model in self.models.items():
            tasks[model_name] = asyncio.create_task(
                self._make_model_decision(model_name, model, user_id, user_state, context)
            )
        
        # Wait for all decisions
        for model_name, task in tasks.items():
            try:
                decisions[model_name] = await task
            except Exception as e:
                logger.error(f"Error getting decision from {model_name}: {e}")
                continue
        
        return decisions
    
    async def _make_model_decision(self, model_name: str, model: Any, user_id: str,
                                  user_state: DynamicEnrichedState, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision for a specific model"""
        
        # Get decision from model
        decision_data = model.make_decision(user_state, context)
        
        # Simulate auction outcome
        environment = self.environments[model_name]
        
        class MockAction:
            def __init__(self, data):
                self.bid_amount = data['bid_amount']
                self.creative_id = data['creative_id'] 
                self.channel = data['channel']
        
        action = MockAction(decision_data)
        
        # Step environment to get realistic outcome
        obs, reward, terminated, truncated, info = environment.step(action)
        
        # Combine decision with outcome
        result = {
            **decision_data,
            'user_id': user_id,
            'model_name': model_name,
            'timestamp': datetime.now(),
            'auction_result': info.get('auction_result'),
            'interaction_result': info.get('interaction_result'),
            'reward': reward,
            'spend': info.get('spend', 0.0),
            'revenue': info.get('revenue', 0.0)
        }
        
        return result
    
    async def _process_decisions(self, user_id: str, decisions: Dict[str, Dict[str, Any]],
                               user_state: DynamicEnrichedState, context: Dict[str, Any]):
        """Process and store decisions"""
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        
        for model_name, decision in decisions.items():
            auction_result = decision.get('auction_result')
            interaction_result = decision.get('interaction_result')
            
            conn.execute('''
                INSERT INTO decisions 
                (session_id, timestamp, model_id, user_id, bid_amount, creative_id, channel,
                 confidence_score, won_auction, clicked, converted, spend, revenue, 
                 user_state, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.session_id,
                decision['timestamp'].isoformat(),
                decision.get('model_name', model_name),
                user_id,
                decision['bid_amount'],
                decision['creative_id'],
                decision['channel'],
                decision.get('confidence', 0.5),
                auction_result.won if auction_result else False,
                interaction_result.clicked if interaction_result else False,
                interaction_result.converted if interaction_result else False,
                decision.get('spend', 0.0),
                decision.get('revenue', 0.0),
                json.dumps(user_state.to_dict()),
                json.dumps(context)
            ))
        
        conn.commit()
        conn.close()
        
        # Update real-time metrics
        for model_name, decision in decisions.items():
            await self._update_model_metrics(model_name, decision)
    
    async def _update_model_metrics(self, model_name: str, decision: Dict[str, Any]):
        """Update metrics for a model"""
        metrics = self.model_metrics[model_name]
        
        metrics.total_decisions += 1
        
        auction_result = decision.get('auction_result')
        interaction_result = decision.get('interaction_result')
        
        if auction_result and auction_result.won:
            metrics.total_impressions += 1
            metrics.total_spend += decision.get('spend', 0.0)
            
            # Update running averages
            n = metrics.total_impressions
            metrics.avg_bid = ((n-1) * metrics.avg_bid + decision['bid_amount']) / n
            metrics.avg_position = ((n-1) * metrics.avg_position + auction_result.position) / n
            metrics.win_rate = metrics.total_impressions / metrics.total_decisions
            
            if interaction_result and interaction_result.clicked:
                metrics.total_clicks += 1
                metrics.ctr = metrics.total_clicks / metrics.total_impressions
                
                if interaction_result.converted:
                    metrics.total_conversions += 1
                    metrics.total_revenue += decision.get('revenue', 0.0)
                    metrics.cvr = metrics.total_conversions / metrics.total_clicks
            
            # Update ROAS
            if metrics.total_spend > 0:
                metrics.roas = metrics.total_revenue / metrics.total_spend
        
        # Update confidence and risk scores
        confidence_scores = [d.get('confidence', 0.5) for d in [decision]]
        metrics.confidence_score = np.mean(confidence_scores)
        
        # Risk score based on bid variance and performance
        recent_bids = [decision['bid_amount']]  # In real system, track more history
        if len(recent_bids) > 1:
            bid_std = np.std(recent_bids)
            avg_bid = np.mean(recent_bids)
            metrics.risk_score = min(1.0, bid_std / max(0.1, avg_bid))
        
        # Update real-time stats
        hour = datetime.now().hour
        self.real_time_stats[model_name][f'hour_{hour}_decisions'] += 1
        self.real_time_stats[model_name]['total_decisions'] += 1
        
        if auction_result and auction_result.won:
            self.real_time_stats[model_name][f'hour_{hour}_impressions'] += 1
            self.real_time_stats[model_name]['total_impressions'] += 1
    
    async def _compare_models(self, user_id: str, decisions: Dict[str, Dict[str, Any]]):
        """Compare decisions between models"""
        
        decision_list = list(decisions.items())
        
        # Compare each pair of models
        for i in range(len(decision_list)):
            for j in range(i + 1, len(decision_list)):
                model1_name, decision1 = decision_list[i]
                model2_name, decision2 = decision_list[j]
                
                # Calculate divergence
                bid_divergence = abs(decision1['bid_amount'] - decision2['bid_amount']) / max(0.01, decision1['bid_amount'])
                creative_divergence = decision1['creative_id'] != decision2['creative_id']
                channel_divergence = decision1['channel'] != decision2['channel']
                
                # Calculate predicted values
                value1 = decision1.get('revenue', 0.0) - decision1.get('spend', 0.0)
                value2 = decision2.get('revenue', 0.0) - decision2.get('spend', 0.0)
                
                # Determine significance
                significant = (
                    bid_divergence > self.config.comparison_threshold or
                    creative_divergence or
                    channel_divergence
                )
                
                # Store comparison
                comparison_data = {
                    'model1': model1_name,
                    'model2': model2_name,
                    'bid_divergence': bid_divergence,
                    'creative_divergence': creative_divergence,
                    'channel_divergence': channel_divergence,
                    'value1': value1,
                    'value2': value2,
                    'significant': significant
                }
                
                self.comparison_history.append(comparison_data)
                
                # Store in database
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT INTO comparisons 
                    (session_id, timestamp, user_id, production_bid, shadow_bid,
                     bid_divergence, creative_divergence, channel_divergence,
                     production_value, shadow_value, significant_divergence, comparison_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.session_id,
                    datetime.now().isoformat(),
                    user_id,
                    decision1['bid_amount'],
                    decision2['bid_amount'],
                    bid_divergence,
                    creative_divergence,
                    channel_divergence,
                    value1,
                    value2,
                    significant,
                    json.dumps(comparison_data)
                ))
                conn.commit()
                conn.close()
    
    def _generate_context(self) -> Dict[str, Any]:
        """Generate realistic context for decisions"""
        return {
            'competition_level': np.random.beta(2, 2),
            'avg_competitor_bid': np.random.lognormal(0.8, 0.4),
            'is_peak_hour': np.random.choice([True, False], p=[0.3, 0.7]),
            'daily_budget': 1000.0,
            'budget_spent': np.random.uniform(0, 800),
            'time_remaining': np.random.uniform(1, 12),
            'market_conditions': {
                'volatility': np.random.beta(2, 3),
                'demand': np.random.beta(3, 2)
            }
        }
    
    async def _periodic_reporting(self):
        """Periodic reporting task"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                if self.config.real_time_reporting:
                    report = self.generate_performance_report()
                    logger.info(f"Performance Report: {json.dumps(report, indent=2)}")
                
                # Save metrics snapshot
                await self._save_metrics_snapshot()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic reporting: {e}")
    
    async def _save_metrics_snapshot(self):
        """Save current metrics to database"""
        conn = sqlite3.connect(self.db_path)
        
        for model_name, metrics in self.model_metrics.items():
            conn.execute('''
                INSERT INTO metrics_snapshots
                (session_id, timestamp, model_id, total_decisions, win_rate, ctr, cvr, roas,
                 avg_bid, confidence_score, risk_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.session_id,
                datetime.now().isoformat(),
                metrics.model_id,
                metrics.total_decisions,
                metrics.win_rate,
                metrics.ctr,
                metrics.cvr,
                metrics.roas,
                metrics.avg_bid,
                metrics.confidence_score,
                metrics.risk_score
            ))
        
        conn.commit()
        conn.close()
    
    async def _run_statistical_analysis(self):
        """Run statistical analysis on model comparisons"""
        logger.info("Running statistical analysis...")
        
        # Basic statistical tests
        if len(self.comparison_history) < self.config.min_sample_size:
            return
        
        recent_comparisons = list(self.comparison_history)[-1000:]  # Last 1000 comparisons
        
        # Analyze bid divergence
        bid_divergences = [c['bid_divergence'] for c in recent_comparisons]
        significant_divergences = [c for c in recent_comparisons if c['significant']]
        
        # Performance comparison
        value_differences = [c['value2'] - c['value1'] for c in recent_comparisons if 'value1' in c and 'value2' in c]
        
        stats_summary = {
            'total_comparisons': len(recent_comparisons),
            'avg_bid_divergence': np.mean(bid_divergences) if bid_divergences else 0,
            'significant_divergence_rate': len(significant_divergences) / len(recent_comparisons),
            'avg_value_difference': np.mean(value_differences) if value_differences else 0,
            'value_difference_std': np.std(value_differences) if value_differences else 0
        }
        
        self.statistical_results = stats_summary
        logger.info(f"Statistical Summary: {stats_summary}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        runtime_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        
        report = {
            'test_info': {
                'test_name': self.config.test_name,
                'session_id': self.session_id,
                'runtime_minutes': runtime_minutes,
                'start_time': self.start_time.isoformat()
            },
            'model_performance': {},
            'comparisons': {
                'total_comparisons': len(self.comparison_history),
                'significant_divergences': sum(1 for c in self.comparison_history if c.get('significant', False))
            },
            'statistical_results': self.statistical_results
        }
        
        # Model performance
        for model_name, metrics in self.model_metrics.items():
            report['model_performance'][model_name] = {
                'total_decisions': metrics.total_decisions,
                'win_rate': metrics.win_rate,
                'ctr': metrics.ctr,
                'cvr': metrics.cvr,
                'roas': metrics.roas,
                'avg_bid': metrics.avg_bid,
                'confidence_score': metrics.confidence_score,
                'risk_score': metrics.risk_score,
                'total_spend': metrics.total_spend,
                'total_revenue': metrics.total_revenue
            }
        
        return report
    
    def stop_testing(self):
        """Stop shadow testing"""
        logger.info("Stopping shadow testing...")
        self.should_stop = True
    
    def get_test_results(self) -> Dict[str, Any]:
        """Get final test results"""
        return {
            'performance_report': self.generate_performance_report(),
            'database_path': self.db_path,
            'comparison_count': len(self.comparison_history),
            'models_tested': list(self.models.keys())
        }

def create_shadow_test_config() -> ShadowTestConfiguration:
    """Create a comprehensive shadow test configuration"""
    return ShadowTestConfiguration(
        test_name="GAELP_Production_vs_Experimental",
        duration_hours=2.0,  # 2 hour test
        models={
            'production': {
                'model_id': 'production_gaelp_v1.2',
                'model_path': 'models/production_model.pt',
                'learning_rate': 1e-4,
                'epsilon': 0.05,
                'bid_bias': 1.0,
                'exploration_rate': 0.05,
                'risk_tolerance': 0.4,
                'creative_preference': 'conservative',
                'channel_preference': 'balanced',
                'description': 'Current production model'
            },
            'experimental': {
                'model_id': 'experimental_gaelp_v2.0',
                'model_path': 'models/experimental_model.pt',
                'learning_rate': 2e-4,
                'epsilon': 0.12,
                'bid_bias': 1.1,
                'exploration_rate': 0.12,
                'risk_tolerance': 0.6,
                'creative_preference': 'aggressive',
                'channel_preference': 'search_focused',
                'description': 'Experimental model with enhanced features'
            },
            'baseline': {
                'model_id': 'random_baseline',
                'learning_rate': 1e-4,
                'epsilon': 0.3,
                'bid_bias': 0.8,
                'exploration_rate': 0.3,
                'risk_tolerance': 0.5,
                'creative_preference': 'balanced',
                'channel_preference': 'balanced',
                'description': 'Random baseline for comparison'
            }
        },
        traffic_percentage=1.0,
        comparison_threshold=0.15,
        statistical_confidence=0.95,
        min_sample_size=200,
        save_all_decisions=True,
        real_time_reporting=True
    )

async def main():
    """Main function for shadow testing demonstration"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('shadow_mode_manager.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("=" * 80)
    logger.info("GAELP SHADOW MODE MANAGER - PRODUCTION TESTING")
    logger.info("=" * 80)
    
    # Create configuration
    config = create_shadow_test_config()
    
    # Initialize manager
    manager = ShadowModeManager(config)
    
    try:
        # Run shadow testing
        await manager.run_shadow_testing()
        
        # Get final results
        results = manager.get_test_results()
        
        # Display results
        logger.info("=" * 80)
        logger.info("SHADOW TESTING RESULTS")
        logger.info("=" * 80)
        
        perf_report = results['performance_report']
        logger.info(f"Test completed: {config.test_name}")
        logger.info(f"Runtime: {perf_report['test_info']['runtime_minutes']:.1f} minutes")
        logger.info(f"Total comparisons: {results['comparison_count']}")
        
        # Model performance
        for model_name, metrics in perf_report['model_performance'].items():
            logger.info(f"\n{model_name.upper()} Model:")
            logger.info(f"  Decisions: {metrics['total_decisions']}")
            logger.info(f"  Win Rate: {metrics['win_rate']:.3f}")
            logger.info(f"  CTR: {metrics['ctr']:.3f}")
            logger.info(f"  CVR: {metrics['cvr']:.3f}")
            logger.info(f"  ROAS: {metrics['roas']:.2f}x")
            logger.info(f"  Avg Bid: ${metrics['avg_bid']:.2f}")
            logger.info(f"  Risk Score: {metrics['risk_score']:.3f}")
        
        # Comparisons
        comp = perf_report['comparisons']
        logger.info(f"\nCOMPARISONS:")
        logger.info(f"  Total: {comp['total_comparisons']}")
        logger.info(f"  Significant Divergences: {comp['significant_divergences']}")
        
        # Statistical results
        if perf_report['statistical_results']:
            stats = perf_report['statistical_results']
            logger.info(f"\nSTATISTICAL ANALYSIS:")
            logger.info(f"  Avg Bid Divergence: {stats.get('avg_bid_divergence', 0):.3f}")
            logger.info(f"  Significant Divergence Rate: {stats.get('significant_divergence_rate', 0):.3f}")
            logger.info(f"  Avg Value Difference: ${stats.get('avg_value_difference', 0):.2f}")
        
        logger.info(f"\nDatabase saved: {results['database_path']}")
        
    except KeyboardInterrupt:
        logger.info("Shadow testing interrupted by user")
        manager.stop_testing()
    except Exception as e:
        logger.error(f"Error in shadow testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())