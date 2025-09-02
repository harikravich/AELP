#!/usr/bin/env python3
"""
ONLINE LEARNING MONITORING DASHBOARD
Real-time monitoring of production continuous learning system

Monitors:
- Thompson Sampling strategy performance
- A/B test results and statistical significance  
- Safety guardrails and circuit breaker status
- Model update frequency and performance
- Production feedback loop health
"""

import asyncio
import logging
import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import deque, defaultdict
import numpy as np
import os

from production_online_learner import create_production_online_learner
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnlineLearningMonitor:
    """Monitor and visualize online learning system performance"""
    
    def __init__(self, discovery_engine: DiscoveryEngine = None):
        self.discovery = discovery_engine or DiscoveryEngine()
        self.metrics_history = deque(maxlen=1000)
        self.performance_data = defaultdict(list)
        
        # Mock agent for monitoring
        self.mock_agent = self._create_mock_agent()
        self.online_learner = create_production_online_learner(self.mock_agent, self.discovery)
        
        # Database for persistence
        self.db_path = "online_learning_monitor.db"
        self._init_monitoring_db()
        
    def _create_mock_agent(self):
        """Create mock agent for monitoring"""
        class MockAgent:
            async def select_action(self, state, deterministic=False):
                return {
                    'bid_amount': np.random.uniform(0.5, 2.0),
                    'budget_allocation': np.random.uniform(0.05, 0.2),
                    'creative_type': np.random.choice(['image', 'video', 'carousel']),
                    'target_audience': 'professionals'
                }
        
        return MockAgent()
    
    def _init_monitoring_db(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                strategy TEXT,
                conversion_rate REAL,
                reward REAL,
                spend REAL,
                revenue REAL,
                circuit_breaker BOOLEAN,
                active_experiments INTEGER,
                model_updates INTEGER
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                variant_id TEXT,
                timestamp TIMESTAMP,
                sample_size INTEGER,
                conversion_rate REAL,
                revenue_per_user REAL,
                statistical_significance REAL,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        status = self.online_learner.get_system_status()
        
        # Add timestamp
        current_time = datetime.now()
        
        # Get strategy performance
        strategy_performance = status.get('strategy_performance', {})
        
        # Calculate aggregate metrics
        metrics = {
            'timestamp': current_time.isoformat(),
            'circuit_breaker_active': status.get('circuit_breaker', False),
            'active_experiments': status.get('active_experiments', 0),
            'model_update_count': status.get('model_update_count', 0),
            'experience_buffer_size': status.get('experience_buffer_size', 0),
            'strategies': strategy_performance
        }
        
        # Add recent performance if available
        recent_perf = status.get('recent_performance', {})
        if recent_perf:
            metrics.update({
                'avg_reward': recent_perf.get('avg_reward', 0),
                'total_spend': recent_perf.get('total_spend', 0),
                'total_conversions': recent_perf.get('total_conversions', 0),
                'episodes': recent_perf.get('episodes', 0)
            })
        
        return metrics
    
    def store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in database"""
        conn = sqlite3.connect(self.db_path)
        
        # Store main metrics
        for strategy_id, strategy_data in metrics.get('strategies', {}).items():
            conn.execute('''
                INSERT INTO monitoring_metrics 
                (timestamp, strategy, conversion_rate, reward, spend, revenue, circuit_breaker, active_experiments, model_updates)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics['timestamp'],
                strategy_id,
                strategy_data.get('expected_conversion_rate', 0),
                0.0,  # Individual reward not available
                0.0,  # Individual spend not available
                0.0,  # Individual revenue not available  
                metrics.get('circuit_breaker_active', False),
                metrics.get('active_experiments', 0),
                metrics.get('model_update_count', 0)
            ))
        
        conn.commit()
        conn.close()
        
        # Add to memory for real-time display
        self.metrics_history.append(metrics)
    
    def simulate_online_learning_session(self, duration_minutes: int = 60):
        """Simulate online learning session for monitoring"""
        logger.info(f"Starting {duration_minutes}-minute simulation for monitoring...")
        
        end_time = time.time() + (duration_minutes * 60)
        episode = 0
        
        while time.time() < end_time:
            try:
                # Simulate user interaction
                user_id = f"sim_user_{episode}_{int(time.time())}"
                
                # Mock state
                state = {
                    'budget_remaining': np.random.uniform(100, 1000),
                    'daily_spend': np.random.uniform(10, 500),
                    'current_roas': np.random.uniform(0.8, 2.5),
                    'competition_level': np.random.uniform(0.3, 0.8)
                }
                
                # Get action from online learner
                action = asyncio.run(
                    self.online_learner.select_production_action(state, user_id)
                )
                
                # Simulate outcome
                conversion_prob = self._calculate_conversion_probability(action, state)
                converted = np.random.random() < conversion_prob
                
                spend = action.get('bid_amount', 1.0) * np.random.uniform(0.8, 1.2)
                revenue = spend * np.random.uniform(1.5, 4.0) if converted else 0
                reward = revenue - spend
                
                outcome = {
                    'conversion': converted,
                    'reward': reward,
                    'spend': spend,
                    'revenue': revenue,
                    'channel': np.random.choice(['google', 'facebook', 'tiktok']),
                    'campaign_id': f"sim_campaign_{episode}"
                }
                
                # Record outcome
                self.online_learner.record_production_outcome(action, outcome, user_id)
                
                # Collect and store metrics every 10 episodes
                if episode % 10 == 0:
                    metrics = self.collect_current_metrics()
                    self.store_metrics(metrics)
                
                episode += 1
                
                # Brief pause
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Simulation error in episode {episode}: {e}")
                time.sleep(5)
        
        logger.info(f"Simulation complete. Ran {episode} episodes.")
    
    def _calculate_conversion_probability(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Calculate realistic conversion probability"""
        base_prob = 0.02  # 2% base conversion rate
        
        # Bid amount effect
        bid_multiplier = min(2.0, action.get('bid_amount', 1.0))
        bid_effect = 1.0 + (bid_multiplier - 1.0) * 0.3  # Diminishing returns
        
        # ROAS effect
        roas_effect = min(1.5, state.get('current_roas', 1.0) / 1.0)
        
        # Competition effect (inverse)
        comp_effect = 1.0 / (1.0 + state.get('competition_level', 0.5))
        
        # Creative type effect
        creative_type = action.get('creative_type', 'image')
        creative_effect = {
            'image': 1.0,
            'video': 1.3,
            'carousel': 1.1
        }.get(creative_type, 1.0)
        
        final_prob = base_prob * bid_effect * roas_effect * comp_effect * creative_effect
        return min(0.15, max(0.001, final_prob))  # Cap between 0.1% and 15%
    
    def create_performance_dashboard(self, save_path: str = "online_learning_dashboard.png"):
        """Create comprehensive performance dashboard"""
        if len(self.metrics_history) < 10:
            logger.warning("Not enough data for dashboard. Running simulation...")
            self.simulate_online_learning_session(5)  # 5 minute sim
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Online Learning System Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Strategy Performance Over Time
        self._plot_strategy_performance(ax1)
        
        # Plot 2: Circuit Breaker and Safety Status
        self._plot_safety_status(ax2)
        
        # Plot 3: A/B Testing Activity
        self._plot_ab_testing_activity(ax3)
        
        # Plot 4: Model Update Frequency
        self._plot_model_updates(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {save_path}")
        
        return save_path
    
    def _plot_strategy_performance(self, ax):
        """Plot strategy performance over time"""
        # Extract data from metrics history
        timestamps = []
        conservative_cvr = []
        balanced_cvr = []
        aggressive_cvr = []
        
        for metrics in self.metrics_history:
            if 'strategies' in metrics:
                timestamps.append(datetime.fromisoformat(metrics['timestamp']))
                
                strategies = metrics['strategies']
                conservative_cvr.append(strategies.get('conservative', {}).get('expected_conversion_rate', 0))
                balanced_cvr.append(strategies.get('balanced', {}).get('expected_conversion_rate', 0))
                aggressive_cvr.append(strategies.get('aggressive', {}).get('expected_conversion_rate', 0))
        
        if timestamps:
            ax.plot(timestamps, conservative_cvr, label='Conservative', color='green', marker='o')
            ax.plot(timestamps, balanced_cvr, label='Balanced', color='blue', marker='s')  
            ax.plot(timestamps, aggressive_cvr, label='Aggressive', color='red', marker='^')
            
            ax.set_title('Strategy Conversion Rates (Thompson Sampling)')
            ax.set_ylabel('Expected Conversion Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        else:
            ax.text(0.5, 0.5, 'No strategy data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Strategy Conversion Rates')
    
    def _plot_safety_status(self, ax):
        """Plot safety and circuit breaker status"""
        timestamps = []
        circuit_breaker_status = []
        active_experiments = []
        
        for metrics in self.metrics_history:
            timestamps.append(datetime.fromisoformat(metrics['timestamp']))
            circuit_breaker_status.append(1 if metrics.get('circuit_breaker_active', False) else 0)
            active_experiments.append(metrics.get('active_experiments', 0))
        
        if timestamps:
            ax2 = ax.twinx()
            
            # Circuit breaker status (binary)
            ax.fill_between(timestamps, circuit_breaker_status, alpha=0.3, color='red', 
                           label='Circuit Breaker Active')
            ax.set_ylabel('Circuit Breaker Status', color='red')
            ax.set_ylim(-0.1, 1.1)
            
            # Active experiments
            ax2.plot(timestamps, active_experiments, color='blue', marker='o', 
                    label='Active Experiments')
            ax2.set_ylabel('Active Experiments', color='blue')
            
            ax.set_title('Safety Status & A/B Testing Activity')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No safety data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Safety Status')
    
    def _plot_ab_testing_activity(self, ax):
        """Plot A/B testing activity"""
        # Get A/B test data from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT timestamp, variant_id, conversion_rate, sample_size 
            FROM ab_test_results 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''')
        ab_data = cursor.fetchall()
        conn.close()
        
        if ab_data:
            # Group by variant
            variants = defaultdict(list)
            for timestamp, variant, cvr, sample_size in ab_data:
                variants[variant].append((datetime.fromisoformat(timestamp), cvr, sample_size))
            
            for variant, data in variants.items():
                timestamps, cvrs, sizes = zip(*data)
                ax.scatter(timestamps, cvrs, s=[s/5 for s in sizes], label=variant, alpha=0.7)
            
            ax.set_title('A/B Test Results (Bubble Size = Sample Size)')
            ax.set_ylabel('Conversion Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Create mock A/B test data for demo
            timestamps = [datetime.now() - timedelta(hours=i) for i in range(10, 0, -1)]
            control_cvr = np.random.normal(0.02, 0.002, 10)
            treatment_cvr = np.random.normal(0.025, 0.003, 10)
            
            ax.plot(timestamps, control_cvr, label='Control', marker='o', color='blue')
            ax.plot(timestamps, treatment_cvr, label='Treatment', marker='s', color='orange')
            ax.set_title('A/B Test Results (Mock Data)')
            ax.set_ylabel('Conversion Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_model_updates(self, ax):
        """Plot model update frequency"""
        timestamps = []
        update_counts = []
        buffer_sizes = []
        
        for metrics in self.metrics_history:
            timestamps.append(datetime.fromisoformat(metrics['timestamp']))
            update_counts.append(metrics.get('model_update_count', 0))
            buffer_sizes.append(metrics.get('experience_buffer_size', 0))
        
        if timestamps and any(update_counts):
            ax2 = ax.twinx()
            
            # Model updates (cumulative)
            ax.plot(timestamps, update_counts, color='green', marker='o', label='Model Updates')
            ax.set_ylabel('Cumulative Model Updates', color='green')
            
            # Experience buffer size
            ax2.plot(timestamps, buffer_sizes, color='purple', marker='s', alpha=0.7, 
                    label='Buffer Size')
            ax2.set_ylabel('Experience Buffer Size', color='purple')
            
            ax.set_title('Model Update Activity')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No model update data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Update Activity')
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {"error": "No metrics data available"}
        
        # Calculate summary statistics
        latest_metrics = list(self.metrics_history)[-1]
        
        # Strategy performance summary
        strategies = latest_metrics.get('strategies', {})
        strategy_summary = {}
        
        for strategy_id, strategy_data in strategies.items():
            strategy_summary[strategy_id] = {
                'expected_cvr': strategy_data.get('expected_conversion_rate', 0),
                'confidence_interval': strategy_data.get('confidence_interval', [0, 0]),
                'total_trials': strategy_data.get('total_trials', 0),
                'last_updated': strategy_data.get('last_updated', 'unknown')
            }
        
        # Safety status
        safety_summary = {
            'circuit_breaker_active': latest_metrics.get('circuit_breaker_active', False),
            'recent_violations': 0,  # Would calculate from history
            'safety_score': 1.0 if not latest_metrics.get('circuit_breaker_active', False) else 0.5
        }
        
        # A/B testing summary
        ab_testing_summary = {
            'active_experiments': latest_metrics.get('active_experiments', 0),
            'total_experiments_run': 0,  # Would calculate from history
            'significant_results': 0     # Would calculate from database
        }
        
        # Model performance
        model_summary = {
            'total_updates': latest_metrics.get('model_update_count', 0),
            'buffer_size': latest_metrics.get('experience_buffer_size', 0),
            'update_frequency': 'unknown'  # Would calculate from history
        }
        
        # Overall health score
        health_components = [
            1.0 if not safety_summary['circuit_breaker_active'] else 0.0,
            min(1.0, latest_metrics.get('experience_buffer_size', 0) / 1000),  # Buffer health
            min(1.0, sum(s.get('total_trials', 0) for s in strategies.values()) / 100)  # Learning activity
        ]
        overall_health = np.mean(health_components)
        
        report = {
            'timestamp': latest_metrics['timestamp'],
            'overall_health_score': overall_health,
            'strategy_performance': strategy_summary,
            'safety_status': safety_summary,
            'ab_testing_status': ab_testing_summary,
            'model_performance': model_summary,
            'recommendations': self._generate_recommendations(latest_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Circuit breaker check
        if metrics.get('circuit_breaker_active', False):
            recommendations.append("⚠️  Circuit breaker is active - investigate recent poor performance")
        
        # Low activity check
        if metrics.get('experience_buffer_size', 0) < 100:
            recommendations.append("📊 Low experience buffer - consider increasing traffic allocation")
        
        # A/B testing check
        if metrics.get('active_experiments', 0) == 0:
            recommendations.append("🧪 No active A/B tests - consider starting new experiments")
        
        # Model update check
        if metrics.get('model_update_count', 0) == 0:
            recommendations.append("🤖 No model updates yet - ensure sufficient diverse data")
        
        # Strategy performance check
        strategies = metrics.get('strategies', {})
        if strategies:
            best_strategy = max(strategies.items(), 
                               key=lambda x: x[1].get('expected_conversion_rate', 0))
            recommendations.append(f"🎯 Best performing strategy: {best_strategy[0]} "
                                 f"(CVR: {best_strategy[1].get('expected_conversion_rate', 0):.3f})")
        
        if not recommendations:
            recommendations.append("✅ System operating normally - continue monitoring")
        
        return recommendations
    
    async def run_continuous_monitoring(self, duration_hours: int = 24):
        """Run continuous monitoring for specified duration"""
        logger.info(f"Starting continuous monitoring for {duration_hours} hours...")
        
        end_time = time.time() + (duration_hours * 3600)
        
        while time.time() < end_time:
            try:
                # Collect metrics
                metrics = self.collect_current_metrics()
                self.store_metrics(metrics)
                
                # Generate dashboard every 30 minutes
                if len(self.metrics_history) % 30 == 0:
                    dashboard_path = self.create_performance_dashboard()
                    logger.info(f"Updated dashboard: {dashboard_path}")
                
                # Generate report every hour
                if len(self.metrics_history) % 60 == 0:
                    report = self.generate_performance_report()
                    logger.info("Performance Report:")
                    logger.info(f"  Health Score: {report.get('overall_health_score', 0):.2f}")
                    
                    for rec in report.get('recommendations', []):
                        logger.info(f"  {rec}")
                
                # Wait 1 minute
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error
        
        logger.info("Continuous monitoring completed")


async def main():
    """Main monitoring entry point"""
    print("=" * 70)
    print(" ONLINE LEARNING MONITORING SYSTEM ".center(70))
    print("=" * 70)
    print("Options:")
    print("1. Run simulation and create dashboard")
    print("2. Generate performance report")
    print("3. Start continuous monitoring")
    print("0. Exit")
    print("=" * 70)
    
    choice = input("Enter choice (1/2/3/0): ").strip()
    
    # Create monitor
    monitor = OnlineLearningMonitor()
    
    if choice == "1":
        print("\nRunning simulation and creating dashboard...")
        monitor.simulate_online_learning_session(10)  # 10 minute simulation
        
        dashboard_path = monitor.create_performance_dashboard()
        print(f"\nDashboard created: {dashboard_path}")
        
        report = monitor.generate_performance_report()
        print("\nPerformance Report:")
        print(json.dumps(report, indent=2))
        
    elif choice == "2":
        print("\nGenerating performance report...")
        # Need some data first
        if not monitor.metrics_history:
            monitor.simulate_online_learning_session(5)
        
        report = monitor.generate_performance_report()
        print("\nPerformance Report:")
        print(json.dumps(report, indent=2))
        
    elif choice == "3":
        try:
            hours = int(input("Enter monitoring duration in hours (default 2): ") or "2")
        except ValueError:
            hours = 2
        
        print(f"\nStarting {hours}-hour continuous monitoring...")
        print("Press Ctrl+C to stop")
        
        await monitor.run_continuous_monitoring(hours)
        
    elif choice == "0":
        print("Exiting...")
        return
    else:
        print("Invalid choice")
        return await main()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nMonitoring failed: {e}")
        import traceback
        traceback.print_exc()