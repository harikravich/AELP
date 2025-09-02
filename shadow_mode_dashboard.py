#!/usr/bin/env python3
"""
SHADOW MODE DASHBOARD
Real-time monitoring dashboard for shadow mode testing
"""

import asyncio
import logging
import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import defaultdict, deque
import pandas as pd

logger = logging.getLogger(__name__)

class ShadowModeDashboard:
    """
    Real-time dashboard for monitoring shadow mode testing
    """
    
    def __init__(self, db_path: str, update_interval: int = 30):
        self.db_path = db_path
        self.update_interval = update_interval
        
        # Data storage
        self.metrics_history = defaultdict(lambda: defaultdict(list))
        self.comparison_data = deque(maxlen=10000)
        self.alert_log = deque(maxlen=1000)
        
        # Dashboard state
        self.last_update = datetime.now()
        self.is_running = False
        
        # Alert thresholds
        self.alert_thresholds = {
            'bid_divergence_high': 0.5,
            'performance_degradation': 0.2,
            'risk_score_high': 0.8,
            'win_rate_low': 0.1,
            'roas_low': 0.5
        }
        
        # Visualization settings
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('GAELP Shadow Mode Testing Dashboard', fontsize=16, fontweight='bold')
        
        logger.info(f"Shadow Mode Dashboard initialized for database: {db_path}")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        logger.info("Starting shadow mode dashboard...")
        self.is_running = True
        
        # Set up animated plotting
        self.animation = FuncAnimation(
            self.fig, self.update_dashboard, interval=self.update_interval * 1000,
            blit=False, cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        logger.info("Stopping shadow mode dashboard...")
        self.is_running = False
    
    def update_dashboard(self, frame):
        """Update dashboard with latest data"""
        try:
            # Load latest data
            self._load_data_from_database()
            
            # Clear all axes
            for ax_row in self.axes:
                for ax in ax_row:
                    ax.clear()
            
            # Plot various metrics
            self._plot_performance_comparison()
            self._plot_bid_divergence_trends()
            self._plot_conversion_rates()
            self._plot_roas_comparison()
            self._plot_risk_metrics()
            self._plot_decision_volume()
            
            # Update title with timestamp
            self.fig.suptitle(
                f'GAELP Shadow Mode Testing Dashboard - Updated: {datetime.now().strftime("%H:%M:%S")}',
                fontsize=16, fontweight='bold'
            )
            
            # Adjust layout
            plt.tight_layout()
            
            # Check for alerts
            self._check_alerts()
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
    
    def _load_data_from_database(self):
        """Load latest data from database"""
        if not Path(self.db_path).exists():
            return
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Load decisions data
            decisions_df = pd.read_sql_query('''
                SELECT * FROM decisions 
                WHERE timestamp > datetime('now', '-1 hour')
                ORDER BY timestamp DESC
                LIMIT 10000
            ''', conn)
            
            # Load comparisons data
            comparisons_df = pd.read_sql_query('''
                SELECT * FROM comparisons
                WHERE timestamp > datetime('now', '-1 hour')
                ORDER BY timestamp DESC
                LIMIT 5000
            ''', conn)
            
            # Load metrics snapshots
            metrics_df = pd.read_sql_query('''
                SELECT * FROM metrics_snapshots
                WHERE timestamp > datetime('now', '-2 hours')
                ORDER BY timestamp ASC
            ''', conn)
            
            # Process data
            self._process_decisions_data(decisions_df)
            self._process_comparisons_data(comparisons_df)
            self._process_metrics_data(metrics_df)
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
        finally:
            conn.close()
    
    def _process_decisions_data(self, df):
        """Process decisions data for visualization"""
        if df.empty:
            return
        
        # Group by model
        for model_id in df['model_id'].unique():
            model_data = df[df['model_id'] == model_id]
            
            # Calculate metrics
            total_decisions = len(model_data)
            won_auctions = model_data['won_auction'].sum()
            clicks = model_data['clicked'].sum()
            conversions = model_data['converted'].sum()
            total_spend = model_data['spend'].sum()
            total_revenue = model_data['revenue'].sum()
            
            # Store metrics
            timestamp = datetime.now()
            self.metrics_history[model_id]['timestamps'].append(timestamp)
            self.metrics_history[model_id]['decisions'].append(total_decisions)
            self.metrics_history[model_id]['win_rate'].append(won_auctions / max(1, total_decisions))
            self.metrics_history[model_id]['ctr'].append(clicks / max(1, won_auctions))
            self.metrics_history[model_id]['cvr'].append(conversions / max(1, clicks))
            self.metrics_history[model_id]['roas'].append(total_revenue / max(0.01, total_spend))
            self.metrics_history[model_id]['avg_bid'].append(model_data['bid_amount'].mean())
            
            # Keep only last 100 points for visualization
            for key in self.metrics_history[model_id]:
                if len(self.metrics_history[model_id][key]) > 100:
                    self.metrics_history[model_id][key] = self.metrics_history[model_id][key][-100:]
    
    def _process_comparisons_data(self, df):
        """Process comparisons data"""
        if df.empty:
            return
        
        # Store comparison data
        self.comparison_data.clear()
        
        for _, row in df.iterrows():
            self.comparison_data.append({
                'timestamp': datetime.fromisoformat(row['timestamp']),
                'bid_divergence': row['bid_divergence'],
                'creative_divergence': row['creative_divergence'],
                'channel_divergence': row['channel_divergence'],
                'significant_divergence': row['significant_divergence'],
                'production_value': row['production_value'],
                'shadow_value': row['shadow_value']
            })
    
    def _process_metrics_data(self, df):
        """Process metrics snapshots"""
        if df.empty:
            return
        
        # Update historical metrics from snapshots
        for _, row in df.iterrows():
            model_id = row['model_id']
            timestamp = datetime.fromisoformat(row['timestamp'])
            
            # Store in history if not already present
            if not self.metrics_history[model_id]['timestamps'] or \
               timestamp > self.metrics_history[model_id]['timestamps'][-1]:
                
                self.metrics_history[model_id]['timestamps'].append(timestamp)
                self.metrics_history[model_id]['win_rate'].append(row['win_rate'])
                self.metrics_history[model_id]['ctr'].append(row['ctr'])
                self.metrics_history[model_id]['cvr'].append(row['cvr'])
                self.metrics_history[model_id]['roas'].append(row['roas'])
                self.metrics_history[model_id]['avg_bid'].append(row['avg_bid'])
    
    def _plot_performance_comparison(self):
        """Plot performance comparison between models"""
        ax = self.axes[0, 0]
        
        models = list(self.metrics_history.keys())
        if not models:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Comparison')
            return
        
        # Get latest metrics for each model
        metrics = ['win_rate', 'ctr', 'cvr', 'roas']
        model_values = {}
        
        for model in models:
            model_values[model] = []
            for metric in metrics:
                if self.metrics_history[model][metric]:
                    model_values[model].append(self.metrics_history[model][metric][-1])
                else:
                    model_values[model].append(0)
        
        # Create bar chart
        x = np.arange(len(metrics))
        width = 0.35
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, (model, values) in enumerate(model_values.items()):
            offset = (i - len(models)/2 + 0.5) * width / len(models)
            bars = ax.bar(x + offset, values, width/len(models), 
                         label=model, alpha=0.8, color=colors[i])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Performance Comparison (Latest)')
        ax.set_xticks(x)
        ax.set_xticklabels(['Win Rate', 'CTR', 'CVR', 'ROAS'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_bid_divergence_trends(self):
        """Plot bid divergence trends over time"""
        ax = self.axes[0, 1]
        
        if not self.comparison_data:
            ax.text(0.5, 0.5, 'No comparison data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Bid Divergence Trends')
            return
        
        # Extract data
        timestamps = [c['timestamp'] for c in self.comparison_data]
        divergences = [c['bid_divergence'] for c in self.comparison_data]
        significant = [c['significant_divergence'] for c in self.comparison_data]
        
        # Plot all divergences
        ax.plot(timestamps, divergences, alpha=0.6, linewidth=1, label='Bid Divergence')
        
        # Highlight significant divergences
        sig_timestamps = [t for t, s in zip(timestamps, significant) if s]
        sig_divergences = [d for d, s in zip(divergences, significant) if s]
        
        if sig_timestamps:
            ax.scatter(sig_timestamps, sig_divergences, color='red', alpha=0.8, 
                      s=20, label='Significant Divergence')
        
        # Add threshold line
        if timestamps:
            ax.axhline(y=self.alert_thresholds['bid_divergence_high'], 
                      color='red', linestyle='--', alpha=0.7, label='Alert Threshold')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Bid Divergence')
        ax.set_title('Bid Divergence Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(timestamps) > 1:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_conversion_rates(self):
        """Plot conversion rate trends"""
        ax = self.axes[0, 2]
        
        if not self.metrics_history:
            ax.text(0.5, 0.5, 'No metrics data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Conversion Rate Trends')
            return
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.metrics_history)))
        
        for i, (model_id, data) in enumerate(self.metrics_history.items()):
            if data['timestamps'] and data['cvr']:
                ax.plot(data['timestamps'], data['cvr'], 
                       label=f'{model_id} CVR', color=colors[i], linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Conversion Rate')
        ax.set_title('Conversion Rate Trends')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if any(data['timestamps'] for data in self.metrics_history.values()):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_roas_comparison(self):
        """Plot ROAS comparison"""
        ax = self.axes[1, 0]
        
        if not self.metrics_history:
            ax.text(0.5, 0.5, 'No ROAS data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('ROAS Comparison')
            return
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.metrics_history)))
        
        for i, (model_id, data) in enumerate(self.metrics_history.items()):
            if data['timestamps'] and data['roas']:
                ax.plot(data['timestamps'], data['roas'], 
                       label=f'{model_id} ROAS', color=colors[i], linewidth=2, marker='o', markersize=3)
        
        # Add target ROAS line
        if any(data['timestamps'] for data in self.metrics_history.values()):
            ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Target ROAS (2.0x)')
            ax.axhline(y=self.alert_thresholds['roas_low'], color='red', linestyle='--', alpha=0.7, label='Alert Threshold')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('ROAS (Return on Ad Spend)')
        ax.set_title('ROAS Trends')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if any(data['timestamps'] for data in self.metrics_history.values()):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_risk_metrics(self):
        """Plot risk metrics"""
        ax = self.axes[1, 1]
        
        if not self.metrics_history:
            ax.text(0.5, 0.5, 'No risk data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk Metrics')
            return
        
        # Calculate risk metrics
        risk_data = {}
        for model_id, data in self.metrics_history.items():
            if data['avg_bid']:
                # Bid volatility as risk indicator
                recent_bids = data['avg_bid'][-20:] if len(data['avg_bid']) >= 20 else data['avg_bid']
                if len(recent_bids) > 1:
                    volatility = np.std(recent_bids) / max(0.01, np.mean(recent_bids))
                    risk_data[model_id] = volatility
        
        if risk_data:
            models = list(risk_data.keys())
            risks = list(risk_data.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            
            bars = ax.bar(models, risks, color=colors, alpha=0.8)
            
            # Add value labels
            for bar, risk in zip(bars, risks):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{risk:.3f}', ha='center', va='bottom')
            
            # Add threshold line
            ax.axhline(y=self.alert_thresholds['risk_score_high'], 
                      color='red', linestyle='--', alpha=0.7, label='Alert Threshold')
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Risk Score (Bid Volatility)')
            ax.set_title('Risk Metrics by Model')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_decision_volume(self):
        """Plot decision volume over time"""
        ax = self.axes[1, 2]
        
        if not self.metrics_history:
            ax.text(0.5, 0.5, 'No volume data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Decision Volume')
            return
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.metrics_history)))
        
        for i, (model_id, data) in enumerate(self.metrics_history.items()):
            if data['timestamps'] and data['decisions']:
                # Calculate decisions per minute
                decisions_per_min = []
                timestamps = []
                
                for j in range(1, len(data['decisions'])):
                    time_diff = (data['timestamps'][j] - data['timestamps'][j-1]).total_seconds() / 60
                    if time_diff > 0:
                        decision_rate = (data['decisions'][j] - data['decisions'][j-1]) / time_diff
                        decisions_per_min.append(decision_rate)
                        timestamps.append(data['timestamps'][j])
                
                if timestamps:
                    ax.plot(timestamps, decisions_per_min, 
                           label=f'{model_id}', color=colors[i], linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Decisions per Minute')
        ax.set_title('Decision Volume Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if any(data['timestamps'] for data in self.metrics_history.values()):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _check_alerts(self):
        """Check for alert conditions"""
        current_time = datetime.now()
        
        # Check each model for alert conditions
        for model_id, data in self.metrics_history.items():
            if not data or not data['timestamps']:
                continue
            
            latest_values = {key: values[-1] if values else 0 
                           for key, values in data.items() if key != 'timestamps'}
            
            # Win rate alert
            if latest_values.get('win_rate', 0) < self.alert_thresholds['win_rate_low']:
                self._create_alert(f"Low win rate for {model_id}: {latest_values['win_rate']:.3f}")
            
            # ROAS alert
            if latest_values.get('roas', 0) < self.alert_thresholds['roas_low']:
                self._create_alert(f"Low ROAS for {model_id}: {latest_values['roas']:.2f}x")
            
            # Bid volatility alert
            if len(data.get('avg_bid', [])) >= 10:
                recent_bids = data['avg_bid'][-10:]
                volatility = np.std(recent_bids) / max(0.01, np.mean(recent_bids))
                if volatility > self.alert_thresholds['risk_score_high']:
                    self._create_alert(f"High bid volatility for {model_id}: {volatility:.3f}")
        
        # Check comparison alerts
        if self.comparison_data:
            recent_comparisons = list(self.comparison_data)[-50:]  # Last 50 comparisons
            high_divergences = [c for c in recent_comparisons 
                              if c['bid_divergence'] > self.alert_thresholds['bid_divergence_high']]
            
            if len(high_divergences) > len(recent_comparisons) * 0.3:  # More than 30%
                self._create_alert(f"High bid divergence rate: {len(high_divergences)}/{len(recent_comparisons)} comparisons")
    
    def _create_alert(self, message: str):
        """Create an alert"""
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'severity': 'warning'
        }
        
        self.alert_log.append(alert)
        logger.warning(f"SHADOW MODE ALERT: {message}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_monitored': list(self.metrics_history.keys()),
            'monitoring_duration_minutes': (datetime.now() - self.last_update).total_seconds() / 60,
            'total_alerts': len(self.alert_log),
            'model_summaries': {}
        }
        
        # Model summaries
        for model_id, data in self.metrics_history.items():
            if data['timestamps']:
                summary = {}
                for metric in ['win_rate', 'ctr', 'cvr', 'roas', 'avg_bid']:
                    if data[metric]:
                        values = data[metric]
                        summary[metric] = {
                            'current': values[-1],
                            'average': np.mean(values),
                            'trend': 'up' if len(values) > 1 and values[-1] > values[-2] else 'down'
                        }
                
                report['model_summaries'][model_id] = summary
        
        # Comparison summary
        if self.comparison_data:
            recent_comparisons = list(self.comparison_data)[-100:]
            significant_count = sum(1 for c in recent_comparisons if c['significant_divergence'])
            
            report['comparison_summary'] = {
                'total_comparisons': len(recent_comparisons),
                'significant_divergences': significant_count,
                'significant_rate': significant_count / len(recent_comparisons) if recent_comparisons else 0,
                'avg_bid_divergence': np.mean([c['bid_divergence'] for c in recent_comparisons])
            }
        
        return report
    
    def save_dashboard_snapshot(self, filename: Optional[str] = None):
        """Save current dashboard as image"""
        if filename is None:
            filename = f"shadow_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard snapshot saved: {filename}")
        
        return filename

def create_dashboard_from_database(db_path: str):
    """Create and start dashboard from existing database"""
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    dashboard = ShadowModeDashboard(db_path)
    
    try:
        dashboard.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
        dashboard.stop_monitoring()
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")

if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Shadow Mode Testing Dashboard')
    parser.add_argument('--db-path', required=True, help='Path to shadow testing database')
    parser.add_argument('--update-interval', type=int, default=30, help='Update interval in seconds')
    parser.add_argument('--save-snapshot', action='store_true', help='Save dashboard snapshot and exit')
    
    args = parser.parse_args()
    
    if args.save_snapshot:
        # Just save snapshot without starting interactive mode
        dashboard = ShadowModeDashboard(args.db_path, args.update_interval)
        dashboard._load_data_from_database()
        dashboard.update_dashboard(0)
        filename = dashboard.save_dashboard_snapshot()
        print(f"Dashboard snapshot saved: {filename}")
    else:
        # Start interactive dashboard
        create_dashboard_from_database(args.db_path)