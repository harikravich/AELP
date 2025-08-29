#!/usr/bin/env python3
"""
REALISTIC GAELP Dashboard - ONLY REAL DATA
Uses only metrics available from actual ad platforms
NO fantasy user tracking or competitor visibility
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import json
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import random
import os
import logging

# Import REALISTIC components only
from realistic_master_integration import RealisticMasterOrchestrator
from realistic_rl_agent import RealisticState

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticDashboard:
    """Dashboard showing ONLY real ad platform metrics"""
    
    def __init__(self):
        # Daily budget (realistic)
        self.daily_budget = 1000.0  # $1k/day to start
        self.today_spend = 0.0
        self.last_reset = datetime.now().date()
        
        # Initialize REALISTIC orchestrator
        self.orchestrator = None
        self.is_running = False
        self.episode_count = 0
        
        # REAL metrics from ad platforms
        self.metrics = {
            # Platform metrics (what you ACTUALLY get)
            'total_impressions': 0,
            'total_clicks': 0,
            'total_spend': 0.0,
            'avg_cpc': 0.0,
            'avg_cpm': 0.0,
            'ctr': 0.0,
            
            # Conversion metrics (YOUR tracking)
            'conversions': 0,
            'conversion_value': 0.0,
            'cvr': 0.0,
            'cpa': 0.0,
            'roas': 0.0,
            
            # Platform-specific (REAL)
            'google_impressions': 0,
            'google_clicks': 0,
            'google_spend': 0.0,
            'facebook_impressions': 0,
            'facebook_clicks': 0,
            'facebook_spend': 0.0,
            
            # NO FANTASY METRICS:
            # - No user journey stages
            # - No competitor bids
            # - No touchpoint tracking
            # - No mental states
        }
        
        # Time series for charts (REAL data only)
        self.time_series = {
            'timestamps': deque(maxlen=100),
            'impressions': deque(maxlen=100),
            'clicks': deque(maxlen=100),
            'spend': deque(maxlen=100),
            'conversions': deque(maxlen=100),
            'ctr': deque(maxlen=100),
            'cpc': deque(maxlen=100),
            'roas': deque(maxlen=100)
        }
        
        # Keyword performance (Google only, REAL)
        self.keyword_performance = defaultdict(lambda: {
            'impressions': 0,
            'clicks': 0,
            'spend': 0.0,
            'conversions': 0,
            'avg_position': 0.0
        })
        
        # Creative performance (REAL)
        self.creative_performance = defaultdict(lambda: {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'ctr': 0.0
        })
        
        # Hourly performance (REAL pattern detection)
        self.hourly_performance = defaultdict(lambda: {
            'impressions': 0,
            'clicks': 0,
            'spend': 0.0,
            'conversions': 0
        })
        
        # RL Agent learning metrics (REAL)
        self.learning_metrics = {
            'epsilon': 1.0,
            'training_steps': 0,
            'avg_reward': 0.0,
            'win_rate': 0.0
        }
        
        # Event log for UI
        self.event_log = deque(maxlen=50)
        
        logger.info("Initialized REALISTIC Dashboard with real metrics only")
    
    def start_simulation(self):
        """Start the REALISTIC simulation"""
        if not self.is_running:
            self.is_running = True
            
            self.log_event("🚀 Starting REALISTIC simulation...", "system")
            self.log_event("📊 Using ONLY real ad platform data", "system")
            
            # Initialize realistic orchestrator
            self.orchestrator = RealisticMasterOrchestrator(daily_budget=self.daily_budget)
            
            self.log_event("✅ Realistic orchestrator initialized", "system")
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(target=self.run_simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            self.log_event("🎯 Simulation started - bidding on real keywords!", "system")
    
    def run_simulation_loop(self):
        """Main loop running REALISTIC simulation"""
        step_count = 0
        
        while self.is_running:
            try:
                # Run one realistic step
                result = self.orchestrator.step()
                step_count += 1
                
                # Update dashboard with REAL data
                self.update_from_realistic_step(result)
                
                # Log significant events
                if step_count % 10 == 0:
                    self.log_event(
                        f"Step {step_count}: {self.metrics['total_impressions']} impressions, "
                        f"${self.metrics['total_spend']:.2f} spent",
                        "info"
                    )
                
                # Control simulation speed
                time.sleep(0.1)  # 10 steps per second
                
            except Exception as e:
                self.log_event(f"❌ Error: {str(e)}", "error")
                logger.error(f"Simulation error: {e}", exc_info=True)
                time.sleep(1)
    
    def update_from_realistic_step(self, result: dict):
        """Update dashboard with REAL step results"""
        
        # Extract REAL metrics
        step_data = result['step_result']
        campaign_metrics = result['campaign_metrics']
        platform_metrics = result['platform_metrics']
        learning = result['learning']
        
        # Update totals (REAL)
        self.metrics['total_impressions'] = campaign_metrics['total_impressions']
        self.metrics['total_clicks'] = campaign_metrics['total_clicks']
        self.metrics['total_spend'] = campaign_metrics['total_spend']
        self.metrics['conversions'] = campaign_metrics['total_conversions']
        self.metrics['conversion_value'] = campaign_metrics['total_revenue']
        
        # Calculate rates (REAL)
        if self.metrics['total_impressions'] > 0:
            self.metrics['ctr'] = self.metrics['total_clicks'] / self.metrics['total_impressions']
        
        if self.metrics['total_clicks'] > 0:
            self.metrics['cvr'] = self.metrics['conversions'] / self.metrics['total_clicks']
            self.metrics['avg_cpc'] = self.metrics['total_spend'] / self.metrics['total_clicks']
        
        if self.metrics['conversions'] > 0:
            self.metrics['cpa'] = self.metrics['total_spend'] / self.metrics['conversions']
        
        if self.metrics['total_spend'] > 0:
            self.metrics['roas'] = self.metrics['conversion_value'] / self.metrics['total_spend']
        
        # Update platform metrics (REAL)
        for platform in ['google', 'facebook', 'tiktok']:
            if platform in platform_metrics:
                data = platform_metrics[platform]
                self.metrics[f'{platform}_impressions'] = data['impressions']
                self.metrics[f'{platform}_clicks'] = data['clicks']
                self.metrics[f'{platform}_spend'] = data['spend']
        
        # Update learning metrics (REAL)
        self.learning_metrics['epsilon'] = learning['epsilon']
        self.learning_metrics['training_steps'] = learning['training_steps']
        
        # Update time series (REAL)
        now = datetime.now().timestamp()
        self.time_series['timestamps'].append(now)
        self.time_series['impressions'].append(self.metrics['total_impressions'])
        self.time_series['clicks'].append(self.metrics['total_clicks'])
        self.time_series['spend'].append(self.metrics['total_spend'])
        self.time_series['conversions'].append(self.metrics['conversions'])
        self.time_series['ctr'].append(self.metrics['ctr'] * 100)  # As percentage
        self.time_series['cpc'].append(self.metrics['avg_cpc'])
        self.time_series['roas'].append(self.metrics['roas'])
        
        # Log significant events (REAL)
        if step_data['won']:
            platform = step_data['platform']
            price = step_data['price_paid']
            
            if step_data['clicked']:
                self.log_event(
                    f"✅ {platform.upper()}: Won & clicked! Paid ${price:.2f}",
                    "success"
                )
        
        if campaign_metrics['total_conversions'] > self.metrics.get('last_conversions', 0):
            self.log_event(
                f"💰 CONVERSION! Total: {campaign_metrics['total_conversions']}",
                "conversion"
            )
            self.metrics['last_conversions'] = campaign_metrics['total_conversions']
    
    def log_event(self, message: str, event_type: str = "info"):
        """Log event for dashboard display"""
        self.event_log.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'type': event_type
        })
    
    def get_dashboard_data(self):
        """Get REAL dashboard data for display"""
        
        # Calculate additional metrics
        win_rate = 0
        if self.orchestrator and self.orchestrator.environment:
            total_steps = self.orchestrator.environment.current_step
            if total_steps > 0:
                win_rate = self.metrics['total_impressions'] / total_steps
        
        return {
            # Summary metrics (REAL)
            'metrics': {
                'impressions': self.metrics['total_impressions'],
                'clicks': self.metrics['total_clicks'],
                'conversions': self.metrics['conversions'],
                'spend': round(self.metrics['total_spend'], 2),
                'revenue': round(self.metrics['conversion_value'], 2),
                'ctr': round(self.metrics['ctr'] * 100, 2),
                'cvr': round(self.metrics['cvr'] * 100, 2),
                'cpa': round(self.metrics['cpa'], 2) if self.metrics['conversions'] > 0 else 0,
                'roas': round(self.metrics['roas'], 2),
                'win_rate': round(win_rate * 100, 1)
            },
            
            # Platform breakdown (REAL)
            'platforms': {
                'google': {
                    'impressions': self.metrics.get('google_impressions', 0),
                    'clicks': self.metrics.get('google_clicks', 0),
                    'spend': round(self.metrics.get('google_spend', 0), 2)
                },
                'facebook': {
                    'impressions': self.metrics.get('facebook_impressions', 0),
                    'clicks': self.metrics.get('facebook_clicks', 0),
                    'spend': round(self.metrics.get('facebook_spend', 0), 2)
                },
                'tiktok': {
                    'impressions': self.metrics.get('tiktok_impressions', 0),
                    'clicks': self.metrics.get('tiktok_clicks', 0),
                    'spend': round(self.metrics.get('tiktok_spend', 0), 2)
                }
            },
            
            # Time series (REAL)
            'time_series': {
                'timestamps': list(self.time_series['timestamps']),
                'impressions': list(self.time_series['impressions']),
                'clicks': list(self.time_series['clicks']),
                'spend': list(self.time_series['spend']),
                'conversions': list(self.time_series['conversions']),
                'ctr': list(self.time_series['ctr']),
                'cpc': list(self.time_series['cpc']),
                'roas': list(self.time_series['roas'])
            },
            
            # Learning progress (REAL)
            'learning': {
                'epsilon': round(self.learning_metrics['epsilon'], 3),
                'training_steps': self.learning_metrics['training_steps'],
                'exploration_rate': round(self.learning_metrics['epsilon'] * 100, 1)
            },
            
            # Events
            'events': list(self.event_log)[-20:],  # Last 20 events
            
            # Status
            'status': {
                'is_running': self.is_running,
                'daily_budget': self.daily_budget,
                'budget_spent': round(self.metrics['total_spend'], 2),
                'budget_remaining': round(self.daily_budget - self.metrics['total_spend'], 2)
            }
        }
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.log_event("🛑 Simulation stopped", "system")
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.__init__()
        self.log_event("🔄 Metrics reset", "system")


# Global dashboard instance
dashboard = RealisticDashboard()

@app.route('/')
def index():
    """Serve the dashboard HTML"""
    return render_template('realistic_dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current dashboard status"""
    return jsonify(dashboard.get_dashboard_data())

@app.route('/api/start', methods=['POST'])
def start_simulation():
    """Start the realistic simulation"""
    dashboard.start_simulation()
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop the simulation"""
    dashboard.stop_simulation()
    return jsonify({'status': 'stopped'})

@app.route('/api/reset', methods=['POST'])
def reset_metrics():
    """Reset all metrics"""
    dashboard.reset_metrics()
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    print("="*60)
    print("REALISTIC GAELP DASHBOARD")
    print("Using ONLY Real Ad Platform Data")
    print("="*60)
    print()
    print("NO FANTASY DATA:")
    print("❌ No user journey tracking")
    print("❌ No competitor bid visibility")
    print("❌ No mental state detection")
    print("❌ No cross-platform user tracking")
    print()
    print("REAL DATA ONLY:")
    print("✅ Platform metrics (impressions, clicks, CPC)")
    print("✅ Your campaign performance")
    print("✅ Post-click tracking on YOUR site")
    print("✅ Conversion tracking with attribution windows")
    print()
    print("="*60)
    print("Starting server at http://localhost:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', debug=True, port=5000)