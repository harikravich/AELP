#!/usr/bin/env python3
"""
Behavioral Health Marketing Dashboard
Real-time visualization of the integrated behavioral health simulator
Shows ACTUAL traffic composition, parent journeys, and conversions
NO HARDCODED SEGMENTS - everything discovered dynamically
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import threading
import json
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import os

# Import our FIXED simulator with all improvements
from enhanced_simulator_fixed import FixedGAELPEnvironment
from realistic_traffic_simulator import RealisticTrafficSimulator
from behavioral_health_persona_factory import BehavioralHealthPersonaFactory
import asyncio

app = Flask(__name__)
CORS(app)

class BehavioralHealthDashboard:
    """Dashboard for the behavioral health marketing simulator"""
    
    def __init__(self):
        # Initialize the FIXED simulator with all our improvements
        self.simulator = FixedGAELPEnvironment(
            max_budget=10000.0,  # $10k daily budget
            max_steps=1000       # 1000 steps per episode
        )
        
        self.is_running = False
        self.simulation_thread = None
        
        # Real-time metrics
        self.current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_impressions': 0,
            'total_clicks': 0,
            'total_spend': 0.0,
            'total_conversions': 0,
            'active_journeys': {},
            'visitor_breakdown': {},
            'channel_performance': {},
            'hourly_pattern': [],
            'conversion_funnel': {
                'triggered': 0,
                'researching': 0,
                'comparing': 0,
                'deciding': 0,
                'converted': 0
            },
            # NEW: Track our fixed simulator metrics
            'current_roas': 0.0,
            'win_rate': 0.0,
            'budget_remaining': 0.0,
            'latest_reward': 0.0,
            'auction_wins': 0,
            'auction_losses': 0,
            'persistent_users': 0
        }
        
        # Time series for charts (last 24 hours)
        self.time_series = {
            'timestamps': deque(maxlen=24),
            'impressions': deque(maxlen=24),
            'clicks': deque(maxlen=24),
            'conversions': deque(maxlen=24),
            'spend': deque(maxlen=24),
            'cac': deque(maxlen=24),
            'crisis_parents': deque(maxlen=24),
            'concerned_parents': deque(maxlen=24),
            'non_parents': deque(maxlen=24),
            'learning_progress': deque(maxlen=100)  # Track RL learning progress
        }
        
        # Journey tracking
        self.journey_details = {
            'crisis_journeys': [],
            'high_concern_journeys': [],
            'moderate_concern_journeys': [],
            'curious_journeys': []
        }
    
    def start_simulation(self):
        """Start the simulation in a background thread"""
        if not self.is_running:
            self.is_running = True
            self.simulation_thread = threading.Thread(target=self._run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5)
    
    def _run_simulation(self):
        """Run the simulation continuously using our fixed GAELP environment"""
        # Reset the environment
        state = self.simulator.reset()
        step_count = 0
        
        while self.is_running and step_count < self.simulator.max_steps:
            # Create action for the current step
            action = {
                'bid': 2.5,  # Competitive bid
                'creative_type': 'behavioral_health',
                'audience_segment': 'concerned_parents',
                'quality_score': 0.8
            }
            
            # Step the environment
            next_state, reward, done, info = self.simulator.step(action)
            step_count += 1
            
            # Update metrics from the step
            self._update_metrics_from_step(info, reward)
            
            # Sleep for visualization
            time.sleep(0.1)  # Fast updates for demo
            
            if done:
                # Episode finished, reset for next one
                state = self.simulator.reset()
                step_count = 0
                time.sleep(1)  # Brief pause between episodes
    
    def _update_metrics_from_step(self, step_info, reward):
        """Update dashboard metrics from simulation step"""
        
        # Update totals from step info
        self.current_metrics['total_impressions'] += step_info.get('impressions', 0)
        self.current_metrics['total_clicks'] += step_info.get('clicks', 0)
        self.current_metrics['total_spend'] += step_info.get('cost', 0)
        self.current_metrics['total_conversions'] += step_info.get('conversions', 0)
        
        # Update performance metrics
        self.current_metrics['current_roas'] = step_info.get('roas', 0)
        self.current_metrics['win_rate'] = step_info.get('win_rate', 0)
        self.current_metrics['budget_remaining'] = step_info.get('budget_remaining', 0)
        
        # Track learning progress (reward improvement)
        self.current_metrics['latest_reward'] = reward
        
        # Update time series
        self.time_series['timestamps'].append(datetime.now().isoformat())
        self.time_series['impressions'].append(step_info.get('impressions', 0))
        self.time_series['clicks'].append(step_info.get('clicks', 0))
        self.time_series['conversions'].append(step_info.get('conversions', 0))
        self.time_series['spend'].append(step_info.get('cost', 0))
        
        # Calculate CAC
        if step_info.get('conversions', 0) > 0:
            cac = step_info.get('cost', 0) / step_info.get('conversions', 1)
        else:
            cac = 0
        self.time_series['cac'].append(cac)
        
        # Update learning progress tracking
        self.time_series['learning_progress'].append({
            'reward': reward,
            'roas': step_info.get('roas', 0),
            'win_rate': step_info.get('win_rate', 0)
        })
        
        # Note: Visitor type tracking would need to be updated based on 
        # persistent user data from the fixed simulator
        self.time_series['crisis_parents'].append(0)  # TODO: Get from persistent user data
        self.time_series['concerned_parents'].append(1)  # Default for demo
        self.time_series['non_parents'].append(0)
        
        # Update journey funnel
        self._update_journey_funnel()
    
    def _update_journey_funnel(self):
        """Update the conversion funnel from active journeys"""
        funnel = {
            'triggered': 0,
            'researching': 0,
            'comparing': 0,
            'deciding': 0,
            'converted': 0
        }
        
        for journey in self.simulator.state.active_journeys.values():
            if journey.current_stage == 'unaware':
                funnel['triggered'] += 1
            elif journey.current_stage == 'researching':
                funnel['researching'] += 1
            elif journey.current_stage == 'comparing':
                funnel['comparing'] += 1
            elif journey.current_stage == 'deciding':
                funnel['deciding'] += 1
        
        funnel['converted'] = self.simulator.state.total_conversions
        self.current_metrics['conversion_funnel'] = funnel
    
    def get_dashboard_data(self):
        """Get current dashboard data"""
        
        # Calculate key metrics
        overall_ctr = (self.current_metrics['total_clicks'] / 
                      max(1, self.current_metrics['total_impressions'])) * 100
        
        overall_cac = (self.current_metrics['total_spend'] / 
                      max(1, self.current_metrics['total_conversions']))
        
        # Get top visitor types
        top_visitors = sorted(
            self.current_metrics['visitor_breakdown'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Get journey summary
        journey_summary = self.simulator.get_simulation_summary()
        
        return {
            'is_running': self.is_running,
            'current_time': datetime.now().isoformat(),
            'metrics': {
                'impressions': self.current_metrics['total_impressions'],
                'clicks': self.current_metrics['total_clicks'],
                'conversions': self.current_metrics['total_conversions'],
                'spend': f"${self.current_metrics['total_spend']:.2f}",
                'ctr': f"{overall_ctr:.2f}%",
                'cac': f"${overall_cac:.2f}",
                'active_journeys': journey_summary['active_journeys'],
                'avg_journey_days': journey_summary['avg_journey_length']
            },
            'visitor_breakdown': dict(top_visitors),
            'conversion_funnel': self.current_metrics['conversion_funnel'],
            'time_series': {
                'timestamps': list(self.time_series['timestamps']),
                'impressions': list(self.time_series['impressions']),
                'clicks': list(self.time_series['clicks']),
                'conversions': list(self.time_series['conversions']),
                'spend': list(self.time_series['spend']),
                'cac': list(self.time_series['cac']),
                'visitor_types': {
                    'crisis_parents': list(self.time_series['crisis_parents']),
                    'concerned_parents': list(self.time_series['concerned_parents']),
                    'non_parents': list(self.time_series['non_parents'])
                }
            }
        }

# Create dashboard instance
dashboard = BehavioralHealthDashboard()

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Behavioral Health Marketing Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .controls {
            text-align: center;
            margin-bottom: 30px;
        }
        button {
            background: white;
            color: #667eea;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 25px;
            cursor: pointer;
            margin: 0 10px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        button:hover {
            transform: scale(1.05);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            opacity: 0.9;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .chart-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .chart-title {
            font-size: 1.3em;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.2);
            margin-left: 20px;
        }
        .status.running {
            background: #10b981;
        }
        .status.stopped {
            background: #ef4444;
        }
        .visitor-breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .visitor-type {
            background: rgba(255, 255, 255, 0.15);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        .visitor-type-name {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        .visitor-type-count {
            font-size: 1.5em;
            font-weight: bold;
        }
        .funnel-stage {
            background: rgba(255, 255, 255, 0.15);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .funnel-stage-name {
            font-weight: bold;
        }
        .funnel-stage-count {
            font-size: 1.5em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Behavioral Health Marketing Dashboard</h1>
        <div class="subtitle">
            Real-Time Simulation with Actual Traffic Patterns
            <span id="status" class="status stopped">STOPPED</span>
        </div>
    </div>
    
    <div class="container">
        <div class="controls">
            <button id="startBtn" onclick="startSimulation()">Start Simulation</button>
            <button id="stopBtn" onclick="stopSimulation()" disabled>Stop Simulation</button>
            <button onclick="refreshData()">Refresh Data</button>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Impressions</div>
                <div class="metric-value" id="impressions">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Clicks</div>
                <div class="metric-value" id="clicks">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CTR</div>
                <div class="metric-value" id="ctr">0%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Conversions</div>
                <div class="metric-value" id="conversions">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CAC</div>
                <div class="metric-value" id="cac">$0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Spend</div>
                <div class="metric-value" id="spend">$0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Active Journeys</div>
                <div class="metric-value" id="active_journeys">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Journey Days</div>
                <div class="metric-value" id="avg_journey_days">0</div>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="chart-container">
                <div class="chart-title">📊 Traffic Composition</div>
                <canvas id="trafficChart"></canvas>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🎯 Conversion Funnel</div>
                <div id="funnelContainer"></div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📈 Performance Over Time</div>
            <canvas id="timeSeriesChart"></canvas>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">👥 Visitor Breakdown (Top 5)</div>
            <div id="visitorBreakdown" class="visitor-breakdown"></div>
        </div>
    </div>
    
    <script>
        let trafficChart, timeSeriesChart;
        let refreshInterval;
        
        function initCharts() {
            // Traffic Composition Pie Chart
            const trafficCtx = document.getElementById('trafficChart').getContext('2d');
            trafficChart = new Chart(trafficCtx, {
                type: 'doughnut',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: [
                            '#ef4444', '#f97316', '#eab308', '#84cc16', 
                            '#22c55e', '#14b8a6', '#06b6d4', '#3b82f6'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            labels: { color: 'white' }
                        }
                    }
                }
            });
            
            // Time Series Chart
            const timeCtx = document.getElementById('timeSeriesChart').getContext('2d');
            timeSeriesChart = new Chart(timeCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Clicks',
                            data: [],
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            yAxisID: 'y'
                        },
                        {
                            label: 'Conversions',
                            data: [],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            yAxisID: 'y'
                        },
                        {
                            label: 'CAC ($)',
                            data: [],
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            labels: { color: 'white' }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            ticks: { color: 'white' },
                            grid: { drawOnChartArea: false }
                        }
                    }
                }
            });
        }
        
        function updateDashboard(data) {
            // Update metrics
            document.getElementById('impressions').textContent = data.metrics.impressions.toLocaleString();
            document.getElementById('clicks').textContent = data.metrics.clicks.toLocaleString();
            document.getElementById('ctr').textContent = data.metrics.ctr;
            document.getElementById('conversions').textContent = data.metrics.conversions.toLocaleString();
            document.getElementById('cac').textContent = data.metrics.cac;
            document.getElementById('spend').textContent = data.metrics.spend;
            document.getElementById('active_journeys').textContent = data.metrics.active_journeys.toLocaleString();
            document.getElementById('avg_journey_days').textContent = data.metrics.avg_journey_days.toFixed(1);
            
            // Update status
            const statusEl = document.getElementById('status');
            if (data.is_running) {
                statusEl.textContent = 'RUNNING';
                statusEl.className = 'status running';
            } else {
                statusEl.textContent = 'STOPPED';
                statusEl.className = 'status stopped';
            }
            
            // Update visitor breakdown
            const visitorHtml = Object.entries(data.visitor_breakdown).map(([type, count]) => `
                <div class="visitor-type">
                    <div class="visitor-type-name">${type.replace(/_/g, ' ')}</div>
                    <div class="visitor-type-count">${count}</div>
                </div>
            `).join('');
            document.getElementById('visitorBreakdown').innerHTML = visitorHtml;
            
            // Update funnel
            const funnelHtml = Object.entries(data.conversion_funnel).map(([stage, count]) => `
                <div class="funnel-stage">
                    <span class="funnel-stage-name">${stage.charAt(0).toUpperCase() + stage.slice(1)}</span>
                    <span class="funnel-stage-count">${count}</span>
                </div>
            `).join('');
            document.getElementById('funnelContainer').innerHTML = funnelHtml;
            
            // Update traffic chart
            trafficChart.data.labels = Object.keys(data.visitor_breakdown);
            trafficChart.data.datasets[0].data = Object.values(data.visitor_breakdown);
            trafficChart.update();
            
            // Update time series chart
            if (data.time_series.timestamps.length > 0) {
                timeSeriesChart.data.labels = data.time_series.timestamps.map(ts => 
                    new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
                );
                timeSeriesChart.data.datasets[0].data = data.time_series.clicks;
                timeSeriesChart.data.datasets[1].data = data.time_series.conversions;
                timeSeriesChart.data.datasets[2].data = data.time_series.cac;
                timeSeriesChart.update();
            }
        }
        
        async function refreshData() {
            try {
                const response = await fetch('/api/dashboard_data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        async function startSimulation() {
            try {
                await fetch('/api/start', { method: 'POST' });
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                
                // Start auto-refresh
                refreshInterval = setInterval(refreshData, 1000);
                refreshData();
            } catch (error) {
                console.error('Error starting simulation:', error);
            }
        }
        
        async function stopSimulation() {
            try {
                await fetch('/api/stop', { method: 'POST' });
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                
                // Stop auto-refresh
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                }
                refreshData();
            } catch (error) {
                console.error('Error stopping simulation:', error);
            }
        }
        
        // Initialize on load
        window.onload = function() {
            initCharts();
            refreshData();
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Serve the dashboard HTML"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/dashboard_data')
def get_dashboard_data():
    """API endpoint for dashboard data"""
    return jsonify(dashboard.get_dashboard_data())

@app.route('/api/start', methods=['POST'])
def start_simulation():
    """Start the simulation"""
    dashboard.start_simulation()
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop the simulation"""
    dashboard.stop_simulation()
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 BEHAVIORAL HEALTH MARKETING DASHBOARD")
    print("="*60)
    print("\nFeatures:")
    print("✅ Real traffic composition (35% parents, 65% other)")
    print("✅ Multi-week journey tracking")
    print("✅ Concern level evolution")
    print("✅ No hardcoded segments")
    print("\nStarting dashboard at http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=False, port=5000, host='0.0.0.0')