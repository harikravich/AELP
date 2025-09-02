#!/usr/bin/env python3
"""
Success Criteria Dashboard Integration

Integrates the GAELP success criteria monitoring system with the live dashboard,
providing real-time KPI tracking, alerts, and performance visualization.

NO FALLBACKS - All metrics are real and monitored against strict thresholds.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit
import threading
import time

from gaelp_success_criteria_monitor import (
    GAELPSuccessCriteriaDefinition,
    PerformanceMonitor,
    KPICategory,
    AlertSeverity,
    KPIMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuccessCriteriaDashboard:
    """
    Real-time dashboard integration for GAELP success criteria monitoring.
    
    Provides live KPI tracking, alert notifications, and performance
    visualization with NO FALLBACKS.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """Initialize dashboard integration"""
        
        self.host = host
        self.port = port
        
        # Initialize success criteria and monitoring
        self.success_criteria = GAELPSuccessCriteriaDefinition()
        self.monitor = PerformanceMonitor(self.success_criteria)
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'gaelp-success-criteria-dashboard'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard state
        self.dashboard_data = {}
        self.alert_queue = []
        self.update_lock = threading.Lock()
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        
        logger.info("Success criteria dashboard initialized")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard_home():
            """Main dashboard page"""
            return render_template('success_criteria_dashboard.html')
        
        @self.app.route('/api/success-criteria')
        def get_success_criteria():
            """Get all success criteria definitions"""
            
            criteria_data = {}
            for name, criteria in self.success_criteria.success_criteria.items():
                criteria_data[name] = {
                    "name": criteria.name,
                    "category": criteria.category.value,
                    "target_value": criteria.target_value,
                    "minimum_acceptable": criteria.minimum_acceptable,
                    "excellence_threshold": criteria.excellence_threshold,
                    "business_critical": criteria.business_critical,
                    "revenue_impact": criteria.revenue_impact,
                    "measurement_window_hours": criteria.measurement_window_hours
                }
            
            return jsonify(criteria_data)
        
        @self.app.route('/api/current-performance')
        def get_current_performance():
            """Get current performance against all KPIs"""
            
            current_metrics = self.monitor.get_current_metrics()
            performance_data = {}
            
            for kpi_name, metrics in current_metrics.items():
                performance_data[kpi_name] = {
                    "current_value": metrics.current_value,
                    "target_value": metrics.target_value,
                    "minimum_acceptable": metrics.minimum_acceptable,
                    "performance_ratio": metrics.performance_ratio,
                    "status": metrics.status,
                    "trend_direction": metrics.trend_direction,
                    "active_alerts": metrics.active_alerts,
                    "last_updated": metrics.last_updated.isoformat()
                }
            
            return jsonify(performance_data)
        
        @self.app.route('/api/system-health')
        def get_system_health():
            """Get overall system health summary"""
            return jsonify(self.monitor.get_system_health_summary())
        
        @self.app.route('/api/performance-report/<int:hours>')
        def get_performance_report(hours):
            """Get performance report for specified time period"""
            
            report = self.monitor.generate_performance_report(hours_back=hours)
            return jsonify(report)
        
        @self.app.route('/api/alerts/active')
        def get_active_alerts():
            """Get all active alerts"""
            
            current_metrics = self.monitor.get_current_metrics()
            active_alerts = []
            
            for kpi_name, metrics in current_metrics.items():
                criteria = self.success_criteria.get_success_criteria(kpi_name)
                
                for alert_message in metrics.active_alerts:
                    # Determine severity
                    if "CRITICAL" in alert_message:
                        severity = AlertSeverity.CRITICAL
                    elif "WARNING" in alert_message:
                        severity = AlertSeverity.HIGH
                    else:
                        severity = AlertSeverity.MEDIUM
                    
                    active_alerts.append({
                        "kpi_name": kpi_name,
                        "message": alert_message,
                        "severity": severity.value,
                        "current_value": metrics.current_value,
                        "target_value": metrics.target_value,
                        "business_critical": criteria.business_critical if criteria else False,
                        "revenue_impact": criteria.revenue_impact if criteria else 0,
                        "timestamp": metrics.last_updated.isoformat()
                    })
            
            # Sort by severity (critical first)
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            active_alerts.sort(key=lambda x: severity_order.get(x["severity"], 4))
            
            return jsonify(active_alerts)
        
        @self.app.route('/api/kpi-categories')
        def get_kpi_categories():
            """Get KPIs organized by category"""
            
            categories = {}
            current_metrics = self.monitor.get_current_metrics()
            
            for category in KPICategory:
                category_kpis = self.success_criteria.get_criteria_by_category(category)
                category_data = {
                    "name": category.value,
                    "kpis": {},
                    "summary": {
                        "total_kpis": len(category_kpis),
                        "meeting_targets": 0,
                        "critical_issues": 0,
                        "avg_performance_ratio": 0
                    }
                }
                
                performance_ratios = []
                
                for kpi_name, criteria in category_kpis.items():
                    kpi_data = {
                        "name": criteria.name,
                        "target": criteria.target_value,
                        "minimum": criteria.minimum_acceptable,
                        "business_critical": criteria.business_critical
                    }
                    
                    if kpi_name in current_metrics:
                        metrics = current_metrics[kpi_name]
                        kpi_data.update({
                            "current_value": metrics.current_value,
                            "performance_ratio": metrics.performance_ratio,
                            "status": metrics.status,
                            "alerts_count": len(metrics.active_alerts)
                        })
                        
                        performance_ratios.append(metrics.performance_ratio)
                        
                        if metrics.current_value >= criteria.target_value:
                            category_data["summary"]["meeting_targets"] += 1
                        
                        if metrics.current_value < criteria.minimum_acceptable:
                            category_data["summary"]["critical_issues"] += 1
                    
                    category_data["kpis"][kpi_name] = kpi_data
                
                if performance_ratios:
                    category_data["summary"]["avg_performance_ratio"] = sum(performance_ratios) / len(performance_ratios)
                
                categories[category.value] = category_data
            
            return jsonify(categories)
    
    def _setup_socket_handlers(self):
        """Setup WebSocket handlers for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Dashboard client connected")
            emit('status', {'message': 'Connected to GAELP Success Criteria Monitor'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("Dashboard client disconnected")
        
        @self.socketio.on('subscribe_updates')
        def handle_subscribe():
            """Subscribe client to real-time updates"""
            logger.info("Client subscribed to real-time updates")
            emit('subscription_confirmed', {'status': 'subscribed'})
    
    def start_dashboard(self):
        """Start the dashboard server"""
        
        logger.info(f"Starting success criteria dashboard on {self.host}:{self.port}")
        
        # Start monitoring
        self.monitor.start_monitoring(check_interval_seconds=30)
        
        # Start real-time update thread
        update_thread = threading.Thread(target=self._real_time_update_loop, daemon=True)
        update_thread.start()
        
        # Start Flask-SocketIO server
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False
        )
    
    def _real_time_update_loop(self):
        """Real-time update loop for dashboard clients"""
        
        logger.info("Starting real-time update loop")
        
        while True:
            try:
                # Get current data
                current_metrics = self.monitor.get_current_metrics()
                system_health = self.monitor.get_system_health_summary()
                
                # Prepare update payload
                update_data = {
                    "timestamp": datetime.now().isoformat(),
                    "system_health": system_health,
                    "kpi_updates": {},
                    "new_alerts": []
                }
                
                # Process KPI updates
                for kpi_name, metrics in current_metrics.items():
                    update_data["kpi_updates"][kpi_name] = {
                        "current_value": metrics.current_value,
                        "performance_ratio": metrics.performance_ratio,
                        "status": metrics.status,
                        "trend_direction": metrics.trend_direction,
                        "alerts_count": len(metrics.active_alerts)
                    }
                    
                    # Check for new alerts
                    for alert in metrics.active_alerts:
                        criteria = self.success_criteria.get_success_criteria(kpi_name)
                        
                        alert_data = {
                            "kpi_name": kpi_name,
                            "message": alert,
                            "severity": "critical" if "CRITICAL" in alert else "high",
                            "business_critical": criteria.business_critical if criteria else False,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Only include if it's a new alert (simplified check)
                        if alert_data not in self.alert_queue[-10:]:  # Check last 10 alerts
                            update_data["new_alerts"].append(alert_data)
                            self.alert_queue.append(alert_data)
                
                # Keep alert queue manageable
                if len(self.alert_queue) > 100:
                    self.alert_queue = self.alert_queue[-50:]
                
                # Emit updates to all connected clients
                with self.update_lock:
                    self.socketio.emit('real_time_update', update_data)
                
                # Send critical alerts separately for immediate attention
                critical_alerts = [alert for alert in update_data["new_alerts"] 
                                 if alert["severity"] == "critical"]
                
                if critical_alerts:
                    self.socketio.emit('critical_alert', {
                        "alerts": critical_alerts,
                        "count": len(critical_alerts),
                        "timestamp": datetime.now().isoformat()
                    })
                    logger.warning(f"Emitted {len(critical_alerts)} critical alerts to dashboard")
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in real-time update loop: {e}")
                time.sleep(30)
    
    def generate_dashboard_config(self) -> Dict[str, Any]:
        """Generate configuration for dashboard frontend"""
        
        config = {
            "title": "GAELP Success Criteria Monitor",
            "refresh_interval_seconds": 30,
            "categories": [],
            "alert_severities": {
                "critical": {"color": "#FF0000", "priority": 1},
                "high": {"color": "#FF6600", "priority": 2},
                "medium": {"color": "#FFAA00", "priority": 3},
                "low": {"color": "#00AA00", "priority": 4}
            },
            "kpi_status_colors": {
                "excellent": "#00FF00",
                "good": "#90EE90", 
                "warning": "#FFA500",
                "critical": "#FF0000"
            }
        }
        
        # Add category configurations
        for category in KPICategory:
            category_kpis = self.success_criteria.get_criteria_by_category(category)
            
            config["categories"].append({
                "name": category.value,
                "display_name": category.value.replace('_', ' ').title(),
                "kpi_count": len(category_kpis),
                "kpis": list(category_kpis.keys())
            })
        
        return config
    
    def export_success_criteria_summary(self) -> Dict[str, Any]:
        """Export comprehensive success criteria summary"""
        
        summary = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_kpis": len(self.success_criteria.success_criteria),
                "business_critical_kpis": len(self.success_criteria.get_business_critical_criteria()),
                "no_fallbacks": True,
                "strict_enforcement": True
            },
            "business_impact": {
                "total_daily_revenue_at_risk": sum(
                    criteria.revenue_impact 
                    for criteria in self.success_criteria.get_business_critical_criteria().values()
                ),
                "highest_risk_kpi": None,
                "highest_impact_amount": 0
            },
            "categories": {},
            "thresholds_summary": {
                "roas_targets": {},
                "efficiency_targets": {},
                "quality_targets": {},
                "operational_targets": {}
            }
        }
        
        # Find highest risk KPI
        for name, criteria in self.success_criteria.get_business_critical_criteria().items():
            if criteria.revenue_impact > summary["business_impact"]["highest_impact_amount"]:
                summary["business_impact"]["highest_impact_amount"] = criteria.revenue_impact
                summary["business_impact"]["highest_risk_kpi"] = name
        
        # Categorize KPIs
        for category in KPICategory:
            category_kpis = self.success_criteria.get_criteria_by_category(category)
            
            summary["categories"][category.value] = {
                "count": len(category_kpis),
                "business_critical_count": sum(
                    1 for criteria in category_kpis.values() 
                    if criteria.business_critical
                ),
                "total_revenue_impact": sum(
                    criteria.revenue_impact for criteria in category_kpis.values()
                ),
                "kpis": list(category_kpis.keys())
            }
        
        # Summarize key thresholds
        roas_kpis = {k: v for k, v in self.success_criteria.success_criteria.items() 
                    if 'roas' in k.lower()}
        
        for name, criteria in roas_kpis.items():
            summary["thresholds_summary"]["roas_targets"][name] = {
                "target": criteria.target_value,
                "minimum": criteria.minimum_acceptable,
                "excellence": criteria.excellence_threshold
            }
        
        return summary


def create_dashboard_template():
    """Create HTML template for the success criteria dashboard"""
    
    template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GAELP Success Criteria Monitor</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .kpi-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid #3498db;
        }
        .kpi-card.excellent { border-left-color: #27ae60; }
        .kpi-card.good { border-left-color: #f39c12; }
        .kpi-card.warning { border-left-color: #e67e22; }
        .kpi-card.critical { border-left-color: #e74c3c; }
        
        .kpi-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .kpi-target {
            color: #666;
            font-size: 0.9em;
        }
        .alert-banner {
            background: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        .system-health {
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .health-score {
            font-size: 3em;
            font-weight: bold;
        }
        .health-good { color: #27ae60; }
        .health-warning { color: #f39c12; }
        .health-critical { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="header">
        <h1>GAELP Success Criteria Monitor</h1>
        <p>Real-time KPI tracking with NO FALLBACKS</p>
        <div id="connection-status">Connecting...</div>
    </div>

    <div id="alert-banner" class="alert-banner">
        <strong>CRITICAL ALERT:</strong> <span id="alert-message"></span>
    </div>

    <div class="system-health">
        <h2>System Health</h2>
        <div id="health-score" class="health-score">--</div>
        <div id="health-status">Loading...</div>
    </div>

    <div id="kpi-grid" class="status-grid">
        <!-- KPI cards will be populated here -->
    </div>

    <script>
        const socket = io();
        
        socket.on('connect', function() {
            document.getElementById('connection-status').textContent = 'Connected';
            socket.emit('subscribe_updates');
            loadInitialData();
        });

        socket.on('real_time_update', function(data) {
            updateKPIs(data.kpi_updates);
            updateSystemHealth(data.system_health);
        });

        socket.on('critical_alert', function(data) {
            showCriticalAlert(data.alerts);
        });

        function loadInitialData() {
            // Load success criteria and current performance
            fetch('/api/current-performance')
                .then(response => response.json())
                .then(data => {
                    createKPICards(data);
                });
            
            fetch('/api/system-health')
                .then(response => response.json())
                .then(data => {
                    updateSystemHealth(data);
                });
        }

        function createKPICards(performanceData) {
            const grid = document.getElementById('kpi-grid');
            grid.innerHTML = '';

            for (const [kpiName, metrics] of Object.entries(performanceData)) {
                const card = document.createElement('div');
                card.className = `kpi-card ${metrics.status}`;
                card.innerHTML = `
                    <h3>${kpiName.replace(/_/g, ' ').toUpperCase()}</h3>
                    <div class="kpi-value">${metrics.current_value.toFixed(2)}</div>
                    <div class="kpi-target">
                        Target: ${metrics.target_value} | 
                        Min: ${metrics.minimum_acceptable} |
                        Status: ${metrics.status.toUpperCase()}
                    </div>
                    <div>Performance: ${(metrics.performance_ratio * 100).toFixed(1)}%</div>
                    <div>Alerts: ${metrics.active_alerts.length}</div>
                `;
                grid.appendChild(card);
            }
        }

        function updateKPIs(updates) {
            // Update existing KPI cards with new values
            for (const [kpiName, metrics] of Object.entries(updates)) {
                const cards = document.querySelectorAll('.kpi-card');
                cards.forEach(card => {
                    if (card.querySelector('h3').textContent.toLowerCase() === 
                        kpiName.replace(/_/g, ' ').toLowerCase()) {
                        
                        card.className = `kpi-card ${metrics.status}`;
                        card.querySelector('.kpi-value').textContent = metrics.current_value.toFixed(2);
                        
                        // Update status and performance info
                        const statusInfo = card.querySelector('.kpi-target').nextElementSibling;
                        statusInfo.textContent = `Performance: ${(metrics.performance_ratio * 100).toFixed(1)}%`;
                        
                        const alertInfo = statusInfo.nextElementSibling;
                        alertInfo.textContent = `Alerts: ${metrics.alerts_count}`;
                    }
                });
            }
        }

        function updateSystemHealth(healthData) {
            const healthScore = document.getElementById('health-score');
            const healthStatus = document.getElementById('health-status');
            
            healthScore.textContent = healthData.health_score ? 
                healthData.health_score.toFixed(1) + '%' : '--';
            
            healthScore.className = 'health-score ';
            if (healthData.health_score >= 80) {
                healthScore.className += 'health-good';
            } else if (healthData.health_score >= 50) {
                healthScore.className += 'health-warning';
            } else {
                healthScore.className += 'health-critical';
            }
            
            healthStatus.textContent = `Status: ${healthData.status || 'Unknown'} | ` +
                `Critical Failures: ${healthData.business_critical_failures || 0} | ` +
                `Active Alerts: ${healthData.total_active_alerts || 0}`;
        }

        function showCriticalAlert(alerts) {
            if (alerts && alerts.length > 0) {
                const banner = document.getElementById('alert-banner');
                const message = document.getElementById('alert-message');
                
                message.textContent = alerts.map(alert => 
                    `${alert.kpi_name}: ${alert.message}`
                ).join('; ');
                
                banner.style.display = 'block';
                
                // Hide after 10 seconds
                setTimeout(() => {
                    banner.style.display = 'none';
                }, 10000);
            }
        }

        // Refresh data every 30 seconds
        setInterval(loadInitialData, 30000);
    </script>
</body>
</html>
    """
    
    # Save template
    template_dir = "/home/hariravichandran/AELP/templates"
    import os
    os.makedirs(template_dir, exist_ok=True)
    
    template_path = f"{template_dir}/success_criteria_dashboard.html"
    with open(template_path, 'w') as f:
        f.write(template)
    
    logger.info(f"Dashboard template created at: {template_path}")


def main():
    """Main function to run the dashboard"""
    
    logger.info("Starting GAELP Success Criteria Dashboard")
    
    # Create dashboard template
    create_dashboard_template()
    
    # Initialize and start dashboard
    dashboard = SuccessCriteriaDashboard()
    
    # Export configuration and summary
    config = dashboard.generate_dashboard_config()
    summary = dashboard.export_success_criteria_summary()
    
    # Save configuration files
    with open('/home/hariravichandran/AELP/dashboard_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open('/home/hariravichandran/AELP/success_criteria_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Total KPIs monitored: {summary['metadata']['total_kpis']}")
    logger.info(f"Business critical KPIs: {summary['metadata']['business_critical_kpis']}")
    logger.info(f"Daily revenue at risk: ${summary['business_impact']['total_daily_revenue_at_risk']:,.0f}")
    
    # Start dashboard server
    dashboard.start_dashboard()


if __name__ == "__main__":
    main()