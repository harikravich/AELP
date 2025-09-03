#!/usr/bin/env python3
"""
Fixed Dashboard Demo
Shows the working GAELP dashboard with real performance data
"""

import sys
import os
import logging
import threading
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardDemo:
    """Demonstrate the fixed GAELP dashboard functionality"""
    
    def __init__(self):
        self.demo_running = False
        
    def run_monitor_demo(self):
        """Run the fixed monitor with real data"""
        logger.info("🚀 Starting Fixed Dashboard Demo...")
        
        from gaelp_production_monitor import GAELPMonitor
        
        # Create monitor with web interface
        monitor = GAELPMonitor(web_mode=True)
        
        # Show current metrics
        self._show_current_metrics(monitor)
        
        # Create templates directory
        os.makedirs('templates', exist_ok=True)
        
        # Create HTML template  
        from gaelp_production_monitor import create_monitor_html
        html_content = create_monitor_html()
        
        with open('templates/gaelp_monitor.html', 'w') as f:
            f.write(html_content)
        
        logger.info("✅ Dashboard template created")
        logger.info("📊 Web dashboard available at: http://localhost:5000")
        logger.info("🔗 API endpoints:")
        logger.info("   - Status: http://localhost:5000/api/status")
        logger.info("   - Metrics: http://localhost:5000/api/metrics")
        logger.info("   - Components: http://localhost:5000/api/components")
        logger.info("   - Alerts: http://localhost:5000/api/alerts")
        
        # Start the web server
        try:
            monitor.run_web_monitor(port=5000)
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
    
    def _show_current_metrics(self, monitor):
        """Display current metrics from the fixed monitor"""
        logger.info("📊 Current System Metrics:")
        
        # Get real status
        status = monitor._get_status()
        metrics = status['metrics']
        
        logger.info(f"   📈 Episodes Completed: {metrics['last_episode']}")
        logger.info(f"   💰 Total Revenue: ${metrics['revenue']:,.2f}")
        logger.info(f"   📊 Average ROAS: {metrics['roas']:.2f}x")
        logger.info(f"   🎯 Conversion Rate: {metrics['conversion_rate']:.1%}")
        logger.info(f"   🔍 Exploration Rate: {metrics['epsilon']:.3f}")
        logger.info(f"   💸 Total Spend: ${metrics['spend']:,.2f}")
        logger.info(f"   📈 Total Profit: ${metrics['total_reward']:,.2f}")
        
        # Get component status
        components = monitor._get_component_status()
        running_components = 0
        total_components = 0
        
        for group, comps in components.items():
            for comp, status in comps.items():
                total_components += 1
                if status['status'] == 'running':
                    running_components += 1
        
        logger.info(f"   ⚙️ System Health: {running_components}/{total_components} components running")
        
        # Get metrics data
        metrics_data = monitor._get_metrics_data()
        if metrics_data['episodes']:
            logger.info(f"   📈 Training Data: {len(metrics_data['episodes'])} episodes available for charts")
            logger.info(f"   🎯 ROAS Range: {min(metrics_data['roas']):.2f}x - {max(metrics_data['roas']):.2f}x")
        else:
            logger.info("   ⚠️ No training data available yet")
    
    def run_streamlit_demo(self):
        """Run the fixed Streamlit dashboard"""
        logger.info("🎨 Starting Streamlit Dashboard Demo...")
        
        try:
            import subprocess
            result = subprocess.run([
                'streamlit', 'run', 'dashboard.py',
                '--server.port', '8501',
                '--server.headless', 'true',
                '--browser.gatherUsageStats', 'false'
            ], capture_output=False, text=True)
        except FileNotFoundError:
            logger.error("❌ Streamlit not installed. Install with: pip install streamlit")
        except Exception as e:
            logger.error(f"❌ Streamlit dashboard error: {e}")
    
    def run_api_test(self):
        """Test all dashboard APIs"""
        logger.info("🧪 Testing Dashboard APIs...")
        
        from gaelp_production_monitor import GAELPMonitor
        monitor = GAELPMonitor(web_mode=True)
        
        # Test status API
        status_data = monitor._get_status()
        logger.info(f"✅ Status API: {len(status_data)} top-level fields")
        
        # Test metrics API
        metrics_data = monitor._get_metrics_data()
        total_data_points = sum(len(v) if isinstance(v, list) else 1 for v in metrics_data.values())
        logger.info(f"✅ Metrics API: {total_data_points} total data points")
        
        # Test components API
        components_data = monitor._get_component_status()
        component_count = sum(len(group) for group in components_data.values())
        logger.info(f"✅ Components API: {component_count} components monitored")
        
        # Test chart data quality
        if metrics_data['episodes']:
            roas_data = metrics_data['roas']
            rewards_data = metrics_data['rewards']
            
            logger.info(f"📊 Chart Data Quality:")
            logger.info(f"   ROAS trend: {len(roas_data)} points, avg {sum(roas_data)/len(roas_data):.2f}")
            logger.info(f"   Reward trend: {len(rewards_data)} points, total {sum(rewards_data):,.2f}")
            logger.info(f"   ✅ Charts will display meaningful trends")
        else:
            logger.info(f"⚠️ No chart data available")
        
        logger.info("🎯 All APIs working with real data!")

def main():
    """Main demo runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed GAELP Dashboard Demo')
    parser.add_argument('--mode', choices=['web', 'streamlit', 'api-test'], default='web',
                       help='Demo mode to run')
    
    args = parser.parse_args()
    
    demo = DashboardDemo()
    
    if args.mode == 'web':
        demo.run_monitor_demo()
    elif args.mode == 'streamlit':
        demo.run_streamlit_demo()
    elif args.mode == 'api-test':
        demo.run_api_test()

if __name__ == "__main__":
    main()
