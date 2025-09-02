#!/usr/bin/env python3
"""
EMERGENCY CONTROLS MONITORING DASHBOARD
Real-time monitoring of emergency systems and kill switches
"""

import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from collections import defaultdict
import os
import signal

from emergency_controls import get_emergency_controller, EmergencyType, EmergencyLevel

logger = logging.getLogger(__name__)

class EmergencyMonitorDashboard:
    """Real-time dashboard for emergency controls monitoring"""
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.controller = get_emergency_controller()
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nReceived shutdown signal, stopping monitor...")
        self.running = False
    
    def get_recent_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent emergency events from database"""
        try:
            conn = sqlite3.connect(self.controller.db_path)
            cutoff = datetime.now() - timedelta(hours=hours)
            
            cursor = conn.execute("""
                SELECT event_id, trigger_type, emergency_level, timestamp, 
                       current_value, threshold_value, message, component, 
                       resolved, actions_taken
                FROM emergency_events
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 50
            """, (cutoff.isoformat(),))
            
            events = []
            for row in cursor.fetchall():
                events.append({
                    'event_id': row[0],
                    'trigger_type': row[1],
                    'emergency_level': row[2],
                    'timestamp': row[3],
                    'current_value': row[4],
                    'threshold_value': row[5],
                    'message': row[6],
                    'component': row[7],
                    'resolved': bool(row[8]),
                    'actions_taken': json.loads(row[9]) if row[9] else []
                })
            
            conn.close()
            return events
            
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        status = self.controller.get_system_status()
        
        # Calculate additional metrics
        metrics = status['metrics'].copy()
        
        # Budget utilization rate
        total_budget = sum(
            v for k, v in self.controller.budget_tracker.items() 
            if k.endswith('_daily_limit')
        )
        total_spend = sum(
            v for k, v in self.controller.budget_tracker.items() 
            if not k.endswith('_daily_limit')
        )
        metrics['budget_utilization_rate'] = (total_spend / total_budget * 100) if total_budget > 0 else 0
        
        # Bid statistics
        if self.controller.bid_history:
            metrics['avg_bid'] = sum(self.controller.bid_history) / len(self.controller.bid_history)
            metrics['max_bid'] = max(self.controller.bid_history)
            metrics['min_bid'] = min(self.controller.bid_history)
            metrics['bid_volatility'] = (max(self.controller.bid_history) - min(self.controller.bid_history))
        else:
            metrics.update({'avg_bid': 0, 'max_bid': 0, 'min_bid': 0, 'bid_volatility': 0})
        
        # Training stability
        if self.controller.training_loss_history:
            recent_losses = self.controller.training_loss_history[-10:]
            metrics['recent_avg_loss'] = sum(recent_losses) / len(recent_losses)
            if len(self.controller.training_loss_history) > 20:
                older_losses = self.controller.training_loss_history[-20:-10]
                older_avg = sum(older_losses) / len(older_losses)
                metrics['loss_trend'] = (metrics['recent_avg_loss'] - older_avg) / older_avg * 100
            else:
                metrics['loss_trend'] = 0
        else:
            metrics.update({'recent_avg_loss': 0, 'loss_trend': 0})
        
        return metrics
    
    def get_trigger_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all emergency triggers"""
        trigger_status = {}
        
        for trigger_type, trigger in self.controller.triggers.items():
            # Get current value based on trigger type
            if trigger_type == EmergencyType.BUDGET_OVERRUN:
                current_values = []
                for budget_name, spend in self.controller.budget_tracker.items():
                    if not budget_name.endswith('_daily_limit'):
                        limit_key = f"{budget_name}_daily_limit"
                        if limit_key in self.controller.budget_tracker:
                            limit = self.controller.budget_tracker[limit_key]
                            if limit > 0:
                                current_values.append(spend / limit)
                current_value = max(current_values) if current_values else 0
                
            elif trigger_type == EmergencyType.ANOMALOUS_BIDDING:
                current_value = max(self.controller.bid_history) if self.controller.bid_history else 0
                
            elif trigger_type == EmergencyType.TRAINING_INSTABILITY:
                if len(self.controller.training_loss_history) >= 20:
                    recent = self.controller.training_loss_history[-10:]
                    baseline = self.controller.training_loss_history[-20:-10]
                    if baseline and sum(baseline) > 0:
                        current_value = sum(recent) / sum(baseline) * 10  # Scale for comparison
                    else:
                        current_value = 1.0
                else:
                    current_value = 1.0
                    
            elif trigger_type == EmergencyType.SYSTEM_ERROR_RATE:
                # Count errors in last minute
                cutoff = datetime.now() - timedelta(minutes=1)
                current_value = sum(
                    count for timestamp_str, count in self.controller.error_counts.items()
                    if datetime.fromisoformat(timestamp_str.split('_')[1]) > cutoff
                )
            else:
                current_value = 0
            
            # Calculate percentage of threshold
            threshold_pct = (current_value / trigger.threshold_value * 100) if trigger.threshold_value > 0 else 0
            
            # Determine status
            if threshold_pct >= 100:
                status = "TRIGGERED"
            elif threshold_pct >= 80:
                status = "WARNING"
            elif threshold_pct >= 60:
                status = "CAUTION"
            else:
                status = "NORMAL"
            
            trigger_status[trigger_type.value] = {
                'current_value': current_value,
                'threshold_value': trigger.threshold_value,
                'threshold_percentage': threshold_pct,
                'status': status,
                'enabled': trigger.enabled
            }
        
        return trigger_status
    
    def print_dashboard(self):
        """Print the emergency monitoring dashboard"""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("=" * 80)
        print(f"🚨 EMERGENCY CONTROLS MONITORING DASHBOARD 🚨".center(80))
        print(f"Last Updated: {now}".center(80))
        print("=" * 80)
        
        # System status
        status = self.controller.get_system_status()
        emergency_level = status['emergency_level'].upper()
        
        level_colors = {
            'GREEN': '🟢',
            'YELLOW': '🟡', 
            'RED': '🔴',
            'BLACK': '⚫'
        }
        
        print(f"\n📊 SYSTEM STATUS")
        print("-" * 40)
        print(f"Emergency Level: {level_colors.get(emergency_level, '❓')} {emergency_level}")
        print(f"System Active: {'✅' if status['active'] else '❌'}")
        print(f"Emergency Stop: {'❌ TRIGGERED' if status['emergency_stop_triggered'] else '✅ Normal'}")
        
        # System metrics
        metrics = self.get_system_metrics()
        print(f"\n📈 SYSTEM METRICS")
        print("-" * 40)
        print(f"Budget Utilization: {metrics['budget_utilization_rate']:.1f}%")
        print(f"Average Bid: ${metrics['avg_bid']:.2f}")
        print(f"Max Bid: ${metrics['max_bid']:.2f}")
        print(f"Recent Avg Loss: {metrics['recent_avg_loss']:.3f}")
        print(f"Loss Trend: {metrics['loss_trend']:+.1f}%")
        print(f"Error Rate: {metrics['error_rate']:.1f}/min")
        
        # Trigger status
        trigger_status = self.get_trigger_status()
        print(f"\n🎯 EMERGENCY TRIGGERS")
        print("-" * 40)
        
        for trigger_name, trigger_info in trigger_status.items():
            status_icon = {
                'NORMAL': '✅',
                'CAUTION': '🟡',
                'WARNING': '🟠',
                'TRIGGERED': '🔴'
            }.get(trigger_info['status'], '❓')
            
            enabled_icon = '🟢' if trigger_info['enabled'] else '⚪'
            
            print(f"{enabled_icon} {status_icon} {trigger_name.replace('_', ' ').title():<25} "
                  f"{trigger_info['threshold_percentage']:6.1f}% "
                  f"({trigger_info['current_value']:.2f}/{trigger_info['threshold_value']:.2f})")
        
        # Circuit breaker status
        print(f"\n⚡ CIRCUIT BREAKERS")
        print("-" * 40)
        
        for component, breaker in self.controller.circuit_breakers.items():
            state_icon = {
                'closed': '✅',
                'half_open': '🟡',
                'open': '❌'
            }.get(breaker.state, '❓')
            
            print(f"{state_icon} {component:<20} {breaker.state.upper():<10} "
                  f"Failures: {breaker.failure_count}/{breaker.failure_threshold}")
        
        # Recent events
        recent_events = self.get_recent_events(hours=1)
        print(f"\n📋 RECENT EVENTS (Last Hour)")
        print("-" * 40)
        
        if recent_events:
            for event in recent_events[:5]:  # Show last 5 events
                timestamp = datetime.fromisoformat(event['timestamp']).strftime("%H:%M:%S")
                level_icon = level_colors.get(event['emergency_level'].upper(), '❓')
                print(f"{level_icon} {timestamp} {event['trigger_type']:<20} {event['component']:<15}")
                print(f"   {event['message'][:60]}...")
        else:
            print("No recent events")
        
        # Control instructions
        print(f"\n🎮 CONTROLS")
        print("-" * 40)
        print("Ctrl+C: Stop monitoring")
        print("Manual emergency stop: Use trigger_manual_emergency_stop() function")
        print("Reset emergency state: Use reset_emergency_state() function")
        
        print("\n" + "=" * 80)
        print(f"Next update in {self.update_interval} seconds...")
    
    def run_interactive_monitor(self):
        """Run interactive monitoring dashboard"""
        self.running = True
        
        print("Starting Emergency Controls Monitor...")
        print("Press Ctrl+C to stop")
        
        try:
            while self.running:
                self.print_dashboard()
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"\nError in monitoring: {e}")
            logger.error(f"Monitoring error: {e}")
        finally:
            self.running = False
    
    def generate_status_report(self, filename: str = None):
        """Generate detailed status report"""
        if filename is None:
            filename = f"emergency_status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.controller.get_system_status(),
            'system_metrics': self.get_system_metrics(),
            'trigger_status': self.get_trigger_status(),
            'recent_events': self.get_recent_events(hours=24),
            'circuit_breaker_status': {
                name: {
                    'state': breaker.state,
                    'failure_count': breaker.failure_count,
                    'failure_threshold': breaker.failure_threshold,
                    'last_failure_time': breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
                }
                for name, breaker in self.controller.circuit_breakers.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Status report saved to: {filename}")
        return filename


def main():
    """Run emergency monitoring dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Emergency Controls Monitoring Dashboard")
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    parser.add_argument('--report', action='store_true', help='Generate status report and exit')
    parser.add_argument('--test', action='store_true', help='Run system tests')
    
    args = parser.parse_args()
    
    dashboard = EmergencyMonitorDashboard(update_interval=args.interval)
    
    if args.test:
        print("Running emergency system tests...")
        # Import and run tests
        from test_emergency_controls import main as run_tests
        success = run_tests()
        return 0 if success else 1
    
    elif args.report:
        print("Generating emergency status report...")
        filename = dashboard.generate_status_report()
        print(f"Report saved as: {filename}")
        return 0
    
    else:
        # Run interactive dashboard
        dashboard.run_interactive_monitor()
        return 0


if __name__ == "__main__":
    sys.exit(main())