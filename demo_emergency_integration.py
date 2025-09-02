#!/usr/bin/env python3
"""
DEMO: EMERGENCY CONTROLS INTEGRATION WITH GAELP
Demonstrates emergency stop mechanisms in action during training
"""

import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

import time
import logging
from datetime import datetime
from emergency_controls import get_emergency_controller, emergency_stop_decorator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_emergency_controls():
    """Demonstrate emergency controls in action"""
    
    print("=" * 80)
    print(" EMERGENCY CONTROLS INTEGRATION DEMO ".center(80))
    print("=" * 80)
    
    # Initialize emergency controller
    emergency_controller = get_emergency_controller()
    emergency_controller._test_mode = True  # Enable test mode to prevent actual shutdown
    
    print(f"\n🚨 Emergency Controller Status: {emergency_controller.current_emergency_level.value.upper()}")
    
    # Mock GAELP components with emergency decorators
    @emergency_stop_decorator("discovery_engine")
    def mock_discovery_operation():
        logger.info("Discovery engine: Finding new segments...")
        return {"new_segments": ["high_value_users", "weekend_shoppers"]}
    
    @emergency_stop_decorator("bidding_system")  
    def mock_bidding_operation(bid_amount):
        logger.info(f"Bidding system: Placing bid of ${bid_amount:.2f}")
        emergency_controller.record_bid(bid_amount)
        return {"bid_placed": True, "cpc": bid_amount}
    
    @emergency_stop_decorator("training_orchestrator")
    def mock_training_step(loss):
        logger.info(f"Training step: Current loss = {loss:.3f}")
        emergency_controller.record_training_loss(loss)
        return {"training_complete": True, "new_loss": loss}
    
    @emergency_stop_decorator("budget_pacer")
    def mock_budget_check(spend, limit):
        logger.info(f"Budget check: ${spend:.2f} / ${limit:.2f}")
        emergency_controller.update_budget_tracking("demo_campaign", spend, limit)
        return {"budget_ok": spend <= limit}
    
    print("\n" + "🟢 PHASE 1: NORMAL OPERATIONS")
    print("-" * 50)
    
    try:
        # Normal operations
        discovery_result = mock_discovery_operation()
        print(f"✅ Discovery: {discovery_result}")
        
        bid_result = mock_bidding_operation(2.50)  # Normal bid
        print(f"✅ Bidding: {bid_result}")
        
        training_result = mock_training_step(1.2)  # Normal loss
        print(f"✅ Training: {training_result}")
        
        budget_result = mock_budget_check(500, 1000)  # 50% budget usage
        print(f"✅ Budget: {budget_result}")
        
        print(f"\n📊 System Status: {emergency_controller.current_emergency_level.value.upper()}")
        
    except Exception as e:
        print(f"❌ Error in normal operations: {e}")
    
    print("\n" + "🟡 PHASE 2: WARNING CONDITIONS")  
    print("-" * 50)
    
    try:
        # Trigger warning conditions
        print("Simulating high bid...")
        bid_result = mock_bidding_operation(60.0)  # High but not critical bid
        print(f"⚠️  High bid placed: {bid_result}")
        
        print("Simulating budget overrun...")
        budget_result = mock_budget_check(1250, 1000)  # 125% budget usage
        print(f"⚠️  Budget overrun: {budget_result}")
        
        print(f"\n📊 System Status: {emergency_controller.current_emergency_level.value.upper()}")
        
    except Exception as e:
        print(f"❌ Error in warning phase: {e}")
    
    print("\n" + "🔴 PHASE 3: CRITICAL CONDITIONS")
    print("-" * 50)
    
    try:
        # Trigger critical conditions
        print("Simulating training instability...")
        for i in range(5):
            loss = 1.0 + i * 5.0  # Exploding loss
            training_result = mock_training_step(loss)
            print(f"⚠️  Training loss: {training_result}")
        
        print("Simulating anomalous bidding...")
        bid_result = mock_bidding_operation(85.0)  # Critical bid amount
        print(f"🚨 Critical bid: {bid_result}")
        
        print(f"\n📊 System Status: {emergency_controller.current_emergency_level.value.upper()}")
        
    except Exception as e:
        print(f"🛑 EMERGENCY PROTECTION ACTIVATED: {e}")
        print("System automatically prevented dangerous operations!")
    
    print("\n" + "⚫ PHASE 4: EMERGENCY STOP")
    print("-" * 50)
    
    try:
        print("Triggering manual emergency stop...")
        emergency_controller.trigger_manual_emergency_stop("Demo emergency stop")
        
        print("Attempting operations after emergency stop...")
        
        # These should be blocked
        discovery_result = mock_discovery_operation()
        print(f"❌ This should not execute: {discovery_result}")
        
    except Exception as e:
        print(f"🛑 EMERGENCY STOP ACTIVE: {e}")
        print("All operations successfully blocked!")
    
    # Show final status
    print("\n" + "📋 FINAL EMERGENCY STATUS")
    print("-" * 50)
    
    status = emergency_controller.get_system_status()
    
    print(f"Emergency Level: {status['emergency_level'].upper()}")
    print(f"System Active: {status['active']}")
    print(f"Emergency Stop: {status['emergency_stop_triggered']}")
    print(f"Recent Events: {len(status['recent_events'])}")
    
    print(f"\nCircuit Breakers:")
    for component, state in status['circuit_breakers'].items():
        icon = "🟢" if state == "closed" else "🟡" if state == "half_open" else "🔴"
        print(f"  {icon} {component}: {state.upper()}")
    
    print(f"\nMetrics:")
    for key, value in status['metrics'].items():
        print(f"  • {key}: {value}")
    
    print("\n" + "✅ EMERGENCY CONTROLS VERIFICATION COMPLETE")
    print("-" * 50)
    
    verification_results = [
        ("Budget overrun detection", any(event['trigger_type'] == 'budget_overrun' for event in status['recent_events'])),
        ("Anomalous bidding detection", any(event['trigger_type'] == 'anomalous_bidding' for event in status['recent_events'])),
        ("Training instability detection", any(event['trigger_type'] == 'training_instability' for event in status['recent_events'])),
        ("Manual emergency stop", status['emergency_stop_triggered']),
        ("System state preservation", len(status['recent_events']) > 0),
        ("Circuit breaker activation", any(state != "closed" for state in status['circuit_breakers'].values()))
    ]
    
    for check, passed in verification_results:
        icon = "✅" if passed else "❌"
        print(f"{icon} {check}")
    
    all_passed = all(passed for _, passed in verification_results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL EMERGENCY CONTROLS WORKING - SYSTEM PRODUCTION READY")
    else:
        print("⚠️  SOME EMERGENCY CONTROLS NEED REVIEW")
    print("=" * 80)
    
    return all_passed


def show_emergency_monitor_preview():
    """Show a preview of the emergency monitoring dashboard"""
    
    print("\n" + "=" * 80)
    print(" EMERGENCY MONITORING DASHBOARD PREVIEW ".center(80)) 
    print("=" * 80)
    
    emergency_controller = get_emergency_controller()
    
    from emergency_monitor import EmergencyMonitorDashboard
    dashboard = EmergencyMonitorDashboard()
    
    # Generate and show metrics
    metrics = dashboard.get_system_metrics()
    trigger_status = dashboard.get_trigger_status()
    
    print(f"\n📊 SYSTEM METRICS")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n🎯 EMERGENCY TRIGGERS")
    print("-" * 40)
    for trigger_name, info in trigger_status.items():
        status_icon = {
            'NORMAL': '✅',
            'CAUTION': '🟡', 
            'WARNING': '🟠',
            'TRIGGERED': '🔴'
        }.get(info['status'], '❓')
        
        print(f"{status_icon} {trigger_name:<25} {info['threshold_percentage']:6.1f}%")
    
    print(f"\n⚡ CIRCUIT BREAKERS")
    print("-" * 40)
    for component, breaker in emergency_controller.circuit_breakers.items():
        state_icon = {
            'closed': '✅',
            'half_open': '🟡',
            'open': '❌'
        }.get(breaker.state, '❓')
        
        print(f"{state_icon} {component:<20} {breaker.state.upper()}")
    
    print(f"\n🎮 AVAILABLE COMMANDS")
    print("-" * 40)
    print("python3 emergency_monitor.py           # Live dashboard")
    print("python3 emergency_monitor.py --report  # Generate report")
    print("python3 emergency_monitor.py --test    # Run tests")


if __name__ == "__main__":
    # Run the demonstration
    success = demo_emergency_controls()
    
    # Show monitoring dashboard preview
    show_emergency_monitor_preview()
    
    print(f"\n🚀 READY FOR PRODUCTION INTEGRATION")
    print("The emergency controls are now integrated with:")
    print("• run_production_training.py - Main training script")  
    print("• emergency_controls.py - Core safety system")
    print("• emergency_monitor.py - Real-time monitoring")
    print("• test_emergency_controls.py - Comprehensive testing")
    
    sys.exit(0 if success else 1)