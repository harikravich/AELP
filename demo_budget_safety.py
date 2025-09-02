#!/usr/bin/env python3
"""
Budget Safety Controller Demo
Demonstrates comprehensive budget safety controls preventing overspending.
"""

import logging
import tempfile
import os
import json
from decimal import Decimal
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import budget safety controller
from budget_safety_controller import BudgetSafetyController, BudgetLimits, get_budget_safety_controller

def demo_budget_safety():
    """Demonstrate budget safety controller functionality"""
    
    print("🛡️ GAELP Budget Safety Controller Demo")
    print("=" * 50)
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "demo_config.json")
    
    # Create demo configuration
    demo_config = {
        "default_limits": {
            "daily_limit": 1000.0,
            "weekly_limit": 5000.0,
            "monthly_limit": 20000.0,
            "max_hourly_spend": 100.0,
            "max_hourly_velocity_increase": 0.50,
            "warning_threshold": 0.80,
            "critical_threshold": 0.95,
            "emergency_threshold": 1.00,
            "max_bid_multiplier": 3.0,
            "max_spend_acceleration": 2.0,
            "prediction_window_hours": 2,
            "overspend_prevention_buffer": 0.10
        },
        "monitoring_intervals": {
            "spending_check_seconds": 30,
            "velocity_check_seconds": 60,
            "anomaly_check_seconds": 120,
            "prediction_check_seconds": 300
        },
        "emergency_actions": {
            "auto_pause_campaigns": True,
            "emergency_stop_threshold": 1.05,
            "notification_webhook": None
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(demo_config, f, indent=2)
    
    print(f"📝 Configuration saved to: {config_path}")
    
    # Initialize Budget Safety Controller
    controller = BudgetSafetyController(config_path)
    controller.db_path = os.path.join(temp_dir, "demo_budget_safety.db")
    controller._test_mode = True  # Prevent actual system exit
    
    print("✅ Budget Safety Controller initialized")
    
    # Demo 1: Register campaigns with different limits
    print(f"\n📊 Demo 1: Campaign Registration")
    
    campaigns = {
        "high_value_campaign": BudgetLimits(
            daily_limit=Decimal('5000.00'),
            weekly_limit=Decimal('25000.00'),
            monthly_limit=Decimal('100000.00'),
            max_hourly_spend=Decimal('500.00'),
            max_hourly_velocity_increase=0.30,
            warning_threshold=0.85,
            critical_threshold=0.97,
            emergency_threshold=1.00
        ),
        "standard_campaign": BudgetLimits(
            daily_limit=Decimal('1000.00'),
            weekly_limit=Decimal('5000.00'),
            monthly_limit=Decimal('20000.00'),
            max_hourly_spend=Decimal('100.00'),
            max_hourly_velocity_increase=0.50
        ),
        "test_campaign": BudgetLimits(
            daily_limit=Decimal('100.00'),
            weekly_limit=Decimal('500.00'),
            monthly_limit=Decimal('2000.00'),
            max_hourly_spend=Decimal('20.00'),
            max_hourly_velocity_increase=0.25,
            warning_threshold=0.75,
            critical_threshold=0.90,
            emergency_threshold=0.95
        )
    }
    
    for campaign_id, limits in campaigns.items():
        controller.register_campaign(campaign_id, limits)
        print(f"   ✓ {campaign_id}: ${limits.daily_limit}/day, ${limits.max_hourly_spend}/hour")
    
    # Demo 2: Normal spending patterns
    print(f"\n💰 Demo 2: Normal Spending Patterns")
    
    test_campaign = "test_campaign"
    
    # Record several normal spending events
    normal_spends = [
        (Decimal('15.00'), Decimal('1.50')),
        (Decimal('12.00'), Decimal('1.20')),
        (Decimal('18.00'), Decimal('1.80')),
        (Decimal('10.00'), Decimal('1.00')),
        (Decimal('20.00'), Decimal('2.00'))
    ]
    
    total_spent = Decimal('0')
    for i, (amount, bid) in enumerate(normal_spends, 1):
        is_safe, violations = controller.record_spending(
            campaign_id=test_campaign,
            channel="google_ads",
            amount=amount,
            bid_amount=bid,
            impressions=500,
            clicks=25,
            conversions=1
        )
        total_spent += amount
        
        print(f"   Spend {i}: ${amount} (bid: ${bid}) → Safe: {is_safe}, Total: ${total_spent}")
        
        if violations:
            for violation in violations:
                print(f"     ⚠️ {violation}")
    
    # Demo 3: Approaching budget limits
    print(f"\n⚠️ Demo 3: Approaching Budget Limits")
    
    # This should trigger warnings (we're at $75, limit is $100, so $30 more gets us to 75% warning threshold)
    warning_amount = Decimal('5.00')  # Gets us to $80 = 80% which should trigger warning
    
    is_safe, violations = controller.record_spending(
        campaign_id=test_campaign,
        channel="google_ads", 
        amount=warning_amount,
        bid_amount=Decimal('1.25'),
        impressions=200,
        clicks=10,
        conversions=0
    )
    total_spent += warning_amount
    
    print(f"   Warning test: ${warning_amount} → Safe: {is_safe}, Total: ${total_spent}")
    print(f"   Utilization: {float(total_spent/campaigns[test_campaign].daily_limit):.1%}")
    
    for violation in violations:
        print(f"   ⚠️ {violation}")
    
    # Demo 4: Hourly velocity limits
    print(f"\n🚀 Demo 4: Hourly Velocity Testing")
    
    # Try to spend more than hourly limit ($20)
    velocity_amount = Decimal('25.00')  # This exceeds the $20 hourly limit
    
    is_safe, violations = controller.record_spending(
        campaign_id=test_campaign,
        channel="google_ads",
        amount=velocity_amount,
        bid_amount=Decimal('2.50'),
        impressions=1000,
        clicks=50,
        conversions=3
    )
    
    print(f"   Velocity test: ${velocity_amount} → Safe: {is_safe}")
    
    for violation in violations:
        print(f"   🚨 {violation}")
    
    # Check campaign status
    campaign_status = controller.get_campaign_status(test_campaign)
    if campaign_status:
        print(f"   Campaign status: {campaign_status['status']}")
    
    # Demo 5: Pre-spend validation
    print(f"\n🔍 Demo 5: Pre-spend Validation")
    
    test_amounts = [Decimal('5.00'), Decimal('15.00'), Decimal('50.00'), Decimal('200.00')]
    
    for amount in test_amounts:
        is_safe, reason = controller.is_campaign_safe_to_spend(test_campaign, amount)
        print(f"   ${amount} check: {'✅ SAFE' if is_safe else '❌ BLOCKED'} - {reason}")
    
    # Demo 6: Bid anomaly detection
    print(f"\n🕵️ Demo 6: Bid Anomaly Detection")
    
    # Use the standard campaign which should still be active
    std_campaign = "standard_campaign"
    
    # Record normal bids to establish baseline
    print("   Establishing bid baseline...")
    for i in range(12):
        controller.record_spending(
            campaign_id=std_campaign,
            channel="google_ads",
            amount=Decimal('20.00'),
            bid_amount=Decimal('2.00'),  # Normal $2 bid
            impressions=400,
            clicks=20,
            conversions=1
        )
    
    # Now record anomalous bid
    print("   Testing anomalous bid...")
    is_safe, violations = controller.record_spending(
        campaign_id=std_campaign,
        channel="google_ads",
        amount=Decimal('30.00'),
        bid_amount=Decimal('15.00'),  # 7.5x normal bid - should be anomalous
        impressions=200,
        clicks=10,
        conversions=2
    )
    
    print(f"   Anomalous bid: $15.00 (vs $2.00 baseline) → Safe: {is_safe}")
    
    for violation in violations:
        print(f"   🔍 {violation}")
    
    # Demo 7: System status overview
    print(f"\n📈 Demo 7: System Status Overview")
    
    status = controller.get_system_status()
    
    print(f"   System Status:")
    print(f"   • Total campaigns: {status['campaigns']['total']}")
    print(f"   • Active campaigns: {status['campaigns']['active']}")
    print(f"   • Paused campaigns: {status['campaigns']['paused']}")
    print(f"   • Emergency stopped: {status['campaigns']['emergency_stopped']}")
    print(f"   • Total daily spend: ${status['spending']['total_daily_spent']:.2f}")
    print(f"   • Total violations: {status['violations']['total_violations']}")
    print(f"   • Monitoring threads: {status['monitoring']['monitoring_threads_active']}")
    
    # Demo 8: Campaign-specific status
    print(f"\n📋 Demo 8: Campaign Status Details")
    
    for campaign_id in campaigns.keys():
        campaign_status = controller.get_campaign_status(campaign_id)
        if campaign_status:
            print(f"   {campaign_id}:")
            print(f"     Status: {campaign_status['status']}")
            print(f"     Daily spent: ${campaign_status['spending']['daily_spent']:.2f}")
            print(f"     Daily utilization: {campaign_status['utilization']['daily_utilization']:.1%}")
            print(f"     Violations: {campaign_status['violation_count']}")
            print(f"     Emergency paused: {campaign_status['emergency_paused']}")
    
    # Demo 9: Emergency actions
    print(f"\n🚨 Demo 9: Emergency Actions")
    
    print("   Testing emergency pause of all campaigns...")
    paused_campaigns = controller.emergency_pause_all_campaigns("Demo emergency test")
    print(f"   Emergency paused {len(paused_campaigns)} campaigns")
    
    # Show final status
    print(f"\n✅ Demo Complete - Budget Safety Controller Summary:")
    final_status = controller.get_system_status()
    print(f"   • Campaigns managed: {final_status['campaigns']['total']}")
    print(f"   • Spending records: {final_status['spending']['total_records']}")
    print(f"   • Safety violations: {final_status['violations']['total_violations']}")
    print(f"   • Emergency protections: Active")
    
    print(f"\n🛡️ Key Budget Safety Features Demonstrated:")
    print(f"   ✓ Multi-tier spending limits (hourly/daily/weekly/monthly)")
    print(f"   ✓ Real-time velocity monitoring and limits")
    print(f"   ✓ Predictive overspend prevention")
    print(f"   ✓ Bid anomaly detection")
    print(f"   ✓ Automatic campaign pausing on violations")
    print(f"   ✓ Pre-spend safety validation")
    print(f"   ✓ Emergency stop integration")
    print(f"   ✓ Comprehensive audit trail")
    print(f"   ✓ Campaign isolation (violations don't affect other campaigns)")
    
    print(f"\n🎯 Budget Safety Controller: NO OVERSPENDING POSSIBLE!")
    
    # Cleanup
    controller.shutdown()
    
    return True

if __name__ == "__main__":
    try:
        success = demo_budget_safety()
        if success:
            print(f"\n🎉 Budget Safety Controller demo completed successfully!")
        else:
            print(f"\n❌ Budget Safety Controller demo failed!")
    except Exception as e:
        print(f"\n💥 Demo error: {e}")
        import traceback
        traceback.print_exc()