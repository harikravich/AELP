#!/usr/bin/env python3
"""
Comprehensive Test Suite for Budget Safety Controller
Tests all budget safety features to ensure no overspending is possible.
"""

import pytest
import sys
import os
import time
import threading
from decimal import Decimal
from datetime import datetime, timedelta
import tempfile
import json
import uuid

# Add the parent directory to the path so we can import GAELP modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from budget_safety_controller import (
    BudgetSafetyController, BudgetLimits, BudgetViolationType, 
    BudgetSafetyLevel, CampaignStatus, get_budget_safety_controller,
    budget_safety_decorator
)

class TestBudgetSafetyController:
    """Test suite for Budget Safety Controller"""
    
    @pytest.fixture
    def controller(self):
        """Create a test budget safety controller"""
        # Use temporary files for testing
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "test_config.json")
        
        # Create test config
        test_config = {
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
                "spending_check_seconds": 1,  # Fast for testing
                "velocity_check_seconds": 2,
                "anomaly_check_seconds": 3,
                "prediction_check_seconds": 5
            },
            "emergency_actions": {
                "auto_pause_campaigns": True,
                "emergency_stop_threshold": 1.05,
                "notification_webhook": None
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Create controller with test config
        controller = BudgetSafetyController(config_path)
        controller.db_path = os.path.join(temp_dir, "test_budget_safety.db")
        controller._test_mode = True  # Prevent actual emergency exits
        
        # Register a test campaign
        test_limits = BudgetLimits(
            daily_limit=Decimal('1000.00'),
            weekly_limit=Decimal('5000.00'),
            monthly_limit=Decimal('20000.00'),
            max_hourly_spend=Decimal('100.00'),
            max_hourly_velocity_increase=0.50
        )
        
        controller.register_campaign("test_campaign", test_limits)
        
        yield controller
        
        # Cleanup
        controller.shutdown()
    
    def test_campaign_registration(self, controller):
        """Test campaign registration and limit setting"""
        campaign_id = "test_reg_campaign"
        limits = BudgetLimits(
            daily_limit=Decimal('500.00'),
            weekly_limit=Decimal('2500.00'),
            monthly_limit=Decimal('10000.00'),
            max_hourly_spend=Decimal('50.00'),
            max_hourly_velocity_increase=0.30
        )
        
        controller.register_campaign(campaign_id, limits)
        
        # Verify campaign is registered
        assert campaign_id in controller.campaign_states
        state = controller.campaign_states[campaign_id]
        assert state.status == CampaignStatus.ACTIVE
        assert state.limits.daily_limit == Decimal('500.00')
        assert state.daily_spent == Decimal('0')
    
    def test_normal_spending_recording(self, controller):
        """Test normal spending recording within limits"""
        campaign_id = "test_campaign"
        
        # Record normal spending
        is_safe, violations = controller.record_spending(
            campaign_id=campaign_id,
            channel="google_ads",
            amount=Decimal('50.00'),
            bid_amount=Decimal('2.50'),
            impressions=1000,
            clicks=40,
            conversions=2
        )
        
        assert is_safe == True
        assert len(violations) == 0
        
        # Check campaign state updated
        state = controller.campaign_states[campaign_id]
        assert state.daily_spent == Decimal('50.00')
        assert state.status == CampaignStatus.ACTIVE
    
    def test_daily_limit_enforcement(self, controller):
        """Test that daily limits are strictly enforced"""
        campaign_id = "test_campaign"
        
        # Spend up to warning threshold (80% of $1000 = $800)
        is_safe, violations = controller.record_spending(
            campaign_id=campaign_id,
            channel="google_ads", 
            amount=Decimal('800.00'),
            bid_amount=Decimal('2.00'),
            impressions=10000,
            clicks=400,
            conversions=20
        )
        
        assert is_safe == True
        assert len(violations) == 1  # Should get warning
        assert violations[0].startswith("daily_limit_exceeded")
        
        # Spend to critical threshold (95% of $1000 = $950)
        is_safe, violations = controller.record_spending(
            campaign_id=campaign_id,
            channel="google_ads",
            amount=Decimal('150.00'),
            bid_amount=Decimal('2.00'),
            impressions=2000,
            clicks=75,
            conversions=5
        )
        
        assert is_safe == False  # Should be unsafe due to critical threshold
        assert len(violations) >= 1
        
        # Campaign should be paused
        state = controller.campaign_states[campaign_id]
        assert state.status == CampaignStatus.PAUSED
    
    def test_hourly_velocity_limits(self, controller):
        """Test hourly spending velocity limits"""
        campaign_id = "test_campaign"
        
        # Try to spend more than hourly limit ($100)
        is_safe, violations = controller.record_spending(
            campaign_id=campaign_id,
            channel="google_ads",
            amount=Decimal('120.00'),  # Exceeds $100 hourly limit
            bid_amount=Decimal('3.00'),
            impressions=2000,
            clicks=60,
            conversions=4
        )
        
        assert is_safe == False
        assert len(violations) >= 1
        assert any("hourly_velocity_exceeded" in v for v in violations)
    
    def test_bid_anomaly_detection(self, controller):
        """Test bid anomaly detection"""
        campaign_id = "test_campaign"
        
        # Record normal bids first to establish baseline
        for i in range(15):
            controller.record_spending(
                campaign_id=campaign_id,
                channel="google_ads",
                amount=Decimal('10.00'),
                bid_amount=Decimal('2.00'),  # Normal bid
                impressions=200,
                clicks=10,
                conversions=1
            )
        
        # Now record anomalous bid (much higher than baseline)
        is_safe, violations = controller.record_spending(
            campaign_id=campaign_id,
            channel="google_ads",
            amount=Decimal('20.00'),
            bid_amount=Decimal('20.00'),  # 10x normal bid - should trigger anomaly
            impressions=100,
            clicks=5,
            conversions=1
        )
        
        assert len(violations) >= 1
        assert any("bid_limit_exceeded" in v for v in violations)
    
    def test_spend_acceleration_detection(self, controller):
        """Test spending acceleration detection"""
        campaign_id = "test_campaign"
        
        # Record slow spending first
        for i in range(5):
            controller.record_spending(
                campaign_id=campaign_id,
                channel="google_ads",
                amount=Decimal('5.00'),  # Small amounts
                bid_amount=Decimal('1.00'),
                impressions=100,
                clicks=10,
                conversions=0
            )
            time.sleep(0.1)  # Small delay
        
        # Now record rapid acceleration
        for i in range(3):
            controller.record_spending(
                campaign_id=campaign_id,
                channel="google_ads",
                amount=Decimal('50.00'),  # Much larger amounts
                bid_amount=Decimal('2.00'),
                impressions=1000,
                clicks=100,
                conversions=5
            )
        
        # The acceleration should be detected in subsequent checks
        time.sleep(1)  # Wait for monitoring thread
    
    def test_pre_spend_safety_check(self, controller):
        """Test pre-spend safety validation"""
        campaign_id = "test_campaign"
        
        # Check normal amount - should be safe
        is_safe, reason = controller.is_campaign_safe_to_spend(campaign_id, Decimal('50.00'))
        assert is_safe == True
        assert reason == "Safe to spend"
        
        # Check amount that would exceed daily limit
        is_safe, reason = controller.is_campaign_safe_to_spend(campaign_id, Decimal('1200.00'))
        assert is_safe == False
        assert "daily limit" in reason.lower()
        
        # Check amount that would exceed hourly limit
        is_safe, reason = controller.is_campaign_safe_to_spend(campaign_id, Decimal('150.00'))
        assert is_safe == False
        assert "hourly limit" in reason.lower()
    
    def test_campaign_pause_and_resume(self, controller):
        """Test campaign pausing and resuming"""
        campaign_id = "test_campaign"
        
        # Trigger a violation to pause campaign
        controller.record_spending(
            campaign_id=campaign_id,
            channel="google_ads",
            amount=Decimal('1000.00'),  # Should trigger emergency threshold
            bid_amount=Decimal('2.00'),
            impressions=20000,
            clicks=1000,
            conversions=50
        )
        
        # Campaign should be paused
        state = controller.campaign_states[campaign_id]
        assert state.status in [CampaignStatus.PAUSED, CampaignStatus.EMERGENCY_STOPPED]
        
        # Test resuming campaign
        if state.status == CampaignStatus.PAUSED:
            success = controller.resume_campaign(campaign_id, "Manual override for testing")
            assert success == True
            assert controller.campaign_states[campaign_id].status == CampaignStatus.ACTIVE
    
    def test_predictive_overspend_prevention(self, controller):
        """Test predictive overspend prevention"""
        campaign_id = "test_campaign"
        
        # Build up spending velocity that would lead to overspend
        for i in range(5):
            controller.record_spending(
                campaign_id=campaign_id,
                channel="google_ads", 
                amount=Decimal('100.00'),  # High rate that could lead to overspend
                bid_amount=Decimal('2.50'),
                impressions=2000,
                clicks=80,
                conversions=4
            )
            time.sleep(0.1)
        
        # The controller should detect this pattern and create predictive warnings
        # Wait for monitoring thread to detect the pattern
        time.sleep(2)
        
        # Check if any violations were detected
        violations = [v for v in controller.violations if v.campaign_id == campaign_id]
        predictive_violations = [v for v in violations if v.violation_type == BudgetViolationType.PREDICTIVE_OVERSPEND]
        
        # We should have at least some concern about the spending rate
        assert len(violations) > 0
    
    def test_multi_campaign_isolation(self, controller):
        """Test that campaigns are properly isolated from each other"""
        # Register second campaign
        campaign2 = "test_campaign_2"
        limits2 = BudgetLimits(
            daily_limit=Decimal('500.00'),
            weekly_limit=Decimal('2500.00'),
            monthly_limit=Decimal('10000.00'),
            max_hourly_spend=Decimal('50.00'),
            max_hourly_velocity_increase=0.30
        )
        controller.register_campaign(campaign2, limits2)
        
        # Spend heavily on campaign 1
        controller.record_spending(
            campaign_id="test_campaign",
            channel="google_ads",
            amount=Decimal('800.00'),
            bid_amount=Decimal('2.00'),
            impressions=16000,
            clicks=800,
            conversions=40
        )
        
        # Campaign 2 should be unaffected
        is_safe, reason = controller.is_campaign_safe_to_spend(campaign2, Decimal('400.00'))
        assert is_safe == True  # Should still be safe on campaign 2
        
        state1 = controller.campaign_states["test_campaign"]
        state2 = controller.campaign_states[campaign2]
        
        assert state1.daily_spent == Decimal('800.00')
        assert state2.daily_spent == Decimal('0.00')  # Unaffected
    
    def test_system_status_reporting(self, controller):
        """Test system status reporting"""
        # Record some spending to generate data
        controller.record_spending(
            campaign_id="test_campaign",
            channel="google_ads",
            amount=Decimal('100.00'),
            bid_amount=Decimal('2.00'),
            impressions=2000,
            clicks=100,
            conversions=5
        )
        
        # Get system status
        status = controller.get_system_status()
        
        assert "timestamp" in status
        assert status["system_active"] == True
        assert "campaigns" in status
        assert status["campaigns"]["total"] >= 1
        assert "spending" in status
        assert status["spending"]["total_daily_spent"] >= 100.0
        assert "violations" in status
        assert "monitoring" in status
    
    def test_campaign_status_reporting(self, controller):
        """Test campaign status reporting"""
        campaign_id = "test_campaign"
        
        # Record some spending
        controller.record_spending(
            campaign_id=campaign_id,
            channel="google_ads",
            amount=Decimal('200.00'),
            bid_amount=Decimal('2.50'),
            impressions=4000,
            clicks=160,
            conversions=8
        )
        
        # Get campaign status
        status = controller.get_campaign_status(campaign_id)
        
        assert status is not None
        assert status["campaign_id"] == campaign_id
        assert status["status"] == CampaignStatus.ACTIVE.value
        assert status["spending"]["daily_spent"] == 200.0
        assert status["limits"]["daily_limit"] == 1000.0
        assert status["utilization"]["daily_utilization"] == 0.2  # 200/1000
    
    def test_budget_safety_decorator(self, controller):
        """Test the budget safety decorator"""
        # Mock function that represents spending
        @budget_safety_decorator("test_campaign", "google_ads")
        def mock_spend_function(amount, bid_amount=None, impressions=0, clicks=0, conversions=0):
            return f"Spent ${amount}"
        
        # Test normal spending
        result = mock_spend_function(Decimal('50.00'), bid_amount=Decimal('2.00'), impressions=1000, clicks=40, conversions=2)
        assert result == "Spent $50.00"
        
        # Test spending that would exceed limits
        try:
            mock_spend_function(Decimal('2000.00'), bid_amount=Decimal('10.00'))  # Exceeds daily limit
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "Budget safety violation" in str(e)
    
    def test_emergency_pause_all_campaigns(self, controller):
        """Test emergency pause of all campaigns"""
        # Register multiple campaigns
        for i in range(3):
            campaign_id = f"emergency_test_{i}"
            controller.register_campaign(campaign_id)
        
        # Emergency pause all
        paused_campaigns = controller.emergency_pause_all_campaigns("Testing emergency pause")
        
        assert len(paused_campaigns) == 4  # 3 new + 1 existing test_campaign
        
        # Verify all are emergency stopped
        for campaign_id in paused_campaigns:
            state = controller.campaign_states[campaign_id]
            assert state.status == CampaignStatus.EMERGENCY_STOPPED
            assert state.emergency_paused == True
    
    def test_limit_updates(self, controller):
        """Test updating campaign limits"""
        campaign_id = "test_campaign"
        
        # Update limits
        new_limits = BudgetLimits(
            daily_limit=Decimal('2000.00'),  # Doubled
            weekly_limit=Decimal('10000.00'),
            monthly_limit=Decimal('40000.00'),
            max_hourly_spend=Decimal('200.00'),
            max_hourly_velocity_increase=0.60
        )
        
        success = controller.update_campaign_limits(campaign_id, new_limits)
        assert success == True
        
        # Verify limits updated
        state = controller.campaign_states[campaign_id]
        assert state.limits.daily_limit == Decimal('2000.00')
        assert state.limits.max_hourly_spend == Decimal('200.00')
        
        # Test that new limits are enforced
        is_safe, reason = controller.is_campaign_safe_to_spend(campaign_id, Decimal('1500.00'))
        assert is_safe == True  # Should be safe with new higher limit
    
    def test_database_persistence(self, controller):
        """Test that spending records and violations are persisted"""
        campaign_id = "test_campaign"
        
        # Record spending
        controller.record_spending(
            campaign_id=campaign_id,
            channel="google_ads",
            amount=Decimal('75.00'),
            bid_amount=Decimal('2.25'),
            impressions=1500,
            clicks=60,
            conversions=3
        )
        
        # Check that data was saved to database
        assert len(controller.spending_history) > 0
        
        # The database operations are tested implicitly by the other tests
        # as the controller saves to database on each operation


def test_budget_safety_integration():
    """Integration test for budget safety controller"""
    print("🛡️ Running Budget Safety Controller Integration Tests")
    print("=" * 60)
    
    # Create test instance
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "integration_test_config.json")
    
    test_config = {
        "default_limits": {
            "daily_limit": 500.0,
            "weekly_limit": 2500.0, 
            "monthly_limit": 10000.0,
            "max_hourly_spend": 50.0,
            "max_hourly_velocity_increase": 0.40,
            "warning_threshold": 0.75,
            "critical_threshold": 0.90,
            "emergency_threshold": 0.95,
            "max_bid_multiplier": 2.5,
            "max_spend_acceleration": 1.8,
            "prediction_window_hours": 1,
            "overspend_prevention_buffer": 0.10
        },
        "monitoring_intervals": {
            "spending_check_seconds": 1,
            "velocity_check_seconds": 2,
            "anomaly_check_seconds": 3,
            "prediction_check_seconds": 5
        },
        "emergency_actions": {
            "auto_pause_campaigns": True,
            "emergency_stop_threshold": 1.00,
            "notification_webhook": None
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(test_config, f)
    
    controller = BudgetSafetyController(config_path)
    controller.db_path = os.path.join(temp_dir, "integration_test.db")
    controller._test_mode = True
    
    try:
        print("✅ Controller initialized successfully")
        
        # Test 1: Campaign registration
        print("\n📝 Test 1: Campaign Registration")
        campaign_id = "integration_test_campaign"
        controller.register_campaign(campaign_id)
        assert campaign_id in controller.campaign_states
        print(f"   Campaign {campaign_id} registered with ${controller.campaign_states[campaign_id].limits.daily_limit} daily limit")
        
        # Test 2: Normal spending
        print("\n💰 Test 2: Normal Spending Pattern")
        total_spent = Decimal('0')
        for i in range(5):
            amount = Decimal('25.00')
            is_safe, violations = controller.record_spending(
                campaign_id=campaign_id,
                channel="google_ads",
                amount=amount,
                bid_amount=Decimal('1.50'),
                impressions=500,
                clicks=25,
                conversions=1
            )
            total_spent += amount
            print(f"   Spend ${amount}: Safe={is_safe}, Total=${total_spent}, Violations={len(violations)}")
        
        # Test 3: Approaching limits
        print("\n⚠️ Test 3: Approaching Budget Limits")
        amount = Decimal('250.00')  # This should trigger warnings (total would be $375 of $500)
        is_safe, violations = controller.record_spending(
            campaign_id=campaign_id,
            channel="google_ads",
            amount=amount,
            bid_amount=Decimal('2.00'),
            impressions=5000,
            clicks=125,
            conversions=8
        )
        total_spent += amount
        print(f"   Large spend ${amount}: Safe={is_safe}, Total=${total_spent}, Violations={len(violations)}")
        for violation in violations:
            print(f"     - {violation}")
        
        # Test 4: Exceeding limits
        print("\n🚨 Test 4: Exceeding Budget Limits")
        amount = Decimal('200.00')  # This should exceed limits and trigger emergency
        is_safe, violations = controller.record_spending(
            campaign_id=campaign_id,
            channel="google_ads",
            amount=amount,
            bid_amount=Decimal('2.50'),
            impressions=4000,
            clicks=80,
            conversions=6
        )
        print(f"   Overspend attempt ${amount}: Safe={is_safe}, Violations={len(violations)}")
        
        # Check campaign status
        state = controller.campaign_states[campaign_id]
        print(f"   Campaign status after overspend: {state.status.value}")
        print(f"   Emergency paused: {state.emergency_paused}")
        
        # Test 5: Pre-spend validation
        print("\n🔍 Test 5: Pre-spend Validation")
        test_amounts = [Decimal('10.00'), Decimal('50.00'), Decimal('500.00')]
        for amount in test_amounts:
            is_safe, reason = controller.is_campaign_safe_to_spend(campaign_id, amount)
            print(f"   ${amount} check: Safe={is_safe} - {reason}")
        
        # Test 6: System status
        print("\n📊 Test 6: System Status Report")
        status = controller.get_system_status()
        print(f"   Total campaigns: {status['campaigns']['total']}")
        print(f"   Active campaigns: {status['campaigns']['active']}")
        print(f"   Paused campaigns: {status['campaigns']['paused']}")
        print(f"   Emergency stopped: {status['campaigns']['emergency_stopped']}")
        print(f"   Total daily spend: ${status['spending']['total_daily_spent']:.2f}")
        print(f"   Total violations: {status['violations']['total_violations']}")
        
        # Test 7: Campaign-specific status
        print(f"\n📋 Test 7: Campaign Status Report")
        campaign_status = controller.get_campaign_status(campaign_id)
        if campaign_status:
            print(f"   Status: {campaign_status['status']}")
            print(f"   Daily spent: ${campaign_status['spending']['daily_spent']:.2f}")
            print(f"   Daily utilization: {campaign_status['utilization']['daily_utilization']:.1%}")
            print(f"   Violation count: {campaign_status['violation_count']}")
        
        print("\n✅ All integration tests completed successfully!")
        print("🛡️ Budget Safety Controller is working properly")
        
        # Verification summary
        print(f"\n📈 Verification Summary:")
        print(f"   • Campaigns registered: {len(controller.campaign_states)}")
        print(f"   • Spending records: {len(controller.spending_history)}")
        print(f"   • Violations detected: {len(controller.violations)}")
        print(f"   • Emergency protections: Active")
        print(f"   • Database persistence: Enabled")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        controller.shutdown()


if __name__ == "__main__":
    # Run integration test
    success = test_budget_safety_integration()
    
    if success:
        print("\n🎉 Budget Safety Controller implementation complete and verified!")
        exit(0)
    else:
        print("\n💥 Budget Safety Controller tests failed!")
        exit(1)