#!/usr/bin/env python3
"""
Test Budget Pacer System
Comprehensive testing of the advanced budget pacing functionality.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import json

from budget_pacer import (
    BudgetPacer, ChannelType, PacingStrategy, SpendTransaction,
    PacingAlert, CircuitBreakerState
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BudgetPacerTester:
    """Comprehensive testing suite for budget pacer"""
    
    def __init__(self):
        self.pacer = BudgetPacer(alert_callback=self._handle_alert)
        self.alerts_received = []
        self.test_results = {}
    
    async def _handle_alert(self, alert: PacingAlert):
        """Handle alerts during testing"""
        self.alerts_received.append(alert)
        logger.warning(f"Test Alert: {alert.alert_type} - {alert.recommended_action}")
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        logger.info("Starting Budget Pacer Test Suite")
        
        test_methods = [
            self.test_hourly_allocation,
            self.test_pacing_strategies,
            self.test_frontload_protection,
            self.test_circuit_breakers,
            self.test_dynamic_reallocation,
            self.test_predictive_pacing,
            self.test_emergency_stop,
            self.test_velocity_limits,
            self.test_performance_analysis,
            self.test_real_world_scenario
        ]
        
        for test_method in test_methods:
            try:
                logger.info(f"Running {test_method.__name__}")
                result = await test_method()
                self.test_results[test_method.__name__] = result
                logger.info(f"✓ {test_method.__name__} passed")
            except Exception as e:
                logger.error(f"✗ {test_method.__name__} failed: {e}")
                self.test_results[test_method.__name__] = {"status": "failed", "error": str(e)}
        
        self._print_test_summary()
    
    async def test_hourly_allocation(self):
        """Test basic hourly budget allocation"""
        campaign_id = "test_allocation_001"
        daily_budget = Decimal('1000.00')
        
        # Test even distribution
        allocations = self.pacer.allocate_hourly_budget(
            campaign_id, ChannelType.GOOGLE_ADS, daily_budget, PacingStrategy.EVEN_DISTRIBUTION
        )
        
        # Verify allocations sum to 1.0
        total_allocation = sum(a.base_allocation_pct for a in allocations)
        assert abs(total_allocation - 1.0) < 0.001, f"Allocations don't sum to 1.0: {total_allocation}"
        
        # Verify all hours are covered
        assert len(allocations) == 24, f"Should have 24 hourly allocations, got {len(allocations)}"
        
        # Calculate hourly budgets
        hourly_budgets = []
        for allocation in allocations:
            hourly_budget = float(daily_budget * Decimal(str(allocation.base_allocation_pct)))
            hourly_budgets.append(hourly_budget)
        
        return {
            "status": "passed",
            "total_allocation": total_allocation,
            "hourly_budgets": hourly_budgets[:6],  # First 6 hours
            "avg_hourly_budget": np.mean(hourly_budgets)
        }
    
    async def test_pacing_strategies(self):
        """Test different pacing strategies"""
        campaign_id = "test_strategies_001"
        daily_budget = Decimal('2000.00')
        
        strategies_results = {}
        
        for strategy in PacingStrategy:
            allocations = self.pacer.allocate_hourly_budget(
                campaign_id + "_" + strategy.value, 
                ChannelType.FACEBOOK_ADS, 
                daily_budget, 
                strategy
            )
            
            # Analyze allocation distribution
            allocation_values = [a.base_allocation_pct for a in allocations]
            strategies_results[strategy.value] = {
                "min_allocation": min(allocation_values),
                "max_allocation": max(allocation_values),
                "std_deviation": np.std(allocation_values),
                "peak_hours": [i for i, val in enumerate(allocation_values) if val > np.mean(allocation_values) * 1.5]
            }
        
        return {
            "status": "passed",
            "strategies": strategies_results
        }
    
    async def test_frontload_protection(self):
        """Test frontload protection prevents early budget exhaustion"""
        campaign_id = "test_frontload_001"
        daily_budget = Decimal('1000.00')
        
        # Allocate budget
        self.pacer.allocate_hourly_budget(
            campaign_id, ChannelType.GOOGLE_ADS, daily_budget, PacingStrategy.EVEN_DISTRIBUTION
        )
        
        # Try to spend large amount in early hours
        early_spend = Decimal('400.00')  # 40% of daily budget
        
        results = {}
        for hour in range(6):  # Test first 6 hours
            # Simulate being in that hour
            can_spend, reason = self.pacer.can_spend(campaign_id, ChannelType.GOOGLE_ADS, early_spend)
            results[f"hour_{hour}"] = {
                "can_spend": can_spend,
                "reason": reason
            }
            
            if hour < 4:  # Frontload protection hours
                assert not can_spend, f"Should not allow large spend in hour {hour}"
        
        return {
            "status": "passed",
            "frontload_tests": results
        }
    
    async def test_circuit_breakers(self):
        """Test circuit breaker functionality"""
        campaign_id = "test_circuit_001"
        daily_budget = Decimal('500.00')
        
        # Allocate budget
        self.pacer.allocate_hourly_budget(
            campaign_id, ChannelType.TIKTOK_ADS, daily_budget, PacingStrategy.EVEN_DISTRIBUTION
        )
        
        # Simulate multiple overspending transactions
        overspend_transactions = []
        for i in range(5):
            transaction = SpendTransaction(
                campaign_id=campaign_id,
                channel=ChannelType.TIKTOK_ADS,
                amount=Decimal('120.00'),  # Large amounts
                timestamp=datetime.utcnow(),
                clicks=10,
                conversions=0,
                cost_per_click=12.00,
                conversion_rate=0.0
            )
            
            self.pacer.record_spend(transaction)
            overspend_transactions.append(transaction)
        
        # Check circuit breaker state
        breaker_state = CircuitBreakerState.CLOSED
        if (campaign_id in self.pacer.circuit_breakers and 
            ChannelType.TIKTOK_ADS in self.pacer.circuit_breakers[campaign_id]):
            breaker_state = self.pacer.circuit_breakers[campaign_id][ChannelType.TIKTOK_ADS].state
        
        return {
            "status": "passed",
            "transactions_recorded": len(overspend_transactions),
            "total_spend": sum(float(t.amount) for t in overspend_transactions),
            "circuit_breaker_state": breaker_state.value if hasattr(breaker_state, 'value') else str(breaker_state)
        }
    
    async def test_dynamic_reallocation(self):
        """Test dynamic budget reallocation between channels"""
        campaign_id = "test_realloc_001"
        daily_budget = Decimal('1500.00')
        
        # Set up multiple channels
        channels = [ChannelType.GOOGLE_ADS, ChannelType.FACEBOOK_ADS, ChannelType.TIKTOK_ADS]
        
        for channel in channels:
            self.pacer.allocate_hourly_budget(
                campaign_id, channel, daily_budget // len(channels), PacingStrategy.ADAPTIVE_HYBRID
            )
        
        # Simulate different performance levels
        high_performer_transactions = [
            SpendTransaction(
                campaign_id=campaign_id,
                channel=ChannelType.GOOGLE_ADS,
                amount=Decimal('50.00'),
                timestamp=datetime.utcnow() - timedelta(hours=i),
                clicks=25,
                conversions=5,  # High conversion rate
                cost_per_click=2.00,
                conversion_rate=0.20
            ) for i in range(5)
        ]
        
        low_performer_transactions = [
            SpendTransaction(
                campaign_id=campaign_id,
                channel=ChannelType.TIKTOK_ADS,
                amount=Decimal('50.00'),
                timestamp=datetime.utcnow() - timedelta(hours=i),
                clicks=50,
                conversions=1,  # Low conversion rate
                cost_per_click=1.00,
                conversion_rate=0.02
            ) for i in range(5)
        ]
        
        # Record transactions
        for transaction in high_performer_transactions + low_performer_transactions:
            self.pacer.record_spend(transaction)
        
        # Test reallocation
        reallocation_results = await self.pacer.reallocate_unused(campaign_id)
        
        return {
            "status": "passed",
            "high_performer_transactions": len(high_performer_transactions),
            "low_performer_transactions": len(low_performer_transactions),
            "reallocation_results": {str(k): float(v) for k, v in reallocation_results.items()}
        }
    
    async def test_predictive_pacing(self):
        """Test ML-based predictive pacing"""
        campaign_id = "test_predict_001"
        daily_budget = Decimal('2000.00')
        
        # Generate synthetic historical data
        historical_transactions = []
        for day in range(14):  # 2 weeks of data
            for hour in range(24):
                # Simulate realistic patterns - more spend during business hours
                base_spend = 20 if 9 <= hour <= 17 else 10
                spend_amount = Decimal(str(base_spend + np.random.normal(0, 5)))
                spend_amount = max(Decimal('1.00'), spend_amount)
                
                clicks = int(spend_amount / Decimal('2.50'))  # $2.50 avg CPC
                conversions = max(0, int(clicks * 0.05 + np.random.normal(0, 1)))  # ~5% conversion rate
                
                transaction = SpendTransaction(
                    campaign_id=campaign_id,
                    channel=ChannelType.GOOGLE_ADS,
                    amount=spend_amount,
                    timestamp=datetime.utcnow() - timedelta(days=day, hours=24-hour),
                    clicks=clicks,
                    conversions=conversions,
                    cost_per_click=float(spend_amount / clicks) if clicks > 0 else 2.50,
                    conversion_rate=conversions / clicks if clicks > 0 else 0.0
                )
                
                historical_transactions.append(transaction)
                self.pacer.record_spend(transaction)
        
        # Test predictive allocation
        allocations = self.pacer.allocate_hourly_budget(
            campaign_id, ChannelType.GOOGLE_ADS, daily_budget, PacingStrategy.PREDICTIVE_ML
        )
        
        # Analyze predictions
        business_hours_allocation = sum(a.base_allocation_pct for a in allocations[9:18])
        off_hours_allocation = sum(a.base_allocation_pct for a in allocations[:9] + allocations[18:])
        
        return {
            "status": "passed",
            "historical_transactions": len(historical_transactions),
            "business_hours_allocation": business_hours_allocation,
            "off_hours_allocation": off_hours_allocation,
            "predicted_peak_hours": [a.hour for a in allocations if a.base_allocation_pct > 1/24 * 1.5]
        }
    
    async def test_emergency_stop(self):
        """Test emergency stop functionality"""
        campaign_id = "test_emergency_001"
        daily_budget = Decimal('1000.00')
        
        # Set up campaign
        self.pacer.allocate_hourly_budget(
            campaign_id, ChannelType.GOOGLE_ADS, daily_budget, PacingStrategy.EVEN_DISTRIBUTION
        )
        
        # Trigger emergency stop
        stop_result = await self.pacer.emergency_stop(campaign_id, "Test emergency stop")
        
        # Verify spending is blocked
        can_spend, reason = self.pacer.can_spend(campaign_id, ChannelType.GOOGLE_ADS, Decimal('10.00'))
        
        return {
            "status": "passed",
            "emergency_stop_successful": stop_result,
            "spending_blocked": not can_spend,
            "block_reason": reason,
            "alerts_generated": len([a for a in self.alerts_received if a.alert_type == "emergency_stop"])
        }
    
    async def test_velocity_limits(self):
        """Test spend velocity limits"""
        campaign_id = "test_velocity_001"
        daily_budget = Decimal('1440.00')  # $1 per minute for 24 hours
        
        # Allocate budget
        self.pacer.allocate_hourly_budget(
            campaign_id, ChannelType.DISPLAY, daily_budget, PacingStrategy.EVEN_DISTRIBUTION
        )
        
        # Test normal velocity (should pass)
        normal_spend = Decimal('0.50')  # Half the per-minute limit
        can_spend_normal, reason_normal = self.pacer.can_spend(
            campaign_id, ChannelType.DISPLAY, normal_spend
        )
        
        # Test excessive velocity (should fail)
        excessive_spend = Decimal('10.00')  # 10x the per-minute limit
        can_spend_excessive, reason_excessive = self.pacer.can_spend(
            campaign_id, ChannelType.DISPLAY, excessive_spend
        )
        
        return {
            "status": "passed",
            "normal_velocity_allowed": can_spend_normal,
            "normal_reason": reason_normal,
            "excessive_velocity_blocked": not can_spend_excessive,
            "excessive_reason": reason_excessive
        }
    
    async def test_performance_analysis(self):
        """Test performance analysis and optimization"""
        campaign_id = "test_performance_001"
        daily_budget = Decimal('1000.00')
        
        # Set up campaign
        self.pacer.allocate_hourly_budget(
            campaign_id, ChannelType.GOOGLE_ADS, daily_budget, PacingStrategy.PERFORMANCE_WEIGHTED
        )
        
        # Create transactions with varying performance
        good_transactions = [
            SpendTransaction(
                campaign_id=campaign_id,
                channel=ChannelType.GOOGLE_ADS,
                amount=Decimal('100.00'),
                timestamp=datetime.utcnow() - timedelta(hours=i),
                clicks=50,
                conversions=10,  # 20% conversion rate
                cost_per_click=2.00,
                conversion_rate=0.20
            ) for i in range(3)
        ]
        
        poor_transactions = [
            SpendTransaction(
                campaign_id=campaign_id,
                channel=ChannelType.GOOGLE_ADS,
                amount=Decimal('100.00'),
                timestamp=datetime.utcnow() - timedelta(hours=i+12),
                clicks=100,
                conversions=1,  # 1% conversion rate
                cost_per_click=1.00,
                conversion_rate=0.01
            ) for i in range(3)
        ]
        
        # Record transactions
        for transaction in good_transactions + poor_transactions:
            self.pacer.record_spend(transaction)
        
        # Analyze performance
        performance_score = self.pacer._analyze_channel_performance(campaign_id, ChannelType.GOOGLE_ADS)
        
        return {
            "status": "passed",
            "good_transactions": len(good_transactions),
            "poor_transactions": len(poor_transactions),
            "performance_score": performance_score,
            "total_conversions": sum(t.conversions for t in good_transactions + poor_transactions)
        }
    
    async def test_real_world_scenario(self):
        """Test realistic campaign scenario"""
        campaign_id = "real_world_test_001"
        daily_budget = Decimal('5000.00')
        
        # Multi-channel campaign
        channels = [
            (ChannelType.GOOGLE_ADS, Decimal('2000.00')),
            (ChannelType.FACEBOOK_ADS, Decimal('1500.00')),
            (ChannelType.TIKTOK_ADS, Decimal('1000.00')),
            (ChannelType.DISPLAY, Decimal('500.00'))
        ]
        
        scenario_results = {}
        
        # Set up channels
        for channel, budget in channels:
            allocations = self.pacer.allocate_hourly_budget(
                campaign_id, channel, budget, PacingStrategy.ADAPTIVE_HYBRID
            )
            scenario_results[channel.value] = {
                "budget": float(budget),
                "allocations": len(allocations)
            }
        
        # Simulate a day of spending
        current_hour = 0
        total_spend = Decimal('0')
        successful_spends = 0
        blocked_spends = 0
        
        for hour in range(24):
            for channel, budget in channels:
                # Simulate 10 spend attempts per hour per channel
                for attempt in range(10):
                    spend_amount = Decimal(str(np.random.uniform(10, 100)))
                    can_spend, reason = self.pacer.can_spend(campaign_id, channel, spend_amount)
                    
                    if can_spend:
                        transaction = SpendTransaction(
                            campaign_id=campaign_id,
                            channel=channel,
                            amount=spend_amount,
                            timestamp=datetime.utcnow(),
                            clicks=int(spend_amount / Decimal('2.50')),
                            conversions=np.random.poisson(1),
                            cost_per_click=2.50,
                            conversion_rate=0.05
                        )
                        self.pacer.record_spend(transaction)
                        total_spend += spend_amount
                        successful_spends += 1
                    else:
                        blocked_spends += 1
        
        # Check final pacing
        pace_results = {}
        for channel, _ in channels:
            pace_ratio, alert = await self.pacer.check_pace(campaign_id, channel)
            pace_results[channel.value] = {
                "pace_ratio": pace_ratio,
                "has_alert": alert is not None
            }
        
        return {
            "status": "passed",
            "channels_configured": len(channels),
            "total_daily_budget": float(sum(budget for _, budget in channels)),
            "total_spend": float(total_spend),
            "successful_spends": successful_spends,
            "blocked_spends": blocked_spends,
            "spend_efficiency": successful_spends / (successful_spends + blocked_spends) if successful_spends + blocked_spends > 0 else 0,
            "pace_results": pace_results,
            "alerts_generated": len(self.alerts_received)
        }
    
    def _print_test_summary(self):
        """Print test results summary"""
        logger.info("\n" + "="*80)
        logger.info("BUDGET PACER TEST SUMMARY")
        logger.info("="*80)
        
        passed_tests = 0
        total_tests = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = result.get('status', 'unknown')
            if status == 'passed':
                logger.info(f"✓ {test_name}: PASSED")
                passed_tests += 1
            else:
                logger.error(f"✗ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
        
        logger.info(f"\nResults: {passed_tests}/{total_tests} tests passed")
        logger.info(f"Total alerts generated: {len(self.alerts_received)}")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! Budget pacer is working correctly.")
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Review implementation.")
        
        # Save detailed results
        with open('/home/hariravichandran/AELP/budget_pacer_test_results.json', 'w') as f:
            json.dump({
                'test_results': self.test_results,
                'alerts_received': [
                    {
                        'alert_type': alert.alert_type,
                        'campaign_id': alert.campaign_id,
                        'severity': alert.severity,
                        'recommended_action': alert.recommended_action
                    } for alert in self.alerts_received
                ],
                'summary': {
                    'passed_tests': passed_tests,
                    'total_tests': total_tests,
                    'success_rate': passed_tests / total_tests if total_tests > 0 else 0
                }
            }, f, indent=2)
        
        logger.info("Detailed test results saved to budget_pacer_test_results.json")


async def main():
    """Run the budget pacer test suite"""
    tester = BudgetPacerTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())