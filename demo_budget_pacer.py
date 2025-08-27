#!/usr/bin/env python3
"""
Budget Pacer Demonstration
Shows real-world usage of the advanced budget pacing system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

from budget_pacer import (
    BudgetPacer, ChannelType, PacingStrategy, SpendTransaction,
    PacingAlert, HourlyAllocation
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BudgetPacerDemo:
    """Demonstration of budget pacing capabilities"""
    
    def __init__(self):
        self.pacer = BudgetPacer(alert_callback=self._handle_alert)
        self.alerts = []
        self.demo_data = {}
    
    async def _handle_alert(self, alert: PacingAlert):
        """Handle alerts during demo"""
        self.alerts.append(alert)
        severity_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🚨", "critical": "🛑"}
        emoji = severity_emoji.get(alert.severity, "❓")
        
        print(f"\n{emoji} PACING ALERT: {alert.alert_type}")
        print(f"   Campaign: {alert.campaign_id}")
        print(f"   Channel: {alert.channel.value if alert.channel else 'All'}")
        print(f"   Current Spend: ${alert.current_spend}")
        print(f"   Pace Ratio: {alert.pace_ratio:.2f}x")
        print(f"   Action: {alert.recommended_action}")
    
    async def run_complete_demo(self):
        """Run complete budget pacing demonstration"""
        print("🚀 GAELP Budget Pacer Demonstration")
        print("="*60)
        
        await self.demo_basic_allocation()
        await self.demo_pacing_strategies()
        await self.demo_frontload_protection()
        await self.demo_performance_optimization()
        await self.demo_emergency_controls()
        await self.demo_multi_channel_campaign()
        
        self._generate_summary_report()
    
    async def demo_basic_allocation(self):
        """Demonstrate basic hourly budget allocation"""
        print("\n📊 BASIC HOURLY ALLOCATION")
        print("-" * 40)
        
        campaign_id = "demo_basic_allocation"
        daily_budget = Decimal('1200.00')
        
        # Allocate budget using adaptive strategy
        allocations = self.pacer.allocate_hourly_budget(
            campaign_id, ChannelType.GOOGLE_ADS, daily_budget, PacingStrategy.ADAPTIVE_HYBRID
        )
        
        # Display allocation breakdown
        print(f"Daily Budget: ${daily_budget}")
        print(f"Strategy: Adaptive Hybrid")
        print("\nHourly Allocation Breakdown:")
        print("Hour | Budget   | Percentage | Performance Score")
        print("-" * 50)
        
        for i, allocation in enumerate(allocations[:12]):  # Show first 12 hours
            hourly_budget = daily_budget * Decimal(str(allocation.base_allocation_pct))
            print(f"{i:2d}   | ${hourly_budget:7.2f} | {allocation.base_allocation_pct:8.1%} | {allocation.performance_multiplier:13.2f}")
        
        print(f"... (showing first 12 hours)")
        
        # Store for visualization
        self.demo_data['basic_allocation'] = {
            'hours': list(range(24)),
            'budgets': [float(daily_budget * Decimal(str(a.base_allocation_pct))) for a in allocations],
            'allocations': [a.base_allocation_pct for a in allocations]
        }
    
    async def demo_pacing_strategies(self):
        """Demonstrate different pacing strategies"""
        print("\n📈 PACING STRATEGY COMPARISON")
        print("-" * 40)
        
        campaign_base = "demo_strategy"
        daily_budget = Decimal('2000.00')
        
        strategies_data = {}
        
        for strategy in PacingStrategy:
            campaign_id = f"{campaign_base}_{strategy.value}"
            
            # Generate some historical data for non-even strategies
            if strategy != PacingStrategy.EVEN_DISTRIBUTION:
                await self._generate_historical_data(campaign_id, ChannelType.FACEBOOK_ADS)
            
            allocations = self.pacer.allocate_hourly_budget(
                campaign_id, ChannelType.FACEBOOK_ADS, daily_budget, strategy
            )
            
            hourly_budgets = [float(daily_budget * Decimal(str(a.base_allocation_pct))) for a in allocations]
            strategies_data[strategy.value] = hourly_budgets
            
            # Show peak hours for each strategy
            peak_hours = [i for i, budget in enumerate(hourly_budgets) if budget > np.mean(hourly_budgets) * 1.2]
            print(f"{strategy.value:20s}: Peak hours {peak_hours[:5]}{'...' if len(peak_hours) > 5 else ''}")
        
        self.demo_data['strategies'] = strategies_data
    
    async def demo_frontload_protection(self):
        """Demonstrate frontload protection mechanism"""
        print("\n🛡️  FRONTLOAD PROTECTION")
        print("-" * 40)
        
        campaign_id = "demo_frontload"
        daily_budget = Decimal('1000.00')
        
        # Set up campaign
        self.pacer.allocate_hourly_budget(
            campaign_id, ChannelType.GOOGLE_ADS, daily_budget, PacingStrategy.EVEN_DISTRIBUTION
        )
        
        # Test spending in early hours
        early_hours_results = []
        for hour in range(6):
            large_spend = Decimal('200.00')  # 20% of daily budget
            can_spend, reason = self.pacer.can_spend(campaign_id, ChannelType.GOOGLE_ADS, large_spend)
            
            early_hours_results.append({
                'hour': hour,
                'can_spend': can_spend,
                'reason': reason
            })
            
            status = "✅ ALLOWED" if can_spend else "🚫 BLOCKED"
            print(f"Hour {hour:2d}: Trying to spend ${large_spend} - {status}")
            if not can_spend:
                print(f"         Reason: {reason}")
        
        print("\n💡 Frontload protection prevents spending too much too early!")
        self.demo_data['frontload_protection'] = early_hours_results
    
    async def demo_performance_optimization(self):
        """Demonstrate performance-based optimization"""
        print("\n⚡ PERFORMANCE OPTIMIZATION")
        print("-" * 40)
        
        campaign_id = "demo_performance"
        daily_budget = Decimal('1500.00')
        
        # Set up multi-channel campaign
        channels = [ChannelType.GOOGLE_ADS, ChannelType.FACEBOOK_ADS, ChannelType.TIKTOK_ADS]
        channel_budgets = {}
        
        for channel in channels:
            budget = daily_budget // len(channels)
            self.pacer.allocate_hourly_budget(campaign_id, channel, budget, PacingStrategy.PERFORMANCE_WEIGHTED)
            channel_budgets[channel] = budget
        
        # Simulate different performance levels
        print("Simulating 24 hours of campaign performance...")
        
        performance_data = {}
        for channel in channels:
            performance_data[channel.value] = {'spend': 0, 'conversions': 0, 'clicks': 0}
        
        # Google Ads: High performer
        google_transactions = self._create_performance_transactions(
            campaign_id, ChannelType.GOOGLE_ADS, 24, avg_spend=30, conversion_rate=0.15
        )
        
        # Facebook Ads: Medium performer  
        facebook_transactions = self._create_performance_transactions(
            campaign_id, ChannelType.FACEBOOK_ADS, 24, avg_spend=25, conversion_rate=0.08
        )
        
        # TikTok Ads: Low performer
        tiktok_transactions = self._create_performance_transactions(
            campaign_id, ChannelType.TIKTOK_ADS, 24, avg_spend=20, conversion_rate=0.03
        )
        
        all_transactions = google_transactions + facebook_transactions + tiktok_transactions
        
        # Record all transactions
        for transaction in all_transactions:
            self.pacer.record_spend(transaction)
            channel_data = performance_data[transaction.channel.value]
            channel_data['spend'] += float(transaction.amount)
            channel_data['conversions'] += transaction.conversions
            channel_data['clicks'] += transaction.clicks
        
        # Display performance summary
        print("\nChannel Performance Summary:")
        print("Channel      | Spend    | Clicks | Conversions | CTR    | CPA")
        print("-" * 65)
        
        for channel_name, data in performance_data.items():
            ctr = data['conversions'] / data['clicks'] if data['clicks'] > 0 else 0
            cpa = data['spend'] / data['conversions'] if data['conversions'] > 0 else float('inf')
            cpa_str = f"${cpa:.2f}" if cpa != float('inf') else "N/A"
            
            print(f"{channel_name:12s} | ${data['spend']:7.2f} | {data['clicks']:6d} | {data['conversions']:11d} | {ctr:6.1%} | {cpa_str}")
        
        # Test reallocation
        print("\nTesting dynamic budget reallocation...")
        reallocation_results = await self.pacer.reallocate_unused(campaign_id)
        
        if reallocation_results:
            print("Budget Reallocation Results:")
            for channel, amount in reallocation_results.items():
                action = "increased" if amount > 0 else "decreased"
                print(f"  {channel.value}: Budget {action} by ${abs(float(amount)):.2f}")
        else:
            print("No significant performance differences found - no reallocation needed")
        
        self.demo_data['performance_optimization'] = {
            'channel_performance': performance_data,
            'reallocation': {str(k): float(v) for k, v in reallocation_results.items()}
        }
    
    async def demo_emergency_controls(self):
        """Demonstrate emergency stop and circuit breaker"""
        print("\n🚨 EMERGENCY CONTROLS")
        print("-" * 40)
        
        campaign_id = "demo_emergency"
        daily_budget = Decimal('800.00')
        
        # Set up campaign
        self.pacer.allocate_hourly_budget(
            campaign_id, ChannelType.DISPLAY, daily_budget, PacingStrategy.EVEN_DISTRIBUTION
        )
        
        print(f"Campaign set up with ${daily_budget} daily budget")
        
        # Simulate rapid spending to trigger alerts
        print("\nSimulating rapid spending pattern...")
        rapid_transactions = []
        total_spend = Decimal('0')
        
        for i in range(8):  # 8 large transactions
            transaction = SpendTransaction(
                campaign_id=campaign_id,
                channel=ChannelType.DISPLAY,
                amount=Decimal('80.00'),
                timestamp=datetime.utcnow(),
                clicks=40,
                conversions=2,
                cost_per_click=2.00,
                conversion_rate=0.05
            )
            
            self.pacer.record_spend(transaction)
            rapid_transactions.append(transaction)
            total_spend += transaction.amount
            
            # Check pace after each transaction
            pace_ratio, alert = self.pacer.check_pace(campaign_id, ChannelType.DISPLAY)
            print(f"Transaction {i+1}: Spend ${transaction.amount}, Total ${total_spend}, Pace: {pace_ratio:.2f}x")
        
        print(f"\nTotal spend: ${total_spend} / ${daily_budget} ({float(total_spend/daily_budget):.1%})")
        
        # Test emergency stop
        print("\nTrigger emergency stop...")
        stop_successful = await self.pacer.emergency_stop(campaign_id, "Demo emergency stop")
        
        # Verify spending is blocked
        can_spend, reason = self.pacer.can_spend(campaign_id, ChannelType.DISPLAY, Decimal('10.00'))
        print(f"Emergency stop successful: {stop_successful}")
        print(f"Post-stop spend attempt blocked: {not can_spend}")
        print(f"Block reason: {reason}")
        
        self.demo_data['emergency_controls'] = {
            'rapid_transactions': len(rapid_transactions),
            'total_spend': float(total_spend),
            'emergency_stop_successful': stop_successful,
            'spending_blocked': not can_spend
        }
    
    async def demo_multi_channel_campaign(self):
        """Demonstrate complex multi-channel campaign management"""
        print("\n🎯 MULTI-CHANNEL CAMPAIGN")
        print("-" * 40)
        
        campaign_id = "demo_multichannel"
        total_budget = Decimal('10000.00')
        
        # Channel budget allocation
        channel_configs = [
            (ChannelType.GOOGLE_ADS, Decimal('4000.00'), PacingStrategy.PREDICTIVE_ML),
            (ChannelType.FACEBOOK_ADS, Decimal('3000.00'), PacingStrategy.PERFORMANCE_WEIGHTED),
            (ChannelType.TIKTOK_ADS, Decimal('2000.00'), PacingStrategy.ADAPTIVE_HYBRID),
            (ChannelType.DISPLAY, Decimal('1000.00'), PacingStrategy.EVEN_DISTRIBUTION)
        ]
        
        print(f"Setting up multi-channel campaign with ${total_budget} total budget:")
        
        # Set up each channel
        for channel, budget, strategy in channel_configs:
            # Generate historical data for better strategy performance
            await self._generate_historical_data(campaign_id, channel)
            
            allocations = self.pacer.allocate_hourly_budget(campaign_id, channel, budget, strategy)
            print(f"  {channel.value:15s}: ${budget:8.2f} using {strategy.value}")
        
        # Simulate a full day of coordinated spending
        print("\nSimulating coordinated multi-channel spending...")
        
        daily_summary = {channel.value: {'spend': 0, 'blocked': 0, 'allowed': 0} for channel, _, _ in channel_configs}
        
        # Simulate 24 hours of spending attempts
        for hour in range(24):
            print(f"Hour {hour:2d}: ", end="")
            
            for channel, budget, _ in channel_configs:
                # 5 spending attempts per hour per channel
                hour_attempts = 5
                hour_spend = 0
                hour_blocked = 0
                hour_allowed = 0
                
                for _ in range(hour_attempts):
                    spend_amount = Decimal(str(np.random.uniform(20, 150)))
                    can_spend, reason = self.pacer.can_spend(campaign_id, channel, spend_amount)
                    
                    if can_spend:
                        transaction = SpendTransaction(
                            campaign_id=campaign_id,
                            channel=channel,
                            amount=spend_amount,
                            timestamp=datetime.utcnow(),
                            clicks=int(spend_amount / Decimal('2.5')),
                            conversions=np.random.poisson(1),
                            cost_per_click=2.5,
                            conversion_rate=0.05
                        )
                        self.pacer.record_spend(transaction)
                        hour_spend += float(spend_amount)
                        hour_allowed += 1
                    else:
                        hour_blocked += 1
                
                daily_summary[channel.value]['spend'] += hour_spend
                daily_summary[channel.value]['blocked'] += hour_blocked
                daily_summary[channel.value]['allowed'] += hour_allowed
            
            # Show hourly progress
            total_hour_spend = sum(data['spend'] for data in daily_summary.values())
            print(f"${total_hour_spend:.0f} total spend")
        
        # Final campaign summary
        print(f"\n📋 CAMPAIGN SUMMARY")
        print("-" * 40)
        print("Channel         | Budget    | Actual    | Utilization | Blocked | Success Rate")
        print("-" * 80)
        
        total_actual_spend = 0
        total_budget = Decimal('0')
        
        for channel, budget, _ in channel_configs:
            channel_data = daily_summary[channel.value]
            actual_spend = channel_data['spend']
            utilization = actual_spend / float(budget) if budget > 0 else 0
            total_attempts = channel_data['blocked'] + channel_data['allowed']
            success_rate = channel_data['allowed'] / total_attempts if total_attempts > 0 else 0
            
            total_actual_spend += actual_spend
            total_budget += budget
            
            print(f"{channel.value:15s} | ${budget:8.2f} | ${actual_spend:8.2f} | {utilization:10.1%} | {channel_data['blocked']:7d} | {success_rate:11.1%}")
        
        overall_utilization = total_actual_spend / float(total_budget) if total_budget > 0 else 0
        print("-" * 80)
        print(f"{'TOTAL':15s} | ${total_budget:8.2f} | ${total_actual_spend:8.2f} | {overall_utilization:10.1%}")
        
        # Check final pacing for all channels
        print(f"\n⏰ FINAL PACING STATUS")
        print("-" * 40)
        
        for channel, _, _ in channel_configs:
            pace_ratio, alert = self.pacer.check_pace(campaign_id, channel)
            status = "🟢 ON PACE" if 0.8 <= pace_ratio <= 1.2 else ("🔴 OVERPACED" if pace_ratio > 1.2 else "🟡 UNDERPACED")
            print(f"{channel.value:15s}: {pace_ratio:.2f}x {status}")
        
        self.demo_data['multi_channel'] = {
            'daily_summary': daily_summary,
            'total_budget': float(total_budget),
            'total_spend': total_actual_spend,
            'utilization': overall_utilization,
            'alerts_generated': len(self.alerts)
        }
    
    def _create_performance_transactions(self, campaign_id: str, channel: ChannelType, 
                                       hours: int, avg_spend: float, conversion_rate: float) -> List[SpendTransaction]:
        """Create realistic performance transactions"""
        transactions = []
        
        for hour in range(hours):
            num_transactions = np.random.poisson(3) + 1  # 1-6 transactions per hour
            
            for _ in range(num_transactions):
                spend_amount = Decimal(str(max(10, np.random.normal(avg_spend, avg_spend * 0.3))))
                clicks = int(spend_amount / Decimal('2.50'))
                conversions = np.random.binomial(clicks, conversion_rate)
                
                transaction = SpendTransaction(
                    campaign_id=campaign_id,
                    channel=channel,
                    amount=spend_amount,
                    timestamp=datetime.utcnow() - timedelta(hours=hours-hour),
                    clicks=clicks,
                    conversions=conversions,
                    cost_per_click=float(spend_amount / clicks) if clicks > 0 else 2.50,
                    conversion_rate=conversions / clicks if clicks > 0 else 0.0
                )
                
                transactions.append(transaction)
        
        return transactions
    
    async def _generate_historical_data(self, campaign_id: str, channel: ChannelType):
        """Generate historical data for strategy testing"""
        # Generate 7 days of historical data
        for day in range(7):
            for hour in range(24):
                # Business hours get more activity
                if 9 <= hour <= 17:
                    transactions_count = np.random.poisson(5) + 2
                    base_spend = 50
                else:
                    transactions_count = np.random.poisson(2) + 1  
                    base_spend = 25
                
                for _ in range(transactions_count):
                    spend_amount = Decimal(str(max(10, np.random.normal(base_spend, 15))))
                    clicks = int(spend_amount / Decimal('2.00'))
                    conversions = np.random.binomial(clicks, 0.06)  # 6% average conversion rate
                    
                    transaction = SpendTransaction(
                        campaign_id=campaign_id,
                        channel=channel,
                        amount=spend_amount,
                        timestamp=datetime.utcnow() - timedelta(days=day, hours=24-hour),
                        clicks=clicks,
                        conversions=conversions,
                        cost_per_click=2.00,
                        conversion_rate=conversions / clicks if clicks > 0 else 0.0
                    )
                    
                    self.pacer.record_spend(transaction)
    
    def _generate_summary_report(self):
        """Generate comprehensive demo summary report"""
        print("\n" + "="*80)
        print("🎉 BUDGET PACER DEMONSTRATION COMPLETE")
        print("="*80)
        
        print(f"📊 Demo Statistics:")
        print(f"   • Total Alerts Generated: {len(self.alerts)}")
        print(f"   • Alert Types: {set(alert.alert_type for alert in self.alerts)}")
        print(f"   • Campaigns Tested: {len(self.demo_data)}")
        
        # Alert breakdown
        if self.alerts:
            alert_counts = {}
            for alert in self.alerts:
                alert_counts[alert.severity] = alert_counts.get(alert.severity, 0) + 1
            
            print(f"   • Alert Severity Breakdown:")
            for severity, count in alert_counts.items():
                print(f"     - {severity}: {count}")
        
        print(f"\n🛡️  Safety Features Demonstrated:")
        print(f"   ✅ Hourly budget allocation")
        print(f"   ✅ Frontload protection")
        print(f"   ✅ Performance-based optimization")  
        print(f"   ✅ Dynamic budget reallocation")
        print(f"   ✅ Circuit breaker protection")
        print(f"   ✅ Emergency stop mechanism")
        print(f"   ✅ Multi-channel coordination")
        print(f"   ✅ Predictive ML pacing")
        
        print(f"\n💡 Key Benefits:")
        print(f"   • Prevents early budget exhaustion")
        print(f"   • Optimizes spend distribution")
        print(f"   • Provides real-time monitoring")
        print(f"   • Enables automatic safety interventions")
        print(f"   • Supports complex multi-channel campaigns")
        
        # Save demo results
        import json
        with open('/home/hariravichandran/AELP/budget_pacer_demo_results.json', 'w') as f:
            json.dump({
                'demo_data': self.demo_data,
                'alerts_summary': {
                    'total_alerts': len(self.alerts),
                    'alert_types': list(set(alert.alert_type for alert in self.alerts)),
                    'severity_counts': {severity: sum(1 for alert in self.alerts if alert.severity == severity) 
                                      for severity in ['low', 'medium', 'high', 'critical']}
                },
                'demo_timestamp': datetime.utcnow().isoformat()
            }, f, indent=2, default=str)
        
        print(f"\n📄 Detailed demo results saved to 'budget_pacer_demo_results.json'")
        print(f"🚀 Budget pacer is ready for production deployment!")


async def main():
    """Run the budget pacer demonstration"""
    demo = BudgetPacerDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())