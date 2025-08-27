#!/usr/bin/env python3
"""
Budget Pacer Integration Example
Shows how to integrate the budget pacer with existing GAELP safety systems.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from budget_pacer import (
    BudgetPacer, ChannelType, PacingStrategy, SpendTransaction,
    PacingAlert
)

# Import existing safety framework components
try:
    from safety_framework.budget_controls import RealMoneyBudgetController, BudgetLimits
    from safety_framework.operational_safety import SafetyOrchestrator
except ImportError:
    # Mock classes for demonstration
    class RealMoneyBudgetController:
        def __init__(self, *args, **kwargs):
            pass
    
    class BudgetLimits:
        def __init__(self, *args, **kwargs):
            pass
    
    class SafetyOrchestrator:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)


class IntegratedBudgetManager:
    """
    Integrated budget management system combining advanced pacing 
    with existing GAELP safety controls.
    """
    
    def __init__(self):
        # Initialize components
        self.budget_pacer = BudgetPacer(alert_callback=self._handle_pacing_alert)
        self.safety_controller = RealMoneyBudgetController(alert_callback=self._handle_safety_alert)
        self.safety_orchestrator = SafetyOrchestrator()
        
        # Integration state
        self.active_campaigns: Dict[str, Dict] = {}
        self.safety_alerts: List[Dict] = []
        self.pacing_alerts: List[PacingAlert] = []
        
        logger.info("Integrated budget manager initialized")
    
    async def setup_campaign(self, campaign_config: Dict) -> bool:
        """
        Set up a new campaign with integrated safety and pacing controls.
        
        Args:
            campaign_config: {
                'campaign_id': str,
                'daily_budget': Decimal,
                'channels': List[Dict],  # [{'channel': ChannelType, 'budget': Decimal, 'strategy': PacingStrategy}]
                'safety_limits': Dict,   # Safety framework limits
                'pacing_params': Dict    # Custom pacing parameters
            }
        """
        try:
            campaign_id = campaign_config['campaign_id']
            daily_budget = campaign_config['daily_budget']
            
            logger.info(f"Setting up integrated campaign: {campaign_id}")
            
            # 1. Set up safety framework limits
            safety_limits = BudgetLimits(
                daily_limit=daily_budget,
                weekly_limit=daily_budget * 7,
                monthly_limit=daily_budget * 30,
                campaign_limit=campaign_config.get('total_budget', daily_budget * 30),
                **campaign_config.get('safety_limits', {})
            )
            
            await self.safety_controller.register_campaign(campaign_id, safety_limits)
            
            # 2. Set up budget pacing for each channel
            for channel_config in campaign_config['channels']:
                channel = channel_config['channel']
                channel_budget = channel_config['budget']
                strategy = channel_config.get('strategy', PacingStrategy.ADAPTIVE_HYBRID)
                
                # Allocate hourly budget with pacing
                allocations = self.budget_pacer.allocate_hourly_budget(
                    campaign_id, channel, channel_budget, strategy
                )
                
                logger.info(f"  Channel {channel.value}: ${channel_budget} with {strategy.value} strategy")
            
            # 3. Store campaign configuration
            self.active_campaigns[campaign_id] = {
                'config': campaign_config,
                'created_at': datetime.utcnow(),
                'status': 'active',
                'total_spend': Decimal('0'),
                'safety_violations': 0,
                'pacing_alerts': 0
            }
            
            logger.info(f"Campaign {campaign_id} setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup campaign {campaign_config.get('campaign_id')}: {e}")
            return False
    
    async def authorize_spend(self, campaign_id: str, channel: ChannelType, 
                            amount: Decimal, context: Dict = None) -> tuple[bool, str, Dict]:
        """
        Authorize a spend transaction through integrated safety and pacing checks.
        
        Returns:
            (authorized: bool, reason: str, additional_info: Dict)
        """
        try:
            if campaign_id not in self.active_campaigns:
                return False, "Campaign not found", {}
            
            campaign_data = self.active_campaigns[campaign_id]
            if campaign_data['status'] != 'active':
                return False, f"Campaign status: {campaign_data['status']}", {}
            
            # 1. Check budget pacing limits
            can_spend_pacing, pacing_reason = self.budget_pacer.can_spend(
                campaign_id, channel, amount
            )
            
            if not can_spend_pacing:
                logger.warning(f"Pacing check failed for {campaign_id}/{channel.value}: {pacing_reason}")
                return False, f"Pacing limit: {pacing_reason}", {"check_failed": "pacing"}
            
            # 2. Check safety framework limits
            # In a real implementation, this would integrate with the actual safety controller
            current_spend = self.budget_pacer._calculate_today_spend(campaign_id, channel)
            projected_total = current_spend + amount
            
            channel_budget = None
            for ch_config in campaign_data['config']['channels']:
                if ch_config['channel'] == channel:
                    channel_budget = ch_config['budget']
                    break
            
            if channel_budget and projected_total > channel_budget * Decimal('0.95'):
                return False, "Safety limit: Approaching daily budget limit", {"check_failed": "safety"}
            
            # 3. Additional context-based checks
            if context:
                # Check for unusual patterns
                if context.get('cost_per_click', 0) > 10.0:
                    return False, "Safety limit: Unusually high cost per click", {"check_failed": "context"}
                
                if context.get('conversion_rate', 0) < 0.001:
                    return False, "Performance limit: Very low conversion rate", {"check_failed": "performance"}
            
            # All checks passed
            return True, "Spend authorized", {
                "pacing_check": "passed",
                "safety_check": "passed", 
                "context_check": "passed"
            }
            
        except Exception as e:
            logger.error(f"Error authorizing spend: {e}")
            return False, f"Authorization error: {e}", {"check_failed": "system_error"}
    
    async def record_transaction(self, transaction: SpendTransaction) -> bool:
        """
        Record a transaction in both pacing and safety systems.
        """
        try:
            campaign_id = transaction.campaign_id
            
            if campaign_id not in self.active_campaigns:
                logger.error(f"Transaction for unknown campaign: {campaign_id}")
                return False
            
            # 1. Record in budget pacer
            self.budget_pacer.record_spend(transaction)
            
            # 2. Update campaign tracking
            campaign_data = self.active_campaigns[campaign_id]
            campaign_data['total_spend'] += transaction.amount
            
            # 3. Check post-transaction pacing
            pace_ratio, alert = self.budget_pacer.check_pace(campaign_id, transaction.channel)
            
            if alert:
                campaign_data['pacing_alerts'] += 1
                await self._handle_pacing_alert(alert)
            
            logger.debug(f"Transaction recorded: ${transaction.amount} for {campaign_id}/{transaction.channel.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record transaction: {e}")
            return False
    
    async def get_campaign_status(self, campaign_id: str) -> Optional[Dict]:
        """Get comprehensive campaign status including pacing and safety metrics."""
        try:
            if campaign_id not in self.active_campaigns:
                return None
            
            campaign_data = self.active_campaigns[campaign_id]
            status = {
                'campaign_id': campaign_id,
                'status': campaign_data['status'],
                'created_at': campaign_data['created_at'],
                'total_spend': campaign_data['total_spend'],
                'channels': {}
            }
            
            # Get status for each channel
            for channel_config in campaign_data['config']['channels']:
                channel = channel_config['channel']
                channel_budget = channel_config['budget']
                
                today_spend = self.budget_pacer._calculate_today_spend(campaign_id, channel)
                pace_ratio, current_alert = self.budget_pacer.check_pace(campaign_id, channel)
                
                # Get circuit breaker status
                breaker_state = "unknown"
                if (campaign_id in self.budget_pacer.circuit_breakers and 
                    channel in self.budget_pacer.circuit_breakers[campaign_id]):
                    breaker_state = self.budget_pacer.circuit_breakers[campaign_id][channel].state.value
                
                status['channels'][channel.value] = {
                    'budget': float(channel_budget),
                    'spend_today': float(today_spend),
                    'utilization': float(today_spend / channel_budget) if channel_budget > 0 else 0,
                    'pace_ratio': pace_ratio,
                    'circuit_breaker_state': breaker_state,
                    'has_alert': current_alert is not None
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting campaign status: {e}")
            return None
    
    async def optimize_campaign_budgets(self, campaign_id: str) -> Dict:
        """
        Optimize campaign budgets based on performance and pacing data.
        """
        try:
            if campaign_id not in self.active_campaigns:
                return {"error": "Campaign not found"}
            
            # Trigger dynamic reallocation
            reallocation_results = await self.budget_pacer.reallocate_unused(campaign_id)
            
            optimization_summary = {
                'campaign_id': campaign_id,
                'optimization_timestamp': datetime.utcnow(),
                'reallocations': reallocation_results,
                'total_reallocated': sum(abs(float(amount)) for amount in reallocation_results.values())
            }
            
            logger.info(f"Budget optimization complete for {campaign_id}: "
                       f"${optimization_summary['total_reallocated']:.2f} reallocated")
            
            return optimization_summary
            
        except Exception as e:
            logger.error(f"Error optimizing campaign budgets: {e}")
            return {"error": str(e)}
    
    async def emergency_pause_campaign(self, campaign_id: str, reason: str = "Emergency pause") -> bool:
        """
        Emergency pause a campaign using integrated safety controls.
        """
        try:
            if campaign_id not in self.active_campaigns:
                logger.error(f"Cannot pause unknown campaign: {campaign_id}")
                return False
            
            # 1. Trigger emergency stop in budget pacer
            pacer_stop = await self.budget_pacer.emergency_stop(campaign_id, reason)
            
            # 2. Update campaign status
            self.active_campaigns[campaign_id]['status'] = 'emergency_paused'
            
            # 3. Log critical event
            logger.critical(f"Emergency pause executed for {campaign_id}: {reason}")
            
            return pacer_stop
            
        except Exception as e:
            logger.error(f"Error in emergency pause: {e}")
            return False
    
    async def _handle_pacing_alert(self, alert: PacingAlert):
        """Handle pacing alerts from budget pacer."""
        try:
            self.pacing_alerts.append(alert)
            
            # Log alert based on severity
            if alert.severity == "critical":
                logger.critical(f"CRITICAL PACING ALERT: {alert.alert_type} - {alert.recommended_action}")
                
                # Auto-trigger emergency pause for critical alerts
                if alert.alert_type in ["emergency_stop_required", "critical_overpacing"]:
                    await self.emergency_pause_campaign(alert.campaign_id, f"Auto-pause: {alert.alert_type}")
            
            elif alert.severity == "high":
                logger.error(f"HIGH PACING ALERT: {alert.alert_type} - {alert.recommended_action}")
            else:
                logger.warning(f"Pacing alert: {alert.alert_type} - {alert.recommended_action}")
            
            # Update campaign alert count
            if alert.campaign_id in self.active_campaigns:
                self.active_campaigns[alert.campaign_id]['pacing_alerts'] += 1
            
        except Exception as e:
            logger.error(f"Error handling pacing alert: {e}")
    
    async def _handle_safety_alert(self, alert: Dict):
        """Handle safety alerts from existing safety framework."""
        try:
            self.safety_alerts.append(alert)
            
            logger.warning(f"Safety framework alert: {alert.get('type', 'unknown')}")
            
            # Coordinate with budget pacer if needed
            if alert.get('type') == 'budget_violation':
                campaign_id = alert.get('campaign_id')
                if campaign_id:
                    # Trigger emergency stop in pacer as well
                    await self.budget_pacer.emergency_stop(campaign_id, "Safety framework violation")
            
        except Exception as e:
            logger.error(f"Error handling safety alert: {e}")
    
    def get_system_health(self) -> Dict:
        """Get overall system health status."""
        return {
            'active_campaigns': len([c for c in self.active_campaigns.values() if c['status'] == 'active']),
            'total_campaigns': len(self.active_campaigns),
            'pacing_alerts_24h': len([a for a in self.pacing_alerts 
                                    if a.timestamp > datetime.utcnow() - timedelta(hours=24)]) if hasattr(self.pacing_alerts[0] if self.pacing_alerts else None, 'timestamp') else len(self.pacing_alerts),
            'safety_alerts_24h': len(self.safety_alerts),
            'total_spend_today': sum(float(c['total_spend']) for c in self.active_campaigns.values()),
            'system_status': 'healthy' if len(self.pacing_alerts) < 10 and len(self.safety_alerts) < 5 else 'degraded'
        }


async def demo_integration():
    """Demonstrate integrated budget management"""
    print("🔗 GAELP Integrated Budget Management Demo")
    print("="*60)
    
    # Initialize integrated manager
    manager = IntegratedBudgetManager()
    
    # Set up a sample campaign
    campaign_config = {
        'campaign_id': 'integrated_demo_001',
        'daily_budget': Decimal('5000.00'),
        'channels': [
            {
                'channel': ChannelType.GOOGLE_ADS,
                'budget': Decimal('2500.00'),
                'strategy': PacingStrategy.PREDICTIVE_ML
            },
            {
                'channel': ChannelType.FACEBOOK_ADS,
                'budget': Decimal('1500.00'),
                'strategy': PacingStrategy.PERFORMANCE_WEIGHTED
            },
            {
                'channel': ChannelType.TIKTOK_ADS,
                'budget': Decimal('1000.00'),
                'strategy': PacingStrategy.ADAPTIVE_HYBRID
            }
        ],
        'safety_limits': {
            'hourly_rate_limit': Decimal('300.00'),
            'roi_threshold': Decimal('0.20')
        },
        'total_budget': Decimal('150000.00')  # 30-day campaign
    }
    
    # Set up campaign
    setup_success = await manager.setup_campaign(campaign_config)
    print(f"✅ Campaign setup: {'Success' if setup_success else 'Failed'}")
    
    if setup_success:
        campaign_id = campaign_config['campaign_id']
        
        # Test spend authorization
        print(f"\n💰 Testing spend authorization:")
        
        test_spends = [
            (ChannelType.GOOGLE_ADS, Decimal('75.00'), {"cost_per_click": 2.50, "conversion_rate": 0.08}),
            (ChannelType.FACEBOOK_ADS, Decimal('150.00'), {"cost_per_click": 1.80, "conversion_rate": 0.05}),
            (ChannelType.GOOGLE_ADS, Decimal('500.00'), {"cost_per_click": 15.00, "conversion_rate": 0.01})  # Should fail
        ]
        
        for channel, amount, context in test_spends:
            authorized, reason, info = await manager.authorize_spend(campaign_id, channel, amount, context)
            status = "✅ APPROVED" if authorized else "❌ BLOCKED"
            print(f"  {channel.value:15s} ${amount:6.2f}: {status}")
            if not authorized:
                print(f"    Reason: {reason}")
        
        # Record some transactions
        print(f"\n📊 Recording transactions:")
        
        sample_transactions = [
            SpendTransaction(
                campaign_id=campaign_id,
                channel=ChannelType.GOOGLE_ADS,
                amount=Decimal('75.00'),
                timestamp=datetime.utcnow(),
                clicks=30,
                conversions=3,
                cost_per_click=2.50,
                conversion_rate=0.10
            ),
            SpendTransaction(
                campaign_id=campaign_id,
                channel=ChannelType.FACEBOOK_ADS,
                amount=Decimal('120.00'),
                timestamp=datetime.utcnow(),
                clicks=67,
                conversions=4,
                cost_per_click=1.79,
                conversion_rate=0.06
            )
        ]
        
        for transaction in sample_transactions:
            success = await manager.record_transaction(transaction)
            print(f"  ${transaction.amount} on {transaction.channel.value}: {'✅ Recorded' if success else '❌ Failed'}")
        
        # Get campaign status
        print(f"\n📈 Campaign status:")
        status = await manager.get_campaign_status(campaign_id)
        if status:
            print(f"  Campaign: {status['campaign_id']}")
            print(f"  Total Spend: ${status['total_spend']}")
            print(f"  Channels:")
            for channel_name, channel_data in status['channels'].items():
                print(f"    {channel_name:15s}: ${channel_data['spend_today']:7.2f} / ${channel_data['budget']:7.2f} "
                      f"({channel_data['utilization']:5.1%}) - Pace: {channel_data['pace_ratio']:.2f}x")
        
        # Test optimization
        print(f"\n⚡ Testing budget optimization:")
        optimization = await manager.optimize_campaign_budgets(campaign_id)
        if 'error' not in optimization:
            print(f"  Optimization completed: ${optimization['total_reallocated']:.2f} reallocated")
        else:
            print(f"  Optimization failed: {optimization['error']}")
        
        # System health
        print(f"\n🏥 System health:")
        health = manager.get_system_health()
        print(f"  Active Campaigns: {health['active_campaigns']}")
        print(f"  Pacing Alerts: {health['pacing_alerts_24h']}")
        print(f"  System Status: {health['system_status'].upper()}")
    
    print(f"\n🎉 Integration demo complete!")
    return manager


if __name__ == "__main__":
    asyncio.run(demo_integration())