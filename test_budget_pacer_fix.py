#!/usr/bin/env python3
"""Test if Budget Pacer fixes work"""

from decimal import Decimal
from datetime import datetime
from budget_pacer import BudgetPacer, PacingStrategy, ChannelType

def test_budget_pacer():
    """Test Budget Pacer with both methods"""
    
    print("="*80)
    print("TESTING BUDGET PACER FIX")
    print("="*80)
    
    # Test 1: Initialize Budget Pacer
    print("\n1. Initializing Budget Pacer...")
    try:
        pacer = BudgetPacer()
        print(f"   ✅ Budget Pacer initialized")
        print(f"      Max hourly spend: {pacer.max_hourly_spend_pct * 100:.0f}%")
        print(f"      Emergency stop threshold: {pacer.emergency_stop_threshold * 100:.0f}%")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Test get_pacing_multiplier method
    print("\n2. Testing get_pacing_multiplier method...")
    scenarios = [
        (10, 30.0, 100.0, "Morning, 30% spent"),
        (14, 50.0, 100.0, "Afternoon, 50% spent"),
        (20, 80.0, 100.0, "Evening, 80% spent"),
        (22, 95.0, 100.0, "Late night, 95% spent"),
        (5, 10.0, 100.0, "Early morning, 10% spent")
    ]
    
    for hour, spent, budget, desc in scenarios:
        try:
            multiplier = pacer.get_pacing_multiplier(hour, spent, budget)
            print(f"   ✅ {desc}:")
            print(f"      Hour {hour}, ${spent:.0f}/${budget:.0f} → multiplier: {multiplier:.2f}")
        except Exception as e:
            print(f"   ❌ {desc} failed: {e}")
            return False
    
    # Test 3: Test can_spend method
    print("\n3. Testing can_spend method...")
    try:
        # First set up a channel budget
        from budget_pacer import ChannelBudget, HourlyAllocation
        
        channel_budget = ChannelBudget(
            channel=ChannelType.SEARCH,
            daily_budget=Decimal('50.00'),
            hourly_allocations=[
                HourlyAllocation(
                    hour=h, 
                    base_allocation_pct=100/24,
                    performance_multiplier=1.0,
                    predicted_conversion_rate=0.02,
                    predicted_cost_per_click=1.5,
                    confidence_score=0.8
                )
                for h in range(24)
            ],
            performance_metrics={'ctr': 0.02, 'conversion_rate': 0.01},
            spend_velocity_limit=Decimal('5.00'),
            circuit_breaker_threshold=Decimal('100.00'),
            last_optimization=datetime.now()
        )
        
        pacer.channel_budgets['test_campaign'] = {
            ChannelType.SEARCH: channel_budget
        }
        
        # Test spending
        can_spend, reason = pacer.can_spend(
            campaign_id='test_campaign',
            channel=ChannelType.SEARCH,
            amount=Decimal('5.00')
        )
        
        print(f"   ✅ can_spend works:")
        print(f"      Can spend $5.00: {can_spend}")
        print(f"      Reason: {reason}")
        
    except Exception as e:
        print(f"   ❌ can_spend failed: {e}")
        return False
    
    # Test 4: Test with MasterOrchestrator
    print("\n4. Testing with MasterOrchestrator...")
    try:
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        config = GAELPConfig()
        master = MasterOrchestrator(config)
        
        if hasattr(master, 'budget_pacer'):
            # Test get_pacing_multiplier through master
            pacing = master.budget_pacer.get_pacing_multiplier(
                hour=10,
                spent_so_far=30.0,
                daily_budget=100.0
            )
            print(f"   ✅ MasterOrchestrator budget pacer works")
            print(f"      Pacing multiplier at hour 10: {pacing:.2f}")
            
            # Test can_spend through master (need to set up budget first)
            from budget_pacer import ChannelBudget
            master.budget_pacer.channel_budgets['master_test'] = {
                ChannelType.DISPLAY: ChannelBudget(
                    channel=ChannelType.DISPLAY,
                    daily_budget=Decimal('75.00'),
                    hourly_allocations=[],
                    performance_metrics={},
                    spend_velocity_limit=Decimal('10.00'),
                    circuit_breaker_threshold=Decimal('150.00'),
                    last_optimization=datetime.now()
                )
            }
            
            can_spend, reason = master.budget_pacer.can_spend(
                campaign_id='master_test',
                channel=ChannelType.DISPLAY,
                amount=Decimal('10.00')
            )
            print(f"      Can spend $10 on display: {can_spend} ({reason})")
            
        else:
            print("   ❌ MasterOrchestrator doesn't have budget_pacer")
            return False
            
    except Exception as e:
        print(f"   ❌ MasterOrchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ BUDGET PACER TEST PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_budget_pacer()
    exit(0 if success else 1)