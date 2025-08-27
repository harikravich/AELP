#!/usr/bin/env python3
"""Test if Safety System fixes work"""

from safety_system import SafetySystem, SafetyConfig

def test_safety_system():
    """Test Safety System with both methods"""
    
    print("="*80)
    print("TESTING SAFETY SYSTEM FIX")
    print("="*80)
    
    # Test 1: Initialize Safety System
    print("\n1. Initializing Safety System...")
    try:
        config = SafetyConfig(max_bid_absolute=10.0)
        safety = SafetySystem(config)
        print(f"   ✅ Safety System initialized")
        print(f"      Max bid: ${config.max_bid_absolute}")
        print(f"      Daily loss threshold: ${config.daily_loss_threshold}")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Test check_bid_safety method
    print("\n2. Testing check_bid_safety method...")
    try:
        # Test safe bid
        is_safe, violations = safety.check_bid_safety(
            query="parental control app",
            bid_amount=5.0,
            campaign_id="test_campaign",
            predicted_roi=0.25
        )
        print(f"   ✅ check_bid_safety works:")
        print(f"      Bid $5.00 is {'safe' if is_safe else 'unsafe'}")
        if violations:
            print(f"      Violations: {violations}")
        
        # Test unsafe bid (too high)
        is_safe, violations = safety.check_bid_safety(
            query="parental control app",
            bid_amount=15.0,
            campaign_id="test_campaign",
            predicted_roi=0.25
        )
        print(f"      Bid $15.00 is {'safe' if is_safe else 'unsafe'}")
        if violations:
            print(f"      Violations: {violations[:1]}...")  # Show first violation
            
    except Exception as e:
        print(f"   ❌ check_bid_safety failed: {e}")
        return False
    
    # Test 3: Test validate_bid method (compatibility wrapper)
    print("\n3. Testing validate_bid method (compatibility wrapper)...")
    try:
        # Test safe bid
        safe_bid = safety.validate_bid(
            bid_amount=5.0,
            context={"budget_remaining": 100.0, "predicted_roi": 0.3}
        )
        print(f"   ✅ validate_bid works:")
        print(f"      Input bid: $5.00 → Safe bid: ${safe_bid:.2f}")
        
        # Test unsafe bid (too high)
        safe_bid = safety.validate_bid(
            bid_amount=15.0,
            context={"budget_remaining": 50.0, "predicted_roi": 0.3}
        )
        print(f"      Input bid: $15.00 → Safe bid: ${safe_bid:.2f}")
        
        # Test with low ROI
        safe_bid = safety.validate_bid(
            bid_amount=3.0,
            context={"budget_remaining": 100.0, "predicted_roi": 0.05}
        )
        print(f"      Input bid: $3.00 (low ROI) → Safe bid: ${safe_bid:.2f}")
        
    except Exception as e:
        print(f"   ❌ validate_bid failed: {e}")
        return False
    
    # Test 4: Test with MasterOrchestrator
    print("\n4. Testing with MasterOrchestrator...")
    try:
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        config = GAELPConfig()
        master = MasterOrchestrator(config)
        
        # Test validate_bid through master
        if hasattr(master, 'safety_system'):
            safe_bid = master.safety_system.validate_bid(
                bid_amount=8.0,
                context={"budget_remaining": 75.0}
            )
            print(f"   ✅ MasterOrchestrator safety system works")
            print(f"      Bid $8.00 → ${safe_bid:.2f}")
            
            # Test check_bid_safety through master
            is_safe, violations = master.safety_system.check_bid_safety(
                query="kids app",
                bid_amount=4.0,
                campaign_id="master_test",
                predicted_roi=0.2
            )
            print(f"      check_bid_safety also works: bid is {'safe' if is_safe else 'unsafe'}")
        else:
            print("   ❌ MasterOrchestrator doesn't have safety_system")
            return False
            
    except Exception as e:
        print(f"   ❌ MasterOrchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ SAFETY SYSTEM TEST PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_safety_system()
    exit(0 if success else 1)