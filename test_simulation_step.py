#!/usr/bin/env python3
"""Test the simulation step to find the error"""

import sys
import traceback
from gaelp_master_integration import MasterOrchestrator, GAELPConfig
from decimal import Decimal

def test_simulation():
    """Test the simulation step by step"""
    print("Initializing MasterOrchestrator...")
    
    config = GAELPConfig()
    config.daily_budget_total = Decimal("1000.0")
    
    try:
        master = MasterOrchestrator(config)
        print("✅ MasterOrchestrator initialized")
        
        print("\nTesting step_fixed_environment...")
        for i in range(3):
            print(f"\nStep {i+1}:")
            result = master.step_fixed_environment()
            if result:
                print(f"  - Reward: {result.get('reward', 0)}")
                print(f"  - Step info: {result.get('step_info', {})}")
                metrics = result.get('metrics', {})
                print(f"  - Impressions: {metrics.get('total_impressions', 0)}")
                print(f"  - Budget spent: {metrics.get('budget_spent', 0)}")
            else:
                print("  - No result returned!")
                
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    
    print("\n✅ Test completed successfully!")
    return True

if __name__ == "__main__":
    test_simulation()