#!/usr/bin/env python3
"""
Test MasterOrchestrator auction to see if it's broken or working
"""

import sys
import os
import numpy as np

def test_master_auction():
    """Test if MasterOrchestrator's auction has the 100% win bug"""
    
    print("=" * 80)
    print("TESTING MASTERORCHESTRATOR AUCTION SYSTEM")
    print("=" * 80)
    
    try:
        # Import and create MasterOrchestrator
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        config = GAELPConfig(
            simulation_episodes=1,
            monte_carlo_sims=1,
            enable_safety_system=False  # Simplify for test
        )
        
        # Create master with minimal setup
        master = MasterOrchestrator(config)
        
        # Test the auction through step_fixed_environment
        print("\n📊 Running 100 auction steps through MasterOrchestrator...")
        
        wins = 0
        total = 0
        
        for i in range(100):
            step_result = master.step_fixed_environment()
            
            if step_result and 'step_info' in step_result:
                info = step_result['step_info']
                if 'auction' in info:
                    auction_result = info['auction']
                    total += 1
                    if auction_result.get('won', False):
                        wins += 1
                        
            # Also check metrics
            if step_result and 'metrics' in step_result:
                metrics = step_result['metrics']
                if i % 20 == 0:  # Print every 20 steps
                    print(f"Step {i}: Wins={metrics.get('auction_wins', 0)}, "
                          f"Losses={metrics.get('auction_losses', 0)}, "
                          f"Win Rate={metrics.get('win_rate', 0):.2%}")
        
        # Final results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        if total > 0:
            actual_win_rate = wins / total
            print(f"\n✅ Total Auctions: {total}")
            print(f"✅ Wins: {wins}")
            print(f"✅ Win Rate: {actual_win_rate:.2%}")
            
            if actual_win_rate > 0.9:
                print("\n❌ PROBLEM DETECTED: Win rate > 90%!")
                print("   The auction system appears to have minimal competition.")
                print("   This is the 100% win rate bug.")
                return False
            elif actual_win_rate < 0.1:
                print("\n⚠️  Win rate very low (< 10%)")
                print("   May need to adjust bid levels.")
                return True
            else:
                print("\n✅ AUCTION SYSTEM WORKING CORRECTLY!")
                print(f"   Win rate of {actual_win_rate:.2%} indicates proper competition.")
                return True
        else:
            print("\n❌ No auction results found in step_fixed_environment")
            return False
            
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("   Cannot test MasterOrchestrator")
        return False
    except Exception as e:
        print(f"\n❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_auction_gym_directly():
    """Test AuctionGymWrapper directly"""
    
    print("\n" + "=" * 80)
    print("TESTING AUCTIONGYM DIRECTLY")
    print("=" * 80)
    
    try:
        from auction_gym_integration import AuctionGymWrapper
        
        wrapper = AuctionGymWrapper({
            'competitors': {'count': 10},
            'num_slots': 5
        })
        
        wins = 0
        for i in range(100):
            bid = np.random.uniform(2.0, 4.0)
            result = wrapper.run_auction(
                our_bid=bid,
                query_value=7.5,
                context={'device': 'mobile', 'hour': 14}
            )
            if result.won:
                wins += 1
        
        win_rate = wins / 100
        print(f"\nDirect AuctionGym Win Rate: {win_rate:.2%}")
        
        if win_rate > 0.9:
            print("❌ AuctionGym has 100% win bug")
            return False
        else:
            print("✅ AuctionGym working correctly")
            return True
            
    except Exception as e:
        print(f"❌ Cannot test AuctionGym: {e}")
        return False

if __name__ == "__main__":
    # Test both systems
    auction_gym_ok = test_auction_gym_directly()
    master_ok = test_master_auction()
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if master_ok:
        print("\n✅ MasterOrchestrator auction system is WORKING")
        print("   No immediate fix needed for auction mechanics.")
        print("\n⚠️  However, check these issues:")
        print("   1. Dashboard win rate display formula (over 100% bug)")
        print("   2. Ensure RL agent receives proper rewards from conversions")
    else:
        print("\n❌ MasterOrchestrator auction system is BROKEN")
        print("   Needs to be fixed to use proper competition.")
        print("\n🔧 FIX REQUIRED:")
        print("   1. Replace with fixed_auction_system.py")
        print("   2. Or fix AuctionGymWrapper competition levels")