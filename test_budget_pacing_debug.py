#!/usr/bin/env python3
"""
Debug budget pacing issues in the enhanced simulator
"""

import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

from enhanced_simulator_fixed import FixedGAELPEnvironment
import numpy as np
import matplotlib.pyplot as plt

def test_budget_progression():
    """Test that budget spending progresses correctly"""
    print("🔍 Testing Budget Progression...")
    
    env = FixedGAELPEnvironment(max_budget=1000.0, max_steps=50)
    state = env.reset()
    
    budget_history = []
    spend_history = []
    step_costs = []
    
    for step in range(50):
        # Simple action - bid consistently
        action = {
            'bid': 2.0,
            'creative_type': 'text',
            'audience_segment': 'parents',
            'quality_score': 0.8
        }
        
        budget_before = env.budget_spent
        state, reward, done, info = env.step(action)
        budget_after = env.budget_spent
        
        step_cost = budget_after - budget_before
        
        budget_history.append(env.budget_spent)
        spend_history.append(step_cost)
        step_costs.append(step_cost)
        
        if step % 10 == 0 or step_cost > 0:
            print(f"Step {step}: Budget spent = {env.budget_spent:.2f}, Step cost = {step_cost:.2f}, Win rate = {info.get('win_rate', 0):.3f}")
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"\n📊 Budget Analysis:")
    print(f"Total budget spent: {env.budget_spent:.2f} / {env.max_budget:.2f}")
    print(f"Average cost per step: {np.mean([c for c in step_costs if c > 0]):.3f}")
    print(f"Steps with spending: {len([c for c in step_costs if c > 0])}")
    print(f"Win rate: {info.get('win_rate', 0):.3f}")
    
    # Check if spending is progressing
    non_zero_spends = [c for c in step_costs if c > 0]
    if len(non_zero_spends) > 0:
        print("✅ Budget spending is working")
        return True
    else:
        print("❌ No budget spent - pacing issue detected")
        return False

def test_different_bid_amounts():
    """Test budget spending with different bid amounts"""
    print("\n🎯 Testing Different Bid Amounts...")
    
    bid_levels = [0.5, 1.0, 2.0, 5.0]
    results = {}
    
    for bid in bid_levels:
        env = FixedGAELPEnvironment(max_budget=1000.0, max_steps=20)
        env.reset()
        
        total_spent = 0
        wins = 0
        
        for step in range(20):
            action = {
                'bid': bid,
                'creative_type': 'text',
                'audience_segment': 'parents',
                'quality_score': 0.8
            }
            
            budget_before = env.budget_spent
            state, reward, done, info = env.step(action)
            spend = env.budget_spent - budget_before
            
            if spend > 0:
                wins += 1
            
            if done:
                break
        
        results[bid] = {
            'total_spent': env.budget_spent,
            'wins': wins,
            'win_rate': wins / 20,
            'avg_cost_per_win': env.budget_spent / max(wins, 1)
        }
        
        print(f"Bid ${bid:.1f}: Spent ${env.budget_spent:.2f}, Wins {wins}/20 ({wins/20:.1%}), Avg cost/win ${env.budget_spent/max(wins,1):.2f}")
    
    # Check if higher bids result in more wins and spending
    bid_0_5_wins = results[0.5]['wins']
    bid_5_0_wins = results[5.0]['wins']
    
    if bid_5_0_wins > bid_0_5_wins:
        print("✅ Higher bids result in more wins")
        return True
    else:
        print("❌ Bid amount doesn't affect win rate properly")
        return False

def test_budget_exhaustion():
    """Test that budget gets exhausted properly"""
    print("\n💸 Testing Budget Exhaustion...")
    
    env = FixedGAELPEnvironment(max_budget=100.0, max_steps=1000)  # Small budget, many steps
    env.reset()
    
    step = 0
    while step < 1000:
        action = {
            'bid': 10.0,  # High bid to spend quickly
            'creative_type': 'text',
            'audience_segment': 'parents',
            'quality_score': 0.9
        }
        
        state, reward, done, info = env.step(action)
        step += 1
        
        if done:
            break
    
    print(f"Budget exhausted after {step} steps")
    print(f"Final budget spent: {env.budget_spent:.2f} / {env.max_budget:.2f}")
    
    if env.budget_spent >= env.max_budget * 0.8:  # At least 80% spent
        print("✅ Budget exhaustion working")
        return True
    else:
        print("❌ Budget not being spent properly")
        return False

def main():
    print("🚀 Budget Pacing Debug Test")
    print("=" * 50)
    
    test1 = test_budget_progression()
    test2 = test_different_bid_amounts() 
    test3 = test_budget_exhaustion()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print(f"Budget progression: {'✅' if test1 else '❌'}")
    print(f"Bid responsiveness: {'✅' if test2 else '❌'}")
    print(f"Budget exhaustion: {'✅' if test3 else '❌'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 Budget pacing is working correctly!")
        return True
    else:
        print("\n❌ Budget pacing has issues that need fixing")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)