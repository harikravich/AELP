#!/usr/bin/env python3
"""Verify auction fix worked"""

def verify_fix():
    """Test that auction now has proper competition"""
    try:
        from enhanced_simulator_fixed import FixedGAELPEnvironment
        
        env = FixedGAELPEnvironment()
        
        wins = 0
        for _ in range(100):
            obs = env.reset()
            action = {
                'bid': 3.0,
                'quality_score': 0.75,
                'channel': 'google',
                'audience_segment': 'concerned_parents'
            }
            obs, reward, done, info = env.step(action)
            if info['auction']['won']:
                wins += 1
        
        win_rate = wins / 100
        print(f"\nWin Rate after fix: {win_rate:.2%}")
        
        if win_rate > 0.9:
            print("❌ Still broken - win rate too high")
            return False
        elif win_rate < 0.1:
            print("⚠️ Win rate very low - may need bid adjustment")
            return True
        else:
            print("✅ FIXED! Auction has proper competition")
            return True
            
    except Exception as e:
        print(f"❌ Error testing: {e}")
        return False

if __name__ == "__main__":
    verify_fix()
