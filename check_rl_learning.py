#!/usr/bin/env python3
"""
Check if RL agent is actually learning by monitoring key metrics
"""

import requests
import json
import time
import numpy as np

def check_learning_progress():
    """Check if the RL agent in the dashboard is actually learning"""
    
    print("🔍 Checking RL Agent Learning Progress...")
    print("=" * 60)
    
    # Get initial metrics
    try:
        response = requests.get("http://localhost:5000/api/ai_arena")
        initial_data = response.json()
        
        print("📊 Initial State:")
        print(f"  Episodes: {initial_data['agent_evolution']['episodes_completed']}")
        print(f"  Learning Rate: {initial_data['agent_evolution']['learning_rate']}")
        print(f"  Skill Points: {initial_data['agent_evolution']['skill_points']}")
        print(f"  ELO Rating: {initial_data['self_play_tournament']['elo_rating']}")
        
        # Check war room for actual performance
        war_response = requests.get("http://localhost:5000/api/war_room")
        war_data = war_response.json()
        
        if 'kpis' in war_data:
            print(f"\n💰 Performance Metrics:")
            print(f"  Impressions: {war_data['kpis']['impressions']}")
            print(f"  Clicks: {war_data['kpis']['clicks']}")
            print(f"  Conversions: {war_data['kpis']['conversions']}")
            print(f"  Spend: ${war_data['kpis']['spend']:.2f}")
            print(f"  Revenue: ${war_data['kpis']['revenue']:.2f}")
        else:
            print(f"\n💰 Performance Metrics: Not yet available")
        
        # Wait and check again
        print("\n⏳ Waiting 10 seconds to observe changes...")
        time.sleep(10)
        
        # Get updated metrics
        response2 = requests.get("http://localhost:5000/api/ai_arena")
        updated_data = response2.json()
        
        war_response2 = requests.get("http://localhost:5000/api/war_room")
        war_data2 = war_response2.json()
        
        print("\n📊 After 10 seconds:")
        print(f"  Episodes: {updated_data['agent_evolution']['episodes_completed']} (Δ: {updated_data['agent_evolution']['episodes_completed'] - initial_data['agent_evolution']['episodes_completed']})")
        print(f"  Skill Points: {updated_data['agent_evolution']['skill_points']} (Δ: {updated_data['agent_evolution']['skill_points'] - initial_data['agent_evolution']['skill_points']})")
        print(f"  ELO Rating: {updated_data['self_play_tournament']['elo_rating']} (Δ: {updated_data['self_play_tournament']['elo_rating'] - initial_data['self_play_tournament']['elo_rating']})")
        
        if 'kpis' in war_data2 and 'kpis' in war_data:
            print(f"\n💰 Performance Changes:")
            print(f"  Impressions: {war_data2['kpis']['impressions']} (Δ: {war_data2['kpis']['impressions'] - war_data['kpis']['impressions']})")
            print(f"  Conversions: {war_data2['kpis']['conversions']} (Δ: {war_data2['kpis']['conversions'] - war_data['kpis']['conversions']})")
        
        # Check if replay buffer is filling
        print("\n🧠 Checking Learning Components:")
        
        # Try to get more detailed metrics
        try:
            exec_response = requests.get("http://localhost:5000/api/executive")
            exec_data = exec_response.json()
            
            if 'learning_metrics' in exec_data:
                print(f"  Loss: {exec_data['learning_metrics'].get('loss', 'N/A')}")
                print(f"  Replay Buffer Size: {exec_data['learning_metrics'].get('buffer_size', 'N/A')}")
                print(f"  Updates Performed: {exec_data['learning_metrics'].get('updates', 'N/A')}")
        except:
            pass
        
        # Diagnosis
        print("\n🔬 DIAGNOSIS:")
        
        episodes_changed = updated_data['agent_evolution']['episodes_completed'] != initial_data['agent_evolution']['episodes_completed']
        metrics_changed = False
        if 'kpis' in war_data2 and 'kpis' in war_data:
            metrics_changed = war_data2['kpis']['impressions'] != war_data['kpis']['impressions']
        
        if not episodes_changed and not metrics_changed:
            print("❌ AGENT NOT LEARNING - No changes detected!")
            print("\nPossible issues:")
            print("  1. Simulation may be paused")
            print("  2. Replay buffer might not be filling")
            print("  3. Update frequency might be too low")
            print("  4. Agent might be in evaluation mode only")
            
            print("\n🔧 Suggested fixes:")
            print("  1. Check if simulation is running: self.master.start_simulation()")
            print("  2. Verify experience collection: self.master.rl_agent.memory")
            print("  3. Check update frequency: self.master.rl_agent.update_interval")
            print("  4. Ensure training mode: self.master.rl_agent.training = True")
            
        elif episodes_changed and not metrics_changed:
            print("⚠️ AGENT RUNNING BUT NOT IMPROVING")
            print("  - Episodes are incrementing but performance is static")
            print("  - Possible issues: Poor reward signal, exploration too low, or stuck in local optimum")
            
        else:
            print("✅ AGENT IS LEARNING!")
            print("  - Episodes incrementing and metrics changing")
            print("  - Monitor for consistent improvement over time")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to dashboard at http://localhost:5000")
        print("Make sure gaelp_live_dashboard_enhanced.py is running")
    except Exception as e:
        print(f"❌ Error checking learning: {e}")

if __name__ == "__main__":
    check_learning_progress()