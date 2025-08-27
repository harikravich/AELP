#!/usr/bin/env python3
"""Check if the GAELP system is actually learning"""

import requests
import json
import time

def check_system():
    r = requests.get('http://localhost:8080/api/status')
    data = r.json()
    
    print('=== CURRENT STATE ===')
    print(f'Episodes: {data["episode_count"]}')
    print(f'Conversions: {data["metrics"]["total_conversions"]}')
    print(f'Spend: ${data["metrics"]["total_spend"]:.2f}')
    print(f'Revenue: ${data["metrics"]["total_revenue"]:.2f}')
    
    print('\n=== THOMPSON SAMPLING ARMS ===')
    arms = data.get('arm_stats', {})
    if arms:
        for name, stats in arms.items():
            print(f'{name}: value={stats["value"]:.3f}, alpha={stats["alpha"]}, beta={stats["beta"]}')
    else:
        print('NO ARMS DATA! The arms are not being tracked!')
    
    print('\n=== RECENT EVENTS ===')
    for event in data.get('event_log', [])[-5:]:
        print(f'{event["type"]}: {event["message"]}')
    
    # Check if touchpoints are being tracked
    print('\n=== CHECKING INTERNALS ===')
    
    # Wait for a conversion
    print('Waiting for a conversion to test learning...')
    for i in range(60):
        time.sleep(5)
        r = requests.get('http://localhost:8080/api/status')
        data = r.json()
        if data["metrics"]["total_conversions"] > 0:
            print(f'\n✅ CONVERSION DETECTED after {data["episode_count"]} episodes!')
            print(f'CAC: ${data["metrics"]["total_spend"] / data["metrics"]["total_conversions"]:.2f}')
            
            # Check if arms updated
            arms = data.get('arm_stats', {})
            if arms and any(arm['alpha'] != 1 or arm['beta'] != 1 for arm in arms.values()):
                print('✅ ARMS ARE UPDATING!')
                for name, stats in arms.items():
                    print(f'  {name}: alpha={stats["alpha"]}, beta={stats["beta"]}')
            else:
                print('❌ ARMS NOT UPDATING DESPITE CONVERSION!')
            
            # Check for learning events
            learning_events = [e for e in data.get('event_log', []) if 'RL Update' in e.get('message', '')]
            if learning_events:
                print(f'✅ Found {len(learning_events)} RL update events')
                for e in learning_events[-3:]:
                    print(f'  {e["message"]}')
            else:
                print('❌ NO RL UPDATE EVENTS FOUND!')
            
            return True
        
        print(f'  Check {i+1}: {data["episode_count"]} episodes, {data["metrics"]["total_conversions"]} conversions')
    
    print('\n❌ NO CONVERSIONS AFTER 5 MINUTES - System may not be learning!')
    return False

if __name__ == '__main__':
    check_system()