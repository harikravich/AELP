#!/usr/bin/env python3
"""Trace competitive bidding and user profiles"""

import requests
import numpy as np
import random
from collections import defaultdict

def trace_competition_and_users():
    r = requests.get('http://localhost:8080/api/status')
    data = r.json()
    
    print("=== COMPETITIVE BIDDING ANALYSIS ===\n")
    
    # Check win rate
    episodes = data['episode_count']
    impressions = data['metrics']['total_impressions']
    win_rate = impressions / episodes * 100 if episodes > 0 else 0
    
    print(f"Episodes (auctions): {episodes}")
    print(f"Impressions won: {impressions}")
    print(f"Win rate: {win_rate:.1f}%")
    
    if win_rate > 90:
        print("⚠️  VERY HIGH WIN RATE - Competition may not be working!")
        print("   Should be losing more auctions if competitors are bidding")
    elif win_rate < 10:
        print("✅ LOW WIN RATE - Strong competition working")
    else:
        print("✅ MODERATE WIN RATE - Competition seems realistic")
    
    # Simulate the auction logic to verify
    print("\n=== SIMULATING AUCTION LOGIC ===")
    
    keyword_value = {
        'urgent parental controls': 3.50,
        'child safety emergency': 4.20,
        'compare parental control apps': 2.80,
        'best screen time app': 2.60,
    }
    
    our_bid = 3.0  # Typical bid
    wins = 0
    total_price = 0
    
    for _ in range(100):
        # Simulate competitor bids (from dashboard code)
        keyword = random.choice(list(keyword_value.keys()))
        base_value = keyword_value[keyword]
        num_competitors = random.randint(3, 8)
        
        competitor_bids = []
        for _ in range(num_competitors):
            comp_bid = base_value * np.random.lognormal(0, 0.3)
            competitor_bids.append(comp_bid)
        
        all_bids = sorted(competitor_bids + [our_bid], reverse=True)
        our_position = all_bids.index(our_bid) + 1
        
        if our_position <= 4:  # Top 4 win
            wins += 1
            winning_price = all_bids[our_position] if our_position < len(all_bids) else our_bid * 0.9
            total_price += winning_price
    
    print(f"Simulated 100 auctions with $3 bid:")
    print(f"  Wins: {wins}/100 ({wins}%)")
    print(f"  Avg winning price: ${total_price/wins:.2f}" if wins > 0 else "  No wins")
    print(f"  Competitor bids ranged from ${min(competitor_bids):.2f} to ${max(competitor_bids):.2f}")
    
    print("\n=== USER PROFILE ANALYSIS ===\n")
    
    # Check segment distribution
    segments = data.get('segment_performance', {})
    total_impressions = sum(s['impressions'] for s in segments.values())
    
    print("Segment distribution of impressions:")
    for seg, stats in segments.items():
        pct = stats['impressions'] / total_impressions * 100 if total_impressions > 0 else 0
        print(f"  {seg}: {stats['impressions']} ({pct:.1f}%)")
    
    # Check for repeat users
    print("\n=== CHECKING USER BEHAVIOR ===")
    
    # The dashboard should be tracking repeat users
    # Let's check the event log for frequency capping
    events = data.get('event_log', [])
    
    frequency_caps = [e for e in events if 'frequency_cap' in e.get('message', '')]
    conversions = [e for e in events if 'Conversion' in e.get('message', '')]
    
    print(f"Frequency cap events: {len(frequency_caps)}")
    if frequency_caps:
        print("✅ Frequency capping is working - users hitting impression limits")
        for e in frequency_caps[-3:]:
            print(f"  {e['message']}")
    else:
        print("⚠️  No frequency capping events - users may not be returning")
    
    print(f"\nConversions with impression count:")
    if conversions:
        for e in conversions:
            print(f"  {e['message']}")
    else:
        print("  No conversions yet")
    
    # Check if CTR is declining with impressions (ad fatigue)
    print("\n=== AD FATIGUE ANALYSIS ===")
    
    if impressions > 0 and data['metrics']['total_clicks'] > 0:
        overall_ctr = data['metrics']['total_clicks'] / impressions * 100
        print(f"Overall CTR: {overall_ctr:.2f}%")
        
        if overall_ctr < 2:
            print("✅ CTR is realistically low - ad fatigue may be working")
        elif overall_ctr > 10:
            print("⚠️  CTR too high - ad fatigue may not be working")
        else:
            print("✅ CTR in reasonable range")

if __name__ == '__main__':
    trace_competition_and_users()