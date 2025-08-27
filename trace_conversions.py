#!/usr/bin/env python3
"""Trace why conversions aren't happening"""

import numpy as np
import random

# Simulate conversion logic from the dashboard
def trace_conversion_probability():
    segments = ['crisis_parent', 'researcher', 'budget_conscious', 'tech_savvy']
    
    base_cvr = {
        'crisis_parent': 0.035,      # 3.5% - urgent need
        'researcher': 0.008,          # 0.8% - just browsing
        'budget_conscious': 0.012,    # 1.2% - need free trial
        'tech_savvy': 0.022           # 2.2% - feature match
    }
    
    print("=== CONVERSION PROBABILITY ANALYSIS ===\n")
    
    total_conversions = 0
    total_clicks = 0
    
    for segment in segments:
        print(f"\n{segment.upper()}:")
        print(f"  Base CVR: {base_cvr[segment]*100:.1f}%")
        
        # Simulate 100 clicks for this segment
        conversions = 0
        for _ in range(100):
            # Quality score
            quality_score = np.random.beta(7, 3)  # Skewed towards good quality
            
            # Price resistance
            price_resistance = 0.7 if segment == 'budget_conscious' else 0.9
            
            # Actual CVR
            actual_cvr = base_cvr[segment] * quality_score * price_resistance
            
            if random.random() < actual_cvr:
                conversions += 1
        
        print(f"  Avg Quality Score: ~{np.mean([np.random.beta(7, 3) for _ in range(1000)]):.2f}")
        print(f"  Price Resistance: {price_resistance}")
        print(f"  Effective CVR: ~{base_cvr[segment] * 0.7 * price_resistance * 100:.2f}%")
        print(f"  Expected conversions per 100 clicks: {conversions}")
        
        total_conversions += conversions
        total_clicks += 100
    
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total clicks simulated: {total_clicks}")
    print(f"Total conversions: {total_conversions}")
    print(f"Overall CVR: {total_conversions/total_clicks*100:.2f}%")
    
    print(f"\n=== WHAT THIS MEANS ===")
    print(f"With 17 clicks, expected conversions: {17 * total_conversions/total_clicks:.2f}")
    print(f"Probability of 0 conversions with 17 clicks: {(1 - total_conversions/total_clicks)**17 * 100:.1f}%")
    
    # Check segment distribution of actual clicks
    import requests
    r = requests.get('http://localhost:8080/api/status')
    data = r.json()
    
    print(f"\n=== ACTUAL SYSTEM DATA ===")
    segments_data = data.get('segment_performance', {})
    for seg, stats in segments_data.items():
        if stats['clicks'] > 0:
            print(f"{seg}: {stats['clicks']} clicks, {stats['conversions']} conversions")
            if stats['conversions'] > 0:
                print(f"  Actual CVR: {stats['conversions']/stats['clicks']*100:.1f}%")

if __name__ == '__main__':
    trace_conversion_probability()