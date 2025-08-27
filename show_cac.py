#!/usr/bin/env python3
"""
Show actual Customer Acquisition Cost (CAC) from the running GAELP system
"""

import requests
import json
import time
from datetime import datetime

def get_cac_metrics():
    """Get current CAC metrics from the dashboard API"""
    try:
        response = requests.get('http://localhost:8080/api/status')
        data = response.json()
        
        metrics = data['metrics']
        segments = data.get('segment_performance', {})
        
        print("\n" + "="*60)
        print("GAELP SYSTEM - ACTUAL CAC ANALYSIS")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Episodes Run: {data['episode_count']:,}")
        
        # Overall metrics
        print("\n📊 OVERALL METRICS:")
        print("-"*40)
        
        impressions = metrics['total_impressions']
        clicks = metrics['total_clicks']
        conversions = metrics['total_conversions']
        spend = metrics['total_spend']
        revenue = metrics['total_revenue']
        
        # Calculate actual CAC
        cac = spend / conversions if conversions > 0 else 0
        
        print(f"Total Ad Spend: ${spend:,.2f}")
        print(f"Total Impressions: {impressions:,}")
        print(f"Total Clicks: {clicks:,}")
        print(f"Total Conversions: {conversions}")
        print(f"Total Revenue: ${revenue:,.2f}")
        
        print("\n💰 CUSTOMER ACQUISITION COST (CAC):")
        print("-"*40)
        print(f"CAC (Cost per Customer): ${cac:.2f}")
        
        # Aura's actual pricing for context
        aura_monthly_price = 12.99  # Aura's actual monthly subscription
        aura_annual_price = 99.99   # Aura's actual annual subscription
        aura_ltv_estimate = aura_annual_price * 2  # Assume 2 year average customer lifetime
        
        print(f"\nAura Pricing Context:")
        print(f"  Monthly Plan: ${aura_monthly_price}")
        print(f"  Annual Plan: ${aura_annual_price}")
        print(f"  Estimated LTV: ${aura_ltv_estimate:.2f} (2-year avg)")
        
        # CAC efficiency
        if cac > 0:
            months_to_breakeven = cac / aura_monthly_price
            cac_to_ltv_ratio = cac / aura_ltv_estimate
            
            print(f"\n📈 CAC Efficiency:")
            print(f"  Months to Break-even: {months_to_breakeven:.1f}")
            print(f"  CAC:LTV Ratio: {cac_to_ltv_ratio:.2%}")
            
            if cac_to_ltv_ratio < 0.33:
                print(f"  Status: ✅ EXCELLENT (CAC < 33% of LTV)")
            elif cac_to_ltv_ratio < 0.5:
                print(f"  Status: 🟡 GOOD (CAC < 50% of LTV)")
            else:
                print(f"  Status: ⚠️  NEEDS OPTIMIZATION (CAC > 50% of LTV)")
        
        # Segment breakdown
        print("\n👥 CAC BY SEGMENT:")
        print("-"*40)
        
        for segment_name, segment_data in segments.items():
            seg_conversions = segment_data.get('conversions', 0)
            seg_spend = segment_data.get('spend', 0)
            seg_revenue = segment_data.get('revenue', 0)
            
            if seg_conversions > 0:
                seg_cac = seg_spend / seg_conversions
                seg_roi = ((seg_revenue - seg_spend) / seg_spend * 100) if seg_spend > 0 else 0
                
                print(f"\n{segment_name.replace('_', ' ').title()}:")
                print(f"  Conversions: {seg_conversions}")
                print(f"  CAC: ${seg_cac:.2f}")
                print(f"  ROI: {seg_roi:.1f}%")
                
                # Segment efficiency
                if seg_cac < aura_annual_price:
                    print(f"  Efficiency: ✅ Profitable on first year")
                elif seg_cac < aura_ltv_estimate:
                    print(f"  Efficiency: 🟡 Profitable over lifetime")
                else:
                    print(f"  Efficiency: ❌ Not profitable")
        
        # Learning progress
        print("\n🧠 LEARNING PROGRESS:")
        print("-"*40)
        
        # Get bid values from Thompson Sampling
        arm_stats = data.get('arm_stats', {})
        if arm_stats:
            best_arm = max(arm_stats.items(), key=lambda x: x[1]['value'])
            print(f"Best Strategy: {best_arm[0]} (value: {best_arm[1]['value']:.3f})")
        
        # Trend analysis
        print(f"\nCurrent ROI: {metrics['current_roi']:.1f}%")
        
        if metrics['current_roi'] > 0:
            print("Trend: ✅ System is profitable!")
        else:
            print("Trend: 🔄 System is learning and optimizing...")
        
        print("\n💡 INSIGHTS:")
        print("-"*40)
        
        # Provide actionable insights
        if cac > aura_ltv_estimate:
            print("⚠️  CAC exceeds LTV - Need to:")
            print("   - Improve targeting to high-value segments")
            print("   - Optimize bid strategy")
            print("   - Improve conversion rate")
        elif cac > aura_annual_price:
            print("🟡 CAC exceeds annual price - Consider:")
            print("   - Focus on crisis parents (higher conversion)")
            print("   - Reduce bids on low-converting segments")
            print("   - Improve creative messaging")
        else:
            print("✅ CAC is healthy! Continue optimizing:")
            print("   - Scale successful segments")
            print("   - Test higher bids on converting keywords")
            print("   - Expand keyword coverage")
        
        # Real-time optimization suggestions
        if segments:
            best_segment = min(
                [(k, v['spend']/v['conversions']) for k, v in segments.items() 
                 if v.get('conversions', 0) > 0],
                key=lambda x: x[1],
                default=(None, 0)
            )
            
            if best_segment[0]:
                print(f"\n🎯 Best performing segment: {best_segment[0].replace('_', ' ').title()}")
                print(f"   CAC: ${best_segment[1]:.2f}")
                print(f"   → Allocate more budget here!")
        
        return cac
        
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None

def monitor_cac(interval=10):
    """Monitor CAC over time"""
    print("Starting CAC monitoring... (Press Ctrl+C to stop)")
    
    cac_history = []
    
    while True:
        try:
            cac = get_cac_metrics()
            if cac:
                cac_history.append(cac)
                
                # Show trend
                if len(cac_history) > 1:
                    change = cac_history[-1] - cac_history[-2]
                    if change < 0:
                        print(f"\n📉 CAC improving! (decreased by ${abs(change):.2f})")
                    elif change > 0:
                        print(f"\n📈 CAC increasing (increased by ${change:.2f})")
                    else:
                        print(f"\n➡️  CAC stable")
            
            print(f"\n{'='*60}")
            print(f"Next update in {interval} seconds...")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            if cac_history:
                print(f"Average CAC over session: ${sum(cac_history)/len(cac_history):.2f}")
                print(f"Best CAC achieved: ${min(cac_history):.2f}")
            break

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        # Continuous monitoring mode
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        monitor_cac(interval)
    else:
        # Single snapshot
        get_cac_metrics()