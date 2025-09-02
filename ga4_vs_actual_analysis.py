#!/usr/bin/env python3
"""
GA4 vs Actual Business Metrics Analysis
Identifying gaps in GA4 tracking
"""

import json
from pathlib import Path

def analyze_gaps():
    """Compare GA4 data with actual business metrics"""
    
    print("="*70)
    print("GA4 vs ACTUAL BUSINESS METRICS - AUGUST 2025")
    print("="*70)
    
    # Your actual numbers (daily averages)
    actual = {
        'web_trial_starts': 298,
        'app_trial_starts': 236,
        'total_trial_starts': 534,
        'd2p_purchases': 861,
        'post_trial_conversions': 160,
        'mobile_subscribers': 112,
        'total_paying_subs': 1134
    }
    
    # GA4 data (August totals)
    ga4_august = {
        'total_conversions': 29745,
        'mobile_conversions': 18054,
        'desktop_conversions': 10769,
        'tablet_conversions': 922,
        'revenue': 2919271
    }
    
    # Calculate GA4 daily averages
    ga4_daily = {
        'total_conversions': ga4_august['total_conversions'] / 31,
        'mobile_conversions': ga4_august['mobile_conversions'] / 31,
        'desktop_conversions': ga4_august['desktop_conversions'] / 31,
        'daily_revenue': ga4_august['revenue'] / 31
    }
    
    print("\n📊 YOUR ACTUAL NUMBERS (Daily):")
    print("-" * 50)
    print(f"  Web Trial Starts:        {actual['web_trial_starts']:,}")
    print(f"  App Trial Starts:        {actual['app_trial_starts']:,}")
    print(f"  Direct-to-Pay:           {actual['d2p_purchases']:,}")
    print(f"  Post-Trial Conversions:  {actual['post_trial_conversions']:,}")
    print(f"  Mobile Subscribers:      {actual['mobile_subscribers']:,}")
    print(f"  TOTAL Paying Subs:       {actual['total_paying_subs']:,}")
    
    print("\n📈 GA4 SHOWS (Daily Avg):")
    print("-" * 50)
    print(f"  Total Conversions:       {ga4_daily['total_conversions']:,.0f}")
    print(f"  Mobile Conversions:      {ga4_daily['mobile_conversions']:,.0f}")
    print(f"  Desktop Conversions:     {ga4_daily['desktop_conversions']:,.0f}")
    print(f"  Daily Revenue:           ${ga4_daily['daily_revenue']:,.2f}")
    
    print("\n🔍 ANALYSIS:")
    print("-" * 50)
    
    # GA4 "conversions" appear to include trial starts
    ga4_conversion_events = ga4_daily['total_conversions']
    likely_composition = actual['web_trial_starts'] + actual['d2p_purchases'] + actual['post_trial_conversions']
    
    print(f"  GA4 Daily Conversions:   {ga4_conversion_events:.0f}")
    print(f"  Your Trial+D2P+PostTrial: {likely_composition}")
    print(f"  Difference:              {ga4_conversion_events - likely_composition:.0f}")
    
    print("\n  GA4 'conversions' likely includes:")
    print(f"    ✓ Web trial starts ({actual['web_trial_starts']})")
    print(f"    ✓ D2P purchases ({actual['d2p_purchases']})")
    print(f"    ✓ Post-trial conversions ({actual['post_trial_conversions']})")
    print(f"    ? App trial starts ({actual['app_trial_starts']}) - MAYBE")
    print(f"    ? Mobile app subscribers ({actual['mobile_subscribers']}) - UNCLEAR")
    
    print("\n❌ WHAT GA4 MIGHT BE MISSING:")
    print("-" * 50)
    print("  1. App trial starts (236/day) - May not fire GA4 events")
    print("  2. Mobile app subscribers (112/day) - App store subscriptions")
    print("  3. Trial-to-paid attribution - Hard to connect trial → payment")
    print("  4. Cohort tracking - Which trial cohort converted when")
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 50)
    
    # Trial conversion rate
    trial_conversion_rate = actual['post_trial_conversions'] / actual['total_trial_starts']
    print(f"  1. Actual Trial→Paid Rate: {trial_conversion_rate:.1%} (160/534)")
    print(f"     Much lower than assumed 70%!")
    
    # Revenue per conversion
    revenue_per_conversion = ga4_daily['daily_revenue'] / ga4_daily['total_conversions']
    print(f"  2. Revenue per 'conversion': ${revenue_per_conversion:.2f}")
    print(f"     Mix of $0 trials and paid purchases")
    
    # Mobile vs Desktop
    mobile_pct = ga4_daily['mobile_conversions'] / ga4_daily['total_conversions']
    print(f"  3. Mobile is {mobile_pct:.1%} of conversions")
    print(f"     Critical for Balance (iOS only!)")
    
    print("\n🎯 FOR GAELP TRAINING:")
    print("-" * 50)
    print("  1. REAL trial conversion: 30% not 70%!")
    print("  2. Must track app vs web separately")
    print("  3. Post-trial conversions lag 14+ days")
    print("  4. Mobile/iOS targeting crucial for Balance")
    print("  5. GA4 missing ~348 app events daily")
    
    # Calculate true CAC implications
    print("\n💰 TRUE CAC IMPLICATIONS:")
    print("-" * 50)
    
    # If trials convert at 30% not 70%
    trial_cpa_target = 60  # Example
    true_trial_cac = trial_cpa_target / 0.30
    assumed_trial_cac = trial_cpa_target / 0.70
    
    print(f"  If target CPA = ${trial_cpa_target}:")
    print(f"    At 70% conversion: CAC = ${assumed_trial_cac:.2f}")
    print(f"    At 30% conversion: CAC = ${true_trial_cac:.2f}")
    print(f"    Difference: ${true_trial_cac - assumed_trial_cac:.2f} higher!")
    print(f"\n  Must bid 57% LESS for trials than we thought!")
    
    # Save analysis
    output_dir = Path("ga4_extracted_data")
    output_dir.mkdir(exist_ok=True)
    
    analysis = {
        'actual_metrics': actual,
        'ga4_metrics': ga4_august,
        'ga4_daily': ga4_daily,
        'gaps': {
            'missing_app_trials': actual['app_trial_starts'],
            'missing_mobile_subs': actual['mobile_subscribers'],
            'total_missing_daily': actual['app_trial_starts'] + actual['mobile_subscribers']
        },
        'insights': {
            'real_trial_conversion_rate': trial_conversion_rate,
            'revenue_per_conversion': revenue_per_conversion,
            'mobile_percentage': mobile_pct,
            'true_trial_cac': true_trial_cac
        }
    }
    
    with open(output_dir / "ga4_vs_actual_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✅ Analysis saved to {output_dir}/ga4_vs_actual_analysis.json")


if __name__ == "__main__":
    analyze_gaps()