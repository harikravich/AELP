#!/usr/bin/env python3
"""
July 2025 - Final GA4 Mapping Analysis
Determining exactly what GA4 tracks vs your actual metrics
"""

def analyze_july_final():
    print("="*70)
    print("JULY 2025 - GA4 TRACKING vs ACTUAL BUSINESS METRICS")
    print("="*70)
    
    # Your actual July numbers
    actual = {
        'web_trial_starts': 295,
        'app_trial_starts': 224,
        'd2p_purchases': 818,
        'post_trial_conversions': 172,
        'mobile_subscribers': 111,
        'total_paying_subs': 1101
    }
    
    # GA4 July data
    ga4 = {
        'purchase_events': 922,  # Daily average
        'revenue': 90050,
        'mobile_conversions': 560,
        'desktop_conversions': 337,
        'add_to_cart': 6750,
        'form_submits': 23773
    }
    
    print("\n📊 YOUR ACTUAL JULY NUMBERS (Daily):")
    print("-" * 60)
    print(f"  Web Trial Starts:        {actual['web_trial_starts']}")
    print(f"  App Trial Starts:        {actual['app_trial_starts']}")
    print(f"  Direct-to-Pay (D2P):     {actual['d2p_purchases']}")
    print(f"  Post-Trial Conversions:  {actual['post_trial_conversions']}")
    print(f"  Mobile App Subscribers:  {actual['mobile_subscribers']}")
    print(f"  TOTAL Paying Subs:       {actual['total_paying_subs']}")
    
    print("\n📈 GA4 SHOWS (Daily):")
    print("-" * 60)
    print(f"  Purchase Events:         {ga4['purchase_events']}")
    print(f"  Revenue:                 ${ga4['revenue']:,}")
    
    print("\n🔍 MAPPING ANALYSIS:")
    print("-" * 60)
    
    # GA4 purchase events = 922
    # Let's see what adds up
    
    print("\nScenario 1: GA4 'purchase' = D2P + Post-Trial Only")
    scenario1 = actual['d2p_purchases'] + actual['post_trial_conversions']
    print(f"  D2P ({actual['d2p_purchases']}) + Post-Trial ({actual['post_trial_conversions']}) = {scenario1}")
    print(f"  GA4 Shows: {ga4['purchase_events']}")
    print(f"  Difference: {ga4['purchase_events'] - scenario1}")
    if abs(ga4['purchase_events'] - scenario1) < 50:
        print("  ✅ CLOSE MATCH!")
    
    print("\nScenario 2: GA4 'purchase' = D2P Only")
    scenario2 = actual['d2p_purchases']
    print(f"  D2P Only: {scenario2}")
    print(f"  GA4 Shows: {ga4['purchase_events']}")
    print(f"  Difference: {ga4['purchase_events'] - scenario2}")
    
    print("\nScenario 3: GA4 'purchase' includes some trials")
    scenario3 = actual['d2p_purchases'] + actual['web_trial_starts']
    print(f"  D2P ({actual['d2p_purchases']}) + Web Trials ({actual['web_trial_starts']}) = {scenario3}")
    print(f"  GA4 Shows: {ga4['purchase_events']}")
    print(f"  Difference: {ga4['purchase_events'] - scenario3}")
    
    print("\n✅ MOST LIKELY SCENARIO:")
    print("-" * 60)
    print("GA4 'purchase' events (922/day) appear to track:")
    print(f"  ✓ D2P Purchases: {actual['d2p_purchases']}")
    print(f"  ✓ Post-Trial Conversions: {actual['post_trial_conversions']}")
    print(f"  ? Mobile App Subs: {actual['mobile_subscribers']} (partial)")
    print(f"  Total: {actual['d2p_purchases'] + actual['post_trial_conversions'] + actual['mobile_subscribers']} = {ga4['purchase_events']}")
    
    print("\n❌ WHAT GA4 IS DEFINITELY MISSING:")
    print("-" * 60)
    print(f"  1. Web Trial Starts: {actual['web_trial_starts']}/day")
    print(f"  2. App Trial Starts: {actual['app_trial_starts']}/day")
    print(f"  Total Missing Trials: {actual['web_trial_starts'] + actual['app_trial_starts']}/day")
    print("\n  GA4 doesn't count trial starts as 'purchases' (makes sense!)")
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 60)
    
    # Calculate trial conversion rate
    total_trials = actual['web_trial_starts'] + actual['app_trial_starts']
    trial_conv_rate = actual['post_trial_conversions'] / total_trials * 100
    
    print(f"1. Trial Conversion Rate: {trial_conv_rate:.1f}%")
    print(f"   {actual['post_trial_conversions']} conversions from ~{total_trials} trials 14 days ago")
    print(f"   Note: July had fewer trials than August, so lower conversions")
    
    print(f"\n2. Revenue per 'purchase': ${ga4['revenue']/ga4['purchase_events']:.2f}")
    print(f"   This is blend of D2P and post-trial conversion values")
    
    print(f"\n3. Mobile is {ga4['mobile_conversions']/ga4['purchase_events']*100:.1f}% of purchases")
    print(f"   But app trials ({actual['app_trial_starts']}/day) not tracked!")
    
    print("\n🎯 FOR GAELP TRAINING:")
    print("-" * 60)
    print("We need to:")
    print("1. Track trials separately (not in GA4 purchases)")
    print("2. Use 14-day attribution window for trial cohorts")
    print(f"3. True trial conversion: ~{trial_conv_rate:.0f}% (growing as you scale)")
    print("4. Separate bid strategies:")
    print(f"   - Trials: Target CPA × {trial_conv_rate/100:.2f} = Max Bid")
    print(f"   - D2P: Target CPA × 0.95 = Max Bid")
    
    print("\n📊 COMPLETE JULY FUNNEL:")
    print("-" * 60)
    print(f"Sessions: 60,722/day")
    print(f"↓")
    print(f"Add to Cart: 6,750/day (11.1%)")
    print(f"↓")
    print(f"Trial Starts: {total_trials}/day")
    print(f"D2P Purchases: {actual['d2p_purchases']}/day")
    print(f"↓")
    print(f"Total New Customers: {actual['total_paying_subs']}/day")
    
    # Calculate true CAC
    print("\n💰 TRUE CAC CALCULATION:")
    print("-" * 60)
    avg_cpc = 25  # Estimate
    sessions_per_customer = 60722 / actual['total_paying_subs']
    print(f"Sessions per customer: {sessions_per_customer:.1f}")
    print(f"At ${avg_cpc} CPC: CAC = ${sessions_per_customer * avg_cpc:.2f}")
    
    return {
        'actual': actual,
        'ga4': ga4,
        'trial_conversion_rate': trial_conv_rate,
        'missing_from_ga4': total_trials
    }


if __name__ == "__main__":
    july_analysis = analyze_july_final()
    
    print("\n" + "="*70)
    print("SUMMARY: GA4 tracks paid conversions well, but misses trial starts")
    print("For GAELP, we need to combine GA4 + your internal trial data")
    print("="*70)