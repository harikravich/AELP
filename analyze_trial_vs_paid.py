#!/usr/bin/env python3
"""
Analyze Trial vs Paid from Product Names
"""

# Product names from our extraction
products = [
    ("D2C Aura 2022 LP FT", 969),  # FT = Free Trial
    ("D2C Aura 2022 - Up to 50%", 1018),  # No FT = Direct to Pay
    ("D2C Aura 2022 - FT + 50%", 304),  # FT = Free Trial
    ("D2C Aura 2022 - FT + 40%", 21),  # FT = Free Trial
    ("Aura-NonCR-Parental-GW-direct-to-pay_d23", 177),  # D2P
    ("AV Intro Pricing - Antivirus, No DBOO - $35.99/yr", 397),  # No DBOO = D2P
    ("Aura Privacy Intro Pricing - Privacy - Free Trial - $35.99", 9),  # Free Trial
    ("D2C Aura – JDI – Antivirus Plus 30d FT", 4),  # FT
]

# Categorize
trial_indicators = ['FT', 'Free Trial', 'free trial', '30d FT', '14D Trial']
d2p_indicators = ['direct-to-pay', 'direct_to_pay', 'd2p', 'D2P', 'No DBOO', 'No CC']

trials = []
direct_pay = []
unclear = []

for name, count in products:
    is_trial = any(indicator in name for indicator in trial_indicators)
    is_d2p = any(indicator in name for indicator in d2p_indicators)
    
    if is_trial:
        trials.append((name, count))
    elif is_d2p:
        direct_pay.append((name, count))
    else:
        # Check for other patterns
        if '%' in name and 'FT' not in name:
            direct_pay.append((name, count))
        else:
            unclear.append((name, count))

print("="*60)
print("TRIAL VS DIRECT-TO-PAY ANALYSIS")
print("="*60)

print("\n🎯 FREE TRIALS (Get CC, bill after trial):")
print("-"*40)
trial_total = 0
for name, count in trials:
    print(f"  {count:4d} - {name}")
    trial_total += count
print(f"\nTOTAL TRIALS: {trial_total}")

print("\n💳 DIRECT-TO-PAY (Immediate payment):")
print("-"*40)
d2p_total = 0
for name, count in direct_pay:
    print(f"  {count:4d} - {name}")
    d2p_total += count
print(f"\nTOTAL D2P: {d2p_total}")

print("\n❓ UNCLEAR (Need more info):")
print("-"*40)
unclear_total = 0
for name, count in unclear:
    print(f"  {count:4d} - {name}")
    unclear_total += count
print(f"\nTOTAL UNCLEAR: {unclear_total}")

print("\n📊 SUMMARY:")
print(f"  Trials: {trial_total} ({trial_total/(trial_total+d2p_total)*100:.1f}%)")
print(f"  D2P: {d2p_total} ({d2p_total/(trial_total+d2p_total)*100:.1f}%)")
print(f"  Trial-to-Paid @ 70%: {trial_total * 0.7:.0f} customers")
print(f"  Total Real Customers: {d2p_total + trial_total * 0.7:.0f}")

print("\n⚠️  WHY THIS MATTERS FOR GAELP:")
print("  1. Can bid MORE for D2P (100% convert)")
print("  2. Must bid LESS for trials (70% convert)")
print("  3. Different LTV curves (trials may churn more)")
print("  4. Attribution windows differ (trials convert over 14 days)")
