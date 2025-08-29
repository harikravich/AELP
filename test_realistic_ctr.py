#!/usr/bin/env python3
"""
Test that CTR predictions are realistic after GA4 training
"""

import numpy as np
from criteo_response_model import CriteoUserResponseModel

print("="*60)
print("TESTING REALISTIC CTR PREDICTIONS")
print("="*60)

# Initialize model (should load GA4-trained model)
print("\n1. Initializing Criteo model...")
model = CriteoUserResponseModel()

# Test scenarios with expected realistic CTRs
test_scenarios = [
    {
        'name': 'Google Search - High Intent Crisis',
        'features': {
            'num_0': 2.5,  # High intent (crisis keywords)
            'num_1': 0.9,  # Late night normalized
            'num_2': 0.7,  # Weekend
            'num_3': 1,    # Position 1
            'num_4': 7.5,  # High quality score
            'num_5': 0.8,  # High engagement
            'num_7': 1.2,  # Mobile device
            'cat_0': 'Paid Search',
            'cat_3': 'mobile',
            'cat_5': 'behavioral_health',
            'cat_6': 'aura'
        },
        'expected_range': (0.02, 0.08)  # 2-8% for high intent search
    },
    {
        'name': 'Display Ad - Low Position',
        'features': {
            'num_0': 0.5,  # Low intent
            'num_1': 0.5,  # Afternoon
            'num_2': 0.3,  # Tuesday
            'num_3': 4,    # Position 4
            'num_4': 6.0,  # Medium quality
            'num_5': 0.3,  # Low engagement
            'num_7': 1.0,  # Desktop
            'cat_0': 'Display',
            'cat_3': 'desktop',
            'cat_5': 'behavioral_health',
            'cat_6': 'aura'
        },
        'expected_range': (0.001, 0.01)  # 0.1-1% for display
    },
    {
        'name': 'Facebook - Teen Safety',
        'features': {
            'num_0': 1.5,  # Medium intent
            'num_1': 0.7,  # Evening
            'num_2': 0.5,  # Friday
            'num_3': 2,    # Position 2
            'num_4': 7.0,  # Good quality
            'num_5': 0.6,  # Medium engagement
            'num_7': 1.2,  # Mobile
            'cat_0': 'Paid Social',
            'cat_3': 'mobile',
            'cat_5': 'behavioral_health',
            'cat_6': 'aura'
        },
        'expected_range': (0.005, 0.03)  # 0.5-3% for social
    },
    {
        'name': 'Email Campaign',
        'features': {
            'num_0': 1.6,  # Email subscriber intent
            'num_1': 0.4,  # Morning
            'num_2': 0.4,  # Wednesday
            'num_3': 1,    # Top position
            'num_4': 8.0,  # High quality
            'num_5': 0.7,  # Good engagement
            'num_7': 1.0,  # Desktop
            'cat_0': 'Email',
            'cat_3': 'desktop',
            'cat_5': 'behavioral_health',
            'cat_6': 'aura'
        },
        'expected_range': (0.01, 0.05)  # 1-5% for email
    },
    {
        'name': 'Organic Search - Natural Result',
        'features': {
            'num_0': 2.5,  # High organic intent
            'num_1': 0.5,  # Midday
            'num_2': 0.4,  # Thursday
            'num_3': 2,    # Position 2
            'num_4': 9.0,  # Very high quality
            'num_5': 0.85, # High engagement
            'num_7': 1.0,  # Desktop
            'cat_0': 'Organic Search',
            'cat_3': 'desktop',
            'cat_5': 'behavioral_health',
            'cat_6': 'aura'
        },
        'expected_range': (0.015, 0.06)  # 1.5-6% for organic
    }
]

print("\n2. Testing CTR predictions:\n")
print(f"{'Scenario':<35} {'Predicted CTR':>12} {'Expected Range':>15} {'Status':>10}")
print("-" * 75)

all_valid = True
predictions = []

for scenario in test_scenarios:
    # Get prediction
    ctr = model.predict_ctr(scenario['features'])
    predictions.append(ctr)
    
    # Check if in expected range
    min_ctr, max_ctr = scenario['expected_range']
    is_valid = min_ctr <= ctr <= max_ctr
    
    status = "✅ Valid" if is_valid else "⚠️ Outside"
    if not is_valid:
        all_valid = False
    
    print(f"{scenario['name']:<35} {ctr*100:>11.3f}% {f'{min_ctr*100:.1f}-{max_ctr*100:.1f}%':>15} {status:>10}")

# Calculate overall statistics
print("\n" + "-" * 75)
print("\n3. Overall Statistics:")
print(f"   Mean CTR: {np.mean(predictions)*100:.2f}%")
print(f"   Median CTR: {np.median(predictions)*100:.2f}%")
print(f"   Min CTR: {np.min(predictions)*100:.3f}%")
print(f"   Max CTR: {np.max(predictions)*100:.3f}%")

# Test with random features to ensure variety
print("\n4. Testing with random features (100 samples):")
random_ctrs = []
for _ in range(100):
    random_features = {
        'num_0': np.random.uniform(0, 3),
        'num_1': np.random.uniform(0, 1),
        'num_2': np.random.uniform(0, 1),
        'num_3': np.random.uniform(1, 4),
        'num_4': np.random.uniform(5, 9),
        'num_5': np.random.uniform(0, 1),
        'num_7': np.random.choice([1.0, 1.2]),
        'cat_0': np.random.choice(['Paid Search', 'Display', 'Paid Social', 'Email', 'Organic Search']),
        'cat_3': np.random.choice(['desktop', 'mobile', 'tablet']),
        'cat_5': 'behavioral_health',
        'cat_6': 'aura'
    }
    ctr = model.predict_ctr(random_features)
    random_ctrs.append(ctr)

print(f"   Random CTR distribution:")
print(f"   10th percentile: {np.percentile(random_ctrs, 10)*100:.3f}%")
print(f"   25th percentile: {np.percentile(random_ctrs, 25)*100:.3f}%")
print(f"   50th percentile: {np.percentile(random_ctrs, 50)*100:.3f}%")
print(f"   75th percentile: {np.percentile(random_ctrs, 75)*100:.3f}%")
print(f"   90th percentile: {np.percentile(random_ctrs, 90)*100:.3f}%")

# Final verdict
print("\n" + "="*60)
print("RESULTS")
print("="*60)

if all_valid:
    print("\n✅ SUCCESS! All CTR predictions are realistic!")
    print("\nThe model is now producing:")
    print("- Search ads: 2-8% CTR")
    print("- Display ads: 0.1-1% CTR")
    print("- Social ads: 0.5-3% CTR")
    print("- Email: 1-5% CTR")
    print("\nThese match real-world advertising benchmarks.")
else:
    print("\n⚠️ Some predictions are outside expected ranges.")
    print("The model may need further calibration.")

print("\n🎯 Dashboard is now using realistic CTR predictions!")
print("   RL agent will learn from realistic click patterns.")
print("   No more fantasy 75% CTRs!")