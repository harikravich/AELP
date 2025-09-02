import numpy as np

# Competitor base bids from fixed_auction_system.py
profiles = [
    {'name': 'Qustodio', 'base_bid': 5.00, 'aggression': 1.2, 'variance': 0.15, 'budget_factor': 1.0},
    {'name': 'Bark', 'base_bid': 6.00, 'aggression': 1.4, 'variance': 0.20, 'budget_factor': 1.2},
    {'name': 'Circle', 'base_bid': 5.50, 'aggression': 1.1, 'variance': 0.10, 'budget_factor': 1.1},
    {'name': 'Norton', 'base_bid': 4.50, 'aggression': 1.0, 'variance': 0.12, 'budget_factor': 0.9},
    {'name': 'Competitor5', 'base_bid': 3.50, 'aggression': 0.9, 'variance': 0.18, 'budget_factor': 0.8},
    {'name': 'Competitor6', 'base_bid': 4.00, 'aggression': 1.0, 'variance': 0.15, 'budget_factor': 1.0}
]

# Calculate bids for crisis hours with crisis intent
print('Competitor bids in crisis scenario (hour=23, intent=crisis):')
print('=' * 60)
for p in profiles:
    base = p['base_bid']
    # Crisis hour multiplier
    base *= 1.45
    # Crisis intent multiplier  
    base *= p['aggression'] * 1.8
    # Average variance (no randomness)
    bid = base * 1.0
    # Budget cap
    max_bid = p['base_bid'] * p['budget_factor'] * 2.5
    final_bid = min(bid, max_bid)
    print(f'{p["name"]:15} Base=${p["base_bid"]:.2f} -> Crisis bid=${final_bid:.2f}')

print('\n' + '=' * 60)
print('Our agent:')
print(f'  Max bid: $8.10 (capped by safety system from $10)')
print(f'  Current bid range: $2.00 - $15.00 (but capped at $10)')
print('\nProblem: ALL competitors bid > $8.10 in crisis scenarios!')
print('Solution: Need to either:')
print('  1. Increase safety system max_bid_absolute')
print('  2. Target non-crisis hours/intents where competition is lower')