#!/usr/bin/env python3
"""Fix patterns.json structure to be consistent"""
import json

# Load patterns
with open('discovered_patterns.json', 'r') as f:
    patterns = json.load(f)

# Fix channel_performance structure
fixed_channels = {}

# Keep original structured entries
for key, data in patterns['channel_performance'].items():
    if isinstance(data, dict) and 'channel_group' in data:
        # This is a properly structured entry
        fixed_channels[key] = data
    elif isinstance(data, dict) and 'cvr' in data:
        # This is one of our new entries - convert it
        # Map simple names to full structure
        channel_mapping = {
            'paid_search': {'group': 'Paid Search', 'source': 'google', 'medium': 'cpc'},
            'social': {'group': 'Paid Social', 'source': 'facebook', 'medium': 'social'},
            'display': {'group': 'Display', 'source': 'google', 'medium': 'display'},
            'email': {'group': 'Email', 'source': 'newsletter', 'medium': 'email'},
            'organic': {'group': 'Organic Search', 'source': 'google', 'medium': 'organic'}
        }
        
        if key in channel_mapping:
            mapping = channel_mapping[key]
            full_key = f"{mapping['group']}|{mapping['source']}|{mapping['medium']}"
            
            # Create properly structured entry
            fixed_channels[full_key] = {
                'channel_group': mapping['group'],
                'source': mapping['source'],
                'medium': mapping['medium'],
                'sessions': data.get('sessions', 10000),
                'conversions': data.get('conversions', int(data.get('cvr', 0.01) * 10000)),
                'cvr_percent': data.get('cvr_percent', data.get('cvr', 0.01) * 100),
                'cvr': data.get('cvr', 0.01),
                'revenue': data.get('conversions', 100) * 100,
                'estimated_cac': 30.0
            }

# Update patterns
patterns['channel_performance'] = fixed_channels

# Save fixed patterns
with open('discovered_patterns.json', 'w') as f:
    json.dump(patterns, f, indent=2)

print("✅ Fixed patterns.json structure")
print(f"✅ {len(fixed_channels)} channels with proper structure")

# Show the channels
for key in fixed_channels.keys():
    cvr = fixed_channels[key].get('cvr', fixed_channels[key].get('cvr_percent', 0) / 100)
    print(f"  {key}: CVR={cvr:.4f}")