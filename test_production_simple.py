#!/usr/bin/env python3
"""
Simple test of production orchestrator with discovered patterns
"""

import os
import json
from pathlib import Path

print("="*60)
print("TESTING PRODUCTION ORCHESTRATOR")
print("="*60)

# Check what data we have
patterns_file = Path("discovered_patterns.json")
if patterns_file.exists():
    with open(patterns_file) as f:
        patterns = json.load(f)
    
    print(f"\n✅ Found discovered_patterns.json")
    print(f"   Segments: {len(patterns.get('user_segments', {}))}")
    print(f"   Channels: {len(patterns.get('channel_performance', {}))}")
    
    # Show segment info
    if 'user_segments' in patterns:
        print("\n📊 User Segments:")
        for seg_name, seg_data in patterns['user_segments'].items():
            print(f"   - {seg_name}: CVR={seg_data.get('conversion_rate', 0)*100:.2f}%")
else:
    print("❌ No discovered_patterns.json found")

# Set environment for production
os.environ['GAELP_ENV'] = 'production'
os.environ['GAELP_DRY_RUN'] = 'true'
os.environ['GAELP_EPISODES'] = '1'  # Just 1 episode for testing

print("\n🚀 Starting production orchestrator...")
print("   Environment: production")
print("   Dry run: true")
print("   Episodes: 1")

# Try importing and running
try:
    from gaelp_production_orchestrator import ProductionOrchestrator
    
    orchestrator = ProductionOrchestrator()
    print("\n✅ Orchestrator initialized")
    
    # Check components
    print("\n📦 Components status:")
    for comp_name, comp in orchestrator.components.items():
        if comp:
            print(f"   ✅ {comp_name}")
        else:
            print(f"   ❌ {comp_name}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)