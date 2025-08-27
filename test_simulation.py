#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

from gaelp_live_dashboard_enhanced import GAELPLiveSystemEnhanced

print("Creating system...")
system = GAELPLiveSystemEnhanced()

print("Starting simulation...")
system.start_simulation()

print(f"Is running: {system.is_running}")
print(f"Episode count: {system.episode_count}")

# Wait a bit
import time
time.sleep(2)

print(f"After 2 seconds - Episode count: {system.episode_count}")
print(f"Metrics: {system.metrics}")
