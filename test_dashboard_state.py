#!/usr/bin/env python3
"""Test dashboard state to see why simulation isn't updating"""

import requests
import time
import json

base_url = "http://localhost:5000"

print("Testing dashboard state...")

# Check initial status
response = requests.get(f"{base_url}/api/status")
data = response.json()
print(f"\nInitial state: is_running={data.get('is_running', False)}")

# Start the simulation
print("\nStarting simulation...")
response = requests.post(f"{base_url}/api/start")
print(f"Start response: {response.json()}")

# Check status after starting
for i in range(5):
    time.sleep(1)
    response = requests.get(f"{base_url}/api/dashboard_data")
    data = response.json()
    print(f"After {i+1}s: running={data.get('is_running')}, episode={data.get('episode_count')}, impressions={data.get('total_impressions')}")
    
# Check the actual dashboard object state
response = requests.get(f"{base_url}/api/status")
status = response.json()
print(f"\nFinal status: {json.dumps(status, indent=2)[:500]}...")