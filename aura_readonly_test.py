#!/usr/bin/env python3
"""
Test if we can access Aura Google Ads with just the developer token
Sometimes read-only access is configured differently
"""

import os
import json
import requests

print("="*60)
print("TESTING AURA READ-ONLY ACCESS")
print("="*60)

DEVELOPER_TOKEN = "uikJ5kqLnGlrXdgzeYwYtg"

print(f"\n🔑 Developer Token: {DEVELOPER_TOKEN}")
print("\n📡 Testing different access methods...")

# Method 1: Test if it's a simplified API key
print("\n1️⃣ Testing as API Key...")
headers = {
    "Authorization": f"Bearer {DEVELOPER_TOKEN}",
    "x-api-key": DEVELOPER_TOKEN,
    "developer-token": DEVELOPER_TOKEN
}

# Try Google Ads REST API v15 (latest)
base_url = "https://googleads.googleapis.com/v15"
endpoints = [
    "/customers:listAccessibleCustomers",
    "/googleAds:search"
]

for endpoint in endpoints:
    url = base_url + endpoint
    print(f"\nTesting: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            print("  ✅ SUCCESS! Got response")
            data = response.json()
            print(f"  Data: {json.dumps(data, indent=2)[:500]}")
        elif response.status_code == 401:
            print("  ❌ Authentication required - need OAuth2")
        elif response.status_code == 403:
            print("  ❌ Forbidden - missing permissions")
        else:
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

# Method 2: Check if there's a custom endpoint for Aura
print("\n2️⃣ Testing Aura-specific endpoints...")
aura_endpoints = [
    "https://api.aurahealth.io/google-ads/data",
    "https://aura.app/api/google-ads",
    "https://life360.com/api/ads/google"
]

for url in aura_endpoints:
    print(f"\nTrying: {url}")
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {DEVELOPER_TOKEN}"},
            timeout=3
        )
        if response.status_code < 500:
            print(f"  Endpoint exists! Status: {response.status_code}")
    except:
        print(f"  Not accessible")

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print("""
The developer token alone is not sufficient for direct API access.
You need OAuth2 credentials to authenticate.

Next steps:
1. Get OAuth2 credentials from the Aura team, OR
2. Create new OAuth2 credentials if you have access to their Google Cloud Project
3. Use the generate_aura_oauth.py script in a separate terminal

In the meantime, we can:
- Use the existing GA4 data for training (751K users)
- Create synthetic Google Ads data based on GA4 patterns
- Set up a pipeline ready for when we get full access
""")