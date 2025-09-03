#!/usr/bin/env python3
"""
Test Google Ads MCP connection with provided API key
"""

import os
import json
import subprocess

print("="*60)
print("TESTING GOOGLE ADS MCP CONNECTION")
print("="*60)

# The API key you provided
API_KEY = "uikJ5kqLnGlrXdgzeYwYtg"

print(f"\n🔑 Using API Key: {API_KEY}")
print("\n📡 Attempting to connect via MCP tools...")

# First, let's try to use the MCP tools directly with claude command
# Note: MCP tools for Google Ads typically need more than just an API key
# They need OAuth2 credentials, but let's test what we can access

print("\n1️⃣ Testing basic MCP connectivity...")

# Try to call a Google Ads MCP function to test connection
# Since we don't have full OAuth credentials, let's see what's accessible

try:
    # Test if we can get any response from the MCP server
    result = subprocess.run(
        ["claude", "mcp", "call", "google-ads", "test_connection"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        print("✅ MCP server is responding")
        print(f"Response: {result.stdout}")
    else:
        print("⚠️ MCP server returned an error")
        print(f"Error: {result.stderr}")
        
except subprocess.TimeoutExpired:
    print("❌ MCP server timed out")
except Exception as e:
    print(f"❌ Error testing MCP: {e}")

print("\n2️⃣ Checking what credentials are needed...")
print("""
Google Ads API typically requires:
1. Developer Token (this might be your API key: {})
2. OAuth2 Client ID
3. OAuth2 Client Secret  
4. OAuth2 Refresh Token
5. Customer ID (the account to access)

The API key alone might be:
- A Developer Token (most likely)
- Or a simplified access token for read-only access
""".format(API_KEY))

print("\n3️⃣ Let's try to use it as a Developer Token...")

# Create a test configuration
test_config = {
    "developer_token": API_KEY,
    "use_proto_plus": True
}

config_file = "/home/hariravichandran/AELP/test_google_ads_config.json"
with open(config_file, "w") as f:
    json.dump(test_config, f, indent=2)

print(f"✅ Created test config at {config_file}")

print("\n4️⃣ Alternative: Direct API test...")
print("Since you have read-only access, the API key might allow direct REST API calls")

# Test with curl to see if it's a REST API key
test_url = "https://googleads.googleapis.com/v14/customers:listAccessibleCustomers"

print(f"\nTesting REST API endpoint: {test_url}")

curl_command = f"""curl -H "Authorization: Bearer {API_KEY}" \
  -H "Developer-Token: {API_KEY}" \
  "{test_url}" 2>/dev/null"""

result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)

if result.stdout:
    try:
        response = json.loads(result.stdout)
        if "error" in response:
            print(f"❌ API Error: {response['error'].get('message', 'Unknown error')}")
            print("\nThis suggests the API key needs additional OAuth2 authentication")
        else:
            print("✅ API Response received!")
            print(json.dumps(response, indent=2))
    except json.JSONDecodeError:
        print(f"Response (non-JSON): {result.stdout[:500]}")
else:
    print("No response from API")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("""
Based on the API key format (uikJ5kqLnGlrXdgzeYwYtg), this appears to be
a Google Ads Developer Token.

To complete the connection, we need:
1. OAuth2 credentials for authentication
2. The Customer ID for the Aura account

Options:
A. Use Google's OAuth2 flow to get credentials (requires browser)
B. Use a service account if one was created for read-only access
C. Check if Aura provided additional credentials beyond the API key

Would you like me to:
1. Set up OAuth2 authentication flow?
2. Try alternative connection methods?
3. Create a mock data pipeline for testing while we get full access?
""")