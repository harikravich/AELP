
#!/usr/bin/env python3
'''Test Aura connection with whatever credentials we have'''

import os
import json
from pathlib import Path

print("\n🔍 Looking for Aura credentials...")

# Check for credential files
cred_files = [
    "aura_credentials.json",
    "aura_oauth_config.json",
    ".env"
]

found_creds = False
for file in cred_files:
    if Path(file).exists():
        print(f"✅ Found {file}")
        found_creds = True
        
        if file.endswith('.json'):
            with open(file) as f:
                creds = json.load(f)
                if 'developer_token' in creds:
                    print(f"   Developer Token: {creds['developer_token']}")
                if 'customer_id' in creds:
                    print(f"   Customer ID: {creds['customer_id']}")

if not found_creds:
    print("❌ No credential files found")
    print("\nUsing known Developer Token: uikJ5kqLnGlrXdgzeYwYtg")
    print("\nTo complete setup, you need:")
    print("1. OAuth2 Client ID & Secret")
    print("2. Refresh Token") 
    print("3. Customer ID")
