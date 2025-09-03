#!/usr/bin/env python3
"""
Set up OAuth2 authentication for Aura Google Ads account
Using the Developer Token: uikJ5kqLnGlrXdgzeYwYtg
"""

import os
import json
import webbrowser
from pathlib import Path

print("="*60)
print("AURA GOOGLE ADS OAUTH2 SETUP")
print("="*60)

DEVELOPER_TOKEN = "uikJ5kqLnGlrXdgzeYwYtg"

print(f"\n✅ Developer Token: {DEVELOPER_TOKEN}")
print("\n📋 To access the Aura Google Ads account, we need OAuth2 credentials.")
print("\nSTEP 1: Create OAuth2 Client")
print("-" * 40)
print("""
1. Go to: https://console.cloud.google.com/apis/credentials
2. Select the project associated with the Aura account
3. Click '+ CREATE CREDENTIALS' → 'OAuth client ID'
4. Choose 'Desktop app' as the application type
5. Name it: 'GAELP Read-Only Access'
6. Click 'CREATE'
7. Download the JSON file with credentials
""")

print("\n⚠️ IMPORTANT: You need access to the Google Cloud Project")
print("   associated with the Aura Google Ads account")

print("\nSTEP 2: Get Customer ID")
print("-" * 40)
print("""
1. Log into Google Ads: https://ads.google.com
2. Look at the top right corner for the Customer ID
3. It's a 10-digit number like: 123-456-7890
4. Remove the dashes when entering it
""")

print("\nSTEP 3: Generate Refresh Token")
print("-" * 40)

# Create the OAuth helper script
oauth_script = """
#!/usr/bin/env python3
import sys
import json
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request

def get_refresh_token(client_id, client_secret):
    # Configure OAuth2 flow
    flow = Flow.from_client_config(
        {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=["https://www.googleapis.com/auth/adwords"],
    )
    
    # Use out-of-band redirect
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    
    # Get authorization URL
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        access_type="offline"
    )
    
    print("\\n🌐 AUTHORIZATION REQUIRED")
    print("="*50)
    print("\\n1. Open this URL in your browser:")
    print(auth_url)
    print("\\n2. Log in with the account that has access to Aura Google Ads")
    print("3. Grant read-only access to Google Ads")
    print("4. Copy the authorization code")
    print("\\n" + "="*50)
    
    # Get authorization code from user
    code = input("\\n📋 Paste authorization code here: ").strip()
    
    # Exchange for tokens
    try:
        flow.fetch_token(code=code)
        refresh_token = flow.credentials.refresh_token
        
        if refresh_token:
            print(f"\\n✅ SUCCESS! Your refresh token:")
            print(f"   {refresh_token}")
            print("\\n⚠️ Keep this token secure - it provides access to the account")
            return refresh_token
        else:
            print("\\n❌ No refresh token received. Make sure to:")
            print("   - Use 'consent' prompt")
            print("   - Grant offline access")
            return None
            
    except Exception as e:
        print(f"\\n❌ Error getting token: {e}")
        return None

if __name__ == "__main__":
    print("\\nEnter your OAuth2 Client credentials:")
    client_id = input("Client ID: ").strip()
    client_secret = input("Client Secret: ").strip()
    
    if client_id and client_secret:
        refresh_token = get_refresh_token(client_id, client_secret)
        
        if refresh_token:
            # Save configuration
            config = {
                "developer_token": "uikJ5kqLnGlrXdgzeYwYtg",
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token
            }
            
            config_file = "aura_oauth_config.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"\\n💾 Configuration saved to {config_file}")
    else:
        print("\\n❌ Client ID and Secret are required")
"""

# Save the OAuth helper script
oauth_file = Path("/home/hariravichandran/AELP/generate_aura_oauth.py")
oauth_file.write_text(oauth_script)
oauth_file.chmod(0o755)

print(f"\n✅ OAuth helper script created: {oauth_file}")

print("\n" + "="*60)
print("QUICK SETUP COMMAND")
print("="*60)
print("\nRun this command to generate your refresh token:")
print(f"  python3 {oauth_file}")

print("\n" + "="*60)
print("ALTERNATIVE: Use Existing Credentials")
print("="*60)
print("""
If Aura already has OAuth2 credentials set up:
1. Ask for the existing refresh token
2. Ask for the Client ID and Client Secret
3. Ask for the Customer ID

Then create a file called 'aura_credentials.json' with:
{
  "developer_token": "uikJ5kqLnGlrXdgzeYwYtg",
  "client_id": "YOUR_CLIENT_ID",
  "client_secret": "YOUR_CLIENT_SECRET", 
  "refresh_token": "YOUR_REFRESH_TOKEN",
  "customer_id": "1234567890"
}
""")

print("\n" + "="*60)
print("ONCE YOU HAVE CREDENTIALS")
print("="*60)
print("""
After you have all credentials, run:
  python3 pull_aura_google_ads_data.py

This will:
1. Connect to the Aura Google Ads account
2. Pull all campaign, keyword, and audience data
3. Create training datasets for GAELP
4. Integrate real Aura patterns into the model
""")

# Create a simplified test script that can work with limited access
test_script = """
#!/usr/bin/env python3
'''Test Aura connection with whatever credentials we have'''

import os
import json
from pathlib import Path

print("\\n🔍 Looking for Aura credentials...")

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
    print("\\nUsing known Developer Token: uikJ5kqLnGlrXdgzeYwYtg")
    print("\\nTo complete setup, you need:")
    print("1. OAuth2 Client ID & Secret")
    print("2. Refresh Token") 
    print("3. Customer ID")
"""

test_file = Path("/home/hariravichandran/AELP/test_aura_connection.py")
test_file.write_text(test_script)
test_file.chmod(0o755)

print(f"\n✅ Test script created: {test_file}")