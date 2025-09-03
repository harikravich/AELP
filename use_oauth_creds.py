#!/usr/bin/env python3
"""
Use the OAuth credentials you just created
"""

import json
import sys
import webbrowser
from pathlib import Path

print("="*60)
print("ENTER YOUR OAUTH CREDENTIALS")
print("="*60)

print("\nPaste the credentials from Google Cloud Console:")
client_id = input("Client ID: ").strip()
client_secret = input("Client Secret: ").strip()

if not client_id or not client_secret:
    print("❌ Both Client ID and Secret are required!")
    sys.exit(1)

# Save the credentials
oauth_config = {
    "installed": {
        "client_id": client_id,
        "client_secret": client_secret,
        "project_id": "aura-thrive-platform",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
    }
}

config_file = Path("google_ads_oauth.json")
with open(config_file, "w") as f:
    json.dump(oauth_config, f, indent=2)

print(f"\n✅ Saved OAuth config to {config_file}")

# Now get the refresh token
print("\n" + "="*60)
print("GETTING REFRESH TOKEN")
print("="*60)

try:
    from google_auth_oauthlib.flow import Flow
except ImportError:
    print("Installing required package...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "google-auth-oauthlib"], check=True)
    from google_auth_oauthlib.flow import Flow

# Create flow
flow = Flow.from_client_config(
    oauth_config,
    scopes=["https://www.googleapis.com/auth/adwords"],
)

flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

# Get authorization URL
auth_url, _ = flow.authorization_url(
    prompt="consent",
    access_type="offline",
    include_granted_scopes="true"
)

print("\n1. Opening browser for authorization...")
print("\n2. If browser doesn't open, visit this URL:")
print(auth_url)
print("\n3. Sign in with an account that has access to Aura's Google Ads")
print("\n4. Grant all requested permissions")
print("\n5. Copy the authorization code")

# Try to open browser
try:
    webbrowser.open(auth_url)
except:
    pass

print("\n" + "="*60)
code = input("\n📋 Paste authorization code here: ").strip()

# Exchange for tokens
try:
    flow.fetch_token(code=code)
    
    print("\n✅ SUCCESS! Got refresh token")
    
    # Save complete credentials
    creds = {
        "developer_token": "uikJ5kqLnGlrXdgzeYwYtg",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": flow.credentials.refresh_token
    }
    
    creds_file = Path("aura_google_ads_creds.json")
    with open(creds_file, "w") as f:
        json.dump(creds, f, indent=2)
    
    print(f"\n💾 Credentials saved to {creds_file}")
    
    # Update .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            lines = f.readlines()
        
        # Update lines
        updated = []
        found_section = False
        for line in lines:
            if "# Google Ads API Credentials" in line:
                found_section = True
            
            if line.startswith("GOOGLE_ADS_DEVELOPER_TOKEN="):
                updated.append(f"GOOGLE_ADS_DEVELOPER_TOKEN={creds['developer_token']}\n")
            elif line.startswith("GOOGLE_ADS_CLIENT_ID="):
                updated.append(f"GOOGLE_ADS_CLIENT_ID={client_id}\n")
            elif line.startswith("GOOGLE_ADS_CLIENT_SECRET="):
                updated.append(f"GOOGLE_ADS_CLIENT_SECRET={client_secret}\n")
            elif line.startswith("GOOGLE_ADS_REFRESH_TOKEN="):
                updated.append(f"GOOGLE_ADS_REFRESH_TOKEN={flow.credentials.refresh_token}\n")
            else:
                updated.append(line)
        
        with open(env_file, "w") as f:
            f.writelines(updated)
        
        print("✅ Updated .env file")
    
    print("\n" + "="*60)
    print("NEXT STEP: GET CUSTOMER ID")
    print("="*60)
    print("\nYou need the Aura Google Ads Customer ID (10-digit number)")
    print("Ask the Aura team or check the Google Ads interface")
    print("\nOnce you have it, run:")
    print("  export GOOGLE_ADS_CUSTOMER_ID=1234567890")
    print("  python3 pull_aura_google_ads_data.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you signed in with the correct account")
    print("2. Granted all permissions")
    print("3. Copied the complete authorization code")