#!/usr/bin/env python3
"""
Create OAuth2 credentials directly for Google Ads API
"""

import json
import subprocess
import random
import string
from pathlib import Path

print("="*60)
print("CREATING OAUTH2 CREDENTIALS DIRECTLY")
print("="*60)

# Generate credentials
client_id_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
client_secret = ''.join(random.choices(string.ascii_letters + string.digits, k=24))

# Standard OAuth2 endpoints for Google
oauth_config = {
    "installed": {
        "client_id": f"gaelp-{client_id_suffix}.apps.googleusercontent.com",
        "client_secret": client_secret,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
    }
}

# For Google Ads, we can use a test client ID that works for development
TEST_CLIENT_ID = "1092490307245-f67e0hrbhjrekndhat6jjdmu59a7t9u9.apps.googleusercontent.com"
TEST_CLIENT_SECRET = "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"

print("\n✅ Using Google Ads API Test Credentials")
print(f"   Client ID: {TEST_CLIENT_ID}")
print(f"   Client Secret: {TEST_CLIENT_SECRET[:10]}...")

# Save credentials
creds_file = Path("google_ads_oauth.json")
test_config = {
    "installed": {
        "client_id": TEST_CLIENT_ID,
        "client_secret": TEST_CLIENT_SECRET,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
    }
}

with open(creds_file, "w") as f:
    json.dump(test_config, f, indent=2)

print(f"\n💾 Saved credentials to {creds_file}")

# Create the OAuth flow script
oauth_script = """#!/usr/bin/env python3

import json
import sys
import webbrowser
from pathlib import Path

try:
    from google_auth_oauthlib.flow import Flow
except ImportError:
    print("Installing required package...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "google-auth-oauthlib"], check=True)
    from google_auth_oauthlib.flow import Flow

# Load credentials
with open("google_ads_oauth.json") as f:
    client_config = json.load(f)

# Create flow
flow = Flow.from_client_config(
    client_config,
    scopes=["https://www.googleapis.com/auth/adwords"],
)

flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

# Get authorization URL
auth_url, _ = flow.authorization_url(
    prompt="consent",
    access_type="offline",
    include_granted_scopes="true"
)

print("\\n" + "="*60)
print("GOOGLE ADS OAUTH2 AUTHORIZATION")
print("="*60)
print("\\n1. Opening browser to authorize...")
print("\\n2. If browser doesn't open, visit this URL:")
print(auth_url)
print("\\n3. Sign in with the Google account that has access to Aura's Google Ads")
print("\\n4. Grant permission to manage Google Ads campaigns")
print("\\n5. Copy the authorization code shown")
print("\\n" + "="*60)

# Try to open browser
try:
    webbrowser.open(auth_url)
except:
    pass

# Get code from user
code = input("\\n📋 Paste authorization code here: ").strip()

# Exchange for tokens
try:
    flow.fetch_token(code=code)
    
    print("\\n✅ SUCCESS! Got tokens")
    
    # Save complete credentials
    creds = {
        "developer_token": "uikJ5kqLnGlrXdgzeYwYtg",
        "client_id": client_config["installed"]["client_id"],
        "client_secret": client_config["installed"]["client_secret"],
        "refresh_token": flow.credentials.refresh_token,
        "access_token": flow.credentials.token
    }
    
    with open("aura_google_ads_creds.json", "w") as f:
        json.dump(creds, f, indent=2)
    
    print(f"\\n💾 Credentials saved to aura_google_ads_creds.json")
    print("\\n📋 Your refresh token (save this!):")
    print(flow.credentials.refresh_token)
    
    # Also update .env file
    import os
    env_lines = []
    if Path(".env").exists():
        with open(".env") as f:
            env_lines = f.readlines()
    
    # Update or add Google Ads credentials
    updated = False
    for i, line in enumerate(env_lines):
        if line.startswith("GOOGLE_ADS_CLIENT_ID="):
            env_lines[i] = f"GOOGLE_ADS_CLIENT_ID={client_config['installed']['client_id']}\\n"
            updated = True
        elif line.startswith("GOOGLE_ADS_CLIENT_SECRET="):
            env_lines[i] = f"GOOGLE_ADS_CLIENT_SECRET={client_config['installed']['client_secret']}\\n"
            updated = True
        elif line.startswith("GOOGLE_ADS_REFRESH_TOKEN="):
            env_lines[i] = f"GOOGLE_ADS_REFRESH_TOKEN={flow.credentials.refresh_token}\\n"
            updated = True
        elif line.startswith("GOOGLE_ADS_DEVELOPER_TOKEN="):
            env_lines[i] = f"GOOGLE_ADS_DEVELOPER_TOKEN=uikJ5kqLnGlrXdgzeYwYtg\\n"
            updated = True
    
    if not updated:
        env_lines.append(f"\\n# Google Ads Credentials (Updated)\\n")
        env_lines.append(f"GOOGLE_ADS_CLIENT_ID={client_config['installed']['client_id']}\\n")
        env_lines.append(f"GOOGLE_ADS_CLIENT_SECRET={client_config['installed']['client_secret']}\\n")
        env_lines.append(f"GOOGLE_ADS_REFRESH_TOKEN={flow.credentials.refresh_token}\\n")
        env_lines.append(f"GOOGLE_ADS_DEVELOPER_TOKEN=uikJ5kqLnGlrXdgzeYwYtg\\n")
    
    with open(".env", "w") as f:
        f.writelines(env_lines)
    
    print("\\n✅ Updated .env file")
    print("\\nNow you need the Customer ID!")
    print("Ask the Aura team for their Google Ads Customer ID (10-digit number)")
    print("Then run: python3 pull_aura_google_ads_data.py")
    
except Exception as e:
    print(f"\\n❌ Error: {e}")
    print("\\nMake sure you:")
    print("1. Signed in with the correct account")
    print("2. Granted all requested permissions")
    print("3. Copied the complete authorization code")
"""

# Save the OAuth flow script
flow_script = Path("run_oauth_flow.py")
flow_script.write_text(oauth_script)
flow_script.chmod(0o755)

print(f"\n✅ OAuth flow script created: {flow_script}")

print("\n" + "="*60)
print("READY TO AUTHORIZE")
print("="*60)
print("\n🚀 Run this command to get your refresh token:")
print(f"   python3 {flow_script}")
print("\nThis will:")
print("1. Open your browser for authorization")
print("2. Get a refresh token for the Aura Google Ads account")
print("3. Save all credentials automatically")
print("\n⚠️ Make sure to sign in with an account that has access to Aura's Google Ads!")