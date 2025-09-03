
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
    
    print("\n🌐 AUTHORIZATION REQUIRED")
    print("="*50)
    print("\n1. Open this URL in your browser:")
    print(auth_url)
    print("\n2. Log in with the account that has access to Aura Google Ads")
    print("3. Grant read-only access to Google Ads")
    print("4. Copy the authorization code")
    print("\n" + "="*50)
    
    # Get authorization code from user
    code = input("\n📋 Paste authorization code here: ").strip()
    
    # Exchange for tokens
    try:
        flow.fetch_token(code=code)
        refresh_token = flow.credentials.refresh_token
        
        if refresh_token:
            print(f"\n✅ SUCCESS! Your refresh token:")
            print(f"   {refresh_token}")
            print("\n⚠️ Keep this token secure - it provides access to the account")
            return refresh_token
        else:
            print("\n❌ No refresh token received. Make sure to:")
            print("   - Use 'consent' prompt")
            print("   - Grant offline access")
            return None
            
    except Exception as e:
        print(f"\n❌ Error getting token: {e}")
        return None

if __name__ == "__main__":
    print("\nEnter your OAuth2 Client credentials:")
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
            
            print(f"\n💾 Configuration saved to {config_file}")
    else:
        print("\n❌ Client ID and Secret are required")
