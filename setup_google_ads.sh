#!/bin/bash

echo "=========================================="
echo "GOOGLE ADS API SETUP FOR GAELP"
echo "=========================================="
echo ""
echo "This script will help you set up Google Ads API access"
echo ""

# Install required packages
echo "📦 Installing Google Ads Python client..."
pip3 install google-ads --quiet

echo ""
echo "🔑 STEP 1: Get Developer Token"
echo "--------------------------------"
echo "1. Go to: https://ads.google.com/aw/apicenter"
echo "2. Sign in with your Google Ads account"
echo "3. Apply for API access (Basic access is free)"
echo "4. Copy your developer token"
echo ""
read -p "Enter your Developer Token: " DEVELOPER_TOKEN

echo ""
echo "🔐 STEP 2: Set up OAuth2 Credentials"
echo "-------------------------------------"
echo "1. Go to: https://console.cloud.google.com/apis/credentials"
echo "2. Create a new OAuth 2.0 Client ID"
echo "3. Application type: Desktop app"
echo "4. Download the credentials JSON"
echo ""
read -p "Enter your Client ID: " CLIENT_ID
read -p "Enter your Client Secret: " CLIENT_SECRET

echo ""
echo "🔄 STEP 3: Get Refresh Token"
echo "-----------------------------"
echo "We'll use Google's OAuth2 flow to get a refresh token..."
echo ""

# Create a temporary Python script to get refresh token
cat > get_refresh_token.py << 'EOF'
import sys
from google_auth_oauthlib.flow import Flow

client_id = sys.argv[1]
client_secret = sys.argv[2]

# OAuth2 flow
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

flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

auth_url, _ = flow.authorization_url(prompt="consent")

print("\n🌐 Open this URL in your browser:")
print(auth_url)
print("\n📋 After authorizing, copy the code and paste it here:")
code = input("Authorization code: ")

flow.fetch_token(code=code)
print(f"\n✅ Your refresh token: {flow.credentials.refresh_token}")
EOF

python3 get_refresh_token.py "$CLIENT_ID" "$CLIENT_SECRET"
echo ""
read -p "Enter your Refresh Token from above: " REFRESH_TOKEN

echo ""
echo "🏢 STEP 4: Customer ID"
echo "----------------------"
echo "Find your Customer ID in Google Ads (top right corner)"
echo "Format: XXX-XXX-XXXX (enter without dashes)"
echo ""
read -p "Enter your Customer ID: " CUSTOMER_ID

echo ""
echo "💾 Saving credentials to .env file..."
cat > .env << EOF
# Google Ads API Credentials
export GOOGLE_ADS_DEVELOPER_TOKEN="$DEVELOPER_TOKEN"
export GOOGLE_ADS_CLIENT_ID="$CLIENT_ID"
export GOOGLE_ADS_CLIENT_SECRET="$CLIENT_SECRET"
export GOOGLE_ADS_REFRESH_TOKEN="$REFRESH_TOKEN"
export GOOGLE_ADS_CUSTOMER_ID="$CUSTOMER_ID"
EOF

echo ""
echo "✅ Setup complete! Credentials saved to .env"
echo ""
echo "To use the credentials, run:"
echo "  source .env"
echo ""
echo "Then test the integration:"
echo "  python3 google_ads_integration.py"
echo ""

# Clean up
rm -f get_refresh_token.py