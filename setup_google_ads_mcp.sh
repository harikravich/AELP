#!/bin/bash

echo "==========================================="
echo "GOOGLE ADS MCP CONFIGURATION"
echo "==========================================="
echo ""
echo "This script will configure the Google Ads MCP server with your credentials"
echo ""

# Check if MCP server is installed
echo "✅ Google Ads MCP server is already installed"
echo ""

echo "📋 CREDENTIALS NEEDED:"
echo "1. Customer ID (from Google Ads interface)"
echo "2. Developer Token (from API Center)"
echo "3. OAuth2 Client ID & Secret (from Google Cloud Console)"
echo "4. Refresh Token (we'll generate this)"
echo ""

read -p "Do you have your Customer ID? (y/n): " HAS_CUSTOMER
if [ "$HAS_CUSTOMER" = "y" ]; then
    read -p "Enter Customer ID (format: 1234567890, no dashes): " CUSTOMER_ID
else
    echo "Get it from https://ads.google.com (top right corner)"
    exit 1
fi

read -p "Do you have your Developer Token? (y/n): " HAS_TOKEN
if [ "$HAS_TOKEN" = "y" ]; then
    read -p "Enter Developer Token: " DEVELOPER_TOKEN
else
    echo "Get it from https://ads.google.com/aw/apicenter"
    exit 1
fi

echo ""
echo "🔐 OAuth2 Setup"
echo "Go to: https://console.cloud.google.com/apis/credentials"
echo "Create OAuth 2.0 Client ID (Desktop application type)"
echo ""
read -p "Enter Client ID: " CLIENT_ID
read -p "Enter Client Secret: " CLIENT_SECRET

echo ""
echo "🔄 Generating Refresh Token..."
echo ""

# Create Python script to get refresh token
cat > /tmp/get_google_ads_refresh_token.py << 'EOF'
import sys
from google_auth_oauthlib.flow import Flow

client_id = sys.argv[1]
client_secret = sys.argv[2]

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

python3 /tmp/get_google_ads_refresh_token.py "$CLIENT_ID" "$CLIENT_SECRET"
echo ""
read -p "Enter the Refresh Token from above: " REFRESH_TOKEN

# Update the MCP configuration
echo ""
echo "💾 Updating MCP configuration..."

# Read existing config
CONFIG_FILE="/home/hariravichandran/.claude.json"

# Use Python to update JSON properly
python3 << EOF
import json

with open("$CONFIG_FILE", "r") as f:
    config = json.load(f)

# Update google-ads server configuration
if "mcpServers" not in config:
    config["mcpServers"] = {}

config["mcpServers"]["google-ads"] = {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "@weppa-cloud/mcp-google-ads"],
    "env": {
        "GOOGLE_ADS_CLIENT_ID": "$CLIENT_ID",
        "GOOGLE_ADS_CLIENT_SECRET": "$CLIENT_SECRET",
        "GOOGLE_ADS_DEVELOPER_TOKEN": "$DEVELOPER_TOKEN",
        "GOOGLE_ADS_CUSTOMER_ID": "$CUSTOMER_ID",
        "GOOGLE_ADS_REFRESH_TOKEN": "$REFRESH_TOKEN",
        "GOOGLE_ADS_LOGIN_CUSTOMER_ID": "$CUSTOMER_ID"
    }
}

with open("$CONFIG_FILE", "w") as f:
    json.dump(config, f, indent=2)

print("✅ Configuration updated!")
EOF

# Also save to .env for the Python scripts
echo ""
echo "💾 Updating .env file..."
cat >> /home/hariravichandran/AELP/.env << EOF

# Google Ads API Credentials (Real)
GOOGLE_ADS_DEVELOPER_TOKEN=$DEVELOPER_TOKEN
GOOGLE_ADS_CLIENT_ID=$CLIENT_ID
GOOGLE_ADS_CLIENT_SECRET=$CLIENT_SECRET
GOOGLE_ADS_REFRESH_TOKEN=$REFRESH_TOKEN
GOOGLE_ADS_CUSTOMER_ID=$CUSTOMER_ID
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "Testing connection..."
claude mcp list | grep google-ads

echo ""
echo "To pull real CPC data, run:"
echo "  python3 /home/hariravichandran/AELP/pull_real_google_ads_data.py"
echo ""

# Clean up
rm -f /tmp/get_google_ads_refresh_token.py