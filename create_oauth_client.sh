#!/bin/bash

echo "Creating OAuth 2.0 Client for Google Ads API..."
echo "================================================"

PROJECT="aura-thrive-platform"

# Enable Google Ads API if not already enabled
echo "Enabling Google Ads API..."
gcloud services enable googleads.googleapis.com --project=$PROJECT

# Create OAuth 2.0 Client
echo ""
echo "Creating OAuth Client..."

# Generate a unique client name with timestamp
CLIENT_NAME="gaelp-oauth-$(date +%s)"

# Create the OAuth client using gcloud
gcloud alpha oauth-clients create $CLIENT_NAME \
  --project=$PROJECT \
  --type=desktop \
  --display-name="GAELP Google Ads Access" \
  --description="OAuth client for GAELP to access Google Ads with read-only permissions" \
  2>/dev/null

if [ $? -eq 0 ]; then
  echo "✅ OAuth Client created successfully!"
  echo ""
  echo "Getting client details..."
  
  # Get the client details
  gcloud alpha oauth-clients describe $CLIENT_NAME --project=$PROJECT --format=json > oauth_client.json
  
  CLIENT_ID=$(cat oauth_client.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('clientId', ''))")
  CLIENT_SECRET=$(cat oauth_client.json | python3 -c "import sys, json; print(json.load(sys.stdin).get('clientSecret', ''))")
  
  echo "CLIENT_ID: $CLIENT_ID"
  echo "CLIENT_SECRET: $CLIENT_SECRET"
  
  # Save to file for easy access
  cat > google_ads_oauth_creds.txt << EOF
CLIENT_ID=$CLIENT_ID
CLIENT_SECRET=$CLIENT_SECRET
DEVELOPER_TOKEN=uikJ5kqLnGlrXdgzeYwYtg
EOF
  
  echo ""
  echo "✅ Credentials saved to google_ads_oauth_creds.txt"
  echo ""
  echo "Now run: python3 generate_aura_oauth.py"
  echo "And use these credentials when prompted"
  
else
  echo ""
  echo "Alternative: Create manually in the Console"
  echo "==========================================="
  echo "1. Go to: https://console.cloud.google.com/apis/credentials?project=$PROJECT"
  echo "2. Click '+ CREATE CREDENTIALS' → 'OAuth client ID'"
  echo "3. Choose 'Desktop app'"
  echo "4. Name it: 'GAELP Google Ads Access'"
  echo "5. Click 'CREATE'"
  echo "6. Copy the Client ID and Client Secret"
fi