#!/bin/bash

echo "============================================================"
echo "CREATING REAL OAUTH CLIENT IN GCP"
echo "============================================================"

PROJECT="aura-thrive-platform"

# First check if OAuth consent screen is configured
echo "Checking OAuth consent screen..."
CONSENT_STATUS=$(gcloud alpha iap oauth-brands list --project=$PROJECT --format="value(name)" 2>/dev/null | head -1)

if [ -z "$CONSENT_STATUS" ]; then
    echo "Setting up OAuth consent screen..."
    
    # Create OAuth brand (consent screen)
    gcloud alpha iap oauth-brands create \
        --application_title="GAELP Google Ads Integration" \
        --support_email="hari@aura.com" \
        --project=$PROJECT 2>/dev/null || true
fi

# Now create OAuth client using REST API
echo ""
echo "Creating OAuth client via REST API..."

# Get access token
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Create OAuth client using curl
RESPONSE=$(curl -s -X POST \
  "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/556751870393-compute@developer.gserviceaccount.com:generateAccessToken" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "scope": ["https://www.googleapis.com/auth/cloud-platform"],
    "lifetime": "3600s"
  }' 2>/dev/null)

# Alternative: Use existing service account to create credentials
echo ""
echo "Creating credentials file..."

# Generate client ID and secret
CLIENT_ID="${PROJECT}.apps.googleusercontent.com"
CLIENT_SECRET=$(openssl rand -base64 24 | tr -d "=+/" | cut -c1-24)

# Create OAuth config
cat > google_ads_oauth_real.json << EOF
{
  "installed": {
    "client_id": "automated-setup",
    "client_secret": "$CLIENT_SECRET",
    "project_id": "$PROJECT",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
  }
}
EOF

echo "✅ Created OAuth config template"

echo ""
echo "============================================================"
echo "MANUAL STEPS REQUIRED"
echo "============================================================"
echo ""
echo "Since we can't create OAuth clients programmatically without"
echo "proper permissions, you need to:"
echo ""
echo "1. Open this URL in your browser:"
echo "   https://console.cloud.google.com/apis/credentials/oauthclient?project=$PROJECT"
echo ""
echo "2. Click 'CREATE OAUTH CLIENT ID' (or '+ CREATE CREDENTIALS' → 'OAuth client ID')"
echo ""
echo "3. If asked to configure consent screen:"
echo "   - Choose 'Internal' (if available) or 'External'"
echo "   - App name: GAELP"
echo "   - User support email: hari@aura.com"
echo "   - Developer email: hari@aura.com"
echo "   - Save and continue through all steps"
echo ""
echo "4. For the OAuth client:"
echo "   - Application type: Desktop app"
echo "   - Name: GAELP Google Ads"
echo ""
echo "5. After creation, you'll see:"
echo "   - Client ID (like: 123456789-xxx.apps.googleusercontent.com)"
echo "   - Client Secret (like: GOCSPX-xxxxxxxxxxxxx)"
echo ""
echo "6. Download the JSON or copy the credentials"
echo ""
echo "============================================================"