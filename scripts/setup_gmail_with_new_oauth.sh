#!/usr/bin/env bash
set -euo pipefail

# Gmail Google Ads Setup with NEW OAuth Client

echo "=========================================="
echo "Gmail Google Ads Setup (New OAuth)"
echo "=========================================="

# Check for Gmail-specific credentials
if [[ -z "${GMAIL_CLIENT_ID:-}" ]] || [[ -z "${GMAIL_REFRESH_TOKEN:-}" ]]; then
    echo "ERROR: Gmail OAuth credentials not found"
    echo "Please add to .env:"
    echo "  GMAIL_CLIENT_ID=your_new_client_id"
    echo "  GMAIL_CLIENT_SECRET=your_new_secret"
    echo "  GMAIL_REFRESH_TOKEN=your_new_refresh_token"
    echo "  GMAIL_CUSTOMER_ID=9704174968"
    exit 1
fi

# Load environment
source /home/hariravichandran/AELP/.env

# Create Gmail config directory
CONFIG_DIR="/home/hariravichandran/AELP/AELP2/config"
mkdir -p "$CONFIG_DIR"

# Create Gmail-specific env file
cat > "$CONFIG_DIR/.google_ads_credentials.gmail.env" << EOF
# Gmail Google Ads Credentials (Separate OAuth)
# Generated: $(date)

export GOOGLE_ADS_DEVELOPER_TOKEN="${GOOGLE_ADS_DEVELOPER_TOKEN}"
export GOOGLE_ADS_CLIENT_ID="${GMAIL_CLIENT_ID}"
export GOOGLE_ADS_CLIENT_SECRET="${GMAIL_CLIENT_SECRET}"
export GOOGLE_ADS_REFRESH_TOKEN="${GMAIL_REFRESH_TOKEN}"
export GOOGLE_ADS_CUSTOMER_ID="${GMAIL_CUSTOMER_ID:-9704174968}"
unset GOOGLE_ADS_LOGIN_CUSTOMER_ID
export GOOGLE_ADS_USE_PROTO_PLUS="true"
EOF

echo "✓ Created Gmail credentials file"

# Test connection
python3 - << 'PYTHON'
import os
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

# Use Gmail credentials
config = {
    "developer_token": os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"],
    "client_id": os.environ["GMAIL_CLIENT_ID"],
    "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
    "refresh_token": os.environ["GMAIL_REFRESH_TOKEN"],
    "use_proto_plus": True
}

customer_id = "9704174968"

try:
    print(f"Testing connection to Gmail Ads account: {customer_id}")
    client = GoogleAdsClient.load_from_dict(config)
    ga_service = client.get_service("GoogleAdsService")
    
    # Simple test query
    query = "SELECT campaign.id FROM campaign LIMIT 1"
    response = ga_service.search_stream(customer_id=customer_id, query=query)
    
    # Just iterate to test connection
    for batch in response:
        pass
    
    print("✓ Successfully connected to Gmail Google Ads!")
    
except GoogleAdsException as ex:
    print(f"ERROR: {ex.error.code().name}")
    for error in ex.failure.errors:
        print(f"  {error.message}")
except Exception as e:
    print(f"ERROR: {e}")
PYTHON

echo ""
echo "Gmail setup complete!"
echo "Credentials saved to: $CONFIG_DIR/.google_ads_credentials.gmail.env"