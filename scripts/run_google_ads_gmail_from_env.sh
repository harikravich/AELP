#!/usr/bin/env bash
set -euo pipefail

# Gmail Google Ads Setup Script - Zero-prompt using environment variables
# This creates a paused campaign in Gmail Ads account without touching Aura credentials

echo "=========================================="
echo "Gmail Google Ads Setup (Separate from Aura)"
echo "=========================================="

# Check for required Gmail environment variables
if [[ -z "${GMAIL_REFRESH_TOKEN:-}" ]]; then
    echo "ERROR: GMAIL_REFRESH_TOKEN not found in environment"
    echo "Please add Gmail OAuth credentials to .env first:"
    echo "  GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN, GMAIL_CUSTOMER_ID"
    exit 1
fi

# Set up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/AELP2/config"
GMAIL_ENV_FILE="$CONFIG_DIR/.google_ads_credentials.gmail.env"

# Create config directory if needed
mkdir -p "$CONFIG_DIR"

# Create Gmail-specific credentials file
cat > "$GMAIL_ENV_FILE" << EOFINNER
# Gmail Google Ads Credentials (Separate from Aura)
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# Customer ID: 9704174968

# Developer token (shared with Aura)
export GOOGLE_ADS_DEVELOPER_TOKEN="${GOOGLE_ADS_DEVELOPER_TOKEN}"

# Gmail OAuth credentials
export GOOGLE_ADS_CLIENT_ID="${GMAIL_CLIENT_ID:-${GOOGLE_ADS_CLIENT_ID}}"
export GOOGLE_ADS_CLIENT_SECRET="${GMAIL_CLIENT_SECRET:-${GOOGLE_ADS_CLIENT_SECRET}}"
export GOOGLE_ADS_REFRESH_TOKEN="${GMAIL_REFRESH_TOKEN}"

# Gmail customer ID (no login customer ID for standalone account)
export GOOGLE_ADS_CUSTOMER_ID="${GMAIL_CUSTOMER_ID:-9704174968}"
unset GOOGLE_ADS_LOGIN_CUSTOMER_ID

# API settings
export GOOGLE_ADS_USE_PROTO_PLUS="true"
EOFINNER

echo "✓ Created Gmail credentials file: $GMAIL_ENV_FILE"
echo ""
echo "Testing Gmail Google Ads connection..."

# Test connection with a simple list campaigns script
python3 - << 'PYTHON'
import os
import sys
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

# Load Gmail credentials from environment
client_config = {
    "developer_token": os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"],
    "client_id": os.environ["GOOGLE_ADS_CLIENT_ID"],
    "client_secret": os.environ["GOOGLE_ADS_CLIENT_SECRET"],
    "refresh_token": os.environ["GOOGLE_ADS_REFRESH_TOKEN"],
    "use_proto_plus": True
}

customer_id = os.environ.get("GOOGLE_ADS_CUSTOMER_ID", "9704174968")

print(f"Connecting to Gmail Ads account: {customer_id}")

try:
    client = GoogleAdsClient.load_from_dict(client_config)
    ga_service = client.get_service("GoogleAdsService")
    
    # Test query
    query = """
        SELECT campaign.id, campaign.name, campaign.status
        FROM campaign
        ORDER BY campaign.id
        LIMIT 5
    """
    
    response = ga_service.search_stream(customer_id=customer_id, query=query)
    
    print("✓ Successfully connected to Gmail Google Ads account!")
    
    campaign_count = 0
    for batch in response:
        for row in batch.results:
            campaign_count += 1
            print(f"  Campaign: {row.campaign.name} ({row.campaign.status.name})")
    
    if campaign_count == 0:
        print("  No campaigns found (this is expected for a new account)")
        
except GoogleAdsException as ex:
    print(f"ERROR: Request failed with status {ex.error.code().name}")
    for error in ex.failure.errors:
        print(f"  {error.message}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print("\n✓ Gmail Google Ads connection successful!")
PYTHON

echo ""
echo "=========================================="
echo "Gmail Google Ads Setup Complete!"
echo "=========================================="
echo ""
echo "Gmail credentials saved to: $GMAIL_ENV_FILE"
echo ""
echo "To use Gmail credentials in future scripts:"
echo "  set -a; source $GMAIL_ENV_FILE; set +a"
echo "  unset GOOGLE_ADS_LOGIN_CUSTOMER_ID"
