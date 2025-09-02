#!/usr/bin/env python3
"""
Google Ads Production Setup for GAELP
Handles authentication, credential management, and initial setup
"""

import os
import json
import webbrowser
import subprocess
from typing import Dict, Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
import logging

logger = logging.getLogger(__name__)

class GoogleAdsAuthenticator:
    """Handles Google Ads API authentication and credential management"""
    
    SCOPES = ['https://www.googleapis.com/auth/adwords']
    
    def __init__(self, credentials_file: str = 'google_ads_credentials.json'):
        self.credentials_file = credentials_file
        self.client_config = None
        
    def setup_oauth_credentials(self) -> Dict[str, str]:
        """
        Guide user through OAuth setup process
        Returns dictionary of credentials to set in environment
        """
        print("=" * 80)
        print("GOOGLE ADS API AUTHENTICATION SETUP")
        print("=" * 80)
        
        print("\n🔐 Setting up Google Ads API authentication...")
        print("\nYou need to complete these steps:")
        print("1. Create a Google Cloud Project")
        print("2. Enable the Google Ads API")
        print("3. Create OAuth 2.0 credentials")
        print("4. Apply for Google Ads API access")
        print("5. Get a Developer Token")
        
        # Check if we have client configuration
        if not self._load_client_config():
            print("\n❌ Client configuration not found.")
            self._guide_client_setup()
            return {}
        
        # Run OAuth flow
        credentials = self._run_oauth_flow()
        
        if credentials:
            # Get developer token and customer ID
            developer_token = self._get_developer_token()
            customer_id = self._get_customer_id()
            
            # Return environment variables
            return {
                'GOOGLE_ADS_CLIENT_ID': self.client_config['web']['client_id'],
                'GOOGLE_ADS_CLIENT_SECRET': self.client_config['web']['client_secret'],
                'GOOGLE_ADS_REFRESH_TOKEN': credentials.refresh_token,
                'GOOGLE_ADS_DEVELOPER_TOKEN': developer_token,
                'GOOGLE_ADS_CUSTOMER_ID': customer_id
            }
        
        return {}
    
    def _load_client_config(self) -> bool:
        """Load OAuth client configuration"""
        try:
            with open('google_ads_client_secret.json', 'r') as f:
                self.client_config = json.load(f)
            return True
        except FileNotFoundError:
            return False
    
    def _guide_client_setup(self):
        """Guide user through client setup"""
        print("\n" + "=" * 60)
        print("STEP 1: CREATE GOOGLE CLOUD PROJECT & OAUTH CREDENTIALS")
        print("=" * 60)
        
        print("\n1. Go to Google Cloud Console:")
        print("   https://console.cloud.google.com/")
        
        print("\n2. Create a new project or select existing project")
        
        print("\n3. Enable Google Ads API:")
        print("   https://console.cloud.google.com/apis/library/googleads.googleapis.com")
        
        print("\n4. Create OAuth 2.0 credentials:")
        print("   a) Go to: https://console.cloud.google.com/apis/credentials")
        print("   b) Click 'Create Credentials' > 'OAuth client ID'")
        print("   c) Choose 'Desktop application'")
        print("   d) Download the JSON file")
        print("   e) Save it as 'google_ads_client_secret.json' in this directory")
        
        print("\n5. Apply for Google Ads API access:")
        print("   https://ads.google.com/aw/apicenter")
        
        input("\nPress Enter after completing these steps...")
    
    def _run_oauth_flow(self) -> Optional[Credentials]:
        """Run OAuth flow to get refresh token"""
        try:
            flow = Flow.from_client_config(
                self.client_config,
                scopes=self.SCOPES,
                redirect_uri='http://localhost:8080'
            )
            
            # Get authorization URL
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'  # Force consent to get refresh token
            )
            
            print(f"\n🌐 Opening browser for authentication...")
            print(f"If browser doesn't open, visit: {auth_url}")
            webbrowser.open(auth_url)
            
            # Get authorization code from user
            auth_code = input("\nPaste the authorization code here: ").strip()
            
            # Exchange code for credentials
            flow.fetch_token(code=auth_code)
            credentials = flow.credentials
            
            print("✅ Authentication successful!")
            return credentials
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return None
    
    def _get_developer_token(self) -> str:
        """Get developer token from user"""
        print("\n" + "=" * 60)
        print("DEVELOPER TOKEN REQUIRED")
        print("=" * 60)
        
        print("\n1. Go to Google Ads API Center:")
        print("   https://ads.google.com/aw/apicenter")
        
        print("\n2. Apply for API access (if not already done)")
        print("   - Basic access is usually sufficient for testing")
        print("   - Standard access requires approval process")
        
        print("\n3. Once approved, get your Developer Token from the API Center")
        
        developer_token = input("\nEnter your Developer Token: ").strip()
        
        if not developer_token:
            print("❌ Developer token is required")
            return ""
        
        return developer_token
    
    def _get_customer_id(self) -> str:
        """Get customer ID from user"""
        print("\n" + "=" * 60)
        print("GOOGLE ADS CUSTOMER ID REQUIRED")
        print("=" * 60)
        
        print("\n1. Go to your Google Ads account:")
        print("   https://ads.google.com/")
        
        print("\n2. Find your Customer ID (10 digits, no hyphens)")
        print("   - Usually shown in the top right corner")
        print("   - Format: 1234567890 (remove any hyphens)")
        
        customer_id = input("\nEnter your Customer ID (10 digits, no hyphens): ").strip()
        
        # Validate customer ID format
        if not customer_id.isdigit() or len(customer_id) != 10:
            print("❌ Customer ID should be 10 digits with no hyphens")
            return ""
        
        return customer_id
    
    def save_credentials_to_env(self, credentials: Dict[str, str]):
        """Save credentials to .env file"""
        env_file_path = '/home/hariravichandran/AELP/.env'
        
        # Read existing .env file
        env_lines = []
        google_ads_keys = set(credentials.keys())
        
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key = line.split('=')[0]
                        if key not in google_ads_keys:
                            env_lines.append(line)
                    else:
                        env_lines.append(line)
        
        # Add Google Ads credentials
        env_lines.append("")
        env_lines.append("# Google Ads API Credentials (Production)")
        for key, value in credentials.items():
            env_lines.append(f"{key}={value}")
        
        # Write updated .env file
        with open(env_file_path, 'w') as f:
            f.write('\n'.join(env_lines))
        
        print(f"✅ Credentials saved to {env_file_path}")
    
    def verify_credentials(self) -> bool:
        """Verify that all required credentials are set"""
        required_vars = [
            'GOOGLE_ADS_DEVELOPER_TOKEN',
            'GOOGLE_ADS_CLIENT_ID',
            'GOOGLE_ADS_CLIENT_SECRET',
            'GOOGLE_ADS_REFRESH_TOKEN',
            'GOOGLE_ADS_CUSTOMER_ID'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            return False
        
        print("✅ All required credentials are set")
        return True

def install_google_ads_library():
    """Install Google Ads Python library"""
    print("\n📦 Installing Google Ads Python library...")
    
    try:
        subprocess.check_call([
            'pip', 'install', 
            'google-ads==24.1.0',
            'google-auth-oauthlib==1.0.0',
            'google-auth==2.23.4'
        ])
        print("✅ Google Ads library installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Google Ads library: {e}")
        return False

def create_google_ads_yaml_config():
    """Create google-ads.yaml configuration file"""
    config_content = f"""# Google Ads API Configuration
# This file is used by the Google Ads Python library

developer_token: {os.environ.get('GOOGLE_ADS_DEVELOPER_TOKEN', 'YOUR_DEVELOPER_TOKEN')}
client_id: {os.environ.get('GOOGLE_ADS_CLIENT_ID', 'YOUR_CLIENT_ID')}
client_secret: {os.environ.get('GOOGLE_ADS_CLIENT_SECRET', 'YOUR_CLIENT_SECRET')}
refresh_token: {os.environ.get('GOOGLE_ADS_REFRESH_TOKEN', 'YOUR_REFRESH_TOKEN')}

# Optional: Uncomment to specify a different login customer ID
# login_customer_id: INSERT_LOGIN_CUSTOMER_ID_HERE

# Optional: Uncomment to specify logging settings
# logging:
#   version: 1
#   disable_existing_loggers: False
#   formatters:
#     default_fmt:
#       format: '[%(asctime)s - %(name)s - %(levelname)s] %(message)s'
#       datefmt: '%Y-%m-%d %H:%M:%S'
#   handlers:
#     default_handler:
#       class: logging.StreamHandler
#       formatter: default_fmt
#   loggers:
#     "":
#       handlers: [default_handler]
#       level: INFO
"""
    
    config_path = '/home/hariravichandran/AELP/google-ads.yaml'
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Google Ads configuration saved to {config_path}")

def test_google_ads_connection():
    """Test Google Ads API connection"""
    print("\n🧪 Testing Google Ads API connection...")
    
    try:
        from google.ads.googleads.client import GoogleAdsClient
        from google.ads.googleads.errors import GoogleAdsException
        
        # Initialize client
        client = GoogleAdsClient.load_from_env()
        customer_id = os.environ.get('GOOGLE_ADS_CUSTOMER_ID')
        
        # Try to get account info
        customer_service = client.get_service("CustomerService")
        customer = customer_service.get_customer(
            resource_name=f"customers/{customer_id}"
        )
        
        print(f"✅ Successfully connected to Google Ads account:")
        print(f"   Customer ID: {customer_id}")
        print(f"   Account Name: {customer.descriptive_name}")
        print(f"   Currency: {customer.currency_code}")
        print(f"   Time Zone: {customer.time_zone}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Google Ads API: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 80)
    print("GOOGLE ADS PRODUCTION SETUP FOR GAELP")
    print("=" * 80)
    
    print("\n🚀 Setting up Google Ads integration for production campaigns...")
    
    # Step 1: Install dependencies
    if not install_google_ads_library():
        print("❌ Failed to install required libraries")
        return
    
    # Step 2: Setup authentication
    authenticator = GoogleAdsAuthenticator()
    
    # Check if credentials already exist
    if authenticator.verify_credentials():
        print("\n✅ Google Ads credentials already configured")
        use_existing = input("Use existing credentials? (y/n): ").strip().lower()
        
        if use_existing == 'n':
            credentials = authenticator.setup_oauth_credentials()
            if credentials:
                authenticator.save_credentials_to_env(credentials)
    else:
        print("\n🔐 Setting up Google Ads authentication...")
        credentials = authenticator.setup_oauth_credentials()
        
        if credentials:
            authenticator.save_credentials_to_env(credentials)
            print("\n✅ Credentials saved successfully!")
            print("\n⚠️  IMPORTANT: Restart your Python session to load new environment variables")
        else:
            print("❌ Failed to setup authentication")
            return
    
    # Step 3: Create configuration files
    create_google_ads_yaml_config()
    
    # Step 4: Test connection
    if test_google_ads_connection():
        print("\n" + "=" * 80)
        print("GOOGLE ADS PRODUCTION SETUP COMPLETE!")
        print("=" * 80)
        print("✅ Google Ads Python library installed")
        print("✅ Authentication configured")
        print("✅ API connection verified")
        print("✅ Ready for production campaign management")
        
        print("\n🚀 Next steps:")
        print("1. Run: python google_ads_production_manager.py")
        print("2. Run: python google_ads_gaelp_integration.py")
        print("3. Integrate with GAELP RL training loop")
        
    else:
        print("\n❌ Setup incomplete - API connection failed")
        print("Please check your credentials and try again")

if __name__ == "__main__":
    main()