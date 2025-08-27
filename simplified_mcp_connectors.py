#!/usr/bin/env python3
"""
GAELP Simplified MCP Connectors - Direct Ad Platform Integration
==============================================================

This module provides a simplified interface to Meta Ads, Google Ads, and TikTok Ads 
using direct API calls and basic HTTP clients. This is a fallback when MCP servers
are not available or not properly configured.

Author: GAELP Team
Version: 1.0.0
Date: 2025-08-21
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
import subprocess

# HTTP clients
import httpx
import requests

# Environment and configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlatformType(Enum):
    """Supported advertising platforms"""
    META = "meta"
    GOOGLE = "google" 
    TIKTOK = "tiktok"


@dataclass
class PlatformCredentials:
    """Credentials for advertising platforms"""
    platform: PlatformType
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    developer_token: Optional[str] = None
    customer_id: Optional[str] = None
    account_id: Optional[str] = None
    app_id: Optional[str] = None
    business_id: Optional[str] = None


@dataclass
class AdCampaign:
    """Unified ad campaign representation"""
    platform: PlatformType
    campaign_id: str
    name: str
    status: str
    budget: float
    start_date: datetime
    end_date: Optional[datetime] = None
    objective: Optional[str] = None


@dataclass
class AdPerformance:
    """Unified ad performance metrics"""
    platform: PlatformType
    campaign_id: str
    impressions: int
    clicks: int
    conversions: int
    spend: float
    cpm: float
    cpc: float
    ctr: float
    conversion_rate: float


class SimplifiedConnector:
    """Base class for simplified platform connectors"""
    
    def __init__(self, platform: PlatformType, credentials: PlatformCredentials):
        self.platform = platform
        self.credentials = credentials
        self.session = httpx.AsyncClient(timeout=30.0)
        
    async def connect(self) -> bool:
        """Test connection to the platform"""
        try:
            return await self._test_connection()
        except Exception as e:
            logger.error(f"Failed to connect to {self.platform.value}: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test connection - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def get_campaigns(self) -> List[AdCampaign]:
        """Get campaigns - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from platform"""
        await self.session.aclose()


class MetaAdsSimplifiedConnector(SimplifiedConnector):
    """Simplified Meta Ads connector using direct API calls"""
    
    def __init__(self, credentials: PlatformCredentials):
        super().__init__(PlatformType.META, credentials)
        self.base_url = "https://graph.facebook.com/v18.0"
        
    async def _test_connection(self) -> bool:
        """Test Meta Ads API connection"""
        if not self.credentials.access_token:
            logger.error("No Meta access token provided")
            return False
            
        try:
            url = f"{self.base_url}/me"
            params = {"access_token": self.credentials.access_token}
            
            response = await self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Connected to Meta Ads as: {data.get('name', 'Unknown')}")
                return True
            else:
                logger.error(f"Meta API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Meta connection test failed: {e}")
            return False
    
    async def get_campaigns(self) -> List[AdCampaign]:
        """Get Meta Ads campaigns"""
        campaigns = []
        
        if not self.credentials.access_token or not self.credentials.account_id:
            logger.error("Missing Meta credentials")
            return campaigns
            
        try:
            url = f"{self.base_url}/{self.credentials.account_id}/campaigns"
            params = {
                "access_token": self.credentials.access_token,
                "fields": "id,name,status,budget_remaining,start_time,stop_time,objective,created_time"
            }
            
            response = await self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                for campaign_data in data.get("data", []):
                    campaigns.append(AdCampaign(
                        platform=PlatformType.META,
                        campaign_id=campaign_data.get("id"),
                        name=campaign_data.get("name"),
                        status=campaign_data.get("status"),
                        budget=float(campaign_data.get("budget_remaining", 0)),
                        start_date=datetime.fromisoformat(campaign_data.get("created_time", datetime.now().isoformat()).replace("Z", "+00:00")),
                        objective=campaign_data.get("objective")
                    ))
                    
                logger.info(f"Retrieved {len(campaigns)} Meta campaigns")
            else:
                logger.error(f"Failed to get Meta campaigns: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error getting Meta campaigns: {e}")
            
        return campaigns


class GoogleAdsSimplifiedConnector(SimplifiedConnector):
    """Simplified Google Ads connector"""
    
    def __init__(self, credentials: PlatformCredentials):
        super().__init__(PlatformType.GOOGLE, credentials)
        self.google_ads_client = None
        
    async def _test_connection(self) -> bool:
        """Test Google Ads API connection"""
        try:
            # Try to import Google Ads library
            from google.ads.googleads.client import GoogleAdsClient
            
            if not all([
                self.credentials.developer_token,
                self.credentials.client_id,
                self.credentials.client_secret,
                self.credentials.refresh_token,
                self.credentials.customer_id
            ]):
                logger.error("Missing Google Ads credentials")
                return False
            
            config = {
                "developer_token": self.credentials.developer_token,
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret,
                "refresh_token": self.credentials.refresh_token,
                "customer_id": self.credentials.customer_id,
            }
            
            self.google_ads_client = GoogleAdsClient.load_from_dict(config)
            
            # Test the connection by getting customer info
            customer_service = self.google_ads_client.get_service("CustomerService")
            customer = customer_service.get_customer(
                customer_id=self.credentials.customer_id
            )
            
            logger.info(f"Connected to Google Ads: {customer.descriptive_name}")
            return True
            
        except ImportError:
            logger.error("Google Ads library not installed. Run: pip install google-ads")
            return False
        except Exception as e:
            logger.error(f"Google Ads connection test failed: {e}")
            return False
    
    async def get_campaigns(self) -> List[AdCampaign]:
        """Get Google Ads campaigns"""
        campaigns = []
        
        if not self.google_ads_client:
            logger.error("Google Ads client not initialized")
            return campaigns
            
        try:
            ga_service = self.google_ads_client.get_service("GoogleAdsService")
            
            query = """
                SELECT 
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign.start_date,
                    campaign.end_date,
                    campaign.advertising_channel_type
                FROM campaign
                WHERE campaign.status != 'REMOVED'
                LIMIT 100
            """
            
            search_request = self.google_ads_client.get_type("SearchGoogleAdsRequest")
            search_request.customer_id = self.credentials.customer_id
            search_request.query = query
            
            response = ga_service.search(request=search_request)
            
            for row in response:
                campaigns.append(AdCampaign(
                    platform=PlatformType.GOOGLE,
                    campaign_id=str(row.campaign.id),
                    name=row.campaign.name,
                    status=row.campaign.status.name,
                    budget=0.0,  # Budget requires separate query
                    start_date=datetime.strptime(row.campaign.start_date, "%Y-%m-%d"),
                    end_date=datetime.strptime(row.campaign.end_date, "%Y-%m-%d") if row.campaign.end_date else None,
                ))
            
            logger.info(f"Retrieved {len(campaigns)} Google Ads campaigns")
            
        except Exception as e:
            logger.error(f"Error getting Google Ads campaigns: {e}")
            
        return campaigns


class TikTokAdsSimplifiedConnector(SimplifiedConnector):
    """Simplified TikTok Ads connector using direct API calls"""
    
    def __init__(self, credentials: PlatformCredentials):
        super().__init__(PlatformType.TIKTOK, credentials)
        self.base_url = "https://business-api.tiktok.com/open_api/v1.3"
        
    async def _test_connection(self) -> bool:
        """Test TikTok Ads API connection"""
        if not self.credentials.access_token:
            logger.error("No TikTok access token provided")
            return False
            
        try:
            url = f"{self.base_url}/advertiser/info/"
            headers = {
                "Access-Token": self.credentials.access_token,
                "Content-Type": "application/json"
            }
            params = {
                "advertiser_ids": json.dumps([self.credentials.account_id]) if self.credentials.account_id else "[]"
            }
            
            response = await self.session.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 0:
                    logger.info("Connected to TikTok Ads successfully")
                    return True
                else:
                    logger.error(f"TikTok API error: {data.get('message')}")
                    return False
            else:
                logger.error(f"TikTok API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"TikTok connection test failed: {e}")
            return False
    
    async def get_campaigns(self) -> List[AdCampaign]:
        """Get TikTok Ads campaigns"""
        campaigns = []
        
        if not self.credentials.access_token or not self.credentials.account_id:
            logger.error("Missing TikTok credentials")
            return campaigns
            
        try:
            url = f"{self.base_url}/campaign/get/"
            headers = {
                "Access-Token": self.credentials.access_token,
                "Content-Type": "application/json"
            }
            params = {
                "advertiser_id": self.credentials.account_id,
                "fields": json.dumps([
                    "campaign_id", "campaign_name", "status", 
                    "budget", "create_time", "objective_type"
                ])
            }
            
            response = await self.session.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("code") == 0:
                    for campaign_data in data.get("data", {}).get("list", []):
                        campaigns.append(AdCampaign(
                            platform=PlatformType.TIKTOK,
                            campaign_id=campaign_data.get("campaign_id"),
                            name=campaign_data.get("campaign_name"),
                            status=campaign_data.get("status"),
                            budget=float(campaign_data.get("budget", 0)),
                            start_date=datetime.fromtimestamp(int(campaign_data.get("create_time", 0))),
                            objective=campaign_data.get("objective_type")
                        ))
                        
                    logger.info(f"Retrieved {len(campaigns)} TikTok campaigns")
                else:
                    logger.error(f"TikTok API error: {data.get('message')}")
            else:
                logger.error(f"Failed to get TikTok campaigns: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error getting TikTok campaigns: {e}")
            
        return campaigns


class SimplifiedAdPlatformManager:
    """Simplified manager for all ad platform connectors"""
    
    def __init__(self):
        self.connectors: Dict[PlatformType, SimplifiedConnector] = {}
        self.credentials: Dict[PlatformType, PlatformCredentials] = {}
        self._load_credentials()
    
    def _load_credentials(self):
        """Load credentials from environment variables"""
        # Meta Ads credentials
        meta_creds = PlatformCredentials(
            platform=PlatformType.META,
            access_token=os.getenv("META_ACCESS_TOKEN"),
            app_id=os.getenv("META_APP_ID"),
            secret_key=os.getenv("META_APP_SECRET"),
            business_id=os.getenv("META_BUSINESS_ID"),
            account_id=os.getenv("META_ACCOUNT_ID")
        )
        self.credentials[PlatformType.META] = meta_creds
        
        # Google Ads credentials
        google_creds = PlatformCredentials(
            platform=PlatformType.GOOGLE,
            developer_token=os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"),
            client_id=os.getenv("GOOGLE_ADS_CLIENT_ID"),
            client_secret=os.getenv("GOOGLE_ADS_CLIENT_SECRET"),
            refresh_token=os.getenv("GOOGLE_ADS_REFRESH_TOKEN"),
            customer_id=os.getenv("GOOGLE_ADS_CUSTOMER_ID")
        )
        self.credentials[PlatformType.GOOGLE] = google_creds
        
        # TikTok Ads credentials
        tiktok_creds = PlatformCredentials(
            platform=PlatformType.TIKTOK,
            access_token=os.getenv("TIKTOK_ACCESS_TOKEN"),
            app_id=os.getenv("TIKTOK_APP_ID"),
            secret_key=os.getenv("TIKTOK_SECRET"),
            account_id=os.getenv("TIKTOK_ADVERTISER_ID")
        )
        self.credentials[PlatformType.TIKTOK] = tiktok_creds
    
    async def initialize_platform(self, platform: PlatformType) -> bool:
        """Initialize a specific platform connector"""
        try:
            if platform in self.connectors:
                return True
            
            creds = self.credentials.get(platform)
            if not creds:
                logger.error(f"No credentials found for {platform.value}")
                return False
            
            if platform == PlatformType.META:
                connector = MetaAdsSimplifiedConnector(creds)
            elif platform == PlatformType.GOOGLE:
                connector = GoogleAdsSimplifiedConnector(creds)
            elif platform == PlatformType.TIKTOK:
                connector = TikTokAdsSimplifiedConnector(creds)
            else:
                logger.error(f"Unsupported platform: {platform.value}")
                return False
            
            success = await connector.connect()
            if success:
                self.connectors[platform] = connector
                logger.info(f"✅ Successfully initialized {platform.value} connector")
                return True
            else:
                logger.error(f"❌ Failed to connect {platform.value} connector")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize {platform.value}: {e}")
            return False
    
    async def initialize_all_platforms(self) -> Dict[PlatformType, bool]:
        """Initialize all available platform connectors"""
        results = {}
        
        for platform in PlatformType:
            results[platform] = await self.initialize_platform(platform)
        
        return results
    
    async def get_all_campaigns(self) -> Dict[PlatformType, List[AdCampaign]]:
        """Get campaigns from all connected platforms"""
        all_campaigns = {}
        
        for platform, connector in self.connectors.items():
            try:
                campaigns = await connector.get_campaigns()
                all_campaigns[platform] = campaigns
                logger.info(f"Retrieved {len(campaigns)} campaigns from {platform.value}")
            except Exception as e:
                logger.error(f"Failed to get campaigns from {platform.value}: {e}")
                all_campaigns[platform] = []
        
        return all_campaigns
    
    def _has_valid_credentials(self, platform: PlatformType) -> bool:
        """Check if platform has valid credentials"""
        creds = self.credentials.get(platform)
        if not creds:
            return False
        
        if platform == PlatformType.META:
            return bool(creds.access_token and creds.account_id)
        elif platform == PlatformType.GOOGLE:
            return bool(creds.developer_token and creds.client_id and 
                       creds.client_secret and creds.refresh_token and creds.customer_id)
        elif platform == PlatformType.TIKTOK:
            return bool(creds.access_token and creds.account_id)
        
        return False
    
    async def get_connection_status(self) -> Dict[PlatformType, Dict[str, Any]]:
        """Get connection status for all platforms"""
        status = {}
        
        for platform in PlatformType:
            is_connected = platform in self.connectors
            has_credentials = self._has_valid_credentials(platform)
            
            creds = self.credentials.get(platform)
            credential_details = {}
            
            if platform == PlatformType.META:
                credential_details = {
                    "access_token": "✅" if creds and creds.access_token and not creds.access_token.startswith("your_") else "❌",
                    "account_id": "✅" if creds and creds.account_id and not creds.account_id.startswith("your_") else "❌",
                    "app_id": "✅" if creds and creds.app_id and not creds.app_id.startswith("your_") else "❌",
                }
            elif platform == PlatformType.GOOGLE:
                credential_details = {
                    "developer_token": "✅" if creds and creds.developer_token and not creds.developer_token.startswith("your_") else "❌",
                    "client_id": "✅" if creds and creds.client_id and not creds.client_id.startswith("your_") else "❌",
                    "customer_id": "✅" if creds and creds.customer_id and not creds.customer_id.startswith("your_") else "❌",
                }
            elif platform == PlatformType.TIKTOK:
                credential_details = {
                    "access_token": "✅" if creds and creds.access_token and not creds.access_token.startswith("your_") else "❌",
                    "account_id": "✅" if creds and creds.account_id and not creds.account_id.startswith("your_") else "❌",
                }
            
            status[platform] = {
                "connected": is_connected,
                "has_credentials": has_credentials,
                "credential_details": credential_details,
                "status": "connected" if is_connected else ("ready" if has_credentials else "missing_credentials")
            }
        
        return status
    
    async def disconnect_all(self):
        """Disconnect from all platforms"""
        for platform, connector in self.connectors.items():
            try:
                await connector.disconnect()
                logger.info(f"Disconnected from {platform.value}")
            except Exception as e:
                logger.error(f"Error disconnecting from {platform.value}: {e}")
        
        self.connectors.clear()


async def test_connections() -> Dict[str, Any]:
    """Test all platform connections and return status report"""
    manager = SimplifiedAdPlatformManager()
    
    logger.info("Testing MCP/Ad Platform connections...")
    init_results = await manager.initialize_all_platforms()
    
    status_report = {
        "timestamp": datetime.now().isoformat(),
        "mcp_packages_installed": await check_mcp_packages(),
        "initialization_results": {p.value: success for p, success in init_results.items()},
        "connection_status": {},
        "sample_data": {}
    }
    
    # Get detailed connection status
    connection_status = await manager.get_connection_status()
    status_report["connection_status"] = {p.value: status for p, status in connection_status.items()}
    
    # Try to fetch sample data from each connected platform
    for platform, connector in manager.connectors.items():
        try:
            campaigns = await connector.get_campaigns()
            status_report["sample_data"][platform.value] = {
                "campaigns_count": len(campaigns),
                "sample_campaigns": [
                    {
                        "id": c.campaign_id,
                        "name": c.name,
                        "status": c.status,
                        "budget": c.budget
                    } for c in campaigns[:3]  # First 3 campaigns
                ]
            }
        except Exception as e:
            status_report["sample_data"][platform.value] = {"error": str(e)}
    
    await manager.disconnect_all()
    return status_report


async def check_mcp_packages() -> Dict[str, Any]:
    """Check which MCP packages are installed and available"""
    packages_status = {}
    
    # Check if uvx is available
    try:
        result = subprocess.run(["which", "uvx"], capture_output=True, text=True)
        packages_status["uvx"] = result.returncode == 0
    except:
        packages_status["uvx"] = False
    
    # Check Python MCP packages
    try:
        import mcp
        packages_status["mcp"] = True
    except ImportError:
        packages_status["mcp"] = False
    
    # Check specific MCP packages
    mcp_packages = ["meta-ads-mcp", "tiktok-ads-mcp", "google-ads"]
    for package in mcp_packages:
        try:
            result = subprocess.run(["pip", "show", package], capture_output=True, text=True)
            packages_status[package] = result.returncode == 0
        except:
            packages_status[package] = False
    
    return packages_status


async def quick_setup() -> Tuple[SimplifiedAdPlatformManager, Dict[PlatformType, bool]]:
    """Quick setup and initialization of all platforms"""
    manager = SimplifiedAdPlatformManager()
    results = await manager.initialize_all_platforms()
    
    # Log results
    connected_platforms = [p.value for p, success in results.items() if success]
    failed_platforms = [p.value for p, success in results.items() if not success]
    
    if connected_platforms:
        logger.info(f"✅ Connected platforms: {', '.join(connected_platforms)}")
    if failed_platforms:
        logger.warning(f"❌ Failed to connect: {', '.join(failed_platforms)}")
    
    return manager, results


# CLI interface
async def main():
    """CLI interface for testing simplified MCP connectors"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GAELP Simplified MCP Connectors")
    parser.add_argument("--test", action="store_true", help="Test all connections")
    parser.add_argument("--platform", choices=["meta", "google", "tiktok"], help="Test specific platform")
    parser.add_argument("--campaigns", action="store_true", help="List campaigns from all platforms")
    parser.add_argument("--status", action="store_true", help="Show connection status")
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing simplified MCP connections...")
        report = await test_connections()
        print(json.dumps(report, indent=2, default=str))
        
    elif args.platform:
        platform = PlatformType(args.platform)
        manager = SimplifiedAdPlatformManager()
        success = await manager.initialize_platform(platform)
        
        if success:
            print(f"✅ Successfully connected to {platform.value}")
            campaigns = await manager.connectors[platform].get_campaigns()
            print(f"Found {len(campaigns)} campaigns")
            for campaign in campaigns[:5]:  # Show first 5
                print(f"  - {campaign.name} ({campaign.status}) - ${campaign.budget}")
        else:
            print(f"❌ Failed to connect to {platform.value}")
        
        await manager.disconnect_all()
        
    elif args.campaigns:
        manager, _ = await quick_setup()
        all_campaigns = await manager.get_all_campaigns()
        
        for platform, campaigns in all_campaigns.items():
            print(f"\n{platform.value.upper()} Campaigns ({len(campaigns)}):")
            for campaign in campaigns:
                print(f"  - {campaign.name} ({campaign.status}) - ${campaign.budget}")
        
        await manager.disconnect_all()
        
    elif args.status:
        manager = SimplifiedAdPlatformManager()
        status = await manager.get_connection_status()
        
        print("\n🔗 Ad Platform Connection Status")
        print("=" * 50)
        
        for platform, platform_status in status.items():
            emoji = "✅" if platform_status["connected"] else ("🟡" if platform_status["has_credentials"] else "❌")
            print(f"\n{emoji} {platform.value.upper()}")
            print(f"   Status: {platform_status['status']}")
            print(f"   Connected: {platform_status['connected']}")
            print(f"   Has Credentials: {platform_status['has_credentials']}")
            
            print("   Credential Details:")
            for cred, status_emoji in platform_status["credential_details"].items():
                print(f"     {cred}: {status_emoji}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())