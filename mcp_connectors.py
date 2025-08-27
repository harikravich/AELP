#!/usr/bin/env python3
"""
GAELP MCP Connectors - Unified Ad Platform Integration
====================================================

This module provides a unified interface to Meta Ads, Google Ads, and TikTok Ads 
through the Model Context Protocol (MCP). It handles authentication, rate limiting,
error handling, and provides a consistent API for all advertising platforms.

Author: GAELP Team
Version: 1.0.0
Date: 2025-08-21
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import time

# MCP and HTTP clients
import httpx
from mcp import Client, ClientSession, StdioServerParameters
from mcp.types import Resource, Tool, TextContent, EmbeddedResource
import subprocess
import sys

# Environment and configuration
from dotenv import load_dotenv
import yaml

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
    additional_config: Optional[Dict[str, Any]] = None


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
    target_audience: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


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
    roas: Optional[float] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None


class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)


class MCPConnector:
    """Base class for MCP platform connectors"""
    
    def __init__(self, platform: PlatformType, credentials: PlatformCredentials):
        self.platform = platform
        self.credentials = credentials
        self.rate_limiter = RateLimiter()
        self.client_session: Optional[ClientSession] = None
        self.mcp_client: Optional[Client] = None
        
    async def connect(self) -> bool:
        """Connect to the MCP server"""
        try:
            await self._setup_mcp_connection()
            logger.info(f"Successfully connected to {self.platform.value} MCP server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.platform.value} MCP server: {e}")
            return False
    
    async def _setup_mcp_connection(self):
        """Setup MCP connection - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.client_session:
            await self.client_session.close()
        self.client_session = None
        self.mcp_client = None
    
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> Optional[AdCampaign]:
        """Create a new ad campaign"""
        raise NotImplementedError
    
    async def get_campaigns(self) -> List[AdCampaign]:
        """Get all campaigns"""
        raise NotImplementedError
    
    async def get_campaign_performance(self, campaign_id: str, 
                                     start_date: datetime, 
                                     end_date: datetime) -> Optional[AdPerformance]:
        """Get campaign performance metrics"""
        raise NotImplementedError
    
    async def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> bool:
        """Update campaign settings"""
        raise NotImplementedError
    
    async def pause_campaign(self, campaign_id: str) -> bool:
        """Pause a campaign"""
        raise NotImplementedError
    
    async def resume_campaign(self, campaign_id: str) -> bool:
        """Resume a paused campaign"""
        raise NotImplementedError


class MetaAdsConnector(MCPConnector):
    """Meta Ads MCP connector"""
    
    def __init__(self, credentials: PlatformCredentials):
        super().__init__(PlatformType.META, credentials)
        
    async def _setup_mcp_connection(self):
        """Setup Meta Ads MCP connection"""
        try:
            # Use the meta-ads-mcp package
            server_params = StdioServerParameters(
                command="uvx",
                args=["meta-ads-mcp"],
                env={
                    "META_ACCESS_TOKEN": self.credentials.access_token or "",
                    "META_APP_ID": self.credentials.app_id or "",
                    "META_APP_SECRET": self.credentials.secret_key or "",
                    "META_BUSINESS_ID": self.credentials.business_id or "",
                    "META_ACCOUNT_ID": self.credentials.account_id or "",
                }
            )
            
            self.mcp_client = Client(server_params)
            self.client_session = await self.mcp_client.__aenter__()
            
        except Exception as e:
            logger.error(f"Failed to setup Meta Ads MCP connection: {e}")
            raise
    
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> Optional[AdCampaign]:
        """Create Meta Ads campaign"""
        await self.rate_limiter.wait_if_needed()
        
        try:
            if not self.client_session:
                await self.connect()
            
            # Use MCP tools to create campaign
            result = await self.client_session.call_tool(
                "create_campaign",
                arguments=campaign_data
            )
            
            if result and result.content:
                campaign_info = json.loads(result.content[0].text)
                return AdCampaign(
                    platform=PlatformType.META,
                    campaign_id=campaign_info.get("id"),
                    name=campaign_data.get("name"),
                    status=campaign_info.get("status", "ACTIVE"),
                    budget=float(campaign_data.get("budget", 0)),
                    start_date=datetime.now(),
                    objective=campaign_data.get("objective"),
                    created_at=datetime.now()
                )
        except Exception as e:
            logger.error(f"Failed to create Meta campaign: {e}")
            return None
    
    async def get_campaigns(self) -> List[AdCampaign]:
        """Get Meta Ads campaigns"""
        await self.rate_limiter.wait_if_needed()
        
        try:
            if not self.client_session:
                await self.connect()
            
            result = await self.client_session.call_tool("get_campaigns", arguments={})
            
            if result and result.content:
                campaigns_data = json.loads(result.content[0].text)
                campaigns = []
                
                for campaign_info in campaigns_data.get("data", []):
                    campaigns.append(AdCampaign(
                        platform=PlatformType.META,
                        campaign_id=campaign_info.get("id"),
                        name=campaign_info.get("name"),
                        status=campaign_info.get("status"),
                        budget=float(campaign_info.get("budget", {}).get("amount", 0)),
                        start_date=datetime.fromisoformat(campaign_info.get("created_time", datetime.now().isoformat())),
                        objective=campaign_info.get("objective")
                    ))
                
                return campaigns
                
        except Exception as e:
            logger.error(f"Failed to get Meta campaigns: {e}")
            return []
    
    async def get_campaign_performance(self, campaign_id: str, 
                                     start_date: datetime, 
                                     end_date: datetime) -> Optional[AdPerformance]:
        """Get Meta Ads campaign performance"""
        await self.rate_limiter.wait_if_needed()
        
        try:
            if not self.client_session:
                await self.connect()
            
            result = await self.client_session.call_tool(
                "get_campaign_insights",
                arguments={
                    "campaign_id": campaign_id,
                    "date_preset": "custom",
                    "time_range": {
                        "since": start_date.strftime("%Y-%m-%d"),
                        "until": end_date.strftime("%Y-%m-%d")
                    }
                }
            )
            
            if result and result.content:
                insights_data = json.loads(result.content[0].text)
                data = insights_data.get("data", [{}])[0]
                
                impressions = int(data.get("impressions", 0))
                clicks = int(data.get("clicks", 0))
                spend = float(data.get("spend", 0))
                
                return AdPerformance(
                    platform=PlatformType.META,
                    campaign_id=campaign_id,
                    impressions=impressions,
                    clicks=clicks,
                    conversions=int(data.get("conversions", 0)),
                    spend=spend,
                    cpm=float(data.get("cpm", 0)),
                    cpc=float(data.get("cpc", 0)),
                    ctr=float(data.get("ctr", 0)),
                    conversion_rate=float(data.get("conversion_rate", 0)),
                    roas=float(data.get("return_on_ad_spend", 0)),
                    date_range_start=start_date,
                    date_range_end=end_date
                )
                
        except Exception as e:
            logger.error(f"Failed to get Meta campaign performance: {e}")
            return None


class GoogleAdsConnector(MCPConnector):
    """Google Ads MCP connector"""
    
    def __init__(self, credentials: PlatformCredentials):
        super().__init__(PlatformType.GOOGLE, credentials)
        
    async def _setup_mcp_connection(self):
        """Setup Google Ads MCP connection"""
        try:
            # Use local implementation or available Google Ads MCP
            server_params = StdioServerParameters(
                command="python",
                args=["-m", "mcp_connectors.google_ads_server"],
                env={
                    "GOOGLE_ADS_DEVELOPER_TOKEN": self.credentials.developer_token or "",
                    "GOOGLE_ADS_CLIENT_ID": self.credentials.client_id or "",
                    "GOOGLE_ADS_CLIENT_SECRET": self.credentials.client_secret or "",
                    "GOOGLE_ADS_REFRESH_TOKEN": self.credentials.refresh_token or "",
                    "GOOGLE_ADS_CUSTOMER_ID": self.credentials.customer_id or "",
                }
            )
            
            self.mcp_client = Client(server_params)
            self.client_session = await self.mcp_client.__aenter__()
            
        except Exception as e:
            logger.warning(f"Failed to setup Google Ads MCP connection: {e}")
            # Fallback to direct API calls
            await self._setup_direct_api_connection()
    
    async def _setup_direct_api_connection(self):
        """Setup direct Google Ads API connection"""
        try:
            from google.ads.googleads.client import GoogleAdsClient
            
            config = {
                "developer_token": self.credentials.developer_token,
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret,
                "refresh_token": self.credentials.refresh_token,
                "customer_id": self.credentials.customer_id,
            }
            
            self.google_ads_client = GoogleAdsClient.load_from_dict(config)
            logger.info("Using direct Google Ads API connection")
            
        except Exception as e:
            logger.error(f"Failed to setup direct Google Ads API: {e}")
            raise
    
    async def get_campaigns(self) -> List[AdCampaign]:
        """Get Google Ads campaigns"""
        await self.rate_limiter.wait_if_needed()
        
        try:
            if self.client_session:
                # Use MCP if available
                result = await self.client_session.call_tool("get_campaigns", arguments={})
                if result and result.content:
                    campaigns_data = json.loads(result.content[0].text)
                    return self._parse_google_campaigns(campaigns_data)
            
            elif hasattr(self, 'google_ads_client'):
                # Use direct API
                return await self._get_campaigns_direct_api()
                
        except Exception as e:
            logger.error(f"Failed to get Google campaigns: {e}")
            return []
    
    async def _get_campaigns_direct_api(self) -> List[AdCampaign]:
        """Get campaigns using direct API"""
        campaigns = []
        try:
            ga_service = self.google_ads_client.get_service("GoogleAdsService")
            
            query = """
                SELECT 
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign.start_date,
                    campaign.end_date,
                    campaign.campaign_budget,
                    campaign.advertising_channel_type
                FROM campaign
                WHERE campaign.status != 'REMOVED'
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
                    budget=0.0,  # Budget needs separate query
                    start_date=datetime.strptime(row.campaign.start_date, "%Y-%m-%d"),
                    end_date=datetime.strptime(row.campaign.end_date, "%Y-%m-%d") if row.campaign.end_date else None,
                ))
            
        except Exception as e:
            logger.error(f"Direct API campaign fetch failed: {e}")
            
        return campaigns
    
    def _parse_google_campaigns(self, campaigns_data: Dict) -> List[AdCampaign]:
        """Parse Google campaigns from MCP response"""
        campaigns = []
        for campaign_info in campaigns_data.get("campaigns", []):
            campaigns.append(AdCampaign(
                platform=PlatformType.GOOGLE,
                campaign_id=campaign_info.get("id"),
                name=campaign_info.get("name"),
                status=campaign_info.get("status"),
                budget=float(campaign_info.get("budget", 0)),
                start_date=datetime.fromisoformat(campaign_info.get("start_date", datetime.now().isoformat())),
                end_date=datetime.fromisoformat(campaign_info.get("end_date")) if campaign_info.get("end_date") else None,
            ))
        return campaigns


class TikTokAdsConnector(MCPConnector):
    """TikTok Ads MCP connector"""
    
    def __init__(self, credentials: PlatformCredentials):
        super().__init__(PlatformType.TIKTOK, credentials)
        
    async def _setup_mcp_connection(self):
        """Setup TikTok Ads MCP connection"""
        try:
            # Use the tiktok-ads-mcp package
            server_params = StdioServerParameters(
                command="uvx",
                args=["tiktok-ads-mcp"],
                env={
                    "TIKTOK_ACCESS_TOKEN": self.credentials.access_token or "",
                    "TIKTOK_APP_ID": self.credentials.app_id or "",
                    "TIKTOK_SECRET": self.credentials.secret_key or "",
                    "TIKTOK_ADVERTISER_ID": self.credentials.account_id or "",
                }
            )
            
            self.mcp_client = Client(server_params)
            self.client_session = await self.mcp_client.__aenter__()
            
        except Exception as e:
            logger.error(f"Failed to setup TikTok Ads MCP connection: {e}")
            raise
    
    async def get_campaigns(self) -> List[AdCampaign]:
        """Get TikTok Ads campaigns"""
        await self.rate_limiter.wait_if_needed()
        
        try:
            if not self.client_session:
                await self.connect()
            
            result = await self.client_session.call_tool("get_campaigns", arguments={})
            
            if result and result.content:
                campaigns_data = json.loads(result.content[0].text)
                campaigns = []
                
                for campaign_info in campaigns_data.get("data", {}).get("list", []):
                    campaigns.append(AdCampaign(
                        platform=PlatformType.TIKTOK,
                        campaign_id=campaign_info.get("campaign_id"),
                        name=campaign_info.get("campaign_name"),
                        status=campaign_info.get("status"),
                        budget=float(campaign_info.get("budget", 0)),
                        start_date=datetime.fromisoformat(campaign_info.get("create_time", datetime.now().isoformat())),
                        objective=campaign_info.get("objective_type")
                    ))
                
                return campaigns
                
        except Exception as e:
            logger.error(f"Failed to get TikTok campaigns: {e}")
            return []


class MCPAdPlatformManager:
    """Unified manager for all ad platform MCP connectors"""
    
    def __init__(self):
        self.connectors: Dict[PlatformType, MCPConnector] = {}
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
                connector = MetaAdsConnector(creds)
            elif platform == PlatformType.GOOGLE:
                connector = GoogleAdsConnector(creds)
            elif platform == PlatformType.TIKTOK:
                connector = TikTokAdsConnector(creds)
            else:
                logger.error(f"Unsupported platform: {platform.value}")
                return False
            
            success = await connector.connect()
            if success:
                self.connectors[platform] = connector
                logger.info(f"Successfully initialized {platform.value} connector")
                return True
            else:
                logger.error(f"Failed to connect {platform.value} connector")
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
    
    async def get_platform_performance(self, platform: PlatformType, 
                                     campaign_id: str,
                                     start_date: datetime, 
                                     end_date: datetime) -> Optional[AdPerformance]:
        """Get performance metrics for a specific platform campaign"""
        connector = self.connectors.get(platform)
        if not connector:
            logger.error(f"No connector found for {platform.value}")
            return None
        
        try:
            return await connector.get_campaign_performance(campaign_id, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get performance from {platform.value}: {e}")
            return None
    
    async def create_cross_platform_campaign(self, campaign_data: Dict[str, Any], 
                                           platforms: List[PlatformType]) -> Dict[PlatformType, Optional[AdCampaign]]:
        """Create campaigns across multiple platforms"""
        results = {}
        
        for platform in platforms:
            connector = self.connectors.get(platform)
            if not connector:
                logger.warning(f"No connector for {platform.value}, skipping")
                results[platform] = None
                continue
            
            try:
                # Adapt campaign data for platform-specific requirements
                platform_data = self._adapt_campaign_data(campaign_data, platform)
                campaign = await connector.create_campaign(platform_data)
                results[platform] = campaign
                
                if campaign:
                    logger.info(f"Created campaign {campaign.campaign_id} on {platform.value}")
                else:
                    logger.error(f"Failed to create campaign on {platform.value}")
                    
            except Exception as e:
                logger.error(f"Error creating campaign on {platform.value}: {e}")
                results[platform] = None
        
        return results
    
    def _adapt_campaign_data(self, campaign_data: Dict[str, Any], platform: PlatformType) -> Dict[str, Any]:
        """Adapt campaign data for platform-specific requirements"""
        adapted_data = campaign_data.copy()
        
        if platform == PlatformType.META:
            # Meta-specific adaptations
            if "objective" not in adapted_data:
                adapted_data["objective"] = "REACH"
            adapted_data["status"] = "PAUSED"  # Start paused for safety
            
        elif platform == PlatformType.GOOGLE:
            # Google Ads-specific adaptations
            if "advertising_channel_type" not in adapted_data:
                adapted_data["advertising_channel_type"] = "SEARCH"
            
        elif platform == PlatformType.TIKTOK:
            # TikTok-specific adaptations
            if "objective_type" not in adapted_data:
                adapted_data["objective_type"] = "REACH"
        
        return adapted_data
    
    async def get_connection_status(self) -> Dict[PlatformType, Dict[str, Any]]:
        """Get connection status for all platforms"""
        status = {}
        
        for platform in PlatformType:
            is_connected = platform in self.connectors
            has_credentials = self._has_valid_credentials(platform)
            
            status[platform] = {
                "connected": is_connected,
                "has_credentials": has_credentials,
                "status": "connected" if is_connected else ("ready" if has_credentials else "missing_credentials")
            }
        
        return status
    
    def _has_valid_credentials(self, platform: PlatformType) -> bool:
        """Check if platform has valid credentials"""
        creds = self.credentials.get(platform)
        if not creds:
            return False
        
        if platform == PlatformType.META:
            return bool(creds.access_token and creds.app_id and creds.secret_key)
        elif platform == PlatformType.GOOGLE:
            return bool(creds.developer_token and creds.client_id and 
                       creds.client_secret and creds.refresh_token and creds.customer_id)
        elif platform == PlatformType.TIKTOK:
            return bool(creds.access_token and creds.app_id and creds.secret_key)
        
        return False
    
    async def disconnect_all(self):
        """Disconnect from all platforms"""
        for platform, connector in self.connectors.items():
            try:
                await connector.disconnect()
                logger.info(f"Disconnected from {platform.value}")
            except Exception as e:
                logger.error(f"Error disconnecting from {platform.value}: {e}")
        
        self.connectors.clear()


# Convenience functions for easy usage

async def get_mcp_manager() -> MCPAdPlatformManager:
    """Get a configured MCP manager instance"""
    manager = MCPAdPlatformManager()
    return manager


async def quick_setup() -> Tuple[MCPAdPlatformManager, Dict[PlatformType, bool]]:
    """Quick setup and initialization of all platforms"""
    manager = await get_mcp_manager()
    results = await manager.initialize_all_platforms()
    
    # Log results
    connected_platforms = [p.value for p, success in results.items() if success]
    failed_platforms = [p.value for p, success in results.items() if not success]
    
    logger.info(f"Connected platforms: {connected_platforms}")
    if failed_platforms:
        logger.warning(f"Failed to connect: {failed_platforms}")
    
    return manager, results


async def test_connections() -> Dict[str, Any]:
    """Test all platform connections and return status report"""
    manager, init_results = await quick_setup()
    
    status_report = {
        "timestamp": datetime.now().isoformat(),
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


# CLI interface
async def main():
    """CLI interface for testing MCP connectors"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GAELP MCP Connectors")
    parser.add_argument("--test", action="store_true", help="Test all connections")
    parser.add_argument("--platform", choices=["meta", "google", "tiktok"], help="Test specific platform")
    parser.add_argument("--campaigns", action="store_true", help="List campaigns from all platforms")
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing MCP connections...")
        report = await test_connections()
        print(json.dumps(report, indent=2, default=str))
        
    elif args.platform:
        platform = PlatformType(args.platform)
        manager = await get_mcp_manager()
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
        
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())