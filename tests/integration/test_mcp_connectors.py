"""
Integration tests for MCP connectors with external ad platforms.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import pytest
from httpx import AsyncClient

from mcp_connectors.meta_ads import MetaAdsConnector
from mcp_connectors.google_ads import GoogleAdsConnector
from mcp_connectors.shared.auth import AuthManager
from mcp_connectors.shared.rate_limiter import RateLimiter


class TestMetaAdsConnector:
    """Test suite for Meta Ads MCP connector integration."""

    @pytest.fixture
    def meta_connector(self, mock_api_responses):
        """Create MetaAdsConnector with mocked API responses."""
        connector = MetaAdsConnector(
            app_id="test_app_id",
            app_secret="test_app_secret",
            access_token="test_access_token"
        )
        
        # Mock HTTP client
        connector._client = AsyncMock()
        connector._client.get.return_value.json.return_value = mock_api_responses["meta_ads"]
        connector._client.post.return_value.json.return_value = {"id": "123456789"}
        
        return connector

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_create_campaign_success(
        self,
        meta_connector: MetaAdsConnector,
        sample_ad_campaign: dict
    ):
        """Test successful campaign creation via Meta Ads API."""
        campaign_config = {
            "name": "Test Campaign",
            "objective": "CONVERSIONS",
            "status": "PAUSED",  # Start paused for safety
            "daily_budget": 10000,  # $100.00 in cents
            "targeting": sample_ad_campaign["targeting"]
        }
        
        campaign_id = await meta_connector.create_campaign(campaign_config)
        
        assert campaign_id == "123456789"
        meta_connector._client.post.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_campaign_insights(
        self,
        meta_connector: MetaAdsConnector
    ):
        """Test retrieving campaign insights from Meta Ads."""
        campaign_id = "123456789"
        date_range = {
            "since": "2024-01-01",
            "until": "2024-01-31"
        }
        
        insights = await meta_connector.get_campaign_insights(campaign_id, date_range)
        
        assert "impressions" in insights
        assert "clicks" in insights
        assert "spend" in insights
        assert insights["impressions"] == 10000

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_update_campaign_budget(
        self,
        meta_connector: MetaAdsConnector
    ):
        """Test updating campaign budget via Meta Ads API."""
        campaign_id = "123456789"
        new_budget = 20000  # $200.00
        
        success = await meta_connector.update_campaign_budget(campaign_id, new_budget)
        
        assert success is True
        meta_connector._client.post.assert_called()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pause_campaign(
        self,
        meta_connector: MetaAdsConnector
    ):
        """Test pausing campaign via Meta Ads API."""
        campaign_id = "123456789"
        
        success = await meta_connector.pause_campaign(campaign_id)
        
        assert success is True
        meta_connector._client.post.assert_called()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_rate_limiting(
        self,
        meta_connector: MetaAdsConnector
    ):
        """Test rate limiting behavior with Meta Ads API."""
        # Simulate rate limit response
        rate_limit_response = AsyncMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {
            "x-app-usage": '{"call_count": 100, "total_cputime": 95, "total_time": 95}',
            "retry-after": "60"
        }
        
        meta_connector._client.get.side_effect = [
            rate_limit_response,  # First call hits rate limit
            AsyncMock(json=lambda: {"data": []})  # Second call succeeds
        ]
        
        with patch("asyncio.sleep") as mock_sleep:
            result = await meta_connector.get_campaigns()
            
            # Should have waited before retrying
            mock_sleep.assert_called_once_with(60)
            assert result == {"data": []}

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_webhook_verification(
        self,
        meta_connector: MetaAdsConnector
    ):
        """Test webhook verification for real-time updates."""
        webhook_data = {
            "hub.mode": "subscribe",
            "hub.challenge": "test_challenge",
            "hub.verify_token": "test_verify_token"
        }
        
        is_valid = meta_connector.verify_webhook(webhook_data)
        assert is_valid is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_handle_webhook_update(
        self,
        meta_connector: MetaAdsConnector
    ):
        """Test handling webhook updates from Meta."""
        webhook_payload = {
            "entry": [
                {
                    "id": "123456789",
                    "changes": [
                        {
                            "field": "campaign",
                            "value": {
                                "campaign_id": "123456789",
                                "event_type": "campaign_delivery",
                                "event_time": datetime.utcnow().isoformat()
                            }
                        }
                    ]
                }
            ]
        }
        
        with patch.object(meta_connector, 'process_campaign_update') as mock_process:
            await meta_connector.handle_webhook(webhook_payload)
            
            mock_process.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_invalid_token(
        self,
        meta_connector: MetaAdsConnector
    ):
        """Test error handling for invalid access token."""
        error_response = AsyncMock()
        error_response.status_code = 401
        error_response.json.return_value = {
            "error": {
                "code": 190,
                "message": "Invalid OAuth access token"
            }
        }
        
        meta_connector._client.get.return_value = error_response
        
        with pytest.raises(Exception, match="Invalid OAuth access token"):
            await meta_connector.get_campaigns()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_operations(
        self,
        meta_connector: MetaAdsConnector
    ):
        """Test batch operations for efficiency."""
        campaign_updates = [
            {"campaign_id": "123", "budget": 10000},
            {"campaign_id": "456", "budget": 15000},
            {"campaign_id": "789", "budget": 20000}
        ]
        
        # Mock batch API response
        meta_connector._client.post.return_value.json.return_value = [
            {"code": 200, "body": {"id": "123"}},
            {"code": 200, "body": {"id": "456"}},
            {"code": 200, "body": {"id": "789"}}
        ]
        
        results = await meta_connector.batch_update_campaigns(campaign_updates)
        
        assert len(results) == 3
        assert all(r["code"] == 200 for r in results)


class TestGoogleAdsConnector:
    """Test suite for Google Ads MCP connector integration."""

    @pytest.fixture
    def google_connector(self, mock_api_responses):
        """Create GoogleAdsConnector with mocked API responses."""
        connector = GoogleAdsConnector(
            developer_token="test_dev_token",
            client_id="test_client_id",
            client_secret="test_client_secret",
            refresh_token="test_refresh_token"
        )
        
        # Mock Google Ads client
        connector._client = Mock()
        connector._client.campaign_service = Mock()
        connector._client.campaign_budget_service = Mock()
        
        return connector

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_create_campaign_budget(
        self,
        google_connector: GoogleAdsConnector
    ):
        """Test creating campaign budget in Google Ads."""
        budget_config = {
            "name": "Test Budget",
            "amount_micros": 10000000000,  # $10,000
            "delivery_method": "STANDARD"
        }
        
        # Mock successful budget creation
        google_connector._client.campaign_budget_service.mutate_campaign_budgets.return_value.results = [
            Mock(resource_name="customers/123/campaignBudgets/456")
        ]
        
        budget_resource = await google_connector.create_campaign_budget("123", budget_config)
        
        assert "campaignBudgets" in budget_resource
        google_connector._client.campaign_budget_service.mutate_campaign_budgets.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_create_campaign(
        self,
        google_connector: GoogleAdsConnector,
        sample_ad_campaign: dict
    ):
        """Test creating campaign in Google Ads."""
        campaign_config = {
            "name": "Test Campaign",
            "advertising_channel_type": "SEARCH",
            "status": "PAUSED",
            "campaign_budget": "customers/123/campaignBudgets/456",
            "bidding_strategy_type": "MANUAL_CPC"
        }
        
        # Mock successful campaign creation
        google_connector._client.campaign_service.mutate_campaigns.return_value.results = [
            Mock(resource_name="customers/123/campaigns/789")
        ]
        
        campaign_resource = await google_connector.create_campaign("123", campaign_config)
        
        assert "campaigns" in campaign_resource
        google_connector._client.campaign_service.mutate_campaigns.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_campaign_performance(
        self,
        google_connector: GoogleAdsConnector
    ):
        """Test retrieving campaign performance metrics."""
        customer_id = "123"
        campaign_resource = "customers/123/campaigns/789"
        
        # Mock Google Ads query response
        mock_row = Mock()
        mock_row.campaign.resource_name = campaign_resource
        mock_row.metrics.impressions = 10000
        mock_row.metrics.clicks = 300
        mock_row.metrics.cost_micros = 700000000  # $700
        
        google_connector._client.search_stream.return_value = [mock_row]
        
        performance = await google_connector.get_campaign_performance(
            customer_id, 
            campaign_resource,
            date_range="LAST_7_DAYS"
        )
        
        assert performance["impressions"] == 10000
        assert performance["clicks"] == 300
        assert performance["cost"] == 700.0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_update_campaign_status(
        self,
        google_connector: GoogleAdsConnector
    ):
        """Test updating campaign status in Google Ads."""
        customer_id = "123"
        campaign_resource = "customers/123/campaigns/789"
        new_status = "ENABLED"
        
        # Mock successful status update
        google_connector._client.campaign_service.mutate_campaigns.return_value.results = [
            Mock(resource_name=campaign_resource)
        ]
        
        success = await google_connector.update_campaign_status(
            customer_id,
            campaign_resource,
            new_status
        )
        
        assert success is True
        google_connector._client.campaign_service.mutate_campaigns.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_keyword_research(
        self,
        google_connector: GoogleAdsConnector
    ):
        """Test keyword research functionality."""
        keywords = ["fitness tracker", "workout monitor", "health device"]
        
        # Mock keyword ideas response
        mock_idea = Mock()
        mock_idea.text = "fitness tracker"
        mock_idea.keyword_idea_metrics.avg_monthly_searches = 10000
        mock_idea.keyword_idea_metrics.competition = "MEDIUM"
        
        google_connector._client.keyword_plan_idea_service.generate_keyword_ideas.return_value.results = [
            mock_idea
        ]
        
        keyword_ideas = await google_connector.get_keyword_ideas("123", keywords)
        
        assert len(keyword_ideas) > 0
        assert keyword_ideas[0]["text"] == "fitness tracker"
        assert keyword_ideas[0]["avg_monthly_searches"] == 10000

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_audience_insights(
        self,
        google_connector: GoogleAdsConnector
    ):
        """Test audience insights functionality."""
        audience_config = {
            "age_ranges": ["AGERANGE_25_34", "AGERANGE_35_44"],
            "genders": ["MALE", "FEMALE"],
            "interests": ["fitness", "health"]
        }
        
        # Mock audience insights response
        mock_insight = Mock()
        mock_insight.dimension = "AGE_RANGE"
        mock_insight.metrics.impressions = 5000
        mock_insight.metrics.clicks = 150
        
        google_connector._client.reach_plan_service.generate_reach_forecast.return_value.forecasts = [
            mock_insight
        ]
        
        insights = await google_connector.get_audience_insights("123", audience_config)
        
        assert "age_range_performance" in insights
        assert insights["total_reach"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_quota_exceeded(
        self,
        google_connector: GoogleAdsConnector
    ):
        """Test error handling for quota exceeded."""
        from google.ads.googleads.errors import GoogleAdsException
        
        # Mock quota exceeded error
        google_connector._client.search_stream.side_effect = GoogleAdsException(
            request_id="test",
            failure=Mock(errors=[Mock(error_code=Mock(quota_error="RESOURCE_EXHAUSTED"))])
        )
        
        with pytest.raises(Exception, match="quota"):
            await google_connector.get_campaign_performance("123", "campaigns/789")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_bulk_campaign_management(
        self,
        google_connector: GoogleAdsConnector
    ):
        """Test bulk campaign management operations."""
        operations = [
            {"action": "pause", "campaign": "customers/123/campaigns/789"},
            {"action": "update_budget", "campaign": "customers/123/campaigns/790", "budget": 15000},
            {"action": "enable", "campaign": "customers/123/campaigns/791"}
        ]
        
        # Mock bulk operation response
        google_connector._client.campaign_service.mutate_campaigns.return_value.results = [
            Mock(resource_name=op["campaign"]) for op in operations
        ]
        
        results = await google_connector.bulk_campaign_operations("123", operations)
        
        assert len(results) == len(operations)
        assert all(r["success"] for r in results)


class TestMCPAuthManager:
    """Test suite for MCP authentication manager."""

    @pytest.fixture
    def auth_manager(self, mock_redis):
        """Create AuthManager with mocked dependencies."""
        with patch("mcp_connectors.shared.auth.redis.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = mock_redis
            
            manager = AuthManager(
                redis_url="redis://localhost:6379",
                encryption_key="test_key_32_chars_long_secret"
            )
            return manager

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_store_and_retrieve_credentials(
        self,
        auth_manager: AuthManager,
        agent_id: str
    ):
        """Test storing and retrieving encrypted credentials."""
        credentials = {
            "platform": "meta_ads",
            "access_token": "test_access_token",
            "app_id": "test_app_id",
            "app_secret": "test_app_secret"
        }
        
        # Store credentials
        await auth_manager.store_credentials(agent_id, "meta_ads", credentials)
        
        # Retrieve credentials
        retrieved = await auth_manager.get_credentials(agent_id, "meta_ads")
        
        assert retrieved["access_token"] == credentials["access_token"]
        assert retrieved["app_id"] == credentials["app_id"]
        auth_manager.redis.set.assert_called()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_token_refresh_meta(
        self,
        auth_manager: AuthManager,
        agent_id: str
    ):
        """Test automatic token refresh for Meta Ads."""
        # Mock expired token scenario
        auth_manager.redis.get.return_value = json.dumps({
            "access_token": "expired_token",
            "refresh_token": "refresh_token",
            "expires_at": (datetime.utcnow() - timedelta(hours=1)).isoformat()
        })
        
        with patch.object(auth_manager, '_refresh_meta_token') as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "new_access_token",
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
            
            credentials = await auth_manager.get_valid_credentials(agent_id, "meta_ads")
            
            assert credentials["access_token"] == "new_access_token"
            mock_refresh.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_token_refresh_google(
        self,
        auth_manager: AuthManager,
        agent_id: str
    ):
        """Test automatic token refresh for Google Ads."""
        # Mock expired token scenario
        auth_manager.redis.get.return_value = json.dumps({
            "access_token": "expired_token",
            "refresh_token": "refresh_token",
            "expires_at": (datetime.utcnow() - timedelta(hours=1)).isoformat()
        })
        
        with patch.object(auth_manager, '_refresh_google_token') as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "new_access_token",
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
            
            credentials = await auth_manager.get_valid_credentials(agent_id, "google_ads")
            
            assert credentials["access_token"] == "new_access_token"
            mock_refresh.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_credentials_encryption(
        self,
        auth_manager: AuthManager,
        agent_id: str
    ):
        """Test that credentials are properly encrypted."""
        sensitive_data = {
            "access_token": "very_secret_token",
            "client_secret": "super_secret_key"
        }
        
        await auth_manager.store_credentials(agent_id, "test_platform", sensitive_data)
        
        # Check that raw stored data is encrypted (not plaintext)
        stored_call = auth_manager.redis.set.call_args[0]
        stored_value = stored_call[1]
        
        assert "very_secret_token" not in stored_value
        assert "super_secret_key" not in stored_value


class TestMCPRateLimiter:
    """Test suite for MCP rate limiter."""

    @pytest.fixture
    def rate_limiter(self, mock_redis):
        """Create RateLimiter with mocked dependencies."""
        with patch("mcp_connectors.shared.rate_limiter.redis.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = mock_redis
            
            limiter = RateLimiter(
                redis_url="redis://localhost:6379",
                default_requests_per_minute=100
            )
            return limiter

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(
        self,
        rate_limiter: RateLimiter,
        agent_id: str
    ):
        """Test rate limit enforcement."""
        # Set low rate limit for testing
        await rate_limiter.set_rate_limit(agent_id, "meta_ads", 5, 60)  # 5 requests per minute
        
        # Mock current request count
        rate_limiter.redis.get.return_value = "4"  # Already made 4 requests
        
        # Should allow 5th request
        can_proceed = await rate_limiter.can_make_request(agent_id, "meta_ads")
        assert can_proceed is True
        
        # Mock hitting limit
        rate_limiter.redis.get.return_value = "5"  # Now at limit
        
        # Should deny 6th request
        can_proceed = await rate_limiter.can_make_request(agent_id, "meta_ads")
        assert can_proceed is False

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_reset(
        self,
        rate_limiter: RateLimiter,
        agent_id: str
    ):
        """Test rate limit reset after time window."""
        # Mock rate limit reached
        rate_limiter.redis.get.return_value = "100"  # At limit
        rate_limiter.redis.ttl.return_value = 0  # TTL expired
        
        can_proceed = await rate_limiter.can_make_request(agent_id, "meta_ads")
        
        # Should reset and allow request
        assert can_proceed is True
        rate_limiter.redis.delete.assert_called()  # Counter reset

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_platform_specific_limits(
        self,
        rate_limiter: RateLimiter,
        agent_id: str
    ):
        """Test platform-specific rate limits."""
        # Set different limits for different platforms
        await rate_limiter.set_rate_limit(agent_id, "meta_ads", 200, 3600)  # 200/hour
        await rate_limiter.set_rate_limit(agent_id, "google_ads", 1000, 3600)  # 1000/hour
        
        # Mock different usage for each platform
        rate_limiter.redis.get.side_effect = lambda key: {
            f"rate_limit:{agent_id}:meta_ads": "150",
            f"rate_limit:{agent_id}:google_ads": "500"
        }.get(key, "0")
        
        # Meta Ads should still allow requests
        can_proceed_meta = await rate_limiter.can_make_request(agent_id, "meta_ads")
        assert can_proceed_meta is True
        
        # Google Ads should also allow requests
        can_proceed_google = await rate_limiter.can_make_request(agent_id, "google_ads")
        assert can_proceed_google is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_burst_protection(
        self,
        rate_limiter: RateLimiter,
        agent_id: str
    ):
        """Test burst protection mechanism."""
        # Enable burst protection
        await rate_limiter.enable_burst_protection(agent_id, max_burst=10, burst_window=5)
        
        # Mock burst of requests
        rate_limiter.redis.get.return_value = "8"  # 8 requests in burst window
        
        # Should allow requests within burst limit
        can_proceed = await rate_limiter.can_make_request(agent_id, "meta_ads")
        assert can_proceed is True
        
        # Mock exceeding burst limit
        rate_limiter.redis.get.return_value = "11"  # Exceeded burst limit
        
        # Should deny request
        can_proceed = await rate_limiter.can_make_request(agent_id, "meta_ads")
        assert can_proceed is False