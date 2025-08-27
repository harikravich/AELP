"""
Unit tests for Environment API endpoints.
"""

import json
import uuid
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import pytest
from httpx import AsyncClient, Response
from fastapi import status

from tests.conftest import TEST_CONFIG


class TestEnvironmentAPI:
    """Test suite for Environment API endpoints."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_environment_reset_success(
        self, 
        async_client: AsyncClient,
        environment_id: str,
        sample_persona_config: dict,
        mock_environment_response: dict
    ):
        """Test successful environment reset."""
        payload = {
            "seed": 42,
            "persona_config": sample_persona_config
        }
        
        with patch("environment_service.reset_environment") as mock_reset:
            mock_reset.return_value = mock_environment_response
            
            response = await async_client.post(
                f"/environments/{environment_id}/reset",
                json=payload
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "initial_state" in data
        assert "reset_id" in data
        assert data["initial_state"]["environment_type"] == "simulated"
        assert data["initial_state"]["available_budget"] == 10000.0
        
        mock_reset.assert_called_once_with(environment_id, payload)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_environment_reset_invalid_persona(
        self,
        async_client: AsyncClient,
        environment_id: str
    ):
        """Test environment reset with invalid persona configuration."""
        payload = {
            "seed": 42,
            "persona_config": {
                "demographics": {
                    "age_range": [200, 300],  # Invalid age range
                    "gender": ["invalid_gender"]  # Invalid gender
                }
            }
        }
        
        response = await async_client.post(
            f"/environments/{environment_id}/reset",
            json=payload
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["code"] == "INVALID_REQUEST"
        assert "persona configuration" in data["error"].lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_environment_reset_missing_persona(
        self,
        async_client: AsyncClient,
        environment_id: str
    ):
        """Test environment reset without persona configuration."""
        payload = {"seed": 42}
        
        response = await async_client.post(
            f"/environments/{environment_id}/reset",
            json=payload
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_environment_step_success(
        self,
        async_client: AsyncClient,
        environment_id: str,
        sample_ad_campaign: dict,
        mock_performance_metrics: dict
    ):
        """Test successful campaign step execution."""
        payload = {"ad_campaign": sample_ad_campaign}
        
        mock_response = {
            "performance_metrics": mock_performance_metrics,
            "reward": 0.85,
            "done": False,
            "info": {
                "campaign_duration": 24,
                "audience_reached": 8500,
                "budget_remaining": 300.0,
                "next_recommended_action": "increase_budget",
                "market_feedback": {
                    "sentiment_score": 0.7,
                    "brand_awareness_lift": 0.15
                }
            }
        }
        
        with patch("environment_service.execute_step") as mock_step:
            mock_step.return_value = mock_response
            
            response = await async_client.post(
                f"/environments/{environment_id}/step",
                json=payload
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "performance_metrics" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data
        
        assert data["reward"] == 0.85
        assert data["done"] is False
        assert data["performance_metrics"]["impressions"] == 10000
        assert data["performance_metrics"]["clicks"] == 300
        
        mock_step.assert_called_once_with(environment_id, payload)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_environment_step_budget_exceeded(
        self,
        async_client: AsyncClient,
        environment_id: str,
        sample_ad_campaign: dict
    ):
        """Test campaign step with budget exceeded."""
        # Modify campaign to exceed budget
        campaign = sample_ad_campaign.copy()
        campaign["budget"]["daily_budget"] = 10000.0  # Excessive budget
        
        payload = {"ad_campaign": campaign}
        
        with patch("environment_service.execute_step") as mock_step:
            mock_step.side_effect = Exception("Budget exceeded safety limit")
            
            response = await async_client.post(
                f"/environments/{environment_id}/step",
                json=payload
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_environment_step_invalid_campaign(
        self,
        async_client: AsyncClient,
        environment_id: str
    ):
        """Test campaign step with invalid campaign structure."""
        payload = {
            "ad_campaign": {
                "creative": {
                    "headline": "",  # Empty headline
                    "description": "x" * 5000  # Too long description
                },
                "budget": {
                    "daily_budget": -100.0  # Negative budget
                }
            }
        }
        
        response = await async_client.post(
            f"/environments/{environment_id}/step",
            json=payload
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_environment_render_json(
        self,
        async_client: AsyncClient,
        environment_id: str
    ):
        """Test environment rendering in JSON format."""
        mock_visualization = {
            "campaign_visualization": {
                "performance_chart": {
                    "metrics_over_time": [
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "impressions": 1000,
                            "clicks": 30,
                            "conversions": 3,
                            "spend": 70.0
                        }
                    ]
                },
                "audience_heatmap": {
                    "demographic_breakdown": {"25-34": 0.4, "35-44": 0.6},
                    "geographic_distribution": {"US": 0.7, "CA": 0.3},
                    "engagement_by_segment": {"high_intent": 0.8, "low_intent": 0.2}
                },
                "creative_performance": [
                    {
                        "creative_id": "creative_001",
                        "performance_score": 0.85,
                        "engagement_metrics": {"ctr": 0.03, "conversion_rate": 0.05}
                    }
                ]
            }
        }
        
        with patch("environment_service.render_environment") as mock_render:
            mock_render.return_value = mock_visualization
            
            response = await async_client.get(
                f"/environments/{environment_id}/render?format=json"
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "campaign_visualization" in data
        assert "performance_chart" in data["campaign_visualization"]
        assert "audience_heatmap" in data["campaign_visualization"]
        assert "creative_performance" in data["campaign_visualization"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_environment_audience_feedback(
        self,
        async_client: AsyncClient,
        environment_id: str
    ):
        """Test audience feedback generation."""
        payload = {
            "ad_creative": {
                "creative_id": "creative_001",
                "headline": "Test Headline",
                "description": "Test Description",
                "image_url": "https://example.com/image.jpg"
            },
            "sample_size": 1000
        }
        
        mock_feedback = {
            "user_responses": [
                {
                    "user_id": str(uuid.uuid4()),
                    "response_type": "click",
                    "engagement_score": 0.8,
                    "sentiment": "positive",
                    "feedback_text": "Great ad!",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ],
            "aggregated_feedback": {
                "overall_sentiment": 0.7,
                "predicted_ctr": 0.035,
                "predicted_conversion_rate": 0.06,
                "audience_fit_score": 0.85
            }
        }
        
        with patch("environment_service.generate_audience_feedback") as mock_feedback_gen:
            mock_feedback_gen.return_value = mock_feedback
            
            response = await async_client.post(
                f"/environments/{environment_id}/audience-feedback",
                json=payload
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "user_responses" in data
        assert "aggregated_feedback" in data
        assert len(data["user_responses"]) > 0
        assert data["aggregated_feedback"]["overall_sentiment"] == 0.7

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_environment_not_found(
        self,
        async_client: AsyncClient
    ):
        """Test API behavior with non-existent environment."""
        non_existent_id = str(uuid.uuid4())
        
        with patch("environment_service.get_environment") as mock_get:
            mock_get.side_effect = Exception("Environment not found")
            
            response = await async_client.get(
                f"/environments/{non_existent_id}/render"
            )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["code"] == "ENVIRONMENT_NOT_FOUND"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unauthorized_access(
        self,
        environment_id: str
    ):
        """Test API behavior without authentication."""
        async with AsyncClient(base_url=TEST_CONFIG["api"]["base_url"]) as client:
            # Don't include authentication headers
            response = await client.post(
                f"/environments/{environment_id}/reset",
                json={"persona_config": {"demographics": {}}}
            )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert data["code"] == "UNAUTHORIZED"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limiting(
        self,
        async_client: AsyncClient,
        environment_id: str,
        sample_persona_config: dict
    ):
        """Test API rate limiting functionality."""
        payload = {
            "seed": 42,
            "persona_config": sample_persona_config
        }
        
        # Simulate rate limiting by making multiple rapid requests
        responses = []
        for _ in range(100):  # Exceed rate limit
            response = await async_client.post(
                f"/environments/{environment_id}/reset",
                json=payload
            )
            responses.append(response)
            
            # Break if we hit rate limit
            if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                break
        
        # Should eventually hit rate limit
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0
        
        # Check rate limit headers
        rate_limited_response = rate_limited_responses[0]
        assert "X-RateLimit-Limit" in rate_limited_response.headers
        assert "X-RateLimit-Remaining" in rate_limited_response.headers
        assert "X-RateLimit-Reset" in rate_limited_response.headers

    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_id", [
        "not-a-uuid",
        "123",
        "",
        "null"
    ])
    @pytest.mark.asyncio
    async def test_invalid_environment_id_format(
        self,
        async_client: AsyncClient,
        invalid_id: str,
        sample_persona_config: dict
    ):
        """Test API behavior with invalid environment ID formats."""
        payload = {
            "seed": 42,
            "persona_config": sample_persona_config
        }
        
        response = await async_client.post(
            f"/environments/{invalid_id}/reset",
            json=payload
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_request_timeout(
        self,
        async_client: AsyncClient,
        environment_id: str,
        sample_persona_config: dict
    ):
        """Test API behavior under request timeout conditions."""
        payload = {
            "seed": 42,
            "persona_config": sample_persona_config
        }
        
        with patch("environment_service.reset_environment") as mock_reset:
            # Simulate slow response
            import asyncio
            async def slow_response(*args):
                await asyncio.sleep(35)  # Longer than timeout
                return {"initial_state": {}}
            
            mock_reset.side_effect = slow_response
            
            response = await async_client.post(
                f"/environments/{environment_id}/reset",
                json=payload,
                timeout=30  # 30 second timeout
            )
        
        # Should timeout
        assert response.status_code == status.HTTP_408_REQUEST_TIMEOUT

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_environment_operations(
        self,
        async_client: AsyncClient,
        environment_id: str,
        sample_persona_config: dict,
        sample_ad_campaign: dict
    ):
        """Test concurrent operations on the same environment."""
        import asyncio
        
        reset_payload = {
            "seed": 42,
            "persona_config": sample_persona_config
        }
        
        step_payload = {
            "ad_campaign": sample_ad_campaign
        }
        
        # Run concurrent operations
        tasks = [
            async_client.post(f"/environments/{environment_id}/reset", json=reset_payload),
            async_client.post(f"/environments/{environment_id}/step", json=step_payload),
            async_client.get(f"/environments/{environment_id}/render")
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least one operation should succeed
        successful_responses = [r for r in responses if not isinstance(r, Exception) and r.status_code == 200]
        assert len(successful_responses) > 0