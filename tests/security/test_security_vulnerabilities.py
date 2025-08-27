"""
Security vulnerability tests for GAELP platform.
"""

import json
import uuid
from base64 import b64encode
from datetime import datetime, timedelta
from urllib.parse import quote

import pytest
from httpx import AsyncClient

from tests.conftest import TEST_CONFIG


class TestInputValidationSecurity:
    """Test security of input validation mechanisms."""

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_sql_injection_protection(
        self,
        async_client: AsyncClient,
        security_test_payloads: dict
    ):
        """Test protection against SQL injection attacks."""
        
        sql_payloads = security_test_payloads["sql_injection"]
        
        for payload in sql_payloads:
            # Test in query parameters
            response = await async_client.get(
                f"/agents/search?name={quote(payload)}"
            )
            
            # Should not return 500 (indicating SQL error)
            assert response.status_code != 500, \
                f"SQL injection payload caused server error: {payload}"
            
            # Should return proper error response for invalid input
            if response.status_code == 400:
                error_data = response.json()
                assert "sql" not in error_data.get("error", "").lower()
            
            # Test in JSON payload
            agent_config = {
                "agent_name": payload,
                "algorithm": "PPO"
            }
            
            response = await async_client.post("/agents", json=agent_config)
            
            # Should handle malicious input gracefully
            assert response.status_code in [400, 422], \
                f"SQL injection in JSON not properly handled: {payload}"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_xss_protection(
        self,
        async_client: AsyncClient,
        security_test_payloads: dict
    ):
        """Test protection against XSS attacks."""
        
        xss_payloads = security_test_payloads["xss_payloads"]
        
        for payload in xss_payloads:
            # Test XSS in campaign creative
            campaign_data = {
                "creative": {
                    "headline": payload,
                    "description": f"Safe description with XSS: {payload}"
                },
                "budget": {"daily_budget": 100.0}
            }
            
            response = await async_client.post(
                f"/agents/{uuid.uuid4()}/campaigns/validate",
                json=campaign_data
            )
            
            # XSS payloads should be caught by content safety
            if response.status_code == 200:
                validation_result = response.json()
                # Should flag as unsafe content
                assert validation_result.get("content_safe") is False, \
                    f"XSS payload not caught by content safety: {payload}"
            
            # Test XSS in API responses
            response = await async_client.get(
                f"/environments/{uuid.uuid4()}/render",
                params={"user_input": payload}
            )
            
            if response.status_code == 200:
                response_text = response.text
                # Response should not contain unescaped script tags
                assert "<script>" not in response_text.lower(), \
                    f"XSS payload in response not escaped: {payload}"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_command_injection_protection(
        self,
        async_client: AsyncClient,
        security_test_payloads: dict
    ):
        """Test protection against command injection attacks."""
        
        command_payloads = security_test_payloads["command_injection"]
        
        for payload in command_payloads:
            # Test in file operations (if any)
            response = await async_client.post(
                "/agents/export",
                json={
                    "agent_id": str(uuid.uuid4()),
                    "filename": payload
                }
            )
            
            # Command injection should be blocked
            assert response.status_code in [400, 422], \
                f"Command injection not blocked: {payload}"
            
            # Test in environment configuration
            env_config = {
                "type": "simulation",
                "config_path": payload
            }
            
            response = await async_client.post("/environments", json=env_config)
            
            assert response.status_code in [400, 422], \
                f"Command injection in environment config not blocked: {payload}"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_path_traversal_protection(
        self,
        async_client: AsyncClient,
        security_test_payloads: dict
    ):
        """Test protection against path traversal attacks."""
        
        path_payloads = security_test_payloads["path_traversal"]
        
        for payload in path_payloads:
            # Test in file download endpoints
            response = await async_client.get(f"/files/download/{quote(payload)}")
            
            # Path traversal should be blocked
            assert response.status_code in [400, 403, 404], \
                f"Path traversal not blocked: {payload}"
            
            # Test in static file serving
            response = await async_client.get(f"/static/{quote(payload)}")
            
            assert response.status_code in [400, 403, 404], \
                f"Path traversal in static files not blocked: {payload}"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_large_payload_protection(
        self,
        async_client: AsyncClient
    ):
        """Test protection against oversized payloads."""
        
        # Create very large payload (10MB)
        large_payload = {
            "data": "x" * (10 * 1024 * 1024),
            "agent_config": {"algorithm": "PPO"}
        }
        
        response = await async_client.post("/agents", json=large_payload)
        
        # Should reject oversized payloads
        assert response.status_code in [400, 413, 422], \
            "Large payload not rejected"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_malformed_json_handling(
        self,
        async_client: AsyncClient
    ):
        """Test handling of malformed JSON payloads."""
        
        malformed_payloads = [
            '{"incomplete": ',
            '{"trailing_comma": "value",}',
            '{"duplicate": "key", "duplicate": "value"}',
            '{"unclosed_string": "value}',
            'not_json_at_all',
            '{"control_chars": "\x00\x01\x02"}'
        ]
        
        for payload in malformed_payloads:
            response = await async_client.post(
                "/agents",
                content=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Should handle malformed JSON gracefully
            assert response.status_code in [400, 422], \
                f"Malformed JSON not handled properly: {payload[:50]}"


class TestAuthenticationSecurity:
    """Test authentication and authorization security."""

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_missing_authentication(self):
        """Test API behavior without authentication."""
        
        # Create client without authentication headers
        async with AsyncClient(base_url=TEST_CONFIG["api"]["base_url"]) as client:
            
            protected_endpoints = [
                ("GET", "/agents"),
                ("POST", "/agents"),
                ("GET", f"/agents/{uuid.uuid4()}"),
                ("DELETE", f"/agents/{uuid.uuid4()}"),
                ("POST", f"/agents/{uuid.uuid4()}/train"),
                ("GET", "/environments"),
                ("POST", "/environments")
            ]
            
            for method, endpoint in protected_endpoints:
                response = await client.request(method, endpoint)
                
                # Should require authentication
                assert response.status_code == 401, \
                    f"Endpoint {method} {endpoint} doesn't require authentication"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_invalid_token_handling(
        self,
        async_client: AsyncClient
    ):
        """Test handling of invalid authentication tokens."""
        
        invalid_tokens = [
            "invalid_token",
            "expired.jwt.token",
            "",
            "Bearer ",
            "Bearer invalid",
            "malformed.jwt"
        ]
        
        for token in invalid_tokens:
            headers = {"Authorization": f"Bearer {token}"}
            
            response = await async_client.get("/agents", headers=headers)
            
            # Should reject invalid tokens
            assert response.status_code in [401, 403], \
                f"Invalid token accepted: {token}"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_token_expiration(
        self,
        async_client: AsyncClient
    ):
        """Test token expiration handling."""
        
        # Create expired token (mock)
        expired_token_payload = {
            "sub": str(uuid.uuid4()),
            "exp": int((datetime.utcnow() - timedelta(hours=1)).timestamp()),
            "iat": int((datetime.utcnow() - timedelta(hours=2)).timestamp())
        }
        
        # This would normally be a properly signed but expired JWT
        expired_token = "expired.jwt.token"
        headers = {"Authorization": f"Bearer {expired_token}"}
        
        response = await async_client.get("/agents", headers=headers)
        
        # Should reject expired token
        assert response.status_code == 401
        
        error_data = response.json()
        assert "expired" in error_data.get("error", "").lower()

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_privilege_escalation_protection(
        self,
        async_client: AsyncClient
    ):
        """Test protection against privilege escalation."""
        
        # Test accessing admin endpoints with regular user token
        regular_user_token = "regular.user.token"
        headers = {"Authorization": f"Bearer {regular_user_token}"}
        
        admin_endpoints = [
            "/admin/users",
            "/admin/system/config",
            "/admin/agents/all",
            "/admin/emergency-shutdown"
        ]
        
        for endpoint in admin_endpoints:
            response = await async_client.get(endpoint, headers=headers)
            
            # Should deny access to admin endpoints
            assert response.status_code in [403, 404], \
                f"Regular user can access admin endpoint: {endpoint}"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_cross_user_data_access(
        self,
        async_client: AsyncClient
    ):
        """Test that users cannot access other users' data."""
        
        user1_token = "user1.token"
        user2_token = "user2.token"
        
        # Create agent as user 1
        headers1 = {"Authorization": f"Bearer {user1_token}"}
        agent_response = await async_client.post(
            "/agents",
            json={"algorithm": "PPO"},
            headers=headers1
        )
        
        if agent_response.status_code == 201:
            agent_id = agent_response.json()["agent_id"]
            
            # Try to access user 1's agent as user 2
            headers2 = {"Authorization": f"Bearer {user2_token}"}
            response = await async_client.get(
                f"/agents/{agent_id}",
                headers=headers2
            )
            
            # Should deny cross-user access
            assert response.status_code in [403, 404], \
                "User can access another user's agent"


class TestDataProtectionSecurity:
    """Test data protection and privacy security."""

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_sensitive_data_exposure(
        self,
        async_client: AsyncClient,
        agent_id: str
    ):
        """Test that sensitive data is not exposed in responses."""
        
        # Get agent configuration
        response = await async_client.get(f"/agents/{agent_id}")
        
        if response.status_code == 200:
            agent_data = response.json()
            
            # Check that sensitive fields are not exposed
            sensitive_fields = [
                "api_key",
                "secret",
                "password",
                "private_key",
                "access_token",
                "refresh_token"
            ]
            
            agent_str = json.dumps(agent_data).lower()
            
            for field in sensitive_fields:
                assert field not in agent_str, \
                    f"Sensitive field '{field}' exposed in agent data"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_pii_data_handling(
        self,
        async_client: AsyncClient
    ):
        """Test handling of personally identifiable information."""
        
        # Test with PII in campaign data
        pii_campaign = {
            "creative": {
                "headline": "Contact John Smith at john.smith@email.com",
                "description": "Call 555-1234 or visit 123 Main St"
            },
            "targeting": {
                "user_data": {
                    "email": "user@example.com",
                    "phone": "555-5678",
                    "address": "456 Oak Ave"
                }
            }
        }
        
        response = await async_client.post(
            f"/agents/{uuid.uuid4()}/campaigns/validate",
            json=pii_campaign
        )
        
        # PII should be flagged by content safety
        if response.status_code == 200:
            validation = response.json()
            # Should detect PII and flag as requiring review
            assert validation.get("requires_review") is True or \
                   validation.get("pii_detected") is True, \
                   "PII not detected in campaign content"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_data_encryption_in_transit(
        self,
        async_client: AsyncClient
    ):
        """Test that data is encrypted in transit."""
        
        # This test would check HTTPS/TLS configuration
        # For testing purposes, verify secure headers are present
        
        response = await async_client.get("/health")
        
        if response.status_code == 200:
            # Check for security headers
            headers = response.headers
            
            security_headers = [
                "strict-transport-security",
                "x-content-type-options",
                "x-frame-options",
                "x-xss-protection"
            ]
            
            for header in security_headers:
                assert header in headers, \
                    f"Missing security header: {header}"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_audit_logging_security(
        self,
        async_client: AsyncClient,
        agent_id: str
    ):
        """Test security of audit logging functionality."""
        
        # Perform actions that should be logged
        actions = [
            ("GET", f"/agents/{agent_id}"),
            ("POST", f"/agents/{agent_id}/train"),
            ("DELETE", f"/agents/{agent_id}")
        ]
        
        for method, endpoint in actions:
            response = await async_client.request(method, endpoint)
            
            # Actions should be logged (verify log injection protection)
            # Try to inject malicious content in user agent
            malicious_headers = {
                "User-Agent": "'; DROP TABLE audit_logs; --",
                "X-Forwarded-For": "127.0.0.1; rm -rf /",
                "Authorization": "Bearer valid.token"
            }
            
            response = await async_client.request(
                method, 
                endpoint, 
                headers=malicious_headers
            )
            
            # Should handle malicious headers safely
            # (Response code may vary, but should not cause server error)
            assert response.status_code != 500, \
                "Malicious headers caused server error"


class TestRateLimitingSecurity:
    """Test rate limiting and DoS protection."""

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_api_rate_limiting(
        self,
        async_client: AsyncClient
    ):
        """Test API rate limiting protection."""
        
        # Make rapid requests to trigger rate limiting
        requests_made = 0
        rate_limited = False
        
        for i in range(1000):  # Try to make 1000 requests rapidly
            response = await async_client.get("/health")
            requests_made += 1
            
            if response.status_code == 429:  # Too Many Requests
                rate_limited = True
                
                # Check rate limiting headers
                assert "X-RateLimit-Limit" in response.headers
                assert "X-RateLimit-Remaining" in response.headers
                assert "Retry-After" in response.headers
                break
            
            # Small delay to avoid overwhelming the test
            if i % 10 == 0:
                await asyncio.sleep(0.01)
        
        # Should eventually hit rate limit
        assert rate_limited, \
            f"Rate limiting not triggered after {requests_made} requests"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_request_size_limiting(
        self,
        async_client: AsyncClient
    ):
        """Test protection against oversized requests."""
        
        # Test various oversized payloads
        size_tests = [
            (1024 * 1024, "1MB payload"),      # 1MB
            (5 * 1024 * 1024, "5MB payload"),  # 5MB
            (10 * 1024 * 1024, "10MB payload") # 10MB
        ]
        
        for size, description in size_tests:
            large_data = "x" * size
            payload = {"data": large_data}
            
            response = await async_client.post("/agents", json=payload)
            
            # Large payloads should be rejected
            assert response.status_code in [400, 413, 422], \
                f"Large payload not rejected: {description}"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_concurrent_request_limiting(
        self,
        async_client: AsyncClient
    ):
        """Test protection against concurrent request flooding."""
        
        import asyncio
        
        async def make_request():
            """Make a single request."""
            return await async_client.get("/health")
        
        # Launch many concurrent requests
        tasks = [make_request() for _ in range(100)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful vs rate-limited responses
        success_count = 0
        rate_limited_count = 0
        error_count = 0
        
        for response in responses:
            if isinstance(response, Exception):
                error_count += 1
            elif response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited_count += 1
            else:
                error_count += 1
        
        # Should have some rate limiting with high concurrency
        assert rate_limited_count > 0 or error_count > 0, \
            "No rate limiting observed with high concurrency"
        
        # But some requests should still succeed
        assert success_count > 0, \
            "All requests failed - system may be overloaded"


class TestBusinessLogicSecurity:
    """Test security of business logic implementation."""

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_budget_manipulation_protection(
        self,
        async_client: AsyncClient,
        agent_id: str
    ):
        """Test protection against budget manipulation attacks."""
        
        # Set budget limit
        budget_config = {
            "daily_limit": 100.0,
            "total_limit": 1000.0
        }
        
        response = await async_client.post(
            f"/agents/{agent_id}/budget-limits",
            json=budget_config
        )
        
        # Try various budget manipulation attacks
        manipulation_attempts = [
            {"daily_budget": -50.0},     # Negative budget
            {"daily_budget": 999999.0},  # Extremely high budget
            {"daily_budget": 0.0},       # Zero budget
            {"daily_budget": "unlimited"}, # Non-numeric value
            {"daily_budget": None}       # Null value
        ]
        
        for attempt in manipulation_attempts:
            campaign_data = {
                "creative": {"headline": "Test", "description": "Test"},
                "budget": attempt
            }
            
            response = await async_client.post(
                f"/agents/{agent_id}/campaigns",
                json=campaign_data
            )
            
            # Invalid budget values should be rejected
            assert response.status_code in [400, 422], \
                f"Budget manipulation not blocked: {attempt}"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_agent_isolation(
        self,
        async_client: AsyncClient
    ):
        """Test that agents are properly isolated from each other."""
        
        # Create two agents
        agent1_response = await async_client.post(
            "/agents",
            json={"algorithm": "PPO", "name": "Agent 1"}
        )
        agent2_response = await async_client.post(
            "/agents", 
            json={"algorithm": "PPO", "name": "Agent 2"}
        )
        
        if agent1_response.status_code == 201 and agent2_response.status_code == 201:
            agent1_id = agent1_response.json()["agent_id"]
            agent2_id = agent2_response.json()["agent_id"]
            
            # Start training for agent 1
            training_response = await async_client.post(
                f"/agents/{agent1_id}/train",
                json={"environment_id": str(uuid.uuid4())}
            )
            
            # Try to stop agent 1's training using agent 2's ID in the request
            stop_response = await async_client.post(
                f"/agents/{agent2_id}/stop-training",
                json={"target_agent": agent1_id}
            )
            
            # Should not allow cross-agent control
            assert stop_response.status_code in [400, 403, 404], \
                "Agent can control another agent"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_campaign_approval_bypass(
        self,
        async_client: AsyncClient,
        agent_id: str
    ):
        """Test protection against campaign approval bypass."""
        
        # Configure agent to require approval for campaigns
        config_response = await async_client.post(
            f"/agents/{agent_id}/config",
            json={"approval_required": True}
        )
        
        # Try to bypass approval with various methods
        bypass_attempts = [
            {"bypass_approval": True},
            {"approved": True},
            {"force_approve": True},
            {"admin_override": True}
        ]
        
        for attempt in bypass_attempts:
            campaign_data = {
                "creative": {"headline": "Test", "description": "Test"},
                "budget": {"daily_budget": 50.0},
                **attempt
            }
            
            response = await async_client.post(
                f"/agents/{agent_id}/campaigns",
                json=campaign_data
            )
            
            if response.status_code == 202:  # Accepted but pending approval
                status_data = response.json()
                # Should still require approval despite bypass attempt
                assert status_data.get("requires_approval") is True, \
                    f"Approval bypass succeeded: {attempt}"