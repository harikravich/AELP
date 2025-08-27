"""
End-to-end tests for complete agent training pipeline.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import pytest
from httpx import AsyncClient

from tests.conftest import TEST_CONFIG


class TestCompleteTrainingPipeline:
    """Test complete agent training pipeline from simulation to real deployment."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_full_training_lifecycle(
        self,
        async_client: AsyncClient,
        mock_agent_config: dict,
        sample_persona_config: dict,
        sample_ad_campaign: dict,
        performance_benchmarks: dict
    ):
        """Test complete agent training lifecycle."""
        
        # Phase 1: Create agent
        agent_response = await async_client.post("/agents", json=mock_agent_config)
        assert agent_response.status_code == 201
        agent_id = agent_response.json()["agent_id"]
        
        # Phase 2: Create simulation environment
        env_response = await async_client.post("/environments", json={
            "type": "simulation",
            "persona_config": sample_persona_config
        })
        assert env_response.status_code == 201
        environment_id = env_response.json()["environment_id"]
        
        # Phase 3: Start training in simulation
        training_config = {
            "environment_id": environment_id,
            "max_episodes": 100,
            "convergence_threshold": 0.85,
            "safety_checks_enabled": True
        }
        
        training_response = await async_client.post(
            f"/agents/{agent_id}/train",
            json=training_config
        )
        assert training_response.status_code == 202
        job_id = training_response.json()["job_id"]
        
        # Phase 4: Monitor training progress
        max_wait_time = 300  # 5 minutes
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < max_wait_time:
            status_response = await async_client.get(f"/jobs/{job_id}/status")
            assert status_response.status_code == 200
            
            status = status_response.json()
            if status["status"] == "COMPLETED":
                break
            elif status["status"] == "FAILED":
                pytest.fail(f"Training failed: {status.get('error')}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        # Verify training completed successfully
        assert status["status"] == "COMPLETED"
        assert status["convergence_achieved"] is True
        
        # Phase 5: Validate simulation performance
        metrics_response = await async_client.get(f"/agents/{agent_id}/metrics")
        assert metrics_response.status_code == 200
        
        metrics = metrics_response.json()
        assert metrics["final_reward"] >= 0.8  # Good performance threshold
        assert metrics["convergence_score"] >= 0.85
        assert len(metrics["episode_rewards"]) >= 50  # Sufficient training
        
        # Phase 6: Safety validation
        safety_response = await async_client.get(f"/agents/{agent_id}/safety-report")
        assert safety_response.status_code == 200
        
        safety_report = safety_response.json()
        assert safety_report["total_violations"] == 0
        assert safety_report["safety_score"] >= 0.9
        
        # Phase 7: Request real environment deployment
        deployment_request = {
            "target_platform": "meta_ads",
            "initial_budget": 100.0,
            "approval_required": True,
            "safety_monitoring": "enhanced"
        }
        
        deployment_response = await async_client.post(
            f"/agents/{agent_id}/deploy",
            json=deployment_request
        )
        assert deployment_response.status_code == 202
        deployment_id = deployment_response.json()["deployment_id"]
        
        # Phase 8: Verify deployment is pending approval
        deployment_status = await async_client.get(f"/deployments/{deployment_id}")
        assert deployment_status.status_code == 200
        
        deployment_data = deployment_status.json()
        assert deployment_data["status"] == "PENDING_APPROVAL"
        assert deployment_data["requires_human_review"] is True

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_simulation_to_real_transfer(
        self,
        async_client: AsyncClient,
        agent_id: str,
        environment_id: str
    ):
        """Test transfer from simulation to real environment."""
        
        # Step 1: Complete simulation training
        simulation_metrics = {
            "episodes_completed": 1000,
            "average_reward": 0.85,
            "convergence_score": 0.92,
            "safety_violations": 0
        }
        
        # Mock successful simulation completion
        with patch("training_orchestrator.get_training_results") as mock_results:
            mock_results.return_value = simulation_metrics
            
            # Step 2: Validate transfer eligibility
            transfer_response = await async_client.post(
                f"/agents/{agent_id}/validate-transfer",
                json={"target_environment": "real"}
            )
            assert transfer_response.status_code == 200
            
            transfer_validation = transfer_response.json()
            assert transfer_validation["eligible"] is True
            assert transfer_validation["confidence_score"] >= 0.8
        
        # Step 3: Create real environment
        real_env_config = {
            "type": "real",
            "platform": "meta_ads",
            "safety_constraints": {
                "max_daily_budget": 50.0,
                "content_safety_level": "strict",
                "human_approval_required": True
            }
        }
        
        real_env_response = await async_client.post(
            "/environments",
            json=real_env_config
        )
        assert real_env_response.status_code == 201
        real_environment_id = real_env_response.json()["environment_id"]
        
        # Step 4: Initiate transfer
        transfer_config = {
            "source_environment": environment_id,
            "target_environment": real_environment_id,
            "transfer_method": "gradual",
            "monitoring_level": "enhanced"
        }
        
        transfer_start = await async_client.post(
            f"/agents/{agent_id}/transfer",
            json=transfer_config
        )
        assert transfer_start.status_code == 202
        transfer_id = transfer_start.json()["transfer_id"]
        
        # Step 5: Monitor transfer progress
        transfer_status = await async_client.get(f"/transfers/{transfer_id}")
        assert transfer_status.status_code == 200
        
        status_data = transfer_status.json()
        assert status_data["status"] in ["INITIATED", "IN_PROGRESS"]
        assert status_data["safety_monitoring_active"] is True

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_budget_control_enforcement(
        self,
        async_client: AsyncClient,
        agent_id: str
    ):
        """Test end-to-end budget control enforcement."""
        
        # Set strict budget limits
        budget_config = {
            "daily_limit": 100.0,
            "total_limit": 1000.0,
            "emergency_stop_threshold": 0.9,
            "approval_required_above": 50.0
        }
        
        budget_response = await async_client.post(
            f"/agents/{agent_id}/budget-limits",
            json=budget_config
        )
        assert budget_response.status_code == 200
        
        # Simulate campaign with budget near limit
        high_budget_campaign = {
            "budget": {"daily_budget": 95.0},  # Close to limit
            "creative": {
                "headline": "Test Campaign",
                "description": "Safe test content"
            }
        }
        
        # Should trigger approval workflow
        campaign_response = await async_client.post(
            f"/agents/{agent_id}/campaigns",
            json=high_budget_campaign
        )
        assert campaign_response.status_code == 202  # Pending approval
        
        approval_data = campaign_response.json()
        assert approval_data["requires_approval"] is True
        assert approval_data["reason"] == "budget_threshold_exceeded"
        
        # Try campaign that exceeds limit
        excessive_budget_campaign = {
            "budget": {"daily_budget": 150.0},  # Exceeds limit
            "creative": {
                "headline": "Expensive Campaign",
                "description": "Too expensive"
            }
        }
        
        # Should be rejected immediately
        rejection_response = await async_client.post(
            f"/agents/{agent_id}/campaigns",
            json=excessive_budget_campaign
        )
        assert rejection_response.status_code == 400
        
        error_data = rejection_response.json()
        assert "budget limit exceeded" in error_data["error"].lower()

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_content_safety_pipeline(
        self,
        async_client: AsyncClient,
        agent_id: str
    ):
        """Test end-to-end content safety validation."""
        
        # Test safe content
        safe_campaign = {
            "creative": {
                "headline": "Discover Amazing Fitness Technology",
                "description": "Transform your workout with innovative fitness tracking",
                "image_url": "https://example.com/safe-image.jpg"
            },
            "budget": {"daily_budget": 50.0}
        }
        
        safe_response = await async_client.post(
            f"/agents/{agent_id}/campaigns/validate",
            json=safe_campaign
        )
        assert safe_response.status_code == 200
        
        validation_result = safe_response.json()
        assert validation_result["content_safe"] is True
        assert validation_result["safety_score"] >= 0.8
        
        # Test potentially problematic content
        problematic_campaign = {
            "creative": {
                "headline": "Questionable content example",
                "description": "Content that might have issues",
                "image_url": "https://example.com/questionable-image.jpg"
            },
            "budget": {"daily_budget": 30.0}
        }
        
        # Mock content safety API response
        with patch("safety_framework.content_safety.validate_content") as mock_validate:
            mock_validate.return_value = {
                "is_safe": False,
                "safety_score": 0.6,
                "violations": [{"category": "questionable", "severity": "medium"}]
            }
            
            problematic_response = await async_client.post(
                f"/agents/{agent_id}/campaigns/validate",
                json=problematic_campaign
            )
            assert problematic_response.status_code == 400
            
            error_data = problematic_response.json()
            assert "content safety" in error_data["error"].lower()

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_emergency_stop_workflow(
        self,
        async_client: AsyncClient,
        agent_id: str
    ):
        """Test emergency stop workflow end-to-end."""
        
        # Start agent in training
        training_response = await async_client.post(
            f"/agents/{agent_id}/train",
            json={"environment_id": str(uuid.uuid4())}
        )
        assert training_response.status_code == 202
        
        # Simulate safety violation that triggers emergency stop
        violation_data = {
            "violation_type": "critical_safety_breach",
            "severity": "critical",
            "details": {
                "safety_score": 0.1,
                "violation_category": "inappropriate_content"
            }
        }
        
        emergency_response = await async_client.post(
            f"/agents/{agent_id}/emergency-stop",
            json=violation_data
        )
        assert emergency_response.status_code == 200
        
        # Verify agent is stopped
        status_response = await async_client.get(f"/agents/{agent_id}/status")
        assert status_response.status_code == 200
        
        agent_status = status_response.json()
        assert agent_status["status"] == "EMERGENCY_STOPPED"
        assert agent_status["emergency_active"] is True
        
        # Try to perform action while in emergency stop
        action_response = await async_client.post(
            f"/agents/{agent_id}/campaigns",
            json={"creative": {"headline": "Test"}}
        )
        assert action_response.status_code == 403
        
        error_data = action_response.json()
        assert "emergency stop" in error_data["error"].lower()
        
        # Release emergency stop after review
        release_response = await async_client.post(
            f"/agents/{agent_id}/release-emergency-stop",
            json={"reason": "Issue resolved after manual review"}
        )
        assert release_response.status_code == 200
        
        # Verify agent can operate again
        final_status = await async_client.get(f"/agents/{agent_id}/status")
        final_data = final_status.json()
        assert final_data["emergency_active"] is False

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(
        self,
        async_client: AsyncClient,
        mock_agent_config: dict
    ):
        """Test coordination between multiple agents."""
        
        # Create multiple agents
        agents = []
        for i in range(3):
            config = mock_agent_config.copy()
            config["agent_name"] = f"Test Agent {i+1}"
            
            response = await async_client.post("/agents", json=config)
            assert response.status_code == 201
            agents.append(response.json()["agent_id"])
        
        # Create shared environment
        env_response = await async_client.post("/environments", json={
            "type": "simulation",
            "shared": True,
            "max_concurrent_agents": 3
        })
        assert env_response.status_code == 201
        environment_id = env_response.json()["environment_id"]
        
        # Start training for all agents
        training_jobs = []
        for agent_id in agents:
            training_response = await async_client.post(
                f"/agents/{agent_id}/train",
                json={
                    "environment_id": environment_id,
                    "max_episodes": 50,
                    "collaborative_mode": True
                }
            )
            assert training_response.status_code == 202
            training_jobs.append(training_response.json()["job_id"])
        
        # Monitor resource allocation
        resource_response = await async_client.get("/system/resources")
        assert resource_response.status_code == 200
        
        resource_data = resource_response.json()
        assert resource_data["active_agents"] == 3
        assert resource_data["cpu_utilization"] <= 0.8  # Within limits
        assert resource_data["memory_utilization"] <= 0.8
        
        # Verify environment coordination
        env_status = await async_client.get(f"/environments/{environment_id}/status")
        assert env_status.status_code == 200
        
        env_data = env_status.json()
        assert env_data["concurrent_agents"] == 3
        assert env_data["coordination_active"] is True

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_data_pipeline_integrity(
        self,
        async_client: AsyncClient,
        agent_id: str,
        environment_id: str
    ):
        """Test end-to-end data pipeline integrity."""
        
        # Generate training data
        training_sessions = []
        for episode in range(10):
            session_data = {
                "episode": episode,
                "actions": [f"action_{i}" for i in range(20)],
                "rewards": [0.1 * i for i in range(20)],
                "states": [f"state_{i}" for i in range(20)],
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "environment_id": environment_id
                }
            }
            training_sessions.append(session_data)
        
        # Submit training data
        for session in training_sessions:
            data_response = await async_client.post(
                f"/agents/{agent_id}/training-data",
                json=session
            )
            assert data_response.status_code == 201
        
        # Verify data storage in BigQuery
        query_response = await async_client.get(
            f"/agents/{agent_id}/training-history",
            params={"limit": 20}
        )
        assert query_response.status_code == 200
        
        history_data = query_response.json()
        assert len(history_data["episodes"]) == 10
        assert all("actions" in ep for ep in history_data["episodes"])
        assert all("rewards" in ep for ep in history_data["episodes"])
        
        # Test data analytics
        analytics_response = await async_client.get(
            f"/agents/{agent_id}/analytics",
            params={"metric": "reward_progression"}
        )
        assert analytics_response.status_code == 200
        
        analytics = analytics_response.json()
        assert "reward_trend" in analytics
        assert "performance_statistics" in analytics
        assert analytics["data_quality_score"] >= 0.9

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_monitoring_and_alerting(
        self,
        async_client: AsyncClient,
        agent_id: str
    ):
        """Test monitoring and alerting system."""
        
        # Configure monitoring
        monitoring_config = {
            "performance_alerts": True,
            "safety_alerts": True,
            "budget_alerts": True,
            "alert_thresholds": {
                "performance_degradation": 0.2,
                "safety_score_minimum": 0.8,
                "budget_utilization": 0.9
            }
        }
        
        monitor_response = await async_client.post(
            f"/agents/{agent_id}/monitoring",
            json=monitoring_config
        )
        assert monitor_response.status_code == 200
        
        # Simulate performance degradation
        degraded_metrics = [
            {"episode": i, "reward": 0.8 - (i * 0.1)} 
            for i in range(5)  # Declining performance
        ]
        
        for metrics in degraded_metrics:
            metrics_response = await async_client.post(
                f"/agents/{agent_id}/metrics",
                json=metrics
            )
            assert metrics_response.status_code == 200
        
        # Check for alerts
        alerts_response = await async_client.get(f"/agents/{agent_id}/alerts")
        assert alerts_response.status_code == 200
        
        alerts_data = alerts_response.json()
        assert len(alerts_data["active_alerts"]) > 0
        assert any(
            "performance degradation" in alert["message"].lower() 
            for alert in alerts_data["active_alerts"]
        )
        
        # Test alert notification
        notifications_response = await async_client.get(
            f"/agents/{agent_id}/notifications"
        )
        assert notifications_response.status_code == 200
        
        notifications = notifications_response.json()
        assert len(notifications["recent_notifications"]) > 0