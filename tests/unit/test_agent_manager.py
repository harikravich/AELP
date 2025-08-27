"""
Unit tests for Agent Manager components.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

import pytest

from agent_manager.core import AgentManager, AgentStatus, TrainingJob
from agent_manager.scheduler import JobScheduler, ResourceManager
from agent_manager.monitoring import AgentMonitor, PerformanceTracker


class TestAgentManager:
    """Test suite for AgentManager core functionality."""

    @pytest.fixture
    def agent_manager(self, mock_redis, mock_bigquery):
        """Create AgentManager instance with mocked dependencies."""
        with patch("agent_manager.core.redis.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = mock_redis
            
            with patch("agent_manager.core.bigquery.Client") as mock_bq_cls:
                mock_bq_cls.return_value = mock_bigquery
                
                manager = AgentManager(
                    redis_url="redis://localhost:6379",
                    bigquery_project="test-project"
                )
                return manager

    @pytest.mark.unit
    async def test_create_agent_success(
        self,
        agent_manager: AgentManager,
        mock_agent_config: dict
    ):
        """Test successful agent creation."""
        agent_id = await agent_manager.create_agent(mock_agent_config)
        
        assert agent_id is not None
        assert isinstance(agent_id, str)
        
        # Verify agent was stored
        agent_manager.redis.set.assert_called()
        
        # Verify agent status is CREATED
        agent_status = await agent_manager.get_agent_status(agent_id)
        assert agent_status == AgentStatus.CREATED

    @pytest.mark.unit
    async def test_create_agent_invalid_config(
        self,
        agent_manager: AgentManager
    ):
        """Test agent creation with invalid configuration."""
        invalid_config = {
            "algorithm": "INVALID_ALGO",
            "hyperparameters": {
                "learning_rate": -0.1  # Invalid negative learning rate
            }
        }
        
        with pytest.raises(ValueError, match="Invalid agent configuration"):
            await agent_manager.create_agent(invalid_config)

    @pytest.mark.unit
    async def test_start_training_success(
        self,
        agent_manager: AgentManager,
        agent_id: str,
        environment_id: str
    ):
        """Test successful training start."""
        # Mock agent exists and is ready
        agent_manager.redis.exists.return_value = True
        agent_manager.redis.get.return_value = '{"status": "CREATED"}'
        
        training_config = {
            "environment_id": environment_id,
            "max_episodes": 1000,
            "max_steps_per_episode": 100,
            "save_frequency": 100
        }
        
        job_id = await agent_manager.start_training(agent_id, training_config)
        
        assert job_id is not None
        assert isinstance(job_id, str)
        
        # Verify training job was created
        agent_manager.redis.set.assert_called()

    @pytest.mark.unit
    async def test_start_training_agent_not_found(
        self,
        agent_manager: AgentManager,
        environment_id: str
    ):
        """Test training start with non-existent agent."""
        non_existent_agent = str(uuid.uuid4())
        agent_manager.redis.exists.return_value = False
        
        training_config = {
            "environment_id": environment_id,
            "max_episodes": 1000
        }
        
        with pytest.raises(ValueError, match="Agent not found"):
            await agent_manager.start_training(non_existent_agent, training_config)

    @pytest.mark.unit
    async def test_start_training_agent_already_training(
        self,
        agent_manager: AgentManager,
        agent_id: str,
        environment_id: str
    ):
        """Test training start when agent is already training."""
        # Mock agent exists and is already training
        agent_manager.redis.exists.return_value = True
        agent_manager.redis.get.return_value = '{"status": "TRAINING"}'
        
        training_config = {
            "environment_id": environment_id,
            "max_episodes": 1000
        }
        
        with pytest.raises(ValueError, match="Agent is already training"):
            await agent_manager.start_training(agent_id, training_config)

    @pytest.mark.unit
    async def test_stop_training_success(
        self,
        agent_manager: AgentManager,
        agent_id: str
    ):
        """Test successful training stop."""
        # Mock agent exists and is training
        agent_manager.redis.exists.return_value = True
        agent_manager.redis.get.return_value = '{"status": "TRAINING", "job_id": "job123"}'
        
        await agent_manager.stop_training(agent_id)
        
        # Verify status was updated
        agent_manager.redis.set.assert_called()

    @pytest.mark.unit
    async def test_get_agent_metrics(
        self,
        agent_manager: AgentManager,
        agent_id: str,
        mock_training_metrics: dict
    ):
        """Test retrieving agent training metrics."""
        # Mock metrics data
        agent_manager.redis.get.return_value = json.dumps(mock_training_metrics)
        
        metrics = await agent_manager.get_agent_metrics(agent_id)
        
        assert metrics is not None
        assert "episode_rewards" in metrics
        assert "policy_loss" in metrics
        assert "convergence_score" in metrics
        assert metrics["convergence_score"] == 0.85

    @pytest.mark.unit
    async def test_update_agent_config(
        self,
        agent_manager: AgentManager,
        agent_id: str
    ):
        """Test updating agent configuration."""
        # Mock agent exists
        agent_manager.redis.exists.return_value = True
        agent_manager.redis.get.return_value = '{"status": "CREATED"}'
        
        config_updates = {
            "hyperparameters": {
                "learning_rate": 0.0001,
                "batch_size": 128
            }
        }
        
        await agent_manager.update_agent_config(agent_id, config_updates)
        
        # Verify config was updated
        agent_manager.redis.set.assert_called()

    @pytest.mark.unit
    async def test_delete_agent_success(
        self,
        agent_manager: AgentManager,
        agent_id: str
    ):
        """Test successful agent deletion."""
        # Mock agent exists and is not training
        agent_manager.redis.exists.return_value = True
        agent_manager.redis.get.return_value = '{"status": "CREATED"}'
        
        await agent_manager.delete_agent(agent_id)
        
        # Verify agent was deleted
        agent_manager.redis.delete.assert_called()

    @pytest.mark.unit
    async def test_delete_agent_while_training(
        self,
        agent_manager: AgentManager,
        agent_id: str
    ):
        """Test agent deletion while training (should fail)."""
        # Mock agent exists and is training
        agent_manager.redis.exists.return_value = True
        agent_manager.redis.get.return_value = '{"status": "TRAINING"}'
        
        with pytest.raises(ValueError, match="Cannot delete agent while training"):
            await agent_manager.delete_agent(agent_id)

    @pytest.mark.unit
    async def test_list_agents(
        self,
        agent_manager: AgentManager
    ):
        """Test listing all agents."""
        # Mock multiple agents
        mock_agent_keys = [
            f"agent:{uuid.uuid4()}",
            f"agent:{uuid.uuid4()}",
            f"agent:{uuid.uuid4()}"
        ]
        agent_manager.redis.keys.return_value = mock_agent_keys
        
        # Mock agent data
        agent_data = {
            "agent_id": "test-id",
            "status": "CREATED",
            "created_at": datetime.utcnow().isoformat()
        }
        agent_manager.redis.get.return_value = json.dumps(agent_data)
        
        agents = await agent_manager.list_agents()
        
        assert len(agents) == len(mock_agent_keys)
        for agent in agents:
            assert "agent_id" in agent
            assert "status" in agent
            assert "created_at" in agent


class TestJobScheduler:
    """Test suite for JobScheduler functionality."""

    @pytest.fixture
    def job_scheduler(self, mock_redis):
        """Create JobScheduler instance with mocked dependencies."""
        with patch("agent_manager.scheduler.redis.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = mock_redis
            
            scheduler = JobScheduler(
                redis_url="redis://localhost:6379",
                max_concurrent_jobs=10
            )
            return scheduler

    @pytest.mark.unit
    async def test_schedule_job_success(
        self,
        job_scheduler: JobScheduler,
        agent_id: str,
        environment_id: str
    ):
        """Test successful job scheduling."""
        job_config = {
            "agent_id": agent_id,
            "environment_id": environment_id,
            "max_episodes": 1000,
            "priority": "normal"
        }
        
        job_id = await job_scheduler.schedule_job(job_config)
        
        assert job_id is not None
        assert isinstance(job_id, str)
        
        # Verify job was queued
        job_scheduler.redis.lpush.assert_called()

    @pytest.mark.unit
    async def test_schedule_high_priority_job(
        self,
        job_scheduler: JobScheduler,
        agent_id: str,
        environment_id: str
    ):
        """Test scheduling high priority job."""
        job_config = {
            "agent_id": agent_id,
            "environment_id": environment_id,
            "max_episodes": 1000,
            "priority": "high"
        }
        
        job_id = await job_scheduler.schedule_job(job_config)
        
        # High priority jobs should use rpush (front of queue)
        job_scheduler.redis.rpush.assert_called()

    @pytest.mark.unit
    async def test_get_next_job(
        self,
        job_scheduler: JobScheduler
    ):
        """Test getting next job from queue."""
        mock_job_data = {
            "job_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "environment_id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat()
        }
        
        job_scheduler.redis.rpop.return_value = json.dumps(mock_job_data)
        
        job = await job_scheduler.get_next_job()
        
        assert job is not None
        assert job["job_id"] == mock_job_data["job_id"]
        assert job["agent_id"] == mock_job_data["agent_id"]

    @pytest.mark.unit
    async def test_cancel_job(
        self,
        job_scheduler: JobScheduler
    ):
        """Test job cancellation."""
        job_id = str(uuid.uuid4())
        
        # Mock job exists
        job_scheduler.redis.exists.return_value = True
        
        await job_scheduler.cancel_job(job_id)
        
        # Verify job was removed
        job_scheduler.redis.delete.assert_called()

    @pytest.mark.unit
    async def test_get_job_status(
        self,
        job_scheduler: JobScheduler
    ):
        """Test getting job status."""
        job_id = str(uuid.uuid4())
        
        mock_job_status = {
            "status": "RUNNING",
            "progress": 0.5,
            "started_at": datetime.utcnow().isoformat()
        }
        
        job_scheduler.redis.get.return_value = json.dumps(mock_job_status)
        
        status = await job_scheduler.get_job_status(job_id)
        
        assert status["status"] == "RUNNING"
        assert status["progress"] == 0.5


class TestResourceManager:
    """Test suite for ResourceManager functionality."""

    @pytest.fixture
    def resource_manager(self):
        """Create ResourceManager instance."""
        return ResourceManager(
            max_cpu_cores=8,
            max_memory_gb=32,
            max_gpu_count=2
        )

    @pytest.mark.unit
    async def test_allocate_resources_success(
        self,
        resource_manager: ResourceManager,
        agent_id: str
    ):
        """Test successful resource allocation."""
        resource_request = {
            "cpu_cores": 2,
            "memory_gb": 4,
            "gpu_count": 1
        }
        
        allocation = await resource_manager.allocate_resources(agent_id, resource_request)
        
        assert allocation is not None
        assert allocation["cpu_cores"] == 2
        assert allocation["memory_gb"] == 4
        assert allocation["gpu_count"] == 1

    @pytest.mark.unit
    async def test_allocate_resources_insufficient(
        self,
        resource_manager: ResourceManager,
        agent_id: str
    ):
        """Test resource allocation with insufficient resources."""
        # Allocate most resources first
        await resource_manager.allocate_resources(
            "agent1", {"cpu_cores": 6, "memory_gb": 28, "gpu_count": 2}
        )
        
        # Try to allocate more than available
        resource_request = {
            "cpu_cores": 4,  # Only 2 remaining
            "memory_gb": 8,  # Only 4 remaining
            "gpu_count": 1   # None remaining
        }
        
        with pytest.raises(ValueError, match="Insufficient resources"):
            await resource_manager.allocate_resources(agent_id, resource_request)

    @pytest.mark.unit
    async def test_release_resources(
        self,
        resource_manager: ResourceManager,
        agent_id: str
    ):
        """Test resource release."""
        # Allocate resources first
        resource_request = {
            "cpu_cores": 2,
            "memory_gb": 4,
            "gpu_count": 1
        }
        
        await resource_manager.allocate_resources(agent_id, resource_request)
        
        # Release resources
        await resource_manager.release_resources(agent_id)
        
        # Verify resources are available again
        available = resource_manager.get_available_resources()
        assert available["cpu_cores"] == 8
        assert available["memory_gb"] == 32
        assert available["gpu_count"] == 2

    @pytest.mark.unit
    async def test_resource_usage_monitoring(
        self,
        resource_manager: ResourceManager
    ):
        """Test resource usage monitoring."""
        # Allocate some resources
        await resource_manager.allocate_resources(
            "agent1", {"cpu_cores": 2, "memory_gb": 4, "gpu_count": 1}
        )
        await resource_manager.allocate_resources(
            "agent2", {"cpu_cores": 1, "memory_gb": 2, "gpu_count": 0}
        )
        
        usage = resource_manager.get_resource_usage()
        
        assert usage["total_allocated"]["cpu_cores"] == 3
        assert usage["total_allocated"]["memory_gb"] == 6
        assert usage["total_allocated"]["gpu_count"] == 1
        
        assert usage["utilization"]["cpu_percentage"] == 37.5  # 3/8
        assert usage["utilization"]["memory_percentage"] == 18.75  # 6/32


class TestAgentMonitor:
    """Test suite for AgentMonitor functionality."""

    @pytest.fixture
    def agent_monitor(self, mock_redis, mock_bigquery):
        """Create AgentMonitor instance with mocked dependencies."""
        with patch("agent_manager.monitoring.redis.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = mock_redis
            
            with patch("agent_manager.monitoring.bigquery.Client") as mock_bq_cls:
                mock_bq_cls.return_value = mock_bigquery
                
                monitor = AgentMonitor(
                    redis_url="redis://localhost:6379",
                    bigquery_project="test-project"
                )
                return monitor

    @pytest.mark.unit
    async def test_track_agent_metrics(
        self,
        agent_monitor: AgentMonitor,
        agent_id: str,
        mock_training_metrics: dict
    ):
        """Test tracking agent training metrics."""
        await agent_monitor.track_metrics(agent_id, mock_training_metrics)
        
        # Verify metrics were stored
        agent_monitor.redis.set.assert_called()
        agent_monitor.bigquery.insert_rows.assert_called()

    @pytest.mark.unit
    async def test_detect_performance_degradation(
        self,
        agent_monitor: AgentMonitor,
        agent_id: str
    ):
        """Test detection of performance degradation."""
        # Mock declining performance metrics
        declining_metrics = [
            {"episode_reward": 0.8, "episode": 100},
            {"episode_reward": 0.7, "episode": 101},
            {"episode_reward": 0.6, "episode": 102},
            {"episode_reward": 0.5, "episode": 103},
            {"episode_reward": 0.4, "episode": 104}
        ]
        
        for metrics in declining_metrics:
            await agent_monitor.track_metrics(agent_id, metrics)
        
        # Check for performance alerts
        alerts = await agent_monitor.check_performance_alerts(agent_id)
        
        assert len(alerts) > 0
        assert any("performance degradation" in alert["message"].lower() for alert in alerts)

    @pytest.mark.unit
    async def test_monitor_training_convergence(
        self,
        agent_monitor: AgentMonitor,
        agent_id: str
    ):
        """Test monitoring training convergence."""
        # Mock stable metrics indicating convergence
        stable_metrics = [
            {"episode_reward": 0.85, "episode": i} 
            for i in range(200, 250)  # 50 episodes of stable performance
        ]
        
        for metrics in stable_metrics:
            await agent_monitor.track_metrics(agent_id, metrics)
        
        convergence_status = await agent_monitor.check_convergence(agent_id)
        
        assert convergence_status["converged"] is True
        assert convergence_status["stability_score"] > 0.9

    @pytest.mark.unit
    async def test_safety_violation_detection(
        self,
        agent_monitor: AgentMonitor,
        agent_id: str,
        mock_safety_violations: list
    ):
        """Test detection of safety violations."""
        for violation in mock_safety_violations:
            await agent_monitor.report_safety_violation(agent_id, violation)
        
        violations = await agent_monitor.get_safety_violations(agent_id)
        
        assert len(violations) == len(mock_safety_violations)
        assert any(v["violation_type"] == "budget_exceeded" for v in violations)
        assert any(v["violation_type"] == "inappropriate_content" for v in violations)

    @pytest.mark.unit
    async def test_resource_utilization_tracking(
        self,
        agent_monitor: AgentMonitor,
        agent_id: str
    ):
        """Test tracking resource utilization."""
        resource_metrics = {
            "cpu_usage_percent": 75.5,
            "memory_usage_mb": 1024,
            "gpu_memory_usage_mb": 2048,
            "network_io_mb": 100,
            "disk_io_mb": 50
        }
        
        await agent_monitor.track_resource_usage(agent_id, resource_metrics)
        
        # Verify resource metrics were stored
        agent_monitor.redis.set.assert_called()
        
        # Get resource usage summary
        summary = await agent_monitor.get_resource_summary(agent_id)
        
        assert "avg_cpu_usage" in summary
        assert "peak_memory_usage" in summary
        assert "total_network_io" in summary