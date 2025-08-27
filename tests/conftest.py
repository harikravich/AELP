"""
Global pytest configuration and fixtures for GAELP testing framework.
"""

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Generator
from unittest.mock import Mock, MagicMock, AsyncMock

import pytest
import docker
import redis
from faker import Faker
from google.cloud import bigquery
from httpx import AsyncClient

# Test configuration
TEST_CONFIG = {
    "database": {
        "url": "postgresql://test:test@localhost:5432/gaelp_test",
        "pool_size": 5,
        "timeout": 30
    },
    "redis": {
        "url": "redis://localhost:6379/0",
        "timeout": 5
    },
    "api": {
        "base_url": "http://localhost:8000",
        "timeout": 30
    },
    "bigquery": {
        "project_id": "gaelp-test",
        "dataset": "test_dataset"
    },
    "docker": {
        "network": "gaelp-test-network"
    }
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return TEST_CONFIG

@pytest.fixture(scope="session")
def fake():
    """Faker instance for generating test data."""
    return Faker()

@pytest.fixture(scope="session")
def docker_client():
    """Docker client for container testing."""
    client = docker.from_env()
    yield client
    client.close()

@pytest.fixture(scope="session")
def test_network(docker_client):
    """Create test Docker network."""
    try:
        network = docker_client.networks.get(TEST_CONFIG["docker"]["network"])
    except docker.errors.NotFound:
        network = docker_client.networks.create(
            TEST_CONFIG["docker"]["network"],
            driver="bridge"
        )
    
    yield network
    
    # Cleanup
    try:
        network.remove()
    except docker.errors.APIError:
        pass  # Network might be in use

@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def environment_id():
    """Generate a test environment ID."""
    return str(uuid.uuid4())

@pytest.fixture
def agent_id():
    """Generate a test agent ID."""
    return str(uuid.uuid4())

@pytest.fixture
def campaign_id():
    """Generate a test campaign ID."""
    return str(uuid.uuid4())

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock_redis = Mock(spec=redis.Redis)
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    mock_redis.expire.return_value = True
    return mock_redis

@pytest.fixture
def mock_bigquery():
    """Mock BigQuery client."""
    mock_client = Mock(spec=bigquery.Client)
    mock_dataset = Mock(spec=bigquery.Dataset)
    mock_table = Mock(spec=bigquery.Table)
    mock_job = Mock(spec=bigquery.QueryJob)
    
    mock_client.dataset.return_value = mock_dataset
    mock_client.get_table.return_value = mock_table
    mock_client.query.return_value = mock_job
    mock_job.result.return_value = []
    
    return mock_client

@pytest.fixture
async def async_client():
    """Async HTTP client for API testing."""
    async with AsyncClient(base_url=TEST_CONFIG["api"]["base_url"]) as client:
        yield client

@pytest.fixture
def sample_persona_config():
    """Sample persona configuration for testing."""
    return {
        "demographics": {
            "age_range": [25, 45],
            "gender": ["male", "female"],
            "income_range": [50000, 100000],
            "location": ["US", "CA", "UK"]
        },
        "interests": ["technology", "fitness", "travel"],
        "behavior_patterns": {
            "engagement_likelihood": 0.3,
            "conversion_rate": 0.05,
            "time_to_convert": 48
        }
    }

@pytest.fixture
def sample_ad_campaign():
    """Sample ad campaign configuration for testing."""
    return {
        "creative": {
            "headline": "Discover the Future of Fitness",
            "description": "Transform your workout routine with AI-powered fitness tracking",
            "image_url": "https://cdn.gaelp.dev/ads/fitness-tracker.jpg",
            "call_to_action": "shop_now"
        },
        "targeting": {
            "demographics": {
                "age_range": [25, 45],
                "gender": ["male", "female"],
                "income_range": [40000, 80000]
            },
            "interests": ["fitness", "health", "technology"],
            "behavioral": {
                "purchase_intent": "high",
                "device_usage": ["mobile", "desktop"]
            }
        },
        "budget": {
            "daily_budget": 100.0,
            "total_budget": 1000.0,
            "bid_strategy": "cpc",
            "max_bid": 2.50
        }
    }

@pytest.fixture
def mock_environment_response():
    """Mock environment API response."""
    return {
        "initial_state": {
            "market_context": {
                "competition_level": 0.7,
                "market_saturation": 0.4,
                "seasonal_factor": 1.2
            },
            "audience_size": 1000000,
            "available_budget": 10000.0,
            "environment_type": "simulated"
        },
        "reset_id": str(uuid.uuid4())
    }

@pytest.fixture
def mock_performance_metrics():
    """Mock campaign performance metrics."""
    return {
        "impressions": 10000,
        "clicks": 300,
        "conversions": 15,
        "ctr": 0.03,
        "conversion_rate": 0.05,
        "cost_per_click": 2.33,
        "cost_per_conversion": 46.67,
        "return_on_ad_spend": 3.2,
        "total_spend": 700.0,
        "revenue": 2240.0
    }

@pytest.fixture
def mock_agent_config():
    """Mock agent configuration."""
    return {
        "agent_id": str(uuid.uuid4()),
        "algorithm": "PPO",
        "hyperparameters": {
            "learning_rate": 0.0003,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2
        },
        "environment_config": {
            "max_episodes": 1000,
            "max_steps_per_episode": 100,
            "reward_function": "roas_optimized"
        },
        "safety_config": {
            "max_daily_budget": 1000.0,
            "content_safety_enabled": True,
            "human_approval_required": True
        }
    }

@pytest.fixture
def mock_training_metrics():
    """Mock training metrics data."""
    return {
        "episode_rewards": [0.1, 0.2, 0.3, 0.4, 0.5],
        "episode_lengths": [50, 45, 60, 55, 50],
        "policy_loss": [0.1, 0.08, 0.06, 0.05, 0.04],
        "value_loss": [0.2, 0.18, 0.15, 0.12, 0.10],
        "learning_rate": 0.0003,
        "exploration_rate": 0.1,
        "convergence_score": 0.85
    }

@pytest.fixture
def mock_safety_violations():
    """Mock safety violation events."""
    return [
        {
            "violation_type": "budget_exceeded",
            "severity": "high",
            "timestamp": datetime.utcnow().isoformat(),
            "details": {
                "requested_budget": 1500.0,
                "max_allowed": 1000.0,
                "agent_id": str(uuid.uuid4())
            }
        },
        {
            "violation_type": "inappropriate_content",
            "severity": "critical",
            "timestamp": datetime.utcnow().isoformat(),
            "details": {
                "content": "Potentially harmful ad creative",
                "content_type": "headline",
                "safety_score": 0.2
            }
        }
    ]

@pytest.fixture
def mock_api_responses():
    """Mock external API responses."""
    return {
        "meta_ads": {
            "campaigns": [
                {
                    "id": "123456789",
                    "name": "Test Campaign",
                    "status": "ACTIVE",
                    "objective": "CONVERSIONS",
                    "daily_budget": 10000,
                    "insights": {
                        "impressions": 10000,
                        "clicks": 300,
                        "spend": 700.0
                    }
                }
            ]
        },
        "google_ads": {
            "campaigns": [
                {
                    "resourceName": "customers/123/campaigns/456",
                    "name": "Test Campaign",
                    "status": "ENABLED",
                    "campaignBudget": {
                        "amountMicros": 10000000000
                    },
                    "metrics": {
                        "impressions": 10000,
                        "clicks": 300,
                        "costMicros": 700000000
                    }
                }
            ]
        }
    }

@pytest.fixture
def load_test_config():
    """Configuration for load testing."""
    return {
        "concurrent_users": [10, 50, 100, 200],
        "requests_per_user": 100,
        "ramp_up_time": 60,
        "test_duration": 300,
        "endpoints": [
            "/environments/{env_id}/reset",
            "/environments/{env_id}/step",
            "/environments/{env_id}/render",
            "/agents/{agent_id}/train",
            "/agents/{agent_id}/status"
        ]
    }

@pytest.fixture
def security_test_payloads():
    """Security test payloads for vulnerability testing."""
    return {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM information_schema.tables --"
        ],
        "xss_payloads": [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>"
        ],
        "command_injection": [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami"
        ],
        "path_traversal": [
            "../../etc/passwd",
            "../../../windows/system32/drivers/etc/hosts",
            "....//....//....//etc/passwd"
        ]
    }

@pytest.fixture
def performance_benchmarks():
    """Performance benchmark targets."""
    return {
        "api_response_time": {
            "p50": 100,  # 100ms
            "p95": 500,  # 500ms
            "p99": 1000  # 1s
        },
        "throughput": {
            "min_requests_per_second": 100,
            "target_requests_per_second": 500
        },
        "resource_usage": {
            "max_cpu_percentage": 80,
            "max_memory_mb": 2048,
            "max_disk_io_mb_per_sec": 100
        },
        "training": {
            "max_episode_time": 300,  # 5 minutes
            "convergence_episodes": 1000,
            "min_success_rate": 0.8
        }
    }

# Environment setup and teardown
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests."""
    # Create test directories
    os.makedirs("reports", exist_ok=True)
    os.makedirs("test_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Set test environment variables
    os.environ["GAELP_ENV"] = "test"
    os.environ["GAELP_LOG_LEVEL"] = "DEBUG"
    os.environ["GAELP_TEST_MODE"] = "true"
    
    yield
    
    # Cleanup after tests
    # Note: Actual cleanup would depend on implementation

@pytest.fixture(autouse=True)
def isolate_tests(monkeypatch):
    """Isolate tests by mocking external dependencies."""
    # Mock environment variables
    monkeypatch.setenv("GAELP_ENV", "test")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gaelp-test")
    monkeypatch.setenv("BIGQUERY_DATASET", "test_dataset")
    
    # Mock external service calls
    mock_requests = Mock()
    monkeypatch.setattr("requests.get", mock_requests)
    monkeypatch.setattr("requests.post", mock_requests)
    monkeypatch.setattr("requests.put", mock_requests)
    monkeypatch.setattr("requests.delete", mock_requests)