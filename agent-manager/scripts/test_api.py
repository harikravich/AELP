#!/usr/bin/env python3
"""
Test script for Agent Manager API
"""
import asyncio
import aiohttp
import json
from typing import Dict, Any


class AgentManagerTester:
    """Test client for Agent Manager API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", token: str = None):
        self.base_url = base_url
        self.token = token
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    async def test_health(self) -> bool:
        """Test health endpoint"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    data = await response.json()
                    print(f"Health check: {data}")
                    return response.status == 200
            except Exception as e:
                print(f"Health check failed: {e}")
                return False
    
    async def create_test_agent(self) -> Dict[str, Any]:
        """Create a test agent"""
        agent_data = {
            "name": "test-agent-001",
            "type": "simulation",
            "version": "1.0.0",
            "docker_image": "python:3.11-slim",
            "description": "Test agent for API validation",
            "config": {
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32
                },
                "environment_selection": "simulation",
                "performance_thresholds": {
                    "accuracy": 0.8
                }
            },
            "resource_requirements": {
                "cpu": "1",
                "memory": "2Gi"
            },
            "budget_limit": 100.0
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/v1/agents",
                    headers=self.headers,
                    json=agent_data
                ) as response:
                    if response.status == 201:
                        data = await response.json()
                        print(f"Created agent: {data['name']} (ID: {data['id']})")
                        return data
                    else:
                        error = await response.text()
                        print(f"Failed to create agent: {response.status} - {error}")
                        return {}
            except Exception as e:
                print(f"Error creating agent: {e}")
                return {}
    
    async def list_agents(self) -> list:
        """List all agents"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/api/v1/agents",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"Found {len(data)} agents")
                        for agent in data:
                            print(f"  - {agent['name']} ({agent['type']}) - {agent['status']}")
                        return data
                    else:
                        error = await response.text()
                        print(f"Failed to list agents: {response.status} - {error}")
                        return []
            except Exception as e:
                print(f"Error listing agents: {e}")
                return []
    
    async def create_training_job(self, agent_id: int) -> Dict[str, Any]:
        """Create a training job for an agent"""
        job_data = {
            "agent_id": agent_id,
            "name": f"test-job-{agent_id}",
            "priority": 5,
            "hyperparameters": {
                "epochs": 10,
                "learning_rate": 0.001
            },
            "training_config": {
                "dataset": "test_dataset",
                "validation_split": 0.2
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/v1/jobs",
                    headers=self.headers,
                    json=job_data
                ) as response:
                    if response.status == 201:
                        data = await response.json()
                        print(f"Created job: {data['name']} (ID: {data['id']})")
                        return data
                    else:
                        error = await response.text()
                        print(f"Failed to create job: {response.status} - {error}")
                        return {}
            except Exception as e:
                print(f"Error creating job: {e}")
                return {}
    
    async def list_jobs(self) -> list:
        """List all training jobs"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/api/v1/jobs",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"Found {len(data)} jobs")
                        for job in data:
                            print(f"  - {job['name']} (Agent: {job['agent_id']}) - {job['status']}")
                        return data
                    else:
                        error = await response.text()
                        print(f"Failed to list jobs: {response.status} - {error}")
                        return []
            except Exception as e:
                print(f"Error listing jobs: {e}")
                return []
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/api/v1/status",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print("System Status:")
                        print(f"  - Queued jobs: {data['job_counts']['queued']}")
                        print(f"  - Running jobs: {data['job_counts']['running']}")
                        print(f"  - Total agents: {data['agent_counts']['total']}")
                        print(f"  - Active agents: {data['agent_counts']['active']}")
                        return data
                    else:
                        error = await response.text()
                        print(f"Failed to get status: {response.status} - {error}")
                        return {}
            except Exception as e:
                print(f"Error getting status: {e}")
                return {}
    
    async def run_full_test(self):
        """Run complete test suite"""
        print("=" * 60)
        print("GAELP Agent Manager API Test")
        print("=" * 60)
        
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        health_ok = await self.test_health()
        if not health_ok:
            print("❌ Health check failed - aborting tests")
            return
        print("✅ Health check passed")
        
        # List existing agents
        print("\n2. Listing existing agents...")
        agents = await self.list_agents()
        
        # Create test agent
        print("\n3. Creating test agent...")
        agent = await self.create_test_agent()
        if not agent:
            print("❌ Failed to create agent - skipping job tests")
            return
        print("✅ Agent created successfully")
        
        # Create training job
        print("\n4. Creating training job...")
        job = await self.create_training_job(agent['id'])
        if job:
            print("✅ Training job created successfully")
        else:
            print("❌ Failed to create training job")
        
        # List jobs
        print("\n5. Listing training jobs...")
        jobs = await self.list_jobs()
        
        # Get system status
        print("\n6. Getting system status...")
        status = await self.get_system_status()
        if status:
            print("✅ System status retrieved successfully")
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)


def create_mock_token() -> str:
    """Create a mock JWT token for testing"""
    import jwt
    from datetime import datetime, timedelta
    
    payload = {
        "sub": "test-user",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "admin": False
    }
    
    # Use the same secret as in the development config
    secret = "your-secret-key-change-in-production"
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token


async def main():
    """Main test function"""
    print("Creating test token...")
    token = create_mock_token()
    print(f"Token: {token[:50]}...")
    
    # Create tester
    tester = AgentManagerTester(token=token)
    
    # Run tests
    await tester.run_full_test()


if __name__ == "__main__":
    asyncio.run(main())