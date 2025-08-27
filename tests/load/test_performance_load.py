"""
Load tests for GAELP performance validation under various loads.
"""

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from statistics import mean, median, stdev
from typing import List, Dict, Any

import pytest
import httpx
from httpx import AsyncClient

from tests.conftest import TEST_CONFIG


class LoadTestMetrics:
    """Collect and analyze load test metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.status_codes: List[int] = []
        self.errors: List[str] = []
        self.start_time: float = 0
        self.end_time: float = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
    
    def add_result(self, response_time: float, status_code: int, error: str = None):
        """Add a test result."""
        self.response_times.append(response_time)
        self.status_codes.append(status_code)
        
        if error:
            self.errors.append(error)
            self.failed_requests += 1
        else:
            self.successful_requests += 1
    
    def start_test(self):
        """Mark test start time."""
        self.start_time = time.time()
    
    def end_test(self):
        """Mark test end time."""
        self.end_time = time.time()
    
    @property
    def duration(self) -> float:
        """Total test duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def total_requests(self) -> int:
        """Total number of requests made."""
        return len(self.response_times)
    
    @property
    def requests_per_second(self) -> float:
        """Requests per second throughput."""
        return self.total_requests / self.duration if self.duration > 0 else 0
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
    
    @property
    def response_time_stats(self) -> Dict[str, float]:
        """Response time statistics."""
        if not self.response_times:
            return {}
        
        sorted_times = sorted(self.response_times)
        return {
            "min": min(sorted_times),
            "max": max(sorted_times),
            "mean": mean(sorted_times),
            "median": median(sorted_times),
            "p95": sorted_times[int(0.95 * len(sorted_times))],
            "p99": sorted_times[int(0.99 * len(sorted_times))],
            "std_dev": stdev(sorted_times) if len(sorted_times) > 1 else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "duration": self.duration,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "requests_per_second": self.requests_per_second,
            "success_rate": self.success_rate,
            "response_times": self.response_time_stats,
            "status_code_distribution": {
                str(code): self.status_codes.count(code)
                for code in set(self.status_codes)
            },
            "error_count": len(self.errors),
            "unique_errors": len(set(self.errors))
        }


class TestAPILoadPerformance:
    """Load tests for API performance."""

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_api_concurrent_requests(
        self,
        load_test_config: dict,
        sample_persona_config: dict,
        performance_benchmarks: dict
    ):
        """Test API performance under concurrent load."""
        
        async def make_request(session: AsyncClient, environment_id: str) -> tuple:
            """Make a single API request and return timing/result."""
            start_time = time.time()
            
            try:
                response = await session.post(
                    f"/environments/{environment_id}/reset",
                    json={
                        "seed": 42,
                        "persona_config": sample_persona_config
                    },
                    timeout=30
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                return response_time, response.status_code, None
                
            except Exception as e:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                return response_time, 0, str(e)
        
        # Test different concurrency levels
        concurrency_levels = [10, 25, 50, 100]
        results = {}
        
        for concurrent_users in concurrency_levels:
            metrics = LoadTestMetrics()
            metrics.start_test()
            
            # Create environment IDs for testing
            environment_ids = [str(uuid.uuid4()) for _ in range(concurrent_users)]
            
            async with AsyncClient(
                base_url=TEST_CONFIG["api"]["base_url"],
                timeout=30
            ) as client:
                
                # Run concurrent requests
                tasks = [
                    make_request(client, env_id) 
                    for env_id in environment_ids
                ]
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect metrics
                for response in responses:
                    if isinstance(response, Exception):
                        metrics.add_result(0, 0, str(response))
                    else:
                        response_time, status_code, error = response
                        metrics.add_result(response_time, status_code, error)
            
            metrics.end_test()
            results[concurrent_users] = metrics.to_dict()
            
            # Validate performance benchmarks
            stats = metrics.response_time_stats
            benchmarks = performance_benchmarks["api_response_time"]
            
            if stats:
                assert stats["p95"] <= benchmarks["p95"], \
                    f"P95 response time {stats['p95']}ms exceeds benchmark {benchmarks['p95']}ms"
                assert stats["p99"] <= benchmarks["p99"], \
                    f"P99 response time {stats['p99']}ms exceeds benchmark {benchmarks['p99']}ms"
            
            assert metrics.success_rate >= 95, \
                f"Success rate {metrics.success_rate}% below 95% threshold"
            
            # Performance should degrade gracefully
            if concurrent_users <= 50:
                assert metrics.requests_per_second >= performance_benchmarks["throughput"]["min_requests_per_second"]
        
        # Generate load test report
        await self._generate_load_test_report("api_concurrent_load", results)

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_training_orchestrator_load(
        self,
        load_test_config: dict,
        mock_agent_config: dict,
        performance_benchmarks: dict
    ):
        """Test training orchestrator under load."""
        
        async def start_training_job(session: AsyncClient) -> tuple:
            """Start a training job and measure response time."""
            start_time = time.time()
            
            try:
                # Create agent
                agent_response = await session.post("/agents", json=mock_agent_config)
                if agent_response.status_code != 201:
                    raise Exception(f"Failed to create agent: {agent_response.status_code}")
                
                agent_id = agent_response.json()["agent_id"]
                
                # Start training
                training_response = await session.post(
                    f"/agents/{agent_id}/train",
                    json={
                        "environment_id": str(uuid.uuid4()),
                        "max_episodes": 10  # Short training for load test
                    }
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                return response_time, training_response.status_code, None
                
            except Exception as e:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                return response_time, 0, str(e)
        
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        # Simulate concurrent training job requests
        concurrent_jobs = 20
        
        async with AsyncClient(
            base_url=TEST_CONFIG["api"]["base_url"],
            timeout=60
        ) as client:
            
            tasks = [start_training_job(client) for _ in range(concurrent_jobs)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for response in responses:
                if isinstance(response, Exception):
                    metrics.add_result(0, 0, str(response))
                else:
                    response_time, status_code, error = response
                    metrics.add_result(response_time, status_code, error)
        
        metrics.end_test()
        
        # Validate training orchestrator performance
        assert metrics.success_rate >= 90, \
            f"Training job success rate {metrics.success_rate}% below 90% threshold"
        
        stats = metrics.response_time_stats
        if stats:
            assert stats["p95"] <= 5000, \
                f"Training job start time P95 {stats['p95']}ms too slow"

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_database_performance_load(
        self,
        load_test_config: dict
    ):
        """Test database performance under load."""
        
        async def database_operation(session: AsyncClient) -> tuple:
            """Perform database-intensive operation."""
            start_time = time.time()
            
            try:
                # Query training history (database-intensive)
                response = await session.get(
                    f"/agents/{uuid.uuid4()}/training-history",
                    params={"limit": 100, "include_metrics": True}
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                return response_time, response.status_code, None
                
            except Exception as e:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                return response_time, 0, str(e)
        
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        # Simulate database load
        concurrent_queries = 50
        
        async with AsyncClient(
            base_url=TEST_CONFIG["api"]["base_url"],
            timeout=30
        ) as client:
            
            tasks = [database_operation(client) for _ in range(concurrent_queries)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for response in responses:
                if isinstance(response, Exception):
                    metrics.add_result(0, 0, str(response))
                else:
                    response_time, status_code, error = response
                    metrics.add_result(response_time, status_code, error)
        
        metrics.end_test()
        
        # Database queries should be reasonably fast even under load
        stats = metrics.response_time_stats
        if stats:
            assert stats["p95"] <= 2000, \
                f"Database query P95 {stats['p95']}ms too slow"
        
        assert metrics.success_rate >= 95, \
            f"Database query success rate {metrics.success_rate}% too low"

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(
        self,
        performance_benchmarks: dict
    ):
        """Test memory usage under various loads."""
        import psutil
        import gc
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        large_datasets = []
        
        try:
            # Create large datasets to simulate training data
            for i in range(100):
                dataset = {
                    "episodes": [
                        {
                            "actions": list(range(1000)),
                            "rewards": [0.1] * 1000,
                            "states": [f"state_{j}" for j in range(1000)]
                        }
                        for _ in range(100)  # 100 episodes per dataset
                    ]
                }
                large_datasets.append(dataset)
                
                # Check memory periodically
                if i % 20 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    # Memory usage should be reasonable
                    max_memory_mb = performance_benchmarks["resource_usage"]["max_memory_mb"]
                    assert current_memory <= max_memory_mb, \
                        f"Memory usage {current_memory}MB exceeds limit {max_memory_mb}MB"
            
            # Final memory check
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory
            
            # Memory increase should be reasonable for the amount of data
            assert memory_increase <= 1000, \
                f"Memory increase {memory_increase}MB seems excessive"
            
        finally:
            # Cleanup
            large_datasets.clear()
            gc.collect()

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_sustained_load_stability(
        self,
        performance_benchmarks: dict
    ):
        """Test system stability under sustained load."""
        
        async def continuous_requests(session: AsyncClient, duration_seconds: int):
            """Make continuous requests for specified duration."""
            end_time = time.time() + duration_seconds
            metrics = LoadTestMetrics()
            
            while time.time() < end_time:
                start_time = time.time()
                
                try:
                    response = await session.get("/health")
                    request_time = (time.time() - start_time) * 1000
                    metrics.add_result(request_time, response.status_code)
                    
                except Exception as e:
                    request_time = (time.time() - start_time) * 1000
                    metrics.add_result(request_time, 0, str(e))
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            return metrics
        
        # Run sustained load for 5 minutes
        test_duration = 300  # 5 minutes
        concurrent_clients = 10
        
        async with AsyncClient(
            base_url=TEST_CONFIG["api"]["base_url"],
            timeout=10
        ) as client:
            
            tasks = [
                continuous_requests(client, test_duration)
                for _ in range(concurrent_clients)
            ]
            
            all_metrics = await asyncio.gather(*tasks)
        
        # Analyze stability over time
        total_requests = sum(m.total_requests for m in all_metrics)
        total_errors = sum(m.failed_requests for m in all_metrics)
        overall_success_rate = ((total_requests - total_errors) / total_requests * 100) if total_requests > 0 else 0
        
        # System should remain stable over time
        assert overall_success_rate >= 99, \
            f"Sustained load success rate {overall_success_rate}% indicates instability"
        
        # Response times should not degrade significantly over time
        for metrics in all_metrics:
            stats = metrics.response_time_stats
            if stats:
                assert stats["p95"] <= 500, \
                    f"Sustained load P95 response time {stats['p95']}ms indicates degradation"

    async def _generate_load_test_report(self, test_name: str, results: Dict[str, Any]):
        """Generate load test report."""
        report = {
            "test_name": test_name,
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
            "summary": {
                "max_concurrent_users": max(results.keys()) if results else 0,
                "best_throughput": max(
                    (r["requests_per_second"] for r in results.values()),
                    default=0
                ),
                "worst_p95_response_time": max(
                    (r["response_times"].get("p95", 0) for r in results.values()),
                    default=0
                )
            }
        }
        
        # Save report to file
        import os
        os.makedirs("reports/load_tests", exist_ok=True)
        
        with open(f"reports/load_tests/{test_name}_{int(time.time())}.json", "w") as f:
            json.dump(report, f, indent=2)


class TestResourceLoadLimits:
    """Test resource limits under load."""

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_cpu_usage_under_load(
        self,
        performance_benchmarks: dict
    ):
        """Test CPU usage under computational load."""
        import psutil
        import multiprocessing
        
        def cpu_intensive_task():
            """CPU-intensive task to simulate training load."""
            # Simulate neural network computation
            for i in range(1000000):
                result = sum(j ** 2 for j in range(100))
            return result
        
        # Monitor CPU usage during load
        initial_cpu = psutil.cpu_percent(interval=1)
        
        # Start CPU-intensive tasks
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [
                executor.submit(cpu_intensive_task)
                for _ in range(multiprocessing.cpu_count() * 2)
            ]
            
            # Monitor CPU during execution
            cpu_readings = []
            for _ in range(10):  # 10 second monitoring
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_readings.append(cpu_percent)
            
            # Wait for tasks to complete
            for future in as_completed(futures):
                future.result()
        
        # Validate CPU usage
        max_cpu = max(cpu_readings)
        avg_cpu = mean(cpu_readings)
        
        max_cpu_limit = performance_benchmarks["resource_usage"]["max_cpu_percentage"]
        
        assert max_cpu <= max_cpu_limit, \
            f"Peak CPU usage {max_cpu}% exceeds limit {max_cpu_limit}%"

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_disk_io_under_load(
        self,
        performance_benchmarks: dict
    ):
        """Test disk I/O performance under load."""
        import tempfile
        import os
        
        def disk_intensive_task(file_path: str, size_mb: int):
            """Disk-intensive task to simulate data storage."""
            data = b"0" * (1024 * 1024)  # 1MB of data
            
            with open(file_path, "wb") as f:
                for _ in range(size_mb):
                    f.write(data)
            
            # Read back the data
            with open(file_path, "rb") as f:
                while f.read(1024 * 1024):
                    pass
            
            os.remove(file_path)
        
        # Create temporary files for testing
        temp_files = [
            tempfile.mktemp(suffix=f"_load_test_{i}.dat")
            for i in range(10)
        ]
        
        start_time = time.time()
        
        # Run disk I/O tasks concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(disk_intensive_task, file_path, 10)  # 10MB per file
                for file_path in temp_files
            ]
            
            for future in as_completed(futures):
                future.result()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate I/O throughput
        total_data_mb = len(temp_files) * 10 * 2  # 10MB write + 10MB read per file
        io_throughput = total_data_mb / duration
        
        max_io_limit = performance_benchmarks["resource_usage"]["max_disk_io_mb_per_sec"]
        
        # I/O throughput should be reasonable
        assert io_throughput <= max_io_limit, \
            f"Disk I/O throughput {io_throughput}MB/s exceeds limit {max_io_limit}MB/s"

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_network_load_handling(
        self,
        load_test_config: dict
    ):
        """Test network load handling capabilities."""
        
        async def network_intensive_request(session: AsyncClient, size_kb: int):
            """Make network request with specified payload size."""
            # Create large payload
            payload = {
                "data": "x" * (size_kb * 1024),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            start_time = time.time()
            
            try:
                response = await session.post("/test/large-payload", json=payload)
                end_time = time.time()
                
                return (end_time - start_time) * 1000, response.status_code, None
                
            except Exception as e:
                end_time = time.time()
                return (end_time - start_time) * 1000, 0, str(e)
        
        # Test with various payload sizes
        payload_sizes = [1, 10, 100, 500]  # KB
        
        for size_kb in payload_sizes:
            metrics = LoadTestMetrics()
            metrics.start_test()
            
            async with AsyncClient(
                base_url=TEST_CONFIG["api"]["base_url"],
                timeout=60
            ) as client:
                
                # Send multiple large requests concurrently
                tasks = [
                    network_intensive_request(client, size_kb)
                    for _ in range(10)
                ]
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for response in responses:
                    if isinstance(response, Exception):
                        metrics.add_result(0, 0, str(response))
                    else:
                        response_time, status_code, error = response
                        metrics.add_result(response_time, status_code, error)
            
            metrics.end_test()
            
            # Network handling should be efficient
            assert metrics.success_rate >= 90, \
                f"Network load success rate {metrics.success_rate}% too low for {size_kb}KB payloads"
            
            stats = metrics.response_time_stats
            if stats:
                # Response time should scale reasonably with payload size
                max_expected_time = size_kb * 10  # 10ms per KB is reasonable
                assert stats["p95"] <= max_expected_time, \
                    f"Network response time {stats['p95']}ms too slow for {size_kb}KB payload"