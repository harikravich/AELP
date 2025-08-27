"""
Main test runner for GAELP testing framework.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pytest
from tests.utils.test_reporting import TestReporter, TestMetrics, TestSuite, CIIntegration


class GAELPTestRunner:
    """Main test runner for GAELP platform."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.reporter = TestReporter(self.config.get("output_dir", "reports"))
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load test configuration."""
        default_config = {
            "output_dir": "reports",
            "test_types": ["unit", "integration", "e2e", "load", "security"],
            "parallel_workers": 4,
            "timeout": 3600,  # 1 hour
            "coverage_threshold": 90.0,
            "success_rate_threshold": 95.0,
            "environments": {
                "test": {
                    "database_url": "postgresql://test:test@localhost:5432/gaelp_test",
                    "redis_url": "redis://localhost:6379/0",
                    "api_base_url": "http://localhost:8000"
                },
                "staging": {
                    "database_url": "postgresql://staging:staging@staging-db:5432/gaelp_staging",
                    "redis_url": "redis://staging-redis:6379/0",
                    "api_base_url": "https://staging-api.gaelp.dev"
                }
            },
            "notifications": {
                "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
                "github_token": os.getenv("GITHUB_TOKEN")
            },
            "quality_gates": {
                "enabled": True,
                "coverage_threshold": 80.0,
                "success_rate_threshold": 95.0,
                "max_duration": 3600,
                "max_failed_tests": 5
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def run_tests(
        self, 
        test_types: Optional[List[str]] = None,
        test_paths: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        environment: str = "test"
    ) -> Dict[str, Any]:
        """Run tests and generate reports."""
        
        print("🚀 Starting GAELP Test Suite")
        print(f"Environment: {environment}")
        
        # Set environment variables
        self._setup_environment(environment)
        
        # Determine which tests to run
        test_types = test_types or self.config["test_types"]
        
        results = {}
        total_start_time = time.time()
        
        for test_type in test_types:
            print(f"\n📋 Running {test_type} tests...")
            
            test_result = await self._run_test_type(
                test_type, 
                test_paths, 
                markers,
                environment
            )
            
            results[test_type] = test_result
            
            # Add results to reporter
            if test_result.get("suite"):
                self.reporter.add_suite_result(test_result["suite"])
        
        total_duration = time.time() - total_start_time
        
        print(f"\n⏱️  Total test execution time: {total_duration:.2f}s")
        
        # Generate comprehensive reports
        print("\n📊 Generating test reports...")
        
        html_report = self.reporter.generate_html_report()
        json_report = self.reporter.generate_json_report()
        junit_report = self.reporter.generate_junit_xml()
        
        print(f"✅ HTML Report: {html_report}")
        print(f"✅ JSON Report: {json_report}")
        print(f"✅ JUnit Report: {junit_report}")
        
        # Save historical data
        self.reporter.save_historical_data()
        
        # Check quality gates
        with open(json_report, 'r') as f:
            report_data = json.load(f)
        
        quality_gates = CIIntegration.check_quality_gates(report_data)
        
        if self.config["quality_gates"]["enabled"]:
            print("\n🚪 Checking Quality Gates...")
            
            for gate_name, gate_result in quality_gates["gates"].items():
                status = "✅ PASS" if gate_result["passed"] else "❌ FAIL"
                print(f"  {gate_name}: {status} ({gate_result['value']} vs {gate_result['threshold']})")
            
            if not quality_gates["passed"]:
                print("\n⚠️  Quality gates failed:")
                for recommendation in quality_gates["recommendations"]:
                    print(f"  - {recommendation}")
        
        # Send notifications
        await self._send_notifications(report_data, quality_gates)
        
        # Return summary
        return {
            "success": quality_gates["passed"] if self.config["quality_gates"]["enabled"] else True,
            "results": results,
            "reports": {
                "html": html_report,
                "json": json_report,
                "junit": junit_report
            },
            "quality_gates": quality_gates,
            "total_duration": total_duration
        }
    
    async def _run_test_type(
        self, 
        test_type: str, 
        test_paths: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        environment: str = "test"
    ) -> Dict[str, Any]:
        """Run a specific type of tests."""
        
        start_time = time.time()
        
        # Build pytest arguments
        pytest_args = []
        
        # Test paths
        if test_paths:
            pytest_args.extend(test_paths)
        else:
            pytest_args.append(f"tests/{test_type}")
        
        # Markers
        test_markers = [f"@pytest.mark.{test_type}"]
        if markers:
            test_markers.extend(markers)
        
        if test_markers:
            pytest_args.extend(["-m", " and ".join(test_markers)])
        
        # Output options
        pytest_args.extend([
            "-v",
            "--tb=short",
            f"--junitxml=reports/{test_type}_junit.xml",
            f"--html=reports/{test_type}_report.html",
            "--self-contained-html"
        ])
        
        # Coverage options
        if test_type in ["unit", "integration"]:
            pytest_args.extend([
                "--cov=.",
                f"--cov-report=html:reports/{test_type}_coverage",
                f"--cov-report=xml:reports/{test_type}_coverage.xml",
                "--cov-branch"
            ])
        
        # Parallel execution for appropriate test types
        if test_type in ["unit", "integration"] and self.config.get("parallel_workers", 1) > 1:
            pytest_args.extend(["-n", str(self.config["parallel_workers"])])
        
        # Timeout
        pytest_args.extend(["--timeout", str(self.config.get("timeout", 3600))])
        
        # Special configurations per test type
        if test_type == "load":
            pytest_args.extend(["--disable-warnings"])
        elif test_type == "security":
            pytest_args.extend(["--tb=long"])  # More detailed output for security issues
        
        print(f"🔧 Running pytest with args: {' '.join(pytest_args)}")
        
        # Run pytest
        exit_code = pytest.main(pytest_args)
        
        duration = time.time() - start_time
        
        # Parse results
        test_result = self._parse_test_results(test_type, exit_code, duration)
        
        print(f"✅ {test_type} tests completed in {duration:.2f}s")
        print(f"   Status: {'PASSED' if exit_code == 0 else 'FAILED'}")
        
        return test_result
    
    def _parse_test_results(self, test_type: str, exit_code: int, duration: float) -> Dict[str, Any]:
        """Parse test results from pytest execution."""
        
        # This would typically parse the JUnit XML or pytest JSON output
        # For now, we'll create a basic result structure
        
        # Try to read JUnit XML for detailed results
        junit_path = Path(f"reports/{test_type}_junit.xml")
        
        if junit_path.exists():
            # Parse JUnit XML (simplified)
            import xml.etree.ElementTree as ET
            
            try:
                tree = ET.parse(junit_path)
                root = tree.getroot()
                
                testsuite = root.find('.//testsuite')
                if testsuite is not None:
                    total_tests = int(testsuite.get('tests', 0))
                    failures = int(testsuite.get('failures', 0))
                    errors = int(testsuite.get('errors', 0))
                    skipped = int(testsuite.get('skipped', 0))
                    passed_tests = total_tests - failures - errors - skipped
                    
                    # Extract individual test results
                    tests = []
                    for testcase in testsuite.findall('testcase'):
                        test_name = testcase.get('name', 'unknown')
                        test_duration = float(testcase.get('time', 0))
                        
                        # Determine status
                        if testcase.find('failure') is not None:
                            status = "failed"
                            error_msg = testcase.find('failure').get('message', '')
                        elif testcase.find('error') is not None:
                            status = "failed"
                            error_msg = testcase.find('error').get('message', '')
                        elif testcase.find('skipped') is not None:
                            status = "skipped"
                            error_msg = None
                        else:
                            status = "passed"
                            error_msg = None
                        
                        test_metric = TestMetrics(
                            test_name=test_name,
                            test_type=test_type,
                            status=status,
                            duration=test_duration,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                            error_message=error_msg
                        )
                        
                        tests.append(test_metric)
                        self.reporter.add_test_result(test_metric)
                    
                    # Try to get coverage data
                    coverage = self._get_coverage_data(test_type)
                    
                    suite = TestSuite(
                        suite_name=test_type,
                        total_tests=total_tests,
                        passed_tests=passed_tests,
                        failed_tests=failures + errors,
                        skipped_tests=skipped,
                        total_duration=duration,
                        coverage=coverage,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        tests=tests
                    )
                    
                    return {
                        "suite": suite,
                        "exit_code": exit_code,
                        "success": exit_code == 0
                    }
                    
            except ET.ParseError as e:
                print(f"⚠️  Could not parse JUnit XML: {e}")
        
        # Fallback result if XML parsing fails
        success = exit_code == 0
        
        suite = TestSuite(
            suite_name=test_type,
            total_tests=1,
            passed_tests=1 if success else 0,
            failed_tests=0 if success else 1,
            skipped_tests=0,
            total_duration=duration,
            coverage=0.0,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            tests=[]
        )
        
        return {
            "suite": suite,
            "exit_code": exit_code,
            "success": success
        }
    
    def _get_coverage_data(self, test_type: str) -> float:
        """Extract coverage data from coverage report."""
        coverage_xml = Path(f"reports/{test_type}_coverage.xml")
        
        if coverage_xml.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_xml)
                root = tree.getroot()
                
                coverage_elem = root.find('.//coverage')
                if coverage_elem is not None:
                    return float(coverage_elem.get('line-rate', 0)) * 100
                    
            except (ET.ParseError, ValueError):
                pass
        
        return 0.0
    
    def _setup_environment(self, environment: str):
        """Set up environment variables for testing."""
        env_config = self.config["environments"].get(environment, {})
        
        for key, value in env_config.items():
            os.environ[key.upper()] = str(value)
        
        # Set additional test environment variables
        os.environ["GAELP_ENV"] = environment
        os.environ["GAELP_TEST_MODE"] = "true"
        os.environ["PYTHONPATH"] = str(Path.cwd())
    
    async def _send_notifications(self, report_data: Dict[str, Any], quality_gates: Dict[str, Any]):
        """Send notifications to configured channels."""
        notifications_config = self.config.get("notifications", {})
        
        # Slack notification
        slack_webhook = notifications_config.get("slack_webhook")
        if slack_webhook:
            try:
                import httpx
                
                slack_payload = CIIntegration.generate_slack_notification(report_data, quality_gates)
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(slack_webhook, json=slack_payload)
                    if response.status_code == 200:
                        print("✅ Slack notification sent")
                    else:
                        print(f"⚠️  Failed to send Slack notification: {response.status_code}")
                        
            except Exception as e:
                print(f"⚠️  Error sending Slack notification: {e}")
        
        # GitHub status check
        github_token = notifications_config.get("github_token")
        if github_token and os.getenv("GITHUB_REPOSITORY"):
            try:
                import httpx
                
                github_payload = CIIntegration.generate_github_status(quality_gates)
                
                repo = os.getenv("GITHUB_REPOSITORY")
                sha = os.getenv("GITHUB_SHA", "main")
                
                url = f"https://api.github.com/repos/{repo}/statuses/{sha}"
                headers = {
                    "Authorization": f"token {github_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=github_payload, headers=headers)
                    if response.status_code == 201:
                        print("✅ GitHub status check posted")
                    else:
                        print(f"⚠️  Failed to post GitHub status: {response.status_code}")
                        
            except Exception as e:
                print(f"⚠️  Error posting GitHub status: {e}")


async def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="GAELP Test Runner")
    parser.add_argument("--config", help="Path to test configuration file")
    parser.add_argument("--types", nargs="+", choices=["unit", "integration", "e2e", "load", "security"], 
                       help="Test types to run")
    parser.add_argument("--paths", nargs="+", help="Specific test paths to run")
    parser.add_argument("--markers", nargs="+", help="Additional pytest markers")
    parser.add_argument("--environment", default="test", help="Test environment")
    parser.add_argument("--no-quality-gates", action="store_true", help="Disable quality gates")
    
    args = parser.parse_args()
    
    runner = GAELPTestRunner(args.config)
    
    if args.no_quality_gates:
        runner.config["quality_gates"]["enabled"] = False
    
    try:
        results = await runner.run_tests(
            test_types=args.types,
            test_paths=args.paths,
            markers=args.markers,
            environment=args.environment
        )
        
        if results["success"]:
            print("\n🎉 All tests passed and quality gates satisfied!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed or quality gates not satisfied!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())