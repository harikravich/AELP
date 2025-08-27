"""
Test reporting and metrics collection for GAELP testing framework.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template


@dataclass
class TestMetrics:
    """Test execution metrics."""
    test_name: str
    test_type: str  # unit, integration, e2e, load, security
    status: str     # passed, failed, skipped
    duration: float
    timestamp: str
    error_message: Optional[str] = None
    coverage: Optional[float] = None
    assertions: Optional[int] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


@dataclass
class TestSuite:
    """Test suite results."""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration: float
    coverage: float
    timestamp: str
    tests: List[TestMetrics]


class TestReporter:
    """Generate comprehensive test reports."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[TestMetrics] = []
        self.suites: List[TestSuite] = []
    
    def add_test_result(self, metrics: TestMetrics):
        """Add test result to reporter."""
        self.metrics.append(metrics)
    
    def add_suite_result(self, suite: TestSuite):
        """Add test suite result."""
        self.suites.append(suite)
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report."""
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>GAELP Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .summary { display: flex; gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; font-size: 0.9em; }
        .test-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        .test-table th, .test-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        .test-table th { background: #f8f9fa; }
        .status-passed { color: #28a745; font-weight: bold; }
        .status-failed { color: #dc3545; font-weight: bold; }
        .status-skipped { color: #ffc107; font-weight: bold; }
        .chart-container { margin: 20px 0; text-align: center; }
        .error-details { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; border-radius: 5px; margin: 5px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>GAELP Test Report</h1>
        <p>Generated on: {{ report_time }}</p>
        <p>Test Environment: {{ environment }}</p>
    </div>
    
    <div class="summary">
        <div class="metric-card">
            <div class="metric-value">{{ total_tests }}</div>
            <div class="metric-label">Total Tests</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ passed_tests }}</div>
            <div class="metric-label">Passed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ failed_tests }}</div>
            <div class="metric-label">Failed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ success_rate }}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ total_duration }}s</div>
            <div class="metric-label">Total Duration</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ coverage }}%</div>
            <div class="metric-label">Code Coverage</div>
        </div>
    </div>
    
    <h2>Test Suite Results</h2>
    <table class="test-table">
        <thead>
            <tr>
                <th>Suite Name</th>
                <th>Total</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Skipped</th>
                <th>Duration</th>
                <th>Coverage</th>
            </tr>
        </thead>
        <tbody>
            {% for suite in suites %}
            <tr>
                <td>{{ suite.suite_name }}</td>
                <td>{{ suite.total_tests }}</td>
                <td class="status-passed">{{ suite.passed_tests }}</td>
                <td class="status-failed">{{ suite.failed_tests }}</td>
                <td class="status-skipped">{{ suite.skipped_tests }}</td>
                <td>{{ "%.2f"|format(suite.total_duration) }}s</td>
                <td>{{ "%.1f"|format(suite.coverage) }}%</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <h2>Individual Test Results</h2>
    <table class="test-table">
        <thead>
            <tr>
                <th>Test Name</th>
                <th>Type</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Timestamp</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>
            {% for test in tests %}
            <tr>
                <td>{{ test.test_name }}</td>
                <td>{{ test.test_type }}</td>
                <td class="status-{{ test.status }}">{{ test.status|upper }}</td>
                <td>{{ "%.3f"|format(test.duration) }}s</td>
                <td>{{ test.timestamp }}</td>
                <td>
                    {% if test.error_message %}
                    <div class="error-details">{{ test.error_message }}</div>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <h2>Performance Metrics</h2>
    <div class="chart-container">
        <img src="test_duration_chart.png" alt="Test Duration Chart" style="max-width: 100%;">
    </div>
    
    <h2>Coverage Details</h2>
    <div class="chart-container">
        <img src="coverage_chart.png" alt="Coverage Chart" style="max-width: 100%;">
    </div>
    
    <h2>Test Trends</h2>
    <div class="chart-container">
        <img src="test_trends.png" alt="Test Trends" style="max-width: 100%;">
    </div>
</body>
</html>
        """
        
        # Calculate summary statistics
        total_tests = len(self.metrics)
        passed_tests = len([t for t in self.metrics if t.status == "passed"])
        failed_tests = len([t for t in self.metrics if t.status == "failed"])
        skipped_tests = len([t for t in self.metrics if t.status == "skipped"])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        total_duration = sum(t.duration for t in self.metrics)
        coverage = sum(s.coverage for s in self.suites) / len(self.suites) if self.suites else 0
        
        template = Template(template_str)
        html_content = template.render(
            report_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            environment=os.getenv("GAELP_ENV", "test"),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=round(success_rate, 1),
            total_duration=round(total_duration, 2),
            coverage=round(coverage, 1),
            suites=self.suites,
            tests=self.metrics
        )
        
        # Generate charts
        self._generate_charts()
        
        # Save HTML report
        report_path = self.output_dir / "test_report.html"
        with open(report_path, "w") as f:
            f.write(html_content)
        
        return str(report_path)
    
    def generate_json_report(self) -> str:
        """Generate JSON report for CI/CD integration."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "environment": os.getenv("GAELP_ENV", "test"),
            "summary": {
                "total_tests": len(self.metrics),
                "passed_tests": len([t for t in self.metrics if t.status == "passed"]),
                "failed_tests": len([t for t in self.metrics if t.status == "failed"]),
                "skipped_tests": len([t for t in self.metrics if t.status == "skipped"]),
                "total_duration": sum(t.duration for t in self.metrics),
                "average_duration": sum(t.duration for t in self.metrics) / len(self.metrics) if self.metrics else 0,
                "coverage": sum(s.coverage for s in self.suites) / len(self.suites) if self.suites else 0
            },
            "suites": [asdict(suite) for suite in self.suites],
            "tests": [asdict(test) for test in self.metrics],
            "performance": self._calculate_performance_metrics(),
            "trends": self._calculate_trends()
        }
        
        report_path = self.output_dir / "test_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        
        return str(report_path)
    
    def generate_junit_xml(self) -> str:
        """Generate JUnit XML report for CI systems."""
        junit_template = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    {% for suite in suites %}
    <testsuite name="{{ suite.suite_name }}" 
               tests="{{ suite.total_tests }}" 
               failures="{{ suite.failed_tests }}" 
               skipped="{{ suite.skipped_tests }}" 
               time="{{ suite.total_duration }}">
        {% for test in suite.tests %}
        <testcase name="{{ test.test_name }}" 
                  classname="{{ test.test_type }}" 
                  time="{{ test.duration }}">
            {% if test.status == "failed" %}
            <failure message="Test failed">{{ test.error_message or "No error message" }}</failure>
            {% elif test.status == "skipped" %}
            <skipped/>
            {% endif %}
        </testcase>
        {% endfor %}
    </testsuite>
    {% endfor %}
</testsuites>"""
        
        template = Template(junit_template)
        xml_content = template.render(suites=self.suites)
        
        report_path = self.output_dir / "junit_report.xml"
        with open(report_path, "w") as f:
            f.write(xml_content)
        
        return str(report_path)
    
    def _generate_charts(self):
        """Generate visualization charts."""
        if not self.metrics:
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Test duration chart
        self._generate_duration_chart()
        
        # Coverage chart
        self._generate_coverage_chart()
        
        # Test trends chart
        self._generate_trends_chart()
    
    def _generate_duration_chart(self):
        """Generate test duration chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Duration by test type
        df = pd.DataFrame([asdict(m) for m in self.metrics])
        duration_by_type = df.groupby('test_type')['duration'].mean()
        
        ax1.bar(duration_by_type.index, duration_by_type.values)
        ax1.set_title('Average Test Duration by Type')
        ax1.set_ylabel('Duration (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Duration distribution
        ax2.hist(df['duration'], bins=20, alpha=0.7)
        ax2.set_title('Test Duration Distribution')
        ax2.set_xlabel('Duration (seconds)')
        ax2.set_ylabel('Number of Tests')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "test_duration_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_coverage_chart(self):
        """Generate coverage chart."""
        if not self.suites:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        suite_names = [s.suite_name for s in self.suites]
        coverages = [s.coverage for s in self.suites]
        
        bars = ax.bar(suite_names, coverages, color=['#28a745' if c >= 80 else '#ffc107' if c >= 60 else '#dc3545' for c in coverages])
        
        ax.set_title('Code Coverage by Test Suite')
        ax.set_ylabel('Coverage (%)')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        
        # Add coverage percentage labels on bars
        for bar, coverage in zip(bars, coverages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{coverage:.1f}%', ha='center', va='bottom')
        
        # Add reference line for 80% coverage
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Target')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "coverage_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_trends_chart(self):
        """Generate test trends chart."""
        # Load historical data if available
        historical_data = self._load_historical_data()
        
        if not historical_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Test success rate trend
        dates = [d['date'] for d in historical_data]
        success_rates = [d['success_rate'] for d in historical_data]
        
        ax1.plot(dates, success_rates, marker='o', linewidth=2)
        ax1.set_title('Test Success Rate Trend')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Coverage trend
        coverages = [d['coverage'] for d in historical_data]
        ax2.plot(dates, coverages, marker='s', linewidth=2, color='orange')
        ax2.set_title('Code Coverage Trend')
        ax2.set_ylabel('Coverage (%)')
        ax2.set_xlabel('Date')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "test_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.metrics:
            return {}
        
        durations = [m.duration for m in self.metrics]
        
        return {
            "total_duration": sum(durations),
            "average_duration": sum(durations) / len(durations),
            "median_duration": sorted(durations)[len(durations) // 2],
            "min_duration": min(durations),
            "max_duration": max(durations),
            "slowest_tests": [
                {"name": m.test_name, "duration": m.duration}
                for m in sorted(self.metrics, key=lambda x: x.duration, reverse=True)[:10]
            ],
            "fastest_tests": [
                {"name": m.test_name, "duration": m.duration}
                for m in sorted(self.metrics, key=lambda x: x.duration)[:10]
            ]
        }
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate trend analysis."""
        # This would typically compare with historical data
        # For now, return basic trend indicators
        
        total_tests = len(self.metrics)
        passed_tests = len([t for t in self.metrics if t.status == "passed"])
        failed_tests = len([t for t in self.metrics if t.status == "failed"])
        
        return {
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "failure_rate": (failed_tests / total_tests * 100) if total_tests > 0 else 0,
            "test_stability": "stable" if failed_tests == 0 else "unstable",
            "performance_trend": "improving",  # Would be calculated from historical data
            "coverage_trend": "stable"  # Would be calculated from historical data
        }
    
    def _load_historical_data(self) -> List[Dict[str, Any]]:
        """Load historical test data for trend analysis."""
        historical_file = self.output_dir / "historical_data.json"
        
        if not historical_file.exists():
            return []
        
        try:
            with open(historical_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def save_historical_data(self):
        """Save current results to historical data."""
        historical_file = self.output_dir / "historical_data.json"
        
        current_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_tests": len(self.metrics),
            "passed_tests": len([t for t in self.metrics if t.status == "passed"]),
            "failed_tests": len([t for t in self.metrics if t.status == "failed"]),
            "success_rate": (len([t for t in self.metrics if t.status == "passed"]) / len(self.metrics) * 100) if self.metrics else 0,
            "coverage": sum(s.coverage for s in self.suites) / len(self.suites) if self.suites else 0,
            "total_duration": sum(t.duration for t in self.metrics)
        }
        
        # Load existing data
        historical_data = self._load_historical_data()
        
        # Add current data
        historical_data.append(current_data)
        
        # Keep only last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        historical_data = [
            d for d in historical_data 
            if datetime.strptime(d["date"], "%Y-%m-%d") > cutoff_date
        ]
        
        # Save updated data
        with open(historical_file, "w") as f:
            json.dump(historical_data, f, indent=2)


class CIIntegration:
    """CI/CD integration utilities."""
    
    @staticmethod
    def check_quality_gates(report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check quality gates for CI/CD pipeline."""
        gates = {
            "coverage_threshold": 80.0,
            "success_rate_threshold": 95.0,
            "max_duration_threshold": 3600.0,  # 1 hour
            "max_failed_tests": 5
        }
        
        summary = report_data.get("summary", {})
        
        results = {
            "passed": True,
            "gates": {},
            "recommendations": []
        }
        
        # Coverage gate
        coverage = summary.get("coverage", 0)
        coverage_passed = coverage >= gates["coverage_threshold"]
        results["gates"]["coverage"] = {
            "passed": coverage_passed,
            "value": coverage,
            "threshold": gates["coverage_threshold"]
        }
        if not coverage_passed:
            results["passed"] = False
            results["recommendations"].append(f"Increase code coverage to at least {gates['coverage_threshold']}%")
        
        # Success rate gate
        total_tests = summary.get("total_tests", 0)
        passed_tests = summary.get("passed_tests", 0)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        success_rate_passed = success_rate >= gates["success_rate_threshold"]
        results["gates"]["success_rate"] = {
            "passed": success_rate_passed,
            "value": success_rate,
            "threshold": gates["success_rate_threshold"]
        }
        if not success_rate_passed:
            results["passed"] = False
            results["recommendations"].append(f"Fix failing tests to achieve {gates['success_rate_threshold']}% success rate")
        
        # Duration gate
        total_duration = summary.get("total_duration", 0)
        duration_passed = total_duration <= gates["max_duration_threshold"]
        results["gates"]["duration"] = {
            "passed": duration_passed,
            "value": total_duration,
            "threshold": gates["max_duration_threshold"]
        }
        if not duration_passed:
            results["passed"] = False
            results["recommendations"].append("Optimize slow tests to reduce total execution time")
        
        # Failed tests gate
        failed_tests = summary.get("failed_tests", 0)
        failed_tests_passed = failed_tests <= gates["max_failed_tests"]
        results["gates"]["failed_tests"] = {
            "passed": failed_tests_passed,
            "value": failed_tests,
            "threshold": gates["max_failed_tests"]
        }
        if not failed_tests_passed:
            results["passed"] = False
            results["recommendations"].append(f"Fix failing tests (current: {failed_tests}, max allowed: {gates['max_failed_tests']})")
        
        return results
    
    @staticmethod
    def generate_slack_notification(report_data: Dict[str, Any], quality_gates: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Slack notification payload."""
        summary = report_data.get("summary", {})
        
        color = "good" if quality_gates["passed"] else "danger"
        
        fields = [
            {
                "title": "Total Tests",
                "value": str(summary.get("total_tests", 0)),
                "short": True
            },
            {
                "title": "Success Rate",
                "value": f"{summary.get('passed_tests', 0) / summary.get('total_tests', 1) * 100:.1f}%",
                "short": True
            },
            {
                "title": "Coverage",
                "value": f"{summary.get('coverage', 0):.1f}%",
                "short": True
            },
            {
                "title": "Duration",
                "value": f"{summary.get('total_duration', 0):.1f}s",
                "short": True
            }
        ]
        
        if not quality_gates["passed"]:
            fields.append({
                "title": "Issues",
                "value": "\n".join(quality_gates["recommendations"]),
                "short": False
            })
        
        return {
            "attachments": [
                {
                    "color": color,
                    "title": "GAELP Test Results",
                    "fields": fields,
                    "footer": "GAELP CI/CD",
                    "ts": int(time.time())
                }
            ]
        }
    
    @staticmethod
    def generate_github_status(quality_gates: Dict[str, Any]) -> Dict[str, Any]:
        """Generate GitHub status check payload."""
        state = "success" if quality_gates["passed"] else "failure"
        
        description = "All quality gates passed" if quality_gates["passed"] else "Quality gates failed"
        
        return {
            "state": state,
            "description": description,
            "context": "gaelp/tests"
        }