# GAELP Testing Framework

A comprehensive testing and validation framework for the Generic Agent Experimentation & Learning Platform (GAELP). This framework ensures reliability, performance, and security before deploying agents with real budgets.

## Overview

The GAELP testing framework provides:

- **Comprehensive Test Coverage**: Unit, integration, end-to-end, load, and security tests
- **Agent Validation**: Simulation-to-real transfer validation and safety mechanism testing
- **Performance Benchmarking**: Load testing and resource utilization monitoring
- **Security Testing**: Vulnerability scanning and safety compliance validation
- **Automated Reporting**: HTML, JSON, and JUnit reports with trend analysis
- **CI/CD Integration**: Automated testing in GitHub Actions with quality gates

## Test Structure

```
tests/
├── unit/                     # Unit tests for individual components
│   ├── test_environment_api.py
│   ├── test_agent_manager.py
│   ├── test_safety_framework.py
│   └── ...
├── integration/              # Integration tests for service interactions
│   ├── test_mcp_connectors.py
│   └── ...
├── e2e/                      # End-to-end workflow tests
│   ├── test_complete_training_pipeline.py
│   └── ...
├── load/                     # Load and performance tests
│   ├── test_performance_load.py
│   └── ...
├── security/                 # Security vulnerability tests
│   ├── test_security_vulnerabilities.py
│   └── ...
└── utils/                    # Testing utilities and helpers
    ├── test_data_generators.py
    ├── validators.py
    ├── test_reporting.py
    └── ...
```

## Running Tests

### Quick Start

```bash
# Install dependencies
pip install -r requirements-test.txt

# Run all tests
python test_runner.py

# Run specific test types
python test_runner.py --types unit integration
python test_runner.py --types e2e --environment staging
python test_runner.py --types security --no-quality-gates
```

### Test Types

#### Unit Tests
```bash
python test_runner.py --types unit
```
- Test individual components in isolation
- Mock external dependencies
- Fast execution (< 5 minutes)
- Target: 90%+ code coverage

#### Integration Tests
```bash
python test_runner.py --types integration
```
- Test service-to-service interactions
- Validate data flow between components
- Test with real databases and message queues
- Medium execution time (10-15 minutes)

#### End-to-End Tests
```bash
python test_runner.py --types e2e
```
- Complete user workflow testing
- Simulation-to-real deployment pipeline
- Full system validation
- Longer execution time (20-30 minutes)

#### Load Tests
```bash
python test_runner.py --types load
```
- Concurrent user simulation
- Performance benchmarking
- Resource utilization testing
- Variable execution time (15-60 minutes)

#### Security Tests
```bash
python test_runner.py --types security
```
- Vulnerability scanning
- Input validation testing
- Authentication/authorization testing
- Medium execution time (10-20 minutes)

## Test Configuration

### Configuration File
Create `test_config.json`:

```json
{
  "output_dir": "reports",
  "test_types": ["unit", "integration", "e2e", "security"],
  "parallel_workers": 4,
  "timeout": 3600,
  "coverage_threshold": 90.0,
  "success_rate_threshold": 95.0,
  "environments": {
    "test": {
      "database_url": "postgresql://test:test@localhost:5432/gaelp_test",
      "redis_url": "redis://localhost:6379/0",
      "api_base_url": "http://localhost:8000"
    }
  },
  "quality_gates": {
    "enabled": true,
    "coverage_threshold": 80.0,
    "success_rate_threshold": 95.0,
    "max_duration": 3600,
    "max_failed_tests": 5
  }
}
```

### Environment Variables
```bash
export GAELP_ENV=test
export DATABASE_URL=postgresql://test:test@localhost:5432/gaelp_test
export REDIS_URL=redis://localhost:6379/0
export GAELP_LOG_LEVEL=DEBUG
```

## Key Testing Areas

### 1. Agent Training Pipeline
- **Simulation Environment Validation**: Verify persona consistency and environment determinism
- **Training Convergence**: Monitor learning curves and convergence metrics
- **Transfer Learning**: Validate simulation-to-real performance correlation
- **Safety Mechanisms**: Test budget controls and emergency stops

### 2. API Endpoints
- **Input Validation**: Test with various valid/invalid inputs
- **Authentication/Authorization**: Verify access controls
- **Rate Limiting**: Test API throttling mechanisms
- **Error Handling**: Validate error responses and codes

### 3. Safety Framework
- **Budget Monitoring**: Test spending limits and alerts
- **Content Safety**: Validate inappropriate content detection
- **Policy Enforcement**: Test compliance with advertising policies
- **Emergency Procedures**: Validate automatic stops and escalation

### 4. Performance & Scalability
- **Concurrent Training**: Multiple agents training simultaneously
- **Database Performance**: Query optimization under load
- **API Throughput**: Requests per second benchmarks
- **Memory Usage**: Memory leak detection and optimization

### 5. Data Pipeline
- **Data Integrity**: Validate data consistency across services
- **BigQuery Integration**: Test analytics and reporting queries
- **Data Privacy**: Ensure PII protection and compliance
- **Backup/Recovery**: Test data restoration procedures

## Test Data Generation

### Synthetic Data
```python
from tests.utils.test_data_generators import TestDataGenerator

generator = TestDataGenerator(seed=42)

# Generate test personas
persona = generator.generate_persona_config()

# Generate test campaigns
campaign = generator.generate_ad_campaign(budget_range=(10.0, 100.0))

# Generate training metrics
metrics = generator.generate_training_metrics(episodes=100)
```

### Mock Services
```python
# Mock external API responses
@pytest.fixture
def mock_meta_ads_api():
    with patch('mcp_connectors.meta_ads.MetaAdsClient') as mock:
        mock.return_value.create_campaign.return_value = {"id": "123456"}
        yield mock
```

## Validation Framework

### Schema Validation
```python
from tests.utils.validators import SchemaValidator

validator = SchemaValidator()
result = validator.validate_schema(campaign_data, "ad_campaign")

assert result.is_valid
assert result.score > 0.9
```

### Business Logic Validation
```python
from tests.utils.validators import BusinessLogicValidator

validator = BusinessLogicValidator()
result = validator.validate_budget_constraints(campaign, agent_limits)

assert result.is_valid
assert len(result.errors) == 0
```

### Safety Validation
```python
from tests.utils.validators import SafetyValidator

validator = SafetyValidator()
result = validator.validate_content_safety(creative_content)

assert result.is_valid
assert result.score > 0.8
```

## Reporting & Analytics

### HTML Reports
Comprehensive HTML reports with:
- Test execution summary
- Coverage analysis
- Performance metrics
- Trend analysis charts
- Failure details

### JSON Reports
Machine-readable reports for CI/CD integration:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "summary": {
    "total_tests": 150,
    "passed_tests": 148,
    "failed_tests": 2,
    "coverage": 92.5
  },
  "quality_gates": {
    "passed": true,
    "gates": {
      "coverage": {"passed": true, "value": 92.5, "threshold": 80.0}
    }
  }
}
```

### JUnit XML
Standard JUnit XML format for CI systems:
```xml
<testsuites>
  <testsuite name="unit" tests="50" failures="0" skipped="0" time="45.2">
    <testcase name="test_agent_creation" classname="unit" time="0.05"/>
    <!-- ... -->
  </testsuite>
</testsuites>
```

## CI/CD Integration

### GitHub Actions
The framework integrates with GitHub Actions for:
- Automated testing on pull requests
- Parallel test execution
- Quality gate enforcement
- Slack notifications
- Coverage reporting

### Quality Gates
Configurable quality gates that must pass before deployment:
- **Code Coverage**: Minimum 80% coverage required
- **Test Success Rate**: Minimum 95% tests must pass
- **Performance**: Maximum execution time limits
- **Security**: No critical vulnerabilities allowed

### Notifications
- **Slack**: Real-time test results and failures
- **GitHub**: PR comments with test summaries
- **Email**: Critical failure notifications

## Best Practices

### Writing Tests
1. **Isolation**: Each test should be independent
2. **Deterministic**: Tests should produce consistent results
3. **Fast**: Unit tests should complete quickly
4. **Clear**: Test names should describe what they validate
5. **Comprehensive**: Cover happy path, edge cases, and error conditions

### Test Data
1. **Realistic**: Use data that mimics production scenarios
2. **Varied**: Test with different data types and sizes
3. **Clean**: Clean up test data after execution
4. **Secure**: Don't use real credentials or PII in tests

### Performance
1. **Benchmarks**: Establish performance baselines
2. **Monitoring**: Track test execution times
3. **Optimization**: Optimize slow tests
4. **Parallel**: Run independent tests in parallel

### Security
1. **Input Validation**: Test with malicious inputs
2. **Authentication**: Verify access controls
3. **Secrets**: Use test credentials only
4. **Compliance**: Validate regulatory requirements

## Troubleshooting

### Common Issues

#### Test Failures
```bash
# Run with verbose output
python test_runner.py --types unit -v

# Run specific failing test
pytest tests/unit/test_agent_manager.py::test_create_agent -v

# Debug with pdb
pytest tests/unit/test_agent_manager.py::test_create_agent --pdb
```

#### Environment Issues
```bash
# Check database connection
psql postgresql://test:test@localhost:5432/gaelp_test

# Check Redis connection
redis-cli -h localhost -p 6379 ping

# Verify API server
curl http://localhost:8000/health
```

#### Performance Issues
```bash
# Profile test execution
python -m cProfile test_runner.py --types unit

# Memory profiling
python -m memory_profiler test_runner.py --types unit
```

### Debugging Tips
1. **Logs**: Check logs in `logs/` directory
2. **Reports**: Review HTML reports for detailed failure information
3. **Isolation**: Run failing tests in isolation
4. **Environment**: Verify test environment setup
5. **Dependencies**: Check for version conflicts

## Contributing

### Adding New Tests
1. Choose appropriate test type (unit/integration/e2e/load/security)
2. Follow naming conventions (`test_*.py`)
3. Use existing fixtures and utilities
4. Add appropriate markers (`@pytest.mark.unit`)
5. Update documentation

### Test Guidelines
1. **One assertion per test** when possible
2. **Descriptive test names** that explain the scenario
3. **Setup and teardown** using fixtures
4. **Mock external dependencies** in unit tests
5. **Test error conditions** not just happy paths

### Code Coverage
- Aim for 90%+ coverage on new code
- Focus on critical business logic
- Don't sacrifice test quality for coverage numbers
- Use coverage reports to identify gaps

## Monitoring & Maintenance

### Regular Tasks
- **Review test results** daily
- **Update test data** monthly
- **Performance benchmarks** quarterly
- **Security scans** weekly
- **Dependency updates** monthly

### Metrics to Track
- Test execution time trends
- Coverage percentage over time
- Flaky test identification
- Performance regression detection
- Security vulnerability trends

## Support

For questions about the testing framework:
- **Documentation**: This README and inline code comments
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and best practices

---

The GAELP testing framework ensures platform reliability and safety through comprehensive automated testing. By following these guidelines and best practices, we maintain high quality standards while enabling rapid development and deployment.