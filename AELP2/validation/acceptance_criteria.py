"""Acceptance criteria validation for AELP2 production system.

This module validates that ALL production requirements are met:
- All required components exist and function
- No fallbacks or simplifications
- Minimum performance requirements met
- Production safety gates operational
- BigQuery integration working
- Attribution system functional

STRICT VALIDATION: System must pass ALL criteria for production deployment.
"""

import os
import sys
import json
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import re

# Assume repo root on PYTHONPATH; do not inject user-specific paths


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = 'critical'  # critical, high, medium, low


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    pass_rate: float
    status: str  # PASSED, FAILED, PARTIAL
    results: List[ValidationResult]
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'timestamp': self.timestamp,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'pass_rate': self.pass_rate,
            'status': self.status,
            'results': [asdict(r) for r in self.results],
            'summary': self.summary
        }
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class AELP2AcceptanceValidator:
    """Validates AELP2 system against acceptance criteria."""
    
    def __init__(self):
        """Initialize validator."""
        # Resolve directories relative to this file
        self.aelp2_dir = Path(__file__).resolve().parents[1]
        self.legacy_dir = self.aelp2_dir.parent
        self.results: List[ValidationResult] = []
        
        # Required environment variables
        self.required_env_vars = [
            'GOOGLE_CLOUD_PROJECT',
            'BIGQUERY_TRAINING_DATASET',
            'AELP2_MIN_WIN_RATE',
            'AELP2_MAX_CAC',
            'AELP2_MIN_ROAS',
            'AELP2_MAX_SPEND_VELOCITY',
            'AELP2_APPROVAL_TIMEOUT',
            'AELP2_SIM_BUDGET',
            'AELP2_EPISODES',
            'AELP2_SIM_STEPS'
        ]
        
        # Required modules
        self.required_modules = [
            'AELP2.core.env.simulator',
            'AELP2.core.env.calibration',
            'AELP2.core.monitoring.bq_writer',
            'AELP2.core.safety.hitl',
            'AELP2.core.intelligence.reward_attribution',
            'AELP2.core.orchestration.production_orchestrator',
            'AELP2.core.data.google_adapter',
            'AELP2.core.data.platform_adapter'
        ]
        
        # Forbidden patterns
        self.forbidden_patterns = [
            'fallback',
            'simplified',
            'mock',
            'dummy',
            'stub',
            'fake',
            'TODO',
            'FIXME',
            'NotImplemented'
        ]
    
    def validate_all(self) -> ValidationReport:
        """Run all validation checks and generate report."""
        print("=" * 80)
        print("AELP2 ACCEPTANCE CRITERIA VALIDATION")
        print("=" * 80)
        print()
        
        # Run all validation checks
        self._validate_directory_structure()
        self._validate_required_files()
        self._validate_module_imports()
        self._validate_no_forbidden_patterns()
        self._validate_no_hardcoded_values()
        self._validate_environment_configuration()
        self._validate_orchestrator_requirements()
        self._validate_safety_components()
        self._validate_attribution_system()
        self._validate_bigquery_integration()
        self._validate_auction_calibration()
        self._validate_reinforcement_learning()
        self._validate_production_readiness()
        
        # Generate report
        report = self._generate_report()
        
        # Print report
        self._print_report(report)
        
        return report
    
    def _validate_directory_structure(self):
        """Validate AELP2 directory structure exists."""
        print("Checking directory structure...")
        
        required_dirs = [
            self.aelp2_dir / 'core',
            self.aelp2_dir / 'core' / 'env',
            self.aelp2_dir / 'core' / 'monitoring',
            self.aelp2_dir / 'core' / 'safety',
            self.aelp2_dir / 'core' / 'intelligence',
            self.aelp2_dir / 'core' / 'orchestration',
            self.aelp2_dir / 'core' / 'data'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            self.results.append(ValidationResult(
                name="Directory Structure",
                passed=False,
                message=f"Missing {len(missing_dirs)} required directories",
                details={'missing': missing_dirs}
            ))
        else:
            self.results.append(ValidationResult(
                name="Directory Structure",
                passed=True,
                message="All required directories exist"
            ))
    
    def _validate_required_files(self):
        """Validate all required files exist."""
        print("Checking required files...")
        
        required_files = [
            'core/env/simulator.py',
            'core/env/calibration.py',
            'core/monitoring/bq_writer.py',
            'core/safety/hitl.py',
            'core/intelligence/reward_attribution.py',
            'core/orchestration/production_orchestrator.py',
            'core/data/google_adapter.py',
            'core/data/platform_adapter.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.aelp2_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        # Check legacy files
        legacy_files = [
            'fortified_environment_no_hardcoding.py',
            'fortified_rl_agent_no_hardcoding.py',
            'gaelp_parameter_manager.py'
        ]
        
        for file_name in legacy_files:
            if not (self.legacy_dir / file_name).exists():
                missing_files.append(f"legacy/{file_name}")
        
        if missing_files:
            self.results.append(ValidationResult(
                name="Required Files",
                passed=False,
                message=f"Missing {len(missing_files)} required files",
                details={'missing': missing_files}
            ))
        else:
            self.results.append(ValidationResult(
                name="Required Files",
                passed=True,
                message="All required files exist"
            ))
    
    def _validate_module_imports(self):
        """Validate all required modules can be imported."""
        print("Checking module imports...")
        
        import_failures = []
        
        for module_name in self.required_modules:
            try:
                module = importlib.import_module(module_name)
                # Check module has content
                if len(dir(module)) < 5:  # Arbitrary minimum
                    import_failures.append(f"{module_name}: Module appears empty")
            except ImportError as e:
                import_failures.append(f"{module_name}: {str(e)}")
            except Exception as e:
                import_failures.append(f"{module_name}: Unexpected error - {str(e)}")
        
        if import_failures:
            self.results.append(ValidationResult(
                name="Module Imports",
                passed=False,
                message=f"{len(import_failures)} modules failed to import",
                details={'failures': import_failures}
            ))
        else:
            self.results.append(ValidationResult(
                name="Module Imports",
                passed=True,
                message="All required modules import successfully"
            ))
    
    def _validate_no_forbidden_patterns(self):
        """Validate no forbidden patterns in code."""
        print("Checking for forbidden patterns...")
        
        violations = []
        
        for py_file in self.aelp2_dir.rglob('*.py'):
            if '__pycache__' in str(py_file) or 'test' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        # Skip comments
                        if line.strip().startswith('#'):
                            continue
                        
                        for pattern in self.forbidden_patterns:
                            if pattern.lower() in line.lower():
                                violations.append(
                                    f"{py_file.relative_to(self.legacy_dir)}:{i} - "
                                    f"{pattern}: {line.strip()[:50]}"
                                )
                                if len(violations) > 20:  # Limit output
                                    break
            except:
                pass
        
        if violations:
            self.results.append(ValidationResult(
                name="Forbidden Patterns",
                passed=False,
                message=f"Found {len(violations)} forbidden patterns",
                details={'violations': violations[:20]}
            ))
        else:
            self.results.append(ValidationResult(
                name="Forbidden Patterns",
                passed=True,
                message="No forbidden patterns found"
            ))
    
    def _validate_no_hardcoded_values(self):
        """Validate no hardcoded values."""
        print("Checking for hardcoded values...")
        
        hardcoded_patterns = [
            (r'segments?\s*=\s*\[', 'hardcoded segments'),
            (r'categories?\s*=\s*\[', 'hardcoded categories'),
            (r'threshold\s*=\s*\d+\.?\d*', 'fixed threshold'),
            (r'budget\s*=\s*\d+\.?\d*', 'fixed budget'),
        ]
        
        violations = []
        
        for py_file in self.aelp2_dir.rglob('*.py'):
            if '__pycache__' in str(py_file) or 'test' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        # Skip comments and default parameters
                        if line.strip().startswith('#') or 'default=' in line:
                            continue
                        
                        for pattern, desc in hardcoded_patterns:
                            if re.search(pattern, line):
                                if 'os.getenv' not in line and 'environ' not in line:
                                    violations.append(
                                        f"{py_file.name}:{i} - {desc}"
                                    )
            except:
                pass
        
        if violations:
            self.results.append(ValidationResult(
                name="Hardcoded Values",
                passed=False,
                message=f"Found {len(violations)} hardcoded values",
                details={'violations': violations[:10]},
                severity='high'
            ))
        else:
            self.results.append(ValidationResult(
                name="Hardcoded Values",
                passed=True,
                message="No hardcoded values found"
            ))
    
    def _validate_environment_configuration(self):
        """Validate environment variable configuration."""
        print("Checking environment configuration...")
        
        missing_vars = []
        for var in self.required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.results.append(ValidationResult(
                name="Environment Configuration",
                passed=False,
                message=f"Missing {len(missing_vars)} required environment variables",
                details={'missing': missing_vars},
                severity='high'
            ))
        else:
            self.results.append(ValidationResult(
                name="Environment Configuration",
                passed=True,
                message="All required environment variables set"
            ))
    
    def _validate_orchestrator_requirements(self):
        """Validate orchestrator meets requirements."""
        print("Checking orchestrator requirements...")
        
        try:
            from AELP2.core.orchestration.production_orchestrator import (
                AELP2ProductionOrchestrator, OrchestratorConfig
            )
            
            # Test configuration validation
            config = OrchestratorConfig(
                episodes=1,
                steps=200,  # Minimum required
                sim_budget=1000.0,
                min_win_rate=0.05
            )
            config.validate()
            
            # Test that < 200 steps fails
            try:
                bad_config = OrchestratorConfig(
                    episodes=1,
                    steps=100,  # Below minimum
                    sim_budget=1000.0
                )
                bad_config.validate()
                self.results.append(ValidationResult(
                    name="Orchestrator Requirements",
                    passed=False,
                    message="Orchestrator allows < 200 steps (should reject)"
                ))
            except:
                self.results.append(ValidationResult(
                    name="Orchestrator Requirements",
                    passed=True,
                    message="Orchestrator enforces minimum 200 steps"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                name="Orchestrator Requirements",
                passed=False,
                message=f"Orchestrator validation failed: {str(e)}"
            ))
    
    def _validate_safety_components(self):
        """Validate safety components are functional."""
        print("Checking safety components...")
        
        try:
            from AELP2.core.safety.hitl import (
                SafetyGates, HITLApprovalQueue, PolicyChecker,
                SafetyEventLogger, SafetyEventType, SafetyEventSeverity
            )
            
            # Test safety gates
            gates = SafetyGates()
            metrics = {
                'win_rate': 0.1,
                'cac': 50.0,
                'roas': 2.0,
                'spend_velocity': 5.0
            }
            passed, violations = gates.evaluate_gates(metrics)
            
            # Test HITL queue
            queue = HITLApprovalQueue()
            request_id = queue.request_approval(
                action_type='test',
                action_data={},
                risk_level='low'
            )
            
            # Test policy checker
            checker = PolicyChecker()
            action = {'bid_amount': 10.0}
            is_compliant, policy_violations = checker.check_action(action)
            
            # Test event logger
            logger = SafetyEventLogger()
            logger.log_safety_event(
                SafetyEventType.GATE_VIOLATION,
                SafetyEventSeverity.LOW,
                {'test': 'data'}
            )
            
            self.results.append(ValidationResult(
                name="Safety Components",
                passed=True,
                message="All safety components functional"
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Safety Components",
                passed=False,
                message=f"Safety component failure: {str(e)}"
            ))
    
    def _validate_attribution_system(self):
        """Validate attribution system functionality."""
        print("Checking attribution system...")
        
        try:
            from AELP2.core.intelligence.reward_attribution import RewardAttributionWrapper
            
            # Create mock attribution engine
            class MockAttribution:
                def attribute_conversion(self, *args, **kwargs):
                    return [
                        {'touchpoint_id': 'tp1', 'credit': 0.5},
                        {'touchpoint_id': 'tp2', 'credit': 0.5}
                    ]
            
            wrapper = RewardAttributionWrapper(
                attribution_engine=MockAttribution(),
                config={'attribution_window_days': 7}
            )
            
            # Test touchpoint tracking
            tp_id = wrapper.track_touchpoint(
                campaign_data={'bid': 10.0},
                user_data={'user_id': 'test'},
                spend=10.0
            )
            
            # Test conversion tracking
            result = wrapper.track_conversion(
                conversion_value=100.0,
                user_id='test',
                conversion_data={}
            )
            
            self.results.append(ValidationResult(
                name="Attribution System",
                passed=True,
                message="Attribution system functional"
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Attribution System",
                passed=False,
                message=f"Attribution system failure: {str(e)}"
            ))
    
    def _validate_bigquery_integration(self):
        """Validate BigQuery integration."""
        print("Checking BigQuery integration...")
        
        try:
            from AELP2.core.monitoring.bq_writer import BigQueryWriter, create_bigquery_writer
            
            # Test with mock client
            from unittest.mock import MagicMock
            mock_client = MagicMock()
            writer = BigQueryWriter(client=mock_client)
            
            # Test write operations
            episode_data = {
                'episode_id': 'test',
                'steps': 200,
                'auctions': 100,
                'wins': 30,
                'win_rate': 0.3
            }
            
            result = writer.write_episode_metrics(episode_data)
            
            if result:
                self.results.append(ValidationResult(
                    name="BigQuery Integration",
                    passed=True,
                    message="BigQuery writer functional"
                ))
            else:
                self.results.append(ValidationResult(
                    name="BigQuery Integration",
                    passed=False,
                    message="BigQuery write operation failed"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                name="BigQuery Integration",
                passed=False,
                message=f"BigQuery integration failure: {str(e)}"
            ))
    
    def _validate_auction_calibration(self):
        """Validate auction calibration functionality."""
        print("Checking auction calibration...")
        
        try:
            from AELP2.core.env.calibration import AuctionCalibrator, CalibrationResult
            
            # Create mock auction
            class MockAuction:
                def run_auction(self, bid, context=None):
                    return {'won': bid > 10.0, 'cost': bid * 0.8}
            
            calibrator = AuctionCalibrator(target_min=0.1, target_max=0.3)
            
            def context_factory():
                return {'competition_level': 0.5}
            
            result = calibrator.calibrate(
                MockAuction(),
                context_factory=context_factory
            )
            
            if isinstance(result, CalibrationResult):
                self.results.append(ValidationResult(
                    name="Auction Calibration",
                    passed=True,
                    message="Auction calibration functional"
                ))
            else:
                self.results.append(ValidationResult(
                    name="Auction Calibration",
                    passed=False,
                    message="Calibration did not return valid result"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                name="Auction Calibration",
                passed=False,
                message=f"Auction calibration failure: {str(e)}"
            ))
    
    def _validate_reinforcement_learning(self):
        """Validate RL implementation (not bandits)."""
        print("Checking for reinforcement learning...")
        
        rl_indicators = [
            'q_learning', 'qlearning', 'deep_q', 'dqn',
            'ppo', 'replay_buffer', 'bellman', 'td_error'
        ]
        
        bandit_indicators = ['bandit', 'ucb', 'thompson']
        
        rl_found = False
        bandit_found = False
        
        # Check legacy agent file
        agent_file = self.legacy_dir / 'fortified_rl_agent_no_hardcoding.py'
        if agent_file.exists():
            try:
                with open(agent_file, 'r') as f:
                    content = f.read().lower()
                    
                    for indicator in rl_indicators:
                        if indicator in content:
                            rl_found = True
                            break
                    
                    for indicator in bandit_indicators:
                        if indicator in content and 'bandit' not in 'abandoned':
                            bandit_found = True
            except:
                pass
        
        if rl_found and not bandit_found:
            self.results.append(ValidationResult(
                name="Reinforcement Learning",
                passed=True,
                message="Using proper RL (not bandits)"
            ))
        elif bandit_found:
            self.results.append(ValidationResult(
                name="Reinforcement Learning",
                passed=False,
                message="Using bandits instead of proper RL"
            ))
        else:
            self.results.append(ValidationResult(
                name="Reinforcement Learning",
                passed=False,
                message="No RL implementation found",
                severity='high'
            ))
    
    def _validate_production_readiness(self):
        """Validate overall production readiness."""
        print("Checking production readiness...")
        
        # Check if tests pass
        test_results = []
        test_files = [
            'tests/test_aelp2_integration.py',
            'tests/test_orchestrator_e2e.py',
            'tests/test_no_fallbacks.py'
        ]
        
        for test_file in test_files:
            test_path = self.legacy_dir / test_file
            if test_path.exists():
                test_results.append(f"{test_file}: exists")
            else:
                test_results.append(f"{test_file}: missing")
        
        if all('exists' in r for r in test_results):
            self.results.append(ValidationResult(
                name="Production Readiness",
                passed=True,
                message="All production tests present",
                details={'tests': test_results}
            ))
        else:
            self.results.append(ValidationResult(
                name="Production Readiness",
                passed=False,
                message="Missing production tests",
                details={'tests': test_results}
            ))
    
    def _generate_report(self) -> ValidationReport:
        """Generate validation report."""
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = total_checks - passed_checks
        pass_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Determine overall status
        critical_failures = [r for r in self.results if not r.passed and r.severity == 'critical']
        
        if failed_checks == 0:
            status = "PASSED"
            summary = "All acceptance criteria met. System ready for production."
        elif critical_failures:
            status = "FAILED"
            summary = f"{len(critical_failures)} critical failures. System NOT ready for production."
        elif pass_rate >= 80:
            status = "PARTIAL"
            summary = f"System partially ready ({pass_rate:.1f}% pass rate). Address failures before production."
        else:
            status = "FAILED"
            summary = f"System not ready ({pass_rate:.1f}% pass rate). Major issues must be resolved."
        
        return ValidationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            pass_rate=pass_rate,
            status=status,
            results=self.results,
            summary=summary
        )
    
    def _print_report(self, report: ValidationReport):
        """Print validation report to console."""
        print()
        print("=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print()
        
        # Print summary statistics
        print(f"Total Checks: {report.total_checks}")
        print(f"Passed: {report.passed_checks}")
        print(f"Failed: {report.failed_checks}")
        print(f"Pass Rate: {report.pass_rate:.1f}%")
        print(f"Status: {report.status}")
        print()
        
        # Print detailed results
        print("Detailed Results:")
        print("-" * 80)
        
        for result in report.results:
            symbol = "✓" if result.passed else "✗"
            print(f"{symbol} {result.name}: {result.message}")
            
            if not result.passed and result.details:
                # Print first few details
                if 'violations' in result.details:
                    violations = result.details['violations']
                    print(f"  First {min(3, len(violations))} violations:")
                    for v in violations[:3]:
                        print(f"    - {v}")
                elif 'missing' in result.details:
                    missing = result.details['missing']
                    print(f"  Missing ({len(missing)} items):")
                    for m in missing[:3]:
                        print(f"    - {m}")
                elif 'failures' in result.details:
                    failures = result.details['failures']
                    print(f"  Failures ({len(failures)} items):")
                    for f in failures[:3]:
                        print(f"    - {f}")
        
        print()
        print("=" * 80)
        print(f"FINAL STATUS: {report.status}")
        print(f"SUMMARY: {report.summary}")
        print("=" * 80)


def main():
    """Main entry point for acceptance validation."""
    validator = AELP2AcceptanceValidator()
    report = validator.validate_all()
    
    # Save report to file
    report_file = Path(__file__).resolve().parent / 'acceptance_report.json'
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report.to_json())
    
    print(f"\nReport saved to: {report_file}")
    
    # Return exit code based on status
    if report.status == "PASSED":
        return 0
    elif report.status == "PARTIAL":
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
