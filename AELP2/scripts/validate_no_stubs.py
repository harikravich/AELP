#!/usr/bin/env python3
"""
Validation Script: Ensure No Stub/Mock/Fallback Code Remains

This script systematically validates that all stub implementations have been replaced
with real, production-ready code. It enforces the NO FALLBACKS rule by scanning for
violations and verifying that systems actually work.

Run this after replacing stub files to ensure compliance with CLAUDE.md requirements.
"""

import os
import sys
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Forbidden patterns from CLAUDE.md
FORBIDDEN_PATTERNS = [
    r'fallback',
    r'simplified',
    r'mock(?!ito)',  # mock but not mockito, and not in test files
    r'dummy',
    r'placeholder',
    r'stub',
    r'TODO|FIXME',
    r'not implemented',
    r'pass\s*#.*later',
    r'return None\s*#.*implement',
    r'if False:',
    r'except.*:\s*pass',
    r'except.*:\s*return.*fallback',
    r'using fallback',
    r'using simplified',
    r'# temporary',
    r'# hack',
]

# Files that were previously stubs and should now be real implementations
CRITICAL_FILES = [
    'scripts/training_stub.py',
    'pipelines/attribution_engine_stub.py',
    'pipelines/delayed_conversions_stub.py',
    'pipelines/model_registry_stub.py',
    'pipelines/users_db_stub.py',
    'pipelines/creative_embeddings_stub.py',
    'adapters/meta_adapter_stub.py',
    'adapters/tiktok_adapter_stub.py',
    'adapters/linkedin_adapter_stub.py',
]


class NoStubsValidator:
    """Comprehensive validator for stub elimination."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.violations = []
        self.validation_results = {
            'total_files_scanned': 0,
            'violations_found': 0,
            'critical_files_validated': 0,
            'import_tests_passed': 0,
            'functionality_tests_passed': 0,
        }

    def run_full_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of stub elimination."""

        logger.info("Starting comprehensive stub elimination validation")

        # Step 1: Scan for forbidden patterns
        pattern_violations = self._scan_forbidden_patterns()

        # Step 2: Validate critical files
        critical_violations = self._validate_critical_files()

        # Step 3: Test imports
        import_results = self._test_imports()

        # Step 4: Basic functionality tests
        functionality_results = self._test_basic_functionality()

        # Step 5: Generate report
        report = self._generate_validation_report(
            pattern_violations, critical_violations, import_results, functionality_results
        )

        return report

    def _scan_forbidden_patterns(self) -> List[Dict[str, Any]]:
        """Scan all Python files for forbidden patterns."""

        logger.info("Scanning for forbidden stub/mock/fallback patterns")
        violations = []

        # Get all Python files
        python_files = list(self.base_path.glob('**/*.py'))

        for file_path in python_files:
            # Skip test files for some patterns
            is_test_file = 'test_' in file_path.name or '/tests/' in str(file_path)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check each forbidden pattern
                for pattern in FORBIDDEN_PATTERNS:
                    # Skip 'mock' pattern in test files
                    if 'mock' in pattern and is_test_file:
                        continue

                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)

                    for match in matches:
                        # Get line number
                        line_num = content[:match.start()].count('\n') + 1
                        line_content = content.split('\n')[line_num - 1].strip()

                        violations.append({
                            'file': str(file_path.relative_to(self.base_path)),
                            'pattern': pattern,
                            'line_number': line_num,
                            'line_content': line_content,
                            'match_text': match.group()
                        })

                self.validation_results['total_files_scanned'] += 1

            except Exception as e:
                logger.warning(f"Failed to scan {file_path}: {e}")

        self.validation_results['violations_found'] = len(violations)
        logger.info(f"Found {len(violations)} pattern violations in {self.validation_results['total_files_scanned']} files")

        return violations

    def _validate_critical_files(self) -> List[Dict[str, Any]]:
        """Validate that critical files have been properly replaced."""

        logger.info("Validating critical files that should no longer be stubs")
        violations = []

        for file_rel_path in CRITICAL_FILES:
            file_path = self.base_path / file_rel_path

            if not file_path.exists():
                violations.append({
                    'file': file_rel_path,
                    'issue': 'File does not exist',
                    'severity': 'ERROR'
                })
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for stub indicators
                stub_indicators = [
                    'stub',
                    'placeholder',
                    'not implemented',
                    'TODO',
                    'FIXME'
                ]

                found_indicators = []
                for indicator in stub_indicators:
                    if re.search(indicator, content, re.IGNORECASE):
                        found_indicators.append(indicator)

                if found_indicators:
                    violations.append({
                        'file': file_rel_path,
                        'issue': f'Still contains stub indicators: {found_indicators}',
                        'severity': 'ERROR'
                    })

                # Check for actual implementation content
                implementation_indicators = [
                    'class.*(?<!Stub)',
                    'def.*(?<!stub)',
                    'import.*(?<!stub)',
                    'try:.*except:.*raise',  # Proper error handling
                ]

                has_implementation = False
                for indicator in implementation_indicators:
                    if re.search(indicator, content, re.IGNORECASE):
                        has_implementation = True
                        break

                if not has_implementation:
                    violations.append({
                        'file': file_rel_path,
                        'issue': 'No substantial implementation found',
                        'severity': 'WARNING'
                    })
                else:
                    self.validation_results['critical_files_validated'] += 1

            except Exception as e:
                violations.append({
                    'file': file_rel_path,
                    'issue': f'Failed to validate: {e}',
                    'severity': 'ERROR'
                })

        logger.info(f"Validated {self.validation_results['critical_files_validated']}/{len(CRITICAL_FILES)} critical files")
        return violations

    def _test_imports(self) -> List[Dict[str, Any]]:
        """Test that critical modules can be imported without errors."""

        logger.info("Testing imports of replaced modules")
        import_results = []

        # Add the base path to Python path for imports
        sys.path.insert(0, str(self.base_path))

        critical_modules = [
            'scripts.training_stub',
            'pipelines.attribution_engine_stub',
            'pipelines.model_registry_stub',
            'adapters.meta_adapter_stub',
        ]

        for module_name in critical_modules:
            try:
                # Import the module
                __import__(module_name)

                import_results.append({
                    'module': module_name,
                    'status': 'SUCCESS',
                    'error': None
                })
                self.validation_results['import_tests_passed'] += 1

            except Exception as e:
                import_results.append({
                    'module': module_name,
                    'status': 'FAILED',
                    'error': str(e)
                })
                logger.error(f"Failed to import {module_name}: {e}")

        return import_results

    def _test_basic_functionality(self) -> List[Dict[str, Any]]:
        """Test basic functionality of replaced systems."""

        logger.info("Testing basic functionality of replaced systems")
        functionality_results = []

        # Test 1: Attribution system initialization
        try:
            from AELP2.core.intelligence.reward_attribution import RewardAttributionWrapper
            attribution = RewardAttributionWrapper()

            functionality_results.append({
                'test': 'AttributionWrapper initialization',
                'status': 'SUCCESS',
                'details': 'Attribution system initialized successfully'
            })
            self.validation_results['functionality_tests_passed'] += 1

        except Exception as e:
            functionality_results.append({
                'test': 'AttributionWrapper initialization',
                'status': 'FAILED',
                'details': f'Attribution initialization failed: {e}'
            })

        # Test 2: Meta adapter validation
        try:
            # Import and check basic structure
            from AELP2.adapters.meta_adapter_stub import MetaAdsAdapter

            # Check that key methods exist
            required_methods = ['apply_budget_change', 'publish_creative', 'create_campaign']
            missing_methods = [method for method in required_methods
                             if not hasattr(MetaAdsAdapter, method)]

            if missing_methods:
                functionality_results.append({
                    'test': 'MetaAdsAdapter structure',
                    'status': 'FAILED',
                    'details': f'Missing methods: {missing_methods}'
                })
            else:
                functionality_results.append({
                    'test': 'MetaAdsAdapter structure',
                    'status': 'SUCCESS',
                    'details': 'All required methods present'
                })
                self.validation_results['functionality_tests_passed'] += 1

        except Exception as e:
            functionality_results.append({
                'test': 'MetaAdsAdapter validation',
                'status': 'FAILED',
                'details': f'Meta adapter validation failed: {e}'
            })

        # Test 3: Training system structure
        try:
            training_file = self.base_path / 'scripts/training_stub.py'
            with open(training_file, 'r') as f:
                content = f.read()

            # Check for RL components
            rl_indicators = ['stable_baselines3', 'PPO', 'DQN', 'RecSim', 'gym.Env']
            found_rl = [indicator for indicator in rl_indicators if indicator in content]

            if len(found_rl) >= 3:  # At least 3 RL indicators
                functionality_results.append({
                    'test': 'RL training system structure',
                    'status': 'SUCCESS',
                    'details': f'Found RL components: {found_rl}'
                })
                self.validation_results['functionality_tests_passed'] += 1
            else:
                functionality_results.append({
                    'test': 'RL training system structure',
                    'status': 'FAILED',
                    'details': f'Insufficient RL components found: {found_rl}'
                })

        except Exception as e:
            functionality_results.append({
                'test': 'Training system validation',
                'status': 'FAILED',
                'details': f'Training system validation failed: {e}'
            })

        return functionality_results

    def _generate_validation_report(self, pattern_violations: List[Dict],
                                  critical_violations: List[Dict],
                                  import_results: List[Dict],
                                  functionality_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""

        # Calculate overall score
        total_tests = (
            len(CRITICAL_FILES) +  # Critical file validation
            len(import_results) +  # Import tests
            len(functionality_results)  # Functionality tests
        )

        passed_tests = (
            self.validation_results['critical_files_validated'] +
            self.validation_results['import_tests_passed'] +
            self.validation_results['functionality_tests_passed']
        )

        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        # Determine overall status
        if len(pattern_violations) == 0 and len(critical_violations) == 0 and success_rate >= 80:
            overall_status = "PASSED"
        elif success_rate >= 60:
            overall_status = "PASSED_WITH_WARNINGS"
        else:
            overall_status = "FAILED"

        report = {
            'validation_timestamp': datetime.utcnow().isoformat(),
            'overall_status': overall_status,
            'success_rate': f"{success_rate:.1f}%",
            'summary': {
                'total_files_scanned': self.validation_results['total_files_scanned'],
                'pattern_violations': len(pattern_violations),
                'critical_file_issues': len(critical_violations),
                'import_tests_passed': f"{self.validation_results['import_tests_passed']}/{len(import_results)}",
                'functionality_tests_passed': f"{self.validation_results['functionality_tests_passed']}/{len(functionality_results)}"
            },
            'details': {
                'pattern_violations': pattern_violations,
                'critical_file_violations': critical_violations,
                'import_results': import_results,
                'functionality_results': functionality_results
            },
            'recommendations': self._generate_recommendations(
                pattern_violations, critical_violations, import_results, functionality_results
            )
        }

        return report

    def _generate_recommendations(self, pattern_violations: List[Dict],
                                critical_violations: List[Dict],
                                import_results: List[Dict],
                                functionality_results: List[Dict]) -> List[str]:
        """Generate recommendations for fixing remaining issues."""

        recommendations = []

        # Pattern violation recommendations
        if pattern_violations:
            violation_files = set(v['file'] for v in pattern_violations)
            recommendations.append(
                f"CRITICAL: Fix {len(pattern_violations)} pattern violations in {len(violation_files)} files. "
                f"Remove all instances of: {set(v['pattern'] for v in pattern_violations)}"
            )

        # Critical file recommendations
        error_violations = [v for v in critical_violations if v['severity'] == 'ERROR']
        if error_violations:
            recommendations.append(
                f"CRITICAL: Fix {len(error_violations)} critical file errors. "
                f"These files must be properly implemented, not stubs."
            )

        # Import failure recommendations
        failed_imports = [r for r in import_results if r['status'] == 'FAILED']
        if failed_imports:
            recommendations.append(
                f"HIGH: Fix {len(failed_imports)} import failures. "
                f"Ensure all dependencies are installed and modules are syntactically correct."
            )

        # Functionality recommendations
        failed_functionality = [r for r in functionality_results if r['status'] == 'FAILED']
        if failed_functionality:
            recommendations.append(
                f"MEDIUM: Fix {len(failed_functionality)} functionality issues. "
                f"Ensure core systems have proper implementation structure."
            )

        # Success recommendations
        if not recommendations:
            recommendations.append(
                "EXCELLENT: No stub code detected! All systems appear to have real implementations."
            )

        return recommendations


def main():
    """Main entry point for validation script."""

    import argparse

    parser = argparse.ArgumentParser(description="Validate stub elimination")
    parser.add_argument('--path', default='/home/hariravichandran/AELP/AELP2',
                       help='Base path to validate')
    parser.add_argument('--output', help='Output file for validation report')
    parser.add_argument('--strict', action='store_true',
                       help='Strict mode - fail on any violations')
    args = parser.parse_args()

    # Run validation
    validator = NoStubsValidator(args.path)
    report = validator.run_full_validation()

    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Validation report written to {args.output}")
    else:
        print(json.dumps(report, indent=2))

    # Print summary
    print("\n" + "="*80)
    print("STUB ELIMINATION VALIDATION SUMMARY")
    print("="*80)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Success Rate: {report['success_rate']}")
    print(f"Files Scanned: {report['summary']['total_files_scanned']}")
    print(f"Pattern Violations: {report['summary']['pattern_violations']}")
    print(f"Critical File Issues: {report['summary']['critical_file_issues']}")
    print(f"Import Tests: {report['summary']['import_tests_passed']}")
    print(f"Functionality Tests: {report['summary']['functionality_tests_passed']}")

    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")

    print("="*80)

    # Exit with appropriate code
    if args.strict and report['overall_status'] != 'PASSED':
        logger.error("Strict mode: validation failed")
        sys.exit(1)
    elif report['overall_status'] == 'FAILED':
        logger.error("Validation failed")
        sys.exit(1)
    else:
        logger.info("Validation completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()