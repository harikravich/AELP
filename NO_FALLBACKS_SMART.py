#!/usr/bin/env python3
"""
GAELP Smart Fallback Validator

This validates against REAL problems that prevent learning, not overly strict rules.
Focus: Prevent fake execution that looks real but isn't actually learning.

CRITICAL VIOLATIONS (Block these):
- Production fallback code that replaces real computation
- Mock implementations outside tests
- Silent failures that hide problems
- Hardcoded values that should be discovered

ACCEPTABLE PATTERNS (Allow these):
- Test file mocks
- Error handling with loud logging
- Safety controls and emergency stops
- Performance optimizations with documentation
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

class FallbackValidator:
    """Smart validator that distinguishes critical from acceptable patterns"""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.violations = defaultdict(list)
        self.skip_dirs = {'.git', '__pycache__', 'node_modules', '.pytest_cache'}
        
    def is_test_file(self, filepath: str) -> bool:
        """Check if file is a test file"""
        filename = os.path.basename(filepath)
        return (filename.startswith('test_') or 
                filename.endswith('_test.py') or
                '/test/' in filepath or
                '/tests/' in filepath)
    
    def check_critical_violations(self, filepath: str, content: str) -> List[Dict]:
        """Check for violations that actually prevent learning"""
        violations = []
        lines = content.split('\n')
        
        # Skip test files for most checks
        if self.is_test_file(filepath):
            return violations
            
        for i, line in enumerate(lines, 1):
            # Skip comments and docstrings
            if line.strip().startswith('#') or '"""' in line:
                continue
                
            # CRITICAL: Production fallback methods
            if re.search(r'def\s+_fallback_\w+|def\s+fallback_\w+', line):
                violations.append({
                    'file': filepath,
                    'line': i,
                    'type': 'CRITICAL_FALLBACK_METHOD',
                    'code': line.strip(),
                    'reason': 'Production fallback method prevents real computation'
                })
            
            # CRITICAL: Mock classes in production
            if re.search(r'class\s+Mock\w+', line) and not self.is_test_file(filepath):
                violations.append({
                    'file': filepath,
                    'line': i,
                    'type': 'CRITICAL_MOCK_CLASS',
                    'code': line.strip(),
                    'reason': 'Mock class in production code prevents real learning'
                })
            
            # CRITICAL: Silent failures
            if re.search(r'except.*:\s*pass\s*$', line):
                violations.append({
                    'file': filepath,
                    'line': i,
                    'type': 'CRITICAL_SILENT_FAILURE',
                    'code': line.strip(),
                    'reason': 'Silent failure hides problems from the system'
                })
            
            # CRITICAL: Component availability flags that create fake paths
            if re.search(r'_AVAILABLE\s*=\s*False', line):
                violations.append({
                    'file': filepath,
                    'line': i,
                    'type': 'CRITICAL_FAKE_PATH',
                    'code': line.strip(),
                    'reason': 'Creates fake execution path when component missing'
                })
            
            # CRITICAL: Simplified algorithms that skip real work
            if 'simplified' in line.lower() and 'algorithm' in line.lower():
                violations.append({
                    'file': filepath,
                    'line': i,
                    'type': 'CRITICAL_SIMPLIFIED',
                    'code': line.strip(),
                    'reason': 'Simplified algorithm skips real computation'
                })
            
            # CRITICAL: Generate mock data in production
            if re.search(r'generate_mock_\w+|create_fake_\w+', line):
                violations.append({
                    'file': filepath,
                    'line': i,
                    'type': 'CRITICAL_MOCK_DATA',
                    'code': line.strip(),
                    'reason': 'Generating fake data prevents learning from real data'
                })
                
        return violations
    
    def check_borderline_patterns(self, filepath: str, content: str) -> List[Dict]:
        """Check patterns that might be acceptable depending on context"""
        warnings = []
        lines = content.split('\n')
        
        if self.is_test_file(filepath):
            return warnings
            
        for i, line in enumerate(lines, 1):
            # BORDERLINE: Emergency fallbacks (might be OK with logging)
            if 'emergency' in line.lower() and 'fallback' in line.lower():
                # Check if there's logging nearby
                surrounding = lines[max(0, i-3):min(len(lines), i+3)]
                has_logging = any('logger' in l or 'print' in l for l in surrounding)
                
                if not has_logging:
                    warnings.append({
                        'file': filepath,
                        'line': i,
                        'type': 'WARNING_EMERGENCY_NO_LOG',
                        'code': line.strip(),
                        'reason': 'Emergency fallback without logging'
                    })
            
            # BORDERLINE: Hardcoded values (might be config defaults)
            if re.search(r'=\s*\d+\.?\d*\s*#.*fallback', line):
                warnings.append({
                    'file': filepath,
                    'line': i,
                    'type': 'WARNING_HARDCODED',
                    'code': line.strip(),
                    'reason': 'Hardcoded value marked as fallback'
                })
                
        return warnings
    
    def validate_file(self, filepath: str) -> Tuple[List[Dict], List[Dict]]:
        """Validate a single file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            critical = self.check_critical_violations(filepath, content)
            warnings = self.check_borderline_patterns(filepath, content) if self.strict_mode else []
            
            return critical, warnings
        except Exception as e:
            return [], []
    
    def validate_codebase(self, root_dir: str = '.') -> Dict:
        """Validate entire codebase"""
        all_critical = []
        all_warnings = []
        files_checked = 0
        
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                filepath = os.path.join(root, file)
                files_checked += 1
                
                critical, warnings = self.validate_file(filepath)
                all_critical.extend(critical)
                all_warnings.extend(warnings)
        
        return {
            'files_checked': files_checked,
            'critical_violations': all_critical,
            'warnings': all_warnings,
            'summary': {
                'critical_count': len(all_critical),
                'warning_count': len(all_warnings),
                'critical_by_type': self._count_by_type(all_critical),
                'warnings_by_type': self._count_by_type(all_warnings)
            }
        }
    
    def _count_by_type(self, violations: List[Dict]) -> Dict[str, int]:
        """Count violations by type"""
        counts = defaultdict(int)
        for v in violations:
            counts[v['type']] += 1
        return dict(counts)


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart GAELP Fallback Validator')
    parser.add_argument('--strict', action='store_true', help='Enable strict mode with warnings')
    parser.add_argument('--fix', action='store_true', help='Show how to fix violations')
    parser.add_argument('--exclude-tests', action='store_true', help='Skip test files')
    parser.add_argument('path', nargs='?', default='.', help='Path to validate')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GAELP SMART FALLBACK VALIDATOR")
    print("=" * 80)
    print(f"Mode: {'STRICT' if args.strict else 'NORMAL'}")
    print(f"Path: {args.path}")
    print("-" * 80)
    
    validator = FallbackValidator(strict_mode=args.strict)
    results = validator.validate_codebase(args.path)
    
    print(f"\nScanned {results['files_checked']} Python files")
    print(f"\nCRITICAL VIOLATIONS: {results['summary']['critical_count']}")
    
    if results['summary']['critical_count'] > 0:
        print("\nViolations by type:")
        for vtype, count in results['summary']['critical_by_type'].items():
            print(f"  {vtype}: {count}")
        
        print("\nTop 10 Critical Violations:")
        for v in results['critical_violations'][:10]:
            print(f"  {v['file']}:{v['line']}")
            print(f"    Type: {v['type']}")
            print(f"    Code: {v['code'][:80]}")
            print(f"    Reason: {v['reason']}")
            print()
    
    if args.strict and results['summary']['warning_count'] > 0:
        print(f"\nWARNINGS: {results['summary']['warning_count']}")
        for vtype, count in results['summary']['warnings_by_type'].items():
            print(f"  {vtype}: {count}")
    
    # Return exit code based on critical violations only
    if results['summary']['critical_count'] > 0:
        print("\n❌ VALIDATION FAILED - Critical violations found")
        print("These violations prevent the system from actually learning.")
        
        if args.fix:
            print("\nTO FIX:")
            print("1. Replace fallback methods with proper error handling")
            print("2. Remove Mock classes from production code")
            print("3. Add logging to exception handlers instead of 'pass'")
            print("4. Remove _AVAILABLE flags that create fake execution paths")
            
        return 1
    else:
        print("\n✅ VALIDATION PASSED - No critical violations")
        print("The system can learn properly without fake execution.")
        return 0


if __name__ == "__main__":
    sys.exit(main())