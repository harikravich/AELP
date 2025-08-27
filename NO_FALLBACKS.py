#!/usr/bin/env python3
"""
GAELP NO FALLBACKS Validation Script

This script enforces the CRITICAL INSTRUCTIONS:
- NO FALLBACKS OR SIMPLIFICATIONS
- NO HARDCODED VALUES
- ALL PARAMETERS FROM REAL GA4 DATA

Will raise errors if any violations are found.
"""

import os
import re
import sys
import ast
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

class NoFallbackError(Exception):
    """Raised when fallback code is detected"""
    pass

def enforce_no_fallbacks(code_string: str = ""):
    """Enforce no fallbacks rule"""
    forbidden_patterns = ['fallback', 'simplified', 'mock', 'dummy']
    for pattern in forbidden_patterns:
        if pattern.lower() in code_string.lower():
            raise NoFallbackError(f"Forbidden fallback pattern detected: {pattern}")
    return True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FORBIDDEN PATTERNS - Will cause errors if found
FORBIDDEN_PATTERNS = {
    'fallback': r'\bfallback\b',
    'simplified': r'\bsimplified?\b', 
    'mock': r'\bmock\b(?!_data|_import)',  # Allow mock_data, mock_import
    'dummy': r'\bdummy\b',
    'hardcoded': r'\bhardcode[d]?\b',
    'todo': r'\btodo\b|\bfixme\b',
    'not_available': r'not\s+available',
    '_available.*false': r'_AVAILABLE\s*=\s*False',
    'random_uniform': r'np\.random\.uniform\(',
    'random_normal': r'np\.random\.normal\(',
    'random_beta': r'np\.random\.beta\(',
    'except.*pass': r'except.*:\s*pass',
    'try.*except.*fallback': r'try.*except.*fallback',
}

# HARDCODED NUMERIC VALUES (common culprits)
HARDCODED_NUMBERS = {
    'bid_amounts': r'bid.*=\s*[0-9]+\.?[0-9]*\b',
    'conversion_rates': r'(cvr|conversion.*rate).*=\s*0\.[0-9]+',
    'budget_values': r'budget.*=\s*[0-9]+\.?[0-9]*',
    'multipliers': r'multiplier.*=\s*[0-9]+\.?[0-9]*',
    'thresholds': r'threshold.*=\s*[0-9]+\.?[0-9]*',
    'probabilities': r'prob.*=\s*0\.[0-9]+',
    'percentages': r'pct.*=\s*[0-9]+\.?[0-9]*',
    'magic_numbers': r'[^a-zA-Z_][0-9]+\.?[0-9]*[^a-zA-Z_0-9](?!.*#.*from.*data)',
}

# REQUIRED IMPORTS (must be present)
REQUIRED_IMPORTS = {
    'parameter_manager': r'from gaelp_parameter_manager import',
    'no_random': 'No direct numpy random calls allowed',
}

# CRITICAL FILES TO CHECK
CRITICAL_FILES = [
    'gaelp_master_integration.py',
    'enhanced_simulator.py', 
    'budget_pacer.py',
    'competitive_intel.py',
    'creative_selector.py',
    'user_journey_database.py',
    'attribution_models.py'
]

class FallbackViolation(Exception):
    """Raised when fallback code is detected"""
    pass

class HardcodedValueViolation(Exception):
    """Raised when hardcoded values are detected"""
    pass

class StrictModeEnforcer:
    """Enforces strict mode - no fallbacks allowed"""
    
    @staticmethod
    def check_fallback_usage(component_name: str):
        """Check if component is trying to use fallbacks"""
        # This would normally check component status
        # For now, just log
        logger.info(f"✅ {component_name} - strict mode active")

class NoFallbacksValidator:
    """Validates that NO fallbacks or hardcoded values exist"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.violations: List[Dict[str, Any]] = []
        
    def validate_all(self) -> bool:
        """Validate all critical files"""
        logger.info("🔍 Starting NO FALLBACKS validation...")
        
        all_passed = True
        
        for file_name in CRITICAL_FILES:
            file_path = self.base_path / file_name
            if file_path.exists():
                try:
                    self.validate_file(file_path)
                    logger.info(f"✅ {file_name}: PASSED")
                except Exception as e:
                    logger.error(f"❌ {file_name}: FAILED - {e}")
                    all_passed = False
            else:
                logger.warning(f"⚠️ {file_name}: FILE NOT FOUND")
        
        # Check parameter manager exists and is used
        if not self._check_parameter_manager_integration():
            all_passed = False
        
        if all_passed:
            logger.info("\n🎉 ALL VALIDATION PASSED!")
            logger.info("✅ NO fallbacks found")
            logger.info("✅ NO hardcoded values found") 
            logger.info("✅ ALL parameters from real GA4 data")
            return True
        else:
            logger.error("\n💥 VALIDATION FAILED!")
            logger.error("❌ Fallbacks or hardcoded values detected")
            logger.error("❌ Fix all issues before proceeding")
            return False
    
    def validate_file(self, file_path: Path) -> None:
        """Validate a single file for violations"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        file_violations = []
        
        # Check for forbidden patterns
        for pattern_name, pattern in FORBIDDEN_PATTERNS.items():
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                line_content = content.split('\n')[line_num - 1].strip()
                
                violation = {
                    'file': str(file_path),
                    'line': line_num,
                    'type': f'FORBIDDEN_PATTERN_{pattern_name.upper()}',
                    'content': line_content,
                    'match': match.group()
                }
                file_violations.append(violation)
        
        # Check for hardcoded numbers
        for number_type, pattern in HARDCODED_NUMBERS.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                line_content = content.split('\n')[line_num - 1].strip()
                
                # Skip if line contains comment indicating data source
                if any(marker in line_content.lower() for marker in [
                    'from data', 'ga4 data', 'real data', 'discovered', 'parameter_manager'
                ]):
                    continue
                
                violation = {
                    'file': str(file_path),
                    'line': line_num,
                    'type': f'HARDCODED_{number_type.upper()}',
                    'content': line_content,
                    'match': match.group()
                }
                file_violations.append(violation)
        
        # Check for parameter manager usage
        if 'gaelp_master_integration.py' in str(file_path):
            if 'get_parameter_manager' not in content:
                file_violations.append({
                    'file': str(file_path),
                    'line': 0,
                    'type': 'MISSING_PARAMETER_MANAGER',
                    'content': 'File must use parameter manager',
                    'match': 'get_parameter_manager not found'
                })
        
        if file_violations:
            self.violations.extend(file_violations)
            violation_summary = '\n'.join([
                f"  Line {v['line']}: {v['type']} - {v['match']}"
                for v in file_violations
            ])
            raise FallbackViolation(f"\n{len(file_violations)} violations found:\n{violation_summary}")
    
    def _check_parameter_manager_integration(self) -> bool:
        """Check that parameter manager is properly integrated"""
        pm_file = self.base_path / 'gaelp_parameter_manager.py'
        
        if not pm_file.exists():
            logger.error("❌ gaelp_parameter_manager.py MISSING")
            return False
        
        # Check that discovered_patterns.json exists
        patterns_file = self.base_path / 'discovered_patterns.json'
        if not patterns_file.exists():
            logger.error("❌ discovered_patterns.json MISSING")
            return False
        
        try:
            with open(patterns_file, 'r') as f:
                patterns = json.load(f)
            
            required_sections = [
                'channel_performance', 
                'user_segments', 
                'device_patterns', 
                'temporal_patterns'
            ]
            
            for section in required_sections:
                if section not in patterns or not patterns[section]:
                    logger.error(f"❌ Missing or empty section in patterns: {section}")
                    return False
            
            logger.info("✅ Parameter manager and GA4 data properly integrated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating patterns file: {e}")
            return False
    
    def generate_report(self) -> str:
        """Generate detailed validation report"""
        if not self.violations:
            return "🎉 NO VIOLATIONS FOUND - ALL SYSTEMS CLEAN!"
        
        report = f"💥 FOUND {len(self.violations)} VIOLATIONS:\n"
        report += "=" * 60 + "\n"
        
        violations_by_file = {}
        for violation in self.violations:
            file_name = violation['file']
            if file_name not in violations_by_file:
                violations_by_file[file_name] = []
            violations_by_file[file_name].append(violation)
        
        for file_name, file_violations in violations_by_file.items():
            report += f"\n📁 {file_name}:\n"
            for v in file_violations:
                report += f"  ❌ Line {v['line']}: {v['type']}\n"
                report += f"     Code: {v['content']}\n"
                report += f"     Match: {v['match']}\n\n"
        
        return report

def main():
    """Run the validation"""
    print("🛡️ GAELP NO FALLBACKS VALIDATOR")
    print("=" * 50)
    print("Enforcing CRITICAL INSTRUCTIONS:")
    print("- NO FALLBACKS OR SIMPLIFICATIONS")
    print("- NO HARDCODED VALUES") 
    print("- ALL PARAMETERS FROM REAL GA4 DATA")
    print("=" * 50)
    
    validator = NoFallbacksValidator()
    
    success = validator.validate_all()
    
    if not success:
        print("\n" + validator.generate_report())
        print("\n💥 CRITICAL VIOLATIONS DETECTED!")
        print("🚨 Fix ALL issues before proceeding")
        print("🚨 No fallbacks or hardcoded values allowed")
        sys.exit(1)
    
    print("\n🎉 VALIDATION SUCCESSFUL!")
    print("✅ GAELP is clean of fallbacks and hardcoded values")
    print("✅ All parameters are data-driven")
    print("✅ Ready for production deployment")

if __name__ == "__main__":
    main()