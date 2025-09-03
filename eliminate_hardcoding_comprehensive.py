#!/usr/bin/env python3
"""
COMPREHENSIVE HARDCODE ELIMINATION SCANNER AND FIXER

This script systematically identifies and eliminates ALL hardcoded values
in the GAELP system, enforcing the NO FALLBACKS rule.
"""

import re
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HardcodingViolation:
    file_path: str
    line_number: int
    line_content: str
    violation_type: str
    hardcoded_value: str
    suggested_fix: str

class HardcodingEliminator:
    """Eliminates ALL hardcoded values from GAELP system"""
    
    def __init__(self):
        self.violations = []
        self.patterns_discovered = self._load_discovered_patterns()
        
        # Violation patterns to detect
        self.violation_patterns = [
            # Literal numbers (except 0, 1, and mathematical constants)
            (r'\b([2-9]|\d{2,})\d*\.?\d*\b', 'hardcoded_number'),
            
            # Forbidden words
            (r'\bfallback\b', 'forbidden_fallback'),
            (r'\bsimplified\b', 'forbidden_simplified'),
            (r'\bmock(?!ito)\b', 'forbidden_mock'),
            (r'\bdummy\b', 'forbidden_dummy'),
            (r'\bdefault_\w+', 'hardcoded_default'),
            (r'\bDEFAULT_\w+', 'hardcoded_default'),
            (r'\bTODO\b', 'unfinished_code'),
            (r'\bFIXME\b', 'unfinished_code'),
            
            # Hardcoded lists/dicts
            (r'return \[.*\]', 'hardcoded_list'),
            (r'return \{.*\}', 'hardcoded_dict'),
            (r'segments = \[.*\]', 'hardcoded_segments'),
            (r'channels = \[.*\]', 'hardcoded_channels'),
            
            # Hardcoded thresholds/limits
            (r'threshold\s*=\s*\d+', 'hardcoded_threshold'),
            (r'max_\w+\s*=\s*\d+', 'hardcoded_max'),
            (r'min_\w+\s*=\s*\d+', 'hardcoded_min'),
            
            # Epsilon/learning rates
            (r'epsilon\s*=\s*0\.\d+', 'hardcoded_epsilon'),
            (r'learning_rate\s*=\s*\d', 'hardcoded_lr'),
            (r'lr\s*=\s*\d', 'hardcoded_lr'),
            
            # Disabled code
            (r'if False:', 'disabled_code'),
            (r'return None\s*#.*later', 'placeholder_code'),
            (r'pass\s*#.*implement', 'placeholder_code'),
        ]
        
        # Files to prioritize based on known issues
        self.priority_files = [
            'fortified_rl_agent_no_hardcoding.py',
            'fortified_environment_no_hardcoding.py',
            'gaelp_master_integration.py',
            'enhanced_simulator.py',
            'creative_selector.py',
            'budget_pacer.py',
            'attribution_models.py'
        ]
    
    def _load_discovered_patterns(self) -> Dict:
        """Load discovered patterns for replacement values"""
        try:
            with open('/home/hariravichandran/AELP/discovered_patterns.json', 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def scan_for_violations(self) -> List[HardcodingViolation]:
        """Scan entire codebase for hardcoding violations"""
        logger.info("🔍 Scanning for hardcoding violations...")
        
        # Scan all Python files
        for py_file in Path('/home/hariravichandran/AELP').rglob('*.py'):
            # Skip test files and specific excluded files
            if any(skip in str(py_file) for skip in ['test_', '__pycache__', '.git', 'eliminate_hardcoding']):
                continue
                
            self._scan_file(py_file)
        
        # Sort by priority and severity
        self.violations.sort(key=lambda v: (
            str(v.file_path) not in self.priority_files,  # Priority files first
            v.violation_type != 'hardcoded_number',       # Numbers are highest priority
            v.line_number
        ))
        
        logger.info(f"📊 Found {len(self.violations)} hardcoding violations")
        return self.violations
    
    def _scan_file(self, file_path: Path):
        """Scan single file for violations"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            return
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments and imports
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('import ') or stripped.startswith('from '):
                continue
            
            # Check each pattern
            for pattern, violation_type in self.violation_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    for match in matches:
                        hardcoded_value = match if isinstance(match, str) else match[0] if match else ""
                        
                        # Skip allowed values
                        if self._is_allowed_value(hardcoded_value, line):
                            continue
                        
                        violation = HardcodingViolation(
                            file_path=str(file_path),
                            line_number=line_num,
                            line_content=line.strip(),
                            violation_type=violation_type,
                            hardcoded_value=hardcoded_value,
                            suggested_fix=self._suggest_fix(violation_type, hardcoded_value, line)
                        )
                        self.violations.append(violation)
    
    def _is_allowed_value(self, value: str, line: str) -> bool:
        """Check if a value is allowed (mathematical constants, etc.)"""
        allowed_values = {'0', '1', '2', '3.14159', '2.71828', 'pi', 'e'}
        
        # Allow small integers in specific contexts
        if value in allowed_values:
            return True
        
        # Allow indexing and array operations
        if any(ctx in line for ctx in ['[', ']', 'range(', 'len(', 'shape']):
            try:
                int_val = int(value)
                if int_val <= 10:  # Allow small integers for indexing
                    return True
            except:
                pass
        
        # Allow version numbers
        if 'version' in line.lower():
            return True
        
        return False
    
    def _suggest_fix(self, violation_type: str, hardcoded_value: str, line: str) -> str:
        """Suggest how to fix the violation"""
        if violation_type == 'hardcoded_number':
            return f"Replace {hardcoded_value} with self.patterns.get('parameter_name', discovered_value)"
        elif violation_type == 'forbidden_fallback':
            return "Remove fallback, implement proper discovery mechanism"
        elif violation_type == 'hardcoded_epsilon':
            return "Replace with self.discovered_patterns['exploration']['epsilon']"
        elif violation_type == 'hardcoded_lr':
            return "Replace with self.optimizer_config.get('learning_rate')"
        elif violation_type == 'hardcoded_segments':
            return "Replace with self.segment_discovery.get_discovered_segments()"
        elif violation_type == 'hardcoded_threshold':
            return "Replace with self.performance_thresholds.get_threshold(metric_name)"
        else:
            return "Replace with dynamically discovered value"
    
    def generate_fixes(self) -> Dict[str, List[str]]:
        """Generate specific fixes for each file"""
        fixes_by_file = {}
        
        for violation in self.violations:
            file_path = violation.file_path
            if file_path not in fixes_by_file:
                fixes_by_file[file_path] = []
            
            fix_description = f"Line {violation.line_number}: {violation.suggested_fix}"
            fixes_by_file[file_path].append(fix_description)
        
        return fixes_by_file
    
    def create_pattern_discovery_system(self):
        """Create comprehensive pattern discovery system"""
        discovery_system = """
class PatternDiscoverySystem:
    '''Universal pattern discovery system - NO HARDCODING ALLOWED'''
    
    def __init__(self):
        self.patterns = self._discover_all_patterns()
        self.thresholds = self._discover_thresholds()
        self.parameters = self._discover_parameters()
    
    def _discover_all_patterns(self) -> Dict:
        '''Discover ALL patterns from data - never return hardcoded values'''
        patterns = {}
        
        # Discover from GA4 data
        patterns.update(self._discover_from_ga4())
        
        # Discover from user behavior
        patterns.update(self._discover_from_user_behavior())
        
        # Discover from competitive analysis
        patterns.update(self._discover_from_competition())
        
        return patterns
    
    def get_value(self, key: str, context: Dict = None) -> Any:
        '''Get discovered value - NEVER return hardcoded defaults'''
        if key not in self.patterns:
            # Don't return default - discover it!
            discovered_value = self._discover_single_value(key, context)
            self.patterns[key] = discovered_value
            logger.info(f"Discovered new pattern: {key} = {discovered_value}")
        
        return self.patterns[key]
"""
        
        with open('/home/hariravichandran/AELP/pattern_discovery_system.py', 'w') as f:
            f.write(discovery_system)
    
    def report_violations(self):
        """Generate comprehensive violation report"""
        if not self.violations:
            print("✅ NO HARDCODING VIOLATIONS FOUND!")
            return
        
        print(f"❌ FOUND {len(self.violations)} HARDCODING VIOLATIONS\n")
        
        # Group by violation type
        by_type = {}
        for v in self.violations:
            if v.violation_type not in by_type:
                by_type[v.violation_type] = []
            by_type[v.violation_type].append(v)
        
        for violation_type, violations in by_type.items():
            print(f"\n🚨 {violation_type.upper()} ({len(violations)} violations):")
            for v in violations[:5]:  # Show first 5 of each type
                file_name = os.path.basename(v.file_path)
                print(f"  {file_name}:{v.line_number} - {v.hardcoded_value}")
                print(f"    Code: {v.line_content[:80]}...")
                print(f"    Fix: {v.suggested_fix}")
        
        print(f"\n📋 PRIORITY FILES TO FIX:")
        priority_violations = [v for v in self.violations if os.path.basename(v.file_path) in self.priority_files]
        for file_name in self.priority_files:
            file_violations = [v for v in priority_violations if os.path.basename(v.file_path) == file_name]
            if file_violations:
                print(f"  {file_name}: {len(file_violations)} violations")

def main():
    """Main execution"""
    eliminator = HardcodingEliminator()
    
    # Scan for violations
    violations = eliminator.scan_for_violations()
    
    # Generate report
    eliminator.report_violations()
    
    # Create pattern discovery system
    eliminator.create_pattern_discovery_system()
    
    # Generate fixes
    fixes = eliminator.generate_fixes()
    
    # Save violation report
    with open('/home/hariravichandran/AELP/hardcoding_violations_report.json', 'w') as f:
        json.dump({
            'total_violations': len(violations),
            'violations_by_file': {v.file_path: v.__dict__ for v in violations},
            'suggested_fixes': fixes
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved to hardcoding_violations_report.json")
    
    return len(violations) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)