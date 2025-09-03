#!/usr/bin/env python3
"""
PRIORITY FILES HARDCODING SCANNER

Focused scan on priority files to identify remaining violations
"""

import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scan_priority_file(file_path: Path) -> dict:
    """Scan a single priority file for hardcoded violations"""
    violations = {
        'hardcoded_numbers': [],
        'forbidden_words': [],
        'thresholds': [],
        'epsilon_lr': [],
        'other': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        return violations
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip comments and imports
        if stripped.startswith('#') or stripped.startswith('import ') or stripped.startswith('from '):
            continue
        
        # Check for hardcoded numbers (excluding allowed ones)
        numbers = re.findall(r'\b([2-9]|\d{2,})\d*\.?\d*\b', line)
        for num in numbers:
            # Skip allowed contexts
            if any(ctx in line for ctx in ['[', ']', 'range(', 'len(', 'shape', 'version']):
                continue
            violations['hardcoded_numbers'].append(f"Line {line_num}: {num} in '{stripped[:60]}...'")
        
        # Check for forbidden words
        forbidden = ['fallback', 'simplified', 'mock', 'dummy', 'TODO', 'FIXME']
        for word in forbidden:
            if re.search(rf'\b{word}\b', line, re.IGNORECASE):
                violations['forbidden_words'].append(f"Line {line_num}: '{word}' in '{stripped[:60]}...'")
        
        # Check for specific threshold patterns
        if re.search(r'threshold\s*=\s*\d', line):
            violations['thresholds'].append(f"Line {line_num}: hardcoded threshold in '{stripped[:60]}...'")
        
        # Check for epsilon/learning rate patterns
        if re.search(r'(epsilon|learning_rate|lr)\s*=\s*[0-9]', line):
            violations['epsilon_lr'].append(f"Line {line_num}: hardcoded param in '{stripped[:60]}...'")
        
        # Check for other patterns
        patterns = [r'return \[\]', r'return \{\}', r'= 0\.[0-9]+', r'= [2-9]\d*']
        for pattern in patterns:
            if re.search(pattern, line):
                violations['other'].append(f"Line {line_num}: pattern '{pattern}' in '{stripped[:60]}...'")
    
    return violations

def main():
    """Scan all priority files"""
    priority_files = [
        'fortified_rl_agent_no_hardcoding.py',
        'fortified_environment_no_hardcoding.py', 
        'gaelp_master_integration.py',
        'enhanced_simulator.py',
        'creative_selector.py',
        'budget_pacer.py',
        'attribution_models.py'
    ]
    
    base_path = Path('/home/hariravichandran/AELP')
    total_violations = 0
    
    print("🔍 SCANNING PRIORITY FILES FOR HARDCODING VIOLATIONS\n")
    
    for file_name in priority_files:
        file_path = base_path / file_name
        if not file_path.exists():
            continue
        
        violations = scan_priority_file(file_path)
        file_violations = sum(len(v) for v in violations.values())
        total_violations += file_violations
        
        print(f"📄 {file_name}: {file_violations} violations")
        
        for category, items in violations.items():
            if items:
                print(f"  {category.replace('_', ' ').title()}: {len(items)}")
                for item in items[:3]:  # Show first 3 of each type
                    print(f"    {item}")
                if len(items) > 3:
                    print(f"    ... and {len(items) - 3} more")
        print()
    
    print(f"📊 TOTAL VIOLATIONS IN PRIORITY FILES: {total_violations}")
    
    if total_violations == 0:
        print("✅ NO HARDCODING VIOLATIONS FOUND IN PRIORITY FILES!")
    else:
        print("❌ HARDCODING VIOLATIONS REMAIN - CONTINUE ELIMINATION")
    
    return total_violations == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)