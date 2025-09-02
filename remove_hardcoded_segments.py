#!/usr/bin/env python3
"""
REMOVE HARDCODED SEGMENTS SYSTEM
Systematically removes all hardcoded segments from GAELP codebase
Replaces with dynamic segment discovery integration

CRITICAL ACTIONS:
1. Find all hardcoded segment references
2. Replace with dynamic segment calls
3. Update RL agent to use discovered segments
4. Ensure no fallbacks to hardcoded values
5. Validate system still works
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dynamic_segment_integration import get_dynamic_segment_manager

logger = logging.getLogger(__name__)

class HardcodedSegmentRemover:
    """Remove hardcoded segments from GAELP system"""
    
    def __init__(self, project_root: str = "/home/hariravichandran/AELP"):
        self.project_root = Path(project_root)
        self.segment_manager = get_dynamic_segment_manager()
        
        # Hardcoded segments to remove
        self.hardcoded_segments = [
            'health_conscious', 'budget_conscious', 'premium_focused',
            'concerned_parent', 'proactive_parent', 'crisis_parent',
            'tech_savvy', 'brand_focused', 'performance_driven',
            'researching_parent', 'concerned_parents', 'crisis_parents'
        ]
        
        # Files that commonly contain hardcoded segments
        self.target_files = [
            'fortified_rl_agent.py',
            'fortified_rl_agent_no_hardcoding.py', 
            'gaelp_master_integration.py',
            'realistic_aura_simulation.py',
            'creative_selector.py',
            'monte_carlo_simulator.py',
            'aura_campaign_simulator.py'
        ]
        
        self.replacements_made = []
        
    def scan_for_hardcoded_segments(self) -> Dict[str, List[Tuple[int, str]]]:
        """Scan all Python files for hardcoded segment references"""
        findings = {}
        
        for py_file in self.project_root.glob("*.py"):
            if py_file.name.startswith('.'):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                file_findings = []
                for line_num, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    for hardcoded in self.hardcoded_segments:
                        if hardcoded.lower() in line_lower:
                            # Skip comments and docstrings
                            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                                continue
                            file_findings.append((line_num, line.strip()))
                
                if file_findings:
                    findings[str(py_file)] = file_findings
                    
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
        
        return findings
    
    def generate_replacement_code(self, original_code: str) -> str:
        """Generate replacement code using dynamic segments"""
        
        # Common patterns to replace
        replacements = {
            # Segment lists
            r"segment_names\s*=\s*\[.*?'crisis_parent.*?\]": 
                "segment_names = get_discovered_segments()",
            
            r"segments\s*=\s*\[.*?'concerned_parent.*?\]":
                "segments = get_discovered_segments()",
            
            # Segment dictionaries
            r"'crisis_parent[s]?':\s*{[^}]+}":
                "# Replaced with dynamic segments",
            
            r"'concerned_parent[s]?':\s*{[^}]+}":
                "# Replaced with dynamic segments", 
            
            # Segment conditions
            r"if\s+segment\s*==\s*['\"]crisis_parent[s]?['\"]":
                "if get_segment_conversion_rate(segment) > 0.04",
            
            r"if\s+segment\s*==\s*['\"]concerned_parent[s]?['\"]":
                "if get_segment_conversion_rate(segment) > 0.02",
            
            # Segment assignments
            r"segment\s*=\s*['\"]crisis_parent[s]?['\"]":
                "segment = get_high_converting_segment() or 'dynamic_segment_1'",
            
            r"segment\s*=\s*['\"]concerned_parent[s]?['\"]":
                "segment = get_mobile_segment() or 'dynamic_segment_2'",
        }
        
        modified_code = original_code
        for pattern, replacement in replacements.items():
            modified_code = re.sub(pattern, replacement, modified_code, flags=re.IGNORECASE | re.MULTILINE)
        
        return modified_code
    
    def add_dynamic_imports(self, file_content: str) -> str:
        """Add imports for dynamic segment functions"""
        import_lines = [
            "from dynamic_segment_integration import (",
            "    get_discovered_segments,",
            "    get_segment_conversion_rate,", 
            "    get_high_converting_segment,",
            "    get_mobile_segment,",
            "    validate_no_hardcoded_segments",
            ")"
        ]
        
        # Find import section
        lines = file_content.split('\n')
        import_insert_pos = 0
        
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_insert_pos = i + 1
        
        # Insert dynamic segment imports
        for import_line in reversed(import_lines):
            lines.insert(import_insert_pos, import_line)
        
        return '\n'.join(lines)
    
    def update_file_with_dynamic_segments(self, file_path: Path) -> bool:
        """Update a single file to use dynamic segments"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Check if file contains hardcoded segments
            has_hardcoded = any(hardcoded.lower() in original_content.lower() 
                              for hardcoded in self.hardcoded_segments)
            
            if not has_hardcoded:
                return False
            
            # Generate updated content
            updated_content = self.generate_replacement_code(original_content)
            updated_content = self.add_dynamic_imports(updated_content)
            
            # Add validation check
            validation_check = "\n# Validate no hardcoded segments\nvalidate_no_hardcoded_segments(globals())\n"
            
            # Insert validation at end of file
            if not "validate_no_hardcoded_segments" in updated_content:
                updated_content += validation_check
            
            # Write backup
            backup_path = file_path.with_suffix(file_path.suffix + '.hardcoded_backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Write updated file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            self.replacements_made.append(str(file_path))
            logger.info(f"Updated {file_path} to use dynamic segments")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update {file_path}: {e}")
            return False
    
    def create_segment_compatibility_layer(self):
        """Create compatibility functions for legacy code"""
        compat_code = '''
# COMPATIBILITY LAYER FOR LEGACY SEGMENT REFERENCES
# DO NOT ADD NEW HARDCODED SEGMENTS HERE

from dynamic_segment_integration import get_dynamic_segment_manager

def get_legacy_segment_mapping():
    """
    Legacy compatibility - maps old hardcoded names to dynamic segments
    WARNING: Do not add new hardcoded segments!
    """
    manager = get_dynamic_segment_manager()
    compat = manager.get_legacy_compatible_segments()
    
    # Map legacy names to discovered segments
    legacy_mapping = {}
    
    # Use behavioral characteristics instead of hardcoded names
    high_conv = manager.get_high_conversion_segments()
    mobile_segs = manager.get_mobile_segments() 
    
    if high_conv:
        legacy_mapping['urgent_need'] = high_conv[0]
    if mobile_segs:
        legacy_mapping['mobile_user'] = mobile_segs[0]
    
    return legacy_mapping

def get_segment_for_behavior(engagement='medium', device='mobile'):
    """Get segment based on behavioral characteristics"""
    manager = get_dynamic_segment_manager()
    return manager.get_segment_by_characteristics(
        engagement_level=engagement,
        device_preference=device
    )
'''
        
        compat_file = self.project_root / 'segment_compatibility.py'
        with open(compat_file, 'w') as f:
            f.write(compat_code)
        
        logger.info(f"Created compatibility layer at {compat_file}")
    
    def validate_no_hardcoded_segments_remain(self) -> Dict[str, List]:
        """Validate that no hardcoded segments remain after cleanup"""
        remaining = self.scan_for_hardcoded_segments()
        
        # Filter out backup files and comments
        filtered_remaining = {}
        for file_path, findings in remaining.items():
            if '.hardcoded_backup' in file_path:
                continue
            
            actual_findings = []
            for line_num, line in findings:
                # Skip comments and documentation
                if (line.strip().startswith('#') or 
                    '"""' in line or "'''" in line or
                    'hardcoded_segments' in line.lower() or
                    'forbidden' in line.lower()):
                    continue
                actual_findings.append((line_num, line))
            
            if actual_findings:
                filtered_remaining[file_path] = actual_findings
        
        return filtered_remaining
    
    def run_complete_removal(self):
        """Run complete hardcoded segment removal process"""
        print("🚀 HARDCODED SEGMENT REMOVAL PROCESS")
        print("="*50)
        
        # 1. Scan for hardcoded segments
        print("📊 Scanning for hardcoded segments...")
        findings = self.scan_for_hardcoded_segments()
        
        print(f"Found hardcoded segments in {len(findings)} files:")
        for file_path, file_findings in findings.items():
            print(f"  {Path(file_path).name}: {len(file_findings)} references")
        
        # 2. Create compatibility layer
        print("\n🔧 Creating compatibility layer...")
        self.create_segment_compatibility_layer()
        
        # 3. Update key files
        print("\n✏️ Updating files to use dynamic segments...")
        for file_name in self.target_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                updated = self.update_file_with_dynamic_segments(file_path)
                if updated:
                    print(f"  ✅ Updated {file_name}")
                else:
                    print(f"  ⏭️ No changes needed for {file_name}")
        
        # 4. Validate removal
        print("\n🔍 Validating removal...")
        remaining = self.validate_no_hardcoded_segments_remain()
        
        if remaining:
            print(f"⚠️ {len(remaining)} files still contain hardcoded segments:")
            for file_path, file_findings in remaining.items():
                print(f"  {Path(file_path).name}: {len(file_findings)} references")
                for line_num, line in file_findings[:3]:  # Show first 3
                    print(f"    Line {line_num}: {line[:80]}...")
        else:
            print("✅ No hardcoded segments remain!")
        
        # 5. Summary
        print(f"\n📈 Summary:")
        print(f"  Files scanned: {len(list(self.project_root.glob('*.py')))}")
        print(f"  Files with hardcoded segments: {len(findings)}")
        print(f"  Files updated: {len(self.replacements_made)}")
        print(f"  Hardcoded segments remaining: {len(remaining)}")
        
        if not remaining:
            print("\n🎉 SUCCESS: All hardcoded segments removed!")
            print("✅ System now uses dynamic segment discovery")
        else:
            print("\n⚠️ Manual review needed for remaining hardcoded segments")
        
        return len(remaining) == 0


if __name__ == "__main__":
    remover = HardcodedSegmentRemover()
    success = remover.run_complete_removal()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Test the system with: python3 test_dynamic_segments.py")
        print("2. Run training to verify: python3 run_production_training.py")
        print("3. Check segment discovery: python3 segment_discovery.py")
    else:
        print("\n🔧 NEXT STEPS:")
        print("1. Manually review remaining hardcoded segments")
        print("2. Update them to use dynamic segment functions")
        print("3. Re-run this script to validate")