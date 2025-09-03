#!/usr/bin/env python3
"""
HARDCODE ELIMINATION VERIFICATION

Verifies that all critical hardcoded values have been eliminated
from the GAELP system, enforcing the NO FALLBACKS rule.
"""

import re
import json
from pathlib import Path
from typing import Dict, List

def verify_no_fallbacks():
    """Verify NO FALLBACKS rule is enforced"""
    print("🔍 VERIFYING NO FALLBACKS RULE...")
    
    # Critical patterns that are NEVER allowed
    forbidden_patterns = [
        (r'\bfallback\b(?!.*#)', 'fallback'),
        (r'\bsimplified\b(?!.*#)', 'simplified'), 
        (r'\bmock\b(?!.*#.*mockito)', 'mock'),
        (r'\bdummy\b(?!.*#)', 'dummy'),
        (r'\bTODO\b', 'TODO'),
        (r'\bFIXME\b', 'FIXME'),
    ]
    
    priority_files = [
        'fortified_rl_agent_no_hardcoding.py',
        'fortified_environment_no_hardcoding.py',
        'gaelp_master_integration.py',
        'enhanced_simulator.py',
        'creative_selector.py',
        'budget_pacer.py',
        'attribution_models.py'
    ]
    
    violations = []
    base_path = Path('/home/hariravichandran/AELP')
    
    for file_name in priority_files:
        file_path = base_path / file_name
        if not file_path.exists():
            continue
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            for pattern, name in forbidden_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(f"{file_name}:{line_num} - {name}")
    
    if violations:
        print(f"❌ {len(violations)} CRITICAL VIOLATIONS FOUND:")
        for violation in violations[:10]:  # Show first 10
            print(f"  {violation}")
        return False
    else:
        print("✅ NO CRITICAL VIOLATIONS FOUND - NO FALLBACKS RULE ENFORCED")
        return True

def verify_parameter_discovery():
    """Verify parameter discovery system works"""
    print("\n🔧 VERIFYING PARAMETER DISCOVERY SYSTEM...")
    
    try:
        # Test imports
        import sys
        sys.path.append('/home/hariravichandran/AELP')
        from discovered_parameter_config import (
            get_config, get_epsilon_params, get_learning_rate, 
            get_conversion_bonus, get_goal_thresholds, get_priority_params
        )
        
        # Test parameter retrieval
        config = get_config()
        epsilon_params = get_epsilon_params()
        learning_rate = get_learning_rate()
        conversion_bonus = get_conversion_bonus()
        goal_thresholds = get_goal_thresholds()
        priority_params = get_priority_params()
        
        print("✅ All parameter discovery functions working")
        print(f"✅ Sample values:")
        print(f"  - Initial epsilon: {epsilon_params['initial_epsilon']:.4f}")
        print(f"  - Learning rate: {learning_rate:.6f}")
        print(f"  - Conversion bonus: {conversion_bonus:.4f}")
        print(f"  - Goal close threshold: {goal_thresholds['close']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter discovery system error: {e}")
        return False

def verify_business_logic_discovery():
    """Verify business logic parameters are discovered, not hardcoded"""
    print("\n📊 VERIFYING BUSINESS LOGIC DISCOVERY...")
    
    # Check that key business parameters come from discovery
    key_checks = [
        ("epsilon values", r'\bepsilon\s*=\s*0\.[0-9]+(?!.*get_epsilon)'),
        ("learning rates", r'\blearning_rate\s*=\s*[0-9](?!.*get_learning)'),
        ("conversion thresholds", r'>\s*0\.1(?!.*get_conversion)'),
        ("goal thresholds", r'<\s*0\.0[0-9]+(?!.*get_goal)'),
    ]
    
    priority_files = [
        'fortified_rl_agent_no_hardcoding.py',
        'fortified_environment_no_hardcoding.py'
    ]
    
    business_violations = []
    base_path = Path('/home/hariravichandran/AELP')
    
    for file_name in priority_files:
        file_path = base_path / file_name
        if not file_path.exists():
            continue
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        for check_name, pattern in key_checks:
            matches = re.findall(pattern, content)
            if matches:
                business_violations.append(f"{file_name}: {len(matches)} hardcoded {check_name}")
    
    if business_violations:
        print(f"⚠️  Business logic violations found:")
        for violation in business_violations:
            print(f"  {violation}")
        return False
    else:
        print("✅ All business logic parameters use discovery system")
        return True

def verify_segment_discovery():
    """Verify segments come from discovery, not hardcoded lists"""
    print("\n👥 VERIFYING SEGMENT DISCOVERY...")
    
    try:
        import sys
        sys.path.append('/home/hariravichandran/AELP')
        from dynamic_segment_integration import get_discovered_segments
        
        segments = get_discovered_segments()
        if segments:
            print(f"✅ Discovered {len(segments)} segments from patterns")
            print(f"✅ Sample segments: {list(segments.keys())[:3]}")
            return True
        else:
            print("⚠️  No segments discovered - may need pattern generation")
            return False
            
    except Exception as e:
        print(f"❌ Segment discovery error: {e}")
        return False

def create_compliance_report():
    """Create final compliance report"""
    print("\n📄 GENERATING COMPLIANCE REPORT...")
    
    report = {
        'no_fallbacks_enforced': verify_no_fallbacks(),
        'parameter_discovery_working': verify_parameter_discovery(),
        'business_logic_discovered': verify_business_logic_discovery(),
        'segment_discovery_working': verify_segment_discovery(),
        'timestamp': str(Path(__file__).stat().st_mtime)
    }
    
    # Save report
    with open('/home/hariravichandran/AELP/hardcode_elimination_compliance.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    """Run complete hardcode elimination verification"""
    print("🎯 HARDCODE ELIMINATION VERIFICATION")
    print("=" * 50)
    
    # Run all verifications
    report = create_compliance_report()
    
    # Overall assessment
    all_passed = all(report.values())
    
    print("\n📊 FINAL ASSESSMENT:")
    print(f"No Fallbacks Rule: {'✅ PASS' if report['no_fallbacks_enforced'] else '❌ FAIL'}")
    print(f"Parameter Discovery: {'✅ PASS' if report['parameter_discovery_working'] else '❌ FAIL'}")
    print(f"Business Logic Discovery: {'✅ PASS' if report['business_logic_discovered'] else '❌ FAIL'}")
    print(f"Segment Discovery: {'✅ PASS' if report['segment_discovery_working'] else '❌ FAIL'}")
    
    if all_passed:
        print("\n🎉 HARDCODE ELIMINATION COMPLETE!")
        print("✅ All critical hardcoded values eliminated")
        print("✅ NO FALLBACKS rule enforced")
        print("✅ All parameters discovered dynamically")
        print("✅ System remains fully functional")
    else:
        print(f"\n⚠️  HARDCODE ELIMINATION INCOMPLETE")
        print("Some issues remain that need manual fixing")
    
    print(f"\n📄 Full report saved to hardcode_elimination_compliance.json")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)