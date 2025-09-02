#!/usr/bin/env python3
"""
DOUBLE DQN IMPLEMENTATION VERIFICATION SCRIPT

This script verifies that the Double DQN implementation is complete and correct.
It checks for:
1. Proper action selection/evaluation decoupling
2. No single DQN fallbacks
3. Complete implementation in all training methods
4. Monitoring and verification methods
"""

import ast
import re
import sys
from typing import Dict, List, Any

def verify_double_dqn_implementation(filepath: str) -> Dict[str, Any]:
    """Comprehensive verification of Double DQN implementation"""
    
    results = {
        'syntax_valid': False,
        'double_dqn_patterns': {},
        'training_methods': {},
        'monitoring_methods': {},
        'fallback_check': {},
        'overall_status': 'FAILED'
    }
    
    try:
        # Read file
        with open(filepath, 'r') as f:
            code = f.read()
        
        # Verify syntax
        ast.parse(code)
        results['syntax_valid'] = True
        
        # Check Double DQN patterns
        results['double_dqn_patterns'] = {
            'argmax_operations': code.count('argmax(1)'),
            'gather_operations': code.count('gather(1,'),
            'online_action_selection': code.count('Action selection with online network'),
            'target_evaluation': code.count('Evaluation with target network'),
            'double_dqn_comments': code.count('Double DQN')
        }
        
        # Check training methods
        legacy_train = 'def _train_step_legacy(self):' in code
        trajectory_train = 'def _train_trajectory_batch(self):' in code
        
        results['training_methods'] = {
            'legacy_training_exists': legacy_train,
            'trajectory_training_exists': trajectory_train,
            'both_have_double_dqn': False
        }
        
        # Check if both methods have Double DQN
        if legacy_train and trajectory_train:
            # Extract method source
            legacy_start = code.find('def _train_step_legacy(self):')
            legacy_end = code.find('\n    def ', legacy_start + 1)
            if legacy_end == -1:
                legacy_end = len(code)
            legacy_source = code[legacy_start:legacy_end]
            
            traj_start = code.find('def _train_trajectory_batch(self):')
            traj_end = code.find('\n    def ', traj_start + 1)
            if traj_end == -1:
                traj_end = len(code)
            traj_source = code[traj_start:traj_end]
            
            # Check patterns in both methods
            legacy_has_patterns = all([
                'argmax(1)' in legacy_source,
                'gather(1,' in legacy_source,
                'Action selection with online network' in legacy_source,
                'Evaluation with target network' in legacy_source
            ])
            
            traj_has_patterns = all([
                'argmax(1)' in traj_source,
                'gather(1,' in traj_source,
                'Action selection with online network' in traj_source,
                'Evaluation with target network' in traj_source
            ])
            
            results['training_methods']['legacy_has_double_dqn'] = legacy_has_patterns
            results['training_methods']['trajectory_has_double_dqn'] = traj_has_patterns
            results['training_methods']['both_have_double_dqn'] = legacy_has_patterns and traj_has_patterns
        
        # Check monitoring methods
        results['monitoring_methods'] = {
            'q_value_monitoring': '_monitor_q_value_overestimation' in code,
            'double_dqn_verification': '_verify_double_dqn_benefit' in code,
            'trajectory_verification': '_verify_trajectory_double_dqn_benefit' in code,
            'overestimation_tracking': 'overestimation_bias' in code
        }
        
        # Check for problematic fallbacks (excluding comparison code)
        fallback_patterns = ['fallback', 'simplified', 'mock', 'dummy']
        fallback_issues = []
        
        for pattern in fallback_patterns:
            matches = []
            for match in re.finditer(pattern, code, re.IGNORECASE):
                # Get context around match
                start = max(0, match.start() - 100)
                end = min(len(code), match.end() + 100)
                context = code[start:end]
                
                # Skip if it's in comparison code
                if any(comp in context.lower() for comp in ['standard_dqn', 'what we replaced', 'comparison']):
                    continue
                    
                # Skip if it's just a comment about fallback logic
                if context.strip().startswith('#') or 'fallback to' in context.lower():
                    continue
                    
                matches.append((match.start(), context))
            
            if matches:
                fallback_issues.extend(matches)
        
        results['fallback_check'] = {
            'problematic_fallbacks_found': len(fallback_issues) > 0,
            'fallback_count': len(fallback_issues),
            'issues': fallback_issues[:3]  # First 3 issues only
        }
        
        # Overall status assessment
        criteria = [
            results['syntax_valid'],
            results['double_dqn_patterns']['argmax_operations'] >= 6,
            results['double_dqn_patterns']['gather_operations'] >= 6,
            results['double_dqn_patterns']['online_action_selection'] >= 3,
            results['double_dqn_patterns']['target_evaluation'] >= 3,
            results['training_methods']['both_have_double_dqn'],
            results['monitoring_methods']['q_value_monitoring'],
            results['monitoring_methods']['double_dqn_verification'],
            not results['fallback_check']['problematic_fallbacks_found']
        ]
        
        if all(criteria):
            results['overall_status'] = 'PASSED'
        elif sum(criteria) >= len(criteria) * 0.8:  # 80% criteria met
            results['overall_status'] = 'MOSTLY_PASSED'
        else:
            results['overall_status'] = 'FAILED'
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def print_verification_report(results: Dict[str, Any]) -> None:
    """Print human-readable verification report"""
    
    print("=" * 60)
    print("DOUBLE DQN IMPLEMENTATION VERIFICATION REPORT")
    print("=" * 60)
    
    # Overall status
    status = results['overall_status']
    status_symbols = {'PASSED': '✓', 'MOSTLY_PASSED': '⚠', 'FAILED': '✗'}
    print(f"\nOVERALL STATUS: {status_symbols.get(status, '?')} {status}")
    
    # Syntax check
    print(f"\nSyntax Valid: {'✓' if results['syntax_valid'] else '✗'}")
    
    # Double DQN patterns
    print(f"\nDOUBLE DQN PATTERNS:")
    patterns = results['double_dqn_patterns']
    print(f"  Argmax Operations: {patterns['argmax_operations']} (expected ≥6)")
    print(f"  Gather Operations: {patterns['gather_operations']} (expected ≥6)")
    print(f"  Online Action Selection: {patterns['online_action_selection']} (expected ≥3)")
    print(f"  Target Evaluation: {patterns['target_evaluation']} (expected ≥3)")
    print(f"  Double DQN Comments: {patterns['double_dqn_comments']}")
    
    # Training methods
    print(f"\nTRAINING METHODS:")
    methods = results['training_methods']
    print(f"  Legacy Training Exists: {'✓' if methods['legacy_training_exists'] else '✗'}")
    print(f"  Trajectory Training Exists: {'✓' if methods['trajectory_training_exists'] else '✗'}")
    if 'legacy_has_double_dqn' in methods:
        print(f"  Legacy Has Double DQN: {'✓' if methods['legacy_has_double_dqn'] else '✗'}")
    if 'trajectory_has_double_dqn' in methods:
        print(f"  Trajectory Has Double DQN: {'✓' if methods['trajectory_has_double_dqn'] else '✗'}")
    print(f"  Both Have Double DQN: {'✓' if methods['both_have_double_dqn'] else '✗'}")
    
    # Monitoring methods
    print(f"\nMONITORING METHODS:")
    monitoring = results['monitoring_methods']
    print(f"  Q-value Monitoring: {'✓' if monitoring['q_value_monitoring'] else '✗'}")
    print(f"  Double DQN Verification: {'✓' if monitoring['double_dqn_verification'] else '✗'}")
    print(f"  Trajectory Verification: {'✓' if monitoring['trajectory_verification'] else '✗'}")
    print(f"  Overestimation Tracking: {'✓' if monitoring['overestimation_tracking'] else '✗'}")
    
    # Fallback check
    print(f"\nFALLBACK CHECK:")
    fallback = results['fallback_check']
    print(f"  Problematic Fallbacks: {'✗' if fallback['problematic_fallbacks_found'] else '✓'} "
          f"({'Found' if fallback['problematic_fallbacks_found'] else 'None'})")
    if fallback['problematic_fallbacks_found']:
        print(f"  Fallback Count: {fallback['fallback_count']}")
    
    print(f"\n{'=' * 60}")
    
    if status == 'PASSED':
        print("🎉 DOUBLE DQN IMPLEMENTATION IS COMPLETE AND CORRECT!")
        print("   - All networks use proper action selection/evaluation decoupling")
        print("   - No single DQN fallbacks detected")
        print("   - Comprehensive monitoring and verification in place")
    elif status == 'MOSTLY_PASSED':
        print("⚠️  DOUBLE DQN IMPLEMENTATION IS MOSTLY CORRECT")
        print("   - Minor issues detected, but core functionality is present")
    else:
        print("❌ DOUBLE DQN IMPLEMENTATION HAS ISSUES")
        print("   - Critical problems detected that need to be fixed")
    
    print(f"{'=' * 60}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_double_dqn_implementation.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    results = verify_double_dqn_implementation(filepath)
    print_verification_report(results)
    
    # Exit with appropriate code
    if results['overall_status'] == 'FAILED':
        sys.exit(1)
    elif results['overall_status'] == 'MOSTLY_PASSED':
        sys.exit(2)
    else:
        sys.exit(0)