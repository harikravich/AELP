#!/usr/bin/env python3
"""
Comprehensive RecSim Fallback Verification Script
Ensures NO FALLBACKS exist in RecSim integration per CLAUDE.md requirements
"""

import os
import re
import sys
from typing import Dict, List, Tuple
import ast

# Import the strict mode enforcer
from NO_FALLBACKS import StrictModeEnforcer

class RecSimFallbackDetector:
    """Detects and reports RecSim integration fallbacks"""
    
    def __init__(self):
        self.violations = []
        self.fallback_patterns = [
            # Import fallbacks
            r'try.*import.*recsim',
            r'except.*recsim',
            r'RECSIM_AVAILABLE.*=.*False',
            r'recsim.*not available',
            r'recsim.*skipping',
            
            # User simulation fallbacks
            r'fallback.*user',
            r'simplified.*user',
            r'mock.*user',
            r'dummy.*user',
            r'random\.choice.*user',
            r'if.*recsim.*else.*simple',
            
            # Behavioral fallbacks
            r'fallback.*behavior',
            r'simplified.*behavior',
            r'basic.*behavior.*model',
            r'default.*behavior',
            
            # Random user generation instead of RecSim
            r'random\.choice.*segment',
            r'np\.random\.choice.*user',
            r'generate.*simple.*user',
            
            # Conditional RecSim usage
            r'if.*recsim.*available',
            r'try.*recsim.*except.*pass',
            r'recsim.*or.*fallback'
        ]
    
    def scan_file(self, filepath: str) -> List[Dict]:
        """Scan a single file for fallback patterns"""
        violations = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.fallback_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            violations.append({
                                'file': filepath,
                                'line': line_num,
                                'content': line.strip(),
                                'pattern': pattern,
                                'severity': self._get_severity(pattern)
                            })
        
        except Exception as e:
            print(f"Error scanning {filepath}: {e}")
        
        return violations
    
    def _get_severity(self, pattern: str) -> str:
        """Get severity level for violation"""
        critical_patterns = [
            r'try.*import.*recsim',
            r'except.*recsim',
            r'RECSIM_AVAILABLE.*=.*False',
            r'fallback.*user',
            r'simplified.*user'
        ]
        
        for crit_pattern in critical_patterns:
            if pattern == crit_pattern:
                return "CRITICAL"
        return "HIGH"
    
    def scan_codebase(self) -> Dict:
        """Scan entire codebase for RecSim fallbacks"""
        
        # Files to scan (RecSim-related and simulation files)
        target_files = []
        
        # Find all Python files in AELP directory
        for root, dirs, files in os.walk('/home/hariravichandran/AELP'):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    
                    # Prioritize RecSim-related files
                    if any(keyword in file.lower() for keyword in 
                           ['recsim', 'simulator', 'user', 'behavior', 'auction', 'enhanced']):
                        target_files.append(filepath)
        
        print(f"Scanning {len(target_files)} files for RecSim fallbacks...")
        
        all_violations = []
        for filepath in target_files:
            file_violations = self.scan_file(filepath)
            all_violations.extend(file_violations)
        
        # Group by severity
        critical_violations = [v for v in all_violations if v['severity'] == 'CRITICAL']
        high_violations = [v for v in all_violations if v['severity'] == 'HIGH']
        
        return {
            'total_violations': len(all_violations),
            'critical_violations': critical_violations,
            'high_violations': high_violations,
            'all_violations': all_violations
        }
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive report"""
        
        report = f"""
{'='*80}
RECSIM FALLBACK VIOLATION REPORT
{'='*80}

SUMMARY:
- Total Violations: {results['total_violations']}
- Critical Violations: {len(results['critical_violations'])}
- High Priority Violations: {len(results['high_violations'])}

CLAUDE.MD COMPLIANCE: {'✅ PASS' if results['total_violations'] == 0 else '❌ FAIL'}

"""
        
        if results['critical_violations']:
            report += f"\n{'='*80}\nCRITICAL VIOLATIONS (MUST FIX IMMEDIATELY):\n{'='*80}\n"
            
            for violation in results['critical_violations']:
                report += f"""
FILE: {violation['file']}
LINE: {violation['line']}
PATTERN: {violation['pattern']}
CODE: {violation['content']}
SEVERITY: {violation['severity']}
---
"""
        
        if results['high_violations']:
            report += f"\n{'='*80}\nHIGH PRIORITY VIOLATIONS:\n{'='*80}\n"
            
            for violation in results['high_violations']:
                report += f"""
FILE: {violation['file']}
LINE: {violation['line']}
PATTERN: {violation['pattern']}
CODE: {violation['content']}
SEVERITY: {violation['severity']}
---
"""
        
        if results['total_violations'] == 0:
            report += f"\n{'='*80}\n🎉 ALL RECSIM INTEGRATION CHECKS PASSED!\n{'='*80}\n"
        else:
            report += f"""
{'='*80}
❌ RECSIM INTEGRATION VIOLATIONS FOUND
{'='*80}

REQUIRED ACTIONS:
1. Remove ALL fallback patterns immediately
2. Ensure RecSim is MANDATORY for user simulation
3. Replace random user generation with RecSim models
4. Remove conditional RecSim usage
5. Enforce strict RecSim dependency

Per CLAUDE.md requirements:
- NO FALLBACKS allowed
- NO SIMPLIFICATIONS allowed  
- RecSim is MANDATORY for realistic user behavior
- System MUST fail if RecSim unavailable
"""
        
        return report


def test_recsim_integration():
    """Test that RecSim integration actually works"""
    
    print("Testing RecSim Integration...")
    
    try:
        # Test 1: Import RecSim components
        import edward2_patch  # Apply compatibility patch first
        from recsim_user_model import RecSimUserModel, UserSegment, UserProfile
        from recsim_auction_bridge import RecSimAuctionBridge
        print("✅ RecSim imports successful")
        
        # Test 2: Create user model
        user_model = RecSimUserModel()
        print("✅ RecSim user model created")
        
        # Test 3: Generate user
        user_profile = user_model.generate_user("test_user", UserSegment.IMPULSE_BUYER)
        print("✅ RecSim user generation works")
        
        # Test 4: Simulate ad response
        ad_content = {
            'creative_quality': 0.8,
            'price_shown': 50.0,
            'brand_match': 0.7,
            'relevance_score': 0.6,
            'product_id': 'test_product'
        }
        
        context = {'hour': 20, 'device': 'mobile'}
        response = user_model.simulate_ad_response("test_user", ad_content, context)
        print("✅ RecSim ad response simulation works")
        
        # Test 5: Verify response format
        required_keys = ['clicked', 'converted', 'revenue', 'user_segment']
        for key in required_keys:
            if key not in response:
                raise ValueError(f"Missing required response key: {key}")
        print("✅ RecSim response format valid")
        
        print("\n🎉 RECSIM INTEGRATION TEST PASSED - NO FALLBACKS DETECTED!")
        return True
        
    except ImportError as e:
        print(f"❌ RecSim import failed: {e}")
        StrictModeEnforcer.enforce('RECSIM_INTEGRATION_TEST', fallback_attempted=True)
        raise ImportError(f"RecSim MUST be available. NO FALLBACKS! Error: {e}")
    
    except Exception as e:
        print(f"❌ RecSim integration test failed: {e}")
        StrictModeEnforcer.enforce('RECSIM_INTEGRATION_TEST', fallback_attempted=True)
        raise RuntimeError(f"RecSim integration MUST work. NO FALLBACKS! Error: {e}")


def main():
    """Main verification function"""
    
    print("🔍 RECSIM FALLBACK VERIFICATION STARTING...")
    print("="*80)
    
    # Step 1: Test actual RecSim integration
    print("\nStep 1: Testing RecSim Integration...")
    try:
        test_recsim_integration()
    except Exception as e:
        print(f"❌ RecSim integration test FAILED: {e}")
        return 1
    
    # Step 2: Scan for fallback patterns
    print("\nStep 2: Scanning for fallback patterns...")
    detector = RecSimFallbackDetector()
    results = detector.scan_codebase()
    
    # Step 3: Generate report
    report = detector.generate_report(results)
    print(report)
    
    # Step 4: Write report to file
    with open('/home/hariravichandran/AELP/recsim_fallback_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: /home/hariravichandran/AELP/recsim_fallback_report.txt")
    
    # Step 5: Enforce compliance
    if results['total_violations'] > 0:
        print(f"\n❌ COMPLIANCE FAILURE: {results['total_violations']} violations found")
        print("Fix all violations before proceeding!")
        StrictModeEnforcer.enforce('RECSIM_FALLBACK_COMPLIANCE', fallback_attempted=True)
        return 1
    else:
        print("\n✅ COMPLIANCE SUCCESS: No fallback violations found")
        print("RecSim integration is properly enforced!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)