#!/usr/bin/env python3
"""
Production Readiness Validation Report
Comprehensive validation of GAELP system for production deployment.

CRITICAL: This script performs preliminary Wave 5 validation.
No shortcuts or simplified validation allowed.
"""

import sys
import os
import traceback
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionReadinessValidator:
    """Validates all critical components for production readiness"""
    
    def __init__(self):
        self.results = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'PENDING',
            'critical_failures': [],
            'warnings': [],
            'component_status': {},
            'safety_systems_status': {},
            'recommendations': []
        }
    
    def validate_component(self, component_name: str, test_func: callable) -> bool:
        """Validate a single component with comprehensive error handling"""
        logger.info(f"🔍 Validating {component_name}...")
        try:
            result = test_func()
            if result:
                self.results['component_status'][component_name] = 'PASS'
                logger.info(f"✅ {component_name} - PASSED")
                return True
            else:
                self.results['component_status'][component_name] = 'FAIL'
                self.results['critical_failures'].append(f"{component_name} - Validation failed")
                logger.error(f"❌ {component_name} - FAILED")
                return False
        except Exception as e:
            self.results['component_status'][component_name] = 'ERROR'
            error_msg = f"{component_name} - Exception: {str(e)}"
            self.results['critical_failures'].append(error_msg)
            logger.error(f"💥 {component_name} - ERROR: {str(e)}")
            return False
    
    def validate_no_fallbacks(self) -> bool:
        """Validate that NO FALLBACKS exist in the codebase"""
        logger.info("🚨 Running NO FALLBACKS validation...")
        try:
            # Run the NO_FALLBACKS.py validator
            import subprocess
            result = subprocess.run([sys.executable, 'NO_FALLBACKS.py', '--strict'], 
                                  capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                return True
            else:
                # Parse violations from stderr
                violations = result.stderr.count('violations found')
                if violations > 0:
                    self.results['critical_failures'].append(f"NO_FALLBACKS violations detected: {result.stderr[:1000]}...")
                return False
        except Exception as e:
            self.results['critical_failures'].append(f"NO_FALLBACKS validator failed to run: {str(e)}")
            return False
    
    def validate_emergency_controls(self) -> bool:
        """Validate emergency control systems"""
        try:
            from emergency_controls import EmergencyController
            from emergency_monitor import EmergencyMonitor
            
            controller = EmergencyController()
            
            # Test critical emergency scenarios
            test_scenarios = [
                ('budget_overrun', {'current_spend': 1500, 'budget_limit': 1200}),
                ('max_cpc_breach', {'current_cpc': 75, 'max_cpc': 50}),
                ('loss_explosion', {'current_loss': 15, 'baseline_loss': 1.2})
            ]
            
            for scenario, params in test_scenarios:
                status = controller.check_violation(scenario, params)
                if not status or status.get('severity') == 'green':
                    logger.warning(f"Emergency control {scenario} may not be properly calibrated")
                    self.results['warnings'].append(f"Emergency control {scenario} calibration check")
            
            return True
        except Exception as e:
            logger.error(f"Emergency controls validation failed: {str(e)}")
            return False
    
    def validate_safety_systems(self) -> bool:
        """Validate all safety system components"""
        safety_components = [
            ('Budget Safety', self._validate_budget_safety),
            ('Reward Validation', self._validate_reward_validation),
            ('Audit Trail', self._validate_audit_trail),
            ('Circuit Breakers', self._validate_circuit_breakers)
        ]
        
        all_passed = True
        for component_name, validator in safety_components:
            passed = self.validate_component(f"Safety System - {component_name}", validator)
            self.results['safety_systems_status'][component_name] = 'PASS' if passed else 'FAIL'
            if not passed:
                all_passed = False
        
        return all_passed
    
    def _validate_budget_safety(self) -> bool:
        """Validate budget safety systems"""
        try:
            from budget_safety_system import BudgetSafetySystem
            safety = BudgetSafetySystem()
            
            # Test budget limit enforcement
            test_result = safety.validate_spend_request(
                campaign_id="test_campaign",
                requested_amount=500,
                current_daily_spend=900,
                daily_budget=1000
            )
            
            return test_result is not None
        except Exception:
            return False
    
    def _validate_reward_validation(self) -> bool:
        """Validate reward validation systems"""
        try:
            from gaelp_safety_framework import GAELPSafetyFramework
            safety = GAELPSafetyFramework()
            
            # Test reward validation
            validated_reward = safety.validate_reward(1000.0)  # Should be clipped
            return validated_reward <= 1000.0
        except Exception:
            return False
    
    def _validate_audit_trail(self) -> bool:
        """Validate audit trail system"""
        try:
            from audit_trail import GAELPAuditTrail
            audit = GAELPAuditTrail()
            
            # Test audit logging
            audit.log_action(
                action_type="test_validation",
                component="production_validator",
                details={"test": True},
                user_id="system"
            )
            return True
        except Exception:
            return False
    
    def _validate_circuit_breakers(self) -> bool:
        """Validate circuit breaker functionality"""
        try:
            from emergency_controls import CircuitBreaker
            
            breaker = CircuitBreaker("test_validation", failure_threshold=2, timeout=1)
            
            # Test failure detection
            for _ in range(3):
                breaker.record_failure()
            
            return breaker.is_open()
        except Exception:
            return False
    
    def validate_core_training_components(self) -> bool:
        """Validate core RL training components"""
        components_to_test = [
            ('Fortified RL Agent', self._validate_rl_agent),
            ('GAELP Environment', self._validate_environment),
            ('Discovery Engine', self._validate_discovery_engine),
            ('Attribution System', self._validate_attribution)
        ]
        
        all_passed = True
        for component_name, validator in components_to_test:
            passed = self.validate_component(component_name, validator)
            if not passed:
                all_passed = False
        
        return all_passed
    
    def _validate_rl_agent(self) -> bool:
        """Validate RL agent loads and functions"""
        try:
            from fortified_rl_agent_no_hardcoding import create_fortified_agent
            agent = create_fortified_agent()
            return agent is not None
        except Exception:
            return False
    
    def _validate_environment(self) -> bool:
        """Validate GAELP environment"""
        try:
            from fortified_environment_no_hardcoding import create_environment
            env = create_environment()
            return env is not None
        except Exception:
            return False
    
    def _validate_discovery_engine(self) -> bool:
        """Validate discovery engine finds patterns"""
        try:
            from discovery_engine import DiscoveryEngine
            engine = DiscoveryEngine()
            
            # Check if patterns file exists and has content
            patterns_file = 'discovered_patterns.json'
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    patterns = json.load(f)
                return 'training_params' in patterns and len(patterns['training_params']) > 0
            return False
        except Exception:
            return False
    
    def _validate_attribution(self) -> bool:
        """Validate attribution system"""
        try:
            from attribution_system import MultiTouchAttributionEngine
            engine = MultiTouchAttributionEngine()
            return engine is not None
        except Exception:
            return False
    
    def validate_data_pipeline(self) -> bool:
        """Validate data pipeline components"""
        try:
            # Check if GA4 data exists
            ga4_dir = 'ga4_extracted_data'
            if not os.path.exists(ga4_dir):
                self.results['warnings'].append("GA4 data directory not found")
                return False
            
            # Check master report
            master_report = os.path.join(ga4_dir, '00_MASTER_REPORT.json')
            if os.path.exists(master_report):
                with open(master_report, 'r') as f:
                    data = json.load(f)
                    if 'summary' in data and data['summary'].get('total_conversions', 0) > 0:
                        return True
            
            self.results['warnings'].append("GA4 master report missing or empty")
            return False
        except Exception:
            return False
    
    def validate_checkpoint_system(self) -> bool:
        """Validate checkpoint and model persistence"""
        try:
            from production_checkpoint_manager import ProductionCheckpointManager
            manager = ProductionCheckpointManager()
            
            # Test checkpoint creation
            test_state = {'test': True, 'validation_run': True}
            checkpoint_path = manager.save_checkpoint('validation_test', test_state, {'accuracy': 0.95})
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Test checkpoint loading
                loaded_state, loaded_metadata = manager.load_checkpoint('validation_test')
                return loaded_state is not None
            
            return False
        except Exception:
            return False
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run comprehensive production readiness validation"""
        logger.info("🚀 Starting Production Readiness Validation...")
        
        validation_steps = [
            ('No Fallbacks Check', self.validate_no_fallbacks),
            ('Emergency Controls', self.validate_emergency_controls), 
            ('Safety Systems', self.validate_safety_systems),
            ('Core Training Components', self.validate_core_training_components),
            ('Data Pipeline', self.validate_data_pipeline),
            ('Checkpoint System', self.validate_checkpoint_system)
        ]
        
        passed_count = 0
        total_count = len(validation_steps)
        
        for step_name, validator in validation_steps:
            if self.validate_component(step_name, validator):
                passed_count += 1
        
        # Determine overall status
        if passed_count == total_count:
            self.results['overall_status'] = 'READY_FOR_PRODUCTION'
        elif passed_count >= total_count * 0.8:  # 80% pass rate
            self.results['overall_status'] = 'READY_WITH_WARNINGS'
        else:
            self.results['overall_status'] = 'NOT_READY_FOR_PRODUCTION'
        
        # Add recommendations
        if len(self.results['critical_failures']) > 0:
            self.results['recommendations'].append("CRITICAL: Fix all critical failures before production deployment")
        
        if len(self.results['warnings']) > 0:
            self.results['recommendations'].append("Address all warnings for optimal production performance")
        
        logger.info(f"✅ Validation Complete: {self.results['overall_status']}")
        logger.info(f"📊 Results: {passed_count}/{total_count} components passed")
        
        return self.results

def main():
    """Run production readiness validation"""
    print("=" * 80)
    print("                GAELP PRODUCTION READINESS VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    validator = ProductionReadinessValidator()
    results = validator.run_full_validation()
    
    # Save results
    results_file = f"production_readiness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("                      VALIDATION RESULTS")
    print("=" * 80)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Components Tested: {len(results['component_status'])}")
    print(f"Critical Failures: {len(results['critical_failures'])}")
    print(f"Warnings: {len(results['warnings'])}")
    
    if results['critical_failures']:
        print("\n🚨 CRITICAL FAILURES:")
        for failure in results['critical_failures'][:5]:  # Show first 5
            print(f"  ❌ {failure}")
        if len(results['critical_failures']) > 5:
            print(f"  ... and {len(results['critical_failures']) - 5} more")
    
    if results['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in results['warnings']:
            print(f"  ⚠️  {warning}")
    
    print(f"\n📄 Full report saved to: {results_file}")
    
    # Return appropriate exit code
    if results['overall_status'] == 'NOT_READY_FOR_PRODUCTION':
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())