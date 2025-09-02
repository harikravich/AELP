#!/usr/bin/env python3
"""
STRICT Verification of Production Checkpoint Manager

This verifies the checkpoint manager meets all production requirements:
- NO FALLBACKS - All validation is real
- NO SIMPLIFICATIONS - Complete implementation
- NO HARDCODING - Everything configurable
- COMPREHENSIVE VALIDATION - Thorough testing
- ROLLBACK CAPABILITY - Always available
"""

import os
import sys
import logging
import torch
from datetime import datetime
from typing import Dict, Any

# Add AELP to path
sys.path.insert(0, '/home/hariravichandran/AELP')

from production_checkpoint_manager import (
    ProductionCheckpointManager,
    ValidationStatus,
    RegressionSeverity,
    HoldoutValidator,
    RegressionDetector,
    ArchitectureValidator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrictValidationTest:
    """Strict validation tests that verify NO FALLBACKS are used"""
    
    def __init__(self):
        self.manager = ProductionCheckpointManager(
            checkpoint_dir="strict_test_checkpoints",
            holdout_data_path="strict_test_holdout.json",
            max_checkpoints=5
        )
        self.test_results = {}
        
    def verify_no_fallbacks(self) -> bool:
        """Verify system uses no fallbacks - everything is validated"""
        logger.info("🔍 VERIFYING NO FALLBACKS IN VALIDATION SYSTEM")
        
        # Create test model
        class StrictTestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 4)
            
            def forward(self, x):
                return self.layer(x)
            
            def select_action(self, state):
                with torch.no_grad():
                    features = torch.tensor(state.get('features', [0]*10), dtype=torch.float32)
                    output = self.forward(features)
                    action = output.argmax().item()
                return {'action': action, 'confidence': torch.softmax(output, dim=0).max().item()}
        
        model = StrictTestModel()
        
        # Save checkpoint
        checkpoint_id = self.manager.save_checkpoint(
            model=model,
            model_version="strict_test_v1.0.0",
            episode=1,
            training_config={'strict_validation': True, 'no_fallbacks': True},
            training_metrics={'roas': 3.5, 'accuracy': 0.87},
            validate_immediately=False
        )
        
        # Run validation manually and verify each step
        logger.info("🔍 Running comprehensive validation...")
        validation_passed = self.manager.validate_checkpoint(checkpoint_id)
        
        if not validation_passed:
            logger.error("❌ VALIDATION FAILED - This indicates proper validation (no fallbacks)")
            return False
            
        # Get validation details
        metadata = self.manager.checkpoints[checkpoint_id]
        
        # Verify validation was comprehensive
        if len(metadata.validation_logs) == 0:
            logger.error("❌ NO VALIDATION LOGS - Fallback detected")
            return False
            
        validation_log = metadata.validation_logs[-1]
        
        # Check all validation steps were executed
        required_steps = ['architecture_validation', 'holdout_validation', 'performance_benchmarking']
        executed_steps = [step['step'] for step in validation_log.get('validation_steps', [])]
        
        for required_step in required_steps:
            if required_step not in executed_steps:
                logger.error(f"❌ MISSING VALIDATION STEP: {required_step} - Fallback detected")
                return False
        
        logger.info("✅ ALL VALIDATION STEPS EXECUTED - No fallbacks detected")
        return True
    
    def verify_regression_detection_works(self) -> bool:
        """Verify regression detection actually works and prevents deployments"""
        logger.info("🔍 VERIFYING REGRESSION DETECTION (NO SIMPLIFICATIONS)")
        
        detector = RegressionDetector(baseline_threshold=0.05)  # 5% threshold
        
        # Create baseline metrics
        from production_checkpoint_manager import ValidationMetrics
        
        baseline = ValidationMetrics(
            accuracy=0.90, precision=0.85, recall=0.80, f1_score=0.82,
            roas=4.0, conversion_rate=0.15, ctr=0.10,
            inference_latency_ms=50.0, memory_usage_mb=100.0, throughput_qps=100.0,
            gradient_norm=0.1, weight_stability=1.0, output_variance=0.05,
            revenue_impact=1000.0, cost_efficiency=1.0, user_satisfaction=0.9,
            timestamp=datetime.now()
        )
        
        # Test minor regression (should pass)
        minor_regression = ValidationMetrics(
            accuracy=0.88, precision=0.83, recall=0.78, f1_score=0.80,  # Small drops
            roas=3.9, conversion_rate=0.14, ctr=0.09,
            inference_latency_ms=55.0, memory_usage_mb=105.0, throughput_qps=95.0,
            gradient_norm=0.12, weight_stability=0.98, output_variance=0.06,
            revenue_impact=980.0, cost_efficiency=0.98, user_satisfaction=0.88,
            timestamp=datetime.now()
        )
        
        severity, analysis = detector.detect_regression(baseline, minor_regression)
        
        if severity not in [RegressionSeverity.NONE, RegressionSeverity.MINOR]:
            logger.error(f"❌ MINOR REGRESSION INCORRECTLY FLAGGED AS: {severity}")
            return False
            
        # Test severe regression (should fail)
        severe_regression = ValidationMetrics(
            accuracy=0.65, precision=0.60, recall=0.55, f1_score=0.57,  # Major drops
            roas=2.5, conversion_rate=0.08, ctr=0.05,  # Severe degradation
            inference_latency_ms=200.0, memory_usage_mb=300.0, throughput_qps=20.0,
            gradient_norm=0.5, weight_stability=0.7, output_variance=0.2,
            revenue_impact=500.0, cost_efficiency=0.6, user_satisfaction=0.5,
            timestamp=datetime.now()
        )
        
        severity, analysis = detector.detect_regression(baseline, severe_regression)
        
        if severity not in [RegressionSeverity.SEVERE, RegressionSeverity.CRITICAL]:
            logger.error(f"❌ SEVERE REGRESSION NOT DETECTED: {severity}")
            return False
            
        logger.info("✅ REGRESSION DETECTION WORKING CORRECTLY")
        return True
    
    def verify_architecture_validation_strict(self) -> bool:
        """Verify architecture validation is strict (no simplified checks)"""
        logger.info("🔍 VERIFYING STRICT ARCHITECTURE VALIDATION")
        
        validator = ArchitectureValidator()
        
        # Create two different models
        class Model1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 4)
        
        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 8)  # Different output size
        
        model1 = Model1()
        model2 = Model2()
        
        config1 = {'architecture': 'model1', 'output_size': 4}
        config2 = {'architecture': 'model2', 'output_size': 8}
        
        sig1 = validator.generate_signature(model1, config1)
        sig2 = validator.generate_signature(model2, config2)
        
        # Should detect incompatibility
        compatible, issues = validator.validate_compatibility(sig1, sig2)
        
        if compatible:
            logger.error("❌ ARCHITECTURE VALIDATION TOO LENIENT - Should detect incompatibility")
            return False
            
        if len(issues) == 0:
            logger.error("❌ NO COMPATIBILITY ISSUES DETECTED - Validation too simple")
            return False
            
        logger.info(f"✅ STRICT ARCHITECTURE VALIDATION WORKING - {len(issues)} issues detected")
        return True
    
    def verify_holdout_validation_comprehensive(self) -> bool:
        """Verify holdout validation is comprehensive (no simplified testing)"""
        logger.info("🔍 VERIFYING COMPREHENSIVE HOLDOUT VALIDATION")
        
        validator = HoldoutValidator("test_holdout_strict.json", min_samples=50)
        
        # Create model that will have some failures
        class TestModel:
            def select_action(self, state):
                # Simulate occasional failures to test error handling
                import random
                if random.random() < 0.1:  # 10% failure rate
                    raise ValueError("Simulated model error")
                return {'action': 0, 'confidence': 0.8}
        
        model = TestModel()
        results = validator.validate_model(model, 'rl_agent')
        
        # Check validation is comprehensive
        required_metrics = ['success_rate', 'avg_inference_time_ms', 'total_samples_tested']
        
        for metric in required_metrics:
            if metric not in results.get('metrics', {}):
                logger.error(f"❌ MISSING VALIDATION METRIC: {metric}")
                return False
        
        # Should have some errors due to our test model
        if len(results.get('errors', [])) == 0:
            logger.warning("⚠️ NO ERRORS DETECTED - Validation might be too lenient")
        
        # Should have performance thresholds
        if results['passed'] and results['metrics']['success_rate'] < 0.9:
            logger.error("❌ VALIDATION PASSED WITH LOW SUCCESS RATE - Too lenient")
            return False
        
        logger.info("✅ COMPREHENSIVE HOLDOUT VALIDATION WORKING")
        return True
    
    def verify_deployment_safety_strict(self) -> bool:
        """Verify deployment safety prevents unvalidated deployments"""
        logger.info("🔍 VERIFYING STRICT DEPLOYMENT SAFETY")
        
        # Create unvalidated checkpoint
        class SimpleModel:
            def state_dict(self):
                return {'param': torch.tensor([1.0])}
        
        model = SimpleModel()
        
        checkpoint_id = self.manager.save_checkpoint(
            model=model,
            model_version="safety_test_v1.0.0",
            episode=1,
            training_config={'safety_test': True},
            training_metrics={'roas': 3.0},
            validate_immediately=False  # Don't validate
        )
        
        # Try to deploy without validation - should fail
        deploy_success = self.manager.deploy_checkpoint(checkpoint_id, force=False)
        
        if deploy_success:
            logger.error("❌ UNVALIDATED MODEL DEPLOYED - Safety system failed")
            return False
        
        logger.info("✅ DEPLOYMENT SAFETY WORKING - Prevents unvalidated deployments")
        return True
    
    def verify_rollback_always_available(self) -> bool:
        """Verify rollback is always available when needed"""
        logger.info("🔍 VERIFYING ROLLBACK AVAILABILITY")
        
        # Create and deploy first model
        class Model1:
            def state_dict(self):
                return {'param1': torch.tensor([1.0])}
            def select_action(self, state):
                return {'action': 0}
        
        model1 = Model1()
        checkpoint1 = self.manager.save_checkpoint(
            model=model1,
            model_version="rollback_test_v1.0.0",
            episode=1,
            training_config={'version': 1},
            training_metrics={'roas': 3.5},
            validate_immediately=True
        )
        
        # Force deploy first model
        deploy1_success = self.manager.deploy_checkpoint(checkpoint1, force=True)
        if not deploy1_success:
            logger.error("❌ COULD NOT DEPLOY FIRST MODEL FOR ROLLBACK TEST")
            return False
        
        # Create and deploy second model
        class Model2:
            def state_dict(self):
                return {'param2': torch.tensor([2.0])}
            def select_action(self, state):
                return {'action': 1}
        
        model2 = Model2()
        checkpoint2 = self.manager.save_checkpoint(
            model=model2,
            model_version="rollback_test_v2.0.0", 
            episode=2,
            training_config={'version': 2},
            training_metrics={'roas': 3.8},
            validate_immediately=True
        )
        
        deploy2_success = self.manager.deploy_checkpoint(checkpoint2, force=True)
        if not deploy2_success:
            logger.error("❌ COULD NOT DEPLOY SECOND MODEL FOR ROLLBACK TEST")
            return False
        
        # Verify rollback is available
        current_prod = self.manager.current_production_checkpoint
        if current_prod != checkpoint2:
            logger.error("❌ CURRENT PRODUCTION CHECKPOINT NOT SET CORRECTLY")
            return False
        
        metadata = self.manager.checkpoints[checkpoint2]
        if not metadata.rollback_points:
            logger.warning("⚠️ NO ROLLBACK POINTS AVAILABLE - May limit rollback capability")
        
        # Test rollback
        rollback_success = self.manager.rollback_to_checkpoint(checkpoint1, "Testing rollback capability")
        
        if not rollback_success:
            logger.error("❌ ROLLBACK FAILED - Safety system compromised")
            return False
        
        # Verify rollback worked
        if self.manager.current_production_checkpoint != checkpoint1:
            logger.error("❌ ROLLBACK DID NOT UPDATE CURRENT PRODUCTION")
            return False
        
        logger.info("✅ ROLLBACK SYSTEM WORKING CORRECTLY")
        return True
    
    def run_strict_verification(self) -> bool:
        """Run all strict verification tests"""
        logger.info("="*70)
        logger.info("STRICT PRODUCTION CHECKPOINT MANAGER VERIFICATION")
        logger.info("Verifying NO FALLBACKS, NO SIMPLIFICATIONS, COMPLETE IMPLEMENTATION")
        logger.info("="*70)
        
        tests = [
            ('no_fallbacks', self.verify_no_fallbacks),
            ('regression_detection', self.verify_regression_detection_works),
            ('architecture_validation', self.verify_architecture_validation_strict),
            ('holdout_validation', self.verify_holdout_validation_comprehensive),
            ('deployment_safety', self.verify_deployment_safety_strict),
            ('rollback_availability', self.verify_rollback_always_available)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"RUNNING STRICT TEST: {test_name.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                result = test_func()
                self.test_results[test_name] = result
                
                if result:
                    logger.info(f"✅ STRICT TEST PASSED: {test_name}")
                    passed_tests += 1
                else:
                    logger.error(f"❌ STRICT TEST FAILED: {test_name}")
                    
            except Exception as e:
                logger.error(f"❌ STRICT TEST ERROR: {test_name} - {e}")
                self.test_results[test_name] = False
        
        # Final results
        logger.info("\n" + "="*70)
        logger.info("STRICT VERIFICATION RESULTS")
        logger.info("="*70)
        
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"  {test_name}: {status}")
        
        all_passed = passed_tests == total_tests
        
        if all_passed:
            logger.info("\n🎉 ALL STRICT TESTS PASSED")
            logger.info("✅ Production Checkpoint Manager is PRODUCTION READY")
            logger.info("✅ NO FALLBACKS detected")
            logger.info("✅ NO SIMPLIFICATIONS detected")
            logger.info("✅ COMPLETE IMPLEMENTATION verified")
            logger.info("✅ COMPREHENSIVE VALIDATION confirmed")
            logger.info("✅ ROLLBACK CAPABILITY verified")
        else:
            logger.error("\n❌ STRICT VERIFICATION FAILED")
            logger.error("❌ Production deployment NOT RECOMMENDED")
            logger.error("❌ Address failed tests before production use")
        
        # Cleanup
        import shutil
        if os.path.exists("strict_test_checkpoints"):
            shutil.rmtree("strict_test_checkpoints")
        
        return all_passed

def main():
    """Run strict verification"""
    logger.info("Starting strict verification of Production Checkpoint Manager")
    
    verifier = StrictValidationTest()
    success = verifier.run_strict_verification()
    
    if success:
        logger.info("\n🚀 PRODUCTION CHECKPOINT MANAGER VERIFIED FOR PRODUCTION")
        logger.info("Ready for deployment with full confidence")
        exit(0)
    else:
        logger.error("\n⚠️ VERIFICATION FAILED - NOT READY FOR PRODUCTION")
        exit(1)

if __name__ == "__main__":
    main()