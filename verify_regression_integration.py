#!/usr/bin/env python3
"""
REGRESSION DETECTION INTEGRATION VERIFICATION
Verifies that the regression detection system properly integrates with GAELP.

VERIFICATION CHECKLIST:
- Integration with existing GAELP training loop
- Emergency controls compatibility  
- Database and checkpoint storage
- Real-time monitoring functionality
- Rollback mechanism execution
- No fallback code introduced
- Performance impact assessment

NO SIMPLIFIED VERIFICATION - Full production integration test
"""

import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

import logging
import tempfile
import shutil
import os
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import GAELP components
from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent
from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
from discovery_engine import GA4DiscoveryEngine
from gaelp_parameter_manager import ParameterManager

# Import regression detection
from regression_detector import RegressionDetector, MetricSnapshot, MetricType
from gaelp_regression_integration import GAELPRegressionMonitor, ProductionTrainingWithRegression
from emergency_controls import get_emergency_controller, EmergencyLevel

# Import testing framework
from test_regression_detection import run_comprehensive_regression_tests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegressionIntegrationVerifier:
    """Comprehensive verification of regression detection integration"""
    
    def __init__(self):
        self.temp_dir = None
        self.verification_results = []
        self.performance_metrics = {}
        
    def setup_test_environment(self):
        """Set up isolated test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix='regression_verify_')
        logger.info(f"Test environment created: {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("Test environment cleaned up")
    
    def verify_no_fallback_code(self) -> Dict[str, Any]:
        """Verify no fallback code was introduced"""
        result = {
            'test_name': 'No Fallback Code Verification',
            'status': 'PASS',
            'details': [],
            'violations': []
        }
        
        # Check regression detector files
        files_to_check = [
            '/home/hariravichandran/AELP/regression_detector.py',
            '/home/hariravichandran/AELP/gaelp_regression_integration.py',
            '/home/hariravichandran/AELP/test_regression_detection.py'
        ]
        
        forbidden_patterns = [
            'fallback', 'simplified', 'mock', 'dummy', 'placeholder',
            'TODO', 'FIXME', 'HACK', 'TEMP', 'quick_fix'
        ]
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                result['violations'].append(f"Missing file: {file_path}")
                continue
                
            with open(file_path, 'r') as f:
                content = f.read().lower()
                
                for pattern in forbidden_patterns:
                    if pattern.lower() in content:
                        # Check if it's in comments/docstrings (which is OK)
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern.lower() in line.lower():
                                stripped = line.strip()
                                if not (stripped.startswith('#') or stripped.startswith('"""') or 
                                       stripped.startswith("'''") or 'docstring' in stripped):
                                    result['violations'].append(
                                        f"{file_path}:{i+1}: {pattern} found in code: {line.strip()[:50]}"
                                    )
        
        if result['violations']:
            result['status'] = 'FAIL'
            result['details'].append(f"Found {len(result['violations'])} fallback violations")
        else:
            result['details'].append("No fallback code detected")
        
        return result
    
    def verify_gaelp_component_integration(self) -> Dict[str, Any]:
        """Verify integration with core GAELP components"""
        result = {
            'test_name': 'GAELP Component Integration',
            'status': 'PASS',
            'details': [],
            'errors': []
        }
        
        try:
            # Test component initialization
            logger.info("Testing GAELP component initialization...")
            
            # Initialize parameter manager
            pm = ParameterManager()
            result['details'].append("ParameterManager initialized successfully")
            
            # Initialize discovery engine
            discovery = GA4DiscoveryEngine(write_enabled=False, cache_only=True)
            result['details'].append("DiscoveryEngine initialized successfully")
            
            # Initialize regression detector with test paths
            regression_detector = RegressionDetector(
                db_path=os.path.join(self.temp_dir, 'test_integration.db'),
                checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
            )
            result['details'].append("RegressionDetector initialized successfully")
            
            # Initialize GAELP monitor
            gaelp_monitor = GAELPRegressionMonitor(regression_detector)
            result['details'].append("GAELPRegressionMonitor initialized successfully")
            
            # Test baseline establishment
            gaelp_monitor._establish_default_baselines()  # Use default since no GA4 data
            result['details'].append("Baselines established successfully")
            
            # Test metric recording
            timestamp = datetime.now()
            test_metrics = [
                (MetricType.ROAS, 2.3),
                (MetricType.CONVERSION_RATE, 0.035),
                (MetricType.REWARD, 85.5)
            ]
            
            for metric_type, value in test_metrics:
                snapshot = MetricSnapshot(
                    metric_type=metric_type,
                    value=value,
                    timestamp=timestamp,
                    episode=1,
                    metadata={'test': 'integration_verification'}
                )
                regression_detector.record_metric(snapshot)
            
            result['details'].append(f"Recorded {len(test_metrics)} test metrics successfully")
            
            # Test regression detection
            alerts = regression_detector.check_for_regressions()
            result['details'].append(f"Regression detection executed (found {len(alerts)} alerts)")
            
            # Clean up
            regression_detector.stop_monitoring()
            
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Integration test failed: {str(e)}")
            logger.error(f"GAELP integration test failed: {e}")
        
        return result
    
    def verify_emergency_controls_integration(self) -> Dict[str, Any]:
        """Verify integration with emergency control system"""
        result = {
            'test_name': 'Emergency Controls Integration',
            'status': 'PASS',
            'details': [],
            'errors': []
        }
        
        try:
            # Initialize emergency controller
            emergency_controller = get_emergency_controller()
            result['details'].append("Emergency controller initialized")
            
            # Initialize regression detector with emergency integration
            regression_detector = RegressionDetector(
                db_path=os.path.join(self.temp_dir, 'test_emergency.db'),
                checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints'),
                emergency_controller=emergency_controller
            )
            result['details'].append("Regression detector with emergency controls initialized")
            
            # Test emergency level reporting
            initial_level = emergency_controller.current_emergency_level
            result['details'].append(f"Initial emergency level: {initial_level.value}")
            
            # Simulate critical regression that should trigger emergency
            from regression_detector import RegressionAlert, RegressionSeverity
            critical_alert = RegressionAlert(
                metric_type=MetricType.ROAS,
                severity=RegressionSeverity.CRITICAL,
                current_value=1.0,
                baseline_mean=2.5,
                baseline_std=0.3,
                z_score=-5.0,
                p_value=0.0001,
                detection_time=datetime.now(),
                confidence=0.99,
                recommended_action="IMMEDIATE_ROLLBACK"
            )
            
            # Test rollback evaluation with emergency integration
            needs_rollback = regression_detector.evaluate_rollback_need([critical_alert])
            result['details'].append(f"Critical alert triggers rollback: {needs_rollback}")
            
            if not needs_rollback:
                result['errors'].append("Critical alert should trigger rollback evaluation")
            
            # Clean up
            regression_detector.stop_monitoring()
            
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Emergency integration test failed: {str(e)}")
            logger.error(f"Emergency integration test failed: {e}")
        
        return result
    
    def verify_database_operations(self) -> Dict[str, Any]:
        """Verify database operations and persistence"""
        result = {
            'test_name': 'Database Operations Verification',
            'status': 'PASS',
            'details': [],
            'errors': []
        }
        
        try:
            db_path = os.path.join(self.temp_dir, 'test_database.db')
            
            # Initialize regression detector
            detector = RegressionDetector(db_path=db_path, checkpoint_dir=self.temp_dir)
            result['details'].append("Database initialized successfully")
            
            # Test metric storage
            metrics_to_store = 50
            timestamp = datetime.now()
            
            for i in range(metrics_to_store):
                snapshot = MetricSnapshot(
                    metric_type=MetricType.ROAS,
                    value=2.0 + np.random.normal(0, 0.3),
                    timestamp=timestamp + timedelta(minutes=i),
                    episode=i,
                    user_id=f"user_{i % 5}",
                    metadata={'batch': 'verification_test'}
                )
                detector.record_metric(snapshot)
            
            result['details'].append(f"Stored {metrics_to_store} metrics")
            
            # Verify database content
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Check metrics table
                cursor.execute("SELECT COUNT(*) FROM metrics")
                stored_count = cursor.fetchone()[0]
                
                if stored_count != metrics_to_store:
                    result['errors'].append(f"Expected {metrics_to_store} metrics, found {stored_count}")
                else:
                    result['details'].append(f"Verified {stored_count} metrics in database")
                
                # Check table structure
                cursor.execute("PRAGMA table_info(metrics)")
                columns = [row[1] for row in cursor.fetchall()]
                
                expected_columns = ['id', 'metric_type', 'value', 'timestamp', 'episode', 'user_id', 'campaign_id', 'metadata']
                missing_columns = set(expected_columns) - set(columns)
                
                if missing_columns:
                    result['errors'].append(f"Missing database columns: {missing_columns}")
                else:
                    result['details'].append("Database schema verified")
            
            # Test database persistence (reinitialize)
            detector.stop_monitoring()
            
            detector2 = RegressionDetector(db_path=db_path, checkpoint_dir=self.temp_dir)
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM metrics")
                persistent_count = cursor.fetchone()[0]
                
                if persistent_count != metrics_to_store:
                    result['errors'].append(f"Persistence failed: expected {metrics_to_store}, found {persistent_count}")
                else:
                    result['details'].append("Database persistence verified")
            
            detector2.stop_monitoring()
            
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Database verification failed: {str(e)}")
        
        if result['errors']:
            result['status'] = 'FAIL'
        
        return result
    
    def verify_checkpoint_management(self) -> Dict[str, Any]:
        """Verify model checkpoint and rollback functionality"""
        result = {
            'test_name': 'Checkpoint Management Verification',
            'status': 'PASS',
            'details': [],
            'errors': []
        }
        
        try:
            # Initialize regression detector
            detector = RegressionDetector(
                db_path=os.path.join(self.temp_dir, 'test_checkpoints.db'),
                checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
            )
            
            # Create test model checkpoints
            test_models = [
                ({'weights': np.random.randn(100)}, {'roas': 2.8, 'conversion_rate': 0.05}, 100),
                ({'weights': np.random.randn(100)}, {'roas': 2.2, 'conversion_rate': 0.04}, 200),
                ({'weights': np.random.randn(100)}, {'roas': 1.9, 'conversion_rate': 0.03}, 300),
            ]
            
            checkpoint_ids = []
            for i, (model, metrics, episodes) in enumerate(test_models):
                validation_scores = {'composite_score': metrics['roas'] * 30}
                
                checkpoint_id = detector.create_model_checkpoint(model, metrics, episodes)
                checkpoint_ids.append(checkpoint_id)
                
                result['details'].append(f"Created checkpoint {i+1}: {checkpoint_id}")
            
            # Verify checkpoint files exist
            for checkpoint_id in checkpoint_ids:
                checkpoint = detector.model_manager.checkpoints[checkpoint_id]
                if not os.path.exists(checkpoint.model_path):
                    result['errors'].append(f"Checkpoint file missing: {checkpoint.model_path}")
            
            # Test rollback candidate selection
            min_performance = {'roas': 2.5, 'conversion_rate': 0.045}
            candidate = detector.model_manager.find_rollback_candidate(min_performance)
            
            if candidate:
                result['details'].append(f"Found rollback candidate: {candidate}")
                
                # Test actual rollback
                rollback_success = detector.model_manager.rollback_to_checkpoint(candidate)
                
                if rollback_success:
                    result['details'].append("Rollback executed successfully")
                    
                    # Verify current model updated
                    current = detector.model_manager.get_current_checkpoint()
                    if current.checkpoint_id == candidate:
                        result['details'].append("Current model correctly updated after rollback")
                    else:
                        result['errors'].append("Current model not updated after rollback")
                else:
                    result['errors'].append("Rollback execution failed")
            else:
                result['errors'].append("No rollback candidate found (should have found one)")
            
            # Test checkpoint metadata persistence
            detector.stop_monitoring()
            
            # Reinitialize and check persistence
            detector2 = RegressionDetector(
                db_path=os.path.join(self.temp_dir, 'test_checkpoints.db'),
                checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
            )
            
            if len(detector2.model_manager.checkpoints) == len(checkpoint_ids):
                result['details'].append("Checkpoint metadata persistence verified")
            else:
                result['errors'].append("Checkpoint metadata not persisted correctly")
            
            detector2.stop_monitoring()
            
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Checkpoint verification failed: {str(e)}")
        
        if result['errors']:
            result['status'] = 'FAIL'
        
        return result
    
    def verify_performance_impact(self) -> Dict[str, Any]:
        """Verify performance impact of regression detection"""
        result = {
            'test_name': 'Performance Impact Assessment',
            'status': 'PASS',
            'details': [],
            'warnings': []
        }
        
        try:
            # Measure baseline performance (without regression detection)
            start_time = time.time()
            
            # Simulate basic training operations
            for i in range(1000):
                # Simulate typical training step operations
                np.random.randn(100)  # Random computation
                time.sleep(0.001)  # Simulate small delay
            
            baseline_time = time.time() - start_time
            result['details'].append(f"Baseline operation time: {baseline_time:.3f}s")
            
            # Measure performance with regression detection
            detector = RegressionDetector(
                db_path=os.path.join(self.temp_dir, 'perf_test.db'),
                checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
            )
            
            # Establish baselines
            for metric_type in [MetricType.ROAS, MetricType.CONVERSION_RATE, MetricType.REWARD]:
                baseline_data = np.random.normal(2.0, 0.5, 100).tolist()
                detector.statistical_detector.update_baseline(metric_type, baseline_data)
            
            start_time = time.time()
            
            # Simulate training with regression monitoring
            for i in range(1000):
                # Record metrics (simulating training)
                snapshot = MetricSnapshot(
                    metric_type=MetricType.ROAS,
                    value=2.0 + np.random.normal(0, 0.3),
                    timestamp=datetime.now(),
                    episode=i
                )
                detector.record_metric(snapshot)
                
                # Simulate computation
                np.random.randn(100)
                time.sleep(0.001)
                
                # Check for regressions periodically
                if i % 50 == 0:
                    detector.check_for_regressions()
            
            monitored_time = time.time() - start_time
            result['details'].append(f"Monitored operation time: {monitored_time:.3f}s")
            
            # Calculate overhead
            overhead_percent = ((monitored_time - baseline_time) / baseline_time) * 100
            result['details'].append(f"Regression monitoring overhead: {overhead_percent:.1f}%")
            
            # Assess impact
            if overhead_percent < 10:
                result['details'].append("✅ Low performance impact (< 10%)")
            elif overhead_percent < 25:
                result['warnings'].append("⚠️ Moderate performance impact (10-25%)")
            else:
                result['warnings'].append("🚨 High performance impact (> 25%)")
            
            detector.stop_monitoring()
            
        except Exception as e:
            result['status'] = 'FAIL'
            result['details'].append(f"Performance assessment failed: {str(e)}")
        
        return result
    
    def verify_real_time_monitoring(self) -> Dict[str, Any]:
        """Verify real-time monitoring capabilities"""
        result = {
            'test_name': 'Real-time Monitoring Verification',
            'status': 'PASS',
            'details': [],
            'errors': []
        }
        
        try:
            detector = RegressionDetector(
                db_path=os.path.join(self.temp_dir, 'monitoring_test.db'),
                checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
            )
            
            # Establish baselines
            detector.statistical_detector.update_baseline(
                MetricType.ROAS, 
                np.random.normal(2.0, 0.3, 100).tolist()
            )
            
            # Start monitoring
            detector.start_monitoring()
            result['details'].append("Background monitoring started")
            
            # Verify monitoring thread is active
            if detector.monitoring_active and detector.monitoring_thread:
                result['details'].append("Monitoring thread active")
            else:
                result['errors'].append("Monitoring thread not started properly")
            
            # Add metrics while monitoring is active
            for i in range(20):
                snapshot = MetricSnapshot(
                    metric_type=MetricType.ROAS,
                    value=2.0 + np.random.normal(0, 0.4),
                    timestamp=datetime.now(),
                    episode=i
                )
                detector.record_metric(snapshot)
                time.sleep(0.1)  # Small delay
            
            # Let monitoring process for a bit
            time.sleep(2.0)
            result['details'].append("Metrics recorded while monitoring active")
            
            # Stop monitoring
            detector.stop_monitoring()
            
            if not detector.monitoring_active:
                result['details'].append("Monitoring stopped cleanly")
            else:
                result['errors'].append("Monitoring did not stop properly")
            
            # Verify metrics were processed
            if len(detector.recent_metrics[MetricType.ROAS]) > 0:
                result['details'].append(f"Processed {len(detector.recent_metrics[MetricType.ROAS])} metrics")
            else:
                result['errors'].append("No metrics processed during monitoring")
            
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Real-time monitoring test failed: {str(e)}")
        
        if result['errors']:
            result['status'] = 'FAIL'
        
        return result
    
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run all verification tests"""
        logger.info("Starting comprehensive regression detection integration verification")
        
        # Setup test environment
        self.setup_test_environment()
        
        verification_tests = [
            self.verify_no_fallback_code,
            self.verify_gaelp_component_integration,
            self.verify_emergency_controls_integration,
            self.verify_database_operations,
            self.verify_checkpoint_management,
            self.verify_performance_impact,
            self.verify_real_time_monitoring
        ]
        
        verification_results = {
            'timestamp': datetime.now().isoformat(),
            'test_results': [],
            'summary': {
                'total_tests': len(verification_tests),
                'passed': 0,
                'failed': 0,
                'warnings': 0
            },
            'overall_status': 'PASS'
        }
        
        # Run each verification test
        for test_func in verification_tests:
            try:
                result = test_func()
                verification_results['test_results'].append(result)
                
                if result['status'] == 'PASS':
                    verification_results['summary']['passed'] += 1
                else:
                    verification_results['summary']['failed'] += 1
                
                if 'warnings' in result and result['warnings']:
                    verification_results['summary']['warnings'] += len(result['warnings'])
                
                logger.info(f"✅ {result['test_name']}: {result['status']}")
                
            except Exception as e:
                error_result = {
                    'test_name': test_func.__name__,
                    'status': 'FAIL',
                    'errors': [f"Test execution failed: {str(e)}"]
                }
                verification_results['test_results'].append(error_result)
                verification_results['summary']['failed'] += 1
                logger.error(f"❌ {test_func.__name__}: FAIL - {str(e)}")
        
        # Determine overall status
        if verification_results['summary']['failed'] > 0:
            verification_results['overall_status'] = 'FAIL'
        elif verification_results['summary']['warnings'] > 0:
            verification_results['overall_status'] = 'PASS_WITH_WARNINGS'
        
        # Cleanup
        self.cleanup_test_environment()
        
        return verification_results

def generate_verification_report(results: Dict[str, Any]):
    """Generate comprehensive verification report"""
    print("\n" + "="*90)
    print("REGRESSION DETECTION INTEGRATION VERIFICATION REPORT")
    print("="*90)
    
    print(f"Verification Date: {results['timestamp']}")
    print(f"Overall Status: {results['overall_status']}")
    print()
    
    # Summary
    summary = results['summary']
    print("📊 VERIFICATION SUMMARY")
    print("-" * 50)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Success Rate: {(summary['passed']/summary['total_tests']*100):.1f}%")
    print()
    
    # Detailed results
    print("📋 DETAILED TEST RESULTS")
    print("-" * 50)
    
    for result in results['test_results']:
        status_icon = "✅" if result['status'] == 'PASS' else "❌"
        print(f"{status_icon} {result['test_name']}: {result['status']}")
        
        if 'details' in result and result['details']:
            for detail in result['details']:
                print(f"   ℹ️  {detail}")
        
        if 'warnings' in result and result['warnings']:
            for warning in result['warnings']:
                print(f"   ⚠️  {warning}")
        
        if 'errors' in result and result['errors']:
            for error in result['errors']:
                print(f"   ❌ {error}")
        
        if 'violations' in result and result['violations']:
            for violation in result['violations']:
                print(f"   🚨 {violation}")
        
        print()
    
    # Final assessment
    print("🎯 FINAL ASSESSMENT")
    print("-" * 50)
    
    if results['overall_status'] == 'PASS':
        print("✅ REGRESSION DETECTION INTEGRATION: FULLY VERIFIED")
        print("✅ System ready for production deployment")
        print("✅ All components properly integrated")
        print("✅ No fallback code detected")
        print("✅ Performance impact acceptable")
        print("✅ Real-time monitoring functional")
        print("✅ Emergency controls integrated")
        print("✅ Database operations verified")
        print("✅ Checkpoint management working")
        
    elif results['overall_status'] == 'PASS_WITH_WARNINGS':
        print("⚠️  REGRESSION DETECTION INTEGRATION: VERIFIED WITH WARNINGS")
        print("✅ Core functionality verified")
        print("⚠️  Review warnings before production deployment")
        print("✅ No critical issues found")
        
    else:
        print("❌ REGRESSION DETECTION INTEGRATION: VERIFICATION FAILED")
        print("❌ Critical issues found - fixes required")
        print("❌ Review failed tests and implement corrections")
        print("❌ Re-run verification after fixes")
    
    print("="*90)

def main():
    """Main verification function"""
    logger.info("Starting regression detection integration verification")
    
    try:
        # Run unit tests first
        logger.info("Running comprehensive regression detection tests...")
        test_success = run_comprehensive_regression_tests()
        
        if not test_success:
            logger.error("Unit tests failed - stopping verification")
            sys.exit(1)
        
        # Run integration verification
        logger.info("Running integration verification...")
        verifier = RegressionIntegrationVerifier()
        results = verifier.run_comprehensive_verification()
        
        # Generate report
        generate_verification_report(results)
        
        # Exit with appropriate code
        if results['overall_status'] == 'FAIL':
            sys.exit(1)
        elif results['overall_status'] == 'PASS_WITH_WARNINGS':
            logger.warning("Verification passed with warnings - review before production")
        else:
            logger.info("Verification completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Verification interrupted by user")
    except Exception as e:
        logger.error(f"Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()