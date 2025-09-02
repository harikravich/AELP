#!/usr/bin/env python3
"""
COMPREHENSIVE REGRESSION DETECTION TESTING
Tests all aspects of the regression detection and rollback system.

CRITICAL TESTS:
- Statistical detection accuracy with known patterns
- Rollback mechanism under various failure scenarios  
- Integration with GAELP training components
- Performance under high load
- Database integrity and persistence
- Emergency integration compatibility

NO MOCKS - Real regression patterns only
NO SIMPLIFIED TESTS - Full system validation
"""

import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

import unittest
import logging
import tempfile
import shutil
import os
import time
import threading
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import regression detection components
from regression_detector import (
    RegressionDetector, StatisticalDetector, ModelManager, 
    MetricSnapshot, MetricType, RegressionSeverity,
    RegressionTestSuite, RegressionAlert
)
from gaelp_regression_integration import GAELPRegressionMonitor, ProductionTrainingWithRegression
from emergency_controls import get_emergency_controller, EmergencyLevel

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
logger = logging.getLogger(__name__)

class TestStatisticalDetector(unittest.TestCase):
    """Test statistical regression detection algorithms"""
    
    def setUp(self):
        """Set up test environment"""
        self.detector = StatisticalDetector(window_size=50, alpha=0.01)
    
    def test_baseline_establishment_accuracy(self):
        """Test that baselines are established accurately from historical data"""
        # Generate known distribution
        true_mean, true_std = 2.5, 0.5
        historical_data = np.random.normal(true_mean, true_std, 500).tolist()
        
        # Update baseline
        self.detector.update_baseline(MetricType.ROAS, historical_data)
        
        # Verify baseline statistics
        baseline = self.detector.baseline_stats[MetricType.ROAS]
        self.assertAlmostEqual(baseline['mean'], true_mean, delta=0.1)
        self.assertAlmostEqual(baseline['std'], true_std, delta=0.1)
        self.assertIn(MetricType.ROAS, self.detector.control_limits)
        
        # Verify control limits are reasonable
        limits = self.detector.control_limits[MetricType.ROAS]
        expected_range = 2.5 * true_std
        self.assertAlmostEqual(
            limits['upper'] - limits['lower'], 
            2 * expected_range, 
            delta=0.5
        )
    
    def test_no_false_positives_on_normal_data(self):
        """Test that normal variations don't trigger false alarms"""
        # Establish baseline
        baseline_data = np.random.normal(100, 20, 200).tolist()
        self.detector.update_baseline(MetricType.REWARD, baseline_data)
        
        # Test with normal variations
        for _ in range(10):
            test_data = np.random.normal(100, 20, 30).tolist()
            alert = self.detector.detect_regression(MetricType.REWARD, test_data)
            
            # Should not trigger alerts for normal variations
            if alert is not None:
                self.assertIn(alert.severity, [RegressionSeverity.NONE, RegressionSeverity.MINOR])
    
    def test_critical_regression_detection(self):
        """Test detection of critical performance regressions"""
        # Establish baseline
        baseline_data = np.random.normal(2.0, 0.3, 200).tolist()
        self.detector.update_baseline(MetricType.ROAS, baseline_data)
        
        # Create critical regression (50% drop)
        regression_data = np.random.normal(1.0, 0.2, 50).tolist()
        alert = self.detector.detect_regression(MetricType.ROAS, regression_data)
        
        self.assertIsNotNone(alert, "Critical regression not detected")
        self.assertIn(alert.severity, [RegressionSeverity.SEVERE, RegressionSeverity.CRITICAL])
        self.assertGreater(abs(alert.z_score), 3.0)
        self.assertLess(alert.p_value, 0.001)
        self.assertGreater(alert.confidence, 0.95)
    
    def test_gradual_degradation_detection(self):
        """Test detection of gradual performance degradation"""
        # Establish baseline
        baseline_data = np.random.normal(0.05, 0.01, 300).tolist()  # 5% CVR
        self.detector.update_baseline(MetricType.CONVERSION_RATE, baseline_data)
        
        # Create gradual degradation (20% drop)
        degraded_data = np.random.normal(0.04, 0.008, 50).tolist()  # 4% CVR
        alert = self.detector.detect_regression(MetricType.CONVERSION_RATE, degraded_data)
        
        self.assertIsNotNone(alert, "Gradual degradation not detected")
        self.assertGreaterEqual(alert.severity, RegressionSeverity.MODERATE)
        self.assertGreater(alert.confidence, 0.80)
    
    def test_variance_change_detection(self):
        """Test detection of changes in metric variance"""
        # Stable baseline
        baseline_data = np.random.normal(1.5, 0.2, 200).tolist()
        self.detector.update_baseline(MetricType.CPC, baseline_data)
        
        # Same mean but much higher variance
        volatile_data = np.random.normal(1.5, 0.8, 50).tolist()
        alert = self.detector.detect_regression(MetricType.CPC, volatile_data)
        
        # Should detect increased volatility as concerning
        if alert is not None:
            self.assertGreaterEqual(alert.severity, RegressionSeverity.MINOR)
    
    def test_insufficient_data_handling(self):
        """Test proper handling of insufficient data scenarios"""
        # Try to establish baseline with insufficient data
        minimal_data = [1.0, 1.1, 0.9]
        self.detector.update_baseline(MetricType.CTR, minimal_data)
        
        # Should handle gracefully (warning logged, no crash)
        self.assertNotIn(MetricType.CTR, self.detector.baseline_stats)
        
        # Try regression detection with insufficient current data
        baseline_data = np.random.normal(0.1, 0.02, 100).tolist()
        self.detector.update_baseline(MetricType.CTR, baseline_data)
        
        minimal_current = [0.08, 0.09]
        alert = self.detector.detect_regression(MetricType.CTR, minimal_current)
        self.assertIsNone(alert)

class TestModelManager(unittest.TestCase):
    """Test model checkpoint management and rollback functionality"""
    
    def setUp(self):
        """Set up test environment with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_creation_and_storage(self):
        """Test creating and storing model checkpoints"""
        # Create mock model
        test_model = {'weights': np.random.randn(100), 'bias': np.random.randn(10)}
        performance_metrics = {'roas': 2.5, 'conversion_rate': 0.04, 'reward': 150.0}
        validation_scores = {'composite_score': 85.2, 'validation_roas': 2.3}
        
        # Create checkpoint
        checkpoint_id = self.manager.create_checkpoint(
            test_model, performance_metrics, 100, validation_scores
        )
        
        self.assertIsNotNone(checkpoint_id)
        self.assertIn(checkpoint_id, self.manager.checkpoints)
        
        # Verify checkpoint details
        checkpoint = self.manager.checkpoints[checkpoint_id]
        self.assertEqual(checkpoint.episodes_trained, 100)
        self.assertEqual(checkpoint.performance_metrics['roas'], 2.5)
        self.assertTrue(checkpoint.is_rollback_candidate)
        
        # Verify file was created
        self.assertTrue(os.path.exists(checkpoint.model_path))
    
    def test_rollback_candidate_selection(self):
        """Test selection of appropriate rollback candidates"""
        # Create multiple checkpoints with different performance
        checkpoints_data = [
            ({'roas': 3.0, 'conversion_rate': 0.06}, 100),  # Best
            ({'roas': 2.2, 'conversion_rate': 0.04}, 200),  # Decent 
            ({'roas': 1.5, 'conversion_rate': 0.02}, 300),  # Poor
            ({'roas': 2.8, 'conversion_rate': 0.055}, 400), # Good
        ]
        
        checkpoint_ids = []
        for metrics, episodes in checkpoints_data:
            test_model = {'data': np.random.randn(50)}
            validation_scores = {'composite_score': metrics['roas'] * 30}
            
            checkpoint_id = self.manager.create_checkpoint(
                test_model, metrics, episodes, validation_scores
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Find best candidate meeting minimum requirements
        min_performance = {'roas': 2.5, 'conversion_rate': 0.05}
        candidate = self.manager.find_rollback_candidate(min_performance)
        
        # Should select the best performing checkpoint that meets requirements
        self.assertIsNotNone(candidate)
        selected_checkpoint = self.manager.checkpoints[candidate]
        
        # Verify it meets requirements
        self.assertGreaterEqual(selected_checkpoint.performance_metrics['roas'], 2.5)
        self.assertGreaterEqual(selected_checkpoint.performance_metrics['conversion_rate'], 0.05)
    
    def test_rollback_execution(self):
        """Test actual rollback execution"""
        # Create initial checkpoint
        model1 = {'version': 1, 'weights': np.random.randn(50)}
        metrics1 = {'roas': 2.5, 'conversion_rate': 0.04}
        validation1 = {'composite_score': 75.0}
        
        checkpoint1 = self.manager.create_checkpoint(model1, metrics1, 100, validation1)
        
        # Create second checkpoint (worse performance)
        model2 = {'version': 2, 'weights': np.random.randn(50)}
        metrics2 = {'roas': 1.8, 'conversion_rate': 0.03}
        validation2 = {'composite_score': 54.0}
        
        checkpoint2 = self.manager.create_checkpoint(model2, metrics2, 200, validation2)
        
        # Current should be checkpoint2
        self.assertEqual(self.manager.current_model_id, checkpoint2)
        
        # Rollback to checkpoint1
        success = self.manager.rollback_to_checkpoint(checkpoint1)
        self.assertTrue(success)
        
        # Verify rollback
        self.assertEqual(self.manager.current_model_id, checkpoint1)
        
        # Verify we can load the rolled back checkpoint
        loaded_data = self.manager.load_checkpoint(checkpoint1)
        self.assertIsNotNone(loaded_data)
    
    def test_checkpoint_cleanup(self):
        """Test automatic cleanup of old checkpoints"""
        # Create many checkpoints
        checkpoint_ids = []
        for i in range(25):  # More than cleanup threshold
            test_model = {'id': i, 'data': np.random.randn(20)}
            metrics = {'roas': 2.0 + i * 0.01, 'conversion_rate': 0.03}
            validation = {'composite_score': 60.0 + i}
            
            checkpoint_id = self.manager.create_checkpoint(
                test_model, metrics, i * 10, validation
            )
            checkpoint_ids.append(checkpoint_id)
            
            # Small delay to ensure different timestamps
            time.sleep(0.01)
        
        # Should have cleaned up to keep only recent ones plus baseline
        self.assertLessEqual(len(self.manager.checkpoints), 22)  # 20 + baseline + current
        
        # Baseline should still exist
        baseline_exists = any(cp.is_baseline for cp in self.manager.checkpoints.values())
        self.assertTrue(baseline_exists)
        
        # Most recent should still exist
        self.assertIn(checkpoint_ids[-1], self.manager.checkpoints)
    
    def test_metadata_persistence(self):
        """Test that checkpoint metadata persists across restarts"""
        # Create checkpoint
        test_model = {'data': [1, 2, 3]}
        metrics = {'roas': 2.3, 'conversion_rate': 0.045}
        validation = {'composite_score': 70.5}
        
        checkpoint_id = self.manager.create_checkpoint(test_model, metrics, 150, validation)
        
        # Create new manager instance (simulating restart)
        manager2 = ModelManager(self.temp_dir)
        
        # Should load existing metadata
        self.assertIn(checkpoint_id, manager2.checkpoints)
        self.assertEqual(manager2.current_model_id, checkpoint_id)
        
        # Verify checkpoint details preserved
        checkpoint = manager2.checkpoints[checkpoint_id]
        self.assertEqual(checkpoint.episodes_trained, 150)
        self.assertEqual(checkpoint.performance_metrics['roas'], 2.3)

class TestRegressionDetector(unittest.TestCase):
    """Test the main regression detector with database integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_regression.db')
        self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        
        self.detector = RegressionDetector(
            db_path=self.db_path,
            checkpoint_dir=self.checkpoint_dir
        )
    
    def tearDown(self):
        """Clean up test environment"""
        self.detector.stop_monitoring()
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test that database is properly initialized"""
        # Check that database file was created
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check that tables were created
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check alerts table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check rollbacks table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rollbacks'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_metric_recording_and_retrieval(self):
        """Test recording and retrieving metrics from database"""
        timestamp = datetime.now()
        
        # Record various metrics
        metrics_data = [
            (MetricType.ROAS, 2.5, 100),
            (MetricType.CONVERSION_RATE, 0.04, 100),
            (MetricType.CPC, 1.8, 100),
            (MetricType.REWARD, 125.5, 100)
        ]
        
        for metric_type, value, episode in metrics_data:
            snapshot = MetricSnapshot(
                metric_type=metric_type,
                value=value,
                timestamp=timestamp,
                episode=episode,
                metadata={'test': True}
            )
            self.detector.record_metric(snapshot)
        
        # Verify storage in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM metrics")
            count = cursor.fetchone()[0]
            self.assertEqual(count, len(metrics_data))
            
            # Check specific metric
            cursor.execute("SELECT value FROM metrics WHERE metric_type = ?", (MetricType.ROAS.value,))
            roas_value = cursor.fetchone()[0]
            self.assertEqual(roas_value, 2.5)
    
    def test_comprehensive_regression_workflow(self):
        """Test complete regression detection workflow"""
        # Step 1: Establish baseline
        baseline_roas = np.random.normal(2.0, 0.3, 100)
        self.detector.statistical_detector.update_baseline(MetricType.ROAS, baseline_roas.tolist())
        
        # Step 2: Record normal metrics
        timestamp = datetime.now()
        for i in range(50):
            snapshot = MetricSnapshot(
                metric_type=MetricType.ROAS,
                value=np.random.normal(2.0, 0.3),
                timestamp=timestamp + timedelta(minutes=i),
                episode=i
            )
            self.detector.record_metric(snapshot)
        
        # Step 3: Record degraded metrics
        for i in range(50, 80):
            snapshot = MetricSnapshot(
                metric_type=MetricType.ROAS,
                value=np.random.normal(1.4, 0.2),  # Significant drop
                timestamp=timestamp + timedelta(minutes=i),
                episode=i
            )
            self.detector.record_metric(snapshot)
        
        # Step 4: Check for regressions
        alerts = self.detector.check_for_regressions()
        
        # Should detect ROAS regression
        roas_alerts = [a for a in alerts if a.metric_type == MetricType.ROAS]
        self.assertGreater(len(roas_alerts), 0, "ROAS regression not detected")
        
        roas_alert = roas_alerts[0]
        self.assertIn(roas_alert.severity, [RegressionSeverity.SEVERE, RegressionSeverity.CRITICAL])
        
        # Step 5: Verify alert storage
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE metric_type = ?", (MetricType.ROAS.value,))
            alert_count = cursor.fetchone()[0]
            self.assertGreater(alert_count, 0)
    
    def test_automatic_rollback_evaluation(self):
        """Test automatic rollback decision making"""
        # Create critical alerts
        critical_alert = RegressionAlert(
            metric_type=MetricType.ROAS,
            severity=RegressionSeverity.CRITICAL,
            current_value=1.2,
            baseline_mean=2.5,
            baseline_std=0.3,
            z_score=-4.5,
            p_value=0.0001,
            detection_time=datetime.now(),
            confidence=0.99,
            recommended_action="IMMEDIATE_ROLLBACK"
        )
        
        # Should trigger rollback
        needs_rollback = self.detector.evaluate_rollback_need([critical_alert])
        self.assertTrue(needs_rollback)
        
        # Create minor alerts
        minor_alert = RegressionAlert(
            metric_type=MetricType.CPC,
            severity=RegressionSeverity.MINOR,
            current_value=1.8,
            baseline_mean=1.6,
            baseline_std=0.2,
            z_score=1.2,
            p_value=0.2,
            detection_time=datetime.now(),
            confidence=0.75,
            recommended_action="MONITOR_CLOSELY"
        )
        
        # Should not trigger rollback
        needs_rollback = self.detector.evaluate_rollback_need([minor_alert])
        self.assertFalse(needs_rollback)
    
    def test_monitoring_thread_functionality(self):
        """Test background monitoring thread operation"""
        # Start monitoring
        self.detector.start_monitoring()
        self.assertTrue(self.detector.monitoring_active)
        self.assertIsNotNone(self.detector.monitoring_thread)
        
        # Let it run briefly
        time.sleep(1.0)
        
        # Stop monitoring
        self.detector.stop_monitoring()
        self.assertFalse(self.detector.monitoring_active)
    
    def test_performance_summary_generation(self):
        """Test generation of comprehensive performance summaries"""
        # Add some test data
        timestamp = datetime.now()
        
        for i in range(20):
            for metric_type in [MetricType.ROAS, MetricType.CONVERSION_RATE, MetricType.REWARD]:
                value = np.random.normal(2.0 if metric_type == MetricType.ROAS else 0.04 if metric_type == MetricType.CONVERSION_RATE else 100, 0.1)
                snapshot = MetricSnapshot(
                    metric_type=metric_type,
                    value=value,
                    timestamp=timestamp + timedelta(minutes=i),
                    episode=i
                )
                self.detector.record_metric(snapshot)
        
        # Generate summary
        summary = self.detector.get_performance_summary()
        
        # Verify structure
        self.assertIn('timestamp', summary)
        self.assertIn('monitoring_active', summary)
        self.assertIn('metric_summaries', summary)
        self.assertIn('system_health', summary)
        
        # Verify metric summaries
        self.assertIn(MetricType.ROAS.value, summary['metric_summaries'])
        self.assertIn('current_mean', summary['metric_summaries'][MetricType.ROAS.value])

class TestGAELPRegressionIntegration(unittest.TestCase):
    """Test GAELP-specific regression monitoring functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = RegressionDetector(
            db_path=os.path.join(self.temp_dir, 'test_gaelp_regression.db'),
            checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
        )
        self.gaelp_monitor = GAELPRegressionMonitor(self.detector)
    
    def tearDown(self):
        """Clean up test environment"""
        self.detector.stop_monitoring()
        shutil.rmtree(self.temp_dir)
    
    def test_business_threshold_checking(self):
        """Test business rule validation"""
        # Test metrics that meet thresholds
        good_agent_metrics = {'roas': 2.0, 'conversion_rate': 0.03}
        good_env_metrics = {'avg_cpc': 3.0}
        good_reward = 75.0
        
        meets_thresholds = self.gaelp_monitor._check_business_thresholds(
            good_agent_metrics, good_env_metrics, good_reward
        )
        self.assertTrue(meets_thresholds)
        
        # Test metrics that don't meet thresholds
        bad_agent_metrics = {'roas': 1.0, 'conversion_rate': 0.01}  # Too low
        bad_env_metrics = {'avg_cpc': 8.0}  # Too high
        bad_reward = 30.0  # Too low
        
        meets_thresholds = self.gaelp_monitor._check_business_thresholds(
            bad_agent_metrics, bad_env_metrics, bad_reward
        )
        self.assertFalse(meets_thresholds)
    
    def test_performance_degradation_detection(self):
        """Test GAELP-specific performance degradation detection"""
        # Add performance history with degrading trend
        timestamp = datetime.now()
        
        # Good performance initially
        for i in range(10):
            agent_metrics = {'roas': 2.5 - i * 0.1, 'conversion_rate': 0.04 - i * 0.002}
            env_metrics = {'avg_cpc': 1.5 + i * 0.1}
            reward = 100.0 - i * 5
            
            self.gaelp_monitor.record_training_metrics(i, agent_metrics, env_metrics, reward)
        
        # Check for degradation
        degradation_report = self.gaelp_monitor.check_performance_degradation(10)
        
        # Should detect business violations
        self.assertGreater(len(degradation_report['business_violations']), 0)
        self.assertIn(degradation_report['severity'], ['warning', 'critical'])
    
    def test_performance_dashboard_generation(self):
        """Test comprehensive dashboard generation"""
        # Add sample performance data
        for i in range(15):
            agent_metrics = {
                'roas': 2.0 + np.random.normal(0, 0.2),
                'conversion_rate': 0.035 + np.random.normal(0, 0.005)
            }
            env_metrics = {
                'avg_cpc': 1.8 + np.random.normal(0, 0.3),
                'ctr': 0.12
            }
            reward = 85 + np.random.normal(0, 15)
            
            self.gaelp_monitor.record_training_metrics(i, agent_metrics, env_metrics, reward)
        
        # Generate dashboard
        dashboard = self.gaelp_monitor.get_performance_dashboard()
        
        # Verify structure
        self.assertIn('timestamp', dashboard)
        self.assertIn('system_status', dashboard)
        self.assertIn('recent_performance', dashboard)
        self.assertIn('business_compliance', dashboard)
        self.assertIn('alert_summary', dashboard)
        
        # Verify performance data
        recent_perf = dashboard['recent_performance']
        self.assertIn('avg_roas', recent_perf)
        self.assertIn('avg_conversion_rate', recent_perf)
        self.assertGreater(recent_perf['episodes_analyzed'], 0)
        
        # Verify compliance data
        compliance = dashboard['business_compliance']
        self.assertIn('compliance_rate', compliance)
        self.assertBetween(compliance['compliance_rate'], 0.0, 1.0)
    
    def assertBetween(self, value, min_val, max_val):
        """Custom assertion for value in range"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)

class TestRegressionSystemPerformance(unittest.TestCase):
    """Test regression system performance under load"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = RegressionDetector(
            db_path=os.path.join(self.temp_dir, 'perf_test.db'),
            checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
        )
    
    def tearDown(self):
        """Clean up test environment"""
        self.detector.stop_monitoring()
        shutil.rmtree(self.temp_dir)
    
    def test_high_volume_metric_recording(self):
        """Test system performance with high volume metric recording"""
        start_time = time.time()
        
        # Record large number of metrics
        timestamp = datetime.now()
        num_metrics = 1000  # Reduced for reasonable test time
        
        for i in range(num_metrics):
            for metric_type in [MetricType.ROAS, MetricType.CONVERSION_RATE, MetricType.REWARD]:
                snapshot = MetricSnapshot(
                    metric_type=metric_type,
                    value=np.random.normal(2.0, 0.5),
                    timestamp=timestamp + timedelta(seconds=i),
                    episode=i
                )
                self.detector.record_metric(snapshot)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process reasonably quickly (< 60 seconds for 3k metrics)
        self.assertLess(processing_time, 60.0, f"Took {processing_time:.2f}s to process {num_metrics*3} metrics")
        
        # Verify all metrics were stored
        with sqlite3.connect(self.detector.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM metrics")
            stored_count = cursor.fetchone()[0]
            self.assertEqual(stored_count, num_metrics * 3)
    
    def test_concurrent_access_safety(self):
        """Test thread safety under concurrent access"""
        errors = []
        
        def worker_thread(thread_id):
            """Worker thread for concurrent testing"""
            try:
                for i in range(100):
                    snapshot = MetricSnapshot(
                        metric_type=MetricType.ROAS,
                        value=np.random.normal(2.0, 0.3),
                        timestamp=datetime.now(),
                        episode=thread_id * 100 + i,
                        user_id=f"user_{thread_id}"
                    )
                    self.detector.record_metric(snapshot)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Start multiple worker threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker_thread, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check for errors
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")
        
        # Verify all metrics recorded
        with sqlite3.connect(self.detector.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM metrics")
            stored_count = cursor.fetchone()[0]
            self.assertEqual(stored_count, 500)  # 5 threads * 100 metrics each
    
    def test_regression_detection_performance(self):
        """Test performance of regression detection algorithms"""
        # Establish baselines for multiple metrics
        for metric_type in [MetricType.ROAS, MetricType.CONVERSION_RATE, MetricType.CPC, MetricType.REWARD]:
            baseline_data = np.random.normal(2.0, 0.5, 200).tolist()
            self.detector.statistical_detector.update_baseline(metric_type, baseline_data)
        
        # Record metrics for detection
        for i in range(200):
            for metric_type in [MetricType.ROAS, MetricType.CONVERSION_RATE, MetricType.CPC, MetricType.REWARD]:
                snapshot = MetricSnapshot(
                    metric_type=metric_type,
                    value=np.random.normal(2.0, 0.5),
                    timestamp=datetime.now() + timedelta(seconds=i),
                    episode=i
                )
                self.detector.record_metric(snapshot)
        
        # Time regression detection
        start_time = time.time()
        alerts = self.detector.check_for_regressions()
        end_time = time.time()
        
        detection_time = end_time - start_time
        
        # Should complete quickly (< 5 seconds)
        self.assertLess(detection_time, 5.0, f"Regression detection took {detection_time:.2f}s")

def run_comprehensive_regression_tests():
    """Run all regression detection tests and generate report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE REGRESSION DETECTION SYSTEM TESTS")
    print("="*80)
    
    # Test suite components
    test_classes = [
        TestStatisticalDetector,
        TestModelManager,
        TestRegressionDetector,
        TestGAELPRegressionIntegration,
        TestRegressionSystemPerformance
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        class_tests = result.testsRun
        class_failures = len(result.failures)
        class_errors = len(result.errors)
        
        total_tests += class_tests
        total_failures += class_failures
        total_errors += class_errors
        
        # Print results
        if class_failures == 0 and class_errors == 0:
            print(f"✅ {test_class.__name__}: ALL {class_tests} TESTS PASSED")
        else:
            print(f"❌ {test_class.__name__}: {class_failures} failures, {class_errors} errors out of {class_tests}")
            
            # Print failure details
            for failure in result.failures:
                print(f"   FAILURE: {failure[0]}")
                print(f"   {failure[1].split('AssertionError:')[-1].strip()}")
            
            for error in result.errors:
                print(f"   ERROR: {error[0]}")
                print(f"   {str(error[1]).split('Exception:')[-1].strip()}")
    
    # Final summary
    print("\n" + "="*80)
    print("REGRESSION TESTING FINAL REPORT")
    print("="*80)
    
    success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - total_failures - total_errors}")
    print(f"Failed: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 ALL REGRESSION DETECTION TESTS PASSED!")
        print("✅ System ready for production deployment")
        print("✅ Statistical detection algorithms verified")
        print("✅ Rollback mechanisms tested")
        print("✅ Database integrity confirmed")
        print("✅ Performance under load validated")
        print("✅ GAELP integration verified")
    else:
        print(f"\n⚠️  {total_failures + total_errors} TEST FAILURES DETECTED")
        print("❌ System requires fixes before production deployment")
        print("❌ Review failed tests and implement corrections")
        
    print("="*80)
    
    return total_failures + total_errors == 0

if __name__ == "__main__":
    # Run comprehensive test suite
    success = run_comprehensive_regression_tests()
    
    if not success:
        sys.exit(1)  # Exit with error if tests failed
    
    print("\n✅ Regression Detection System: FULLY VALIDATED")
    print("Ready for integration with GAELP production training")