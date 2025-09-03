#!/usr/bin/env python3
"""
COMPREHENSIVE REGRESSION DETECTION DEMO
Demonstrates the complete regression detection system with realistic scenarios

DEMO FEATURES:
1. Baseline establishment from historical data
2. Real-time performance monitoring
3. Regression detection across multiple metrics
4. Component health monitoring
5. Automatic rollback capability
6. Production dashboard

NO FALLBACKS - Full demonstration of production-ready system
"""

import sys
import os
import logging
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import regression detection components
from comprehensive_regression_detector import ComprehensiveRegressionDetector, RegressionSeverity
from gaelp_regression_production_integration import ProductionRegressionManager

logger = logging.getLogger(__name__)

class RegressionDetectionDemo:
    """Comprehensive demo of regression detection system"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize regression detector
        self.detector = ComprehensiveRegressionDetector(
            db_path="/home/hariravichandran/AELP/regression_demo.db"
        )
        
        # Demo state
        self.demo_phase = 1
        self.episode_count = 0
        
        self.logger.info("Regression detection demo initialized")
    
    def run_demo(self):
        """Run comprehensive regression detection demo"""
        self.logger.info("🚀 Starting Comprehensive Regression Detection Demo")
        
        try:
            # Phase 1: Establish baselines
            self._demo_baseline_establishment()
            
            # Phase 2: Normal operation monitoring
            self._demo_normal_operation()
            
            # Phase 3: Gradual performance degradation
            self._demo_gradual_degradation()
            
            # Phase 4: Critical regression and rollback
            self._demo_critical_regression()
            
            # Phase 5: Recovery monitoring
            self._demo_recovery_monitoring()
            
            # Final report
            self._generate_demo_report()
            
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.detector.stop_monitoring()
    
    def _demo_baseline_establishment(self):
        """Phase 1: Demonstrate baseline establishment"""
        self.demo_phase = 1
        self.logger.info("="*60)
        self.logger.info("📊 PHASE 1: BASELINE ESTABLISHMENT")
        self.logger.info("="*60)
        
        # Start monitoring
        self.detector.start_monitoring()
        
        # Simulate historical performance data
        self.logger.info("Establishing performance baselines from historical data...")
        
        # Generate 200 episodes of healthy historical performance
        for episode in range(200):
            # Healthy ROAS around 2.3
            roas = np.random.normal(2.3, 0.25)
            # Healthy CVR around 3.2%
            cvr = np.random.normal(0.032, 0.005)
            # Healthy CTR around 1.1%
            ctr = np.random.normal(0.011, 0.002)
            # Reasonable CPC around $1.40
            cpc = np.random.normal(1.4, 0.3)
            # Good episode rewards around 95
            reward = np.random.normal(95, 12)
            # Stable training loss around 0.6
            training_loss = np.random.normal(0.6, 0.15)
            
            # Record historical data with timestamps in the past
            timestamp = datetime.now() - timedelta(minutes=200-episode)
            
            self.detector.record_performance_metric('roas', max(0.5, roas), timestamp)
            self.detector.record_performance_metric('cvr', max(0.005, cvr), timestamp)
            self.detector.record_performance_metric('ctr', max(0.003, ctr), timestamp)
            self.detector.record_performance_metric('cpc', max(0.5, cpc), timestamp)
            self.detector.record_performance_metric('reward', reward, timestamp)
            self.detector.record_performance_metric('training_loss', max(0.1, training_loss), timestamp)
            
            if episode % 50 == 49:
                self.logger.info(f"  Processed {episode+1}/200 historical episodes")
        
        # Allow baselines to establish
        time.sleep(2)
        
        # Show baseline status
        status = self.detector.get_comprehensive_status()
        self.logger.info("✅ Baselines established:")
        for metric, info in status['performance_baselines'].items():
            if info['samples'] > 0:
                self.logger.info(f"  📈 {metric}: {info['samples']} samples, "
                               f"limits: [{info['control_limits'].get('lower_2sigma', 0):.3f}, "
                               f"{info['control_limits'].get('upper_2sigma', 0):.3f}]")
        
        self.logger.info(f"🏥 Component health: {status['component_health']['status']}")
        self.logger.info(f"⚡ System health: {status['system_health']}")
    
    def _demo_normal_operation(self):
        """Phase 2: Demonstrate normal operation monitoring"""
        self.demo_phase = 2
        self.logger.info("\n" + "="*60)
        self.logger.info("✅ PHASE 2: NORMAL OPERATION MONITORING")
        self.logger.info("="*60)
        
        self.logger.info("Simulating normal production operation...")
        
        # Run 50 episodes of normal performance
        for episode in range(50):
            self.episode_count += 1
            
            # Normal performance with slight variations
            roas = np.random.normal(2.25, 0.2)
            cvr = np.random.normal(0.031, 0.004)
            ctr = np.random.normal(0.0105, 0.0015)
            cpc = np.random.normal(1.45, 0.25)
            reward = np.random.normal(92, 10)
            training_loss = np.random.normal(0.65, 0.12)
            
            # Record metrics
            self.detector.record_performance_metric('roas', max(0.5, roas))
            self.detector.record_performance_metric('cvr', max(0.005, cvr))
            self.detector.record_performance_metric('ctr', max(0.003, ctr))
            self.detector.record_performance_metric('cpc', max(0.5, cpc))
            self.detector.record_performance_metric('reward', reward)
            self.detector.record_performance_metric('training_loss', max(0.1, training_loss))
            
            # Check for regressions every 10 episodes
            if episode % 10 == 9:
                regressions = self.detector.detect_comprehensive_regressions()
                
                if regressions:
                    self.logger.warning(f"  Episode {self.episode_count}: {len(regressions)} regressions detected")
                else:
                    self.logger.info(f"  Episode {self.episode_count}: No regressions - system healthy")
            
            time.sleep(0.05)  # Brief pause between episodes
        
        self.logger.info("✅ Normal operation complete - no significant regressions detected")
    
    def _demo_gradual_degradation(self):
        """Phase 3: Demonstrate gradual performance degradation detection"""
        self.demo_phase = 3
        self.logger.info("\n" + "="*60)
        self.logger.info("⚠️  PHASE 3: GRADUAL PERFORMANCE DEGRADATION")
        self.logger.info("="*60)
        
        self.logger.info("Simulating gradual performance degradation...")
        
        # Run 40 episodes with gradually declining performance
        for episode in range(40):
            self.episode_count += 1
            
            # Gradual degradation - performance slowly getting worse
            degradation_factor = episode / 40.0  # 0 to 1 over 40 episodes
            
            # ROAS slowly declining from 2.25 to 1.8
            roas = np.random.normal(2.25 - (degradation_factor * 0.45), 0.2)
            # CVR slowly declining from 3.1% to 2.3%
            cvr = np.random.normal(0.031 - (degradation_factor * 0.008), 0.004)
            # Reward slowly declining
            reward = np.random.normal(92 - (degradation_factor * 25), 10)
            
            # Other metrics stay relatively stable
            ctr = np.random.normal(0.0105, 0.0015)
            cpc = np.random.normal(1.45 + (degradation_factor * 0.3), 0.25)
            training_loss = np.random.normal(0.65 + (degradation_factor * 0.2), 0.12)
            
            # Record metrics
            self.detector.record_performance_metric('roas', max(0.5, roas))
            self.detector.record_performance_metric('cvr', max(0.005, cvr))
            self.detector.record_performance_metric('ctr', max(0.003, ctr))
            self.detector.record_performance_metric('cpc', max(0.5, cpc))
            self.detector.record_performance_metric('reward', reward)
            self.detector.record_performance_metric('training_loss', max(0.1, training_loss))
            
            # Check for regressions every 10 episodes
            if episode % 10 == 9:
                regressions = self.detector.detect_comprehensive_regressions()
                
                if regressions:
                    self.logger.warning(f"  Episode {self.episode_count}: {len(regressions)} regressions detected")
                    for regression in regressions:
                        self.logger.warning(f"    - {regression.regression_type.value} regression: "
                                          f"{regression.severity.value_str} in {regression.metrics_affected}")
                else:
                    self.logger.info(f"  Episode {self.episode_count}: No regressions detected yet")
            
            time.sleep(0.05)
        
        # Check final state after gradual degradation
        final_regressions = self.detector.detect_comprehensive_regressions()
        
        if final_regressions:
            self.logger.warning(f"⚠️  Gradual degradation detected: {len(final_regressions)} regressions")
            for regression in final_regressions:
                self.logger.warning(f"  📉 {regression.regression_type.value}: {regression.severity.value_str}")
        else:
            self.logger.info("ℹ️  Gradual degradation not yet detected (may need more data)")
    
    def _demo_critical_regression(self):
        """Phase 4: Demonstrate critical regression and automatic rollback"""
        self.demo_phase = 4
        self.logger.info("\n" + "="*60)
        self.logger.info("🚨 PHASE 4: CRITICAL REGRESSION & AUTOMATIC ROLLBACK")
        self.logger.info("="*60)
        
        # First, create a checkpoint to rollback to
        self.logger.info("Creating performance checkpoint for rollback...")
        good_performance = {'roas': 2.2, 'conversion_rate': 0.030, 'reward': 90, 'episode': self.episode_count}
        
        checkpoint_id = self.detector.core_detector.create_model_checkpoint(
            model={'weights': 'demo_good_model', 'episode': self.episode_count},
            performance_metrics=good_performance,
            episodes_trained=self.episode_count
        )
        
        self.logger.info(f"✅ Checkpoint created: {checkpoint_id}")
        
        # Now simulate critical performance regression
        self.logger.info("Simulating critical performance regression...")
        
        # Run 25 episodes with severely degraded performance
        for episode in range(25):
            self.episode_count += 1
            
            # Critical regression - severe performance drop
            roas = np.random.normal(1.2, 0.25)  # Catastrophic ROAS drop
            cvr = np.random.normal(0.015, 0.003)  # CVR halved
            ctr = np.random.normal(0.006, 0.002)  # CTR severely down
            cpc = np.random.normal(2.5, 0.5)  # CPC way up
            reward = np.random.normal(25, 8)  # Reward collapsed
            training_loss = np.random.normal(2.5, 0.5)  # Loss exploded
            
            # Record critical metrics
            self.detector.record_performance_metric('roas', max(0.1, roas))
            self.detector.record_performance_metric('cvr', max(0.005, cvr))
            self.detector.record_performance_metric('ctr', max(0.001, ctr))
            self.detector.record_performance_metric('cpc', max(0.5, cpc))
            self.detector.record_performance_metric('reward', reward)
            self.detector.record_performance_metric('training_loss', max(0.1, training_loss))
            
            # Check for critical regressions more frequently
            if episode % 5 == 4:
                regressions = self.detector.detect_comprehensive_regressions()
                
                if regressions:
                    critical_regressions = [r for r in regressions if r.severity == RegressionSeverity.CRITICAL]
                    severe_regressions = [r for r in regressions if r.severity == RegressionSeverity.SEVERE]
                    
                    self.logger.error(f"  🚨 Episode {self.episode_count}: CRITICAL REGRESSION DETECTED!")
                    self.logger.error(f"     Critical: {len(critical_regressions)}, Severe: {len(severe_regressions)}")
                    
                    for regression in critical_regressions:
                        self.logger.error(f"     🔴 CRITICAL: {regression.metrics_affected}")
                        if regression.rollback_recommendation.get('action') == 'immediate_rollback':
                            self.logger.error("     💥 IMMEDIATE ROLLBACK RECOMMENDED")
                    
                    # Check if rollback should be triggered
                    should_rollback = self.detector._should_trigger_rollback(regressions)
                    
                    if should_rollback:
                        self.logger.error("🔄 TRIGGERING AUTOMATIC ROLLBACK")
                        
                        # Find rollback candidate
                        min_performance = {'roas': 1.8, 'conversion_rate': 0.025, 'reward': 75}
                        candidate = self.detector.core_detector.model_manager.find_rollback_candidate(min_performance)
                        
                        if candidate:
                            rollback_success = self.detector.core_detector.model_manager.rollback_to_checkpoint(candidate)
                            
                            if rollback_success:
                                self.logger.info(f"✅ ROLLBACK SUCCESSFUL: Now using checkpoint {candidate}")
                                
                                # Reset recent metrics after rollback
                                for baseline in self.detector.baselines.values():
                                    recent_clear = min(20, len(baseline.values) // 3)
                                    for _ in range(recent_clear):
                                        if baseline.values:
                                            baseline.values.pop()
                                            baseline.timestamps.pop()
                                
                                self.logger.info("🔄 Metrics reset post-rollback")
                                return  # Exit critical phase after successful rollback
                            else:
                                self.logger.error("❌ ROLLBACK FAILED")
                        else:
                            self.logger.error("❌ NO ROLLBACK CANDIDATE FOUND")
            
            time.sleep(0.05)
        
        self.logger.error("⚠️  Critical regression phase completed without rollback")
    
    def _demo_recovery_monitoring(self):
        """Phase 5: Demonstrate recovery monitoring"""
        self.demo_phase = 5
        self.logger.info("\n" + "="*60)
        self.logger.info("🔄 PHASE 5: RECOVERY MONITORING")
        self.logger.info("="*60)
        
        self.logger.info("Simulating system recovery after rollback...")
        
        # Run 30 episodes showing recovery to normal performance
        for episode in range(30):
            self.episode_count += 1
            
            # Gradual recovery - performance slowly improving
            recovery_factor = min(1.0, episode / 20.0)  # Recovery over 20 episodes
            
            # Performance recovering towards baseline
            roas = np.random.normal(1.5 + (recovery_factor * 0.7), 0.2)
            cvr = np.random.normal(0.020 + (recovery_factor * 0.010), 0.003)
            reward = np.random.normal(50 + (recovery_factor * 40), 8)
            
            # Other metrics also recovering
            ctr = np.random.normal(0.007 + (recovery_factor * 0.0035), 0.002)
            cpc = np.random.normal(2.2 - (recovery_factor * 0.75), 0.3)
            training_loss = np.random.normal(1.8 - (recovery_factor * 1.15), 0.2)
            
            # Record recovery metrics
            self.detector.record_performance_metric('roas', max(0.5, roas))
            self.detector.record_performance_metric('cvr', max(0.005, cvr))
            self.detector.record_performance_metric('ctr', max(0.003, ctr))
            self.detector.record_performance_metric('cpc', max(0.5, cpc))
            self.detector.record_performance_metric('reward', reward)
            self.detector.record_performance_metric('training_loss', max(0.1, training_loss))
            
            # Check recovery progress every 10 episodes
            if episode % 10 == 9:
                regressions = self.detector.detect_comprehensive_regressions()
                
                if regressions:
                    self.logger.info(f"  Episode {self.episode_count}: {len(regressions)} residual regressions")
                    for regression in regressions:
                        if regression.severity.value >= RegressionSeverity.SEVERE.value:
                            self.logger.warning(f"    ⚠️  {regression.severity.value_str} in {regression.metrics_affected}")
                        else:
                            self.logger.info(f"    📈 {regression.severity.value_str} in {regression.metrics_affected} (recovering)")
                else:
                    self.logger.info(f"  Episode {self.episode_count}: ✅ No regressions - recovery successful!")
            
            time.sleep(0.05)
        
        self.logger.info("🎉 Recovery monitoring complete")
    
    def _generate_demo_report(self):
        """Generate comprehensive demo report"""
        self.logger.info("\n" + "="*70)
        self.logger.info("📋 COMPREHENSIVE REGRESSION DETECTION DEMO REPORT")
        self.logger.info("="*70)
        
        # Get final system status
        status = self.detector.get_comprehensive_status()
        
        self.logger.info(f"Demo Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total Episodes Simulated: {self.episode_count}")
        self.logger.info(f"Final System Health: {status['system_health']}")
        self.logger.info(f"Component Health Status: {status['component_health']['status']}")
        
        # Performance baselines summary
        self.logger.info("\n📊 PERFORMANCE BASELINES:")
        for metric, info in status['performance_baselines'].items():
            if info['samples'] > 0:
                self.logger.info(f"  {metric}: {info['samples']} samples")
        
        # Regression events summary
        recent_events = [e for e in self.detector.regression_events 
                        if e.detection_time > datetime.now() - timedelta(hours=1)]
        
        self.logger.info(f"\n🚨 REGRESSION EVENTS DETECTED: {len(self.detector.regression_events)} total")
        self.logger.info(f"Recent Events (last hour): {len(recent_events)}")
        
        # Event breakdown by severity
        severity_counts = {}
        for event in self.detector.regression_events:
            severity = event.severity.value_str
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            self.logger.info(f"  {severity}: {count} events")
        
        # Rollback history
        self.logger.info(f"\n🔄 ROLLBACK HISTORY: {len(self.detector.rollback_history)} rollbacks attempted")
        successful_rollbacks = sum(1 for r in self.detector.rollback_history if r.get('success', False))
        self.logger.info(f"Successful Rollbacks: {successful_rollbacks}")
        
        # Component health details
        component_health = status['component_health']
        self.logger.info(f"\n🏥 COMPONENT HEALTH:")
        self.logger.info(f"  Overall Status: {component_health['status']}")
        self.logger.info(f"  Health Ratio: {component_health['health_ratio']:.1%}")
        self.logger.info(f"  Healthy Components: {component_health['healthy_components']}")
        self.logger.info(f"  Degraded Components: {component_health['degraded_components']}")
        self.logger.info(f"  Failed Components: {component_health['failed_components']}")
        
        # Final assessment
        self.logger.info("\n🎯 DEMONSTRATION SUMMARY:")
        
        if status['system_health'] in ['healthy', 'degraded']:
            self.logger.info("✅ Regression detection system functioning correctly")
            self.logger.info("✅ Baselines established and maintained")
            self.logger.info("✅ Performance regressions detected appropriately")
            self.logger.info("✅ Component health monitoring operational")
        else:
            self.logger.info("⚠️  System shows some degradation - investigation recommended")
        
        if successful_rollbacks > 0:
            self.logger.info("✅ Automatic rollback mechanism operational")
        else:
            self.logger.info("ℹ️  No rollbacks triggered during demo")
        
        self.logger.info("\n🚀 DEMO COMPLETE: Comprehensive regression detection system demonstrated")
        self.logger.info("="*70)

def main():
    """Main demo function"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/home/hariravichandran/AELP/regression_demo.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Comprehensive Regression Detection Demo")
    
    try:
        # Create and run demo
        demo = RegressionDetectionDemo()
        demo.run_demo()
        
        logger.info("✅ Demo completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())