#!/usr/bin/env python3
"""
Demonstration of convergence monitoring integration in training loops.
Shows how to use the comprehensive convergence monitoring system.
"""

import sys
import os
import torch
import numpy as np
import json
import logging
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_training_with_convergence_monitoring():
    """
    Demonstrate how to integrate convergence monitoring into training.
    """
    print("=== CONVERGENCE MONITORING INTEGRATION DEMO ===")
    print("This shows how to use the convergence monitoring system in production.\n")
    
    from unittest.mock import Mock
    from fortified_rl_agent_no_hardcoding import ConvergenceMonitor
    
    # Mock setup (in production, use real agent and discovery engine)
    discovery_engine = Mock()
    discovery_engine._load_discovered_patterns = Mock(return_value={
        'channels': {
            'google_ads': {'performance': 0.8, 'cost': 0.5},
            'facebook': {'performance': 0.7, 'cost': 0.4},
            'display': {'performance': 0.6, 'cost': 0.3}
        },
        'segments': {
            'high_value': {'cvr': 0.08, 'value': 25.0},
            'medium_value': {'cvr': 0.05, 'value': 15.0}
        },
        'performance_metrics': {
            'cvr_stats': {'mean': 0.05, 'std': 0.02},
            'revenue_stats': {'mean': 15.0, 'std': 5.0}
        },
        'training_params': {
            'buffer_size': 1000,
            'learning_rate': 0.001,
            'epsilon': 0.3,
            'epsilon_min': 0.1,
            'min_episodes': 200  # Shorter for demo
        }
    })
    
    # Mock agent
    mock_agent = Mock()
    mock_agent.epsilon = 0.3
    mock_agent.epsilon_min = 0.1
    mock_agent.dropout_rate = 0.2
    mock_agent.discovered_channels = ['google_ads', 'facebook', 'display']
    mock_agent.optimizer_bid = Mock()
    mock_agent.optimizer_creative = Mock()
    mock_agent.optimizer_channel = Mock()
    mock_agent.optimizer_bid.param_groups = [{'lr': 0.001}]
    mock_agent.optimizer_creative.param_groups = [{'lr': 0.001}]
    mock_agent.optimizer_channel.param_groups = [{'lr': 0.001}]
    
    # Initialize convergence monitor
    monitor = ConvergenceMonitor(
        agent=mock_agent,
        discovery_engine=discovery_engine,
        checkpoint_dir="./demo_checkpoints"
    )
    
    print("✓ Convergence monitor initialized")
    print(f"  Emergency stop threshold: {monitor.thresholds['emergency_gradient_threshold']}")
    print(f"  Plateau detection threshold: {monitor.thresholds['plateau_threshold']}")
    print(f"  Exploration diversity threshold: {monitor.thresholds['exploration_diversity_threshold']}\n")
    
    # SIMULATION: Training loop with convergence monitoring
    print("Starting simulated training with real-time monitoring...")
    print("=" * 60)
    
    episode = 0
    total_steps = 0
    should_stop = False
    
    # Simulate 10 episodes of training
    while episode < 10 and not should_stop:
        episode += 1
        episode_reward = 0
        steps_in_episode = 0
        
        print(f"\n--- Episode {episode} ---")
        
        # Simulate steps within episode
        for step in range(20):  # 20 steps per episode for demo
            total_steps += 1
            steps_in_episode += 1
            
            # Simulate training metrics (realistic patterns)
            if step < 5:
                # Early in episode - higher loss, more exploration
                loss = np.random.uniform(0.5, 1.0)
                reward = np.random.uniform(-0.1, 0.3)
                gradient_norm = np.random.uniform(0.5, 2.0)
            elif step < 15:
                # Mid episode - decreasing loss, steady rewards
                loss = np.random.uniform(0.2, 0.6)
                reward = np.random.uniform(0.1, 0.7)
                gradient_norm = np.random.uniform(0.3, 1.5)
            else:
                # Late in episode - low loss, higher rewards
                loss = np.random.uniform(0.1, 0.4)
                reward = np.random.uniform(0.3, 1.0)
                gradient_norm = np.random.uniform(0.2, 1.0)
            
            episode_reward += reward
            
            # Create varied actions (good exploration)
            action = {
                'channel_action': step % 3,  # Cycle through channels
                'creative_action': (step // 2) % 2,  # Some creative variety
                'bid_action': step % 5  # Bid variety
            }
            
            # CRITICAL: Real-time convergence monitoring
            should_stop = monitor.monitor_step(
                loss=loss,
                reward=reward,
                gradient_norm=gradient_norm,
                action=action
            )
            
            # Print step info periodically
            if step % 10 == 0:
                print(f"  Step {step}: loss={loss:.3f}, reward={reward:.3f}, grad_norm={gradient_norm:.3f}")
            
            # Check if emergency stop triggered
            if should_stop:
                print(f"  ⚠️ EMERGENCY STOP triggered at step {step}!")
                break
        
        # End of episode monitoring
        should_stop_episode = monitor.end_episode(episode_reward)
        
        print(f"Episode {episode} completed: total_reward={episode_reward:.3f}, steps={steps_in_episode}")
        print(f"  Current epsilon: {mock_agent.epsilon:.3f}")
        print(f"  Alerts: {len(monitor.alerts)}, Critical: {len(monitor.critical_alerts)}")
        
        if should_stop_episode:
            print(f"  ⚠️ Training convergence detected or emergency stop!")
            should_stop = True
            break
        
        # Generate periodic convergence reports
        if episode % 3 == 0:
            report = monitor.generate_report()
            print(f"\n--- Convergence Report (Episode {episode}) ---")
            print(f"Training Status: {report['training_status']}")
            print(f"Current Metrics: {report['current_metrics']}")
            print(f"Alerts Summary: {report['alerts_summary']}")
            print(f"Exploration Metrics: {report['exploration_metrics']}")
    
    # Final report
    final_report = monitor.generate_report()
    print("\n" + "=" * 60)
    print("FINAL CONVERGENCE REPORT")
    print("=" * 60)
    print(f"Training completed after {episode} episodes, {total_steps} steps")
    print(f"Emergency stop: {monitor.emergency_stop_triggered}")
    print(f"Converged: {monitor.convergence_detected}")
    print(f"Total alerts: {len(monitor.alerts)}")
    print(f"Critical alerts: {len(monitor.critical_alerts)}")
    print(f"Interventions taken: {len(monitor.intervention_history)}")
    
    if monitor.alerts:
        print("\nRecent Alerts:")
        for alert in monitor.alerts[-5:]:
            print(f"  - {alert['message']}")
    
    if monitor.intervention_history:
        print("\nInterventions Taken:")
        for intervention in monitor.intervention_history:
            print(f"  - Step {intervention['step']}: {intervention['intervention']}")
    
    print(f"\nFinal agent state:")
    print(f"  Epsilon: {mock_agent.epsilon:.3f}")
    print(f"  Learning rates: {mock_agent.optimizer_bid.param_groups[0]['lr']:.6f}")
    print(f"  Dropout rate: {mock_agent.dropout_rate:.3f}")
    
    return final_report

def demo_problematic_training():
    """
    Demonstrate convergence monitoring with problematic training scenarios.
    """
    print("\n\n=== PROBLEMATIC TRAINING SCENARIO DEMO ===")
    print("Simulating training issues to demonstrate monitoring capabilities.\n")
    
    from unittest.mock import Mock
    from fortified_rl_agent_no_hardcoding import ConvergenceMonitor
    
    # Setup for problematic scenario
    discovery_engine = Mock()
    discovery_engine._load_discovered_patterns = Mock(return_value={
        'channels': {'google_ads': {}, 'facebook': {}},
        'segments': {'high_value': {}},
        'performance_metrics': {
            'cvr_stats': {'mean': 0.05, 'std': 0.02},
            'revenue_stats': {'mean': 15.0, 'std': 5.0}
        },
        'training_params': {'buffer_size': 1000, 'learning_rate': 0.001}
    })
    
    mock_agent = Mock()
    mock_agent.epsilon = 0.05  # Very low - will trigger premature convergence
    mock_agent.epsilon_min = 0.01
    mock_agent.dropout_rate = 0.2
    mock_agent.discovered_channels = ['google_ads', 'facebook']
    mock_agent.optimizer_bid = Mock()
    mock_agent.optimizer_creative = Mock()
    mock_agent.optimizer_channel = Mock()
    mock_agent.optimizer_bid.param_groups = [{'lr': 0.01}]  # High LR - may cause instability
    mock_agent.optimizer_creative.param_groups = [{'lr': 0.01}]
    mock_agent.optimizer_channel.param_groups = [{'lr': 0.01}]
    
    monitor = ConvergenceMonitor(
        agent=mock_agent,
        discovery_engine=discovery_engine,
        checkpoint_dir="./problem_demo_checkpoints"
    )
    
    monitor.episode = 50  # Simulate we're early in training
    
    print("Simulating problematic scenarios...")
    
    # Scenario 1: Loss explosion
    print("\n1. Testing loss explosion detection...")
    should_stop = monitor.monitor_step(
        loss=float('inf'),  # Infinite loss
        reward=0.5,
        gradient_norm=1.0,
        action={'channel_action': 0, 'creative_action': 0, 'bid_action': 0}
    )
    print(f"   Loss explosion detected: {should_stop}")
    print(f"   Emergency stop: {monitor.emergency_stop_triggered}")
    
    # Reset for next test
    monitor.emergency_stop_triggered = False
    
    # Scenario 2: Gradient explosion
    print("\n2. Testing gradient explosion detection...")
    should_stop = monitor.monitor_step(
        loss=0.3,
        reward=0.5,
        gradient_norm=1000.0,  # Huge gradient
        action={'channel_action': 0, 'creative_action': 0, 'bid_action': 0}
    )
    print(f"   Gradient explosion detected: {should_stop}")
    print(f"   Emergency stop: {monitor.emergency_stop_triggered}")
    
    # Reset for next test
    monitor.emergency_stop_triggered = False
    
    # Scenario 3: Premature convergence (low epsilon early)
    print("\n3. Testing premature convergence detection...")
    should_stop = monitor.monitor_step(
        loss=0.1,
        reward=0.5,
        gradient_norm=0.5,
        action={'channel_action': 0, 'creative_action': 0, 'bid_action': 0}
    )
    print(f"   Premature convergence detected: {len([a for a in monitor.alerts if 'TOO EARLY' in a['message']]) > 0}")
    print(f"   Epsilon after intervention: {mock_agent.epsilon:.3f}")
    
    # Scenario 4: Action repetition (memorization)
    print("\n4. Testing memorization detection...")
    repetitive_action = {'channel_action': 0, 'creative_action': 0, 'bid_action': 0}
    for i in range(101):  # Feed same action repeatedly
        monitor.monitor_step(
            loss=0.1,
            reward=0.5,
            gradient_norm=0.3,
            action=repetitive_action
        )
    
    memorization_alerts = [a for a in monitor.alerts if 'MEMORIZATION' in a['message'] or 'collapsed' in a['message']]
    print(f"   Memorization/exploration collapse detected: {len(memorization_alerts) > 0}")
    print(f"   Number of alerts: {len(memorization_alerts)}")
    
    print(f"\nFinal problematic training report:")
    print(f"  Total alerts: {len(monitor.alerts)}")
    print(f"  Critical alerts: {len(monitor.critical_alerts)}")
    print(f"  Interventions: {len(monitor.intervention_history)}")
    print(f"  Recent interventions:")
    for intervention in monitor.intervention_history[-3:]:
        print(f"    - {intervention['intervention']}")

def main():
    """Run convergence monitoring demonstrations"""
    
    # Demo 1: Normal training with monitoring
    report = demo_training_with_convergence_monitoring()
    
    # Demo 2: Problematic training scenarios  
    demo_problematic_training()
    
    print("\n\n=== INTEGRATION SUMMARY ===")
    print("The convergence monitoring system provides:")
    print("✓ Real-time training instability detection (NaN/Inf, gradient explosion)")
    print("✓ Premature convergence prevention (epsilon management)")
    print("✓ Action diversity monitoring (exploration tracking)")
    print("✓ Performance plateau detection (learning stagnation)")
    print("✓ Automatic intervention system (LR adjustment, exploration boost)")
    print("✓ Emergency checkpoint saving (crash recovery)")
    print("✓ Comprehensive reporting (training diagnostics)")
    print("✓ Early stopping capabilities (save compute time)")
    
    print("\nIntegration in production training:")
    print("1. Initialize ConvergenceMonitor with your agent and discovery engine")
    print("2. Call monitor.monitor_step() after each training step")
    print("3. Call monitor.end_episode() at the end of each episode")
    print("4. Check monitor.should_stop() to determine if training should halt")
    print("5. Use monitor.generate_report() for training diagnostics")
    print("6. Monitor logs for real-time alerts and interventions")
    
    print("\nNo fallback monitoring allowed - this is the production system!")

if __name__ == "__main__":
    main()