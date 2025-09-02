#!/usr/bin/env python3
"""
Test convergence monitoring functionality in the fortified RL agent.
"""

import sys
import os
import torch
import numpy as np
import json
import tempfile
from unittest.mock import Mock, MagicMock

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

def setup_test_environment():
    """Setup test environment with mocked components"""
    
    # Create mock discovery engine with realistic patterns
    discovery_engine = Mock()
    discovery_engine._load_discovered_patterns = Mock(return_value={
        'channels': {
            'google_ads': {'performance': 0.8, 'cost': 0.5},
            'facebook': {'performance': 0.7, 'cost': 0.4},
            'display': {'performance': 0.6, 'cost': 0.3}
        },
        'segments': {
            'high_value': {'cvr': 0.08, 'value': 25.0},
            'medium_value': {'cvr': 0.05, 'value': 15.0},
            'low_value': {'cvr': 0.02, 'value': 8.0}
        },
        'performance_metrics': {
            'cvr_stats': {'mean': 0.05, 'std': 0.02},
            'revenue_stats': {'mean': 15.0, 'std': 5.0}
        },
        'training_params': {
            'buffer_size': 1000,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epsilon': 0.3,
            'epsilon_decay': 0.999,
            'epsilon_min': 0.1,
            'gamma': 0.95,
            'training_frequency': 10,
            'target_update_frequency': 100,
            'dropout_rate': 0.2,
            'warm_start_steps': 100,
            'plateau_patience': 50,
            'min_episodes': 500
        }
    })
    
    # Create mock agent
    mock_agent = Mock()
    mock_agent.epsilon = 0.3
    mock_agent.epsilon_min = 0.1
    mock_agent.discovered_channels = ['google_ads', 'facebook', 'display']
    mock_agent.discovered_segments = ['high_value', 'medium_value', 'low_value']
    mock_agent.q_network_bid = Mock()
    mock_agent.q_network_creative = Mock()
    mock_agent.q_network_channel = Mock()
    mock_agent.optimizer_bid = Mock()
    mock_agent.optimizer_creative = Mock()
    mock_agent.optimizer_channel = Mock()
    
    # Mock state dictionaries
    mock_agent.q_network_bid.state_dict = Mock(return_value={})
    mock_agent.q_network_creative.state_dict = Mock(return_value={})
    mock_agent.q_network_channel.state_dict = Mock(return_value={})
    mock_agent.optimizer_bid.state_dict = Mock(return_value={})
    mock_agent.optimizer_creative.state_dict = Mock(return_value={})
    mock_agent.optimizer_channel.state_dict = Mock(return_value={})
    
    # Mock parameter groups for learning rate adjustment
    mock_agent.optimizer_bid.param_groups = [{'lr': 0.001}]
    mock_agent.optimizer_creative.param_groups = [{'lr': 0.001}]
    mock_agent.optimizer_channel.param_groups = [{'lr': 0.001}]
    
    # Mock Q-value tracking
    mock_agent.q_value_tracking = {
        'overestimation_bias': [0.1, 0.12, 0.08, 0.15, 0.09]
    }
    
    # Mock dropout rate for regularization testing
    mock_agent.dropout_rate = 0.2
    
    return discovery_engine, mock_agent

def test_convergence_monitor_initialization():
    """Test convergence monitor initialization"""
    print("Testing convergence monitor initialization...")
    
    from fortified_rl_agent_no_hardcoding import ConvergenceMonitor
    
    discovery_engine, mock_agent = setup_test_environment()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = ConvergenceMonitor(
            agent=mock_agent,
            discovery_engine=discovery_engine,
            checkpoint_dir=temp_dir
        )
        
        assert monitor.agent == mock_agent
        assert monitor.discovery == discovery_engine
        assert monitor.checkpoint_dir == temp_dir
        assert len(monitor.thresholds) > 0
        assert len(monitor.convergence_criteria) > 0
        
        print("✓ Convergence monitor initialized successfully")
        print(f"  Thresholds: {monitor.thresholds}")
        print(f"  Criteria: {monitor.convergence_criteria}")
        
        return monitor

def test_training_instability_detection():
    """Test training instability detection"""
    print("\nTesting training instability detection...")
    
    monitor = test_convergence_monitor_initialization()
    
    # Test NaN detection
    should_stop = monitor.monitor_step(
        loss=float('nan'), 
        reward=0.5, 
        gradient_norm=1.0,
        action={'channel_action': 0, 'creative_action': 1, 'bid_action': 2}
    )
    
    assert should_stop == True
    assert monitor.emergency_stop_triggered == True
    assert len(monitor.critical_alerts) > 0
    print("✓ NaN detection working")
    
    # Reset for next test
    monitor.emergency_stop_triggered = False
    monitor.critical_alerts = []
    
    # Test gradient explosion
    should_stop = monitor.monitor_step(
        loss=0.5,
        reward=0.3,
        gradient_norm=1000.0,  # Very high gradient
        action={'channel_action': 0, 'creative_action': 1, 'bid_action': 2}
    )
    
    assert should_stop == True
    assert monitor.emergency_stop_triggered == True
    print("✓ Gradient explosion detection working")

def test_premature_convergence_detection():
    """Test premature convergence detection"""
    print("\nTesting premature convergence detection...")
    
    monitor = test_convergence_monitor_initialization()
    monitor.episode = 100  # Early episode
    
    # Force epsilon to minimum to trigger premature convergence
    monitor.agent.epsilon = monitor.agent.epsilon_min
    
    should_stop = monitor.monitor_step(
        loss=0.1,
        reward=0.3, 
        gradient_norm=0.5,
        action={'channel_action': 0, 'creative_action': 1, 'bid_action': 2}
    )
    
    # Should detect premature convergence
    assert len([alert for alert in monitor.alerts if 'TOO EARLY' in alert['message']]) > 0
    print("✓ Premature convergence detection working")

def test_action_diversity_monitoring():
    """Test action diversity monitoring"""
    print("\nTesting action diversity monitoring...")
    
    monitor = test_convergence_monitor_initialization()
    
    # Feed repetitive actions to trigger low diversity alert
    repetitive_action = {'channel_action': 0, 'creative_action': 0, 'bid_action': 0}
    
    for i in range(101):  # Need 100+ actions for diversity check
        monitor.monitor_step(
            loss=0.1,
            reward=0.3,
            gradient_norm=0.5,
            action=repetitive_action
        )
    
    # Should detect exploration collapse or low diversity
    diversity_alerts = [alert for alert in monitor.alerts 
                       if 'unique actions' in alert['message'] or 'collapsed' in alert['message']]
    assert len(diversity_alerts) > 0
    print("✓ Action diversity monitoring working")

def test_plateau_detection():
    """Test performance plateau detection"""
    print("\nTesting plateau detection...")
    
    monitor = test_convergence_monitor_initialization()
    
    # Feed constant rewards to create plateau
    for i in range(250):  # Need 200+ rewards for plateau detection
        monitor.monitor_step(
            loss=0.1,
            reward=0.5,  # Constant reward
            gradient_norm=0.3,
            action={'channel_action': i % 3, 'creative_action': i % 2, 'bid_action': i % 5}
        )
    
    # Should detect plateau
    plateau_alerts = [alert for alert in monitor.alerts if 'plateau' in alert['message'].lower()]
    assert len(plateau_alerts) > 0
    print("✓ Plateau detection working")

def test_intervention_system():
    """Test automatic intervention system"""
    print("\nTesting intervention system...")
    
    monitor = test_convergence_monitor_initialization()
    
    # Test exploration increase intervention
    old_epsilon = monitor.agent.epsilon
    monitor.increase_exploration()
    
    assert monitor.agent.epsilon > old_epsilon
    assert len(monitor.intervention_history) > 0
    print("✓ Exploration increase intervention working")
    
    # Test learning rate adjustment
    old_lr = monitor.agent.optimizer_bid.param_groups[0]['lr']
    monitor.adjust_learning_parameters()
    
    assert monitor.agent.optimizer_bid.param_groups[0]['lr'] < old_lr
    print("✓ Learning rate adjustment working")

def test_emergency_checkpoint():
    """Test emergency checkpoint saving"""
    print("\nTesting emergency checkpoint system...")
    
    monitor = test_convergence_monitor_initialization()
    
    # Test emergency checkpoint
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor.checkpoint_dir = temp_dir
        monitor.training_step = 100
        monitor.episode = 50
        
        monitor.save_emergency_checkpoint()
        
        # Check if checkpoint file was created
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.startswith('emergency_checkpoint')]
        assert len(checkpoint_files) > 0
        print("✓ Emergency checkpoint system working")

def test_convergence_report():
    """Test convergence report generation"""
    print("\nTesting convergence report generation...")
    
    monitor = test_convergence_monitor_initialization()
    
    # Add some sample data
    for i in range(50):
        monitor.monitor_step(
            loss=0.1 + 0.01 * i,
            reward=0.3 + 0.01 * i,
            gradient_norm=0.5,
            action={'channel_action': i % 3, 'creative_action': i % 2, 'bid_action': i % 5}
        )
    
    report = monitor.generate_report()
    
    assert 'training_status' in report
    assert 'current_metrics' in report
    assert 'stability_metrics' in report
    assert 'alerts_summary' in report
    assert 'exploration_metrics' in report
    
    print("✓ Convergence report generation working")
    print(f"  Sample report keys: {list(report.keys())}")

def main():
    """Run all convergence monitoring tests"""
    print("=== CONVERGENCE MONITORING TESTS ===")
    
    try:
        test_convergence_monitor_initialization()
        test_training_instability_detection()
        test_premature_convergence_detection()
        test_action_diversity_monitoring()
        test_plateau_detection()
        test_intervention_system()
        test_emergency_checkpoint()
        test_convergence_report()
        
        print("\n✅ ALL CONVERGENCE MONITORING TESTS PASSED!")
        print("\nConvergence monitoring system is ready for production use:")
        print("- Real-time instability detection ✓")
        print("- Premature convergence detection ✓")  
        print("- Action diversity monitoring ✓")
        print("- Performance plateau detection ✓")
        print("- Automatic intervention system ✓")
        print("- Emergency checkpoint system ✓")
        print("- Comprehensive reporting ✓")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)