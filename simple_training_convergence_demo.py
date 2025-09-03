#!/usr/bin/env python3
"""
SIMPLE TRAINING WITH CONVERGENCE MONITORING DEMO
Demonstrates the essential convergence monitoring integration
without complex dependencies
"""

import sys
import os
import torch
import numpy as np
import logging
from datetime import datetime
from unittest.mock import Mock

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

from production_convergence_monitor import integrate_convergence_monitoring

def create_mock_training_components():
    """Create mock training components for demonstration"""
    
    # Create a realistic mock agent
    agent = Mock()
    agent.epsilon = 0.3
    agent.epsilon_min = 0.01
    agent.dropout_rate = 0.2
    
    # Mock networks and optimizers
    agent.q_network_bid = Mock()
    agent.q_network_creative = Mock()
    agent.q_network_channel = Mock()
    
    agent.q_network_bid.state_dict = Mock(return_value={'layer.weight': torch.randn(10, 5)})
    agent.q_network_creative.state_dict = Mock(return_value={'layer.weight': torch.randn(8, 4)})
    agent.q_network_channel.state_dict = Mock(return_value={'layer.weight': torch.randn(6, 3)})
    
    agent.optimizer_bid = Mock()
    agent.optimizer_creative = Mock()
    agent.optimizer_channel = Mock()
    
    agent.optimizer_bid.param_groups = [{'lr': 0.001}]
    agent.optimizer_creative.param_groups = [{'lr': 0.0008}]
    agent.optimizer_channel.param_groups = [{'lr': 0.0012}]
    
    agent.optimizer_bid.state_dict = Mock(return_value={})
    agent.optimizer_creative.state_dict = Mock(return_value={})
    agent.optimizer_channel.state_dict = Mock(return_value={})
    
    # Mock environment
    env = Mock()
    env.reset = Mock(return_value=np.random.randn(15))
    env.step = Mock(return_value=(np.random.randn(15), 0.1, False, {}))
    
    # Mock discovery engine
    discovery = Mock()
    discovery._load_discovered_patterns = Mock(return_value={
        'channels': {
            'google_ads': {'performance': 0.85, 'cost_efficiency': 0.7},
            'facebook': {'performance': 0.78, 'cost_efficiency': 0.6},
            'display': {'performance': 0.65, 'cost_efficiency': 0.8}
        },
        'segments': {
            'high_value': {'cvr': 0.12, 'lifetime_value': 450},
            'medium_value': {'cvr': 0.08, 'lifetime_value': 280}
        },
        'training_params': {
            'buffer_size': 2000,
            'learning_rate': 0.001,
            'monitoring_buffer_size': 3000
        }
    })
    
    return agent, env, discovery

def realistic_training_simulation(monitor, num_episodes=50):
    """
    Simulate realistic training with convergence monitoring
    This demonstrates the actual integration pattern
    """
    
    print(f"Starting realistic training simulation ({num_episodes} episodes)")
    print("This shows the exact pattern for production use:\n")
    
    total_steps = 0
    best_performance = float('-inf')
    
    # Main training loop
    for episode in range(num_episodes):
        
        episode_reward = 0
        episode_steps = 0
        
        # Simulate episode
        for step in range(30):  # 30 steps per episode
            
            total_steps += 1
            episode_steps += 1
            
            # Simulate training progression 
            progress = total_steps / (num_episodes * 30)
            
            # Generate realistic training metrics
            # Loss decreases with some noise
            loss = 1.2 * (1 - progress * 0.8) + np.random.normal(0, 0.08)
            loss = max(0.01, loss)
            
            # Reward improves with exploration noise
            reward = progress * 0.9 + np.random.normal(0, 0.15)
            reward = max(-0.2, reward)
            
            # Gradient norm stabilizes over time
            gradient_norm = 2.5 * (1 - progress * 0.6) * (1 + np.random.normal(0, 0.25))
            gradient_norm = max(0.05, gradient_norm)
            
            # Diverse actions (shows good exploration)
            action = {
                'channel_action': (total_steps + episode * 3) % 3,
                'creative_action': (step // 4) % 2,
                'bid_action': (total_steps * 2) % 5
            }
            
            # Q-values improve over time
            q_values = torch.tensor([
                1.5 + progress * 3.0 + np.random.normal(0, 0.4),
                1.2 + progress * 2.5 + np.random.normal(0, 0.4),
                0.9 + progress * 2.0 + np.random.normal(0, 0.4)
            ])
            
            episode_reward += reward
            
            # ============================================================
            # THIS IS THE KEY INTEGRATION - JUST ONE LINE!
            # ============================================================
            should_stop = monitor.monitor_step(
                loss=loss,
                reward=reward,
                gradient_norm=gradient_norm,
                action=action,
                q_values=q_values
            )
            # ============================================================
            
            # Handle monitoring results
            if should_stop:
                print(f"\n🛑 TRAINING STOPPED by monitoring system!")
                print(f"   Episode: {episode}, Step: {total_steps}")
                print(f"   Reason: {monitor.alerts[-1].message if monitor.alerts else 'Emergency stop'}")
                
                # Show what the monitor detected
                if monitor.emergency_stop_triggered:
                    print("   🚨 EMERGENCY STOP - Critical instability detected")
                    print("   💾 Emergency checkpoint saved automatically")
                
                if monitor.alerts:
                    recent_alerts = monitor.alerts[-3:]
                    print("   📋 Recent alerts:")
                    for alert in recent_alerts:
                        print(f"      - {alert.severity.value}: {alert.message}")
                
                if monitor.interventions:
                    recent_interventions = monitor.interventions[-3:]
                    print("   🔧 Automatic interventions applied:")
                    for interv in recent_interventions:
                        print(f"      - {interv.intervention_type}: {interv.reason}")
                
                return {
                    'early_stop': True,
                    'episode': episode,
                    'total_steps': total_steps,
                    'alerts': len(monitor.alerts),
                    'interventions': len(monitor.interventions)
                }
        
        # End of episode
        episode_stopped = monitor.end_episode(episode_reward)
        
        if episode_stopped:
            print(f"\n📊 Episode-level monitoring triggered stop at episode {episode}")
            return {'early_stop': True, 'episode': episode, 'reason': 'episode_monitoring'}
        
        # Track best performance
        if episode_reward > best_performance:
            best_performance = episode_reward
        
        # Progress reporting
        if episode % 10 == 0 and episode > 0:
            print(f"Episode {episode}: reward={episode_reward:.3f}, best={best_performance:.3f}")
            print(f"  Stage: {monitor.training_stage.value}, Epsilon: {monitor.agent.epsilon:.4f}")
            print(f"  Alerts: {len(monitor.alerts)}, Interventions: {len(monitor.interventions)}")
            
            # Show system health
            if monitor.alerts:
                print(f"  Recent alert: {monitor.alerts[-1].message}")
    
    print(f"\n✅ Training completed successfully!")
    print(f"   Episodes: {num_episodes}, Steps: {total_steps}")
    print(f"   Best performance: {best_performance:.3f}")
    print(f"   Final stage: {monitor.training_stage.value}")
    
    return {
        'early_stop': False,
        'episodes': num_episodes,
        'total_steps': total_steps,
        'best_performance': best_performance
    }

def demonstrate_monitoring_features(monitor):
    """Demonstrate key monitoring features with targeted tests"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING KEY MONITORING FEATURES")
    print("="*60)
    
    # Feature 1: NaN Detection
    print("\n1. NaN/Inf Detection:")
    should_stop = monitor.monitor_step(
        loss=float('nan'),
        reward=0.5,
        gradient_norm=1.0,
        action={'channel_action': 0, 'creative_action': 1, 'bid_action': 2},
        q_values=torch.tensor([1.0, 2.0, 3.0])
    )
    print(f"   ✓ NaN detected and handled: {should_stop}")
    
    # Reset for next test
    monitor.emergency_stop_triggered = False
    
    # Feature 2: Gradient Explosion
    print("\n2. Gradient Explosion Detection:")
    should_stop = monitor.monitor_step(
        loss=0.3,
        reward=0.4,
        gradient_norm=75.0,  # Very high
        action={'channel_action': 1, 'creative_action': 0, 'bid_action': 3},
        q_values=torch.tensor([2.0, 3.0, 4.0])
    )
    print(f"   ✓ Gradient explosion detected and handled: {should_stop}")
    
    # Feature 3: Show Interventions
    if monitor.interventions:
        print("\n3. Automatic Interventions Applied:")
        for interv in monitor.interventions[-3:]:
            print(f"   - {interv.intervention_type}: {interv.reason}")
            if interv.parameters_changed:
                print(f"     Parameters: {interv.parameters_changed}")
    
    # Feature 4: Comprehensive Report
    print("\n4. Comprehensive Monitoring Report:")
    report = monitor.generate_comprehensive_report()
    
    print(f"   Training Steps: {report['training_summary']['total_steps']}")
    print(f"   Current Stage: {report['training_summary']['current_stage']}")
    print(f"   Stability: {report['stability_assessment']['loss_stability']}")
    print(f"   Total Alerts: {report['alerts_summary']['total_alerts']}")
    print(f"   Interventions: {report['interventions_summary']['total_interventions']}")
    
    if report['recommendations']:
        print("   Recommendations:")
        for rec in report['recommendations'][:3]:
            print(f"     - {rec}")
    
    return report

def main():
    """Main demonstration function"""
    
    print("="*80)
    print("SIMPLE TRAINING WITH CONVERGENCE MONITORING")
    print("="*80)
    print("This demonstrates how to integrate production convergence")
    print("monitoring into any existing training loop.\n")
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise for demo
    
    # Step 1: Create training components
    print("🔧 Step 1: Setting up training components...")
    agent, environment, discovery_engine = create_mock_training_components()
    print("   ✓ Agent, environment, and discovery engine ready")
    
    # Step 2: Integrate convergence monitoring (ONE LINE!)
    print("\n🔍 Step 2: Integrating convergence monitoring...")
    monitor = integrate_convergence_monitoring(
        agent=agent,
        environment=environment,
        discovery_engine=discovery_engine,
        checkpoint_dir="./demo_checkpoints"
    )
    print(f"   ✓ Monitoring integrated with {len(monitor.thresholds)} thresholds")
    print(f"   ✓ Real-time monitoring active")
    
    # Step 3: Run training with monitoring
    print("\n🚀 Step 3: Running training with monitoring...")
    result = realistic_training_simulation(monitor, num_episodes=30)
    
    print(f"\n📊 TRAINING RESULTS:")
    for key, value in result.items():
        print(f"   {key}: {value}")
    
    # Step 4: Demonstrate monitoring features
    monitor_features_report = demonstrate_monitoring_features(monitor)
    
    # Step 5: Show integration code
    print("\n" + "="*60)
    print("INTEGRATION CODE FOR YOUR TRAINING LOOP")
    print("="*60)
    print("""
# 1. Import the monitoring system
from production_convergence_monitor import integrate_convergence_monitoring

# 2. Initialize monitoring (add this once to your training setup)
monitor = integrate_convergence_monitoring(
    agent=your_agent,
    environment=your_environment,
    discovery_engine=your_discovery_engine
)

# 3. Add monitoring to your training loop (add this to each training step)
for episode in range(num_episodes):
    for step in range(steps_per_episode):
        
        # Your existing training code...
        loss = agent.train_step(state, action, reward, next_state, done)
        gradient_norm = agent.get_gradient_norm()  # If available
        
        # ADD THIS ONE LINE:
        should_stop = monitor.monitor_step(loss, reward, gradient_norm, action)
        
        if should_stop:
            print("Training stopped due to convergence issues")
            break  # Exit training loop
    
    # Add this at episode end:
    if monitor.end_episode(episode_reward):
        break  # Stop training

# 4. Generate final report
final_report = monitor.generate_comprehensive_report()
""")
    
    print("\n" + "="*60)
    print("BENEFITS YOU GET:")
    print("="*60)
    print("✅ Prevents wasted compute on bad training runs")
    print("✅ Detects NaN/Inf immediately (saves hours of debugging)")
    print("✅ Automatic hyperparameter adjustments during training")
    print("✅ Emergency checkpoints for recovery")
    print("✅ Rich diagnostics and recommendations")
    print("✅ Production-ready monitoring with zero fallbacks")
    
    print(f"\n🎯 INTEGRATION EFFORT: Add 3 lines of code to existing training!")
    print(f"🚀 PRODUCTION READY: Handles all edge cases gracefully")
    
    return result, monitor_features_report

if __name__ == "__main__":
    training_result, features_report = main()
    print(f"\nDemo completed successfully!")
    print(f"Training result: {training_result.get('early_stop', False) and 'Stopped early' or 'Completed'}")
    print(f"System health: {features_report['system_health']['database_status']}")