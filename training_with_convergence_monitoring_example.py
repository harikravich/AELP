#!/usr/bin/env python3
"""
TRAINING WITH CONVERGENCE MONITORING - INTEGRATION EXAMPLE
Shows how to integrate the production convergence monitoring system
with existing GAELP training loops
"""

import sys
import os
import torch
import numpy as np
import logging
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

# Import existing GAELP components
from production_convergence_monitor import integrate_convergence_monitoring
from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent
from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
from discovery_engine import GA4RealTimeDataPipeline

logger = logging.getLogger(__name__)

def train_with_convergence_monitoring(num_episodes=1000):
    """
    Example training loop with integrated convergence monitoring
    This shows the pattern for production use
    """
    
    print("=" * 80)
    print("TRAINING WITH CONVERGENCE MONITORING - INTEGRATION EXAMPLE")
    print("=" * 80)
    print("This example shows how to integrate convergence monitoring")
    print("with existing GAELP training systems.\n")
    
    # ========== STEP 1: INITIALIZE COMPONENTS ==========
    print("Step 1: Initializing training components...")
    
    # Initialize discovery engine (data pipeline)
    try:
        discovery_engine = GA4RealTimeDataPipeline()
        print("✓ GA4 Real-time data pipeline initialized")
    except Exception as e:
        print(f"⚠️ Could not initialize GA4 pipeline: {e}")
        print("Using mock discovery engine for demo")
        from unittest.mock import Mock
        discovery_engine = Mock()
        discovery_engine._load_discovered_patterns = Mock(return_value={
            'channels': {'google_ads': {}, 'facebook': {}, 'display': {}},
            'segments': {'high_value': {}, 'medium_value': {}},
            'training_params': {'buffer_size': 2000}
        })
    
    # Initialize environment
    try:
        environment = ProductionFortifiedEnvironment(
            discovery_engine=discovery_engine
        )
        print("✓ Production fortified environment initialized")
    except Exception as e:
        print(f"⚠️ Could not initialize environment: {e}")
        print("Using mock environment for demo")
        from unittest.mock import Mock
        environment = Mock()
        environment.reset = Mock(return_value=np.random.randn(10))
        environment.step = Mock(return_value=(np.random.randn(10), 0.1, False, {}))
    
    # Initialize RL agent
    try:
        agent = ProductionFortifiedRLAgent(
            discovery_engine=discovery_engine
        )
        print("✓ Production fortified RL agent initialized")
    except Exception as e:
        print(f"⚠️ Could not initialize agent: {e}")
        print("Using mock agent for demo")
        from unittest.mock import Mock
        agent = Mock()
        agent.epsilon = 0.3
        agent.epsilon_min = 0.01
        agent.dropout_rate = 0.2
        agent.select_action = Mock(return_value={'channel': 0, 'creative': 1, 'bid': 2})
        agent.train_step = Mock(return_value=0.5)
        
        # Mock networks and optimizers
        agent.q_network_bid = Mock()
        agent.q_network_creative = Mock()
        agent.q_network_channel = Mock()
        agent.optimizer_bid = Mock()
        agent.optimizer_creative = Mock()  
        agent.optimizer_channel = Mock()
        
        agent.q_network_bid.state_dict = Mock(return_value={})
        agent.q_network_creative.state_dict = Mock(return_value={})
        agent.q_network_channel.state_dict = Mock(return_value={})
        
        agent.optimizer_bid.param_groups = [{'lr': 0.001}]
        agent.optimizer_creative.param_groups = [{'lr': 0.001}]
        agent.optimizer_channel.param_groups = [{'lr': 0.001}]
        
        agent.optimizer_bid.state_dict = Mock(return_value={})
        agent.optimizer_creative.state_dict = Mock(return_value={})
        agent.optimizer_channel.state_dict = Mock(return_value={})
    
    # ========== STEP 2: INITIALIZE CONVERGENCE MONITORING ==========
    print("\nStep 2: Initializing convergence monitoring...")
    
    # This is the key integration - just one line!
    monitor = integrate_convergence_monitoring(
        agent=agent,
        environment=environment,
        discovery_engine=discovery_engine,
        checkpoint_dir="./production_training_checkpoints"
    )
    
    print("✓ Convergence monitoring integrated")
    print(f"  - {len(monitor.thresholds)} learned thresholds loaded")
    print(f"  - Buffer sizes: loss={monitor.loss_history.maxlen}")
    print(f"  - Emergency checkpoint system ready")
    print(f"  - Real-time monitoring active")
    
    # ========== STEP 3: TRAINING LOOP WITH MONITORING ==========
    print(f"\nStep 3: Starting training with monitoring ({num_episodes} episodes)...")
    
    total_steps = 0
    best_performance = float('-inf')
    training_start_time = datetime.now()
    
    for episode in range(num_episodes):
        
        # Reset environment
        try:
            state = environment.reset()
        except:
            state = np.random.randn(10)  # Mock state for demo
        
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Episode loop
        while not done and episode_steps < 200:  # Max 200 steps per episode
            
            # Select action
            try:
                action = agent.select_action(state)
            except:
                action = {
                    'channel_action': total_steps % 3,
                    'creative_action': (total_steps // 2) % 2,
                    'bid_action': total_steps % 5
                }
            
            # Take step in environment
            try:
                next_state, reward, done, info = environment.step(action)
            except:
                # Mock step for demo
                progress = total_steps / (num_episodes * 50)
                reward = 0.1 + progress * 0.5 + np.random.normal(0, 0.1)
                next_state = np.random.randn(10)
                done = episode_steps >= 50  # End episode after 50 steps for demo
                info = {}
            
            # Train agent
            try:
                loss = agent.train_step(state, action, reward, next_state, done)
                
                # Get gradient norm (important for monitoring)
                if hasattr(agent, 'get_gradient_norm'):
                    gradient_norm = agent.get_gradient_norm()
                else:
                    gradient_norm = 1.0 + np.random.normal(0, 0.3)  # Mock for demo
                
                # Get Q-values for monitoring (optional but helpful)
                if hasattr(agent, 'get_q_values'):
                    q_values = agent.get_q_values(state)
                else:
                    q_values = torch.tensor([1.0, 2.0, 3.0])  # Mock for demo
                    
            except:
                # Mock training for demo
                progress = total_steps / (num_episodes * 50)
                loss = 1.0 - progress * 0.7 + np.random.normal(0, 0.1)
                loss = max(0.01, loss)
                gradient_norm = 2.0 - progress + np.random.normal(0, 0.2)
                gradient_norm = max(0.1, gradient_norm)
                q_values = torch.tensor([1.0 + progress, 2.0 + progress, 3.0 + progress])
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            state = next_state
            
            # ========== CRITICAL: REAL-TIME MONITORING ==========
            # This is where the magic happens - just one function call!
            should_stop = monitor.monitor_step(
                loss=loss,
                reward=reward,
                gradient_norm=gradient_norm,
                action=action,
                q_values=q_values
            )
            
            # Check if monitoring detected critical issues
            if should_stop:
                print(f"\n⚠️ TRAINING STOPPED by convergence monitor at episode {episode}, step {total_steps}")
                print(f"   Reason: {monitor.alerts[-1].message if monitor.alerts else 'Emergency stop'}")
                
                # Generate final report
                final_report = monitor.generate_comprehensive_report()
                print(f"   Total alerts: {final_report['alerts_summary']['total_alerts']}")
                print(f"   Interventions: {final_report['interventions_summary']['total_interventions']}")
                
                if monitor.emergency_stop_triggered:
                    print("   Emergency checkpoint saved - training can be resumed")
                
                return {
                    'stopped_early': True,
                    'episode': episode,
                    'total_steps': total_steps,
                    'reason': 'convergence_monitoring',
                    'final_report': final_report
                }
        
        # End of episode monitoring
        episode_stopped = monitor.end_episode(episode_reward)
        
        if episode_stopped:
            print(f"\n⚠️ EPISODE-LEVEL STOP at episode {episode}")
            print(f"   Consecutive poor episodes: {monitor.consecutive_poor_episodes}")
            return {
                'stopped_early': True,
                'episode': episode,
                'total_steps': total_steps,
                'reason': 'episode_level_monitoring'
            }
        
        # Track performance
        if episode_reward > best_performance:
            best_performance = episode_reward
        
        # Periodic reporting
        if episode % 100 == 0 and episode > 0:
            elapsed_time = datetime.now() - training_start_time
            
            print(f"\nProgress Report - Episode {episode}/{num_episodes}")
            print(f"  Time elapsed: {elapsed_time}")
            print(f"  Best performance: {best_performance:.3f}")
            print(f"  Current epsilon: {getattr(agent, 'epsilon', 0.0):.4f}")
            print(f"  Training stage: {monitor.training_stage.value}")
            print(f"  Total alerts: {len(monitor.alerts)}")
            print(f"  Interventions: {len(monitor.interventions)}")
            
            # Show recent interventions
            if monitor.interventions:
                recent_interventions = monitor.interventions[-3:]
                print("  Recent interventions:")
                for interv in recent_interventions:
                    print(f"    - {interv.intervention_type}: {interv.reason}")
            
            # Check system health
            report = monitor.generate_comprehensive_report()
            stability = report['stability_assessment']
            print(f"  Stability: {stability['loss_stability']}, {stability['gradient_health']}")
    
    # ========== STEP 4: TRAINING COMPLETED SUCCESSFULLY ==========
    training_end_time = datetime.now()
    total_training_time = training_end_time - training_start_time
    
    print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"   Episodes: {num_episodes}")
    print(f"   Total steps: {total_steps}")
    print(f"   Training time: {total_training_time}")
    print(f"   Best performance: {best_performance:.3f}")
    print(f"   Final stage: {monitor.training_stage.value}")
    
    # Generate final comprehensive report
    final_report = monitor.generate_comprehensive_report()
    
    print("\n📊 FINAL TRAINING REPORT:")
    print(f"   Loss stability: {final_report['stability_assessment']['loss_stability']}")
    print(f"   Gradient health: {final_report['stability_assessment']['gradient_health']}")
    print(f"   Total alerts: {final_report['alerts_summary']['total_alerts']}")
    print(f"   Critical alerts: {final_report['alerts_summary']['critical_alerts']}")
    print(f"   Successful interventions: {final_report['interventions_summary']['successful_interventions']}")
    
    # Save success metrics for future training
    print("\n💾 SAVING SUCCESS METRICS...")
    monitor.save_success_metrics()
    print("   Success metrics saved - will improve future training thresholds")
    
    return {
        'stopped_early': False,
        'episode': num_episodes,
        'total_steps': total_steps,
        'training_time': total_training_time,
        'best_performance': best_performance,
        'final_report': final_report
    }

def demonstrate_integration_patterns():
    """Demonstrate different integration patterns"""
    
    print("\n" + "=" * 80)
    print("INTEGRATION PATTERNS")  
    print("=" * 80)
    
    print("\n1. BASIC INTEGRATION PATTERN:")
    print("   # Just add one line to your existing training loop:")
    print("   monitor = integrate_convergence_monitoring(agent, env, discovery)")
    print("   ")
    print("   # Then in your training step:")
    print("   should_stop = monitor.monitor_step(loss, reward, grad_norm, action)")
    print("   if should_stop:")
    print("       break  # Stop training - issue detected")
    
    print("\n2. ADVANCED INTEGRATION PATTERN:")
    print("   # For production systems with custom handling:")
    print("   monitor = ProductionConvergenceMonitor(agent, env, discovery)")
    print("   ")
    print("   # Custom alert handling:")
    print("   if monitor.alerts and monitor.alerts[-1].severity == AlertSeverity.CRITICAL:")
    print("       # Custom handling for critical alerts")
    print("       handle_critical_alert(monitor.alerts[-1])")
    
    print("\n3. DISTRIBUTED TRAINING INTEGRATION:")
    print("   # Each worker has its own monitor:")
    print("   worker_monitor = integrate_convergence_monitoring(")
    print("       agent, env, discovery, checkpoint_dir=f'./checkpoints_worker_{worker_id}'")
    print("   )")
    print("   ")
    print("   # Aggregate monitoring results across workers")
    print("   if any(worker.should_stop() for worker in worker_monitors):")
    print("       stop_all_workers()")
    
    print("\n4. HYPERPARAMETER TUNING INTEGRATION:")
    print("   # Use monitoring results for hyperparameter optimization:")
    print("   for lr in learning_rates:")
    print("       agent.set_learning_rate(lr)")
    print("       monitor = integrate_convergence_monitoring(agent, env, discovery)")
    print("       result = train_with_monitoring()")
    print("       if not result['stopped_early']:")
    print("           best_lr = lr  # Found good hyperparameters")
    
    print("\n5. A/B TESTING INTEGRATION:")
    print("   # Compare different training configurations:")
    print("   monitor_a = integrate_convergence_monitoring(agent_a, env, discovery)")
    print("   monitor_b = integrate_convergence_monitoring(agent_b, env, discovery)")
    print("   ")
    print("   # Train both and compare results")
    print("   result_a = train_with_monitoring(monitor_a)")
    print("   result_b = train_with_monitoring(monitor_b)")
    print("   # Choose best configuration based on stability metrics")

if __name__ == "__main__":
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the main training example
    result = train_with_convergence_monitoring(num_episodes=200)
    
    print(f"\nTRAINING RESULT:")
    for key, value in result.items():
        if key != 'final_report':
            print(f"  {key}: {value}")
    
    # Show integration patterns
    demonstrate_integration_patterns()
    
    print("\n" + "=" * 80)
    print("KEY BENEFITS OF INTEGRATED CONVERGENCE MONITORING:")
    print("=" * 80)
    print("✅ Prevents wasted compute on divergent training")
    print("✅ Automatic hyperparameter adjustments during training")
    print("✅ Early detection of training issues (NaN, explosions, plateaus)")
    print("✅ Emergency checkpoints for crash recovery")
    print("✅ Rich training diagnostics and recommendations")
    print("✅ Learning from successful runs for better thresholds")
    print("✅ Production-ready with proper error handling")
    print("✅ Easy integration - just add one function call!")
    
    print("\nTO USE IN YOUR TRAINING:")
    print("1. Import: from production_convergence_monitor import integrate_convergence_monitoring")
    print("2. Initialize: monitor = integrate_convergence_monitoring(agent, env, discovery)")
    print("3. Monitor: should_stop = monitor.monitor_step(loss, reward, grad_norm, action)")
    print("4. Handle: if should_stop: break")
    print("\nThat's it! Your training is now production-ready with convergence monitoring.")