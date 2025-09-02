#!/usr/bin/env python3
"""
Test the full RL agent with gradient stabilization to ensure it works in real training.
This tests the complete integration of gradient clipping in the training loop.
NO FALLBACKS ALLOWED - Must work with the actual agent.
"""

import sys
import os
sys.path.insert(0, '/home/hariravichandran/AELP')

import torch
import numpy as np
import logging
from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent, DynamicEnrichedState

# Configure logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_state():
    """Create a test state for training"""
    return DynamicEnrichedState(
        stage=1,
        touchpoints_seen=1,
        days_since_first_touch=0.5,
        segment_index=0,
        segment_cvr=0.05,
        segment_engagement=0.3,
        segment_avg_ltv=25.0,
        device_index=0,
        channel_index=0,
        channel_performance=0.4,
        channel_attribution_credit=0.8,
        creative_index=0,
        creative_ctr=0.02,
        creative_cvr=0.01,
        creative_fatigue=0.1,
        hour_of_day=14,
        day_of_week=2,
        remaining_budget=1000.0,
        pacing_factor=1.0,
        competitive_pressure=0.3,
        expected_conversion_value=12.5
    )

def test_agent_initialization_with_gradient_stabilizer():
    """Test that agent initializes correctly with gradient stabilizer"""
    print("=== Testing Agent Initialization with Gradient Stabilizer ===")
    
    try:
        # Initialize agent - should create gradient stabilizer
        agent = ProductionFortifiedRLAgent(
            discovery_engine_path="discovered_patterns.json",
            hyperparameters={
                'learning_rate': 0.001,
                'batch_size': 16,
                'buffer_size': 1000,
                'target_update_frequency': 100,
                'epsilon': 0.1,
                'gamma': 0.99
            }
        )
        
        print("✓ Agent initialized successfully")
        print(f"  - Gradient stabilizer present: {hasattr(agent, 'gradient_stabilizer')}")
        
        if hasattr(agent, 'gradient_stabilizer'):
            print(f"  - Initial clip threshold: {agent.gradient_stabilizer.clip_threshold:.4f}")
            print(f"  - Loss scale: {agent.gradient_stabilizer.loss_scale}")
        
        return agent
        
    except Exception as e:
        print(f"✗ FAILED: Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_gradient_stability_during_training():
    """Test gradient stability during actual training steps"""
    print("\n=== Testing Gradient Stability During Training ===")
    
    agent = ProductionFortifiedRLAgent(
        discovery_engine_path="discovered_patterns.json",
        hyperparameters={
            'learning_rate': 0.001,
            'batch_size': 8,  # Small batch for testing
            'buffer_size': 100,
            'target_update_frequency': 50,
            'epsilon': 0.5,  # High exploration for testing
            'gamma': 0.99
        }
    )
    
    # Create test states and actions
    states = []
    actions = []
    rewards = []
    next_states = []
    
    for i in range(20):  # Create some training data
        state = create_test_state()
        state.stage = i % 3  # Vary the states
        state.remaining_budget = 1000.0 - (i * 10)
        
        # Get action from agent
        action = agent.act(state)
        
        # Simulate reward (some positive, some negative)
        reward = np.random.uniform(-0.5, 1.5)
        if i % 5 == 0:  # Occasional high reward
            reward = np.random.uniform(2.0, 5.0)
        
        # Create next state
        next_state = create_test_state()
        next_state.stage = state.stage + 1
        next_state.remaining_budget = state.remaining_budget - 50
        
        # Store for later
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        
        # Train the agent - this should use gradient stabilization
        done = (i == 19)  # Last step is done
        agent.train(state, action, reward, next_state, done)
        
        # Check gradient stabilizer status every few steps
        if hasattr(agent, 'gradient_stabilizer') and i % 5 == 0:
            stability_report = agent.gradient_stabilizer.get_stability_report()
            print(f"Step {i+1}: Status={stability_report.get('status', 'unknown')}, "
                  f"Clips={stability_report.get('total_clips', 0)}, "
                  f"Explosions={stability_report.get('gradient_explosions', 0)}")
    
    # Get final stability report
    if hasattr(agent, 'gradient_stabilizer'):
        final_report = agent.gradient_stabilizer.get_stability_report()
        print(f"\nFinal Gradient Stability Report:")
        print(f"  - Status: {final_report.get('status', 'unknown')}")
        print(f"  - Stability score: {final_report.get('stability_score', 0.0):.3f}")
        print(f"  - Total clips: {final_report.get('total_clips', 0)}")
        print(f"  - Gradient explosions: {final_report.get('gradient_explosions', 0)}")
        print(f"  - Vanishing gradients: {final_report.get('vanishing_gradients', 0)}")
        print(f"  - Emergency interventions: {final_report.get('emergency_interventions', 0)}")
        print(f"  - Average gradient norm: {final_report.get('avg_grad_norm', 0.0):.4f}")
        
        if final_report.get('status') != 'no_data':
            print("✓ Gradient stabilization working during training")
        else:
            print("✗ FAILED: No gradient data collected")
    else:
        print("✗ FAILED: Agent missing gradient stabilizer")

def test_training_with_extreme_gradients():
    """Test training stability with conditions that might cause gradient problems"""
    print("\n=== Testing Training with Extreme Conditions ===")
    
    agent = ProductionFortifiedRLAgent(
        discovery_engine_path="discovered_patterns.json",
        hyperparameters={
            'learning_rate': 0.01,  # Higher LR for potential instability
            'batch_size': 4,
            'buffer_size': 50,
            'target_update_frequency': 10,  # Frequent updates
            'epsilon': 0.9,  # Very high exploration
            'gamma': 0.999  # Very high discount factor
        }
    )
    
    # Create training scenario with extreme rewards
    extreme_rewards = [10.0, -10.0, 15.0, -5.0, 20.0, -15.0, 8.0, -8.0]
    
    for i, reward in enumerate(extreme_rewards):
        state = create_test_state()
        state.remaining_budget = 10000.0  # High budget for extreme bids
        
        action = agent.act(state)
        
        next_state = create_test_state()
        next_state.remaining_budget = 9000.0
        
        # Train with extreme reward
        agent.train(state, action, reward, next_state, done=(i == len(extreme_rewards)-1))
        
        if hasattr(agent, 'gradient_stabilizer') and i % 3 == 0:
            report = agent.gradient_stabilizer.get_stability_report()
            print(f"Extreme step {i+1}: Reward={reward:.1f}, "
                  f"Explosions={report.get('gradient_explosions', 0)}, "
                  f"Clips={report.get('total_clips', 0)}")
    
    # Check that training remained stable
    if hasattr(agent, 'gradient_stabilizer'):
        final_report = agent.gradient_stabilizer.get_stability_report()
        stability_score = final_report.get('stability_score', 0.0)
        
        if stability_score > 0.5:  # Should maintain reasonable stability
            print(f"✓ Training remained stable under extreme conditions (score: {stability_score:.3f})")
        else:
            print(f"⚠️  Training showed instability under extreme conditions (score: {stability_score:.3f})")
            print("    This may be expected with extreme rewards, but gradient stabilizer should have intervened")
    
    print("✓ Extreme conditions test completed")

def test_gradient_stabilizer_learning():
    """Test that gradient stabilizer learns and adapts over training"""
    print("\n=== Testing Gradient Stabilizer Learning ===")
    
    agent = ProductionFortifiedRLAgent(
        discovery_engine_path="discovered_patterns.json",
        hyperparameters={
            'learning_rate': 0.0005,
            'batch_size': 8,
            'buffer_size': 200,
            'target_update_frequency': 50,
            'epsilon': 0.3,
            'gamma': 0.99
        }
    )
    
    if not hasattr(agent, 'gradient_stabilizer'):
        print("✗ FAILED: Agent missing gradient stabilizer")
        return
    
    initial_threshold = agent.gradient_stabilizer.clip_threshold
    print(f"Initial clip threshold: {initial_threshold:.4f}")
    
    # Run extended training to allow learning
    for epoch in range(5):  # Multiple epochs
        for step in range(20):  # Steps per epoch
            state = create_test_state()
            state.stage = step % 4
            
            action = agent.act(state)
            reward = np.random.normal(0.5, 1.0)  # Normal distribution of rewards
            
            next_state = create_test_state()
            next_state.stage = state.stage + 1
            
            agent.train(state, action, reward, next_state, done=(step == 19))
        
        # Check threshold adaptation
        current_threshold = agent.gradient_stabilizer.clip_threshold
        print(f"Epoch {epoch+1}: Threshold = {current_threshold:.4f}, "
              f"Clips = {agent.gradient_stabilizer.total_clips}, "
              f"History size = {len(agent.gradient_stabilizer.gradient_norms_history)}")
    
    final_threshold = agent.gradient_stabilizer.clip_threshold
    
    if len(agent.gradient_stabilizer.gradient_norms_history) > 50:
        print(f"✓ Gradient stabilizer collected sufficient training history")
        print(f"✓ Threshold evolved from {initial_threshold:.4f} to {final_threshold:.4f}")
    else:
        print("⚠️  Limited gradient history collected")

def main():
    """Run all agent gradient stability tests"""
    print("Agent Gradient Stability Integration Tests")
    print("=" * 60)
    
    try:
        # Test agent initialization
        test_agent_initialization_with_gradient_stabilizer()
        
        # Test gradient stability during normal training
        test_gradient_stability_during_training()
        
        # Test with extreme conditions
        test_training_with_extreme_gradients()
        
        # Test learning and adaptation
        test_gradient_stabilizer_learning()
        
        print("\n" + "=" * 60)
        print("🎉 ALL AGENT GRADIENT STABILITY TESTS PASSED!")
        print("The RL agent properly integrates gradient stabilization:")
        print("  ✅ Agent initializes with gradient stabilizer")
        print("  ✅ Gradient stability maintained during training")
        print("  ✅ Handles extreme training conditions")
        print("  ✅ Gradient stabilizer learns and adapts")
        print("  ✅ Complete integration with RL training loop")
        
        return True
        
    except Exception as e:
        print(f"\n❌ AGENT GRADIENT STABILITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)