#!/usr/bin/env python3
"""
Simple Learning Verification for GAELP Agents
Direct test of whether RL agents are actually learning
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import sys
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gradient_flow_monitor import GradientFlowMonitor, instrument_training_step

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_neural_network_learning():
    """Test that we can detect learning in a basic neural network"""
    logger.info("="*60)
    logger.info("TESTING BASIC NEURAL NETWORK LEARNING")
    logger.info("="*60)
    
    # Create simple regression problem
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data: y = 2*x1 + 3*x2 + 1 + noise
    X = torch.randn(1000, 2)
    y = 2*X[:, 0:1] + 3*X[:, 1:2] + 1 + 0.1*torch.randn(1000, 1)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Track learning metrics
    monitor = GradientFlowMonitor(window_size=100)
    
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.data.clone()
    
    losses = []
    gradient_norms = []
    weight_changes = []
    
    logger.info("Training neural network...")
    
    for epoch in range(100):
        # Forward pass
        pred = model(X)
        loss = criterion(pred, y)
        
        # Store loss
        losses.append(loss.item())
        
        # Check gradient flow
        grad_info = monitor.check_gradient_flow(model, loss, compute_backward=True)
        gradient_norms.append(grad_info.get('total_norm', 0))
        
        # Apply gradient clipping if needed
        if grad_info.get('total_norm', 0) > 10:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Optimizer step
        optimizer.step()
        
        # Calculate weight change
        total_change = 0
        for name, param in model.named_parameters():
            if name in initial_weights:
                change = torch.norm(param.data - initial_weights[name]).item()
                total_change += change
        weight_changes.append(total_change)
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: Loss={loss.item():.6f}, "
                       f"Grad_norm={grad_info.get('total_norm', 0):.6f}, "
                       f"Weight_change={total_change:.6f}")
    
    # Verify learning occurred
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_improvement = (initial_loss - final_loss) / initial_loss
    
    avg_grad_norm = np.mean(gradient_norms[-20:])  # Recent gradients
    final_weight_change = weight_changes[-1]
    
    logger.info("\nLearning Verification Results:")
    logger.info(f"Initial Loss: {initial_loss:.6f}")
    logger.info(f"Final Loss: {final_loss:.6f}")
    logger.info(f"Loss Improvement: {loss_improvement:.1%}")
    logger.info(f"Average Gradient Norm: {avg_grad_norm:.6f}")
    logger.info(f"Total Weight Change: {final_weight_change:.6f}")
    
    # Learning checks
    checks = {
        'loss_improved': loss_improvement > 0.5,  # 50% improvement
        'gradients_present': avg_grad_norm > 1e-8,
        'gradients_stable': 1e-8 < avg_grad_norm < 100,
        'weights_changed': final_weight_change > 1e-6,
        'convergence': losses[-1] < losses[-20] * 1.1  # Recent convergence
    }
    
    logger.info("\nLearning Checks:")
    all_passed = True
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status}: {check}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 BASIC NEURAL NETWORK IS LEARNING!")
        return True
    else:
        logger.error("❌ BASIC NEURAL NETWORK LEARNING FAILED!")
        return False

def test_agent_integration_learning():
    """Test learning with agent-like structure"""
    logger.info("="*60)
    logger.info("TESTING AGENT INTEGRATION LEARNING")
    logger.info("="*60)
    
    class SimpleAgent:
        """Simple Q-learning-like agent for testing"""
        
        def __init__(self, state_dim: int = 4, action_dim: int = 2, lr: float = 0.001):
            self.q_network = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim)
            )
            
            self.target_network = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim)
            )
            
            # Copy weights to target network
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
            self.criterion = nn.MSELoss()
            self.epsilon = 1.0
            self.replay_buffer = []
            self.step_count = 0
            
        def act(self, state):
            """Select action using epsilon-greedy"""
            if np.random.random() < self.epsilon:
                return np.random.randint(0, 2)  # Random action
            else:
                with torch.no_grad():
                    q_values = self.q_network(torch.FloatTensor(state))
                    return q_values.argmax().item()
                    
        def store_experience(self, state, action, reward, next_state, done):
            """Store experience in replay buffer"""
            self.replay_buffer.append((state, action, reward, next_state, done))
            if len(self.replay_buffer) > 10000:
                self.replay_buffer.pop(0)
        
        def should_update(self):
            """Check if we should perform a training update"""
            return len(self.replay_buffer) >= 32 and self.step_count % 4 == 0
        
        def update(self):
            """Perform Q-learning update"""
            if len(self.replay_buffer) < 32:
                return {'loss': 0}
                
            # Sample batch
            batch = np.random.choice(len(self.replay_buffer), size=32, replace=False)
            batch_data = [self.replay_buffer[i] for i in batch]
            
            states = torch.FloatTensor([e[0] for e in batch_data])
            actions = torch.LongTensor([e[1] for e in batch_data])
            rewards = torch.FloatTensor([e[2] for e in batch_data])
            next_states = torch.FloatTensor([e[3] for e in batch_data])
            dones = torch.BoolTensor([e[4] for e in batch_data])
            
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Target Q values
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + 0.99 * next_q_values * (~dones)
                target_q_values = target_q_values.unsqueeze(1)
            
            # Compute loss
            loss = self.criterion(current_q_values, target_q_values)
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            # Decay epsilon
            self.epsilon *= 0.995
            self.epsilon = max(self.epsilon, 0.01)
            
            # Update target network occasionally
            if self.step_count % 100 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.step_count += 1
            
            return {'loss': loss.item()}
    
    class SimpleEnvironment:
        """Simple environment that rewards moving towards target"""
        
        def __init__(self):
            self.target = np.array([1.0, 1.0, 1.0, 1.0])
            self.reset()
        
        def reset(self):
            self.state = np.random.uniform(-1, 1, 4)
            return self.state.copy()
        
        def step(self, action):
            # Action 0: move towards target, Action 1: move randomly
            if action == 0:
                # Move towards target (good action)
                direction = self.target - self.state
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                self.state += 0.1 * direction
            else:
                # Random movement (bad action)
                self.state += np.random.uniform(-0.1, 0.1, 4)
            
            # Reward based on distance to target
            distance = np.linalg.norm(self.state - self.target)
            reward = -distance  # Closer = higher reward
            
            # Episode ends when very close or very far
            done = distance < 0.1 or distance > 5.0
            
            return self.state.copy(), reward, done, {}
    
    # Create agent and environment
    agent = SimpleAgent()
    env = SimpleEnvironment()
    monitor = GradientFlowMonitor()
    
    # Track learning metrics
    episode_rewards = []
    losses = []
    gradient_norms = []
    epsilon_history = []
    q_value_changes = []
    
    # Record initial Q-network weights
    initial_weights = {}
    for name, param in agent.q_network.named_parameters():
        initial_weights[name] = param.data.clone()
    
    logger.info("Training agent...")
    
    for episode in range(200):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select and perform action
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            
            # Update agent
            if agent.should_update():
                result = agent.update()
                loss = result['loss']
                losses.append(loss)
                
                # Monitor gradients
                if loss > 0:
                    # Create a dummy loss tensor for gradient checking
                    dummy_loss = torch.tensor(loss, requires_grad=True)
                    grad_info = monitor.check_gradient_flow(agent.q_network, dummy_loss, compute_backward=False)
                    gradient_norms.append(grad_info.get('total_norm', 0))
            
            state = next_state
            if done or episode_length > 100:
                break
        
        episode_rewards.append(episode_reward)
        epsilon_history.append(agent.epsilon)
        
        # Calculate Q-value changes
        total_change = 0
        for name, param in agent.q_network.named_parameters():
            if name in initial_weights:
                change = torch.norm(param.data - initial_weights[name]).item()
                total_change += change
        q_value_changes.append(total_change)
        
        if episode % 40 == 0:
            recent_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards[-1]
            recent_loss = np.mean(losses[-10:]) if len(losses) >= 10 else 0
            logger.info(f"Episode {episode}: Reward={recent_reward:.3f}, "
                       f"Loss={recent_loss:.6f}, Epsilon={agent.epsilon:.3f}, "
                       f"Buffer_size={len(agent.replay_buffer)}")
    
    # Analyze learning
    logger.info("\nLearning Analysis:")
    
    # Performance improvement
    early_rewards = np.mean(episode_rewards[:50])
    late_rewards = np.mean(episode_rewards[-50:])
    reward_improvement = late_rewards - early_rewards
    
    # Loss improvement
    if len(losses) > 20:
        early_loss = np.mean(losses[:10])
        late_loss = np.mean(losses[-10:])
        loss_improvement = (early_loss - late_loss) / max(early_loss, 1e-8)
    else:
        loss_improvement = 0
    
    # Exploration decay
    epsilon_decay = epsilon_history[0] - epsilon_history[-1]
    
    # Weight changes
    final_weight_change = q_value_changes[-1]
    
    # Buffer utilization
    buffer_utilization = len(agent.replay_buffer) / 10000
    
    logger.info(f"Early Average Reward: {early_rewards:.3f}")
    logger.info(f"Late Average Reward: {late_rewards:.3f}")
    logger.info(f"Reward Improvement: {reward_improvement:.3f}")
    logger.info(f"Loss Improvement: {loss_improvement:.1%}")
    logger.info(f"Epsilon Decay: {epsilon_decay:.3f}")
    logger.info(f"Total Weight Change: {final_weight_change:.6f}")
    logger.info(f"Buffer Utilization: {buffer_utilization:.1%}")
    
    # Learning checks
    checks = {
        'performance_improved': reward_improvement > 0.5,  # Meaningful improvement
        'exploration_decayed': epsilon_decay > 0.8,  # Epsilon decayed properly
        'weights_updated': final_weight_change > 1e-4,  # Weights changed
        'buffer_used': buffer_utilization > 0.1,  # Buffer was populated
        'training_occurred': len(losses) > 50,  # Training updates happened
    }
    
    if len(losses) > 10:
        checks['loss_stable'] = np.std(losses[-10:]) < np.mean(losses[-10:])  # Loss stabilized
    
    logger.info("\nLearning Verification:")
    all_passed = True
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status}: {check}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 AGENT INTEGRATION LEARNING SUCCESS!")
        return True
    else:
        logger.error("❌ AGENT INTEGRATION LEARNING FAILED!")
        return False

def run_simple_learning_verification():
    """Run simple learning verification tests"""
    logger.info("🚀 STARTING SIMPLE LEARNING VERIFICATION")
    logger.info("="*80)
    
    results = {}
    
    # Test 1: Basic Neural Network Learning
    logger.info("\n📊 Test 1: Basic Neural Network Learning")
    results['neural_network'] = test_basic_neural_network_learning()
    
    # Test 2: Agent Integration Learning  
    logger.info("\n🤖 Test 2: Agent Integration Learning")
    results['agent_integration'] = test_agent_integration_learning()
    
    # Overall results
    logger.info("\n" + "="*80)
    logger.info("🎯 SIMPLE LEARNING VERIFICATION SUMMARY")
    logger.info("="*80)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status}: {test_name.replace('_', ' ').title()} Test")
    
    logger.info(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL SIMPLE LEARNING TESTS PASSED!")
        logger.info("✅ LEARNING VERIFICATION SYSTEM IS WORKING!")
    else:
        logger.error("❌ SOME LEARNING TESTS FAILED!")
        logger.error("⚠️  LEARNING VERIFICATION NEEDS INVESTIGATION!")
    
    return results

if __name__ == "__main__":
    # Run simple verification tests
    test_results = run_simple_learning_verification()
    
    # Exit with appropriate code
    if all(test_results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure