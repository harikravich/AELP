#!/usr/bin/env python3
"""
Comprehensive Learning Verification System for GAELP RL Agents
Verifies that reinforcement learning agents are actually learning by checking:
1. Gradient flow - Non-zero, non-exploding gradients
2. Weight updates - Neural network parameters changing over time
3. Experience replay - Proper sampling and utilization
4. Performance improvement - Rewards increasing, entropy decreasing
5. Loss convergence - Training loss improving over time
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque, defaultdict
from dataclasses import dataclass
import json
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningCheckpoint:
    """Snapshot of learning state for comparison"""
    episode: int
    weights: Dict[str, torch.Tensor]
    loss: float
    reward: float
    entropy: float
    gradient_norm: float
    timestamp: datetime

class LearningMetricsTracker:
    """Comprehensive tracker for all learning indicators"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            # Weight tracking
            'weight_changes': deque(maxlen=window_size),
            'weight_magnitudes': deque(maxlen=window_size),
            
            # Gradient tracking  
            'gradient_norms': deque(maxlen=window_size),
            'gradient_means': deque(maxlen=window_size),
            'gradient_stds': deque(maxlen=window_size),
            
            # Loss tracking
            'losses': deque(maxlen=window_size),
            'value_losses': deque(maxlen=window_size),
            'policy_losses': deque(maxlen=window_size),
            
            # Performance tracking
            'episode_rewards': deque(maxlen=window_size),
            'episode_lengths': deque(maxlen=window_size),
            'success_rates': deque(maxlen=window_size),
            'roas_scores': deque(maxlen=window_size),
            
            # Policy tracking
            'entropy_values': deque(maxlen=window_size),
            'exploration_rates': deque(maxlen=window_size),
            'action_probabilities': deque(maxlen=window_size),
            
            # Experience replay tracking
            'buffer_sizes': deque(maxlen=window_size),
            'sampling_rates': deque(maxlen=window_size),
            'priority_errors': deque(maxlen=window_size)
        }
        
        self.initial_weights = None
        self.checkpoints = []
        self.learning_phases = []
        self.anomalies = []
        
    def record_initial_weights(self, model: nn.Module):
        """Record initial weights for comparison"""
        self.initial_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone().detach()
        logger.info(f"Recorded initial weights for {len(self.initial_weights)} parameters")
        
    def record_gradient_flow(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, float]:
        """Verify and record gradient flow through the network"""
        gradient_info = {}
        total_norm = 0
        param_count = 0
        zero_grad_count = 0
        
        # Calculate gradients if not already done
        if not any(p.grad is not None for p in model.parameters()):
            loss.backward(retain_graph=True)
        
        gradient_norms = []
        gradient_means = []
        gradient_stds = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                gradient_norms.append(grad_norm)
                gradient_means.append(grad_mean)
                gradient_stds.append(grad_std)
                
                total_norm += grad_norm ** 2
                param_count += 1
                
                # Check for problems
                if torch.isnan(param.grad).any():
                    self.anomalies.append(f"NaN gradient in {name}")
                elif torch.isinf(param.grad).any():
                    self.anomalies.append(f"Inf gradient in {name}")
                elif grad_norm == 0:
                    zero_grad_count += 1
                elif grad_norm > 100:
                    self.anomalies.append(f"Exploding gradient in {name}: {grad_norm}")
                    
                gradient_info[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std
                }
            else:
                gradient_info[name] = {'error': 'No gradient'}
                zero_grad_count += 1
        
        total_norm = total_norm ** 0.5
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0
        avg_grad_mean = np.mean(gradient_means) if gradient_means else 0
        avg_grad_std = np.mean(gradient_stds) if gradient_stds else 0
        
        # Record metrics
        self.metrics['gradient_norms'].append(total_norm)
        self.metrics['gradient_means'].append(avg_grad_mean)
        self.metrics['gradient_stds'].append(avg_grad_std)
        
        # Check for gradient flow problems
        flow_health = {
            'total_norm': total_norm,
            'avg_norm': avg_grad_norm,
            'zero_grad_ratio': zero_grad_count / max(param_count, 1),
            'has_flow': total_norm > 1e-8,
            'is_stable': total_norm < 100,
            'param_count': param_count,
            'zero_grad_count': zero_grad_count
        }
        
        return flow_health
        
    def record_weight_update(self, model: nn.Module) -> Dict[str, float]:
        """Record weight changes and magnitudes"""
        if self.initial_weights is None:
            self.record_initial_weights(model)
            
        total_change = 0
        total_magnitude = 0
        param_count = 0
        
        weight_changes = {}
        
        for name, param in model.named_parameters():
            if name in self.initial_weights and param.requires_grad:
                # Calculate change from initial
                initial = self.initial_weights[name]
                current = param.data
                
                change = torch.norm(current - initial).item()
                magnitude = torch.norm(current).item()
                
                total_change += change
                total_magnitude += magnitude
                param_count += 1
                
                weight_changes[name] = {
                    'change_from_initial': change,
                    'current_magnitude': magnitude,
                    'relative_change': change / (magnitude + 1e-8)
                }
        
        avg_change = total_change / max(param_count, 1)
        avg_magnitude = total_magnitude / max(param_count, 1)
        
        # Record metrics
        self.metrics['weight_changes'].append(total_change)
        self.metrics['weight_magnitudes'].append(total_magnitude)
        
        update_health = {
            'total_change': total_change,
            'avg_change': avg_change,
            'avg_magnitude': avg_magnitude,
            'is_updating': total_change > 1e-6,
            'param_count': param_count
        }
        
        return update_health
        
    def record_loss_metrics(self, total_loss: float, value_loss: float = None, 
                          policy_loss: float = None):
        """Record training loss metrics"""
        self.metrics['losses'].append(total_loss)
        
        if value_loss is not None:
            self.metrics['value_losses'].append(value_loss)
        if policy_loss is not None:
            self.metrics['policy_losses'].append(policy_loss)
            
    def record_performance_metrics(self, episode_reward: float, episode_length: int = None,
                                 success_rate: float = None, roas: float = None):
        """Record episode performance metrics"""
        self.metrics['episode_rewards'].append(episode_reward)
        
        if episode_length is not None:
            self.metrics['episode_lengths'].append(episode_length)
        if success_rate is not None:
            self.metrics['success_rates'].append(success_rate)
        if roas is not None:
            self.metrics['roas_scores'].append(roas)
            
    def record_policy_metrics(self, entropy: float, exploration_rate: float = None,
                            action_probs: np.ndarray = None):
        """Record policy-specific metrics"""
        self.metrics['entropy_values'].append(entropy)
        
        if exploration_rate is not None:
            self.metrics['exploration_rates'].append(exploration_rate)
        if action_probs is not None:
            # Record max probability as measure of determinism
            max_prob = np.max(action_probs)
            self.metrics['action_probabilities'].append(max_prob)
            
    def record_replay_metrics(self, buffer_size: int, sampling_rate: float = None,
                            priority_error: float = None):
        """Record experience replay metrics"""
        self.metrics['buffer_sizes'].append(buffer_size)
        
        if sampling_rate is not None:
            self.metrics['sampling_rates'].append(sampling_rate)
        if priority_error is not None:
            self.metrics['priority_errors'].append(priority_error)
            
    def verify_learning(self, min_episodes: int = 50) -> Dict[str, bool]:
        """Comprehensive learning verification"""
        checks = {
            'weights_updating': False,
            'gradients_flowing': False,
            'loss_improving': False,
            'entropy_decreasing': False,
            'performance_improving': False,
            'replay_functioning': False,
            'exploration_decaying': False,
            'convergence_stable': False
        }
        
        # Need minimum data for verification
        if len(self.metrics['losses']) < min_episodes:
            logger.warning(f"Insufficient data for verification: {len(self.metrics['losses'])} < {min_episodes}")
            return checks
            
        # 1. Check weight updates
        if len(self.metrics['weight_changes']) > 10:
            recent_changes = list(self.metrics['weight_changes'])[-10:]
            checks['weights_updating'] = max(recent_changes) > 1e-6
            
        # 2. Check gradient flow
        if len(self.metrics['gradient_norms']) > 10:
            recent_grads = list(self.metrics['gradient_norms'])[-10:]
            checks['gradients_flowing'] = (
                min(recent_grads) > 1e-8 and  # Not zero
                max(recent_grads) < 100      # Not exploding
            )
            
        # 3. Check loss improvement
        if len(self.metrics['losses']) > min_episodes:
            losses = list(self.metrics['losses'])
            early_loss = np.mean(losses[:min_episodes//2])
            late_loss = np.mean(losses[-min_episodes//2:])
            checks['loss_improving'] = late_loss < early_loss * 0.95  # 5% improvement
            
        # 4. Check entropy decrease (policy becoming more deterministic)
        if len(self.metrics['entropy_values']) > min_episodes:
            entropies = list(self.metrics['entropy_values'])
            early_entropy = np.mean(entropies[:min_episodes//2])
            late_entropy = np.mean(entropies[-min_episodes//2:])
            checks['entropy_decreasing'] = late_entropy < early_entropy * 0.8  # 20% reduction
            
        # 5. Check performance improvement
        if len(self.metrics['episode_rewards']) > min_episodes:
            rewards = list(self.metrics['episode_rewards'])
            early_reward = np.mean(rewards[:min_episodes//2])
            late_reward = np.mean(rewards[-min_episodes//2:])
            checks['performance_improving'] = late_reward > early_reward * 1.1  # 10% improvement
            
        # 6. Check replay buffer functioning
        if len(self.metrics['buffer_sizes']) > 10:
            buffer_sizes = list(self.metrics['buffer_sizes'])
            checks['replay_functioning'] = max(buffer_sizes) > 100  # Buffer has experiences
            
        # 7. Check exploration decay
        if len(self.metrics['exploration_rates']) > min_episodes:
            exploration = list(self.metrics['exploration_rates'])
            early_explore = np.mean(exploration[:min_episodes//2])
            late_explore = np.mean(exploration[-min_episodes//2:])
            checks['exploration_decaying'] = late_explore < early_explore * 0.7  # 30% reduction
            
        # 8. Check convergence stability (not too much variance in recent performance)
        if len(self.metrics['episode_rewards']) > 20:
            recent_rewards = list(self.metrics['episode_rewards'])[-20:]
            reward_std = np.std(recent_rewards)
            reward_mean = np.mean(recent_rewards)
            cv = reward_std / (abs(reward_mean) + 1e-8)  # Coefficient of variation
            checks['convergence_stable'] = cv < 0.5  # Less than 50% variance
            
        return checks
        
    def diagnose_learning_problems(self) -> List[str]:
        """Diagnose specific learning problems"""
        problems = []
        
        # Check recent metrics
        if len(self.metrics['gradient_norms']) > 5:
            recent_grads = list(self.metrics['gradient_norms'])[-5:]
            if max(recent_grads) == 0:
                problems.append("CRITICAL: Zero gradients - no backpropagation happening")
            elif min(recent_grads) > 100:
                problems.append("CRITICAL: Exploding gradients - need gradient clipping")
                
        if len(self.metrics['weight_changes']) > 5:
            recent_changes = list(self.metrics['weight_changes'])[-5:]
            if max(recent_changes) < 1e-8:
                problems.append("CRITICAL: Weights not updating - check optimizer.step()")
                
        if len(self.metrics['losses']) > 10:
            recent_losses = list(self.metrics['losses'])[-10:]
            if all(np.isnan(loss) for loss in recent_losses):
                problems.append("CRITICAL: All losses are NaN - numerical instability")
            elif len(set(recent_losses)) == 1:
                problems.append("WARNING: Loss not changing - possible dead network")
                
        if len(self.metrics['entropy_values']) > 10:
            recent_entropy = list(self.metrics['entropy_values'])[-10:]
            if all(e < 0.001 for e in recent_entropy):
                problems.append("WARNING: Very low entropy - agent may be stuck in local optimum")
                
        # Add recorded anomalies
        problems.extend(self.anomalies[-10:])  # Last 10 anomalies
        
        return problems
        
    def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning verification report"""
        checks = self.verify_learning()
        problems = self.diagnose_learning_problems()
        
        # Calculate summary statistics
        summary_stats = {}
        for metric_name, metric_values in self.metrics.items():
            if len(metric_values) > 0:
                values = list(metric_values)
                summary_stats[metric_name] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'recent_mean': float(np.mean(values[-10:])) if len(values) >= 10 else float(np.mean(values))
                }
                
        report = {
            'timestamp': datetime.now().isoformat(),
            'learning_verified': all(checks.values()),
            'verification_checks': checks,
            'problems_found': problems,
            'summary_statistics': summary_stats,
            'total_anomalies': len(self.anomalies),
            'data_points': len(self.metrics['losses'])
        }
        
        return report
        
    def save_report(self, filepath: str):
        """Save learning report to file"""
        report = self.generate_learning_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Saved learning verification report to {filepath}")
        
    def plot_learning_metrics(self, save_path: str = "learning_metrics.png"):
        """Create comprehensive visualization of learning metrics"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('GAELP Agent Learning Verification', fontsize=16, fontweight='bold')
        
        # Plot 1: Weight Changes
        if len(self.metrics['weight_changes']) > 0:
            axes[0, 0].plot(list(self.metrics['weight_changes']))
            axes[0, 0].set_title('Weight Changes Over Time')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Total Weight Change')
            axes[0, 0].grid(True, alpha=0.3)
            
        # Plot 2: Gradient Norms
        if len(self.metrics['gradient_norms']) > 0:
            axes[0, 1].plot(list(self.metrics['gradient_norms']))
            axes[0, 1].set_title('Gradient Norms')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
            
        # Plot 3: Training Loss
        if len(self.metrics['losses']) > 0:
            axes[0, 2].plot(list(self.metrics['losses']))
            axes[0, 2].set_title('Training Loss')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].grid(True, alpha=0.3)
            
        # Plot 4: Episode Rewards
        if len(self.metrics['episode_rewards']) > 0:
            rewards = list(self.metrics['episode_rewards'])
            axes[1, 0].plot(rewards)
            # Add trend line
            if len(rewards) > 10:
                z = np.polyfit(range(len(rewards)), rewards, 1)
                p = np.poly1d(z)
                axes[1, 0].plot(range(len(rewards)), p(range(len(rewards))), "r--", alpha=0.8)
            axes[1, 0].set_title('Episode Rewards (with trend)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Total Reward')
            axes[1, 0].grid(True, alpha=0.3)
            
        # Plot 5: Entropy Decay
        if len(self.metrics['entropy_values']) > 0:
            axes[1, 1].plot(list(self.metrics['entropy_values']))
            axes[1, 1].set_title('Policy Entropy (Should Decrease)')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].grid(True, alpha=0.3)
            
        # Plot 6: ROAS Performance
        if len(self.metrics['roas_scores']) > 0:
            roas = list(self.metrics['roas_scores'])
            axes[1, 2].plot(roas)
            if len(roas) > 10:
                z = np.polyfit(range(len(roas)), roas, 1)
                p = np.poly1d(z)
                axes[1, 2].plot(range(len(roas)), p(range(len(roas))), "r--", alpha=0.8)
            axes[1, 2].set_title('ROAS Over Time (with trend)')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('ROAS')
            axes[1, 2].grid(True, alpha=0.3)
            
        # Plot 7: Buffer Size
        if len(self.metrics['buffer_sizes']) > 0:
            axes[2, 0].plot(list(self.metrics['buffer_sizes']))
            axes[2, 0].set_title('Experience Buffer Size')
            axes[2, 0].set_xlabel('Training Step')
            axes[2, 0].set_ylabel('Buffer Size')
            axes[2, 0].grid(True, alpha=0.3)
            
        # Plot 8: Exploration Rate
        if len(self.metrics['exploration_rates']) > 0:
            axes[2, 1].plot(list(self.metrics['exploration_rates']))
            axes[2, 1].set_title('Exploration Rate (Should Decay)')
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Exploration Rate')
            axes[2, 1].grid(True, alpha=0.3)
            
        # Plot 9: Learning Summary
        checks = self.verify_learning()
        check_names = list(checks.keys())
        check_values = [1 if v else 0 for v in checks.values()]
        
        colors = ['green' if v else 'red' for v in check_values]
        axes[2, 2].barh(check_names, check_values, color=colors, alpha=0.7)
        axes[2, 2].set_title('Learning Verification Checks')
        axes[2, 2].set_xlabel('Passing (1) / Failing (0)')
        axes[2, 2].set_xlim(0, 1.2)
        
        # Rotate labels for readability
        for tick in axes[2, 2].get_yticklabels():
            tick.set_rotation(0)
            tick.set_fontsize(8)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved learning metrics plot to {save_path}")

class LearningVerifier:
    """Main class for verifying RL agent learning"""
    
    def __init__(self):
        self.tracker = LearningMetricsTracker()
        
    def instrument_training_loop(self, agent, env, num_episodes: int = 100,
                                min_buffer_size: int = 1000):
        """Instrument a training loop with comprehensive learning verification"""
        logger.info("Starting instrumented training with learning verification...")
        
        # Record initial state
        if hasattr(agent, 'q_network'):
            self.tracker.record_initial_weights(agent.q_network)
        elif hasattr(agent, 'actor_critic'):
            self.tracker.record_initial_weights(agent.actor_critic)
        elif hasattr(agent, 'policy_net'):
            self.tracker.record_initial_weights(agent.policy_net)
        else:
            logger.warning("Could not find model to track weights")
            
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Get action from agent
                if hasattr(agent, 'act'):
                    action = agent.act(obs)
                elif hasattr(agent, 'select_action'):
                    action = agent.select_action(obs)
                else:
                    # Fallback random action
                    action = env.action_space.sample()
                    
                # Step environment
                if hasattr(env, 'step'):
                    next_obs, reward, done, info = env.step(action)
                else:
                    # Mock step for testing
                    next_obs, reward, done, info = obs, 0.1, episode_length > 50, {}
                
                episode_reward += reward
                episode_length += 1
                
                # Store experience if agent has replay buffer
                if hasattr(agent, 'store_experience'):
                    agent.store_experience(obs, action, reward, next_obs, done)
                elif hasattr(agent, 'remember'):
                    agent.remember(obs, action, reward, next_obs, done)
                    
                # Update agent if conditions met
                update_occurred = False
                if hasattr(agent, 'should_update') and agent.should_update():
                    update_occurred = self._perform_update_with_verification(agent, episode_length)
                elif hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) > min_buffer_size:
                    if episode_length % 4 == 0:  # Update every 4 steps
                        update_occurred = self._perform_update_with_verification(agent, episode_length)
                        
                # Record replay metrics
                if hasattr(agent, 'replay_buffer'):
                    buffer_size = len(agent.replay_buffer)
                    self.tracker.record_replay_metrics(buffer_size)
                    
                obs = next_obs
                if done:
                    break
                    
            # Record episode metrics
            episode_rewards.append(episode_reward)
            self.tracker.record_performance_metrics(episode_reward, episode_length)
            
            # Calculate ROAS if possible
            if hasattr(info, 'get') and 'roas' in info:
                self.tracker.record_performance_metrics(episode_reward, episode_length, roas=info['roas'])
                
            # Periodic verification and reporting
            if episode % 10 == 0 and episode > 0:
                self._periodic_verification_report(episode, episode_rewards[-10:])
                
        # Final comprehensive report
        self._generate_final_report()
        
        return self.tracker
        
    def _perform_update_with_verification(self, agent, step: int) -> bool:
        """Perform agent update with comprehensive verification"""
        try:
            # Get model reference
            model = None
            if hasattr(agent, 'q_network'):
                model = agent.q_network
            elif hasattr(agent, 'actor_critic'):
                model = agent.actor_critic  
            elif hasattr(agent, 'policy_net'):
                model = agent.policy_net
                
            if model is None:
                logger.warning("Could not find model for gradient verification")
                return False
                
            # Record weights before update
            weight_health_before = self.tracker.record_weight_update(model)
            
            # Perform the actual update
            loss = None
            if hasattr(agent, 'update'):
                result = agent.update()
                if isinstance(result, dict):
                    loss = result.get('loss', result.get('total_loss'))
                elif isinstance(result, (float, int)):
                    loss = result
            elif hasattr(agent, 'train_step'):
                loss = agent.train_step()
            elif hasattr(agent, 'learn'):
                loss = agent.learn()
                
            if loss is not None:
                # Verify gradient flow
                gradient_health = self.tracker.record_gradient_flow(model, torch.tensor(loss) if not isinstance(loss, torch.Tensor) else loss)
                
                # Record loss metrics
                self.tracker.record_loss_metrics(float(loss))
                
                # Record weight changes after update
                weight_health_after = self.tracker.record_weight_update(model)
                
                # Calculate entropy if possible
                if hasattr(agent, 'get_policy_entropy'):
                    entropy = agent.get_policy_entropy()
                    self.tracker.record_policy_metrics(entropy)
                elif hasattr(agent, 'entropy'):
                    entropy = float(agent.entropy)
                    self.tracker.record_policy_metrics(entropy)
                    
                # Record exploration rate if available
                if hasattr(agent, 'epsilon'):
                    self.tracker.record_policy_metrics(0.5, exploration_rate=agent.epsilon)  # Dummy entropy
                    
                return True
            else:
                logger.warning("No loss returned from agent update")
                return False
                
        except Exception as e:
            logger.error(f"Error during update verification: {e}")
            return False
            
    def _periodic_verification_report(self, episode: int, recent_rewards: List[float]):
        """Generate periodic verification report"""
        checks = self.tracker.verify_learning(min_episodes=max(10, episode//10))
        problems = self.tracker.diagnose_learning_problems()
        
        avg_reward = np.mean(recent_rewards)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"LEARNING VERIFICATION REPORT - Episode {episode}")
        logger.info(f"{'='*60}")
        logger.info(f"Average Recent Reward: {avg_reward:.3f}")
        
        # Check results
        passing_checks = sum(checks.values())
        total_checks = len(checks)
        
        logger.info(f"Learning Checks: {passing_checks}/{total_checks} passing")
        for check, result in checks.items():
            status = "✅" if result else "❌"
            logger.info(f"  {status} {check}")
            
        if problems:
            logger.warning("Problems detected:")
            for problem in problems:
                logger.warning(f"  ⚠️  {problem}")
        else:
            logger.info("  No problems detected!")
            
        if all(checks.values()):
            logger.info("🎉 AGENT IS LEARNING SUCCESSFULLY!")
        else:
            logger.warning("⚠️  LEARNING ISSUES DETECTED")
            
    def _generate_final_report(self):
        """Generate final comprehensive learning report"""
        logger.info(f"\n{'='*80}")
        logger.info("FINAL LEARNING VERIFICATION REPORT")
        logger.info(f"{'='*80}")
        
        report = self.tracker.generate_learning_report()
        
        logger.info(f"Learning Verified: {'YES' if report['learning_verified'] else 'NO'}")
        logger.info(f"Total Data Points: {report['data_points']}")
        logger.info(f"Total Anomalies: {report['total_anomalies']}")
        
        # Save report and plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"learning_verification_report_{timestamp}.json"
        plot_path = f"learning_metrics_{timestamp}.png"
        
        self.tracker.save_report(report_path)
        self.tracker.plot_learning_metrics(plot_path)
        
        logger.info(f"Detailed report saved to: {report_path}")
        logger.info(f"Learning plots saved to: {plot_path}")
        
        return report

def verify_agent_learning(agent_or_training_function, env=None, num_episodes: int = 100):
    """Convenience function to verify agent learning"""
    verifier = LearningVerifier()
    
    if callable(agent_or_training_function):
        # If it's a training function, call it with verification
        logger.info("Instrumenting custom training function...")
        return agent_or_training_function(verifier.tracker)
    else:
        # If it's an agent, run standard verification
        if env is None:
            logger.error("Environment required when passing agent directly")
            return None
            
        return verifier.instrument_training_loop(agent_or_training_function, env, num_episodes)

if __name__ == "__main__":
    # Example usage
    logger.info("Learning Verification System - Running example...")
    
    # This would normally be your actual training code
    class MockAgent:
        def __init__(self):
            self.q_network = nn.Linear(4, 2)
            self.optimizer = torch.optim.Adam(self.q_network.parameters())
            self.replay_buffer = deque(maxlen=10000)
            self.epsilon = 1.0
            
        def act(self, state):
            return 0 if np.random.random() < self.epsilon else 1
            
        def should_update(self):
            return len(self.replay_buffer) > 32
            
        def update(self):
            loss = torch.tensor(np.random.exponential(1.0))  # Simulated decreasing loss
            self.epsilon *= 0.995  # Decay exploration
            return {'loss': loss.item()}
            
        def store_experience(self, s, a, r, ns, d):
            self.replay_buffer.append((s, a, r, ns, d))
            
    class MockEnv:
        def __init__(self):
            self.step_count = 0
            
        def reset(self):
            self.step_count = 0
            return np.random.random(4)
            
        def step(self, action):
            self.step_count += 1
            reward = np.random.random() + self.step_count * 0.01  # Improving reward
            done = self.step_count > 50
            return np.random.random(4), reward, done, {}
    
    # Run verification
    agent = MockAgent()
    env = MockEnv()
    
    verifier = LearningVerifier()
    tracker = verifier.instrument_training_loop(agent, env, num_episodes=50)
    
    logger.info("Example verification complete!")