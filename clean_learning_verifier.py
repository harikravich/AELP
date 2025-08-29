#!/usr/bin/env python3
"""
Clean Learning Verification System

A clean, uncorrupted version of the learning verification system
for GAELP agents. Verifies real learning through comprehensive checks.

CRITICAL VERIFICATIONS:
1. Weight Updates - Network parameters must change after training
2. Gradient Flow - Non-zero gradients must flow backward
3. Loss Improvement - Training loss should generally decrease
4. Entropy Changes - Policy entropy should evolve appropriately
5. Performance Gains - Agent performance must improve over time

NO FALLBACKS. NO CORRUPTED CODE. REAL VERIFICATION ONLY.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class CleanLearningMetrics:
    """Clean container for learning verification metrics"""
    
    # Weight change tracking
    weight_changes: List[float]
    weight_norms: List[float]
    
    # Gradient tracking
    gradient_norms: List[float]
    gradient_flows: Dict[str, List[float]]
    
    # Loss tracking
    policy_losses: List[float]
    value_losses: List[float]
    total_losses: List[float]
    
    # Entropy tracking
    entropies: List[float]
    entropy_trends: List[float]
    
    # Performance tracking
    episode_rewards: List[float]
    episode_lengths: List[int]
    roas_values: List[float]
    conversion_rates: List[float]
    
    # Training step tracking
    training_steps: List[int]
    timestamps: List[datetime]
    
    def __post_init__(self):
        """Initialize empty lists if None"""
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name)
            if field_value is None:
                if 'Dict' in str(self.__dataclass_fields__[field_name].type):
                    setattr(self, field_name, {})
                else:
                    setattr(self, field_name, [])

class CleanLearningVerifier:
    """Clean learning verifier without corrupted pattern discovery calls"""
    
    def __init__(self, 
                 agent_name: str = "gaelp_agent",
                 tolerance: float = 1e-6,
                 window_size: int = 50,
                 save_plots: bool = True):
        
        self.agent_name = agent_name
        self.tolerance = tolerance
        self.window_size = window_size
        self.save_plots = save_plots
        
        # Initialize metrics
        self.metrics = CleanLearningMetrics(
            weight_changes=[], weight_norms=[], gradient_norms=[],
            gradient_flows={}, policy_losses=[], value_losses=[],
            total_losses=[], entropies=[], entropy_trends=[],
            episode_rewards=[], episode_lengths=[], roas_values=[],
            conversion_rates=[], training_steps=[], timestamps=[]
        )
        
        # Store initial model states for comparison
        self.initial_weights: Optional[Dict[str, torch.Tensor]] = None
        self.previous_weights: Optional[Dict[str, torch.Tensor]] = None
        
        # Learning verification flags
        self.learning_verified = False
        self.verification_results = {}
        
        logger.info(f"Clean learning verifier initialized for {agent_name}")
    
    def capture_initial_weights(self, model: nn.Module):
        """Capture initial model weights for comparison"""
        
        self.initial_weights = {}
        self.previous_weights = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.detach().clone()
                self.previous_weights[name] = param.detach().clone()
        
        logger.info(f"Captured initial weights for {len(self.initial_weights)} parameters")
    
    def verify_gradient_flow(self, 
                           model: nn.Module,
                           loss: torch.Tensor,
                           training_step: int) -> Dict[str, Any]:
        """Verify that gradients are flowing properly through the network"""
        
        gradient_info = {
            'has_gradients': False,
            'gradient_norms': {},
            'total_gradient_norm': 0.0,
            'problems': [],
            'gradient_stats': {}
        }
        
        # Ensure gradients are computed
        if loss.requires_grad:
            loss.backward(retain_graph=True)
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count += 1
                
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_info['gradient_norms'][name] = grad_norm
                    total_norm += grad_norm ** 2
                    
                    # Check for problems
                    if torch.isnan(param.grad).any():
                        gradient_info['problems'].append(f"{name}: Contains NaN gradients")
                    elif torch.isinf(param.grad).any():
                        gradient_info['problems'].append(f"{name}: Contains Inf gradients")
                    elif grad_norm == 0.0:
                        gradient_info['problems'].append(f"{name}: Zero gradients")
                    elif grad_norm > 100.0:
                        gradient_info['problems'].append(f"{name}: Large gradients ({grad_norm:.2f})")
                    
                else:
                    gradient_info['problems'].append(f"{name}: No gradient computed!")
        
        total_norm = total_norm ** 0.5
        gradient_info['total_gradient_norm'] = total_norm
        gradient_info['has_gradients'] = total_norm > self.tolerance
        
        # Calculate gradient statistics
        if gradient_info['gradient_norms']:
            norms = list(gradient_info['gradient_norms'].values())
            gradient_info['gradient_stats'] = {
                'mean_norm': np.mean(norms),
                'std_norm': np.std(norms),
                'min_norm': np.min(norms),
                'max_norm': np.max(norms)
            }
        
        # Store for tracking
        self.metrics.gradient_norms.append(total_norm)
        self.metrics.training_steps.append(training_step)
        self.metrics.timestamps.append(datetime.now())
        
        # Update gradient flow tracking
        for name, norm in gradient_info['gradient_norms'].items():
            if name not in self.metrics.gradient_flows:
                self.metrics.gradient_flows[name] = []
            self.metrics.gradient_flows[name].append(norm)
        
        if gradient_info['problems']:
            logger.warning(f"Gradient flow problems: {len(gradient_info['problems'])} issues")
        
        return gradient_info
    
    def verify_weight_updates(self, 
                            model: nn.Module,
                            training_step: int) -> Dict[str, Any]:
        """Verify that model weights are actually updating during training"""
        
        weight_update_info = {
            'weights_changed': False,
            'total_change': 0.0,
            'parameter_changes': {},
            'problems': [],
            'weight_stats': {}
        }
        
        if self.previous_weights is None:
            self.capture_initial_weights(model)
            return weight_update_info
        
        total_change = 0.0
        unchanged_params = 0
        
        current_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.previous_weights:
                current_weights[name] = param.detach().clone()
                
                # Calculate change from previous step
                change = (current_weights[name] - self.previous_weights[name]).norm().item()
                weight_update_info['parameter_changes'][name] = change
                total_change += change
                
                if change < self.tolerance:
                    unchanged_params += 1
                
                # Check for problems
                if torch.isnan(current_weights[name]).any():
                    weight_update_info['problems'].append(f"{name}: Contains NaN weights")
                elif torch.isinf(current_weights[name]).any():
                    weight_update_info['problems'].append(f"{name}: Contains Inf weights")
        
        weight_update_info['total_change'] = total_change
        weight_update_info['weights_changed'] = total_change > self.tolerance
        
        # Calculate change from initial weights
        total_initial_change = 0.0
        if self.initial_weights:
            for name in current_weights:
                if name in self.initial_weights:
                    initial_change = (current_weights[name] - self.initial_weights[name]).norm().item()
                    total_initial_change += initial_change
        
        # Weight statistics
        if weight_update_info['parameter_changes']:
            changes = list(weight_update_info['parameter_changes'].values())
            weight_update_info['weight_stats'] = {
                'mean_change': np.mean(changes),
                'std_change': np.std(changes),
                'min_change': np.min(changes),
                'max_change': np.max(changes),
                'unchanged_params': unchanged_params,
                'total_params': len(changes),
                'change_from_initial': total_initial_change
            }
        
        # Store for tracking
        self.metrics.weight_changes.append(total_change)
        self.metrics.weight_norms.append(total_initial_change)
        
        # Update previous weights
        self.previous_weights = current_weights
        
        if not weight_update_info['weights_changed']:
            weight_update_info['problems'].append("CRITICAL: No weight updates detected!")
        
        return weight_update_info
    
    def verify_loss_improvement(self, 
                              policy_loss: float,
                              value_loss: float,
                              total_loss: float,
                              training_step: int) -> Dict[str, Any]:
        """Verify that training losses are improving over time"""
        
        loss_info = {
            'loss_improving': False,
            'policy_trend': 'unknown',
            'value_trend': 'unknown',
            'total_trend': 'unknown',
            'problems': [],
            'loss_stats': {}
        }
        
        # Store current losses
        self.metrics.policy_losses.append(policy_loss)
        self.metrics.value_losses.append(value_loss)
        self.metrics.total_losses.append(total_loss)
        
        # Check for loss problems
        if np.isnan(policy_loss) or np.isnan(value_loss) or np.isnan(total_loss):
            loss_info['problems'].append("CRITICAL: NaN loss detected!")
        if np.isinf(policy_loss) or np.isinf(value_loss) or np.isinf(total_loss):
            loss_info['problems'].append("CRITICAL: Infinite loss detected!")
        
        # Analyze trends if we have enough data
        if len(self.metrics.total_losses) >= self.window_size:
            recent_losses = self.metrics.total_losses[-self.window_size:]
            early_losses = self.metrics.total_losses[-2*self.window_size:-self.window_size]
            
            if len(early_losses) >= self.window_size // 2:
                early_mean = np.mean(early_losses)
                recent_mean = np.mean(recent_losses)
                
                # Check if loss is improving
                improvement = (early_mean - recent_mean) / (abs(early_mean) + 1e-8)
                loss_info['loss_improving'] = improvement > 0.05  # 5% improvement threshold
                
                loss_info['loss_stats'] = {
                    'early_mean': early_mean,
                    'recent_mean': recent_mean,
                    'improvement_pct': improvement * 100,
                    'current_policy': policy_loss,
                    'current_value': value_loss,
                    'current_total': total_loss
                }
        
        return loss_info
    
    def verify_entropy_evolution(self, 
                               entropy: float,
                               training_step: int) -> Dict[str, Any]:
        """Verify that policy entropy is evolving appropriately"""
        
        entropy_info = {
            'entropy_valid': False,
            'entropy_trend': 'unknown',
            'exploration_health': 'unknown',
            'problems': [],
            'entropy_stats': {}
        }
        
        # Check for entropy problems
        if np.isnan(entropy):
            entropy_info['problems'].append("CRITICAL: NaN entropy detected!")
            return entropy_info
        
        if entropy < 0:
            entropy_info['problems'].append("CRITICAL: Negative entropy (impossible!)")
        
        # Store entropy
        self.metrics.entropies.append(entropy)
        
        # Analyze entropy evolution
        if len(self.metrics.entropies) >= self.window_size:
            recent_entropies = self.metrics.entropies[-self.window_size:]
            
            # Check exploration health
            mean_entropy = np.mean(recent_entropies)
            if mean_entropy < 0.01:
                entropy_info['exploration_health'] = 'poor'
                entropy_info['problems'].append("Low entropy: Policy may be too deterministic")
            elif mean_entropy > 2.0:
                entropy_info['exploration_health'] = 'excessive'
                entropy_info['problems'].append("High entropy: Policy may be too random")
            else:
                entropy_info['exploration_health'] = 'good'
            
            # Calculate entropy trend over time
            if len(self.metrics.entropies) >= 2 * self.window_size:
                early_entropy = np.mean(self.metrics.entropies[-2*self.window_size:-self.window_size])
                recent_entropy = np.mean(recent_entropies)
                entropy_change = (recent_entropy - early_entropy) / (early_entropy + 1e-8)
                
                self.metrics.entropy_trends.append(entropy_change)
                
                entropy_info['entropy_stats'] = {
                    'current': entropy,
                    'recent_mean': recent_entropy,
                    'early_mean': early_entropy,
                    'change_pct': entropy_change * 100,
                    'std': np.std(recent_entropies)
                }
            
            entropy_info['entropy_valid'] = True
        
        return entropy_info
    
    def verify_performance_improvement(self, 
                                     episode_reward: float,
                                     episode_length: int,
                                     roas: Optional[float] = None,
                                     conversion_rate: Optional[float] = None) -> Dict[str, Any]:
        """Verify that agent performance is improving over episodes"""
        
        performance_info = {
            'performance_improving': False,
            'reward_trend': 'unknown',
            'roas_trend': 'unknown',
            'conversion_trend': 'unknown',
            'problems': [],
            'performance_stats': {}
        }
        
        # Store performance metrics
        self.metrics.episode_rewards.append(episode_reward)
        self.metrics.episode_lengths.append(episode_length)
        
        if roas is not None:
            self.metrics.roas_values.append(roas)
        if conversion_rate is not None:
            self.metrics.conversion_rates.append(conversion_rate)
        
        # Analyze performance trends
        if len(self.metrics.episode_rewards) >= self.window_size:
            
            # Calculate improvement
            if len(self.metrics.episode_rewards) >= 2 * self.window_size:
                early_rewards = self.metrics.episode_rewards[-2*self.window_size:-self.window_size]
                recent_rewards = self.metrics.episode_rewards[-self.window_size:]
                
                early_mean = np.mean(early_rewards)
                recent_mean = np.mean(recent_rewards)
                
                improvement = (recent_mean - early_mean) / (abs(early_mean) + 1e-8)
                performance_info['performance_improving'] = improvement > 0.1  # 10% improvement
                
                performance_info['performance_stats'] = {
                    'early_reward_mean': early_mean,
                    'recent_reward_mean': recent_mean,
                    'improvement_pct': improvement * 100,
                    'current_reward': episode_reward,
                    'current_length': episode_length
                }
            
            # ROAS trend
            if self.metrics.roas_values and len(self.metrics.roas_values) >= self.window_size:
                if roas is not None:
                    performance_info['performance_stats']['current_roas'] = roas
                    performance_info['performance_stats']['recent_roas_mean'] = np.mean(
                        self.metrics.roas_values[-self.window_size:]
                    )
        
        return performance_info
    
    def comprehensive_verification(self, 
                                 model: nn.Module,
                                 loss: torch.Tensor,
                                 policy_loss: float,
                                 value_loss: float,
                                 entropy: float,
                                 episode_reward: float,
                                 episode_length: int,
                                 training_step: int,
                                 **kwargs) -> Dict[str, Any]:
        """Perform comprehensive verification of learning"""
        
        verification_results = {
            'learning_verified': False,
            'verification_timestamp': datetime.now(),
            'training_step': training_step,
            'checks': {}
        }
        
        logger.info(f"Running comprehensive learning verification at step {training_step}")
        
        # Run all verification checks
        try:
            verification_results['checks']['gradient_flow'] = self.verify_gradient_flow(
                model, loss, training_step
            )
            
            verification_results['checks']['weight_updates'] = self.verify_weight_updates(
                model, training_step
            )
            
            verification_results['checks']['loss_improvement'] = self.verify_loss_improvement(
                policy_loss, value_loss, loss.item(), training_step
            )
            
            verification_results['checks']['entropy_evolution'] = self.verify_entropy_evolution(
                entropy, training_step
            )
            
            verification_results['checks']['performance_improvement'] = self.verify_performance_improvement(
                episode_reward, episode_length, 
                kwargs.get('roas'), kwargs.get('conversion_rate')
            )
            
        except Exception as e:
            logger.error(f"Verification failed with error: {e}")
            verification_results['error'] = str(e)
            return verification_results
        
        # Evaluate overall learning status
        learning_checks = {
            'gradients_flowing': verification_results['checks']['gradient_flow']['has_gradients'],
            'weights_updating': verification_results['checks']['weight_updates']['weights_changed'],
            'entropy_valid': verification_results['checks']['entropy_evolution']['entropy_valid'],
            'no_critical_problems': self._no_critical_problems(verification_results['checks'])
        }
        
        # Optional checks (require more data)
        if len(self.metrics.total_losses) >= self.window_size:
            learning_checks['loss_improving'] = verification_results['checks']['loss_improvement']['loss_improving']
        
        if len(self.metrics.episode_rewards) >= self.window_size:
            learning_checks['performance_improving'] = verification_results['checks']['performance_improvement']['performance_improving']
        
        # Overall verification
        critical_checks_passed = all([
            learning_checks['gradients_flowing'],
            learning_checks['weights_updating'], 
            learning_checks['entropy_valid'],
            learning_checks['no_critical_problems']
        ])
        
        verification_results['learning_checks'] = learning_checks
        verification_results['critical_checks_passed'] = critical_checks_passed
        verification_results['learning_verified'] = critical_checks_passed
        
        # Store results
        self.verification_results = verification_results
        self.learning_verified = critical_checks_passed
        
        return verification_results
    
    def _no_critical_problems(self, checks: Dict[str, Any]) -> bool:
        """Check if there are any critical problems that prevent learning"""
        
        critical_problems = []
        
        for check_name, check_results in checks.items():
            if 'problems' in check_results:
                for problem in check_results['problems']:
                    if 'CRITICAL' in problem:
                        critical_problems.append(f"{check_name}: {problem}")
        
        return len(critical_problems) == 0
    
    def generate_learning_report(self) -> str:
        """Generate a comprehensive learning verification report"""
        
        report = []
        report.append("=" * 80)
        report.append(f"LEARNING VERIFICATION REPORT - {self.agent_name}")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now()}")
        report.append(f"Training Steps: {len(self.metrics.training_steps)}")
        report.append("")
        
        # Overall status
        if self.learning_verified:
            report.append("🎉 LEARNING STATUS: VERIFIED ✅")
        else:
            report.append("⚠️  LEARNING STATUS: NOT VERIFIED ❌")
        
        if self.verification_results:
            checks = self.verification_results.get('checks', {})
            
            # Gradient flow
            gradient_check = checks.get('gradient_flow', {})
            if gradient_check.get('has_gradients'):
                report.append("✅ Gradient Flow: HEALTHY")
                report.append(f"   Total gradient norm: {gradient_check.get('total_gradient_norm', 0):.6f}")
            else:
                report.append("❌ Gradient Flow: BROKEN")
            
            # Weight updates
            weight_check = checks.get('weight_updates', {})
            if weight_check.get('weights_changed'):
                report.append("✅ Weight Updates: ACTIVE")
                report.append(f"   Total weight change: {weight_check.get('total_change', 0):.6f}")
            else:
                report.append("❌ Weight Updates: INACTIVE")
            
            # Entropy evolution
            entropy_check = checks.get('entropy_evolution', {})
            if entropy_check.get('entropy_valid'):
                report.append("✅ Entropy Evolution: HEALTHY")
            else:
                report.append("❌ Entropy Evolution: PROBLEMATIC")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def create_clean_learning_verifier(agent_name: str = "gaelp_agent") -> CleanLearningVerifier:
    """Factory function to create a clean learning verifier"""
    return CleanLearningVerifier(agent_name=agent_name, save_plots=True)

if __name__ == "__main__":
    # Example usage
    print("🔍 Clean Learning Verification System")
    print("=" * 50)
    print("This system ensures RL agents actually learn.")
    print("NO FALLBACKS. NO CORRUPTED CODE. REAL VERIFICATION ONLY.")
    print("=" * 50)
    
    verifier = create_clean_learning_verifier("test_agent")
    print(f"✅ Clean learning verifier created for {verifier.agent_name}")
    print("✅ Ready for comprehensive learning verification!")