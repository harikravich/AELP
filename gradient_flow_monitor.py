#!/usr/bin/env python3
"""
Gradient Flow Monitor for GAELP RL Agents
Lightweight tool to verify gradients are flowing during training
Can be easily integrated into existing training loops
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, deque
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradientFlowMonitor:
    """Lightweight monitor for gradient flow in neural networks"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.gradient_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.step_count = 0
        self.problems_detected = []
        
    def check_gradient_flow(self, model: nn.Module, loss: torch.Tensor = None, 
                           compute_backward: bool = True) -> Dict[str, Any]:
        """
        Check if gradients are flowing through the network
        
        Args:
            model: PyTorch model to check
            loss: Loss tensor (will compute backward if needed)
            compute_backward: Whether to compute gradients
            
        Returns:
            Dictionary with gradient flow information
        """
        self.step_count += 1
        
        # Compute gradients if needed
        if compute_backward and loss is not None:
            if not loss.requires_grad:
                self.problems_detected.append("Loss doesn't require gradients!")
                return {'error': 'Loss does not require gradients'}
                
            # Clear existing gradients
            model.zero_grad()
            
            try:
                loss.backward(retain_graph=True)
            except RuntimeError as e:
                self.problems_detected.append(f"Backward pass failed: {e}")
                return {'error': f'Backward pass failed: {e}'}
        
        # Check gradients
        gradient_info = self._analyze_gradients(model)
        
        # Record loss if provided
        if loss is not None:
            self.loss_history.append(loss.item())
        
        # Record gradient norm
        if 'total_norm' in gradient_info:
            self.gradient_history.append(gradient_info['total_norm'])
        
        # Check for problems
        self._detect_problems(gradient_info)
        
        return gradient_info
    
    def _analyze_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze gradients in the model"""
        
        total_norm = 0.0
        param_count = 0
        grad_count = 0
        zero_grad_count = 0
        nan_grad_count = 0
        inf_grad_count = 0
        
        layer_info = {}
        grad_norms = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            param_count += 1
            
            if param.grad is None:
                zero_grad_count += 1
                layer_info[name] = {'status': 'no_gradient'}
                continue
            
            grad_count += 1
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            total_norm += grad_norm ** 2
            
            # Check for NaN/Inf
            if torch.isnan(param.grad).any():
                nan_grad_count += 1
                layer_info[name] = {'status': 'nan_gradient', 'norm': grad_norm}
            elif torch.isinf(param.grad).any():
                inf_grad_count += 1  
                layer_info[name] = {'status': 'inf_gradient', 'norm': grad_norm}
            elif grad_norm == 0:
                layer_info[name] = {'status': 'zero_gradient', 'norm': 0}
            else:
                layer_info[name] = {
                    'status': 'normal', 
                    'norm': grad_norm,
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item()
                }
        
        total_norm = total_norm ** 0.5
        
        return {
            'total_norm': total_norm,
            'param_count': param_count,
            'grad_count': grad_count,
            'zero_grad_count': zero_grad_count,
            'nan_grad_count': nan_grad_count,
            'inf_grad_count': inf_grad_count,
            'avg_grad_norm': np.mean(grad_norms) if grad_norms else 0,
            'max_grad_norm': max(grad_norms) if grad_norms else 0,
            'min_grad_norm': min(grad_norms) if grad_norms else 0,
            'grad_flow_ratio': grad_count / max(param_count, 1),
            'layer_info': layer_info,
            'step': self.step_count
        }
    
    def _detect_problems(self, gradient_info: Dict[str, Any]):
        """Detect gradient flow problems"""
        
        # Problem 1: No gradients at all
        if gradient_info['grad_count'] == 0:
            self.problems_detected.append(f"Step {self.step_count}: No gradients computed!")
        
        # Problem 2: Very low gradient flow
        elif gradient_info['grad_flow_ratio'] < 0.5:
            self.problems_detected.append(
                f"Step {self.step_count}: Low gradient flow "
                f"({gradient_info['grad_flow_ratio']:.2%})"
            )
        
        # Problem 3: Zero total norm
        if gradient_info['total_norm'] == 0:
            self.problems_detected.append(f"Step {self.step_count}: Zero gradient norm!")
        
        # Problem 4: Very small gradients (vanishing)
        elif gradient_info['total_norm'] < 1e-8:
            self.problems_detected.append(
                f"Step {self.step_count}: Vanishing gradients "
                f"(norm={gradient_info['total_norm']:.2e})"
            )
        
        # Problem 5: Very large gradients (exploding)
        elif gradient_info['total_norm'] > 1000:
            self.problems_detected.append(
                f"Step {self.step_count}: Exploding gradients "
                f"(norm={gradient_info['total_norm']:.2e})"
            )
        
        # Problem 6: NaN gradients
        if gradient_info['nan_grad_count'] > 0:
            self.problems_detected.append(
                f"Step {self.step_count}: {gradient_info['nan_grad_count']} NaN gradients!"
            )
        
        # Problem 7: Inf gradients  
        if gradient_info['inf_grad_count'] > 0:
            self.problems_detected.append(
                f"Step {self.step_count}: {gradient_info['inf_grad_count']} Inf gradients!"
            )
    
    def verify_learning_progress(self, min_steps: int = 20) -> Dict[str, bool]:
        """Verify that learning is progressing based on gradient history"""
        
        if len(self.gradient_history) < min_steps:
            return {'insufficient_data': True}
        
        gradient_norms = list(self.gradient_history)
        losses = list(self.loss_history) if self.loss_history else []
        
        checks = {
            'gradients_present': len(gradient_norms) > 0 and max(gradient_norms) > 0,
            'gradients_stable': self._check_gradient_stability(gradient_norms),
            'loss_decreasing': self._check_loss_improvement(losses) if losses else False,
            'no_critical_problems': len(self.problems_detected) == 0
        }
        
        return checks
    
    def _check_gradient_stability(self, gradient_norms: List[float]) -> bool:
        """Check if gradients are stable (not exploding or vanishing)"""
        if not gradient_norms:
            return False
            
        recent_norms = gradient_norms[-10:] if len(gradient_norms) >= 10 else gradient_norms
        
        # Check bounds
        min_norm = min(recent_norms)
        max_norm = max(recent_norms)
        
        # Stable if: not too small, not too large, not too variable
        stable = (
            min_norm > 1e-8 and  # Not vanishing
            max_norm < 100 and   # Not exploding
            (max_norm / (min_norm + 1e-8)) < 1000  # Not too variable
        )
        
        return stable
    
    def _check_loss_improvement(self, losses: List[float]) -> bool:
        """Check if loss is generally improving"""
        if len(losses) < 10:
            return False
            
        # Compare first half with second half
        mid_point = len(losses) // 2
        early_loss = np.mean(losses[:mid_point])
        late_loss = np.mean(losses[mid_point:])
        
        # Loss improved if recent loss is lower
        return late_loss < early_loss * 0.95  # 5% improvement threshold
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of gradient flow monitoring"""
        
        verification = self.verify_learning_progress()
        
        summary = {
            'total_steps': self.step_count,
            'total_problems': len(self.problems_detected),
            'recent_problems': self.problems_detected[-5:],
            'verification_checks': verification
        }
        
        if self.gradient_history:
            summary['gradient_stats'] = {
                'mean': np.mean(self.gradient_history),
                'std': np.std(self.gradient_history),
                'min': np.min(self.gradient_history),
                'max': np.max(self.gradient_history),
                'recent_mean': np.mean(list(self.gradient_history)[-10:])
            }
        
        if self.loss_history:
            summary['loss_stats'] = {
                'mean': np.mean(self.loss_history),
                'std': np.std(self.loss_history),
                'min': np.min(self.loss_history),
                'max': np.max(self.loss_history),
                'recent_mean': np.mean(list(self.loss_history)[-10:])
            }
            
        return summary
    
    def print_summary(self):
        """Print a human-readable summary"""
        summary = self.get_summary()
        verification = summary['verification_checks']
        
        print("\n" + "="*60)
        print("GRADIENT FLOW MONITORING SUMMARY")
        print("="*60)
        
        print(f"Total steps monitored: {summary['total_steps']}")
        print(f"Total problems detected: {summary['total_problems']}")
        
        print("\nVerification Checks:")
        for check, passed in verification.items():
            if check == 'insufficient_data':
                continue
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {check.replace('_', ' ').title()}")
        
        if summary['recent_problems']:
            print(f"\nRecent Problems:")
            for problem in summary['recent_problems']:
                print(f"  ⚠️  {problem}")
        
        if 'gradient_stats' in summary:
            stats = summary['gradient_stats']
            print(f"\nGradient Statistics:")
            print(f"  Mean norm: {stats['mean']:.6f}")
            print(f"  Recent mean: {stats['recent_mean']:.6f}")
            print(f"  Range: {stats['min']:.6f} to {stats['max']:.6f}")
        
        overall_health = all(verification.values()) if 'insufficient_data' not in verification else False
        
        if overall_health:
            print("\n🎉 GRADIENT FLOW IS HEALTHY!")
        else:
            print("\n⚠️  GRADIENT FLOW ISSUES DETECTED!")

def instrument_training_step(model: nn.Module, loss: torch.Tensor, 
                           optimizer: torch.optim.Optimizer, 
                           monitor: GradientFlowMonitor = None) -> Dict[str, Any]:
    """
    Instrument a single training step with gradient monitoring
    
    Args:
        model: PyTorch model
        loss: Loss tensor
        optimizer: PyTorch optimizer
        monitor: Existing monitor (will create new if None)
        
    Returns:
        Gradient flow information
    """
    if monitor is None:
        monitor = GradientFlowMonitor()
    
    # Check gradients before backward
    optimizer.zero_grad()
    
    # Perform backward pass and check gradients
    gradient_info = monitor.check_gradient_flow(model, loss, compute_backward=True)
    
    # Apply gradient clipping if gradients are too large
    if 'total_norm' in gradient_info and gradient_info['total_norm'] > 10:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        logger.warning(f"Applied gradient clipping (norm was {gradient_info['total_norm']:.2f})")
    
    # Optimizer step
    optimizer.step()
    
    return gradient_info

# Convenience decorator for automatic monitoring
def monitor_gradients(monitor: GradientFlowMonitor = None):
    """Decorator to automatically monitor gradients in training functions"""
    
    def decorator(training_function):
        def wrapper(*args, **kwargs):
            nonlocal monitor
            if monitor is None:
                monitor = GradientFlowMonitor()
            
            # Inject monitor into kwargs
            kwargs['gradient_monitor'] = monitor
            
            result = training_function(*args, **kwargs)
            
            # Print periodic summaries
            if monitor.step_count % 50 == 0:
                print(f"\n--- Gradient Monitoring Update (Step {monitor.step_count}) ---")
                checks = monitor.verify_learning_progress()
                for check, passed in checks.items():
                    if check != 'insufficient_data':
                        status = "✅" if passed else "❌"
                        print(f"{status} {check.replace('_', ' ').title()}")
            
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage
    print("Testing Gradient Flow Monitor...")
    
    # Create test model and data
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(), 
        nn.Linear(16, 1)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    monitor = GradientFlowMonitor()
    
    # Generate test data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    print("Running training steps with gradient monitoring...")
    
    for step in range(30):
        # Forward pass
        pred = model(X)
        loss = nn.MSELoss()(pred, y)
        
        # Monitored training step
        grad_info = instrument_training_step(model, loss, optimizer, monitor)
        
        if step % 10 == 0:
            print(f"Step {step}: Loss={loss.item():.6f}, "
                  f"Grad_norm={grad_info.get('total_norm', 0):.6f}")
    
    # Print final summary
    monitor.print_summary()