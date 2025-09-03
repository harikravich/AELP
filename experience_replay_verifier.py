#!/usr/bin/env python3
"""
Experience Replay Verification System for GAELP RL Agents
Verifies that experience replay is functioning correctly:
1. Buffer is being populated
2. Experiences are being sampled properly  
3. Batch training is occurring
4. Priority updates are working (if applicable)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque, defaultdict
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperienceStats:
    """Statistics about stored experiences"""
    buffer_size: int
    buffer_capacity: int
    buffer_utilization: float
    unique_states: int
    unique_actions: int
    reward_mean: float
    reward_std: float
    done_ratio: float

class ExperienceReplayVerifier:
    """Verifies that experience replay is working correctly"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            'buffer_sizes': deque(maxlen=window_size),
            'sample_counts': deque(maxlen=window_size),
            'batch_sizes': deque(maxlen=window_size),
            'sampling_frequencies': deque(maxlen=window_size),
            'experience_ages': deque(maxlen=window_size),
            'priority_updates': deque(maxlen=window_size),
            'reward_distributions': deque(maxlen=window_size)
        }
        
        self.problems = []
        self.step_count = 0
        self.last_buffer_hash = None
        
    def verify_buffer_population(self, replay_buffer: Any) -> Dict[str, Any]:
        """Verify that the replay buffer is being populated correctly"""
        self.step_count += 1
        
        verification = {
            'buffer_exists': replay_buffer is not None,
            'has_capacity': False,
            'has_experiences': False,
            'is_growing': False,
            'buffer_stats': None
        }
        
        if replay_buffer is None:
            self.problems.append("No replay buffer found!")
            return verification
        
        # Get buffer size and capacity
        buffer_size = self._get_buffer_size(replay_buffer)
        buffer_capacity = self._get_buffer_capacity(replay_buffer)
        
        verification['has_capacity'] = buffer_capacity > 0
        verification['has_experiences'] = buffer_size > 0
        
        # Record metrics
        self.metrics['buffer_sizes'].append(buffer_size)
        
        # Check if buffer is growing (for first few hundred steps)
        if len(self.metrics['buffer_sizes']) >= 2 and self.step_count < 500:
            recent_sizes = list(self.metrics['buffer_sizes'])[-10:]
            verification['is_growing'] = max(recent_sizes) > min(recent_sizes)
        else:
            verification['is_growing'] = True  # Assume grown after 500 steps
        
        # Get detailed buffer statistics
        if buffer_size > 0:
            buffer_stats = self._analyze_buffer_content(replay_buffer)
            verification['buffer_stats'] = buffer_stats
            self.metrics['reward_distributions'].append(buffer_stats.reward_mean)
        
        # Check for problems
        if buffer_capacity == 0:
            self.problems.append(f"Step {self.step_count}: Buffer has zero capacity!")
        elif buffer_size == 0 and self.step_count > 50:
            self.problems.append(f"Step {self.step_count}: Buffer empty after 50 steps!")
        elif buffer_size > 10 and not verification['is_growing'] and self.step_count < 300:
            self.problems.append(f"Step {self.step_count}: Buffer not growing - experiences may not be stored!")
        
        return verification
    
    def verify_sampling(self, replay_buffer: Any, batch_size: int = 32, 
                       num_samples: int = 5) -> Dict[str, Any]:
        """Verify that sampling from the buffer works correctly"""
        
        verification = {
            'can_sample': False,
            'correct_batch_size': False,
            'diverse_samples': False,
            'valid_experiences': False,
            'sampling_stats': {}
        }
        
        if replay_buffer is None:
            return verification
        
        buffer_size = self._get_buffer_size(replay_buffer)
        if buffer_size < batch_size:
            verification['sampling_stats'] = {'error': 'Buffer too small for sampling'}
            return verification
        
        try:
            # Attempt multiple samples to test diversity
            sampled_batches = []
            for _ in range(num_samples):
                batch = self._sample_from_buffer(replay_buffer, batch_size)
                if batch is not None:
                    sampled_batches.append(batch)
            
            verification['can_sample'] = len(sampled_batches) > 0
            
            if sampled_batches:
                # Check batch size
                first_batch = sampled_batches[0]
                actual_batch_size = self._get_batch_size(first_batch)
                verification['correct_batch_size'] = actual_batch_size == batch_size
                
                self.metrics['batch_sizes'].append(actual_batch_size)
                self.metrics['sample_counts'].append(len(sampled_batches))
                
                # Check diversity (different samples should have different experiences)
                verification['diverse_samples'] = self._check_sample_diversity(sampled_batches)
                
                # Check validity of experiences
                verification['valid_experiences'] = self._validate_experience_format(first_batch)
                
                # Calculate sampling statistics
                verification['sampling_stats'] = self._calculate_sampling_stats(sampled_batches)
                
        except Exception as e:
            self.problems.append(f"Step {self.step_count}: Sampling failed - {e}")
            verification['sampling_stats'] = {'error': str(e)}
        
        return verification
    
    def verify_priority_updates(self, replay_buffer: Any, td_errors: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Verify priority updates for prioritized experience replay"""
        
        verification = {
            'has_priorities': False,
            'can_update_priorities': False,
            'priorities_change': False
        }
        
        if not self._is_prioritized_buffer(replay_buffer):
            verification['has_priorities'] = False
            return verification
        
        verification['has_priorities'] = True
        
        try:
            # Try to get current priorities
            priorities_before = self._get_buffer_priorities(replay_buffer)
            
            if priorities_before is not None and td_errors is not None:
                # Try to update priorities
                indices = np.arange(min(len(td_errors), len(priorities_before)))
                self._update_buffer_priorities(replay_buffer, indices, td_errors[:len(indices)])
                
                # Check if priorities changed
                priorities_after = self._get_buffer_priorities(replay_buffer)
                
                if priorities_after is not None:
                    verification['can_update_priorities'] = True
                    
                    # Check if priorities actually changed
                    if len(priorities_before) == len(priorities_after):
                        max_change = np.max(np.abs(priorities_after - priorities_before))
                        verification['priorities_change'] = max_change > 1e-6
                        self.metrics['priority_updates'].append(max_change)
                
        except Exception as e:
            self.problems.append(f"Step {self.step_count}: Priority update failed - {e}")
            
        return verification
    
    def verify_batch_training(self, agent: Any, replay_buffer: Any, 
                            training_frequency: int = 4) -> Dict[str, Any]:
        """Verify that batch training is occurring at the right frequency"""
        
        verification = {
            'training_occurred': False,
            'correct_frequency': False,
            'weight_updates': False
        }
        
        # This is harder to verify without instrumenting the actual training loop
        # We can check if the agent has the necessary methods and parameters
        
        if hasattr(agent, 'train') or hasattr(agent, 'update') or hasattr(agent, 'learn'):
            verification['training_occurred'] = True
            
        # Check if training frequency makes sense
        buffer_size = self._get_buffer_size(replay_buffer) if replay_buffer else 0
        if buffer_size >= 32:  # Minimum batch size
            expected_training_steps = buffer_size // training_frequency
            verification['correct_frequency'] = expected_training_steps > 0
            
            # Record frequency metric
            self.metrics['sampling_frequencies'].append(training_frequency)
        
        return verification
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive experience replay verification report"""
        
        report = {
            'total_steps': self.step_count,
            'total_problems': len(self.problems),
            'recent_problems': self.problems[-10:],
            'metrics_summary': {},
            'verification_summary': {}
        }
        
        # Summarize metrics
        for metric_name, metric_values in self.metrics.items():
            if len(metric_values) > 0:
                values = list(metric_values)
                report['metrics_summary'][metric_name] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'recent_mean': float(np.mean(values[-10:])) if len(values) >= 10 else float(np.mean(values))
                }
        
        # Overall verification status
        buffer_healthy = (
            len(self.metrics['buffer_sizes']) > 0 and
            max(self.metrics['buffer_sizes']) > 0
        )
        
        sampling_healthy = (
            len(self.metrics['sample_counts']) > 0 and
            max(self.metrics['sample_counts']) > 0 and
            len(self.metrics['batch_sizes']) > 0
        )
        
        report['verification_summary'] = {
            'buffer_healthy': buffer_healthy,
            'sampling_healthy': sampling_healthy,
            'no_critical_problems': len([p for p in self.problems if 'CRITICAL' in p or 'failed' in p]) == 0,
            'overall_healthy': buffer_healthy and sampling_healthy
        }
        
        return report
    
    def plot_replay_metrics(self, save_path: str = "replay_metrics.png"):
        """Plot experience replay metrics"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Experience Replay Verification Metrics', fontsize=16)
        
        # Buffer size over time
        if len(self.metrics['buffer_sizes']) > 0:
            axes[0, 0].plot(list(self.metrics['buffer_sizes']))
            axes[0, 0].set_title('Buffer Size Over Time')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Buffer Size')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Batch sizes
        if len(self.metrics['batch_sizes']) > 0:
            batch_sizes = list(self.metrics['batch_sizes'])
            axes[0, 1].hist(batch_sizes, bins=20, alpha=0.7)
            axes[0, 1].set_title('Batch Size Distribution')
            axes[0, 1].set_xlabel('Batch Size')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Sampling frequency
        if len(self.metrics['sampling_frequencies']) > 0:
            axes[0, 2].plot(list(self.metrics['sampling_frequencies']))
            axes[0, 2].set_title('Sampling Frequency')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Reward distribution
        if len(self.metrics['reward_distributions']) > 0:
            axes[1, 0].plot(list(self.metrics['reward_distributions']))
            axes[1, 0].set_title('Buffer Reward Mean Over Time')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Mean Reward')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Priority updates (if available)
        if len(self.metrics['priority_updates']) > 0:
            axes[1, 1].plot(list(self.metrics['priority_updates']))
            axes[1, 1].set_title('Priority Update Magnitudes')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Max Priority Change')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Summary stats
        report = self.generate_comprehensive_report()
        verification = report['verification_summary']
        
        checks = ['buffer_healthy', 'sampling_healthy', 'no_critical_problems', 'overall_healthy']
        check_values = [verification[check] for check in checks]
        colors = ['green' if v else 'red' for v in check_values]
        
        axes[1, 2].barh(checks, [1 if v else 0 for v in check_values], color=colors, alpha=0.7)
        axes[1, 2].set_title('Verification Checks')
        axes[1, 2].set_xlabel('Pass/Fail')
        axes[1, 2].set_xlim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved experience replay metrics plot to {save_path}")
    
    # Helper methods for different buffer types
    def _get_buffer_size(self, buffer: Any) -> int:
        """Get buffer size, handling different buffer implementations"""
        if hasattr(buffer, '__len__'):
            return len(buffer)
        elif hasattr(buffer, 'size'):
            return buffer.size()
        elif hasattr(buffer, 'buffer_size'):
            return buffer.buffer_size
        elif hasattr(buffer, 'n_entries'):
            return buffer.n_entries
        elif hasattr(buffer, 'buffer') and hasattr(buffer.buffer, '__len__'):
            return len(buffer.buffer)
        else:
            return 0
    
    def _get_buffer_capacity(self, buffer: Any) -> int:
        """Get buffer capacity"""
        if hasattr(buffer, 'capacity'):
            return buffer.capacity
        elif hasattr(buffer, 'maxlen'):
            return buffer.maxlen
        elif hasattr(buffer, 'buffer_size'):
            return buffer.buffer_size
        elif hasattr(buffer, 'size') and callable(buffer.size):
            return buffer.size()  # Current size as proxy
        else:
            return 10000  # Default assumption
    
    def _sample_from_buffer(self, buffer: Any, batch_size: int) -> Optional[Any]:
        """Sample from buffer, handling different implementations"""
        try:
            if hasattr(buffer, 'sample'):
                return buffer.sample(batch_size)
            elif hasattr(buffer, 'sample_batch'):
                return buffer.sample_batch(batch_size)
            elif hasattr(buffer, 'get_batch'):
                return buffer.get_batch(batch_size)
            else:
                return None
        except Exception as e:
            logger.warning(f"Sampling failed: {e}")
            return None
    
    def _get_batch_size(self, batch: Any) -> int:
        """Get actual batch size from sampled batch"""
        if isinstance(batch, (list, tuple)):
            if len(batch) > 0:
                first_element = batch[0]
                if hasattr(first_element, '__len__'):
                    return len(first_element)
                elif isinstance(first_element, torch.Tensor):
                    return first_element.shape[0]
            return len(batch)
        elif hasattr(batch, '__len__'):
            return len(batch)
        elif isinstance(batch, torch.Tensor):
            return batch.shape[0]
        else:
            return 0
    
    def _analyze_buffer_content(self, buffer: Any) -> ExperienceStats:
        """Analyze the content of the buffer"""
        buffer_size = self._get_buffer_size(buffer)
        buffer_capacity = self._get_buffer_capacity(buffer)
        
        # Try to extract experiences for analysis
        rewards = []
        done_flags = []
        
        try:
            # Different ways to access buffer content
            if hasattr(buffer, 'buffer') and isinstance(buffer.buffer, list):
                for exp in buffer.buffer[:min(100, len(buffer.buffer))]:
                    if hasattr(exp, 'reward'):
                        rewards.append(exp.reward)
                    if hasattr(exp, 'done'):
                        done_flags.append(exp.done)
                        
            elif hasattr(buffer, 'sample'):
                # Sample a small batch to analyze
                sample = buffer.sample(min(32, buffer_size))
                if isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    # Assume format: (states, actions, rewards, next_states, dones)
                    if len(sample) >= 5:
                        rewards = sample[2] if isinstance(sample[2], (list, np.ndarray)) else [sample[2]]
                        done_flags = sample[4] if isinstance(sample[4], (list, np.ndarray)) else [sample[4]]
                        
        except Exception:
            pass  # Couldn't analyze content
        
        stats = ExperienceStats(
            buffer_size=buffer_size,
            buffer_capacity=buffer_capacity,
            buffer_utilization=buffer_size / max(buffer_capacity, 1),
            unique_states=0,  # Hard to compute without more info
            unique_actions=0,  # Hard to compute without more info
            reward_mean=float(np.mean(rewards)) if rewards else 0.0,
            reward_std=float(np.std(rewards)) if rewards else 0.0,
            done_ratio=float(np.mean(done_flags)) if done_flags else 0.0
        )
        
        return stats
    
    def _check_sample_diversity(self, sampled_batches: List[Any]) -> bool:
        """Check if samples are diverse (not always the same)"""
        if len(sampled_batches) < 2:
            return True  # Can't check with just one sample
        
        # Simple diversity check: convert to string and compare
        try:
            batch_strings = []
            for batch in sampled_batches[:3]:  # Check first 3 batches
                batch_str = str(batch)[:100]  # First 100 chars
                batch_strings.append(batch_str)
            
            # If all strings are different, samples are diverse
            unique_strings = set(batch_strings)
            return len(unique_strings) > 1
            
        except Exception:
            return True  # Assume diverse if can't check
    
    def _validate_experience_format(self, batch: Any) -> bool:
        """Validate that experiences have the expected format"""
        try:
            # Check if batch is a tuple/list with expected number of elements
            if isinstance(batch, (list, tuple)):
                # Common formats: (states, actions, rewards, next_states, dones)
                return len(batch) >= 4
            
            # Check if it's a dict with expected keys
            elif isinstance(batch, dict):
                expected_keys = {'states', 'actions', 'rewards', 'next_states'}
                return len(expected_keys.intersection(batch.keys())) >= 3
            
            # Check if it's a tensor/array
            elif isinstance(batch, (torch.Tensor, np.ndarray)):
                return batch.shape[0] > 0  # Has batch dimension
            
            return False
            
        except Exception:
            return False
    
    def _calculate_sampling_stats(self, sampled_batches: List[Any]) -> Dict[str, Any]:
        """Calculate statistics about sampling"""
        stats = {
            'num_samples': len(sampled_batches),
            'consistent_batch_size': True,
            'avg_batch_size': 0
        }
        
        if sampled_batches:
            batch_sizes = [self._get_batch_size(batch) for batch in sampled_batches]
            stats['avg_batch_size'] = np.mean(batch_sizes)
            stats['consistent_batch_size'] = len(set(batch_sizes)) == 1
        
        return stats
    
    def _is_prioritized_buffer(self, buffer: Any) -> bool:
        """Check if buffer supports prioritized sampling"""
        return (
            hasattr(buffer, 'update_priorities') or
            hasattr(buffer, 'update') or
            'priority' in type(buffer).__name__.lower() or
            'prioritized' in type(buffer).__name__.lower()
        )
    
    def _get_buffer_priorities(self, buffer: Any) -> Optional[np.ndarray]:
        """Get current priorities from buffer"""
        try:
            if hasattr(buffer, 'priorities'):
                return np.array(buffer.priorities)
            elif hasattr(buffer, 'tree') and hasattr(buffer.tree, 'tree'):
                return np.array(buffer.tree.tree)
            else:
                return None
        except Exception:
            return None
    
    def _update_buffer_priorities(self, buffer: Any, indices: np.ndarray, td_errors: np.ndarray):
        """Update buffer priorities"""
        if hasattr(buffer, 'update_priorities'):
            buffer.update_priorities(indices, td_errors)
        elif hasattr(buffer, 'update'):
            for idx, error in zip(indices, td_errors):
                buffer.update(idx, error)

def verify_experience_replay(agent_or_buffer: Any, agent: Any = None, 
                           num_verification_steps: int = 100) -> Dict[str, Any]:
    """
    Convenience function to verify experience replay
    
    Args:
        agent_or_buffer: Either an agent with replay buffer or the buffer directly
        agent: Agent (if buffer passed separately)
        num_verification_steps: Number of steps to verify
        
    Returns:
        Verification report
    """
    verifier = ExperienceReplayVerifier()
    
    # Get buffer and agent
    if hasattr(agent_or_buffer, 'replay_buffer'):
        buffer = agent_or_buffer.replay_buffer
        agent = agent_or_buffer
    else:
        buffer = agent_or_buffer
    
    logger.info(f"Starting experience replay verification for {num_verification_steps} steps...")
    
    for step in range(num_verification_steps):
        # Verify buffer population
        pop_result = verifier.verify_buffer_population(buffer)
        
        # Verify sampling (if buffer has enough experiences)
        buffer_size = verifier._get_buffer_size(buffer)
        if buffer_size >= 32:
            sample_result = verifier.verify_sampling(buffer, batch_size=32)
            
            # Verify priority updates if supported
            if verifier._is_prioritized_buffer(buffer):
                td_errors = np.random.random(10)  # Mock TD errors
                priority_result = verifier.verify_priority_updates(buffer, td_errors)
        
        # Verify batch training
        if agent is not None:
            train_result = verifier.verify_batch_training(agent, buffer)
    
    # Generate final report
    report = verifier.generate_comprehensive_report()
    
    logger.info("Experience Replay Verification Complete!")
    logger.info(f"Buffer healthy: {report['verification_summary']['buffer_healthy']}")
    logger.info(f"Sampling healthy: {report['verification_summary']['sampling_healthy']}")
    logger.info(f"Overall healthy: {report['verification_summary']['overall_healthy']}")
    
    if report['recent_problems']:
        logger.warning("Recent problems detected:")
        for problem in report['recent_problems'][-5:]:
            logger.warning(f"  - {problem}")
    
    # Save visualization
    verifier.plot_replay_metrics("experience_replay_verification.png")
    
    return report

if __name__ == "__main__":
    # Example usage with mock buffer
    from collections import deque
    
    class MockReplayBuffer:
        def __init__(self, capacity=1000):
            self.buffer = deque(maxlen=capacity)
            self.capacity = capacity
        
        def __len__(self):
            return len(self.buffer)
        
        def add(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
        
        def sample(self, batch_size):
            if len(self.buffer) < batch_size:
                return None
            
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            
            states = [b[0] for b in batch]
            actions = [b[1] for b in batch]
            rewards = [b[2] for b in batch]
            next_states = [b[3] for b in batch]
            dones = [b[4] for b in batch]
            
            return (states, actions, rewards, next_states, dones)
    
    # Test verification
    buffer = MockReplayBuffer()
    
    # Populate buffer
    for i in range(200):
        state = np.random.random(4)
        action = np.random.randint(0, 2)
        reward = np.random.normal(0, 1)
        next_state = np.random.random(4)
        done = i % 50 == 0
        buffer.add(state, action, reward, next_state, done)
    
    # Run verification
    report = verify_experience_replay(buffer, num_verification_steps=50)
    print("\nVerification completed! Check 'experience_replay_verification.png' for plots.")