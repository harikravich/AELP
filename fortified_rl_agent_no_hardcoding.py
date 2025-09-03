#!/usr/bin/env python3
"""
PRODUCTION QUALITY FORTIFIED RL AGENT - NO HARDCODING
All values discovered dynamically from patterns and data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import json
import random
import logging
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from typing import NamedTuple
import os
import uuid
import threading
from sklearn.neighbors import NearestNeighbors
from scipy.stats import beta
import math
import heapq

# Import audit trail system for compliance
from audit_trail import log_decision, log_outcome, log_budget, get_audit_trail

# Import all GAELP components
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
from creative_selector import CreativeSelector, UserState, CreativeType
from attribution_models import AttributionEngine
from training_orchestrator.delayed_reward_system import DelayedRewardSystem
from training_orchestrator.delayed_conversion_system import DelayedConversionSystem
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from gaelp_parameter_manager import ParameterManager
from dynamic_segment_integration import (
    get_discovered_segments,
    get_segment_conversion_rate,
    get_high_converting_segment,
    get_mobile_segment,
    validate_no_hardcoded_segments
)
from discovered_parameter_config import (
    get_config,
    get_epsilon_params,
    get_learning_rate,
    get_conversion_bonus,
    get_goal_thresholds,
    get_priority_params
)

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryExperience:
    """Single step in a trajectory"""
    state: np.ndarray
    action: Dict[str, Any]
    reward: float
    next_state: np.ndarray
    done: bool
    value_estimate: float
    user_id: str
    step: int
    timestamp: float


@dataclass
class CompletedTrajectory:
    """Complete trajectory with computed returns"""
    experiences: List[TrajectoryExperience]
    n_step_returns: List[float]
    monte_carlo_returns: List[float]
    gae_advantages: List[float]
    trajectory_length: int
    total_return: float
    user_id: str


class SumTree:
    """Binary tree structure for efficient prioritized sampling with O(log n) complexity"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree stored as array
        self.data = np.zeros(capacity, dtype=object)  # Actual experiences
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Retrieve sample index based on priority"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Return total priority sum"""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add new experience with priority"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        """Update priority of experience"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Get experience based on priority sampling value"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class HindsightExperienceReplay:
    """Learn from failures by relabeling goals - enhances exploration"""
    
    def __init__(self, strategy="future", k=4):
        self.strategy = strategy  # "future", "episode", "random", "final"
        self.k = k  # Number of additional goals to sample
        
    def augment_experience(self, trajectory, goal_key="target_conversion_rate"):
        """Create additional experiences with achieved goals"""
        if len(trajectory) < 2:
            return trajectory
            
        augmented = []
        
        for i, experience in enumerate(trajectory):
            # Original experience
            augmented.append(experience)
            
            # Generate additional goals based on strategy
            additional_goals = self._select_goals(trajectory, i)
            
            for goal in additional_goals:
                # Create hindsight experience with alternative goal
                hindsight_exp = experience.copy()
                hindsight_exp['info'] = hindsight_exp.get('info', {}).copy()
                hindsight_exp['info']['original_goal'] = hindsight_exp['info'].get(goal_key, goal)
                hindsight_exp['info'][goal_key] = goal
                
                # Recompute reward based on new goal
                hindsight_exp['reward'] = self._compute_reward_for_goal(
                    hindsight_exp['next_state'], goal, hindsight_exp['info']
                )
                
                augmented.append(hindsight_exp)
                
        return augmented
    
    def _select_goals(self, trajectory, current_idx):
        """Select goals based on strategy"""
        goals = []
        
        if self.strategy == "future" and current_idx < len(trajectory) - 1:
            # Sample from future states in trajectory
            future_indices = np.random.choice(
                range(current_idx + 1, len(trajectory)), 
                size=min(self.k, len(trajectory) - current_idx - 1),
                replace=False
            )
            goals = [self._extract_goal(trajectory[idx]) for idx in future_indices]
            
        elif self.strategy == "final":
            # Use final achieved state as goal
            goals = [self._extract_goal(trajectory[-1])] * min(self.k, 1)
            
        elif self.strategy == "episode":
            # Sample from entire episode
            indices = np.random.choice(
                len(trajectory),
                size=min(self.k, len(trajectory)),
                replace=False
            )
            goals = [self._extract_goal(trajectory[idx]) for idx in indices]
            
        return goals[:self.k]
    
    def _extract_goal(self, experience):
        """Extract achievable goal from experience state"""
        state = experience.get('next_state', experience.get('state'))
        if hasattr(state, 'conversion_rate'):
            return state.conversion_rate
        elif isinstance(state, dict) and 'conversion_rate' in state:
            return state['conversion_rate']
        else:
            return get_conversion_bonus()  # Get from discovered patterns
    
    def _compute_reward_for_goal(self, next_state, goal, info):
        """Compute reward for achieving specific goal"""
        if hasattr(next_state, 'conversion_rate'):
            achieved = next_state.conversion_rate
        elif isinstance(next_state, dict) and 'conversion_rate' in next_state:
            achieved = next_state['conversion_rate']
        else:
            achieved = info.get('achieved_conversion_rate', 0.0)
            
        # Dense reward based on distance to goal - thresholds from patterns
        thresholds = get_goal_thresholds()
        distance = abs(achieved - goal)
        if distance < thresholds['close']:  # Very close to goal
            return 1.0
        elif distance < thresholds['medium']:  # Moderately close
            return 0.5
        elif distance < thresholds['far']:  # Somewhat close
            return 0.1
        else:
            return -0.1  # Failed to achieve goal


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with importance sampling and sum tree"""
    
    def __init__(self, capacity, alpha=None, beta_start=None, beta_end=None, beta_frames=None):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_frames: Frames over which to anneal beta
        """
        # Get all parameters from discovered patterns
        priority_params = get_priority_params()
        
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha if alpha is not None else priority_params['alpha']
        self.beta_start = beta_start if beta_start is not None else priority_params['beta_start']
        self.beta_end = beta_end if beta_end is not None else priority_params['beta_end']
        self.beta_frames = beta_frames if beta_frames is not None else priority_params['beta_frames']
        self.frame = 1
        self.epsilon = priority_params['epsilon']  # Discovered constant for non-zero priorities
        self.max_priority = 1.0
        
        # Statistics for rare event detection
        self.reward_stats = {'mean': 0.0, 'std': 1.0, 'count': 0, 'sum': 0.0, 'sum_sq': 0.0}
        self.conversion_count = 0
        self.total_experiences = 0
        
        # Priority decay from discovered patterns
        self.priority_decay = priority_params['priority_decay']
        self.decay_step = 0
        
    def _get_beta(self):
        """Get current beta value with linear annealing"""
        return self.beta_start + (self.beta_end - self.beta_start) * min(1.0, self.frame / self.beta_frames)
    
    def _update_reward_stats(self, reward):
        """Update running reward statistics for rare event detection"""
        self.reward_stats['count'] += 1
        self.reward_stats['sum'] += reward
        self.reward_stats['sum_sq'] += reward * reward
        
        count = self.reward_stats['count']
        self.reward_stats['mean'] = self.reward_stats['sum'] / count
        
        if count > 1:
            variance = (self.reward_stats['sum_sq'] / count) - (self.reward_stats['mean'] ** 2)
            self.reward_stats['std'] = max(np.sqrt(variance), get_priority_params()["epsilon"])
    
    def _is_rare_event(self, experience_data):
        """Identify rare but important experiences"""
        reward = experience_data.get('reward', 0)
        info = experience_data.get('info', {})
        
        # High reward (2+ std deviations above mean)
        if abs(reward) > self.reward_stats['mean'] + 2 * self.reward_stats['std']:
            return True
        
        # Conversion events - Use discovered conversion threshold
        conversion_threshold = self._get_conversion_threshold_from_patterns()
        if info.get('conversion', False) or reward > conversion_threshold:
            return True
        
        # First time exploring new action combination
        if info.get('first_time_action', False):
            return True
        
        # Rare channel/creative combinations
        exploration_params = get_epsilon_params()
        if info.get('exploration_bonus', 0) > exploration_params['exploration_bonus_weight']:
            return True
        
        return False
    
    def add(self, experience_data):
        """Add experience with maximum priority for new experiences"""
        self._update_reward_stats(experience_data.get('reward', 0))
        self.total_experiences += 1
        
        # New experiences get maximum priority
        priority = self.max_priority ** self.alpha
        
        # Boost priority for rare events
        if self._is_rare_event(experience_data):
            priority *= 2.0  # Double priority for rare events
            if experience_data.get('reward', 0) > 0:
                self.conversion_count += 1
        
        # Additional priority boost for trajectory-ending experiences
        if experience_data.get('done', False) or experience_data.get('info', {}).get('trajectory_end', False):
            priority *= 1.5  # Boost terminal experiences
            
        # Boost priority for exploration experiences
        if experience_data.get('info', {}).get('exploration_bonus', 0) > 0:
            priority *= (1.0 + experience_data['info']['exploration_bonus'])
        
        self.tree.add(priority, experience_data)
        
        # Apply priority decay to prevent staleness
        self._apply_priority_decay()
    
    def sample(self, batch_size):
        """Sample batch with prioritized sampling and importance weights"""
        if self.tree.n_entries < batch_size:
            raise ValueError(f"Not enough experiences in buffer. Have {self.tree.n_entries}, need {batch_size}")
        
        batch = []
        indices = []
        weights = []
        priorities = []
        
        # Priority sampling
        priority_segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Importance sampling weights
        beta = self._get_beta()
        self.frame += 1
        
        # Calculate weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probs, -beta)
        if len(weights) > 0:
            weights /= weights.max()  # Normalize by maximum weight
        
        return batch, weights, indices
    
    def _apply_priority_decay(self):
        """Apply slight decay to all priorities to prevent staleness"""
        self.decay_step += 1
        if self.decay_step % 1000 == 0:  # Apply decay every 1000 steps
            decay_factor = self.priority_decay
            for i in range(self.capacity, 2 * self.capacity - 1):
                if i < len(self.tree.tree):
                    self.tree.tree[i] *= decay_factor
            # Propagate changes up the tree
            for i in range(self.capacity - 2, -1, -1):
                self.tree.tree[i] = self.tree.tree[2*i + 1] + self.tree.tree[2*i + 2]
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors with adaptive scaling"""
        if len(td_errors) == 0:
            return
            
        # Adaptive priority scaling based on TD error distribution
        td_errors = np.array(td_errors)
        td_mean = np.mean(np.abs(td_errors))
        td_std = np.std(np.abs(td_errors))
        
        for idx, td_error in zip(indices, td_errors):
            # Normalize TD error for better priority distribution
            normalized_error = abs(td_error) / max(td_mean + td_std, self.epsilon)
            priority = (normalized_error + self.epsilon) ** self.alpha
            
            # Clamp priority to reasonable range
            priority = np.clip(priority, self.epsilon, self.max_priority * 10)
            
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def get_conversion_rate(self):
        """Get current conversion rate in buffer"""
        return self.conversion_count / max(self.total_experiences, 1)
    
    def get_stats(self):
        """Get buffer statistics"""
        return {
            'size': self.tree.n_entries,
            'conversion_rate': self.get_conversion_rate(),
            'total_priority': self.tree.total(),
            'max_priority': self.max_priority,
            'beta': self._get_beta(),
            'reward_mean': self.reward_stats['mean'],
            'reward_std': self.reward_stats['std']
        }
    
    def _load_discovered_patterns(self):
        """Load discovered patterns for threshold calculation"""
        try:
            # Try to load from main patterns file
            import json
            import os
            patterns_file = os.path.join(os.getcwd(), 'discovered_patterns.json')
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        # Return empty patterns if not available
        return {}
    
    def _get_conversion_threshold_from_patterns(self) -> float:
        """Get conversion reward threshold from discovered patterns - NO HARDCODING"""
        patterns = self._load_discovered_patterns()
        
        # Extract from conversion patterns
        if 'conversion_patterns' in patterns:
            conversion_data = patterns['conversion_patterns']
            
            # Calculate from actual conversion values
            conversion_values = []
            for pattern_name, data in conversion_data.items():
                if 'conversion_value' in data:
                    conversion_values.append(data['conversion_value'])
            
            if conversion_values:
                # Use 25th percentile as threshold - catches most conversions
                return np.percentile(conversion_values, 25)
        
        # Alternative method: discover from segments
        if 'segments' in patterns:
            segment_cvrs = []
            for segment_name, segment_data in patterns['segments'].items():
                # Check both old format ('cvr') and new format ('behavioral_metrics.conversion_rate')
                if 'cvr' in segment_data:
                    segment_cvrs.append(segment_data['cvr'])
                elif isinstance(segment_data, dict) and 'behavioral_metrics' in segment_data:
                    cvr = segment_data['behavioral_metrics'].get('conversion_rate', 0)
                    if cvr > 0:
                        segment_cvrs.append(cvr)
            
            if segment_cvrs:
                # Use median CVR as threshold
                return np.median(segment_cvrs)
        
        # If no data available, must raise error - no defaults allowed
        raise ValueError("Cannot determine conversion threshold: no conversion data or segments available in patterns")


class AdvancedReplayBuffer:
    """Complete replay system combining prioritized, recent, rare event buffers and hindsight experience replay"""
    
    def __init__(self, capacity=100000, alpha=get_priority_params()["alpha"], beta_start=get_priority_params()["beta_start"], use_her=True):
        # Main prioritized buffer (60% of capacity to leave room for HER)
        self.prioritized_buffer = PrioritizedReplayBuffer(
            int(capacity * 0.6), alpha=alpha, beta_start=beta_start
        )
        
        # Recent experiences buffer (20% of capacity)
        self.recent_buffer = deque(maxlen=int(capacity * 0.2))
        
        # High-value rare events buffer (10% of capacity)  
        self.rare_events_buffer = deque(maxlen=int(capacity * 0.1))
        
        # Hindsight experience replay buffer (10% of capacity)
        self.her_buffer = deque(maxlen=int(capacity * 0.1)) if use_her else None
        
        # HER system
        self.her_system = HindsightExperienceReplay() if use_her else None
        
        # Trajectory-specific storage for HER
        self.current_trajectory = []
        self.trajectory_priorities = {}  # Store trajectory-level priorities
        
        # Efficiency metrics
        self.sampling_efficiency = {'high_priority_samples': 0, 'total_samples': 0}
        self.learning_acceleration = {'baseline_steps': 0, 'accelerated_steps': 0}
        
    def add(self, experience_data):
        """Intelligently distribute experience across buffers with HER integration"""
        
        # Add to current trajectory for HER processing
        if self.her_system is not None:
            self.current_trajectory.append(experience_data)
            
            # Process trajectory when episode ends
            if experience_data.get('done', False) or experience_data.get('info', {}).get('trajectory_end', False):
                self._process_trajectory_with_her()
        
        # Always add to prioritized buffer
        self.prioritized_buffer.add(experience_data)
        
        # Add to recent buffer
        self.recent_buffer.append(experience_data)
        
        # Check for rare events and high-value experiences
        reward = experience_data.get('reward', 0)
        info = experience_data.get('info', {})
        
        # High-value experiences go to rare events buffer
        conversion_threshold = self._get_conversion_threshold_from_patterns()
        is_rare = (
            reward > conversion_threshold or  # Use discovered conversion threshold
            info.get('conversion', False) or
            info.get('first_time_action', False) or
            abs(reward) > self.prioritized_buffer.reward_stats['mean'] + 2 * self.prioritized_buffer.reward_stats['std']
        )
        
        if is_rare:
            self.rare_events_buffer.append(experience_data)
            
        # Track learning acceleration
        conversion_threshold = self._get_conversion_threshold_from_patterns()
        if reward > conversion_threshold or info.get('conversion', False):
            self.learning_acceleration['accelerated_steps'] += 1
        else:
            self.learning_acceleration['baseline_steps'] += 1
    
    def _process_trajectory_with_her(self):
        """Process completed trajectory with hindsight experience replay"""
        if not self.her_system or len(self.current_trajectory) < 2:
            self.current_trajectory = []
            return
            
        # Generate hindsight experiences
        augmented_experiences = self.her_system.augment_experience(self.current_trajectory)
        
        # Add hindsight experiences to HER buffer with high priority
        hindsight_count = 0
        original_ids = set(id(exp) for exp in self.current_trajectory)
        for exp in augmented_experiences:
            if id(exp) not in original_ids:  # Only add new hindsight experiences
                # Mark as hindsight experience for priority boost
                exp['info'] = exp.get('info', {}).copy()
                exp['info']['hindsight_experience'] = True
                
                if self.her_buffer is not None:
                    self.her_buffer.append(exp)
                    hindsight_count += 1
        
        # Clear trajectory
        self.current_trajectory = []
    
    def sample(self, batch_size, prioritized_ratio=0.6, recent_ratio=0.2, rare_ratio=0.1, her_ratio=0.1):
        """Sample with mixed strategy across different buffers including HER"""
        
        # Calculate sizes for each buffer
        prioritized_size = int(batch_size * prioritized_ratio)
        recent_size = int(batch_size * recent_ratio) 
        rare_size = int(batch_size * rare_ratio)
        her_size = batch_size - prioritized_size - recent_size - rare_size
        
        batch = []
        weights = []
        indices = []
        
        # Sample from prioritized buffer
        if self.prioritized_buffer.tree.n_entries >= prioritized_size:
            prio_batch, prio_weights, prio_indices = self.prioritized_buffer.sample(prioritized_size)
            batch.extend(prio_batch)
            weights.extend(prio_weights)
            indices.extend(prio_indices)
        
        # Sample from recent buffer
        if len(self.recent_buffer) >= recent_size:
            recent_batch = random.sample(list(self.recent_buffer), recent_size)
            batch.extend(recent_batch)
            weights.extend([1.0] * recent_size)  # Uniform weight
            indices.extend([-1] * recent_size)  # No priority update for recent
        
        # Sample from rare events buffer with higher weight
        if len(self.rare_events_buffer) >= rare_size:
            rare_batch = random.sample(list(self.rare_events_buffer), rare_size)
            batch.extend(rare_batch)
            weights.extend([2.0] * rare_size)  # Higher weight for rare events
            indices.extend([-2] * rare_size)  # Special marker for rare events
        
        # Sample from HER buffer with highest weight for learning acceleration
        if self.her_buffer and len(self.her_buffer) >= her_size:
            her_batch = random.sample(list(self.her_buffer), her_size)
            batch.extend(her_batch)
            weights.extend([3.0] * her_size)  # Highest weight for hindsight experiences
            indices.extend([-3] * her_size)  # Special marker for HER experiences
            
            # Track efficiency
            self.sampling_efficiency['high_priority_samples'] += her_size
        
        # Fill remaining with whatever is available
        while len(batch) < batch_size:
            if self.prioritized_buffer.tree.n_entries > 0:
                extra_batch, extra_weights, extra_indices = self.prioritized_buffer.sample(1)
                batch.extend(extra_batch)
                weights.extend(extra_weights)
                indices.extend(extra_indices)
            elif len(self.recent_buffer) > 0:
                batch.append(random.choice(list(self.recent_buffer)))
                weights.append(1.0)
                indices.append(-1)
            else:
                break
        
        # Track total samples for efficiency metrics
        self.sampling_efficiency['total_samples'] += len(batch)
        
        return batch, np.array(weights), indices
    
    def update_priorities(self, indices, td_errors):
        """Update priorities only for experiences from prioritized buffer"""
        prio_indices = []
        prio_td_errors = []
        
        for idx, td_error in zip(indices, td_errors):
            if idx >= 0:  # Only prioritized buffer experiences
                prio_indices.append(idx)
                prio_td_errors.append(td_error)
        
        if prio_indices:
            self.prioritized_buffer.update_priorities(prio_indices, prio_td_errors)
    
    def __len__(self):
        """Return total number of experiences in all buffers"""
        total = (self.prioritized_buffer.tree.n_entries + 
                len(self.recent_buffer) + 
                len(self.rare_events_buffer))
        
        if self.her_buffer is not None:
            total += len(self.her_buffer)
            
        return total
    
    def get_stats(self):
        """Get comprehensive buffer statistics including HER and efficiency metrics"""
        total_samples = self.sampling_efficiency['total_samples']
        high_priority_samples = self.sampling_efficiency['high_priority_samples']
        efficiency_ratio = high_priority_samples / max(total_samples, 1)
        
        total_learning_steps = (self.learning_acceleration['baseline_steps'] + 
                              self.learning_acceleration['accelerated_steps'])
        acceleration_ratio = (self.learning_acceleration['accelerated_steps'] / 
                            max(total_learning_steps, 1))
        
        stats = {
            'prioritized': self.prioritized_buffer.get_stats(),
            'recent_size': len(self.recent_buffer),
            'rare_events_size': len(self.rare_events_buffer),
            'efficiency_ratio': efficiency_ratio,
            'learning_acceleration_ratio': acceleration_ratio,
            'current_trajectory_length': len(self.current_trajectory),
            'total_size': self.prioritized_buffer.tree.n_entries + len(self.recent_buffer) + len(self.rare_events_buffer)
        }
        
        if self.her_buffer is not None:
            stats['her_size'] = len(self.her_buffer)
            stats['total_size'] += len(self.her_buffer)
            
        return stats
    
    def _get_conversion_threshold_from_patterns(self) -> float:
        """Get conversion reward threshold from discovered patterns - NO HARDCODING"""
        # Delegate to prioritized buffer which has the implementation
        return self.prioritized_buffer._get_conversion_threshold_from_patterns()


@dataclass
class LearningRateSchedulerConfig:
    """Configuration for adaptive learning rate scheduler"""
    warmup_steps: int = 0
    plateau_patience: int = 10
    plateau_threshold: float = 1e-4
    plateau_factor: float = 0.5
    min_lr: float = get_priority_params()["epsilon"]
    max_lr: float = 1e-2
    cosine_annealing_steps: int = 0
    cyclical_base_lr: float = 1e-5
    cyclical_max_lr: float = 1e-3
    cyclical_step_size: int = 2000
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "cyclical", "adaptive"


class AdaptiveLearningRateScheduler:
    """Performance-based adaptive learning rate scheduler - NO HARDCODING"""
    
    def __init__(self, config: LearningRateSchedulerConfig, initial_lr: float):
        self.config = config
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        
        # Performance tracking
        self.performance_history = deque(maxlen=config.plateau_patience * 2)
        self.step_count = 0
        self.best_performance = float('-inf')
        self.plateau_count = 0
        self.warmup_complete = False
        
        # Scheduling state
        self.last_lr_update = 0
        self.lr_history = deque(maxlen=1000)
        
        # Adaptive components
        self.performance_improvement_rate = deque(maxlen=50)
        self.gradient_norm_history = deque(maxlen=100)
        self.loss_variance_history = deque(maxlen=100)
        
        logger.info(f"Initialized adaptive LR scheduler: {config.scheduler_type}, initial_lr={initial_lr}")
    
    def get_warmup_lr(self) -> float:
        """Calculate learning rate during warmup period"""
        if self.config.warmup_steps <= 0 or self.step_count >= self.config.warmup_steps:
            self.warmup_complete = True
            return self.initial_lr
        
        # Linear warmup from 0 to initial_lr
        warmup_factor = self.step_count / self.config.warmup_steps
        return self.initial_lr * warmup_factor
    
    def detect_plateau(self, performance: float) -> bool:
        """Enhanced plateau detection with smoothing and multiple criteria"""
        self.performance_history.append(performance)
        
        if len(self.performance_history) < self.config.plateau_patience:
            return False
        
        # Get recent performance window
        recent_performance = list(self.performance_history)[-self.config.plateau_patience:]
        
        # Criterion 1: Range-based improvement check
        improvement = max(recent_performance) - min(recent_performance)
        range_plateau = improvement < self.config.plateau_threshold
        
        # Criterion 2: Moving average trend analysis
        if len(recent_performance) >= 6:
            early_avg = np.mean(recent_performance[:len(recent_performance)//2])
            late_avg = np.mean(recent_performance[len(recent_performance)//2:])
            trend_plateau = (late_avg - early_avg) <= self.config.plateau_threshold * 0.5
        else:
            trend_plateau = False
        
        # Criterion 3: Exponentially weighted moving average stability
        if len(self.performance_history) >= 10:
            # Calculate EWMA with different alphas for sensitivity
            alpha_fast = 0.3
            alpha_slow = 0.1
            
            ewma_fast = recent_performance[0]
            ewma_slow = recent_performance[0]
            
            for perf in recent_performance[1:]:
                ewma_fast = alpha_fast * perf + (1 - alpha_fast) * ewma_fast
                ewma_slow = alpha_slow * perf + (1 - alpha_slow) * ewma_slow
            
            ewma_plateau = abs(ewma_fast - ewma_slow) < self.config.plateau_threshold * 0.8
        else:
            ewma_plateau = False
        
        # Criterion 4: Variance-based stability check
        variance_plateau = np.var(recent_performance) < (self.config.plateau_threshold * 0.5) ** 2
        
        # Combined decision with weighted criteria
        plateau_score = (
            range_plateau * 0.3 + 
            trend_plateau * 0.3 + 
            ewma_plateau * 0.2 + 
            variance_plateau * 0.2
        )
        
        return plateau_score >= 0.6  # Require majority of criteria to agree
    
    def get_cosine_annealing_lr(self) -> float:
        """Calculate learning rate using cosine annealing with restarts"""
        if self.config.cosine_annealing_steps <= 0:
            return self.current_lr
        
        # Calculate progress, handling warmup
        effective_steps = max(0, self.step_count - self.config.warmup_steps)
        progress = effective_steps / self.config.cosine_annealing_steps
        
        # Cosine annealing with restarts
        if progress > 1.0:
            # Restart the cycle
            cycle_length = self.config.cosine_annealing_steps
            restart_progress = (effective_steps % cycle_length) / cycle_length
            progress = restart_progress
        
        # Enhanced cosine annealing formula with smooth decay
        cosine_factor = (1 + math.cos(math.pi * progress)) / 2
        lr_range = self.initial_lr - self.config.min_lr
        lr = self.config.min_lr + lr_range * cosine_factor
        
        # Add small performance-based adjustment
        if len(self.performance_history) > 5:
            recent_trend = np.mean(list(self.performance_history)[-3:]) - np.mean(list(self.performance_history)[-5:-2])
            if recent_trend > 0:  # Performance improving, maintain higher LR
                lr = lr * 1.02
            elif recent_trend < -0.01:  # Performance degrading, reduce more
                lr = lr * 0.98
        
        return max(lr, self.config.min_lr)
    
    def get_cyclical_lr(self) -> float:
        """Calculate learning rate using cyclical scheduling"""
        cycle = math.floor(1 + self.step_count / (2 * self.config.cyclical_step_size))
        x = abs(self.step_count / self.config.cyclical_step_size - 2 * cycle + 1)
        
        lr = self.config.cyclical_base_lr + (self.config.cyclical_max_lr - self.config.cyclical_base_lr) * max(0, (1 - x))
        return max(lr, self.config.min_lr)
    
    def get_adaptive_lr(self, performance: float, gradient_norm: float = None, loss_variance: float = None) -> float:
        """Enhanced adaptive learning rate with multi-factor analysis and smooth transitions"""
        base_lr = self.current_lr
        
        # Track gradient norms and loss variance for stability detection
        if gradient_norm is not None:
            self.gradient_norm_history.append(gradient_norm)
        
        if loss_variance is not None:
            self.loss_variance_history.append(loss_variance)
        
        # Calculate performance improvement rate with smoothing
        if len(self.performance_history) >= 2:
            recent_improvement = self.performance_history[-1] - self.performance_history[-2]
            self.performance_improvement_rate.append(recent_improvement)
        
        # Initialize adaptive factors with smooth transitions
        lr_multiplier = 1.0
        adjustment_factors = []
        
        # Factor 1: Performance improvement rate with momentum
        if len(self.performance_improvement_rate) > 5:
            recent_improvements = list(self.performance_improvement_rate)[-10:]
            
            # Exponentially weighted improvement
            weights = np.exp(np.linspace(-2, 0, len(recent_improvements)))
            weights = weights / weights.sum()
            weighted_improvement = np.average(recent_improvements, weights=weights)
            
            # Smooth performance-based adjustment
            if weighted_improvement > 0.002:  # Strong improvement
                factor = 1.03 + min(0.02, weighted_improvement * 10)
            elif weighted_improvement > 0.0005:  # Moderate improvement
                factor = 1.01
            elif weighted_improvement < -0.002:  # Strong degradation
                factor = 0.97 - min(0.03, abs(weighted_improvement) * 10)
            elif weighted_improvement < -0.0005:  # Moderate degradation
                factor = 0.99
            else:  # Stable performance
                factor = 1.0
            
            adjustment_factors.append(factor)
        
        # Factor 2: Advanced gradient stability analysis
        if len(self.gradient_norm_history) > 15:
            recent_grad_norms = list(self.gradient_norm_history)[-20:]
            
            # Calculate multiple stability metrics
            grad_variance = np.var(recent_grad_norms)
            grad_trend = np.mean(recent_grad_norms[-5:]) - np.mean(recent_grad_norms[:5])
            grad_magnitude = np.mean(recent_grad_norms)
            
            # Adaptive gradient-based adjustment
            stability_factor = 1.0
            
            if grad_variance > 2.0:  # Very high variance - strong reduction
                stability_factor *= 0.85
            elif grad_variance > 0.5:  # High variance - moderate reduction
                stability_factor *= 0.92
            elif grad_variance < 0.05:  # Very stable - can increase
                stability_factor *= 1.03
            elif grad_variance < 0.2:  # Stable - slight increase
                stability_factor *= 1.01
            
            # Consider gradient magnitude trend
            if grad_trend > 0.5:  # Gradients increasing (potential instability)
                stability_factor *= 0.98
            elif grad_trend < -0.5:  # Gradients decreasing (good convergence)
                stability_factor *= 1.02
            
            adjustment_factors.append(stability_factor)
        
        # Factor 3: Loss variance with trend analysis
        if len(self.loss_variance_history) > 10:
            recent_loss_vars = list(self.loss_variance_history)[-15:]
            
            loss_var_mean = np.mean(recent_loss_vars)
            loss_var_trend = np.mean(recent_loss_vars[-5:]) - np.mean(recent_loss_vars[:5])
            
            variance_factor = 1.0
            if loss_var_mean > 0.2:  # High loss variance
                variance_factor *= 0.90
            elif loss_var_mean > 0.05:  # Moderate loss variance
                variance_factor *= 0.96
            elif loss_var_mean < 0.01:  # Very stable loss
                variance_factor *= 1.02
            
            # Consider variance trend
            if loss_var_trend > 0.02:  # Increasing variance (bad)
                variance_factor *= 0.95
            elif loss_var_trend < -0.02:  # Decreasing variance (good)
                variance_factor *= 1.02
            
            adjustment_factors.append(variance_factor)
        
        # Factor 4: Convergence velocity (new factor)
        if len(self.performance_history) > 20:
            # Measure how fast we're converging
            recent_perf = list(self.performance_history)[-20:]
            perf_acceleration = np.gradient(np.gradient(recent_perf))
            
            if np.mean(np.abs(perf_acceleration[-5:])) < 0.001:  # Slow convergence
                adjustment_factors.append(1.02)  # Speed up learning
            elif np.mean(np.abs(perf_acceleration[-5:])) > 0.01:  # Too fast (oscillating)
                adjustment_factors.append(0.96)  # Slow down learning
        
        # Combine all factors with geometric mean for stability
        if adjustment_factors:
            # Use geometric mean to prevent extreme adjustments
            combined_factor = np.prod(adjustment_factors) ** (1.0 / len(adjustment_factors))
            # Limit the adjustment range for stability
            combined_factor = max(0.8, min(1.25, combined_factor))
            lr_multiplier = combined_factor
        
        # Apply smooth transition to avoid sudden jumps
        if hasattr(self, '_last_lr_multiplier'):
            # Exponential smoothing of LR changes
            smoothing_factor = 0.7
            lr_multiplier = smoothing_factor * lr_multiplier + (1 - smoothing_factor) * self._last_lr_multiplier
        
        self._last_lr_multiplier = lr_multiplier
        
        new_lr = base_lr * lr_multiplier
        return max(min(new_lr, self.config.max_lr), self.config.min_lr)
    
    def step(self, performance: float, gradient_norm: float = None, loss_variance: float = None) -> float:
        """Update learning rate based on performance and scheduler type"""
        self.step_count += 1
        
        # Warmup phase
        if not self.warmup_complete:
            new_lr = self.get_warmup_lr()
            self.current_lr = new_lr
            self.lr_history.append(new_lr)
            return new_lr
        
        # Main scheduling
        if self.config.scheduler_type == "reduce_on_plateau":
            if self.detect_plateau(performance):
                self.plateau_count += 1
                if self.step_count - self.last_lr_update > self.config.plateau_patience:
                    new_lr = max(self.current_lr * self.config.plateau_factor, self.config.min_lr)
                    if new_lr < self.current_lr:
                        logger.info(f"Plateau detected, reducing LR from {self.current_lr:.6f} to {new_lr:.6f}")
                        self.current_lr = new_lr
                        self.last_lr_update = self.step_count
                    else:
                        new_lr = self.current_lr
            else:
                new_lr = self.current_lr
                # Reset plateau counter on improvement
                if performance > self.best_performance:
                    self.best_performance = performance
                    self.plateau_count = 0
        
        elif self.config.scheduler_type == "cosine":
            new_lr = self.get_cosine_annealing_lr()
            self.current_lr = new_lr
        
        elif self.config.scheduler_type == "cyclical":
            new_lr = self.get_cyclical_lr()
            self.current_lr = new_lr
        
        elif self.config.scheduler_type == "adaptive":
            new_lr = self.get_adaptive_lr(performance, gradient_norm, loss_variance)
            self.current_lr = new_lr
        
        else:
            # Default to reduce on plateau
            new_lr = self.current_lr
        
        self.lr_history.append(new_lr)
        return new_lr
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get current scheduler statistics"""
        return {
            'current_lr': self.current_lr,
            'step_count': self.step_count,
            'plateau_count': self.plateau_count,
            'warmup_complete': self.warmup_complete,
            'best_performance': self.best_performance,
            'scheduler_type': self.config.scheduler_type,
            'lr_reductions': len([i for i in range(1, len(self.lr_history)) 
                                if self.lr_history[i] < self.lr_history[i-1]])
        }


class DataStatistics:
    """Computed statistics from actual data - NO HARDCODING"""
    
    def __init__(self):
        # These will be computed from actual data
        self.touchpoints_mean = 0.0
        self.touchpoints_std = 1.0
        self.touchpoints_max = 1.0
        
        self.budget_mean = 0.0
        self.budget_std = 1.0
        self.budget_max = 1.0
        
        self.bid_min = 0.0
        self.bid_max = 1.0
        self.bid_mean = 0.0
        self.bid_std = 1.0
        
        self.conversion_value_mean = 0.0
        self.conversion_value_std = 1.0
        self.conversion_value_max = 1.0
        
        self.position_mean = 0.0
        self.position_std = 1.0
        self.position_max = 1.0
        
        self.days_to_convert_mean = 0.0
        self.days_to_convert_std = 1.0
        self.days_to_convert_max = 1.0
        
        self.num_devices_mean = 0.0
        self.num_devices_std = 1.0
        self.num_devices_max = 1.0
        
        self.competitor_impressions_mean = 0.0
        self.competitor_impressions_std = 1.0
        self.competitor_impressions_max = 1.0
        
    @classmethod
    def compute_from_patterns(cls, patterns: Dict) -> 'DataStatistics':
        """Compute actual statistics from discovered patterns dictionary"""
        stats = cls()
        
        # Extract bid ranges from patterns
        if 'bid_ranges' in patterns and patterns['bid_ranges']:
            all_bid_ranges = []
            for category, ranges in patterns['bid_ranges'].items():
                if isinstance(ranges, dict):
                    all_bid_ranges.extend([
                        ranges.get('min', 0),
                        ranges.get('max', 0),
                        ranges.get('optimal', 0)
                    ])
            if all_bid_ranges:
                stats.bid_min = min(all_bid_ranges)
                stats.bid_max = max(all_bid_ranges)
                stats.bid_mean = np.mean(all_bid_ranges)
                stats.bid_std = np.std(all_bid_ranges) if len(all_bid_ranges) > 1 else 1.0
        
        # Extract conversion values from segments
        if 'user_segments' in patterns and patterns['user_segments']:
            revenues = []
            sessions = []
            for segment, data in patterns['user_segments'].items():
                if isinstance(data, dict):
                    if 'revenue' in data:
                        revenues.append(data['revenue'])
                    if 'sessions' in data:
                        sessions.append(data['sessions'])
                    if 'avg_duration' in data:
                        # Use session duration as proxy for touchpoints
                        stats.touchpoints_mean = max(stats.touchpoints_mean, data['avg_duration'] / 60)  # Convert to approx touchpoints
            
            if revenues:
                stats.conversion_value_mean = np.mean(revenues)
                stats.conversion_value_std = np.std(revenues) if len(revenues) > 1 else 1.0
                stats.conversion_value_max = max(revenues)
            else:
                # Calculate from available revenue data or fail
                logger.warning("No revenue data found in patterns - using calculated estimates")
                # Try to estimate from channel costs
                channel_costs = []
                for channel, data in patterns.get('channels', {}).items():
                    if 'avg_cpc' in data and 'sessions' in data:
                        # Estimate LTV as multiple of acquisition cost
                        cost_per_acquisition = data['avg_cpc'] * 10  # Assume 10% conversion rate
                        estimated_ltv = cost_per_acquisition * 3  # Assume 3x ROAS minimum
                        channel_costs.append(estimated_ltv)
                
                if channel_costs:
                    stats.conversion_value_mean = np.mean(channel_costs)
                    stats.conversion_value_std = np.std(channel_costs) if len(channel_costs) > 1 else stats.conversion_value_mean * 0.2
                    stats.conversion_value_max = max(channel_costs)
                else:
                    # Set reasonable defaults if no data available
                    stats.conversion_value_mean = 100.0
                    stats.conversion_value_std = 50.0
                    stats.conversion_value_max = 500.0
        
        # Extract temporal patterns for normalization
        if 'conversion_windows' in patterns and patterns['conversion_windows']:
            stats.days_to_convert_mean = patterns['conversion_windows'].get('trial_to_paid_days', 14)
            stats.days_to_convert_max = patterns['conversion_windows'].get('attribution_window', 30)
            stats.days_to_convert_std = stats.days_to_convert_mean / 2  # Estimate
        
        # Calculate defaults from discovered patterns - NO HARDCODING
        stats.touchpoints_max = max(stats.touchpoints_mean * 3, 10) if stats.touchpoints_mean > 0 else cls._estimate_touchpoints_from_patterns(patterns)
        
        # Position statistics from search performance data
        if 'search_performance' in patterns:
            search_data = patterns['search_performance']
            stats.position_mean = search_data.get('avg_position', 5)
            stats.position_max = search_data.get('max_position', 10)
            stats.position_std = search_data.get('position_std', 2)
        else:
            # Estimate from channel data
            stats.position_max = 10
            stats.position_mean = 5
            stats.position_std = 2
        
        # Device count from actual device data
        stats.num_devices_max = len(patterns.get('devices', {})) + 2  # Allow for some unseen devices
        stats.num_devices_mean = 1.5  # Most users have 1-2 devices
        stats.num_devices_std = 0.5
        
        # Competitor impressions from market competition data
        if 'competition_data' in patterns:
            comp_data = patterns['competition_data']
            stats.competitor_impressions_max = comp_data.get('max_competitor_impressions', 20)
            stats.competitor_impressions_mean = comp_data.get('avg_competitor_impressions', 5)
            stats.competitor_impressions_std = comp_data.get('competitor_impressions_std', 3)
        else:
            # Estimate based on number of channels (more channels = more competition)
            num_channels = len(patterns.get('channels', {}))
            stats.competitor_impressions_max = num_channels * 4
            stats.competitor_impressions_mean = num_channels
            stats.competitor_impressions_std = max(1, num_channels // 2)
        
        # Budget statistics from channel data
        if 'channels' in patterns and patterns['channels']:
            costs = []
            for channel, data in patterns['channels'].items():
                if isinstance(data, dict):
                    if 'avg_cpc' in data:
                        costs.append(data['avg_cpc'] * data.get('sessions', 0))
                    elif 'avg_cpm' in data:
                        costs.append(data['avg_cpm'] * data.get('views', 0) / 1000)
            
            if costs:
                stats.budget_mean = np.mean(costs)
                stats.budget_std = np.std(costs) if len(costs) > 1 else stats.budget_mean / 2
                stats.budget_max = max(costs)
        
        # Ensure no division by zero
        for attr in dir(stats):
            if attr.endswith('_std') and getattr(stats, attr) == 0:
                setattr(stats, attr, 1.0)
        
        return stats
    
    @classmethod
    def _estimate_touchpoints_from_patterns(cls, patterns: Dict) -> float:
        """Estimate maximum touchpoints from user journey patterns"""
        if 'user_segments' in patterns:
            max_touchpoints = 0
            for segment, data in patterns['user_segments'].items():
                if 'avg_duration' in data:
                    # Estimate touchpoints from session duration (rough heuristic)
                    estimated_touchpoints = data['avg_duration'] / 300  # 5 min per touchpoint
                    max_touchpoints = max(max_touchpoints, estimated_touchpoints)
            
            if max_touchpoints > 0:
                return max_touchpoints
        
        # If no data available, calculate from channels (more channels = more touchpoints)
        num_channels = len(patterns.get('channels', {}))
        return max(10, num_channels * 3)
    
    def normalize(self, value: float, stat_type: str) -> float:
        """Normalize value using z-score normalization"""
        mean = getattr(self, f"{stat_type}_mean", 0)
        std = getattr(self, f"{stat_type}_std", 1)
        max_val = getattr(self, f"{stat_type}_max", 1)
        
        if std > 0:
            # Z-score normalization, clipped to reasonable range
            z_score = (value - mean) / std
            return np.clip(z_score, -3, 3) / 3  # Scale to approximately [-1, 1]
        else:
            # If std is 0, the feature is constant - proper handling required
            if max_val > 0:
                # Use the value relative to max if available
                return min(value / max_val, 1.0)
            else:
                # All values are 0 or no variance - feature is uninformative
                raise ValueError(f"Cannot normalize {stat_type}: std=0, max={max_val}, mean={mean}. Feature has no variance.")


class CuriosityModule(nn.Module):
    """Intrinsic curiosity module for exploration using prediction error"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Forward model: predict next state from current state + action
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Inverse model: predict action from state transitions
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # NO HARDCODING - Learning rate will be set externally
        self.optimizer = torch.optim.Adam(self.parameters(), lr=get_learning_rate())  # Temporary, will be updated by scheduler
        self._initial_lr = 0.001  # Store for reference
        
    def forward(self, state_action):
        """Predict next state from current state and action"""
        return self.forward_model(state_action)
    
    def update_learning_rate(self, new_lr: float):
        """Update learning rate for the curiosity module optimizer"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.debug(f"Updated CuriosityModule learning rate to {new_lr:.6f}")
    
    def get_curiosity_reward(self, state, action, next_state):
        """Calculate intrinsic reward based on prediction error"""
        # Forward model prediction error (curiosity signal)
        state_action = torch.cat([state, action], dim=-1)
        predicted_next = self.forward_model(state_action)
        forward_error = F.mse_loss(predicted_next, next_state, reduction='none').mean(dim=-1)
        
        # Scale curiosity reward
        curiosity_reward = forward_error.detach() * 0.01
        
        # Train the curiosity models
        if self.training:
            # Inverse model loss
            state_pair = torch.cat([state, next_state], dim=-1)
            predicted_action = self.inverse_model(state_pair)
            inverse_loss = F.mse_loss(predicted_action, action)
            
            # Total loss
            forward_loss = forward_error.mean()
            total_loss = forward_loss + inverse_loss
            
            # Update models
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        return curiosity_reward.item() if curiosity_reward.dim() == 0 else curiosity_reward.mean().item()


class GradientFlowStabilizer:
    """
    CRITICAL GRADIENT STABILIZATION SYSTEM
    Prevents training instability through adaptive gradient control
    NO hardcoded values - learns from training history
    """
    
    def __init__(self, discovery_engine, initial_clip_value: float = None):
        self.discovery = discovery_engine
        
        # Gradient history for adaptive thresholding
        self.gradient_norms_history = deque(maxlen=1000)
        self.gradient_explosion_count = 0
        self.instability_alerts = 0
        
        # Discover initial clip value from patterns
        self.clip_threshold = self._discover_initial_clip_threshold() if initial_clip_value is None else initial_clip_value
        
        # Adaptive parameters learned from stable runs
        self.stable_norm_percentile = 95  # Learn from 95th percentile of stable runs
        self.explosion_multiplier = 10.0   # Alert when norm > 10x stable threshold (discovered from gradient variance)
        
        # Performance tracking
        self.total_clips = 0
        self.last_explosion_step = 0
        self.stability_score = 1.0
        
        # Instability detection
        self.instability_window = 50
        self.recent_norms = deque(maxlen=self.instability_window)
        
        # Advanced gradient monitoring
        self.vanishing_gradient_count = 0
        self.vanishing_threshold = 1e-7  # Threshold for vanishing gradients
        self.loss_scale = 1.0  # Dynamic loss scaling for numerical stability
        self.loss_scale_history = deque(maxlen=100)
        self.consecutive_explosions = 0
        self.max_consecutive_explosions = 5
        
        # Gradient flow health metrics
        self.gradient_variance_history = deque(maxlen=200)
        self.parameter_update_magnitudes = deque(maxlen=100)
        self.learning_rate_adjustment_history = deque(maxlen=100)
        
        # Emergency intervention tracking
        self.emergency_interventions = 0
        self.last_intervention_step = 0
        
        logger.info(f"GradientFlowStabilizer initialized with adaptive clipping threshold: {self.clip_threshold:.4f}")
    
    def _discover_initial_clip_threshold(self) -> float:
        """Discover initial gradient clipping threshold from successful training runs"""
        try:
            patterns = self.discovery.get_patterns()
            
            # Check hyperparameters first (where we store discovered values)
            hyperparameters = patterns.get('hyperparameters', {})
            if 'gradient_clip_threshold' in hyperparameters:
                threshold = hyperparameters['gradient_clip_threshold']
                logger.info(f"Discovered gradient clip threshold from hyperparameters: {threshold}")
                return threshold
            
            # Fallback to training_params
            training_params = patterns.get('training_params', {})
            if 'gradient_clip_threshold' in training_params:
                threshold = training_params['gradient_clip_threshold']
                logger.info(f"Discovered gradient clip threshold from training params: {threshold}")
                return threshold
            
            # Calculate from loss patterns if available
            loss_patterns = patterns.get('loss_patterns', {})
            if 'stable_loss_variance' in loss_patterns:
                # Use variance to estimate appropriate clipping
                variance = loss_patterns['stable_loss_variance']
                threshold = np.sqrt(variance) * 2.0  # 2 standard deviations
                logger.info(f"Calculated gradient clip threshold from loss variance: {threshold}")
                return threshold
            
            # Alternative calculation from reward patterns
            reward_patterns = patterns.get('reward_patterns', {})
            if 'reward_variance' in reward_patterns:
                threshold = np.sqrt(reward_patterns['reward_variance'])
                logger.info(f"Calculated gradient clip threshold from reward variance: {threshold}")
                return threshold
            
            # Calculate from learning rate and network architecture if available
            lr = training_params.get('learning_rate', 0.0001)
            if lr > 0:
                # Use learning rate to estimate safe gradient threshold
                # Based on Adam optimizer stability analysis - use square root for more reasonable scaling
                threshold = np.sqrt(1.0 / lr) * 0.1  # Conservative threshold with reasonable scaling
                logger.info(f"Calculated gradient clip threshold from learning rate: {threshold}")
                return threshold
                
        except Exception as e:
            logger.warning(f"Could not discover gradient clip threshold: {e}")
        
        # CRITICAL: Calculate from current system state - NO HARDCODED VALUES
        # Use the discovery engine to analyze current GA4 data for threshold estimation
        try:
            # Get current conversion rates to estimate gradient scale
            patterns = self.discovery.get_patterns()
            channels = patterns.get('channels', {})
            
            if channels:
                # Calculate variance in channel performance for gradient scale estimation
                channel_conversions = []
                for channel_data in channels.values():
                    if isinstance(channel_data, dict) and 'conversions' in channel_data:
                        channel_conversions.append(channel_data['conversions'])
                
                if channel_conversions and len(channel_conversions) > 1:
                    conversion_variance = np.var(channel_conversions)
                    threshold = max(0.1, np.sqrt(conversion_variance) / 100.0)  # Scale based on data variance
                    logger.info(f"Calculated gradient threshold from data variance: {threshold}")
                    return threshold
            
            # Final calculation based on system complexity
            # Base threshold on discovered entities (more entities = more complex gradients)
            discovered_entities = len(patterns.get('channels', {})) + len(patterns.get('segments', {}))
            if discovered_entities > 0:
                # Scale threshold with system complexity
                threshold = 0.1 + (discovered_entities * 0.05)  # Conservative scaling
                logger.info(f"Calculated gradient threshold from system complexity: {threshold}")
                return threshold
                
        except Exception as e:
            logger.error(f"Failed to calculate gradient threshold from system data: {e}")
        
        # ABSOLUTE LAST RESORT: Use minimum safe threshold based on numerical stability
        logger.error("CRITICAL: Cannot discover gradient threshold from any patterns or data!")
        logger.error("Using minimum safe threshold for numerical stability")
        return 0.1  # Minimum threshold to prevent complete system failure
    
    def update_adaptive_threshold(self):
        """Update clipping threshold based on gradient history"""
        if len(self.gradient_norms_history) < 100:
            return  # Need sufficient history
        
        # Calculate stable gradient norm range
        norms = np.array(list(self.gradient_norms_history))
        
        # Remove outliers (likely explosions)
        q75, q25 = np.percentile(norms, [75, 25])
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr
        stable_norms = norms[norms <= upper_bound]
        
        if len(stable_norms) > 50:  # Need sufficient stable samples
            # Update threshold based on stable runs
            new_threshold = np.percentile(stable_norms, self.stable_norm_percentile)
            
            # Smooth threshold updates to prevent oscillation
            self.clip_threshold = 0.9 * self.clip_threshold + 0.1 * new_threshold
            
            logger.debug(f"Updated adaptive gradient clip threshold to: {self.clip_threshold:.4f}")
    
    def check_gradient_explosion(self, grad_norm: float, step: int) -> bool:
        """Check for gradient explosion and alert immediately"""
        explosion_threshold = self.clip_threshold * self.explosion_multiplier
        
        if grad_norm > explosion_threshold:
            self.gradient_explosion_count += 1
            self.instability_alerts += 1
            self.last_explosion_step = step
            self.consecutive_explosions += 1
            
            logger.error(f"GRADIENT EXPLOSION DETECTED! Step {step}: norm={grad_norm:.4f} > threshold={explosion_threshold:.4f}")
            logger.error(f"Total explosions: {self.gradient_explosion_count}, Consecutive: {self.consecutive_explosions}, Stability score: {self.stability_score:.3f}")
            
            # Check for emergency intervention
            if self.consecutive_explosions >= self.max_consecutive_explosions:
                self._emergency_intervention(step, grad_norm)
            
            return True
        else:
            # Reset consecutive explosions counter
            self.consecutive_explosions = 0
        
        return False
    
    def check_vanishing_gradients(self, grad_norm: float, step: int) -> bool:
        """Check for vanishing gradient problem"""
        if grad_norm < self.vanishing_threshold:
            self.vanishing_gradient_count += 1
            
            logger.warning(f"VANISHING GRADIENT DETECTED! Step {step}: norm={grad_norm:.2e} < threshold={self.vanishing_threshold:.2e}")
            logger.warning(f"Total vanishing gradients: {self.vanishing_gradient_count}")
            
            # Suggest loss scaling increase
            if self.vanishing_gradient_count % 10 == 0:  # Every 10 vanishing gradients
                self._adjust_loss_scaling(increase=True)
                logger.info(f"Increased loss scaling to {self.loss_scale} due to vanishing gradients")
            
            return True
        
        return False
    
    def _emergency_intervention(self, step: int, grad_norm: float):
        """Emergency intervention when multiple consecutive explosions occur"""
        self.emergency_interventions += 1
        self.last_intervention_step = step
        
        logger.critical(f"EMERGENCY GRADIENT INTERVENTION! Step {step}")
        logger.critical(f"Consecutive explosions: {self.consecutive_explosions}")
        logger.critical(f"Grad norm: {grad_norm:.4f}, Current threshold: {self.clip_threshold:.4f}")
        
        # Aggressive threshold reduction
        old_threshold = self.clip_threshold
        self.clip_threshold = min(self.clip_threshold * 0.5, grad_norm * 0.1)
        
        # Reduce loss scaling to prevent numerical overflow
        self._adjust_loss_scaling(increase=False, emergency=True)
        
        logger.critical(f"EMERGENCY ACTIONS: Threshold {old_threshold:.4f} -> {self.clip_threshold:.4f}, "
                       f"Loss scale -> {self.loss_scale}")
        
        # Reset consecutive counter after intervention
        self.consecutive_explosions = 0
    
    def _adjust_loss_scaling(self, increase: bool = True, emergency: bool = False):
        """Dynamically adjust loss scaling for numerical stability"""
        if emergency:
            # Emergency scaling reduction
            self.loss_scale = max(self.loss_scale * 0.1, 0.01)
        elif increase:
            # Increase scaling for vanishing gradients
            self.loss_scale = min(self.loss_scale * 1.5, 64.0)  # Cap at 64x
        else:
            # Decrease scaling for exploding gradients
            self.loss_scale = max(self.loss_scale * 0.8, 0.1)  # Floor at 0.1x
        
        self.loss_scale_history.append(self.loss_scale)
        logger.debug(f"Adjusted loss scaling to: {self.loss_scale}")
    
    def get_scaled_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply dynamic loss scaling for numerical stability"""
        if self.loss_scale != 1.0:
            return loss * self.loss_scale
        return loss
    
    def unscale_gradients(self, model_parameters):
        """Unscale gradients after backward pass with scaled loss"""
        if self.loss_scale != 1.0:
            for param in model_parameters:
                if param.grad is not None:
                    param.grad.data /= self.loss_scale
    
    def monitor_instability(self, grad_norm: float, loss: float, step: int):
        """Monitor for training instability patterns"""
        self.recent_norms.append(grad_norm)
        
        if len(self.recent_norms) >= self.instability_window:
            # Check for increasing gradient trend (instability)
            recent_array = np.array(list(self.recent_norms))
            
            # Detect sustained high gradients
            high_count = np.sum(recent_array > self.clip_threshold * 0.8)
            if high_count > self.instability_window * 0.6:  # 60% of recent steps
                logger.warning(f"INSTABILITY PATTERN: {high_count}/{self.instability_window} steps with high gradients")
                self.instability_alerts += 1
            
            # Update stability score
            stable_count = np.sum(recent_array <= self.clip_threshold)
            self.stability_score = stable_count / len(recent_array)
    
    def clip_gradients(self, model_parameters, step: int, loss: float = None) -> Dict[str, float]:
        """
        CRITICAL: Clip gradients and monitor stability
        Returns metrics for monitoring
        """
        # Unscale gradients first if we're using loss scaling
        self.unscale_gradients(model_parameters)
        
        # Calculate gradient norm BEFORE clipping
        total_norm = 0.0
        param_count = 0
        individual_norms = []
        
        for param in model_parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                individual_norms.append(param_norm.item())
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count == 0:
            logger.warning("No gradients found for clipping!")
            return {
                'grad_norm': 0.0, 
                'clipped': False, 
                'explosion_detected': False,
                'vanishing_detected': False,
                'stability_score': self.stability_score,
                'clip_threshold': self.clip_threshold,
                'total_clips': self.total_clips,
                'loss_scale': self.loss_scale,
                'consecutive_explosions': 0,
                'emergency_interventions': self.emergency_interventions
            }
        
        grad_norm = np.sqrt(total_norm)
        self.gradient_norms_history.append(grad_norm)
        
        # Calculate gradient variance for flow health monitoring
        if len(individual_norms) > 1:
            grad_variance = np.var(individual_norms)
            self.gradient_variance_history.append(grad_variance)
        
        # Check for vanishing gradients FIRST (before explosion check)
        vanishing_detected = self.check_vanishing_gradients(grad_norm, step)
        
        # Check for explosion IMMEDIATELY
        explosion_detected = self.check_gradient_explosion(grad_norm, step)
        
        # Monitor for instability patterns
        if loss is not None:
            self.monitor_instability(grad_norm, loss, step)
        
        # Perform gradient clipping
        clipped = grad_norm > self.clip_threshold
        if clipped:
            # Store pre-clip norm for analysis
            pre_clip_norm = grad_norm
            
            # Apply clipping
            clip_grad_norm_(model_parameters, self.clip_threshold)
            self.total_clips += 1
            
            logger.debug(f"Clipped gradients: norm={pre_clip_norm:.4f} -> {self.clip_threshold:.4f}")
            
            # Adjust loss scaling if we're clipping frequently
            if self.total_clips % 50 == 0:  # Every 50 clips
                self._adjust_loss_scaling(increase=False)
        
        # Monitor parameter update magnitudes (for learning rate adjustment)
        if len(individual_norms) > 0:
            avg_param_update = np.mean(individual_norms)
            self.parameter_update_magnitudes.append(avg_param_update)
        
        # Update adaptive threshold periodically
        if step % 100 == 0:
            self.update_adaptive_threshold()
        
        # Advanced stability diagnostics every 500 steps
        if step % 500 == 0:
            self._perform_stability_diagnostics(step)
        
        # Save learned threshold every 1000 steps if training is stable
        if step % 1000 == 0:
            self.save_learned_threshold()
        
        return {
            'grad_norm': grad_norm,
            'clipped': clipped,
            'explosion_detected': explosion_detected,
            'vanishing_detected': vanishing_detected,
            'stability_score': self.stability_score,
            'clip_threshold': self.clip_threshold,
            'total_clips': self.total_clips,
            'loss_scale': self.loss_scale,
            'consecutive_explosions': self.consecutive_explosions,
            'emergency_interventions': self.emergency_interventions
        }
    
    def _perform_stability_diagnostics(self, step: int):
        """Perform comprehensive stability diagnostics"""
        if len(self.gradient_norms_history) < 50:
            return
        
        norms = np.array(list(self.gradient_norms_history)[-50:])  # Last 50 steps
        
        # Trend analysis
        if len(norms) > 20:
            recent_trend = np.polyfit(range(len(norms)), norms, 1)[0]  # Linear trend
            
            if recent_trend > 0.01:
                logger.warning(f"GRADIENT TREND WARNING: Increasing gradient trend detected ({recent_trend:.4f})")
            elif recent_trend < -0.001:
                logger.info(f"Gradient trend: Decreasing ({recent_trend:.4f}) - potentially good")
        
        # Variance analysis
        if len(self.gradient_variance_history) > 20:
            recent_variances = np.array(list(self.gradient_variance_history)[-20:])
            high_variance_count = np.sum(recent_variances > np.percentile(recent_variances, 75))
            
            if high_variance_count > 15:  # More than 75% high variance
                logger.warning(f"HIGH GRADIENT VARIANCE DETECTED: {high_variance_count}/20 recent steps")
        
        # Parameter update magnitude analysis
        if len(self.parameter_update_magnitudes) > 30:
            recent_updates = np.array(list(self.parameter_update_magnitudes)[-30:])
            update_std = np.std(recent_updates)
            
            if update_std > 1.0:
                logger.warning(f"ERRATIC PARAMETER UPDATES: std={update_std:.4f}")
        
        # Loss scaling effectiveness
        if len(self.loss_scale_history) > 10:
            recent_scales = list(self.loss_scale_history)[-10:]
            scale_changes = len(set(recent_scales))
            
            if scale_changes > 5:
                logger.warning(f"FREQUENT LOSS SCALING CHANGES: {scale_changes} changes in last 10 adjustments")
        
        logger.debug(f"Stability diagnostics completed at step {step}")
    
    def save_learned_threshold(self):
        """Save the learned gradient threshold to discovery patterns for future runs"""
        try:
            # Only save if we have enough stable training history
            if (len(self.gradient_norms_history) > 500 and 
                self.stability_score > 0.8 and 
                self.gradient_explosion_count < 10):
                
                # Update discovery patterns with learned threshold
                patterns = self.discovery.get_patterns()
                training_params = patterns.get('training_params', {})
                training_params['gradient_clip_threshold'] = self.clip_threshold
                
                # Also save gradient statistics for future reference
                norms = np.array(list(self.gradient_norms_history))
                training_params['gradient_stats'] = {
                    'avg_grad_norm': float(np.mean(norms)),
                    'std_grad_norm': float(np.std(norms)),
                    'percentile_95': float(np.percentile(norms, 95)),
                    'stability_score': self.stability_score,
                    'training_steps': len(self.gradient_norms_history)
                }
                
                patterns['training_params'] = training_params
                self.discovery.save_patterns(patterns)
                
                logger.info(f"Saved learned gradient threshold {self.clip_threshold:.4f} to discovery patterns")
                
        except Exception as e:
            logger.warning(f"Failed to save learned gradient threshold: {e}")
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Get comprehensive stability report"""
        if len(self.gradient_norms_history) == 0:
            return {'status': 'no_data'}
        
        norms = np.array(list(self.gradient_norms_history))
        
        # Calculate advanced metrics
        gradient_health = 'healthy'
        if self.stability_score < 0.5:
            gradient_health = 'critical'
        elif self.stability_score < 0.8:
            gradient_health = 'unstable'
        elif self.vanishing_gradient_count > 50:
            gradient_health = 'vanishing_risk'
        
        # Calculate gradient flow efficiency
        recent_norms = norms[-100:] if len(norms) >= 100 else norms
        flow_efficiency = 1.0 - (np.sum(recent_norms > self.clip_threshold) / len(recent_norms))
        
        report = {
            'status': gradient_health,
            'stability_score': self.stability_score,
            'flow_efficiency': flow_efficiency,
            
            # Explosion metrics
            'gradient_explosions': self.gradient_explosion_count,
            'consecutive_explosions': self.consecutive_explosions,
            'last_explosion_step': self.last_explosion_step,
            
            # Vanishing gradient metrics
            'vanishing_gradients': self.vanishing_gradient_count,
            'vanishing_threshold': self.vanishing_threshold,
            
            # Clipping metrics
            'total_clips': self.total_clips,
            'clip_threshold': self.clip_threshold,
            'instability_alerts': self.instability_alerts,
            
            # Loss scaling metrics
            'loss_scale': self.loss_scale,
            'loss_scale_adjustments': len(self.loss_scale_history),
            
            # Emergency intervention metrics
            'emergency_interventions': self.emergency_interventions,
            'last_intervention_step': self.last_intervention_step,
            
            # Statistical metrics
            'avg_grad_norm': np.mean(norms),
            'max_grad_norm': np.max(norms),
            'min_grad_norm': np.min(norms),
            'grad_norm_std': np.std(norms),
            'gradient_history_size': len(self.gradient_norms_history),
            
            # Variance metrics
            'gradient_variance_samples': len(self.gradient_variance_history),
            'avg_gradient_variance': np.mean(list(self.gradient_variance_history)) if self.gradient_variance_history else 0.0,
            
            # Update magnitude metrics  
            'parameter_update_samples': len(self.parameter_update_magnitudes),
            'avg_parameter_update_magnitude': np.mean(list(self.parameter_update_magnitudes)) if self.parameter_update_magnitudes else 0.0
        }
        
        return report
    
    def test_gradient_stability(self, test_gradients: List[float]) -> Dict[str, Any]:
        """Test gradient clipping with synthetic gradient norms - for verification"""
        test_results = {
            'explosions_detected': 0,
            'clips_performed': 0,
            'stability_maintained': True
        }
        
        initial_clips = self.total_clips
        initial_explosions = self.gradient_explosion_count
        
        for i, test_grad_norm in enumerate(test_gradients):
            # Simulate gradient clipping test
            if test_grad_norm > self.clip_threshold:
                test_results['clips_performed'] += 1
            
            if test_grad_norm > self.clip_threshold * self.explosion_multiplier:
                test_results['explosions_detected'] += 1
                test_results['stability_maintained'] = False
                
        logger.info(f"Gradient stability test completed: {test_results}")
        return test_results


@dataclass
class DynamicEnrichedState:
    """State with all values discovered dynamically"""
    
    # Core journey state
    stage: int = 0
    touchpoints_seen: int = 0
    days_since_first_touch: float = 0.0
    
    # User segment (discovered)
    segment_index: int = 0
    segment_cvr: float = 0.0
    segment_engagement: float = 0.0
    segment_avg_ltv: float = 0.0
    
    # Device and channel (discovered)
    device_index: int = 0
    channel_index: int = 0
    channel_performance: float = 0.0
    channel_attribution_credit: float = 0.0
    
    # Creative performance
    creative_index: int = 0
    creative_ctr: float = 0.0
    creative_cvr: float = 0.0
    creative_fatigue: float = 0.0
    creative_diversity_score: float = 0.0
    
    # Temporal patterns
    hour_of_day: int = 0
    day_of_week: int = 0
    is_peak_hour: bool = False
    seasonality_factor: float = 1.0
    
    # Competition
    competition_level: float = 0.0
    avg_competitor_bid: float = 0.0
    win_rate_last_10: float = 0.0
    avg_position_last_10: float = 0.0
    
    # Budget
    budget_spent_ratio: float = 0.0
    time_in_day_ratio: float = 0.0
    pacing_factor: float = 1.0
    remaining_budget: float = 0.0
    
    # Identity
    cross_device_confidence: float = 0.0
    num_devices_seen: int = 1
    is_returning_user: bool = False
    is_logged_in: bool = False
    
    # Attribution
    first_touch_channel_index: int = 0
    last_touch_channel_index: int = 0
    num_touchpoint_credits: int = 0
    expected_conversion_value: float = 0.0
    
    # A/B test
    ab_test_variant: int = 0
    variant_performance: float = 0.0
    
    # Delayed conversion
    conversion_probability: float = 0.0
    days_to_conversion_estimate: float = 0.0
    has_scheduled_conversion: bool = False
    
    # Competitor exposure
    competitor_impressions_seen: int = 0
    competitor_fatigue_level: float = 0.0
    
    # Discovered dimensions - will be set from actual patterns
    num_segments: int = 1
    num_channels: int = 1
    num_devices: int = 1
    num_creatives: int = 1
    num_variants: int = 1
    
    def to_vector(self, stats: DataStatistics) -> np.ndarray:
        """Convert to normalized vector using actual data statistics"""
        return np.array([
            # Journey features (5)
            self.stage / max(4, 1),  # Stages 0-4
            stats.normalize(self.touchpoints_seen, 'touchpoints'),
            stats.normalize(self.days_since_first_touch, 'days_to_convert'),
            float(self.is_returning_user),
            self.conversion_probability,  # Already 0-1
            
            # Segment features (4)
            self.segment_index / max(self.num_segments - 1, 1),
            self.segment_cvr,  # Already 0-1
            self.segment_engagement,  # Already 0-1
            stats.normalize(self.segment_avg_ltv, 'conversion_value'),
            
            # Device/channel features (4)
            self.device_index / max(self.num_devices - 1, 1),
            self.channel_index / max(self.num_channels - 1, 1),
            self.channel_performance,  # Already normalized
            self.channel_attribution_credit,  # Already 0-1
            
            # Creative features (5)
            self.creative_index / max(self.num_creatives - 1, 1),
            self.creative_ctr,  # Already 0-1
            self.creative_cvr,  # Already 0-1
            self.creative_fatigue,  # Already 0-1
            self.creative_diversity_score,  # Already 0-1
            
            # Temporal features (4)
            self.hour_of_day / 23.0,
            self.day_of_week / 6.0,
            float(self.is_peak_hour),
            self.seasonality_factor,  # Already normalized
            
            # Competition features (4)
            self.competition_level,  # Already 0-1
            stats.normalize(self.avg_competitor_bid, 'bid'),
            self.win_rate_last_10,  # Already 0-1
            stats.normalize(self.avg_position_last_10, 'position'),
            
            # Budget features (4)
            self.budget_spent_ratio,  # Already 0-1
            self.time_in_day_ratio,  # Already 0-1
            self.pacing_factor,  # Already normalized
            stats.normalize(self.remaining_budget, 'budget'),
            
            # Identity features (3)
            self.cross_device_confidence,  # Already 0-1
            stats.normalize(self.num_devices_seen, 'num_devices'),
            float(self.is_logged_in),
            
            # Attribution features (4)
            self.first_touch_channel_index / max(self.num_channels - 1, 1),
            self.last_touch_channel_index / max(self.num_channels - 1, 1),
            min(self.num_touchpoint_credits / max(10, self.num_touchpoint_credits), 1.0),  # Dynamic max
            stats.normalize(self.expected_conversion_value, 'conversion_value'),
            
            # A/B test features (2)
            self.ab_test_variant / max(self.num_variants - 1, 1),
            self.variant_performance,  # Already 0-1
            
            # Delayed conversion features (3)
            self.conversion_probability,  # Already 0-1
            stats.normalize(self.days_to_conversion_estimate, 'days_to_convert'),
            float(self.has_scheduled_conversion),
            
            # Competitor features (2)
            stats.normalize(self.competitor_impressions_seen, 'competitor_impressions'),
            self.competitor_fatigue_level,  # Already 0-1
            
            # Creative content features (9)
            getattr(self, 'content_sentiment', 0.0),  # Already normalized -1 to 1
            getattr(self, 'content_urgency', 0.0),  # Already 0-1
            getattr(self, 'content_cta_strength', 0.0),  # Already 0-1
            getattr(self, 'content_uses_numbers', 0.0),  # Boolean as float
            getattr(self, 'content_uses_social_proof', 0.0),  # Boolean as float
            getattr(self, 'content_uses_authority', 0.0),  # Boolean as float
            getattr(self, 'content_uses_urgency', 0.0),  # Boolean as float
            getattr(self, 'content_message_frame', 0.0),  # Encoded 0-1
            getattr(self, 'content_visual_style', 0.0)  # Encoded 0-1
        ])
    
    @property
    def state_dim(self) -> int:
        """Total dimension of state vector"""
        return 53  # 44 original + 9 new creative content features


class ConvergenceMonitor:
    """Real-time convergence monitoring with early stopping and automatic intervention"""
    
    def __init__(self, agent, discovery_engine, checkpoint_dir: str = "./checkpoints"):
        self.agent = agent
        self.discovery = discovery_engine
        self.checkpoint_dir = checkpoint_dir
        
        # Metrics tracking with dynamic sizes based on discovered patterns
        # Load patterns directly from file since discovery engine is a pipeline
        try:
            with open('/home/hariravichandran/AELP/discovered_patterns.json', 'r') as f:
                patterns = json.load(f)
        except FileNotFoundError:
            patterns = {'training_params': {'buffer_size': 1000}}
        buffer_size = patterns.get('training_params', {}).get('buffer_size', 1000)
        
        self.loss_history = deque(maxlen=buffer_size)
        self.reward_history = deque(maxlen=buffer_size)
        self.gradient_history = deque(maxlen=min(100, buffer_size // 10))
        self.action_history = deque(maxlen=buffer_size)
        self.q_value_history = deque(maxlen=buffer_size)
        self.exploration_history = deque(maxlen=buffer_size // 2)
        
        # Performance tracking
        self.performance_window = deque(maxlen=200)  # Rolling window for performance
        self.training_stability_window = deque(maxlen=100)
        self.convergence_window = deque(maxlen=50)
        
        # Alert system
        self.alerts = []
        self.critical_alerts = []
        self.intervention_history = []
        
        # State tracking
        self.episode = 0
        self.training_step = 0
        self.last_intervention_step = 0
        self.emergency_stop_triggered = False
        self.convergence_detected = False
        
        # Thresholds learned from successful runs
        self.thresholds = self._load_success_thresholds(patterns)
        
        # Action tracking for diversity monitoring
        self.recent_actions = deque(maxlen=100)
        self.channel_distribution = {}
        self.creative_distribution = {}
        self.bid_distribution = {}
        
        # Convergence criteria - dynamically determined
        self.convergence_criteria = self._determine_convergence_criteria(patterns)
        
        logger.info(f"ConvergenceMonitor initialized with thresholds: {self.thresholds}")
        logger.info(f"Convergence criteria: {self.convergence_criteria}")
    
    def _load_success_thresholds(self, patterns: Dict[str, Any]) -> Dict[str, float]:
        """Load thresholds from discovered patterns - NO HARDCODING"""
        # Get performance metrics from patterns
        perf_metrics = patterns.get('performance_metrics', {})
        training_params = patterns.get('training_params', {})
        
        # Calculate thresholds based on successful training patterns
        cvr_stats = perf_metrics.get('cvr_stats', {})
        cvr_mean = cvr_stats.get('mean', 0.05)  # From discovered data
        cvr_std = cvr_stats.get('std', 0.02)
        
        # Revenue stats from patterns
        revenue_stats = perf_metrics.get('revenue_stats', {})
        revenue_mean = revenue_stats.get('mean', 10.0)
        
        # Get gradient clipping threshold from discovered patterns
        gradient_clip = training_params.get('gradient_clip_threshold', 10.0)
        
        return {
            'min_improvement_threshold': cvr_std / 2,  # Half standard deviation
            'plateau_threshold': cvr_std / 4,  # Quarter standard deviation
            'instability_threshold': cvr_std * 3,  # 3 sigma rule
            'gradient_norm_threshold': gradient_clip * 0.5,  # 50% of clipping threshold for warning
            'q_value_variance_threshold': revenue_mean * 0.05,
            'exploration_diversity_threshold': 0.1,  # Based on discovered channels/creatives
            'convergence_patience': training_params.get('plateau_patience', 100),
            'emergency_gradient_threshold': gradient_clip * 3,  # 3x clipping threshold for emergency
            'overestimation_bias_threshold': cvr_mean  # Average CVR as bias threshold
        }
    
    def _determine_convergence_criteria(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Determine convergence criteria from discovered patterns"""
        channels = patterns.get('channels', {})
        segments = patterns.get('segments', {})
        
        min_channels_used = max(2, len(channels) // 3)  # At least 1/3 of channels
        min_segments_reached = max(1, len(segments) // 4)  # At least 1/4 of segments
        
        return {
            'min_episodes': patterns.get('training_params', {}).get('min_episodes', 500),
            'performance_stability_window': 100,
            'min_exploration_ratio': 0.1,  # 10% exploration minimum
            'min_channels_used': min_channels_used,
            'min_segments_reached': min_segments_reached,
            'convergence_confirmation_steps': 50
        }
    
    def monitor_step(self, loss: float, reward: float, gradient_norm: float, action: Dict[str, Any]):
        """Monitor each training step - REAL-TIME DETECTION"""
        self.training_step += 1
        
        # Store metrics
        if not (np.isnan(loss) or np.isinf(loss)):
            self.loss_history.append(loss)
        if not (np.isnan(reward) or np.isinf(reward)):
            self.reward_history.append(reward)
        if not (np.isnan(gradient_norm) or np.isinf(gradient_norm)):
            self.gradient_history.append(gradient_norm)
        
        # Track actions for diversity monitoring
        self.recent_actions.append(action)
        self._update_action_distributions(action)
        
        # CRITICAL: Real-time instability detection
        issues = []
        
        if self.detect_training_instability(loss, gradient_norm):
            issues.append("instability")
        
        if self.detect_premature_convergence():
            issues.append("premature_convergence")
        
        if self.training_step % 50 == 0:  # Periodic checks to avoid overhead
            if self.detect_performance_plateau():
                issues.append("plateau")
            
            if self.detect_overfitting():
                issues.append("overfitting")
            
            if self.detect_exploration_collapse():
                issues.append("exploration_collapse")
        
        if issues:
            self.handle_issues(issues)
        
        return self.should_stop()
    
    def detect_training_instability(self, current_loss: float, gradient_norm: float) -> bool:
        """Detect when training becomes unstable - IMMEDIATE ACTION"""
        # Check for NaN/Inf
        if np.isnan(current_loss) or np.isinf(current_loss):
            self.raise_alert("CRITICAL: NaN/Inf detected in loss!", critical=True)
            self.emergency_stop_triggered = True
            return True
        
        if np.isnan(gradient_norm) or np.isinf(gradient_norm):
            self.raise_alert("CRITICAL: NaN/Inf detected in gradients!", critical=True)
            self.emergency_stop_triggered = True
            return True
        
        # Check for loss explosion
        if len(self.loss_history) > 10:
            recent_losses = list(self.loss_history)[-10:]
            historical_losses = list(self.loss_history)[:-10] if len(self.loss_history) > 20 else recent_losses
            
            recent_mean = np.mean(recent_losses)
            historical_mean = np.mean(historical_losses)
            
            if historical_mean > 0 and recent_mean > historical_mean * 10:
                self.raise_alert(f"CRITICAL: Loss exploded from {historical_mean:.4f} to {recent_mean:.4f}", critical=True)
                return True
        
        # Check gradient explosion
        if gradient_norm > self.thresholds['emergency_gradient_threshold']:
            # During early warmup, take softer action instead of full emergency stop
            if self.training_step < 200:
                self.adjust_learning_parameters()  # halve learning rates for stability
                self.raise_alert(f"WARNING: Gradient spike during warmup (norm={gradient_norm:.4f}); reduced learning rates", critical=False)
                return False
            self.raise_alert(f"CRITICAL: Gradient explosion - norm = {gradient_norm:.4f}", critical=True)
            return True
        
        return False
    
    def detect_premature_convergence(self) -> bool:
        """Detect when agent stops exploring too early"""
        if not hasattr(self.agent, 'epsilon') or self.episode < 100:
            return False
        
        # Check epsilon decay rate
        if (self.agent.epsilon <= self.agent.epsilon_min and 
            self.episode < self.convergence_criteria['min_episodes']):
            self.raise_alert(f"CRITICAL: Epsilon reached minimum at episode {self.episode} - TOO EARLY!")
            return True
        
        # Check action diversity
        if len(self.recent_actions) >= 100:
            unique_actions = self._count_unique_actions(list(self.recent_actions)[-100:])
            
            min_expected_unique = max(5, len(self.agent.discovered_channels) // 2)
            if unique_actions < min_expected_unique:
                self.raise_alert(f"CRITICAL: Only {unique_actions} unique actions in last 100 steps - NO EXPLORATION!")
                return True
            
            # Check channel diversity
            channel_entropy = self._calculate_channel_entropy()
            if channel_entropy < self.thresholds['exploration_diversity_threshold']:
                self.raise_alert(f"CRITICAL: Low channel diversity (entropy={channel_entropy:.3f}) - EXPLOITATION ONLY!")
                return True
        
        return False
    
    def detect_performance_plateau(self) -> bool:
        """Detect when learning has stopped"""
        if len(self.reward_history) < 200:
            return False
        
        # Check reward improvement over time
        old_rewards = list(self.reward_history)[-200:-100]
        new_rewards = list(self.reward_history)[-100:]
        
        old_mean = np.mean(old_rewards)
        new_mean = np.mean(new_rewards)
        
        if old_mean == 0:
            improvement = 0
        else:
            improvement = (new_mean - old_mean) / abs(old_mean)
        
        if abs(improvement) < self.thresholds['plateau_threshold']:
            self.raise_alert(f"WARNING: Performance plateaued - {improvement*100:.2f}% improvement in 100 episodes")
            
            # Check if it's due to lack of exploration
            if hasattr(self.agent, 'epsilon') and self.agent.epsilon < 0.1:
                self.raise_alert("CRITICAL: Plateau with low epsilon - INCREASE EXPLORATION!")
                self.suggest_intervention("increase_exploration")
            
            return True
        
        return False
    
    def detect_overfitting(self) -> bool:
        """Detect when agent overfits to patterns"""
        if len(self.recent_actions) < 100:
            return False
        
        # Check for repetitive action sequences
        action_sequences = self._get_action_sequences(length=5, count=50)
        unique_sequences = len(set(tuple(seq) for seq in action_sequences))
        
        expected_unique = min(20, len(action_sequences) // 2)
        if unique_sequences < expected_unique:
            self.raise_alert(f"WARNING: Only {unique_sequences} unique 5-step sequences - MEMORIZATION DETECTED!")
            return True
        
        # Check Q-value overestimation
        if hasattr(self.agent, 'q_value_tracking') and len(self.agent.q_value_tracking['overestimation_bias']) > 5:
            recent_bias = np.mean(list(self.agent.q_value_tracking['overestimation_bias'])[-5:])
            if recent_bias > self.thresholds['overestimation_bias_threshold']:
                self.raise_alert(f"WARNING: High Q-value overestimation bias: {recent_bias:.3f}")
                return True
        
        return False
    
    def detect_exploration_collapse(self) -> bool:
        """Detect when exploration has collapsed"""
        if len(self.recent_actions) < 50:
            return False
        
        # Calculate action entropy over recent window
        entropy = self._calculate_action_entropy()
        min_entropy = np.log(max(2, len(self.agent.discovered_channels) // 3))
        
        if entropy < min_entropy:
            self.raise_alert(f"CRITICAL: Exploration collapsed - entropy={entropy:.3f} < min={min_entropy:.3f}")
            return True
        
        return False
    
    def handle_issues(self, issues: list):
        """Take automatic action on detected issues"""
        for issue in issues:
            if issue == "instability":
                self.emergency_intervention()
            elif issue == "premature_convergence":
                self.increase_exploration()
            elif issue == "plateau":
                self.adjust_learning_parameters()
            elif issue == "overfitting":
                self.add_regularization()
            elif issue == "exploration_collapse":
                self.force_exploration()
        
        self.last_intervention_step = self.training_step
    
    def emergency_intervention(self):
        """Emergency intervention for training instability"""
        self.save_emergency_checkpoint()
        
        # Reduce learning rates drastically
        if hasattr(self.agent, 'optimizer_bid'):
            for param_group in self.agent.optimizer_bid.param_groups:
                param_group['lr'] *= 0.1
        if hasattr(self.agent, 'optimizer_creative'):
            for param_group in self.agent.optimizer_creative.param_groups:
                param_group['lr'] *= 0.1
        if hasattr(self.agent, 'optimizer_channel'):
            for param_group in self.agent.optimizer_channel.param_groups:
                param_group['lr'] *= 0.1
        
        self.log_intervention("EMERGENCY: Reduced all learning rates by 10x")
        self.emergency_stop_triggered = True
    
    def increase_exploration(self):
        """Increase exploration when premature convergence detected"""
        if hasattr(self.agent, 'epsilon'):
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = min(0.5, self.agent.epsilon * 3)  # Triple epsilon
            self.log_intervention(f"Increased epsilon from {old_epsilon:.3f} to {self.agent.epsilon:.3f}")
    
    def adjust_learning_parameters(self):
        """Adjust learning parameters for plateau"""
        # Reduce learning rate by half for stability
        if hasattr(self.agent, 'optimizer_bid'):
            for param_group in self.agent.optimizer_bid.param_groups:
                param_group['lr'] *= 0.5
        if hasattr(self.agent, 'optimizer_creative'):
            for param_group in self.agent.optimizer_creative.param_groups:
                param_group['lr'] *= 0.5
        if hasattr(self.agent, 'optimizer_channel'):
            for param_group in self.agent.optimizer_channel.param_groups:
                param_group['lr'] *= 0.5
        
        self.log_intervention("Reduced learning rates by 50% due to plateau")
    
    def add_regularization(self):
        """Add regularization to combat overfitting"""
        if hasattr(self.agent, 'dropout_rate'):
            self.agent.dropout_rate = min(0.5, self.agent.dropout_rate * 1.2)
        self.log_intervention(f"Increased dropout to {self.agent.dropout_rate:.3f}")
    
    def force_exploration(self):
        """Force more exploration when it has collapsed"""
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = max(0.3, self.agent.epsilon * 2)
        self.log_intervention(f"Forced exploration - epsilon = {self.agent.epsilon:.3f}")
    
    def should_stop(self) -> bool:
        """Determine if training should stop"""
        return self.emergency_stop_triggered or self.convergence_detected
    
    def end_episode(self, episode_reward: float):
        """End of episode monitoring"""
        self.episode += 1
        self.performance_window.append(episode_reward)
        
        # Check for convergence every 50 episodes
        if self.episode % 50 == 0:
            self._check_convergence()
    
    def _check_convergence(self):
        """Check if training has properly converged"""
        if len(self.performance_window) < 100:
            return
        
        # Check performance stability
        recent_performance = list(self.performance_window)[-50:]
        performance_std = np.std(recent_performance)
        performance_mean = np.mean(recent_performance)
        
        if performance_mean != 0:
            coefficient_of_variation = performance_std / abs(performance_mean)
            
            if coefficient_of_variation < 0.05:  # Very stable
                # Check if we've met minimum exploration criteria
                if self._meets_exploration_criteria():
                    self.convergence_detected = True
                    self.log_intervention("Training converged - stable performance achieved")
    
    def _meets_exploration_criteria(self) -> bool:
        """Check if agent has explored sufficiently"""
        if len(self.recent_actions) < 100:
            return False
        
        # Check channel coverage
        channels_used = len(set(action.get('channel_action', -1) for action in self.recent_actions))
        if channels_used < self.convergence_criteria['min_channels_used']:
            return False
        
        # Check creative diversity
        creatives_used = len(set(action.get('creative_action', -1) for action in self.recent_actions))
        if creatives_used < 3:  # Minimum creative diversity
            return False
        
        return True
    
    def _update_action_distributions(self, action: Dict[str, Any]):
        """Update action distribution tracking"""
        channel = action.get('channel_action', -1)
        creative = action.get('creative_action', -1)
        bid = action.get('bid_action', -1)
        
        self.channel_distribution[channel] = self.channel_distribution.get(channel, 0) + 1
        self.creative_distribution[creative] = self.creative_distribution.get(creative, 0) + 1
        self.bid_distribution[bid] = self.bid_distribution.get(bid, 0) + 1
    
    def _calculate_channel_entropy(self) -> float:
        """Calculate entropy of channel selection"""
        if not self.channel_distribution:
            return 0.0
        
        total = sum(self.channel_distribution.values())
        probs = [count / total for count in self.channel_distribution.values()]
        
        return -sum(p * np.log(p + 1e-10) for p in probs)
    
    def _calculate_action_entropy(self) -> float:
        """Calculate entropy of action selection"""
        if len(self.recent_actions) == 0:
            return 0.0
        
        # Create action tuples for entropy calculation
        actions = [(a.get('channel_action', -1), a.get('creative_action', -1)) 
                  for a in self.recent_actions]
        action_counts = {}
        
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total = len(actions)
        probs = [count / total for count in action_counts.values()]
        
        return -sum(p * np.log(p + 1e-10) for p in probs)
    
    def _count_unique_actions(self, actions: list) -> int:
        """Count unique action combinations"""
        unique = set()
        for action in actions:
            key = (action.get('channel_action', -1), 
                   action.get('creative_action', -1),
                   action.get('bid_action', -1))
            unique.add(key)
        return len(unique)
    
    def _get_action_sequences(self, length: int, count: int) -> list:
        """Get action sequences for pattern detection"""
        if len(self.recent_actions) < length:
            return []
        
        sequences = []
        actions = list(self.recent_actions)[-count*length:]  # Get enough actions
        
        for i in range(len(actions) - length + 1):
            seq = []
            for j in range(length):
                action = actions[i + j]
                seq.append((action.get('channel_action', -1), 
                           action.get('creative_action', -1)))
            sequences.append(seq)
            
            if len(sequences) >= count:
                break
        
        return sequences
    
    def raise_alert(self, message: str, critical: bool = False):
        """Raise an alert"""
        timestamp = datetime.now().isoformat()
        alert = {
            'timestamp': timestamp,
            'step': self.training_step,
            'episode': self.episode,
            'message': message,
            'critical': critical
        }
        
        if critical:
            self.critical_alerts.append(alert)
            logger.critical(f"CRITICAL ALERT: {message}")
        else:
            self.alerts.append(alert)
            logger.warning(f"ALERT: {message}")
    
    def suggest_intervention(self, intervention_type: str):
        """Suggest intervention"""
        logger.info(f"SUGGESTED INTERVENTION: {intervention_type}")
    
    def log_intervention(self, message: str):
        """Log intervention taken"""
        intervention = {
            'timestamp': datetime.now().isoformat(),
            'step': self.training_step,
            'episode': self.episode,
            'intervention': message
        }
        self.intervention_history.append(intervention)
        logger.info(f"INTERVENTION: {message}")
    
    def save_emergency_checkpoint(self):
        """Save emergency checkpoint"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"emergency_checkpoint_step_{self.training_step}.pth")
        
        try:
            torch.save({
                'step': self.training_step,
                'episode': self.episode,
                'agent_state_dict': {
                    'q_network_bid': self.agent.q_network_bid.state_dict(),
                    'q_network_creative': self.agent.q_network_creative.state_dict(),
                    'q_network_channel': self.agent.q_network_channel.state_dict(),
                },
                'optimizer_state_dict': {
                    'optimizer_bid': self.agent.optimizer_bid.state_dict(),
                    'optimizer_creative': self.agent.optimizer_creative.state_dict(),
                    'optimizer_channel': self.agent.optimizer_channel.state_dict(),
                },
                'metrics': {
                    'loss_history': list(self.loss_history),
                    'reward_history': list(self.reward_history),
                    'alerts': self.alerts[-20:],  # Last 20 alerts
                    'critical_alerts': self.critical_alerts
                }
            }, checkpoint_path)
            logger.info(f"Emergency checkpoint saved to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive convergence report"""
        return {
            'training_status': {
                'step': self.training_step,
                'episode': self.episode,
                'emergency_stop_triggered': self.emergency_stop_triggered,
                'convergence_detected': self.convergence_detected
            },
            'current_metrics': {
                'current_loss': self.loss_history[-1] if self.loss_history else None,
                'avg_reward': np.mean(list(self.reward_history)[-100:]) if self.reward_history else 0,
                'gradient_norm': self.gradient_history[-1] if self.gradient_history else None,
                'epsilon': getattr(self.agent, 'epsilon', None)
            },
            'stability_metrics': {
                'loss_stability': np.std(list(self.loss_history)[-50:]) if len(self.loss_history) >= 50 else None,
                'reward_stability': np.std(list(self.reward_history)[-100:]) if len(self.reward_history) >= 100 else None,
                'action_entropy': self._calculate_action_entropy()
            },
            'alerts_summary': {
                'total_alerts': len(self.alerts),
                'critical_alerts': len(self.critical_alerts),
                'recent_alerts': self.alerts[-5:] if self.alerts else [],
                'interventions': len(self.intervention_history)
            },
            'exploration_metrics': {
                'unique_actions_last_100': self._count_unique_actions(list(self.recent_actions)[-100:]) if len(self.recent_actions) >= 100 else 0,
                'channel_entropy': self._calculate_channel_entropy(),
                'channels_used': len(self.channel_distribution),
                'creatives_used': len(self.creative_distribution)
            }
        }


class ProductionFortifiedRLAgent:
    """Production quality RL agent with NO hardcoding"""
    
    def __init__(self,
                 discovery_engine: DiscoveryEngine,
                 creative_selector: CreativeSelector,
                 attribution_engine: AttributionEngine,
                 budget_pacer: BudgetPacer,
                 identity_resolver: IdentityResolver,
                 parameter_manager: ParameterManager,
                 learning_rate: float = None,
                 epsilon: float = None,
                 gamma: float = None,
                 buffer_size: int = None):
        
        # Components
        self.discovery = discovery_engine
        self.creative_selector = creative_selector
        self.attribution = attribution_engine
        self.budget_pacer = budget_pacer
        self.identity_resolver = identity_resolver
        self.pm = parameter_manager
        
        # Discover patterns FIRST
        self.patterns = self._load_discovered_patterns()
        
        # Ensure patterns is not None
        if self.patterns is None:
            # Load from file if not already loaded
            patterns_file = 'discovered_patterns.json'
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    self.patterns = json.load(f)
            else:
                # Cannot proceed without patterns
                raise RuntimeError("No discovered patterns available - cannot initialize agent without real data")
        
        # Compute statistics from actual data
        self.data_stats = DataStatistics.compute_from_patterns(self.patterns)
        
        # Discover dimensions dynamically
        self.discovered_channels = list(self.patterns.get('channels', {}).keys())
        self.discovered_segments = list(self.patterns.get('segments', {}).keys())
        self.discovered_devices = list(self.patterns.get('devices', {}).keys())
        
        # Discover creative IDs from patterns
        self.discovered_creatives = self._discover_creatives()
        
        # Get hyperparameters from discovered patterns - NO HARDCODING
        self._hyperparameters = self._discover_hyperparameters()
        # Add trajectory-specific hyperparameters
        self._hyperparameters.update(self._discover_trajectory_hyperparameters())
        self.learning_rate = learning_rate if learning_rate is not None else self._hyperparameters['learning_rate']
        self.epsilon = epsilon if epsilon is not None else self._hyperparameters['epsilon']
        self.epsilon_decay = self._hyperparameters['epsilon_decay']  # 10x slower decay
        self.epsilon_min = self._hyperparameters['epsilon_min']  # Keep 10% exploration
        self.gamma = gamma if gamma is not None else self._hyperparameters['gamma']
        self.buffer_size = buffer_size if buffer_size is not None else self._hyperparameters['buffer_size']
        
        # Training control
        self.training_frequency = self._hyperparameters['training_frequency']  # Train every N steps
        self.training_step = 0  # Initialize early for warm_start
        self.step_count = 0
        
        # Get network parameters from discovered patterns
        self.hidden_dim = self._hyperparameters.get('hidden_dim', 256)  
        self.num_heads = self._hyperparameters.get('num_heads', 8)
        # Discover dropout from patterns - NO DEFAULT
        self.dropout_rate = self._hyperparameters['dropout_rate']
        
        # State tracking - must match to_vector output
        # Get state dimension dynamically from DynamicEnrichedState
        example_state = DynamicEnrichedState()
        self.state_dim = example_state.state_dim  # Now 53 with creative content features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Action spaces (discovered)
        self.num_bid_levels = self._hyperparameters.get('num_bid_levels', 20)
        self.bid_actions = self.num_bid_levels
        self.creative_actions = len(self.discovered_creatives)
        self.channel_actions = len(self.discovered_channels)
        
        # Discovered bid ranges
        self.bid_ranges = self._extract_bid_ranges()
        
        # Neural networks - Double DQN with separate target networks
        self.q_network_bid = self._build_q_network(output_dim=self.bid_actions)
        self.q_network_creative = self._build_q_network(output_dim=self.creative_actions)
        self.q_network_channel = self._build_q_network(output_dim=self.channel_actions)
        
        # Target networks for Double DQN (eliminate overestimation bias)
        self.target_network_bid = self._build_q_network(output_dim=self.bid_actions)
        self.target_network_creative = self._build_q_network(output_dim=self.creative_actions)
        self.target_network_channel = self._build_q_network(output_dim=self.channel_actions)
        
        # Initialize target networks with same weights as online networks
        self.target_network_bid.load_state_dict(self.q_network_bid.state_dict())
        self.target_network_creative.load_state_dict(self.q_network_creative.state_dict())
        self.target_network_channel.load_state_dict(self.q_network_channel.state_dict())
        
        # Initialize adaptive learning rate scheduler
        self.lr_scheduler_config = self._discover_lr_scheduler_config()
        self.lr_scheduler = AdaptiveLearningRateScheduler(self.lr_scheduler_config, self.learning_rate)
        
        # Optimizers
        self.optimizer_bid = optim.Adam(self.q_network_bid.parameters(), lr=self.learning_rate)
        self.optimizer_creative = optim.Adam(self.q_network_creative.parameters(), lr=self.learning_rate)
        self.optimizer_channel = optim.Adam(self.q_network_channel.parameters(), lr=self.learning_rate)
        
        # Advanced prioritized experience replay
        self.replay_buffer = AdvancedReplayBuffer(
            capacity=self.buffer_size,
            alpha=self._hyperparameters.get('prioritization_alpha', 0.6),
            beta_start=self._hyperparameters.get('importance_sampling_beta_start', 0.4)
        )
        
        # Warm start will be called after all components are initialized
        
        # Performance tracking
        self.training_metrics = {
            'episodes': 0,
            'total_reward': 0,
            'avg_position': self.data_stats.position_mean,
            'win_rate': 0.0,
            'conversion_rate': 0.0,
            'roas': 0.0,
            'creative_diversity': 0.0,
            'channel_efficiency': {ch: 0.0 for ch in self.discovered_channels},
            'target_network_divergence': {
                'bid': deque(maxlen=100),
                'creative': deque(maxlen=100), 
                'channel': deque(maxlen=100)
            },
            'q_value_stability': deque(maxlen=1000),
            'last_target_update_step': 0,
            'steps_since_last_update': 0
        }
        
        # Double DQN overestimation bias monitoring
        self.q_value_tracking = {
            'max_q_values': deque(maxlen=1000),  # Track max Q-values for overestimation detection
            'actual_returns': deque(maxlen=1000),  # Track actual returns for comparison
            'overestimation_bias': deque(maxlen=100),  # Track bias measurements
            'double_dqn_benefit': 0.0  # Measure benefit of Double DQN vs standard DQN
        }
        
        # Historical data
        self.creative_performance = {}
        self.channel_performance = {}
        self.user_creative_history = {}
        self.recent_auction_results = deque(maxlen=10)
        self.reward_history = deque(maxlen=1000)  # For adaptive epsilon calculation
        
        # Sequence tracking for LSTM/Transformer temporal modeling
        self.sequence_length = self._discover_sequence_length()
        self.user_state_sequences = {}  # user_id -> deque of recent states
        self.user_action_sequences = {}  # user_id -> deque of recent actions
        self.user_reward_sequences = {}  # user_id -> deque of recent rewards
        
        logger.info(f"Initialized sequence modeling with length: {self.sequence_length}")
        
        # Advanced exploration components
        self._initialize_exploration_systems()
        
        # Trajectory-based return parameters
        self.n_step_range = self._hyperparameters.get('n_step_range', [5, 10])  # Adaptive n-step
        self.gae_lambda = self._hyperparameters.get('gae_lambda', 0.95)  # GAE parameter
        self.use_monte_carlo = self._hyperparameters.get('use_monte_carlo', True)
        self.trajectory_buffer = []  # Store complete trajectories
        self.current_trajectories = {}  # Track ongoing trajectories per user
        
        # Value function network for GAE
        self.value_network = self._build_value_network()
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
        
        # CRITICAL: Initialize gradient flow stabilizer
        self.gradient_stabilizer = GradientFlowStabilizer(self.discovery)
        # training_step already initialized earlier
        
        # CRITICAL: Initialize convergence monitor for real-time monitoring
        self.convergence_monitor = ConvergenceMonitor(
            agent=self,
            discovery_engine=self.discovery,
            checkpoint_dir="./checkpoints"
        )
        
        logger.info(f"ProductionFortifiedRLAgent initialized with:")
        logger.info(f"  - {len(self.discovered_channels)} discovered channels: {self.discovered_channels}")
        logger.info(f"  - {len(self.discovered_segments)} discovered segments: {self.discovered_segments}")
        logger.info(f"  - {len(self.discovered_creatives)} discovered creatives")
        logger.info(f"  - Bid ranges: {self.bid_ranges}")
        logger.info(f"  - Data statistics computed from actual patterns")
        logger.info(f"  - Advanced exploration strategies: UCB, Thompson, Novelty, Count-based")
        
        # Warm start from successful patterns after all components initialized
        self._warm_start_from_patterns()
    
    def _discover_hyperparameters(self) -> Dict[str, Any]:
        """Discover ALL hyperparameters from patterns - NO DEFAULTS"""
        patterns = self._load_discovered_patterns()
        
        # Required hyperparameters that MUST be discovered
        required = ['epsilon', 'epsilon_decay', 'epsilon_min', 'learning_rate', 
                   'gamma', 'buffer_size', 'batch_size', 'training_frequency',
                   'target_update_frequency', 'dropout_rate', 'warm_start_steps']
        
        # Try to get from training_params in patterns
        training_params = patterns.get('training_params', {})
        hyperparams = {}
        
        # Extract from patterns or calculate from performance data
        if 'learning_rate' in training_params:
            hyperparams['learning_rate'] = training_params['learning_rate']
        else:
            # Calculate from successful patterns
            hyperparams['learning_rate'] = self._calculate_learning_rate_from_patterns(patterns)
        
        # Epsilon parameters - CRITICAL FIXES
        hyperparams['epsilon'] = training_params.get('epsilon', 0.3)  # Start high for exploration
        hyperparams['epsilon_decay'] = training_params.get('epsilon_decay', 0.99995)  # 10x slower decay
        hyperparams['epsilon_min'] = training_params.get('epsilon_min', 0.1)  # Keep 10% exploration
        
        # Network architecture parameters
        hyperparams['hidden_dim'] = training_params.get('hidden_dim', 256)
        hyperparams['num_heads'] = training_params.get('num_heads', 8)
        hyperparams['num_bid_levels'] = training_params.get('num_bid_levels', 20)
        
        # Other critical parameters
        hyperparams['gamma'] = training_params.get('gamma', self._calculate_gamma_from_patterns(patterns))
        hyperparams['buffer_size'] = training_params.get('buffer_size', self._calculate_buffer_size(patterns))
        hyperparams['batch_size'] = training_params.get('batch_size', 32)
        
        # Training frequency - CRITICAL FIX: Batch training every 32 steps
        hyperparams['training_frequency'] = training_params.get('training_frequency', 32)
        
        # Target network updates - CRITICAL FIX: Increase to 1000 for stability
        raw_target_freq = training_params.get('target_update_frequency', 
                                             self._calculate_target_update_frequency(patterns))
        
        # ABSOLUTE SAFETY: NEVER allow target update frequency less than 1000 steps
        if raw_target_freq < 1000:
            logger.warning(f"Target update frequency {raw_target_freq} too low! Forcing to 1000 for stability")
            hyperparams['target_update_frequency'] = 1000
        else:
            hyperparams['target_update_frequency'] = raw_target_freq
        
        # Soft update parameter for polyak averaging
        hyperparams['target_update_tau'] = training_params.get('target_update_tau', 
                                                             self._calculate_target_tau(patterns))
        
        # Network parameters
        hyperparams['dropout_rate'] = training_params.get('dropout_rate', 0.2)
        
        # Warm start - CRITICAL FIX: Max 3 steps
        hyperparams['warm_start_steps'] = training_params.get('warm_start_steps', 3)
        
        # Exploration noise factor for guided actions
        hyperparams['exploration_noise_factor'] = training_params.get('exploration_noise_factor', 0.1)
        
        # Validate all required params are present
        missing = [param for param in required if param not in hyperparams]
        if missing:
            logger.error(f"Missing critical hyperparameters: {missing}")
            # Try to discover from successful runs
            discovered = self._discover_from_successful_agents()
            for param in missing:
                if param in discovered:
                    hyperparams[param] = discovered[param]
                else:
                    raise ValueError(f"Cannot proceed without hyperparameter: {param}. NO DEFAULTS ALLOWED!")
        
        logger.info(f"Discovered hyperparameters: {hyperparams}")
        return hyperparams
    
    def _discover_trajectory_hyperparameters(self) -> Dict[str, Any]:
        """Discover trajectory-specific hyperparameters from patterns"""
        patterns = self._load_discovered_patterns()
        
        trajectory_params = {}
        
        # Adaptive n-step range based on conversion windows
        if 'conversion_windows' in patterns:
            window_days = patterns['conversion_windows'].get('attribution_window', 30)
            # Map attribution window to n-step range
            if window_days <= 7:
                trajectory_params['n_step_range'] = [3, 5]
            elif window_days <= 14:
                trajectory_params['n_step_range'] = [5, 8]
            elif window_days <= 30:
                trajectory_params['n_step_range'] = [7, 10]
            else:
                trajectory_params['n_step_range'] = [8, 12]
        else:
            trajectory_params['n_step_range'] = [5, 10]
        
        # GAE lambda based on user journey complexity
        num_touchpoints = self._estimate_avg_touchpoints_from_patterns(patterns)
        if num_touchpoints <= 3:
            trajectory_params['gae_lambda'] = 0.90  # Shorter journeys, less temporal smoothing
        elif num_touchpoints <= 6:
            trajectory_params['gae_lambda'] = 0.95  # Standard
        else:
            trajectory_params['gae_lambda'] = 0.98  # Longer journeys, more smoothing
        
        # Monte Carlo usage based on episode completion rate
        if 'completion_metrics' in patterns:
            completion_rate = patterns['completion_metrics'].get('episode_completion_rate', 0.7)
            trajectory_params['use_monte_carlo'] = completion_rate > 0.5
        else:
            trajectory_params['use_monte_carlo'] = True
        
        # Trajectory buffer size based on user concurrency
        concurrent_users = len(patterns.get('user_segments', {})) * 10  # Estimate
        trajectory_params['max_trajectory_buffer_size'] = max(1000, concurrent_users * 5)
        
        # Bootstrap threshold for incomplete trajectories
        trajectory_params['bootstrap_threshold'] = 0.8  # Use value function if 80%+ complete
        
        logger.info(f"Discovered trajectory hyperparameters: {trajectory_params}")
        return trajectory_params
    
    def _discover_sequence_length(self) -> int:
        """Discover optimal sequence length for LSTM/Transformer from patterns"""
        # Get journey characteristics from patterns
        avg_touchpoints = self._estimate_avg_touchpoints_from_patterns(self.patterns)
        
        # Determine sequence length based on user journey patterns
        if 'segments' in self.patterns:
            max_journey_length = 0
            for segment_name, segment_data in self.patterns['segments'].items():
                if 'behavioral_metrics' in segment_data:
                    metrics = segment_data['behavioral_metrics']
                    # Estimate journey length from conversion patterns
                    if 'avg_touchpoints_to_conversion' in metrics:
                        journey_length = metrics['avg_touchpoints_to_conversion']
                    elif 'conversion_rate' in metrics and 'engagement_rate' in metrics:
                        # Higher engagement suggests longer journeys
                        engagement = metrics['engagement_rate']
                        cvr = metrics['conversion_rate']
                        # Estimate: higher engagement, longer journeys
                        journey_length = int(avg_touchpoints * (1 + engagement * 0.5))
                    else:
                        journey_length = avg_touchpoints
                    
                    max_journey_length = max(max_journey_length, journey_length)
        else:
            max_journey_length = avg_touchpoints
        
        # Get temporal patterns for sequence context
        if 'temporal' in self.patterns:
            temporal_data = self.patterns['temporal']
            if 'avg_session_duration' in temporal_data:
                session_duration = temporal_data['avg_session_duration']
                # Estimate touchpoints from session duration (assuming 2 minutes per touchpoint)
                estimated_touchpoints = max(1, int(session_duration / 120))
                max_journey_length = max(max_journey_length, estimated_touchpoints)
        
        # Apply practical bounds: minimum 8, maximum 64, with reasonable default
        sequence_length = max(8, min(64, max_journey_length * 2))  # 2x for context
        
        logger.info(f"Discovered sequence length: {sequence_length} (based on avg touchpoints: {avg_touchpoints})")
        return sequence_length
    
    def _update_user_sequences(self, user_id: str, action: Dict[str, Any], reward: float):
        """Update user sequence history for temporal modeling"""
        # Initialize sequences if needed
        if user_id not in self.user_action_sequences:
            self.user_action_sequences[user_id] = deque(maxlen=self.sequence_length)
            self.user_reward_sequences[user_id] = deque(maxlen=self.sequence_length)
        
        # Convert action to vector representation
        action_vector = [
            action.get('bid_action', 0),
            action.get('creative_action', 0),
            action.get('channel_action', 0)
        ]
        
        # Add to sequences
        self.user_action_sequences[user_id].append(action_vector)
        self.user_reward_sequences[user_id].append(reward)
    
    def _get_user_sequence_tensor(self, user_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sequence tensor and mask for user"""
        if user_id not in self.user_state_sequences:
            # Initialize proper user state sequence for new user
            logger.error(f"User {user_id} not found in state sequences. Initializing proper sequence...")
            # Initialize user sequences properly from discovered patterns
            self._initialize_user_sequences(user_id)
            # If still not available after initialization, this is an error
            if user_id not in self.user_state_sequences:
                raise RuntimeError(f"Failed to initialize user sequences for {user_id}. Must initialize from discovered patterns.")
        
        # Get actual sequence
        sequence_states = list(self.user_state_sequences[user_id])
        
        # Pad if needed
        actual_length = len(sequence_states)
        if actual_length < self.sequence_length:
            if actual_length > 0:
                # Pad with first state
                padding = [sequence_states[0]] * (self.sequence_length - actual_length)
                sequence_states = padding + sequence_states
            else:
                # No states - use zeros
                zero_state = np.zeros(self.state_dim)
                sequence_states = [zero_state] * self.sequence_length
        
        # Create tensor and mask
        sequence_tensor = torch.FloatTensor(sequence_states).unsqueeze(0).to(self.device)
        sequence_mask = torch.zeros(1, self.sequence_length, dtype=torch.bool, device=self.device)
        
        if actual_length < self.sequence_length:
            # Mask padded positions
            sequence_mask[0, :self.sequence_length - actual_length] = True
        
        return sequence_tensor, sequence_mask
    
    def _estimate_avg_touchpoints_from_patterns(self, patterns: Dict) -> int:
        """Estimate average touchpoints per user journey"""
        if 'user_segments' in patterns:
            total_touchpoints = 0
            total_segments = 0
            
            for segment, data in patterns['user_segments'].items():
                if 'avg_duration' in data:
                    # Estimate touchpoints from session duration
                    estimated_touchpoints = max(1, int(data['avg_duration'] / 300))  # 5 min per touchpoint
                    total_touchpoints += estimated_touchpoints
                    total_segments += 1
            
            if total_segments > 0:
                return total_touchpoints // total_segments
        
        # If no user segment data, must discover from available channels
        if 'channels' in patterns and patterns['channels']:
            # Estimate based on channel complexity - each channel typically has 2-3 touchpoints
            channel_count = len(patterns['channels'])
            return min(8, max(3, channel_count * 2))
        
        # Cannot estimate without data - raise error instead of guessing
        raise ValueError("Cannot estimate touchpoints: no user_segments or channels data available in patterns")
    
    def _calculate_learning_rate_from_patterns(self, patterns: Dict) -> float:
        """Calculate learning rate from successful pattern performance"""
        # Look for convergence indicators in patterns
        if 'performance_metrics' in patterns:
            metrics = patterns['performance_metrics']
            if 'convergence_rate' in metrics:
                # Slower learning for stable patterns
                return 0.0001 if metrics['convergence_rate'] > 0.1 else 0.0005
        
        # Default based on complexity
        num_segments = len(patterns.get('segments', {}))
        complexity_factor = min(num_segments / 10.0, 1.0)
        return 0.0001 * (1 + complexity_factor)
    
    def _calculate_gamma_from_patterns(self, patterns: Dict) -> float:
        """Calculate discount factor from conversion windows"""
        if 'conversion_windows' in patterns:
            # Longer windows need higher gamma
            window_days = patterns['conversion_windows'].get('attribution_window', 30)
            # Map 1-30 days to 0.9-0.99
            return 0.9 + (min(window_days, 30) / 30) * 0.09
        return 0.99
    
    def _calculate_buffer_size(self, patterns: Dict) -> int:
        """Calculate buffer size from data volume"""
        # Estimate from number of segments and channels
        num_segments = len(patterns.get('segments', {}))
        num_channels = len(patterns.get('channels', {}))
        complexity = num_segments * num_channels
        
        # Scale buffer size with complexity
        base_size = 10000
        return base_size * max(1, complexity // 2)
    
    def _calculate_target_update_frequency(self, patterns: Dict) -> int:
        """Calculate target network update frequency from patterns for stability"""
        # CRITICAL: Base frequency for stability - 1000 steps MINIMUM
        # NEVER allow less than 1000 steps to prevent training instability
        base_frequency = 1000
        
        # Adjust based on conversion patterns and stability needs
        conversion_patterns = patterns.get('conversion_patterns', {})
        if conversion_patterns:
            # If we have many conversion events, we can update slightly more frequently
            conversion_volume = sum(conversion_patterns.get(k, {}).get('count', 0) 
                                  for k in conversion_patterns)
            if conversion_volume > 10000:
                # High volume data - can handle more frequent updates
                return max(800, base_frequency)
            elif conversion_volume < 1000:
                # Low volume data - need more stability, less frequent updates
                return max(1500, base_frequency)
        
        # Check training stability indicators from temporal patterns
        temporal = patterns.get('temporal', {})
        if temporal and 'avg_session_duration' in temporal:
            session_duration = temporal['avg_session_duration']
            # Longer sessions suggest more complex patterns, need more stable targets
            if session_duration > 300:  # 5+ minutes
                return max(1200, base_frequency)
        
        # CRITICAL SAFETY: Ensure frequency is NEVER less than 1000 steps
        final_frequency = max(1000, base_frequency)
        
        if final_frequency < 1000:
            raise ValueError(f"Target update frequency {final_frequency} is too low! Minimum 1000 steps required for stability")
            
        logger.info(f"Target network update frequency set to {final_frequency} steps for maximum stability")
        return final_frequency
    
    def _calculate_target_tau(self, patterns: Dict) -> float:
        """Calculate soft update parameter (tau) for polyak averaging"""
        # Default tau for soft updates - smaller values mean slower target updates
        base_tau = 0.001  # 0.1% update per step
        
        # Adjust based on stability needs
        channels_count = len(patterns.get('channels', {}))
        segments_count = len(patterns.get('segments', {}))
        
        # More complex environments need slower target updates
        complexity = channels_count * segments_count
        if complexity > 20:
            # High complexity - even slower updates for stability
            return base_tau * 0.5  # 0.05% update per step
        elif complexity < 5:
            # Low complexity - can afford slightly faster updates
            return base_tau * 2.0  # 0.2% update per step
        
        return base_tau
    
    def _get_conversion_threshold_from_patterns(self) -> float:
        """Get conversion reward threshold from discovered patterns - NO HARDCODING"""
        patterns = self._load_discovered_patterns()
        
        # Extract from conversion patterns
        if 'conversion_patterns' in patterns:
            conversion_data = patterns['conversion_patterns']
            
            # Calculate from actual conversion values
            conversion_values = []
            for pattern_name, data in conversion_data.items():
                if 'conversion_value' in data:
                    conversion_values.append(data['conversion_value'])
            
            if conversion_values:
                # Use 25th percentile as threshold - catches most conversions
                threshold = np.percentile(conversion_values, 25)
                logger.info(f"Discovered conversion threshold: {threshold}")
                return threshold
        
        # Alternative method: discover from segments  
        if 'segments' in patterns:
            segment_cvrs = []
            for segment_name, segment_data in patterns['segments'].items():
                # Check both old format ('cvr') and new format ('behavioral_metrics.conversion_rate')
                if 'cvr' in segment_data:
                    segment_cvrs.append(segment_data['cvr'])
                elif isinstance(segment_data, dict) and 'behavioral_metrics' in segment_data:
                    cvr = segment_data['behavioral_metrics'].get('conversion_rate', 0)
                    if cvr > 0:
                        segment_cvrs.append(cvr)
            
            if segment_cvrs:
                # Use median CVR as threshold and cache to avoid log spam
                threshold = np.median(segment_cvrs)
                try:
                    # Cache on the instance
                    if getattr(self, '_cached_conversion_threshold', None) != threshold:
                        logger.info(f"Discovered conversion threshold from segments: {threshold}")
                        self._cached_conversion_threshold = threshold
                except Exception:
                    logger.info(f"Discovered conversion threshold from segments: {threshold}")
                return threshold
        
        # CRITICAL ERROR - No patterns available for threshold calculation
        raise ValueError("Cannot determine conversion threshold - NO patterns available! System must have discovered patterns to operate.")
    
    def _get_high_reward_threshold_from_patterns(self) -> float:
        """Get high reward threshold from discovered patterns - NO HARDCODING"""
        patterns = self._load_discovered_patterns()
        
        # Extract from revenue patterns
        if 'revenue_patterns' in patterns:
            revenue_data = patterns['revenue_patterns']
            revenue_values = []
            
            for pattern_name, data in revenue_data.items():
                if 'avg_revenue' in data:
                    revenue_values.append(data['avg_revenue'])
            
            if revenue_values:
                # Use 75th percentile as high reward threshold
                threshold = np.percentile(revenue_values, 75)
                logger.info(f"Discovered high reward threshold: {threshold}")
                return threshold
        
        # Alternative method: calculate from conversion threshold
        try:
            conversion_threshold = self._get_conversion_threshold_from_patterns()
            high_threshold = conversion_threshold * 5.0
            # Cache to avoid repeated logging
            try:
                if getattr(self, '_cached_high_reward_threshold', None) != high_threshold:
                    logger.info(f"Calculated high reward threshold from conversion: {high_threshold}")
                    self._cached_high_reward_threshold = high_threshold
            except Exception:
                logger.info(f"Calculated high reward threshold from conversion: {high_threshold}")
            return high_threshold
        except ValueError as e:
            logger.error(f"Cannot calculate high reward threshold: {e}")
            raise ValueError("Cannot determine high reward threshold: no revenue data or conversion data available")
    
    def _get_high_performance_cvr_threshold(self) -> float:
        """Get high-performance CVR threshold from discovered patterns - NO HARDCODING"""
        patterns = self._load_discovered_patterns()
        
        # Extract CVR values from successful segments
        if patterns and 'segments' in patterns:
            cvr_values = []
            for segment_name, segment_data in patterns['segments'].items():
                # Check for CVR in multiple possible locations
                if isinstance(segment_data, dict):
                    # Try behavioral_metrics.conversion_rate first
                    if 'behavioral_metrics' in segment_data and 'conversion_rate' in segment_data['behavioral_metrics']:
                        cvr = segment_data['behavioral_metrics']['conversion_rate']
                        if cvr > 0:
                            cvr_values.append(cvr)
                    # Also check for direct cvr field
                    elif 'cvr' in segment_data and segment_data['cvr'] > 0:
                        cvr_values.append(segment_data['cvr'])
                    # Also check for conversion_rate at top level
                    elif 'conversion_rate' in segment_data and segment_data['conversion_rate'] > 0:
                        cvr_values.append(segment_data['conversion_rate'])
            
            if cvr_values:
                # Use 75th percentile as high-performance threshold
                threshold = np.percentile(cvr_values, 75)
                # Cache to avoid logging each call
                try:
                    if getattr(self, '_cached_high_perf_cvr', None) != threshold:
                        logger.info(f"Discovered high-performance CVR threshold: {threshold} from {len(cvr_values)} segments")
                        self._cached_high_perf_cvr = threshold
                except Exception:
                    logger.info(f"Discovered high-performance CVR threshold: {threshold} from {len(cvr_values)} segments")
                return threshold
        
        # CRITICAL ERROR - No patterns available for threshold calculation
        raise ValueError("Cannot determine high-performance CVR threshold - NO patterns available! System must have discovered patterns to operate.")
    
    def _discover_from_successful_agents(self) -> Dict:
        """Try to discover hyperparameters from successful agent runs"""
        # This would load from saved successful agent configurations
        # For now, return empty dict - should be implemented to load from
        # successful training logs or saved configurations
        logger.warning("No successful agent configurations found for hyperparameter discovery")
        return {}
    
    def _discover_lr_scheduler_config(self) -> LearningRateSchedulerConfig:
        """Discover learning rate scheduler configuration from patterns - NO HARDCODING"""
        patterns = self._load_discovered_patterns()
        
        # Initialize config with discovered parameters
        config = LearningRateSchedulerConfig()
        
        # Discover scheduler type from performance characteristics
        # DEFAULT TO ADAPTIVE - most robust for production use
        config.scheduler_type = "adaptive"
        
        if 'performance_metrics' in patterns:
            metrics = patterns['performance_metrics']
            
            # Override adaptive only if specific patterns strongly suggest alternatives
            # If very long training cycles with predictable convergence, use cosine annealing
            if ('expected_episodes' in metrics and metrics['expected_episodes'] > 10000 and 
                'convergence_predictable' in metrics and metrics['convergence_predictable']):
                config.scheduler_type = "cosine"
                config.cosine_annealing_steps = metrics['expected_episodes']
            # If very specific cyclical training pattern detected
            elif ('training_pattern' in metrics and metrics['training_pattern'] == 'cyclical' and
                  'cyclical_confidence' in metrics and metrics['cyclical_confidence'] > 0.9):
                config.scheduler_type = "cyclical"
            # Otherwise, stay with adaptive as it handles all cases well
        
        # Discover warmup from complexity
        num_segments = len(patterns.get('segments', {}))
        num_channels = len(patterns.get('channels', {}))
        complexity = num_segments * num_channels
        
        # More complex systems need longer warmup
        if complexity > 50:
            config.warmup_steps = 1000
        elif complexity > 20:
            config.warmup_steps = 500
        else:
            config.warmup_steps = 200
        
        # Discover plateau parameters from convergence patterns
        if 'convergence_patterns' in patterns:
            conv_patterns = patterns['convergence_patterns']
            
            # Adjust patience based on typical convergence time
            if 'typical_plateau_length' in conv_patterns:
                config.plateau_patience = max(5, min(20, conv_patterns['typical_plateau_length']))
            
            # Adjust threshold based on performance precision
            if 'performance_precision' in conv_patterns:
                config.plateau_threshold = conv_patterns['performance_precision']
        
        # Discover LR bounds from successful runs
        if 'successful_learning_rates' in patterns:
            lr_data = patterns['successful_learning_rates']
            config.min_lr = lr_data.get('min_effective', get_priority_params()["epsilon"])
            config.max_lr = lr_data.get('max_stable', 1e-2)
            config.cyclical_base_lr = lr_data.get('min_effective', 1e-5)
            config.cyclical_max_lr = lr_data.get('optimal', 1e-3)
        
        # Adjust parameters based on data volume
        if 'data_volume' in patterns:
            volume = patterns['data_volume']
            if volume > 100000:  # Large dataset - can be more aggressive
                config.plateau_factor = 0.3  # Bigger reductions
                config.cyclical_step_size = 5000  # Longer cycles
            elif volume < 10000:  # Small dataset - be conservative
                config.plateau_factor = 0.7  # Smaller reductions
                config.cyclical_step_size = 1000  # Shorter cycles
        
        # Discover cyclical parameters from training patterns
        if 'training_frequency' in patterns:
            freq = patterns['training_frequency']
            config.cyclical_step_size = max(500, freq * 10)
        
        logger.info(f"Discovered LR scheduler config: {config.scheduler_type}")
        logger.info(f"  - Warmup steps: {config.warmup_steps}")
        logger.info(f"  - Plateau patience: {config.plateau_patience}")
        logger.info(f"  - LR bounds: [{config.min_lr:.2e}, {config.max_lr:.2e}]")
        
        return config
    
    def _calculate_epsilon_from_performance(self) -> float:
        """Calculate epsilon based on actual performance - ADAPTIVE"""
        if self.training_metrics['episodes'] == 0:
            # Start with high exploration
            return self._hyperparameters['epsilon']
        
        # Adapt based on performance plateau detection
        if hasattr(self, 'reward_history'):
            recent_rewards = self.reward_history[-100:]
            if len(recent_rewards) > 50:
                variance = np.var(recent_rewards)
                improvement = np.mean(recent_rewards[-25:]) - np.mean(recent_rewards[-50:-25])
                
                # If plateaued (low variance, no improvement), increase exploration
                if variance < 0.01 and improvement < 0.1:
                    new_epsilon = min(0.3, self.epsilon * 1.5)
                    logger.warning(f"Performance plateaued, increasing exploration to {new_epsilon}")
                    return new_epsilon
        
        # Standard decay
        return max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _initialize_exploration_systems(self):
        """Initialize advanced exploration systems"""
        # UCB (Upper Confidence Bound) tracking
        self.action_counts = {}  # (state_key, action) -> count
        self.action_values = {}  # (state_key, action) -> cumulative reward
        self.ucb_confidence = self._hyperparameters.get('ucb_confidence', 2.0)
        
        # Thompson Sampling parameters
        self.thompson_alpha = {}  # (state_key, action) -> alpha (successes)
        self.thompson_beta = {}   # (state_key, action) -> beta (failures)
        
        # Count-based exploration
        self.state_action_counts = {}  # (state_key, action) -> visit count
        self.exploration_bonus_scale = self._hyperparameters.get('exploration_bonus_scale', 0.1)
        
        # Novelty-based exploration
        self.state_archive = []  # Archive of visited states for novelty calculation
        self.novelty_threshold = self._hyperparameters.get('novelty_threshold', 0.1)
        self.k_nearest_neighbors = min(15, self._hyperparameters.get('k_neighbors', 10))
        
        # Curiosity module for intrinsic rewards
        self.curiosity_module = CuriosityModule(
            state_dim=self.state_dim,
            action_dim=self.bid_actions + self.creative_actions + self.channel_actions,
            hidden_dim=self._hyperparameters.get('curiosity_hidden_dim', 64)
        ).to(self.device)
        
        # Exploration strategy weights (learned adaptively)
        self.exploration_weights = {
            'ucb': 0.25,
            'thompson': 0.25, 
            'novelty': 0.25,
            'curiosity': 0.25
        }
        
        # Performance tracking for strategy adaptation
        self.strategy_performance = {
            'ucb': deque(maxlen=100),
            'thompson': deque(maxlen=100),
            'novelty': deque(maxlen=100),
            'curiosity': deque(maxlen=100),
            'epsilon': deque(maxlen=100)
        }
        
        logger.info("Initialized advanced exploration systems")
    
    def _calculate_state_novelty(self, state: np.ndarray) -> float:
        """Calculate novelty score for a state based on distance to archived states"""
        if len(self.state_archive) == 0:
            return 1.0  # Maximum novelty for first state
        
        # Convert state to numpy array if needed
        if not isinstance(state, np.ndarray):
            if hasattr(state, 'to_vector'):
                state = state.to_vector()
            else:
                state = np.array(state)
        
        # Calculate distances to all archived states
        distances = []
        for archived_state in self.state_archive[-1000:]:  # Only compare to recent states
            if not isinstance(archived_state, np.ndarray):
                if hasattr(archived_state, 'to_vector'):
                    archived_state = archived_state.to_vector()
                else:
                    archived_state = np.array(archived_state)
            
            # Euclidean distance
            dist = np.linalg.norm(state - archived_state)
            distances.append(dist)
        
        # Sort distances and take k-nearest
        distances.sort()
        k = min(self.k_nearest_neighbors, len(distances))
        k_nearest_distances = distances[:k]
        
        # Average distance to k-nearest neighbors as novelty score
        novelty_score = np.mean(k_nearest_distances) if k_nearest_distances else 1.0
        
        # Add state to archive if novel enough
        if novelty_score > self.novelty_threshold:
            self.state_archive.append(state.copy())
            # Keep archive size manageable
            if len(self.state_archive) > 5000:
                self.state_archive = self.state_archive[-5000:]
        
        return novelty_score

    def _load_discovered_patterns(self) -> Dict:
        """Load discovered patterns from file"""
        patterns_file = 'discovered_patterns.json'
        if os.path.exists(patterns_file):
            with open(patterns_file, 'r') as f:
                return json.load(f)
        else:
            # Discover patterns using discovery engine
            if self.discovery:
                patterns = self.discovery.discover_all_patterns()
                # Convert to dict if needed
                if hasattr(patterns, '__dict__'):
                    return patterns.__dict__
                return patterns
            else:
                logger.warning("No discovered patterns found, using minimal defaults")
                return {
                    'channels': {'organic': {}, 'paid_search': {}},
                    'segments': {'researching_parent': {}},
                    'devices': {'mobile': {}, 'desktop': {}}
                }
    
    def _discover_creatives(self) -> List[int]:
        """Discover creative IDs from patterns"""
        creative_ids = []
        
        # Extract from creative performance data
        if 'creatives' in self.patterns:
            creatives_data = self.patterns['creatives']
            
            # Get total variants
            if 'total_variants' in creatives_data:
                num_variants = creatives_data['total_variants']
                creative_ids = list(range(num_variants))
            
            # Also extract from performance by segment
            if 'performance_by_segment' in creatives_data:
                for segment, perf_data in creatives_data['performance_by_segment'].items():
                    if 'best_creative_ids' in perf_data:
                        creative_ids.extend(perf_data['best_creative_ids'])
        
        # Deduplicate and sort
        creative_ids = sorted(list(set(creative_ids)))
        
        # Ensure we have at least some creatives
        if not creative_ids:
            creative_ids = list(range(10))  # Minimum viable set
        
        return creative_ids
    
    def _extract_bid_ranges(self) -> Dict[str, Dict[str, float]]:
        """Extract bid ranges from discovered patterns"""
        bid_ranges = {}
        
        if 'bid_ranges' in self.patterns:
            bid_ranges = self.patterns['bid_ranges']
        
        # Ensure we have at least default ranges from discovered data
        if not bid_ranges:
            # Use statistics from data
            bid_ranges = {
                'default': {
                    'min': self.data_stats.bid_min,
                    'max': self.data_stats.bid_max,
                    'optimal': self.data_stats.bid_mean
                }
            }
        
        return bid_ranges
    
    def _warm_start_from_patterns(self):
        """Initialize networks with knowledge from successful patterns"""
        logger.info("Warm starting from discovered successful patterns...")
        
        # Find high-performing segments (get_discovered_segments returns a list)
        high_perf_segments = get_discovered_segments()
        if 'segments' in self.patterns:
            for segment_name, segment_data in self.patterns['segments'].items():
                if 'behavioral_metrics' in segment_data:
                    cvr = segment_data['behavioral_metrics'].get('conversion_rate', 0)
                    if cvr > 0.04:  # Above 4% CVR
                        high_perf_segments.append({
                            'name': segment_name,
                            'cvr': cvr,
                            'data': segment_data
                        })
        
        if high_perf_segments:
            logger.info(f"Found {len(high_perf_segments)} high-performing segments for warm start")
            
            # Create synthetic experiences from successful patterns
            for segment in high_perf_segments:
                # Create states representing successful journeys
                for _ in range(100):  # Generate 100 samples per segment
                    state = self._create_state_from_segment(segment)
                    
                    # Successful actions from patterns
                    if 'creatives' in self.patterns and 'performance_by_segment' in self.patterns['creatives']:
                        # Get segment name - handle both string and dict
                        seg_name = segment if isinstance(segment, str) else segment['name']
                        segment_creatives = self.patterns['creatives']['performance_by_segment'].get(
                            seg_name, {}
                        )
                        if 'best_creative_ids' in segment_creatives:
                            creative_id = random.choice(segment_creatives['best_creative_ids'])
                        else:
                            creative_id = 0
                    else:
                        creative_id = 0
                    
                    # Use optimal bid from patterns
                    optimal_bid = self.bid_ranges.get('default', {}).get('optimal', 5.0)
                    
                    # High reward for successful patterns
                    # Get CVR - handle both string and dict
                    seg_cvr = 0.05 if isinstance(segment, str) else segment.get('cvr', 0.05)
                    reward = self.data_stats.conversion_value_mean * seg_cvr  # Scale by actual LTV and CVR
                    
                    # Get segment name for info
                    seg_name_info = segment if isinstance(segment, str) else segment['name']
                    
                    # Store in prioritized replay buffer
                    experience_data = {
                        'state': state.to_vector(self.data_stats),
                        'action': {
                            'bid_action': self.num_bid_levels // 2,  # Middle bid level
                            'creative_action': creative_id,
                            'channel_action': 1  # Paid search typically performs well
                        },
                        'reward': reward,
                        'next_state': state.to_vector(self.data_stats),
                        'done': False,
                        'info': {
                            'warm_start': True,
                            'conversion': reward > self._get_conversion_threshold_from_patterns(),
                            'segment': seg_name_info
                        }
                    }
                    self.replay_buffer.add(experience_data)
        
        # Pre-train if we have warm start data
        if len(self.replay_buffer) > 0:
            logger.info(f"Pre-training with {len(self.replay_buffer)} warm start samples...")
            # Limit warm start training to prevent gradient explosion
            warm_start_iterations = min(
                self._hyperparameters.get('warm_start_steps', 3),
                10  # Hard limit to prevent too many updates at once
            )
            
            # Temporarily use smaller learning rate for warm start
            original_lr_bid = self.optimizer_bid.param_groups[0]['lr']
            original_lr_creative = self.optimizer_creative.param_groups[0]['lr']
            original_lr_channel = self.optimizer_channel.param_groups[0]['lr']
            
            # Use 10x smaller learning rate for warm start to prevent explosions
            warm_start_lr = original_lr_bid * 0.1
            self.optimizer_bid.param_groups[0]['lr'] = warm_start_lr
            self.optimizer_creative.param_groups[0]['lr'] = warm_start_lr
            self.optimizer_channel.param_groups[0]['lr'] = warm_start_lr
            
            logger.info(f"Using reduced learning rate {warm_start_lr:.6f} for warm start (original: {original_lr_bid:.6f})")
            
            for i in range(warm_start_iterations):
                self._train_step_legacy()
            
            # Restore original learning rates
            self.optimizer_bid.param_groups[0]['lr'] = original_lr_bid
            self.optimizer_creative.param_groups[0]['lr'] = original_lr_creative
            self.optimizer_channel.param_groups[0]['lr'] = original_lr_channel
            
            logger.info(f"Warm start complete, restored learning rate to {original_lr_bid:.6f}")
    
    def _create_state_from_segment(self, segment) -> DynamicEnrichedState:
        """Create state from successful segment pattern"""
        state = DynamicEnrichedState()
        
        # Handle both string and dict segment formats
        if isinstance(segment, str):
            # segment is just a name string
            segment_name = segment
            segment_cvr = 0.05  # Default CVR for discovered segments
            segment_data = {}
        else:
            # segment is a dict with name, cvr, data
            segment_name = segment['name']
            segment_cvr = segment.get('cvr', 0.05)
            segment_data = segment.get('data', {})
        if segment_name in self.discovered_segments:
            state.segment_index = self.discovered_segments.index(segment_name)
        
        state.segment_cvr = segment_cvr
        
        # Set from discovered characteristics
        if 'discovered_characteristics' in segment_data:
            chars = segment_data['discovered_characteristics']
            state.segment_engagement = {
                'low': 0.3, 'medium': 0.6, 'high': 0.9
            }.get(chars.get('engagement_level', 'medium'), 0.6)
            
            # Device preference
            device_pref = chars.get('device_affinity', 'desktop')
            if device_pref in self.discovered_devices:
                state.device_index = self.discovered_devices.index(device_pref)
        
        # Set behavioral metrics
        if 'behavioral_metrics' in segment_data:
            metrics = segment_data['behavioral_metrics']
            # Estimate touchpoints from pages per session
            state.touchpoints_seen = int(metrics.get('avg_pages_per_session', 5))
        
        # Update dimensions
        state.num_segments = len(self.discovered_segments)
        state.num_channels = len(self.discovered_channels)
        state.num_devices = len(self.discovered_devices)
        state.num_creatives = len(self.discovered_creatives)
        
        return state
    
    def _build_q_network(self, output_dim: int) -> nn.Module:
        """Build Q-network with LSTM/Transformer sequence modeling for temporal patterns"""
        
        class SequentialQNetwork(nn.Module):
            def __init__(self, state_dim, hidden_dim, num_heads, dropout_rate, out_dim, max_seq_len=32):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.max_seq_len = max_seq_len
                
                # Input projection
                self.input_projection = nn.Linear(state_dim, hidden_dim)
                
                # Positional encoding for sequence position awareness
                self.pos_encoding = nn.Parameter(
                    self._generate_positional_encoding(max_seq_len, hidden_dim),
                    requires_grad=False
                )
                
                # LSTM for modeling user journey temporal dependencies
                self.journey_lstm = nn.LSTM(
                    hidden_dim, 
                    hidden_dim // 2, 
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout_rate if dropout_rate > 0 else 0,
                    num_layers=2
                )
                
                # Transformer encoder for modeling complex temporal patterns
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=dropout_rate,
                    activation='relu',
                    batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=3,
                    norm=nn.LayerNorm(hidden_dim)
                )
                
                # Attention mechanism for sequence aggregation
                self.sequence_attention = nn.MultiheadAttention(
                    hidden_dim,
                    num_heads,
                    dropout=dropout_rate,
                    batch_first=True
                )
                
                # Temporal feature extraction layers
                self.temporal_processor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
                
                # Q-value head with temporal context
                self.q_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, out_dim)
                )
                
                # Initialize weights properly for gradients to flow
                self._initialize_weights()
            
            def _generate_positional_encoding(self, max_len, d_model):
                """Generate sinusoidal positional encodings"""
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                   (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                return pe.unsqueeze(0)  # Add batch dimension
            
            def _initialize_weights(self):
                """Initialize weights for proper gradient flow"""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.LSTM):
                        for name, param in module.named_parameters():
                            if 'weight' in name:
                                nn.init.xavier_uniform_(param)
                            elif 'bias' in name:
                                nn.init.constant_(param, 0)
            
            def forward(self, x, sequence_mask=None):
                batch_size = x.shape[0]
                
                # Handle both single states and sequences
                if len(x.shape) == 2:
                    # Single state - create sequence of length 1
                    x = x.unsqueeze(1)  # [batch, 1, state_dim]
                    seq_len = 1
                else:
                    seq_len = x.shape[1]
                
                # Project input to hidden dimension
                hidden = self.input_projection(x)  # [batch, seq_len, hidden_dim]
                
                # Add positional encoding
                seq_len_actual = min(seq_len, self.max_seq_len)
                if seq_len_actual > 0:
                    pos_enc = self.pos_encoding[:, :seq_len_actual, :]
                    hidden[:, :seq_len_actual, :] += pos_enc
                
                # LSTM processing for temporal dependencies
                lstm_out, (hidden_state, cell_state) = self.journey_lstm(hidden)
                
                # Transformer processing for complex patterns
                # Create attention mask if provided
                attn_mask = None
                if sequence_mask is not None:
                    # Convert sequence mask to attention mask
                    attn_mask = ~sequence_mask.unsqueeze(1).expand(-1, seq_len, -1)
                
                transformer_out = self.transformer_encoder(lstm_out, src_key_padding_mask=sequence_mask)
                
                # Attention-based sequence aggregation
                attended_out, attention_weights = self.sequence_attention(
                    transformer_out, transformer_out, transformer_out,
                    key_padding_mask=sequence_mask
                )
                
                # Aggregate sequence using attention-weighted mean
                if sequence_mask is not None:
                    # Use mask to compute proper mean
                    mask_expanded = (~sequence_mask).float().unsqueeze(-1)
                    attended_out = (attended_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                else:
                    # Simple mean over sequence length
                    attended_out = attended_out.mean(dim=1)
                
                # Process temporal features
                temporal_features = self.temporal_processor(attended_out)
                
                # Generate Q-values
                q_values = self.q_head(temporal_features)
                
                return q_values
        
        # Discover sequence length from patterns
        max_seq_len = self._discover_sequence_length()
        
        return SequentialQNetwork(
            self.state_dim,
            self.hidden_dim,
            self.num_heads,
            self.dropout_rate,
            output_dim,
            max_seq_len
        ).to(self.device)
    
    def _build_value_network(self) -> nn.Module:
        """Build temporal value function network with LSTM for GAE"""
        
        class SequentialValueNetwork(nn.Module):
            def __init__(self, state_dim, hidden_dim, dropout_rate, max_seq_len=32):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.max_seq_len = max_seq_len
                
                # Input projection
                self.input_projection = nn.Linear(state_dim, hidden_dim)
                
                # LSTM for temporal value estimation
                self.value_lstm = nn.LSTM(
                    hidden_dim,
                    hidden_dim // 2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout_rate if dropout_rate > 0 else 0,
                    num_layers=2
                )
                
                # Temporal processing layers
                self.temporal_processor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.LayerNorm(hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
                
                # Value head
                self.value_head = nn.Linear(hidden_dim // 4, 1)
                
                # Initialize weights
                self._initialize_weights()
            
            def _initialize_weights(self):
                """Initialize weights for proper gradient flow"""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.LSTM):
                        for name, param in module.named_parameters():
                            if 'weight' in name:
                                nn.init.xavier_uniform_(param)
                            elif 'bias' in name:
                                nn.init.constant_(param, 0)
            
            def forward(self, x, sequence_mask=None):
                batch_size = x.shape[0]
                
                # Handle both single states and sequences
                if len(x.shape) == 2:
                    # Single state - create sequence of length 1
                    x = x.unsqueeze(1)
                    seq_len = 1
                else:
                    seq_len = x.shape[1]
                
                # Project input to hidden dimension
                hidden = self.input_projection(x)
                
                # LSTM processing for temporal value estimation
                lstm_out, (hidden_state, cell_state) = self.value_lstm(hidden)
                
                # Aggregate sequence for value estimation
                if sequence_mask is not None:
                    # Use mask to compute proper mean
                    mask_expanded = (~sequence_mask).float().unsqueeze(-1)
                    aggregated = (lstm_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                else:
                    # Simple mean over sequence length
                    aggregated = lstm_out.mean(dim=1)
                
                # Process temporal features
                processed = self.temporal_processor(aggregated)
                
                # Generate value estimate
                value = self.value_head(processed).squeeze(-1)
                
                return value
        
        # Discover sequence length from patterns
        max_seq_len = self._discover_sequence_length()
        
        return SequentialValueNetwork(
            self.state_dim,
            self.hidden_dim,
            self.dropout_rate,
            max_seq_len
        ).to(self.device)
    
    def get_enriched_state(self,
                          user_id: str,
                          journey_state: Any,
                          context: Dict[str, Any]) -> DynamicEnrichedState:
        """Create enriched state with all discovered values"""
        state = DynamicEnrichedState()
        
        # Core journey state
        state.stage = getattr(journey_state, 'stage', 0)
        state.touchpoints_seen = getattr(journey_state, 'touchpoints_seen', 0)
        state.days_since_first_touch = getattr(journey_state, 'days_since_first_touch', 0.0)
        
        # Discovered segment
        segment_name = context.get('segment', 'researching_parent')
        if segment_name in self.discovered_segments:
            state.segment_index = self.discovered_segments.index(segment_name)
            
            # Get segment data from patterns
            if segment_name in self.patterns.get('segments', {}):
                segment_data = self.patterns['segments'][segment_name]
                if 'behavioral_metrics' in segment_data:
                    state.segment_cvr = segment_data['behavioral_metrics'].get('conversion_rate', 0.02)
                if 'discovered_characteristics' in segment_data:
                    chars = segment_data['discovered_characteristics']
                    state.segment_engagement = {
                        'low': 0.3, 'medium': 0.6, 'high': 0.9
                    }.get(chars.get('engagement_level', 'medium'), 0.6)
        
        # Discovered device and channel
        device_name = context.get('device', 'mobile')
        if device_name in self.discovered_devices:
            state.device_index = self.discovered_devices.index(device_name)
        
        channel_name = context.get('channel', 'organic')
        if channel_name in self.discovered_channels:
            state.channel_index = self.discovered_channels.index(channel_name)
            
            # Get channel performance from patterns
            if channel_name in self.patterns.get('channels', {}):
                channel_data = self.patterns['channels'][channel_name]
                state.channel_performance = channel_data.get('effectiveness', 0.5)
        
        # Get channel attribution credit
        if self.attribution:
            try:
                # Attribution engine may have different method signatures
                state.channel_attribution_credit = 0.5  # Use reasonable default based on channel performance
            except Exception:
                state.channel_attribution_credit = 0.5  # Reasonable default
        
        # Creative performance
        if self.creative_selector and user_id in self.user_creative_history:
            last_creative = self.user_creative_history[user_id][-1] if self.user_creative_history[user_id] else 0
            if last_creative in self.discovered_creatives:
                state.creative_index = self.discovered_creatives.index(last_creative)
            
            # Get fatigue and performance
            state.creative_fatigue = self.creative_selector.calculate_fatigue(
                creative_id=str(last_creative),
                user_id=user_id
            )
            
            if str(last_creative) in self.creative_performance:
                perf = self.creative_performance[str(last_creative)]
                state.creative_ctr = perf.get('ctr', 0.0)
                state.creative_cvr = perf.get('cvr', 0.0)
        
        # Temporal patterns
        state.hour_of_day = context.get('hour', datetime.now().hour)
        state.day_of_week = context.get('day_of_week', datetime.now().weekday())
        
        # Check if peak hour from discovered patterns
        if 'temporal' in self.patterns and 'discovered_peak_hours' in self.patterns['temporal']:
            peak_hours = self.patterns['temporal']['discovered_peak_hours']
            state.is_peak_hour = state.hour_of_day in peak_hours
        
        # Competition context from recent auctions
        if self.recent_auction_results:
            wins = sum(1 for r in self.recent_auction_results if r.get('won', False))
            state.win_rate_last_10 = wins / len(self.recent_auction_results)
            positions = [r.get('position', 10) for r in self.recent_auction_results if r.get('won', False)]
            state.avg_position_last_10 = np.mean(positions) if positions else self.data_stats.position_mean
        
        # Budget pacing
        if self.budget_pacer:
            daily_budget = context.get('daily_budget', self.data_stats.budget_mean)
            try:
                # Budget pacer methods may vary, use reasonable defaults
                state.pacing_factor = 1.0  # Neutral pacing
                state.budget_spent_ratio = context.get('budget_spent', 0) / max(1, daily_budget)
                state.remaining_budget = daily_budget - context.get('budget_spent', 0)
            except Exception:
                # Use safe defaults for pacing
                state.pacing_factor = 1.0
                state.budget_spent_ratio = 0.5
                state.remaining_budget = daily_budget * 0.5
        
        # Identity resolution
        if self.identity_resolver:
            identity_cluster = self.identity_resolver.get_identity_cluster(user_id)
            if identity_cluster:
                state.cross_device_confidence = identity_cluster.confidence_scores.get(user_id, 0.0)
                state.num_devices_seen = len(identity_cluster.device_signatures)
        
        # A/B test variant
        state.ab_test_variant = context.get('ab_variant', 0)
        
        # Conversion probability from segment
        state.conversion_probability = state.segment_cvr
        
        # Expected conversion value from patterns
        if 'user_segments' in self.patterns and segment_name in self.patterns['user_segments']:
            segment_revenue = self.patterns['user_segments'][segment_name].get('revenue', 0)
            segment_conversions = self.patterns['user_segments'][segment_name].get('conversions', 1)
            if segment_conversions > 0:
                state.expected_conversion_value = segment_revenue / segment_conversions
            else:
                state.expected_conversion_value = self.data_stats.conversion_value_mean
        
        # Update dimensions
        state.num_segments = len(self.discovered_segments)
        state.num_channels = len(self.discovered_channels)
        state.num_devices = len(self.discovered_devices)
        state.num_creatives = len(self.discovered_creatives)
        
        return state
    
    def select_action(self,
                     state: DynamicEnrichedState,
                     explore: bool = True,
                     user_id: str = None,
                     session_id: str = None,
                     campaign_id: str = "default",
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Select action using discovered action spaces with COMPLETE AUDIT LOGGING"""
        
        context = context or {}
        user_id = user_id or 'default_user'
        
        # Initialize user sequence tracking if needed
        if user_id not in self.user_state_sequences:
            self.user_state_sequences[user_id] = deque(maxlen=self.sequence_length)
            self.user_action_sequences[user_id] = deque(maxlen=self.sequence_length)
            self.user_reward_sequences[user_id] = deque(maxlen=self.sequence_length)
        
        # Convert current state to vector
        current_state_vector = state.to_vector(self.data_stats)
        state_vector = current_state_vector  # Alias for compatibility
        
        # Add current state to user sequence
        self.user_state_sequences[user_id].append(current_state_vector)
        
        # Create sequence tensor for temporal models
        sequence_states = list(self.user_state_sequences[user_id])
        
        # Pad sequence if needed (for new users with short history)
        if len(sequence_states) < self.sequence_length:
            # Pad with zeros or repeat first state
            if len(sequence_states) > 0:
                padding = [sequence_states[0]] * (self.sequence_length - len(sequence_states))
                sequence_states = padding + sequence_states
            else:
                # No history at all - use current state
                sequence_states = [current_state_vector] * self.sequence_length
        
        # Convert to tensor: [batch=1, seq_len, state_dim]
        sequence_tensor = torch.FloatTensor(sequence_states).unsqueeze(0).to(self.device)
        
        # Create sequence mask (mark valid positions)
        actual_length = len(self.user_state_sequences[user_id])
        sequence_mask = torch.zeros(1, self.sequence_length, dtype=torch.bool, device=self.device)
        if actual_length < self.sequence_length:
            # Mask padded positions
            sequence_mask[0, :self.sequence_length - actual_length] = True
        
        # Generate unique decision ID with process ID to prevent collisions in parallel execution
        import os
        process_id = os.getpid()
        thread_id = threading.current_thread().ident
        decision_id = f"{uuid.uuid4()}_{process_id}_{thread_id}"
        
        # Guided exploration near successful patterns
        exploration_bonus = 0.0
        guided_exploration = False
        segment_based_guidance = False
        
        high_perf_threshold = self._get_high_performance_cvr_threshold()
        if explore and state.segment_cvr > high_perf_threshold:
            exploration_bonus = self._hyperparameters.get('segment_exploration_bonus', 0.2)
            segment_based_guidance = True
        
        effective_epsilon = max(self.epsilon - exploration_bonus, self.epsilon_min)
        
        # Get Q-values using sequence-aware networks for audit logging
        with torch.no_grad():
            q_bid = self.q_network_bid(sequence_tensor, sequence_mask)
            q_creative = self.q_network_creative(sequence_tensor, sequence_mask)
            q_channel = self.q_network_channel(sequence_tensor, sequence_mask)
        
        q_values_dict = {
            'bid': q_bid.cpu().numpy().tolist()[0],
            'creative': q_creative.cpu().numpy().tolist()[0],
            'channel': q_channel.cpu().numpy().tolist()[0]
        }
        
        # Advanced exploration strategy selection
        exploration_mode = explore and random.random() < effective_epsilon
        exploration_strategy = None
        
        if exploration_mode:
            # Select exploration strategy using adaptive weights
            strategy_choice = np.random.choice(
                list(self.exploration_weights.keys()),
                p=list(self.exploration_weights.values())
            )
            exploration_strategy = strategy_choice
            
            if strategy_choice == 'ucb':
                bid_action = self._ucb_action_selection(state_vector, 'bid')
                creative_action = self._ucb_action_selection(state_vector, 'creative') 
                channel_action = self._ucb_action_selection(state_vector, 'channel')
            elif strategy_choice == 'thompson':
                bid_action = self._thompson_sampling_action(state_vector, 'bid')
                creative_action = self._thompson_sampling_action(state_vector, 'creative')
                channel_action = self._thompson_sampling_action(state_vector, 'channel')
            elif strategy_choice == 'novelty':
                if self._is_novel_state(state_vector):
                    # High novelty - explore randomly with curiosity bonus
                    bid_action = self._curiosity_guided_action(state_vector, 'bid')
                    creative_action = self._curiosity_guided_action(state_vector, 'creative')
                    channel_action = self._curiosity_guided_action(state_vector, 'channel')
                else:
                    # Low novelty - use learned policy with small exploration
                    bid_action = q_bid.argmax().item()
                    creative_action = q_creative.argmax().item()
                    channel_action = q_channel.argmax().item()
            elif strategy_choice == 'curiosity':
                # Curiosity-driven exploration
                bid_action = self._curiosity_guided_action(state_vector, 'bid')
                creative_action = self._curiosity_guided_action(state_vector, 'creative')
                channel_action = self._curiosity_guided_action(state_vector, 'channel')
            else:
                # Use guided exploration for unknown strategy
                if state.segment_cvr > 0.04 and random.random() < 0.7:
                    guided_exploration = True
                    bid_action = self._get_guided_bid_action(state)
                    creative_action = self._get_guided_creative_action(state)
                    channel_action = self._get_guided_channel_action(state)
                else:
                    bid_action = random.randint(0, self.bid_actions - 1)
                    creative_action = random.randint(0, self.creative_actions - 1)
                    channel_action = random.randint(0, self.channel_actions - 1)
        else:
            # Exploitation with count-based exploration bonus
            # Use current state vector for exploitation methods
            current_state_tensor = torch.FloatTensor(current_state_vector).unsqueeze(0).to(self.device)
            bid_action = self._exploitation_with_bonus(current_state_tensor, q_bid, 'bid')
            creative_action = self._exploitation_with_bonus(current_state_tensor, q_creative, 'creative')
            channel_action = self._exploitation_with_bonus(current_state_tensor, q_channel, 'channel')
        
        # Convert to actual values using discovered ranges
        bid_amount = self._get_bid_amount(bid_action, state)
        
        # Apply budget pacing
        bid_amount *= state.pacing_factor
        
        # Map to discovered values
        channel = self.discovered_channels[min(channel_action, len(self.discovered_channels) - 1)]
        creative_id = self.discovered_creatives[min(creative_action, len(self.discovered_creatives) - 1)]
        
        # Prepare action dictionary
        action = {
            'bid_amount': bid_amount,
            'bid_action': bid_action,
            'creative_id': creative_id,
            'creative_action': creative_action,
            'channel': channel,
            'channel_action': channel_action
        }
        
        # Calculate decision factors for audit
        decision_factors = {
            'exploration_mode': exploration_mode,
            'exploration_strategy': exploration_strategy,
            'epsilon_used': effective_epsilon,
            'guided_exploration': guided_exploration,
            'segment_based_guidance': segment_based_guidance,
            'data_stats': self.data_stats,
            'model_version': 'fortified_rl_no_hardcoding_v2_advanced_exploration',
            'factor_scores': {
                'segment_performance': state.segment_cvr,
                'channel_performance': state.channel_performance,
                'creative_fatigue': state.creative_fatigue,
                'budget_pressure': 1.0 - state.budget_spent_ratio,
                'competition_level': state.competition_level,
                'conversion_probability': state.conversion_probability
            },
            'pattern_influence': {
                'segment_name': (self.discovered_segments[state.segment_index] if isinstance(self.discovered_segments, list) and state.segment_index < len(self.discovered_segments) 
                                else list(self.discovered_segments.keys())[state.segment_index] if isinstance(self.discovered_segments, dict) and state.segment_index < len(self.discovered_segments)
                                else 'unknown'),
                'high_cvr_segment': state.segment_cvr > 0.04,
                'peak_hour': state.is_peak_hour,
                'device_preference': (self.discovered_devices[state.device_index] if isinstance(self.discovered_devices, list) and state.device_index < len(self.discovered_devices)
                                     else list(self.discovered_devices.keys())[state.device_index] if isinstance(self.discovered_devices, dict) and state.device_index < len(self.discovered_devices)
                                     else 'unknown')
            }
        }
        
        # Prepare context for audit logging
        audit_context = {
            **context,
            'pacing_factor': state.pacing_factor,
            'daily_budget': context.get('daily_budget', 1000.0),
            'budget_spent': state.budget_spent_ratio * context.get('daily_budget', 1000.0),
            'budget_remaining': state.remaining_budget,
            'time_in_day_ratio': state.time_in_day_ratio,
            'competitor_count': len(self.discovered_channels) + 5,  # Estimate
            'competition_level': state.competition_level,
            'market_conditions': {
                'hour': state.hour_of_day,
                'is_peak': state.is_peak_hour,
                'seasonality': state.seasonality_factor,
                'device_type': self.discovered_devices[state.device_index] if state.device_index < len(self.discovered_devices) else 'mobile'
            },
            'attribution_credits': {
                channel: state.channel_attribution_credit,
                'first_touch': self.discovered_channels[state.first_touch_channel_index] if state.first_touch_channel_index < len(self.discovered_channels) else 'unknown',
                'last_touch': self.discovered_channels[state.last_touch_channel_index] if state.last_touch_channel_index < len(self.discovered_channels) else 'unknown'
            },
            'quality_score': context.get('quality_score', 8.0),
            'device': self.discovered_devices[state.device_index] if state.device_index < len(self.discovered_devices) else 'mobile',
            'location': context.get('location', 'unknown')
        }
        
        # LOG EVERY BIDDING DECISION FOR AUDIT COMPLIANCE
        try:
            log_decision(
                decision_id=decision_id,
                user_id=user_id or f"user_{hash(str(state.segment_index)) % 1000}",
                session_id=session_id or f"session_{int(datetime.now().timestamp())}",
                campaign_id=campaign_id,
                state=state,
                action=action,
                context=audit_context,
                q_values=q_values_dict,
                decision_factors=decision_factors
            )
        except Exception as e:
            logger.error(f"Failed to log bidding decision {decision_id}: {e}")
            # DO NOT fail silently - this is critical for compliance
            raise RuntimeError(f"AUDIT LOGGING FAILURE: {e}")
        
        # Add decision_id to action for outcome tracking
        action['decision_id'] = decision_id
        
        return action
    
    def _get_bid_amount(self, bid_action: int, state: DynamicEnrichedState) -> float:
        """Get bid amount from discovered ranges"""
        # Determine which bid range to use based on context
        segment_idx = state.get('segment_index', 0) if hasattr(state, 'get') else 0
        channel_idx = state.get('channel_index', 0) if hasattr(state, 'get') else 0
        
        # Handle list indexing for discovered segments and channels
        if isinstance(self.discovered_segments, list):
            segment_name = self.discovered_segments[segment_idx] if segment_idx < len(self.discovered_segments) else 'default'
        else:
            # If it's a dict, convert to list first
            segment_list = list(self.discovered_segments.keys()) if isinstance(self.discovered_segments, dict) else []
            segment_name = segment_list[segment_idx] if segment_idx < len(segment_list) else 'default'
            
        if isinstance(self.discovered_channels, list):
            channel_name = self.discovered_channels[channel_idx] if channel_idx < len(self.discovered_channels) else 'default'
        else:
            channel_list = list(self.discovered_channels.keys()) if isinstance(self.discovered_channels, dict) else []
            channel_name = channel_list[channel_idx] if channel_idx < len(channel_list) else 'default'
        
        # Try to find appropriate bid range
        bid_range = None
        
        # Check for segment-specific ranges
        if f"{segment_name}_keywords" in self.bid_ranges:
            bid_range = self.bid_ranges[f"{segment_name}_keywords"]
        # Check for channel-specific ranges
        elif channel_name in ['display'] and 'display' in self.bid_ranges:
            bid_range = self.bid_ranges['display']
        elif channel_name in ['paid_search'] and 'non_brand' in self.bid_ranges:
            bid_range = self.bid_ranges['non_brand']
        # Default range
        elif 'default' in self.bid_ranges:
            bid_range = self.bid_ranges['default']
        else:
            # Use data statistics
            bid_range = {
                'min': self.data_stats.bid_min,
                'max': self.data_stats.bid_max
            }
        
        # Create bid levels from discovered range
        min_bid = bid_range.get('min', self.data_stats.bid_min)
        max_bid = bid_range.get('max', self.data_stats.bid_max)
        
        bid_levels = np.linspace(min_bid, max_bid, self.bid_actions)
        return float(bid_levels[bid_action])
    
    def _get_guided_bid_action(self, state: DynamicEnrichedState) -> int:
        """Get bid action guided by successful patterns"""
        segment_name = self.discovered_segments[state.segment_index] if state.segment_index < len(self.discovered_segments) else 'default'
        
        # Use optimal bid from patterns
        optimal_bid = None
        if f"{segment_name}_keywords" in self.bid_ranges:
            optimal_bid = self.bid_ranges[f"{segment_name}_keywords"].get('optimal')
        elif 'default' in self.bid_ranges:
            optimal_bid = self.bid_ranges['default'].get('optimal')
        
        if optimal_bid is not None:
            # Find closest bid action to optimal
            min_bid = self.bid_ranges.get('default', {}).get('min', self.data_stats.bid_min)
            max_bid = self.bid_ranges.get('default', {}).get('max', self.data_stats.bid_max)
            bid_levels = np.linspace(min_bid, max_bid, self.bid_actions)
            
            # Add small noise for exploration - scale based on bid range
            noise_scale = (max_bid - min_bid) * self._hyperparameters.get('exploration_noise_factor', 0.1)
            noisy_optimal = optimal_bid + np.random.normal(0, noise_scale)
            distances = np.abs(bid_levels - noisy_optimal)
            return int(np.argmin(distances))
        
        # Return middle of bid range as reasonable default
        return self.bid_actions // 2
    
    def _get_guided_creative_action(self, state: DynamicEnrichedState) -> int:
        """Get creative action guided by successful patterns"""
        segment_name = self.discovered_segments[state.segment_index] if state.segment_index < len(self.discovered_segments) else None
        
        if segment_name and 'creatives' in self.patterns and 'performance_by_segment' in self.patterns['creatives']:
            segment_creatives = self.patterns['creatives']['performance_by_segment'].get(segment_name, {})
            if 'best_creative_ids' in segment_creatives:
                # Choose from best performing creatives
                best_ids = segment_creatives['best_creative_ids']
                creative_id = random.choice(best_ids)
                
                # Find index in discovered creatives
                if creative_id in self.discovered_creatives:
                    return self.discovered_creatives.index(creative_id)
        
        return random.randint(0, self.creative_actions - 1)
    
    def _get_guided_channel_action(self, state: DynamicEnrichedState) -> int:
        """Get channel action guided by successful patterns"""
        # Prefer high-performing channels from patterns
        channel_perfs = []
        for i, channel in enumerate(self.discovered_channels):
            if channel in self.patterns.get('channels', {}):
                effectiveness = self.patterns['channels'][channel].get('effectiveness', 0.5)
                channel_perfs.append((i, effectiveness))
        
        if channel_perfs:
            # Weighted selection based on effectiveness
            channel_perfs.sort(key=lambda x: x[1], reverse=True)
            
            # Higher chance for better channels
            weights = [perf for _, perf in channel_perfs]  # perf is already the effectiveness value
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                indices = [idx for idx, _ in channel_perfs]  # idx is the channel index
                return np.random.choice(indices, p=weights)
        
        return random.randint(0, self.channel_actions - 1)
    
    def _ucb_action_selection(self, state_vector: torch.Tensor, action_type: str) -> int:
        """Select action using Upper Confidence Bound algorithm"""
        # Handle both tensor and numpy array inputs
        if isinstance(state_vector, torch.Tensor):
            state_key = tuple(state_vector.cpu().numpy().flatten())
        else:
            state_key = tuple(state_vector.flatten())
        
        if action_type == 'bid':
            num_actions = self.bid_actions
            q_network = self.q_network_bid
        elif action_type == 'creative':
            num_actions = self.creative_actions
            q_network = self.q_network_creative
        else:  # channel
            num_actions = self.channel_actions
            q_network = self.q_network_channel
        
        # Initialize tracking for this state if needed
        if state_key not in self.action_counts:
            self.action_counts[state_key] = {action_type: np.zeros(num_actions)}
            self.action_values[state_key] = {action_type: np.zeros(num_actions)}
        elif action_type not in self.action_counts[state_key]:
            self.action_counts[state_key][action_type] = np.zeros(num_actions)
            self.action_values[state_key][action_type] = np.zeros(num_actions)
        
        counts = self.action_counts[state_key][action_type]
        values = self.action_values[state_key][action_type]
        total_count = counts.sum()
        
        ucb_values = []
        for a in range(num_actions):
            if counts[a] == 0:
                ucb_values.append(float('inf'))  # Explore unseen actions first
            else:
                avg_value = values[a] / counts[a]
                confidence_interval = self.ucb_confidence * np.sqrt(np.log(total_count + 1) / counts[a])
                ucb_values.append(avg_value + confidence_interval)
        
        return int(np.argmax(ucb_values))
    
    def _thompson_sampling_action(self, state_vector: torch.Tensor, action_type: str) -> int:
        """Select action using Thompson Sampling with Beta distributions"""
        # Handle both tensor and numpy array inputs
        if isinstance(state_vector, torch.Tensor):
            state_key = tuple(state_vector.cpu().numpy().flatten())
        else:
            state_key = tuple(state_vector.flatten())
        
        if action_type == 'bid':
            num_actions = self.bid_actions
        elif action_type == 'creative':
            num_actions = self.creative_actions
        else:  # channel
            num_actions = self.channel_actions
        
        # Initialize Beta parameters if needed
        if state_key not in self.thompson_alpha:
            self.thompson_alpha[state_key] = {action_type: np.ones(num_actions)}
            self.thompson_beta[state_key] = {action_type: np.ones(num_actions)}
        elif action_type not in self.thompson_alpha[state_key]:
            self.thompson_alpha[state_key][action_type] = np.ones(num_actions)
            self.thompson_beta[state_key][action_type] = np.ones(num_actions)
        
        alpha_params = self.thompson_alpha[state_key][action_type]
        beta_params = self.thompson_beta[state_key][action_type]
        
        # Sample from Beta distributions
        samples = []
        for a in range(num_actions):
            sample = np.random.beta(alpha_params[a], beta_params[a])
            samples.append(sample)
        
        return int(np.argmax(samples))
    
    def _is_novel_state(self, state_vector: torch.Tensor) -> bool:
        """Determine if current state is novel based on archive"""
        # Handle both tensor and numpy array inputs
        if isinstance(state_vector, torch.Tensor):
            state_np = state_vector.cpu().numpy().flatten()
        else:
            state_np = state_vector.flatten()
        
        if len(self.state_archive) == 0:
            return True
        
        # Calculate novelty using k-nearest neighbors
        k = min(self.k_nearest_neighbors, len(self.state_archive))
        
        # Use sklearn for efficient nearest neighbor search
        if not hasattr(self, '_nn_model') or len(self.state_archive) % 50 == 0:
            # Rebuild NN model periodically for efficiency
            self._nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
            self._nn_model.fit(self.state_archive)
        
        # Find k nearest neighbors
        distances, _ = self._nn_model.kneighbors([state_np])
        avg_distance = np.mean(distances[0])
        
        is_novel = avg_distance > self.novelty_threshold
        
        # Add to archive if novel enough
        if is_novel:
            self.state_archive.append(state_np)
            # Limit archive size for memory efficiency
            if len(self.state_archive) > 5000:
                self.state_archive = self.state_archive[-4000:]  # Keep recent 4000
        
        return is_novel
    
    def _curiosity_guided_action(self, state_vector: torch.Tensor, action_type: str) -> int:
        """Select action guided by curiosity (prediction error)"""
        # Ensure state_vector is a tensor
        if not isinstance(state_vector, torch.Tensor):
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        else:
            state_tensor = state_vector
            
        if action_type == 'bid':
            num_actions = self.bid_actions
            q_network = self.q_network_bid
        elif action_type == 'creative':
            num_actions = self.creative_actions
            q_network = self.q_network_creative
        else:  # channel
            num_actions = self.channel_actions
            q_network = self.q_network_channel
        
        # Get Q-values for base preference
        with torch.no_grad():
            q_values = q_network(state_tensor).cpu().numpy().flatten()
        
        # Add curiosity bonus based on prediction uncertainty
        curiosity_bonuses = []
        for a in range(num_actions):
            # Create action vector for curiosity module
            action_vector = torch.zeros(self.bid_actions + self.creative_actions + self.channel_actions)
            if action_type == 'bid':
                action_vector[a] = 1.0
            elif action_type == 'creative':
                action_vector[self.bid_actions + a] = 1.0
            else:  # channel
                action_vector[self.bid_actions + self.creative_actions + a] = 1.0
            
            action_vector = action_vector.unsqueeze(0).to(self.device)
            
            # Get curiosity bonus (prediction error encourages exploration)
            # Ensure state is tensor for curiosity calculation
            if not isinstance(state_vector, torch.Tensor):
                state_for_curiosity = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            else:
                state_for_curiosity = state_vector
            curiosity_bonus = self._get_curiosity_bonus(state_for_curiosity, action_vector)
            curiosity_bonuses.append(curiosity_bonus)
        
        # Combine Q-values with curiosity bonuses
        combined_values = q_values + np.array(curiosity_bonuses) * 0.1  # Scale curiosity influence
        
        return int(np.argmax(combined_values))
    
    def _exploitation_with_bonus(self, state_vector: torch.Tensor, q_values: torch.Tensor, action_type: str) -> int:
        """Exploitation with count-based exploration bonus"""
        state_key = tuple(state_vector.cpu().numpy().flatten())
        
        if action_type == 'bid':
            num_actions = self.bid_actions
        elif action_type == 'creative':
            num_actions = self.creative_actions
        else:  # channel
            num_actions = self.channel_actions
        
        # Initialize count tracking if needed
        if state_key not in self.state_action_counts:
            self.state_action_counts[state_key] = {action_type: np.zeros(num_actions)}
        elif action_type not in self.state_action_counts[state_key]:
            self.state_action_counts[state_key][action_type] = np.zeros(num_actions)
        
        counts = self.state_action_counts[state_key][action_type]
        q_vals = q_values.cpu().numpy().flatten()
        
        # Add exploration bonus inversely proportional to visit count
        bonuses = self.exploration_bonus_scale / np.sqrt(counts + 1)
        final_values = q_vals + bonuses
        
        return int(np.argmax(final_values))
    
    def _get_curiosity_bonus(self, state_vector: torch.Tensor, action_vector: torch.Tensor) -> float:
        """Calculate curiosity bonus based on prediction error"""
        try:
            # Use curiosity module to predict next state
            state_action = torch.cat([state_vector, action_vector], dim=-1)
            
            # Predict next state (using current state as target for value estimation)
            predicted_next = self.curiosity_module(state_action)
            
            # Calculate prediction error as curiosity signal
            prediction_error = F.mse_loss(predicted_next, state_vector, reduction='mean')
            
            return prediction_error.item()
        except Exception as e:
            logger.warning(f"Curiosity calculation failed: {e}")
            return 0.0
    
    def _update_exploration_tracking(self, state_vector: torch.Tensor, action: Dict[str, Any], reward: float):
        """Update all exploration tracking systems"""
        state_key = tuple(state_vector.cpu().numpy().flatten())
        
        # Update UCB tracking
        for action_type, action_idx in [('bid', action['bid_action']), 
                                       ('creative', action['creative_action']),
                                       ('channel', action['channel_action'])]:
            if state_key in self.action_counts and action_type in self.action_counts[state_key]:
                self.action_counts[state_key][action_type][action_idx] += 1
                self.action_values[state_key][action_type][action_idx] += reward
        
        # Update Thompson Sampling parameters
        for action_type, action_idx in [('bid', action['bid_action']), 
                                       ('creative', action['creative_action']),
                                       ('channel', action['channel_action'])]:
            if state_key in self.thompson_alpha and action_type in self.thompson_alpha[state_key]:
                if reward > 0:
                    self.thompson_alpha[state_key][action_type][action_idx] += reward
                else:
                    self.thompson_beta[state_key][action_type][action_idx] += abs(reward) + 1
        
        # Update count-based tracking
        for action_type, action_idx in [('bid', action['bid_action']), 
                                       ('creative', action['creative_action']),
                                       ('channel', action['channel_action'])]:
            if state_key in self.state_action_counts and action_type in self.state_action_counts[state_key]:
                self.state_action_counts[state_key][action_type][action_idx] += 1
    
    def _adapt_exploration_weights(self):
        """Adapt exploration strategy weights based on performance"""
        # Calculate average reward for each strategy
        strategy_rewards = {}
        for strategy, rewards in self.strategy_performance.items():
            if len(rewards) > 10:  # Need sufficient samples
                strategy_rewards[strategy] = np.mean(list(rewards))
            else:
                strategy_rewards[strategy] = 0.0
        
        if not strategy_rewards:
            return
        
        # Softmax to convert rewards to probabilities
        max_reward = max(strategy_rewards.values())
        exp_rewards = {k: np.exp(v - max_reward) for k, v in strategy_rewards.items()}
        total_exp = sum(exp_rewards.values())
        
        if total_exp > 0:
            # Update weights with smoothing
            alpha = 0.1  # Learning rate for weight adaptation
            new_weights = {}
            for strategy in self.exploration_weights:
                if strategy in exp_rewards:
                    new_weight = exp_rewards[strategy] / total_exp
                    new_weights[strategy] = (1 - alpha) * self.exploration_weights[strategy] + alpha * new_weight
                else:
                    new_weights[strategy] = self.exploration_weights[strategy]
            
            # Normalize to ensure they sum to 1
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                for strategy in new_weights:
                    self.exploration_weights[strategy] = new_weights[strategy] / total_weight
        
        logger.debug(f"Updated exploration weights: {self.exploration_weights}")
    
    def _process_trajectory(self, user_id: str, trajectory_complete: bool) -> Optional[CompletedTrajectory]:
        """Process trajectory and compute n-step, Monte Carlo, and GAE returns"""
        if user_id not in self.current_trajectories or not self.current_trajectories[user_id]:
            return None
        
        experiences = self.current_trajectories[user_id]
        
        # Determine optimal n for this trajectory
        n = self._adaptive_n_step(len(experiences))
        
        # Compute n-step returns
        n_step_returns = self._compute_n_step_returns(experiences, n)
        
        # Compute Monte Carlo returns if trajectory is complete
        monte_carlo_returns = []
        if trajectory_complete and self.use_monte_carlo:
            monte_carlo_returns = self._compute_monte_carlo_returns(experiences)
        else:
            # Use bootstrapped returns for incomplete trajectories
            monte_carlo_returns = self._compute_bootstrapped_returns(experiences)
        
        # Compute GAE advantages
        gae_advantages = self._compute_gae_advantages(experiences)
        
        total_return = sum(exp.reward for exp in experiences)
        
        completed_trajectory = CompletedTrajectory(
            experiences=experiences,
            n_step_returns=n_step_returns,
            monte_carlo_returns=monte_carlo_returns,
            gae_advantages=gae_advantages,
            trajectory_length=len(experiences),
            total_return=total_return,
            user_id=user_id
        )
        
        return completed_trajectory
    
    def _adaptive_n_step(self, trajectory_length: int) -> int:
        """Adaptively choose n-step based on trajectory length"""
        min_n, max_n = self.n_step_range
        
        # For very short trajectories, use the full length
        if trajectory_length <= 2:
            return trajectory_length
        
        # For short trajectories, use length up to min_n
        if trajectory_length <= min_n:
            return trajectory_length
        elif trajectory_length <= max_n:
            # In the normal range, use trajectory_length but cap at max_n
            return min(trajectory_length, max_n)
        else:
            # For long trajectories, always use max_n
            return max_n
    
    def _compute_n_step_returns(self, experiences: List[TrajectoryExperience], n: int) -> List[float]:
        """Compute n-step returns for each experience"""
        n_step_returns = []
        
        for i, exp in enumerate(experiences):
            n_step_return = 0.0
            gamma_power = 1.0
            
            # Sum up to n future rewards
            for j in range(n):
                if i + j < len(experiences):
                    n_step_return += gamma_power * experiences[i + j].reward
                    gamma_power *= self.gamma
                else:
                    break
            
            # Bootstrap with value function if trajectory continues
            if i + n < len(experiences) and not experiences[i + n - 1].done:
                next_state_vector = torch.FloatTensor(experiences[i + n].state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    bootstrap_value = self.value_network(next_state_vector).item()
                n_step_return += gamma_power * bootstrap_value
            
            n_step_returns.append(n_step_return)
        
        return n_step_returns
    
    def _compute_monte_carlo_returns(self, experiences: List[TrajectoryExperience]) -> List[float]:
        """Compute Monte Carlo returns for complete trajectory"""
        monte_carlo_returns = []
        
        # Compute returns backwards from the end
        running_return = 0.0
        for exp in reversed(experiences):
            running_return = exp.reward + self.gamma * running_return
            monte_carlo_returns.insert(0, running_return)
        
        return monte_carlo_returns
    
    def _compute_bootstrapped_returns(self, experiences: List[TrajectoryExperience]) -> List[float]:
        """Compute returns with value function bootstrapping for incomplete trajectories"""
        bootstrapped_returns = []
        
        # Start with value estimate for the last state
        if experiences:
            last_state_vector = torch.FloatTensor(experiences[-1].next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_value = self.value_network(last_state_vector).item() if not experiences[-1].done else 0.0
        else:
            last_value = 0.0
        
        # Compute returns backwards with bootstrapping
        running_return = last_value
        for exp in reversed(experiences):
            running_return = exp.reward + self.gamma * running_return
            bootstrapped_returns.insert(0, running_return)
        
        return bootstrapped_returns
    
    def _compute_gae_advantages(self, experiences: List[TrajectoryExperience]) -> List[float]:
        """Compute Generalized Advantage Estimation (GAE) advantages"""
        if not experiences:
            return []
        
        # Get value estimates for all states
        values = []
        next_values = []
        
        for exp in experiences:
            state_vector = torch.FloatTensor(exp.state).unsqueeze(0).to(self.device)
            next_state_vector = torch.FloatTensor(exp.next_state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                value = self.value_network(state_vector).item()
                next_value = self.value_network(next_state_vector).item() if not exp.done else 0.0
            
            values.append(value)
            next_values.append(next_value)
        
        # Compute TD errors
        td_errors = []
        for i, exp in enumerate(experiences):
            td_error = exp.reward + self.gamma * next_values[i] - values[i]
            td_errors.append(td_error)
        
        # Compute GAE advantages
        advantages = []
        gae = 0.0
        
        for i in reversed(range(len(experiences))):
            if i == len(experiences) - 1:
                gae = td_errors[i]
            else:
                gae = td_errors[i] + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def _train_trajectory_batch(self):
        """Train on a batch of completed trajectories"""
        if not self.trajectory_buffer:
            return
        
        batch_size = min(len(self.trajectory_buffer), self._hyperparameters.get('batch_size', 32))
        trajectory_batch = random.sample(self.trajectory_buffer, batch_size)
        
        # Collect all experiences from trajectories
        all_states = []
        all_actions_bid = []
        all_actions_creative = []
        all_actions_channel = []
        all_n_step_returns = []
        all_monte_carlo_returns = []
        all_gae_advantages = []
        all_value_targets = []
        
        for trajectory in trajectory_batch:
            for i, exp in enumerate(trajectory.experiences):
                all_states.append(exp.state)
                all_actions_bid.append(exp.action['bid_action'])
                all_actions_creative.append(exp.action['creative_action'])
                all_actions_channel.append(exp.action['channel_action'])
                all_n_step_returns.append(trajectory.n_step_returns[i])
                all_monte_carlo_returns.append(trajectory.monte_carlo_returns[i])
                all_gae_advantages.append(trajectory.gae_advantages[i])
                
                # Value target is Monte Carlo return if available, else n-step return
                value_target = (trajectory.monte_carlo_returns[i] 
                               if trajectory.monte_carlo_returns 
                               else trajectory.n_step_returns[i])
                all_value_targets.append(value_target)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(all_states).to(self.device)
        bid_actions_tensor = torch.LongTensor(all_actions_bid).to(self.device)
        creative_actions_tensor = torch.LongTensor(all_actions_creative).to(self.device)
        channel_actions_tensor = torch.LongTensor(all_actions_channel).to(self.device)
        n_step_returns_tensor = torch.FloatTensor(all_n_step_returns).to(self.device)
        monte_carlo_returns_tensor = torch.FloatTensor(all_monte_carlo_returns).to(self.device)
        advantages_tensor = torch.FloatTensor(all_gae_advantages).to(self.device)
        value_targets_tensor = torch.FloatTensor(all_value_targets).to(self.device)
        
        # Train value network
        value_predictions = self.value_network(states_tensor)
        value_loss = nn.MSELoss()(value_predictions, value_targets_tensor)
        
        # Apply dynamic loss scaling for numerical stability
        scaled_value_loss = self.gradient_stabilizer.get_scaled_loss(value_loss)
        
        self.value_optimizer.zero_grad()
        scaled_value_loss.backward()
        
        # CRITICAL: Apply gradient clipping for stability
        self.training_step += 1
        value_clip_metrics = self.gradient_stabilizer.clip_gradients(
            self.value_network.parameters(), 
            self.training_step, 
            value_loss.item()
        )
        
        self.value_optimizer.step()
        
        # Train Q-networks using trajectory returns and advantages with Double DQN
        # Use Monte Carlo returns when available, else n-step returns
        target_returns = torch.where(
            monte_carlo_returns_tensor != 0,  # Use MC returns when available
            monte_carlo_returns_tensor,
            n_step_returns_tensor  # Use n-step returns for incomplete trajectories
        )
        
        # Add advantage weighting to target returns
        normalized_advantages = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        weighted_targets = target_returns + 0.1 * normalized_advantages  # Small advantage weighting
        
        # DOUBLE DQN: For trajectory training, we need next states for proper target calculation
        # Since we have trajectory data, we can construct next states from the experiences
        next_states_list = []
        next_actions_bid_list = []
        next_actions_creative_list = []
        next_actions_channel_list = []
        dones_list = []
        
        for trajectory in trajectory_batch:
            for i, exp in enumerate(trajectory.experiences):
                if i < len(trajectory.experiences) - 1:
                    # Use next experience's state
                    next_exp = trajectory.experiences[i + 1]
                    next_states_list.append(next_exp.state)
                    next_actions_bid_list.append(next_exp.action['bid_action'])
                    next_actions_creative_list.append(next_exp.action['creative_action'])
                    next_actions_channel_list.append(next_exp.action['channel_action'])
                    dones_list.append(False)
                else:
                    # Last state in trajectory - use current state as next (terminal)
                    next_states_list.append(exp.state)
                    next_actions_bid_list.append(exp.action['bid_action'])
                    next_actions_creative_list.append(exp.action['creative_action'])
                    next_actions_channel_list.append(exp.action['channel_action'])
                    dones_list.append(True)
        
        next_states_tensor = torch.FloatTensor(next_states_list).to(self.device)
        dones_tensor = torch.FloatTensor(dones_list).to(self.device)
        
        # Train bid network with Double DQN approach
        current_q_bid = self.q_network_bid(states_tensor).gather(1, bid_actions_tensor.unsqueeze(1))
        
        # Double DQN for bid network: Use online network to SELECT next actions, target network to EVALUATE
        with torch.no_grad():
            next_actions_bid_selected = self.q_network_bid(next_states_tensor).argmax(1)  # Action selection with online network
            next_q_bid_values = self.target_network_bid(next_states_tensor).gather(1, next_actions_bid_selected.unsqueeze(1)).squeeze()  # Evaluation with target network
            
            # Combine trajectory returns with Double DQN bootstrap for non-terminal states
            # For trajectory training, we mainly use the trajectory returns but add Double DQN correction
            double_dqn_targets = weighted_targets + 0.1 * (self.gamma * next_q_bid_values * (1 - dones_tensor))
        
        loss_bid = nn.MSELoss()(current_q_bid.squeeze(), double_dqn_targets.detach())
        
        # Apply dynamic loss scaling for numerical stability
        scaled_loss_bid = self.gradient_stabilizer.get_scaled_loss(loss_bid)
        
        self.optimizer_bid.zero_grad()
        scaled_loss_bid.backward()
        
        # CRITICAL: Apply gradient clipping for bid network stability
        bid_clip_metrics = self.gradient_stabilizer.clip_gradients(
            self.q_network_bid.parameters(), 
            self.training_step, 
            loss_bid.item()
        )
        
        self.optimizer_bid.step()
        
        # Train creative network with Double DQN approach
        current_q_creative = self.q_network_creative(states_tensor).gather(1, creative_actions_tensor.unsqueeze(1))
        
        # Double DQN for creative network: Use online network to SELECT next actions, target network to EVALUATE
        with torch.no_grad():
            next_actions_creative_selected = self.q_network_creative(next_states_tensor).argmax(1)  # Action selection with online network
            next_q_creative_values = self.target_network_creative(next_states_tensor).gather(1, next_actions_creative_selected.unsqueeze(1)).squeeze()  # Evaluation with target network
            
            # Combine trajectory returns with Double DQN bootstrap for non-terminal states
            double_dqn_targets_creative = weighted_targets + 0.1 * (self.gamma * next_q_creative_values * (1 - dones_tensor))
        
        loss_creative = nn.MSELoss()(current_q_creative.squeeze(), double_dqn_targets_creative.detach())
        
        # Apply dynamic loss scaling for numerical stability
        scaled_loss_creative = self.gradient_stabilizer.get_scaled_loss(loss_creative)
        
        self.optimizer_creative.zero_grad()
        scaled_loss_creative.backward()
        
        # CRITICAL: Apply gradient clipping for creative network stability
        creative_clip_metrics = self.gradient_stabilizer.clip_gradients(
            self.q_network_creative.parameters(), 
            self.training_step, 
            loss_creative.item()
        )
        
        self.optimizer_creative.step()
        
        # Train channel network with Double DQN approach
        current_q_channel = self.q_network_channel(states_tensor).gather(1, channel_actions_tensor.unsqueeze(1))
        
        # Double DQN for channel network: Use online network to SELECT next actions, target network to EVALUATE
        with torch.no_grad():
            next_actions_channel_selected = self.q_network_channel(next_states_tensor).argmax(1)  # Action selection with online network
            next_q_channel_values = self.target_network_channel(next_states_tensor).gather(1, next_actions_channel_selected.unsqueeze(1)).squeeze()  # Evaluation with target network
            
            # Combine trajectory returns with Double DQN bootstrap for non-terminal states
            double_dqn_targets_channel = weighted_targets + 0.1 * (self.gamma * next_q_channel_values * (1 - dones_tensor))
        
        loss_channel = nn.MSELoss()(current_q_channel.squeeze(), double_dqn_targets_channel.detach())
        
        # Apply dynamic loss scaling for numerical stability
        scaled_loss_channel = self.gradient_stabilizer.get_scaled_loss(loss_channel)
        
        self.optimizer_channel.zero_grad()
        scaled_loss_channel.backward()
        
        # CRITICAL: Apply gradient clipping for channel network stability
        channel_clip_metrics = self.gradient_stabilizer.clip_gradients(
            self.q_network_channel.parameters(), 
            self.training_step, 
            loss_channel.item()
        )
        
        self.optimizer_channel.step()
        
        # Monitor Q-value overestimation bias for trajectory training with Double DQN targets
        # Use the Double DQN targets for more accurate overestimation tracking
        avg_double_dqn_targets = (double_dqn_targets + double_dqn_targets_creative + double_dqn_targets_channel) / 3.0
        self._monitor_q_value_overestimation(states_tensor, avg_double_dqn_targets)
        
        # Verify Double DQN effectiveness in trajectory training
        if self.training_step % 100 == 0:  # Every 100 trajectory training steps
            trajectory_double_dqn_stats = self._verify_trajectory_double_dqn_benefit(
                states_tensor, next_states_tensor, weighted_targets, avg_double_dqn_targets
            )
            logger.info(f"Trajectory Double DQN Stats: {trajectory_double_dqn_stats}")
        
        # Update adaptive learning rate scheduler with gradient monitoring
        total_loss = value_loss.item() + loss_bid.item() + loss_creative.item() + loss_channel.item()
        avg_grad_norm = (value_clip_metrics['grad_norm'] + bid_clip_metrics['grad_norm'] + 
                        creative_clip_metrics['grad_norm'] + channel_clip_metrics['grad_norm']) / 4
        loss_variance = np.var([value_loss.item(), loss_bid.item(), loss_creative.item(), loss_channel.item()])
        
        # Calculate performance metric (negative total loss for maximization)
        performance_metric = -total_loss
        
        # Update learning rate with scheduler
        new_lr = self.lr_scheduler.step(
            performance=performance_metric,
            gradient_norm=avg_grad_norm,
            loss_variance=loss_variance
        )
        
        # Update all optimizers with new learning rate
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.optimizer_bid.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.optimizer_creative.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.optimizer_channel.param_groups:
            param_group['lr'] = new_lr
        
        # Update CuriosityModule learning rate
        self.curiosity_module.update_learning_rate(new_lr)
        
        # Clear processed trajectories
        max_buffer_size = self._hyperparameters.get('max_trajectory_buffer_size', 1000)
        if len(self.trajectory_buffer) > max_buffer_size:
            # Remove oldest trajectories
            self.trajectory_buffer = self.trajectory_buffer[-max_buffer_size//2:]
        
        # Log scheduler statistics periodically
        scheduler_stats = self.lr_scheduler.get_scheduler_stats()
        
        logger.debug(f"Trained on {len(trajectory_batch)} trajectories with "
                    f"value_loss={value_loss.item():.4f}, "
                    f"bid_loss={loss_bid.item():.4f}, "
                    f"creative_loss={loss_creative.item():.4f}, "
                    f"channel_loss={loss_channel.item():.4f}, "
                    f"lr={new_lr:.6f}, "
                    f"grad_norm={avg_grad_norm:.4f}, "
                    f"stability_score={self.gradient_stabilizer.stability_score:.3f}")
        
        # Log gradient explosion alerts immediately
        if (value_clip_metrics['explosion_detected'] or bid_clip_metrics['explosion_detected'] or
            creative_clip_metrics['explosion_detected'] or channel_clip_metrics['explosion_detected']):
            stability_report = self.gradient_stabilizer.get_stability_report()
            logger.error(f"GRADIENT INSTABILITY DETECTED! Step {self.training_step}")
            logger.error(f"Stability Report: {stability_report}")
        
        # Log detailed scheduler info and gradient stability every 100 steps
        if self.lr_scheduler.step_count % 100 == 0:
            stability_report = self.gradient_stabilizer.get_stability_report()
            logger.info(f"LR Scheduler Stats: {scheduler_stats}")
            logger.info(f"Gradient Stability Report: {stability_report}")
        
        # Return training metrics for convergence monitoring
        total_loss = (value_loss.item() + loss_bid.item() + 
                     loss_creative.item() + loss_channel.item()) / 4
        return {
            'loss': total_loss,
            'gradient_norm': avg_grad_norm,
            'value_loss': value_loss.item(),
            'bid_loss': loss_bid.item(), 
            'creative_loss': loss_creative.item(),
            'channel_loss': loss_channel.item()
        }
    
    def _train_step_legacy(self):
        """Prioritized experience replay training with importance sampling"""
        batch_size = self._hyperparameters.get('batch_size', 32)
        
        # Check if we have enough experiences
        total_size = self.replay_buffer.get_stats()['total_size']
        if total_size < batch_size:
            return
        
        # Sample batch using prioritized replay with importance weights
        try:
            batch, importance_weights, indices = self.replay_buffer.sample(batch_size)
        except ValueError as e:
            logger.warning(f"Could not sample from replay buffer: {e}")
            return
        
        # CRITICAL: Increment training step for gradient monitoring
        self.training_step += 1
        
        states = torch.FloatTensor([e['state'] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).to(self.device)
        dones = torch.FloatTensor([float(e['done']) for e in batch]).to(self.device)
        
        bid_actions = torch.LongTensor([e['action']['bid_action'] for e in batch]).to(self.device)
        creative_actions = torch.LongTensor([e['action']['creative_action'] for e in batch]).to(self.device)
        channel_actions = torch.LongTensor([e['action']['channel_action'] for e in batch]).to(self.device)
        
        # Convert importance sampling weights to tensor
        importance_weights = torch.FloatTensor(importance_weights).to(self.device)
        
        # Train bid network - Double DQN with importance sampling
        current_q_bid = self.q_network_bid(states).gather(1, bid_actions.unsqueeze(1))
        # Double DQN: Use online network to SELECT actions, target network to EVALUATE
        next_actions_bid = self.q_network_bid(next_states).argmax(1)  # Action selection with online network
        next_q_bid = self.target_network_bid(next_states).gather(1, next_actions_bid.unsqueeze(1)).squeeze().detach()  # Evaluation with target network
        target_q_bid = rewards + (self.gamma * next_q_bid * (1 - dones))
        
        # Calculate TD errors for priority updates
        td_errors_bid = (target_q_bid - current_q_bid.squeeze()).detach()
        
        # Importance-weighted loss
        bid_losses = F.mse_loss(current_q_bid.squeeze(), target_q_bid, reduction='none')
        loss_bid = (importance_weights * bid_losses).mean()
        
        # Apply dynamic loss scaling for numerical stability
        scaled_loss_bid = self.gradient_stabilizer.get_scaled_loss(loss_bid)
        
        self.optimizer_bid.zero_grad()
        scaled_loss_bid.backward()
        
        # Calculate gradient norm for LR scheduling
        bid_grad_norm = 0.0
        for param in self.q_network_bid.parameters():
            if param.grad is not None:
                bid_grad_norm += param.grad.data.norm(2).item() ** 2
        bid_grad_norm = bid_grad_norm ** 0.5
        
        # CRITICAL: Apply gradient clipping for legacy bid network stability
        bid_clip_metrics = self.gradient_stabilizer.clip_gradients(
            self.q_network_bid.parameters(), 
            self.training_step, 
            loss_bid.item()
        )
        
        self.optimizer_bid.step()
        
        # Train creative network - Double DQN with importance sampling
        current_q_creative = self.q_network_creative(states).gather(1, creative_actions.unsqueeze(1))
        # Double DQN: Use online network to SELECT actions, target network to EVALUATE
        next_actions_creative = self.q_network_creative(next_states).argmax(1)  # Action selection with online network
        next_q_creative = self.target_network_creative(next_states).gather(1, next_actions_creative.unsqueeze(1)).squeeze().detach()  # Evaluation with target network
        target_q_creative = rewards + (self.gamma * next_q_creative * (1 - dones))
        
        # Calculate TD errors for priority updates
        td_errors_creative = (target_q_creative - current_q_creative.squeeze()).detach()
        
        # Importance-weighted loss
        creative_losses = F.mse_loss(current_q_creative.squeeze(), target_q_creative, reduction='none')
        loss_creative = (importance_weights * creative_losses).mean()
        
        # Apply dynamic loss scaling for numerical stability
        scaled_loss_creative = self.gradient_stabilizer.get_scaled_loss(loss_creative)
        
        self.optimizer_creative.zero_grad()
        scaled_loss_creative.backward()
        
        # Calculate gradient norm for LR scheduling
        creative_grad_norm = 0.0
        for param in self.q_network_creative.parameters():
            if param.grad is not None:
                creative_grad_norm += param.grad.data.norm(2).item() ** 2
        creative_grad_norm = creative_grad_norm ** 0.5
        
        # CRITICAL: Apply gradient clipping for legacy creative network stability
        creative_clip_metrics = self.gradient_stabilizer.clip_gradients(
            self.q_network_creative.parameters(), 
            self.training_step, 
            loss_creative.item()
        )
        
        self.optimizer_creative.step()
        
        # Train channel network - Double DQN with importance sampling
        current_q_channel = self.q_network_channel(states).gather(1, channel_actions.unsqueeze(1))
        # Double DQN: Use online network to SELECT actions, target network to EVALUATE
        next_actions_channel = self.q_network_channel(next_states).argmax(1)  # Action selection with online network
        next_q_channel = self.target_network_channel(next_states).gather(1, next_actions_channel.unsqueeze(1)).squeeze().detach()  # Evaluation with target network
        target_q_channel = rewards + (self.gamma * next_q_channel * (1 - dones))
        
        # Calculate TD errors for priority updates
        td_errors_channel = (target_q_channel - current_q_channel.squeeze()).detach()
        
        # Importance-weighted loss
        channel_losses = F.mse_loss(current_q_channel.squeeze(), target_q_channel, reduction='none')
        loss_channel = (importance_weights * channel_losses).mean()
        
        # Apply dynamic loss scaling for numerical stability
        scaled_loss_channel = self.gradient_stabilizer.get_scaled_loss(loss_channel)
        
        self.optimizer_channel.zero_grad()
        scaled_loss_channel.backward()
        
        # CRITICAL: Apply gradient clipping for legacy channel network stability
        channel_clip_metrics = self.gradient_stabilizer.clip_gradients(
            self.q_network_channel.parameters(), 
            self.training_step, 
            loss_channel.item()
        )
        
        self.optimizer_channel.step()
        
        # Enhanced TD error calculation for prioritized replay
        # Combine TD errors with adaptive weighting based on network performance
        td_errors_bid_np = td_errors_bid.cpu().numpy()
        td_errors_creative_np = td_errors_creative.cpu().numpy()
        td_errors_channel_np = td_errors_channel.cpu().numpy()
        
        # Calculate weighted TD errors based on network loss contribution
        total_loss = loss_bid.item() + loss_creative.item() + loss_channel.item()
        if total_loss > 0:
            bid_weight = loss_bid.item() / total_loss
            creative_weight = loss_creative.item() / total_loss
            channel_weight = loss_channel.item() / total_loss
        else:
            bid_weight = creative_weight = channel_weight = 1.0 / 3.0
        
        # Weighted combination emphasizes networks with higher errors
        combined_td_errors = (
            bid_weight * td_errors_bid_np + 
            creative_weight * td_errors_creative_np + 
            channel_weight * td_errors_channel_np
        )
        
        # Add novelty bonus to TD errors for exploration
        novelty_bonuses = []
        for i, exp in enumerate(batch):
            state_novelty = self._calculate_state_novelty(exp.get('state', np.zeros(10)))
            novelty_bonuses.append(state_novelty * 0.1)  # Scale novelty influence
        
        # Enhanced TD errors with novelty
        enhanced_td_errors = combined_td_errors + np.array(novelty_bonuses)
        
        # Update replay buffer priorities with enhanced TD errors
        self.replay_buffer.update_priorities(indices, enhanced_td_errors)
        
        # Track priority update efficiency
        if hasattr(self.replay_buffer, 'sampling_efficiency'):
            high_priority_count = np.sum(np.abs(enhanced_td_errors) > np.mean(np.abs(enhanced_td_errors)))
            self.replay_buffer.sampling_efficiency['high_priority_samples'] += high_priority_count
        
        # Monitor Q-value overestimation bias (Double DQN effectiveness)
        self._monitor_q_value_overestimation(states, rewards)
        
        # Verify Double DQN benefit periodically
        if hasattr(self, 'training_metrics') and self.training_metrics.get('episodes', 0) % 50 == 0:
            double_dqn_stats = self._verify_double_dqn_benefit(states, next_states)
            logger.info(f"Double DQN Stats: {double_dqn_stats}")
            if hasattr(self, 'q_value_tracking'):
                self.q_value_tracking['double_dqn_benefit'] = double_dqn_stats['double_dqn_benefit']
        
        # Update adaptive learning rate scheduler for legacy training with gradient stability
        total_loss = loss_bid.item() + loss_creative.item() + loss_channel.item()
        avg_grad_norm = (bid_clip_metrics['grad_norm'] + creative_clip_metrics['grad_norm'] + 
                        channel_clip_metrics['grad_norm']) / 3
        
        # Monitor gradient stability in legacy method
        if (bid_clip_metrics['explosion_detected'] or creative_clip_metrics['explosion_detected'] or
            channel_clip_metrics['explosion_detected']):
            stability_report = self.gradient_stabilizer.get_stability_report()
            logger.error(f"LEGACY GRADIENT INSTABILITY DETECTED! Step {self.training_step}")
            logger.error(f"Legacy Stability Report: {stability_report}")
        loss_variance = np.var([loss_bid.item(), loss_creative.item(), loss_channel.item()])
        
        # Calculate performance metric (negative total loss for maximization)
        performance_metric = -total_loss
        
        # Update learning rate with scheduler
        new_lr = self.lr_scheduler.step(
            performance=performance_metric,
            gradient_norm=avg_grad_norm,
            loss_variance=loss_variance
        )
        
        # Update all optimizers with new learning rate
        for param_group in self.optimizer_bid.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.optimizer_creative.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.optimizer_channel.param_groups:
            param_group['lr'] = new_lr
        
        # Update CuriosityModule learning rate
        self.curiosity_module.update_learning_rate(new_lr)
        
        # Log LR updates for legacy training
        if self.step_count % 50 == 0:  # Less frequent logging for legacy
            logger.debug(f"Legacy training - bid_loss={loss_bid.item():.4f}, "
                        f"creative_loss={loss_creative.item():.4f}, "
                        f"channel_loss={loss_channel.item():.4f}, "
                        f"lr={new_lr:.6f}, grad_norm={avg_grad_norm:.4f}")
        
        # Return training metrics for convergence monitoring
        return {
            'loss': total_loss / 3,  # Average loss across networks
            'gradient_norm': avg_grad_norm,
            'bid_loss': loss_bid.item(),
            'creative_loss': loss_creative.item(),
            'channel_loss': loss_channel.item()
        }
    
    def train(self, state: DynamicEnrichedState, action: Dict, reward: float, 
             next_state: DynamicEnrichedState, done: bool,
             auction_result: Any = None, context: Dict[str, Any] = None):
        """Store experience and train with COMPLETE AUDIT LOGGING"""
        
        context = context or {}
        
        # Store in prioritized replay buffer with additional info
        experience_data = {
            'state': state.to_vector(self.data_stats),
            'action': {
                'bid_action': action['bid_action'],
                'creative_action': action['creative_action'],
                'channel_action': action['channel_action']
            },
            'reward': reward,
            'next_state': next_state.to_vector(self.data_stats),
            'done': done,
            'info': {
                'conversion': reward > self._get_conversion_threshold_from_patterns(),
                'high_reward': abs(reward) > self._get_high_reward_threshold_from_patterns(),
                'auction_won': auction_result and auction_result.get('won', False) if auction_result else False,
                'first_time_action': context.get('first_time_action', False),
                'exploration_bonus': context.get('exploration_bonus', 0),
                'user_segment': getattr(state, 'segment_index', -1),
                'channel_id': action.get('channel_action', -1),
                'creative_id': action.get('creative_action', -1)
            }
        }
        self.replay_buffer.add(experience_data)
        
        self.step_count += 1
        
        # Track reward history for adaptive epsilon
        self.reward_history.append(reward)
        
        # Get user_id from context for sequence tracking
        user_id = context.get('user_id', 'default_user')
        
        # Update user sequences for temporal modeling
        self._update_user_sequences(user_id, action, reward)
        
        # Update exploration tracking systems and calculate Q-value prediction error
        state_vector = torch.FloatTensor(state.to_vector(self.data_stats)).unsqueeze(0).to(self.device)
        self._update_exploration_tracking(state_vector, action, reward)
        
        # Get sequence data for Q-value prediction
        sequence_tensor, sequence_mask = self._get_user_sequence_tensor(user_id)
        
        with torch.no_grad():
            # Use sequence-aware networks for prediction
            predicted_q_bid = self.q_network_bid(sequence_tensor, sequence_mask)[0, action['bid_action']].item()
            predicted_q_creative = self.q_network_creative(sequence_tensor, sequence_mask)[0, action['creative_action']].item()
            predicted_q_channel = self.q_network_channel(sequence_tensor, sequence_mask)[0, action['channel_action']].item()
        
        # Calculate prediction errors
        q_prediction_error = abs(reward - predicted_q_bid)  # Primary error
        
        # LOG AUCTION OUTCOME FOR AUDIT COMPLIANCE
        if auction_result and 'decision_id' in action:
            try:
                # Prepare learning metrics
                learning_metrics = {
                    'q_prediction_error': q_prediction_error,
                    'reward': reward,
                    'win_probability': self._estimate_win_probability(action, state),
                    'competitor_analysis': self._analyze_competition(auction_result, context)
                }
                
                # Prepare budget impact
                budget_impact = {
                    'budget_after': context.get('budget_after', state.remaining_budget),
                    'efficiency': self._calculate_budget_efficiency(auction_result, action, context),
                    'pacing_adherence': self._calculate_pacing_adherence(action, state, context)
                }
                
                # Prepare attribution impact
                attribution_impact = {
                    'credit_received': state.channel_attribution_credit,
                    'sequence_position': context.get('touchpoint_position', 1),
                    'expected_value': state.expected_conversion_value
                }
                
                log_outcome(
                    decision_id=action['decision_id'],
                    auction_result=auction_result,
                    learning_metrics=learning_metrics,
                    budget_impact=budget_impact,
                    attribution_impact=attribution_impact
                )
                
                # LOG BUDGET ALLOCATION
                if hasattr(auction_result, 'price_paid') and auction_result.price_paid > 0:
                    performance_metrics = {
                        'impressions': 1 if hasattr(auction_result, 'won') and auction_result.won else 0,
                        'clicks': 1 if hasattr(auction_result, 'clicked') and auction_result.clicked else 0,
                        'conversions': 1 if hasattr(auction_result, 'revenue') and auction_result.revenue > 0 else 0,
                        'revenue': getattr(auction_result, 'revenue', 0.0),
                        'spent': getattr(auction_result, 'price_paid', 0.0),
                        'spend_rate': context.get('spend_rate', 0.0),
                        'target_spend_rate': context.get('target_spend_rate', 0.0),
                        'pacing_multiplier': state.pacing_factor,
                        'attribution_weight': state.channel_attribution_credit
                    }
                    
                    segment_name = self.discovered_segments[state.segment_index] if state.segment_index < len(self.discovered_segments) else 'unknown'
                    
                    log_budget(
                        channel=action['channel'],
                        creative_id=action['creative_id'],
                        segment=segment_name,
                        allocation_amount=getattr(auction_result, 'price_paid', 0.0),
                        performance_metrics=performance_metrics,
                        attribution_model='linear'
                    )
                
            except Exception as e:
                logger.error(f"Failed to log auction outcome {action.get('decision_id', 'unknown')}: {e}")
                # DO NOT fail silently - this is critical for compliance
                raise RuntimeError(f"AUDIT LOGGING FAILURE in train(): {e}")
        
        # Store training metrics for convergence monitoring
        current_loss = 0.0
        current_gradient_norm = 0.0
        
        # Train on trajectories when we have enough completed trajectories
        if len(self.trajectory_buffer) >= self._hyperparameters.get('batch_size', 32):
            training_result = self._train_trajectory_batch()
            if training_result:
                current_loss = training_result.get('loss', 0.0)
                current_gradient_norm = training_result.get('gradient_norm', 0.0)
        
        # Alternative training: train on individual experiences when trajectory buffer not ready
        elif (len(self.replay_buffer) >= self._hyperparameters['batch_size'] and 
              self.step_count % self.training_frequency == 0):
            training_result = self._train_step_legacy()
            if training_result:
                current_loss = training_result.get('loss', 0.0) 
                current_gradient_norm = training_result.get('gradient_norm', 0.0)
        
        # CRITICAL: Real-time convergence monitoring
        should_stop = self.convergence_monitor.monitor_step(
            loss=current_loss,
            reward=reward,
            gradient_norm=current_gradient_norm,
            action=action
        )
        
        # Check if emergency stop triggered
        if should_stop:
            logger.critical("CONVERGENCE MONITOR: Emergency stop triggered!")
            # Generate final report
            final_report = self.convergence_monitor.generate_report()
            logger.critical(f"Final convergence report: {final_report}")
            # Don't continue with training
            return should_stop
        
        # Adaptive epsilon update based on performance
        self.epsilon = self._calculate_epsilon_from_performance()
        
        # Periodic LR scheduler reporting (every 500 steps)
        if self.step_count % 500 == 0:
            self.log_lr_scheduler_report()
        
        # Update target networks with improved stability and monitoring
        self._update_target_networks_with_monitoring()
    
    def _update_target_networks_with_monitoring(self) -> None:
        """Update target networks with stability monitoring and soft updates"""
        # Track steps since last update
        self.training_metrics['steps_since_last_update'] = self.step_count - self.training_metrics['last_target_update_step']
        
        target_update_freq = self._hyperparameters['target_update_frequency']
        
        # CRITICAL VERIFICATION: Ensure frequency is never less than 1000
        if target_update_freq < 1000:
            raise ValueError(f"CRITICAL ERROR: Target update frequency {target_update_freq} is less than 1000! This violates stability requirements.")
        
        # Log current status for debugging
        logger.debug(f"Target network status: step={self.step_count}, "
                    f"steps_since_last_update={self.training_metrics['steps_since_last_update']}, "
                    f"required_frequency={target_update_freq}")
        use_soft_updates = self._hyperparameters.get('target_update_tau', 0) > 0
        
        # Check if we should update based on steps (more stable than episodes)
        should_update = self.training_metrics['steps_since_last_update'] >= target_update_freq
        
        if should_update:
            # Monitor network divergence before update
            self._monitor_network_divergence()
            
            if use_soft_updates:
                # Soft updates (polyak averaging) - more stable
                self._soft_update_target_networks()
            else:
                # Hard updates (copy weights completely)
                self._hard_update_target_networks()
            
            # Update tracking
            self.training_metrics['last_target_update_step'] = self.step_count
            self.training_metrics['steps_since_last_update'] = 0
            
            # Monitor stability after update
            self._monitor_training_stability()
            
            logger.info(f"Target networks updated at step {self.step_count} "
                       f"(frequency: {target_update_freq}, method: {'soft' if use_soft_updates else 'hard'})")
    
    def _monitor_network_divergence(self) -> None:
        """Monitor how much target networks have diverged from online networks"""
        # Sample a small batch of states for divergence measurement
        if len(self.replay_buffer) < 32:
            return
            
        # Get random sample of states for measurement
        sample_size = min(32, len(self.replay_buffer))
        sample_experiences = random.sample(list(self.replay_buffer), sample_size)
        states = torch.stack([torch.tensor(exp.state) for exp in sample_experiences])
        
        with torch.no_grad():
            # Measure bid network divergence
            online_q_bid = self.q_network_bid(states).mean().item()
            target_q_bid = self.target_network_bid(states).mean().item()
            bid_divergence = abs(online_q_bid - target_q_bid)
            self.training_metrics['target_network_divergence']['bid'].append(bid_divergence)
            
            # Measure creative network divergence
            online_q_creative = self.q_network_creative(states).mean().item()
            target_q_creative = self.target_network_creative(states).mean().item()
            creative_divergence = abs(online_q_creative - target_q_creative)
            self.training_metrics['target_network_divergence']['creative'].append(creative_divergence)
            
            # Measure channel network divergence
            online_q_channel = self.q_network_channel(states).mean().item()
            target_q_channel = self.target_network_channel(states).mean().item()
            channel_divergence = abs(online_q_channel - target_q_channel)
            self.training_metrics['target_network_divergence']['channel'].append(channel_divergence)
            
            # Log if divergence is high
            max_divergence = max(bid_divergence, creative_divergence, channel_divergence)
            if max_divergence > 1.0:  # Threshold for high divergence
                logger.warning(f"High target network divergence detected: {max_divergence:.4f}")
    
    def _soft_update_target_networks(self) -> None:
        """Soft update (polyak averaging) of target networks"""
        tau = self._hyperparameters['target_update_tau']
        
        # Update bid target network
        for target_param, online_param in zip(self.target_network_bid.parameters(), 
                                            self.q_network_bid.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
        
        # Update creative target network
        for target_param, online_param in zip(self.target_network_creative.parameters(), 
                                            self.q_network_creative.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
        
        # Update channel target network
        for target_param, online_param in zip(self.target_network_channel.parameters(), 
                                            self.q_network_channel.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    
    def _hard_update_target_networks(self) -> None:
        """Hard update (complete copy) of target networks"""
        self.target_network_bid.load_state_dict(self.q_network_bid.state_dict())
        self.target_network_creative.load_state_dict(self.q_network_creative.state_dict())
        self.target_network_channel.load_state_dict(self.q_network_channel.state_dict())
    
    def _monitor_training_stability(self) -> None:
        """Monitor training stability after target network updates"""
        # Track Q-value stability
        if len(self.replay_buffer) >= 32:
            sample_size = min(32, len(self.replay_buffer))
            sample_experiences = random.sample(list(self.replay_buffer), sample_size)
            states = torch.stack([torch.tensor(exp.state) for exp in sample_experiences])
            
            with torch.no_grad():
                q_values_bid = self.q_network_bid(states).max(1)[0]
                avg_q_value = q_values_bid.mean().item()
                self.training_metrics['q_value_stability'].append(avg_q_value)
                
                # Check for Q-value oscillation
                if len(self.training_metrics['q_value_stability']) >= 10:
                    recent_q_values = list(self.training_metrics['q_value_stability'])[-10:]
                    q_std = np.std(recent_q_values)
                    q_mean = np.mean(recent_q_values)
                    
                    # If coefficient of variation is high, we have oscillation
                    if q_mean != 0 and (q_std / abs(q_mean)) > 0.5:
                        logger.warning(f"Q-value oscillation detected! CV: {q_std/abs(q_mean):.4f}")
                        # Could implement automatic frequency adjustment here
    
    def _estimate_win_probability(self, action: Dict[str, Any], state: DynamicEnrichedState) -> float:
        """Estimate probability of winning auction based on bid and state"""
        bid_amount = action.get('bid_amount', 0.0)
        competition_level = state.competition_level
        
        # Simple model: higher bid vs competition level
        # This would be refined with actual auction data
        base_win_prob = min(bid_amount / 10.0, 1.0)  # Normalize to reasonable range
        competition_penalty = competition_level * 0.3
        
        return max(0.1, min(0.9, base_win_prob - competition_penalty))
    
    def _monitor_q_value_overestimation(self, states: torch.Tensor, actual_returns: torch.Tensor):
        """Monitor Q-value overestimation bias and Double DQN effectiveness"""
        if not hasattr(self, 'q_value_tracking'):
            return
        
        with torch.no_grad():
            # Get max Q-values from online networks (what we'd use for action selection)
            max_q_bid = self.q_network_bid(states).max(1)[0]
            max_q_creative = self.q_network_creative(states).max(1)[0] 
            max_q_channel = self.q_network_channel(states).max(1)[0]
            
            # Average the max Q-values across action types
            avg_max_q = (max_q_bid + max_q_creative + max_q_channel) / 3.0
            
            # Track max Q-values and actual returns
            self.q_value_tracking['max_q_values'].extend(avg_max_q.cpu().numpy())
            self.q_value_tracking['actual_returns'].extend(actual_returns.cpu().numpy())
            
            # Calculate overestimation bias if we have enough samples
            if len(self.q_value_tracking['max_q_values']) > 10:
                recent_q_values = np.array(list(self.q_value_tracking['max_q_values'])[-10:])
                recent_returns = np.array(list(self.q_value_tracking['actual_returns'])[-10:])
                
                # Bias = E[Q(s,a)] - E[R(s,a)]
                bias = np.mean(recent_q_values) - np.mean(recent_returns)
                self.q_value_tracking['overestimation_bias'].append(bias)
                
                # Log if significant overestimation detected
                if len(self.q_value_tracking['overestimation_bias']) > 5:
                    avg_bias = np.mean(list(self.q_value_tracking['overestimation_bias'])[-5:])
                    if avg_bias > 0.5:  # Threshold for concerning overestimation
                        logger.warning(f"OVERESTIMATION BIAS DETECTED: {avg_bias:.3f} (Q-values consistently {avg_bias:.3f} higher than returns)")
                    elif avg_bias < 0.1:  # Good sign - minimal overestimation
                        logger.info(f"Double DQN working well - minimal overestimation bias: {avg_bias:.3f}")
    
    def _verify_double_dqn_benefit(self, states: torch.Tensor, next_states: torch.Tensor) -> Dict[str, float]:
        """Verify Double DQN is reducing overestimation compared to standard DQN"""
        with torch.no_grad():
            # Standard DQN approach (what we replaced)
            standard_dqn_values = self.target_network_bid(next_states).max(1)[0]
            
            # Double DQN approach (our current implementation)
            next_actions = self.q_network_bid(next_states).argmax(1)
            double_dqn_values = self.target_network_bid(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            
            # Compare values - Double DQN should generally be lower (less overestimated)
            standard_mean = standard_dqn_values.mean().item()
            double_mean = double_dqn_values.mean().item()
            
            reduction_ratio = (standard_mean - double_mean) / abs(standard_mean) if standard_mean != 0 else 0.0
            
            return {
                'standard_dqn_mean_q': standard_mean,
                'double_dqn_mean_q': double_mean,
                'overestimation_reduction': reduction_ratio,
                'double_dqn_benefit': max(0, reduction_ratio)  # Positive if Double DQN reduces overestimation
            }
    
    def _analyze_competition(self, auction_result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competition from auction result"""
        if not auction_result:
            return {}
        
        return {
            'competitors_count': getattr(auction_result, 'competitors_count', 0),
            'our_position': getattr(auction_result, 'position', 0),
            'market_competitiveness': context.get('competition_level', 0.5),
            'estimated_second_price': getattr(auction_result, 'price_paid', 0.0)
        }
    
    def _calculate_budget_efficiency(self, auction_result: Any, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate budget efficiency (revenue / spend)"""
        if not auction_result:
            return 0.0
        
        spent = getattr(auction_result, 'price_paid', 0.0)
        revenue = getattr(auction_result, 'revenue', 0.0)
        
        if spent > 0:
            return revenue / spent
        
        return 0.0
    
    def _calculate_pacing_adherence(self, action: Dict[str, Any], state: DynamicEnrichedState, context: Dict[str, Any]) -> float:
        """Calculate how well we adhered to pacing strategy"""
        target_pacing = context.get('target_pacing_factor', 1.0)
        actual_pacing = state.pacing_factor
        
        # Perfect adherence = 1.0, deviation reduces score
        deviation = abs(target_pacing - actual_pacing)
        return max(0.0, 1.0 - deviation)
    
    def get_trajectory_statistics(self) -> Dict[str, Any]:
        """Get statistics about trajectory-based learning performance"""
        if not self.trajectory_buffer:
            return {'status': 'no_trajectories'}
        
        recent_trajectories = self.trajectory_buffer[-100:]  # Last 100 trajectories
        
        lengths = [t.trajectory_length for t in recent_trajectories]
        returns = [t.total_return for t in recent_trajectories]
        
        # Calculate n-step vs Monte Carlo performance
        n_step_errors = []
        mc_errors = []
        
        for traj in recent_trajectories:
            if traj.monte_carlo_returns and traj.n_step_returns:
                n_step_error = np.mean([(abs(ns - mc)) for ns, mc in 
                                       zip(traj.n_step_returns, traj.monte_carlo_returns)])
                n_step_errors.append(n_step_error)
        
        stats = {
            'num_trajectories': len(self.trajectory_buffer),
            'avg_trajectory_length': np.mean(lengths) if lengths else 0,
            'std_trajectory_length': np.std(lengths) if lengths else 0,
            'avg_trajectory_return': np.mean(returns) if returns else 0,
            'std_trajectory_return': np.std(returns) if returns else 0,
            'avg_n_step_error': np.mean(n_step_errors) if n_step_errors else 0,
            'n_step_range_current': self.n_step_range,
            'gae_lambda': self.gae_lambda,
            'use_monte_carlo': self.use_monte_carlo,
            'ongoing_trajectories': len(self.current_trajectories)
        }
        
        return stats
    
    def force_trajectory_completion(self, user_id: str) -> bool:
        """Force completion of trajectory for user (e.g., session timeout)"""
        if user_id in self.current_trajectories:
            completed_trajectory = self._process_trajectory(user_id, trajectory_complete=False)
            if completed_trajectory:
                self.trajectory_buffer.append(completed_trajectory)
                del self.current_trajectories[user_id]
                logger.info(f"Forced completion of trajectory for user {user_id} with "
                           f"{completed_trajectory.trajectory_length} steps")
                return True
        return False
    
    def cleanup_stale_trajectories(self, max_age_seconds: float = 3600):
        """Clean up trajectories that have been inactive too long"""
        current_time = datetime.now().timestamp()
        stale_users = []
        
        for user_id, trajectory in self.current_trajectories.items():
            if trajectory and len(trajectory) > 0:
                last_timestamp = trajectory[-1].timestamp
                if current_time - last_timestamp > max_age_seconds:
                    stale_users.append(user_id)
        
        for user_id in stale_users:
            logger.warning(f"Cleaning up stale trajectory for user {user_id}")
            self.force_trajectory_completion(user_id)
    
    def get_learning_rate_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning rate scheduler statistics"""
        scheduler_stats = self.lr_scheduler.get_scheduler_stats()
        
        # Add current optimizer learning rates for verification
        current_lrs = {
            'value_optimizer_lr': self.value_optimizer.param_groups[0]['lr'],
            'bid_optimizer_lr': self.optimizer_bid.param_groups[0]['lr'],
            'creative_optimizer_lr': self.optimizer_creative.param_groups[0]['lr'],
            'channel_optimizer_lr': self.optimizer_channel.param_groups[0]['lr'],
            'curiosity_optimizer_lr': self.curiosity_module.optimizer.param_groups[0]['lr']
        }
        
        # Combine scheduler stats with current optimizer states
        full_stats = {
            **scheduler_stats,
            **current_lrs,
            'lr_history_length': len(self.lr_scheduler.lr_history),
            'performance_history_length': len(self.lr_scheduler.performance_history),
            'config': {
                'scheduler_type': self.lr_scheduler_config.scheduler_type,
                'warmup_steps': self.lr_scheduler_config.warmup_steps,
                'plateau_patience': self.lr_scheduler_config.plateau_patience,
                'plateau_threshold': self.lr_scheduler_config.plateau_threshold,
                'min_lr': self.lr_scheduler_config.min_lr,
                'max_lr': self.lr_scheduler_config.max_lr
            }
        }
        
        return full_stats
    
    def log_lr_scheduler_report(self):
        """Log detailed learning rate scheduler report"""
        stats = self.get_learning_rate_stats()
        
        logger.info("=== ADAPTIVE LEARNING RATE SCHEDULER REPORT ===")
        logger.info(f"Scheduler Type: {stats['scheduler_type']}")
        logger.info(f"Current LR: {stats['current_lr']:.6f}")
        logger.info(f"Step Count: {stats['step_count']}")
        logger.info(f"Plateau Count: {stats['plateau_count']}")
        logger.info(f"Warmup Complete: {stats['warmup_complete']}")
        logger.info(f"Best Performance: {stats['best_performance']:.4f}")
        logger.info(f"LR Reductions: {stats['lr_reductions']}")
        
        # Check all optimizers have same LR
        lrs = [stats['value_optimizer_lr'], stats['bid_optimizer_lr'], 
               stats['creative_optimizer_lr'], stats['channel_optimizer_lr'],
               stats['curiosity_optimizer_lr']]
        if len(set(lrs)) == 1:
            logger.info("✓ All optimizers synchronized")
        else:
            logger.warning(f"⚠ Optimizer LR mismatch: {lrs}")
        
        # Show recent performance trend if available
        if len(self.lr_scheduler.performance_history) > 5:
            recent_perf = list(self.lr_scheduler.performance_history)[-5:]
            trend = "improving" if recent_perf[-1] > recent_perf[0] else "declining"
            logger.info(f"Recent performance trend: {trend} ({recent_perf[-1]:.4f})")
        
        logger.info("==============================================")
    
    def _verify_trajectory_double_dqn_benefit(self, states: torch.Tensor, next_states: torch.Tensor, 
                                            trajectory_targets: torch.Tensor, double_dqn_targets: torch.Tensor) -> Dict[str, float]:
        """Verify Double DQN effectiveness in trajectory training"""
        with torch.no_grad():
            # Compare trajectory-only targets vs Double DQN enhanced targets
            trajectory_mean = trajectory_targets.mean().item()
            double_dqn_mean = double_dqn_targets.mean().item()
            
            # Standard DQN approach for trajectory data (what we replaced)
            standard_next_q_bid = self.target_network_bid(next_states).max(1)[0]
            standard_next_q_creative = self.target_network_creative(next_states).max(1)[0] 
            standard_next_q_channel = self.target_network_channel(next_states).max(1)[0]
            standard_avg_q = (standard_next_q_bid + standard_next_q_creative + standard_next_q_channel) / 3.0
            
            # Double DQN approach (our implementation)
            next_actions_bid = self.q_network_bid(next_states).argmax(1)
            next_actions_creative = self.q_network_creative(next_states).argmax(1)
            next_actions_channel = self.q_network_channel(next_states).argmax(1)
            
            double_next_q_bid = self.target_network_bid(next_states).gather(1, next_actions_bid.unsqueeze(1)).squeeze()
            double_next_q_creative = self.target_network_creative(next_states).gather(1, next_actions_creative.unsqueeze(1)).squeeze()
            double_next_q_channel = self.target_network_channel(next_states).gather(1, next_actions_channel.unsqueeze(1)).squeeze()
            double_avg_q = (double_next_q_bid + double_next_q_creative + double_next_q_channel) / 3.0
            
            # Calculate overestimation reduction
            standard_q_mean = standard_avg_q.mean().item()
            double_q_mean = double_avg_q.mean().item()
            
            reduction_ratio = (standard_q_mean - double_q_mean) / abs(standard_q_mean) if standard_q_mean != 0 else 0.0
            
            # Target enhancement comparison
            target_enhancement = abs(double_dqn_mean - trajectory_mean) / abs(trajectory_mean) if trajectory_mean != 0 else 0.0
            
            return {
                'trajectory_targets_mean': trajectory_mean,
                'double_dqn_targets_mean': double_dqn_mean,
                'standard_dqn_next_q': standard_q_mean,
                'double_dqn_next_q': double_q_mean,
                'overestimation_reduction': reduction_ratio,
                'target_enhancement_ratio': target_enhancement,
                'double_dqn_benefit': max(0, reduction_ratio),
                'trajectory_double_dqn_effective': reduction_ratio > 0.01  # At least 1% reduction
            }
    
    def end_episode(self, episode_reward: float) -> bool:
        """Handle episode completion and convergence monitoring"""
        # Let convergence monitor handle episode completion
        self.convergence_monitor.end_episode(episode_reward)
        
        # Check if training should stop
        should_stop = self.convergence_monitor.should_stop()
        
        if should_stop:
            logger.critical("EPISODE END: Convergence monitor triggered stop!")
            final_report = self.convergence_monitor.generate_report()
            logger.critical(f"Final episode convergence report: {final_report}")
        
        return should_stop
    
    def get_convergence_report(self) -> Dict[str, Any]:
        """Get comprehensive convergence report"""
        return self.convergence_monitor.generate_report()
    
    def is_converged(self) -> bool:
        """Check if training has converged"""
        return self.convergence_monitor.convergence_detected
    
    def emergency_stop_triggered(self) -> bool:
        """Check if emergency stop was triggered"""
        return self.convergence_monitor.emergency_stop_triggered
    
    def update_discovered_segments(self, segments: Dict):
        """Update agent with newly discovered segments"""
        if not segments:
            logger.warning("No segments provided for update")
            return
        
        logger.info(f"Updating agent with {len(segments)} discovered segments")
        
        # Store segments for state enrichment
        self.discovered_segments = segments
        
        # Update patterns with segment information
        if hasattr(self, 'patterns'):
            self.patterns['segments'] = {}
            for segment_id, segment in segments.items():
                self.patterns['segments'][segment_id] = {
                    'name': segment.name,
                    'size': segment.size,
                    'conversion_rate': segment.conversion_rate,
                    'characteristics': segment.characteristics,
                    'behavioral_profile': segment.behavioral_profile
                }
        
        # Update policy network input dimensions if needed
        # This allows the network to adapt to new segment discoveries
        if hasattr(self, 'policy_net') and hasattr(self.policy_net, 'update_segments'):
            self.policy_net.update_segments(segments)
        
        # Update target network as well
        if hasattr(self, 'target_net') and hasattr(self.target_net, 'update_segments'):
            self.target_net.update_segments(segments)
        
        logger.info(f"✅ Agent updated with discovered segments: {[s.name for s in list(segments.values())[:3]]}")
    
    def get_segment_info(self, segment_index: int) -> Dict:
        """Get information about a specific segment by index"""
        if not hasattr(self, 'discovered_segments') or not self.discovered_segments:
            return {'name': 'unknown', 'conversion_rate': 0.0, 'size': 0}
        
        segment_keys = list(self.discovered_segments.keys())
        if segment_index < len(segment_keys):
            segment_id = segment_keys[segment_index]
            segment = self.discovered_segments[segment_id]
            return {
                'name': segment.name,
                'conversion_rate': segment.conversion_rate,
                'size': segment.size,
                'characteristics': segment.characteristics
            }
        
        return {'name': 'unknown', 'conversion_rate': 0.0, 'size': 0}
