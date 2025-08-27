"""
Experience Replay Buffers

Implements various replay buffer types for efficient experience storage
and sampling in reinforcement learning agents.
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from importance_sampler import ImportanceSampler, Experience

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Standard experience replay buffer with uniform sampling.
    """
    
    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.logger = logging.getLogger(f"{__name__}.ReplayBuffer")
    
    def add(self, state: torch.Tensor, action: Any, reward: float, 
            next_state: torch.Tensor, done: bool):
        """Add experience tuple to buffer"""
        
        experience = {
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done
        }
        
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences uniformly"""
        
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
        
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to tensors
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.bool)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_state(self) -> Dict[str, Any]:
        """Get buffer state for checkpointing"""
        return {
            'buffer': list(self.buffer),
            'capacity': self.capacity,
            'position': self.position
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load buffer state from checkpoint"""
        self.buffer = deque(state['buffer'], maxlen=state['capacity'])
        self.capacity = state['capacity']
        self.position = state['position']


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer that samples experiences
    based on their TD error magnitude.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 epsilon: float = 1e-6, seed: int = 42):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.epsilon = epsilon  # Small constant to prevent zero priorities
        
        # Storage
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Sum tree for efficient sampling
        self._build_sum_tree()
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.logger = logging.getLogger(f"{__name__}.PrioritizedReplayBuffer")
    
    def _build_sum_tree(self):
        """Build sum tree for efficient priority-based sampling"""
        # Tree size is 2 * capacity - 1
        tree_size = 2 * self.capacity - 1
        self.sum_tree = np.zeros(tree_size, dtype=np.float32)
        self.min_tree = np.full(tree_size, float('inf'), dtype=np.float32)
        
        # Leaf nodes start at index capacity - 1
        self.leaf_start = self.capacity - 1
    
    def add(self, state: torch.Tensor, action: Any, reward: float,
            next_state: torch.Tensor, done: bool, priority: Optional[float] = None):
        """Add experience with priority"""
        
        experience = {
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done
        }
        
        # Use maximum priority for new experiences if not specified
        if priority is None:
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        # Store experience
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Update priority
        self.priorities[self.position] = priority
        self._update_tree(self.position, priority)
        
        # Update pointers
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _update_tree(self, index: int, priority: float):
        """Update sum tree with new priority"""
        priority = priority ** self.alpha
        
        # Update leaf node
        tree_index = index + self.leaf_start
        self.sum_tree[tree_index] = priority
        self.min_tree[tree_index] = priority
        
        # Propagate changes up the tree
        while tree_index > 0:
            tree_index = (tree_index - 1) // 2
            left_child = 2 * tree_index + 1
            right_child = 2 * tree_index + 2
            
            self.sum_tree[tree_index] = (
                self.sum_tree[left_child] + self.sum_tree[right_child]
            )
            self.min_tree[tree_index] = min(
                self.min_tree[left_child], self.min_tree[right_child]
            )
    
    def _sample_index(self, value: float) -> int:
        """Sample index based on priority value"""
        tree_index = 0
        
        while tree_index < self.leaf_start:
            left_child = 2 * tree_index + 1
            if value <= self.sum_tree[left_child]:
                tree_index = left_child
            else:
                value -= self.sum_tree[left_child]
                tree_index = 2 * tree_index + 2
        
        return tree_index - self.leaf_start
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], List[int], np.ndarray]:
        """Sample batch with importance sampling weights"""
        
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer: {self.size} < {batch_size}")
        
        indices = []
        priorities = []
        
        # Sample indices based on priorities
        total_priority = self.sum_tree[0]
        segment_size = total_priority / batch_size
        
        for i in range(batch_size):
            start = segment_size * i
            end = segment_size * (i + 1)
            value = np.random.uniform(start, end)
            
            index = self._sample_index(value)
            indices.append(index)
            priorities.append(self.priorities[index])
        
        # Compute importance sampling weights
        min_priority = self.min_tree[0]
        max_weight = (min_priority * self.size) ** (-self.beta)
        
        weights = []
        for priority in priorities:
            weight = (priority * self.size) ** (-self.beta)
            weights.append(weight / max_weight)
        
        weights = np.array(weights, dtype=np.float32)
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        
        # Convert to tensors
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.bool)
        
        batch_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        
        return batch_dict, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for given indices"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self.size:
                # Add small epsilon to prevent zero priorities
                priority = max(priority, self.epsilon)
                self.priorities[idx] = priority
                self._update_tree(idx, priority)
    
    def __len__(self) -> int:
        return self.size
    
    def get_state(self) -> Dict[str, Any]:
        """Get buffer state for checkpointing"""
        return {
            'buffer': self.buffer[:self.size],
            'priorities': self.priorities[:self.size].tolist(),
            'capacity': self.capacity,
            'alpha': self.alpha,
            'beta': self.beta,
            'epsilon': self.epsilon,
            'position': self.position,
            'size': self.size
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load buffer state from checkpoint"""
        self.buffer = state['buffer']
        self.capacity = state['capacity']
        self.alpha = state['alpha']
        self.beta = state['beta']
        self.epsilon = state['epsilon']
        self.position = state['position']
        self.size = state['size']
        
        # Rebuild sum tree
        self._build_sum_tree()
        
        # Restore priorities
        priorities = np.array(state['priorities'])
        self.priorities[:self.size] = priorities
        
        for i in range(self.size):
            self._update_tree(i, priorities[i])


class HindsightExperienceReplay:
    """
    Hindsight Experience Replay (HER) buffer for goal-conditioned RL.
    Useful for ad campaign optimization with multiple objectives.
    """
    
    def __init__(self, capacity: int, replay_strategy: str = "future", 
                 replay_k: int = 4, seed: int = 42):
        self.capacity = capacity
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k  # Number of HER transitions per episode
        
        self.buffer = ReplayBuffer(capacity, seed)
        self.episode_buffer = []  # Store current episode
        
        self.logger = logging.getLogger(f"{__name__}.HindsightExperienceReplay")
    
    def add(self, state: torch.Tensor, action: Any, reward: float,
            next_state: torch.Tensor, done: bool, goal: torch.Tensor,
            achieved_goal: torch.Tensor):
        """Add experience with goal information"""
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'goal': goal,
            'achieved_goal': achieved_goal
        }
        
        self.episode_buffer.append(experience)
        
        # If episode is done, process with HER
        if done:
            self._process_episode()
            self.episode_buffer = []
    
    def _process_episode(self):
        """Process episode with hindsight experience replay"""
        episode_length = len(self.episode_buffer)
        
        # Add original transitions
        for exp in self.episode_buffer:
            self.buffer.add(
                exp['state'], exp['action'], exp['reward'],
                exp['next_state'], exp['done']
            )
        
        # Generate HER transitions
        for t in range(episode_length):
            for _ in range(self.replay_k):
                
                # Select future goal based on strategy
                if self.replay_strategy == "future":
                    future_indices = range(t, episode_length)
                    if future_indices:
                        future_t = np.random.choice(future_indices)
                        new_goal = self.episode_buffer[future_t]['achieved_goal']
                    else:
                        continue
                elif self.replay_strategy == "episode":
                    future_t = np.random.randint(0, episode_length)
                    new_goal = self.episode_buffer[future_t]['achieved_goal']
                else:
                    continue
                
                # Create HER transition
                exp = self.episode_buffer[t]
                
                # Recompute reward with new goal
                new_reward = self._compute_reward(
                    exp['achieved_goal'], new_goal, exp
                )
                
                # Check if goal is achieved
                new_done = self._goal_achieved(exp['achieved_goal'], new_goal)
                
                # Add HER transition
                self.buffer.add(
                    exp['state'], exp['action'], new_reward,
                    exp['next_state'], new_done
                )
    
    def _compute_reward(self, achieved_goal: torch.Tensor, 
                       goal: torch.Tensor, experience: Dict) -> float:
        """Compute reward for achieving goal"""
        # Simple sparse reward: 0 if goal achieved, -1 otherwise
        distance = torch.norm(achieved_goal - goal)
        return 0.0 if distance < 0.1 else -1.0
    
    def _goal_achieved(self, achieved_goal: torch.Tensor, 
                      goal: torch.Tensor) -> bool:
        """Check if goal is achieved"""
        distance = torch.norm(achieved_goal - goal)
        return distance < 0.1
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from underlying buffer"""
        return self.buffer.sample(batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class CircularBuffer:
    """
    Circular buffer for storing fixed-size sequences efficiently.
    Useful for recurrent networks and sequence-based learning.
    """
    
    def __init__(self, capacity: int, sequence_length: int):
        self.capacity = capacity
        self.sequence_length = sequence_length
        
        self.states = np.zeros((capacity, sequence_length + 1, 128), dtype=np.float32)
        self.actions = np.zeros((capacity, sequence_length), dtype=np.int32)
        self.rewards = np.zeros((capacity, sequence_length), dtype=np.float32)
        self.dones = np.zeros((capacity, sequence_length), dtype=np.bool_)
        
        self.position = 0
        self.size = 0
        
        self.logger = logging.getLogger(f"{__name__}.CircularBuffer")
    
    def add_sequence(self, states: np.ndarray, actions: np.ndarray,
                    rewards: np.ndarray, dones: np.ndarray):
        """Add complete sequence to buffer"""
        
        if len(states) != self.sequence_length + 1:
            raise ValueError(f"State sequence length mismatch: {len(states)} != {self.sequence_length + 1}")
        
        self.states[self.position] = states
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards
        self.dones[self.position] = dones
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of sequences"""
        
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: {self.size} < {batch_size}")
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices]
        }
    
    def __len__(self) -> int:
        return self.size