"""
Importance Sampling Integration for Training Orchestrator

This module provides components to integrate the ImportanceSampler with the training
loop for crisis parent weighting, ensuring rare but valuable events (crisis parents)
get proper representation in training batches.
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


class ImportanceSamplingReplayBuffer:
    """
    Replay buffer that integrates with the ImportanceSampler for crisis parent weighting.
    
    This buffer ensures that crisis parents (10% population, 50% value) get proper
    5x weight representation in training batches through importance sampling.
    """
    
    def __init__(
        self,
        capacity: int,
        importance_sampler: Optional[ImportanceSampler] = None,
        seed: int = 42
    ):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        # Initialize importance sampler if not provided
        self.importance_sampler = importance_sampler or ImportanceSampler(
            population_ratios={"crisis_parent": 0.1, "regular_parent": 0.9},
            conversion_ratios={"crisis_parent": 0.5, "regular_parent": 0.5},
            max_weight=5.0,  # 5x weight for crisis parents
            alpha=0.6,
            beta_start=0.4
        )
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.logger = logging.getLogger(f"{__name__}.ImportanceSamplingReplayBuffer")
    
    def add(
        self,
        state: torch.Tensor,
        action: Any,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        event_type: str = "regular_parent",
        value: float = None,
        metadata: Dict = None
    ):
        """Add experience with event type for importance weighting"""
        
        # Store standard replay buffer format
        experience_dict = {
            'state': state.cpu() if hasattr(state, 'cpu') else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu() if hasattr(next_state, 'cpu') else next_state,
            'done': done,
            'event_type': event_type,
            'value': value or reward,
            'metadata': metadata or {}
        }
        
        self.buffer.append(experience_dict)
        
        # Also add to importance sampler
        importance_exp = Experience(
            state=state.cpu().numpy() if hasattr(state, 'cpu') else np.array(state),
            action=action if isinstance(action, (int, np.ndarray)) else (
                action.cpu().numpy() if hasattr(action, 'cpu') else np.array(action)
            ),
            reward=reward,
            next_state=next_state.cpu().numpy() if hasattr(next_state, 'cpu') else np.array(next_state),
            done=done,
            value=value or reward,
            event_type=event_type,
            timestamp=len(self.buffer),
            metadata=metadata or {}
        )
        
        self.importance_sampler.add_experience(importance_exp)
        
        self.logger.debug(f"Added {event_type} experience with value {importance_exp.value}")
    
    def sample(self, batch_size: int, use_importance_sampling: bool = True) -> Tuple[Dict[str, torch.Tensor], List[float], List[int]]:
        """Sample batch with optional importance weighting"""
        
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
        
        if use_importance_sampling and len(self.importance_sampler.experiences) >= batch_size:
            # Use importance sampling
            sampled_experiences, importance_weights, sampled_indices = self.importance_sampler.weighted_sampling(
                batch_size=batch_size,
                temperature=1.0
            )
            
            # Convert importance sampler experiences back to buffer format
            batch = []
            for i, idx in enumerate(sampled_indices):
                if idx < len(self.buffer):
                    batch.append(self.buffer[idx])
                else:
                    # Fallback to random sampling if index out of range
                    batch.append(random.choice(self.buffer))
            
            crisis_count = sum(1 for exp in sampled_experiences if exp.event_type == 'crisis_parent')
            self.logger.debug(f"Importance sampling: crisis_parent ratio in batch: {crisis_count / batch_size:.2%}")
            
        else:
            # Standard uniform sampling
            batch = random.sample(self.buffer, batch_size)
            importance_weights = [1.0] * batch_size
            sampled_indices = list(range(batch_size))  # Placeholder indices
            
            self.logger.debug("Using uniform sampling")
        
        # Convert batch to tensors
        states = torch.stack([exp['state'] if hasattr(exp['state'], 'dim') else torch.tensor(exp['state']) for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp['next_state'] if hasattr(exp['next_state'], 'dim') else torch.tensor(exp['next_state']) for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.bool)
        
        batch_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'event_types': [exp['event_type'] for exp in batch],
            'values': [exp['value'] for exp in batch]
        }
        
        return batch_dict, importance_weights, sampled_indices
    
    def sample_uniform(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch uniformly (standard replay buffer behavior)"""
        batch_dict, _, _ = self.sample(batch_size, use_importance_sampling=False)
        return batch_dict
    
    def sample_importance_weighted(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], List[float], List[int]]:
        """Sample batch with importance weighting (crisis parent prioritization)"""
        return self.sample(batch_size, use_importance_sampling=True)
    
    def update_importance_weights(
        self,
        population_ratios: Dict[str, float] = None,
        conversion_ratios: Dict[str, float] = None
    ):
        """Update importance sampling ratios based on observed data"""
        
        if population_ratios:
            self.importance_sampler.population_ratios.update(population_ratios)
        
        if conversion_ratios:
            self.importance_sampler.conversion_ratios.update(conversion_ratios)
        
        self.importance_sampler._update_importance_weights()
        
        self.logger.info(f"Updated importance weights: {self.importance_sampler._importance_weights}")
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get statistics about sampling behavior and event distribution"""
        
        # Buffer statistics
        event_counts = {}
        total_value_by_type = {}
        
        for exp in self.buffer:
            event_type = exp['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            total_value_by_type[event_type] = total_value_by_type.get(event_type, 0) + exp['value']
        
        buffer_stats = {
            'buffer_size': len(self.buffer),
            'event_counts': event_counts,
            'event_ratios': {k: v/len(self.buffer) for k, v in event_counts.items()} if self.buffer else {},
            'avg_value_by_type': {k: v/event_counts[k] for k, v in total_value_by_type.items() if event_counts.get(k, 0) > 0}
        }
        
        # Importance sampler statistics
        importance_stats = self.importance_sampler.get_sampling_statistics()
        
        # Combined statistics
        return {
            'buffer_stats': buffer_stats,
            'importance_stats': importance_stats,
            'crisis_parent_weight': importance_stats.get('importance_weights', {}).get('crisis_parent', 1.0),
            'regular_parent_weight': importance_stats.get('importance_weights', {}).get('regular_parent', 1.0)
        }
    
    def clear_buffer(self):
        """Clear the replay buffer"""
        self.buffer.clear()
        self.importance_sampler.clear_buffer()
        self.logger.info("Cleared importance sampling replay buffer")
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_state(self) -> Dict[str, Any]:
        """Get buffer state for checkpointing"""
        return {
            'buffer': list(self.buffer),
            'capacity': self.capacity,
            'importance_sampler_state': self.importance_sampler.get_sampling_statistics()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load buffer state from checkpoint"""
        self.buffer = deque(state['buffer'], maxlen=state['capacity'])
        self.capacity = state['capacity']
        
        # Restore importance sampler state
        if 'importance_sampler_state' in state:
            importance_state = state['importance_sampler_state']
            if 'population_ratios' in importance_state:
                self.importance_sampler.population_ratios = importance_state['population_ratios']
            if 'conversion_ratios' in importance_state:
                self.importance_sampler.conversion_ratios = importance_state['conversion_ratios']
            if 'frame_count' in importance_state:
                self.importance_sampler.frame_count = importance_state['frame_count']
            self.importance_sampler._update_importance_weights()
        
        # Re-add experiences to importance sampler
        for exp in self.buffer:
            importance_exp = Experience(
                state=exp['state'].numpy() if hasattr(exp['state'], 'numpy') else np.array(exp['state']),
                action=exp['action'],
                reward=exp['reward'],
                next_state=exp['next_state'].numpy() if hasattr(exp['next_state'], 'numpy') else np.array(exp['next_state']),
                done=exp['done'],
                value=exp['value'],
                event_type=exp['event_type'],
                timestamp=len(self.importance_sampler.experiences),
                metadata=exp.get('metadata', {})
            )
            self.importance_sampler.add_experience(importance_exp)


class ExperienceAggregator:
    """
    Aggregates experiences from Monte Carlo simulator for importance sampling.
    
    This component identifies crisis parents based on simulation data and ensures
    they are properly labeled for importance weighting.
    """
    
    def __init__(self, crisis_indicators: List[str] = None):
        self.crisis_indicators = crisis_indicators or [
            'crisis', 'urgent', 'emergency', 'high_priority', 'critical'
        ]
        self.logger = logging.getLogger(f"{__name__}.ExperienceAggregator")
    
    def identify_event_type(self, experience: Dict[str, Any]) -> str:
        """
        Identify whether an experience is from a crisis parent or regular parent.
        
        Args:
            experience: Experience dictionary with metadata
            
        Returns:
            Event type: 'crisis_parent' or 'regular_parent'
        """
        
        # Check metadata for crisis indicators
        metadata = experience.get('metadata', {})
        
        # User profile analysis
        user_profile = metadata.get('user_profile', {})
        if isinstance(user_profile, dict):
            # Check for crisis indicators in user profile
            for field, value in user_profile.items():
                if isinstance(value, str) and any(indicator in value.lower() for indicator in self.crisis_indicators):
                    return 'crisis_parent'
        elif isinstance(user_profile, str):
            # Check string user profile
            if any(indicator in user_profile.lower() for indicator in self.crisis_indicators):
                return 'crisis_parent'
        
        # Check behavior patterns
        behavior = metadata.get('behavior', {})
        if isinstance(behavior, dict):
            # High engagement with crisis-related content
            if behavior.get('crisis_content_engagement', 0) > 0.7:
                return 'crisis_parent'
            
            # Urgent search patterns
            if behavior.get('search_urgency', 0) > 0.8:
                return 'crisis_parent'
        
        # Check campaign context
        campaign_context = metadata.get('campaign_context', {})
        if isinstance(campaign_context, dict):
            # Crisis-targeted campaigns
            if campaign_context.get('crisis_targeted', False):
                return 'crisis_parent'
        
        # Check reward value - high value experiences are more likely crisis parents
        reward = experience.get('reward', 0)
        value = experience.get('value', reward)
        
        # Threshold-based classification (top 10% of values are likely crisis parents)
        if value > metadata.get('value_threshold_90th', 2.0):
            return 'crisis_parent'
        
        # Default to regular parent
        return 'regular_parent'
    
    def aggregate_experiences(
        self,
        raw_experiences: List[Dict[str, Any]],
        importance_buffer: ImportanceSamplingReplayBuffer
    ) -> Dict[str, int]:
        """
        Aggregate experiences from simulation and add to importance sampling buffer.
        
        Args:
            raw_experiences: List of raw experiences from Monte Carlo simulator
            importance_buffer: Buffer to add processed experiences to
            
        Returns:
            Statistics about processed experiences
        """
        
        processed_stats = {'crisis_parent': 0, 'regular_parent': 0}
        
        for exp in raw_experiences:
            # Identify event type
            event_type = self.identify_event_type(exp)
            processed_stats[event_type] += 1
            
            # Extract required fields with defaults
            state = exp.get('state', torch.zeros(128))
            action = exp.get('action', 0)
            reward = exp.get('reward', 0.0)
            next_state = exp.get('next_state', torch.zeros(128))
            done = exp.get('done', False)
            value = exp.get('value', reward)
            metadata = exp.get('metadata', {})
            
            # Add to importance sampling buffer
            importance_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                event_type=event_type,
                value=value,
                metadata=metadata
            )
        
        self.logger.info(f"Processed {len(raw_experiences)} experiences: {processed_stats}")
        return processed_stats


def integrate_importance_sampling_with_training(
    agent,
    replay_buffer_capacity: int = 10000,
    importance_sampler: Optional[ImportanceSampler] = None
) -> Tuple[ImportanceSamplingReplayBuffer, ExperienceAggregator]:
    """
    Integrate importance sampling with existing training components.
    
    Args:
        agent: The RL agent (PPO, DQN, etc.)
        replay_buffer_capacity: Capacity of the importance sampling replay buffer
        importance_sampler: Optional pre-configured importance sampler
        
    Returns:
        Tuple of (importance_buffer, experience_aggregator)
    """
    
    # Create importance sampling replay buffer
    importance_buffer = ImportanceSamplingReplayBuffer(
        capacity=replay_buffer_capacity,
        importance_sampler=importance_sampler
    )
    
    # Create experience aggregator
    experience_aggregator = ExperienceAggregator()
    
    logger.info(f"Integrated importance sampling with {type(agent).__name__} agent")
    
    return importance_buffer, experience_aggregator


# Example usage and testing functions
def test_crisis_parent_weighting():
    """Test function to verify crisis parent weighting works correctly"""
    
    # Create buffer
    buffer = ImportanceSamplingReplayBuffer(capacity=1000)
    
    # Add experiences - 90% regular parents, 10% crisis parents
    for i in range(100):
        state = torch.randn(10)
        action = i % 4
        next_state = torch.randn(10)
        
        if i % 10 == 0:  # 10% crisis parents
            event_type = "crisis_parent"
            reward = 5.0  # Higher reward for crisis parents
            value = 5.0
        else:  # 90% regular parents
            event_type = "regular_parent"  
            reward = 1.0
            value = 1.0
        
        buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
            event_type=event_type,
            value=value
        )
    
    # Sample with importance weighting
    batch_dict, importance_weights, indices = buffer.sample_importance_weighted(32)
    
    # Count crisis parents in batch
    crisis_count = sum(1 for event_type in batch_dict['event_types'] if event_type == 'crisis_parent')
    
    print(f"Crisis parents in batch: {crisis_count}/32 ({crisis_count/32:.1%})")
    print(f"Expected higher than 10% due to 5x weighting")
    print(f"Average importance weight: {np.mean(importance_weights):.3f}")
    
    # Get statistics
    stats = buffer.get_sampling_statistics()
    print(f"Crisis parent weight: {stats['crisis_parent_weight']:.1f}x")
    print(f"Regular parent weight: {stats['regular_parent_weight']:.1f}x")
    
    return buffer, stats


if __name__ == "__main__":
    # Run test
    buffer, stats = test_crisis_parent_weighting()
    print("Crisis parent weighting test completed successfully!")