"""
Importance Sampling for Rare Valuable Events

This module implements importance sampling techniques to weight experiences by their
value and rarity, ensuring that critical but infrequent events (like crisis parents
who represent 10% of population but 50% of conversions) receive appropriate attention
during training.

The importance sampler addresses the challenge where rare events with high value
would otherwise be underrepresented in training data, leading to suboptimal policy
learning for these critical scenarios.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Container for a single experience/transition."""
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    value: float  # Estimated value of this experience
    event_type: str  # Type of event (e.g., "crisis_parent", "regular_parent")
    timestamp: float
    metadata: Dict = None


class ImportanceSampler:
    """
    Importance sampler that weights experiences based on their value and rarity.
    
    This implementation specifically handles the scenario where crisis parents
    represent 10% of the population but 50% of conversions, ensuring they
    receive proper weighting in training.
    """
    
    def __init__(
        self,
        population_ratios: Dict[str, float] = None,
        conversion_ratios: Dict[str, float] = None,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
        alpha: float = 0.6,  # Prioritization exponent
        beta_start: float = 0.4,  # Initial importance sampling correction
        beta_frames: int = 100000,  # Frames to anneal beta to 1.0
        epsilon: float = 1e-6
    ):
        """
        Initialize the importance sampler.
        
        Args:
            population_ratios: Ratio of each event type in population
            conversion_ratios: Ratio of each event type in successful outcomes
            min_weight: Minimum importance weight
            max_weight: Maximum importance weight
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling correction
            beta_frames: Number of frames to anneal beta to 1.0
            epsilon: Small constant to avoid zero probabilities
        """
        # Default ratios based on the crisis parent scenario
        self.population_ratios = population_ratios or {
            "crisis_parent": 0.1,
            "regular_parent": 0.9
        }
        
        self.conversion_ratios = conversion_ratios or {
            "crisis_parent": 0.5,
            "regular_parent": 0.5
        }
        
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        
        # Experience buffer and statistics
        self.experiences = []
        self.event_counts = defaultdict(int)
        self.event_values = defaultdict(list)
        self.total_experiences = 0
        self.frame_count = 0
        
        # Precomputed importance weights for efficiency
        self._importance_weights = {}
        self._update_importance_weights()
    
    def _update_importance_weights(self):
        """Update precomputed importance weights based on population and conversion ratios."""
        for event_type in self.population_ratios:
            pop_ratio = self.population_ratios[event_type]
            conv_ratio = self.conversion_ratios.get(event_type, pop_ratio)
            
            # Calculate importance weight as conversion_ratio / population_ratio
            # This gives higher weights to events that are rare in population
            # but common in successful outcomes
            raw_weight = conv_ratio / (pop_ratio + self.epsilon)
            
            # Clip weights to prevent extreme values
            weight = np.clip(raw_weight, self.min_weight, self.max_weight)
            self._importance_weights[event_type] = weight
            
        logger.info(f"Updated importance weights: {self._importance_weights}")
    
    def calculate_importance_weight(
        self,
        experience: Experience,
        use_value: bool = True
    ) -> float:
        """
        Calculate importance weight for a given experience.
        
        Args:
            experience: The experience to weight
            use_value: Whether to incorporate experience value in weighting
            
        Returns:
            Importance weight for the experience
        """
        # Base weight from event type
        base_weight = self._importance_weights.get(
            experience.event_type, 1.0
        )
        
        if not use_value:
            return base_weight
        
        # Incorporate experience value
        # Higher value experiences get higher weights
        value_multiplier = 1.0 + np.tanh(experience.value)
        
        # Apply prioritization exponent
        final_weight = base_weight * (value_multiplier ** self.alpha)
        
        return np.clip(final_weight, self.min_weight, self.max_weight)
    
    def add_experience(self, experience: Experience):
        """Add an experience to the buffer."""
        self.experiences.append(experience)
        self.event_counts[experience.event_type] += 1
        self.event_values[experience.event_type].append(experience.value)
        self.total_experiences += 1
        
        # Periodically update importance weights based on observed data
        if self.total_experiences % 1000 == 0:
            self._update_empirical_ratios()
    
    def _update_empirical_ratios(self):
        """Update population ratios based on observed data."""
        if self.total_experiences == 0:
            return
        
        # Update population ratios based on observed frequencies
        for event_type, count in self.event_counts.items():
            self.population_ratios[event_type] = count / self.total_experiences
        
        # Update conversion ratios based on observed values
        # Assume higher average value indicates higher conversion rate
        total_avg_value = np.mean([
            np.mean(values) for values in self.event_values.values()
            if len(values) > 0
        ])
        
        for event_type, values in self.event_values.items():
            if len(values) > 0:
                avg_value = np.mean(values)
                # Normalize by total average value to get relative conversion rate
                self.conversion_ratios[event_type] = avg_value / (total_avg_value + self.epsilon)
        
        self._update_importance_weights()
    
    def weighted_sampling(
        self,
        batch_size: int,
        temperature: float = 1.0
    ) -> Tuple[List[Experience], List[float], List[int]]:
        """
        Sample experiences using importance weights.
        
        Args:
            batch_size: Number of experiences to sample
            temperature: Temperature parameter for softmax sampling
            
        Returns:
            Tuple of (sampled_experiences, importance_weights, indices)
        """
        if len(self.experiences) < batch_size:
            # If we don't have enough experiences, return all
            weights = [self.calculate_importance_weight(exp) for exp in self.experiences]
            return self.experiences[:], weights, list(range(len(self.experiences)))
        
        # Calculate importance weights for all experiences
        weights = np.array([
            self.calculate_importance_weight(exp) for exp in self.experiences
        ])
        
        # Apply temperature scaling
        if temperature != 1.0:
            weights = weights ** (1.0 / temperature)
        
        # Convert to probabilities
        probabilities = weights / (np.sum(weights) + self.epsilon)
        
        # Sample indices based on probabilities
        indices = np.random.choice(
            len(self.experiences),
            size=batch_size,
            replace=True,
            p=probabilities
        )
        
        # Get sampled experiences and their weights
        sampled_experiences = [self.experiences[i] for i in indices]
        sampled_weights = weights[indices]
        
        return sampled_experiences, sampled_weights.tolist(), indices.tolist()
    
    def bias_correction(
        self,
        gradients: np.ndarray,
        importance_weights: List[float],
        batch_size: int
    ) -> np.ndarray:
        """
        Apply bias correction to gradients using importance sampling weights.
        
        Args:
            gradients: Raw gradients from the batch
            importance_weights: Importance weights for the sampled experiences
            batch_size: Size of the sampled batch
            
        Returns:
            Bias-corrected gradients
        """
        self.frame_count += 1
        
        # Calculate current beta (annealing from beta_start to 1.0)
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * 
                  self.frame_count / self.beta_frames)
        
        # Calculate importance sampling correction weights
        # IS weight = (1 / (N * P(i)))^beta where P(i) is sampling probability
        max_weight = max(importance_weights)
        is_weights = np.array([(max_weight / w) ** beta for w in importance_weights])
        
        # Normalize IS weights
        is_weights = is_weights / np.mean(is_weights)
        
        # Apply bias correction to gradients
        if gradients.ndim == 1:
            # Single gradient vector
            corrected_gradients = gradients * np.mean(is_weights)
        else:
            # Batch of gradients
            corrected_gradients = gradients * is_weights.reshape(-1, 1)
        
        return corrected_gradients
    
    def get_sampling_statistics(self) -> Dict:
        """Get statistics about sampling behavior."""
        if not self.experiences:
            return {}
        
        stats = {
            "total_experiences": self.total_experiences,
            "event_counts": dict(self.event_counts),
            "population_ratios": dict(self.population_ratios),
            "conversion_ratios": dict(self.conversion_ratios),
            "importance_weights": dict(self._importance_weights),
            "frame_count": self.frame_count
        }
        
        # Add value statistics per event type
        stats["event_value_stats"] = {}
        for event_type, values in self.event_values.items():
            if values:
                stats["event_value_stats"][event_type] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values)
                }
        
        return stats
    
    def save_state(self, filepath: str):
        """Save the sampler state to disk."""
        state = {
            "population_ratios": self.population_ratios,
            "conversion_ratios": self.conversion_ratios,
            "event_counts": dict(self.event_counts),
            "event_values": {k: list(v) for k, v in self.event_values.items()},
            "total_experiences": self.total_experiences,
            "frame_count": self.frame_count,
            "_importance_weights": self._importance_weights
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved importance sampler state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load the sampler state from disk."""
        if not os.path.exists(filepath):
            logger.warning(f"State file {filepath} not found, using defaults")
            return
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.population_ratios = state["population_ratios"]
        self.conversion_ratios = state["conversion_ratios"]
        self.event_counts = defaultdict(int, state["event_counts"])
        self.event_values = defaultdict(list, state["event_values"])
        self.total_experiences = state["total_experiences"]
        self.frame_count = state["frame_count"]
        self._importance_weights = state["_importance_weights"]
        
        logger.info(f"Loaded importance sampler state from {filepath}")
    
    def clear_buffer(self):
        """Clear the experience buffer while keeping statistics."""
        self.experiences.clear()
        logger.info("Cleared experience buffer")


class PrioritizedExperienceReplay(ImportanceSampler):
    """
    Extended importance sampler that implements prioritized experience replay
    with TD-error based prioritization in addition to importance sampling.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.td_errors = []
        self.priorities = []
    
    def add_experience_with_priority(
        self,
        experience: Experience,
        td_error: float
    ):
        """Add experience with TD-error for prioritization."""
        self.add_experience(experience)
        self.td_errors.append(abs(td_error) + self.epsilon)
        
        # Calculate combined priority from importance weight and TD-error
        importance_weight = self.calculate_importance_weight(experience)
        priority = importance_weight * (abs(td_error) + self.epsilon)
        self.priorities.append(priority)
    
    def weighted_sampling_with_td_priority(
        self,
        batch_size: int,
        temperature: float = 1.0
    ) -> Tuple[List[Experience], List[float], List[int]]:
        """Sample using combined importance weights and TD-error priorities."""
        if len(self.experiences) < batch_size:
            weights = [self.calculate_importance_weight(exp) for exp in self.experiences]
            return self.experiences[:], weights, list(range(len(self.experiences)))
        
        # Use combined priorities for sampling
        if self.priorities:
            weights = np.array(self.priorities)
        else:
            weights = np.array([
                self.calculate_importance_weight(exp) for exp in self.experiences
            ])
        
        # Apply temperature scaling
        if temperature != 1.0:
            weights = weights ** (1.0 / temperature)
        
        # Convert to probabilities
        probabilities = weights / (np.sum(weights) + self.epsilon)
        
        # Sample indices
        indices = np.random.choice(
            len(self.experiences),
            size=batch_size,
            replace=True,
            p=probabilities
        )
        
        sampled_experiences = [self.experiences[i] for i in indices]
        sampled_weights = weights[indices]
        
        return sampled_experiences, sampled_weights.tolist(), indices.tolist()
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities for sampled experiences based on new TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                experience = self.experiences[idx]
                importance_weight = self.calculate_importance_weight(experience)
                new_priority = importance_weight * (abs(td_error) + self.epsilon)
                self.priorities[idx] = new_priority
                
                if idx < len(self.td_errors):
                    self.td_errors[idx] = abs(td_error) + self.epsilon


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import random
    
    # Initialize sampler with crisis parent scenario
    sampler = ImportanceSampler(
        population_ratios={"crisis_parent": 0.1, "regular_parent": 0.9},
        conversion_ratios={"crisis_parent": 0.5, "regular_parent": 0.5}
    )
    
    # Generate sample experiences
    np.random.seed(42)
    for i in range(1000):
        # 90% regular parents, 10% crisis parents
        if np.random.random() < 0.1:
            event_type = "crisis_parent"
            value = np.random.normal(2.0, 0.5)  # Higher value for crisis parents
        else:
            event_type = "regular_parent"
            value = np.random.normal(1.0, 0.3)  # Lower value for regular parents
        
        experience = Experience(
            state=np.random.randn(10),
            action=np.random.randint(0, 4),
            reward=np.random.randn(),
            next_state=np.random.randn(10),
            done=np.random.random() < 0.1,
            value=value,
            event_type=event_type,
            timestamp=i,
            metadata={}
        )
        
        sampler.add_experience(experience)
    
    # Sample a batch
    batch_size = 32
    experiences, weights, indices = sampler.weighted_sampling(batch_size)
    
    # Check sampling distribution
    crisis_count = sum(1 for exp in experiences if exp.event_type == "crisis_parent")
    regular_count = batch_size - crisis_count
    
    print(f"Sampled batch composition:")
    print(f"Crisis parents: {crisis_count}/{batch_size} ({crisis_count/batch_size:.2%})")
    print(f"Regular parents: {regular_count}/{batch_size} ({regular_count/batch_size:.2%})")
    print(f"Average importance weights: {np.mean(weights):.3f}")
    
    # Print statistics
    stats = sampler.get_sampling_statistics()
    print(f"\nSampling statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")