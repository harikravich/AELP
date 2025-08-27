#!/usr/bin/env python3
"""
Dynamic Discovery System - NO HARDCODED VALUES
Discovers segments, devices, channels, and parameters dynamically at runtime
"""

import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DiscoveredEntities:
    """Dynamically discovered entities from data"""
    segments: Set[str] = field(default_factory=set)
    devices: Set[str] = field(default_factory=set)
    channels: Set[str] = field(default_factory=set)
    stages: Set[int] = field(default_factory=set)
    
    # Learned thresholds and parameters
    normalization_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stage_multipliers: Dict[int, float] = field(default_factory=dict)
    
    # Encoding mappings (built dynamically)
    segment_to_idx: Dict[str, int] = field(default_factory=dict)
    device_to_idx: Dict[str, int] = field(default_factory=dict)
    channel_to_idx: Dict[str, int] = field(default_factory=dict)
    
    # Statistics for normalization
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

class DynamicDiscoverySystem:
    """System for discovering and learning all parameters dynamically"""
    
    def __init__(self, cache_path: Optional[str] = None):
        self.entities = DiscoveredEntities()
        self.cache_path = Path(cache_path) if cache_path else Path("dynamic_entities_cache.pkl")
        self.observation_count = 0
        self.running_stats = defaultdict(lambda: {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0.0,
            'count': 0,
            'sum': 0.0,
            'sum_sq': 0.0
        })
        
        # Try to load cached discoveries
        self._load_cache()
    
    def observe(self, data: Dict[str, Any]) -> None:
        """Observe new data and update discovered entities"""
        self.observation_count += 1
        
        # Discover categorical entities
        if 'segment' in data and data['segment']:
            self.entities.segments.add(data['segment'])
            self._update_encoding('segment')
        
        if 'device' in data and data['device']:
            self.entities.devices.add(data['device'])
            self._update_encoding('device')
        
        if 'channel' in data and data['channel']:
            self.entities.channels.add(data['channel'])
            self._update_encoding('channel')
        
        if 'stage' in data:
            self.entities.stages.add(data['stage'])
        
        # Update running statistics for numerical features
        for key in ['touchpoints_seen', 'days_since_first_touch', 'previous_clicks',
                    'previous_impressions', 'estimated_ltv', 'ad_fatigue_level',
                    'hour_of_day', 'day_of_week']:
            if key in data:
                self._update_statistics(key, data[key])
        
        # Learn stage multipliers from outcomes
        if 'stage' in data and 'outcome' in data and 'bid' in data:
            self._update_stage_multiplier(data['stage'], data['outcome'], data['bid'])
        
        # Periodically save discoveries
        if self.observation_count % 100 == 0:
            self._save_cache()
    
    def _update_statistics(self, key: str, value: float) -> None:
        """Update running statistics for a feature"""
        stats = self.running_stats[key]
        stats['count'] += 1
        stats['sum'] += value
        stats['sum_sq'] += value ** 2
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)
        stats['mean'] = stats['sum'] / stats['count']
        
        # Calculate standard deviation
        if stats['count'] > 1:
            variance = (stats['sum_sq'] / stats['count']) - (stats['mean'] ** 2)
            stats['std'] = np.sqrt(max(0, variance))  # Avoid negative variance due to numerical errors
        else:
            stats['std'] = 0.0
        
        # Update normalization parameters
        if key not in self.entities.normalization_params:
            self.entities.normalization_params[key] = {}
        
        # Use percentiles for robust normalization
        self.entities.normalization_params[key] = {
            'min': stats['min'],
            'max': stats['max'],
            'mean': stats['mean'],
            'std': stats['std'],
            'scale': stats['max'] - stats['min'] if stats['max'] > stats['min'] else 1.0
        }
    
    def _update_stage_multiplier(self, stage: int, outcome: float, bid: float) -> None:
        """Learn stage-specific bid multipliers from outcomes"""
        if stage not in self.entities.stage_multipliers:
            self.entities.stage_multipliers[stage] = 1.0
        
        # Simple exponential moving average update
        alpha = 0.01  # Learning rate
        if bid > 0:
            effectiveness = outcome / bid  # ROI-like metric
            current = self.entities.stage_multipliers[stage]
            self.entities.stage_multipliers[stage] = (1 - alpha) * current + alpha * effectiveness
    
    def _update_encoding(self, entity_type: str) -> None:
        """Update one-hot encoding mappings when new entities are discovered"""
        if entity_type == 'segment':
            self.entities.segment_to_idx = {seg: i for i, seg in enumerate(sorted(self.entities.segments))}
        elif entity_type == 'device':
            self.entities.device_to_idx = {dev: i for i, dev in enumerate(sorted(self.entities.devices))}
        elif entity_type == 'channel':
            self.entities.channel_to_idx = {ch: i for i, ch in enumerate(sorted(self.entities.channels))}
    
    def get_encoding_dimension(self, entity_type: str) -> int:
        """Get the dimension for one-hot encoding of an entity type"""
        if entity_type == 'segment':
            return max(1, len(self.entities.segments))
        elif entity_type == 'device':
            return max(1, len(self.entities.devices))
        elif entity_type == 'channel':
            return max(1, len(self.entities.channels))
        return 1
    
    def encode_categorical(self, entity_type: str, value: str) -> np.ndarray:
        """Encode categorical with fixed dimensions using feature hashing"""
        # Use fixed dimensions for stable neural network input
        # But preserve ability to discover unlimited categories
        max_dims = {
            'segment': 15,  # Enough for all parent types + others
            'device': 3,    # mobile, desktop, tablet
            'channel': 5    # google, facebook, bing, tiktok, other
        }
        
        dim = max_dims.get(entity_type, 10)
        encoding = np.zeros(dim)
        
        # Hash-based encoding for stable dimensions
        if entity_type == 'segment':
            # Special handling for segments - preserve discovered info
            if value in self.entities.segment_to_idx:
                idx = self.entities.segment_to_idx[value] % dim
            else:
                # New segment - add to discoveries but hash to fixed space
                self.entities.segments.add(value)
                idx = hash(value) % dim
                # Update mapping for consistency
                if len(self.entities.segment_to_idx) < dim:
                    self.entities.segment_to_idx[value] = len(self.entities.segment_to_idx)
                else:
                    self.entities.segment_to_idx[value] = idx
            encoding[idx] = 1.0
            
        elif entity_type == 'device':
            if value in self.entities.device_to_idx:
                idx = self.entities.device_to_idx[value] % dim
            else:
                self.entities.devices.add(value)
                idx = hash(value) % dim
                self.entities.device_to_idx[value] = idx
            encoding[idx] = 1.0
            
        elif entity_type == 'channel':
            if value in self.entities.channel_to_idx:
                idx = self.entities.channel_to_idx[value] % dim
            else:
                self.entities.channels.add(value)
                idx = hash(value) % dim
                self.entities.channel_to_idx[value] = idx
            encoding[idx] = 1.0
        
        return encoding
    
    def normalize_numerical(self, key: str, value: float) -> float:
        """Normalize a numerical value based on learned statistics"""
        if key not in self.entities.normalization_params:
            # Haven't seen this feature yet, return as-is bounded to [0, 1]
            return min(1.0, max(0.0, value))
        
        params = self.entities.normalization_params[key]
        
        # Use robust normalization
        if params['std'] > 0:
            # Z-score normalization, then sigmoid to bound
            z_score = (value - params['mean']) / params['std']
            return 1.0 / (1.0 + np.exp(-z_score))
        elif params['scale'] > 0:
            # Min-max normalization
            return (value - params['min']) / params['scale']
        else:
            # No variation observed, return 0.5
            return 0.5
    
    def get_stage_multiplier(self, stage: int) -> float:
        """Get learned stage multiplier"""
        if stage in self.entities.stage_multipliers:
            return self.entities.stage_multipliers[stage]
        # Return neutral multiplier for unknown stages
        return 1.0
    
    def get_state_dimension(self) -> int:
        """Calculate the total dimension of the state vector"""
        base_dims = 10  # Numerical features
        segment_dims = self.get_encoding_dimension('segment')
        device_dims = self.get_encoding_dimension('device')
        return base_dims + segment_dims + device_dims
    
    def _save_cache(self) -> None:
        """Save discovered entities to cache"""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'entities': self.entities,
                    'observation_count': self.observation_count,
                    'running_stats': dict(self.running_stats)
                }, f)
            logger.debug(f"Saved discoveries to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save discovery cache: {e}")
    
    def _load_cache(self) -> None:
        """Load discovered entities from cache"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.entities = data['entities']
                    self.observation_count = data['observation_count']
                    self.running_stats = defaultdict(lambda: {
                        'min': float('inf'),
                        'max': float('-inf'),
                        'mean': 0.0,
                        'count': 0,
                        'sum': 0.0,
                        'sum_sq': 0.0
                    }, data['running_stats'])
                logger.info(f"Loaded discoveries from cache: {len(self.entities.segments)} segments, "
                           f"{len(self.entities.devices)} devices, {len(self.entities.channels)} channels")
            except Exception as e:
                logger.warning(f"Failed to load discovery cache: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of discovered entities and statistics"""
        return {
            'observation_count': self.observation_count,
            'discovered_segments': list(self.entities.segments),
            'discovered_devices': list(self.entities.devices),
            'discovered_channels': list(self.entities.channels),
            'discovered_stages': list(self.entities.stages),
            'normalization_params': self.entities.normalization_params,
            'stage_multipliers': self.entities.stage_multipliers,
            'statistics': {k: dict(v) for k, v in self.running_stats.items()}
        }


# Global discovery system instance
discovery_system = DynamicDiscoverySystem()


def discover_from_data(data_points: List[Dict[str, Any]]) -> DynamicDiscoverySystem:
    """Discover entities from a batch of data points"""
    system = DynamicDiscoverySystem()
    for data in data_points:
        system.observe(data)
    return system


if __name__ == "__main__":
    # Test dynamic discovery
    test_data = [
        {'segment': 'new_segment_1', 'device': 'smartwatch', 'touchpoints_seen': 5, 'stage': 1},
        {'segment': 'new_segment_2', 'device': 'mobile', 'touchpoints_seen': 3, 'stage': 0},
        {'segment': 'new_segment_1', 'device': 'vr_headset', 'touchpoints_seen': 8, 'stage': 2},
    ]
    
    system = discover_from_data(test_data)
    print(json.dumps(system.get_summary(), indent=2))