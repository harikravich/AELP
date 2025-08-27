"""
Journey State Encoder with Enhanced Sequence Processing

Comprehensive state encoder that captures journey history, touchpoint sequences,
and temporal dynamics for sequential decision making in customer journey optimization.
Integrates LSTM embeddings for sequence features and provides PPO-compatible output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
from enum import Enum
import logging

from enhanced_journey_tracking import UserState, Channel, TouchpointType

logger = logging.getLogger(__name__)


class JourneyStage(Enum):
    """Customer journey stages for progression tracking"""
    AWARENESS = 0
    CONSIDERATION = 1
    CONVERSION = 2
    RETENTION = 3
    ADVOCACY = 4


@dataclass
class JourneySequence:
    """Sequence data structure for journey history"""
    touchpoints: List[int]  # Channel sequence
    states: List[int]       # State progression
    costs: List[float]      # Cost sequence
    timestamps: List[float] # Normalized time deltas
    
    def __len__(self):
        return len(self.touchpoints)
    
    def get_last_n(self, n: int) -> 'JourneySequence':
        """Get last n elements of sequence"""
        if len(self.touchpoints) <= n:
            return self
        return JourneySequence(
            touchpoints=self.touchpoints[-n:],
            states=self.states[-n:],
            costs=self.costs[-n:],
            timestamps=self.timestamps[-n:]
        )


@dataclass
class JourneyStateEncoderConfig:
    """Configuration for journey state encoder"""
    
    # Sequence processing
    max_sequence_length: int = 5
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    
    # Channel and state dimensions
    num_channels: int = 8
    num_states: int = 7
    num_stages: int = 5
    
    # Feature dimensions
    channel_embedding_dim: int = 16
    state_embedding_dim: int = 12
    temporal_embedding_dim: int = 8
    
    # Normalization parameters
    max_journey_days: int = 30
    max_touches: int = 20
    max_cost: float = 1000.0
    max_fatigue_level: float = 1.0
    
    # Output dimensions
    encoded_state_dim: int = 256
    
    # Training parameters
    normalize_features: bool = True
    use_layer_norm: bool = True
    use_attention: bool = True
    attention_heads: int = 4


class ChannelEmbedding(nn.Module):
    """Learnable embeddings for marketing channels"""
    
    def __init__(self, num_channels: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_channels, embedding_dim)
        self.num_channels = num_channels
        
    def forward(self, channel_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_ids: Tensor of shape (batch_size, seq_len) or (batch_size,)
        Returns:
            Channel embeddings of shape (batch_size, seq_len, embedding_dim) or (batch_size, embedding_dim)
        """
        # Clamp channel IDs to valid range
        channel_ids = torch.clamp(channel_ids.long(), 0, self.num_channels - 1)
        return self.embedding(channel_ids)


class StateEmbedding(nn.Module):
    """Learnable embeddings for user states"""
    
    def __init__(self, num_states: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_states, embedding_dim)
        self.num_states = num_states
        
    def forward(self, state_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_ids: Tensor of shape (batch_size, seq_len) or (batch_size,)
        Returns:
            State embeddings of shape (batch_size, seq_len, embedding_dim) or (batch_size, embedding_dim)
        """
        # Clamp state IDs to valid range
        state_ids = torch.clamp(state_ids.long(), 0, self.num_states - 1)
        return self.embedding(state_ids)


class TemporalEmbedding(nn.Module):
    """Sinusoidal temporal embeddings for time-based features"""
    
    def __init__(self, embedding_dim: int, max_len: int = 1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Create sinusoidal position embeddings
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-np.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: Normalized timestamps of shape (batch_size, seq_len)
        Returns:
            Temporal embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        # Scale timestamps to position indices
        positions = (timestamps * 999).long().clamp(0, 999)
        
        # Get embeddings
        batch_size, seq_len = positions.shape
        embeddings = self.pe[positions.view(-1)].view(batch_size, seq_len, self.embedding_dim)
        
        return embeddings


class SequenceLSTM(nn.Module):
    """LSTM for processing touchpoint sequences"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, sequence_embeddings: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence_embeddings: Tensor of shape (batch_size, seq_len, input_dim)
            lengths: Actual sequence lengths for packing
        Returns:
            output: LSTM output of shape (batch_size, seq_len, hidden_dim)
            final_state: Final hidden state of shape (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = sequence_embeddings.shape
        
        # Pack sequences if lengths provided
        if lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                sequence_embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_input)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(sequence_embeddings)
        
        # Get final state from last layer
        final_state = hidden[-1]  # Shape: (batch_size, hidden_dim)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        final_state = self.layer_norm(final_state)
        
        return output, final_state


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence features"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, sequence_features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            sequence_features: Tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)
        Returns:
            Pooled features of shape (batch_size, hidden_dim)
        """
        # Self-attention
        attended_features, _ = self.attention(
            sequence_features, sequence_features, sequence_features,
            key_padding_mask=mask
        )
        
        # Residual connection and layer norm
        features = self.layer_norm(sequence_features + attended_features)
        
        # Global average pooling (excluding masked positions)
        if mask is not None:
            # Mask is True for padded positions
            mask_expanded = mask.unsqueeze(-1).expand_as(features)
            features = features.masked_fill(mask_expanded, 0)
            pooled = features.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float()
        else:
            pooled = features.mean(dim=1)
        
        return pooled


class JourneyStateEncoder(nn.Module):
    """
    Enhanced journey state encoder with LSTM embeddings for sequence features.
    
    Encodes comprehensive journey context including:
    - Touchpoint sequence (last 3-5 channels)
    - Days in journey and stage progression
    - Competitors seen and user fatigue level
    - Channel distribution and time since last touch
    """
    
    def __init__(self, config: JourneyStateEncoderConfig):
        super().__init__()
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Embedding layers
        self.channel_embedding = ChannelEmbedding(
            config.num_channels, config.channel_embedding_dim
        )
        self.state_embedding = StateEmbedding(
            config.num_states, config.state_embedding_dim
        )
        self.temporal_embedding = TemporalEmbedding(
            config.temporal_embedding_dim
        )
        
        # Sequence processing
        sequence_input_dim = (config.channel_embedding_dim + 
                            config.state_embedding_dim + 
                            config.temporal_embedding_dim + 1)  # +1 for cost
        
        self.sequence_lstm = SequenceLSTM(
            input_dim=sequence_input_dim,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout
        )
        
        # Attention pooling
        if config.use_attention:
            self.attention_pooling = AttentionPooling(
                hidden_dim=config.lstm_hidden_dim,
                num_heads=config.attention_heads
            )
        
        # Static feature processing
        self.static_features_dim = self._calculate_static_features_dim()
        
        # Feature fusion network
        fusion_input_dim = (config.lstm_hidden_dim + self.static_features_dim)
        
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, config.encoded_state_dim),
            nn.LayerNorm(config.encoded_state_dim) if config.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.encoded_state_dim, config.encoded_state_dim),
            nn.LayerNorm(config.encoded_state_dim) if config.use_layer_norm else nn.Identity(),
            nn.ReLU()
        )
        
        # Feature normalization parameters (learned during training)
        self.register_buffer('feature_means', torch.zeros(self.static_features_dim))
        self.register_buffer('feature_stds', torch.ones(self.static_features_dim))
        self.register_buffer('normalization_fitted', torch.tensor(False))
        
    def _calculate_static_features_dim(self) -> int:
        """Calculate dimension of static (non-sequence) features"""
        dim = 0
        
        # Current state and journey metrics
        dim += 1  # Current state
        dim += 1  # Days in journey
        dim += 1  # Journey stage
        dim += 1  # Total touches
        dim += 1  # Conversion probability
        dim += 1  # User fatigue level
        
        # Time features
        dim += 1  # Time since last touch
        dim += 2  # Hour of day (sin/cos encoding)
        dim += 2  # Day of week (sin/cos encoding)
        dim += 2  # Day of month (sin/cos encoding)
        
        # Channel distribution features
        dim += self.config.num_channels  # Touch count per channel
        dim += self.config.num_channels  # Cost per channel
        dim += self.config.num_channels  # Days since last touch per channel
        
        # Performance features
        dim += 1  # CTR
        dim += 1  # Engagement rate
        dim += 1  # Bounce rate
        dim += 1  # Conversion rate
        
        # Competitor features
        dim += 1  # Competitors seen count
        dim += 1  # Competitor engagement rate
        
        return dim
    
    def encode_journey(self, journey_data: Dict[str, Any]) -> torch.Tensor:
        """
        Main encoding method that processes complete journey state.
        
        Args:
            journey_data: Dictionary containing all journey information
            
        Returns:
            Encoded state tensor compatible with PPO agent
        """
        # Extract sequence features
        sequence_features = self.get_sequence_features(journey_data)
        
        # Extract static features
        static_features = self._extract_static_features(journey_data)
        
        # Process sequence through LSTM
        batch_size = 1  # Single journey
        device = next(self.parameters()).device
        
        if sequence_features['touchpoints']:
            # Create sequence tensors
            seq_len = len(sequence_features['touchpoints'])
            
            channels = torch.tensor(sequence_features['touchpoints'], 
                                  dtype=torch.long, device=device).unsqueeze(0)
            states = torch.tensor(sequence_features['states'], 
                                dtype=torch.long, device=device).unsqueeze(0)
            costs = torch.tensor(sequence_features['costs'], 
                               dtype=torch.float, device=device).unsqueeze(0).unsqueeze(-1)
            timestamps = torch.tensor(sequence_features['timestamps'], 
                                    dtype=torch.float, device=device).unsqueeze(0)
            
            # Get embeddings
            channel_emb = self.channel_embedding(channels)  # (1, seq_len, emb_dim)
            state_emb = self.state_embedding(states)        # (1, seq_len, emb_dim)
            temporal_emb = self.temporal_embedding(timestamps)  # (1, seq_len, emb_dim)
            
            # Concatenate all sequence features
            sequence_input = torch.cat([channel_emb, state_emb, temporal_emb, costs], dim=-1)
            
            # Process through LSTM
            sequence_output, sequence_final = self.sequence_lstm(sequence_input)
            
            # Apply attention pooling if enabled
            if self.config.use_attention:
                sequence_encoding = self.attention_pooling(sequence_output)
            else:
                sequence_encoding = sequence_final
                
        else:
            # No sequence data available - use zero encoding
            sequence_encoding = torch.zeros(1, self.config.lstm_hidden_dim, device=device)
        
        # Normalize static features
        static_tensor = torch.tensor(static_features, dtype=torch.float, device=device)
        if self.config.normalize_features:
            static_tensor = self.normalize_features(static_tensor.unsqueeze(0)).squeeze(0)
        
        # Combine sequence and static features
        combined_features = torch.cat([
            sequence_encoding.squeeze(0),  # Remove batch dimension
            static_tensor
        ], dim=0)
        
        # Apply fusion network
        encoded_state = self.fusion_network(combined_features.unsqueeze(0)).squeeze(0)
        
        return encoded_state
    
    def get_sequence_features(self, journey_data: Dict[str, Any]) -> Dict[str, List]:
        """
        Extract sequence features from journey data.
        
        Args:
            journey_data: Raw journey data dictionary
            
        Returns:
            Dictionary with sequence feature lists
        """
        journey_history = journey_data.get('journey_history', [])
        
        # Limit to last N touchpoints
        max_len = self.config.max_sequence_length
        recent_history = journey_history[-max_len:] if len(journey_history) > max_len else journey_history
        
        sequence_features = {
            'touchpoints': [],
            'states': [],
            'costs': [],
            'timestamps': []
        }
        
        if not recent_history:
            return sequence_features
        
        # Extract features from each touchpoint
        base_timestamp = journey_data.get('current_timestamp', 0)
        
        for touchpoint in recent_history:
            # Channel (enum to int mapping)
            channel_name = touchpoint.get('channel', 'search')
            channel_id = self._channel_to_id(channel_name)
            sequence_features['touchpoints'].append(channel_id)
            
            # State (enum to int mapping)
            state_name = touchpoint.get('user_state_after', 'unaware')
            state_id = self._state_to_id(state_name)
            sequence_features['states'].append(state_id)
            
            # Cost
            cost = float(touchpoint.get('cost', 0.0))
            sequence_features['costs'].append(cost)
            
            # Normalized timestamp (days from current)
            timestamp = touchpoint.get('timestamp', base_timestamp)
            if isinstance(timestamp, str):
                from datetime import datetime
                timestamp = datetime.fromisoformat(timestamp).timestamp()
            
            time_delta = abs(base_timestamp - timestamp) / 86400.0  # Convert to days
            normalized_time = min(time_delta / self.config.max_journey_days, 1.0)
            sequence_features['timestamps'].append(normalized_time)
        
        return sequence_features
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize static features using learned statistics.
        
        Args:
            features: Raw feature tensor of shape (batch_size, feature_dim)
            
        Returns:
            Normalized feature tensor
        """
        if not self.normalization_fitted:
            # Initialize normalization parameters if not fitted
            self.feature_means = features.mean(dim=0)
            self.feature_stds = features.std(dim=0) + 1e-8
            self.normalization_fitted = torch.tensor(True)
        
        # Apply z-score normalization
        normalized = (features - self.feature_means) / self.feature_stds
        
        # Clamp to reasonable range
        normalized = torch.clamp(normalized, -5.0, 5.0)
        
        return normalized
    
    def _extract_static_features(self, journey_data: Dict[str, Any]) -> List[float]:
        """Extract static (non-sequence) features from journey data"""
        features = []
        
        # Current state and journey metrics
        current_state = journey_data.get('current_state', 'unaware')
        features.append(float(self._state_to_id(current_state)))
        
        days_in_journey = min(float(journey_data.get('days_in_journey', 0)), 
                             self.config.max_journey_days) / self.config.max_journey_days
        features.append(days_in_journey)
        
        journey_stage = float(journey_data.get('journey_stage', 0)) / (self.config.num_stages - 1)
        features.append(journey_stage)
        
        total_touches = min(float(journey_data.get('total_touches', 0)), 
                          self.config.max_touches) / self.config.max_touches
        features.append(total_touches)
        
        conversion_prob = float(journey_data.get('conversion_probability', 0.0))
        features.append(conversion_prob)
        
        fatigue_level = min(float(journey_data.get('user_fatigue_level', 0.0)), 
                          self.config.max_fatigue_level) / self.config.max_fatigue_level
        features.append(fatigue_level)
        
        # Time features
        time_since_last = min(float(journey_data.get('time_since_last_touch', 0)), 
                             self.config.max_journey_days) / self.config.max_journey_days
        features.append(time_since_last)
        
        # Cyclical time encoding
        hour = float(journey_data.get('hour_of_day', 12))
        features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24)
        ])
        
        day_of_week = float(journey_data.get('day_of_week', 3))
        features.extend([
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7)
        ])
        
        day_of_month = float(journey_data.get('day_of_month', 15))
        features.extend([
            np.sin(2 * np.pi * day_of_month / 31),
            np.cos(2 * np.pi * day_of_month / 31)
        ])
        
        # Channel distribution features
        channel_distribution = journey_data.get('channel_distribution', {})
        channel_costs = journey_data.get('channel_costs', {})
        channel_last_touch = journey_data.get('channel_last_touch', {})
        
        for i in range(self.config.num_channels):
            channel_name = self._id_to_channel(i)
            
            # Touch count per channel (normalized)
            touch_count = float(channel_distribution.get(channel_name, 0))
            normalized_count = min(touch_count / self.config.max_touches, 1.0)
            features.append(normalized_count)
            
            # Cost per channel (normalized)
            cost = float(channel_costs.get(channel_name, 0))
            normalized_cost = min(cost / self.config.max_cost, 1.0)
            features.append(normalized_cost)
            
            # Days since last touch per channel (normalized)
            days_since = float(channel_last_touch.get(channel_name, self.config.max_journey_days))
            normalized_days = min(days_since / self.config.max_journey_days, 1.0)
            features.append(normalized_days)
        
        # Performance features
        features.append(float(journey_data.get('click_through_rate', 0.025)))
        features.append(float(journey_data.get('engagement_rate', 0.1)))
        features.append(float(journey_data.get('bounce_rate', 0.5)))
        features.append(float(journey_data.get('conversion_rate', 0.05)))
        
        # Competitor features
        features.append(min(float(journey_data.get('competitors_seen', 0)), 10.0) / 10.0)
        features.append(float(journey_data.get('competitor_engagement_rate', 0.0)))
        
        return features
    
    def _channel_to_id(self, channel_name: str) -> int:
        """Map channel name to integer ID"""
        channel_mapping = {
            'search': 0, 'social': 1, 'display': 2, 'video': 3,
            'email': 4, 'direct': 5, 'affiliate': 6, 'retargeting': 7
        }
        return channel_mapping.get(channel_name.lower(), 0)
    
    def _id_to_channel(self, channel_id: int) -> str:
        """Map integer ID to channel name"""
        id_to_channel = {
            0: 'search', 1: 'social', 2: 'display', 3: 'video',
            4: 'email', 5: 'direct', 6: 'affiliate', 7: 'retargeting'
        }
        return id_to_channel.get(channel_id, 'search')
    
    def _state_to_id(self, state_name: str) -> int:
        """Map state name to integer ID"""
        state_mapping = {
            'unaware': 0, 'aware': 1, 'interested': 2, 'considering': 3,
            'intent': 4, 'converted': 5, 'churned': 6
        }
        return state_mapping.get(state_name.lower(), 0)
    
    def get_output_dim(self) -> int:
        """Get the output dimension of encoded states"""
        return self.config.encoded_state_dim
    
    def update_normalization_stats(self, feature_batch: torch.Tensor):
        """Update normalization statistics with new batch of features"""
        with torch.no_grad():
            batch_mean = feature_batch.mean(dim=0)
            batch_std = feature_batch.std(dim=0) + 1e-8
            
            if self.normalization_fitted:
                # Exponential moving average
                momentum = 0.1
                self.feature_means = (1 - momentum) * self.feature_means + momentum * batch_mean
                self.feature_stds = (1 - momentum) * self.feature_stds + momentum * batch_std
            else:
                self.feature_means = batch_mean
                self.feature_stds = batch_std
                self.normalization_fitted = torch.tensor(True)
    
    def get_config(self) -> JourneyStateEncoderConfig:
        """Get encoder configuration"""
        return self.config
    
    def forward(self, journey_data: Union[Dict[str, Any], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the encoder.
        
        Args:
            journey_data: Either journey data dict or pre-encoded tensor
            
        Returns:
            Encoded state tensor
        """
        if isinstance(journey_data, dict):
            return self.encode_journey(journey_data)
        else:
            # Assume already encoded tensor
            return journey_data


def create_journey_encoder(
    max_sequence_length: int = 5,
    lstm_hidden_dim: int = 64,
    encoded_state_dim: int = 256,
    **kwargs
) -> JourneyStateEncoder:
    """
    Factory function to create a journey state encoder with common configurations.
    
    Args:
        max_sequence_length: Maximum length of touchpoint sequences
        lstm_hidden_dim: Hidden dimension for LSTM
        encoded_state_dim: Output dimension of encoded states
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured JourneyStateEncoder instance
    """
    config = JourneyStateEncoderConfig(
        max_sequence_length=max_sequence_length,
        lstm_hidden_dim=lstm_hidden_dim,
        encoded_state_dim=encoded_state_dim,
        **kwargs
    )
    
    return JourneyStateEncoder(config)


# Example usage and testing functions
if __name__ == "__main__":
    # Example journey data structure
    example_journey = {
        'current_state': 'considering',
        'days_in_journey': 5,
        'journey_stage': 1,
        'total_touches': 3,
        'conversion_probability': 0.3,
        'user_fatigue_level': 0.2,
        'time_since_last_touch': 2.0,
        'hour_of_day': 14,
        'day_of_week': 2,
        'day_of_month': 15,
        'current_timestamp': 1640995200,  # Unix timestamp
        'journey_history': [
            {
                'channel': 'search',
                'user_state_after': 'aware',
                'cost': 2.50,
                'timestamp': 1640908800
            },
            {
                'channel': 'social',
                'user_state_after': 'interested',
                'cost': 1.20,
                'timestamp': 1640951200
            },
            {
                'channel': 'display',
                'user_state_after': 'considering',
                'cost': 3.80,
                'timestamp': 1640994000
            }
        ],
        'channel_distribution': {
            'search': 1, 'social': 1, 'display': 1, 'video': 0,
            'email': 0, 'direct': 0, 'affiliate': 0, 'retargeting': 0
        },
        'channel_costs': {
            'search': 2.50, 'social': 1.20, 'display': 3.80, 'video': 0.0,
            'email': 0.0, 'direct': 0.0, 'affiliate': 0.0, 'retargeting': 0.0
        },
        'channel_last_touch': {
            'search': 3.0, 'social': 1.5, 'display': 0.1, 'video': 30.0,
            'email': 30.0, 'direct': 30.0, 'affiliate': 30.0, 'retargeting': 30.0
        },
        'click_through_rate': 0.035,
        'engagement_rate': 0.15,
        'bounce_rate': 0.4,
        'conversion_rate': 0.08,
        'competitors_seen': 2,
        'competitor_engagement_rate': 0.12
    }
    
    # Create and test encoder
    encoder = create_journey_encoder(
        max_sequence_length=5,
        lstm_hidden_dim=64,
        encoded_state_dim=256
    )
    
    # Encode journey
    encoded_state = encoder.encode_journey(example_journey)
    print(f"Encoded state shape: {encoded_state.shape}")
    print(f"Encoded state sample: {encoded_state[:10]}")
    
    # Test sequence feature extraction
    sequence_features = encoder.get_sequence_features(example_journey)
    print(f"Sequence features: {sequence_features}")
    
    logger.info("Journey state encoder test completed successfully")