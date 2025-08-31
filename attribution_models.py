"""
Multi-Touch Attribution Models for GAELP

This module implements various attribution models for assigning credit to touchpoints
in multi-touch customer journeys, enabling proper reward assignment in reinforcement
learning environments.

Now enhanced with conversion lag model integration for dynamic attribution windows.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging

# Import conversion lag model for dynamic attribution windows - REQUIRED
from conversion_lag_model import ConversionLagModel, ConversionJourney

logger = logging.getLogger(__name__)


@dataclass
class Touchpoint:
    """Represents a single touchpoint in a customer journey."""
    id: str
    timestamp: datetime
    channel: str
    action: str
    value: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Journey:
    """Represents a complete customer journey with touchpoints and conversion."""
    id: str
    touchpoints: List[Touchpoint]
    conversion_value: float
    conversion_timestamp: datetime
    converted: bool = True

    def __post_init__(self):
        # Sort touchpoints by timestamp
        self.touchpoints.sort(key=lambda x: x.timestamp)


class AttributionModel(ABC):
    """Abstract base class for attribution models.
    
    NOW WITH iOS 14.5+ PRIVACY NOISE:
    - Adds 20-30% attribution uncertainty to reflect real-world privacy limitations
    - All derived models automatically inherit this noise
    """
    
    # iOS 14.5+ Reality: Attribution has significant uncertainty
    IOS_PRIVACY_NOISE_LEVEL = 0.25  # 25% uncertainty baseline
    
    def __init__(self, add_privacy_noise: bool = True):
        """Initialize with optional privacy noise for realism."""
        self.add_privacy_noise = add_privacy_noise

    @abstractmethod
    def calculate_attribution(self, journey: Journey) -> Dict[str, float]:
        """Calculate attribution weights for each touchpoint in the journey."""
        pass
    
    def _apply_privacy_noise(self, attributions: Dict[str, float]) -> Dict[str, float]:
        """Apply iOS 14.5+ privacy noise to attribution values.
        
        Adds realistic uncertainty to attribution to reflect:
        - Limited cross-device tracking
        - Delayed/missing conversion signals
        - Privacy-preserving aggregation
        """
        if not self.add_privacy_noise:
            return attributions
            
        noisy_attributions = {}
        total_after_noise = 0.0
        
        for touchpoint_id, value in attributions.items():
            # Add Gaussian noise with 25% standard deviation
            noise = np.random.normal(0, self.IOS_PRIVACY_NOISE_LEVEL)
            # Ensure non-negative values
            noisy_value = max(0, value * (1 + noise))
            noisy_attributions[touchpoint_id] = noisy_value
            total_after_noise += noisy_value
        
        # Renormalize to sum to 1.0 (preserving relative weights after noise)
        if total_after_noise > 0:
            for touchpoint_id in noisy_attributions:
                noisy_attributions[touchpoint_id] /= total_after_noise
        
        return noisy_attributions

    def distribute_credit(self, journey: Journey) -> List[Tuple[Touchpoint, float]]:
        """Distribute conversion credit among touchpoints with privacy noise."""
        attributions = self.calculate_attribution(journey)
        # Apply privacy noise
        attributions = self._apply_privacy_noise(attributions)
        return [
            (tp, attributions.get(tp.id, 0.0) * journey.conversion_value)
            for tp in journey.touchpoints
        ]

    def get_touchpoint_value(self, touchpoint_id: str, journey: Journey) -> float:
        """Get the attributed value for a specific touchpoint with privacy noise."""
        attributions = self.calculate_attribution(journey)
        # Apply privacy noise
        attributions = self._apply_privacy_noise(attributions)
        return attributions.get(touchpoint_id, 0.0) * journey.conversion_value


class TimeDecayAttribution(AttributionModel):
    """
    Time-decay attribution model with exponential decay.
    More recent touchpoints receive higher attribution.
    """

    def __init__(self, half_life_days: int = 7, add_privacy_noise: bool = True):
        """
        Initialize time-decay attribution model.
        
        Args:
            half_life_days: Number of days for attribution to decay by half
            add_privacy_noise: Whether to add iOS 14.5+ privacy noise
        """
        super().__init__(add_privacy_noise=add_privacy_noise)
        self.half_life_days = half_life_days
        self.decay_constant = np.log(2) / (half_life_days * 24 * 3600)  # per second

    def calculate_attribution(self, journey: Journey) -> Dict[str, float]:
        """Calculate time-decay attribution weights."""
        if not journey.touchpoints:
            return {}

        weights = {}
        total_weight = 0.0
        conversion_time = journey.conversion_timestamp

        for touchpoint in journey.touchpoints:
            # Calculate time difference in seconds
            time_diff = (conversion_time - touchpoint.timestamp).total_seconds()
            
            # Apply exponential decay
            weight = np.exp(-self.decay_constant * time_diff)
            weights[touchpoint.id] = weight
            total_weight += weight

        # Normalize weights to sum to 1
        if total_weight > 0:
            for tp_id in weights:
                weights[tp_id] /= total_weight

        return weights


class PositionBasedAttribution(AttributionModel):
    """
    Position-based (U-shaped) attribution model.
    First and last touchpoints get higher attribution.
    """

    def __init__(self, first_weight: float = 0.4, last_weight: float = 0.4, 
                 middle_weight: float = 0.2, add_privacy_noise: bool = True):
        """
        Initialize position-based attribution model.
        
        Args:
            first_weight: Attribution weight for first touchpoint
            last_weight: Attribution weight for last touchpoint
            middle_weight: Attribution weight for middle touchpoints (distributed equally)
            add_privacy_noise: Whether to add iOS 14.5+ privacy noise
        """
        super().__init__(add_privacy_noise=add_privacy_noise)
        if abs(first_weight + last_weight + middle_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.first_weight = first_weight
        self.last_weight = last_weight
        self.middle_weight = middle_weight

    def calculate_attribution(self, journey: Journey) -> Dict[str, float]:
        """Calculate position-based attribution weights."""
        if not journey.touchpoints:
            return {}

        num_touchpoints = len(journey.touchpoints)
        weights = {}

        if num_touchpoints == 1:
            # Single touchpoint gets all attribution
            weights[journey.touchpoints[0].id] = 1.0
        elif num_touchpoints == 2:
            # First and last touchpoints split attribution
            weights[journey.touchpoints[0].id] = self.first_weight / (self.first_weight + self.last_weight)
            weights[journey.touchpoints[1].id] = self.last_weight / (self.first_weight + self.last_weight)
        else:
            # First touchpoint
            weights[journey.touchpoints[0].id] = self.first_weight
            
            # Last touchpoint
            weights[journey.touchpoints[-1].id] = self.last_weight
            
            # Middle touchpoints (equal distribution)
            middle_touchpoints = num_touchpoints - 2
            middle_weight_per_touchpoint = self.middle_weight / middle_touchpoints
            
            for i in range(1, num_touchpoints - 1):
                weights[journey.touchpoints[i].id] = middle_weight_per_touchpoint

        return weights


class LinearAttribution(AttributionModel):
    """
    Linear attribution model.
    All touchpoints receive equal credit.
    """
    
    def __init__(self, add_privacy_noise: bool = True):
        """Initialize linear attribution with optional privacy noise."""
        super().__init__(add_privacy_noise=add_privacy_noise)

    def calculate_attribution(self, journey: Journey) -> Dict[str, float]:
        """Calculate linear attribution weights."""
        if not journey.touchpoints:
            return {}

        num_touchpoints = len(journey.touchpoints)
        weight_per_touchpoint = 1.0 / num_touchpoints

        return {tp.id: weight_per_touchpoint for tp in journey.touchpoints}


class DataDrivenAttribution(AttributionModel):
    """
    Data-driven attribution model based on conversion probability lift.
    Uses machine learning to determine optimal attribution weights.
    """

    def __init__(self, add_privacy_noise: bool = True):
        """Initialize data-driven attribution model."""
        super().__init__(add_privacy_noise=add_privacy_noise)
        self.conversion_probability_model = None
        self.baseline_conversion_rate = 0.0
        self.is_trained = False

    def train(self, journeys: List[Journey], validation_split: float = 0.2):
        """
        Train the data-driven attribution model.
        
        Args:
            journeys: List of customer journeys for training
            validation_split: Fraction of data to use for validation
        """
        # NO FALLBACKS - sklearn is REQUIRED
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        # Prepare training data
        X, y = self._prepare_training_data(journeys)
        
        if len(X) < 10:  # Minimum samples required
            raise ValueError("Insufficient training data. Need at least 10 samples. NO FALLBACKS!")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Train model
        self.conversion_probability_model = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.conversion_probability_model.fit(X_train, y_train)

        # Calculate baseline conversion rate
        self.baseline_conversion_rate = np.mean(y_train)

        # Validate model
        val_predictions = self.conversion_probability_model.predict(X_val)
        accuracy = accuracy_score(y_val, val_predictions)
        
        logger.info(f"Data-driven attribution model trained with accuracy: {accuracy:.3f}")
        self.is_trained = True

    def _prepare_training_data(self, journeys: List[Journey]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from journeys."""
        features = []
        labels = []

        for journey in journeys:
            if not journey.touchpoints:
                continue

            # Extract features for each touchpoint combination
            for i, touchpoint in enumerate(journey.touchpoints):
                feature_vector = self._extract_features(touchpoint, journey, i)
                features.append(feature_vector)
                labels.append(1 if journey.converted else 0)

        return np.array(features), np.array(labels)

    def _extract_features(self, touchpoint: Touchpoint, journey: Journey, 
                         position: int) -> List[float]:
        """Extract features for a touchpoint."""
        features = []
        
        # Position features
        features.append(position)  # Absolute position
        features.append(position / len(journey.touchpoints))  # Relative position
        
        # Time features
        time_to_conversion = (journey.conversion_timestamp - touchpoint.timestamp).total_seconds()
        features.append(time_to_conversion / (24 * 3600))  # Days to conversion
        
        # Channel encoding (simple one-hot for common channels)
        common_channels = ['email', 'social', 'search', 'direct', 'referral']
        for channel in common_channels:
            features.append(1.0 if touchpoint.channel == channel else 0.0)
        
        # Journey features
        features.append(len(journey.touchpoints))  # Journey length
        features.append(journey.conversion_value)  # Conversion value
        
        return features

    def calculate_attribution(self, journey: Journey) -> Dict[str, float]:
        """Calculate data-driven attribution weights."""
        if not self.is_trained or not journey.touchpoints:
            # Fallback to linear attribution
            return LinearAttribution().calculate_attribution(journey)

        attribution_scores = {}
        total_score = 0.0

        for i, touchpoint in enumerate(journey.touchpoints):
            # Calculate probability lift
            feature_vector = self._extract_features(touchpoint, journey, i)
            
            try:
                probability = self.conversion_probability_model.predict_proba(
                    [feature_vector]
                )[0][1]  # Probability of conversion
                
                # Calculate lift over baseline
                lift = max(0.0, probability - self.baseline_conversion_rate)
                attribution_scores[touchpoint.id] = lift
                total_score += lift
                
            except Exception as e:
                logger.warning(f"Error calculating attribution for touchpoint {touchpoint.id}: {e}")
                attribution_scores[touchpoint.id] = 0.0

        # Normalize scores
        if total_score > 0:
            for tp_id in attribution_scores:
                attribution_scores[tp_id] /= total_score
        else:
            # Fallback to equal attribution
            equal_weight = 1.0 / len(journey.touchpoints)
            attribution_scores = {tp.id: equal_weight for tp in journey.touchpoints}

        return attribution_scores


class AttributionEngine:
    """
    Main attribution engine that manages different attribution models
    and provides a unified interface for calculating attributions.
    Enhanced with conversion lag model integration for dynamic attribution windows.
    """

    def __init__(self, conversion_lag_model: Optional['ConversionLagModel'] = None):
        """
        Initialize attribution engine with available models.
        
        Args:
            conversion_lag_model: Optional ConversionLagModel for dynamic attribution windows
        """
        self.models = {
            'time_decay': TimeDecayAttribution(),
            'position_based': PositionBasedAttribution(),
            'linear': LinearAttribution(),
            'data_driven': DataDrivenAttribution()
        }
        
        # Conversion lag model for dynamic attribution windows
        self.conversion_lag_model = conversion_lag_model
        self.dynamic_attribution_enabled = (conversion_lag_model is not None and 
                                          CONVERSION_LAG_MODEL_AVAILABLE)

    def add_model(self, name: str, model: AttributionModel):
        """Add a custom attribution model."""
        self.models[name] = model

    def calculate_attribution(self, journey: Journey, model_name: str = 'linear') -> Dict[str, float]:
        """
        Calculate attribution using specified model.
        
        Args:
            journey: Customer journey to analyze
            model_name: Name of attribution model to use
            
        Returns:
            Dictionary mapping touchpoint IDs to attribution weights
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")

        model = self.models[model_name]
        return model.calculate_attribution(journey)

    def distribute_credit(self, journey: Journey, model_name: str = 'linear') -> List[Tuple[Touchpoint, float]]:
        """
        Distribute conversion credit among touchpoints.
        
        Args:
            journey: Customer journey to analyze
            model_name: Name of attribution model to use
            
        Returns:
            List of tuples (touchpoint, attributed_value)
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")

        model = self.models[model_name]
        return model.distribute_credit(journey)

    def get_touchpoint_value(self, touchpoint_id: str, journey: Journey, 
                           model_name: str = 'linear') -> float:
        """
        Get attributed value for a specific touchpoint.
        
        Args:
            touchpoint_id: ID of the touchpoint
            journey: Customer journey containing the touchpoint
            model_name: Name of attribution model to use
            
        Returns:
            Attributed value for the touchpoint
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")

        model = self.models[model_name]
        return model.get_touchpoint_value(touchpoint_id, journey)

    def compare_models(self, journey: Journey, model_names: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare attribution results across multiple models.
        
        Args:
            journey: Customer journey to analyze
            model_names: List of model names to compare (defaults to all models)
            
        Returns:
            Dictionary with model names as keys and attribution results as values
        """
        if model_names is None:
            model_names = list(self.models.keys())

        results = {}
        for model_name in model_names:
            if model_name in self.models:
                try:
                    results[model_name] = self.calculate_attribution(journey, model_name)
                except Exception as e:
                    logger.error(f"Error calculating attribution for model {model_name}: {e}")
                    results[model_name] = {}

        return results

    def train_data_driven_model(self, journeys: List[Journey]):
        """Train the data-driven attribution model."""
        if 'data_driven' in self.models:
            self.models['data_driven'].train(journeys)
    
    def calculate_dynamic_attribution_window(self, journey: Journey) -> int:
        """
        Calculate dynamic attribution window based on conversion lag predictions.
        
        Args:
            journey: Customer journey to analyze
            
        Returns:
            Recommended attribution window in days
        """
        if not self.dynamic_attribution_enabled or not self.conversion_lag_model.is_fitted:
            return 30  # Default attribution window
        
        try:
            # Convert Journey to ConversionJourney format
            conversion_journey = self._convert_to_conversion_journey(journey)
            
            # Get conversion predictions
            predictions = self.conversion_lag_model.predict_conversion_time([conversion_journey])
            hazard_rates = self.conversion_lag_model.calculate_hazard_rate([conversion_journey])
            
            # Calculate optimal window based on where conversion probability plateaus
            conversion_probs = predictions.get(conversion_journey.user_id, np.array([]))
            
            if len(conversion_probs) == 0:
                return 30
            
            # Find point where 95% of conversion probability is reached
            max_prob = np.max(conversion_probs)
            target_prob = max_prob * 0.95
            
            optimal_window = 30  # Default
            for day, prob in enumerate(conversion_probs, 1):
                if prob >= target_prob:
                    optimal_window = min(day + 7, 60)  # Add buffer, max 60 days
                    break
            
            logger.info(f"Dynamic attribution window calculated: {optimal_window} days")
            return optimal_window
            
        except Exception as e:
            logger.error(f"Error calculating dynamic attribution window: {e}")
            return 30
    
    def _convert_to_conversion_journey(self, journey: Journey) -> 'ConversionJourney':
        """
        Convert Journey object to ConversionJourney for lag model compatibility.
        
        Args:
            journey: Journey object to convert
            
        Returns:
            ConversionJourney object
        """
        if not CONVERSION_LAG_MODEL_AVAILABLE:
            raise ImportError("ConversionLagModel not available")
        
        # Convert touchpoints to the format expected by ConversionJourney
        touchpoints_data = []
        for tp in journey.touchpoints:
            touchpoints_data.append({
                'timestamp': tp.timestamp,
                'channel': tp.channel,
                'action': tp.action,
                'value': tp.value,
                'metadata': tp.metadata
            })
        
        # Extract features from journey
        features = {
            'touchpoint_count': len(journey.touchpoints),
            'total_value': journey.conversion_value,
            'journey_length_hours': (journey.conversion_timestamp - journey.touchpoints[0].timestamp).total_seconds() / 3600 if journey.touchpoints else 0
        }
        
        return ConversionJourney(
            user_id=journey.id.split('_')[0] if '_' in journey.id else journey.id,  # Extract user_id
            start_time=journey.touchpoints[0].timestamp if journey.touchpoints else journey.conversion_timestamp,
            end_time=journey.conversion_timestamp if journey.converted else None,
            converted=journey.converted,
            touchpoints=touchpoints_data,
            features=features
        )
    
    def calculate_attribution_with_dynamic_window(self, 
                                                journey: Journey, 
                                                model_name: str = 'linear',
                                                use_dynamic_window: bool = True) -> Tuple[Dict[str, float], int]:
        """
        Calculate attribution with dynamically sized attribution window.
        
        Args:
            journey: Customer journey to analyze
            model_name: Attribution model to use
            use_dynamic_window: Whether to use dynamic attribution window
            
        Returns:
            Tuple of (attribution_weights, attribution_window_days)
        """
        if use_dynamic_window and self.dynamic_attribution_enabled:
            window_days = self.calculate_dynamic_attribution_window(journey)
            
            # Filter touchpoints based on dynamic window
            cutoff_time = journey.conversion_timestamp - timedelta(days=window_days)
            filtered_touchpoints = [
                tp for tp in journey.touchpoints 
                if tp.timestamp >= cutoff_time
            ]
            
            # Create new journey with filtered touchpoints
            filtered_journey = Journey(
                id=journey.id,
                touchpoints=filtered_touchpoints,
                conversion_value=journey.conversion_value,
                conversion_timestamp=journey.conversion_timestamp,
                converted=journey.converted
            )
            
            attributions = self.calculate_attribution(filtered_journey, model_name)
            
            logger.info(f"Applied dynamic attribution window of {window_days} days, "
                       f"included {len(filtered_touchpoints)}/{len(journey.touchpoints)} touchpoints")
            
            return attributions, window_days
        else:
            # Use standard attribution
            attributions = self.calculate_attribution(journey, model_name)
            return attributions, 30  # Default window
    
    def get_conversion_timing_insights(self, journey: Journey) -> Optional[Dict[str, Any]]:
        """
        Get conversion timing insights from the lag model.
        
        Args:
            journey: Customer journey to analyze
            
        Returns:
            Dictionary with conversion timing insights or None if not available
        """
        if not self.dynamic_attribution_enabled:
            return None
        
        try:
            conversion_journey = self._convert_to_conversion_journey(journey)
            
            # Get predictions and hazard rates
            predictions = self.conversion_lag_model.predict_conversion_time([conversion_journey])
            hazard_rates = self.conversion_lag_model.calculate_hazard_rate([conversion_journey])
            
            conversion_probs = predictions.get(conversion_journey.user_id, np.array([]))
            hazard_data = hazard_rates.get(conversion_journey.user_id, np.array([]))
            
            if len(conversion_probs) == 0:
                return None
            
            # Calculate key metrics
            peak_conversion_day = np.argmax(hazard_data) + 1 if len(hazard_data) > 0 else 1
            median_conversion_day = None
            
            # Find median (50% probability)
            cumulative_prob = 0
            for day, prob in enumerate(conversion_probs, 1):
                cumulative_prob = prob
                if cumulative_prob >= 0.5:
                    median_conversion_day = day
                    break
            
            return {
                'peak_conversion_day': int(peak_conversion_day),
                'median_conversion_day': int(median_conversion_day) if median_conversion_day else None,
                'max_conversion_probability': float(np.max(conversion_probs)),
                'conversion_probabilities_7_days': conversion_probs[:7].tolist() if len(conversion_probs) >= 7 else conversion_probs.tolist(),
                'conversion_probabilities_30_days': conversion_probs[:30].tolist() if len(conversion_probs) >= 30 else conversion_probs.tolist(),
                'recommended_attribution_window': self.calculate_dynamic_attribution_window(journey)
            }
            
        except Exception as e:
            logger.error(f"Error getting conversion timing insights: {e}")
            return None


# Utility functions for GAELP integration

def create_journey_from_episode(episode_data: Dict[str, Any]) -> Journey:
    """
    Create a Journey object from GAELP episode data.
    
    Args:
        episode_data: Dictionary containing episode information
        
    Returns:
        Journey object for attribution analysis
    """
    touchpoints = []
    
    for i, (state, action, reward, timestamp) in enumerate(
        zip(episode_data.get('states', []),
            episode_data.get('actions', []),
            episode_data.get('rewards', []),
            episode_data.get('timestamps', []))
    ):
        touchpoint = Touchpoint(
            id=f"{episode_data.get('episode_id', 'unknown')}_{i}",
            timestamp=timestamp,
            channel=action.get('channel', 'unknown'),
            action=action.get('type', 'unknown'),
            metadata={'state': state, 'reward': reward}
        )
        touchpoints.append(touchpoint)
    
    journey = Journey(
        id=episode_data.get('episode_id', 'unknown'),
        touchpoints=touchpoints,
        conversion_value=episode_data.get('total_reward', 0.0),
        conversion_timestamp=episode_data.get('end_timestamp', datetime.now()),
        converted=episode_data.get('success', False)
    )
    
    return journey


def calculate_multi_touch_rewards(episode_data: Dict[str, Any], 
                                model_name: str = 'time_decay',
                                conversion_lag_model: Optional['ConversionLagModel'] = None,
                                use_dynamic_window: bool = True) -> Tuple[List[float], Dict[str, Any]]:
    """
    Calculate multi-touch attribution rewards for a GAELP episode with dynamic attribution windows.
    
    Args:
        episode_data: Dictionary containing episode information
        model_name: Attribution model to use
        conversion_lag_model: Optional ConversionLagModel for dynamic windows
        use_dynamic_window: Whether to use dynamic attribution windows
        
    Returns:
        Tuple of (attributed_rewards, attribution_metadata)
    """
    engine = AttributionEngine(conversion_lag_model=conversion_lag_model)
    journey = create_journey_from_episode(episode_data)
    
    # Calculate attribution with dynamic windows if enabled
    if use_dynamic_window and conversion_lag_model and CONVERSION_LAG_MODEL_AVAILABLE:
        attributions, window_days = engine.calculate_attribution_with_dynamic_window(
            journey=journey,
            model_name=model_name,
            use_dynamic_window=True
        )
        
        # Get conversion timing insights
        timing_insights = engine.get_conversion_timing_insights(journey)
        
        metadata = {
            'attribution_window_days': window_days,
            'dynamic_window_used': True,
            'timing_insights': timing_insights
        }
    else:
        # Standard attribution
        attributions = engine.calculate_attribution(journey, model_name)
        metadata = {
            'attribution_window_days': 30,  # Default
            'dynamic_window_used': False,
            'timing_insights': None
        }
    
    # Get attributed values for each touchpoint
    credited_touchpoints = engine.distribute_credit(journey, model_name)
    
    # Extract attributed rewards in order
    attributed_rewards = [credit for _, credit in credited_touchpoints]
    
    return attributed_rewards, metadata


def calculate_multi_touch_rewards_with_timing(episode_data: Dict[str, Any],
                                           conversion_lag_model: 'ConversionLagModel',
                                           model_name: str = 'time_decay') -> Dict[str, Any]:
    """
    Enhanced multi-touch attribution with conversion timing predictions and censored data handling.
    
    Args:
        episode_data: Dictionary containing episode information
        conversion_lag_model: Trained ConversionLagModel for timing predictions
        model_name: Attribution model to use
        
    Returns:
        Dictionary with attributed rewards and comprehensive timing analysis
    """
    if not CONVERSION_LAG_MODEL_AVAILABLE:
        raise ImportError("ConversionLagModel not available")
    
    engine = AttributionEngine(conversion_lag_model=conversion_lag_model)
    journey = create_journey_from_episode(episode_data)
    
    # Convert to ConversionJourney for lag model
    conversion_journey = engine._convert_to_conversion_journey(journey)
    
    # Handle censored data if journey is ongoing
    processed_journeys = conversion_lag_model.handle_censored_data([conversion_journey])
    processed_journey = processed_journeys[0] if processed_journeys else conversion_journey
    
    # Get conversion predictions
    predictions = conversion_lag_model.predict_conversion_time([processed_journey])
    hazard_rates = conversion_lag_model.calculate_hazard_rate([processed_journey])
    
    # Calculate attribution with dynamic window
    attributions, window_days = engine.calculate_attribution_with_dynamic_window(
        journey=journey,
        model_name=model_name,
        use_dynamic_window=True
    )
    
    # Get timing insights
    timing_insights = engine.get_conversion_timing_insights(journey)
    
    # Calculate attributed rewards
    attributed_rewards, metadata = calculate_multi_touch_rewards(
        episode_data=episode_data,
        model_name=model_name,
        conversion_lag_model=conversion_lag_model,
        use_dynamic_window=True
    )
    
    return {
        'attributed_rewards': attributed_rewards,
        'attribution_weights': attributions,
        'dynamic_attribution_window_days': window_days,
        'conversion_predictions': predictions.get(processed_journey.user_id, []),
        'hazard_rates': hazard_rates.get(processed_journey.user_id, []),
        'timing_insights': timing_insights,
        'journey_is_censored': processed_journey.is_censored,
        'timeout_reason': processed_journey.timeout_reason,
        'metadata': metadata
    }