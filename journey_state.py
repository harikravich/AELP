"""
GAELP Journey State Management System
Handles user journey state transitions and progression logic.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class JourneyState(Enum):
    """User journey states following the customer journey funnel."""
    UNAWARE = "UNAWARE"
    AWARE = "AWARE"
    CONSIDERING = "CONSIDERING"
    INTENT = "INTENT"
    CONVERTED = "CONVERTED"

class TransitionTrigger(Enum):
    """Events that can trigger state transitions."""
    IMPRESSION = "IMPRESSION"
    CLICK = "CLICK"
    ENGAGEMENT = "ENGAGEMENT"
    CONTENT_VIEW = "CONTENT_VIEW"
    PRODUCT_VIEW = "PRODUCT_VIEW"
    ADD_TO_CART = "ADD_TO_CART"
    CHECKOUT_START = "CHECKOUT_START"
    PURCHASE = "PURCHASE"
    COMPETITOR_EXPOSURE = "COMPETITOR_EXPOSURE"
    TIMEOUT = "TIMEOUT"
    MANUAL = "MANUAL"

@dataclass
class StateTransition:
    """Represents a state transition event."""
    from_state: JourneyState
    to_state: JourneyState
    timestamp: datetime
    trigger: TransitionTrigger
    confidence: float
    context: Dict[str, Any]
    touchpoint_id: Optional[str] = None
    channel: Optional[str] = None

@dataclass
class JourneyStateConfig:
    """Configuration for journey state management."""
    # Transition probabilities based on triggers
    transition_probabilities: Dict[Tuple[JourneyState, TransitionTrigger], Dict[JourneyState, float]]
    
    # Minimum confidence thresholds for transitions
    confidence_thresholds: Dict[Tuple[JourneyState, JourneyState], float]
    
    # Engagement scoring weights
    engagement_weights: Dict[str, float]
    
    # Intent signals and their weights
    intent_signals: Dict[str, float]
    
    # Decay rates for various metrics
    time_decay_rate: float = 0.1
    engagement_decay_rate: float = 0.05

class JourneyStateManager:
    """Manages user journey state transitions and scoring."""
    
    def __init__(self, config: Optional[JourneyStateConfig] = None):
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> JourneyStateConfig:
        """Returns default configuration for journey state management."""
        
        # Default transition probabilities
        transition_probs = {
            # From UNAWARE
            (JourneyState.UNAWARE, TransitionTrigger.IMPRESSION): {
                JourneyState.AWARE: 0.3,
                JourneyState.UNAWARE: 0.7
            },
            (JourneyState.UNAWARE, TransitionTrigger.CLICK): {
                JourneyState.AWARE: 0.7,
                JourneyState.CONSIDERING: 0.2,
                JourneyState.UNAWARE: 0.1
            },
            
            # From AWARE
            (JourneyState.AWARE, TransitionTrigger.IMPRESSION): {
                JourneyState.AWARE: 0.8,
                JourneyState.CONSIDERING: 0.2
            },
            (JourneyState.AWARE, TransitionTrigger.CLICK): {
                JourneyState.CONSIDERING: 0.6,
                JourneyState.AWARE: 0.4
            },
            (JourneyState.AWARE, TransitionTrigger.ENGAGEMENT): {
                JourneyState.CONSIDERING: 0.8,
                JourneyState.AWARE: 0.2
            },
            
            # From CONSIDERING
            (JourneyState.CONSIDERING, TransitionTrigger.PRODUCT_VIEW): {
                JourneyState.INTENT: 0.4,
                JourneyState.CONSIDERING: 0.6
            },
            (JourneyState.CONSIDERING, TransitionTrigger.ADD_TO_CART): {
                JourneyState.INTENT: 0.9,
                JourneyState.CONSIDERING: 0.1
            },
            (JourneyState.CONSIDERING, TransitionTrigger.COMPETITOR_EXPOSURE): {
                JourneyState.AWARE: 0.3,
                JourneyState.CONSIDERING: 0.7
            },
            
            # From INTENT
            (JourneyState.INTENT, TransitionTrigger.CHECKOUT_START): {
                JourneyState.CONVERTED: 0.6,
                JourneyState.INTENT: 0.4
            },
            (JourneyState.INTENT, TransitionTrigger.PURCHASE): {
                JourneyState.CONVERTED: 1.0
            },
            (JourneyState.INTENT, TransitionTrigger.COMPETITOR_EXPOSURE): {
                JourneyState.CONSIDERING: 0.4,
                JourneyState.INTENT: 0.6
            }
        }
        
        # Confidence thresholds for transitions
        confidence_thresholds = {
            (JourneyState.UNAWARE, JourneyState.AWARE): 0.6,
            (JourneyState.AWARE, JourneyState.CONSIDERING): 0.7,
            (JourneyState.CONSIDERING, JourneyState.INTENT): 0.8,
            (JourneyState.INTENT, JourneyState.CONVERTED): 0.9,
            # Backward transitions have lower thresholds
            (JourneyState.CONSIDERING, JourneyState.AWARE): 0.5,
            (JourneyState.INTENT, JourneyState.CONSIDERING): 0.6
        }
        
        # Engagement scoring weights
        engagement_weights = {
            'dwell_time': 0.3,
            'scroll_depth': 0.2,
            'click_depth': 0.2,
            'interaction_count': 0.15,
            'content_completion': 0.15
        }
        
        # Intent signals and weights
        intent_signals = {
            'product_view': 0.3,
            'price_check': 0.4,
            'comparison_view': 0.35,
            'add_to_cart': 0.8,
            'checkout_start': 0.9,
            'support_contact': 0.6,
            'review_read': 0.25,
            'specification_view': 0.5
        }
        
        return JourneyStateConfig(
            transition_probabilities=transition_probs,
            confidence_thresholds=confidence_thresholds,
            engagement_weights=engagement_weights,
            intent_signals=intent_signals
        )
    
    def calculate_transition_probability(
        self, 
        current_state: JourneyState, 
        trigger: TransitionTrigger,
        context: Dict[str, Any]
    ) -> Dict[JourneyState, float]:
        """Calculate probabilities for all possible state transitions."""
        
        # Get transition probabilities safely
        if hasattr(self, 'config') and self.config and hasattr(self.config, 'transition_probabilities'):
            base_probs = self.config.transition_probabilities.get(
                (current_state, trigger), {}
            )
        else:
            base_probs = {}
        
        if not base_probs:
            # No transition defined, stay in current state
            return {current_state: 1.0}
        
        # Adjust probabilities based on context
        adjusted_probs = base_probs.copy()
        
        # Consider engagement score
        engagement_score = context.get('engagement_score', 0.0)
        if engagement_score > 0.7:
            # High engagement increases progression probability
            for state, prob in adjusted_probs.items():
                if self._is_progression(current_state, state):
                    adjusted_probs[state] = min(1.0, prob * 1.3)
        
        # Consider intent signals
        intent_score = self._calculate_intent_score(context)
        if intent_score > 0.6:
            # High intent increases progression to intent/conversion states
            for state in [JourneyState.INTENT, JourneyState.CONVERTED]:
                if state in adjusted_probs:
                    adjusted_probs[state] = min(1.0, adjusted_probs[state] * 1.5)
        
        # Consider competitor exposure
        if context.get('competitor_exposure', False):
            # Competitor exposure decreases progression probability
            for state, prob in adjusted_probs.items():
                if self._is_progression(current_state, state):
                    adjusted_probs[state] = prob * 0.7
        
        # Normalize probabilities
        total_prob = sum(adjusted_probs.values())
        if total_prob > 0:
            adjusted_probs = {
                state: prob / total_prob 
                for state, prob in adjusted_probs.items()
            }
        
        return adjusted_probs
    
    def predict_next_state(
        self, 
        current_state: JourneyState, 
        trigger: TransitionTrigger,
        context: Dict[str, Any]
    ) -> Tuple[JourneyState, float]:
        """Predict the most likely next state with confidence."""
        
        probs = self.calculate_transition_probability(current_state, trigger, context)
        
        # Find the state with highest probability
        next_state = max(probs.items(), key=lambda x: x[1])
        
        return next_state[0], next_state[1]
    
    def should_transition(
        self, 
        from_state: JourneyState, 
        to_state: JourneyState, 
        confidence: float
    ) -> bool:
        """Determine if a state transition should occur based on confidence."""
        
        # Get confidence threshold safely
        if hasattr(self, 'config') and self.config and hasattr(self.config, 'confidence_thresholds'):
            threshold = self.config.confidence_thresholds.get(
                (from_state, to_state), 0.5
            )
        else:
            threshold = 0.5
        
        return confidence >= threshold
    
    def calculate_engagement_score(self, touchpoint_data: Dict[str, Any]) -> float:
        """Calculate engagement score based on touchpoint data."""
        
        score = 0.0
        
        # Default weights if config is missing or None
        default_weights = {
            'dwell_time': 0.3,
            'scroll_depth': 0.2,
            'click_depth': 0.2,
            'interaction_count': 0.15,
            'content_completion': 0.15
        }
        
        # Get weights safely
        if not hasattr(self, 'config') or self.config is None:
            weights = default_weights
        elif not hasattr(self.config, 'engagement_weights') or self.config.engagement_weights is None:
            weights = default_weights
        else:
            weights = self.config.engagement_weights
        
        # Dwell time (normalized to 0-1)
        dwell_time = touchpoint_data.get('dwell_time_seconds', 0)
        dwell_score = min(1.0, dwell_time / 300.0) if dwell_time else 0.0  # 5 minutes = 1.0
        score += dwell_score * weights.get('dwell_time', 0.3)
        
        # Scroll depth
        scroll_depth = touchpoint_data.get('scroll_depth', 0.0) or 0.0
        score += scroll_depth * weights.get('scroll_depth', 0.2)
        
        # Click depth
        click_depth = touchpoint_data.get('click_depth', 0) or 0
        click_score = min(1.0, click_depth / 5.0) if click_depth else 0.0  # 5+ clicks = 1.0
        score += click_score * weights.get('click_depth', 0.2)
        
        # Interaction count (pages viewed, etc.)
        interaction_count = touchpoint_data.get('interaction_count', 0) or 0
        interaction_score = min(1.0, interaction_count / 10.0) if interaction_count else 0.0
        score += interaction_score * weights.get('interaction_count', 0.15)
        
        # Content completion (for video/article content)
        completion_rate = touchpoint_data.get('content_completion_rate', 0.0) or 0.0
        score += completion_rate * weights.get('content_completion', 0.15)
        
        return min(1.0, score)
    
    def _calculate_intent_score(self, context: Dict[str, Any]) -> float:
        """Calculate intent score based on user signals."""
        
        score = 0.0
        intent_signals = context.get('intent_signals', [])
        
        # Default intent signals if config is missing
        default_signals = {
            'product_view': 0.3,
            'price_check': 0.4,
            'comparison_view': 0.35,
            'add_to_cart': 0.8,
            'checkout_start': 0.9,
            'support_contact': 0.6,
            'review_read': 0.25,
            'specification_view': 0.5
        }
        
        # Get signals safely
        if not hasattr(self, 'config') or self.config is None:
            signals_config = default_signals
        elif not hasattr(self.config, 'intent_signals') or self.config.intent_signals is None:
            signals_config = default_signals
        else:
            signals_config = self.config.intent_signals
        
        for signal in intent_signals:
            if signal in signals_config:
                score += signals_config[signal]
        
        return min(1.0, score)
    
    def _is_progression(self, from_state: JourneyState, to_state: JourneyState) -> bool:
        """Check if transition represents forward progression in journey."""
        
        state_order = {
            JourneyState.UNAWARE: 0,
            JourneyState.AWARE: 1,
            JourneyState.CONSIDERING: 2,
            JourneyState.INTENT: 3,
            JourneyState.CONVERTED: 4
        }
        
        return state_order[to_state] > state_order[from_state]
    
    def calculate_journey_score(
        self, 
        touchpoints: List[Dict[str, Any]], 
        current_state: JourneyState
    ) -> float:
        """Calculate overall journey score based on touchpoint history."""
        
        if not touchpoints:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for i, touchpoint in enumerate(touchpoints):
            # Calculate base score for touchpoint
            engagement_score = self.calculate_engagement_score(touchpoint)
            
            # Apply time decay (more recent touchpoints weighted higher)
            age_days = (datetime.now() - touchpoint.get('timestamp', datetime.now())).days
            decay_rate = getattr(self.config, 'time_decay_rate', 0.05) if hasattr(self, 'config') and self.config else 0.05
            time_weight = max(0.1, 1.0 - (age_days * decay_rate))
            
            # Apply position weight (later touchpoints weighted higher)
            position_weight = (i + 1) / max(1, len(touchpoints))
            
            # Combine weights
            final_weight = time_weight * position_weight
            
            total_score += engagement_score * final_weight
            total_weight += final_weight
        
        base_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Adjust based on current state
        state_multiplier = {
            JourneyState.UNAWARE: 0.2,
            JourneyState.AWARE: 0.4,
            JourneyState.CONSIDERING: 0.6,
            JourneyState.INTENT: 0.8,
            JourneyState.CONVERTED: 1.0
        }
        
        return base_score * state_multiplier[current_state]
    
    def calculate_conversion_probability(
        self, 
        current_state: JourneyState,
        journey_score: float,
        days_in_journey: int,
        touchpoint_count: int,
        context: Dict[str, Any]
    ) -> float:
        """Calculate probability of conversion based on journey data."""
        
        # Base probability by state
        base_probs = {
            JourneyState.UNAWARE: 0.01,
            JourneyState.AWARE: 0.05,
            JourneyState.CONSIDERING: 0.15,
            JourneyState.INTENT: 0.45,
            JourneyState.CONVERTED: 1.0
        }
        
        base_prob = base_probs[current_state]
        
        # Journey score multiplier
        score_multiplier = 1.0 + (journey_score * 0.5)
        
        # Touchpoint count factor (more touchpoints = higher engagement)
        touchpoint_factor = min(2.0, 1.0 + (touchpoint_count * 0.05))
        
        # Time factor (urgency increases over time, then decreases)
        if days_in_journey <= 7:
            time_factor = 1.0 + (days_in_journey * 0.1)
        else:
            time_factor = 1.7 - ((days_in_journey - 7) * 0.05)
        time_factor = max(0.5, time_factor)
        
        # Intent signals boost
        intent_score = self._calculate_intent_score(context)
        intent_factor = 1.0 + (intent_score * 0.3)
        
        # Competitor exposure penalty
        competitor_penalty = 1.0
        if context.get('recent_competitor_exposure', False):
            competitor_penalty = 0.8
        
        # Calculate final probability
        final_prob = (
            base_prob * 
            score_multiplier * 
            touchpoint_factor * 
            time_factor * 
            intent_factor * 
            competitor_penalty
        )
        
        return min(1.0, final_prob)

def create_state_transition(
    from_state: JourneyState,
    to_state: JourneyState,
    trigger: TransitionTrigger,
    confidence: float,
    touchpoint_id: Optional[str] = None,
    channel: Optional[str] = None,
    **context_kwargs
) -> StateTransition:
    """Helper function to create a state transition."""
    
    return StateTransition(
        from_state=from_state,
        to_state=to_state,
        timestamp=datetime.now(),
        trigger=trigger,
        confidence=confidence,
        touchpoint_id=touchpoint_id,
        channel=channel,
        context=context_kwargs
    )

# Example usage and testing
if __name__ == "__main__":
    # Initialize state manager
    manager = JourneyStateManager()
    
    # Test state transition prediction
    context = {
        'engagement_score': 0.8,
        'intent_signals': ['product_view', 'price_check'],
        'competitor_exposure': False
    }
    
    next_state, confidence = manager.predict_next_state(
        JourneyState.AWARE, 
        TransitionTrigger.CLICK, 
        context
    )
    
    print(f"Predicted transition: AWARE -> {next_state.value} (confidence: {confidence:.3f})")
    
    # Test journey scoring
    touchpoints = [
        {
            'timestamp': datetime.now() - timedelta(days=1),
            'dwell_time_seconds': 120,
            'scroll_depth': 0.8,
            'click_depth': 2,
            'interaction_count': 3
        },
        {
            'timestamp': datetime.now(),
            'dwell_time_seconds': 200,
            'scroll_depth': 0.9,
            'click_depth': 4,
            'interaction_count': 6
        }
    ]
    
    journey_score = manager.calculate_journey_score(touchpoints, JourneyState.CONSIDERING)
    print(f"Journey score: {journey_score:.3f}")
    
    # Test conversion probability
    conversion_prob = manager.calculate_conversion_probability(
        current_state=JourneyState.CONSIDERING,
        journey_score=journey_score,
        days_in_journey=5,
        touchpoint_count=8,
        context=context
    )
    
    print(f"Conversion probability: {conversion_prob:.3f}")