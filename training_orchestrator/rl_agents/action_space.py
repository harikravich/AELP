"""
Action space management for GAELP RL agents.
Handles discretization, continuous actions, and multi-dimensional action spaces.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions available in ad campaigns."""
    BID_ADJUSTMENT = "bid_adjustment"
    BUDGET_ALLOCATION = "budget_allocation"
    CREATIVE_SELECTION = "creative_selection"
    AUDIENCE_TARGETING = "audience_targeting"
    PLATFORM_SELECTION = "platform_selection"
    TIME_SCHEDULING = "time_scheduling"


class ActionSpaceManager:
    """
    Manages the action space for RL agents in ad campaign optimization.
    Supports both discrete and continuous action spaces.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Define action dimensions
        self.continuous_actions = {
            ActionType.BID_ADJUSTMENT: (-1.0, 1.0),  # -100% to +100%
            ActionType.BUDGET_ALLOCATION: (0.0, 1.0),  # 0% to 100%
        }
        
        self.discrete_actions = {
            ActionType.CREATIVE_SELECTION: ['image', 'video', 'carousel', 'text', 'collection'],
            ActionType.AUDIENCE_TARGETING: [
                'young_adults_18_24',
                'adults_25_34', 
                'adults_35_44',
                'adults_45_54',
                'adults_55_64',
                'seniors_65_plus',
                'professionals',
                'families',
                'students'
            ],
            ActionType.PLATFORM_SELECTION: [
                'google_ads',
                'facebook_ads',
                'instagram_ads',
                'tiktok_ads',
                'linkedin_ads',
                'twitter_ads',
                'snapchat_ads'
            ],
            ActionType.TIME_SCHEDULING: [
                'always_on',
                'business_hours',
                'evening_hours',
                'weekend_only',
                'custom_schedule'
            ]
        }
        
        # Calculate total action dimensions
        self.continuous_dim = len(self.continuous_actions)
        self.discrete_dims = {k: len(v) for k, v in self.discrete_actions.items()}
        self.total_discrete_dim = sum(self.discrete_dims.values())
        
        # For hybrid action space
        self.action_dim = self.continuous_dim + len(self.discrete_actions)
        
        logger.info(f"Initialized ActionSpaceManager with {self.action_dim} dimensions")
    
    def sample_action(self) -> np.ndarray:
        """Sample a random action from the action space."""
        # Sample continuous components
        continuous = []
        for action_type, (min_val, max_val) in self.continuous_actions.items():
            continuous.append(np.random.uniform(min_val, max_val))
        
        # Sample discrete components (as continuous values to be discretized)
        discrete = []
        for action_type in self.discrete_actions:
            discrete.append(np.random.random())  # 0 to 1, will be discretized
        
        return np.array(continuous + discrete)
    
    def parse_action(self, action: np.ndarray) -> Dict[ActionType, Any]:
        """
        Parse a raw action vector into meaningful campaign parameters.
        
        Args:
            action: Raw action vector from RL agent
            
        Returns:
            Dictionary mapping action types to their values
        """
        parsed = {}
        idx = 0
        
        # Parse continuous actions
        for action_type in self.continuous_actions:
            min_val, max_val = self.continuous_actions[action_type]
            value = np.clip(action[idx], min_val, max_val)
            parsed[action_type] = value
            idx += 1
        
        # Parse discrete actions
        for action_type, options in self.discrete_actions.items():
            # Convert continuous value to discrete index
            continuous_val = np.clip(action[idx], 0, 0.999999)
            discrete_idx = int(continuous_val * len(options))
            parsed[action_type] = options[discrete_idx]
            idx += 1
        
        return parsed
    
    def action_to_campaign_params(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Convert an action vector to campaign parameters.
        
        Returns:
            Dictionary of campaign parameters ready for API calls
        """
        parsed = self.parse_action(action)
        
        # Convert to campaign API parameters
        params = {
            'bid_strategy': {
                'type': 'manual_cpc',
                'bid_adjustment': float(parsed[ActionType.BID_ADJUSTMENT]),
            },
            'budget': {
                'daily_amount': float(parsed[ActionType.BUDGET_ALLOCATION] * 100),  # Scale to dollars
                'type': 'daily'
            },
            'creative': {
                'format': parsed[ActionType.CREATIVE_SELECTION],
            },
            'targeting': {
                'audience_segment': parsed[ActionType.AUDIENCE_TARGETING],
                'platforms': [parsed[ActionType.PLATFORM_SELECTION]],
                'schedule': parsed[ActionType.TIME_SCHEDULING],
            }
        }
        
        # Add platform-specific parameters
        platform = parsed[ActionType.PLATFORM_SELECTION]
        if platform == 'google_ads':
            params['google_specific'] = {
                'campaign_type': 'search',
                'network': 'search_and_partners'
            }
        elif platform in ['facebook_ads', 'instagram_ads']:
            params['meta_specific'] = {
                'objective': 'conversions',
                'optimization_goal': 'purchase'
            }
        
        return params
    
    def get_action_mask(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action mask based on current state constraints.
        
        Args:
            state: Current campaign state
            
        Returns:
            Binary mask indicating valid actions (1 = valid, 0 = invalid)
        """
        mask = np.ones(self.action_dim)
        
        # Mask budget allocation if budget exhausted
        if state.get('budget_remaining', float('inf')) <= 0:
            mask[1] = 0  # Budget allocation index
        
        # Mask platforms based on account status
        if not state.get('google_ads_enabled', True):
            # Find and mask Google Ads platform selection
            # This would need more sophisticated indexing in production
            pass
        
        return mask
    
    def discretize_continuous(self, value: float, num_bins: int = 10) -> int:
        """
        Discretize a continuous value into bins.
        
        Args:
            value: Continuous value (0 to 1)
            num_bins: Number of discrete bins
            
        Returns:
            Discrete bin index
        """
        value = np.clip(value, 0, 0.999999)
        return int(value * num_bins)
    
    def continuous_from_discrete(self, bin_idx: int, num_bins: int = 10) -> float:
        """
        Convert discrete bin index back to continuous value.
        
        Args:
            bin_idx: Discrete bin index
            num_bins: Number of discrete bins
            
        Returns:
            Continuous value (midpoint of bin)
        """
        return (bin_idx + 0.5) / num_bins
    
    def get_action_names(self) -> List[str]:
        """Get human-readable names for all action dimensions."""
        names = []
        
        # Continuous action names
        for action_type in self.continuous_actions:
            names.append(action_type.value)
        
        # Discrete action names
        for action_type in self.discrete_actions:
            names.append(action_type.value)
        
        return names
    
    def validate_action(self, action: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate an action vector.
        
        Args:
            action: Action vector to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if len(action) != self.action_dim:
            errors.append(f"Action dimension mismatch: expected {self.action_dim}, got {len(action)}")
            return False, errors
        
        # Check continuous action bounds
        idx = 0
        for action_type, (min_val, max_val) in self.continuous_actions.items():
            if not min_val <= action[idx] <= max_val:
                errors.append(f"{action_type.value} out of bounds: {action[idx]} not in [{min_val}, {max_val}]")
            idx += 1
        
        # Check discrete action bounds (should be 0 to 1 for discretization)
        for action_type in self.discrete_actions:
            if not 0 <= action[idx] <= 1:
                errors.append(f"{action_type.value} out of bounds: {action[idx]} not in [0, 1]")
            idx += 1
        
        return len(errors) == 0, errors


class HierarchicalActionSpace:
    """
    Hierarchical action space for multi-level decision making.
    High-level: Campaign strategy decisions
    Low-level: Tactical optimizations
    """
    
    def __init__(self):
        # High-level strategic actions
        self.high_level_actions = [
            'aggressive_growth',     # High spend, broad targeting
            'conservative_testing',  # Low spend, narrow targeting
            'balanced_optimization', # Medium spend, data-driven
            'exploratory_learning',  # Varied experiments
            'exploitation_focus'     # Stick to winning formula
        ]
        
        # Low-level tactical managers
        self.low_level_managers = {
            'aggressive_growth': ActionSpaceManager({'default_budget_multiplier': 1.5}),
            'conservative_testing': ActionSpaceManager({'default_budget_multiplier': 0.5}),
            'balanced_optimization': ActionSpaceManager({'default_budget_multiplier': 1.0}),
            'exploratory_learning': ActionSpaceManager({'exploration_bonus': 0.2}),
            'exploitation_focus': ActionSpaceManager({'exploration_penalty': -0.2})
        }
        
        self.current_strategy = None
    
    def select_strategy(self, state: Dict[str, Any]) -> str:
        """
        Select high-level strategy based on campaign state.
        
        Args:
            state: Current campaign performance metrics
            
        Returns:
            Selected strategy name
        """
        # Simple rule-based strategy selection (could be learned)
        roas = state.get('roas', 0)
        confidence = state.get('data_confidence', 0)  # Based on data volume
        
        if confidence < 0.3:
            # Not enough data, need exploration
            strategy = 'exploratory_learning'
        elif roas < 1.0:
            # Losing money, be conservative
            strategy = 'conservative_testing'
        elif roas > 3.0 and confidence > 0.7:
            # Winning formula, exploit it
            strategy = 'exploitation_focus'
        elif roas > 2.0:
            # Good performance, push for growth
            strategy = 'aggressive_growth'
        else:
            # Moderate performance, optimize
            strategy = 'balanced_optimization'
        
        self.current_strategy = strategy
        return strategy
    
    def get_tactical_action(self, strategy: str, state: np.ndarray) -> np.ndarray:
        """
        Get low-level tactical action based on strategy.
        
        Args:
            strategy: High-level strategy
            state: Current state vector
            
        Returns:
            Tactical action vector
        """
        manager = self.low_level_managers[strategy]
        
        # Strategy-specific action sampling
        if strategy == 'exploratory_learning':
            # More random exploration
            return manager.sample_action()
        elif strategy == 'exploitation_focus':
            # Stick to defaults with small variations
            action = manager.sample_action()
            action = action * 0.2 + 0.4  # Center around moderate values
            return action
        else:
            # Standard action sampling
            return manager.sample_action()