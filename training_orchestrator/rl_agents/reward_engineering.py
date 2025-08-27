"""
Reward Engineering for Ad Campaign Optimization

Implements sophisticated reward functions that balance multiple objectives
including ROAS, brand safety, exploration, and business constraints.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RewardComponent(Enum):
    """Types of reward components"""
    ROAS = "roas"
    CTR = "ctr"
    CONVERSION_RATE = "conversion_rate"
    BRAND_SAFETY = "brand_safety"
    BUDGET_EFFICIENCY = "budget_efficiency"
    EXPLORATION_BONUS = "exploration_bonus"
    DIVERSITY_BONUS = "diversity_bonus"
    CONSTRAINT_PENALTY = "constraint_penalty"
    RISK_PENALTY = "risk_penalty"


@dataclass
class RewardConfig:
    """Configuration for reward engineering"""
    
    # Primary reward weights
    roas_weight: float = 1.0
    ctr_weight: float = 0.3
    conversion_weight: float = 0.5
    
    # Safety and compliance weights
    brand_safety_weight: float = 0.8
    policy_compliance_weight: float = 1.0
    
    # Efficiency and optimization weights
    budget_efficiency_weight: float = 0.4
    frequency_penalty_weight: float = 0.2
    
    # Exploration and diversity weights
    exploration_bonus_weight: float = 0.1
    diversity_bonus_weight: float = 0.15
    
    # Constraint penalty weights
    budget_violation_penalty: float = 2.0
    safety_violation_penalty: float = 5.0
    
    # Reward shaping parameters
    reward_normalization: bool = True
    reward_clipping: bool = True
    reward_clip_min: float = -10.0
    reward_clip_max: float = 10.0
    
    # ROAS targets and thresholds
    target_roas: float = 3.0
    min_acceptable_roas: float = 1.2
    roas_bonus_threshold: float = 4.0
    
    # Risk management
    max_risk_tolerance: float = 0.3
    risk_adjusted_rewards: bool = True
    
    # Temporal considerations
    decay_factor: float = 0.99
    long_term_bonus: float = 0.2


class RewardEngineer:
    """
    Advanced reward engineering system for ad campaign optimization.
    
    Combines multiple reward signals to create a comprehensive reward function
    that encourages profitable, safe, and diverse ad campaign strategies.
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        
        # Track reward history for normalization
        self.reward_history = []
        self.component_history = {component.value: [] for component in RewardComponent}
        
        # Exploration tracking
        self.action_counts = {}
        self.state_action_visits = {}
        
        # Diversity tracking
        self.campaign_types_used = set()
        self.audience_segments_targeted = set()
        
        self.logger = logging.getLogger(__name__)
    
    def compute_reward(self, 
                      state: Dict[str, Any],
                      action: Dict[str, Any], 
                      next_state: Dict[str, Any],
                      campaign_results: Dict[str, Any],
                      episode_step: int) -> Tuple[float, Dict[str, float]]:
        """
        Compute comprehensive reward signal.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            campaign_results: Results from campaign execution
            episode_step: Current step in episode
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        
        components = {}
        
        # Primary performance rewards
        components[RewardComponent.ROAS.value] = self._compute_roas_reward(campaign_results)
        components[RewardComponent.CTR.value] = self._compute_ctr_reward(campaign_results)
        components[RewardComponent.CONVERSION_RATE.value] = self._compute_conversion_reward(campaign_results)
        
        # Safety and compliance rewards
        components[RewardComponent.BRAND_SAFETY.value] = self._compute_brand_safety_reward(
            campaign_results, action
        )
        
        # Efficiency rewards
        components[RewardComponent.BUDGET_EFFICIENCY.value] = self._compute_budget_efficiency_reward(
            action, campaign_results
        )
        
        # Exploration and diversity bonuses
        components[RewardComponent.EXPLORATION_BONUS.value] = self._compute_exploration_bonus(
            state, action
        )
        components[RewardComponent.DIVERSITY_BONUS.value] = self._compute_diversity_bonus(action)
        
        # Constraint penalties
        components[RewardComponent.CONSTRAINT_PENALTY.value] = self._compute_constraint_penalties(
            action, campaign_results
        )
        components[RewardComponent.RISK_PENALTY.value] = self._compute_risk_penalty(
            action, campaign_results
        )
        
        # Combine components with weights
        total_reward = self._combine_reward_components(components)
        
        # Apply reward shaping
        total_reward = self._apply_reward_shaping(total_reward, episode_step)
        
        # Update tracking
        self._update_tracking(action, components, total_reward)
        
        return total_reward, components
    
    def _compute_roas_reward(self, campaign_results: Dict[str, Any]) -> float:
        """Compute ROAS-based reward"""
        
        revenue = campaign_results.get("revenue", 0.0)
        cost = campaign_results.get("cost", 1.0)
        
        # Avoid division by zero
        roas = revenue / max(cost, 0.01)
        
        # Progressive reward based on ROAS thresholds
        if roas < self.config.min_acceptable_roas:
            # Penalty for poor ROAS
            reward = -1.0 + (roas / self.config.min_acceptable_roas)
        elif roas < self.config.target_roas:
            # Linear reward towards target
            reward = (roas - self.config.min_acceptable_roas) / (
                self.config.target_roas - self.config.min_acceptable_roas
            )
        elif roas < self.config.roas_bonus_threshold:
            # Bonus for exceeding target
            reward = 1.0 + 0.5 * (roas - self.config.target_roas) / (
                self.config.roas_bonus_threshold - self.config.target_roas
            )
        else:
            # High bonus for exceptional performance
            reward = 1.5 + 0.1 * np.log(1 + roas - self.config.roas_bonus_threshold)
        
        return reward * self.config.roas_weight
    
    def _compute_ctr_reward(self, campaign_results: Dict[str, Any]) -> float:
        """Compute CTR-based reward"""
        
        clicks = campaign_results.get("clicks", 0)
        impressions = campaign_results.get("impressions", 1)
        ctr = clicks / max(impressions, 1)
        
        # Normalize CTR (typical range 0.5% - 5%)
        baseline_ctr = 0.02  # 2% baseline
        normalized_ctr = ctr / baseline_ctr
        
        # Logarithmic reward to prevent excessive optimization for CTR
        reward = np.log(1 + normalized_ctr) - np.log(2)  # Zero reward at baseline
        
        return reward * self.config.ctr_weight
    
    def _compute_conversion_reward(self, campaign_results: Dict[str, Any]) -> float:
        """Compute conversion rate reward"""
        
        conversions = campaign_results.get("conversions", 0)
        clicks = campaign_results.get("clicks", 1)
        conversion_rate = conversions / max(clicks, 1)
        
        # Normalize conversion rate (typical range 1% - 10%)
        baseline_conversion = 0.03  # 3% baseline
        normalized_conversion = conversion_rate / baseline_conversion
        
        # Square root reward to encourage conversions while preventing over-optimization
        reward = np.sqrt(normalized_conversion) - 1.0
        
        return reward * self.config.conversion_weight
    
    def _compute_brand_safety_reward(self, campaign_results: Dict[str, Any], 
                                   action: Dict[str, Any]) -> float:
        """Compute brand safety reward"""
        
        # Brand safety score from content analysis
        brand_safety_score = campaign_results.get("brand_safety_score", 0.8)
        
        # Penalty for low brand safety
        if brand_safety_score < 0.6:
            reward = -2.0 * (0.6 - brand_safety_score)
        elif brand_safety_score < 0.8:
            reward = -0.5 * (0.8 - brand_safety_score)
        else:
            reward = 0.5 * (brand_safety_score - 0.8)
        
        # Additional penalty for risky creative types or placements
        creative_type = action.get("creative_type", "image")
        if creative_type == "video" and brand_safety_score < 0.7:
            reward -= 0.5  # Extra caution for video content
        
        return reward * self.config.brand_safety_weight
    
    def _compute_budget_efficiency_reward(self, action: Dict[str, Any],
                                        campaign_results: Dict[str, Any]) -> float:
        """Compute budget efficiency reward"""
        
        budget = action.get("budget", 100.0)
        actual_spend = campaign_results.get("cost", budget)
        revenue = campaign_results.get("revenue", 0.0)
        
        # Efficiency as revenue per dollar spent
        efficiency = revenue / max(actual_spend, 0.01)
        
        # Bonus for under-budget performance
        budget_utilization = actual_spend / max(budget, 0.01)
        if budget_utilization < 0.9 and efficiency > 2.0:
            efficiency_bonus = 0.2 * (0.9 - budget_utilization)
        else:
            efficiency_bonus = 0.0
        
        # Penalty for over-spending
        overspend_penalty = max(0, actual_spend - budget) / budget
        
        reward = np.log(1 + efficiency) - overspend_penalty + efficiency_bonus
        
        return reward * self.config.budget_efficiency_weight
    
    def _compute_exploration_bonus(self, state: Dict[str, Any], 
                                 action: Dict[str, Any]) -> float:
        """Compute exploration bonus based on action novelty"""
        
        # Create action signature
        action_signature = (
            action.get("creative_type", "image"),
            action.get("target_audience", "general"),
            action.get("bid_strategy", "cpc"),
            round(action.get("budget", 50.0) / 10) * 10  # Discretize budget
        )
        
        # Count action occurrences
        self.action_counts[action_signature] = self.action_counts.get(action_signature, 0) + 1
        
        # Exploration bonus inversely proportional to frequency
        visit_count = self.action_counts[action_signature]
        exploration_bonus = 1.0 / np.sqrt(visit_count)
        
        # State-action exploration bonus
        state_signature = self._create_state_signature(state)
        state_action_key = (state_signature, action_signature)
        
        self.state_action_visits[state_action_key] = (
            self.state_action_visits.get(state_action_key, 0) + 1
        )
        
        sa_visit_count = self.state_action_visits[state_action_key]
        sa_exploration_bonus = 0.5 / np.sqrt(sa_visit_count)
        
        total_bonus = exploration_bonus + sa_exploration_bonus
        
        return total_bonus * self.config.exploration_bonus_weight
    
    def _compute_diversity_bonus(self, action: Dict[str, Any]) -> float:
        """Compute diversity bonus for trying different campaign types"""
        
        creative_type = action.get("creative_type", "image")
        audience = action.get("target_audience", "general")
        
        self.campaign_types_used.add(creative_type)
        self.audience_segments_targeted.add(audience)
        
        # Bonus for diversity in campaign portfolio
        creative_diversity = len(self.campaign_types_used) / 3.0  # 3 creative types available
        audience_diversity = len(self.audience_segments_targeted) / 3.0  # 3 audiences available
        
        diversity_score = (creative_diversity + audience_diversity) / 2.0
        
        # Diminishing returns on diversity
        reward = np.sqrt(diversity_score) - 0.5
        
        return reward * self.config.diversity_bonus_weight
    
    def _compute_constraint_penalties(self, action: Dict[str, Any],
                                    campaign_results: Dict[str, Any]) -> float:
        """Compute penalties for constraint violations"""
        
        penalty = 0.0
        
        # Budget constraint violation
        budget = action.get("budget", 100.0)
        actual_spend = campaign_results.get("cost", budget)
        
        if actual_spend > budget * 1.1:  # 10% overspend tolerance
            budget_violation = (actual_spend - budget * 1.1) / budget
            penalty += budget_violation * self.config.budget_violation_penalty
        
        # Safety constraint violations
        brand_safety_score = campaign_results.get("brand_safety_score", 0.8)
        if brand_safety_score < 0.5:
            safety_violation = 0.5 - brand_safety_score
            penalty += safety_violation * self.config.safety_violation_penalty
        
        # Frequency cap violations
        frequency = campaign_results.get("frequency", 2.0)
        if frequency > 5.0:  # Max frequency of 5
            frequency_violation = (frequency - 5.0) / 5.0
            penalty += frequency_violation * self.config.frequency_penalty_weight
        
        return -penalty  # Return negative penalty
    
    def _compute_risk_penalty(self, action: Dict[str, Any],
                            campaign_results: Dict[str, Any]) -> float:
        """Compute risk-adjusted penalty"""
        
        if not self.config.risk_adjusted_rewards:
            return 0.0
        
        # Compute campaign risk factors
        budget = action.get("budget", 100.0)
        creative_type = action.get("creative_type", "image")
        bid_strategy = action.get("bid_strategy", "cpc")
        
        # Risk factors
        budget_risk = min(budget / 200.0, 1.0)  # Higher budget = higher risk
        creative_risk = {"image": 0.1, "video": 0.3, "carousel": 0.2}[creative_type]
        bid_risk = {"cpc": 0.1, "cpm": 0.2, "cpa": 0.3}[bid_strategy]
        
        total_risk = (budget_risk + creative_risk + bid_risk) / 3.0
        
        # Penalty if risk exceeds tolerance
        if total_risk > self.config.max_risk_tolerance:
            risk_penalty = (total_risk - self.config.max_risk_tolerance) ** 2
            return -risk_penalty
        
        return 0.0
    
    def _combine_reward_components(self, components: Dict[str, float]) -> float:
        """Combine weighted reward components"""
        
        total_reward = sum(components.values())
        
        return total_reward
    
    def _apply_reward_shaping(self, reward: float, episode_step: int) -> float:
        """Apply reward shaping techniques"""
        
        # Reward normalization
        if self.config.reward_normalization and len(self.reward_history) > 10:
            reward_mean = np.mean(self.reward_history[-100:])
            reward_std = np.std(self.reward_history[-100:])
            reward = (reward - reward_mean) / max(reward_std, 0.1)
        
        # Reward clipping
        if self.config.reward_clipping:
            reward = np.clip(reward, self.config.reward_clip_min, self.config.reward_clip_max)
        
        # Long-term bonus for consistent performance
        if episode_step > 10 and len(self.reward_history) >= 10:
            recent_rewards = self.reward_history[-10:]
            if all(r > 0 for r in recent_rewards):
                reward += self.config.long_term_bonus
        
        return reward
    
    def _create_state_signature(self, state: Dict[str, Any]) -> Tuple:
        """Create hashable state signature for exploration tracking"""
        
        market_context = state.get("market_context", {})
        persona = state.get("persona", {})
        
        signature = (
            round(market_context.get("competition_level", 0.5) * 10) / 10,
            market_context.get("seasonality_factor", 1.0),
            persona.get("demographics", {}).get("age_group", "unknown"),
            persona.get("demographics", {}).get("income", "unknown"),
            tuple(sorted(persona.get("interests", [])))
        )
        
        return signature
    
    def _update_tracking(self, action: Dict[str, Any], 
                        components: Dict[str, float], total_reward: float):
        """Update reward and action tracking"""
        
        self.reward_history.append(total_reward)
        
        # Keep only recent history to prevent memory growth
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-500:]
        
        # Update component history
        for component, value in components.items():
            self.component_history[component].append(value)
            if len(self.component_history[component]) > 1000:
                self.component_history[component] = self.component_history[component][-500:]
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get reward statistics for monitoring"""
        
        stats = {
            "total_reward_mean": np.mean(self.reward_history) if self.reward_history else 0.0,
            "total_reward_std": np.std(self.reward_history) if self.reward_history else 0.0,
            "total_episodes": len(self.reward_history),
            "component_means": {},
            "component_stds": {},
            "exploration_stats": {
                "unique_actions": len(self.action_counts),
                "unique_state_actions": len(self.state_action_visits),
                "creative_types_used": len(self.campaign_types_used),
                "audience_segments_used": len(self.audience_segments_targeted)
            }
        }
        
        # Component statistics
        for component, history in self.component_history.items():
            if history:
                stats["component_means"][component] = np.mean(history)
                stats["component_stds"][component] = np.std(history)
        
        return stats
    
    def reset_episode_tracking(self):
        """Reset per-episode tracking variables"""
        # Keep global tracking but reset episode-specific variables
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get reward engineer state for checkpointing"""
        return {
            "config": self.config.__dict__,
            "reward_history": self.reward_history[-100:],  # Keep recent history
            "component_history": {k: v[-100:] for k, v in self.component_history.items()},
            "action_counts": dict(self.action_counts),
            "state_action_visits": dict(self.state_action_visits),
            "campaign_types_used": list(self.campaign_types_used),
            "audience_segments_targeted": list(self.audience_segments_targeted)
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load reward engineer state from checkpoint"""
        self.reward_history = state["reward_history"]
        self.component_history = state["component_history"]
        self.action_counts = state["action_counts"]
        self.state_action_visits = state["state_action_visits"]
        self.campaign_types_used = set(state["campaign_types_used"])
        self.audience_segments_targeted = set(state["audience_segments_targeted"])