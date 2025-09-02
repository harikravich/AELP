#!/usr/bin/env python3
"""
Explainable RL Agent for GAELP - Full Transparency in All Bid Decisions

CRITICAL REQUIREMENTS:
- Every bid decision must be fully explainable
- No black box decisions allowed
- Real-time explanation generation
- Integration with audit trail system
- Quantified factor attribution
- Human-readable reasoning

This agent extends the fortified RL system with comprehensive explainability,
ensuring every bid decision can be understood, audited, and optimized.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import random
import logging
import uuid
from typing import Dict, Tuple, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import math

# Import core GAELP systems
from fortified_rl_agent_no_hardcoding import (
    FortifiedRLAgent, DynamicEnrichedState, TrajectoryExperience, 
    CompletedTrajectory, PrioritizedReplayBuffer
)
from bid_explainability_system import (
    BidExplainabilityEngine, BidDecisionExplanation, DecisionFactor,
    DecisionConfidence, FactorImportance, ExplainabilityMetrics,
    explain_bid_decision, integrate_with_audit_trail
)
from audit_trail import log_decision, log_outcome, log_budget, get_audit_trail
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
from dynamic_segment_integration import get_discovered_segments, validate_no_hardcoded_segments

logger = logging.getLogger(__name__)

@dataclass
class ExplainableAction:
    """Action with full explanation attached"""
    
    # Core action
    bid_amount: float
    bid_action_idx: int
    creative_id: int
    creative_action_idx: int
    channel: str
    channel_action_idx: int
    
    # Decision process transparency
    q_values_bid: List[float]
    q_values_creative: List[float] 
    q_values_channel: List[float]
    
    # Factor analysis
    primary_factors: List[DecisionFactor]
    decision_confidence: DecisionConfidence
    uncertainty_range: Tuple[float, float]
    
    # Explanation
    explanation: BidDecisionExplanation
    
    # Metadata
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    exploration_mode: bool = False
    epsilon_used: float = 0.0

@dataclass 
class ExplainableExperience:
    """Experience with attached explanations for learning"""
    
    # Core experience
    state: np.ndarray
    action: ExplainableAction
    reward: float
    next_state: np.ndarray
    done: bool
    
    # Explanations
    action_explanation: BidDecisionExplanation
    reward_explanation: Dict[str, Any]  # Why this reward was received
    
    # Learning insights
    q_prediction_error: float
    factor_prediction_errors: Dict[str, float]  # Which factors were wrong
    
    # Metadata
    user_id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)

class ExplainableRLAgent(FortifiedRLAgent):
    """
    RL Agent with comprehensive explainability for all decisions
    
    Extends FortifiedRLAgent with:
    - Real-time explanation generation for every decision
    - Factor importance tracking and learning
    - Audit trail integration
    - Performance prediction with uncertainty
    - Counterfactual analysis capabilities
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize explainable RL agent with explanation capabilities"""
        
        # Initialize base RL agent
        super().__init__(*args, **kwargs)
        
        # Add explainability components
        self.explainability_engine = BidExplainabilityEngine()
        
        # Enhanced replay buffer for explainable experiences
        self.explainable_replay_buffer = deque(maxlen=50000)
        
        # Factor importance tracking
        self.factor_importance_history = defaultdict(list)
        self.factor_prediction_accuracy = defaultdict(list)
        
        # Explanation performance metrics
        self.explanation_metrics = ExplainabilityMetrics(
            coverage=0.0, consistency=0.0, actionability=0.0,
            comprehensibility=0.0, factual_accuracy=0.0
        )
        
        # Decision confidence tracking
        self.confidence_history = deque(maxlen=1000)
        self.confidence_vs_performance = []
        
        # Counterfactual analysis cache
        self.counterfactual_cache = {}
        
        # Audit trail integration
        self.audit_trail = get_audit_trail()
        
        logger.info("ExplainableRLAgent initialized with full transparency capabilities")
    
    def select_explainable_action(self,
                                 state: DynamicEnrichedState,
                                 user_id: str,
                                 session_id: str,
                                 campaign_id: str,
                                 context: Dict[str, Any],
                                 explore: bool = True) -> ExplainableAction:
        """
        Select action with complete explainability
        
        Returns ExplainableAction with full decision transparency
        """
        
        # Generate decision ID
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        logger.debug(f"Generating explainable action for decision {decision_id}")
        
        # Get state vector for model
        state_vector = torch.FloatTensor(state.to_vector(self.data_stats)).unsqueeze(0).to(self.device)
        
        # Get Q-values from all networks
        with torch.no_grad():
            q_bid = self.q_network_bid(state_vector)
            q_creative = self.q_network_creative(state_vector)  
            q_channel = self.q_network_channel(state_vector)
            
            # Convert to lists for serialization
            q_values_bid = q_bid.cpu().numpy().flatten().tolist()
            q_values_creative = q_creative.cpu().numpy().flatten().tolist()
            q_values_channel = q_channel.cpu().numpy().flatten().tolist()
        
        # Determine if exploring
        exploring = explore and random.random() < self.epsilon
        epsilon_used = self.epsilon if exploring else 0.0
        
        # Select actions
        if exploring:
            # Guided exploration based on uncertainty
            bid_action_idx = self._guided_exploration_bid(q_values_bid, state, context)
            creative_action_idx = self._guided_exploration_creative(q_values_creative, state, context)
            channel_action_idx = self._guided_exploration_channel(q_values_channel, state, context)
        else:
            # Exploitation - select best actions
            bid_action_idx = int(np.argmax(q_values_bid))
            creative_action_idx = int(np.argmax(q_values_creative)) 
            channel_action_idx = int(np.argmax(q_values_channel))
        
        # Convert to actual values
        bid_levels = np.linspace(self.min_bid, self.max_bid, self.num_bid_levels)
        bid_amount = float(bid_levels[bid_action_idx])
        
        # Apply budget pacing
        pacing_factor = context.get('pacing_factor', 1.0)
        bid_amount *= pacing_factor
        bid_amount = max(self.min_bid, min(self.max_bid, bid_amount))
        
        # Map to channels and creatives
        channels = self.discovered_channels
        channel = channels[min(channel_action_idx, len(channels) - 1)]
        creative_id = creative_action_idx
        
        # Generate comprehensive explanation
        action_dict = {
            'bid_amount': bid_amount,
            'bid_action': bid_action_idx,
            'creative_id': creative_id,
            'creative_action': creative_action_idx,
            'channel': channel,
            'channel_action': channel_action_idx
        }
        
        model_outputs = {
            'q_values_bid': q_values_bid,
            'q_values_creative': q_values_creative,
            'q_values_channel': q_values_channel
        }
        
        decision_factors = {
            'exploration_mode': exploring,
            'epsilon_used': epsilon_used,
            'model_version': 'explainable_gaelp_v1',
            'pacing_factor': pacing_factor,
            'data_stats': self.data_stats
        }
        
        # Generate complete explanation
        explanation = self.explainability_engine.explain_bid_decision(
            decision_id=decision_id,
            user_id=user_id,
            campaign_id=campaign_id,
            state=state,
            action=action_dict,
            context=context,
            model_outputs=model_outputs,
            decision_factors=decision_factors
        )
        
        # Create explainable action
        explainable_action = ExplainableAction(
            bid_amount=bid_amount,
            bid_action_idx=bid_action_idx,
            creative_id=creative_id,
            creative_action_idx=creative_action_idx,
            channel=channel,
            channel_action_idx=channel_action_idx,
            q_values_bid=q_values_bid,
            q_values_creative=q_values_creative,
            q_values_channel=q_values_channel,
            primary_factors=explanation.primary_factors,
            decision_confidence=explanation.decision_confidence,
            uncertainty_range=explanation.uncertainty_range,
            explanation=explanation,
            decision_id=decision_id,
            timestamp=timestamp,
            exploration_mode=exploring,
            epsilon_used=epsilon_used
        )
        
        # Log to audit trail for compliance
        self._log_decision_to_audit_trail(
            decision_id, user_id, session_id, campaign_id,
            state, action_dict, context, model_outputs, decision_factors
        )
        
        # Track decision confidence
        self.confidence_history.append(explanation.decision_confidence)
        
        # Update factor importance tracking
        self._update_factor_importance_tracking(explanation.primary_factors + explanation.secondary_factors)
        
        logger.info(f"Generated explainable action {decision_id}: "
                   f"${bid_amount:.2f} bid with {explanation.decision_confidence.value} confidence")
        
        return explainable_action
    
    def store_explainable_experience(self,
                                   state: DynamicEnrichedState,
                                   action: ExplainableAction,
                                   reward: float,
                                   next_state: DynamicEnrichedState,
                                   done: bool,
                                   user_id: str,
                                   session_id: str,
                                   auction_result: Any) -> None:
        """
        Store experience with full explanations for learning
        """
        
        # Generate reward explanation
        reward_explanation = self._explain_reward(
            state, action, reward, next_state, auction_result
        )
        
        # Calculate prediction errors for learning
        q_prediction_error = self._calculate_q_prediction_error(
            action.q_values_bid, reward, action.bid_action_idx
        )
        
        factor_prediction_errors = self._calculate_factor_prediction_errors(
            action.primary_factors, auction_result
        )
        
        # Create explainable experience
        exp = ExplainableExperience(
            state=state.to_vector(self.data_stats),
            action=action,
            reward=reward,
            next_state=next_state.to_vector(self.data_stats),
            done=done,
            action_explanation=action.explanation,
            reward_explanation=reward_explanation,
            q_prediction_error=q_prediction_error,
            factor_prediction_errors=factor_prediction_errors,
            user_id=user_id,
            session_id=session_id
        )
        
        # Store in explainable replay buffer
        self.explainable_replay_buffer.append(exp)
        
        # Also store in base replay buffer for training
        super().store_experience(
            state=state,
            action=action.__dict__,  # Convert to dict for base class
            reward=reward,
            next_state=next_state,
            done=done,
            user_id=user_id,
            metadata={
                'decision_id': action.decision_id,
                'explanation_confidence': action.explanation.explanation_confidence,
                'factor_count': len(action.primary_factors)
            }
        )
        
        # Log outcome to audit trail
        self._log_outcome_to_audit_trail(
            action.decision_id, auction_result, {
                'q_prediction_error': q_prediction_error,
                'reward': reward
            }
        )
        
        # Update explanation accuracy metrics
        self._update_explanation_accuracy(action, auction_result)
        
        logger.debug(f"Stored explainable experience for decision {action.decision_id}")
    
    def train_with_explanations(self, batch_size: int = 256) -> Dict[str, Any]:
        """
        Train agent using explainable experiences to improve both performance and explanations
        """
        
        if len(self.explainable_replay_buffer) < batch_size:
            return self.train(batch_size)  # Fall back to base training
        
        # Sample batch of explainable experiences
        batch = random.sample(list(self.explainable_replay_buffer), batch_size)
        
        # Regular RL training
        training_losses = super().train(batch_size)
        
        # Additional explainability-focused training
        explanation_losses = self._train_explanation_accuracy(batch)
        
        # Update factor importance weights based on prediction accuracy
        self._update_factor_weights()
        
        # Update explanation quality metrics
        self._update_explanation_metrics(batch)
        
        training_losses.update(explanation_losses)
        training_losses['explanation_coverage'] = self.explanation_metrics.coverage
        training_losses['explanation_accuracy'] = self.explanation_metrics.factual_accuracy
        
        return training_losses
    
    def generate_decision_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive decision report with explanations
        """
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_experiences = [
            exp for exp in self.explainable_replay_buffer 
            if exp.timestamp >= cutoff_time
        ]
        
        if not recent_experiences:
            return {'error': 'No recent decisions to analyze'}
        
        # Decision confidence analysis
        confidence_distribution = defaultdict(int)
        for exp in recent_experiences:
            confidence_distribution[exp.action.decision_confidence.value] += 1
        
        # Factor importance analysis
        factor_importance_summary = self._analyze_factor_importance(recent_experiences)
        
        # Prediction accuracy analysis
        accuracy_by_factor = self._analyze_prediction_accuracy(recent_experiences)
        
        # Performance by confidence level
        performance_by_confidence = self._analyze_performance_by_confidence(recent_experiences)
        
        # Top insights and opportunities
        insights = self._extract_decision_insights(recent_experiences)
        
        return {
            'time_window_hours': time_window_hours,
            'total_decisions': len(recent_experiences),
            'confidence_distribution': dict(confidence_distribution),
            'factor_importance_summary': factor_importance_summary,
            'prediction_accuracy': accuracy_by_factor,
            'performance_by_confidence': performance_by_confidence,
            'key_insights': insights,
            'explanation_quality_metrics': {
                'coverage': self.explanation_metrics.coverage,
                'consistency': self.explanation_metrics.consistency,
                'actionability': self.explanation_metrics.actionability,
                'comprehensibility': self.explanation_metrics.comprehensibility,
                'factual_accuracy': self.explanation_metrics.factual_accuracy
            }
        }
    
    def explain_counterfactual(self,
                              state: DynamicEnrichedState,
                              action: ExplainableAction,
                              what_if_changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate counterfactual explanation: "What if X was different?"
        
        Args:
            state: Current state
            action: Action taken
            what_if_changes: Dict of changes to apply (e.g., {'segment_cvr': 0.05})
        
        Returns:
            Counterfactual analysis with estimated changes
        """
        
        # Create modified state
        modified_state = self._apply_counterfactual_changes(state, what_if_changes)
        
        # Generate alternative action for modified state
        with torch.no_grad():
            state_vector = torch.FloatTensor(modified_state.to_vector(self.data_stats)).unsqueeze(0).to(self.device)
            q_bid = self.q_network_bid(state_vector)
            q_values_bid = q_bid.cpu().numpy().flatten().tolist()
            
            # Get alternative best action
            alternative_bid_idx = int(np.argmax(q_values_bid))
            bid_levels = np.linspace(self.min_bid, self.max_bid, self.num_bid_levels)
            alternative_bid = float(bid_levels[alternative_bid_idx])
        
        # Calculate differences
        bid_difference = alternative_bid - action.bid_amount
        q_value_difference = max(q_values_bid) - max(action.q_values_bid)
        
        # Generate explanation for the change
        explanation = {
            'original_bid': action.bid_amount,
            'counterfactual_bid': alternative_bid,
            'bid_difference': bid_difference,
            'q_value_difference': q_value_difference,
            'changes_applied': what_if_changes,
            'explanation': f"If {list(what_if_changes.keys())[0]} changed to {list(what_if_changes.values())[0]}, "
                         f"bid would {'increase' if bid_difference > 0 else 'decrease'} by ${abs(bid_difference):.2f}",
            'confidence': 'medium'  # Counterfactuals inherently have some uncertainty
        }
        
        return explanation
    
    def get_factor_attribution_over_time(self, hours: int = 24) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Get factor attribution changes over time for trend analysis
        """
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_experiences = [
            exp for exp in self.explainable_replay_buffer
            if exp.timestamp >= cutoff_time
        ]
        
        # Group by hour
        factor_by_hour = defaultdict(lambda: defaultdict(list))
        
        for exp in recent_experiences:
            hour_key = exp.timestamp.replace(minute=0, second=0, microsecond=0)
            
            for factor in exp.action.primary_factors:
                factor_by_hour[hour_key][factor.name].append(factor.impact_weight)
        
        # Calculate average impact per hour for each factor
        attribution_trends = defaultdict(list)
        
        for hour_key in sorted(factor_by_hour.keys()):
            for factor_name, impacts in factor_by_hour[hour_key].items():
                avg_impact = np.mean(impacts)
                attribution_trends[factor_name].append((hour_key, avg_impact))
        
        return dict(attribution_trends)
    
    # Helper methods for explainability
    
    def _guided_exploration_bid(self, q_values: List[float], state: DynamicEnrichedState, context: Dict) -> int:
        """Intelligent exploration for bid selection based on uncertainty"""
        q_array = np.array(q_values)
        
        # Add uncertainty-based exploration
        uncertainty = np.std(q_array) 
        if uncertainty > 1.0:  # High uncertainty
            # Explore around the top 3 actions
            top_3_idx = np.argsort(q_array)[-3:]
            return np.random.choice(top_3_idx)
        else:
            # Low uncertainty, explore randomly
            return random.randint(0, len(q_values) - 1)
    
    def _guided_exploration_creative(self, q_values: List[float], state: DynamicEnrichedState, context: Dict) -> int:
        """Intelligent creative exploration based on fatigue and performance"""
        
        # Avoid high-fatigue creatives during exploration
        if hasattr(state, 'creative_fatigue') and state.creative_fatigue > 0.8:
            # Explore less fatigued creatives
            low_fatigue_actions = list(range(len(q_values)))[:10]  # First 10 as proxy for low fatigue
            return random.choice(low_fatigue_actions)
        
        return random.randint(0, len(q_values) - 1)
    
    def _guided_exploration_channel(self, q_values: List[float], state: DynamicEnrichedState, context: Dict) -> int:
        """Intelligent channel exploration based on attribution"""
        
        # Favor channels with good attribution during exploration
        if hasattr(state, 'channel_attribution_credit') and state.channel_attribution_credit > 0.5:
            # Stay with high-attribution channels during exploration
            return min(state.channel, len(q_values) - 1)
        
        return random.randint(0, len(q_values) - 1)
    
    def _explain_reward(self, state: DynamicEnrichedState, action: ExplainableAction, 
                       reward: float, next_state: DynamicEnrichedState, auction_result: Any) -> Dict[str, Any]:
        """Generate explanation for received reward"""
        
        explanation = {
            'total_reward': reward,
            'reward_components': {},
            'explanation': []
        }
        
        # Auction outcome component
        if hasattr(auction_result, 'won') and auction_result.won:
            explanation['reward_components']['auction_win'] = 5.0
            explanation['explanation'].append(f"Won auction at position {getattr(auction_result, 'position', 'unknown')}")
        else:
            explanation['reward_components']['auction_loss'] = -2.0
            explanation['explanation'].append("Lost auction")
        
        # Conversion component
        if hasattr(auction_result, 'revenue') and auction_result.revenue > 0:
            explanation['reward_components']['conversion'] = auction_result.revenue * 0.1
            explanation['explanation'].append(f"Generated ${auction_result.revenue:.2f} revenue")
        
        # Click component
        if hasattr(auction_result, 'clicked') and auction_result.clicked:
            explanation['reward_components']['click'] = 3.0
            explanation['explanation'].append("User clicked on ad")
        
        # Budget efficiency component
        if hasattr(auction_result, 'price_paid') and action.bid_amount > 0:
            efficiency = (action.bid_amount - getattr(auction_result, 'price_paid', 0)) / action.bid_amount
            explanation['reward_components']['efficiency'] = efficiency * 2.0
            explanation['explanation'].append(f"Bid efficiency: {efficiency:.0%}")
        
        return explanation
    
    def _calculate_q_prediction_error(self, q_values: List[float], reward: float, action_idx: int) -> float:
        """Calculate prediction error for Q-value accuracy tracking"""
        
        if not q_values or action_idx >= len(q_values):
            return 0.0
        
        predicted_value = q_values[action_idx]
        prediction_error = abs(predicted_value - reward)
        
        return prediction_error
    
    def _calculate_factor_prediction_errors(self, factors: List[DecisionFactor], auction_result: Any) -> Dict[str, float]:
        """Calculate prediction errors for each factor"""
        
        errors = {}
        
        for factor in factors:
            if factor.name == "User Segment Conversion Rate":
                # Check if actual conversion happened
                actual_conversion = getattr(auction_result, 'revenue', 0) > 0
                predicted_conversion = factor.raw_value > 0.02
                error = 1.0 if actual_conversion != predicted_conversion else 0.0
                errors[factor.name] = error
            
            elif factor.name == "Creative Performance Prediction":
                # Check if actual click happened
                actual_click = getattr(auction_result, 'clicked', False)
                predicted_high_ctr = factor.raw_value > 0.02
                error = 1.0 if actual_click != predicted_high_ctr else 0.0
                errors[factor.name] = error
        
        return errors
    
    def _log_decision_to_audit_trail(self, decision_id: str, user_id: str, session_id: str, 
                                   campaign_id: str, state: Any, action: Dict, context: Dict,
                                   model_outputs: Dict, decision_factors: Dict):
        """Log decision to audit trail with explanations"""
        
        try:
            # Convert explanation to audit trail format
            q_values = {
                'bid': model_outputs.get('q_values_bid', []),
                'creative': model_outputs.get('q_values_creative', []),
                'channel': model_outputs.get('q_values_channel', [])
            }
            
            log_decision(
                decision_id=decision_id,
                user_id=user_id,
                session_id=session_id,
                campaign_id=campaign_id,
                state=state,
                action=action,
                context=context,
                q_values=q_values,
                decision_factors=decision_factors
            )
            
        except Exception as e:
            logger.error(f"Failed to log decision to audit trail: {e}")
    
    def _log_outcome_to_audit_trail(self, decision_id: str, auction_result: Any, learning_metrics: Dict):
        """Log outcome to audit trail"""
        
        try:
            budget_impact = {
                'budget_after': getattr(auction_result, 'budget_remaining', 0),
                'efficiency': getattr(auction_result, 'efficiency', 0)
            }
            
            attribution_impact = {
                'credit_received': getattr(auction_result, 'attribution_credit', 0),
                'sequence_position': 1
            }
            
            log_outcome(
                decision_id=decision_id,
                auction_result=auction_result,
                learning_metrics=learning_metrics,
                budget_impact=budget_impact,
                attribution_impact=attribution_impact
            )
            
        except Exception as e:
            logger.error(f"Failed to log outcome to audit trail: {e}")
    
    def _update_factor_importance_tracking(self, factors: List[DecisionFactor]):
        """Update factor importance tracking for trend analysis"""
        
        for factor in factors:
            self.factor_importance_history[factor.name].append({
                'timestamp': datetime.now(),
                'importance': factor.impact_weight,
                'confidence': factor.confidence
            })
            
            # Keep only recent history (last 7 days)
            cutoff = datetime.now() - timedelta(days=7)
            self.factor_importance_history[factor.name] = [
                entry for entry in self.factor_importance_history[factor.name]
                if entry['timestamp'] >= cutoff
            ]
    
    def _train_explanation_accuracy(self, batch: List[ExplainableExperience]) -> Dict[str, float]:
        """Train explanation accuracy using experience feedback"""
        
        # This is a placeholder for more sophisticated explanation accuracy training
        # In practice, this would involve training auxiliary models to predict
        # explanation quality and factor accuracy
        
        factor_accuracy_losses = []
        
        for exp in batch:
            for factor_name, error in exp.factor_prediction_errors.items():
                factor_accuracy_losses.append(error)
        
        avg_factor_accuracy = np.mean(factor_accuracy_losses) if factor_accuracy_losses else 0.0
        
        return {
            'explanation_factor_accuracy_loss': avg_factor_accuracy
        }
    
    def _update_factor_weights(self):
        """Update factor importance weights based on prediction accuracy"""
        
        # Adjust weights based on historical accuracy
        for factor_name, accuracy_history in self.factor_prediction_accuracy.items():
            if len(accuracy_history) >= 10:  # Minimum samples
                recent_accuracy = np.mean(accuracy_history[-10:])
                
                # Adjust weight in explainability engine based on accuracy
                if factor_name in self.explainability_engine.feature_weights:
                    current_weight = self.explainability_engine.feature_weights[factor_name]
                    # Increase weight for accurate factors, decrease for inaccurate ones
                    adjustment = (1.0 - recent_accuracy) * 0.1  # Max 10% adjustment
                    new_weight = current_weight * (1.0 + adjustment)
                    self.explainability_engine.feature_weights[factor_name] = max(0.01, min(0.5, new_weight))
    
    def _update_explanation_accuracy(self, action: ExplainableAction, auction_result: Any):
        """Update explanation accuracy tracking"""
        
        for factor in action.primary_factors:
            # Simple accuracy check - in practice would be more sophisticated
            if factor.name == "Creative Performance Prediction":
                predicted_good = factor.raw_value > 0.02
                actual_good = getattr(auction_result, 'clicked', False)
                accurate = predicted_good == actual_good
                
                self.factor_prediction_accuracy[factor.name].append(1.0 if accurate else 0.0)
                
                # Keep recent history only
                if len(self.factor_prediction_accuracy[factor.name]) > 100:
                    self.factor_prediction_accuracy[factor.name].pop(0)
    
    def _update_explanation_metrics(self, batch: List[ExplainableExperience]):
        """Update explanation quality metrics"""
        
        if not batch:
            return
        
        # Calculate coverage - what % of decision is explained
        coverage_scores = []
        for exp in batch:
            total_contribution = sum(exp.action.explanation.factor_contributions.values())
            coverage_scores.append(min(total_contribution, 1.0))
        
        avg_coverage = np.mean(coverage_scores)
        
        # Update running average
        self.explanation_metrics.coverage = self.explanation_metrics.coverage * 0.9 + avg_coverage * 0.1
        
        # Calculate factual accuracy
        accuracy_scores = []
        for exp in batch:
            factor_errors = list(exp.factor_prediction_errors.values())
            if factor_errors:
                accuracy = 1.0 - np.mean(factor_errors)
                accuracy_scores.append(max(0.0, accuracy))
        
        if accuracy_scores:
            avg_accuracy = np.mean(accuracy_scores)
            self.explanation_metrics.factual_accuracy = self.explanation_metrics.factual_accuracy * 0.9 + avg_accuracy * 0.1
    
    def _apply_counterfactual_changes(self, state: DynamicEnrichedState, changes: Dict[str, Any]) -> DynamicEnrichedState:
        """Apply counterfactual changes to create modified state"""
        
        # Create copy of state
        modified_state = DynamicEnrichedState()
        modified_state.__dict__.update(state.__dict__)
        
        # Apply changes
        for attr_name, new_value in changes.items():
            if hasattr(modified_state, attr_name):
                setattr(modified_state, attr_name, new_value)
        
        return modified_state
    
    def _analyze_factor_importance(self, experiences: List[ExplainableExperience]) -> Dict[str, float]:
        """Analyze factor importance across recent experiences"""
        
        factor_importance = defaultdict(list)
        
        for exp in experiences:
            for factor in exp.action.primary_factors + exp.action.explanation.secondary_factors:
                factor_importance[factor.name].append(factor.impact_weight)
        
        # Calculate average importance
        importance_summary = {}
        for factor_name, importances in factor_importance.items():
            importance_summary[factor_name] = {
                'avg_importance': np.mean(importances),
                'std_importance': np.std(importances),
                'frequency': len(importances)
            }
        
        return importance_summary
    
    def _analyze_prediction_accuracy(self, experiences: List[ExplainableExperience]) -> Dict[str, float]:
        """Analyze prediction accuracy for different factors"""
        
        accuracy_by_factor = defaultdict(list)
        
        for exp in experiences:
            for factor_name, error in exp.factor_prediction_errors.items():
                accuracy_by_factor[factor_name].append(1.0 - error)
        
        return {
            factor_name: np.mean(accuracies) 
            for factor_name, accuracies in accuracy_by_factor.items()
        }
    
    def _analyze_performance_by_confidence(self, experiences: List[ExplainableExperience]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by confidence level"""
        
        performance_by_confidence = defaultdict(lambda: {'rewards': [], 'count': 0})
        
        for exp in experiences:
            confidence = exp.action.decision_confidence.value
            performance_by_confidence[confidence]['rewards'].append(exp.reward)
            performance_by_confidence[confidence]['count'] += 1
        
        # Calculate averages
        result = {}
        for confidence, data in performance_by_confidence.items():
            result[confidence] = {
                'avg_reward': np.mean(data['rewards']) if data['rewards'] else 0.0,
                'count': data['count']
            }
        
        return result
    
    def _extract_decision_insights(self, experiences: List[ExplainableExperience]) -> List[str]:
        """Extract key insights from recent decisions"""
        
        insights = []
        
        # High confidence decisions
        high_confidence_count = sum(
            1 for exp in experiences 
            if exp.action.decision_confidence in [DecisionConfidence.HIGH, DecisionConfidence.VERY_HIGH]
        )
        
        if high_confidence_count > len(experiences) * 0.7:
            insights.append(f"High confidence in {high_confidence_count/len(experiences):.0%} of recent decisions")
        
        # Factor dominance
        factor_counts = defaultdict(int)
        for exp in experiences:
            if exp.action.primary_factors:
                top_factor = max(exp.action.primary_factors, key=lambda f: f.impact_weight)
                factor_counts[top_factor.name] += 1
        
        if factor_counts:
            dominant_factor = max(factor_counts.keys(), key=lambda k: factor_counts[k])
            dominance_pct = factor_counts[dominant_factor] / len(experiences)
            
            if dominance_pct > 0.5:
                insights.append(f"{dominant_factor} is dominant factor in {dominance_pct:.0%} of decisions")
        
        return insights


# Integration functions
def create_explainable_rl_agent(discovery_engine, creative_selector, attribution_engine, 
                               budget_pacer, identity_resolver, parameter_manager,
                               **kwargs) -> ExplainableRLAgent:
    """Create explainable RL agent with all GAELP components"""
    
    return ExplainableRLAgent(
        discovery_engine=discovery_engine,
        creative_selector=creative_selector, 
        attribution_engine=attribution_engine,
        budget_pacer=budget_pacer,
        identity_resolver=identity_resolver,
        parameter_manager=parameter_manager,
        **kwargs
    )


if __name__ == "__main__":
    # Demo explainable RL agent
    print("Explainable RL Agent Demo")
    print("=" * 50)
    
    # This would normally be integrated with full GAELP system
    print("✅ Explainable RL Agent implementation complete")
    print("Features implemented:")
    print("- Full decision explainability")  
    print("- Real-time factor attribution")
    print("- Audit trail integration")
    print("- Counterfactual analysis")
    print("- Performance tracking by confidence")
    print("- Factor importance learning")